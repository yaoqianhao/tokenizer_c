/* normalize.c - full NFKC normalization (Unicode UAX #15) plus tokenizer
 * space-marking rules.
 *
 * Pipeline (backwards compatible with the previous "basic NFKC subset"):
 *   1. C0/C1 control removal (keeps \n \r \t before whitespace mapping)
 *   2. Unicode whitespace family -> U+0020 space
 *   3. zero-width / BOM stripping
 *   4. FULL NFKC: NFKD decomposition (canonical + compatibility: full-width,
 *      circled digits, ligatures, roman numerals, super/subscripts,
 *      compat currency signs, ...), canonical ordering, recomposition
 *      (NFKC(NFC(NFKD(x)))), tables generated from UnicodeData.txt via
 *      scripts/gen_nfkc_tables.py
 *   5. optional ASCII case folding
 *   6. space -> ▁ (U+2581) marking (encode path)
 *
 * Extension hooks (unchanged):
 *   - runtime: tokenizer_set_normalizer_hook()
 *   - compile time: TOKENIZER_FULL_NFKC_TABLE nfkc_lookup() provider (kept
 *     for source compatibility; the built-in full NFKC already covers it).
 */
#include "normalize.h"
#include "nfkc.h"
#include "utf8.h"

#include <stdlib.h>
#include <string.h>

/* Extension point kept for ABI/source compatibility (previously the only way
 * to reach full NFKC; the built-in implementation now supersedes it). */
#ifdef TOKENIZER_FULL_NFKC_TABLE
int nfkc_lookup(uint32_t cp, uint32_t *mapped);
#define NFKC_LOOKUP(cp, out) nfkc_lookup((cp), (out))
#else
#define NFKC_LOOKUP(cp, out) 0
#endif

/* U+2581 (▁) encoded as UTF-8. */
static const char kSpaceMark[3] = { (char)0xE2, (char)0x96, (char)0x81 };

static int is_whitespace_cp(uint32_t cp) {
    switch (cp) {
        case 0x0009: /* \t */
        case 0x000A: /* \n */
        case 0x000B:
        case 0x000C:
        case 0x000D:
        case 0x0020:
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return 1;
        default:
            return (cp >= 0x2000 && cp <= 0x200A) ? 1 : 0;
    }
}

/* Zero-width / formatting chars stripped before NFKC. */
static int is_strip_cp(uint32_t cp) {
    switch (cp) {
        case 0x200B: /* zero width space */
        case 0x200C: /* zero width non-joiner */
        case 0x200D: /* zero width joiner */
        case 0xFEFF: /* BOM */
            return 1;
        default:
            return 0;
    }
}

static int is_control_cp(uint32_t cp) {
    /* keep \n \r \t */
    if (cp == 0x000A || cp == 0x000D || cp == 0x0009) return 0;
    if (cp <= 0x001F) return 1;
    if (cp >= 0x007F && cp <= 0x009F) return 1;
    return 0;
}

char *normalize_text(const char *text, size_t len,
                     int mark_spaces, int fold_case, size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!text) return NULL;
    if (len == 0) {
        char *r = (char *)malloc(1);
        if (r) r[0] = '\0';
        return r;
    }
    /* Pass 1: decode to code points, applying control / whitespace /
     * zero-width rules exactly as the legacy basic subset did. */
    size_t cap = len + 1;
    uint32_t *cps = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!cps) return NULL;
    size_t nc = 0;
    size_t i = 0;
    while (i < len) {
        uint32_t cp = 0;
        size_t adv = 0;
        int ok = utf8_decode_at(text, len, i, &cp, &adv);
        (void)ok;
        if (is_control_cp(cp)) {
            i += adv;
            continue;
        }
        if (is_whitespace_cp(cp)) {
            cp = 0x0020;
        }
        if (is_strip_cp(cp)) {
            i += adv;
            continue;
        }
        /* compile-time full-NFKC extension hook (legacy path) */
        uint32_t mapped = 0;
        if (NFKC_LOOKUP(cp, &mapped)) {
            cp = mapped;
        }
        if (nc + 1 >= cap) {
            size_t ncap = cap * 2;
            uint32_t *nb = (uint32_t *)realloc(cps, ncap * sizeof(uint32_t));
            if (!nb) { free(cps); return NULL; }
            cps = nb;
            cap = ncap;
        }
        cps[nc++] = cp;
        i += adv;
    }
    if (nc == 0) {
        free(cps);
        char *r = (char *)malloc(1);
        if (r) r[0] = '\0';
        if (out_len) *out_len = 0;
        return r;
    }
    /* Pass 2: full NFKC (decompose / order / recompose). */
    uint32_t *norm = NULL;
    size_t nn = 0;
    nfkc_normalize_cps(cps, nc, &norm, &nn);
    uint32_t *work = norm ? norm : cps; /* OOM fallback: identity */
    size_t wn = norm ? nn : nc;
    /* Pass 3: optional ASCII case folding. */
    if (fold_case) {
        for (size_t k = 0; k < wn; k++) {
            if (work[k] >= 'A' && work[k] <= 'Z') work[k] += ('a' - 'A');
        }
    }
    /* Pass 4: encode back to UTF-8 (space -> ▁ when mark_spaces). */
    size_t outcap = wn * 4 + 8;
    char *out = (char *)malloc(outcap);
    if (!out) {
        free(cps);
        if (norm) free(norm);
        return NULL;
    }
    size_t o = 0;
    for (size_t k = 0; k < wn; k++) {
        uint32_t cp = work[k];
        if (mark_spaces && cp == 0x0020) {
            if (o + 3 >= outcap) break;
            memcpy(out + o, kSpaceMark, 3);
            o += 3;
            continue;
        }
        char eb[4];
        int nb = utf8_encode(cp, eb);
        if (nb <= 0) {
            if (o + 1 >= outcap) break;
            out[o++] = '?'; /* unmappable cp -> replacement char */
        } else {
            if (o + (size_t)nb >= outcap) break;
            memcpy(out + o, eb, (size_t)nb);
            o += (size_t)nb;
        }
    }
    if (o >= outcap) o = outcap - 1;
    out[o] = '\0';
    free(cps);
    if (norm) free(norm);
    if (out_len) *out_len = o;
    return out;
}
