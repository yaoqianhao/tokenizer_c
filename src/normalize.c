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
#include "nfc.h"
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

/* Canonical (NFC-style) composition only.
 *
 * NORM_MODE_REVERSIBLE must not apply *compatibility* decomposition, because
 * that folding is not invertible (U+FF0C ，-> U+002C ,). We therefore look up
 * canonical composition pairs only: if cps[i] is a combining mark (nonzero
 * canonical combining class) we try to compose it with the base collected in
 * the current (maximal-ccc==0) starter segment. Uncomposable marks are kept
 * as-is; no compatibility mapping is ever applied, so the transform is
 * byte-lossless for already-NFC text. */
static void nfc_compose_cps(const uint32_t *cps, size_t n, uint32_t **out,
                            size_t *out_count) {
    *out = NULL;
    *out_count = 0;
    uint32_t *buf = (uint32_t *)malloc((n ? n : 1) * sizeof(uint32_t));
    if (!buf) return;
    size_t on = 0;
    size_t starter = (size_t)-1; /* index in buf of the current starter */
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = cps[i];
        if (nfkc_ccc(cp) != 0) {
            int composed = 0;
            if (starter != (size_t)-1) {
                uint32_t l = buf[starter];
                uint32_t c = nfkc_compose(l, cp);
                if (c != 0) {
                    buf[starter] = c;
                    composed = 1;
                }
            }
            if (composed) continue;
            buf[on++] = cp;
            continue;
        }
        buf[on++] = cp;
        starter = on - 1;
    }
    *out = buf;
    *out_count = on;
}

char *normalize_text(const char *text, size_t len,
                     int mark_spaces, int fold_case, size_t *out_len) {
    return normalize_text_ex(text, len, mark_spaces, fold_case,
                             NORM_MODE_LEGACY_NFKC, out_len);
}

char *normalize_text_ex(const char *text, size_t len, int mark_spaces,
                        int fold_case, int mode, size_t *out_len) {
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
            /* 额外发现 3（可逆性缺口）: U+3000 / U+00A0 / U+202F / U+2000-200A
             * 等 Unicode 空白在 NFKC 下折叠为 U+0020（兼容折叠，不可逆）。
             * 可逆模式只折叠 ASCII 空白，非 ASCII 空白原样保留，保证
             * encode -> decode 的字节级往返；ASCII 空格仍按原规则标记为 ▁。 */
            if (mode != NORM_MODE_REVERSIBLE || cp <= 0x007F) {
                cp = 0x0020;
            }
        }
        if (is_strip_cp(cp)) {
            /* 额外发现 3: 零宽字符 / BOM 的删除同样是不可逆变换，可逆模式
             * 原样保留（由 byte fallback 承载），否则往返丢字节。 */
            if (mode != NORM_MODE_REVERSIBLE) {
                i += adv;
                continue;
            }
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
    /* Pass 2: normalization stage.
     *   mode 0 : full NFKC (compatibility decomposition + recomposition).
     *   mode 1 : lossless canonical composition only - so full-width
     *            punctuation is NOT folded to ASCII.
     *   mode 2 : strict NFC (canonical decomposition + canonical ordering +
     *            canonical composition, UAX #15) - same non-folding property
     *            as mode 1, but fully canonical (adds decomposition of
     *            precomposed forms and the ordering/blocking rules). */
    uint32_t *norm = NULL;
    size_t nn = 0;
    if (mode == NORM_MODE_REVERSIBLE) {
        nfc_compose_cps(cps, nc, &norm, &nn);
    } else if (mode == NORM_MODE_NFC) {
        /* nfc_normalize_cps returns 0 on OOM (norm stays NULL -> hard fail
         * below, consistent with the NFKC path's failure semantics). */
        if (!nfc_normalize_cps(cps, nc, &norm, &nn)) norm = NULL;
    } else {
        nfkc_normalize_cps(cps, nc, &norm, &nn);
    }
    if (!norm) {
        /* 额外发现 2: OOM during normalization — fail loudly instead of
         * silently degrading to identity (consistent with "truncation ==
         * failure"). */
        free(cps);
        return NULL;
    }
    uint32_t *work = norm;
    size_t wn = nn;
    /* Pass 3: optional ASCII case folding. */
    if (fold_case) {
        for (size_t k = 0; k < wn; k++) {
            if (work[k] >= 'A' && work[k] <= 'Z') work[k] += ('a' - 'A');
        }
    }
    /* Pass 4: encode back to UTF-8 (space -> ▁ when mark_spaces). */
    size_t outcap = 64;
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
            if (o + 3 > outcap) {
                size_t ncap = outcap * 2;
                char *nb = (char *)realloc(out, ncap);
                if (!nb) goto oom_out;
                out = nb; outcap = ncap;
            }
            memcpy(out + o, kSpaceMark, 3);
            o += 3;
            continue;
        }
        char eb[4];
        int nb = utf8_encode(cp, eb);
        if (nb <= 0) { nb = 1; eb[0] = '?'; }
        if (o + (size_t)nb > outcap) {
            size_t ncap = (outcap * 2 > o + (size_t)nb) ? outcap * 2
                                                        : o + (size_t)nb;
            char *nb2 = (char *)realloc(out, ncap);
            if (!nb2) goto oom_out;
            out = nb2; outcap = ncap;
        }
        memcpy(out + o, eb, (size_t)nb);
        o += (size_t)nb;
    }
    out[o] = '\0';
    free(cps);
    if (norm) free(norm);
    if (out_len) *out_len = o;
    return out;
oom_out:
    free(cps);
    if (norm) free(norm);
    free(out);
    return NULL;
}
