/* nfc.c - Unicode NFC (canonical decomposition + ordering + composition).
 *
 * Why this file exists (2026-10-14): the encoder can now run in three modes
 * (see normalize.h):
 *   NORM_MODE_LEGACY_NFKC : full NFKC               (historical default)
 *   NORM_MODE_NFC         : strict NFC              (this file)
 *   NORM_MODE_REVERSIBLE  : lossless composition    (src/normalize.c)
 *
 * NFC and NFKC share the same canonical combining classes and the same
 * canonical composition pairs, so those are reused from nfkc.c via
 * nfkc_ccc() / nfkc_compose(). Only the *canonical decomposition* table is new
 * (src/nfc_tables.inc) because nfkc_tables.inc stores the compatibility
 * (NFKD) expansion, which would fold U+FF0C to U+002C -- exactly what NFC
 * must not do. The legacy NFKC code path is not modified in any way.
 */
#include "nfc.h"
#include "nfkc.h"

#include "nfc_tables.inc"

#include <stdlib.h>
#include <string.h>

int nfc_decompose_cp(uint32_t cp, const uint32_t **out_seq, size_t *out_len) {
    uint32_t lo = 0, hi = NFC_DECOMP_N;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (kNfcDecompEntries[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < NFC_DECOMP_N && kNfcDecompEntries[lo].cp == cp) {
        const nfc_decomp_entry *e = &kNfcDecompEntries[lo];
        if (out_seq) *out_seq = &kNfcDecompPool[e->off];
        if (out_len) *out_len = (size_t)e->len;
        return 1;
    }
    return 0;
}

void nfc_prewarm(void) {
    /* Tables are sorted static arrays: nothing to build. */
}

/* Canonical ordering (UAX #15 step 2): stable insertion sort by canonical
 * combining class, never reordering across a starter (ccc == 0). */
static void canonical_order(uint32_t *cps, size_t n) {
    for (size_t i = 1; i < n; i++) {
        int cc = nfkc_ccc(cps[i]);
        if (cc == 0) continue;
        size_t j = i;
        while (j > 0) {
            int prev = nfkc_ccc(cps[j - 1]);
            if (prev == 0 || prev <= cc) break;
            uint32_t t = cps[j];
            cps[j] = cps[j - 1];
            cps[j - 1] = t;
            j--;
        }
    }
}

/* Canonical composition (UAX #15 step 3) with the blocking rule: a combining
 * mark composes with the current starter only when no character with an equal
 * or higher combining class sits between them. Composes in place and returns
 * the new length (<= n). */
static size_t canonical_compose(uint32_t *cps, size_t n) {
    size_t on = 0;
    size_t starter = (size_t)-1; /* index of the current starter inside cps */
    int last_cc = 0;             /* ccc of the last non-starter appended */
    for (size_t i = 0; i < n; i++) {
        uint32_t cp = cps[i];
        int cc = nfkc_ccc(cp);
        if (starter != (size_t)-1 && last_cc < cc) {
            uint32_t c = nfkc_compose(cps[starter], cp);
            if (c != 0) {
                cps[starter] = c;
                continue;
            }
        }
        cps[on] = cp;
        if (cc == 0) {
            starter = on;
            last_cc = 0;
        } else {
            last_cc = cc;
        }
        on++;
    }
    return on;
}

int nfc_normalize_cps(const uint32_t *cps, size_t n, uint32_t **out,
                      size_t *out_count) {
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!out || !out_count) return 0;
    if (n == 0) {
        uint32_t *empty = (uint32_t *)malloc(sizeof(uint32_t));
        if (!empty) return 0;
        *out = empty;
        *out_count = 0;
        return 1;
    }

    /* Pass A: canonical decomposition (the table is fully expanded, so one
     * lookup per input code point is enough). */
    size_t cap = n + 1, on = 0;
    uint32_t *buf = (uint32_t *)malloc(cap * sizeof(uint32_t));
    if (!buf) return 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t *seq = NULL;
        size_t sl = 0;
        if (nfc_decompose_cp(cps[i], &seq, &sl) && sl > 0) {
            if (on + sl > cap) {
                size_t ncap = cap;
                while (ncap < on + sl) ncap *= 2;
                uint32_t *nb = (uint32_t *)realloc(buf, ncap * sizeof(uint32_t));
                if (!nb) { free(buf); return 0; }
                buf = nb;
                cap = ncap;
            }
            memcpy(buf + on, seq, sl * sizeof(uint32_t));
            on += sl;
        } else {
            if (on + 1 > cap) {
                size_t ncap = cap * 2;
                uint32_t *nb = (uint32_t *)realloc(buf, ncap * sizeof(uint32_t));
                if (!nb) { free(buf); return 0; }
                buf = nb;
                cap = ncap;
            }
            buf[on++] = cps[i];
        }
    }

    canonical_order(buf, on);
    size_t rn = canonical_compose(buf, on);
    *out = buf;
    *out_count = rn;
    return 1;
}
