/* segment_rules.c - deterministic segmentation rules (see header). */
#include "segment_rules.h"
#include "utf8.h"

#include <stdlib.h>
#include <string.h>

int segment_rules_is_digit_cp(unsigned cp) { return cp >= '0' && cp <= '9'; }

int segment_rules_is_operator_cp(unsigned cp) {
    switch (cp) {
        case '+': case '-': case '*': case '/': case '=': case '<': case '>':
        case '!': case '&': case '|': case '^': case '~': case '%':
            return 1;
        default:
            return 0;
    }
}

/* Decode one char, always returning an advance >= 1 (malformed bytes consume
 * one byte, matching the encoder's byte fallback). */
static size_t next_char(const char *s, size_t n, size_t i, uint32_t *cp) {
    size_t adv = 0;
    if (!utf8_decode_at(s, n, i, cp, &adv) || adv == 0) adv = 1;
    if (adv > n - i) adv = n - i;
    return adv;
}

int segment_rules_build(const char *s, size_t n, int digit_grid,
                        int op_isolation, size_t min_digit_run,
                        seg_rules_t *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!s || n == 0) return 0;
    if (!digit_grid && !op_isolation) return 0;

    unsigned char *cut = (unsigned char *)calloc(n, 1);
    if (!cut) return -1;
    int any = 0;
    size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        size_t adv = next_char(s, n, i, &cp);

        if (digit_grid && segment_rules_is_digit_cp(cp)) {
            size_t start = i;
            size_t end = i + adv;
            while (end < n) {
                uint32_t c2 = 0;
                size_t a2 = next_char(s, n, end, &c2);
                if (!segment_rules_is_digit_cp(c2)) break;
                end += a2;
            }
            size_t digits = end - start; /* ASCII digits are 1 byte each */
            if (digits >= min_digit_run && digits > 0) {
                size_t off = 0;
                size_t chunk = (digits % 3) ? (digits % 3) : 3;
                cut[start] = 1; /* cannot fuse with what precedes the run */
                while (off < digits) {
                    size_t nxt = off + chunk;
                    if (nxt < digits) cut[start + nxt] = 1; /* group boundary */
                    off = nxt;
                    chunk = 3;
                }
                if (end < n) cut[end] = 1; /* cannot fuse with what follows */
                any = 1;
            }
            i = end;
            continue;
        }

        if (op_isolation && segment_rules_is_operator_cp(cp)) {
            size_t start = i;
            size_t end = i + adv;
            while (end < n) {
                uint32_t c2 = 0;
                size_t a2 = next_char(s, n, end, &c2);
                if (!segment_rules_is_operator_cp(c2)) break;
                end += a2;
            }
            cut[start] = 1;
            if (end < n) cut[end] = 1;
            any = 1;
            i = end;
            continue;
        }
        i += adv;
    }

    if (!any) {
        free(cut);
        return 0; /* no rule fired: table stays inactive (fast path) */
    }
    out->cut = cut;
    out->n = n;
    out->active = 1;
    return 0;
}

void segment_rules_free(seg_rules_t *r) {
    if (!r) return;
    free(r->cut);
    r->cut = NULL;
    r->n = 0;
    r->active = 0;
}

int segment_rules_edge_ok(const seg_rules_t *r, size_t i, size_t j) {
    if (!r || !r->active || !r->cut) return 1;
    if (j <= i) return 0;
    for (size_t b = i + 1; b < j; b++) {
        if (r->cut[b]) return 0;
    }
    return 1;
}

size_t segment_rules_digit_groups(size_t n, unsigned char *lens, size_t max) {
    if (n == 0) return 0;
    size_t cnt = 0;
    size_t off = 0;
    size_t chunk = (n % 3) ? (n % 3) : 3;
    while (off < n) {
        if (lens && cnt < max) lens[cnt] = (unsigned char)chunk;
        cnt++;
        off += chunk;
        chunk = 3;
    }
    return cnt;
}
