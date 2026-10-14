/* segment_rules.h - deterministic segmentation rules (P1-6).
 *
 * These rules are *encoding-time constraints* layered on top of the Unigram
 * Viterbi search. They do not touch the piece table: they only forbid some
 * piece edges, so the search stays a generic Viterbi DP and stays lossless.
 *
 * Rules:
 *   digit grid  : a maximal ASCII digit run with >= min_digit_run digits is cut
 *                 into right-aligned groups of three digits (the leading group
 *                 may be 1 or 2 digits, exactly like the TokEval "ideal"
 *                 grouping n % 3 || 3, +3, ...). Both ends of the run are cut
 *                 as well, so nothing can fuse an operator/letter/space with
 *                 the number.
 *   operator    : both ends of every maximal operator run are cut, so no piece
 *                 can fuse an operator with its operands.
 *
 * A "cut" at byte offset i means: no piece may span the boundary between
 * i-1 and i. Single bytes (byte fallback) can never straddle a cut, so the
 * byte-level fallback guarantee (OOV == 0) is preserved.
 */
#ifndef TOKENIZER_SEGMENT_RULES_H
#define TOKENIZER_SEGMENT_RULES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char *cut; /* cut[i] != 0: no piece may straddle byte offset i */
    size_t n;           /* byte length of the analysed text */
    int active;         /* 1 when at least one cut was produced */
} seg_rules_t;

/* Build the rule table for s[0..n). Returns 0 on success (also when no rule
 * fires: out->active stays 0), -1 on OOM. */
int segment_rules_build(const char *s, size_t n, int digit_grid,
                        int op_isolation, size_t min_digit_run,
                        seg_rules_t *out);

void segment_rules_free(seg_rules_t *r);

/* 1 when the piece edge [i, j) is allowed (never straddles a cut). */
int segment_rules_edge_ok(const seg_rules_t *r, size_t i, size_t j);

/* Right-aligned 3-digit grouping of a run of n digits: fills lens[] with the
 * group lengths (1..3, summing to n) up to max entries and returns the total
 * number of groups. Used by the determinism pass so that exactly the required
 * digit-group pieces exist in the vocabulary. */
size_t segment_rules_digit_groups(size_t n, unsigned char *lens, size_t max);

/* ASCII operator set, kept in sync with TokEval's operator regex
 * [+-*\/=<>!&|^~%]. */
int segment_rules_is_operator_cp(unsigned cp);
int segment_rules_is_digit_cp(unsigned cp);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_SEGMENT_RULES_H */
