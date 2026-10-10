/* nfkc.h - full NFKC normalization core (UAX #15): decomposition,
 * canonical ordering and recomposition, driven by compact static tables. */
#ifndef TOKENIZER_NFKC_H
#define TOKENIZER_NFKC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NFKD decomposition of one code point. On success writes *out_len sequence
 * elements to *out_seq (static storage, valid until next call) and returns 1.
 * Returns 0 when the code point has no decomposition. */
int nfkc_decompose_cp(uint32_t cp, const uint32_t **out_seq, size_t *out_len);

/* Canonical combining class (0 for non-combining). */
int nfkc_ccc(uint32_t cp);

/* Canonical composition: if (a,b) recomposes to c returns c, else 0.
 * Respects UAX #15 composition exclusions. */
uint32_t nfkc_compose(uint32_t a, uint32_t b);

/* NFKC normalize a NUL-terminated array of code points into *out (caller
 * frees) writing count to *out_count.  *out may be NULL on OOM. */
void nfkc_normalize_cps(const uint32_t *cps, size_t n,
                        uint32_t **out, size_t *out_count);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_NFKC_H */
