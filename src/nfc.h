/* nfc.h - Unicode NFC (canonical normalization) for NORM_MODE_NFC.
 *
 * New module (2026-10-14). The legacy NFKC implementation (src/nfkc.c and
 * src/nfkc_tables.inc) is untouched; this module only adds the canonical
 * path so that a third, switchable normalization mode is available.
 */
#ifndef TOKENIZER_NFC_H
#define TOKENIZER_NFC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Full canonical (non-compatibility) decomposition of cp, i.e. the recursive
 * canonical mapping with compatibility tags (<compat>, <wide>, <narrow>, ...)
 * ignored. Returns 1 and fills *out_seq / *out_len when cp has a canonical
 * mapping, 0 (and leaves the outputs untouched) otherwise. The returned
 * sequence points into static storage and must not be freed. */
int nfc_decompose_cp(uint32_t cp, const uint32_t **out_seq, size_t *out_len);

/* Unicode NFC (UAX #15) over a code point array:
 *   1. canonical decomposition (recursive, non-compatibility)
 *   2. canonical ordering (stable sort by combining class inside each
 *      starter segment)
 *   3. canonical composition with the blocking rule
 * No compatibility mapping is ever applied, so full-width punctuation
 * (U+FF0C) and other compatibility forms survive unchanged -- this is what
 * distinguishes NFC from NFKC. Returns 1 and stores a malloc'ed result in
 * *out / *out_count (caller frees), 0 on OOM. */
int nfc_normalize_cps(const uint32_t *cps, size_t n, uint32_t **out,
                      size_t *out_count);

/* Kept for symmetry with nfkc_prewarm(); the NFC tables are static and sorted,
 * so this is a no-op placeholder for future index caching. */
void nfc_prewarm(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TOKENIZER_NFC_H */
