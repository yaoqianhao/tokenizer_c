/* normalize.h - basic NFKC-like normalization subset (internal). */
#ifndef TOKENIZER_NORMALIZE_H
#define TOKENIZER_NORMALIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Normalize a UTF-8 string.
 *   mark_spaces : replace U+0020 spaces with the ▁ (U+2581) sentencepiece
 *                 marker so encode/decode round-trips losslessly.
 *   fold_case   : ASCII upper -> lower case folding.
 * Returns a malloc'ed NUL-terminated UTF-8 buffer or NULL on OOM.
 * *out_len receives the byte length (excluding NUL).
 */
char *normalize_text(const char *text, size_t len,
                     int mark_spaces, int fold_case, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_NORMALIZE_H */
