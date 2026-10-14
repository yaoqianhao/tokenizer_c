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
 *
 * This is the legacy entry point: mode NORM_MODE_LEGACY_NFKC (full NFKC,
 * including lossy compatibility folding such as U+FF0C -> U+002C).
 */
char *normalize_text(const char *text, size_t len,
                     int mark_spaces, int fold_case, size_t *out_len);

/* Normalization modes (P0-1 reversible normalization, extended 2026-10-14
 * with a strict NFC mode). Values are stable: tokenizer.h mirrors them as
 * TOKENIZER_NORM_MODE_* and the CLI maps --norm=nfkc|nfc|reversible onto
 * them. */
#define NORM_MODE_LEGACY_NFKC 0 /* full NFKC: compatibility folding (lossy)   */
#define NORM_MODE_REVERSIBLE  1 /* canonical composition only (lossless)     */
#define NORM_MODE_NFC         2 /* strict NFC: canonical decomposition +
                                 * canonical ordering + canonical
                                 * composition; never applies compatibility
                                 * folding (U+FF0C stays U+FF0C) */

/*
 * Mode-aware variant used by the encode path.
 *   NORM_MODE_LEGACY_NFKC : full NFKC, keeps historical behaviour.
 *   NORM_MODE_REVERSIBLE  : canonical (NFC-style) composition only, i.e.
 *                           compatibility folding is disabled. Full-width
 *                           punctuation such as ，(U+FF0C) survives unchanged
 *                           so encode -> decode is byte-exact.
 * Same ownership rules as normalize_text().
 */
char *normalize_text_ex(const char *text, size_t len, int mark_spaces,
                        int fold_case, int mode, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_NORMALIZE_H */
