/* utf8.h - minimal UTF-8 decode/encode helpers (internal). */
#ifndef TOKENIZER_UTF8_H
#define TOKENIZER_UTF8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode one UTF-8 code point starting at s[i].
 * On success stores the code point in *cp, the number of bytes consumed in
 * *advance (1..4) and returns 1. On malformed input stores the raw byte in
 * *cp, sets *advance = 1 and returns 0 (so callers can do byte-fallback). */
int utf8_decode_at(const char *s, size_t len, size_t i,
                   uint32_t *cp, size_t *advance);

/* Encode a code point into out[0..3]. Returns bytes written (1..4),
 * 0 if the code point is invalid. */
int utf8_encode(uint32_t cp, char out[4]);

/* Number of bytes for a lead byte, 0 if invalid. */
int utf8_seq_len(uint8_t lead);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_UTF8_H */
