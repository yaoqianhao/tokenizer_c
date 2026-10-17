/* tiktoken_bpe.h - tiktoken-compatible byte-level BPE (public API).
 *
 * Independent C11 implementation of the tiktoken encoder/decoder semantics:
 *   - vocabulary key  = raw bytes, value = rank ("rank is the id")
 *   - encoding        = repeated merge of the adjacent pair whose concatenated
 *                       bytes have the smallest rank, until no pair is known
 *                       (never frequency based)
 *   - pre-tokenization = optional regex-like splitting, always approximate
 *                        (see the "APPROXIMATION" notes in tiktoken_bpe.c)
 *   - special tokens  = longest-first literal match, split off before the
 *                       ordinary text is encoded
 *   - decoding        = id -> bytes, concatenated, UTF-8 decoded with invalid
 *                       bytes replaced by U+FFFD (tiktoken errors="replace")
 *
 * Style follows tokenizer.h: opaque handle, 0/-1 returns, malloc'ed outputs
 * released by the matching ttbpe_free_* helper, thread-local last_error.
 * No third-party dependency, no C++ and no allocation in the caller.
 */
#ifndef TIKTOKEN_BPE_H
#define TIKTOKEN_BPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tiktoken_bpe tiktoken_bpe_t;

/* Zero-copy view of one pre-tokenized piece; ptr points into the text buffer
 * that was passed to ttbpe_split() and stays valid as long as that buffer. */
typedef struct {
    const char *ptr;
    size_t len;
} ttbpe_piece;

/* Pre-tokenization modes. */
#define TTBPE_SPLIT_NONE   0 /* no split: encode the whole byte range at once
                              * (identical to tiktoken _encode_single_piece) */
#define TTBPE_SPLIT_GPT2   1 /* GPT-2 / r50k style pattern */
#define TTBPE_SPLIT_CL100K 2 /* cl100k / o200k style pattern */

/* Character classes used by the approximate pre-tokenizer.  A classifier hook
 * (see ttbpe_set_char_classifier) may return one of these for a code point, or
 * a negative value to fall back to the built-in table. */
#define TTBPE_CC_SPACE  0
#define TTBPE_CC_LETTER 1
#define TTBPE_CC_DIGIT  2
#define TTBPE_CC_OTHER  3

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Load a vocabulary.  The format is detected from the content (first
 * non-whitespace byte) and, when that is inconclusive, from the extension:
 *   .tok.json  - JSON vocabulary (xAI Grok style, see docs/ttbpe_spec.md 4.1)
 *   .tiktoken  - one base64 line per token, line index (0 based) = id
 * Returns NULL and sets last_error() on failure. */
tiktoken_bpe_t *ttbpe_load(const char *path);

/* Same, for a vocabulary already held in memory. */
tiktoken_bpe_t *ttbpe_load_from_memory(const void *data, size_t size);

void ttbpe_free(tiktoken_bpe_t *tt);

/* ------------------------------------------------------------------ */
/* introspection                                                       */
/* ------------------------------------------------------------------ */

size_t      ttbpe_vocab_size(const tiktoken_bpe_t *tt);
const char *ttbpe_format(const tiktoken_bpe_t *tt);    /* "tok.json" | "tiktoken" */
const char *ttbpe_word_split(const tiktoken_bpe_t *tt);/* e.g. "V1", NULL if absent */
size_t      ttbpe_special_count(const tiktoken_bpe_t *tt);
const char *ttbpe_last_error(void);

/* Raw bytes behind a token id (borrowed pointer).  Returns -1 when the id is
 * not defined in this vocabulary. */
int ttbpe_id_bytes(const tiktoken_bpe_t *tt, int id, const unsigned char **bytes,
                   size_t *len);

/* ------------------------------------------------------------------ */
/* configuration                                                       */
/* ------------------------------------------------------------------ */

/* Register a special token.  It is added to the vocabulary and gets the next
 * free id (vocab_size, vocab_size+1, ...) unless the vocabulary already
 * carried that token.  Returns 0 on success, -1 on failure. */
int ttbpe_add_special(tiktoken_bpe_t *tt, const char *token, size_t len);

int ttbpe_set_split_mode(tiktoken_bpe_t *tt, int mode);      /* TTBPE_SPLIT_* */
int ttbpe_set_cache_enabled(tiktoken_bpe_t *tt, int enabled);/* default: on */

/* Install a global code-point classifier (NULL restores the built-in one).
 * Called as fn(cp); return TTBPE_CC_* or a negative value for "use built-in". */
void ttbpe_set_char_classifier(int (*fn)(uint32_t cp));

/* ------------------------------------------------------------------ */
/* encoding                                                            */
/* ------------------------------------------------------------------ */

/* Default semantics == tiktoken Encoding.encode(text) with
 * allowed_special=set(), disallowed_special="all": an unregistered-but-allowed
 * special token occurrence is an error (-1, last_error set). */
int ttbpe_encode(const tiktoken_bpe_t *tt, const char *text, size_t len,
                 int **out_ids, size_t *out_n);

/* allow_all_special = 1  -> allowed_special="all": specials are emitted as ids
 * reject_disallowed = 1  -> a special that is not allowed is an error
 * reject_disallowed = 0  -> such a special is encoded as ordinary text */
int ttbpe_encode_ex(const tiktoken_bpe_t *tt, const char *text, size_t len,
                    int allow_all_special, int reject_disallowed,
                    int **out_ids, size_t *out_n);

/* == tiktoken encode_ordinary: special tokens are plain text. */
int ttbpe_encode_ordinary(const tiktoken_bpe_t *tt, const char *text, size_t len,
                          int **out_ids, size_t *out_n);

/* == tiktoken _encode_single_piece: no split, no special handling; the whole
 * byte range is byte-pair encoded. */
int ttbpe_encode_single_piece(const tiktoken_bpe_t *tt, const char *bytes, size_t len,
                              int **out_ids, size_t *out_n);

/* ------------------------------------------------------------------ */
/* decoding                                                            */
/* ------------------------------------------------------------------ */

/* id -> bytes -> UTF-8, invalid byte sequences replaced by U+FFFD. */
int ttbpe_decode(const tiktoken_bpe_t *tt, const int *ids, size_t n,
                 char **out_text, size_t *out_len);

/* id -> bytes, concatenated, no UTF-8 interpretation. */
int ttbpe_decode_bytes(const tiktoken_bpe_t *tt, const int *ids, size_t n,
                       unsigned char **out_bytes, size_t *out_len);

/* ------------------------------------------------------------------ */
/* pre-tokenization / debugging                                        */
/* ------------------------------------------------------------------ */

/* Zero-copy piece list for the handle's current split mode. */
int ttbpe_split(const tiktoken_bpe_t *tt, const char *text, size_t len,
                ttbpe_piece **out, size_t *out_n);

/* UTF-8 encode a code point the way ttbpe_decode() would (U+FFFD for cp 0xFFFD
 * and for out-of-range values). */
int ttbpe_utf8_encode(uint32_t cp, char *dst, size_t dst_cap, size_t *written);

/* ------------------------------------------------------------------ */
/* memory helpers                                                      */
/* ------------------------------------------------------------------ */

void ttbpe_free_pieces(ttbpe_piece *p);
void ttbpe_free_ids(int *ids);
void ttbpe_free_buf(void *p);

#ifdef __cplusplus
}
#endif

#endif /* TIKTOKEN_BPE_H */
