/* tokenizer_internal.h - shared internal layout & helpers across modules. */
#ifndef TOKENIZER_INTERNAL_H
#define TOKENIZER_INTERNAL_H

#include "tokenizer.h"
#include "trie.h"
#include "utf8.h"
#include "normalize.h"
#include "nfkc.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 64-bit file size helper (P0-1) ---- */
#if defined(_WIN32)
static int file_size64(FILE *fp, int64_t *out) {
    __int64 cur = _ftelli64(fp);
    if (cur < 0) return -1;
    if (_fseeki64(fp, 0, SEEK_END) != 0) return -1;
    __int64 sz = _ftelli64(fp);
    _fseeki64(fp, cur, SEEK_SET);
    if (sz < 0) return -1;
    *out = (int64_t)sz;
    return 0;
}
#else
static int file_size64(FILE *fp, int64_t *out) {
    off_t cur = ftello(fp);
    if (cur < 0) return -1;
    if (fseeko(fp, 0, SEEK_END) != 0) return -1;
    off_t sz = ftello(fp);
    fseeko(fp, cur, SEEK_SET);
    if (sz < 0) return -1;
    *out = (int64_t)sz;
    return 0;
}
#endif

struct tokenizer {
    int vocab_size;
    int cap_slots;      /* allocated slots (>= vocab_size) */
    char **pieces;      /* piece byte strings (malloc'ed) */
    int *piece_lens;    /* piece byte lengths */
    float *scores;      /* log-probabilities (negative) */
    int unk_id;
    int bos_id;
    int eos_id;
    int pad_id;
    int byte_fallback;  /* nonzero => <0xXX> fallback enabled */
    int fold_case;      /* nonzero => normalize folded case at encode (P1-3) */
    trie_t *trie;       /* rebuilt by tokenizer_build_trie() */
    tokenizer_normalizer_hook_fn norm_hook;
    void *norm_userdata;
};

/* Set the thread-local last-error message (printf-style). */
void tokenizer_set_error(const char *fmt, ...);

/* Clear the thread-local last-error message (success path). */
void tokenizer_clear_error(void);

/* Allocate an empty tokenizer shell (no pieces). */
tokenizer_t *tokenizer_create(int unk_id, int bos_id, int eos_id, int pad_id,
                              int byte_fallback);

/* Set / overwrite one piece entry (copy of bytes). */
int tokenizer_set_piece(tokenizer_t *tok, int id, const char *piece,
                        size_t len, float score);

/* Rebuild the internal trie from current pieces. Returns 0 on success. */
int tokenizer_build_trie(tokenizer_t *tok);

/* model.c: binary model format:
 *   magic "TKCZ" (4B), version u32 (=1), vocab u32,
 *   unk,bos,eos,pad i32 x4, flags u32 (bit0=byte_fallback),
 *   then vocab entries: u32 len, len bytes, f32 score. Little-endian. */
int tokenizer_save_binary(const tokenizer_t *tok, const char *path);
tokenizer_t *tokenizer_load_binary_file(const char *path);
tokenizer_t *tokenizer_load_binary_mem(const void *data, size_t size);

/* sentencepiece.c: SentencePiece ModelProto (.model) load. */
tokenizer_t *tokenizer_load_sentencepiece(const char *path);
int tokenizer_export_sentencepiece(const tokenizer_t *tok, const char *path);
int tokenizer_export_json(const tokenizer_t *tok, const char *json_path);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_INTERNAL_H */
