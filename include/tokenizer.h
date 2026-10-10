/*
 * tokenizer.h - Industrial-grade Unigram tokenizer (Gemini/SentencePiece
 * algorithm framework), pure C11 implementation.
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tokenizer handle. Thread-safe for concurrent encode/decode after load. */
typedef struct tokenizer tokenizer_t;

/* Fixed special-token id block: reserved at training time (see design doc §3). */
typedef struct tokenizer_special_tokens {
    int unk_id;
    int bos_id;
    int eos_id;
    int pad_id;
} tokenizer_special_tokens;

/* ------------------------------------------------------------------ */
/* Model loading / lifecycle                                          */
/* ------------------------------------------------------------------ */

/* Load a model from a custom binary model file (.tkm). Returns NULL on
 * failure; inspect tokenizer_last_error() for the reason. */
tokenizer_t *tokenizer_load(const char *model_path);

/* Load a model from an in-memory binary image (never takes ownership). */
tokenizer_t *tokenizer_load_from_memory(const void *data, size_t size);

/* Free a tokenizer and all owned memory. Safe to pass NULL. */
void tokenizer_free(tokenizer_t *tok);

/* ------------------------------------------------------------------ */
/* Model metadata                                                      */
/* ------------------------------------------------------------------ */

size_t tokenizer_vocab_size(const tokenizer_t *tok);
tokenizer_special_tokens tokenizer_special_tokens_get(const tokenizer_t *tok);
int tokenizer_unk_id(const tokenizer_t *tok);
int tokenizer_bos_id(const tokenizer_t *tok);
int tokenizer_eos_id(const tokenizer_t *tok);
int tokenizer_pad_id(const tokenizer_t *tok);

/* Piece <-> id lookups. Returns -1 when not found. */
int tokenizer_piece_to_id(const tokenizer_t *tok, const char *piece, size_t len);
int tokenizer_id_to_piece(const tokenizer_t *tok, int id, const char **piece, size_t *len);

/* ------------------------------------------------------------------ */
/* Encoding / decoding                                                 */
/* ------------------------------------------------------------------ */

/* Encode a UTF-8 string into token ids using Viterbi DP over a Trie
 * accelerated DAG. out_ids is malloc'ed (caller frees with tokenizer_free_ids).
 * Returns 0 on success, -1 on failure. Guarantees: every input byte is
 * represented (byte-fallback keeps OOV == 0). */
int tokenizer_encode(const tokenizer_t *tok, const char *text, size_t len,
                     int **out_ids, size_t *out_n);
void tokenizer_free_ids(int *ids);

/* Decode token ids back to UTF-8 text. Restores U+2581 marks (▁) to plain
 * spaces and expands <0xXX> byte-fallback pieces to raw bytes.
 * out_text is malloc'ed (caller frees with tokenizer_free_text). */
int tokenizer_decode(const tokenizer_t *tok, const int *ids, size_t n,
                     char **out_text, size_t *out_len);
void tokenizer_free_text(char *text);

/* ------------------------------------------------------------------ */
/* Normalization                                                       */
/* ------------------------------------------------------------------ */

/* Basic normalization subset (NFKC-like):
 *   - strip C0/C1 control chars (keeps \n \r \t)
 *   - map Unicode whitespace family to ASCII space
 *   - map full-width (U+FF01..U+FF5E) to half-width ASCII, U+3000 -> space
 *   - optional ASCII case folding
 *   - mark plain spaces with ▁ (U+2581) so encode/decode round-trips
 * Returns malloc'ed UTF-8 string (caller frees with tokenizer_free_text). */
char *tokenizer_normalize(const char *text, size_t len, size_t *out_len);

/* Extension hook for custom normalization (full NFKC tables, etc.).
 * Set to NULL (default) to disable. The hook receives the raw input and
 * must produce a malloc'ed normalized string in *out (or set *out = NULL
 * to fall back to the built-in basic normalizer). */
typedef char *(*tokenizer_normalizer_hook_fn)(const char *text, size_t len,
                                              size_t *out_len, void *userdata);
void tokenizer_set_normalizer_hook(tokenizer_t *tok,
                                   tokenizer_normalizer_hook_fn hook,
                                   void *userdata);

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */

typedef struct tokenizer_train_options {
    /* Required */
    const char *corpus_path; /* UTF-8 plain text, one document per line */
    const char *model_path;  /* output binary model (.tkm) */
    /* Optional (0 / negative means defaults) */
    size_t  vocab_size;      /* target vocab size; 0 => 4096 (256K needs big corpus) */
    int     max_piece_len;   /* seed n-gram length cap in bytes; 0 => 8 */
    int     max_candidates;  /* seed candidate hash cap; 0 => 1000000 */
    int     max_iters;       /* max EM iterations; 0 => 10 */
    double  em_eps;          /* convergence threshold (log-likelihood delta); 0 => 1e-4 */
    int     shrink_pct;      /* keep top-% pieces per pruning round; 0 => 60 */
    int     min_rounds;      /* min EM rounds after reaching target size; 0 => 3 */
    int     add_byte_fallback; /* add 256 <0xXX> pieces; 0 => on (1) */
    int     fold_case;         /* ASCII case folding in normalization */
    int     verbose;           /* progress logging; 0 => off */
    /* Optional export */
    const char *json_path;   /* optional HF-style tokenizer.json export */
} tokenizer_train_options;

#define TOKENIZER_TRAIN_OPTIONS_DEFAULT { \
    /* corpus_path */ NULL, /* model_path */ NULL, \
    /* vocab_size */ 0, /* max_piece_len */ 0, /* max_candidates */ 0, \
    /* max_iters */ 0, /* em_eps */ 0.0, /* shrink_pct */ 0, /* min_rounds */ 0, \
    /* add_byte_fallback */ 1, /* fold_case */ 0, /* verbose */ 0, \
    /* json_path */ NULL \
}

/* Train a Unigram model from a corpus and save it. Returns 0 on success. */
int tokenizer_train(const tokenizer_train_options *opts);

/* Export a loaded model to simple JSON (HF tokenizer.json-like shape). */
int tokenizer_export_json(const tokenizer_t *tok, const char *json_path);

/* Export a model as a SentencePiece ModelProto (.model, protobuf wire
 * format). The result is loadable by SentencePieceProcessor in
 * transformers / vLLM as a UNIGRAM model. */
int tokenizer_export_sentencepiece(const tokenizer_t *tok, const char *path);

/* Load a SentencePiece .model (any model type). Returns NULL on failure;
 * inspect tokenizer_last_error() for the reason. */
tokenizer_t *tokenizer_load_sentencepiece(const char *path);

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

/* Thread-local last error message (NULL when no error). */
const char *tokenizer_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TOKENIZER_H */
