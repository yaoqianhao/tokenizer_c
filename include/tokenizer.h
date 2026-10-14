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

/* Full NFKC normalization (Unicode UAX #15):
 *   - strip C0/C1 control chars (keeps \n \r \t)
 *   - map the Unicode whitespace family to ASCII space
 *   - full NFKC decomposition / canonical ordering / recomposition
 *     (full-width, circled digits, ligatures, roman numerals,
 *      super/subscripts, compat currency signs, ...)
 *   - strip zero-width chars and BOM
 *   - optional ASCII case folding
 *   - mark plain spaces with U+2581 (▁) when mark_spaces=1
 * Returns a malloc'ed UTF-8 string (free with tokenizer_free_text). */
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
/* Runtime inference-flag overrides (P0-1 / P1-4 / P1-6)               */
/* ------------------------------------------------------------------ */

/* Override the encode-time flags that a model file carries in its header.
 * These setters exist so that already-trained models can be re-evaluated with
 * a fixed encoder without retraining. They only affect encode(), never the
 * piece table, so decode() stays byte-exact either way.
 *
 *   tokenizer_set_norm_reversible(tok, 1):
 *       switch the encoder to lossless (NFC-only) normalization, i.e. stop
 *       folding full-width / compatibility forms (U+FF0C -> U+002C). Fixes the
 *       encode->decode round-trip failures. 0 = legacy full NFKC.
 *   tokenizer_set_segment_rules(tok, 1):
 *       enable the deterministic digit 3-grouping + operator isolation rules.
 *   tokenizer_set_byte_real(tok, 1):
 *       emit the real "<0xXX>" piece id on the byte-fallback path when the
 *       vocabulary contains one (models trained before P1-4 do not, so this is
 *       a no-op for them and needs a retrain to matter). */
void tokenizer_set_norm_reversible(tokenizer_t *tok, int enable);
void tokenizer_set_segment_rules(tokenizer_t *tok, int enable);
void tokenizer_set_byte_real(tokenizer_t *tok, int enable);

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
    int     add_byte_fallback; /* add 256 <0xXX> pieces; 1 => on (default), 0 => off */
    int     fold_case;         /* ASCII case folding in normalization */
    int     verbose;           /* progress logging; 0 => off */
    /* Optional export */
    const char *json_path;   /* optional HF-style tokenizer.json export */
    /* Optional checkpoint / resume */
    const char *checkpoint_path; /* if set, save a per-round checkpoint after
                                    each EM iteration; if a valid checkpoint
                                    already exists at this path, training
                                    resumes from it (skipping re-seeding) */
} tokenizer_train_options;

#define TOKENIZER_TRAIN_OPTIONS_DEFAULT { \
    /* corpus_path */ NULL, /* model_path */ NULL, \
    /* vocab_size */ 0, /* max_piece_len */ 0, /* max_candidates */ 0, \
    /* max_iters */ 0, /* em_eps */ 0.0, /* shrink_pct */ 0, /* min_rounds */ 0, \
    /* add_byte_fallback */ 1, /* fold_case */ 0, /* verbose */ 0, \
    /* json_path */ NULL, /* checkpoint_path */ NULL \
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
