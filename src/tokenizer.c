/* tokenizer.c - inference engine.
 *
 * Encoding = Viterbi DP over a byte-level DAG accelerated by a prefix Trie:
 *   dp[i]            : max log-likelihood of the prefix text[0..i)
 *   best_piece[i]    : piece id chosen for the edge ending at i
 *   prev[i]          : start byte offset of that edge
 * Byte-fallback guarantees OOV == 0: when no piece matches at position i, the
 * current byte is emitted as <0xXX> with a fixed low score.
 *
 * Decoding = piece lookup + join, then ▁ -> space restoration and <0xXX> ->
 * raw byte expansion. Pure table lookups, no branching hot spots.
 */
#include "tokenizer_internal.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* forward declaration (used by tokenizer_build_trie) */
static int is_byte_fallback_piece(const char *p, int len, unsigned *byte);

/* ------------------------------------------------------------------ */
/* thread-local error                                                  */
/* ------------------------------------------------------------------ */
static _Thread_local char tls_error[512];

void tokenizer_set_error(const char *fmt, ...); /* defined below */

const char *tokenizer_last_error(void) {
    return tls_error[0] ? tls_error : NULL;
}

void tokenizer_clear_error(void) {
    tls_error[0] = '\0';
}

static void set_error(const char *msg) {
    snprintf(tls_error, sizeof(tls_error), "%s", msg);
}

void tokenizer_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tls_error, sizeof(tls_error), fmt, ap);
    va_end(ap);
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */
tokenizer_t *tokenizer_create(int unk_id, int bos_id, int eos_id, int pad_id,
                              int byte_fallback) {
    tokenizer_t *tok = (tokenizer_t *)calloc(1, sizeof(*tok));
    if (!tok) return NULL;
    tok->unk_id = unk_id;
    tok->bos_id = bos_id;
    tok->eos_id = eos_id;
    tok->pad_id = pad_id;
    tok->byte_fallback = byte_fallback;
    tok->trie = trie_new();
    if (!tok->trie) {
        free(tok);
        return NULL;
    }
    return tok;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    if (tok->pieces) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->pieces[i]);
        }
    }
    free(tok->pieces);
    free(tok->piece_lens);
    free(tok->scores);
    trie_free(tok->trie);
    free(tok);
}

/* ------------------------------------------------------------------ */
/* piece management                                                    */
/* ------------------------------------------------------------------ */
static int tokenizer_ensure_piece_slots(tokenizer_t *tok, int need) {
    if (need < 0 || need > 100000000) return -1; /* sanity */
    char **np = (char **)malloc((size_t)need * sizeof(char *));
    int *nl = (int *)malloc((size_t)need * sizeof(int));
    float *ns = (float *)malloc((size_t)need * sizeof(float));
    if (!np || !nl || !ns) {
        free(np); free(nl); free(ns);
        return -1;
    }
    if (tok->vocab_size > 0) {
        memcpy(np, tok->pieces, (size_t)tok->vocab_size * sizeof(char *));
        memcpy(nl, tok->piece_lens, (size_t)tok->vocab_size * sizeof(int));
        memcpy(ns, tok->scores, (size_t)tok->vocab_size * sizeof(float));
    }
    free(tok->pieces);
    free(tok->piece_lens);
    free(tok->scores);
    tok->pieces = np;
    tok->piece_lens = nl;
    tok->scores = ns;
    for (int i = tok->vocab_size; i < need; i++) {
        tok->pieces[i] = NULL;
        tok->piece_lens[i] = 0;
        tok->scores[i] = 0.0f;
    }
    tok->vocab_size = need;
    return 0;
}

int tokenizer_set_piece(tokenizer_t *tok, int id, const char *piece,
                        size_t len, float score) {
    if (!tok || id < 0 || !piece) return -1;
    if (id >= tok->vocab_size) {
        if (tokenizer_ensure_piece_slots(tok, id + 1) != 0) return -1;
    }
    char *copy = (char *)malloc(len + 1);
    if (!copy) return -1;
    memcpy(copy, piece, len);
    copy[len] = '\0';
    free(tok->pieces[id]);
    tok->pieces[id] = copy;
    tok->piece_lens[id] = (int)len;
    tok->scores[id] = score;
    return 0;
}

int tokenizer_build_trie(tokenizer_t *tok) {
    trie_free(tok->trie);
    tok->trie = trie_new();
    if (!tok->trie) return -1;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->pieces[i]) continue;
        /* Byte-fallback placeholders are virtual: they are never matched as
         * literal text (this keeps encode/decode round-trips lossless).
         * They are expanded at decode time only. */
        unsigned dummy = 0;
        if (is_byte_fallback_piece(tok->pieces[i], tok->piece_lens[i], &dummy))
            continue;
        if (trie_insert(tok->trie, tok->pieces[i], (size_t)tok->piece_lens[i], i) != 0) {
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* loading / metadata                                                  */
/* ------------------------------------------------------------------ */
static const char *file_ext(const char *path) {
    const char *dot = NULL;
    for (const char *p = path; *p; p++) {
        if (*p == '/' || *p == '\\') dot = NULL;
        else if (*p == '.') dot = p;
    }
    return dot ? dot + 1 : "";
}

tokenizer_t *tokenizer_load(const char *model_path) {
    tokenizer_t *tok = NULL;
    const char *ext = file_ext(model_path);
    if (strcmp(ext, "model") == 0) {
        tok = tokenizer_load_sentencepiece(model_path);
    } else if (strcmp(ext, "json") == 0) {
        set_error("tokenizer.json loading is not supported (export only)");
        return NULL;
    } else {
        tok = tokenizer_load_binary_file(model_path);
        if (tok && tokenizer_build_trie(tok) != 0) {
            tokenizer_free(tok);
            set_error("failed to build trie");
            return NULL;
        }
    }
    if (!tok) return NULL;
    if (tok->trie == NULL) {
        if (tokenizer_build_trie(tok) != 0) {
            tokenizer_free(tok);
            set_error("failed to build trie");
            return NULL;
        }
    }
    tokenizer_clear_error();
    return tok;
}

tokenizer_t *tokenizer_load_from_memory(const void *data, size_t size) {
    tokenizer_t *tok = tokenizer_load_binary_mem(data, size);
    if (!tok) return NULL;
    if (tokenizer_build_trie(tok) != 0) {
        tokenizer_free(tok);
        set_error("failed to build trie");
        return NULL;
    }
    tokenizer_clear_error();
    return tok;
}

size_t tokenizer_vocab_size(const tokenizer_t *tok) {
    return tok ? (size_t)tok->vocab_size : 0;
}

tokenizer_special_tokens tokenizer_special_tokens_get(const tokenizer_t *tok) {
    tokenizer_special_tokens st = { 0, 0, 0, 0 };
    if (tok) {
        st.unk_id = tok->unk_id;
        st.bos_id = tok->bos_id;
        st.eos_id = tok->eos_id;
        st.pad_id = tok->pad_id;
    }
    return st;
}
int tokenizer_unk_id(const tokenizer_t *tok) { return tok ? tok->unk_id : -1; }
int tokenizer_bos_id(const tokenizer_t *tok) { return tok ? tok->bos_id : -1; }
int tokenizer_eos_id(const tokenizer_t *tok) { return tok ? tok->eos_id : -1; }
int tokenizer_pad_id(const tokenizer_t *tok) { return tok ? tok->pad_id : -1; }

int tokenizer_piece_to_id(const tokenizer_t *tok, const char *piece, size_t len) {
    if (!tok || !piece) return -1;
    return trie_lookup(tok->trie, piece, len);
}

int tokenizer_id_to_piece(const tokenizer_t *tok, int id, const char **piece,
                          size_t *len) {
    if (!tok || id < 0 || id >= tok->vocab_size) return -1;
    if (piece) *piece = tok->pieces[id];
    if (len) *len = (size_t)tok->piece_lens[id];
    return 0;
}

void tokenizer_set_normalizer_hook(tokenizer_t *tok,
                                   tokenizer_normalizer_hook_fn hook,
                                   void *userdata) {
    if (tok) {
        tok->norm_hook = hook;
        tok->norm_userdata = userdata;
    }
}

/* ------------------------------------------------------------------ */
/* encode                                                              */
/* ------------------------------------------------------------------ */

/* Byte-fallback pieces "<0xXX>" are virtual: encode emits them as negative
 * virtual ids (-1 - byte), decode expands them back to raw bytes. Negative
 * ids never collide with real piece ids (vocab_size < INT_MAX). */
static const float kByteFallbackScore = -10.0f;

/* is the piece the byte-fallback marker for byte b? */
static int is_byte_fallback_piece(const char *p, int len, unsigned *byte) {
    if (len != 6 || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>')
        return 0;
    if (!isxdigit((unsigned char)p[3]) || !isxdigit((unsigned char)p[4]))
        return 0;
    unsigned v = 0;
    for (int k = 3; k <= 4; k++) {
        char c = p[k];
        unsigned d = (c >= '0' && c <= '9') ? (unsigned)(c - '0')
                  : (c >= 'A' && c <= 'F') ? (unsigned)(c - 'A' + 10)
                  : (unsigned)(c - 'a' + 10);
        v = v * 16 + d;
    }
    *byte = v;
    return 1;
}

static char *normalize_for_encode(const tokenizer_t *tok, const char *text,
                                  size_t len, size_t *out_len) {
    if (tok->norm_hook) {
        char *r = tok->norm_hook(text, len, out_len, tok->norm_userdata);
        if (r) return r;
        /* hook returned NULL => fall through to builtin */
    }
    return normalize_text(text, len, 1 /*mark_spaces*/, tok->fold_case,
                          out_len);
}

int tokenizer_encode(const tokenizer_t *tok, const char *text, size_t len,
                     int **out_ids, size_t *out_n) {
    if (out_ids) *out_ids = NULL;
    if (out_n) *out_n = 0;
    if (!tok || !text) {
        set_error("tokenizer_encode: NULL argument");
        return -1;
    }
    size_t nlen = 0;
    char *norm = normalize_for_encode(tok, text, len, &nlen);
    if (!norm) {
        set_error("tokenizer_encode: normalization failed");
        return -1;
    }
    const char *s = norm;
    size_t n = nlen;

    double *dp = (double *)malloc((n + 1) * sizeof(double));
    int *best = (int *)malloc((n + 1) * sizeof(int));
    size_t *prev = (size_t *)malloc((n + 1) * sizeof(size_t));
    int *path = (int *)malloc((n + 1) * sizeof(int));
    if (!dp || !best || !prev || !path) {
        free(norm); free(dp); free(best); free(prev); free(path);
        set_error("tokenizer_encode: OOM");
        return -1;
    }

    for (size_t i = 0; i <= n; i++) {
        dp[i] = -INFINITY;
        best[i] = -1;
        prev[i] = 0;
    }
    dp[0] = 0.0;

    enum { MAX_MATCH = 64 };
    int ids[MAX_MATCH];
    size_t ends[MAX_MATCH];

    for (size_t i = 0; i < n; i++) {
        if (dp[i] == -INFINITY) {
            /* unreachable (should never trigger): repair by falling back one
             * byte from i-1 so the DP table stays contiguous */
            dp[i] = (i == 0) ? 0.0 : dp[i - 1] + kByteFallbackScore;
            best[i] = (i == 0) ? -1 : -1 - (int)(unsigned char)s[i - 1];
            prev[i] = (i == 0) ? 0 : i - 1;
        }
        int m = trie_match_prefixes(tok->trie, s, n, i, ids, ends, MAX_MATCH);
        for (int k = 0; k < m; k++) {
            size_t j = ends[k];
            int pid = ids[k];
            double cand = dp[i] + (double)tok->scores[pid];
            if (cand > dp[j]) {
                dp[j] = cand;
                best[j] = pid;
                prev[j] = i;
            }
        }
        /* P1-2: unconditional 1-byte edge so every position keeps a default
         * byte-fallback transition; the old "only when unreachable" repair
         * could leave dp holes that backtracking never resolved. */
        double fb = dp[i] + kByteFallbackScore;
        if (fb > dp[i + 1]) {
            dp[i + 1] = fb;
            best[i + 1] = -1 - (int)(unsigned char)s[i];
            prev[i + 1] = i;
        }
    }

    /* backtrack */
    size_t cnt = 0;
    for (size_t i = n; i > 0;) {
        int pid = best[i];
        if (pid < 0) {
            /* virtual byte-fallback markers (id = -1 - byte) emitted by the
             * repair/fallback paths above; decode() expands them back to raw
             * bytes. They are the ONLY legitimate negative ids here. */
            if (pid < -256) {
                free(norm); free(dp); free(best); free(prev); free(path);
                set_error("tokenizer_encode: unreachable state");
                return -1;
            }
            path[cnt++] = pid;
            i = prev[i];
            continue;
        }
        path[cnt++] = pid;
        i = prev[i];
    }
    /* reverse into output */
    int *res = (int *)malloc(cnt * sizeof(int));
    if (!res) {
        free(norm); free(dp); free(best); free(prev); free(path);
        set_error("tokenizer_encode: OOM");
        return -1;
    }
    for (size_t k = 0; k < cnt; k++) res[k] = path[cnt - 1 - k];

    free(norm); free(dp); free(best); free(prev); free(path);
    if (out_ids) *out_ids = res; else free(res);
    if (out_n) *out_n = cnt;
    tokenizer_clear_error();
    return 0;
}

void tokenizer_free_ids(int *ids) { free(ids); }

/* ------------------------------------------------------------------ */
/* decode                                                              */
/* ------------------------------------------------------------------ */
int tokenizer_decode(const tokenizer_t *tok, const int *ids, size_t n,
                     char **out_text, size_t *out_len) {
    if (out_text) *out_text = NULL;
    if (out_len) *out_len = 0;
    if (!tok || !ids) {
        set_error("tokenizer_decode: NULL argument");
        return -1;
    }
    size_t cap = 0;
    for (size_t k = 0; k < n; k++) {
        int id = ids[k];
        if (id < 0 && id >= -256) {
            cap += 2; /* virtual byte-fallback marker -> 1 raw byte */
            continue;
        }
        if (id < 0 || id >= tok->vocab_size) {
            set_error("tokenizer_decode: id out of range");
            return -1;
        }
        cap += (size_t)tok->piece_lens[id] + 1;
    }
    cap = cap * 2 + 8; /* ▁ expands 3B -> 1B, so 2x is plenty */
    char *buf = (char *)malloc(cap);
    if (!buf) {
        set_error("tokenizer_decode: OOM");
        return -1;
    }
    size_t o = 0;
    for (size_t k = 0; k < n; k++) {
        int id = ids[k];
        /* virtual byte-fallback markers emitted by encode */
        if (id < 0 && id >= -256) {
            if (o + 1 >= cap) goto overflow;
            buf[o++] = (char)(-id - 1);
            continue;
        }
        const char *p = tok->pieces[id];
        int plen = tok->piece_lens[id];
        unsigned byte = 0;
        if (is_byte_fallback_piece(p, plen, &byte)) {
            if (o + 1 >= cap) goto overflow;
            buf[o++] = (char)byte;
            continue;
        }
        /* ▁ -> ' ' (and any other occurrence of the mark) */
        for (int b = 0; b < plen; b++) {
            if ((unsigned char)p[b] == 0xE2 && b + 2 < plen &&
                (unsigned char)p[b+1] == 0x96 && (unsigned char)p[b+2] == 0x81) {
                if (o + 1 >= cap) goto overflow;
                buf[o++] = ' ';
                b += 2;
            } else {
                if (o + 1 >= cap) goto overflow;
                buf[o++] = p[b];
            }
        }
    }
    buf[o] = '\0';
    if (out_text) *out_text = buf; else free(buf);
    if (out_len) *out_len = o;
    tokenizer_clear_error();
    return 0;
overflow:
    free(buf);
    set_error("tokenizer_decode: output buffer overflow");
    return -1;
}

void tokenizer_free_text(char *text) { free(text); }

/* ------------------------------------------------------------------ */
/* public normalize                                                    */
/* ------------------------------------------------------------------ */
char *tokenizer_normalize(const char *text, size_t len, size_t *out_len) {
    return normalize_text(text, len, 1 /*mark_spaces*/, 0, out_len);
}
