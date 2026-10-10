/* model.c - custom binary model serialization & simple JSON export.
 *
 * Binary format (little-endian):
 *   offset 0   : magic "TKCZ"                 (4 bytes)
 *   offset 4   : version   u32 = 1
 *   offset 8   : vocab     u32
 *   offset 12  : unk,bos,eos,pad              (4 x i32)
 *   offset 28  : flags     u32 (bit0 = byte_fallback enabled)
 *   offset 32  : vocab entries:
 *                  u32 len, len bytes (piece UTF-8), f32 score (log-prob)
 */
#include "tokenizer_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* little-endian helpers                                               */
/* ------------------------------------------------------------------ */
static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}
static void put_i32(unsigned char *p, int32_t v) { put_u32(p, (uint32_t)v); }
static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static int32_t get_i32(const unsigned char *p) { return (int32_t)get_u32(p); }

/* float bits (assumes IEEE-754, true for all supported targets) */
static void put_f32(unsigned char *p, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    put_u32(p, bits);
}
static float get_f32(const unsigned char *p) {
    uint32_t bits = get_u32(p);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ------------------------------------------------------------------ */
/* binary save                                                         */
/* ------------------------------------------------------------------ */
int tokenizer_save_binary(const tokenizer_t *tok, const char *path) {
    if (!tok || !path) return -1;
    /* compute size */
    size_t size = 32;
    for (int i = 0; i < tok->vocab_size; i++) {
        size += 4 + (size_t)tok->piece_lens[i] + 4;
    }
    unsigned char *buf = (unsigned char *)malloc(size ? size : 1);
    if (!buf) return -1;
    size_t o = 0;
    memcpy(buf + o, "TKCZ", 4); o += 4;
    put_u32(buf + o, 1); o += 4;
    put_u32(buf + o, (uint32_t)tok->vocab_size); o += 4;
    put_i32(buf + o, tok->unk_id); o += 4;
    put_i32(buf + o, tok->bos_id); o += 4;
    put_i32(buf + o, tok->eos_id); o += 4;
    put_i32(buf + o, tok->pad_id); o += 4;
    put_u32(buf + o, tok->byte_fallback ? 1u : 0u); o += 4;
    for (int i = 0; i < tok->vocab_size; i++) {
        put_u32(buf + o, (uint32_t)tok->piece_lens[i]); o += 4;
        memcpy(buf + o, tok->pieces[i], (size_t)tok->piece_lens[i]);
        o += (size_t)tok->piece_lens[i];
        put_f32(buf + o, tok->scores[i]); o += 4;
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        tokenizer_set_error("save: cannot open %s", path);
        free(buf);
        return -1;
    }
    size_t w = fwrite(buf, 1, o, fp);
    fclose(fp);
    free(buf);
    if (w != o) {
        tokenizer_set_error("save: short write");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* binary load                                                         */
/* ------------------------------------------------------------------ */
tokenizer_t *tokenizer_load_binary_mem(const void *data, size_t size) {
    const unsigned char *p = (const unsigned char *)data;
    if (!p || size < 32) {
        tokenizer_set_error("load: truncated model");
        return NULL;
    }
    if (memcmp(p, "TKCZ", 4) != 0) {
        tokenizer_set_error("load: bad magic (not a .tkm model)");
        return NULL;
    }
    uint32_t version = get_u32(p + 4);
    if (version != 1) {
        tokenizer_set_error("load: unsupported version %u", (unsigned)version);
        return NULL;
    }
    uint32_t vocab = get_u32(p + 8);
    int unk = get_i32(p + 12);
    int bos = get_i32(p + 16);
    int eos = get_i32(p + 20);
    int pad = get_i32(p + 24);
    uint32_t flags = get_u32(p + 28);
    int byte_fb = (int)(flags & 1u);

    if (vocab > 100000000) {
        tokenizer_set_error("load: implausible vocab size %u", (unsigned)vocab);
        return NULL;
    }
    tokenizer_t *tok = tokenizer_create(unk, bos, eos, pad, byte_fb);
    if (!tok) {
        tokenizer_set_error("load: OOM");
        return NULL;
    }
    size_t o = 32;
    for (uint32_t i = 0; i < vocab; i++) {
        if (o + 8 > size) {
            tokenizer_set_error("load: truncated vocab entry");
            tokenizer_free(tok);
            return NULL;
        }
        uint32_t plen = get_u32(p + o);
        o += 4;
        if (o + plen + 4 > size) {
            tokenizer_set_error("load: truncated piece bytes");
            tokenizer_free(tok);
            return NULL;
        }
        float score = get_f32(p + o + plen);
        if (tokenizer_set_piece(tok, (int)i, (const char *)(p + o), plen, score) != 0) {
            tokenizer_set_error("load: OOM");
            tokenizer_free(tok);
            return NULL;
        }
        o += plen + 4;
    }
    return tok;
}

tokenizer_t *tokenizer_load_binary_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tokenizer_set_error("load: cannot open %s", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        tokenizer_set_error("load: empty model file");
        return NULL;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        tokenizer_set_error("load: OOM");
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) {
        free(buf);
        tokenizer_set_error("load: short read");
        return NULL;
    }
    tokenizer_t *tok = tokenizer_load_binary_mem(buf, (size_t)sz);
    free(buf);
    return tok;
}

/* JSON / SentencePiece / HuggingFace exports live in sentencepiece.c. */
