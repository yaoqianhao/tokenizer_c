/* sentencepiece.c - SentencePiece ModelProto (.model) serialization &
 * parsing plus full HuggingFace tokenizer.json export.
 *
 * The .model writer emits the wire format of
 *   message ModelProto {
 *     repeated SentencePiece piece = 1;
 *     optional TrainerSpec trainer_spec = 2;
 *     optional NormalizerSpec normalizer_spec = 3;
 *     optional bool denormalizer_exists = 6;
 *     optional string decoder = 7;
 *   }
 * with UNIGRAM model_type, pieces carrying piece/score/type, and a
 * minimal-but-valid TrainerSpec / NormalizerSpec, so transformers / vLLM
 * SentencePieceProcessor can load the file directly.
 *
 * The loader accepts any ModelProto produced by SentencePiece itself
 * (BPE/Unigram/word/char models): pieces are adopted in file order and the
 * special ids are resolved from trainer_spec when present, falling back to
 * type==UNKNOWN / CONTROL scanning.
 */
#include "tokenizer_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* protobuf wire helpers                                               */
/* ------------------------------------------------------------------ */
static void pb_u64(unsigned char *buf, size_t cap, size_t *o, uint64_t v) {
    while (v >= 0x80) {
        if (*o >= cap) return;
        buf[(*o)++] = (unsigned char)(v | 0x80);
        v >>= 7;
    }
    if (*o >= cap) return;
    buf[(*o)++] = (unsigned char)v;
}

static void pb_tag(unsigned char *buf, size_t cap, size_t *o,
                   int field, int wire) {
    pb_u64(buf, cap, o, ((uint64_t)field << 3) | (uint64_t)wire);
}

static void pb_str(unsigned char *buf, size_t cap, size_t *o,
                   int field, const char *s, size_t len) {
    pb_tag(buf, cap, o, field, 2);
    pb_u64(buf, cap, o, len);
    if (*o + len > cap) return;
    memcpy(buf + *o, s, len);
    *o += len;
}

static void pb_varint(unsigned char *buf, size_t cap, size_t *o,
                      int field, uint64_t v) {
    pb_tag(buf, cap, o, field, 0);
    pb_u64(buf, cap, o, v);
}

static void pb_fixed32(unsigned char *buf, size_t cap, size_t *o,
                       int field, float f) {
    pb_tag(buf, cap, o, field, 5);
    if (*o + 4 > cap) return;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    for (int k = 0; k < 4; k++) buf[(*o)++] = (unsigned char)(bits >> (8 * k));
}

/* ------------------------------------------------------------------ */
/* .model export                                                       */
/* ------------------------------------------------------------------ */
enum {
    SP_NORMAL = 1, SP_UNKNOWN = 2, SP_CONTROL = 3,
    SP_USER_DEFINED = 4, SP_BYTE = 5, SP_UNUSED = 6
};

static int is_byte_piece(const char *p, int len) {
    if (len != 6 || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>')
        return 0;
    return isxdigit((unsigned char)p[3]) && isxdigit((unsigned char)p[4]);
}

int tokenizer_export_sentencepiece(const tokenizer_t *tok, const char *path) {
    if (!tok || !path) return -1;
    /* rough size: pieces (overhead 16 + piece bytes + 8) * N + specs */
    size_t cap = 4096;
    for (int i = 0; i < tok->vocab_size; i++) {
        cap += 24 + (size_t)tok->piece_lens[i];
    }
    cap += 1024;
    unsigned char *buf = (unsigned char *)malloc(cap);
    if (!buf) return -1;
    size_t o = 0;

    for (int i = 0; i < tok->vocab_size; i++) {
        /* SentencePiece message */
        size_t msg_start = o;
        /* placeholder tag+len: 1 byte tag + 1 byte length varint. Reserve
         * exactly 2 so the body starts right after the patched length. */
        if (o + 2 > cap) { free(buf); return -1; }
        o += 2;
        size_t body = o;
        pb_str(buf, cap, &o, 1, tok->pieces[i], (size_t)tok->piece_lens[i]);
        pb_fixed32(buf, cap, &o, 2, tok->scores[i]);
        int type = SP_NORMAL;
        if (i == tok->unk_id) type = SP_UNKNOWN;
        else if (i == tok->bos_id || i == tok->eos_id || i == tok->pad_id)
            type = SP_CONTROL;
        else if (is_byte_piece(tok->pieces[i], tok->piece_lens[i]))
            type = SP_BYTE;
        pb_varint(buf, cap, &o, 3, (uint64_t)type);
        size_t len = o - body;
        /* patch length: write the tag via a throwaway cursor so msg_start is
         * NOT advanced (pb_u64 advances whatever pointer it gets) */
        size_t tag_pos = msg_start;
        pb_u64(buf, cap, &tag_pos, ((uint64_t)1 << 3) | 2u);
        /* we wrote 1 tag byte + varint length byte(s): re-encode by shifting
         * body right when the length needs a second byte. Simpler: rebuild
         * this entry in place by moving body forward. */
        size_t varint_len = 0;
        {
            uint64_t v = len;
            do { varint_len++; v >>= 7; } while (v);
        }
        if (varint_len == 1) {
            /* overwrite the placeholder length byte */
            buf[msg_start + 1] = (unsigned char)len;
        } else {
            /* shift body by (varint_len - 1) bytes to make room */
            memmove(buf + msg_start + 1 + varint_len, buf + body, len);
            size_t lpos = msg_start + 1;
            uint64_t v = len;
            while (v >= 0x80) {
                buf[lpos++] = (unsigned char)(v | 0x80);
                v >>= 7;
            }
            buf[lpos] = (unsigned char)v;
            o += (varint_len - 1);
        }
    }

    /* TrainerSpec */
    {
        size_t spec_start = o;
        if (o + 2 > cap) { free(buf); return -1; }
        o += 2;
        size_t body = o;
        pb_varint(buf, cap, &o, 3, 1);      /* model_type = UNIGRAM */
        pb_varint(buf, cap, &o, 4, (uint64_t)tok->vocab_size);
        pb_fixed32(buf, cap, &o, 10, 0.9995f); /* character_coverage */
        pb_varint(buf, cap, &o, 35, tok->byte_fallback ? 1u : 0u);
        pb_varint(buf, cap, &o, 40, (uint64_t)tok->unk_id);
        pb_varint(buf, cap, &o, 41, (uint64_t)tok->bos_id);
        pb_varint(buf, cap, &o, 42, (uint64_t)tok->eos_id);
        pb_varint(buf, cap, &o, 43, (uint64_t)tok->pad_id);
        const char *unk_p = (tok->unk_id >= 0 && tok->unk_id < tok->vocab_size)
            ? tok->pieces[tok->unk_id] : "<unk>";
        const char *bos_p = (tok->bos_id >= 0 && tok->bos_id < tok->vocab_size)
            ? tok->pieces[tok->bos_id] : "<s>";
        const char *eos_p = (tok->eos_id >= 0 && tok->eos_id < tok->vocab_size)
            ? tok->pieces[tok->eos_id] : "</s>";
        const char *pad_p = (tok->pad_id >= 0 && tok->pad_id < tok->vocab_size)
            ? tok->pieces[tok->pad_id] : "<pad>";
        pb_str(buf, cap, &o, 45, unk_p, strlen(unk_p));
        pb_str(buf, cap, &o, 46, bos_p, strlen(bos_p));
        pb_str(buf, cap, &o, 47, eos_p, strlen(eos_p));
        pb_str(buf, cap, &o, 48, pad_p, strlen(pad_p));
        size_t len = o - body;
        size_t tag_pos = spec_start;
        pb_u64(buf, cap, &tag_pos, ((uint64_t)2 << 3) | 2u);
        size_t varint_len = 0;
        { uint64_t v = len; do { varint_len++; v >>= 7; } while (v); }
        if (varint_len == 1) {
            buf[spec_start + 1] = (unsigned char)len;
        } else {
            memmove(buf + spec_start + 1 + varint_len, buf + body, len);
            size_t lpos = spec_start + 1;
            uint64_t v = len;
            while (v >= 0x80) { buf[lpos++] = (unsigned char)(v | 0x80); v >>= 7; }
            buf[lpos] = (unsigned char)v;
            o += (varint_len - 1);
        }
    }

    /* NormalizerSpec */
    {
        size_t spec_start = o;
        if (o + 2 > cap) { free(buf); return -1; }
        o += 2;
        size_t body = o;
        pb_str(buf, cap, &o, 1, "identity", 8);
        pb_varint(buf, cap, &o, 3, 0); /* add_dummy_prefix = false */
        pb_varint(buf, cap, &o, 4, 0); /* remove_extra_whitespaces = false */
        pb_varint(buf, cap, &o, 5, 1); /* escape_whitespaces = true */
        size_t len = o - body;
        size_t tag_pos = spec_start;
        pb_u64(buf, cap, &tag_pos, ((uint64_t)3 << 3) | 2u);
        size_t varint_len = 0;
        { uint64_t v = len; do { varint_len++; v >>= 7; } while (v); }
        if (varint_len == 1) {
            buf[spec_start + 1] = (unsigned char)len;
        } else {
            memmove(buf + spec_start + 1 + varint_len, buf + body, len);
            size_t lpos = spec_start + 1;
            uint64_t v = len;
            while (v >= 0x80) { buf[lpos++] = (unsigned char)(v | 0x80); v >>= 7; }
            buf[lpos] = (unsigned char)v;
            o += (varint_len - 1);
        }
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        tokenizer_set_error("sp: cannot open %s", path);
        free(buf);
        return -1;
    }
    size_t w = fwrite(buf, 1, o, fp);
    fclose(fp);
    free(buf);
    if (w != o) {
        tokenizer_set_error("sp: short write");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* .model load                                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    char **pieces;
    float *scores;
    int *types;
    int count, cap;
} sp_piece_list;

static void sp_list_free(sp_piece_list *l) {
    for (int i = 0; i < l->count; i++) free(l->pieces[i]);
    free(l->pieces);
    free(l->scores);
    free(l->types);
    memset(l, 0, sizeof(*l));
}

static int sp_list_push(sp_piece_list *l, const char *p, size_t plen,
                        float score, int type) {
    if (l->count >= l->cap) {
        int ncap = l->cap ? l->cap * 2 : 64;
        char **np = (char **)realloc(l->pieces, (size_t)ncap * sizeof(char *));
        if (!np) return -1;
        l->pieces = np;
        float *ns = (float *)realloc(l->scores, (size_t)ncap * sizeof(float));
        if (!ns) return -1;
        l->scores = ns;
        int *nt = (int *)realloc(l->types, (size_t)ncap * sizeof(int));
        if (!nt) return -1;
        l->types = nt;
        l->cap = ncap;
    }
    char *copy = (char *)malloc(plen + 1);
    if (!copy) return -1;
    memcpy(copy, p, plen);
    copy[plen] = '\0';
    l->pieces[l->count] = copy;
    l->scores[l->count] = score;
    l->types[l->count] = type;
    l->count++;
    return 0;
}

/* parse one varint field value from buf at *pos (advances) */
static int pb_parse_u64(const unsigned char *buf, size_t size, size_t *pos,
                        uint64_t *out) {
    uint64_t v = 0;
    int shift = 0;
    while (*pos < size && shift < 64) {
        unsigned char b = buf[(*pos)++];
        v |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) { *out = v; return 0; }
        shift += 7;
    }
    return -1;
}

static int pb_parse_fixed32(const unsigned char *buf, size_t size, size_t *pos,
                            float *out) {
    if (*pos + 4 > size) return -1;
    uint32_t bits = (uint32_t)buf[*pos] | ((uint32_t)buf[*pos + 1] << 8) |
                    ((uint32_t)buf[*pos + 2] << 16) |
                    ((uint32_t)buf[*pos + 3] << 24);
    *pos += 4;
    memcpy(out, &bits, 4);
    return 0;
}

static int pb_parse_str(const unsigned char *buf, size_t size, size_t *pos,
                        const char **out, size_t *olen) {
    uint64_t len = 0;
    if (pb_parse_u64(buf, size, pos, &len) != 0) return -1;
    if (len > (uint64_t)(size - *pos)) return -1;
    *out = (const char *)(buf + *pos);
    *olen = (size_t)len;
    *pos += (size_t)len;
    return 0;
}

/* parse a SentencePiece submessage; returns -1 on malformed */
static int sp_parse_piece(const unsigned char *buf, size_t size, size_t *pos,
                          sp_piece_list *l) {
    uint64_t plen = 0;
    if (pb_parse_u64(buf, size, pos, &plen) != 0) return -1;
    size_t end = *pos + (size_t)plen;
    if (end > size) return -1;
    const char *piece = NULL;
    size_t piece_len = 0;
    float score = 0.0f;
    int type = SP_NORMAL;
    int got = 0;
    while (*pos < end) {
        uint64_t tag = 0;
        if (pb_parse_u64(buf, size, pos, &tag) != 0) return -1;
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);
        if (field == 1 && wire == 2) {
            if (pb_parse_str(buf, size, pos, &piece, &piece_len) != 0) return -1;
            got |= 1;
        } else if (field == 2 && wire == 5) {
            if (pb_parse_fixed32(buf, size, pos, &score) != 0) return -1;
            got |= 2;
        } else if (field == 3 && wire == 0) {
            uint64_t v = 0;
            if (pb_parse_u64(buf, size, pos, &v) != 0) return -1;
            type = (int)v;
            got |= 4;
        } else {
            /* skip unknown field */
            if (wire == 0) {
                uint64_t v = 0;
                if (pb_parse_u64(buf, size, pos, &v) != 0) return -1;
            } else if (wire == 1) {
                if (*pos + 8 > size) return -1;
                *pos += 8;
            } else if (wire == 5) {
                if (*pos + 4 > size) return -1;
                *pos += 4;
            } else if (wire == 2) {
                uint64_t slen = 0;
                if (pb_parse_u64(buf, size, pos, &slen) != 0) return -1;
                if (slen > (uint64_t)(size - *pos)) return -1;
                *pos += (size_t)slen;
            } else {
                return -1;
            }
        }
    }
    if (!(got & 1)) return -1; /* piece string is mandatory */
    return sp_list_push(l, piece, piece_len, score, type);
}

/* trainer_spec field resolution */
typedef struct {
    int unk_id, bos_id, eos_id, pad_id;
    int has_unk, has_bos, has_eos, has_pad;
    int byte_fallback;
} sp_trainer_info;

static void sp_parse_trainer(const unsigned char *buf, size_t size, size_t *pos,
                             sp_trainer_info *info) {
    uint64_t slen = 0;
    if (pb_parse_u64(buf, size, pos, &slen) != 0) return;
    size_t end = *pos + (size_t)slen;
    if (end > size) return;
    while (*pos < end) {
        uint64_t tag = 0;
        if (pb_parse_u64(buf, size, pos, &tag) != 0) return;
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);
        uint64_t v = 0;
        switch (field) {
            case 3: /* model_type */
            case 4: /* vocab_size */
            case 40: case 41: case 42: case 43: /* ids */
            case 35: /* byte_fallback */
                if (wire != 0) return;
                if (pb_parse_u64(buf, size, pos, &v) != 0) return;
                if (field == 40) { info->unk_id = (int)v; info->has_unk = 1; }
                else if (field == 41) { info->bos_id = (int)v; info->has_bos = 1; }
                else if (field == 42) { info->eos_id = (int)v; info->has_eos = 1; }
                else if (field == 43) { info->pad_id = (int)v; info->has_pad = 1; }
                else if (field == 35) info->byte_fallback = v ? 1 : 0;
                break;
            default:
                if (wire == 0) { if (pb_parse_u64(buf, size, pos, &v) != 0) return; }
                else if (wire == 2) {
                    const char *dummy; size_t dl;
                    if (pb_parse_str(buf, size, pos, &dummy, &dl) != 0) return;
                } else if (wire == 5) {
                    float f; if (pb_parse_fixed32(buf, size, pos, &f) != 0) return;
                } else if (wire == 1) {
                    if (*pos + 8 > size) return;
                    *pos += 8;
                } else return;
                break;
        }
    }
}

tokenizer_t *tokenizer_load_sentencepiece(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tokenizer_set_error("sp: cannot open %s", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); tokenizer_set_error("sp: empty model file"); return NULL; }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) { fclose(fp); tokenizer_set_error("sp: OOM"); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) { free(buf); tokenizer_set_error("sp: short read"); return NULL; }

    sp_piece_list pieces = { 0 };
    sp_trainer_info trainer = { -1, -1, -1, -1, 0, 0, 0, 0, 0 };
    size_t pos = 0;
    while (pos < (size_t)sz) {
        uint64_t tag = 0;
        if (pb_parse_u64(buf, (size_t)sz, &pos, &tag) != 0) break;
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);
        if (field == 1 && wire == 2) {
            if (sp_parse_piece(buf, (size_t)sz, &pos, &pieces) != 0) {
                sp_list_free(&pieces);
                free(buf);
                tokenizer_set_error("sp: malformed piece");
                return NULL;
            }
        } else if (field == 2 && wire == 2) {
            sp_parse_trainer(buf, (size_t)sz, &pos, &trainer);
        } else if (wire == 2) {
            const char *dummy; size_t dl;
            if (pb_parse_str(buf, (size_t)sz, &pos, &dummy, &dl) != 0) break;
        } else if (wire == 0) {
            uint64_t v; if (pb_parse_u64(buf, (size_t)sz, &pos, &v) != 0) break;
        } else if (wire == 5) {
            float f; if (pb_parse_fixed32(buf, (size_t)sz, &pos, &f) != 0) break;
        } else if (wire == 1) {
            pos += 8;
        } else break;
    }
    free(buf);

    if (pieces.count == 0) {
        sp_list_free(&pieces);
        tokenizer_set_error("sp: no pieces found");
        return NULL;
    }

    /* resolve special ids */
    int vsz = pieces.count;
    int unk_id = -1, bos_id = -1, eos_id = -1, pad_id = -1;
    int byte_fallback = trainer.byte_fallback;
    for (int i = 0; i < vsz; i++) {
        if (pieces.types[i] == SP_BYTE) byte_fallback = 1;
    }
    /* unk */
    if (trainer.has_unk && trainer.unk_id >= 0 && trainer.unk_id < vsz)
        unk_id = trainer.unk_id;
    if (unk_id < 0) {
        for (int i = 0; i < vsz; i++)
            if (pieces.types[i] == SP_UNKNOWN) { unk_id = i; break; }
    }
    if (unk_id < 0) {
        for (int i = 0; i < vsz; i++)
            if (strcmp(pieces.pieces[i], "<unk>") == 0) { unk_id = i; break; }
    }
    if (unk_id < 0 && vsz > 0) unk_id = 0;
    /* control ids by trainer spec, else by name */
    if (trainer.has_bos && trainer.bos_id >= 0 && trainer.bos_id < vsz)
        bos_id = trainer.bos_id;
    if (bos_id < 0)
        for (int i = 0; i < vsz; i++)
            if (strcmp(pieces.pieces[i], "<s>") == 0) { bos_id = i; break; }
    if (trainer.has_eos && trainer.eos_id >= 0 && trainer.eos_id < vsz)
        eos_id = trainer.eos_id;
    if (eos_id < 0)
        for (int i = 0; i < vsz; i++)
            if (strcmp(pieces.pieces[i], "</s>") == 0) { eos_id = i; break; }
    if (trainer.has_pad && trainer.pad_id >= 0 && trainer.pad_id < vsz)
        pad_id = trainer.pad_id;
    if (pad_id < 0)
        for (int i = 0; i < vsz; i++)
            if (strcmp(pieces.pieces[i], "<pad>") == 0) { pad_id = i; break; }

    tokenizer_t *tok = tokenizer_create(unk_id, bos_id, eos_id, pad_id,
                                        byte_fallback);
    if (!tok) {
        sp_list_free(&pieces);
        tokenizer_set_error("sp: OOM");
        return NULL;
    }
    for (int i = 0; i < vsz; i++) {
        if (tokenizer_set_piece(tok, i, pieces.pieces[i],
                                strlen(pieces.pieces[i]), pieces.scores[i]) != 0) {
            sp_list_free(&pieces);
            tokenizer_free(tok);
            tokenizer_set_error("sp: OOM");
            return NULL;
        }
    }
    sp_list_free(&pieces);
    if (tokenizer_build_trie(tok) != 0) {
        tokenizer_free(tok);
        tokenizer_set_error("sp: trie build failed");
        return NULL;
    }
    return tok;
}

/* ------------------------------------------------------------------ */
/* HuggingFace tokenizer.json export (full format)                     */
/* ------------------------------------------------------------------ */
static void jstr(FILE *fp, const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            case 0x08: fputs("\\b", fp); break;
            case 0x0C: fputs("\\f", fp); break;
            default:
                if (c < 0x20) fprintf(fp, "\\u%04x", c);
                else fputc((int)c, fp);
        }
    }
}

static void jscore(FILE *fp, float s) {
    if (s != s || s == (float)1.0 / 0.0f || s == -(float)1.0 / 0.0f) {
        fputs("null", fp);
        return;
    }
    fprintf(fp, "%.6g", (double)s);
}

static void jspecial(FILE *fp, const tokenizer_t *tok, int id, const char *name) {
    if (id < 0 || id >= tok->vocab_size) {
        fprintf(fp, "\"%s_token\": null", name);
        return;
    }
    fprintf(fp, "\"%s_token\": {\"content\": \"", name);
    jstr(fp, tok->pieces[id], (size_t)tok->piece_lens[id]);
    fprintf(fp, "\", \"lstrip\": false, \"rstrip\": false, "
                "\"normalized\": false, \"single_word\": false, \"id\": %d}", id);
}

int tokenizer_export_json(const tokenizer_t *tok, const char *json_path) {
    if (!tok || !json_path) return -1;
    FILE *fp = fopen(json_path, "wb");
    if (!fp) {
        tokenizer_set_error("json: cannot open %s", json_path);
        return -1;
    }
    const int specials[4] = { tok->unk_id, tok->bos_id, tok->eos_id, tok->pad_id };
    int first_added = 1;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": \"1.0\",\n");
    fprintf(fp, "  \"truncation\": null,\n");
    fprintf(fp, "  \"padding\": null,\n");
    fprintf(fp, "  \"added_tokens\": [\n");
    for (int k = 0; k < 4; k++) {
        int id = specials[k];
        if (id < 0 || id >= tok->vocab_size) continue;
        fprintf(fp, "    %s{\"id\": %d, \"content\": \"", first_added ? "" : ",", id);
        jstr(fp, tok->pieces[id], (size_t)tok->piece_lens[id]);
        fprintf(fp, "\", \"lstrip\": false, \"rstrip\": false, "
                    "\"normalized\": false, \"single_word\": false, "
                    "\"special\": true}");
        first_added = 0;
    }
    fprintf(fp, "\n  ],\n");
    fprintf(fp, "  \"normalizer\": {\"type\": \"NFKC\"},\n");
    fprintf(fp, "  \"pre_tokenizer\": null,\n");
    fprintf(fp, "  \"post_processor\": null,\n");
    fprintf(fp, "  \"decoder\": {\"type\": \"Metaspace\", "
                "\"replacement\": \"\u2581\", \"add_prefix_space\": false},\n");
    fprintf(fp, "  \"model\": {\n");
    fprintf(fp, "    \"type\": \"Unigram\",\n");
    fprintf(fp, "    \"byte_fallback\": %s,\n", tok->byte_fallback ? "true" : "false");
    fprintf(fp, "    \"unk_id\": %d,\n", tok->unk_id);
    fprintf(fp, "    \"vocab\": {\n");
    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(fp, "      \"");
        jstr(fp, tok->pieces[i], (size_t)tok->piece_lens[i]);
        fprintf(fp, "\": ");
        jscore(fp, tok->scores[i]);
        fprintf(fp, "%s\n", (i + 1 < tok->vocab_size) ? "," : "");
    }
    fprintf(fp, "    }\n");
    fprintf(fp, "  },\n");
    jspecial(fp, tok, tok->unk_id, "unk");
    fprintf(fp, ",\n");
    jspecial(fp, tok, tok->bos_id, "bos");
    fprintf(fp, ",\n");
    jspecial(fp, tok, tok->eos_id, "eos");
    fprintf(fp, ",\n");
    jspecial(fp, tok, tok->pad_id, "pad");
    fprintf(fp, ",\n");
    fprintf(fp, "  \"tokenizer_class\": \"Unigram\"\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}
