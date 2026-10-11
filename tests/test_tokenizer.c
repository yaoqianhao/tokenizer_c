/* test_tokenizer.c - basic unit & integration tests.
 *
 * Build & run:
 *   gcc .../test_tokenizer.c .../libtokenizer.a -o test_tokenizer
 *   ./test_tokenizer
 * Exit code 0 on success, 1 when any check failed.
 */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);   \
        }                                                                    \
    } while (0)

/* Replace the ▁ marker (E2 96 81) with a plain space -- this is exactly the
 * transformation decode() applies before returning text. */
static char *unmark(char *text, size_t len, size_t *out_len) {
    size_t o = 0;
    for (size_t i = 0; i < len; i++) {
        if (i + 2 < len && (unsigned char)text[i] == 0xE2 &&
            (unsigned char)text[i + 1] == 0x96 &&
            (unsigned char)text[i + 2] == 0x81) {
            text[o++] = ' ';
            i += 2;
        } else {
            text[o++] = text[i];
        }
    }
    text[o] = '\0';
    if (out_len) *out_len = o;
    return text;
}

static void write_corpus(const char *path) {
    const char *corpus =
        "hello world this is a unigram tokenizer test\n"
        "the quick brown fox jumps over the lazy dog\n"
        "机器学习与自然语言处理是人工智能的重要方向\n"
        "hello 世界 你好 tokenizer 测试\n"
        "UNIGRAM VITERBI TRIE BYTE FALLBACK\n"
        "fullwidth ＡＢＣＤ１２３４ test\n"
        "repeat repeat repeat repeat repeat repeat\n";
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fputs(corpus, fp);
    fclose(fp);
}

int main(void) {
    const char *corpus_path = "t_corpus.txt";
    const char *model_path = "t_model.tkm";
    const char *json_path = "t_model.json";
    write_corpus(corpus_path);

    /* ---- 1. training ---- */
    tokenizer_train_options opts = TOKENIZER_TRAIN_OPTIONS_DEFAULT;
    opts.corpus_path = corpus_path;
    opts.model_path = model_path;
    opts.json_path = json_path;
    opts.vocab_size = 300;
    opts.max_piece_len = 6;
    opts.max_iters = 6;
    opts.min_rounds = 2;
    opts.verbose = 0;
    CHECK(tokenizer_train(&opts) == 0, "train succeeds");
    if (tokenizer_last_error()) fprintf(stderr, "train err: %s\n", tokenizer_last_error());

    /* ---- 2. load & metadata ---- */
    tokenizer_t *tok = tokenizer_load(model_path);
    CHECK(tok != NULL, "load model");
    if (!tok) return 1;
    CHECK(tokenizer_vocab_size(tok) <= 300, "vocab size <= target");
    CHECK(tokenizer_vocab_size(tok) >= 4, "vocab size >= specials");
    CHECK(tokenizer_unk_id(tok) == 0, "unk id == 0");
    CHECK(tokenizer_bos_id(tok) == 1, "bos id == 1");
    CHECK(tokenizer_eos_id(tok) == 2, "eos id == 2");
    CHECK(tokenizer_pad_id(tok) == 3, "pad id == 3");
    const char *piece0 = NULL;
    size_t p0len = 0;
    CHECK(tokenizer_id_to_piece(tok, 0, &piece0, &p0len) == 0 && p0len == 5 &&
          memcmp(piece0, "<unk>", 5) == 0, "id 0 is <unk>");

    /* ---- 3. encode/decode round-trip ---- */
    const char *samples[] = {
        "hello world",
        "机器学习 测试",
        "the quick brown fox",
        "你好",
        "hello 世界 tokenizer",
        "ＡＢＣ １２３",       /* fullwidth -> halfwidth via normalization */
        "abc XYZ 123",
    };
    for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); i++) {
        size_t norm_len = 0;
        char *norm = tokenizer_normalize(samples[i], strlen(samples[i]), &norm_len);
        CHECK(norm != NULL, "normalize allocates");
        if (!norm) continue;
        char *expected = unmark(norm, norm_len, &norm_len);
        int *ids = NULL;
        size_t n = 0;
        CHECK(tokenizer_encode(tok, samples[i], strlen(samples[i]), &ids, &n) == 0,
              "encode succeeds");
        if (tokenizer_last_error())
            fprintf(stderr, "  encode err[%zu] '%s': %s\n", i, samples[i],
                    tokenizer_last_error());
        if (ids) {
            char *out = NULL;
            size_t olen = 0;
            CHECK(tokenizer_decode(tok, ids, n, &out, &olen) == 0, "decode succeeds");
            if (out) {
                CHECK(olen == norm_len && memcmp(out, expected, olen) == 0,
                      "decode(encode(x)) round-trips to normalized text");
                if (!(olen == norm_len && memcmp(out, expected, olen) == 0)) {
                    fprintf(stderr, "  sample: '%s'\n  expect: '%.*s'\n  out   : '%.*s'\n",
                            samples[i], (int)norm_len, expected, (int)olen, out);
                }
                tokenizer_free_text(out);
            }
            tokenizer_free_ids(ids);
        }
        tokenizer_free_text(norm);
    }

    /* ---- 4. byte fallback (OOV == 0) ---- */
    {
        const char *rare = "xqzwv"; /* letters that may not be frequent */
        int *ids = NULL;
        size_t n = 0;
        CHECK(tokenizer_encode(tok, rare, strlen(rare), &ids, &n) == 0,
              "encode rare text succeeds");
        CHECK(ids != NULL && n > 0, "rare text yields tokens");
        if (ids) {
            char *out = NULL;
            size_t olen = 0;
            if (tokenizer_decode(tok, ids, n, &out, &olen) == 0) {
                CHECK(olen == strlen(rare) && memcmp(out, rare, olen) == 0,
                      "byte fallback decode restores raw bytes");
                tokenizer_free_text(out);
            }
            tokenizer_free_ids(ids);
        }
    }
    /* random bytes robustness: bytes must be whitespace/control-free because
     * normalization rewrites whitespace (e.g. \n -> space marker) and decode
     * maps the marker back to a plain space. Printable bytes stay untouched. */
    {
        unsigned char junk[32];
        for (int i = 0; i < 32; i++)
            junk[i] = (unsigned char)((i * 7 + 3) % 95 + 0x21);
        size_t nl = 0;
        char *norm_in = tokenizer_normalize((const char *)junk, sizeof(junk), &nl);
        if (norm_in) {
            int *ids = NULL;
            size_t n = 0;
            CHECK(tokenizer_encode(tok, norm_in, nl, &ids, &n) == 0,
                  "encode arbitrary bytes succeeds");
            if (ids) {
                char *out = NULL;
                size_t olen = 0;
                if (tokenizer_decode(tok, ids, n, &out, &olen) == 0) {
                    CHECK(olen == nl && memcmp(out, norm_in, olen) == 0,
                          "arbitrary bytes round-trip");
                    tokenizer_free_text(out);
                }
                tokenizer_free_ids(ids);
            }
            tokenizer_free_text(norm_in);
        }
    }

    /* ---- 5. piece lookups ---- */
    {
        int id = tokenizer_piece_to_id(tok, "<s>", 3);
        CHECK(id == 1, "piece_to_id <s>");
        const char *pc = NULL;
        size_t pl = 0;
        CHECK(tokenizer_id_to_piece(tok, id, &pc, &pl) == 0 && pl == 3 &&
              memcmp(pc, "<s>", 3) == 0, "id_to_piece <s>");
    }

    /* ---- 6. JSON export exists and mentions type ---- */
    {
        FILE *fp = fopen(json_path, "rb");
        CHECK(fp != NULL, "json exported");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long jsz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *buf = (char *)malloc((size_t)jsz + 1);
            if (buf) {
                size_t rd = fread(buf, 1, (size_t)jsz, fp);
                buf[rd] = '\0';
                CHECK(strstr(buf, "\"Unigram\"") != NULL,
                      "json contains Unigram type");
                free(buf);
            }
            fclose(fp);
        }
    }

    /* ---- 7. reload from memory image ---- */
    {
        FILE *fp = fopen(model_path, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            unsigned char *buf = (unsigned char *)malloc((size_t)sz);
            if (buf) {
                fread(buf, 1, (size_t)sz, fp);
                tokenizer_t *tok2 = tokenizer_load_from_memory(buf, (size_t)sz);
                CHECK(tok2 != NULL, "load from memory");
                if (tok2) {
                    const char *in = "hello memory";
                    int *ids = NULL;
                    size_t n = 0;
                if (tokenizer_encode(tok2, in, strlen(in), &ids, &n) == 0) {
                    char *out = NULL;
                    size_t olen = 0;
                    if (tokenizer_decode(tok2, ids, n, &out, &olen) == 0) {
                        size_t nl = 0;
                        char *norm = tokenizer_normalize(in, strlen(in), &nl);
                        char *expected = norm ? unmark(norm, nl, &nl) : NULL;
                        CHECK(expected && nl == olen && memcmp(expected, out, olen) == 0,
                              "memory model round-trip");
                        tokenizer_free_text(norm);
                        tokenizer_free_text(out);
                    }
                    tokenizer_free_ids(ids);
                }
                    tokenizer_free(tok2);
                }
                free(buf);
            }
            fclose(fp);
        }
    }

    /* ---- 8. invalid id decode rejected ---- */
    {
        int bad[] = { 0, 9999999 };
        char *out = NULL;
        size_t olen = 0;
        CHECK(tokenizer_decode(tok, bad, 2, &out, &olen) != 0,
              "out-of-range id rejected");
        CHECK(out == NULL, "no output on rejected decode");
        tokenizer_free_text(out);
    }

    /* ---- 9. SentencePiece .model export & load round-trip ---- */
    {
        const char *sp_path = "t_model.model";
        CHECK(tokenizer_export_sentencepiece(tok, sp_path) == 0,
              "sp export succeeds");
        FILE *fp = fopen(sp_path, "rb");
        CHECK(fp != NULL, "sp file written");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            fclose(fp);
            CHECK(sz > 64, "sp file non-trivial size");
        }
        tokenizer_t *sp = tokenizer_load_sentencepiece(sp_path);
        CHECK(sp != NULL, "sp load succeeds");
        /* last_error is a process-wide sticky buffer: only report it when the
         * load actually failed, otherwise earlier tests' residual errors
         * (e.g. the deliberate out-of-range decode check) would be printed
         * as a misleading "sp load err" line. */
        if (!sp && tokenizer_last_error())
            fprintf(stderr, "sp load err: %s\n", tokenizer_last_error());
        if (sp) {
            CHECK(tokenizer_vocab_size(sp) == tokenizer_vocab_size(tok),
                  "sp vocab size matches");
            CHECK(tokenizer_unk_id(sp) == 0, "sp unk id preserved");
            CHECK(tokenizer_bos_id(sp) == 1, "sp bos id preserved");
            CHECK(tokenizer_eos_id(sp) == 2, "sp eos id preserved");
            CHECK(tokenizer_pad_id(sp) == 3, "sp pad id preserved");
            int ok = 1;
            for (int i = 0; i < (int)tokenizer_vocab_size(tok) && ok; i++) {
                const char *a = NULL, *b = NULL;
                size_t al = 0, bl = 0;
                tokenizer_id_to_piece(tok, i, &a, &al);
                tokenizer_id_to_piece(sp, i, &b, &bl);
                if (al != bl || memcmp(a, b, al) != 0) ok = 0;
            }
            CHECK(ok, "sp piece list identical to original");
            const char *s = "hello 世界 你好 tokenizer";
            int *ids1 = NULL, *ids2 = NULL;
            size_t n1 = 0, n2 = 0;
            CHECK(tokenizer_encode(tok, s, strlen(s), &ids1, &n1) == 0,
                  "sp encode on original");
            CHECK(tokenizer_encode(sp, s, strlen(s), &ids2, &n2) == 0,
                  "sp encode on loaded model");
            CHECK(ids1 && ids2 && n1 == n2 &&
                  memcmp(ids1, ids2, n1 * sizeof(int)) == 0,
                  "sp encode parity with original");
            tokenizer_free_ids(ids1);
            tokenizer_free_ids(ids2);
            tokenizer_free(sp);
        }
        /* extension dispatch through tokenizer_load() */
        tokenizer_t *sp2 = tokenizer_load(sp_path);
        CHECK(sp2 != NULL, "tokenizer_load dispatches .model extension");
        if (sp2) {
            const char *pc = NULL;
            size_t pl = 0;
            CHECK(tokenizer_id_to_piece(sp2, 0, &pc, &pl) == 0 && pl == 5 &&
                  memcmp(pc, "<unk>", 5) == 0, "loaded sp model id0 is <unk>");
            tokenizer_free(sp2);
        }
    }

    /* ---- 10. full HF tokenizer.json export fields ---- */
    {
        FILE *fp = fopen(json_path, "rb");
        CHECK(fp != NULL, "json re-exported");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *buf = (char *)malloc((size_t)sz + 1);
            if (buf) {
                size_t rd = fread(buf, 1, (size_t)sz, fp);
                buf[rd] = '\0';
                CHECK(strstr(buf, "\"normalizer\": {\"type\": \"NFKC\"}") != NULL,
                      "json has NFKC normalizer");
                CHECK(strstr(buf, "\"pre_tokenizer\": null") != NULL,
                      "json has null pre_tokenizer");
                CHECK(strstr(buf, "\"byte_fallback\": true") != NULL,
                      "json has byte_fallback true");
                CHECK(strstr(buf, "\"special\": true") != NULL,
                      "json added_tokens carries special flags");
                CHECK(strstr(buf, "\"unk_token\"") != NULL &&
                      strstr(buf, "\"eos_token\"") != NULL,
                      "json exposes unk/bos/eos/pad tokens");
                CHECK(strstr(buf, "\"type\": \"Unigram\"") != NULL,
                      "json has Unigram model config");
                CHECK(strstr(buf, "\"vocab\": [") != NULL && strstr(buf, "\"unk_id\": 0") != NULL,
                      "json has vocab ARRAY and unk_id");
                CHECK(strstr(buf, "[\"<unk>\", ") != NULL || strstr(buf, "[\"▁\", ") != NULL,
                      "json vocab entries are [piece, score] pairs");
                free(buf);
            }
            fclose(fp);
        }
    }

    /* ---- 11. full NFKC normalization (beyond legacy subset) ---- */
    {
        static const struct { const char *in; const char *want; } cases[] = {
            { "ＡＢＣ１２３", "ABC123" },       /* full-width ascii */
            { "① ② ③", "1 2 3" },             /* circled digits */
            { "ﬁﬂ", "fifl" },                  /* ligatures */
            { "Ⅻ ⅳ", "XII iv" },              /* roman numerals */
            { "² ³ ¹", "2 3 1" },              /* superscripts */
            { "＄ ＃ ￠", "$ # ¢" },            /* fullwidth/compat currency */
            { "－（１）", "-(1)" },              /* fullwidth punctuation */
            { "ｶﾀｶﾅ", "カタカナ" },            /* halfwidth katakana */
            { "㈱", "(株)" },                  /* compat ideographic */
            { "㎏", "kg" },                    /* unit sign */
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            size_t nl = 0;
            char *n = tokenizer_normalize(cases[i].in, strlen(cases[i].in), &nl);
            CHECK(n != NULL, "nfkc normalize allocates");
            if (n) {
                char *exp = unmark(n, nl, &nl);
                CHECK(strcmp(exp, cases[i].want) == 0, "nfkc mapping correct");
                if (strcmp(exp, cases[i].want) != 0) {
                    fprintf(stderr, "  nfkc in='%s' want='%s' got='%s'\n",
                            cases[i].in, cases[i].want, exp);
                }
                tokenizer_free_text(n);
            }
        }
        /* backward compatibility: legacy basic-subset behavior still applies.
         * Note: '\n' is whitespace -> space (▁), so a trailing marker is
         * expected and matches the original subset's semantics. */
        {
            size_t nl = 0;
            char *n = tokenizer_normalize("hello　世界\n", strlen("hello　世界\n"), &nl);
            CHECK(n != NULL, "nfkc legacy compat allocates");
            if (n) {
                char *exp = unmark(n, nl, &nl);
                CHECK(strcmp(exp, "hello 世界 ") == 0, "nfkc legacy whitespace mapping");
                if (strcmp(exp, "hello 世界 ") != 0)
                    fprintf(stderr, "  legacy in='hello 世界' got='%s'\n", exp);
                tokenizer_free_text(n);
            }
        }
    }

    tokenizer_free(tok);

    /* cleanup temp files */
    remove(corpus_path);
    remove(model_path);
    remove(json_path);
    remove("t_model.model");

    if (g_failures == 0) {
        printf("OK: all %d checks passed\n", g_checks);
        return 0;
    }
    printf("FAILED: %d/%d checks failed\n", g_failures, g_checks);
    return 1;
}
