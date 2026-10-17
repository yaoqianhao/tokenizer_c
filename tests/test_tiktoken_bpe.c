/* test_tiktoken_bpe.c - self contained unit tests for the tiktoken compatible
 * byte-level BPE module (include/tiktoken_bpe.h + src/tiktoken_bpe.c).
 *
 * The tests build two miniature vocabularies in memory (one .tiktoken, one
 * .tok.json) so that no external asset is needed, then check
 *   - vocabulary loading / introspection (format, vocab_size, word_split)
 *   - base64 and JSON parsing, including malformed input rejection
 *   - BPE merge order (rank decides, never frequency)
 *   - the approximate pre-tokenizer against hand computed Python regex output
 *   - special token handling (allow / reject / ordinary)
 *   - decoding with tiktoken errors="replace" semantics
 *   - error paths and the piece cache
 *
 * Build: see CMakeLists.txt (target: test_tiktoken_bpe).
 * Exit code 0 = all checks passed, 1 = at least one failure.
 */
#include "tiktoken_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;
static int g_case_failures = 0;

static void check_impl(int ok, const char *expr, int line) {
    g_checks++;
    if (!ok) {
        g_failures++;
        g_case_failures++;
        printf("  FAIL line %d: %s\n", line, expr);
    }
}

#define CHECK(cond) check_impl((cond) != 0, #cond, __LINE__)

static void begin_case(const char *name) {
    g_case_failures = 0;
    printf("[ case ] %s\n", name);
}

static void end_case(const char *name) {
    if (g_case_failures == 0) printf("  ok    %s\n", name);
    else printf("  FAIL  %s (%d failed checks)\n", name, g_case_failures);
}

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static tiktoken_bpe_t *load_memory(const char *text) {
    return ttbpe_load_from_memory(text, strlen(text));
}

static void print_ids(const int *ids, size_t n) {
    printf("    got ids:");
    for (size_t i = 0; i < n; i++) printf(" %d", ids[i]);
    if (n == 0) printf(" (empty)");
    printf("\n");
}

static int ids_equal(const int *a, size_t n, const int *b, size_t m) {
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static void check_encode(const tiktoken_bpe_t *tt, const char *text,
                         const int *want, size_t want_n) {
    int *ids = NULL;
    size_t n = 0;
    if (ttbpe_encode_single_piece(tt, text, strlen(text), &ids, &n) != 0) {
        printf("  FAIL encode_single_piece(\"%s\") returned -1: %s\n", text,
               ttbpe_last_error() ? ttbpe_last_error() : "(no error)");
        g_failures++;
        g_case_failures++;
        return;
    }
    if (!ids_equal(ids, n, want, want_n)) {
        printf("  FAIL encode_single_piece(\"%s\") mismatch, want", text);
        print_ids(want, want_n);
        print_ids(ids, n);
        g_failures++;
        g_case_failures++;
    } else {
        g_checks++;
    }
    ttbpe_free_ids(ids);
}

static void check_ordinary(const tiktoken_bpe_t *tt, const char *text,
                           const int *want, size_t want_n) {
    int *ids = NULL;
    size_t n = 0;
    if (ttbpe_encode_ordinary(tt, text, strlen(text), &ids, &n) != 0) {
        printf("  FAIL encode_ordinary(\"%s\") returned -1: %s\n", text,
               ttbpe_last_error() ? ttbpe_last_error() : "(no error)");
        g_failures++;
        g_case_failures++;
        return;
    }
    if (!ids_equal(ids, n, want, want_n)) {
        printf("  FAIL encode_ordinary(\"%s\") mismatch, want", text);
        print_ids(want, want_n);
        print_ids(ids, n);
        g_failures++;
        g_case_failures++;
    } else {
        g_checks++;
    }
    ttbpe_free_ids(ids);
}

/* Compare the pieces produced by the handle's current split mode. */
static void check_split(const tiktoken_bpe_t *tt, const char *text,
                        const char *const *want, size_t want_n) {
    ttbpe_piece *pieces = NULL;
    size_t n = 0;
    if (ttbpe_split(tt, text, strlen(text), &pieces, &n) != 0) {
        printf("  FAIL split(\"%s\") returned -1: %s\n", text,
               ttbpe_last_error() ? ttbpe_last_error() : "(no error)");
        g_failures++;
        g_case_failures++;
        return;
    }
    int ok = (n == want_n);
    if (ok) {
        for (size_t i = 0; i < n; i++) {
            if (strlen(want[i]) != pieces[i].len ||
                memcmp(want[i], pieces[i].ptr, pieces[i].len) != 0) {
                ok = 0;
                break;
            }
        }
    }
    if (!ok) {
        printf("  FAIL split(\"%s\") mismatch\n    want:", text);
        for (size_t i = 0; i < want_n; i++) printf(" [%s]", want[i]);
        printf("\n    got :");
        for (size_t i = 0; i < n; i++)
            printf(" [%.*s]", (int)pieces[i].len, pieces[i].ptr);
        printf("\n");
        g_failures++;
        g_case_failures++;
    } else {
        g_checks++;
    }
    ttbpe_free_pieces(pieces);
}

/* ------------------------------------------------------------------ */
/* miniature vocabularies                                             */
/* ------------------------------------------------------------------ */

/* .tiktoken style: one base64 line per token, line index = id. */
static const char *TIKTOKEN_VOCAB =
    "YQ==\n"      /*  0  "a"           */
    "Yg==\n"      /*  1  "b"           */
    "Yw==\n"      /*  2  "c"           */
    "ZA==\n"      /*  3  "d"           */
    "IA==\n"      /*  4  " "           */
    "Jw==\n"      /*  5  "'"           */
    "cw==\n"      /*  6  "s"           */
    "YmE=\n"      /*  7  "ba"          */
    "YWI=\n"      /*  8  "ab"          */
    "YWJh\n"      /*  9  "aba"         */
    "YmFi\n"      /* 10  "bab"         */
    "YWJhYg==\n"  /* 11  "abab"        */
    "Cg==\n"      /* 12  "\n"          */
    "YWJjZA==\n"  /* 13  "abcd"        */
    "YmM=\n"      /* 14  "bc"          */
    "YWJj\n"      /* 15  "abc"         */
    "/w==\n"      /* 16  "\xff"        */
    "ww==\n"      /* 17  "\xc3"        */
    "4oI=\n"      /* 18  "\xe2\x82"    */
    "w6k=\n";     /* 19  "\xc3\xa9"    */

/* .tok.json style vocabulary (xAI Grok layout) with one special token. */
static const char *JSON_VOCAB =
    "{"
    "\"special_tokens\":[{\"bytes\":[60,124,101,110,100,124,62],"
    "\"token\":100}],"
    "\"reserved_tokens\":0,"
    "\"regular_tokens\":["
    "{\"bytes\":[97],\"token\":0},"
    "{\"bytes\":[98],\"token\":1},"
    "{\"bytes\":[97,98],\"token\":2},"
    "{\"bytes\":[32],\"token\":3},"
    "{\"bytes\":[10],\"token\":4}],"
    "\"word_split\":\"V1\","
    "\"vocab_size\":101"
    "}";

/* "<|end|>" as a C string, used by the special token tests. */
#define SPECIAL_TOK "<|end|>"

/* ------------------------------------------------------------------ */
/* tests                                                              */
/* ------------------------------------------------------------------ */

static void test_tiktoken_format(void) {
    begin_case("tiktoken (base64) vocabulary");

    tiktoken_bpe_t *tt = load_memory(TIKTOKEN_VOCAB);
    CHECK(tt != NULL);
    if (!tt) {
        printf("  load error: %s\n", ttbpe_last_error());
        end_case("tiktoken (base64) vocabulary");
        return;
    }
    CHECK(ttbpe_vocab_size(tt) == 20);
    CHECK(ttbpe_format(tt) != NULL && strcmp(ttbpe_format(tt), "tiktoken") == 0);
    CHECK(ttbpe_special_count(tt) == 0);
    CHECK(ttbpe_word_split(tt) == NULL);
    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_NONE) == 0);

    /* rank decides, not frequency: "ba" (rank 7) beats "ab" (rank 8) even
     * though "ab" occurs twice in "ababa". */
    const int want_ababa[] = {9, 7}; /* "aba" + "ba" */
    check_encode(tt, "ababa", want_ababa, 2);

    /* a high ranked merge still happens once every other pair is gone */
    const int want_abc[] = {15};
    check_encode(tt, "abc", want_abc, 1);
    const int want_abcd[] = {13};
    check_encode(tt, "abcd", want_abcd, 1);
    const int want_bc[] = {14};
    check_encode(tt, "bc", want_bc, 1);

    /* every single byte exists in this vocabulary, so "abcd 's" only merges
     * through the ranks it knows: ab (8) -> abc (15) -> abcd (13); the space,
     * "'" and "s" stay single byte tokens because " ", " '" and "'s" are not
     * in the vocabulary at all. */
    const int want_bytes[] = {13, 4, 5, 6};
    check_encode(tt, "abcd 's", want_bytes, 4);

    /* decoding: bytes -> text */
    const int ids_abc[] = {15};
    char *text = NULL;
    size_t text_len = 0;
    CHECK(ttbpe_decode(tt, ids_abc, 1, &text, &text_len) == 0);
    CHECK(text != NULL && text_len == 3 && memcmp(text, "abc", 3) == 0);
    ttbpe_free_buf(text);

    /* unknown byte -> error, not a silent fallback */
    int *ids = NULL;
    size_t n = 0;
    CHECK(ttbpe_encode_single_piece(tt, "z", 1, &ids, &n) == -1);
    CHECK(ttbpe_last_error() != NULL);
    free(ids);

    /* empty input is a success with zero ids */
    ids = NULL;
    n = 0;
    CHECK(ttbpe_encode_single_piece(tt, "", 0, &ids, &n) == 0);
    CHECK(n == 0);
    ttbpe_free_ids(ids);

    /* out of range id and NULL arguments */
    CHECK(ttbpe_decode(tt, ids_abc, 1, &text, &text_len) == 0);
    ttbpe_free_buf(text);
    const int bad_id = 9999;
    CHECK(ttbpe_decode(tt, &bad_id, 1, &text, &text_len) == -1);
    CHECK(ttbpe_decode(tt, NULL, 0, &text, &text_len) == 0);
    ttbpe_free_buf(text);
    CHECK(ttbpe_encode(NULL, "a", 1, &ids, &n) == -1);
    CHECK(ttbpe_vocab_size(NULL) == 0);

    /* raw bytes are preserved for invalid UTF-8 */
    const int ids_invalid[] = {16, 0};
    unsigned char *raw = NULL;
    size_t raw_len = 0;
    CHECK(ttbpe_decode_bytes(tt, ids_invalid, 2, &raw, &raw_len) == 0);
    CHECK(raw_len == 2 && raw[0] == 0xff && raw[1] == 'a');
    ttbpe_free_buf(raw);

    ttbpe_free(tt);
    end_case("tiktoken (base64) vocabulary");
}

static void test_json_format(void) {
    begin_case("tok.json vocabulary + special tokens");

    tiktoken_bpe_t *tt = load_memory(JSON_VOCAB);
    CHECK(tt != NULL);
    if (!tt) {
        printf("  load error: %s\n", ttbpe_last_error());
        end_case("tok.json vocabulary + special tokens");
        return;
    }
    CHECK(ttbpe_vocab_size(tt) == 101);
    CHECK(ttbpe_format(tt) != NULL && strcmp(ttbpe_format(tt), "tok.json") == 0);
    CHECK(ttbpe_special_count(tt) == 1);
    CHECK(ttbpe_word_split(tt) != NULL && strcmp(ttbpe_word_split(tt), "V1") == 0);

    const unsigned char *bytes = NULL;
    size_t len = 0;
    CHECK(ttbpe_id_bytes(tt, 100, &bytes, &len) == 0);
    CHECK(len == 7 && memcmp(bytes, SPECIAL_TOK, 7) == 0);
    CHECK(ttbpe_id_bytes(tt, 2, &bytes, &len) == 0);
    CHECK(len == 2 && memcmp(bytes, "ab", 2) == 0);
    CHECK(ttbpe_id_bytes(tt, 50, &bytes, &len) == -1); /* id 50 was never defined */

    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_CL100K) == 0);

    int *ids = NULL;
    size_t n = 0;

    /* text without specials; "ab" is a single rank 2 token here */
    const int want_ab[] = {2};
    check_ordinary(tt, "ab", want_ab, 1);

    /* specials are plain text for encode_ordinary -> '<' has no rank -> error */
    CHECK(ttbpe_encode_ordinary(tt, "a" SPECIAL_TOK "b", 1 + 7 + 1, &ids, &n) == -1);
    CHECK(ttbpe_last_error() != NULL);

    /* default encode() rejects a special that is not allowed */
    CHECK(ttbpe_encode(tt, "a" SPECIAL_TOK "b", 1 + 7 + 1, &ids, &n) == -1);
    CHECK(ttbpe_last_error() != NULL);

    /* allowed_special = "all" emits the special id */
    CHECK(ttbpe_encode_ex(tt, "a" SPECIAL_TOK "b", 1 + 7 + 1, 1, 1, &ids, &n) == 0);
    const int want_special[] = {0, 100, 1};
    if (!ids_equal(ids, n, want_special, 3)) {
        printf("  FAIL special encode mismatch, want");
        print_ids(want_special, 3);
        print_ids(ids, n);
        g_failures++;
        g_case_failures++;
    } else {
        g_checks++;
    }
    ttbpe_free_ids(ids);

    /* a second, longer special is registered and matched greedily */
    CHECK(ttbpe_add_special(tt, SPECIAL_TOK "extra", 12) == 0);
    CHECK(ttbpe_vocab_size(tt) == 102);
    CHECK(ttbpe_special_count(tt) == 2);
    CHECK(ttbpe_encode_ex(tt, SPECIAL_TOK "extra", 12, 1, 1, &ids, &n) == 0);
    const int want_long[] = {101};
    if (!ids_equal(ids, n, want_long, 1)) {
        printf("  FAIL longest special mismatch, want");
        print_ids(want_long, 1);
        print_ids(ids, n);
        g_failures++;
        g_case_failures++;
    } else {
        g_checks++;
    }
    ttbpe_free_ids(ids);

    /* disallowed but allowed_special="all" is rejected when asked to */
    CHECK(ttbpe_encode_ex(tt, "a" SPECIAL_TOK "b", 9, 0, 0, &ids, &n) == -1);
    ttbpe_free_ids(ids);

    /* unknown byte */
    CHECK(ttbpe_encode_ordinary(tt, "z", 1, &ids, &n) == -1);
    ttbpe_free_ids(ids);

    /* decode round trip through the special */
    const int round[] = {0, 100, 1};
    char *out = NULL;
    size_t out_len = 0;
    CHECK(ttbpe_decode(tt, round, 3, &out, &out_len) == 0);
    CHECK(out_len == 9 && memcmp(out, "a" SPECIAL_TOK "b", 9) == 0);
    ttbpe_free_buf(out);

    ttbpe_free(tt);
    end_case("tok.json vocabulary + special tokens");
}

static void test_split_modes(void) {
    begin_case("pre-tokenizer boundaries");

    tiktoken_bpe_t *tt = load_memory(TIKTOKEN_VOCAB);
    CHECK(tt != NULL);
    if (!tt) {
        end_case("pre-tokenizer boundaries");
        return;
    }

    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_NONE) == 0);
    const char *none[] = {"abc de"};
    check_split(tt, "abc de", none, 1);

    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_CL100K) == 0);
    const char *c1[] = {"Hello", ",", " world", "!", " ", "123"};
    check_split(tt, "Hello, world! 123", c1, 6);
    const char *c2[] = {"a", " ", " b"};
    check_split(tt, "a  b", c2, 3);
    const char *c3[] = {"a", "\n\n", "b"};
    check_split(tt, "a\n\nb", c3, 3);
    const char *c4[] = {"a", " \n"};
    check_split(tt, "a \n", c4, 2);
    const char *c5[] = {"   "};
    check_split(tt, "   ", c5, 1);
    const char *c6[] = {"12", " ", "34"};
    check_split(tt, "12 34", c6, 3);
    const char *c7[] = {"don", "'t"};
    check_split(tt, "don't", c7, 2);
    const char *c8[] = {"a", " \n", "b"};
    check_split(tt, "a \nb", c8, 3);

    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_GPT2) == 0);
    const char *g1[] = {"Hello", ",", " world", "!"};
    check_split(tt, "Hello, world!", g1, 4);
    const char *g2[] = {"don", "'t"};
    check_split(tt, "don't", g2, 2);
    const char *g3[] = {"I", "'ll"};
    check_split(tt, "I'll", g3, 2);
    const char *g4[] = {"a", " ", " b"};
    check_split(tt, "a  b", g4, 3);
    const char *g5[] = {"a", " \n"};
    check_split(tt, "a \n", g5, 2);

    /* an invalid mode is refused and leaves the previous mode in place */
    CHECK(ttbpe_set_split_mode(tt, 42) == -1);

    ttbpe_free(tt);
    end_case("pre-tokenizer boundaries");
}

static void test_cache_and_errors(void) {
    begin_case("piece cache, malformed input");

    tiktoken_bpe_t *tt = load_memory(TIKTOKEN_VOCAB);
    CHECK(tt != NULL);
    if (!tt) {
        end_case("piece cache, malformed input");
        return;
    }
    CHECK(ttbpe_set_split_mode(tt, TTBPE_SPLIT_CL100K) == 0);

    int *a = NULL, *b = NULL;
    size_t na = 0, nb = 0;
    const char *text = "abc ab ababa abc";
    CHECK(ttbpe_encode(tt, text, strlen(text), &a, &na) == 0);
    CHECK(ttbpe_encode(tt, text, strlen(text), &b, &nb) == 0); /* cache hit */
    CHECK(ids_equal(a, na, b, nb));
    ttbpe_free_ids(a);
    ttbpe_free_ids(b);

    CHECK(ttbpe_set_cache_enabled(tt, 0) == 0);
    a = b = NULL;
    na = nb = 0;
    CHECK(ttbpe_encode(tt, text, strlen(text), &a, &na) == 0);
    CHECK(ttbpe_encode(tt, text, strlen(text), &b, &nb) == 0);
    CHECK(ids_equal(a, na, b, nb));
    ttbpe_free_ids(a);
    ttbpe_free_ids(b);
    CHECK(ttbpe_set_cache_enabled(tt, 1) == 0);

    ttbpe_free(tt);

    /* malformed vocabularies must fail loudly instead of returning a handle */
    CHECK(load_memory("not a vocabulary") == NULL);
    CHECK(ttbpe_last_error() != NULL);
    CHECK(load_memory("YQ==\n!!!not base64!!!\n") == NULL);
    CHECK(load_memory("{\"regular_tokens\":[{\"bytes\":[97],\"token\":0}") == NULL);
    CHECK(load_memory("{\"regular_tokens\":[{\"bytes\":[999],\"token\":0}]}") == NULL);
    CHECK(load_memory("") == NULL);

    end_case("piece cache, malformed input");
}

static void test_decode_replace(void) {
    begin_case("decode errors=replace");

    tiktoken_bpe_t *tt = load_memory(TIKTOKEN_VOCAB);
    CHECK(tt != NULL);
    if (!tt) {
        end_case("decode errors=replace");
        return;
    }

    char *out = NULL;
    size_t out_len = 0;

    /* valid 2 byte sequence survives */
    const int ok_ids[] = {19};
    CHECK(ttbpe_decode(tt, ok_ids, 1, &out, &out_len) == 0);
    CHECK(out_len == 2 && (unsigned char)out[0] == 0xc3 &&
          (unsigned char)out[1] == 0xa9);
    ttbpe_free_buf(out);

    /* lone 0xff becomes one U+FFFD */
    const int ff_ids[] = {16};
    CHECK(ttbpe_decode(tt, ff_ids, 1, &out, &out_len) == 0);
    CHECK(out_len == 3 && memcmp(out, "\xef\xbf\xbd", 3) == 0);
    ttbpe_free_buf(out);

    /* truncated 2 byte sequence becomes one U+FFFD */
    const int trunc2[] = {17};
    CHECK(ttbpe_decode(tt, trunc2, 1, &out, &out_len) == 0);
    CHECK(out_len == 3 && memcmp(out, "\xef\xbf\xbd", 3) == 0);
    ttbpe_free_buf(out);

    /* truncated 3 byte sequence (E2 82) becomes one U+FFFD, no extra 'a' */
    const int trunc3[] = {18, 0};
    CHECK(ttbpe_decode(tt, trunc3, 2, &out, &out_len) == 0);
    CHECK(out_len == 4 && memcmp(out, "\xef\xbf\xbd" "a", 4) == 0);
    ttbpe_free_buf(out);

    /* "a" + U+FFFD + "a" keeps the valid parts intact */
    const int mixed[] = {0, 16, 0};
    CHECK(ttbpe_decode(tt, mixed, 3, &out, &out_len) == 0);
    CHECK(out_len == 5 && memcmp(out, "a\xef\xbf\xbd" "a", 5) == 0);
    ttbpe_free_buf(out);

    ttbpe_free(tt);
    end_case("decode errors=replace");
}

int main(void) {
    printf("tiktoken_bpe unit tests\n");
    test_tiktoken_format();
    test_json_format();
    test_split_modes();
    test_cache_and_errors();
    test_decode_replace();
    printf("\n%d checks, %d failures -> %s\n", g_checks, g_failures,
           g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
