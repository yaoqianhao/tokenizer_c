/* test_cli_tt_raw.c - CLI level regression test for the byte exact output of
 * `tok-cli tt-decode --raw`.
 *
 * Background: `tt-decode --raw` must write exactly the bytes produced by
 * ttbpe_decode_bytes() (tiktoken Encoding.decode_bytes() semantics): no added
 * trailing newline, no Windows text mode CRLF translation, invalid UTF-8 bytes
 * and NUL bytes preserved.  A regression here is invisible to the in-process
 * unit tests of tests/test_tiktoken_bpe.c because those never touch stdout,
 * so this test drives the real CLI binary through a pipe and compares the raw
 * byte stream.
 *
 * The fixture is a miniature .tiktoken vocabulary generated at run time:
 *   id 0..255 -> the single byte with that value
 *   id 256    -> "hello"
 *   id 257    -> " world"
 *
 * Usage: test_cli_tt_raw <path to the tok-cli executable>
 *   CMake/CTest : add_test(NAME cli_tt_raw COMMAND test_cli_tt_raw $<TARGET_FILE:tok-cli>)
 * Exit code 0 = all checks passed, 1 = at least one failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define POPEN  _popen
#define PCLOSE _pclose
#else
#define POPEN  popen
#define PCLOSE pclose
#endif

static int g_checks = 0;
static int g_failures = 0;

static void check_impl(int ok, const char *expr, int line) {
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("  FAIL line %d: %s\n", line, expr);
    }
}

#define CHECK(cond) check_impl((cond) != 0, #cond, __LINE__)

/* ------------------------------------------------------------------ */
/* fixture: a miniature .tiktoken vocabulary                           */
/* ------------------------------------------------------------------ */

static size_t b64_encode(const unsigned char *in, size_t n, char *out) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t i = 0, o = 0;
    while (i + 3 <= n) {
        unsigned v = ((unsigned)in[i] << 16) | ((unsigned)in[i + 1] << 8) |
                     (unsigned)in[i + 2];
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = tab[(v >> 6) & 63];
        out[o++] = tab[v & 63];
        i += 3;
    }
    if (n - i == 1) {
        unsigned v = (unsigned)in[i] << 16;
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = '=';
        out[o++] = '=';
    } else if (n - i == 2) {
        unsigned v = ((unsigned)in[i] << 16) | ((unsigned)in[i + 1] << 8);
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = tab[(v >> 6) & 63];
        out[o++] = '=';
    }
    out[o] = '\0';
    return o;
}

static int write_line(FILE *f, const unsigned char *bytes, size_t n) {
    char b64[64];
    if (n > 45) return -1;
    b64_encode(bytes, n, b64);
    if (fputs(b64, f) == EOF) return -1;
    if (fputc('\n', f) == EOF) return -1;
    return 0;
}

static int write_vocab(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (int b = 0; b < 256; b++) {
        unsigned char c = (unsigned char)b;
        if (write_line(f, &c, 1) != 0) {
            fclose(f);
            return -1;
        }
    }
    if (write_line(f, (const unsigned char *)"hello", 5) != 0 ||
        write_line(f, (const unsigned char *)" world", 6) != 0) {
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* running the CLI                                                     */
/* ------------------------------------------------------------------ */

static int capture(const char *cli, const char *args, unsigned char *out,
                   size_t cap, size_t *out_len) {
    char cmd[4096];
#ifdef _WIN32
    /* _popen runs the command through `cmd /c`; when that command line holds
     * more than two quotes, cmd strips the leading and the trailing quote,
     * which mangles a plain "\"prog\" args" form.  The extra outer pair is the
     * documented workaround and keeps both the program and its arguments
     * intact. */
    snprintf(cmd, sizeof(cmd), "\"\"%s\" %s\"", cli, args);
#else
    snprintf(cmd, sizeof(cmd), "\"%s\" %s", cli, args);
#endif
    FILE *p = POPEN(cmd, "rb");
    if (!p) {
        printf("  FAIL cannot spawn: %s\n", cmd);
        return -1;
    }
    size_t n = fread(out, 1, cap, p);
    int too_long = !feof(p);
    int rc = PCLOSE(p);
    if (too_long) {
        printf("  FAIL output exceeds %llu bytes\n", (unsigned long long)cap);
        return -1;
    }
    if (rc != 0) {
        printf("  FAIL non zero exit status %d: %s\n", rc, cmd);
        return -1;
    }
    *out_len = n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tok-cli executable>\n", argv[0]);
        return 1;
    }
    const char *cli = argv[1];
    const char *dir = getenv("TEMP");
    if (!dir || !*dir) dir = getenv("TMP");
    if (!dir || !*dir) dir = ".";
    char vocab[1024];
    snprintf(vocab, sizeof(vocab), "%s/test_cli_tt_raw_vocab.tiktoken", dir);

    printf("cli_tt_raw tests\n  cli   : %s\n  vocab : %s\n", cli, vocab);
    if (write_vocab(vocab) != 0) {
        printf("  FAIL cannot write the vocabulary fixture\n");
        return 1;
    }

    char args[2048];
    unsigned char out[4096];
    size_t n = 0;

    /* 1. --raw adds nothing: a single 0x0A byte in, a single 0x0A byte out
     *    (a missing binary mode would turn it into CRLF). */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"10\" --raw", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n == 1 && out[0] == 0x0a);

    /* 2. --raw keeps invalid UTF-8 bytes untouched (E4 B8 is a truncated
     *    sequence that ttbpe_decode() would replace by U+FFFD). */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"228,184\" --raw", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n == 2 && out[0] == 0xe4 && out[1] == 0xb8);

    /* 3. multi byte tokens are concatenated unchanged. */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"256,257\" --raw", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n == 11 && memcmp(out, "hello world", 11) == 0);

    /* 4. control bytes (including NUL) survive the pipe. */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"0,1,2\" --raw", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n == 3 && out[0] == 0x00 && out[1] == 0x01 && out[2] == 0x02);

    /* 5. contrast: without --raw the truncated sequence becomes U+FFFD and a
     *    line terminator is appended. */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"228,184\"", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n >= 4 && memcmp(out, "\xef\xbf\xbd", 3) == 0);
    {
        int tail_ok = 1;
        for (size_t i = 3; i < n; i++)
            if (out[i] != '\r' && out[i] != '\n') tail_ok = 0;
        CHECK(tail_ok);
        CHECK(n > 0 && out[n - 1] == '\n');
    }

    /* 6. contrast: valid text output ends with a line terminator. */
    n = 0;
    snprintf(args, sizeof(args),
             "tt-decode --vocab \"%s\" --ids \"256,257\"", vocab);
    CHECK(capture(cli, args, out, sizeof(out), &n) == 0);
    CHECK(n >= 12 && memcmp(out, "hello world", 11) == 0);
    CHECK(n > 0 && out[n - 1] == '\n');

    remove(vocab);
    printf("\n%d checks, %d failures -> %s\n", g_checks, g_failures,
           g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
