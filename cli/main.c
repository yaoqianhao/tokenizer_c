/* main.c - tokenizer CLI.
 *
 * Subcommands:
 *   train   --corpus <file> --model <out.tkm> [options]
 *   encode  --model <file> (--text <text> | --file <in.txt>)
 *   decode  --model <file> --ids <id,id,...>
 */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Windows: command-line arguments arrive in the system ANSI code page (e.g.
 * GBK on zh-CN systems). Re-acquire them via the wide-character command line
 * and convert to UTF-8 so all text handling in the library stays UTF-8.
 * Returns a heap argv (caller frees) or NULL (keep original argv). */
static char **win32_argv_to_utf8(int *argc_out) {
#ifdef _WIN32
    int wargc = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return NULL;
    char **argv = (char **)calloc((size_t)wargc + 1, sizeof(char *));
    if (!argv) {
        LocalFree(wargv);
        return NULL;
    }
    for (int i = 0; i < wargc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        if (n <= 0) {
            argv[i] = (char *)malloc(1);
            if (argv[i]) argv[i][0] = '\0';
            continue;
        }
        argv[i] = (char *)malloc((size_t)n);
        if (argv[i]) {
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], n, NULL, NULL);
        }
    }
    LocalFree(wargv);
    *argc_out = wargc;
    return argv;
#else
    (void)argc_out;
    return NULL;
#endif
}

/* P0-1: 64-bit file size (Windows). */
static int cli_file_size64(FILE *fp, long long *out) {
    __int64 cur = _ftelli64(fp);
    if (cur < 0) return -1;
    if (_fseeki64(fp, 0, SEEK_END) != 0) return -1;
    __int64 sz = _ftelli64(fp);
    _fseeki64(fp, cur, SEEK_SET);
    if (sz < 0) return -1;
    *out = (long long)sz;
    return 0;
}

static const char *g_prog = "tok-cli";

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  %s train --corpus <file> --model <out.tkm> [options]\n"
        "    options: --vocab-size N --max-piece-len N --max-candidates N\n"
        "             --iters N --eps F --shrink PCT --min-rounds N\n"
        "             --no-byte-fallback --fold-case --json <out.json>\n"
        "             --ckpt <file> --sp-model <out.model> -v\n"
        "  %s encode --model <file> (--text <text> | --file <in.txt>)\n"
        "             [--norm=nfkc|nfc|reversible] [--segment-rules] [--byte-real]\n"
        "  %s decode --model <file> --ids <id,id,...>\n"
        "  %s bench  --model <file> (--file <f> | --list <paths.txt>)"
        " [--repeat N] [--decode-cap N]\n"
        "             [--norm=nfkc|nfc|reversible] [--segment-rules] [--byte-real]\n"
        "    inference overrides (applied after load, for re-evaluating an\n"
        "    already-trained model without a retrain):\n"
        "      --norm=<mode>      normalization mode (2026-10-14):\n"
        "                           nfkc       full NFKC (default, lossy)\n"
        "                           nfc        strict NFC, no compat folding\n"
        "                           reversible lossless canonical composition\n"
        "      --norm-reversible  alias of --norm=reversible (P0-1)\n"
        "      --segment-rules    digit 3-grouping + operator isolation (P1-6)\n"
        "      --byte-real        emit real <0xXX> piece ids (P1-4)\n",
        g_prog, g_prog, g_prog, g_prog);
}

static const char *arg_value(int argc, char **argv, int i, const char *opt,
                             int *next) {
    if (i + 1 < argc) {
        *next = i + 2;
        return argv[i + 1];
    }
    fprintf(stderr, "missing value for %s\n", opt);
    return NULL;
}

static int run_train(int argc, char **argv) {
    tokenizer_train_options opts = TOKENIZER_TRAIN_OPTIONS_DEFAULT;
    const char *sp_model = NULL;
    for (int i = 0; i < argc; i++) {
        int next = i + 1;
        const char *v = NULL;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        }
        if (strcmp(argv[i], "--corpus") == 0) {
            v = arg_value(argc, argv, i, "--corpus", &next);
            if (!v) return 1;
            opts.corpus_path = v;
        } else if (strcmp(argv[i], "--model") == 0) {
            v = arg_value(argc, argv, i, "--model", &next);
            if (!v) return 1;
            opts.model_path = v;
        } else if (strcmp(argv[i], "--json") == 0) {
            v = arg_value(argc, argv, i, "--json", &next);
            if (!v) return 1;
            opts.json_path = v;
        } else if (strcmp(argv[i], "--sp-model") == 0) {
            v = arg_value(argc, argv, i, "--sp-model", &next);
            if (!v) return 1;
            sp_model = v;
        } else if (strcmp(argv[i], "--ckpt") == 0) {
            v = arg_value(argc, argv, i, "--ckpt", &next);
            if (!v) return 1;
            opts.checkpoint_path = v;
        } else if (strcmp(argv[i], "--vocab-size") == 0) {
            v = arg_value(argc, argv, i, "--vocab-size", &next);
            if (!v) return 1;
            opts.vocab_size = (size_t)strtoul(v, NULL, 10);
        } else if (strcmp(argv[i], "--max-piece-len") == 0) {
            v = arg_value(argc, argv, i, "--max-piece-len", &next);
            if (!v) return 1;
            opts.max_piece_len = atoi(v);
        } else if (strcmp(argv[i], "--max-candidates") == 0) {
            v = arg_value(argc, argv, i, "--max-candidates", &next);
            if (!v) return 1;
            opts.max_candidates = atoi(v);
        } else if (strcmp(argv[i], "--iters") == 0) {
            v = arg_value(argc, argv, i, "--iters", &next);
            if (!v) return 1;
            opts.max_iters = atoi(v);
        } else if (strcmp(argv[i], "--eps") == 0) {
            v = arg_value(argc, argv, i, "--eps", &next);
            if (!v) return 1;
            opts.em_eps = atof(v);
        } else if (strcmp(argv[i], "--shrink") == 0) {
            v = arg_value(argc, argv, i, "--shrink", &next);
            if (!v) return 1;
            opts.shrink_pct = atoi(v);
        } else if (strcmp(argv[i], "--min-rounds") == 0) {
            v = arg_value(argc, argv, i, "--min-rounds", &next);
            if (!v) return 1;
            opts.min_rounds = atoi(v);
        } else if (strcmp(argv[i], "--no-byte-fallback") == 0) {
            opts.add_byte_fallback = 0;
        } else if (strcmp(argv[i], "--fold-case") == 0) {
            opts.fold_case = 1;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = 1;
        } else {
            fprintf(stderr, "unknown train option: %s\n", argv[i]);
            return 1;
        }
        i = next - 1;
    }
    if (!opts.corpus_path || !opts.model_path) {
        fprintf(stderr, "train requires --corpus and --model\n");
        return 1;
    }
    if (tokenizer_train(&opts) != 0) {
        fprintf(stderr, "train failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        return 1;
    }
    printf("trained model saved to %s (vocab_size=%zu)\n", opts.model_path,
           opts.vocab_size ? opts.vocab_size : 4096);
    if (opts.json_path) printf("json export: %s\n", opts.json_path);
    if (sp_model) {
        tokenizer_t *tok = tokenizer_load(opts.model_path);
        if (!tok) {
            fprintf(stderr, "load for sp export failed: %s\n",
                    tokenizer_last_error() ? tokenizer_last_error() : "unknown");
            return 1;
        }
        if (tokenizer_export_sentencepiece(tok, sp_model) != 0) {
            fprintf(stderr, "sp export failed: %s\n",
                    tokenizer_last_error() ? tokenizer_last_error() : "unknown");
            tokenizer_free(tok);
            return 1;
        }
        tokenizer_free(tok);
        printf("sentencepiece model saved to %s\n", sp_model);
    }
    return 0;
}

/* Load .tkm or SentencePiece .model by extension */
static tokenizer_t *load_model(const char *model) {
    size_t ml = strlen(model);
    if (ml > 6 && strcmp(model + ml - 6, ".model") == 0)
        return tokenizer_load_sentencepiece(model);
    return tokenizer_load(model);
}

/* 2026-10-14: --norm=<mode> / --norm <mode> parsing.
 * Returns the TOKENIZER_NORM_MODE_* value, -1 when the option is absent and
 * -2 for an unrecognised value (callers report the error and exit). */
static int cli_parse_norm_mode(const char *v) {
    if (!v || !*v) return -2;
    if (strcmp(v, "nfkc") == 0 || strcmp(v, "legacy") == 0)
        return TOKENIZER_NORM_MODE_NFKC;
    if (strcmp(v, "nfc") == 0) return TOKENIZER_NORM_MODE_NFC;
    if (strcmp(v, "reversible") == 0 || strcmp(v, "rev") == 0)
        return TOKENIZER_NORM_MODE_REVERSIBLE;
    return -2;
}

static const char *cli_norm_mode_name(int mode) {
    switch (mode) {
        case TOKENIZER_NORM_MODE_NFKC: return "nfkc";
        case TOKENIZER_NORM_MODE_NFC: return "nfc";
        case TOKENIZER_NORM_MODE_REVERSIBLE: return "reversible";
        default: return "unknown";
    }
}

/* P0-1 / P1-4 / P1-6 runtime overrides (see tokenizer.h). Applied after load so
 * an already-trained model can be re-evaluated with the fixed encoder without
 * a retrain. Flags only ever turn an override ON (0 = keep the model header);
 * norm_mode >= 0 is the explicit 2026-10-14 mode switch and wins over the
 * --norm-reversible alias. */
static void apply_infer_flags(tokenizer_t *tok, int norm_mode, int norm_rev,
                              int seg_rules, int byte_real) {
    if (!tok) return;
    if (norm_mode >= 0) {
        if (tokenizer_set_norm_mode(tok, norm_mode) != 0)
            fprintf(stderr, "warning: unsupported norm mode id %d ignored\n",
                    norm_mode);
    } else if (norm_rev) {
        tokenizer_set_norm_reversible(tok, 1);
    }
    if (seg_rules) tokenizer_set_segment_rules(tok, 1);
    if (byte_real) tokenizer_set_byte_real(tok, 1);
}

static int run_encode(int argc, char **argv) {
    const char *model = NULL, *text = NULL, *file = NULL;
    int norm_rev = 0, seg_rules = 0, byte_real = 0;
    int norm_mode = -1; /* 2026-10-14: explicit --norm=<mode> override */
    for (int i = 0; i < argc; i++) {
        int next = i + 1;
        const char *v = NULL;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        }
        if (strcmp(argv[i], "--model") == 0) {
            v = arg_value(argc, argv, i, "--model", &next);
            if (!v) return 1;
            model = v;
        } else if (strcmp(argv[i], "--text") == 0) {
            v = arg_value(argc, argv, i, "--text", &next);
            if (!v) return 1;
            text = v;
        } else if (strcmp(argv[i], "--file") == 0) {
            v = arg_value(argc, argv, i, "--file", &next);
            if (!v) return 1;
            file = v;
        } else if (strcmp(argv[i], "--norm-reversible") == 0) {
            norm_rev = 1;
        } else if (strncmp(argv[i], "--norm=", 7) == 0) {
            norm_mode = cli_parse_norm_mode(argv[i] + 7);
            if (norm_mode < 0) {
                fprintf(stderr, "encode: unknown norm mode '%s' "
                                "(expected nfkc|nfc|reversible)\n", argv[i] + 7);
                return 1;
            }
        } else if (strcmp(argv[i], "--norm") == 0) {
            v = arg_value(argc, argv, i, "--norm", &next);
            if (!v) return 1;
            norm_mode = cli_parse_norm_mode(v);
            if (norm_mode < 0) {
                fprintf(stderr, "encode: unknown norm mode '%s' "
                                "(expected nfkc|nfc|reversible)\n", v);
                return 1;
            }
        } else if (strcmp(argv[i], "--segment-rules") == 0) {
            seg_rules = 1;
        } else if (strcmp(argv[i], "--byte-real") == 0) {
            byte_real = 1;
        } else {
            fprintf(stderr, "unknown encode option: %s\n", argv[i]);
            return 1;
        }
        i = next - 1;
    }
    if (!model || (!text && !file)) {
        fprintf(stderr, "encode requires --model and (--text | --file)\n");
        return 1;
    }
    tokenizer_t *tok = load_model(model);
    if (!tok) {
        fprintf(stderr, "load failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        return 1;
    }
    apply_infer_flags(tok, norm_mode, norm_rev, seg_rules, byte_real);
    char *content = NULL;
    size_t content_len = 0;
    if (text) {
        content = (char *)text;
        content_len = strlen(text);
    } else {
        FILE *fp = fopen(file, "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", file); tokenizer_free(tok); return 1; }
        long long sz = 0;
        if (cli_file_size64(fp, &sz) != 0 || sz <= 0) {
            fclose(fp); tokenizer_free(tok); return 1;
        }
        content = (char *)malloc((size_t)sz + 1);
        if (!content) { fclose(fp); tokenizer_free(tok); return 1; }
        size_t rd = fread(content, 1, (size_t)sz, fp);
        fclose(fp);
        content[rd] = '\0';
        content_len = rd;
    }
    int *ids = NULL;
    size_t n = 0;
    if (tokenizer_encode(tok, content, content_len, &ids, &n) != 0) {
        fprintf(stderr, "encode failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        if (content != text) free(content);
        tokenizer_free(tok);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        if (i) printf(" ");
        printf("%d", ids[i]);
    }
    printf("\n");
    tokenizer_free_ids(ids);
    if (content != text) free(content);
    tokenizer_free(tok);
    return 0;
}

static int run_decode(int argc, char **argv) {
    const char *model = NULL, *ids_str = NULL;
    for (int i = 0; i < argc; i++) {
        int next = i + 1;
        const char *v = NULL;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        }
        if (strcmp(argv[i], "--model") == 0) {
            v = arg_value(argc, argv, i, "--model", &next);
            if (!v) return 1;
            model = v;
        } else if (strcmp(argv[i], "--ids") == 0) {
            v = arg_value(argc, argv, i, "--ids", &next);
            if (!v) return 1;
            ids_str = v;
        } else {
            fprintf(stderr, "unknown decode option: %s\n", argv[i]);
            return 1;
        }
        i = next - 1;
    }
    if (!model || !ids_str) {
        fprintf(stderr, "decode requires --model and --ids\n");
        return 1;
    }
    tokenizer_t *tok = load_model(model);
    if (!tok) {
        fprintf(stderr, "load failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        return 1;
    }
    /* parse space/comma separated ids */
    size_t cap = strlen(ids_str) / 2 + 8;
    int *ids = (int *)malloc(cap * sizeof(int));
    size_t n = 0;
    const char *p = ids_str;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (n >= cap) {
            size_t ncap = cap ? cap * 2 : 256;
            int *nids = (int *)realloc(ids, ncap * sizeof(int));
            if (!nids) {
                fprintf(stderr, "decode: OOM growing id buffer\n");
                free(ids);
                tokenizer_free(tok);
                return 1;
            }
            ids = nids;
            cap = ncap;
        }
        ids[n++] = (int)v;
        p = end;
    }
    char *out = NULL;
    size_t out_len = 0;
    if (tokenizer_decode(tok, ids, n, &out, &out_len) != 0) {
        fprintf(stderr, "decode failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        free(ids);
        tokenizer_free(tok);
        return 1;
    }
    fwrite(out, 1, out_len, stdout);
    fputc('\n', stdout);
    tokenizer_free_text(out);
    free(ids);
    tokenizer_free(tok);
    return 0;
}

/* ------------------------------------------------------------------ */
/* bench: in-process throughput measurement (P0-2)                     */
/*                                                                     */
/* One process: load once, encode every input file, then decode the    */
/* resulting id sequences --repeat times. The kernel timings therefore */
/* exclude per-call process start and repeated model loading, which    */
/* the legacy "one CLI call per metric" harness folded into load_ms.   */
/* Emits one single-line JSON object on stdout; progress goes to       */
/* stderr, so a caller can json-parse stdout directly.                 */
/* ------------------------------------------------------------------ */

static double bench_now_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static void json_puts(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        if (*p == '\\' || *p == '"') { putchar('\\'); putchar((char)*p); }
        else if (*p < 0x20) printf("\\u%04x", (unsigned)*p);
        else putchar((char)*p);
    }
    putchar('"');
}

static int bench_read_file(const char *path, char **out, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    long long sz = 0;
    if (cli_file_size64(fp, &sz) != 0 || sz < 0) { fclose(fp); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\0';
    *out = buf;
    *out_len = rd;
    return 0;
}

static int run_bench(int argc, char **argv) {
    const char *model = NULL, *list_path = NULL;
    const char *paths[32];
    static char list_buf[32][1024];
    int npaths = 0, repeat = 1, decode_cap = 0;
    int norm_rev = 0, seg_rules = 0, byte_real = 0;
    int norm_mode = -1; /* 2026-10-14: explicit --norm=<mode> override */
    for (int i = 0; i < argc; i++) {
        int next = i + 1;
        const char *v = NULL;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        } else if (strcmp(argv[i], "--model") == 0) {
            v = arg_value(argc, argv, i, "--model", &next);
            if (!v) return 1;
            model = v;
        } else if (strcmp(argv[i], "--list") == 0) {
            v = arg_value(argc, argv, i, "--list", &next);
            if (!v) return 1;
            list_path = v;
        } else if (strcmp(argv[i], "--file") == 0) {
            v = arg_value(argc, argv, i, "--file", &next);
            if (!v) return 1;
            if (npaths < 32) paths[npaths++] = v;
        } else if (strcmp(argv[i], "--repeat") == 0) {
            v = arg_value(argc, argv, i, "--repeat", &next);
            if (!v) return 1;
            repeat = atoi(v);
        } else if (strcmp(argv[i], "--decode-cap") == 0) {
            v = arg_value(argc, argv, i, "--decode-cap", &next);
            if (!v) return 1;
            decode_cap = atoi(v);
        } else if (strcmp(argv[i], "--norm-reversible") == 0) {
            norm_rev = 1;
        } else if (strncmp(argv[i], "--norm=", 7) == 0) {
            norm_mode = cli_parse_norm_mode(argv[i] + 7);
            if (norm_mode < 0) {
                fprintf(stderr, "bench: unknown norm mode '%s' "
                                "(expected nfkc|nfc|reversible)\n", argv[i] + 7);
                return 1;
            }
        } else if (strcmp(argv[i], "--norm") == 0) {
            v = arg_value(argc, argv, i, "--norm", &next);
            if (!v) return 1;
            norm_mode = cli_parse_norm_mode(v);
            if (norm_mode < 0) {
                fprintf(stderr, "bench: unknown norm mode '%s' "
                                "(expected nfkc|nfc|reversible)\n", v);
                return 1;
            }
        } else if (strcmp(argv[i], "--segment-rules") == 0) {
            seg_rules = 1;
        } else if (strcmp(argv[i], "--byte-real") == 0) {
            byte_real = 1;
        } else {
            fprintf(stderr, "unknown bench option: %s\n", argv[i]);
            return 1;
        }
        i = next - 1;
    }
    if (list_path) {
        FILE *fp = fopen(list_path, "rb");
        if (!fp) {
            fprintf(stderr, "bench: cannot open list %s\n", list_path);
            return 1;
        }
        char line[1024];
        while (npaths < 32 && fgets(line, sizeof(line), fp)) {
            size_t l = strlen(line);
            while (l && (line[l - 1] == '\n' || line[l - 1] == '\r' ||
                         line[l - 1] == ' ')) line[--l] = '\0';
            if (!l) continue;
            memcpy(list_buf[npaths], line, l + 1);
            paths[npaths] = list_buf[npaths];
            npaths++;
        }
        fclose(fp);
    }
    if (!model || npaths == 0) {
        fprintf(stderr, "bench requires --model and (--file | --list)\n");
        return 1;
    }
    if (repeat < 1) repeat = 1;

    double t0 = bench_now_ms();
    tokenizer_t *tok = load_model(model);
    double load_ms = bench_now_ms() - t0;
    if (!tok) {
        fprintf(stderr, "load failed: %s\n",
                tokenizer_last_error() ? tokenizer_last_error() : "unknown");
        return 1;
    }
    apply_infer_flags(tok, norm_mode, norm_rev, seg_rules, byte_real);

    /* 2026-10-14: the mode actually used by encode(), after resolving the
     * --norm override / --norm-reversible alias / model header default. */
    int eff_norm_mode = norm_mode;
    if (eff_norm_mode < 0)
        eff_norm_mode = norm_rev ? TOKENIZER_NORM_MODE_REVERSIBLE
                                 : TOKENIZER_NORM_MODE_NFKC;

    double enc_ms_total = 0.0, dec_ms_total = 0.0, dec_tokens_total = 0.0;
    unsigned long long bytes_total = 0, tokens_total = 0, decoded_bytes_sum = 0;

    printf("{");
    printf("\"model\": ");
    json_puts(model);
    printf(", \"mode\": \"in-process\", \"vocab_size\": %d, \"load_ms\": %.3f,"
           " \"repeat\": %d, \"norm_mode\": \"%s\", \"norm_mode_id\": %d,"
           " \"norm_reversible\": %d, \"segment_rules\": %d,"
           " \"byte_real\": %d, \"files\": [",
           (int)tokenizer_vocab_size(tok), load_ms, repeat,
           cli_norm_mode_name(eff_norm_mode), eff_norm_mode,
           eff_norm_mode == TOKENIZER_NORM_MODE_REVERSIBLE ? 1 : 0,
           seg_rules ? 1 : 0, byte_real ? 1 : 0);

    for (int fi = 0; fi < npaths; fi++) {
        char *content = NULL;
        size_t clen = 0;
        if (bench_read_file(paths[fi], &content, &clen) != 0) {
            fprintf(stderr, "bench: cannot read %s (skipped)\n", paths[fi]);
            continue;
        }
        int *ids = NULL;
        size_t n = 0;
        double e0 = bench_now_ms();
        int rc = tokenizer_encode(tok, content, clen, &ids, &n);
        double e1 = bench_now_ms();
        if (rc != 0) {
            fprintf(stderr, "bench: encode failed on %s (skipped)\n", paths[fi]);
            free(content);
            continue;
        }
        size_t dn = (decode_cap > 0 && n > (size_t)decode_cap)
                        ? (size_t)decode_cap : n;
        double d0 = bench_now_ms();
        for (int r = 0; r < repeat && dn > 0; r++) {
            char *out = NULL;
            size_t olen = 0;
            if (tokenizer_decode(tok, ids, dn, &out, &olen) != 0) {
                fprintf(stderr, "bench: decode failed on %s\n", paths[fi]);
                break;
            }
            decoded_bytes_sum += (unsigned long long)olen;
            tokenizer_free_text(out);
        }
        double d1 = bench_now_ms();
        double enc_ms = e1 - e0, dec_ms = d1 - d0;
        enc_ms_total += enc_ms;
        dec_ms_total += dec_ms;
        dec_tokens_total += (double)dn * (double)repeat;
        bytes_total += (unsigned long long)clen;
        tokens_total += (unsigned long long)n;

        if (fi) printf(", ");
        printf("{\"path\": ");
        json_puts(paths[fi]);
        printf(", \"bytes\": %llu, \"tokens\": %llu, \"encode_ms\": %.3f,"
               " \"encode_tps\": %.3f, \"decoded_tokens\": %llu,"
               " \"decode_ms\": %.3f, \"decode_tps\": %.3f}",
               (unsigned long long)clen, (unsigned long long)n, enc_ms,
               enc_ms > 0 ? (double)clen / (enc_ms / 1000.0) : 0.0,
               (unsigned long long)(dn * (size_t)repeat), dec_ms,
               dec_ms > 0 ? ((double)dn * (double)repeat) / (dec_ms / 1000.0)
                          : 0.0);
        fprintf(stderr, "bench: %s bytes=%llu tokens=%llu encode_ms=%.3f "
                        "decode_ms=%.3f\n",
                paths[fi], (unsigned long long)clen,
                (unsigned long long)n, enc_ms, dec_ms);
        tokenizer_free_ids(ids);
        free(content);
    }

    printf("], \"bytes_total\": %llu, \"tokens_total\": %llu,"
           " \"encode_ms_total\": %.3f, \"decode_ms_total\": %.3f,"
           " \"encode_tps\": %.3f, \"decode_tps\": %.3f,"
           " \"decoded_bytes_sum\": %llu}\n",
           bytes_total, tokens_total, enc_ms_total, dec_ms_total,
           enc_ms_total > 0 ? (double)bytes_total / (enc_ms_total / 1000.0) : 0.0,
           dec_ms_total > 0 ? dec_tokens_total / (dec_ms_total / 1000.0) : 0.0,
           decoded_bytes_sum);
    tokenizer_free(tok);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    char **utf8_argv = win32_argv_to_utf8(&argc);
    if (utf8_argv) argv = utf8_argv;
#endif
    if (argc < 2) {
        usage();
        return 1;
    }
    int rc = 0;
    g_prog = argv[0];
    if (strcmp(argv[1], "train") == 0) rc = run_train(argc - 2, argv + 2);
    else if (strcmp(argv[1], "encode") == 0) rc = run_encode(argc - 2, argv + 2);
    else if (strcmp(argv[1], "decode") == 0) rc = run_decode(argc - 2, argv + 2);
    else if (strcmp(argv[1], "bench") == 0) rc = run_bench(argc - 2, argv + 2);
    else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        usage();
        rc = 0;
    } else {
        usage();
        rc = 1;
    }
#ifdef _WIN32
    if (utf8_argv) {
        for (int i = 0; i < argc; i++) free(utf8_argv[i]);
        free(utf8_argv);
    }
#endif
    return rc;
}
