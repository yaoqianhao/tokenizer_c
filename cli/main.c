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

static const char *g_prog = "tok-cli";

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  %s train --corpus <file> --model <out.tkm> [options]\n"
        "    options: --vocab-size N --max-piece-len N --max-candidates N\n"
        "             --iters N --eps F --shrink PCT --min-rounds N\n"
        "             --no-byte-fallback --fold-case --json <out.json>\n"
        "             --sp-model <out.model> -v\n"
        "  %s encode --model <file> (--text <text> | --file <in.txt>)\n"
        "  %s decode --model <file> --ids <id,id,...>\n",
        g_prog, g_prog, g_prog);
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

static int run_encode(int argc, char **argv) {
    const char *model = NULL, *text = NULL, *file = NULL;
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
    char *content = NULL;
    size_t content_len = 0;
    if (text) {
        content = (char *)text;
        content_len = strlen(text);
    } else {
        FILE *fp = fopen(file, "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", file); tokenizer_free(tok); return 1; }
        fseek(fp, 0, SEEK_END);
        long sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);
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
            cap *= 2;
            ids = (int *)realloc(ids, cap * sizeof(int));
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
