/* bench_enc.c - encoding throughput + Trie lookup benchmark for a trained
 * model (.tkm). Measures:
 *   1. tokenizer_encode() latency per input line (p50/p90/p99/max, ns)
 *      and throughput in MB/s and tokens/s.
 *   2. Trie lookup: build a fresh trie from the model pieces, then
 *      (a) exact trie_lookup() of each piece (ns/op, p50/p99),
 *      (b) trie_match_prefixes() walking a sample text (matches/op).
 * Usage: bench_enc <model.tkm> <sample.txt> [iterations]
 */
#include "../include/tokenizer.h"
#include "../src/trie.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
static double now_s(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }
#endif

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static uint64_t percentile(uint64_t *arr, size_t n, double p) {
    if (n == 0) return 0;
    size_t idx = (size_t)(p * (double)(n - 1));
    return arr[idx];
}

static char **read_lines(const char *path, size_t *nlines, uint64_t *total_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("open sample"); return NULL; }
    char **lines = NULL;
    size_t cap = 0, cnt = 0;
    *total_bytes = 0;
    char buf[65536];
    while (fgets(buf, sizeof(buf), fp)) {
        size_t l = strlen(buf);
        while (l && (buf[l - 1] == '\n' || buf[l - 1] == '\r')) buf[--l] = '\0';
        if (l == 0) continue;
        char *cp = (char *)malloc(l + 1);
        if (!cp) { fclose(fp); return NULL; }
        memcpy(cp, buf, l);
        cp[l] = '\0';
        if (cnt >= cap) {
            size_t ncap = cap ? cap * 2 : 1024;
            char **nl = (char **)realloc(lines, ncap * sizeof(char *));
            if (!nl) { fclose(fp); return NULL; }
            lines = nl;
            cap = ncap;
        }
        lines[cnt++] = cp;
        *total_bytes += l;
    }
    fclose(fp);
    *nlines = cnt;
    return lines;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: bench_enc <model.tkm> <sample.txt> [iters]\n");
        return 1;
    }
    int iters = argc > 3 ? atoi(argv[3]) : 5;
    if (iters < 1) iters = 1;

    tokenizer_t *tok = tokenizer_load(argv[1]);
    if (!tok) {
        fprintf(stderr, "load failed: %s\n", tokenizer_last_error() ? tokenizer_last_error() : "?");
        return 1;
    }
    int vsz = (int)tokenizer_vocab_size(tok);
    fprintf(stderr, "model loaded: vocab=%d\n", vsz);

    size_t nl = 0;
    uint64_t total_bytes = 0;
    char **lines = read_lines(argv[2], &nl, &total_bytes);
    if (!lines || nl == 0) { fprintf(stderr, "no sample lines\n"); return 1; }
    fprintf(stderr, "sample lines=%zu bytes=%" PRIu64 "\n", nl, total_bytes);

    /* ---- 1. encode latency ---- */
    uint64_t *lats = (uint64_t *)malloc(nl * sizeof(uint64_t));
    uint64_t total_ns = 0, total_tok = 0;
    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i < nl; i++) {
            int *ids = NULL;
            size_t n = 0;
            double t0 = now_s();
            tokenizer_encode(tok, lines[i], strlen(lines[i]), &ids, &n);
            double dt = (now_s() - t0) * 1e9;
            lats[i] = (uint64_t)dt;
            total_ns += (uint64_t)dt;
            total_tok += n;
            tokenizer_free_ids(ids);
        }
    }
    qsort(lats, nl, sizeof(uint64_t), cmp_u64);
    double encode_s = (double)total_ns / 1e9;
    printf("=== encode (%d iters) ===\n", iters);
    printf("lines=%zu total_bytes=%" PRIu64 " total_tokens=%" PRIu64 "\n",
           nl, total_bytes, total_tok);
    printf("latency_ns: min=%" PRIu64 " p50=%" PRIu64 " p90=%" PRIu64
           " p99=%" PRIu64 " max=%" PRIu64 "\n",
           lats[0], percentile(lats, nl, 0.50), percentile(lats, nl, 0.90),
           percentile(lats, nl, 0.99), lats[nl - 1]);
    printf("throughput: %.2f MB/s, %.2f lines/s, %.2f tokens/s\n",
           (double)total_bytes / encode_s / 1e6,
           (double)nl * iters / encode_s,
           (double)total_tok / encode_s);
    free(lats);

    /* ---- 2. trie build + exact lookup ---- */
    trie_t *trie = trie_new();
    if (!trie) { fprintf(stderr, "trie_new failed\n"); return 1; }
    double t0 = now_s();
    for (int i = 0; i < vsz; i++) {
        const char *p = NULL;
        size_t pl = 0;
        tokenizer_id_to_piece(tok, i, &p, &pl);
        if (trie_insert(trie, p, pl, i) != 0) { fprintf(stderr, "insert failed\n"); return 1; }
    }
    double build_s = now_s() - t0;
    printf("\n=== trie ===\n");
    printf("nodes=%zu build_time=%.3f s (%.1f ns/piece)\n",
           trie_node_count(trie), build_s, build_s * 1e9 / (double)vsz);

    uint64_t *lt = (uint64_t *)malloc((size_t)vsz * sizeof(uint64_t));
    for (int i = 0; i < vsz; i++) {
        const char *p = NULL;
        size_t pl = 0;
        tokenizer_id_to_piece(tok, i, &p, &pl);
        double t1 = now_s();
        trie_lookup(trie, p, pl);
        lt[i] = (uint64_t)((now_s() - t1) * 1e9);
    }
    qsort(lt, vsz, sizeof(uint64_t), cmp_u64);
    printf("exact lookup: min=%" PRIu64 " ns p50=%" PRIu64 " p90=%" PRIu64
           " p99=%" PRIu64 " max=%" PRIu64 " (%.1f Mop/s)\n",
           lt[0], percentile(lt, vsz, 0.50), percentile(lt, vsz, 0.90),
           percentile(lt, vsz, 0.99), lt[vsz - 1],
           (double)vsz / ((double)(lt[0] + lt[vsz - 1]) / 2e9) > 0 ? 0 : 0);
    /* recompute Mop/s cleanly */
    {
        double avg = 0;
        for (int i = 0; i < vsz; i++) avg += (double)lt[i];
        avg /= vsz;
        printf("exact lookup avg=%.1f ns (~%.2f Mop/s by avg)\n",
               avg, 1e3 / avg);
    }
    free(lt);

    /* ---- 3. prefix match on sample text ---- */
    uint64_t *mt = (uint64_t *)malloc(nl * sizeof(uint64_t));
    uint64_t total_matches = 0, total_steps = 0;
    int maxr = 64;
    int *ids = (int *)malloc((size_t)maxr * sizeof(int));
    size_t *ends = (size_t *)malloc((size_t)maxr * sizeof(size_t));
    for (size_t i = 0; i < nl; i++) {
        const char *s = lines[i];
        size_t len = strlen(s);
        double t1 = now_s();
        size_t pos = 0, steps = 0, matches = 0;
        while (pos < len) {
            int m = trie_match_prefixes(trie, s, len, pos, ids, ends, maxr);
            matches += (size_t)m;
            steps++;
            if (m == 0) pos++;
            else {
                size_t mx = ends[0];
                for (int k = 1; k < m; k++) if (ends[k] > mx) mx = ends[k];
                pos += mx;
            }
        }
        mt[i] = (uint64_t)((now_s() - t1) * 1e9);
        total_matches += matches;
        total_steps += steps;
    }
    qsort(mt, nl, sizeof(uint64_t), cmp_u64);
    printf("\n=== trie prefix match (walk sample) ===\n");
    printf("line cost ns: min=%" PRIu64 " p50=%" PRIu64 " p90=%" PRIu64
           " p99=%" PRIu64 " max=%" PRIu64 "\n",
           mt[0], percentile(mt, nl, 0.50), percentile(mt, nl, 0.90),
           percentile(mt, nl, 0.99), mt[nl - 1]);
    printf("total_match_steps=%" PRIu64 " prefix_matches=%" PRIu64
           " (avg %.2f match/pos)\n",
           total_steps, total_matches,
           (double)total_matches / (double)(total_steps ? total_steps : 1));
    free(mt); free(ids); free(ends);

    for (size_t i = 0; i < nl; i++) free(lines[i]);
    free(lines);
    trie_free(trie);
    tokenizer_free(tok);
    return 0;
}
