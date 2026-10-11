/* trainer.c - Unigram training.
 *
 * Pipeline (design doc §5):
 *   1. seed vocabulary: character n-gram (byte-level) frequency counts from
 *      the normalized corpus, capped at max_candidates.
 *   2. EM iterations:
 *        E step : Viterbi optimal segmentation per sequence, accumulate
 *                 piece usage counts.
 *        M step : re-estimate P(t) = count(t) / sum(counts), score = log P.
 *        prune  : keep the top-k pieces by score (k shrinks towards the
 *                 target vocab size), special tokens & byte-fallback pieces
 *                 are always retained.
 *   3. convergence: vocab reached target size and log-likelihood gain stayed
 *      below em_eps for min_rounds consecutive rounds (or max_iters reached).
 *   4. serialize to binary model + optional JSON export.
 */
#include "tokenizer_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAIN_MAX_LINE 4096 /* per-sequence byte cap (design doc §5.4) */

/* ------------------------------------------------------------------ */
/* corpus                                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    char **lines;
    size_t *lens;
    size_t count;
    size_t cap;
} corpus_t;

static int corpus_load(const char *path, corpus_t *c) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tokenizer_set_error("train: cannot open corpus %s", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        tokenizer_set_error("train: empty corpus");
        return -1;
    }
    char *data = (char *)malloc((size_t)sz + 1);
    if (!data) {
        fclose(fp);
        tokenizer_set_error("train: OOM reading corpus");
        return -1;
    }
    size_t rd = fread(data, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) {
        free(data);
        tokenizer_set_error("train: short read on corpus");
        return -1;
    }
    data[rd] = '\0';

    c->lines = NULL;
    c->lens = NULL;
    c->count = 0;
    c->cap = 0;

    /* skip UTF-8 BOM if present */
    size_t start = 0;
    if (rd >= 3 && (unsigned char)data[0] == 0xEF &&
        (unsigned char)data[1] == 0xBB && (unsigned char)data[2] == 0xBF) {
        start = 3;
    }
    for (size_t i = 0; i <= rd; i++) {
        if (i == rd || data[i] == '\n') {
            size_t end = i;
            while (end > start && data[end - 1] == '\r') end--;
            if (end - start > 0) {
                if (c->count >= c->cap) {
                    size_t ncap = c->cap ? c->cap * 2 : 1024;
                    char **nl = (char **)realloc(c->lines, ncap * sizeof(char *));
                    if (!nl) goto oom;
                    c->lines = nl;
                    size_t *nlens = (size_t *)realloc(c->lens, ncap * sizeof(size_t));
                    if (!nlens) goto oom;
                    c->lens = nlens;
                    c->cap = ncap;
                }
                size_t llen = end - start;
                if (llen > TRAIN_MAX_LINE) {
                    llen = TRAIN_MAX_LINE; /* cap per-sequence length */
                }
                char *copy = (char *)malloc(llen + 1);
                if (!copy) goto oom;
                memcpy(copy, data + start, llen);
                copy[llen] = '\0';
                c->lines[c->count] = copy;
                c->lens[c->count] = llen;
                c->count++;
            }
            start = i + 1;
        }
    }
    free(data);
    return 0;
oom:
    free(data);
    for (size_t i = 0; i < c->count; i++) free(c->lines[i]);
    free(c->lines);
    free(c->lens);
    tokenizer_set_error("train: OOM in corpus_load");
    return -1;
}

static void corpus_free(corpus_t *c) {
    for (size_t i = 0; i < c->count; i++) free(c->lines[i]);
    free(c->lines);
    free(c->lens);
    memset(c, 0, sizeof(*c));
}

/* ------------------------------------------------------------------ */
/* byte-string hash map (FNV-1a, open addressing)                      */
/* ------------------------------------------------------------------ */
typedef struct {
    char *key;
    size_t key_len;
    double count;
    int occupied;
} hash_entry_t;

typedef struct {
    hash_entry_t *entries;
    size_t cap;      /* power of two */
    size_t size;     /* occupied entries */
    size_t max_size; /* hard cap of distinct keys */
    int full;        /* max_size reached: do not insert new keys */
} strhash_t;

static size_t fnv1a(const char *s, size_t n) {
    size_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static strhash_t *hash_new(size_t cap, size_t max_size) {
    strhash_t *h = (strhash_t *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    size_t c = 65536;
    while (c < cap) c *= 2;
    h->entries = (hash_entry_t *)calloc(c, sizeof(hash_entry_t));
    if (!h->entries) {
        free(h);
        return NULL;
    }
    h->cap = c;
    h->size = 0;
    h->max_size = max_size;
    h->full = 0;
    return h;
}

static void hash_free(strhash_t *h) {
    if (!h) return;
    for (size_t i = 0; i < h->cap; i++) {
        if (h->entries[i].occupied) free(h->entries[i].key);
    }
    free(h->entries);
    free(h);
}

/* Grow to at least 2x. Returns 0 on success. */
static int hash_grow(strhash_t *h) {
    size_t ncap = h->cap * 2;
    hash_entry_t *ne = (hash_entry_t *)calloc(ncap, sizeof(hash_entry_t));
    if (!ne) return -1;
    for (size_t i = 0; i < h->cap; i++) {
        if (!h->entries[i].occupied) continue;
        size_t idx = fnv1a(h->entries[i].key, h->entries[i].key_len) & (ncap - 1);
        while (ne[idx].occupied) idx = (idx + 1) & (ncap - 1);
        ne[idx] = h->entries[i];
    }
    free(h->entries);
    h->entries = ne;
    h->cap = ncap;
    return 0;
}

/* Returns pointer to the count slot for key, or NULL when the hash is full
 * and the key is new (key not inserted). */
static double *hash_get_or_create(strhash_t *h, const char *key, size_t len) {
    if (h->cap * 3 < h->size * 4) {
        if (hash_grow(h) != 0) return NULL;
    }
    size_t idx = fnv1a(key, len) & (h->cap - 1);
    while (1) {
        hash_entry_t *e = &h->entries[idx];
        if (!e->occupied) {
            if (h->full) return NULL;
            if (h->size >= h->max_size) {
                h->full = 1;
                return NULL;
            }
            char *k = (char *)malloc(len + 1);
            if (!k) return NULL;
            memcpy(k, key, len);
            k[len] = '\0';
            e->key = k;
            e->key_len = len;
            e->count = 0.0;
            e->occupied = 1;
            h->size++;
            return &e->count;
        }
        if (e->key_len == len && memcmp(e->key, key, len) == 0) {
            return &e->count;
        }
        idx = (idx + 1) & (h->cap - 1);
    }
}

/* Collect all entries into parallel arrays (caller frees). */
static size_t hash_collect(const strhash_t *h, char ***keys, size_t **lens,
                           double **counts) {
    char **ks = (char **)malloc(h->size * sizeof(char *));
    size_t *ls = (size_t *)malloc(h->size * sizeof(size_t));
    double *cs = (double *)malloc(h->size * sizeof(double));
    if (!ks || !ls || !cs) {
        free(ks); free(ls); free(cs);
        return 0;
    }
    size_t n = 0;
    for (size_t i = 0; i < h->cap; i++) {
        if (h->entries[i].occupied) {
            ks[n] = h->entries[i].key;
            ls[n] = h->entries[i].key_len;
            cs[n] = h->entries[i].count;
            n++;
        }
    }
    *keys = ks;
    *lens = ls;
    *counts = cs;
    return n;
}

/* ------------------------------------------------------------------ */
/* seed vocabulary                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    char **pieces;
    int *lens;
    double *counts; /* frequency for initial score estimation */
    int size;
    int cap;
} seed_t;

static int seed_push(seed_t *s, const char *piece, size_t len, double count) {
    if (s->size >= s->cap) {
        int ncap = s->cap ? s->cap * 2 : 4096;
        char **np = (char **)realloc(s->pieces, (size_t)ncap * sizeof(char *));
        if (!np) return -1;
        s->pieces = np;
        int *nl = (int *)realloc(s->lens, (size_t)ncap * sizeof(int));
        if (!nl) return -1;
        s->lens = nl;
        double *nc = (double *)realloc(s->counts, (size_t)ncap * sizeof(double));
        if (!nc) return -1;
        s->counts = nc;
        s->cap = ncap;
    }
    char *copy = (char *)malloc(len + 1);
    if (!copy) return -1;
    memcpy(copy, piece, len);
    copy[len] = '\0';
    s->pieces[s->size] = copy;
    s->lens[s->size] = (int)len;
    s->counts[s->size] = count;
    s->size++;
    return 0;
}

static void seed_free(seed_t *s) {
    for (int i = 0; i < s->size; i++) free(s->pieces[i]);
    free(s->pieces);
    free(s->lens);
    free(s->counts);
    memset(s, 0, sizeof(*s));
}

typedef struct {
    char *key;
    size_t len;
    double count;
} seed_sort_item_t;

static int seed_sort_item_cmp(const void *a, const void *b) {
    const seed_sort_item_t *x = (const seed_sort_item_t *)a;
    const seed_sort_item_t *y = (const seed_sort_item_t *)b;
    if (y->count != x->count) return y->count > x->count ? 1 : -1;
    if (y->len != x->len) return y->len > x->len ? 1 : -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* piece helpers                                                       */
/* ------------------------------------------------------------------ */
static int is_special_id(int id, int unk, int bos, int eos, int pad) {
    return id == unk || id == bos || id == eos || id == pad;
}

static int is_bytefb_style(const char *p, int len) {
    if (len != 6 || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>')
        return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Viterbi segmentation over a vocab trie (shared with inference)      */
/* ------------------------------------------------------------------ */
typedef struct {
    int *ids;      /* malloc'ed path (ids >= vocab_size mark byte fallback) */
    size_t cnt;
    double ll;
} seg_t;

static void seg_free(seg_t *s) {
    free(s->ids);
    s->ids = NULL;
    s->cnt = 0;
}

static int viterbi_segment(const trie_t *trie, const int vocab_size,
                           const float *scores, const char *s, size_t n,
                           seg_t *out) {
    const double kFbScore = -10.0;
    double *dp = (double *)malloc((n + 1) * sizeof(double));
    int *best = (int *)malloc((n + 1) * sizeof(int));
    size_t *prev = (size_t *)malloc((n + 1) * sizeof(size_t));
    int *path = (int *)malloc((n + 1) * sizeof(int));
    if (!dp || !best || !prev || !path) {
        free(dp); free(best); free(prev); free(path);
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
        if (dp[i] == -INFINITY) dp[i] = (i == 0) ? 0.0 : dp[i - 1] + kFbScore;
        int m = trie_match_prefixes(trie, s, n, i, ids, ends, MAX_MATCH);
        int any = 0;
        for (int k = 0; k < m; k++) {
            size_t j = ends[k];
            int pid = ids[k];
            if (pid >= vocab_size) continue;
            double cand = dp[i] + (double)scores[pid];
            if (cand > dp[j]) {
                dp[j] = cand;
                best[j] = pid;
                prev[j] = i;
                any = 1;
            }
        }
        if (dp[i + 1] == -INFINITY && !any) {
            dp[i + 1] = dp[i] + kFbScore;
            best[i + 1] = vocab_size + (unsigned char)s[i]; /* fallback marker */
            prev[i + 1] = i;
        }
    }
    size_t cnt = 0;
    for (size_t i = n; i > 0;) {
        path[cnt++] = best[i];
        i = prev[i];
    }
    int *res = (int *)malloc((cnt ? cnt : 1) * sizeof(int));
    if (!res) {
        free(dp); free(best); free(prev); free(path);
        return -1;
    }
    for (size_t k = 0; k < cnt; k++) res[k] = path[cnt - 1 - k];
    out->ids = res;
    out->cnt = cnt;
    out->ll = dp[n];
    free(dp); free(best); free(prev); free(path);
    return 0;
}

/* ------------------------------------------------------------------ */
/* checkpoint / resume (train can be interrupted & continued)          */
/* ------------------------------------------------------------------ */
#define CKPT_MAGIC "TKCH"
#define CKPT_VERSION 1u

/* Binary layout (little-endian native):
 *   magic[4] + version(u32)
 *   vocab_target(u64) max_piece_len(i32) max_candidates(i32)
 *   max_iters(i32) em_eps(f64) shrink_pct(i32) min_rounds(i32)
 *   add_fb(i32) fold_case(i32)
 *   vsz(i32) round(i32) n_rounds_at_target(i32) prev_ll(f64)
 *   for each piece: plen(i32) bytes float score
 */
static int ckpt_save(const tokenizer_train_options *opts, const char *path,
                     int vsz, char **pieces, int *plens, float *scores,
                     int round, int n_rounds_at_target, double prev_ll) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        tokenizer_set_error("train: cannot open checkpoint %s", tmp);
        return -1;
    }
    fwrite(CKPT_MAGIC, 1, 4, fp);
    uint32_t ver = CKPT_VERSION;
    fwrite(&ver, 1, sizeof(ver), fp);
    uint64_t vt = (uint64_t)(opts->vocab_size ? opts->vocab_size : 4096);
    fwrite(&vt, 1, sizeof(vt), fp);
    int32_t mpl = opts->max_piece_len ? opts->max_piece_len : 8;
    fwrite(&mpl, 1, sizeof(mpl), fp);
    int32_t mc = opts->max_candidates ? (int32_t)opts->max_candidates : 1000000;
    fwrite(&mc, 1, sizeof(mc), fp);
    int32_t mi = opts->max_iters ? opts->max_iters : 10;
    fwrite(&mi, 1, sizeof(mi), fp);
    double ee = opts->em_eps > 0 ? opts->em_eps : 1e-4;
    fwrite(&ee, 1, sizeof(ee), fp);
    int32_t sp = opts->shrink_pct ? opts->shrink_pct : 60;
    fwrite(&sp, 1, sizeof(sp), fp);
    int32_t mr = opts->min_rounds ? opts->min_rounds : 3;
    fwrite(&mr, 1, sizeof(mr), fp);
    int32_t af = opts->add_byte_fallback;
    fwrite(&af, 1, sizeof(af), fp);
    int32_t fc = opts->fold_case;
    fwrite(&fc, 1, sizeof(fc), fp);
    int32_t vs = (int32_t)vsz;
    fwrite(&vs, 1, sizeof(vs), fp);
    int32_t rd = (int32_t)round;
    fwrite(&rd, 1, sizeof(rd), fp);
    int32_t nr = (int32_t)n_rounds_at_target;
    fwrite(&nr, 1, sizeof(nr), fp);
    double pl = prev_ll;
    fwrite(&pl, 1, sizeof(pl), fp);
    for (int i = 0; i < vsz; i++) {
        int32_t l = (int32_t)plens[i];
        fwrite(&l, 1, sizeof(l), fp);
        fwrite(pieces[i], 1, (size_t)l, fp);
        fwrite(&scores[i], 1, sizeof(float), fp);
    }
    if (ferror(fp)) {
        fclose(fp);
        remove(tmp);
        tokenizer_set_error("train: write error on checkpoint");
        return -1;
    }
    if (fclose(fp) != 0) {
        remove(tmp);
        tokenizer_set_error("train: close error on checkpoint");
        return -1;
    }
    /* Windows rename() cannot overwrite an existing file */
    remove(path);
    if (rename(tmp, path) != 0) {
        remove(tmp);
        tokenizer_set_error("train: cannot finalize checkpoint %s", path);
        return -1;
    }
    return 0;
}

/* Returns 1 if a valid checkpoint exists at path (magic ok), 0 otherwise. */
static int ckpt_exists(const char *path) {
    if (!path || !*path) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    char magic[4];
    int ok = (fread(magic, 1, 4, fp) == 4 && memcmp(magic, CKPT_MAGIC, 4) == 0);
    fclose(fp);
    return ok;
}

/* Load checkpoint; validates parameters against opts. On success caller
 * owns pieces/plens/scores. Returns 0 on success, -1 on failure. */
static int ckpt_load(const tokenizer_train_options *opts, const char *path,
                     int *vsz_out, char ***pieces_out, int **plens_out,
                     float **scores_out, int *round_out,
                     int *n_rounds_at_target_out, double *prev_ll_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tokenizer_set_error("train: cannot open checkpoint %s", path);
        return -1;
    }
    char magic[4];
    uint32_t ver;
    uint64_t vt;
    int32_t mpl, mc, mi, sp, mr, af, fc, vs, rd, nr;
    double ee, pl;
    size_t got = 0;
    got += fread(magic, 1, 4, fp) == 4 ? 4 : 0;
    got += fread(&ver, 1, sizeof(ver), fp) == sizeof(ver) ? sizeof(ver) : 0;
    got += fread(&vt, 1, sizeof(vt), fp) == sizeof(vt) ? sizeof(vt) : 0;
    got += fread(&mpl, 1, sizeof(mpl), fp) == sizeof(mpl) ? sizeof(mpl) : 0;
    got += fread(&mc, 1, sizeof(mc), fp) == sizeof(mc) ? sizeof(mc) : 0;
    got += fread(&mi, 1, sizeof(mi), fp) == sizeof(mi) ? sizeof(mi) : 0;
    got += fread(&ee, 1, sizeof(ee), fp) == sizeof(ee) ? sizeof(ee) : 0;
    got += fread(&sp, 1, sizeof(sp), fp) == sizeof(sp) ? sizeof(sp) : 0;
    got += fread(&mr, 1, sizeof(mr), fp) == sizeof(mr) ? sizeof(mr) : 0;
    got += fread(&af, 1, sizeof(af), fp) == sizeof(af) ? sizeof(af) : 0;
    got += fread(&fc, 1, sizeof(fc), fp) == sizeof(fc) ? sizeof(fc) : 0;
    got += fread(&vs, 1, sizeof(vs), fp) == sizeof(vs) ? sizeof(vs) : 0;
    got += fread(&rd, 1, sizeof(rd), fp) == sizeof(rd) ? sizeof(rd) : 0;
    got += fread(&nr, 1, sizeof(nr), fp) == sizeof(nr) ? sizeof(nr) : 0;
    got += fread(&pl, 1, sizeof(pl), fp) == sizeof(pl) ? sizeof(pl) : 0;
    size_t need = 4 + sizeof(ver) + sizeof(vt) + 9 * sizeof(int32_t) +
                  sizeof(double) + sizeof(double);
    if (got < need) {
        fclose(fp);
        tokenizer_set_error("train: truncated checkpoint %s", path);
        return -1;
    }
    if (memcmp(magic, CKPT_MAGIC, 4) != 0 || ver != CKPT_VERSION) {
        fclose(fp);
        tokenizer_set_error("train: unsupported checkpoint %s", path);
        return -1;
    }
    size_t vtarget = opts->vocab_size ? opts->vocab_size : 4096;
    int pmpl = opts->max_piece_len ? opts->max_piece_len : 8;
    size_t pmc = opts->max_candidates ? (size_t)opts->max_candidates : 1000000;
    double pee = opts->em_eps > 0 ? opts->em_eps : 1e-4;
    int psp = opts->shrink_pct ? opts->shrink_pct : 60;
    int pmr = opts->min_rounds ? opts->min_rounds : 3;
    /* max_iters is intentionally excluded from the consistency check so a
     * resume can continue past the previously planned number of rounds. */
    if (vt != (uint64_t)vtarget || mpl != pmpl || (size_t)mc != pmc ||
        ee != pee || sp != psp || mr != pmr ||
        af != opts->add_byte_fallback || fc != opts->fold_case) {
        fclose(fp);
        tokenizer_set_error("train: checkpoint parameters mismatch with current options");
        return -1;
    }
    if (vs <= 0 || vs > 100000000) {
        fclose(fp);
        tokenizer_set_error("train: checkpoint vsz out of range");
        return -1;
    }
    char **ps = (char **)calloc((size_t)vs, sizeof(char *));
    int *ls = (int *)malloc((size_t)vs * sizeof(int));
    float *sc = (float *)malloc((size_t)vs * sizeof(float));
    if (!ps || !ls || !sc) {
        free(ps); free(ls); free(sc);
        fclose(fp);
        tokenizer_set_error("train: OOM loading checkpoint");
        return -1;
    }
    for (int i = 0; i < vs; i++) {
        int32_t l;
        if (fread(&l, 1, sizeof(l), fp) != sizeof(l) ||
            l < 0 || l > (int32_t)(1 << 20)) {
            for (int j = 0; j < i; j++) free(ps[j]);
            free(ps); free(ls); free(sc);
            fclose(fp);
            tokenizer_set_error("train: corrupt checkpoint piece header");
            return -1;
        }
        char *buf = (char *)malloc((size_t)l + 1);
        if (!buf) {
            for (int j = 0; j < i; j++) free(ps[j]);
            free(ps); free(ls); free(sc);
            fclose(fp);
            tokenizer_set_error("train: OOM loading checkpoint piece");
            return -1;
        }
        if (fread(buf, 1, (size_t)l, fp) != (size_t)l) {
            free(buf);
            for (int j = 0; j < i; j++) free(ps[j]);
            free(ps); free(ls); free(sc);
            fclose(fp);
            tokenizer_set_error("train: corrupt checkpoint piece data");
            return -1;
        }
        buf[l] = '\0';
        float s;
        if (fread(&s, 1, sizeof(s), fp) != sizeof(s)) {
            free(buf);
            for (int j = 0; j < i; j++) free(ps[j]);
            free(ps); free(ls); free(sc);
            fclose(fp);
            tokenizer_set_error("train: corrupt checkpoint piece score");
            return -1;
        }
        ps[i] = buf;
        ls[i] = (int)l;
        sc[i] = s;
    }
    fclose(fp);
    *vsz_out = (int)vs;
    *pieces_out = ps;
    *plens_out = ls;
    *scores_out = sc;
    *round_out = (int)rd;
    *n_rounds_at_target_out = (int)nr;
    *prev_ll_out = pl;
    return 0;
}

/* ------------------------------------------------------------------ */
/* train                                                               */
/* ------------------------------------------------------------------ */
int tokenizer_train(const tokenizer_train_options *opts) {
    if (!opts || !opts->corpus_path || !opts->model_path) {
        tokenizer_set_error("train: corpus_path and model_path required");
        return -1;
    }
    size_t vocab_target = opts->vocab_size ? opts->vocab_size : 4096;
    int max_piece_len = opts->max_piece_len ? opts->max_piece_len : 8;
    size_t max_candidates = opts->max_candidates ? (size_t)opts->max_candidates : 1000000;
    int max_iters = opts->max_iters ? opts->max_iters : 10;
    double em_eps = opts->em_eps > 0 ? opts->em_eps : 1e-4;
    int shrink_pct = opts->shrink_pct ? opts->shrink_pct : 60;
    int min_rounds = opts->min_rounds ? opts->min_rounds : 3;
    int add_fb = opts->add_byte_fallback;
    int fold_case = opts->fold_case;
    int verbose = opts->verbose;

    const int UNK = 0, BOS = 1, EOS = 2, PAD = 3;
    const float kSpecialScore = -100.0f;
    const float kFbScore = -10.0f;

    /* ---- load & normalize corpus ---- */
    corpus_t corpus;
    if (corpus_load(opts->corpus_path, &corpus) != 0) return -1;
    if (verbose) fprintf(stderr, "[train] corpus lines: %zu\n", corpus.count);

    /* ---- optional resume from checkpoint ---- */
    /* EM working state; on resume, vsz/pieces/plens/scores come from the
     * checkpoint instead of re-seeding, and the round loop starts at the
     * saved round. */
    int vsz = 0;
    char **pieces = NULL;
    int *plens = NULL;
    float *scores = NULL;
    double *counts = NULL;
    int n_rounds_at_target = 0;
    double prev_ll = -INFINITY;
    int round = 0;
    int resumed = 0;
    if (opts->checkpoint_path && *opts->checkpoint_path &&
        ckpt_exists(opts->checkpoint_path)) {
        if (ckpt_load(opts, opts->checkpoint_path, &vsz, &pieces, &plens,
                      &scores, &round, &n_rounds_at_target, &prev_ll) != 0) {
            corpus_free(&corpus);
            return -1;
        }
        resumed = 1;
        if (verbose)
            fprintf(stderr,
                    "[train] resumed from checkpoint %s (round=%d vsz=%d)\n",
                    opts->checkpoint_path, round, vsz);
    }

    /* ---- seed n-gram counting (skipped when resuming) ---- */
    strhash_t *h = NULL;
    double total_ngram = 0.0;
    seed_t seed;
    memset(&seed, 0, sizeof(seed));
    int n_fb = 0;
    if (resumed) {
        /* byte-fallback count is deterministic when enabled */
        n_fb = add_fb ? 256 : 0;
    }
    if (!resumed) {
    if (verbose) fprintf(stderr, "[train] seeding...\n");
    h = hash_new(65536, max_candidates);
    if (!h) {
        corpus_free(&corpus);
        tokenizer_set_error("train: OOM creating seed hash");
        return -1;
    }
    for (size_t li = 0; li < corpus.count; li++) {
        size_t nlen = 0;
        char *norm = normalize_text(corpus.lines[li], corpus.lens[li],
                                    1 /*mark_spaces*/, fold_case, &nlen);
        if (!norm) continue;
        if (verbose) fprintf(stderr, "[train] seed line %zu norm_len=%zu\n", li, nlen);
        for (size_t i = 0; i < nlen; i++) {
            /* only start windows at UTF-8 character boundaries */
            int lead = utf8_seq_len((uint8_t)norm[i]);
            if (lead == 0) {
                if ((unsigned char)norm[i] < 0x80) {
                    lead = 1; /* ASCII byte: single-char boundary */
                } else {
                    continue; /* continuation byte: not a boundary */
                }
            }
            int inserted_this_pos = 0;
            size_t wlen = 0;
            for (size_t j = i; j < nlen;) {
                int s = utf8_seq_len((uint8_t)norm[j]);
                if (s == 0) s = 1;
                size_t nb = (size_t)s;
                if (j + nb > nlen) break;
                wlen += nb;
                if (wlen > (size_t)max_piece_len) break;
                double *cnt = hash_get_or_create(h, norm + i, wlen);
                if (cnt) {
                    *cnt += 1.0;
                    total_ngram += 1.0;
                    inserted_this_pos = 1;
                } else if (h->full) {
                    break; /* hash full: stop extending at this position */
                } else {
                    break; /* allocation failure: stop extending */
                }
                j += nb;
            }
            (void)inserted_this_pos;
        }
        free(norm);
    }
    if (verbose)
        fprintf(stderr, "[train] seed distinct n-grams: %zu (count total %.0f)\n",
                h->size, total_ngram);

    /* ---- sort seed by frequency, truncate to max_candidates ---- */
    char **keys = NULL;
    size_t *klen = NULL;
    double *kcount = NULL;
    size_t nkeys = hash_collect(h, &keys, &klen, &kcount);
    seed_sort_item_t *items = (seed_sort_item_t *)malloc((nkeys ? nkeys : 1) * sizeof(*items));
    if (!items) {
        free(keys); free(klen); free(kcount);
        hash_free(h); corpus_free(&corpus);
        tokenizer_set_error("train: OOM sorting seeds");
        return -1;
    }
    for (size_t i = 0; i < nkeys; i++) {
        items[i].key = keys[i];
        items[i].len = klen[i];
        items[i].count = kcount[i];
    }
    qsort(items, nkeys, sizeof(*items), seed_sort_item_cmp);
    size_t keep = nkeys < max_candidates ? nkeys : max_candidates;

    /* special tokens first (fixed ids) */
    seed_push(&seed, "<unk>", 5, 0.0);
    seed_push(&seed, "<s>", 3, 0.0);
    seed_push(&seed, "</s>", 4, 0.0);
    seed_push(&seed, "<pad>", 5, 0.0);
    for (size_t i = 0; i < keep; i++) {
        if (seed_push(&seed, items[i].key, items[i].len, items[i].count) != 0) {
            break;
        }
    }
    free(items);
    free(keys); free(klen); free(kcount);
    hash_free(h);

    /* ---- append byte-fallback placeholders (optional) ---- */
    if (add_fb) {
        for (int b = 0; b < 256; b++) {
            char fb[6];
            static const char hexd[] = "0123456789ABCDEF";
            fb[0] = '<'; fb[1] = '0'; fb[2] = 'x';
            fb[3] = hexd[(b >> 4) & 0xF]; fb[4] = hexd[b & 0xF]; fb[5] = '>';
            if (seed_push(&seed, fb, 6, 0.0) != 0) break;
            n_fb++;
        }
    }
    if (verbose) fprintf(stderr, "[train] seed vocab: %d (byte-fb: %d)\n",
                        seed.size, n_fb);
    } /* end if (!resumed) */

    /* ---- EM ---- */
    /* work copy: pieces/lens/scores/counts over EM lifetime.
     * Fresh start: arrays come from the seed. Resume: arrays come from the
     * checkpoint; only counts needs a fresh allocation (E step zeroes it). */
    if (!resumed) {
        vsz = seed.size;
        pieces = seed.pieces;
        plens = seed.lens;
        scores = (float *)malloc((size_t)vsz * sizeof(float));
        counts = (double *)calloc((size_t)vsz, sizeof(double));
        if (!scores || !counts) {
            seed_free(&seed);
            corpus_free(&corpus);
            tokenizer_set_error("train: OOM allocating EM buffers");
            return -1;
        }
        /* initial scores from seed frequencies */
        double sum_freq = 0.0;
        for (int i = 4; i < vsz; i++) sum_freq += seed.counts[i];
        for (int i = 0; i < vsz; i++) {
            if (is_special_id(i, UNK, BOS, EOS, PAD)) {
                scores[i] = kSpecialScore;
            } else if (is_bytefb_style(pieces[i], plens[i])) {
                scores[i] = kFbScore;
            } else {
                double p = sum_freq > 0 ? seed.counts[i] / sum_freq : 1e-9;
                scores[i] = (float)log(p);
            }
        }
    } else {
        counts = (double *)calloc((size_t)vsz, sizeof(double));
        if (!counts) {
            for (int i = 0; i < vsz; i++) free(pieces[i]);
            free(pieces); free(plens); free(scores);
            corpus_free(&corpus);
            tokenizer_set_error("train: OOM allocating EM buffers");
            return -1;
        }
    }

    trie_t *trie = trie_new();
    if (!trie) {
        if (resumed) {
            for (int i = 0; i < vsz; i++) free(pieces[i]);
            free(pieces); free(plens); free(scores); free(counts);
        } else {
            seed_free(&seed); free(scores); free(counts);
        }
        corpus_free(&corpus);
        tokenizer_set_error("train: OOM creating trie");
        return -1;
    }
    if (verbose) fprintf(stderr, "[train] EM start vsz=%d target=%zu\n", vsz, vocab_target);
    for (; round < max_iters; round++) {
        /* rebuild trie from current vocab (skip byte-fb placeholders) */
        trie_free(trie);
        trie = trie_new();
        if (!trie) break;
        for (int i = 0; i < vsz; i++) {
            if (is_bytefb_style(pieces[i], plens[i])) continue;
            trie_insert(trie, pieces[i], (size_t)plens[i], i);
        }

        /* E step */
        for (int i = 0; i < vsz; i++) counts[i] = 0.0;
        double ll = 0.0;
        for (size_t li = 0; li < corpus.count; li++) {
            size_t nlen = 0;
            char *norm = normalize_text(corpus.lines[li], corpus.lens[li],
                                        1 /*mark_spaces*/, fold_case, &nlen);
            if (!norm) continue;
            if (nlen > TRAIN_MAX_LINE) nlen = TRAIN_MAX_LINE;
            seg_t seg = { NULL, 0, 0.0 };
            if (viterbi_segment(trie, vsz, scores, norm, nlen, &seg) != 0) {
                free(norm);
                continue;
            }
            ll += seg.ll;
            for (size_t k = 0; k < seg.cnt; k++) {
                int id = seg.ids[k];
                if (id >= 0 && id < vsz &&
                    !is_special_id(id, UNK, BOS, EOS, PAD) &&
                    !is_bytefb_style(pieces[id], plens[id])) {
                    counts[id] += 1.0;
                }
            }
            seg_free(&seg);
            free(norm);
        }

        /* M step: re-estimate scores */
        double sumc = 0.0;
        for (int i = 0; i < vsz; i++) sumc += counts[i];
        for (int i = 0; i < vsz; i++) {
            if (is_special_id(i, UNK, BOS, EOS, PAD)) {
                scores[i] = kSpecialScore;
            } else if (is_bytefb_style(pieces[i], plens[i])) {
                scores[i] = kFbScore;
            } else {
                double p = sumc > 0 ? counts[i] / sumc : 1e-9;
                scores[i] = (float)log(p);
            }
        }

        /* prune toward target */
        if (vsz > (int)vocab_target) {
            /* index sort: specials first, byte-fb last, others by score desc */
            int *order = (int *)malloc((size_t)vsz * sizeof(int));
            if (!order) break;
            for (int i = 0; i < vsz; i++) order[i] = i;
            /* simple insertion of comparator via qsort_r-like closure: use a
             * static context through the callback-free approach: sort manually
             * with qsort and a comparator that consults globals is unsafe, so
             * we do a stable manual selection below instead. */
            /* --- manual ordering by category --- */
            /* bucket: special(0), normal(1), bytefb(2) */
            int *buckets[3];
            int bn[3] = {0, 0, 0};
            for (int i = 0; i < vsz; i++) {
                int b;
                if (is_special_id(i, UNK, BOS, EOS, PAD)) b = 0;
                else if (is_bytefb_style(pieces[i], plens[i])) b = 2;
                else b = 1;
                bn[b]++;
            }
            for (int b = 0; b < 3; b++) {
                buckets[b] = (int *)malloc((size_t)(bn[b] ? bn[b] : 1) * sizeof(int));
                if (!buckets[b]) {
                    for (int t = 0; t < 3; t++) free(buckets[t]);
                    free(order);
                    order = NULL;
                    break;
                }
            }
            if (!order) break;
            int oi[3] = {0, 0, 0};
            for (int i = 0; i < vsz; i++) {
                int b;
                if (is_special_id(i, UNK, BOS, EOS, PAD)) b = 0;
                else if (is_bytefb_style(pieces[i], plens[i])) b = 2;
                else b = 1;
                buckets[b][oi[b]++] = i;
            }
            /* sort normal bucket by score desc (qsort needs comparator with
             * access to scores -> use a global pointer workaround is unsafe;
             * instead sort (score,id) pairs). */
            if (bn[1] > 1) {
                typedef struct { float score; int id; } pair_t;
                pair_t *pairs = (pair_t *)malloc((size_t)bn[1] * sizeof(pair_t));
                if (pairs) {
                    for (int i = 0; i < bn[1]; i++) {
                        pairs[i].score = scores[buckets[1][i]];
                        pairs[i].id = buckets[1][i];
                    }
                    /* simple insertion sort by score desc (stable-ish) */
                    for (int i = 1; i < bn[1]; i++) {
                        pair_t key = pairs[i];
                        int j = i - 1;
                        while (j >= 0 && pairs[j].score < key.score) {
                            pairs[j + 1] = pairs[j];
                            j--;
                        }
                        pairs[j + 1] = key;
                    }
                    for (int i = 0; i < bn[1]; i++) buckets[1][i] = pairs[i].id;
                    free(pairs);
                }
            }
            /* concat: specials + top normals + bytefb */
            int pos = 0;
            for (int b = 0; b < 3; b++)
                for (int i = 0; i < bn[b]; i++)
                    order[pos++] = buckets[b][i];
            int nkeep = vsz;
            if (shrink_pct > 0 && vsz > (int)vocab_target) {
                int by_shrink = (int)((size_t)vsz * (size_t)shrink_pct / 100);
                int by_target = (int)vocab_target;
                nkeep = by_shrink > by_target ? by_shrink : by_target;
                if (nkeep < 4 + n_fb) nkeep = 4 + n_fb;
                if (nkeep > vsz) nkeep = vsz;
            }
            /* rebuild arrays in the new order */
            char **np = (char **)malloc((size_t)nkeep * sizeof(char *));
            int *nl = (int *)malloc((size_t)nkeep * sizeof(int));
            float *ns = (float *)malloc((size_t)nkeep * sizeof(float));
            if (!np || !nl || !ns) {
                free(np); free(nl); free(ns);
                for (int t = 0; t < 3; t++) free(buckets[t]);
                free(order);
                break;
            }
            for (int i = 0; i < nkeep; i++) {
                int src = order[i];
                np[i] = pieces[src];
                nl[i] = plens[src];
                ns[i] = scores[src];
                pieces[src] = NULL; /* ownership moved */
            }
            /* free the pruned tail pieces */
            for (int i = nkeep; i < vsz; i++) {
                if (pieces[order[i]]) free(pieces[order[i]]);
            }
            free(pieces); free(plens); free(scores);
            pieces = np;
            plens = nl;
            scores = ns;
            vsz = nkeep;
            /* counts are re-collected from scratch every E step, so simply
             * allocate a zeroed buffer sized to the new vocab; failure to do
             * so aborts training. */
            double *ncounts = (double *)calloc((size_t)vsz, sizeof(double));
            if (!ncounts) {
                free(counts);
                counts = NULL;
                for (int t = 0; t < 3; t++) free(buckets[t]);
                free(order);
                break;
            }
            free(counts);
            counts = ncounts;
            for (int t = 0; t < 3; t++) free(buckets[t]);
            free(order);
        }

        if (verbose)
            fprintf(stderr, "[train] round %d: vocab=%d ll=%.4f\n",
                    round + 1, vsz, ll);

        /* convergence: at target and ll gain small */
        if (vsz <= (int)vocab_target) {
            double gain = prev_ll == -INFINITY ? 0.0 : ll - prev_ll;
            if (gain < em_eps) n_rounds_at_target++;
            else n_rounds_at_target = 0;
            if (n_rounds_at_target >= min_rounds) break;
        }
        prev_ll = ll;

        /* ---- save per-round checkpoint (for resume) ---- */
        if (opts->checkpoint_path && *opts->checkpoint_path) {
            if (ckpt_save(opts, opts->checkpoint_path, vsz, pieces, plens,
                          scores, round + 1, n_rounds_at_target, prev_ll) != 0) {
                if (verbose)
                    fprintf(stderr, "[train] warn: checkpoint save failed: %s\n",
                            tokenizer_last_error() ? tokenizer_last_error()
                                                    : "unknown");
            } else if (verbose) {
                fprintf(stderr, "[train] checkpoint saved (round %d)\n",
                        round + 1);
            }
        }
    }

    /* ---- save model ---- */
    tokenizer_t *tok = tokenizer_create(UNK, BOS, EOS, PAD, add_fb);
    int rc = -1;
    if (tok) {
        for (int i = 0; i < vsz; i++) {
            if (tokenizer_set_piece(tok, i, pieces[i], (size_t)plens[i], scores[i]) != 0) {
                tokenizer_set_error("train: OOM building model");
                goto done;
            }
        }
        if (tokenizer_save_binary(tok, opts->model_path) != 0) goto done;
        if (opts->json_path && *opts->json_path) {
            if (tokenizer_export_json(tok, opts->json_path) != 0) goto done;
        }
        rc = 0;
    }
done:
    if (tok) tokenizer_free(tok);
    for (int i = 0; i < vsz; i++) free(pieces[i]);
    free(pieces);
    free(plens);
    free(scores);
    free(counts);
    trie_free(trie);
    corpus_free(&corpus);
    return rc;
}
