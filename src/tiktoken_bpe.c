/* tiktoken_bpe.c - tiktoken-compatible byte-level BPE (implementation).
 *
 * This file is self-contained C11: no JSON library, no regex library, no
 * base64 library.  It implements exactly the semantics described in
 * docs/ttbpe_spec.md section 3:
 *
 *   1. "rank is the id": a merge step always picks the adjacent pair whose
 *      concatenated bytes carry the smallest rank and repeats until no pair is
 *      known.  Frequency plays no role whatsoever.
 *   2. Vocabulary keys are raw byte strings and may be invalid UTF-8
 *      (e.g. b"\xb7", b" \xe4\xb8").
 *   3. Pieces produced by the pre-tokenizer are encoded independently; two
 *      pieces never merge across their boundary.
 *   4. Special tokens are split off first with longest-literal-match; an
 *      allowed special is emitted as its id, a disallowed one either raises an
 *      error (reject_disallowed) or falls through to ordinary encoding.
 *   5. Decoding concatenates the raw bytes of the ids and then performs a
 *      UTF-8 decode that replaces every invalid sequence (per the Unicode
 *      "maximal subpart" rule) with U+FFFD, i.e. tiktoken errors="replace".
 *   6. Empty input yields 0 ids and success; NULL tokenizer or an out-of-range
 *      id fails.
 *   7. A vocabulary that lacks a prefix of a piece never crashes: the encoder
 *      merges whatever it can and reports an error when a leftover part has no
 *      rank at all (tiktoken would raise KeyError here).
 *   8. The piece -> ids result cache can be switched off and never changes the
 *      result, only the amount of recomputation.
 *
 * ============================ APPROXIMATION ==========================
 * The regex based pre-tokenizer (docs/ttbpe_spec.md section 5) is reimplemented
 * without a regex engine, and the character classes are simplified:
 *
 *   - ASCII is exact: whitespace / letters / digits / everything else.
 *   - Non-ASCII: a small set of Unicode whitespace code points is treated as
 *     whitespace, a small set of format/private-use/control code points is
 *     treated as "other", and EVERY remaining non-ASCII code point is treated
 *     as a letter.  Emoji, CJK punctuation and non-ASCII digits therefore join
 *     the letter class instead of being symbols/digits, which is why
 *     tt-split differs from Python `regex.findall` on non-ASCII input while
 *     being 100% identical on pure ASCII input (verified in the report).
 *   - Invalid UTF-8 bytes are classified as "other" and never split a byte
 *     sequence, they just advance one byte at a time.
 *
 * ttbpe_set_char_classifier() installs a real Unicode classifier once a table
 * is available; nothing else in this file has to change.
 * =====================================================================
 */
#include "tiktoken_bpe.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* error reporting (thread-local, mirroring tokenizer_last_error)      */
/* ------------------------------------------------------------------ */

static _Thread_local char g_tls_error[512];

static void tt_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_tls_error, sizeof(g_tls_error), fmt, ap);
    va_end(ap);
}

const char *ttbpe_last_error(void) {
    return g_tls_error[0] ? g_tls_error : NULL;
}

static void tt_clear_error(void) { g_tls_error[0] = '\0'; }

/* ------------------------------------------------------------------ */
/* growable byte buffer                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    unsigned char *p;
    size_t len, cap;
} buf_t;

static void buf_free(buf_t *b) {
    free(b->p);
    b->p = NULL;
    b->len = b->cap = 0;
}

static int buf_reserve(buf_t *b, size_t extra) {
    if (extra <= b->cap - b->len) return 0;
    size_t need = b->len + extra;
    if (need < b->len) return -1; /* overflow */
    size_t ncap = b->cap ? b->cap : 64;
    while (ncap < need) {
        if (ncap > (SIZE_MAX / 2)) return -1;
        ncap *= 2;
    }
    unsigned char *np = (unsigned char *)realloc(b->p, ncap);
    if (!np) return -1;
    b->p = np;
    b->cap = ncap;
    return 0;
}

static int buf_append(buf_t *b, const void *data, size_t n) {
    if (n == 0) return 0;
    if (buf_reserve(b, n) != 0) return -1;
    memcpy(b->p + b->len, data, n);
    b->len += n;
    return 0;
}

static void buf_reset(buf_t *b) { b->len = 0; }

/* ------------------------------------------------------------------ */
/* growable int buffer                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    int *p;
    size_t n, cap;
} idbuf_t;

static int idbuf_push(idbuf_t *b, int v) {
    if (b->n == b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 64;
        int *np = (int *)realloc(b->p, ncap * sizeof(int));
        if (!np) return -1;
        b->p = np;
        b->cap = ncap;
    }
    b->p[b->n++] = v;
    return 0;
}

/* ------------------------------------------------------------------ */
/* binary min-heap over (rank, index)                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t rank;
    int32_t idx;
} heap_item;

typedef struct {
    heap_item *a;
    size_t n, cap;
} minheap_t;

/* Ordering is (rank, index) lexicographic, i.e. the smallest rank wins and
 * ties are broken by the left-most pair, which reproduces the deterministic
 * "first minimal pair" scan order of the tiktoken reference implementation. */
static int heap_less(const heap_item *x, const heap_item *y) {
    if (x->rank != y->rank) return x->rank < y->rank;
    return x->idx < y->idx;
}

static int heap_push(minheap_t *h, int32_t rank, int32_t idx) {
    if (h->n + 1 > h->cap) {
        size_t ncap = h->cap ? h->cap * 2 : 64;
        heap_item *na = (heap_item *)realloc(h->a, ncap * sizeof(heap_item));
        if (!na) return -1;
        h->a = na;
        h->cap = ncap;
    }
    size_t i = h->n++;
    h->a[i].rank = rank;
    h->a[i].idx = idx;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!heap_less(&h->a[i], &h->a[parent])) break;
        heap_item t = h->a[parent];
        h->a[parent] = h->a[i];
        h->a[i] = t;
        i = parent;
    }
    return 0;
}

/* Returns 1 and fills *out when an item was available, 0 on empty. */
static int heap_pop(minheap_t *h, heap_item *out) {
    if (h->n == 0) return 0;
    *out = h->a[0];
    h->a[0] = h->a[--h->n];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < h->n && heap_less(&h->a[l], &h->a[m])) m = l;
        if (r < h->n && heap_less(&h->a[r], &h->a[m])) m = r;
        if (m == i) break;
        heap_item t = h->a[m];
        h->a[m] = h->a[i];
        h->a[i] = t;
        i = m;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* handle                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t off; /* offset into arena */
    uint32_t len;
} span_t;

typedef struct {
    uint32_t off; /* offset into arena */
    uint32_t len;
    int32_t id;
} ht_slot;

typedef struct {
    ht_slot *slots;
    size_t cap; /* power of two, or 0 */
    size_t count;
} ht_t;

typedef struct {
    uint32_t off, len;
    int32_t id;
} special_t;

/* piece -> ids cache entry (chained hash table) */
#define TTBPE_CACHE_MAX_ENTRIES 65536u
#define TTBPE_CACHE_MAX_PIECE_LEN 128u

typedef struct {
    uint32_t koff, klen;
    uint32_t ioff; /* offset into cache_ids */
    uint32_t nids;
    int32_t next; /* chain */
} cache_entry_t;

struct tiktoken_bpe {
    buf_t arena;          /* all vocabulary key bytes */
    span_t *ids;          /* id -> span in arena */
    unsigned char *present;/* id -> defined flag */
    int32_t id_cap;       /* allocated slots in ids/present */
    int32_t max_id;       /* highest defined id */
    int32_t vocab_size;   /* reported vocabulary size */
    ht_t ht;              /* key bytes -> id */

    special_t *specials;  /* sorted by length, longest first */
    size_t n_specials, cap_specials;
    size_t min_special_len;
    int32_t next_special_id;

    char format[16];      /* "tok.json" | "tiktoken" */
    char *word_split;     /* JSON field, may be NULL */
    int split_mode;
    int cache_enabled;

    cache_entry_t *cache;
    size_t n_cache;
    int32_t *cache_head;
    size_t cache_head_cap;
    int32_t *cache_ids;
    size_t cache_ids_len, cache_ids_cap;
};

/* global (not per handle) code point classifier hook */
static int (*g_classifier)(uint32_t cp) = NULL;

void ttbpe_set_char_classifier(int (*fn)(uint32_t cp)) { g_classifier = fn; }

/* ------------------------------------------------------------------ */
/* vocabulary hash table                                               */
/* ------------------------------------------------------------------ */

static uint64_t hash_bytes(const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL; /* FNV-1a 64 */
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Place (off,len,id) into an explicit slot array.  Duplicate key bytes update
 * the stored id ("last definition wins", exactly like the Python reference
 * dict comprehension in the spec). */
static void ht_place(ht_slot *slots, size_t cap, const unsigned char *arena,
                     uint32_t off, uint32_t len, int32_t id) {
    size_t mask = cap - 1;
    size_t i = (size_t)(hash_bytes(arena + off, len) & mask);
    for (;;) {
        ht_slot *s = &slots[i];
        if (s->id < 0) {
            s->off = off;
            s->len = len;
            s->id = id;
            return;
        }
        if (s->len == len && memcmp(arena + s->off, arena + off, len) == 0) {
            s->id = id;
            return;
        }
        i = (i + 1) & mask;
    }
}

static int ht_grow(tiktoken_bpe_t *tt) {
    size_t ncap = tt->ht.cap ? tt->ht.cap * 2 : 1024;
    ht_slot *slots = (ht_slot *)malloc(ncap * sizeof(ht_slot));
    if (!slots) return -1;
    for (size_t i = 0; i < ncap; i++) slots[i].id = -1;
    for (int32_t id = 0; id < tt->id_cap; id++) {
        if (!tt->present[id]) continue;
        span_t sp = tt->ids[id];
        ht_place(slots, ncap, tt->arena.p, sp.off, sp.len, id);
    }
    free(tt->ht.slots);
    tt->ht.slots = slots;
    tt->ht.cap = ncap;
    return 0;
}

static int ht_insert(tiktoken_bpe_t *tt, uint32_t off, uint32_t len, int32_t id) {
    if (tt->ht.cap == 0 && ht_grow(tt) != 0) return -1;
    if ((tt->ht.count + 1) * 10 >= tt->ht.cap * 7) {
        if (ht_grow(tt) != 0) return -1;
    }
    size_t mask = tt->ht.cap - 1;
    size_t i = (size_t)(hash_bytes(tt->arena.p + off, len) & mask);
    for (;;) {
        ht_slot *s = &tt->ht.slots[i];
        if (s->id < 0) {
            s->off = off;
            s->len = len;
            s->id = id;
            tt->ht.count++;
            return 0;
        }
        if (s->len == len && memcmp(tt->arena.p + s->off, tt->arena.p + off, len) == 0) {
            s->id = id;
            return 0;
        }
        i = (i + 1) & mask;
    }
}

/* Rank (== id) of a byte string, -1 when unknown. */
static int32_t rank_of(const tiktoken_bpe_t *tt, const void *bytes, size_t len) {
    if (!tt || !tt->ht.cap || len == 0) return -1;
    size_t mask = tt->ht.cap - 1;
    size_t i = (size_t)(hash_bytes(bytes, len) & mask);
    const unsigned char *b = (const unsigned char *)bytes;
    for (;;) {
        const ht_slot *s = &tt->ht.slots[i];
        if (s->id < 0) return -1;
        if (s->len == len && memcmp(tt->arena.p + s->off, b, len) == 0) return s->id;
        i = (i + 1) & mask;
    }
}

/* ------------------------------------------------------------------ */
/* vocabulary construction                                             */
/* ------------------------------------------------------------------ */

static int tt_ensure_ids(tiktoken_bpe_t *tt, int64_t id) {
    if (id < 0 || id > 0x7fffffff) return -1;
    if (id < tt->id_cap) return 0;
    int64_t ncap = tt->id_cap ? tt->id_cap : 256;
    while (ncap <= id) {
        if (ncap > (int64_t)1 << 30) return -1;
        ncap *= 2;
    }
    span_t *ni = (span_t *)realloc(tt->ids, (size_t)ncap * sizeof(span_t));
    if (!ni) return -1;
    tt->ids = ni;
    unsigned char *np = (unsigned char *)realloc(tt->present, (size_t)ncap);
    if (!np) return -1;
    tt->present = np;
    memset(tt->ids + tt->id_cap, 0, (size_t)(ncap - tt->id_cap) * sizeof(span_t));
    memset(tt->present + tt->id_cap, 0, (size_t)(ncap - tt->id_cap));
    tt->id_cap = (int32_t)ncap;
    return 0;
}

static int32_t tt_next_special_id(tiktoken_bpe_t *tt) {
    int32_t id = tt->next_special_id;
    if (id < tt->max_id + 1) id = tt->max_id + 1;
    if (id < tt->vocab_size) id = tt->vocab_size;
    tt->next_special_id = id + 1;
    return id;
}

/* Register bytes -> id in both directions.  Returns 0/-1. */
static int tt_insert_rank(tiktoken_bpe_t *tt, const unsigned char *bytes,
                          size_t len, int32_t id) {
    if (!tt || !bytes || len == 0 || len > 0xffffffffu) return -1;
    if (tt_ensure_ids(tt, id) != 0) {
        tt_set_error("token id %d out of range", id);
        return -1;
    }
    if (tt->present[id]) {
        span_t sp = tt->ids[id];
        if (sp.len != len || memcmp(tt->arena.p + sp.off, bytes, len) != 0) {
            tt_set_error("token id %d defined twice with different bytes", id);
            return -1;
        }
        return 0;
    }
    uint32_t off = (uint32_t)tt->arena.len;
    if (buf_append(&tt->arena, bytes, len) != 0) {
        tt_set_error("out of memory growing vocabulary arena");
        return -1;
    }
    if (ht_insert(tt, off, len, id) != 0) {
        tt_set_error("out of memory growing vocabulary hash table");
        return -1;
    }
    tt->ids[id].off = off;
    tt->ids[id].len = (uint32_t)len;
    tt->present[id] = 1;
    if (id > tt->max_id) tt->max_id = id;
    if (id + 1 > tt->vocab_size) tt->vocab_size = id + 1;
    return 0;
}

static int special_cmp_desc(const void *a, const void *b) {
    const special_t *x = (const special_t *)a;
    const special_t *y = (const special_t *)b;
    if (x->len != y->len) return x->len > y->len ? -1 : 1; /* longest first */
    return x->id < y->id ? -1 : (x->id > y->id ? 1 : 0);
}

/* Attach an already inserted token to the special-token match list. */
static int tt_register_special(tiktoken_bpe_t *tt, uint32_t off, uint32_t len,
                               int32_t id) {
    for (size_t i = 0; i < tt->n_specials; i++) {
        if (tt->specials[i].len == len &&
            memcmp(tt->arena.p + tt->specials[i].off, tt->arena.p + off, len) == 0) {
            return 0; /* already known */
        }
    }
    if (tt->n_specials == tt->cap_specials) {
        size_t ncap = tt->cap_specials ? tt->cap_specials * 2 : 16;
        special_t *ns = (special_t *)realloc(tt->specials, ncap * sizeof(special_t));
        if (!ns) {
            tt_set_error("out of memory growing special token list");
            return -1;
        }
        tt->specials = ns;
        tt->cap_specials = ncap;
    }
    tt->specials[tt->n_specials].off = off;
    tt->specials[tt->n_specials].len = len;
    tt->specials[tt->n_specials].id = id;
    tt->n_specials++;
    if (tt->min_special_len == 0 || len < tt->min_special_len) tt->min_special_len = len;
    return 0;
}

int ttbpe_add_special(tiktoken_bpe_t *tt, const char *token, size_t len) {
    if (!tt || !token || len == 0) {
        tt_set_error("ttbpe_add_special: invalid argument");
        return -1;
    }
    const unsigned char *bytes = (const unsigned char *)token;
    int32_t existing = rank_of(tt, bytes, len);
    int32_t id;
    if (existing >= 0) {
        id = existing;
    } else {
        id = tt_next_special_id(tt);
        if (tt_insert_rank(tt, bytes, len, id) != 0) return -1;
    }
    span_t sp = tt->ids[id];
    if (tt_register_special(tt, sp.off, sp.len, id) != 0) return -1;
    if (tt->n_specials > 1) {
        qsort(tt->specials, tt->n_specials, sizeof(special_t), special_cmp_desc);
    }
    return 0;
}

int ttbpe_set_split_mode(tiktoken_bpe_t *tt, int mode) {
    if (!tt) return -1;
    if (mode != TTBPE_SPLIT_NONE && mode != TTBPE_SPLIT_GPT2 &&
        mode != TTBPE_SPLIT_CL100K) {
        tt_set_error("unknown split mode %d", mode);
        return -1;
    }
    tt->split_mode = mode;
    return 0;
}

int ttbpe_set_cache_enabled(tiktoken_bpe_t *tt, int enabled) {
    if (!tt) return -1;
    tt->cache_enabled = enabled ? 1 : 0;
    return 0;
}

/* ------------------------------------------------------------------ */
/* UTF-8 helpers                                                       */
/* ------------------------------------------------------------------ */

/* Decode one code point at s[i].  *adv receives the number of bytes consumed
 * (>= 1).  Returns -1 for an invalid sequence, consuming exactly the "maximal
 * subpart" so that a subsequent U+FFFD replacement matches Python's
 * bytes.decode("utf-8", errors="replace"). */
static int32_t utf8_read(const unsigned char *s, size_t len, size_t i, size_t *adv) {
    unsigned char b0 = s[i];
    if (b0 < 0x80) {
        *adv = 1;
        return b0;
    }
    size_t need;
    unsigned char lo = 0x80, hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        need = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        need = 3;
        if (b0 == 0xE0) lo = 0xA0;
        if (b0 == 0xED) hi = 0x9F; /* reject surrogates */
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        need = 4;
        if (b0 == 0xF0) lo = 0x90;
        if (b0 == 0xF4) hi = 0x8F;
    } else {
        *adv = 1;
        return -1;
    }
    size_t k = 1;
    int32_t cp = (int32_t)(b0 & (0xFF >> (need + 1)));
    while (k < need) {
        if (i + k >= len) break;
        unsigned char c = s[i + k];
        if (c < lo || c > hi) break;
        cp = (cp << 6) | (int32_t)(c & 0x3F);
        k++;
        lo = 0x80;
        hi = 0xBF;
    }
    if (k < need) {
        *adv = k;
        return -1;
    }
    *adv = need;
    return cp;
}

int ttbpe_utf8_encode(uint32_t cp, char *dst, size_t dst_cap, size_t *written) {
    if (!dst) return -1;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
    size_t n;
    unsigned char tmp[4];
    if (cp < 0x80) {
        tmp[0] = (unsigned char)cp;
        n = 1;
    } else if (cp < 0x800) {
        tmp[0] = (unsigned char)(0xC0 | (cp >> 6));
        tmp[1] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        tmp[0] = (unsigned char)(0xE0 | (cp >> 12));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = (unsigned char)(0xF0 | (cp >> 18));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    if (dst_cap < n) return -1;
    memcpy(dst, tmp, n);
    if (written) *written = n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* character classes (see the APPROXIMATION note at the top)           */
/* ------------------------------------------------------------------ */

/* Unicode whitespace outside ASCII (Zs/Zl/Zp subset, matching Python \s). */
static int is_unicode_space(uint32_t cp) {
    switch (cp) {
        case 0x00A0: case 0x1680: case 0x2028: case 0x2029: case 0x202F:
        case 0x205F: case 0x3000:
            return 1;
        default:
            break;
    }
    return cp >= 0x2000 && cp <= 0x200A;
}

/* Non-ASCII code points that are neither whitespace nor letters in the real
 * regex classes and are therefore folded into "other": C1 controls, format
 * characters, private use, surrogates (never reachable), unassigned gaps that
 * matter in practice are deliberately NOT listed -> they stay "letters". */
static int is_unicode_other(uint32_t cp) {
    if (cp < 0xA0) return 1; /* C1 controls */
    if (cp >= 0x200B && cp <= 0x200F) return 1;
    if (cp >= 0x202A && cp <= 0x202E) return 1;
    if (cp >= 0x2060 && cp <= 0x206F) return 1;
    if (cp == 0xFEFF) return 1;
    if (cp >= 0xFFF9 && cp <= 0xFFFB) return 1;
    if (cp >= 0xE000 && cp <= 0xF8FF) return 1;   /* private use area */
    if (cp >= 0xF0000 && cp <= 0x10FFFF) return 1; /* private use planes */
    return 0;
}

static int classify_ascii(unsigned char c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r')
        return TTBPE_CC_SPACE;
    if (c >= '0' && c <= '9') return TTBPE_CC_DIGIT;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return TTBPE_CC_LETTER;
    return TTBPE_CC_OTHER;
}

static int classify_builtin(uint32_t cp) {
    if (cp < 0x80) return classify_ascii((unsigned char)cp);
    if (is_unicode_space(cp)) return TTBPE_CC_SPACE;
    if (is_unicode_other(cp)) return TTBPE_CC_OTHER;
    return TTBPE_CC_LETTER; /* approximation: all other non-ASCII = letters */
}

typedef struct {
    int cls;
    size_t nbytes;
} cls_info;

static cls_info cls_of(const unsigned char *s, size_t len, size_t i) {
    cls_info r;
    unsigned char c = s[i];
    if (c < 0x80) {
        r.cls = classify_ascii(c);
        r.nbytes = 1;
        return r;
    }
    size_t adv = 1;
    int32_t cp = utf8_read(s, len, i, &adv);
    if (adv == 0) adv = 1;
    if (cp < 0) {
        r.cls = TTBPE_CC_OTHER; /* invalid byte: symbol-like */
        r.nbytes = adv;
        return r;
    }
    if (g_classifier) {
        int over = g_classifier((uint32_t)cp);
        if (over >= 0 && over <= 3) {
            r.cls = over;
            r.nbytes = adv;
            return r;
        }
    }
    r.cls = classify_builtin((uint32_t)cp);
    r.nbytes = adv;
    return r;
}

/* Byte length of the maximal run of code points of class cls at pos. */
static size_t run_of(const unsigned char *s, size_t len, size_t pos, int cls) {
    size_t i = pos;
    while (i < len) {
        cls_info ci = cls_of(s, len, i);
        if (ci.cls != cls) break;
        i += ci.nbytes;
    }
    return i - pos;
}

/* Length of a run of the two literal newline bytes \r \n (regex [\r\n]). */
static size_t newline_run(const unsigned char *s, size_t len, size_t pos) {
    size_t i = pos;
    while (i < len && (s[i] == '\r' || s[i] == '\n')) i++;
    return i - pos;
}

/* ------------------------------------------------------------------ */
/* pre-tokenization                                                    */
/* ------------------------------------------------------------------ */

/* GPT-2 / r50k pattern (from tiktoken _educational.py):
 *   's|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * Case sensitive; no possessive quantifiers. */
static size_t gpt2_match(const unsigned char *s, size_t len, size_t pos) {
    /* 1. English contraction suffixes */
    if (s[pos] == '\'') {
        if (pos + 1 < len) {
            unsigned char c1 = s[pos + 1];
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
            if (pos + 2 < len) {
                unsigned char c2 = s[pos + 2];
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'l' && c2 == 'l'))
                    return 3;
            }
        }
    }
    /* 2..4: optional single ASCII space followed by one or more letters,
     * digits or "other" code points. */
    size_t p = (s[pos] == ' ') ? pos + 1 : pos;
    if (p < len) {
        size_t n = run_of(s, len, p, TTBPE_CC_LETTER);
        if (n > 0) return (p - pos) + n;
        n = run_of(s, len, p, TTBPE_CC_DIGIT);
        if (n > 0) return (p - pos) + n;
        n = run_of(s, len, p, TTBPE_CC_OTHER);
        if (n > 0) return (p - pos) + n;
    }
    /* 5. \s+(?!\S) : greedy on the whole run, then the lookahead forces a
     *    one-character backtrack unless the run ends the text. */
    size_t sp = run_of(s, len, pos, TTBPE_CC_SPACE);
    if (sp > 0) {
        if (pos + sp == len) return sp;
        return sp >= 2 ? sp - 1 : 1;
    }
    /* 6. \s+ */
    return 1; /* unreachable for well formed classes; never stall */
}

/* cl100k / o200k pattern (verbatim from tiktoken_ext/openai_public.py, the
 * possessive quantifiers of the original are preserved):
 *   '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+
 *   | ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s */
static size_t cl100k_match(const unsigned char *s, size_t len, size_t pos) {
    unsigned char b0 = s[pos];
    /* 1. case insensitive contraction suffixes */
    if (b0 == '\'') {
        if (pos + 1 < len) {
            unsigned char c1 = (unsigned char)tolower(s[pos + 1]);
            if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') return 2;
            if (pos + 2 < len) {
                unsigned char c2 = (unsigned char)tolower(s[pos + 2]);
                if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') ||
                    (c1 == 'r' && c2 == 'e'))
                    return 3;
            }
        }
    }
    /* 2. [^\r\n\p{L}\p{N}]?+\p{L}++ */
    cls_info ci = cls_of(s, len, pos);
    if (ci.cls == TTBPE_CC_LETTER) {
        return run_of(s, len, pos, TTBPE_CC_LETTER);
    }
    int is_prefix = (ci.cls == TTBPE_CC_OTHER) ||
                    (ci.cls == TTBPE_CC_SPACE && b0 != '\r' && b0 != '\n');
    if (is_prefix) {
        size_t q = pos + ci.nbytes;
        if (q < len && cls_of(s, len, q).cls == TTBPE_CC_LETTER) {
            return ci.nbytes + run_of(s, len, q, TTBPE_CC_LETTER);
        }
        /* possessive prefix without a following letter: alternative fails */
    }
    /* 3. \p{N}{1,3}+ */
    if (ci.cls == TTBPE_CC_DIGIT) {
        size_t n = run_of(s, len, pos, TTBPE_CC_DIGIT);
        return n > 3 ? 3 : n;
    }
    /* 4.  ?[^\s\p{L}\p{N}]++[\r\n]*+ */
    {
        size_t p = (b0 == ' ') ? pos + 1 : pos;
        if (p < len) {
            size_t sym = run_of(s, len, p, TTBPE_CC_OTHER);
            if (sym > 0) {
                size_t q = p + sym;
                return (q - pos) + newline_run(s, len, q);
            }
        }
    }
    /* 5. \s++$ */
    size_t sp = run_of(s, len, pos, TTBPE_CC_SPACE);
    if (sp > 0 && pos + sp == len) return sp;
    /* 6. \s*[\r\n] : the last newline inside the whitespace run wins */
    for (size_t k = sp; k-- > 0;) {
        unsigned char c = s[pos + k];
        if (c == '\r' || c == '\n') return k + 1;
    }
    /* 7. \s+(?!\S) */
    if (sp >= 2) return sp - 1;
    /* 8. \s (also covers a single whitespace not at the end of the text) */
    if (sp >= 1) return 1;
    return 1; /* never stall */
}

static size_t split_next_piece(int mode, const unsigned char *s, size_t len,
                               size_t pos) {
    if (pos >= len) return 0;
    if (mode == TTBPE_SPLIT_NONE) return len - pos;
    if (mode == TTBPE_SPLIT_GPT2) return gpt2_match(s, len, pos);
    return cl100k_match(s, len, pos);
}

int ttbpe_split(const tiktoken_bpe_t *tt, const char *text, size_t len,
                ttbpe_piece **out, size_t *out_n) {
    if (!tt || !out || !out_n || (!text && len)) {
        tt_set_error("ttbpe_split: invalid argument");
        return -1;
    }
    *out = NULL;
    *out_n = 0;
    if (len == 0) return 0;
    ttbpe_piece *pieces = (ttbpe_piece *)malloc((len + 1) * sizeof(ttbpe_piece));
    if (!pieces) {
        tt_set_error("ttbpe_split: out of memory");
        return -1;
    }
    size_t n = 0, pos = 0;
    while (pos < len) {
        size_t pl = split_next_piece(tt->split_mode, (const unsigned char *)text,
                                     len, pos);
        if (pl == 0) pl = 1;
        pieces[n].ptr = text + pos;
        pieces[n].len = pl;
        n++;
        pos += pl;
    }
    *out = pieces;
    *out_n = n;
    return 0;
}

void ttbpe_free_pieces(ttbpe_piece *p) { free(p); }

/* ------------------------------------------------------------------ */
/* byte pair encoding of one piece                                     */
/* ------------------------------------------------------------------ */

/* Merge the adjacent pair with the smallest rank until no pair is known.
 * Writes at most len ids into out[] and returns their number, or -1 when a
 * leftover part has no rank (tiktoken raises KeyError in that case). */
static int bpe_piece_raw(const tiktoken_bpe_t *tt, const unsigned char *s,
                         size_t len, int *out) {
    if (len == 1) {
        int32_t id = rank_of(tt, s, 1);
        if (id < 0) {
            tt_set_error("byte 0x%02x is not in the vocabulary", s[0]);
            return -1;
        }
        out[0] = (int)id;
        return 1;
    }

    size_t n = len + 1; /* nodes 0..len, node len is the sentinel */
    unsigned char *blk = (unsigned char *)malloc(3 * n * sizeof(int32_t) + n + 64);
    if (!blk) {
        tt_set_error("out of memory in BPE merge");
        return -1;
    }
    int32_t *start = (int32_t *)blk;                 /* n int32 */
    int32_t *nxt = (int32_t *)(blk + n * 4);         /* n int32 */
    int32_t *prv = (int32_t *)(blk + 2 * n * 4);     /* n int32 */
    unsigned char *alive = blk + 3 * n * 4;          /* n bytes */

    for (size_t i = 0; i < n; i++) {
        start[i] = (int32_t)i; /* covers node len (the sentinel) as well */
        nxt[i] = (i + 1 < n) ? (int32_t)(i + 1) : -1;
        prv[i] = (i > 0) ? (int32_t)(i - 1) : -1;
        alive[i] = 1;
    }

    minheap_t heap;
    heap.a = NULL;
    heap.n = heap.cap = 0;

    /* push the pair whose left node is i (needs a live successor < len) */
#define PUSH_PAIR(i)                                                          \
    do {                                                                       \
        int32_t _i = (i);                                                      \
        if (_i >= 0 && (size_t)_i < len) {                                     \
            int32_t _j = nxt[_i];                                              \
            if (_j >= 0 && (size_t)_j < len) {                                 \
                int32_t _r = rank_of(tt, s + start[_i],                         \
                                     (size_t)(start[nxt[_j]] - start[_i]));    \
                if (_r >= 0 && heap_push(&heap, _r, _i) != 0) goto oom;        \
            }                                                                  \
        }                                                                      \
    } while (0)

    for (size_t i = 0; i + 1 < len; i++) PUSH_PAIR((int32_t)i);

    heap_item it;
    while (heap_pop(&heap, &it)) {
        int32_t i = it.idx;
        if (i < 0 || (size_t)i >= len || !alive[i]) continue;
        int32_t j = nxt[i];
        if (j < 0 || (size_t)j >= len) continue;
        int32_t cur = rank_of(tt, s + start[i], (size_t)(start[nxt[j]] - start[i]));
        if (cur != it.rank) continue; /* stale entry */

        /* merge i and j */
        alive[j] = 0;
        nxt[i] = nxt[j];
        if (nxt[i] >= 0) prv[nxt[i]] = i;
        PUSH_PAIR(prv[i]);
        PUSH_PAIR(i);
    }

    /* collect the surviving parts */
    int cnt = 0;
    for (int32_t i = 0; i >= 0 && (size_t)i < len; i = nxt[i]) {
        size_t span = (size_t)(start[nxt[i]] - start[i]);
        int32_t id = rank_of(tt, s + start[i], span);
        if (id < 0) {
            tt_set_error("no rank for merged part of length %zu", span);
            free(heap.a);
            free(blk);
            return -1;
        }
        out[cnt++] = (int)id;
    }
    free(heap.a);
    free(blk);
    return cnt;

oom:
    free(heap.a);
    free(blk);
    tt_set_error("out of memory in BPE merge");
    return -1;
#undef PUSH_PAIR
}

/* ------------------------------------------------------------------ */
/* piece -> ids cache                                                  */
/* ------------------------------------------------------------------ */

static int32_t cache_head_index(const tiktoken_bpe_t *tt, const void *bytes,
                                size_t len) {
    return (int32_t)(hash_bytes(bytes, len) & (uint64_t)(tt->cache_head_cap - 1));
}

static int cache_lookup(const tiktoken_bpe_t *tt, const unsigned char *bytes,
                        size_t len, const int **ids_out, size_t *n_out) {
    if (!tt->cache_enabled || !tt->cache_head_cap || len > TTBPE_CACHE_MAX_PIECE_LEN)
        return 0;
    int32_t e = tt->cache_head[cache_head_index(tt, bytes, len)];
    while (e >= 0) {
        const cache_entry_t *c = &tt->cache[e];
        if (c->klen == len &&
            memcmp(tt->arena.p + c->koff, bytes, len) == 0) {
            *ids_out = tt->cache_ids + c->ioff;
            *n_out = c->nids;
            return 1;
        }
        e = c->next;
    }
    return 0;
}

static void cache_store(tiktoken_bpe_t *tt, const unsigned char *bytes, size_t len,
                        const int *ids, size_t n) {
    if (!tt->cache_enabled || len > TTBPE_CACHE_MAX_PIECE_LEN) return;
    if (tt->n_cache >= TTBPE_CACHE_MAX_ENTRIES) return; /* bounded, best effort */
    if (tt->n_cache >= tt->cache_head_cap) {
        size_t ncap = tt->cache_head_cap ? tt->cache_head_cap * 2 : 1024;
        int32_t *nh = (int32_t *)realloc(tt->cache_head, ncap * sizeof(int32_t));
        if (!nh) return;
        for (size_t i = tt->cache_head_cap; i < ncap; i++) nh[i] = -1;
        tt->cache_head = nh;
        tt->cache_head_cap = ncap;
    }
    if (!tt->cache) {
        tt->cache = (cache_entry_t *)malloc(TTBPE_CACHE_MAX_ENTRIES *
                                            sizeof(cache_entry_t));
        if (!tt->cache) return;
    }
    if (tt->cache_ids_len + n > tt->cache_ids_cap) {
        size_t ncap = tt->cache_ids_cap ? tt->cache_ids_cap : 4096;
        while (ncap < tt->cache_ids_len + n) ncap *= 2;
        int32_t *ni = (int32_t *)realloc(tt->cache_ids, ncap * sizeof(int32_t));
        if (!ni) return;
        tt->cache_ids = ni;
        tt->cache_ids_cap = ncap;
    }
    uint32_t koff = (uint32_t)tt->arena.len;
    if (buf_append(&tt->arena, bytes, len) != 0) return;
    size_t idx = tt->n_cache++;
    cache_entry_t *c = &tt->cache[idx];
    c->koff = koff;
    c->klen = (uint32_t)len;
    c->ioff = (uint32_t)tt->cache_ids_len;
    c->nids = (uint32_t)n;
    memcpy(tt->cache_ids + tt->cache_ids_len, ids, n * sizeof(int32_t));
    tt->cache_ids_len += n;
    int32_t h = cache_head_index(tt, bytes, len);
    c->next = tt->cache_head[h];
    tt->cache_head[h] = (int32_t)idx;
}

/* Encode one piece, using the cache when possible. */
static int encode_one_piece(const tiktoken_bpe_t *tt, const unsigned char *s,
                            size_t len, idbuf_t *out) {
    if (len == 0) return 0;
    tiktoken_bpe_t *mtt = (tiktoken_bpe_t *)tt; /* cache is mutable state */
    const int *cached = NULL;
    size_t cn = 0;
    if (cache_lookup(tt, s, len, &cached, &cn)) {
        if (out->n + cn > out->cap) {
            size_t ncap = out->cap ? out->cap : 64;
            while (ncap < out->n + cn) ncap *= 2;
            int *np = (int *)realloc(out->p, ncap * sizeof(int));
            if (!np) {
                tt_set_error("out of memory growing token buffer");
                return -1;
            }
            out->p = np;
            out->cap = ncap;
        }
        memcpy(out->p + out->n, cached, cn * sizeof(int));
        out->n += cn;
        return 0;
    }

    int *tmp = (int *)malloc(len * sizeof(int));
    if (!tmp) {
        tt_set_error("out of memory allocating merge buffer");
        return -1;
    }
    int n = bpe_piece_raw(tt, s, len, tmp);
    if (n < 0) {
        free(tmp);
        return -1;
    }
    int rc = 0;
    for (int i = 0; i < n; i++) {
        if (idbuf_push(out, tmp[i]) != 0) {
            tt_set_error("out of memory growing token buffer");
            rc = -1;
            break;
        }
    }
    if (rc == 0) cache_store(mtt, s, len, tmp, (size_t)n);
    free(tmp);
    return rc;
}

/* ------------------------------------------------------------------ */
/* encoding                                                            */
/* ------------------------------------------------------------------ */

/* Longest special token matching at s, returns its length (0 = none). */
static size_t special_match(const tiktoken_bpe_t *tt, const unsigned char *s,
                            size_t len, int32_t *id_out) {
    if (tt->n_specials == 0 || len < tt->min_special_len) return 0;
    for (size_t i = 0; i < tt->n_specials; i++) {
        const special_t *sp = &tt->specials[i];
        if (sp->len > len) continue;
        if (memcmp(tt->arena.p + sp->off, s, sp->len) == 0) {
            if (id_out) *id_out = sp->id;
            return sp->len;
        }
    }
    return 0;
}

/* special_mode: 0 = ignore specials (ordinary), 1 = reject, 2 = allow all */
static int encode_core(const tiktoken_bpe_t *tt, const char *text, size_t len,
                       int use_split, int special_mode, int **out_ids,
                       size_t *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    if (!tt) {
        tt_set_error("ttbpe_encode: NULL tokenizer");
        return -1;
    }
    if (!text && len != 0) {
        tt_set_error("ttbpe_encode: NULL text with non-zero length");
        return -1;
    }
    if (len == 0) return 0;

    idbuf_t ib;
    ib.p = NULL;
    ib.n = ib.cap = 0;
    const unsigned char *s = (const unsigned char *)text;

    if (!use_split) {
        if (encode_one_piece(tt, s, len, &ib) != 0) {
            free(ib.p);
            return -1;
        }
    } else {
        size_t pos = 0;
        while (pos < len) {
            if (special_mode != 0 && tt->n_specials) {
                int32_t sid = -1;
                size_t sl = special_match(tt, s + pos, len - pos, &sid);
                if (sl > 0) {
                    if (special_mode == 1) {
                        tt_set_error("encountered disallowed special token "
                                     "(id %d) at byte offset %zu",
                                     sid, pos);
                        free(ib.p);
                        return -1;
                    }
                    if (idbuf_push(&ib, (int)sid) != 0) {
                        tt_set_error("out of memory growing token buffer");
                        free(ib.p);
                        return -1;
                    }
                    pos += sl;
                    continue;
                }
            }
            size_t pl = split_next_piece(tt->split_mode, s, len, pos);
            if (pl == 0 || pl > len - pos) pl = len - pos;
            if (encode_one_piece(tt, s + pos, pl, &ib) != 0) {
                free(ib.p);
                return -1;
            }
            pos += pl;
        }
    }
    *out_ids = ib.p;
    *out_n = ib.n;
    return 0;
}

int ttbpe_encode(const tiktoken_bpe_t *tt, const char *text, size_t len,
                 int **out_ids, size_t *out_n) {
    return encode_core(tt, text, len, 1, 1, out_ids, out_n);
}

int ttbpe_encode_ex(const tiktoken_bpe_t *tt, const char *text, size_t len,
                    int allow_all_special, int reject_disallowed,
                    int **out_ids, size_t *out_n) {
    int mode;
    if (allow_all_special) {
        mode = 2;
    } else if (reject_disallowed) {
        mode = 1;
    } else {
        mode = 0;
    }
    return encode_core(tt, text, len, 1, mode, out_ids, out_n);
}

int ttbpe_encode_ordinary(const tiktoken_bpe_t *tt, const char *text, size_t len,
                          int **out_ids, size_t *out_n) {
    return encode_core(tt, text, len, 1, 0, out_ids, out_n);
}

int ttbpe_encode_single_piece(const tiktoken_bpe_t *tt, const char *bytes, size_t len,
                              int **out_ids, size_t *out_n) {
    return encode_core(tt, bytes, len, 0, 0, out_ids, out_n);
}

/* ------------------------------------------------------------------ */
/* decoding                                                            */
/* ------------------------------------------------------------------ */

int ttbpe_decode_bytes(const tiktoken_bpe_t *tt, const int *ids, size_t n,
                       unsigned char **out_bytes, size_t *out_len) {
    if (!out_bytes || !out_len) return -1;
    *out_bytes = NULL;
    *out_len = 0;
    if (!tt) {
        tt_set_error("ttbpe_decode: NULL tokenizer");
        return -1;
    }
    if (!ids && n) {
        tt_set_error("ttbpe_decode: NULL ids with non-zero length");
        return -1;
    }
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= tt->id_cap || !tt->present[id]) {
            tt_set_error("token id %d is out of range (vocab_size=%d)", id,
                         tt->vocab_size);
            return -1;
        }
        total += tt->ids[id].len;
    }
    unsigned char *buf = (unsigned char *)malloc(total ? total : 1);
    if (!buf) {
        tt_set_error("ttbpe_decode: out of memory");
        return -1;
    }
    size_t off = 0;
    for (size_t i = 0; i < n; i++) {
        span_t sp = tt->ids[ids[i]];
        memcpy(buf + off, tt->arena.p + sp.off, sp.len);
        off += sp.len;
    }
    *out_bytes = buf;
    *out_len = total;
    return 0;
}

/* Output length of the U+FFFD replacement decoding. */
static size_t utf8_replace_len(const unsigned char *s, size_t len) {
    size_t i = 0, out = 0;
    while (i < len) {
        if (s[i] < 0x80) {
            i++;
            out++;
            continue;
        }
        size_t adv = 1;
        int32_t cp = utf8_read(s, len, i, &adv);
        if (cp < 0) {
            out += 3; /* U+FFFD */
        } else if (cp < 0x800) {
            out += 2;
        } else if (cp < 0x10000) {
            out += 3;
        } else {
            out += 4;
        }
        i += adv;
    }
    return out;
}

static void utf8_replace_write(const unsigned char *s, size_t len,
                               unsigned char *dst) {
    size_t i = 0, o = 0;
    while (i < len) {
        if (s[i] < 0x80) {
            dst[o++] = s[i++];
            continue;
        }
        size_t adv = 1;
        int32_t cp = utf8_read(s, len, i, &adv);
        if (cp < 0) {
            /* U+FFFD, one replacement per maximal invalid subpart */
            dst[o++] = 0xEF;
            dst[o++] = 0xBF;
            dst[o++] = 0xBD;
        } else {
            size_t n = 0;
            ttbpe_utf8_encode((uint32_t)cp, (char *)(dst + o), 4, &n);
            o += n;
        }
        i += adv;
    }
}

int ttbpe_decode(const tiktoken_bpe_t *tt, const int *ids, size_t n,
                 char **out_text, size_t *out_len) {
    if (!out_text || !out_len) return -1;
    *out_text = NULL;
    *out_len = 0;
    unsigned char *raw = NULL;
    size_t raw_len = 0;
    if (ttbpe_decode_bytes(tt, ids, n, &raw, &raw_len) != 0) return -1;
    size_t need = utf8_replace_len(raw, raw_len);
    char *text = (char *)malloc(need + 1);
    if (!text) {
        free(raw);
        tt_set_error("ttbpe_decode: out of memory");
        return -1;
    }
    utf8_replace_write(raw, raw_len, (unsigned char *)text);
    text[need] = '\0';
    free(raw);
    *out_text = text;
    *out_len = need;
    return 0;
}

/* ------------------------------------------------------------------ */
/* JSON vocabulary (.tok.json, xAI Grok style)                         */
/* ------------------------------------------------------------------ */

typedef struct {
    const unsigned char *p, *end;
} jscan_t;

static void j_skip_ws(jscan_t *s) {
    while (s->p < s->end) {
        unsigned char c = *s->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s->p++;
        else break;
    }
}

static int j_expect(jscan_t *s, unsigned char c) {
    j_skip_ws(s);
    if (s->p < s->end && *s->p == c) {
        s->p++;
        return 0;
    }
    return -1;
}

static int j_skip_string(jscan_t *s) {
    if (s->p >= s->end || *s->p != '"') return -1;
    s->p++;
    while (s->p < s->end) {
        unsigned char c = *s->p++;
        if (c == '\\') {
            if (s->p < s->end) s->p++;
            else return -1;
        } else if (c == '"') {
            return 0;
        }
    }
    return -1;
}

static int j_skip_value(jscan_t *s, int depth) {
    if (depth > 64) return -1;
    j_skip_ws(s);
    if (s->p >= s->end) return -1;
    unsigned char c = *s->p;
    if (c == '"') return j_skip_string(s);
    if (c == '{') {
        s->p++;
        j_skip_ws(s);
        if (s->p < s->end && *s->p == '}') {
            s->p++;
            return 0;
        }
        for (;;) {
            j_skip_ws(s);
            if (j_skip_string(s) != 0) return -1;
            if (j_expect(s, ':') != 0) return -1;
            if (j_skip_value(s, depth + 1) != 0) return -1;
            j_skip_ws(s);
            if (s->p < s->end && *s->p == ',') {
                s->p++;
                continue;
            }
            if (s->p < s->end && *s->p == '}') {
                s->p++;
                return 0;
            }
            return -1;
        }
    }
    if (c == '[') {
        s->p++;
        j_skip_ws(s);
        if (s->p < s->end && *s->p == ']') {
            s->p++;
            return 0;
        }
        for (;;) {
            if (j_skip_value(s, depth + 1) != 0) return -1;
            j_skip_ws(s);
            if (s->p < s->end && *s->p == ',') {
                s->p++;
                continue;
            }
            if (s->p < s->end && *s->p == ']') {
                s->p++;
                return 0;
            }
            return -1;
        }
    }
    if (c == 't' || c == 'f' || c == 'n') {
        while (s->p < s->end && *s->p >= 'a' && *s->p <= 'z') s->p++;
        return 0;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        s->p++;
        while (s->p < s->end) {
            unsigned char d = *s->p;
            if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' ||
                d == '+' || d == '-')
                s->p++;
            else break;
        }
        return 0;
    }
    return -1;
}

static int j_parse_int(jscan_t *s, long long *out) {
    j_skip_ws(s);
    int neg = 0;
    if (s->p < s->end && *s->p == '-') {
        neg = 1;
        s->p++;
    }
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return -1;
    long long v = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') {
        v = v * 10 + (*s->p - '0');
        if (v > ((long long)1 << 40)) return -1;
        s->p++;
    }
    *out = neg ? -v : v;
    return 0;
}

/* Parses a JSON string into a NUL terminated copy inside out. */
static int j_parse_string(jscan_t *s, buf_t *out) {
    if (j_expect(s, '"') != 0) return -1;
    buf_reset(out);
    while (s->p < s->end) {
        unsigned char c = *s->p++;
        if (c == '"') {
            return buf_append(out, "", 1); /* NUL terminator */
        }
        if (c == '\\') {
            if (s->p >= s->end) return -1;
            unsigned char e = *s->p++;
            unsigned char rep = 0;
            switch (e) {
                case '"': rep = '"'; break;
                case '\\': rep = '\\'; break;
                case '/': rep = '/'; break;
                case 'b': rep = '\b'; break;
                case 'f': rep = '\f'; break;
                case 'n': rep = '\n'; break;
                case 'r': rep = '\r'; break;
                case 't': rep = '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    for (int k = 0; k < 4; k++) {
                        if (s->p >= s->end) return -1;
                        unsigned char h = *s->p++;
                        uint32_t d;
                        if (h >= '0' && h <= '9') d = (uint32_t)(h - '0');
                        else if (h >= 'a' && h <= 'f') d = (uint32_t)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') d = (uint32_t)(h - 'A' + 10);
                        else return -1;
                        cp = (cp << 4) | d;
                    }
                    char enc[4];
                    size_t n = 0;
                    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
                    ttbpe_utf8_encode(cp, enc, sizeof(enc), &n);
                    if (buf_append(out, enc, n) != 0) return -1;
                    continue;
                }
                default: return -1;
            }
            if (buf_append(out, &rep, 1) != 0) return -1;
            continue;
        }
        if (buf_append(out, &c, 1) != 0) return -1;
    }
    return -1;
}

/* Parses [12, 34, ...] of values in 0..255 into out. */
static int j_parse_byte_array(jscan_t *s, buf_t *out) {
    if (j_expect(s, '[') != 0) return -1;
    buf_reset(out);
    j_skip_ws(s);
    if (s->p < s->end && *s->p == ']') {
        s->p++;
        return 0;
    }
    for (;;) {
        long long v = 0;
        if (j_parse_int(s, &v) != 0 || v < 0 || v > 255) return -1;
        unsigned char b = (unsigned char)v;
        if (buf_append(out, &b, 1) != 0) return -1;
        j_skip_ws(s);
        if (s->p < s->end && *s->p == ',') {
            s->p++;
            continue;
        }
        if (s->p < s->end && *s->p == ']') {
            s->p++;
            return 0;
        }
        return -1;
    }
}

/* {"bytes": [..], "token": N} in any key order. */
static int j_parse_token_entry(jscan_t *s, buf_t *bytes, buf_t *key,
                               long long *token_out) {
    if (j_expect(s, '{') != 0) return -1;
    int have_token = 0, have_bytes = 0;
    long long tok = -1;
    for (;;) {
        j_skip_ws(s);
        if (j_parse_string(s, key) != 0) return -1;
        if (j_expect(s, ':') != 0) return -1;
        const char *k = (const char *)key->p;
        if (strcmp(k, "bytes") == 0) {
            if (j_parse_byte_array(s, bytes) != 0) return -1;
            have_bytes = 1;
        } else if (strcmp(k, "token") == 0) {
            if (j_parse_int(s, &tok) != 0) return -1;
            have_token = 1;
        } else if (j_skip_value(s, 0) != 0) {
            return -1;
        }
        j_skip_ws(s);
        if (s->p < s->end && *s->p == ',') {
            s->p++;
            continue;
        }
        if (s->p < s->end && *s->p == '}') {
            s->p++;
            break;
        }
        return -1;
    }
    if (!have_token || !have_bytes || bytes->len == 0 || tok < 0) return -1;
    *token_out = tok;
    return 0;
}

static int parse_tok_json(tiktoken_bpe_t *tt, const unsigned char *data,
                          size_t size) {
    jscan_t s;
    s.p = data;
    s.end = data + size;
    buf_t bytes, key, entry_key;
    bytes.p = NULL;
    bytes.len = bytes.cap = 0;
    key.p = NULL;
    key.len = key.cap = 0;
    entry_key.p = NULL;
    entry_key.len = entry_key.cap = 0;
    long long declared_vocab = -1;
    int rc = -1;

    if (j_expect(&s, '{') != 0) goto done;
    j_skip_ws(&s);
    if (s.p < s.end && *s.p == '}') {
        s.p++;
    } else {
        for (;;) {
            j_skip_ws(&s);
            if (j_parse_string(&s, &key) != 0) goto done;
            if (j_expect(&s, ':') != 0) goto done;
            const char *k = (const char *)key.p;
            if (strcmp(k, "regular_tokens") == 0 ||
                strcmp(k, "special_tokens") == 0) {
                int is_special = (k[0] == 's');
                if (j_expect(&s, '[') != 0) goto done;
                j_skip_ws(&s);
                if (s.p < s.end && *s.p == ']') {
                    s.p++;
                } else {
                    for (;;) {
                        long long tok = -1;
                        if (j_parse_token_entry(&s, &bytes, &entry_key, &tok) != 0) {
                            tt_set_error("malformed token entry near offset %zu",
                                         (size_t)(s.p - data));
                            goto done;
                        }
                        if (tt_insert_rank(tt, bytes.p, bytes.len, (int32_t)tok) != 0)
                            goto done;
                        if (is_special) {
                            int32_t id = (int32_t)tok;
                            if (tt_register_special(tt, tt->ids[id].off,
                                                    tt->ids[id].len, id) != 0)
                                goto done;
                        }
                        j_skip_ws(&s);
                        if (s.p < s.end && *s.p == ',') {
                            s.p++;
                            continue;
                        }
                        if (s.p < s.end && *s.p == ']') {
                            s.p++;
                            break;
                        }
                        tt_set_error("malformed token array near offset %zu",
                                     (size_t)(s.p - data));
                        goto done;
                    }
                }
            } else if (strcmp(k, "vocab_size") == 0) {
                if (j_parse_int(&s, &declared_vocab) != 0) goto done;
            } else if (strcmp(k, "word_split") == 0) {
                if (j_parse_string(&s, &key) != 0) goto done;
                free(tt->word_split);
                tt->word_split = NULL;
                if (key.len > 1) { /* more than the NUL terminator */
                    tt->word_split = (char *)malloc(key.len);
                    if (!tt->word_split) goto done;
                    memcpy(tt->word_split, key.p, key.len);
                }
            } else if (j_skip_value(&s, 0) != 0) {
                goto done;
            }
            j_skip_ws(&s);
            if (s.p < s.end && *s.p == ',') {
                s.p++;
                continue;
            }
            if (s.p < s.end && *s.p == '}') {
                s.p++;
                break;
            }
            goto done;
        }
    }

    if (tt->max_id < 0 && declared_vocab <= 0) {
        tt_set_error("empty vocabulary");
        goto done;
    }
    if (tt->n_specials > 1)
        qsort(tt->specials, tt->n_specials, sizeof(special_t), special_cmp_desc);
    tt->vocab_size = (int32_t)(tt->max_id + 1);
    if (declared_vocab > tt->vocab_size) tt->vocab_size = (int32_t)declared_vocab;
    tt->next_special_id = tt->vocab_size > tt->max_id + 1 ? tt->vocab_size
                                                          : tt->max_id + 1;
    rc = 0;

done:
    buf_free(&bytes);
    buf_free(&key);
    buf_free(&entry_key);
    return rc;
}

/* ------------------------------------------------------------------ */
/* .tiktoken vocabulary (one base64 line per token)                    */
/* ------------------------------------------------------------------ */

static int b64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Strict RFC 4648 standard alphabet decode: the payload length must be a
 * multiple of four, only the last group may carry one or two '=' characters,
 * and the unused low bits of the final group must be zero.  This mirrors the
 * strict base64 engine tiktoken itself uses, so malformed vocabularies are
 * rejected instead of being silently mis-decoded. */
static int b64_decode(const unsigned char *s, size_t n, buf_t *out) {
    buf_reset(out);
    if (n == 0 || (n % 4) != 0) return -1;
    size_t pad = 0;
    if (s[n - 1] == '=') pad++;
    if (n >= 2 && s[n - 2] == '=') pad++;
    if (pad > 2) return -1;
    size_t body = n - pad;
    if (pad != 0 && pad != 4 - (body % 4)) return -1;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < body; i++) {
        int v = b64_value(s[i]);
        if (v < 0) return -1; /* includes padding in the middle */
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            unsigned char b = (unsigned char)((acc >> bits) & 0xFF);
            if (buf_append(out, &b, 1) != 0) return -1;
        }
    }
    if (bits >= 6) return -1;              /* stray character */
    if (bits > 0 && (acc & ((1u << bits) - 1u)) != 0) return -1;
    if (pad == 0 && (body % 4) != 0) return -1;
    return 0;
}

static int parse_tiktoken_vocab(tiktoken_bpe_t *tt, const unsigned char *data,
                                size_t size) {
    buf_t raw;
    raw.p = NULL;
    raw.len = raw.cap = 0;
    size_t i = 0;
    int32_t id = 0;
    int rc = 0;
    while (i < size) {
        while (i < size && (data[i] == ' ' || data[i] == '\t')) i++;
        size_t j = i;
        while (j < size && data[j] != '\n' && data[j] != '\r') j++;
        size_t line_len = j - i;
        while (line_len > 0 && (data[i + line_len - 1] == ' ' ||
                                data[i + line_len - 1] == '\t'))
            line_len--;
        if (line_len == 0) {
            /* only tolerated as trailing whitespace */
            size_t k = j;
            while (k < size && (data[k] == '\n' || data[k] == '\r' ||
                                data[k] == ' ' || data[k] == '\t'))
                k++;
            if (k < size) {
                tt_set_error("empty line in .tiktoken vocabulary at offset %zu",
                             i);
                rc = -1;
            }
            break;
        }
        if (b64_decode(data + i, line_len, &raw) != 0) {
            tt_set_error("invalid base64 on line %d of .tiktoken vocabulary", id);
            rc = -1;
            break;
        }
        if (tt_insert_rank(tt, raw.p, raw.len, id) != 0) {
            rc = -1;
            break;
        }
        id++;
        i = j;
        while (i < size && (data[i] == '\n' || data[i] == '\r')) i++;
    }
    if (rc == 0) {
        if (id == 0) {
            tt_set_error("empty .tiktoken vocabulary");
            rc = -1;
        } else {
            tt->vocab_size = id;
            tt->next_special_id = tt->max_id + 1 > tt->vocab_size
                                      ? tt->max_id + 1
                                      : tt->vocab_size;
        }
    }
    buf_free(&raw);
    return rc;
}

/* ------------------------------------------------------------------ */
/* loading                                                             */
/* ------------------------------------------------------------------ */

static tiktoken_bpe_t *tt_new(void) {
    tiktoken_bpe_t *tt = (tiktoken_bpe_t *)calloc(1, sizeof(*tt));
    if (!tt) return NULL;
    tt->max_id = -1;
    tt->split_mode = TTBPE_SPLIT_CL100K; /* default approximation, see spec 5 */
    tt->cache_enabled = 1;
    snprintf(tt->format, sizeof(tt->format), "%s", "tiktoken");
    return tt;
}

static int looks_like_json(const unsigned char *data, size_t size) {
    size_t i = 0;
    if (size >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF) i = 3;
    while (i < size && (data[i] == ' ' || data[i] == '\t' || data[i] == '\n' ||
                        data[i] == '\r'))
        i++;
    return i < size && data[i] == '{';
}

tiktoken_bpe_t *ttbpe_load_from_memory(const void *data, size_t size) {
    tt_clear_error();
    if (!data || size == 0) {
        tt_set_error("ttbpe_load_from_memory: empty input");
        return NULL;
    }
    tiktoken_bpe_t *tt = tt_new();
    if (!tt) {
        tt_set_error("out of memory");
        return NULL;
    }
    int ok;
    if (looks_like_json((const unsigned char *)data, size)) {
        ok = parse_tok_json(tt, (const unsigned char *)data, size) == 0;
        if (ok) snprintf(tt->format, sizeof(tt->format), "%s", "tok.json");
    } else {
        ok = parse_tiktoken_vocab(tt, (const unsigned char *)data, size) == 0;
        if (ok) snprintf(tt->format, sizeof(tt->format), "%s", "tiktoken");
    }
    if (!ok) {
        ttbpe_free(tt);
        return NULL;
    }
    if (tt->ht.cap == 0) { /* nothing was inserted at all */
        tt_set_error("vocabulary contains no tokens");
        ttbpe_free(tt);
        return NULL;
    }
    return tt;
}

static int read_whole_file(const char *path, buf_t *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tt_set_error("cannot open vocabulary file: %s", path);
        return -1;
    }
#if defined(_WIN32)
    if (_fseeki64(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        tt_set_error("cannot seek in %s", path);
        return -1;
    }
    __int64 sz = _ftelli64(fp);
    _fseeki64(fp, 0, SEEK_SET);
#else
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        tt_set_error("cannot seek in %s", path);
        return -1;
    }
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
#endif
    if (sz < 0) {
        fclose(fp);
        tt_set_error("cannot size %s", path);
        return -1;
    }
    if (buf_reserve(out, (size_t)sz) != 0) {
        fclose(fp);
        tt_set_error("out of memory reading %s", path);
        return -1;
    }
    size_t rd = fread(out->p, 1, (size_t)sz, fp);
    fclose(fp);
    if (rd != (size_t)sz) {
        tt_set_error("short read on %s", path);
        return -1;
    }
    out->len = rd;
    return 0;
}

tiktoken_bpe_t *ttbpe_load(const char *path) {
    tt_clear_error();
    if (!path) {
        tt_set_error("ttbpe_load: NULL path");
        return NULL;
    }
    buf_t file;
    file.p = NULL;
    file.len = file.cap = 0;
    if (read_whole_file(path, &file) != 0) return NULL;
    tiktoken_bpe_t *tt = ttbpe_load_from_memory(file.p, file.len);
    buf_free(&file);
    return tt;
}

void ttbpe_free(tiktoken_bpe_t *tt) {
    if (!tt) return;
    buf_free(&tt->arena);
    free(tt->ids);
    free(tt->present);
    free(tt->ht.slots);
    free(tt->specials);
    free(tt->word_split);
    free(tt->cache);
    free(tt->cache_head);
    free(tt->cache_ids);
    free(tt);
}

/* ------------------------------------------------------------------ */
/* introspection                                                       */
/* ------------------------------------------------------------------ */

size_t ttbpe_vocab_size(const tiktoken_bpe_t *tt) {
    return tt ? (size_t)tt->vocab_size : 0;
}

const char *ttbpe_format(const tiktoken_bpe_t *tt) {
    return tt ? tt->format : NULL;
}

const char *ttbpe_word_split(const tiktoken_bpe_t *tt) {
    return tt ? tt->word_split : NULL;
}

size_t ttbpe_special_count(const tiktoken_bpe_t *tt) {
    return tt ? tt->n_specials : 0;
}

int ttbpe_id_bytes(const tiktoken_bpe_t *tt, int id, const unsigned char **bytes,
                   size_t *len) {
    if (!tt || id < 0 || id >= tt->id_cap || !tt->present[id]) return -1;
    span_t sp = tt->ids[id];
    if (bytes) *bytes = tt->arena.p + sp.off;
    if (len) *len = sp.len;
    return 0;
}

void ttbpe_free_ids(int *ids) { free(ids); }
void ttbpe_free_buf(void *p) { free(p); }
