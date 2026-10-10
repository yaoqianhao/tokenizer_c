/* trie.c - array-based prefix Trie.
 *
 * Nodes are kept in a growable array. Each node owns a small sorted-by-byte
 * child vector; lookups walk the trie byte-by-byte, giving O(piece_len *
 * log(degree)) matching (degree is tiny in practice, effectively O(piece_len)).
 */
#include "trie.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct child {
    uint8_t byte;
    int node;
} child_t;

typedef struct node {
    child_t *children;
    int n_children;
    int cap_children;
    int piece_id; /* -1 => not a terminal */
} node_t;

struct trie {
    node_t *nodes;
    size_t n_nodes;
    size_t cap_nodes;
    int failed;
};

static node_t *trie_node_at(trie_t *t, size_t i) {
    return &t->nodes[i];
}

static int node_child_index(const node_t *n, uint8_t byte) {
    int lo = 0, hi = n->n_children - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint8_t b = n->children[mid].byte;
        if (b == byte) return mid;
        if (b < byte) lo = mid + 1;
        else hi = mid - 1;
    }
    return -(lo + 1); /* insertion point */
}

static int node_child_get(const node_t *n, uint8_t byte) {
    int idx = node_child_index(n, byte);
    if (idx < 0) return -1;
    return n->children[idx].node;
}

trie_t *trie_new(void) {
    trie_t *t = (trie_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->cap_nodes = 16;
    t->nodes = (node_t *)calloc(t->cap_nodes, sizeof(node_t));
    if (!t->nodes) {
        free(t);
        return NULL;
    }
    t->n_nodes = 1; /* root */
    t->nodes[0].piece_id = -1; /* root is never a terminal */
    return t;
}

void trie_free(trie_t *t) {
    if (!t) return;
    for (size_t i = 0; i < t->n_nodes; i++) {
        free(t->nodes[i].children);
    }
    free(t->nodes);
    free(t);
}

static int trie_grow_node(trie_t *t, node_t *n, int need) {
    (void)t;
    if (n->cap_children >= need) return 0;
    int newcap = n->cap_children ? n->cap_children * 2 : 4;
    while (newcap < need) newcap *= 2;
    child_t *nc = (child_t *)realloc(n->children, (size_t)newcap * sizeof(child_t));
    if (!nc) return -1;
    n->children = nc;
    n->cap_children = newcap;
    return 0;
}

static int trie_add_node(trie_t *t) {
    if (t->n_nodes >= t->cap_nodes) {
        size_t newcap = t->cap_nodes * 2;
        node_t *nn = (node_t *)realloc(t->nodes, newcap * sizeof(node_t));
        if (!nn) return -1;
        t->nodes = nn;
        memset(t->nodes + t->cap_nodes, 0, (newcap - t->cap_nodes) * sizeof(node_t));
        t->cap_nodes = newcap;
    }
    int idx = (int)(t->n_nodes++);
    t->nodes[idx].piece_id = -1; /* new nodes are not terminals */
    return idx;
}

int trie_insert(trie_t *t, const char *key, size_t len, int id) {
    if (!t || !key) return -1;
    int cur = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t b = (uint8_t)key[i];
        node_t *n = trie_node_at(t, (size_t)cur);
        int idx = node_child_index(n, b);
        if (idx < 0) {
            int nxt = trie_add_node(t);
            if (nxt < 0) return -1;
            /* trie_add_node may have reallocated t->nodes; refresh n */
            n = trie_node_at(t, (size_t)cur);
            if (trie_grow_node(t, n, n->n_children + 1) != 0) return -1;
            int ipos = -(idx + 1);
            memmove(&n->children[ipos + 1], &n->children[ipos],
                    sizeof(child_t) * (size_t)(n->n_children - ipos));
            n->children[ipos].byte = b;
            n->children[ipos].node = nxt;
            n->n_children++;
            cur = nxt;
        } else {
            cur = n->children[idx].node;
        }
    }
    node_t *n = trie_node_at(t, (size_t)cur);
    n->piece_id = id;
    return 0;
}

int trie_lookup(const trie_t *t, const char *key, size_t len) {
    if (!t || !key) return -1;
    int cur = 0;
    for (size_t i = 0; i < len; i++) {
        node_t *n = (node_t *)trie_node_at((trie_t *)t, (size_t)cur);
        int c = node_child_get(n, (uint8_t)key[i]);
        if (c < 0) return -1;
        cur = c;
    }
    return trie_node_at((trie_t *)t, (size_t)cur)->piece_id;
}

int trie_match_prefixes(const trie_t *t, const char *data, size_t dlen,
                        size_t pos, int *ids, size_t *ends, int max_results) {
    if (!t || !data || pos >= dlen || max_results <= 0) return 0;
    int count = 0;
    int cur = 0;
    for (size_t i = pos; i < dlen; i++) {
        node_t *n = (node_t *)trie_node_at((trie_t *)t, (size_t)cur);
        int c = node_child_get(n, (uint8_t)data[i]);
        if (c < 0) break;
        cur = c;
        node_t *m = (node_t *)trie_node_at((trie_t *)t, (size_t)cur);
        if (m->piece_id >= 0) {
            ids[count] = m->piece_id;
            ends[count] = i + 1;
            count++;
            if (count >= max_results) break;
        }
    }
    return count;
}

size_t trie_node_count(const trie_t *t) {
    return t ? t->n_nodes : 0;
}
