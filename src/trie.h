/* trie.h - array-based prefix Trie used to accelerate piece matching during
 * Viterbi DP encoding (internal). */
#ifndef TOKENIZER_TRIE_H
#define TOKENIZER_TRIE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trie trie_t;

trie_t *trie_new(void);
void trie_free(trie_t *t);

/* Insert key -> id. Returns 0 on success, -1 on OOM. Re-insertion of an
 * existing key overwrites the stored id. */
int trie_insert(trie_t *t, const char *key, size_t len, int id);

/* Exact lookup. Returns stored id or -1. */
int trie_lookup(const trie_t *t, const char *key, size_t len);

/* Match all pieces starting at data[pos].
 * ids[] / ends[] receive the piece id and the end byte offset (exclusive).
 * Returns the number of matches (<= max_results). Ends are strictly > pos. */
int trie_match_prefixes(const trie_t *t, const char *data, size_t dlen,
                        size_t pos, int *ids, size_t *ends, int max_results);

/* Number of nodes (diagnostics). */
size_t trie_node_count(const trie_t *t);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_TRIE_H */
