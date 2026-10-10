#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main(void) {
    tokenizer_t *tok = tokenizer_load("t_model.tkm");
    if (!tok) { printf("load FAIL: %s\n", tokenizer_last_error()); return 1; }
    printf("vocab=%d\n", (int)tokenizer_vocab_size(tok));
    const char *p = NULL; size_t l = 0;
    for (int i = 0; i < 12 && i < (int)tokenizer_vocab_size(tok); i++) {
        tokenizer_id_to_piece(tok, i, &p, &l);
        printf("  id %d len %zu: ", i, l);
        for (size_t k = 0; k < l; k++) printf("%02X ", (unsigned char)p[k]);
        printf(" | '%.*s'\n", (int)l, p);
    }
    int id = tokenizer_piece_to_id(tok, "hello", 5);
    printf("piece_to_id('hello')=%d\n", id);
    id = tokenizer_piece_to_id(tok, "\xE2\x96\x81", 3);
    printf("piece_to_id('mark')=%d\n", id);
    int *ids = NULL; size_t n = 0;
    int rc = tokenizer_encode(tok, "hello world", 11, &ids, &n);
    printf("encode rc=%d n=%zu err=%s\n", rc, n, tokenizer_last_error() ? tokenizer_last_error() : "none");
    tokenizer_free_ids(ids);
    tokenizer_free(tok);
    return 0;
}
