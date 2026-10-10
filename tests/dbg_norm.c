#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main(void) {
    const char *inputs[] = { "hello world", "the quick brown fox", "abc XYZ 123", "hello 世界 tokenizer" };
    for (int i = 0; i < 4; i++) {
        size_t nl = 0;
        char *n = tokenizer_normalize(inputs[i], strlen(inputs[i]), &nl);
        printf("[%d] in='%s' normlen=%zu", i, inputs[i], nl);
        if (n) {
            printf(" hex=");
            for (size_t k = 0; k < nl; k++) printf("%02X ", (unsigned char)n[k]);
            printf("\n");
            tokenizer_free_text(n);
        } else {
            printf(" norm FAIL: %s\n", tokenizer_last_error() ? tokenizer_last_error() : "?");
        }
    }
    return 0;
}
