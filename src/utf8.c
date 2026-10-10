/* utf8.c - minimal UTF-8 decode/encode helpers (internal). */
#include "utf8.h"

int utf8_seq_len(uint8_t lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0;
}

int utf8_decode_at(const char *s, size_t len, size_t i,
                   uint32_t *cp, size_t *advance) {
    if (i >= len) {
        *cp = 0;
        *advance = 0;
        return 0;
    }
    const uint8_t *b = (const uint8_t *)s;
    uint8_t c0 = b[i];
    if (c0 < 0x80) {
        *cp = c0;
        *advance = 1;
        return 1;
    }
    int n = utf8_seq_len(c0);
    if (n == 0 || i + (size_t)n > len) {
        /* Malformed: treat the raw byte as a code point for byte-fallback. */
        *cp = c0;
        *advance = 1;
        return 0;
    }
    uint32_t cpv = 0;
    switch (n) {
        case 2:
            if ((b[i+1] & 0xC0) != 0x80) break;
            cpv = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(b[i+1] & 0x3F);
            if (cpv < 0x80) break;             /* overlong */
            *cp = cpv; *advance = 2; return 1;
        case 3:
            if ((b[i+1] & 0xC0) != 0x80 || (b[i+2] & 0xC0) != 0x80) break;
            cpv = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(b[i+1] & 0x3F) << 6)
                | (uint32_t)(b[i+2] & 0x3F);
            if (cpv < 0x800) break;            /* overlong */
            if (cpv >= 0xD800 && cpv <= 0xDFFF) break; /* surrogate */
            *cp = cpv; *advance = 3; return 1;
        case 4:
            if ((b[i+1] & 0xC0) != 0x80 || (b[i+2] & 0xC0) != 0x80 ||
                (b[i+3] & 0xC0) != 0x80) break;
            cpv = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(b[i+1] & 0x3F) << 12)
                | ((uint32_t)(b[i+2] & 0x3F) << 6) | (uint32_t)(b[i+3] & 0x3F);
            if (cpv < 0x10000 || cpv > 0x10FFFF) break; /* overlong / out of range */
            *cp = cpv; *advance = 4; return 1;
        default:
            break;
    }
    *cp = c0;
    *advance = 1;
    return 0;
}

int utf8_encode(uint32_t cp, char out[4]) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}
