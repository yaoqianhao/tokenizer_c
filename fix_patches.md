---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_9d245ec3c56a11f19063525400393706
    ReservedCode1: 3kiPvD0kVcZnRtHBe0zbs9iwCSJ6cb/XrsuyXShkOjGISJSFZPqTuEEP1KA4WPbDKSk1au27CjBHNQo5NEp6h6kTQixScK+4bS24GjkoEbWHsL2tJiZMv+5mONotXfc34NTdEnWZP661B5+/l3lvoT0Rd7Qze3/TLk5Cll8uKaRUT4ddGcEvttr0e24=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_9d245ec3c56a11f19063525400393706
    ReservedCode2: 3kiPvD0kVcZnRtHBe0zbs9iwCSJ6cb/XrsuyXShkOjGISJSFZPqTuEEP1KA4WPbDKSk1au27CjBHNQo5NEp6h6kTQixScK+4bS24GjkoEbWHsL2tJiZMv+5mONotXfc34NTdEnWZP661B5+/l3lvoT0Rd7Qze3/TLk5Cll8uKaRUT4ddGcEvttr0e24=
---

# tokenizer_c 修改建议修复代码文档（fix_patches.md）

- 日期：2026-10-11
- 依据：《2026-10-11-code-review-fix-plan.md》（21 条 P0/P1/P2）与 `output/fix_plan_verification.md` 核实报告（含 2 条额外发现）
- 范围：P0-1 ~ P0-4、P1-1 ~ P1-9、P2-1 ~ P2-8 全部 21 条 + 额外发现 1、2，逐项给出可直接替换编译的修复代码
- 约定：行号以 `E:\tokenizer_c` 当前源码为准（标注“约”表示函数内部近似位置）；所有补丁保持现有 C11 风格（英文注释、`malloc`/`free`、`goto done` 清理模式）

## 覆盖统计

| 类别 | 条目 | 状态 |
|---|---|---|
| P0 | P0-1 / P0-2 / P0-3 / P0-4 | 4 条均给出代码 |
| P1 | P1-1 ~ P1-9 | 9 条均给出代码（P1-5、P1-7 按“部分属实”口径修复） |
| P2 | P2-1 ~ P2-8 | 8 条均给出代码（P2-2 为可选补丁，建议按 fix-plan 暂缓） |
| 额外发现 | 发现 1 / 发现 2 | 2 条均给出代码（发现 1 并入 P1-2 补丁） |
| 无法给出代码 | 无 | P2-2 已给可选实现，是否落地由用户决定 |

---

## 前置：公共 64 位文件大小辅助（P0-1 共用）

新增到 `src/tokenizer_internal.h`（在 `#include <stdint.h>` 之后、`extern "C"` 之前）：

```c
/* ---- 64-bit file size helper (P0-1) ---- */
#if defined(_WIN32)
static int file_size64(FILE *fp, int64_t *out) {
    __int64 cur = _ftelli64(fp);
    if (cur < 0) return -1;
    if (_fseeki64(fp, 0, SEEK_END) != 0) return -1;
    __int64 sz = _ftelli64(fp);
    _fseeki64(fp, cur, SEEK_SET);
    if (sz < 0) return -1;
    *out = (int64_t)sz;
    return 0;
}
#else
static int file_size64(FILE *fp, int64_t *out) {
    off_t cur = ftello(fp);
    if (cur < 0) return -1;
    if (fseeko(fp, 0, SEEK_END) != 0) return -1;
    off_t sz = ftello(fp);
    fseeko(fp, cur, SEEK_SET);
    if (sz < 0) return -1;
    *out = (int64_t)sz;
    return 0;
}
#endif
```

接口说明：新增内部静态函数，无公共头文件变更；Windows 走 `_fseeki64/_ftelli64`，POSIX 走 `fseeko/ftello`。`tokenizer_internal.h` 需补充 `#include <stdio.h>`（若当前未间接引入）。

---

## P0 严重级

### P0-1 32-bit `ftell`（4 处调用点）

**文件 / 位置**
- `src/trainer.c` `corpus_load`（约 L42-43）
- `src/model.c` `tokenizer_load_binary_file`（约 L131）
- `src/sentencepiece.c` `tokenizer_load_sentencepiece`（约 L430）
- `cli/main.c` `run_encode` `--file` 分支（约 L215）

**修复前（以 trainer.c 为例，其余 3 处同构）**

```c
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        tokenizer_set_error("train: empty corpus");
        return -1;
    }
```

**修复后**

```c
    int64_t sz = 0;
    if (file_size64(fp, &sz) != 0 || sz <= 0) {
        fclose(fp);
        tokenizer_set_error("train: empty corpus");
        return -1;
    }
```

后续 `(size_t)sz` 的用法不变（P0-2 之后 trainer.c 不再整文件读入，仅用 sz 做非空校验；model.c / sentencepiece.c / main.c 仍 `malloc((size_t)sz)` 后 `fread`）。

`cli/main.c` 不引用 `tokenizer_internal.h`，在该文件内直接内联等价实现（避免引入内部头）：

```c
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
```

调用点替换为：

```c
        long long sz = 0;
        if (cli_file_size64(fp, &sz) != 0 || sz <= 0) {
            fclose(fp); tokenizer_free(tok); return 1;
        }
        content = (char *)malloc((size_t)sz + 1);
```

**验证方法**
1. 构造 >2GB 语料（或临时用 `fsutil file createnew` 生成 3GB 文件）跑 `train --corpus`，确认不再报空/错误且能读取。
2. `encode --file <2GB 以上文件>` 输出非空。
3. 小文件回归：`train` + `encode` + `decode` 结果与修复前一致。

---

### P0-2 `corpus_load` 整文件加载 + 按行再拷贝（峰值 2×）

**文件 / 位置**：`src/trainer.c` `corpus_load`（函数整体替换，约 L31-101）

**问题摘要**：`char *data = malloc((size_t)sz + 1)` 整文件读入后又为每行 `memcpy` 一份，峰值内存约 2× 语料体积。改为流式按行读取，每行只保留一份拷贝。`corpus_t` 结构与 E step 遍历逻辑不变。

**修复前**（整函数，节选关键段）

```c
    char *data = (char *)malloc((size_t)sz + 1);
    if (!data) { ... }
    size_t rd = fread(data, 1, (size_t)sz, fp);
    ...
    data[rd] = '\0';
    ...
    for (size_t i = 0; i <= rd; i++) {
        if (i == rd || data[i] == '\n') {
            ...
            char *copy = (char *)malloc(llen + 1);
            memcpy(copy, data + start, llen);
            ...
        }
    }
    free(data);
```

**修复后**（整函数替换；新增一个自实现 `read_line`，不依赖 POSIX `getline`）

```c
/* Read one line (including '\n') into *out (NUL-terminated).
 * Returns line length >= 0, or -1 on EOF-without-data / OOM. */
static ssize_t read_line(char **out, size_t *cap, FILE *fp) {
    if (*out == NULL || *cap == 0) {
        *cap = 256;
        *out = (char *)malloc(*cap);
        if (!*out) return -1;
    }
    size_t n = 0;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (n + 1 >= *cap) {
            size_t ncap = *cap * 2;
            char *nb = (char *)realloc(*out, ncap);
            if (!nb) return -1;
            *out = nb;
            *cap = ncap;
        }
        (*out)[n++] = (char)c;
        if (c == '\n') break;
    }
    if (n == 0 && c == EOF) return -1; /* clean EOF */
    (*out)[n] = '\0';
    return (ssize_t)n;
}

static int corpus_load(const char *path, corpus_t *c) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        tokenizer_set_error("train: cannot open corpus %s", path);
        return -1;
    }
    int64_t sz = 0;
    if (file_size64(fp, &sz) != 0 || sz <= 0) {
        fclose(fp);
        tokenizer_set_error("train: empty corpus");
        return -1;
    }

    c->lines = NULL;
    c->lens = NULL;
    c->count = 0;
    c->cap = 0;

    /* skip UTF-8 BOM if present (peek 3 bytes, rewind if not BOM) */
    int bom = 0;
    {
        unsigned char hdr[3];
        size_t h = fread(hdr, 1, 3, fp);
        if (h == 3 && hdr[0] == 0xEF && hdr[1] == 0xBB && hdr[2] == 0xBF) {
            bom = 1;
        } else if (h > 0) {
            if (fseek(fp, (long)h, SEEK_CUR) != 0) { /* rewind already read bytes */
                fclose(fp);
                tokenizer_set_error("train: cannot seek corpus");
                return -1;
            }
        }
    }

    char *line = NULL;
    size_t linecap = 0;
    ssize_t llen = 0;
    size_t li = 0;
    while ((llen = read_line(&line, &linecap, fp)) >= 0) {
        size_t start = 0;
        size_t end = (size_t)llen;
        if (li == 0 && bom && end >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
            start = 3;
        }
        while (end > start && line[end - 1] == '\r') end--;
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
            size_t llen_c = end - start;
            if (llen_c > TRAIN_MAX_LINE) llen_c = TRAIN_MAX_LINE; /* cap */
            char *copy = (char *)malloc(llen_c + 1);
            if (!copy) goto oom;
            memcpy(copy, line + start, llen_c);
            copy[llen_c] = '\0';
            c->lines[c->count] = copy;
            c->lens[c->count] = llen_c;
            c->count++;
        }
        li++;
    }
    free(line);
    if (ferror(fp)) {
        fclose(fp);
        tokenizer_set_error("train: read error on corpus");
        return -1;
    }
    fclose(fp);
    return 0;
oom:
    free(line);
    fclose(fp);
    for (size_t i = 0; i < c->count; i++) free(c->lines[i]);
    free(c->lines);
    free(c->lens);
    tokenizer_set_error("train: OOM in corpus_load");
    return -1;
}
```

说明：行缓冲全程只有 1 份（每行 `copy` 是唯一驻留拷贝），峰值内存 ≈ 语料体积 + 单行缓冲。`ssize_t` 需 `<sys/types.h>`（Windows MSVC 无 `ssize_t`，可改为 `long long` 或 `int`；本补丁按 gcc/mingw 风格给出，MSVC 下把 `ssize_t` 换成 `ptrdiff_t` 即可）。

**验证方法**
1. 修复前后同一小语料训练，词表与 ll 曲线一致（95 项测试全过）。
2. `train_big.bat` 3GB 语料下观察峰值内存：`Get-Process` 或任务管理器确认峰值由 ~6GB 降为 ~3GB。
3. 含 BOM、CRLF、超长行（>4096）的语料均正确截断/剥离。

---

### P0-3 逐行 verbose 日志

**文件 / 位置**：`src/trainer.c` seed 循环内（约 L757，位于 `if (!resumed)` 块中）

**修复前**

```c
        if (verbose) fprintf(stderr, "[train] seed line %zu norm_len=%zu\n", li, nlen);
```

**修复后**

```c
        if (verbose && (li % 100000 == 0))
            fprintf(stderr, "[train] seed line %zu / %zu\n", li, corpus.count);
```

**验证方法**：`train -v` 输出行数由 N 条变为 ~N/100000 条；其余行为不变。`train_big.bat` 日志体积显著下降。

---

### P0-4 剪枝插入排序 O(n²) 改 qsort

**文件 / 位置**：`src/trainer.c` 剪枝段（L935-938 的插入排序）；比较器提升为文件级 static

**修复前**

```c
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
```

**修复后**（先在文件级加入类型与比较器，替换块内排序）

```c
/* P0-4: pruned-pair sort key (defined at file scope for qsort comparator) */
typedef struct {
    float score;
    int id;
} pruned_pair_t;

static int pruned_pair_cmp_desc(const void *a, const void *b) {
    const pruned_pair_t *x = (const pruned_pair_t *)a;
    const pruned_pair_t *y = (const pruned_pair_t *)b;
    if (y->score != x->score) return y->score > x->score ? 1 : -1;
    /* tie-break by ascending id: deterministic, stable-ish */
    return (x->id > y->id) - (x->id < y->id);
}
```

```c
                    pruned_pair_t *pairs =
                        (pruned_pair_t *)malloc((size_t)bn[1] * sizeof(pruned_pair_t));
                    if (pairs) {
                        for (int i = 0; i < bn[1]; i++) {
                            pairs[i].score = scores[buckets[1][i]];
                            pairs[i].id = buckets[1][i];
                        }
                        qsort(pairs, (size_t)bn[1], sizeof(pruned_pair_t),
                              pruned_pair_cmp_desc);
                        for (int i = 0; i < bn[1]; i++) buckets[1][i] = pairs[i].id;
                        free(pairs);
                    }
```

**验证方法**
1. 排序结果与插入排序在无并列分数时完全一致；并列分数下顺序确定（id 升序）。
2. 大词表（>10 万）单轮剪枝耗时从秒级/分钟级降为毫秒级。
3. 95 项测试回归通过。

---

## P1 高优先级

### P1-1 种子不是 Top-K

**文件 / 位置**：`src/trainer.c` `tokenizer_train`（`h = hash_new(...)` 约 L741；`keep = nkeys < max_candidates ...` L750）；`src/trainer.c` `hash_get_or_create`（L194-196 硬 cap 拒绝）

**问题摘要**：满哈希拒绝新 key，种子是“先出现的前 max_candidates 条”而非全量 Top-K。

**修复前**

```c
    h = hash_new(65536, max_candidates);
```

**修复后**（主方案：放宽硬 cap 8 倍，避免内存失控；若需严格 Top-K 见“严格选项”）

```c
    /* P1-1: 种子 Top-K。hash 硬 cap 放宽 8 倍：3GB 语料 distinct n-gram
     * 远超 max_candidates 时仍会拒绝新 key，但绝大多数场景下可收集到
     * 足够覆盖 Top-K 的候选，随后统一 qsort 截断。内存成本约每 entry
     * 24~32B，按 max_candidates=1000000 计约 256MB 上限内。 */
    size_t seed_cap = max_candidates > SIZE_MAX / 8 ? SIZE_MAX : max_candidates * 8;
    h = hash_new(65536, seed_cap);
```

**严格选项（真正全量 Top-K，内存风险需自行评估）**：`hash_new(65536, SIZE_MAX)`——取消硬 cap，seed 阶段收集全部 distinct n-gram 后 `qsort` 再截断 `keep`。3GB 语料 distinct n-gram 可能达数千万至上亿条，内存 4~10GB，仅在机器内存充足时使用。

**验证方法**
1. 构造含大量低频长尾的语料（distinct n-gram > max_candidates），对比修复前后种子词表：修复后应包含更多高覆盖候选。
2. 训练收敛后词表 PPL 不劣化。

---

### P1-2 Viterbi 每位置缺默认 byte 边 + 额外发现 1（训练端修补分支未写 best/prev）

**文件 / 位置**
- `src/tokenizer.c` `tokenizer_encode`（DP 循环，约 L270-295）
- `src/trainer.c` `viterbi_segment`（DP 循环，L338 起）

**修复前（推理端 tokenizer.c）**

```c
        int m = trie_match_prefixes(tok->trie, s, n, i, ids, ends, MAX_MATCH);
        int any_updated = 0;
        for (int k = 0; k < m; k++) {
            size_t j = ends[k];
            int pid = ids[k];
            double cand = dp[i] + (double)tok->scores[pid];
            if (cand > dp[j]) {
                dp[j] = cand;
                best[j] = pid;
                prev[j] = i;
                any_updated = 1;
            }
        }
        /* byte-fallback: guarantee progress by at least one byte */
        if (dp[i + 1] == -INFINITY && !any_updated) {
            dp[i + 1] = dp[i] + kByteFallbackScore;
            best[i + 1] = -1 - (int)(unsigned char)s[i];
            prev[i + 1] = i;
        }
```

**修复后（推理端 tokenizer.c）**：每位置无条件写入 1-byte 边，移除条件兜底

```c
        int m = trie_match_prefixes(tok->trie, s, n, i, ids, ends, MAX_MATCH);
        for (int k = 0; k < m; k++) {
            size_t j = ends[k];
            int pid = ids[k];
            double cand = dp[i] + (double)tok->scores[pid];
            if (cand > dp[j]) {
                dp[j] = cand;
                best[j] = pid;
                prev[j] = i;
            }
        }
        /* P1-2: unconditional 1-byte edge so every position keeps a default
         * byte-fallback transition; the old "only when unreachable" repair
         * could leave dp holes that backtracking never resolved. */
        double fb = dp[i] + kByteFallbackScore;
        if (fb > dp[i + 1]) {
            dp[i + 1] = fb;
            best[i + 1] = -1 - (int)(unsigned char)s[i];
            prev[i + 1] = i;
        }
```

（`dp[i] == -INFINITY` 的防御修补保留；由于 byte 边保证连续性，正常情况下不会触发。）

**修复前（训练端 trainer.c viterbi_segment）**

```c
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
```

**修复后（训练端 trainer.c viterbi_segment）**：与推理端统一；同时修复额外发现 1（修补分支补齐 `best`/`prev` 写入）

```c
        int m = trie_match_prefixes(trie, s, n, i, ids, ends, MAX_MATCH);
        for (int k = 0; k < m; k++) {
            size_t j = ends[k];
            int pid = ids[k];
            if (pid >= vocab_size) continue;
            double cand = dp[i] + (double)scores[pid];
            if (cand > dp[j]) {
                dp[j] = cand;
                best[j] = pid;
                prev[j] = i;
            }
        }
        /* P1-2 + 额外发现 1: unconditional byte edge AND best/prev are
         * always written, so DP holes are fully repaired on the train side
         * (the old branch wrote dp but not best/prev). */
        double fb = dp[i] + kFbScore;
        if (fb > dp[i + 1]) {
            dp[i + 1] = fb;
            best[i + 1] = vocab_size + (unsigned char)s[i]; /* fallback marker */
            prev[i + 1] = i;
        }
```

**MAX_MATCH=64 动态化（可选增强）**：需扩展 `src/trie.h` 的 `trie_match_prefixes`（增加可重复调用的 offset 参数或在 trie.c 提供全量收集接口），本补丁不包含该接口变更；若长前缀丢失影响实测，再按“trie 接口扩展”单独排期。

**验证方法**
1. 编码任意二进制/生僻文本：`encode | decode` 往返无损，OOV=0。
2. 训练端：对比修复前后 E step 累计 `ll`，修复后 ll 单调合理（不再出现 DP 空洞导致的异常路径）。
3. 95 项测试回归通过。

---

### P1-3 `fold_case` 不入模型

**文件 / 位置**
- `src/tokenizer_internal.h`：`struct tokenizer`（增加字段）
- `src/model.c`：`tokenizer_save_binary`（flags 写入）、`tokenizer_load_binary_mem`（flags 读取）
- `src/tokenizer.c`：`normalize_for_encode`（约 L190）
- `src/trainer.c`：`tokenizer_train` 建模型处（`tokenizer_create` 后）

**修复前**

```c
    /* tokenizer_internal.h */
    int byte_fallback;  /* nonzero => <0xXX> fallback enabled */
```

```c
    /* model.c save */
    put_u32(buf + o, tok->byte_fallback ? 1u : 0u); o += 4;
    /* model.c load */
    int byte_fb = (int)(flags & 1u);
```

```c
    /* tokenizer.c normalize_for_encode */
    return normalize_text(text, len, 1 /*mark_spaces*/, 0 /*fold_case*/,
                          out_len);
```

**修复后**

```c
    /* tokenizer_internal.h */
    int byte_fallback;  /* nonzero => <0xXX> fallback enabled */
    int fold_case;      /* nonzero => normalize folded case at encode (P1-3) */
```

```c
    /* model.c save: flags bit0 = byte_fallback, bit1 = fold_case */
    put_u32(buf + o, (tok->byte_fallback ? 1u : 0u) |
                     (tok->fold_case ? 2u : 0u)); o += 4;
    /* model.c load */
    int byte_fb = (int)(flags & 1u);
    int fold_case = (int)((flags >> 1) & 1u);
```

```c
    /* tokenizer.c normalize_for_encode */
    return normalize_text(text, len, 1 /*mark_spaces*/, tok->fold_case,
                          out_len);
```

```c
    /* trainer.c: after tokenizer_create(...) */
    tok->fold_case = fold_case;
```

**验证方法**
1. `train --fold-case` 后 `encode "ABC"` 与 `encode "abc"` 输出相同 id 序列（修复前不同）。
2. 旧模型（v1，bit1=0）加载后 `fold_case=0`，行为与修复前一致——向后兼容无需升版本。
3. `.tkm` 与 `.model` 往返后 `fold_case` 标志保留。

---

### P1-4 last error 粘滞

**文件 / 位置**：`src/tokenizer.c`（`tls_error` / `tokenizer_last_error`，约 L35-50）；成功出口

**修复前**

```c
const char *tokenizer_last_error(void) {
    return tls_error[0] ? tls_error : NULL;
}
```

**修复后**：新增内部清除接口，所有成功出口调用

```c
void tokenizer_clear_error(void) {
    tls_error[0] = '\0';
}
```

`src/tokenizer_internal.h` 增加声明：

```c
/* Clear the thread-local last-error buffer (call on every success path). */
void tokenizer_clear_error(void);
```

在各成功出口追加 `tokenizer_clear_error();`：
- `tokenizer_encode` 末尾 `return 0;` 前
- `tokenizer_decode` 末尾 `return 0;` 前
- `tokenizer_load` / `tokenizer_load_from_memory` 成功 `return tok;` 前
- `src/model.c` `tokenizer_load_binary_mem` 成功 `return tok;` 前
- `src/sentencepiece.c` `tokenizer_load_sentencepiece` 成功 `return tok;` 前、`tokenizer_export_json` / `tokenizer_export_sentencepiece` 成功 `return 0;` 前
- `src/trainer.c` `tokenizer_train` 的 `done:` 中 `rc == 0` 时

**验证方法**
1. 先触发一次失败（如 `decode` 越界 id），再执行一次成功 `load`，`tokenizer_last_error()` 返回 NULL（修复前仍返回旧错误）。
2. 95 项测试通过（尤其 `tests/test_tokenizer.c` L277-280 的粘滞注释场景）。

---

### P1-5 `realloc` 先写后失败（3 处）

**文件 / 位置**
- `src/tokenizer.c` `tokenizer_ensure_piece_slots`（约 L60）
- `src/trainer.c` `seed_push`（L263-275）
- `src/sentencepiece.c` `sp_list_push`（约 L215）

**修复前（tokenizer.c 示例）**

```c
static int tokenizer_ensure_piece_slots(tokenizer_t *tok, int need) {
    if (need < 0 || need > 100000000) return -1; /* sanity */
    char **np = (char **)realloc(tok->pieces, (size_t)need * sizeof(char *));
    if (!np) return -1;
    tok->pieces = np;
    int *nl = (int *)realloc(tok->piece_lens, (size_t)need * sizeof(int));
    if (!nl) return -1;
    tok->piece_lens = nl;
    float *ns = (float *)realloc(tok->scores, (size_t)need * sizeof(float));
    if (!ns) return -1;
    tok->scores = ns;
    ...
}
```

**修复后（tokenizer.c）**：三块先独立 malloc+拷贝，全部成功再统一替换

```c
static int tokenizer_ensure_piece_slots(tokenizer_t *tok, int need) {
    if (need < 0 || need > 100000000) return -1; /* sanity */
    char **np = (char **)malloc((size_t)need * sizeof(char *));
    int *nl = (int *)malloc((size_t)need * sizeof(int));
    float *ns = (float *)malloc((size_t)need * sizeof(float));
    if (!np || !nl || !ns) {
        free(np); free(nl); free(ns);
        return -1;
    }
    size_t old = (size_t)tok->vocab_size;
    if (old > 0) {
        memcpy(np, tok->pieces, old * sizeof(char *));
        memcpy(nl, tok->piece_lens, old * sizeof(int));
        memcpy(ns, tok->scores, old * sizeof(float));
    }
    /* commit only after all allocations succeeded (P1-5) */
    free(tok->pieces);
    free(tok->piece_lens);
    free(tok->scores);
    tok->pieces = np;
    tok->piece_lens = nl;
    tok->scores = ns;
    for (int i = (int)old; i < need; i++) {
        tok->pieces[i] = NULL;
        tok->piece_lens[i] = 0;
        tok->scores[i] = 0.0f;
    }
    tok->vocab_size = need;
    return 0;
}
```

**修复后（trainer.c seed_push，仅替换扩容块）**

```c
    if (s->size >= s->cap) {
        int ncap = s->cap ? s->cap * 2 : 4096;
        char **np = (char **)malloc((size_t)ncap * sizeof(char *));
        int *nl = (int *)malloc((size_t)ncap * sizeof(int));
        double *nc = (double *)malloc((size_t)ncap * sizeof(double));
        if (!np || !nl || !nc) {
            free(np); free(nl); free(nc);
            return -1;
        }
        if (s->size > 0) {
            memcpy(np, s->pieces, (size_t)s->size * sizeof(char *));
            memcpy(nl, s->lens, (size_t)s->size * sizeof(int));
            memcpy(nc, s->counts, (size_t)s->size * sizeof(double));
        }
        free(s->pieces); free(s->lens); free(s->counts);
        s->pieces = np; s->lens = nl; s->counts = nc;
        s->cap = ncap;
    }
```

**修复后（sentencepiece.c sp_list_push，仅替换扩容块）**

```c
    if (l->count >= l->cap) {
        int ncap = l->cap ? l->cap * 2 : 64;
        char **np = (char **)malloc((size_t)ncap * sizeof(char *));
        float *ns = (float *)malloc((size_t)ncap * sizeof(float));
        int *nt = (int *)malloc((size_t)ncap * sizeof(int));
        if (!np || !ns || !nt) {
            free(np); free(ns); free(nt);
            return -1;
        }
        if (l->count > 0) {
            memcpy(np, l->pieces, (size_t)l->count * sizeof(char *));
            memcpy(ns, l->scores, (size_t)l->count * sizeof(float));
            memcpy(nt, l->types, (size_t)l->count * sizeof(int));
        }
        free(l->pieces); free(l->scores); free(l->types);
        l->pieces = np; l->scores = ns; l->types = nt;
        l->cap = ncap;
    }
```

**验证方法**：三处扩容路径在低内存注入（如 `ulimit`/Job 对象限内存）下失败时槽位状态保持一致；`tokenizer_free` 不触碰未初始化槽位；95 项测试回归通过。

---

### P1-6 截断当成功

**文件 / 位置**
- `src/tokenizer.c` `tokenizer_decode`（约 L330-360，3 处 `break`）
- `src/normalize.c` `normalize_text` Pass 4（约 L155-175，3 处 `break`）

**修复前（tokenizer.c）**

```c
        if (id < 0 && id >= -256) {
            if (o + 1 >= cap) break;
            buf[o++] = (char)(-id - 1);
            continue;
        }
        ...
        if (is_byte_fallback_piece(p, plen, &byte)) {
            if (o + 1 >= cap) break;
            buf[o++] = (char)byte;
            continue;
        }
        ...
            if ((unsigned char)p[b] == 0xE2 && ...) {
                if (o + 1 >= cap) break;
                buf[o++] = ' ';
                ...
            } else {
                if (o + 1 >= cap) break;
                buf[o++] = p[b];
            }
    }
    buf[o] = '\0';
    ...
    return 0;
```

**修复后（tokenizer.c）**：三处 `break` 改为 `goto overflow;`，末尾统一失败返回

```c
        if (id < 0 && id >= -256) {
            if (o + 1 >= cap) goto overflow;
            buf[o++] = (char)(-id - 1);
            continue;
        }
        ...
        if (is_byte_fallback_piece(p, plen, &byte)) {
            if (o + 1 >= cap) goto overflow;
            buf[o++] = (char)byte;
            continue;
        }
        ...
            if ((unsigned char)p[b] == 0xE2 && ...) {
                if (o + 1 >= cap) goto overflow;
                buf[o++] = ' ';
                ...
            } else {
                if (o + 1 >= cap) goto overflow;
                buf[o++] = p[b];
            }
    }
    buf[o] = '\0';
    ...
    return 0;
overflow:
    free(buf);
    set_error("tokenizer_decode: output buffer overflow");
    return -1;
```

**修复前（normalize.c Pass 4，节选）**

```c
    for (size_t k = 0; k < wn; k++) {
        uint32_t cp = work[k];
        if (mark_spaces && cp == 0x0020) {
            if (o + 3 >= outcap) break;
            memcpy(out + o, kSpaceMark, 3);
            o += 3;
            continue;
        }
        char eb[4];
        int nb = utf8_encode(cp, eb);
        if (nb <= 0) {
            if (o + 1 >= outcap) break;
            out[o++] = '?';
        } else {
            if (o + (size_t)nb >= outcap) break;
            memcpy(out + o, eb, (size_t)nb);
            o += (size_t)nb;
        }
    }
    if (o >= outcap) o = outcap - 1;
    out[o] = '\0';
```

**修复后（normalize.c Pass 4）**：改为动态扩容，容量不足时失败而非截断成功

```c
    size_t outcap = 64;
    char *out = (char *)malloc(outcap);
    if (!out) {
        free(cps);
        if (norm) free(norm);
        return NULL;
    }
    size_t o = 0;
    for (size_t k = 0; k < wn; k++) {
        uint32_t cp = work[k];
        if (mark_spaces && cp == 0x0020) {
            if (o + 3 > outcap) {
                size_t ncap = outcap * 2;
                char *nb = (char *)realloc(out, ncap);
                if (!nb) goto oom_out;
                out = nb; outcap = ncap;
            }
            memcpy(out + o, kSpaceMark, 3);
            o += 3;
            continue;
        }
        char eb[4];
        int nb = utf8_encode(cp, eb);
        if (nb <= 0) { nb = 1; eb[0] = '?'; }
        if (o + (size_t)nb > outcap) {
            size_t ncap = (outcap * 2 > o + (size_t)nb) ? outcap * 2
                                                        : o + (size_t)nb;
            char *nb2 = (char *)realloc(out, ncap);
            if (!nb2) goto oom_out;
            out = nb2; outcap = ncap;
        }
        memcpy(out + o, eb, (size_t)nb);
        o += (size_t)nb;
    }
    out[o] = '\0';
    free(cps);
    if (norm) free(norm);
    if (out_len) *out_len = o;
    return out;
oom_out:
    free(cps);
    if (norm) free(norm);
    free(out);
    return NULL;
```

**验证方法**
1. 构造超长 `ids` 列表 decode：输出完整且无截断（正常输入本不触发，防御性修复）。
2. 极端 Unicode 输入 normalize：输出与预期完整一致；低内存注入时返回 NULL（而非截断成功）。

---

### P1-7 SP 加载用 `strlen` 当 piece 长度

**文件 / 位置**：`src/sentencepiece.c` `sp_piece_list` 结构、`sp_list_push`、`sp_list_free`、`tokenizer_load_sentencepiece` 加载循环（约 L466）

**问题摘要**：`sp_list_push` 已保存真实 `plen` 但未存，加载时用 `strlen` 重算；对含内嵌 NUL 字节的 piece 会截断（实际影响低，仍顺手修复）。

**修复前**

```c
typedef struct {
    char **pieces;
    float *scores;
    int *types;
    int count, cap;
} sp_piece_list;
```

```c
    /* sp_list_push 内 */
    char *copy = (char *)malloc(plen + 1);
    if (!copy) return -1;
    memcpy(copy, p, plen);
    copy[plen] = '\0';
    l->pieces[l->count] = copy;
    l->scores[l->count] = score;
    l->types[l->count] = type;
    l->count++;
    return 0;
```

```c
    /* 加载循环 */
        if (tokenizer_set_piece(tok, i, pieces.pieces[i],
                                strlen(pieces.pieces[i]), pieces.scores[i]) != 0) {
```

**修复后**

```c
typedef struct {
    char **pieces;
    size_t *lens;       /* real piece byte lengths (P1-7) */
    float *scores;
    int *types;
    int count, cap;
} sp_piece_list;
```

`sp_list_push` 扩容块同步为“三块先齐后换”（同 P1-5 模式，`lens` 替换 `types` 位置即可）：

```c
    if (l->count >= l->cap) {
        int ncap = l->cap ? l->cap * 2 : 64;
        char **np = (char **)malloc((size_t)ncap * sizeof(char *));
        size_t *nl = (size_t *)malloc((size_t)ncap * sizeof(size_t));
        float *ns = (float *)malloc((size_t)ncap * sizeof(float));
        int *nt = (int *)malloc((size_t)ncap * sizeof(int));
        if (!np || !nl || !ns || !nt) {
            free(np); free(nl); free(ns); free(nt);
            return -1;
        }
        if (l->count > 0) {
            memcpy(np, l->pieces, (size_t)l->count * sizeof(char *));
            memcpy(nl, l->lens, (size_t)l->count * sizeof(size_t));
            memcpy(ns, l->scores, (size_t)l->count * sizeof(float));
            memcpy(nt, l->types, (size_t)l->count * sizeof(int));
        }
        free(l->pieces); free(l->lens); free(l->scores); free(l->types);
        l->pieces = np; l->lens = nl; l->scores = ns; l->types = nt;
        l->cap = ncap;
    }
```

入列时保存真实长度：

```c
    char *copy = (char *)malloc(plen + 1);
    if (!copy) return -1;
    memcpy(copy, p, plen);
    copy[plen] = '\0';
    l->pieces[l->count] = copy;
    l->lens[l->count] = plen;
    l->scores[l->count] = score;
    l->types[l->count] = type;
    l->count++;
    return 0;
```

`sp_list_free` 增加 `free(l->lens);`；加载循环改为：

```c
        if (tokenizer_set_piece(tok, i, pieces.pieces[i],
                                pieces.lens[i], pieces.scores[i]) != 0) {
```

**验证方法**
1. 加载 SentencePiece 官方 unigram 模型，词表长度与 `pieces.count` 一致。
2. 构造含内嵌 NUL 字节 piece 的测试 `.model`，加载后该 piece 不被截断（`piece_lens` 为真实长度）。
3. 95 项测试回归通过。

---

### P1-8 HF JSON `vocab` 形状错误（object 改数组）

**文件 / 位置**：`src/sentencepiece.c` `tokenizer_export_json`（约 L560）；`tests/test_tokenizer.c`（L350-352 断言）

**修复前**

```c
    fprintf(fp, "    \"vocab\": {\n");
    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(fp, "      \"");
        jstr(fp, tok->pieces[i], (size_t)tok->piece_lens[i]);
        fprintf(fp, "\": ");
        jscore(fp, tok->scores[i]);
        fprintf(fp, "%s\n", (i + 1 < tok->vocab_size) ? "," : "");
    }
    fprintf(fp, "    }\n");
```

**修复后**（官方 Unigram 契约：`[["<piece>", score], ...]`）

```c
    fprintf(fp, "    \"vocab\": [\n");
    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(fp, "      [\"");
        jstr(fp, tok->pieces[i], (size_t)tok->piece_lens[i]);
        fprintf(fp, "\", ");
        jscore(fp, tok->scores[i]);
        fprintf(fp, "]%s\n", (i + 1 < tok->vocab_size) ? "," : "");
    }
    fprintf(fp, "    ]\n");
```

**测试断言更新**（`tests/test_tokenizer.c` L350-352 区域）：把只搜 `"vocab"` 的弱断言升级为校验数组形状：

```c
    CHECK(strstr(buf, "\"vocab\": [") != NULL && strstr(buf, "\"unk_id\": 0") != NULL,
          "json has vocab ARRAY and unk_id");
    CHECK(strstr(buf, "[\"<unk>\", ") != NULL || strstr(buf, "[\"▁\", ") != NULL,
          "json vocab entries are [piece, score] pairs");
```

**验证方法**
1. 导出后 `python -c "import json,io; d=json.load(open('tokenizer.json')); print(type(d['model']['vocab']), len(d['model']['vocab']))"` 输出 `<class 'list'>`。
2. HuggingFace `tokenizers` 库直接加载 `tokenizer.json` 成功。

---

### P1-9 SP NormalizerSpec 与真实管线不符

**文件 / 位置**：`src/sentencepiece.c` `tokenizer_export_sentencepiece` NormalizerSpec（约 L236）

**问题摘要**：导出 NormalizerSpec `name="identity"`，而 `normalize.c` 实际执行完整 NFKC + 空格标记（▁）。

**修复前**

```c
        pb_str(buf, cap, &o, 1, "identity", 8);
```

**修复后**

```c
        /* P1-9: 与真实管线对齐——normalize.c 执行完整 NFKC（UAX #15）；
         * nmt_nfkc 是 SentencePiece 中最接近的实现（full-width 折叠、
         * 兼容分解等），并保留 escape_whitespaces=true（field 5 不变）。
         * 注意：控制符剥离 / 零宽字符剔除等细节在 NormalizerSpec 中
         * 无一一对应的字段，外部加载方行为仍可能存在细微偏差。 */
        pb_str(buf, cap, &o, 1, "nmt_nfkc", 8);
```

**验证方法**
1. 导出 `.model`，用 SentencePiece `SentencePieceProcessor` 加载并对同一文本执行 `Encode`，与本地 `tokenizer_encode` 输出对照（偏差仅限 spec 无法表达的剥离规则）。
2. `tokenizer.json` 侧 `"normalizer": {"type": "NFKC"}` 保持不变，两端口径一致。

---

## P2 中优先级

### P2-1 `seed.counts` 所有权转移后未 free

**文件 / 位置**：`src/trainer.c` `tokenizer_train` EM 入口（L787-790 转移块之后）

**问题摘要**：EM 接管 `seed.pieces/lens` 但不接管 `seed.counts`（`counts` 重新 `calloc`），成功路径不释放 `seed.counts`，泄漏。

**修复前**

```c
        vsz = seed.size;
        pieces = seed.pieces;
        plens = seed.lens;
        scores = (float *)malloc((size_t)vsz * sizeof(float));
        counts = (double *)calloc((size_t)vsz, sizeof(double));
        if (!scores || !counts) {
            seed_free(&seed);
            ...
        }
        /* initial scores from seed frequencies */
        double sum_freq = 0.0;
        for (int i = 4; i < vsz; i++) sum_freq += seed.counts[i];
        for (int i = 0; i < vsz; i++) {
            ...
                double p = sum_freq > 0 ? seed.counts[i] / sum_freq : 1e-9;
            ...
        }
```

**修复后**：在初始 score 计算完成后（`if (!resumed)` 块末尾）释放并置 NULL

```c
        /* P2-1: seed.counts 仅供初始 score 估计，EM 不接管（counts 为独立
         * calloc），释放防泄漏；置 NULL 使错误路径 seed_free() 安全。 */
        free(seed.counts);
        seed.counts = NULL;
```

**验证方法**：`train` 完成后 ASan/Valgrind 无 `seed.counts` 泄漏报告；分配失败路径（`!scores || !counts`）仍由 `seed_free` 正常释放（`free(NULL)` 安全）。

---

### P2-2 checkpoint 原生 endian（建议暂缓；可选补丁）

**文件 / 位置**：`src/trainer.c` `ckpt_save`（L413 起）与 `ckpt_load`；`src/model.c` 提供 `put_u32/get_u32` 可复用

**问题摘要**：`ckpt_save` 直接 `fwrite` 原生整数/浮点，非小端，跨机器加载失效。fix-plan 已将其列为“明确暂缓”；本项提供可选实现，是否落地由用户决定。

**修复后（可选）**：为 checkpoint 引入显式小端序列化（新增文件级 helper，`ckpt_save` / `ckpt_load` 逐字段替换）

```c
/* ---- P2-2 (optional): little-endian checkpoint I/O ---- */
static void ckpt_w32(FILE *fp, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    fwrite(b, 1, 4, fp);
}
static void ckpt_w64(FILE *fp, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    fwrite(b, 1, 8, fp);
}
static void ckpt_wf(FILE *fp, float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    ckpt_w32(fp, bits);
}
static void ckpt_wstr(FILE *fp, const char *s, size_t n) {
    ckpt_w64(fp, (uint64_t)n);
    fwrite(s, 1, n, fp);
}
```

```c
static uint32_t ckpt_r32(FILE *fp) {
    unsigned char b[4];
    if (fread(b, 1, 4, fp) != 4) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}
static uint64_t ckpt_r64(FILE *fp) {
    unsigned char b[8];
    if (fread(b, 1, 8, fp) != 8) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}
static float ckpt_rf(FILE *fp) {
    uint32_t bits = ckpt_r32(fp);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
```

用法示例（`ckpt_save` 头部段）：

```c
    ckpt_w32(fp, 0x54434b50u);           /* "TCKP" magic (was fwrite) */
    ckpt_w32(fp, 2u);                    /* checkpoint version bump */
    ckpt_w32(fp, (uint32_t)vsz);
    ckpt_w32(fp, (uint32_t)round_no);
    ckpt_w32(fp, (uint32_t)n_at_target);
    ckpt_wf(fp, (float)prev_ll);
```

piece 循环同理：`ckpt_wstr(fp, pieces[i], (size_t)plens[i])` + `ckpt_wf(fp, scores[i])`；`ckpt_load` 用 `ckpt_r*` 对称读取。若落地，请同步 `ckpt_load` 的版本号分支（读 v1 原生格式向后兼容）。

**验证方法（若实施）**：小语料训 1 轮存 checkpoint → 换机/模拟大小端加载 resume，ll 曲线连续；95 项测试通过。

---

### P2-3 `add_byte_fallback` 注释语义矛盾

**文件 / 位置**：`include/tokenizer.h`（`struct tokenizer_train_options` 内 `add_byte_fallback` 字段，约 L58）

**修复前**

```c
    int add_byte_fallback; /* add 256 <0xXX> pieces; 0 => on (1) */
```

**修复后**（统一为“1=开启（默认），0=关闭”，与实现一致）

```c
    int add_byte_fallback; /* add 256 <0xXX> pieces; 1 => on (default), 0 => off */
```

**验证方法**：文档级改动，编译通过即可；`--no-byte-fallback` 行为不变。

---

### P2-4 CLI decode `realloc` 不检查 NULL

**文件 / 位置**：`cli/main.c` `run_decode`（约 L305）

**修复前**

```c
        if (n >= cap) {
            cap *= 2;
            ids = (int *)realloc(ids, cap * sizeof(int));
        }
        ids[n++] = id;
```

**修复后**

```c
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
        ids[n++] = id;
```

**验证方法**：超长输入 decode 可正常扩容；低内存注入返回错误码而非崩溃。

---

### P2-5 protobuf `pos + plen` 溢出绕过边界检查

**文件 / 位置**
- `src/sentencepiece.c` `sp_parse_piece`（约 L300）
- `src/sentencepiece.c` `sp_parse_trainer`（约 L390）

**修复前（sp_parse_piece）**

```c
    uint64_t plen = 0;
    if (pb_parse_u64(buf, size, pos, &plen) != 0) return -1;
    size_t end = *pos + (size_t)plen;
    if (end > size) return -1;
```

**修复后**

```c
    uint64_t plen = 0;
    if (pb_parse_u64(buf, size, pos, &plen) != 0) return -1;
    /* P2-5: check BEFORE adding — plen near UINT64_MAX would wrap */
    if (plen > (uint64_t)(size - *pos)) return -1;
    size_t end = *pos + (size_t)plen;
```

**修复前（sp_parse_trainer）**

```c
    uint64_t slen = 0;
    if (pb_parse_u64(buf, size, pos, &slen) != 0) return;
    size_t end = *pos + (size_t)slen;
    if (end > size) return;
```

**修复后**

```c
    uint64_t slen = 0;
    if (pb_parse_u64(buf, size, pos, &slen) != 0) return;
    if (slen > (uint64_t)(size - *pos)) return; /* P2-5: check before adding */
    size_t end = *pos + (size_t)slen;
```

**验证方法**：构造 `plen`/`slen` 接近 `UINT64_MAX` 的畸形 `.model`，加载返回失败且不越界读写（ASan 无报错）；合法模型加载不受影响。

---

### P2-6 EM 中 `trie_insert` 失败被忽略

**文件 / 位置**：`src/trainer.c` EM 循环内 trie 重建（L841）

**修复前**

```c
        for (int i = 0; i < vsz; i++) {
            if (is_bytefb_style(pieces[i], plens[i])) continue;
            trie_insert(trie, pieces[i], (size_t)plens[i], i);
        }
```

**修复后**

```c
        for (int i = 0; i < vsz; i++) {
            if (is_bytefb_style(pieces[i], plens[i])) continue;
            if (trie_insert(trie, pieces[i], (size_t)plens[i], i) != 0) {
                /* P2-6: OOM mid-round — abort and let done: clean up */
                tokenizer_set_error("train: OOM rebuilding trie");
                goto done;
            }
        }
```

**验证方法**：低内存注入时训练明确失败（错误信息）而非静默缺词表继续；正常训练行为不变。

---

### P2-7 `nfkc` 懒建表非线程安全

**文件 / 位置**：`src/nfkc.c` `ensure_index` / `decomp_index`（约 L23-34）；`src/nfkc.h` 增加预热接口

**问题摘要**：`s_decomp_index` 懒初始化无锁，多线程首次并发调用存在数据竞争。

**修复后**：C11 `call_once` 初始化 + 显式预热入口（load/train 时主动建表）

```c
#include <threads.h>   /* C11; MSVC 无 threads.h 时用下方 nfkc_prewarm() 预热方案 */

static once_flag s_index_once = ONCE_FLAG_INIT;

static void build_index(void) {
    uint32_t *idx = (uint32_t *)malloc(0x30000 * sizeof(uint32_t));
    if (!idx) return; /* fallback: binary search still works */
    memset(idx, 0xFF, 0x30000 * sizeof(uint32_t));
    for (uint32_t i = 0; i < NFKC_DECOMP_N; i++) {
        uint32_t cp = kNfkcDecompEntries[i].cp;
        if (cp < 0x30000) idx[cp] = i;
    }
    s_decomp_index = idx;
}

static void ensure_index(void) {
    call_once(&s_index_once, build_index);
}

/* Explicit prewarm for callers that can do it on a single-threaded entry
 * point (load / train), eliminating the lazy race entirely. */
void nfkc_prewarm(void) {
    ensure_index();
}
```

`src/nfkc.h` 增加声明 `void nfkc_prewarm(void);`。入口预热：`src/model.c` `tokenizer_load_binary_file`、`src/sentencepiece.c` `tokenizer_load_sentencepiece`、`src/trainer.c` `tokenizer_train` 开头各加一行 `nfkc_prewarm();`。

MSVC 备注：无 `<threads.h>`，可仅依赖 `nfkc_prewarm()` 预热（load/train 均为单线程入口，预热后 `decomp_index` 不再懒建）；或改用 `InitOnceExecuteOnce`。gcc/clang 直接用 `call_once`。

**验证方法**：多线程并发首调 `tokenizer_normalize`（如 8 线程同时 normalize 不同文本）无数据竞争（TSan 无报错）；`decomp_index` 结果一致。

---

### P2-8 公开头 `tokenizer_normalize` 注释过时

**文件 / 位置**：`include/tokenizer.h`（约 L81）

**修复前**

```c
    /* Basic normalization subset (NFKC-like): ... */
```

**修复后**

```c
    /* Full NFKC normalization (Unicode UAX #15):
     *   - strip C0/C1 control chars (keeps \n \r \t)
     *   - map the Unicode whitespace family to ASCII space
     *   - full NFKC decomposition / canonical ordering / recomposition
     *     (full-width, circled digits, ligatures, roman numerals,
     *      super/subscripts, compat currency signs, ...)
     *   - strip zero-width chars and BOM
     *   - optional ASCII case folding
     *   - mark plain spaces with U+2581 (▁) when mark_spaces=1
     * Returns a malloc'ed UTF-8 string (free with tokenizer_free_text). */
```

**验证方法**：文档级改动，编译通过即可；`normalize.c` 行为不变。

---

## 额外发现

### 额外发现 1：训练端 `viterbi_segment` 修补分支未写 best/prev

已并入 **P1-2** 修复：训练端 DP 循环改为“每位置无条件 byte 边且同步写 `best[i+1]` / `prev[i+1]`”，修补分支不再存在，推理端与训练端逻辑统一。无需单独补丁。

### 额外发现 2：`nfkc_normalize_cps` OOM 静默降级 identity

**文件 / 位置**：`src/normalize.c` `normalize_text` Pass 2（约 L140）

**修复前**

```c
    uint32_t *norm = NULL;
    size_t nn = 0;
    nfkc_normalize_cps(cps, nc, &norm, &nn);
    uint32_t *work = norm ? norm : cps; /* OOM fallback: identity */
    size_t wn = norm ? nn : nc;
```

**修复后**（NFKC 表构建/展开 OOM 时明确失败，与“截断即失败”口径一致）

```c
    uint32_t *norm = NULL;
    size_t nn = 0;
    nfkc_normalize_cps(cps, nc, &norm, &nn);
    if (!norm) {
        /* 额外发现 2: OOM during NFKC — fail loudly instead of silently
         * degrading to identity (consistent with "truncation == failure"). */
        free(cps);
        return NULL;
    }
    uint32_t *work = norm;
    size_t wn = nn;
```

**验证方法**：低内存注入下 `normalize_text` 返回 NULL（调用方按“normalization failed”处理），不再输出未归一化的身份结果；正常输入行为不变。

---

## 汇总说明

- 覆盖条目：21 条（P0×4 + P1×9 + P2×8）+ 2 条额外发现，全部给出可直接替换的代码；P2-2 为“建议暂缓 + 可选补丁”，属可落地实现。
- 接口变更汇总：`tokenizer_internal.h` 新增 `file_size64`（static）、`tokenizer_clear_error()` 声明；`struct tokenizer` 新增 `fold_case` 字段（P1-3）；`src/nfkc.h` 新增 `nfkc_prewarm()`（P2-7）；`sp_piece_list` 新增 `lens` 字段（P1-7）；`tests/test_tokenizer.c` JSON 断言升级（P1-8）。
- 无新增第三方依赖；C11 特性 `threads.h` 仅 P2-7 使用且提供 MSVC 替代方案。
- 回归顺序建议：先落地 A 批次（P0-1/2/3/4）与 B 批次（P1-1/2），跑 95 项测试 + 小语料训练；再落地 C/D 批次并逐项做验证方法中的互操作检查。

*（内容由AI生成，仅供参考）*
