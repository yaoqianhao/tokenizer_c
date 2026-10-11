# tokenizer-c 代码审阅与修改建议

- 日期：2026-10-11
- 范围：`E:\tokenizer_c` 当前主干（训练 / 推理 / 序列化 / CLI / 测试）
- 状态：审阅完成，待按批次落地
- 目的：把「能跑通小语料」推进到「3GB / 256K 可训练、契约与生态对齐」

本文是实现侧的问题清单与改法，不替代 `2026-10-09-tokenizer-design.md`。设计目标仍以设计文档为准；下文指出的是实现与设计/README 之间的差距。

---

## 1. 结论摘要

工程骨架可用：模块边界清楚，C API、TKCZ、checkpoint、NFKC、单元测试（95 项）已形成闭环。

作为「工业级」实现，当前最大缺口不在 Trie/NFKC 细节，而在：

1. **Windows 上大文件 I/O 不可用**（`long` + `ftell`，语料 >2GB 会失败）。
2. **训练算法近似过度**（种子先到先得、Viterbi 非每位置 byte 边、剪枝插入排序）。
3. **导出格式与官方契约不一致**（HF `vocab` 用 object、SP normalizer 写成 identity）。
4. **若干 API 失败语义不干净**（last error 粘滞、realloc 半失败、decode 截断仍返回 0）。

这与评测中 overall BPT 落后 SentencePiece、以及 `train_big.bat` 预估 110~140 小时，是同一组根因。

---

## 2. 现状与设计差距

| 设计/README 承诺 | 当前实现 | 影响 |
|---|---|---|
| 3.28GB 海量语料训练 | `corpus_load` 整文件读入，`ftell` 返回 32-bit `long` | Windows 上 >2GB 语料大概率直接失败 |
| 种子 n-gram 按频次截断到 `max_candidates` | 哈希满后不再插入新 key，再对已有 key 排序 | 大语料只统计文件前部，词表质量偏斜 |
| Unigram Viterbi 全局最优 | 仅当本位置「完全无匹配」才走 1-byte fallback；`MAX_MATCH=64` | 路径不保证最优，存在空洞再修补 |
| `--fold-case` 训练/推理一致 | TKCZ 只存 `byte_fallback`；encode 写死 `fold_case=0` | 折叠训练的模型推理不折叠 |
| HF `tokenizer.json` 可被 tokenizers 加载 | `vocab` 导出为 JSON object | 丢失 piece 顺序，官方 Unigram 要的是数组 |
| SP `.model` 行为对齐 | NormalizerSpec 为 `identity`，实际 encode 做 NFKC + `▁` | 官方 SentencePieceProcessor 加载后行为不一致 |
| load 后并发 encode/decode | `nfkc.c` 懒建 `s_decomp_index` 无同步 | 首次 encode 存在数据竞争 |
| 行长度 4096 为已知限制 | 静默截断，可切在 UTF-8 中间 | 长文档/代码块被破坏，且无统计 |

---

## 3. 问题清单（按严重度）

### 3.1 严重：大语料训练可能根本跑不起来

**P0-1 32-bit `ftell`**

位置：`src/trainer.c` `corpus_load`，以及 `src/model.c`、`src/sentencepiece.c`、`cli/main.c`。

Windows LLP64 下 `long` 为 32 位。`fseek`/`ftell` 对 >2GB 文件返回负数或截断，随后被当成空文件拒绝。`train_big.bat` 指向的 `corpus_big/corpus_big.txt`（约 3.28GB）命中该路径。

**P0-2 整文件加载 + 按行再拷贝**

`corpus_load` 先 `malloc(文件大小)`，再为每一行再 `malloc`。峰值约 **2× 语料体积**，3GB 语料在 16GB 机器上极易 OOM。

**P0-3 逐行 verbose**

`trainer.c` seed 循环：`fprintf(stderr, "[train] seed line %zu ...")`。`train_big.bat` 固定 `-v`，海量行会把日志 I/O 打满。

**P0-4 剪枝插入排序**

`trainer.c` 对 normal bucket 做 O(n²) insertion sort。百万级种子一轮可达 \(10^{11}\)～\(10^{12}\) 量级比较，与「每轮 EM 10~14 小时」量级相符。

### 3.2 高：正确性与契约

**P1-1 种子不是 Top-K**

`strhash_t.max_size` 满后 `full=1`，新 n-gram 丢弃。最终 `qsort` 只在「先出现的 100 万条」里选。SentencePiece 是全量计数再截断。

**P1-2 Viterbi 缺默认 byte 边**

`tokenizer_encode` / `viterbi_segment`：

```c
if (dp[i + 1] == -INFINITY && !any_updated) { /* fallback */ }
```

标准 Unigram：每个位置始终保留 1-byte（或 unk）边。当前若只匹配到长度 ≥2 的 piece，`i+1` 可能空洞，再靠 `-INFINITY` 修补，最优性不保证。`MAX_MATCH = 64` 会丢掉更长前缀。

**P1-3 `fold_case` 不入模型**

TKCZ flags 仅 `bit0 = byte_fallback`。推理 `normalize_for_encode(..., fold_case=0)`。

**P1-4 last error 粘滞**

`tls_error` 成功路径不清空。测试代码已注明该问题。调用方不能用「有没有 last error」判断本次调用。

**P1-5 `realloc` 半失败**

`tokenizer_ensure_piece_slots`、`seed_push`、`sp_list_push`：第一块 realloc 成功并写回指针，后续失败时 `cap`/`vocab_size` 未同步，`tokenizer_free` 可能漏释放或读野指针。

**P1-6 截断当成功**

`tokenizer_decode`、`normalize_text` 在 `o + 1 >= cap` 时 `break` 仍返回成功，输出变短且调用方不知情。

**P1-7 SP 加载用 `strlen` 当 piece 长度**

`tokenizer_load_sentencepiece` 调用 `tokenizer_set_piece(..., strlen(pieces.pieces[i]), ...)`。含 NUL 的 byte piece 会被截断。`sp_list_push` 已保存真实长度，load 循环未用。

**P1-8 HF JSON 形状错误**

官方 Unigram：

```json
"vocab": [["<unk>", -100.0], ["hello", -1.23]]
```

当前：

```json
"vocab": { "hello": -1.23 }
```

object 丢失顺序；测试只 `strstr("vocab")`，测不出互操作。

**P1-9 SP NormalizerSpec 与真实管线不符**

导出 `name = "identity"`，实际 `normalize.c` 走完整 NFKC + 空格标记。

### 3.3 中：内存、并发、健壮性

| ID | 问题 | 位置 |
|---|---|---|
| P2-1 | `seed.counts` 所有权转移后未 free | `tokenizer_train` EM 起始 |
| P2-2 | checkpoint 原生 endian，不可跨机器 | `ckpt_save` / `ckpt_load` |
| P2-3 | 注释写 `add_byte_fallback` 0 表示开启，实现把 0 当关闭 | `tokenizer.h` vs `tokenizer_train` |
| P2-4 | CLI decode `realloc` 不检查 NULL | `cli/main.c` |
| P2-5 | protobuf `pos + plen` 可能溢出后绕过边界检查 | `sentencepiece.c` |
| P2-6 | EM 中 `trie_insert` 失败被忽略 | `trainer.c` 重建 trie |
| P2-7 | `nfkc` 懒建表非线程安全 | `src/nfkc.c` `ensure_index` |
| P2-8 | 公开头 `tokenizer_normalize` 仍写「basic NFKC subset」 | `include/tokenizer.h` |

---

## 4. 修改建议（按落地顺序）

### 批次 A — 让 3GB 训练先能跑完（优先做）

目标：Windows 可读 >2GB 语料；内存峰值可控；每轮 EM 不再被排序拖死。

#### A1. 64 位文件偏移

统一封装，替换所有 `fseek`/`ftell` 测文件大小的路径：

```c
#ifdef _WIN32
#define FILE_SEEK64(fp, off, wh) _fseeki64((fp), (off), (wh))
#define FILE_TELL64(fp)          _ftelli64(fp)
#else
#define FILE_SEEK64(fp, off, wh) fseeko((fp), (off), (wh))
#define FILE_TELL64(fp)          (int64_t)ftello(fp)
#endif
```

涉及：`corpus_load`、`tokenizer_load_binary_file`、`tokenizer_load_sentencepiece`、CLI `--file`。

#### A2. 语料流式读行

`corpus_load` 改为按 `\n` 缓冲读取（8–64KB），不要整文件映射。

两种可接受形态：

- **训练中两遍扫描**：seed 一遍、每轮 EM 再扫文件（内存最小，实现与现循环最接近：把 `corpus.lines[i]` 换成「再读一遍」）。
- **只缓存归一化后的行**：仍占内存，但去掉「原始 + 拷贝」双份，且避开每次 EM 重复 NFKC。

3GB 场景推荐第一种，或「归一化行落临时文件再扫」。

行过长：按 UTF-8 边界截断，累计 `truncated_lines`，verbose 只打汇总。

#### A3. 进度日志

删除 seed 逐行 `fprintf`。`-v` 改为每 N 秒或每 10 万行：

```
[train] seed  lines=1200000  ngrams=850000  elapsed=32s
[train] round 3  vocab=180000  ll=...  elapsed=...
```

`train_big.bat` 可继续带 `-v`。

#### A4. 剪枝改 `qsort`

将 insertion sort 换为：

```c
static int pair_cmp(const void *a, const void *b) {
    const pair_t *x = (const pair_t *)a;
    const pair_t *y = (const pair_t *)b;
    if (x->score < y->score) return 1;
    if (x->score > y->score) return -1;
    return (x->id > y->id) - (x->id < y->id);
}
qsort(pairs, (size_t)bn[1], sizeof(pair_t), pair_cmp);
```

验收：

- 现有 `test_tokenizer` 仍全部通过。
- 人为构造 >2GB 稀疏语料（或 `fsutil file createnew` 后再写几行），`corpus_load` 不再报 empty。
- 同配置下一轮 EM 墙钟应明显下降（排序项变为可忽略）。

---

### 批次 B — 词表与切分质量

目标：缩小与 SentencePiece 的 BPT 差距。

#### B1. 种子改为全量计数再 Top-K

不要在插入期用 `max_size` 拒绝新 key。

推荐：

1. 哈希 cap 至少 `max_candidates * 4`～`* 8`，或取消硬 cap、仅在 collect 后截断。
2. `hash_collect` → 按 `count` 降序 `qsort` → 保留 `max_candidates`。
3. 内存不够时：第一遍精确 1-gram/2-gram，更长窗口仅当所有真前缀已入选再扩展。

#### B2. 每位置无条件 1-byte 边

`tokenizer_encode` 与 `viterbi_segment` 使用同一规则：

```c
double fb = dp[i] + kByteFallbackScore;
if (fb > dp[i + 1]) {
    dp[i + 1] = fb;
    best[i + 1] = /* 负虚拟 id（推理）或 vocab_size + byte（训练） */;
    prev[i + 1] = i;
}
```

`MAX_MATCH` 改为不小于 `max_piece_len`（字节），或动态增长。训练/推理必须同一套 DP。

#### B3. 行长度

`TRAIN_MAX_LINE` 默认提到至少 64KiB（或 1MiB 可配置）。截断必须在码点边界，并计入日志。

验收：在 `benchmark/corpus_64mb.txt` 上用相同 `vocab_size=262144` 重训，对比 `eval_metrics.py` 的 overall BPT（应对齐或优于当前 7.49，向 SP 的 5.87 靠近）。

---

### 批次 C — API / 模型契约

#### C1. `fold_case` 写入 TKCZ

- `flags bit1 = fold_case`
- `struct tokenizer` 增加字段
- save/load 读写；`normalize_for_encode` 使用该字段
- 版本：保持 v1 且缺省 0，或升 v2 并兼容 v1

#### C2. 成功路径 `clear_error`

```c
static void clear_error(void) { tls_error[0] = '\0'; }
```

encode/decode/load/train 成功返回前调用。

#### C3. realloc 先齐后换

三块都分配成功再替换旧指针；失败时旧对象仍可 `tokenizer_free`。同类改 `seed_push`、`sp_list_push`、corpus 扩容。

#### C4. 输出截断视为失败

decode/normalize 缓冲不够则扩容或返回错误，禁止 `break` 后 `return 0`。

验收：加 `--fold-case` 往返用例；故意越界 decode 后下一次成功 encode 的 `tokenizer_last_error()` 为 NULL。

---

### 批次 D — 生态导出与并发

#### D1. HF JSON `vocab` 改为数组

```c
fprintf(fp, "    \"vocab\": [\n");
/* ["piece", score], ... */
```

测试断言出现 `"vocab": [`，不要只搜 `"vocab"`。

#### D2. SP 导出/加载

- NormalizerSpec 与真实管线对齐：`nfkc` / `nmt_nfkc`，保留 `escape_whitespaces`
- `tokenizer_set_piece` 使用已解析的 `plen`，禁止 `strlen`
- 子消息长度：`if (plen > size - *pos) return -1`，避免 `pos + plen` 溢出

#### D3. NFKC 索引 `call_once`

```c
static once_flag nfkc_once = ONCE_FLAG_INIT;
/* C11: call_once(&nfkc_once, ensure_index); */
```

或在 `tokenizer_load` / `tokenizer_build_trie` 时主动建表。

#### D4. 其它清理

- EM 接管后 `free(seed.counts)`，并把 `seed.pieces/lens` 置 NULL，避免 `seed_free` 双释放
- `trie_insert` 失败则中止该轮并 `goto done`
- 统一 `add_byte_fallback`：0 表示默认开启，或删除「0 => on」注释，只保留 DEFAULT 宏
- 更新 `tokenizer.h` 中 normalize 注释为完整 NFKC

验收：Python 侧用 `tokenizers` 加载 json、用 `sentencepiece` 加载 `.model` 能打开；含 `<0x00>` 的模型 round-trip 长度不变。

---

## 5. 明确暂缓

以下不是当前瓶颈，不要和控制路径搅在一起：

- 训练多线程 / OpenMP（先做 A/B，单线程也会快一个数量级）
- 自研 protobuf 大重构
- CMake 对 MSVC 的 `/W4` 适配（现状是 MinGW/gcc）
- Trie 换哈希子节点（当前数组 + 二分足够）
- checkpoint 改小端跨平台（可随 C1 顺手，但不是 P0）

---

## 6. 建议的提交切分

| 提交 | 内容 | 风险 |
|---|---|---|
| 1 | A1+A2+A3：64-bit I/O、流式语料、进度日志 | 中（训练入口行为变） |
| 2 | A4：剪枝 qsort | 低 |
| 3 | B1：种子 Top-K | 中（词表分布变，需重训才可比 BPT） |
| 4 | B2+B3：Viterbi byte 边、行上限 | 中（encode 路径变，必须训推一致） |
| 5 | C 整批：flag / error / realloc / decode | 低～中 |
| 6 | D 整批：JSON / SP / once / 泄漏 | 低 |

每提交后跑 `build/test_tokenizer`（或 `ctest --test-dir build`）。涉及 encode 语义的提交（4）后，用现有 `t_model` 往返不够，应用同一小语料重新 `tokenizer_train` 再测。

---

## 7. 若只改三处

时间不够时，只做下面三项，3GB 脚本才有意义：

1. `_fseeki64` / `_ftelli64` + 流式读行  
2. 去掉逐行 verbose  
3. 剪枝 `qsort`

算法质量与 HF/SP 兼容放到第二轮。

---

## 8. 回归清单

- [ ] `ctest --test-dir build`：原 95 项仍通过（语义变更提交允许更新期望，但须说明）
- [ ] Windows：对 >2GB 文件 `tok-cli train` 不再报 `empty corpus`
- [ ] `-v` 日志行数与语料行数脱钩（seed 阶段不应每行一条）
- [ ] `--fold-case` 存盘后再 encode，大小写行为与训练一致
- [ ] `tokenizer_decode` 非法 id 失败后，下一次 encode 成功则 `tokenizer_last_error()` 为空
- [ ] 导出 json 可被 HuggingFace `tokenizers` Unigram 加载
- [ ] 导出 `.model` 可被 `SentencePieceProcessor` 加载；piece 含 NUL 时长度不丢
- [ ] 可选：`benchmark/eval_metrics.py` 同口径对比 SP 基线 BPT

---

## 9. 相关文件

| 文件 | 角色 |
|---|---|
| `src/trainer.c` | P0 训练 I/O、种子、EM、剪枝、checkpoint |
| `src/tokenizer.c` | encode/decode、error、piece 槽 |
| `src/model.c` | TKCZ 读写 |
| `src/sentencepiece.c` | `.model` / `tokenizer.json` |
| `src/nfkc.c` | 并发建表 |
| `src/normalize.c` | 截断、NFKC 调用 |
| `cli/main.c` | 大文件 encode、decode realloc |
| `include/tokenizer.h` | 契约注释、flags 语义 |
| `tests/test_tokenizer.c` | 增补 fold_case / JSON 数组 / error 清空 |
| `train_big.bat` | 依赖 A 批次完成后才值得再跑 |

---

## 10. 文档关系

- `2026-10-09-tokenizer-design.md`：需求与目标架构（不变）
- 本文：相对当前代码的差距与改法
- `README.md`：落地后应同步更新「已知限制」「P1 BPT 门槛」「训练 I/O 说明」
- `benchmark/*.md`：批次 B 重训后再刷新数字，不要在只改 I/O 时改评测结论
