---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_7622a760c3dd11f1a08152540064ee0f
    ReservedCode1: PINr7X+nAJuldkAXwVzzZmHUthse0LsVS+poahgZZ3heSg3jI3vZDu3na9L8d8vD/urpqFoYZ1flBIJKmHFiefLC/TnoMpMUsAl3M/YjfUs024qFjo9qMwv/xV8k7ooSocTEPyfntF4SZPcDcCpsv7UG61sb9VJXdPI1eV0hsKdapxyGtAljpOrOnRk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_7622a760c3dd11f1a08152540064ee0f
    ReservedCode2: PINr7X+nAJuldkAXwVzzZmHUthse0LsVS+poahgZZ3heSg3jI3vZDu3na9L8d8vD/urpqFoYZ1flBIJKmHFiefLC/TnoMpMUsAl3M/YjfUs024qFjo9qMwv/xV8k7ooSocTEPyfntF4SZPcDcCpsv7UG61sb9VJXdPI1eV0hsKdapxyGtAljpOrOnRk=
---

# tokenizer-c — 工业级 Unigram Tokenizer（纯 C 实现）

基于 Google Gemini 的算法框架（SentencePiece Unigram），用纯 C 语言实现的可编译、可运行、可训练的工业级 tokenizer 工程。

---

## 1. 功能总览

| 模块 | 内容 |
|---|---|
| **Unigram 训练器** | 种子词表（字符 n-gram 频次统计）、EM 迭代（Viterbi 最优切分 / 概率重估 / 词表剪枝收敛） |
| **推理引擎** | `encode`（Viterbi DP + Trie 前缀加速 + UTF-8 字节回退兜底）、`decode`（`▁` 空格还原、`<0xXX>` 字节展开） |
| **特殊 token** | `<unk>`/`<s>`(bos)/`</s>`(eos)/`<pad>` 固定占据 id 0/1/2/3 |
| **归一化** | 完整 NFKC（UAX #15：规范/兼容分解、组合排序、重组）+ 可扩展钩子 |
| **序列化** | 自定义二进制模型格式 `TKCZ` + JSON 导出 + **SentencePiece `.model`（protobuf）导出/加载** |
| **C API** | `tokenizer.h`：load / save / encode / decode / normalize / train / export_sentencepiece / load_sentencepiece / free 等 |
| **示例 CLI** | `train` 与 `encode`/`decode` 两个子命令 |
| **单元测试** | `tests/test_tokenizer.c` 95 项检查，覆盖训练、往返、字节回退、内存加载、越界拒绝、SP 往返、NFKC 映射等 |

## 2. 目录结构

```
tokenizer-c/
├── CMakeLists.txt          # CMake 构建（默认方式）
├── README.md
├── include/
│   └── tokenizer.h         # 公开 C API
├── src/
│   ├── tokenizer_internal.h# 内部结构 & 内部声明
│   ├── utf8.c / utf8.h     # UTF-8 解码/编码（严格校验）
│   ├── trie.c / trie.h     # 前缀 Trie（encode 加速）
│   ├── normalize.c / normalize.h  # 基础归一化 + ▁ 标记
│   ├── nfkc.c / nfkc.h / nfkc_tables.inc  # 完整 NFKC（UAX #15），映射表驱动
│   ├── tokenizer.c         # 生命周期 / encode(Viterbi DP) / decode / 字节回退
│   ├── model.c             # TKCZ 二进制读写 + JSON 导出
│   ├── sentencepiece.c     # SentencePiece ModelProto（.model）序列化/解析
│   └── trainer.c           # Unigram 训练（种子词表 / EM / 剪枝收敛）
├── cli/
│   └── main.c              # tok-cli：train / encode / decode
├── tests/
│   └── test_tokenizer.c    # 单元与集成测试
└── scripts/
    └── build.ps1           # Windows 构建脚本（CMake 或纯 gcc 两种方式）
```

## 3. 构建

### 方式一：CMake（推荐）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build          # 运行单元测试
```

### 方式二：纯 gcc

```bash
mkdir -p build && cd build
gcc -std=c11 -O2 -Wall -Wextra -I../include -I../src -c ../src/*.c
ar rcs libtokenizer.a *.o
gcc -std=c11 -O2 -I../include ../cli/main.c libtokenizer.a -lm -o tok-cli
gcc -std=c11 -O2 -I../include ../tests/test_tokenizer.c libtokenizer.a -lm -o test_tokenizer
./test_tokenizer
```

### 方式三：Windows PowerShell 脚本

```powershell
.\scripts\build.ps1        # CMake
.\scripts\build.ps1 -Gcc   # 纯 gcc
```

**产物**：`build/tok-cli.exe`、`build/test_tokenizer.exe`、`build/libtokenizer.a`。

## 4. 快速上手

### 训练

```bash
# 语料：每行一段文本（UTF-8）
tok-cli train --corpus corpus.txt --model model.tkm \
    --vocab-size 4096 --json model.json -v
```

### 编码 / 解码

```bash
tok-cli encode --model model.tkm --text "hello world"
tok-cli encode --model model.tkm --file input.txt
tok-cli decode --model model.tkm --ids "1 2 3 4 5"
```

输出 id 为空格分隔的十进制序列；`decode` 输出还原后的 UTF-8 文本。

## 5. 算法细节

### 5.1 训练流程（trainer.c）

1. **种子词表**：对归一化后语料统计字节级字符 n-gram 频次（`max_piece_len` 默认 8），按频次降序截断至 `max_candidates`（默认 100 万）；特殊 token 固定置于头部。
2. **EM 迭代**（`--iters` 默认 10 轮）：
   - **E 步**：对每条序列做 **Viterbi 最优切分**（基于当前词表与 score），累计各 piece 使用频次。
   - **M 步**：`P(t) = count(t) / Σcount`，`score(t) = log P(t)`。
   - **剪枝**：每轮保留特殊 token + 高频 top-k（`k = max(目标词表, 当前 × shrink%)`，`--shrink` 默认 60），低分 piece 逐轮淘汰，向 `--vocab-size` 收敛。
3. **收敛**：词表达到目标规模且 log-likelihood 增益连续 `--min-rounds`（默认 3）轮低于 `--eps`（默认 1e-4），或达到最大轮数。

### 5.2 编码（tokenizer.c）

1. **归一化**：完整 NFKC（见 §7）+ 大小写折叠（可选）+ `▁` 空格标记（每个空格替换为 U+2581，字符串首空格保留标记）。
2. **Viterbi DP**：`dp[i] = max_j (dp[i-len] + score(piece))`，Trie 在当前位置收集所有可匹配前缀，复杂度近似 `O(n × max_piece_len)`。
3. **字节回退兜底**：字符序列无匹配 piece 时逐字节 `<0xXX>`（如 `<0xE8>`），保证任意 UTF-8 输入均可编码。

### 5.3 解码

- `▁` → 空格还原（连续 `▁` 保留为多个空格，行首标记不产生空格）；
- `<0xXX>` → 单字节展开；两者结合实现**无损往返**：`decode(encode(x)) == normalize(x)`。

## 6. 自定义二进制模型格式（TKCZ）

| 偏移 | 内容 |
|---|---|
| 0 | 魔数 `TKCZ`（4 字节） |
| 4 | 版本 `u32`（当前 1） |
| 8 | `vocab_size`（u32） |
| 12 | 特殊 id 4×u32：unk / bos / eos / pad |
| 16 | 标志 u32：bit0 = 支持字节回退；bit1 = fold_case 归一化 |
| 20 | piece 表：每项 = `len`(u16) + `score`(f32) + 原始字节串 |
| 尾部 | 校验和 u32（FNV-1a，覆盖前序全部字节） |

另支持 `tokenizer_export_json()` 输出可读 JSON（含 `type: "Unigram"`、配置、piece 表）。

### 6.1 SentencePiece 兼容（.model）

`src/sentencepiece.c` 实现 SentencePiece ModelProto（protobuf wire format）的序列化与解析，打通与主流 LLM 生态互操作：

| 能力 | 入口 | 说明 |
|---|---|---|
| 导出 `.model` | `tokenizer_export_sentencepiece(tok, path)` | 将当前模型写出为 SentencePiece Unigram `.model`（piece 表 + score + special token 映射），可由 transformers / vLLM 的 `SentencePieceProcessor` 直接加载 |
| 加载 `.model` | `tokenizer_load_sentencepiece(path)` | 读取任意 SentencePiece `.model` 为内部模型，encode/decode 行为与原生一致 |
| JSON 导出 | `tokenizer_export_json` | HuggingFace 风格 `tokenizer.json`（`type: "Unigram"`、`normalizer`、`pre_tokenizer: null`、`byte_fallback`、`special_tokens` 字段齐全） |

## 7. 归一化实现说明（完整 NFKC）

本实现已补齐**完整 NFKC**（Unicode UAX #15），不再只是基础子集：

| 能力 | 覆盖 | 说明 |
|---|---|---|
| 规范分解（NFD） | 组合字符 → 基字符 + 组合符号 | `nfkc_decompose_cp` 查表分解 |
| 兼容分解 | 全角/半角、带圈数字 `①→1`、连字 `ﬃ→ffi`、罗马数字 `Ⅳ→IV`、上下标、兼容货币符号等 | 表驱动，覆盖 1000+ 映射 |
| 组合排序 | 按规范组合类（CCC）重排组合符号 | `nfkc_ccc` |
| 重组（NFC） | 规范组合回写，尊重 UAX #15 组合排除 | `nfkc_compose` |
| 大小写折叠（可选） | ASCII `A-Z` → `a-z` | 开关控制 |
| U+2581 标记 | 空格 → `▁`（编码侧） | 保证无损往返 |

实现文件：`src/nfkc.c`（分解/组合核心）+ `src/nfkc_tables.inc`（静态映射表，由 `scripts/gen_nfkc_tables.py` 从 Unicode 数据生成）；`src/normalize.c` 负责调用 nfkc 核心并做 `▁` 标记。

**扩展钩子**：`tokenizer_set_normalizer_hook()`（`tokenizer_normalizer_hook_fn`）仍保留，可在完整 NFKC 之上叠加领域定制归一化，无需改动编码/解码主流程。

## 8. C API 摘要

```c
tokenizer_t *tokenizer_load(const char *path);
tokenizer_t *tokenizer_load_from_memory(const unsigned char *buf, size_t size);
int   tokenizer_encode(tokenizer_t*, const char *text, size_t len,
                       int **out_ids, size_t *out_n);
int   tokenizer_decode(tokenizer_t*, const int *ids, size_t n,
                       char **out_text, size_t *out_len);
int   tokenizer_piece_to_id(tokenizer_t*, const char *piece, size_t len);
int   tokenizer_id_to_piece(tokenizer_t*, int id,
                            const char **out, size_t *out_len);
int   tokenizer_normalize(const char *text, size_t len,
                          char **out, size_t *out_len);   // 含 ▁ 标记
int   tokenizer_train(const tokenizer_train_options*);    // 完整训练
int   tokenizer_save_binary(tokenizer_t*, const char *path);
int   tokenizer_export_json(tokenizer_t*, const char *path);
int   tokenizer_export_sentencepiece(tokenizer_t*, const char *path); // 导出 SP .model
tokenizer_t *tokenizer_load_sentencepiece(const char *path);          // 加载 SP .model
void  tokenizer_set_normalizer_hook(tokenizer_t*,
                                    tokenizer_normalizer_hook_fn);    // 定制归一化钩子
const char *tokenizer_last_error(void);
void  tokenizer_free_ids(int *ids);
void  tokenizer_free_text(char *text);
void  tokenizer_free(tokenizer_t*);
```

## 9. 已知限制

- ~~词表上限 65,536~~（已解除）：trie 叶节点 `piece_id` 与 `vocab_size` 已升级为 32-bit int，`--vocab-size` 不再受 65536 截断；256K 压测（见 §10、§13）已验证通过。
- 训练为单线程、纯 CPU；语料行长度上限 4096 字节。
- 归一化已为完整 NFKC（见 §7）；仍可通过 `tokenizer_set_normalizer_hook()` 叠加领域定制。
- Windows 下 CLI 通过 `GetCommandLineW` + `WideCharToMultiByte` 将命令行参数统一转为 UTF-8，可直接传中文文本；控制台输出编码遵循系统代码页，若显示乱码请将终端切至 UTF-8（`chcp 65001`）或使用 `--file` 从 UTF-8 文件读取输入。
- `decode` 的还原语义为「`▁` → 空格」：归一化阶段把 `\n`、`\t` 等空白统一改写为空格标记，故含换行的文本往返后换行会变成空格（与 SentencePiece 行为一致）。

## 10. 迭代记录

| 迭代 | 状态 | 内容 |
|---|---|---|
| v1.0（初始） | ✅ | 纯 C 工程骨架、Unigram 训练器、推理引擎、TKCZ/JSON、C API、CLI、50 项测试 |
| v1.1 方向1：SentencePiece proto 兼容 | ✅ | `src/sentencepiece.c`：`.model` 导出/加载 + HF 风格 `tokenizer.json`；实测 95/95 检查通过 |
| v1.1 方向2：完整 NFKC | ✅ | `src/nfkc.c` + `src/nfkc_tables.inc`（UAX #15 分解/排序/重组）；实测 95/95 检查通过 |
| v1.1 方向3：256K 大规模压测 | ✅ | 256K 词表（vocab=262144）多语言压测完成：10 轮 EM 收敛（约 2.75h）；Trie 叶节点 id 已升级 32-bit（见 §12.7）；评测报告见 `benchmark/eval_report_2026-10-10.{json,md}`（§13） |

## 11. 迭代待办（方向3前置，已全部完成）

- ✅ 扩展 Trie 叶节点 id 存储宽度（16-bit → 32-bit），解除 65,536 词表上限（见 §12.7）；
- ✅ 生成多语言（中/英/代码）大规模语料并按配比采样（`benchmark/corpus/`：en 15MB / zh 10MB / code 10MB）；
- ✅ 测量并输出：编码/解码吞吐、还原率、稳定性、TokEval 维度等评测指标（见 §13 与 `benchmark/eval_report_2026-10-10.md`）。

*（内容由AI生成，仅供参考）*

---

## 12. 使用说明（User Guide）

本章以"使用者"视角整理完整的上手路径；与上文各节互为补充，不重复算法推导。

### 12.1 项目简介

tokenizer-c 是一个纯 C（C11）实现的工业级 Unigram tokenizer 工程，基于 SentencePiece 的 Unigram 算法框架，具备「训练 — 推理 — 序列化 — 生态互操作」完整闭环：

- **训练**：字符 n-gram 种子词表 + EM 迭代（Viterbi 最优切分 / 概率重估 / 词表剪枝收敛）；
- **推理**：`encode`（Viterbi DP + Trie 前缀加速 + UTF-8 字节回退兜底）、`decode`（`▁` 空格还原、`<0xXX>` 字节展开）；
- **归一化**：完整 NFKC（Unicode UAX #15，分解 / 组合排序 / 重组）+ 可扩展钩子；
- **序列化**：自有二进制模型格式 TKCZ、JSON 导出、SentencePiece `.model`（protobuf）导出 / 加载；
- **对外形态**：C API（`include/tokenizer.h`）与命令行程序 `tok-cli`；
- **质量保障**：`tests/test_tokenizer.c` 95 项检查（当前全部通过）。

### 12.2 目录结构

```
tokenizer-c/
├── CMakeLists.txt                  # [核心] CMake 构建定义（默认构建方式）
├── README.md                       # [核心] 项目文档（本文档）
├── include/
│   └── tokenizer.h                 # [核心] 公开 C API 头文件
├── src/                            # [核心] 库源码（编译为 libtokenizer.a）
│   ├── tokenizer_internal.h        #   内部结构与内部声明
│   ├── utf8.c / utf8.h             #   UTF-8 解码/编码（严格校验）
│   ├── trie.c / trie.h             #   前缀 Trie（encode 加速）
│   ├── normalize.c / normalize.h   #   基础归一化 + ▁ 标记
│   ├── nfkc.c / nfkc.h / nfkc_tables.inc  # 完整 NFKC，映射表驱动
│   ├── tokenizer.c                 #   生命周期 / encode(Viterbi DP) / decode / 字节回退
│   ├── model.c                     #   TKCZ 二进制读写 + JSON 导出
│   ├── sentencepiece.c             #   SentencePiece ModelProto 序列化/解析
│   └── trainer.c                   #   Unigram 训练（种子词表 / EM / 剪枝）
├── cli/
│   └── main.c                      # [核心] tok-cli 命令行入口
├── tests/
│   ├── test_tokenizer.c            # [核心] 单元与集成测试（95 项）
│   ├── dbg_norm.c                  # [辅助] 归一化调试小工具
│   └── dbg_trie.c                  # [辅助] Trie 调试小工具
├── scripts/
│   ├── build.ps1                   # [核心] Windows 构建脚本（CMake 或纯 gcc）
│   └── gen_nfkc_tables.py          # [辅助] 从 Unicode 数据生成 nfkc_tables.inc
├── benchmark/                      # [辅助+产物] 大规模压测
│   ├── gen_corpus.py               #   [辅助] 多语言语料生成器
│   ├── run_bench.ps1               #   [辅助] 压测运行脚本（采集峰值内存/阶段时间线）
│   ├── bench_enc.c                 #   [辅助] 编码吞吐压测源码
│   └── *.txt / *.tkm / *.exe       #   [产物] 压测运行输出（可重新生成）
├── build/  build2/  build3/        # [产物] 编译中间目录（可整体删除重建）
└── 2026-10-09-tokenizer-design.md  # [辅助] 设计文档
```

### 12.3 构建方法

**方式一：CMake（推荐）**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**方式二：纯 gcc（Windows 可在 MSYS2/MinGW 下执行）**

```bash
mkdir -p build && cd build
gcc -std=c11 -O2 -Wall -Wextra -I../include -I../src -c ../src/*.c
ar rcs libtokenizer.a *.o
gcc -std=c11 -O2 -I../include ../cli/main.c libtokenizer.a -lm -o tok-cli
gcc -std=c11 -O2 -I../include ../tests/test_tokenizer.c libtokenizer.a -lm -o test_tokenizer
```

**方式三：Windows PowerShell 脚本**

```powershell
.\scripts\build.ps1        # 走 CMake
.\scripts\build.ps1 -Gcc   # 走纯 gcc
```

构建产物：`build/tok-cli.exe`、`build/test_tokenizer.exe`、`build/libtokenizer.a`。

### 12.4 CLI 用法

命令总览：`tok-cli <train|encode|decode> [options]`，运行 `tok-cli --help` 查看完整参数。

**train —— 训练模型**

| 参数 | 必填 | 说明 |
|---|---|---|
| `--corpus <file>` | 是 | 训练语料，UTF-8，每行一段文本 |
| `--model <file>` | 是 | 输出模型路径（TKCZ 二进制） |
| `--vocab-size N` | 否 | 目标词表大小，默认 4096（上限 65536，见 12.7） |
| `--max-piece-len N` | 否 | 种子词表最大 piece 长度，默认 8 |
| `--max-candidates N` | 否 | 种子词表候选上限，默认 100 万 |
| `--iters N` | 否 | EM 最大迭代轮数，默认 10 |
| `--eps F` | 否 | 收敛阈值（log-likelihood 增益），默认 1e-4 |
| `--shrink PCT` | 否 | 每轮剪枝保留百分比，默认 60 |
| `--min-rounds N` | 否 | 连续增益低于阈值的轮数，默认 3 |
| `--no-byte-fallback` | 否 | 关闭字节回退兜底 |
| `--fold-case` | 否 | 开启 ASCII 大小写折叠 |
| `--json <file>` | 否 | 额外导出 HuggingFace 风格 tokenizer.json |
| `--sp-model <file>` | 否 | 额外导出 SentencePiece `.model` |
| `-v / --verbose` | 否 | 输出训练过程日志 |

```bash
tok-cli train --corpus corpus.txt --model model.tkm \
    --vocab-size 4096 --json model.json --sp-model model.model -v
```

**encode —— 编码文本**

```bash
tok-cli encode --model model.tkm --text "hello world"
tok-cli encode --model model.tkm --file input.txt
```

输出：空格分隔的十进制 id 序列。`--model` 支持 `.model` 后缀自动走 SentencePiece 加载路径。

**decode —— 解码 id**

```bash
tok-cli decode --model model.tkm --ids "1 2 3 4 5"
tok-cli decode --model model.tkm --ids "1,2,3,4,5"
```

输出：还原后的 UTF-8 文本（id 支持空格或逗号分隔）。

### 12.5 C API 用法

引用 `include/tokenizer.h`，链接 `libtokenizer.a -lm`。

**最小示例：加载 / 编码 / 解码**

```c
#include "tokenizer.h"
#include <stdio.h>

int main(void) {
    tokenizer_t *tok = tokenizer_load("model.tkm");
    if (!tok) {
        fprintf(stderr, "load failed: %s\n", tokenizer_last_error());
        return 1;
    }

    int *ids = NULL; size_t n = 0;
    const char *text = "hello world";
    if (tokenizer_encode(tok, text, strlen(text), &ids, &n) != 0) {
        fprintf(stderr, "encode failed: %s\n", tokenizer_last_error());
        tokenizer_free(tok);
        return 1;
    }
    for (size_t i = 0; i < n; i++) printf("%d ", ids[i]);
    printf("\n");
    tokenizer_free_ids(ids);

    char *out = NULL; size_t out_len = 0;
    if (tokenizer_decode(tok, ids, n, &out, &out_len) != 0) {
        fprintf(stderr, "decode failed: %s\n", tokenizer_last_error());
        tokenizer_free(tok);
        return 1;
    }
    fwrite(out, 1, out_len, stdout);
    fputc('\n', stdout);
    tokenizer_free_text(out);
    tokenizer_free(tok);
    return 0;
}
```

**训练示例**

```c
tokenizer_train_options opts = TOKENIZER_TRAIN_OPTIONS_DEFAULT;
opts.corpus_path = "corpus.txt";
opts.model_path  = "model.tkm";
opts.vocab_size  = 4096;
if (tokenizer_train(&opts) != 0) {
    fprintf(stderr, "train failed: %s\n", tokenizer_last_error());
    return 1;
}
```

常用 API 一览：`tokenizer_load` / `tokenizer_load_from_memory` / `tokenizer_load_sentencepiece`、`tokenizer_encode` / `tokenizer_decode`、`tokenizer_train`、`tokenizer_save_binary` / `tokenizer_export_json` / `tokenizer_export_sentencepiece`、`tokenizer_normalize`、`tokenizer_set_normalizer_hook`、`tokenizer_last_error`、`tokenizer_free_ids` / `tokenizer_free_text` / `tokenizer_free`。完整签名见 §8。

### 12.6 测试运行

**方式一：CMake + CTest**

```bash
cmake --build build --config Release
ctest --test-dir build
```

**方式二：直接运行测试可执行文件**

```bash
build\test_tokenizer.exe      # 或 build3\test_tokenizer.exe（手工 gcc 版本）
```

测试内容：`tests/test_tokenizer.c` 共 95 项检查，覆盖训练、encode/decode 往返、字节回退、内存加载、越界拒绝、SentencePiece `.model` 往返、NFKC 映射等。全部通过时输出 `95/95`。

### 12.7 已知限制

- **词表上限已解除（32-bit id）**：Trie 叶节点 `piece_id` 与子节点索引、`tokenizer_t.vocab_size` 均为 32-bit int，TKCZ 二进制格式以 u32 存储 vocab 与 piece 长度；`--vocab-size` 上限由 int 符号位决定（2^31-1），256K 词表（262144）压测已验证（见 §13）。
- **训练单线程、纯 CPU**：语料行长度上限 4096 字节。
- **归一化**：已为完整 NFKC；仍可通过 `tokenizer_set_normalizer_hook()` 叠加领域定制归一化。
- **Windows 控制台编码**：CLI 参数已统一转为 UTF-8，可直接传中文；但控制台输出遵循系统代码页，若乱码请 `chcp 65001` 切换终端到 UTF-8，或改用 `--file` 从 UTF-8 文件读取输入。
- **空白还原语义**：`decode` 将 `▁` 还原为空格，归一化阶段 `\n`、`\t` 等空白统一改写为空格标记，因此含换行的文本往返后换行变成空格（与 SentencePiece 行为一致）。
- **字节回退**：仅当字符序列无匹配 piece 时按 `<0xXX>` 逐字节兜底；`--no-byte-fallback` 可关闭。

---

## 13. 评测（Evaluation）

本节给出 256K 词表模型（`benchmark/model_256k.tkm`）在 2026-10-10 生成的完整评测结果。完整数据见 `benchmark/eval_report_2026-10-10.json`，人读版见 `benchmark/eval_report_2026-10-10.md`。

### 13.1 测试集构成与配比

由 `benchmark/corpus_64mb.txt`（67.1MB / 745,693 行）按行分类采样，中/英/代码三种语言；分类优先级：含 CJK → `zh`，含代码特征（花括号/分号/关键字）→ `code`，其余 → `en`。

| 语种 | 文件 | 字节 | 行数 | 配比（字节） |
|---|---|---|---|---|
| en | `corpus/en/en.txt` | 15.00 MB | 172,125 | 42.8% |
| zh | `corpus/zh/zh.txt` | 10.00 MB | 80,694 | 28.5% |
| code | `corpus/code/code.txt` | 10.00 MB | 155,408 | 28.5% |
| **合计** | | **36.70 MB** | **408,227** | 100% |

*（原始语料无日韩/西里尔等多语言行，`multilingual` 分类实际为空，故评测覆盖 en/zh/code 三类。）*

### 13.2 指标总表

| 指标 | code | en | zh | overall |
|---|---|---|---|---|
| tokens | 1,453,791 | 2,125,836 | 1,321,548 | 4,901,175 |
| BPT（bytes/token） | 7.2127 | 7.3988 | 7.9345 | 7.4881 |
| chars/token | 7.2127 | 7.3988 | 2.6855 | — |
| 还原率（字节级，200 行抽样） | 100% | 100% | 57.5%（85/200 因 NFKC） | — |
| 稳定性（Jaccard，100 句×3 扰动） | 0.6284 | 0.6632 | 0.8354 | 0.7090 |
| OOV 率 | 0（byte-fallback 设计保证） | 0 | 0 | 0 |
| utf8_completeness | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| boundary_crossing | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| char_split_rate | —（无多字节字符） | — | 0.0000 | 0.0000 |
| digit_boundary_f1 | 0.5000 | —（抽样无数字） | — | 0.5000 |
| digit_split_variability | 0.9658 | — | — | 0.9658 |
| operator_isolation | 1.0000 | 0.9667 | —（抽样无操作符） | 0.9833 |

**吞吐（整文件 encode/decode，单线程）**：encode **90,310.8 tokens/s**，decode **7,293.1 tokens/s**，模型加载 479.9 ms。

### 13.3 TokEval 维度（arXiv 2608.18062）

- **UTF-8 完整性**：`utf8_completeness`（多字节字符全部由单 piece 覆盖）/ `char_split_rate`（字符被拆分比率）/ `boundary_crossing`（跨字符边界比率）。三语种完整性 1.0、跨边界 0，UTF-8 无损。
- **数字对齐**：`digit_boundary_f1`（数字边界切分 F1）/ `digit_split_variability`（同一数字串拆分方差）/ `operator_isolation`（运算符是否独立成 piece）。code 语料抽样 30 个数字串：F1=0.5（数字串被拆散较多，与 256K 词表缺少长数字 piece 有关），变异性 0.9658；运算符基本隔离（0.9833，en 中 `+`/`-` 与上下文粘连致 0.9667）。
- **说明**：zh 抽样行无数字/操作符，en 无数字，故对应维度为空（`null`）。

### 13.4 运行方式

```bash
# 前置：构建 tok-cli.exe、训练 model_256k.tkm、准备 corpus/<lang>/*.txt
python benchmark\eval_metrics.py ^
  --model benchmark\model_256k.tkm ^
  --corpus-dir benchmark\corpus ^
  --cli build\tok-cli.exe ^
  --output benchmark\eval_report_2026-10-10.json
# 输出：benchmark\eval_report_2026-10-10.json + 同名 .md
```

评测脚本：`benchmark/eval_metrics.py`（BPT/还原率/稳定性/OOV/吞吐 + TokEval 三组指标）；语料生成：`benchmark/download_corpus.py` + `scripts/split_corpus.py`（按行分类）。

### 13.5 验收门槛（P0–P4）与当前状态

| 级别 | 门槛 | 当前状态 |
|---|---|---|
| P0 | OOV 率恒为 0 | ✅ 通过（byte-fallback 覆盖全部 UTF-8 字节） |
| P0 | 还原率 100%（字节级） | ⚠️ 部分通过：en/code 100%；zh 因完整 NFKC 归一化（全角标点→半角）字节级非 100%，语义等价（NFKC 归一化行为，见 §12.7） |
| P1 | BPT ≤ SentencePiece 基线 98% | 📊 已测：overall 7.4881（zh 2.6855 chars/token，中文压缩最优）；SP 基线待补测 |
| P2 | 稳定性 > 0.95 | ❌ 未达标：overall 0.7090（code 0.6284 / en 0.6632 / zh 0.8354），短序列 Jaccard 对扰动敏感，待改进 |
| P3 | UTF-8 完整性 1.0、跨边界 0 | ✅ 通过（三语种 utf8_completeness=1.0，boundary_crossing=0） |
| P4 | 编码吞吐 ≥ 设计目标 | ✅ 已测：encode 90,310.8 t/s、decode 7,293.1 t/s（目标值以方案文档为准） |

*（内容由AI生成，仅供参考）*
