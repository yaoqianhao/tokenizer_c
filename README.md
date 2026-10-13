---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_6e03136fc5e111f1a08152540064ee0f
    ReservedCode1: OWHMxvw+noJwEHJjJbzI0rBLDMshzMnNKxPRwWq7n3YY1zZxgsfLyjj/uXhaRL13PoeS4F4KdSOcFcH4/SrCwoGWwIcft4oYprDw1Amhm6BUE9Tq2CszVFaV/faAjgqTpQNbvqpgcWFS8yASpfjlFJreWKDzirIhsklaMsUxjmqK3aKkQ2hVvm/MJuM=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_6e03136fc5e111f1a08152540064ee0f
    ReservedCode2: OWHMxvw+noJwEHJjJbzI0rBLDMshzMnNKxPRwWq7n3YY1zZxgsfLyjj/uXhaRL13PoeS4F4KdSOcFcH4/SrCwoGWwIcft4oYprDw1Amhm6BUE9Tq2CszVFaV/faAjgqTpQNbvqpgcWFS8yASpfjlFJreWKDzirIhsklaMsUxjmqK3aKkQ2hVvm/MJuM=
---



# tokenizer-c — 工业级 Unigram Tokenizer（纯 C 实现）

基于 Google Gemini / SentencePiece 的 Unigram 算法框架，用纯 C（C11）实现的可编译、可运行、可训练的工业级 tokenizer 工程，具备「训练 — 推理 — 序列化 — 生态互操作」完整闭环。

---

## 1. 项目简介

tokenizer-c 是一个从零实现的 Unigram 分词器：

- **训练**：字符 n-gram 种子词表 + EM 迭代（Viterbi 最优切分 / 概率重估 / 词表剪枝收敛），支持断点续训；
- **推理**：`encode`（Viterbi DP + Trie 前缀加速 + UTF-8 字节回退兜底）、`decode`（`▁` 空格还原、`<0xXX>` 字节展开），字节回退保证 **OOV 恒为 0**；
- **归一化**：完整 NFKC（Unicode UAX #15：规范/兼容分解、组合排序、重组）+ 可选 ASCII 大小写折叠 + 扩展钩子；
- **序列化**：自定义二进制模型格式 `TKCZ`、HF 风格 `tokenizer.json` 导出、**SentencePiece `.model`（protobuf 线格式）导出/加载**；
- **对外形态**：C API（`include/tokenizer.h`）与命令行程序 `tok-cli`；
- **质量保障**：`tests/test_tokenizer.c` 95 项检查（当前全部通过）。

## 2. 功能总览

| 模块 | 内容 |
|---|---|
| **Unigram 训练器** | 种子词表（字符 n-gram 频次统计）、EM 迭代（Viterbi 最优切分 / 概率重估 / 词表剪枝收敛）、`TKCH` checkpoint 断点续训 |
| **推理引擎** | `encode`（Viterbi DP + Trie 前缀加速 + UTF-8 字节回退兜底）、`decode`（`▁` 空格还原、`<0xXX>` 字节展开） |
| **特殊 token** | `<unk>` / `<s>`(bos) / `</s>`(eos) / `<pad>` 固定占据 id 0 / 1 / 2 / 3 |
| **归一化** | 完整 NFKC（UAX #15）+ `▁` 空格标记 + 可选 ASCII 大小写折叠 + 定制归一化钩子 |
| **序列化** | 自定义二进制模型格式 `TKCZ` + HF 风格 JSON 导出 + **SentencePiece `.model`（protobuf）导出/加载** |
| **C API** | `tokenizer.h`：load / encode / decode / normalize / train / export_json / export_sentencepiece / load_sentencepiece 等 |
| **示例 CLI** | `tok-cli`：`train` / `encode` / `decode` 三个子命令 |
| **单元测试** | `tests/test_tokenizer.c` 95 项检查（训练、往返、字节回退、内存加载、越界拒绝、SP 往返、NFKC 映射等） |

## 3. 目录结构

```
tokenizer-c/
├── CMakeLists.txt                       # CMake 构建定义（默认构建方式）
├── README.md                            # 项目文档（本文档）
├── train_big.bat                        # 海量语料训练批处理（断点续训版）
├── 2026-10-09-tokenizer-design.md       # 设计文档（架构/算法/接口/里程碑）
├── include/
│   └── tokenizer.h                      # 公开 C API 头文件
├── src/                                 # 库源码（编译为 libtokenizer.a）
│   ├── tokenizer_internal.h             # 内部结构体与内部声明
│   ├── utf8.c / utf8.h                  # UTF-8 解码/编码（畸形字节按原始字节处理）
│   ├── trie.c / trie.h                  # 数组式前缀 Trie（encode 加速）
│   ├── normalize.c / normalize.h        # 归一化管线 + ▁ 空格标记
│   ├── nfkc.c / nfkc.h / nfkc_tables.inc# 完整 NFKC（UAX #15），映射表驱动
│   ├── tokenizer.c                      # 生命周期 / encode(Viterbi DP) / decode / 字节回退
│   ├── model.c                          # TKCZ 二进制读写
│   ├── sentencepiece.c                  # SentencePiece ModelProto 序列化/解析 + JSON 导出
│   └── trainer.c                        # Unigram 训练（种子词表 / EM / 剪枝 / checkpoint）
├── cli/
│   └── main.c                           # tok-cli 命令行入口（train / encode / decode）
├── tests/
│   ├── test_tokenizer.c                 # 单元与集成测试（95 项检查）
│   ├── dbg_norm.c                       # 归一化调试小工具
│   └── dbg_trie.c                       # Trie 调试小工具
├── scripts/
│   ├── build.ps1                        # Windows 构建脚本（CMake 或纯 gcc）
│   ├── gen_nfkc_tables.py               # 从 Unicode 数据生成 nfkc_tables.inc
│   └── split_corpus.py                  # 语料按语言分类切分
├── benchmark/                           # 大规模压测与评测
│   ├── gen_corpus.py / download_corpus.py # 多语言语料生成/下载
│   ├── eval_metrics.py                  # 评测脚本（BPT/还原率/稳定性/OOV/TokEval/吞吐）
│   ├── run_bench.ps1 / bench_enc.c      # 压测运行脚本 / 编码吞吐压测源码
│   ├── corpus/                          # 评测语料（en/zh/code 子目录）
│   ├── model_256k.tkm / model_small256k.tkm # 训练好的模型（产物）
│   ├── eval_report_2026-10-10.{json,md} # 256K 模型评测报告（2026-10-10）
│   ├── tokenizer_comparison_report.md   # 多厂商 tokenizer 对比报告（2026-10-11）
│   └── tokenizer_c_benchmark_comparison.md # 厂商 vs tokenizer_c 汇总对比（2026-10-11）
├── corpus_big/                          # 3.28GB 海量训练语料（en/zh/code + 合并文件）
├── build/  build2/  build3/             # 编译中间目录（可整体删除重建）
└── *.tkm / *.json / *.ckpt              # 训练产物（模型 / HF JSON / checkpoint，可重新生成）
```

## 4. 构建

### 方式一：CMake（推荐）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build          # 运行单元测试
```

### 方式二：纯 gcc（Windows 可在 MSYS2/MinGW 下执行）

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

**产物**：`build/tok-cli.exe`、`build/test_tokenizer.exe`、`build/libtokenizer.a`（Windows 下额外链接 shell32）。

## 5. 快速上手

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

- `encode` 输出空格分隔的十进制 id 序列；未匹配的字节以**负虚拟 id**（`-1-byte`）输出，`decode` 时展开为原始字节。
- `decode` 的 `--ids` 支持空格或逗号分隔；输出还原后的 UTF-8 文本。
- `--model` 传 `.model` 后缀时自动走 SentencePiece 加载路径，可对 SP 模型直接 encode/decode。

## 6. CLI 用法

命令总览：`tok-cli <train|encode|decode> [options]`，运行 `tok-cli --help` 查看完整参数。

### 6.1 train —— 训练模型

| 参数 | 必填 | 说明 |
|---|---|---|
| `--corpus <file>` | 是 | 训练语料，UTF-8，每行一段文本 |
| `--model <file>` | 是 | 输出模型路径（TKCZ 二进制） |
| `--vocab-size N` | 否 | 目标词表大小，默认 4096；无硬性上限（32-bit int，256K 已验证） |
| `--max-piece-len N` | 否 | 种子词表最大 piece 长度（字节），默认 8 |
| `--max-candidates N` | 否 | 种子词表候选上限，默认 1,000,000 |
| `--iters N` | 否 | EM 最大迭代轮数，默认 10 |
| `--eps F` | 否 | 收敛阈值（log-likelihood 增益），默认 1e-4 |
| `--shrink PCT` | 否 | 每轮剪枝保留百分比，默认 60 |
| `--min-rounds N` | 否 | 连续增益低于阈值的轮数，默认 3 |
| `--no-byte-fallback` | 否 | 关闭字节回退兜底 |
| `--fold-case` | 否 | 开启 ASCII 大小写折叠 |
| `--ckpt <file>` | 否 | 断点续训：每轮 EM 后保存 checkpoint；中断后同命令重跑自动恢复 |
| `--json <file>` | 否 | 额外导出 HuggingFace 风格 tokenizer.json |
| `--sp-model <file>` | 否 | 额外导出 SentencePiece `.model` |
| `-v / --verbose` | 否 | 输出训练过程日志 |

```bash
tok-cli train --corpus corpus.txt --model model.tkm \
    --vocab-size 4096 --json model.json --sp-model model.model -v
```

### 6.2 断点续训（Checkpoint Resume）

训练命令加 `--ckpt <file>` 后，训练器在**每轮 EM 结束后**自动将当前词表与训练状态保存为二进制 checkpoint（魔数 `TKCH` + 版本号 + 参数一致性校验 + 临时文件 rename 原子写入）。

- **断点恢复**：训练中断 / 断电后，以**相同命令**重跑即可自动检测 checkpoint，**跳过 seed 种子词表构建**，直接从上一轮次继续 EM 迭代。
- **参数一致性校验**：checkpoint 内记录 `vocab_size / max_piece_len / max_candidates / em_eps / shrink_pct / min_rounds / add_byte_fallback / fold_case`，恢复时参数必须一致，否则拒绝加载。
- **max_iters 不受限制**：`--iters` 不参与一致性校验，**可提高轮数继续训练**（如原 `--iters 8` 中断后可改 `--iters 10` 重跑续训）。
- **重复运行安全**：checkpoint 已存在时重复执行会直接续训、不重复 seed，因此可直接重跑训练命令或批处理脚本续训。

```bash
# 256K 大语料续训示例
tok-cli train --corpus corpus_big.txt --model model_256k.tkm \
    --vocab-size 262144 --json model_256k.json \
    --iters 8 --shrink 60 --ckpt model_256k.ckpt -v
```

### 6.3 海量语料训练脚本（train_big.bat）

仓库根目录提供 `train_big.bat`（Windows 批处理），一键训练 3.28GB 海量语料（`corpus_big/corpus_big.txt`，由 `corpus_big/en|zh|code/` 子目录语料合并）：

- **运行方式**：`train_big.bat [iters] [shrink] [vocab_size]`，默认 `iters=8  shrink=60  vocab_size=262144`。
- **产物路径**：模型 `model_big.tkm`、JSON `model_big.json`、checkpoint `model_big.ckpt` 均输出到仓库根目录；CLI 固定使用 `build3\tok-cli.exe`（首次运行自动补齐 `libwinpthread-1.dll`）。
- **日志**：训练在后台最小化窗口运行，日志写入 `train_big.log`（stdout）与 `train_big.log.err`（stderr）。
- **查看进度**：`type train_big.log.err | findstr /i "seed EM round"`；进程/内存：`tasklist /fi "imagename eq tok-cli.exe"`。
- **断点续训**：脚本内置 `--ckpt E:\tokenizer_c\model_big.ckpt`，若该文件已存在会自动检测并从断点续训（跳过 seed），**直接重跑本脚本即可续训**。
- **耗时预估**：`iters=8` 约 110~140 小时（种子约 3~4h + 每轮 EM 约 10~14h）。

### 6.4 训练语料说明

训练语料 `corpus_big/corpus_big.txt` 约 3.28 GB（精确 3,522,679,232 字节，UTF-8，每行一段文本），由 `corpus_big/` 下三个子目录语料按 **en → zh → code 顺序直接拼接** 而成，随提交 `585102a`（"add big lan data and train code"）加入仓库：

| 子语料 | 文件 | 大小（字节） | 行数 | 原始来源（HuggingFace 数据集） |
|---|---|---|---|---|
| 英文 | `corpus_big/en/en.txt` | 2,312,810,695 | 9,093,878 | `allenai/c4`（en，train 分片） |
| 中文 | `corpus_big/zh/zh.txt` | 504,969,474 | 3,679,146 | `pleisto/wikipedia-cn-20230720-filtered`（train） |
| 代码 | `corpus_big/code/code.txt` | 704,899,063 | 17,303,336 | 仓库未保留原始下载缓存（内容以 JavaScript 等开源代码文本为主） |

**拼接校验**：合并文件行数 30,076,360 与字节数均精确等于三者之和（9,093,878 + 3,679,146 + 17,303,336 = 30,076,360）；边界行（第 9,093,879 行、第 12,773,025 行）与 `zh/zh.txt`、`code/code.txt` 首行逐字节一致；合并文件首行为 `en/en.txt` 首行。训练日志 `train_big.log.err` 确认加载 `corpus lines: 30,076,360`。

**下载方式**：经 HuggingFace `datasets` 库 streaming 模式下载，原始分片缓存在 `corpus_big/_hf_cache/`（标准 HF 缓存目录，含下载元数据）：

- **英文**：缓存 `_hf_cache/en/c4-train.00000~00002-of-01024.json.gz`（3 个分片，各约 318~320 MB）；
- **中文**：缓存 `_hf_cache/wikipedia-cn-20230720-filtered.json`（JSONL，含 prompt/completion 字段，约 101.8 万条）与 `_hf_cache/data/train-00000~00002-of-01126.parquet`（分片）；
- **代码**：`_hf_cache/` 中无对应缓存，具体数据集不可从仓库证实。

**数据源配置**（与 `benchmark/download_corpus.py` 声明一致，`datasets.load_dataset(..., streaming=True)`，主源加载失败自动回退）：

| 语言 | 首选数据源 | 回退数据源 |
|---|---|---|
| en | `wikitext/wikitext-103-raw-v1`（split=test） | `allenai/c4`（split=train）——缓存证实实际使用 |
| zh | `pleisto/wikipedia-cn-20230720-filtered`（split=train）——缓存证实实际使用 | `csebuetnlp/xwikis` |
| code | `codeparrot/github-code`（py/js/java） | `bigcode/the-stack`（split=train） |
| multilingual | `facebook/flores`（devtest） | `Helsinki-NLP/opus-100`（en-de，train） |

> 说明：海量训练语料仅含 en/zh/code 三类（无 multilingual 行）；原始下载脚本中单行超过 4000 字节会被截断（`download_corpus.py` 行为）；仓库内未见针对这 3.28 GB 语料的独立过滤/分类脚本（与评测语料 `corpus_64mb.txt` 由 `scripts/split_corpus.py` 按 CJK / 代码特征 / 其余规则分类切分不同，详见 §13.1）。

### 6.5 encode —— 编码文本

```bash
tok-cli encode --model model.tkm --text "hello world"
tok-cli encode --model model.tkm --file input.txt
```

输出：空格分隔的十进制 id 序列；`--model` 传 `.model` 后缀自动走 SentencePiece 加载路径。

### 6.6 decode —— 解码 id

```bash
tok-cli decode --model model.tkm --ids "1 2 3 4 5"
tok-cli decode --model model.tkm --ids "1,2,3,4,5"
```

输出：还原后的 UTF-8 文本（id 支持空格或逗号分隔）。

## 7. 算法说明

### 7.1 训练流程（trainer.c）

1. **种子词表**：对归一化后语料统计字节级字符 n-gram 频次（`max_piece_len` 默认 8），按频次降序截断至 `max_candidates`（默认 100 万）；特殊 token 固定置于头部。
2. **EM 迭代**（`--iters` 默认 10 轮）：
   - **E 步**：对每条序列做 **Viterbi 最优切分**（基于当前词表与 score），累计各 piece 使用频次。
   - **M 步**：`P(t) = count(t) / Σcount`，`score(t) = log P(t)`。
   - **剪枝**：每轮保留特殊 token + 高频 top-k（`k = max(目标词表, 当前 × shrink%)`，`--shrink` 默认 60），低分 piece 逐轮淘汰，向 `--vocab-size` 收敛。
3. **收敛**：词表达到目标规模且 log-likelihood 增益连续 `--min-rounds`（默认 3）轮低于 `--eps`（默认 1e-4），或达到最大轮数。
4. **字节回退**：开启时（默认）在词表中追加 256 个 `<0xXX>` 特殊 piece，用于 SentencePiece 导出兼容与 OOV 兜底。

### 7.2 编码（tokenizer.c）

1. **归一化**：完整 NFKC（见 §9）+ 可选大小写折叠 + `▁` 空格标记（每个空格替换为 U+2581，字符串首空格保留标记）。
2. **Viterbi DP**：`dp[i] = max_j (dp[i-len] + score(piece))`，Trie 在当前位置收集所有可匹配前缀，复杂度近似 `O(n × max_piece_len)`。
3. **字节回退兜底**：字符序列无匹配 piece 时，按字节输出**负虚拟 id**（`-1-byte`，score 固定为 -10.0），保证任意 UTF-8 输入均可编码、OOV 恒为 0。

### 7.3 解码

- `▁` → 空格还原（连续 `▁` 保留为多个空格，行首标记不产生空格）；
- `<0xXX>` / 负 id → 原始单字节展开；
- 两者结合实现**无损往返**：`decode(encode(x)) == normalize(x)`（在归一化语义下等价）。

## 8. 模型格式

### 8.1 TKCZ（自定义二进制格式）

小端序布局：

| 偏移 | 内容 |
|---|---|
| 0 | 魔数 `TKCZ`（4 字节） |
| 4 | 版本 `u32`（当前 1） |
| 8 | `vocab_size`（u32） |
| 12 | 特殊 id 4×`i32`：unk / bos / eos / pad |
| 28 | 标志 `u32`：bit0 = 字节回退开启 |
| 32 | piece 表：每项 = `len`(u32) + 原始字节串(len 字节) + `score`(f32, log-prob) |

加载时对 `vocab_size` 有合理性上限校验（> 1 亿拒绝）。另有内部 `tokenizer_save_binary` / `tokenizer_load_from_memory` 提供文件与内存两种读写。

### 8.2 JSON 导出

`tokenizer_export_json()` 输出 HuggingFace 风格 `tokenizer.json`（含 `type: "Unigram"`、normalizer、pre_tokenizer、byte_fallback、special_tokens、piece 表），可由 HF `tokenizers` 库加载使用。

### 8.3 SentencePiece 兼容（.model）

`src/sentencepiece.c` 实现 SentencePiece ModelProto（protobuf wire format）的序列化与解析，打通与主流 LLM 生态互操作：

| 能力 | 入口 | 说明 |
|---|---|---|
| 导出 `.model` | `tokenizer_export_sentencepiece(tok, path)` | 将当前模型写出为 SentencePiece Unigram `.model`（piece 表 + score + special token 映射），可由 transformers / vLLM 的 `SentencePieceProcessor` 直接加载 |
| 加载 `.model` | `tokenizer_load_sentencepiece(path)` | 读取任意 SentencePiece `.model`（Unigram / BPE / word / char 均可）为内部模型，encode/decode 行为一致 |
| JSON 导出 | `tokenizer_export_json` | HuggingFace 风格 `tokenizer.json`（字段齐全） |

## 9. 归一化实现说明（完整 NFKC）

本实现为**完整 NFKC**（Unicode UAX #15）：

| 能力 | 覆盖 | 说明 |
|---|---|---|
| 规范分解（NFD） | 组合字符 → 基字符 + 组合符号 | `nfkc_decompose_cp` 查表分解 |
| 兼容分解 | 全角/半角、带圈数字 `①→1`、连字 `ﬃ→ffi`、罗马数字 `Ⅳ→IV`、上下标、兼容货币符号等 | 表驱动，覆盖 1000+ 映射 |
| 组合排序 | 按规范组合类（CCC）重排组合符号 | `nfkc_ccc` |
| 重组（NFC） | 规范组合回写，尊重 UAX #15 组合排除 | `nfkc_compose` |
| 大小写折叠（可选） | ASCII `A-Z` → `a-z` | 开关控制 |
| U+2581 标记 | 空格 → `▁`（编码侧） | 保证无损往返 |

实现文件：`src/nfkc.c`（分解/组合核心）+ `src/nfkc_tables.inc`（静态映射表，8,422 行，由 `scripts/gen_nfkc_tables.py` 从 Unicode 数据生成）；`src/normalize.c` 负责调用 nfkc 核心并做 `▁` 标记。

**扩展钩子**：`tokenizer_set_normalizer_hook()`（`tokenizer_normalizer_hook_fn`）可在完整 NFKC 之上叠加领域定制归一化，无需改动编码/解码主流程。

## 10. C API

公开头文件：`include/tokenizer.h`（MIT License，C11，支持 C++ 调用，`extern "C"` 包裹）。

### 10.1 生命周期与元数据

```c
tokenizer_t *tokenizer_load(const char *model_path);              // 加载 .tkm
tokenizer_t *tokenizer_load_from_memory(const void *data, size_t size); // 内存加载
void         tokenizer_free(tokenizer_t *tok);                    // 释放（可传 NULL）

size_t tokenizer_vocab_size(const tokenizer_t *tok);
tokenizer_special_tokens tokenizer_special_tokens_get(const tokenizer_t *tok);
int tokenizer_unk_id(const tokenizer_t *tok);
int tokenizer_bos_id(const tokenizer_t *tok);
int tokenizer_eos_id(const tokenizer_t *tok);
int tokenizer_pad_id(const tokenizer_t *tok);

int tokenizer_piece_to_id(const tokenizer_t *tok, const char *piece, size_t len); // 未找到返回 -1
int tokenizer_id_to_piece(const tokenizer_t *tok, int id, const char **piece, size_t *len);
```

### 10.2 编码 / 解码

```c
/* 编码：out_ids 由 malloc 分配，调用者用 tokenizer_free_ids 释放 */
int tokenizer_encode(const tokenizer_t *tok, const char *text, size_t len,
                     int **out_ids, size_t *out_n);
void tokenizer_free_ids(int *ids);

/* 解码：out_text 由 malloc 分配，调用者用 tokenizer_free_text 释放 */
int tokenizer_decode(const tokenizer_t *tok, const int *ids, size_t n,
                     char **out_text, size_t *out_len);
void tokenizer_free_text(char *text);
```

### 10.3 归一化

```c
/* 返回 malloc 的 UTF-8 字符串（含 ▁ 标记），调用者用 tokenizer_free_text 释放 */
char *tokenizer_normalize(const char *text, size_t len, size_t *out_len);

/* 定制归一化钩子：返回 malloc 字符串，或置 *out = NULL 回退到内置归一化 */
typedef char *(*tokenizer_normalizer_hook_fn)(const char *text, size_t len,
                                              size_t *out_len, void *userdata);
void tokenizer_set_normalizer_hook(tokenizer_t *tok,
                                   tokenizer_normalizer_hook_fn hook,
                                   void *userdata);
```

### 10.4 训练与导出

```c
typedef struct tokenizer_train_options {
    const char *corpus_path;   /* 必填：语料路径 */
    const char *model_path;    /* 必填：输出 .tkm 路径 */
    size_t  vocab_size;        /* 0 => 4096 */
    int     max_piece_len;     /* 0 => 8 */
    int     max_candidates;    /* 0 => 1000000 */
    int     max_iters;         /* 0 => 10 */
    double  em_eps;            /* 0 => 1e-4 */
    int     shrink_pct;        /* 0 => 60 */
    int     min_rounds;        /* 0 => 3 */
    int     add_byte_fallback; /* 0 => 开(1) */
    int     fold_case;         /* ASCII 大小写折叠 */
    int     verbose;
    const char *json_path;         /* 可选 HF tokenizer.json 导出 */
    const char *checkpoint_path;   /* 可选 checkpoint 断点续训 */
} tokenizer_train_options;

int tokenizer_train(const tokenizer_train_options *opts);          // 成功返回 0
int tokenizer_export_json(const tokenizer_t *tok, const char *path);
int tokenizer_export_sentencepiece(const tokenizer_t *tok, const char *path);
tokenizer_t *tokenizer_load_sentencepiece(const char *path);
```

### 10.5 错误处理与线程安全

```c
const char *tokenizer_last_error(void);  // 线程局部错误信息（无错误时为 NULL）
```

- 加载完成后，`tokenizer_t` 可被多个线程**并发**执行 encode/decode（只读共享，无锁）。
- 训练为单线程接口；错误查询为线程局部（thread-local）。

### 10.6 C API 最小示例

```c
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>

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

编译：引用 `include/tokenizer.h`，链接 `libtokenizer.a -lm`（Windows 另加 shell32）。

## 11. 测试运行

**方式一：CMake + CTest**

```bash
cmake --build build --config Release
ctest --test-dir build
```

**方式二：直接运行测试可执行文件**

```bash
build\test_tokenizer.exe      # 或 build3\test_tokenizer.exe（手工 gcc 版本）
```

测试内容：`tests/test_tokenizer.c` 共 95 项检查，覆盖训练、encode/decode 往返、字节回退、内存加载、越界拒绝、SentencePiece `.model` 往返、NFKC 映射等。全部通过时输出 `OK: all 95 checks passed`。

## 12. 已知限制

- **词表上限已解除（32-bit id）**：Trie 叶节点 `piece_id` 与子节点索引、`tokenizer_t.vocab_size` 均为 32-bit int，TKCZ 以 u32 存储 vocab 与 piece 长度；`--vocab-size` 上限由 int 符号位决定（2^31-1），256K 词表（262144）压测已验证。
- **训练单线程、纯 CPU**：语料行长度上限 4096 字节。
- **归一化**：已为完整 NFKC；仍可通过 `tokenizer_set_normalizer_hook()` 叠加领域定制。
- **Windows 控制台编码**：CLI 通过 `GetCommandLineW` + `WideCharToMultiByte` 将命令行参数统一转为 UTF-8，可直接传中文；但控制台输出遵循系统代码页，若乱码请 `chcp 65001` 切换终端到 UTF-8，或改用 `--file` 从 UTF-8 文件读取输入。
- **空白还原语义**：`decode` 将 `▁` 还原为空格；归一化阶段把 `\n`、`\t` 等空白统一改写为空格标记，故含换行的文本往返后换行会变成空格（与 SentencePiece 行为一致）。
- **字节回退**：仅当字符序列无匹配 piece 时按字节兜底（负虚拟 id）；`--no-byte-fallback` 可关闭。

## 13. 评测

### 13.1 256K 模型评测（2026-10-10）

模型 `benchmark/model_256k.tkm`（vocab=262144），测试集由 `benchmark/corpus_64mb.txt`（67.1MB / 745,693 行）按行分类采样：en 15.00MB / zh 10.00MB / code 10.00MB（合计 36.70MB / 408,227 行）。

| 指标 | code | en | zh | overall |
|---|---|---|---|---|
| tokens | 1,453,791 | 2,125,836 | 1,321,548 | 4,901,175 |
| BPT（bytes/token） | 7.2127 | 7.3988 | 7.9345 | 7.4881 |
| chars/token | 7.2127 | 7.3988 | 2.6855 | — |
| 还原率（字节级，200 行抽样） | 100% | 100% | 57.5%（85/200 因 NFKC 归一化） | — |
| 稳定性（Jaccard，100 句×3 扰动） | 0.6284 | 0.6632 | 0.8354 | 0.7090 |
| OOV 率 | 0（byte-fallback 设计保证） | 0 | 0 | 0 |
| utf8_completeness | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| boundary_crossing | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| char_split_rate | —（无多字节字符） | — | 0.0000 | 0.0000 |
| digit_boundary_f1 | 0.5000 | —（抽样无数字） | — | 0.5000 |
| digit_split_variability | 0.9658 | — | — | 0.9658 |
| operator_isolation | 1.0000 | 0.9667 | —（抽样无操作符） | 0.9833 |

**吞吐（整文件 encode/decode，单线程）**：encode **90,310.8 tokens/s**，decode **7,293.1 tokens/s**，模型加载 479.9 ms。

### 13.2 SP 基线对比（2026-10-11）

用官方 SentencePiece（0.2.2）在**同一训练语料** `benchmark/corpus_64mb.txt` 上训练 262,144 词表 Unigram 模型（`--model_type=unigram --normalization_rule_name=nfkc --character_coverage=0.9995 --hard_vocab_limit=true`），同一套 `eval_metrics.py` 同语料评测：

| 语种 | 项目 BPT | SP 基线 BPT | 项目/基线 |
|---|---|---|---|
| code | 7.2127 | 4.5740 | 157.7% |
| en | 7.3988 | 5.7108 | 129.6% |
| zh | 7.9345 | 8.6622 | **91.6%（领先 8.4%）** |
| **overall** | **7.4881** | **5.8653** | 127.7% |

其他基线观测：roundtrip en/code 100%、zh 非 100%（同 NFKC 语义等价）；stability overall ≈0.8617；OOV=0；UTF-8 完整性 0.9957、跨边界 0；encode 1,252,416.5 t/s、decode 9,829.7 t/s。

结论：项目对**中文压缩更优**（zh BPT 领先 8.4%）；code/en 落后，差距主因是训练收敛轮数（项目 10 轮 EM vs SP 默认 2 轮子迭代）与剪枝策略差异，可提高 `--iters`/降低 `--shrink` 或扩充 code/en 语料后重训。

### 13.3 多厂商横向对比（2026-10-11）

详见 `benchmark/tokenizer_comparison_report.md`（五家厂商逐项评测）与 `benchmark/tokenizer_c_benchmark_comparison.md`（汇总对比）。

| tokenizer | 类型/词表 | overall BPT | 还原率 | 稳定性 | UTF-8 完整性 | 操作符隔离 |
|---|---|---|---|---|---|---|
| tokenizer_c 256K | Unigram 262144 | 7.4881 | en/code 100%、zh 57.5% | 0.7090 | 1.0000 | 0.9833 |
| SentencePiece 基线 | Unigram 262144 | 5.8653 | en/code 100%、zh <100% | ≈0.8617 | 0.9957 | - |
| Qwen2.5 | BPE 151643 | 5.1523 | 100% | 0.8230 | 1.0000 | 0.1667 |
| gpt6 (gpt-oss-120b) | BPE 199998 | 4.9335 | 100% | 0.8269 | 1.0000 | 0.1667 |
| grok (grok-2) | BPE 131072 | **4.7766** | 100% | 0.8272 | 0.9924 | 0.9667 |
| doubao (Seed-OSS-36B) | BPE 155121 | 4.8973 | 100% | 0.8283 | 1.0000 | 0.1667 |
| hunyuan (Hunyuan-A13B) | BPE 127957 | 5.1990 | 100% | 0.8220 | 1.0000 | 0.1667 |

要点：

1. **压缩效率**：五家厂商整体 BPT 全部优于 tokenizer_c 256K（低 30%~36%）；中文差距最大（zh BPT 4.13~5.49 vs 7.93）。
2. **设计取向**：tokenizer_c 256K 与 grok 属"整字优先"路线（字符拆分率≈0、操作符隔离 0.96+）；Qwen/gpt6/doubao/hunyuan 属字节级 BPE（中文按字节拆分，换来更高压缩率）。
3. **tokenizer_c 独特优势**：中文整字覆盖、操作符独立成 token、UTF-8 无损，适合"字符语义可解释"的应用。
4. **吞吐仅量级参考**：tokenizer_c 为 C CLI、SP 为 C++，五家为 tokenizers/tiktoken（Rust），实现与优化程度不同，不可直接比较。

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

评测脚本：`benchmark/eval_metrics.py`（BPT / 还原率 / 稳定性 / OOV / 吞吐 + TokEval 三组指标）；语料生成：`benchmark/download_corpus.py` + `scripts/split_corpus.py`（按行分类）。

## 14. 验收门槛（P0–P4）与当前状态

| 级别 | 门槛 | 当前状态 |
|---|---|---|
| P0 | OOV 率恒为 0 | 通过（byte-fallback 覆盖全部 UTF-8 字节） |
| P0 | 还原率 100%（字节级） | 部分通过：en/code 100%；zh 因完整 NFKC 归一化（全角标点→半角）字节级非 100%，语义等价 |
| P1 | BPT ≤ SentencePiece 基线 98% | 未达标：overall 7.4881 vs SP 5.8653（127.7%）；zh 领先 8.4%，差距集中在 code/en |
| P2 | 稳定性 > 0.95 | 未达标：overall 0.7090，短序列 Jaccard 对扰动敏感 |
| P3 | UTF-8 完整性 1.0、跨边界 0 | 通过（三语种 utf8_completeness=1.0，boundary_crossing=0） |
| P4 | 编码吞吐 ≥ 设计目标 | 已测：encode 90,310.8 t/s、decode 7,293.1 t/s |

## 15. 迭代记录

| 迭代 | 状态 | 内容 |
|---|---|---|
| v1.0（初始） | 完成 | 纯 C 工程骨架、Unigram 训练器、推理引擎、TKCZ/JSON、C API、CLI、测试 |
| v1.1 方向1：SentencePiece proto 兼容 | 完成 | `src/sentencepiece.c`：`.model` 导出/加载 + HF 风格 `tokenizer.json`；95/95 检查通过 |
| v1.1 方向2：完整 NFKC | 完成 | `src/nfkc.c` + `src/nfkc_tables.inc`（UAX #15 分解/排序/重组）；95/95 检查通过 |
| v1.1 方向3：256K 大规模压测 | 完成 | 256K 词表多语言压测：10 轮 EM 收敛（约 2.75h）；Trie 叶节点 id 升级 32-bit；评测报告 `benchmark/eval_report_2026-10-10.{json,md}` |
| v1.2：SP 基线补测与多厂商对比 | 完成 | 官方 SP 262144 同口径基线补测（§13.2）；五家厂商 tokenizer 横向评测（§13.3，报告见 benchmark/） |
| v1.3：3.28GB 海量语料训练 | 进行中 | `train_big.bat` 一键训练 256K 词表（`iters=8  shrink=60`，含断点续训），产物 `model_big.tkm` / `model_big.json` / `model_big.ckpt` |

## 16. 相关文档

- `2026-10-09-tokenizer-design.md`：设计文档（需求、架构、算法、接口、里程碑、验证标准）
- `2026-10-11-code-review-fix-plan.md`：代码审阅与修改建议（P0–P2 问题清单、分批改法、回归清单）
- `benchmark/eval_report_2026-10-10.md`：256K 模型评测报告
- `benchmark/tokenizer_comparison_report.md`：多厂商 tokenizer 对比报告
- `benchmark/tokenizer_c_benchmark_comparison.md`：厂商 vs tokenizer_c 汇总对比
- `benchmark/qwen_tokenizer_eval_report.md`：Qwen2.5 tokenizer 评测
*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*
