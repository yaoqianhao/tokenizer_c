---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_c8048890c76f11f1820a525400638852
    ReservedCode1: jEkbcsdFKZ92aO2kpE5oapAq/W9JC9zlc4zWKKBR7ryKTOoCdn8H2dJs4e5e/tf88c+eSCj9EYpJ13NwEkLhbnh5ZT3zRf7cywR8/Hvyk3fzAyWFSsJOBx3ffNA7jhhMmuBfk3I4FXGHE/Tu67/MoJI+8OUODXgh9G8mC4lBlr8A2tRWqqizUrw3tfU=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_c8048890c76f11f1820a525400638852
    ReservedCode2: jEkbcsdFKZ92aO2kpE5oapAq/W9JC9zlc4zWKKBR7ryKTOoCdn8H2dJs4e5e/tf88c+eSCj9EYpJ13NwEkLhbnh5ZT3zRf7cywR8/Hvyk3fzAyWFSsJOBx3ffNA7jhhMmuBfk3I4FXGHE/Tu67/MoJI+8OUODXgh9G8mC4lBlr8A2tRWqqizUrw3tfU=
---

# tokenizer-c 初学者学习材料

> 面向对象：第一次接触分词器（tokenizer）与 Unigram 算法的同学。
> 编写日期：2026-10-14
> 代码基线：`E:\tokenizer_c`（纯 C11 实现，最新模型 `model_big2`，词表 262,144）
> 阅读前提：会一点 C 语言、听说过"把文本转成 id 序列"这件事即可，不需要预先懂 Unigram。

---

## 0. 怎么用这份材料

建议顺序：

1. 先看第 1 节建立概念（tokenizer 在干什么、Unigram 在优化什么）。
2. 再看第 2 节，把工程编译跑起来（**一定要亲手跑通，否则后面全是纸上谈兵**）。
3. 然后带着"这个词表长什么样"的问题看第 3 节数据结构。
4. 第 4、5 节是核心算法，建议对着源码读（每小节都给了精确文件/行号/函数名）。
5. 第 6、7 节是动手环节：CLI 与评测。
6. 最后第 8 节是"踩坑速查表"，遇到问题先查它。

文档里所有命令**都在本机实测跑通过**，输出样例为真实输出（语料是一份 5 行的迷你语料）。

---

## 1. Tokenizer 基本概念与这份代码的定位

### 1.1 概念：tokenizer 到底解决什么问题

神经网络（语言模型）不认识文字，只认识整数 id。tokenizer 的职责就是把一段 UTF-8 文本变成 `int` 序列：

```
"hello world"  --encode-->  [8, 16, -109, -101]
[8, 16, -109, -101] --decode--> "hello world"      （必须能原样还原）
```

三个必须先建立的概念：

| 概念 | 含义 | 在本工程中的体现 |
|---|---|---|
| **词表（vocab）** | 允许出现的"片段（piece）"全集，每条有唯一 id | `src/tokenizer_internal.h` 的 `struct tokenizer`：`pieces / piece_lens / scores` |
| **切分（segmentation）** | 把一句话切成词表里片段的序列 | `tokenizer_encode()`，用 Viterbi 动态规划求最优切分 |
| **打分（score）** | 每个 piece 一个分数（对数概率），用于比较哪种切分更好 | `struct tokenizer.scores`（float，恒为负数） |

衡量分词器好坏常用的一个指标是 **BPT（Bytes Per Token，每 token 承载多少字节）**：BPT 越大说明同样长度的文本用更少 token 表示，压缩效率越高。第 7 节会看到它。

### 1.2 Unigram 算法的核心假设

Unigram 语言模型假设一句话的生成过程是：**独立**地从词表里挑出一串 piece 拼起来。于是：

```
P(句子) = Π P(piece_i)          （各 piece 相互独立）
```

因为每个 piece 的概率都小于 1，直接乘容易下溢，所以工程里存的是 **对数概率（log-prob）**，乘积变加法：

```
log P(句子) = Σ log P(piece_i)
```

给定一句话，可能有非常多切法（例如 "hello" 可以切成 `h e l l o`，也可以切成 `hel lo`）。Unigram 的选择原则是：

> **在所有能拼出这句话的切分里，选 log 概率之和最大的那一个。**

求这个最优切分用 **Viterbi 动态规划**（本工程 `src/tokenizer.c` 的 `tokenizer_encode()` 与 `src/trainer.c` 的 `viterbi_segment()`）。

而"每个 piece 的概率到底是多少"就是**训练**要解决的事：用 EM 算法从语料里迭代估计。

### 1.3 这份代码的定位

`E:\tokenizer_c` 是一套**工业级、纯 C11、无第三方运行时依赖**的 Unigram tokenizer：

- 算法骨架对标 Google **Gemini / SentencePiece 的 Unigram 框架**（见 `2026-10-09-tokenizer-design.md` §2）；
- 支持**训练 → 推理（encode/decode）→ 多格式导出**全链路；
- 导出格式兼容生态：
  - `.tkm`：本工程自定义二进制（TKCZ），加载最快；
  - `.json`：HF `tokenizer.json` 风格（`tokenizer_export_json()`）；
  - `.model`：SentencePiece `ModelProto`，可被 transformers / vLLM 的 `SentencePieceProcessor` 当 UNIGRAM 模型加载（`src/sentencepiece.c`）。
- 提供 CLI（`tok-cli.exe`）与 C API（`include/tokenizer.h`）两种使用方式。

一句话总结 **"它是谁"**：它是一个**可以自己训练、自己推理、还能导出给 Python 生态用**的 Unigram 分词器 C 实现。

### 1.4 一份代码的运行链路（先建立全局图）

```
                 cli/main.c  （命令行入口，UTF-8 参数处理）
                      |
        train --------+-------- encode / decode
          |                            |
  src/trainer.c                 src/tokenizer.c
  （种子 → EM → 剪枝）            （归一化 → Viterbi → ids / 反向拼回文本）
          |                            |
          |                     src/normalize.c + src/nfkc.c + src/nfkc_tables.inc
          |                            |
          +-------> src/model.c  （TKCZ 二进制保存/加载）
                    src/sentencepiece.c（.model / .json 导出与加载）
                    src/trie.c  （前缀 Trie，加速匹配）
                    src/utf8.c  （UTF-8 编解码工具）
```

**初学者常见疑问**

- Q：为什么既要在 C 里写训练，又有 `train_big2.py`？
  A：Python 脚本只是**薄薄的调度壳**（`E:\tokenizer_c\train_big2.py`），它检查 DLL、拼命令行、后台启动 `tok-cli.exe train` 并把日志重定向到文件；真正的训练逻辑全在 C 里。
- Q：`.tkm` 和 `.model` 谁更快？
  A：`.tkm` 是自定义格式，解析简单，加载最快；`.model` 是为了兼容 Python 生态。二者编码结果一致（`tests/test_tokenizer.c` 第 9 组检查里有"sp encode parity with original"）。

---

## 2. 目录地图与构建

### 2.1 真实目录结构

```
E:\tokenizer_c\
├── include\tokenizer.h          ← 唯一的公开 C API 头文件（166 行）
├── src\
│   ├── tokenizer.c              ← encode / decode / 加载 / 元数据
│   ├── trainer.c                ← 训练：种子 → EM → 剪枝 → 落盘（1096 行）
│   ├── model.c                  ← TKCZ 二进制保存/加载 + JSON 导出
│   ├── sentencepiece.c          ← SentencePiece .model 导出/加载
│   ├── normalize.c              ← 基础归一化（空格标记、大小写折叠）
│   ├── nfkc.c                   ← 完整 NFKC 归一化
│   ├── nfkc_tables.inc          ← 自动生成的 Unicode 表（勿手改）
│   ├── trie.c / trie.h          ← 前缀 Trie
│   ├── utf8.c / utf8.h          ← UTF-8 工具
│   └── tokenizer_internal.h     ← 内部结构体 & 内部函数声明
├── cli\main.c                   ← CLI：train / encode / decode
├── tests\test_tokenizer.c       ← 单元测试（420 行，11 组检查）
├── scripts\
│   ├── build.ps1                ← Windows 一键编译
│   ├── gen_nfkc_tables.py       ← 生成 src\nfkc_tables.inc
│   └── split_corpus.py          ← 把大语料拆成评测子集
├── benchmark\eval_metrics.py    ← 评测脚本（指标计算）
├── benchmark\corpus\{en,zh,code}\*.txt  ← 评测语料
├── build\                       ← 编译产物（tok-cli.exe、test_tokenizer.exe）
├── CMakeLists.txt
├── README.md                    ← 权威使用说明（32KB，建议通读）
├── 2026-10-09-tokenizer-design.md  ← 设计文档（算法选型、架构、格式约定）
└── model_big2.tkm / .json / .ckpt  ← 最新 262K 模型
```

### 2.2 三种构建方式

**方式一：CMake（推荐）**

```powershell
cd E:\tokenizer_c
cmake -B build -S .
cmake --build build --config Release
```

**方式二：脚本（Windows 一键）**

```powershell
cd E:\tokenizer_c
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

`scripts/build.ps1` 会优先用 CMake，找不到 CMake 就退回逐个文件调用 `gcc`，产出：

| 产物 | 说明 |
|---|---|
| `build\tok-cli.exe` | 命令行工具（本材料的主角） |
| `build\test_tokenizer.exe` | 单元测试 |
| `build\libtokenizer.a` | 静态库，供 C 项目链接 |

**方式三：手工 gcc（MSYS2/MinGW）**

```bash
cd /e/tokenizer_c
gcc -std=c11 -O2 -Iinclude -Isrc src/*.c cli/main.c -o build/tok-cli.exe -lm
```

**验证编译成功**（真实输出）：

```powershell
E:\tokenizer_c\build\tok-cli.exe
# Usage:
#   tok-cli train --corpus <file> --model <out.tkm> [options]
#   ...
```

**初学者常见疑问 / 易错点**

- Q：为什么双击 `tok-cli.exe` 窗口一闪而过？
  A：它是命令行程序。必须**先进 PowerShell，再输入命令**（或 `cd` 到 exe 所在目录后 `.\tok-cli.exe`）。
- ⚠️ 易错点：跑训练时如果提示缺少 `libwinpthread-1.dll`，把该 DLL 复制到 exe 同目录（`train_big2.py` 里就是自动做这件事）。
- ⚠️ 易错点：改过 `src\normalize.c` 里的 NFKC 逻辑后忘了重新编译，直接用旧 exe 测，会得到"改了没用"的错觉。
- ⚠️ 易错点：`src\nfkc_tables.inc` 是 `scripts\gen_nfkc_tables.py` 自动生成的，**手改会被下次生成覆盖**；要改表先改脚本。

---

## 3. 关键数据结构

### 3.1 内存中的词表：`struct tokenizer`

位置：`src/tokenizer_internal.h`（约第 50 行起）

```c
struct tokenizer {
    int vocab_size;          /* 词表大小，如 262144 */
    char **pieces;           /* 每个 piece 的字节串（piece 是"字节串"不是"字符串"！） */
    int *piece_lens;         /* 每个 piece 的字节长度 */
    float *scores;           /* 对数概率，负数 */
    int unk_id, bos_id, eos_id, pad_id;   /* 特殊 token 固定占 0/1/2/3 */
    int byte_fallback;       /* 是否启用 <0xXX> 字节兜底 */
    int fold_case;           /* 是否在编码前做 ASCII 大小写折叠 */
    trie_t *trie;            /* 由 tokenizer_build_trie() 重建，加速匹配 */
    tokenizer_normalizer_hook_fn norm_hook;
    void *norm_userdata;
};
```

要点：

1. **piece 用 `(指针, 长度)` 表示而不是 `char*`**。因为 piece 内部可能含 `\0` 之类非文本字节（byte fallback 场景），用 NUL 结尾字符串会出错。所有 API 都成对出现：`tokenizer_id_to_piece(tok, id, &piece, &len)`。
2. **特殊 token 固定占前 4 个 id**：`<unk>=0, <bos>=1, <eos>=2, <pad>=3`，训练时预留、导出的 `.model`/`.json` 里也会带上。
3. **`scores` 是 log 概率（负数）**，Viterbi 最大化它的和。
4. **`trie` 是"派生缓存"**：加载模型后由 `tokenizer_build_trie()` 构建（见 `src/tokenizer.c` 的 `tokenizer_load()` / `tokenizer_load_from_memory()`）。你手工 `tokenizer_set_piece()` 之后必须重建 Trie，否则新 piece 不会被匹配到。
5. 加载完成后 `encode/decode` 是**并发安全**的（头文件注释写明）。

**最小示例：打印词表前几条（C）**

```c
#include "tokenizer.h"
#include <stdio.h>
int main(void) {
    tokenizer_t *tok = tokenizer_load("model_big2.tkm");
    if (!tok) { printf("load failed: %s\n", tokenizer_last_error()); return 1; }
    printf("vocab_size = %zu\n", tokenizer_vocab_size(tok));
    for (int i = 0; i < 5; i++) {
        const char *p = NULL; size_t n = 0;
        tokenizer_id_to_piece(tok, i, &p, &n);
        printf("id=%d piece=[%.*s] len=%zu\n", i, (int)n, p, n);
    }
    tokenizer_free(tok);
    return 0;
}
```

编译运行：

```powershell
cd E:\tokenizer_c
gcc -std=c11 -O2 -Iinclude -Isrc src\*.c demo.c -o build\demo.exe -lm
build\demo.exe
```

### 3.2 TKCZ：`.tkm` 模型二进制格式

权威定义在 `src/model.c` 顶部注释与 `src/model.c` 的 `tokenizer_save_binary()` / `tokenizer_load_binary_mem()`；`src/tokenizer_internal.h` 也有摘要。

| 偏移 | 类型 | 内容 |
|---|---|---|
| 0 | 4 字节 | 魔数 `"TKCZ"` |
| 4 | u32 | 版本号（当前 `= 1`） |
| 8 | u32 | `vocab_size` |
| 12 | i32 ×4 | `unk, bos, eos, pad` 四个特殊 id |
| 28 | u32 | flags：bit0 = 启用 byte_fallback，bit1 = fold_case |
| 32 | 重复 `vocab_size` 次 | 每条：`u32 len` + `len` 字节 piece + `f32 score` |

全部**小端序（little-endian）**。

**初学者可以直接用 Python 验证**（不用编译 C，很适合第一次接触）：

```python
import struct
data = open(r"E:\tokenizer_c\model_big2.tkm", "rb").read()
assert data[:4] == b"TKCZ", "不是 TKCZ 模型"
ver, vocab = struct.unpack_from("<II", data, 4)
unk, bos, eos, pad = struct.unpack_from("<iiii", data, 12)
flags, = struct.unpack_from("<I", data, 28)
print(f"version={ver} vocab={vocab} unk={unk} bos={bos} eos={eos} pad={pad} "
      f"byte_fallback={flags & 1} fold_case={(flags >> 1) & 1}")
off = 32
for i in range(5):                      # 只看前 5 条
    (n,) = struct.unpack_from("<I", data, off); off += 4
    piece = data[off:off+n]; off += n
    (score,) = struct.unpack_from("<f", data, off); off += 4
    print(i, piece, score)
```

评测脚本里也用同样的思路：`benchmark\eval_metrics.py` 的 `parse_tkcz_vocab()` 就是解析这个头部拿 `vocab_size`。

**常见疑问 / 易错点**

- Q：为什么不用 protobuf/JSON 做首选格式？
  A：`.tkm` 结构极简、加载快、零依赖；跨生态的兼容交给 `.model` / `.json` 两种导出格式。
- ⚠️ 易错点：`f32` 是 IEEE-754 单精度，**同样的分数在 Python 里读出来和 C 里可能有最后一位差异**，比较时不要用 `==` 比浮点。
- ⚠️ 易错点：解析时 `piece` 长度是**字节数**，不是字符数。"机"在 UTF-8 里是 3 字节。
- ⚠️ 易错点：`flags` 的 bit1 是 `fold_case`，早期版本没有这一位；若你手工造 TKCZ 文件，务必写对 flags，否则编码行为会变。

### 3.3 ckpt：断点续训文件

位置：`src/trainer.c` 约第 424-481 行，`ckpt_save()` / `ckpt_load()` / `ckpt_exists()`。

```
magic "TKCH" + version(u32)
vocab_target(u64) max_piece_len(i32) max_candidates(i32)
max_iters(i32) em_eps(f64) shrink_pct(i32) min_rounds(i32)
add_fb(i32) fold_case(i32)
vsz(i32) round(i32) n_rounds_at_target(i32) prev_ll(f64)
然后每个 piece：plen(i32) + bytes + score(float)
```

设计要点（也是初学者最容易踩的坑）：

1. **ckpt 里存了训练超参并做一致性校验**。用 `--vocab-size 262144` 训到第 5 轮中断，如果下一轮你改成 `--vocab-size 30000` 续训，会被拒绝（参数不匹配）。这避免了"续训出一个四不像模型"。
2. **存在 ckpt 就自动续训、跳过种子阶段**。这是 `tokenizer_train()` 里的逻辑：`ckpt_exists()` 命中则从上次轮次继续。
3. **每轮 EM 结束写一次**（先写 `.tmp` 再改名，保证原子性）。
4. ⚠️ **严禁把别的训练的 ckpt 改名混用**。`train_big2.py` 里专门打印了警告：`model_big.ckpt` 的语料分布与 `model_big2` 不同，改名混用会得到错误的续训结果（tok-cli 原生不支持跨语料 resume）。

**跑一个"故意中断再续训"的实验**（理解 ckpt 最快的方式）：

```powershell
cd $env:TEMP
# 第一次：只跑 2 轮
E:\tokenizer_c\build\tok-cli.exe train --corpus mini.txt --model m.tkm --vocab-size 300 --iters 2 --ckpt m.ckpt -v
# 第二次：改成 6 轮，观察它是否跳过 seeding 直接进 EM round 3
E:\tokenizer_c\build\tok-cli.exe train --corpus mini.txt --model m.tkm --vocab-size 300 --iters 6 --ckpt m.ckpt -v
```

---

## 4. Unigram 训练流程（`src/trainer.c`，核心章节）

文件头注释已经把四步写清楚了：

```
E step : Viterbi optimal segmentation per sequence, accumulate counts
M step : re-estimate log-probabilities
prune  : keep the top-k pieces by score (k shrinks towards the target)
stop   : when the log-likelihood gain stays below em_eps for min_rounds rounds
```

### 4.1 语料加载

- `read_line()`（第 49 行）：逐行读，**单行上限 `TRAIN_MAX_LINE = 4096` 字节**（第 25 行，对应设计文档 §5.4）。
- `corpus_load()`（第 72 行）/ `corpus_free()`（第 132 行）：一次性把整个语料读进内存的 `corpus_t`。

⚠️ 关键认知：**训练语料是"一行 = 一个序列"**。所以语料要按"句子/文档"分行；全塞成一行不但会触发 4096 截断，还会让 EM 的统计失真。

⚠️ 另一个易错点：`read_line` 是**按字节**读的，不做编码转换。语料必须是 UTF-8，GBK 文件不会报错但会训出乱码 piece。

### 4.2 种子构建（seeding）

从零开始的词表不是空的，而是先统计出现过的 n-gram 作为"候选池"：

| 环节 | 函数（`src/trainer.c`） | 说明 |
|---|---|---|
| 字符串哈希 | `fnv1a()`（第 157 行） | FNV-1a 64 位哈希 |
| 哈希表 | `hash_new / hash_grow / hash_get_or_create`（166 / 193 / 211 行） | 计数 n-gram 出现次数，受 `max_candidates`（默认 1,000,000）限制 |
| 种子容器 | `seed_t`、`seed_push()`（第 278 行） | 收集候选 piece 与计数 |
| 排序 | `seed_sort_item_cmp()`（第 322 行） | 计数降序；**平局时更长的 piece 优先**（第 28 行注释） |
| 特殊/兜底识别 | `is_special_id()`（333 行）、`is_bytefb_style()`（337 行） | 区分特殊 token 与 `<0xXX>` 形态的 piece |

种子阶段结束时会追加 **256 个 byte-fallback piece**（`<0x00>`…`<0xFF>`），日志里体现为 `byte-fb: 256`。

**真实训练日志**（本机 5 行迷你语料实测）：

```
[train] corpus lines: 5
[train] seeding...
[train] seed line 0 / 5
[train] seed distinct n-grams: 361 (count total 523)
[train] seed vocab: 621 (byte-fb: 256)
[train] EM start vsz=621 target=300
```

读法：语料里出现 361 个不同的 n-gram（总出现 523 次），加上 256 个字节兜底 piece 共 621 条种子，目标是压缩到 300。

⚠️ 易错点：`--max-piece-len`（默认 8 字节）是**种子阶段**的 n-gram 长度上限，直接决定候选池大小与训练耗时；调大它训练会显著变慢（`train_big2.py` 就注释了 15GB 语料的耗时量级）。

### 4.3 EM 迭代：E 步与 M 步

主循环在 `tokenizer_train()`（`src/trainer.c` 第 639 行起），EM 主体在第 883-1050 行。

- **重建 Trie**：每轮开始按当前词表重建（剪枝会删 piece，Trie 必须同步更新）。
- **E 步**：对语料每一行调用 `viterbi_segment()`（第 358 行）取得最优切分，累计每个 piece 的**使用次数**，并把该行的最优 `ll`（log 似然）累加到全局 `ll`（第 883-891 行）。
  - `viterbi_segment()` 与推理期的 `tokenizer_encode()` 是**同一套 Viterbi**，区别只是它用 Trie 输出（`seg_t`）供统计用。
- **M 步**：把每轮累计的计数归一化成概率 `p_i = count_i / total`，再取 `log` 写回 `scores[]`。

⚠️ 关键理解：E 步是"给定当前概率，把语料按最优方式切开数一遍"；M 步是"按数出来的频次重新定概率"。反复迭代，模型会自己找到"哪些片段值得成为 token"。

**真实日志**：

```
[train] round 1: vocab=372 ll=-228.8982
[train] round 2: vocab=300 ll=-134.0379
[train] round 3: vocab=300 ll=-134.0379
```

`ll` 是"越大越好"的（都是负数，越接近 0 越好）；第 3 轮开始不再变化，说明在当前 settings 下已收敛。

### 4.4 剪枝（pruning）：把词表压到目标大小

这是 Unigram 训练里**最容易被忽略但最关键**的一步（第 922-1035 行）。种子池（621）远大于目标（300），必须删掉那些"又低分又少用"的 piece。

实现要点：

1. **按类别分桶**（第 931-957 行）：

   ```
   buckets[0] = special   （<unk>/<bos>/<eos>/<pad>，永不删）
   buckets[1] = normal    （普通 piece，参与排序剪枝）
   buckets[2] = bytefb    （<0xXX>，永不删，保证 OOV=0）
   ```

   ⚠️ 这就是**为什么剪枝不能简单"按分数砍尾部"**：字节兜底 piece 分数是固定的很低值（`kByteFallbackScore = -10.0f`），如果按分数全局排序，它们会被第一批删光，直接破坏"任意文本都能编码"的保证。

2. **普通 piece 排序**：用 `pruned_pair_t`（第 28-33 行注释：**分数降序，平局时更长的 piece 优先**），这里用的是插入排序（小规模数据上够用且稳定）。

3. **决定保留多少**（第 989-992 行）：

   ```c
   if (shrink_pct > 0 && vsz > (int)vocab_target) {
       int by_shrink = (int)((size_t)vsz * (size_t)shrink_pct / 100);
       nkeep = by_shrink > by_target ? by_shrink : by_target;
   }
   ```

   即每轮最多砍到原来的 `shrink_pct`%（默认 60%），但**不低于最终目标**。这是"渐进收缩"：621 → 372 → 300，而不是一刀砍到 300。

4. **删除尾部 piece 并释放内存**（第 1013 行起），顺序重排后 id 会重新编号成连续。

⚠️ 易错点：**剪枝会改变 piece 的 id**。所以"边训边用某个 id 做实验"是不可靠的；要稳定的 id 就等训练结束拿最终 `.tkm`。

⚠️ 易错点：`--shrink 100` 等于关闭渐进收缩（一步到位），小语料上容易把好 piece 误删导致 `ll` 反而变差。

### 4.5 收敛判断

```c
double gain = prev_ll == -INFINITY ? 0.0 : ll - prev_ll;
if (gain < em_eps) n_rounds_at_target++;
if (n_rounds_at_target >= min_rounds) break;   /* 第 1045-1048 行 */
```

- `em_eps`（默认 `1e-4`）：进步小于这个值就算"没进步"。
- `min_rounds`（默认 3）：**连续**这么多轮没进步才停。
- `max_iters`（默认 10）：硬上限。

⚠️ 易错点：停训的判据是"**log 似然不再提升**"，不是"词表到目标大小了"。词表在第 2 轮就到 300，但训练还继续跑了 3 轮，这是正常的（在等收敛）。

### 4.6 落盘：训练产物

`tokenizer_train()` 结尾会把模型写成：

| 产物 | 写入函数 | 说明 |
|---|---|---|
| `model.tkm` | `tokenizer_save_binary()`（`src/model.c`） | 主产物 |
| `model.json` | `tokenizer_export_json()` | 传了 `--json` 才生成 |
| `model.model` | `tokenizer_export_sentencepiece()`（`src/sentencepiece.c`） | 传了 `--sp-model` 才生成 |
| `model.ckpt` | `ckpt_save()` | 传了 `--ckpt` 时每轮写 |

**训练参数默认值一览**（`include/tokenizer.h` 的 `TOKENIZER_TRAIN_OPTIONS_DEFAULT`，CLI 里 0 值会被替换成默认）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `vocab_size` | 4096 | 目标词表大小（262K 需要大语料） |
| `max_piece_len` | 8 | 种子 n-gram 长度上限（字节） |
| `max_candidates` | 1000000 | 种子候选哈希上限 |
| `max_iters` | 10 | 最大 EM 轮数 |
| `em_eps` | 1e-4 | 收敛阈值 |
| `shrink_pct` | 60 | 每轮保留百分比 |
| `min_rounds` | 3 | 达到目标后最少再跑几轮 |
| `add_byte_fallback` | 1（开） | 是否加 256 个 `<0xXX>` |
| `fold_case` | 0（关） | 是否 ASCII 大小写折叠 |

---

## 5. 归一化与 Byte Fallback

这一节的两个机制，一个是"编码前把文本变规整"，一个是"遇到不认识的字符也不能崩"。

### 5.1 基础归一化 `normalize_text()`

位置：`src/normalize.c`（声明在 `src/normalize.h`）。

```c
char *normalize_text(const char *text, size_t len,
                     int mark_spaces, int fold_case, size_t *out_len);
```

它做三件基础的事：

1. **空白归一化**：各种 Unicode 空白 → ASCII 空格。
2. **空格标记**：`mark_spaces = 1` 时，把普通空格 U+0020 换成 `▁`（U+2581）。这是 SentencePiece 的经典技巧——**让空格成为可以被打包的普通字符**，从而让"词首空格"能被吸收进 piece（如 `▁hello`），解码时再还原。
3. **大小写折叠**：`fold_case = 1` 时 ASCII 大写转小写（训练选项 `--fold-case` / TKCZ flags bit1）。

`tokenizer_encode()` 内部通过 `normalize_for_encode()`（`src/tokenizer.c` 约第 237-244 行）调用它：

```c
if (tok->norm_hook) { r = tok->norm_hook(...); if (r) return r; }   /* 自定义钩子优先 */
return normalize_text(text, len, /*mark_spaces=*/1, tok->fold_case, out_len);
```

### 5.2 完整 NFKC 归一化

对外暴露的 `tokenizer_normalize()`（`src/normalize.c` 第 139 行附近的 `nfkc_normalize_cps()` 是核心）实现了**真正的 NFKC**（Unicode UAX #15）：

- 剥离 C0/C1 控制字符（保留 `\n \r \t`）；
- 全角转半角（`ＡＢＣ１２３` → `ABC123`）；
- 带圈数字、连字（`ﬁ` → `fi`）、罗马数字、上下标（`²` → `2`）、兼容货币符号、全角标点、半角片假名等兼容字符映射；
- 剥离零宽字符与 BOM；
- 分解 → 规范排序 → 重新组合。

字符表是**代码生成**的，不是手写：

```powershell
python scripts\gen_nfkc_tables.py     # 生成 src\nfkc_tables.inc
```

生成三类静态表（`scripts/gen_nfkc_tables.py` 顶部注释）：`kDecompEntries/kDecompPool`（NFKD 分解）、`kNfkcCccEntries`（组合类）、`kNfkcComposeEntries`（可重新组合的对），均按序排好以便**二分查找**。

⚠️ 易错点：`tokenizer_normalize()` 不是 `normalize_text()`。前者是完整 NFKC 且**不带空格标记**（返回结果里空格还是空格），后者是内部基础归一化且默认带 `▁` 标记。学习/测试时混用会看到"空格变成了 ▁"或"没变成 ▁"的困惑。

⚠️ 易错点：`\n` 属于空白，会被归一化成空格（→ `▁`）。所以**逐行评测时不要带尾部换行**，否则会多出一个空格 token，导致往返比对误判——`benchmark\eval_metrics.py` 里 `encode_text()` 就是专门为此"不写尾部换行"。

### 5.3 Byte Fallback：绝不出现 OOV

设计目标写在 `include/tokenizer.h`：

> Guarantees: every input byte is represented (byte-fallback keeps OOV == 0).

机制（`src/tokenizer.c`）：

1. 词表里保留 256 个形态为 `<0xXX>` 的 piece（`is_byte_fallback_piece()` 第 268 行附近做形态判定：长度 6、`<0x` 前缀、`>` 结尾、中间两个十六进制位）。
2. 编码时，任何"词表里匹配不到的字节"都可以用一步"降级边"跳过，代价是 `kByteFallbackScore = -10.0f`（固定低分，让正常 piece 永远优先）。
3. **P1-2 改动（重要）**：每步都无条件加一条"1 字节边"（源码第 350-360 行的注释说明），而不是"只在不可达时补救"。早期实现会因为 DP 出现空洞而在回溯时解不出来。
4. **虚拟 id**：byte fallback **不占用真实 id**，而是编码成负数 `-1 - byte`。解码时展开回原始字节。

**实测验证**（迷你模型，词表里没有 'A' 'B' 'C'）：

```powershell
E:\tokenizer_c\build\tok-cli.exe encode --model mini.tkm --text "ＡＢＣ12"
# -66 -67 -68 -50 -51
```

对照：`'A'=65 → -1-65 = -66`，`'1'=49 → -50`。**全角 `ＡＢＣ` 先被 NFKC 归一化成 `ABC`，再走字节兜底。**

```powershell
E:\tokenizer_c\build\tok-cli.exe decode --model mini.tkm --ids "8,16,-109,-101"
# hello world
```

⚠️ 初学者必知的三个坑：

1. **负数是合法的**。看到 encode 结果里有 `< 0` 的 id 不要慌，那是 byte fallback 虚拟 id，`tokenizer_decode()` 认识它们；但**不要把它们存进你自己的 embedding 表**——真实 token 的 id 范围是 `[0, vocab_size)`，虚拟 id 只应存在于"编码结果 → 立刻解码"的流水线里。
2. **byte fallback 打开时 OOV 恒为 0**。如果你在评测里看到 `unk` 占比不为 0，先检查是不是用了 `--no-byte-fallback` 训的模型。
3. **BPT 会因此变低**。全走字节兜底意味着一个字被拆成 3 个 token（UTF-8 三字节）。byte fallback 是"保底不崩"，不是"高效表达"——高效表达要靠训练把常见词放进来。

---

## 6. CLI 的编译与使用（`cli/main.c`）

### 6.1 总览（真实 usage 输出）

```
Usage:
  tok-cli train --corpus <file> --model <out.tkm> [options]
    options: --vocab-size N --max-piece-len N --max-candidates N
             --iters N --eps F --shrink PCT --min-rounds N
             --no-byte-fallback --fold-case --json <out.json>
             --ckpt <file> --sp-model <out.model> -v
  tok-cli encode --model <file> (--text <text> | --file <in.txt>)
  tok-cli decode --model <file> --ids <id,id,...>
```

Windows 专属细节：`cli/main.c` 开头用 `WideCharToMultiByte(CP_UTF8, ...)` 把宽字符参数转成 UTF-8，所以**中文参数可以直接在 PowerShell 里传**（`--text "机器学习"` 可用）。

### 6.2 train：训练

```powershell
cd E:\tokenizer_c
build\tok-cli.exe train `
  --corpus corpus.txt `
  --model model.tkm `
  --vocab-size 32000 `
  --max-piece-len 8 `
  --iters 10 `
  --shrink 60 `
  --json model.json `
  --ckpt model.ckpt `
  --sp-model model.model `
  -v
```

| 选项 | 作用 |
|---|---|
| `--corpus` | 输入语料（UTF-8、每行一个序列） |
| `--model` | 输出 `.tkm` |
| `--vocab-size` | 目标词表大小（默认 4096） |
| `--max-piece-len` | 种子 n-gram 长度上限（默认 8） |
| `--max-candidates` | 种子候选上限（默认 1000000） |
| `--iters` | 最大 EM 轮数（默认 10） |
| `--eps` | 收敛阈值（默认 1e-4） |
| `--shrink` | 每轮保留百分比（默认 60） |
| `--min-rounds` | 收敛后最少再跑几轮（默认 3） |
| `--no-byte-fallback` | 关闭 256 个 `<0xXX>`（**不推荐**） |
| `--fold-case` | ASCII 大小写折叠 |
| `--json` / `--sp-model` / `--ckpt` | 额外导出 JSON / SentencePiece / 检查点 |
| `-v` | 打印进度日志到 **stderr** |

⚠️ 易错点：`-v` 的日志走 **stderr**，进度信息走 **stdout**。用重定向看日志时别搞混：

```powershell
build\tok-cli.exe train ... -v > out.txt 2> err.txt     # out=结果，err=进度
```

### 6.3 最小可跑通示例（5 行语料，实测）

```powershell
cd $env:TEMP
'hello world this is a unigram tokenizer test',
'the quick brown fox jumps over the lazy dog',
'机器学习与自然语言处理',
'hello 世界 你好 tokenizer 测试',
'ＡＢＣ１２３ fullwidth' | Set-Content mini.txt -Encoding UTF8

E:\tokenizer_c\build\tok-cli.exe train --corpus mini.txt --model mini.tkm `
    --vocab-size 300 --max-piece-len 6 --iters 6 --json mini.json -v
```

实测 stderr（进度）：

```
[train] corpus lines: 5
[train] seeding...
[train] seed distinct n-grams: 361 (count total 523)
[train] seed vocab: 621 (byte-fb: 256)
[train] EM start vsz=621 target=300
[train] round 1: vocab=372 ll=-228.8982
[train] round 2: vocab=300 ll=-134.0379
[train] round 3: vocab=300 ll=-134.0379
...
```

实测 stdout（结果）：

```
trained model saved to mini.tkm (vocab_size=300)
json export: mini.json
```

### 6.4 encode / decode

```powershell
# 编码一段文本（直接传文本）
build\tok-cli.exe encode --model mini.tkm --text "hello world"
# 8 16 -109 -101

# 编码一段文本（中文同样可用）
build\tok-cli.exe encode --model mini.tkm --text "机器学习"
# 36 22 -229 -186 -161

# 编码整个文件
build\tok-cli.exe encode --model mini.tkm --file one.txt
# 7 32 4 31 25

# 解码（把上面的 ids 还原回文本）
build\tok-cli.exe decode --model mini.tkm --ids "8,16,-109,-101"
# hello world
```

⚠️ 易错点：**不要用 `cmd /c` 加多层转义来传中文/带空格参数**，引号会被吃掉，报 `unknown encode option: hello`。直接用 PowerShell 调用即可。

⚠️ 易错点：`--ids` 是**逗号分隔**的（`8,16,-109,-101`），不是空格分隔。

### 6.5 用已有大模型直接推理

```powershell
E:\tokenizer_c\build\tok-cli.exe encode --model E:\tokenizer_c\model_big2.tkm --text "今天天气不错"
```

`model_big2.tkm`（vocab 262,144，2026-10-13 训练）是当前最新模型；同目录还有 `.json`（HF 格式）与 `.ckpt`（续训用）。

---

## 7. 测试与评测

### 7.1 单元测试：`tests\test_tokenizer.c`

```powershell
cd E:\tokenizer_c
build\test_tokenizer.exe
# OK: all N checks passed
```

它是一条**端到端流水线**：自己写一份小语料 → 训练（`vocab_size=300`）→ 加载校验 → 编码解码 → 清理临时文件。共 11 组检查，初学者可以按组读源码：

| 组 | 检查内容 | 学到的知识点 |
|---|---|---|
| 1-3 | 写语料、训练、加载元数据 | 训练入口与 `vocab_size` 校验 |
| 4-6 | encode/decode 往返、byte fallback、piece 查表、JSON 导出字段 | 往返必须逐字节相等 |
| 7 | `tokenizer_load_from_memory()` 内存加载 | 模型可以吃内存镜像 |
| 8 | 非法 id 解码必须被拒绝 | `out-of-range id rejected` |
| 9 | SentencePiece `.model` 导出 + 加载 + **与原模型逐条 id 比对编码结果一致** | 跨格式一致性 |
| 10 | `.json` 字段检查（`"type": "Unigram"`、`byte_fallback: true`、`normalizer: NFKC`…） | 导出契约 |
| 11 | 完整 NFKC 映射用例（全角、带圈数字、连字、罗马数字、上下标、半角片假名、`㎏`→`kg`…） | 归一化边界 |

⚠️ 易错点：第 11 组里 `"hello　世界\n"` 期望归一化结果是 `"hello 世界 "`（**末尾保留一个空格**），因为 `\n` 是空白 → 空格 → `▁`。新手常误以为"换行应该被删掉"。

### 7.2 评测脚本：`benchmark\eval_metrics.py`

```powershell
# 前置：build\tok-cli.exe 存在；benchmark\corpus\{en,zh,code}\*.txt 已准备
python benchmark\eval_metrics.py `
  --model E:\tokenizer_c\model_big2.tkm `
  --corpus-dir E:\tokenizer_c\benchmark\corpus `
  --cli E:\tokenizer_c\build\tok-cli.exe `
  --output E:\tokenizer_c\benchmark\eval_report_2026-10-14.json
```

设计原则（脚本 docstring 明确写了）：**所有指标都通过真实调用 CLI（`tok-cli.exe`）测量，CLI 返回非零退出码即报错终止**——不模拟、不估算。

指标分四类，含义如下：

**① 编码效率**

| 指标 | 含义 | 怎么读 |
|---|---|---|
| `bytes` / `tokens` | 语料字节数 / 切出的 token 数 | 基础计数 |
| **BPT** = bytes / tokens | 每个 token 平均承载多少**字节** | **越大越好**（压缩越强） |
| `chars_per_token` | 每个 token 平均承载多少字符 | 中文场景更直观 |

**② 还原率（round-trip）**：每语种抽前 200 行，逐行 `encode → decode` 与原文**逐字节比对**。任何一行不等就是 bug（归一化是"有损"的前提是它自己可逆）。

**③ 稳定性**：每语种抽 100 句，做 3 种扰动后与原 token 序列算 **Jaccard 相似度**：

1. 首字母大小写互换；
2. 随机插入拼写错误；
3. 换行符变化（行内有 `\n` 就换成 `\t`，否则把第一个空格换成 `\n`）。

相似度高说明分词结果对输入噪声不敏感（下游模型更稳）。

**④ 吞吐与加载**：最大语种文件整体 encode 计时得 `tokens/sec`；CLI 启动 + 模型加载时间作为近似 `load_ms`。

**⑤ OOV / 字节回退**：仅当模型是 `.model` 且环境装了 `sentencepiece` 时统计 `unk` / `<0xXX>` 占比，否则该项输出 `null` 并继续（不报错）。

**⑥ TokEval 指标**（来源：TokEval, arXiv 2608.18062）：

- §3.4 编码保真：`utf8_completeness`（UTF-8 字节序列是否被完整表达）、`char_split_rate`（一个字符被拆开的比例）、`boundary_crossing_rate`（token 边界是否跨越字符边界）。
- §3.5 数字对齐：`digit_boundary_f1`、`digit_split_variability`（同一数字串在不同上下文被切得是否一致）、`operator_isolation`（运算符是否被单独切出）。

**⑦ 协议鲁棒性**（可选，**必须显式传 `--protocol-slices-dir` 才启用**，不传则行为与旧版完全一致）：

- **双口径比值** `D_lang = BPT_eval / BPT_real`（en/zh/code 各一个）——衡量"用评测集口径算出的效率"与"用真实语料口径算出的效率"差多少；
- **模型内离散度** `dispersion = max(D) - min(D)`（外加三语种标准差）；
- **评测集质量审计** `protocol_audit`（逐文件的重复行率、总字符数、不同字符数、不同汉字数、行数）；
- 判定阈值：`≤ 0.35` 稳定 / `0.35–1.0` 敏感 / `> 1.0` 口径依赖。

**当前基线（`model_big2`）**：`D = (1.0065, 1.0465, 1.2134)`，`dispersion = 0.2069` → 落在"稳定"区间。语料分布为 en 56.42% / zh 19.99% / code 23.59%（按字节）。

⚠️ 易错点：`--corpus-dir` 要有 `en/zh/code` 子目录（脚本按 `corpus/<lang>/*.txt` 收集），直接指到一个扁平目录会取不到文件。可以用 `scripts\split_corpus.py` 从大语料切出评测子集。

⚠️ 易错点：`dispersion` 不是"越低越好"的无脑指标，它衡量的是**评测口径与真实口径的一致性**；数值大时应该先怀疑评测集本身有偏（脚本会顺带输出 `protocol_audit` 帮你查）。

### 7.3 训练性能观测：`benchmark\run_bench.ps1`

```powershell
powershell -ExecutionPolicy Bypass -File benchmark\run_bench.ps1 `
  -Exe E:\tokenizer_c\build\tok-cli.exe -Corpus corpus.txt -Model out.tkm `
  -Vocab 262144 -ExtraArgs "-v --max-candidates 1000000 --max-piece-len 24"
```

它轮询被重定向的 stderr 文件（~150ms 精度）产出**各阶段耗时时间线**，同时采样**峰值工作集/私有内存（MB）**，最后打印：

```
WALL_S=...      # 总墙钟时间（秒）
PEAK_WS_MB=...  # 峰值工作集
PEAK_PRIV_MB=...  # 峰值私有内存
EXIT=0
```

学这个脚本能顺带学到"C 进程性能观测"的基本手法。

---

## 8. 初学者常见疑问与易错点速查表

| # | 问题 / 现象 | 原因与对策 |
|---|---|---|
| 1 | encode 结果里有负数 | 正常。byte fallback 虚拟 id = `-1 - byte`；不要写进 embedding 表 |
| 2 | 中文 encode 出来 id 数特别多 | 可能是 byte fallback 在兜底（词表没覆盖）；检查 `--vocab-size` 与语料覆盖，观察 `unk`/`<0xXX>` 占比 |
| 3 | 训练日志里 `round 2: vocab=300` 之后还在跑 | 正常。停训看的是"log 似然不再提升"（`em_eps`+`min_rounds`），不是词表大小 |
| 4 | 续训被拒绝 | ckpt 里存了超参并做一致性校验；续训必须用**同样的** `--vocab-size` / `--max-piece-len` 等 |
| 5 | 把别的 ckpt 改名来续训 | **禁止**。语料分布不同会导致续训结果无意义（`train_big2.py` 里有明确警告） |
| 6 | 改了 piece 但 encode 匹配不到 | 手工 `tokenizer_set_piece()` 之后必须调 `tokenizer_build_trie()` |
| 7 | 逐行往返比对失败，多出一个空格 token | `\n` 被归一化成空格（`▁`）；单行 encode 时**不要带尾部换行** |
| 8 | `tokenizer_normalize()` 和 `normalize_text()` 行为不一致 | 前者是完整 NFKC、不带空格标记；后者是内部基础归一化、默认 `mark_spaces=1` |
| 9 | 手改 `nfkc_tables.inc` 后被覆盖 | 该文件由 `scripts\gen_nfkc_tables.py` 生成，改脚本再重新生成 |
| 10 | 词表 id 在训练中变化 | 剪枝会重排 id；要稳定 id 等训练结束用最终 `.tkm` |
| 11 | `cmd /c` 传中文参数报 `unknown encode option` | 引号被 cmd 吃掉；直接用 PowerShell 调用 |
| 12 | `--ids` 传空格分隔不生效 | 必须**逗号**分隔 |
| 13 | 训练语料全写一行 | 单行上限 `TRAIN_MAX_LINE = 4096` 字节，且 EM 统计会失真；一行一个序列 |
| 14 | 语料是 GBK | 不会报错但会训出乱码 piece；必须是 UTF-8 |
| 15 | 训 262K 词表却给 5 行语料 | 词表规模必须匹配语料量；小语料 + 大 vocab 只会得到一堆 byte fallback |
| 16 | `-v` 的日志看不到 | 日志走 stderr，`2> err.txt` 或 `2>&1` 才看得到 |
| 17 | 找不到 `libwinpthread-1.dll` | 把 DLL 放到 exe 同目录（`train_big2.py` 自动处理） |
| 18 | 评测脚本报"取不到文件" | `--corpus-dir` 必须是包含 `en/zh/code` 子目录的父目录 |
| 19 | 浮点分数比较不相等 | `.tkm` 里是 f32；跨语言读取有末位差异，比较用容差 |
| 20 | 想把 tokenizer 给 Python 用 | 直接导出 `.model`（`--sp-model`）给 `SentencePieceProcessor`，或用 `.json` |

---

## 9. 建议的学习路径

1. **跑通**：第 2 节编译 → 第 6.3 节 5 行语料训练 → 第 6.4 节 encode/decode。（半天）
2. **看懂数据**：第 3 节三个结构 + 用 Python 手撕 TKCZ 头部。（一天）
3. **理解算法**：按 `src/trainer.c` 的 `tokenizer_train()` 主线读（seed → EM → prune → 收敛），配合第 4 节的真实日志对照。（两到三天）
4. **动手改**：试着把 `--shrink` 改成 100、把 `--max-piece-len` 改成 4，对比 `ll` 与最终词表大小，理解超参影响。
5. **闭环评测**：用第 7 节跑一次 `eval_metrics.py`，看 BPT / dispersion 如何反哺训练决策。

配套阅读（同目录，均为真实文档）：

| 文档 | 什么时候看 |
|---|---|
| `README.md` | 权威使用手册：构建、CLI、API、已知限制（§12）、验收门槛（§14） |
| `2026-10-09-tokenizer-design.md` | 设计动机：为什么是 Unigram（§2）、分层架构（§4）、训练管线（§5）、模块职责（§4.1） |
| `2026-10-13-improvement-plan.md` | 当前优化方向 |
| `2026-10-14-tokenizer-optimization-plan-papers.md` | 带论文依据的六项改造方案（配额、种子、长度、byte fallback、归一化可逆、数字规则）与 D/dispersion 验收闸门 |

---

## 附：本材料中所有已实测命令清单

```powershell
# 构建
cmake -B build -S . ; cmake --build build --config Release
powershell -ExecutionPolicy Bypass -File scripts\build.ps1

# 训练（5 行迷你语料，300 词表）
build\tok-cli.exe train --corpus mini.txt --model mini.tkm --vocab-size 300 `
    --max-piece-len 6 --iters 6 --json mini.json -v

# 训练（导出 SentencePiece）
build\tok-cli.exe train --corpus mini.txt --model mini2.tkm --vocab-size 300 `
    --max-piece-len 6 --iters 4 --sp-model mini.model

# 编码 / 解码
build\tok-cli.exe encode --model mini.tkm --text "hello world"     # 8 16 -109 -101
build\tok-cli.exe encode --model mini.tkm --text "机器学习"          # 36 22 -229 -186 -161
build\tok-cli.exe encode --model mini.tkm --file one.txt           # 7 32 4 31 25
build\tok-cli.exe decode --model mini.tkm --ids "8,16,-109,-101"    # hello world

# 测试
build\test_tokenizer.exe

# 评测
python benchmark\eval_metrics.py --model model_big2.tkm `
    --corpus-dir benchmark\corpus --cli build\tok-cli.exe `
    --output benchmark\eval_report_2026-10-14.json
```
*（内容由AI生成，仅供参考）*
