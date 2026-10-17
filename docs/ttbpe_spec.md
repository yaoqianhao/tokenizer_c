# ttbpe_spec.md — tiktoken 兼容 byte-level BPE 模块规格

本文件是 `include/tiktoken_bpe.h` 与 `src/tiktoken_bpe.c` 共同引用的规格说明，描述该模块**按真实实现**所遵循的语义：
`src/tiktoken_bpe.c` 头注释引用的 "section 3" 即本文第 3 节，"section 5 of the spec" 即本文第 5 节，
`include/tiktoken_bpe.h` 中 `.tok.json` 格式的引用 "4.1" 即本文第 4.1 节。

## 1. 范围

| 项目 | 内容 |
| --- | --- |
| 目标 | 用纯 C11 实现 tiktoken 的 encoder/decoder 语义，用于与本工程 Unigram 分词器做同口径对比与互操作 |
| 公开头文件 | `include/tiktoken_bpe.h` |
| 实现文件 | `src/tiktoken_bpe.c`（自包含：不使用 JSON / 正则 / base64 第三方库） |
| 单元测试 | `tests/test_tiktoken_bpe.c`（97 项检查） |
| CLI 回归 | `tests/test_cli_tt_raw.c`（15 项检查，`tt-decode --raw` 字节级） |
| 风格 | 与 `tokenizer.h` 一致：不透明句柄、0/-1 返回码、输出由对应 `ttbpe_free_*` 释放、线程局部 `last_error` |

## 2. 公开 API（`include/tiktoken_bpe.h`）

| 分类 | 接口 |
| --- | --- |
| 生命周期 | `ttbpe_load` / `ttbpe_load_from_memory` / `ttbpe_free` |
| 元信息 | `ttbpe_vocab_size` / `ttbpe_format` / `ttbpe_word_split` / `ttbpe_special_count` / `ttbpe_last_error` / `ttbpe_id_bytes` |
| 配置 | `ttbpe_add_special` / `ttbpe_set_split_mode` / `ttbpe_set_cache_enabled` / `ttbpe_set_char_classifier` |
| 编码 | `ttbpe_encode` / `ttbpe_encode_ex` / `ttbpe_encode_ordinary` / `ttbpe_encode_single_piece` |
| 解码 | `ttbpe_decode`（UTF-8 文本，`errors="replace"`）/ `ttbpe_decode_bytes`（原始字节） |
| 预切分 / 调试 | `ttbpe_split`（零拷贝 piece 列表）/ `ttbpe_utf8_encode` |
| 内存 | `ttbpe_free_pieces` / `ttbpe_free_ids` / `ttbpe_free_buf` |

预切分模式常量：`TTBPE_SPLIT_NONE`(0) / `TTBPE_SPLIT_GPT2`(1) / `TTBPE_SPLIT_CL100K`(2)，加载后默认 `TTBPE_SPLIT_CL100K`。
字符类常量（供分类器钩子使用）：`TTBPE_CC_SPACE` / `TTBPE_CC_LETTER` / `TTBPE_CC_DIGIT` / `TTBPE_CC_OTHER`。

## 3. 编码 / 解码语义

本节的 8 条与 `src/tiktoken_bpe.c` 头注释中 "ttbpe_spec.md section 3" 逐条对应。

### 3.1 rank 即 id

合并过程始终选择"拼接后字节的 rank **最小**"的相邻对，重复该步骤直到没有任何已知对为止；**频次不参与任何决策**（区别于 Unigram/BPE 训练时的统计合并）。词表键为原始字节串、值为 rank，rank 同时就是该 token 的 id。

### 3.2 词表键是原始字节

词表键是原始字节串，允许**非法 UTF-8**（例如 `b"\xb7"`、`b" \xe4\xb8"`），不做任何文本层校验。

### 3.3 piece 独立编码

预切分产生的每个 piece 独立做字节对合并，**两个 piece 之间不跨边界合并**（对应 tiktoken 的 `_encode_single_piece` 逐片调用）。

### 3.4 special token 优先切分

special token 以**最长字面匹配**先从文本中切出。切出后：允许的 special 直接输出其 id；不允许的 special 由 `reject_disallowed` 决定——为真则报错（`ttbpe_encode` 的默认行为，等价 `allowed_special=set(), disallowed_special="all"`），为假则回落为普通文本编码。

### 3.5 解码：字节拼接 + `errors="replace"`

解码先按 id 顺序把各 token 的原始字节拼接，再做一次 UTF-8 解码：所有非法序列按 Unicode 的 **maximal subpart** 规则替换为 U+FFFD，即 tiktoken `errors="replace"`。需要原始字节时使用 `ttbpe_decode_bytes`（不做 UTF-8 解释）。

### 3.6 边界与错误

空输入返回 0 个 id 且成功；`tt` 为 NULL、id 越界等情况失败并设置 `last_error`。

### 3.7 词表缺前缀时不崩溃

若词表中缺少某个 piece 的某个前缀，编码器不会崩溃：能合并多少合并多少；当剩余部分在整个词表中没有任何 rank 时报错（tiktoken 在同样输入下会抛 `KeyError`）。

### 3.8 结果缓存

piece → ids 的缓存可关闭（`ttbpe_set_cache_enabled`）。开关只影响重复计算量，**不改变任何结果**。

## 4. 词表格式

### 4.1 `.tok.json`（xAI Grok 风格 JSON 词表）

顶层为 JSON 对象，识别以下键（**其余键一律跳过**，以保持前向兼容）：

| 键 | 类型 | 语义 |
| --- | --- | --- |
| `regular_tokens` | 数组 | 普通 token 列表 |
| `special_tokens` | 数组 | special token 列表，其中每条同时登记为 special token（参与第 3.4 节的最长字面匹配） |
| `vocab_size` | 整数 | 词表规模**声明值**，用于收尾校验（见下） |
| `word_split` | 字符串 | 元信息（如 `"V1"`），经 `ttbpe_word_split()` 暴露，不参与切分 |

数组元素为对象，必须**同时包含** `token`（整数 id）与 `bytes`（base64 原始字节）两个键，键序无关；`bytes` 不能为空：

```json
{
  "word_split": "V1",
  "vocab_size": 3,
  "regular_tokens": [
    { "token": 0, "bytes": "aGVsbG8=" },
    { "token": 1, "bytes": "IHdvcmxk" }
  ],
  "special_tokens": [
    { "token": 2, "bytes": "PHxFTkRPRlRURVhUPHw=" }
  ]
}
```

- **id 取自条目的 `token` 字段**，不是数组下标。
- 收尾时词表规模取 `max(max_id + 1, vocab_size 声明值)`；后续 `ttbpe_add_special()` 从该规模起分配新 id。
- special token 表按字节长度**降序**排序，保证最长优先匹配。
- 完全空的词表报错 `empty vocabulary`。

### 4.2 `.tiktoken`（每行一个 base64，行号 = id）

- 每行一个 base64 字符串，**行号（0 起）即该 token 的 id**。
- base64 为严格 RFC 4648 标准字母表：长度必须是 4 的倍数，只有最后一组允许 1~2 个 `=`，末组未使用低位必须为 0；不合法直接拒绝，避免静默错误解码（与 tiktoken 使用的严格 base64 引擎一致）。
- 该格式本身不携带 special token 声明，special token 需通过 `ttbpe_add_special()` 注册。

### 4.3 格式探测

先按内容判定（首个非空白字节），内容无法判定时再看扩展名；`.tok.json` → JSON 词表，`.tiktoken` → 每行 base64。`ttbpe_format()` 返回 `"tok.json"` 或 `"tiktoken"`。

## 5. 预切分（不使用正则引擎的近似实现）

按模式选择近似正则：`TTBPE_SPLIT_NONE`（整段一次编码）、`TTBPE_SPLIT_GPT2`（GPT-2 / r50k 风格）、`TTBPE_SPLIT_CL100K`（cl100k / o200k 风格，**默认**）。

字符类简化规则：

- **ASCII 精确**：空白 / 字母 / 数字 / 其它，四类判断与 Python 正则一致。
- **非 ASCII 简化**：少量 Unicode 空白码点算空白；少量 format / private-use / control 码点算 `other`；**其余所有非 ASCII 码点一律算字母**。因此 emoji、CJK 标点、非 ASCII 数字会被归入字母类，而不是符号/数字类。
- **非法 UTF-8 字节**归入 `other` 类，且逐字节推进，不会与后续字节拼成多字节序列参与分类。

由此得到的结论：**纯 ASCII 输入下 `tt-split` 与 Python `regex.findall` 的结果 100% 一致；非 ASCII 输入下存在可预期的差异**（差异范围即上述字符类简化，已在评测报告中验证）。

`ttbpe_set_char_classifier()` 可注入真正的 Unicode 分类器（返回 `TTBPE_CC_*`，返回负值表示回退到内置表）；接入后本文件其它逻辑无需改动。

## 6. CLI 子命令（`tok-cli`）

| 子命令 | 作用 | 主要选项 |
| --- | --- | --- |
| `tt-encode` | 编码为 id 序列 | `--vocab`；`--text \| --file`；`--split none\|gpt2\|cl100k` 或 `--split=<mode>`；`--ordinary`；`--allow-special`；`--special <token>`（可多次）；`--repeat <n>`；`--ids-only` |
| `tt-decode` | id 序列解码 | `--vocab`；`--ids "1,2,3"`（逗号分隔）；`--raw` |
| `tt-split` | 查看预切分结果 | `--vocab`；`--text \| --file`；`--split=<mode>` |
| `tt-info` | 查看词表元信息 | `--vocab` |

输出约定：

- `tt-encode`：stdout 为空格分隔的 id 加一个换行；除非 `--ids-only`，否则向 stderr 打印统计行（`vocab` / `format` / `split` / `specials` / `bytes` / `tokens` / `repeat` / `total_ms` / `ms_per_pass` / `mb_per_s`）。
- `tt-decode --raw`：**只输出 `ttbpe_decode_bytes()` 产生的原始字节**，不追加换行、不做文本模式换行转换（stdout 临时切到二进制模式），从而与 tiktoken `Encoding.decode_bytes()` 字节完全一致；不加 `--raw` 时输出 UTF-8 文本（非法序列替换为 U+FFFD）并追加一个换行。
- `tt-encode` 默认语义等价 tiktoken `Encoding.encode(text)`（`allowed_special=set(), disallowed_special="all"`）；`--ordinary` 等价 `encode_ordinary`；`--allow-special` 配合 `ttbpe_encode_ex(allow_all_special=1, reject_disallowed=1)`。

## 7. 回归测试

| 用例 | 覆盖点 |
| --- | --- |
| `tests/test_tiktoken_bpe.c`（97 项检查） | 词表加载（两种格式）、rank 即 id、字节级合并、special token 最长匹配与 `reject_disallowed` 两种行为、缺前缀错误、`decode` 的 `errors="replace"`、`decode_bytes` 原始字节、`split` 各模式、缓存开关等价性 |
| `tests/test_cli_tt_raw.c`（15 项检查） | CLI 层：`tt-decode --raw` 不追加换行（`0x0A` 单字节进出）、保留非法 UTF-8 字节、多字节 token 直接拼接、控制字节（含 NUL）穿过管道、与非 `--raw` 文本输出的对照 |
| `ctest` | 上述两个可执行测试 + `cli_tt_raw` 共 3 项，`ctest --test-dir build` 全绿 |

`tests/test_cli_tt_raw.c` 在运行期生成一份迷你 `.tiktoken` 词表（id 0..255 为单字节、256 为 `hello`、257 为 ` world`），因此不依赖仓库内的大词表产物。
