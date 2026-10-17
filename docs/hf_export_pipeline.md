---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_ada9909ec96411f18e0e52540064ee0f
    ReservedCode1: o2TLFwki5NL5gVtj0P8PLhKNDy1LPXuUA4koYXI0TXhASaC/D2j+gylLvhgdt4WZOekKxt2IHghsMUIETPA4CWxRg5+pTCeZUsOzU5nIlZZtnI5bymlMex6nqmffgVXf9gBRfiX7YIS5t/ErK5jmU8ufc7lcodq3GQNWSA8a7bojTrUrb7ug16j7vK8=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_ada9909ec96411f18e0e52540064ee0f
    ReservedCode2: o2TLFwki5NL5gVtj0P8PLhKNDy1LPXuUA4koYXI0TXhASaC/D2j+gylLvhgdt4WZOekKxt2IHghsMUIETPA4CWxRg5+pTCeZUsOzU5nIlZZtnI5bymlMex6nqmffgVXf9gBRfiX7YIS5t/ErK5jmU8ufc7lcodq3GQNWSA8a7bojTrUrb7ug16j7vK8=
---



# tokenizer_c → HuggingFace fast tokenizer 转换流水线

> 文档版本 v1.1 ｜ 落盘日期 2026-10-15 ｜ 最近更新 2026-10-16（默认归一化模式改 NFC） ｜ 适用仓库 `E:\tokenizer_c`
> 对应实现：[`tools\hf_export.py`](<E:\tokenizer_c\tools\hf_export.py>)（43,882 B，转换器/校验器）、[`tools\hf_pipeline.py`](<E:\tokenizer_c\tools\hf_pipeline.py>)（16,764 B，端到端编排器）
> 行为准绳：附件脚本 `patch_model_big2_to_hf.py` 的定稿行为 + `HF_fast_格式核验报告.md` 的 parity 口径

---

## 1. 这条流水线解决什么问题

`tok-cli` 训练产出的是**自研格式**的 `.tkm`（二进制模型）+ `model.json`（导出词表）。生态侧（`transformers` / `tokenizers` / Tokka-Bench / TokEval）要的是 **HuggingFace fast** 目录。二者之间存在四类结构性差异，直接喂 `model.json` 给 HF 必然加载失败或静默错切：

1. `model.json` 顶层带有 `unk_token` / `bos_token` 等**非 `tokenizer.json` 合法键**；
2. `decoder` 缺 `ByteFallback` 解码链，导致 `<0xXX>` 字节序列无法还原成文本；
3. `pre_tokenizer` / `normalizer` 缺失（HF 侧需要 Metaspace + 归一化链，默认 NFC）；
4. 词表里约 4.3% 的 piece 分数是 `null`（训练剪枝残留），以及 256 个 `<0xXX>` 字节 piece 缺失。

本流水线把这些差异**一次性、可复现、可自检**地抹平，并给出"转换后能不能真的替代 C 路径"的**逐 piece 判定**，而不是靠 token 数容差拍脑袋放行。

---

## 2. 流水线总览与数据流

### 2.1 四段式

```
        ┌────────────────────────────────────────────────────────────┐
        │ [1/4] build    （可选）scripts\build.ps1 → build\tok-cli.exe│
        └────────────────────────────────────────────────────────────┘
                                  │
        ┌────────────────────────────────────────────────────────────┐
        │ [2/4] train    （可选）tok-cli train --corpus C --model M   │
        │                     --json J [--vocab-size N --iters K ...] │
        └────────────────────────────────────────────────────────────┘
                                  │  J = model.json（Unigram：model.vocab = [[piece, score], ...]）
                                  ▼
        ┌────────────────────────────────────────────────────────────┐
        │ [3/4] export   体检（8 项）→ 7 处修复 → 写 4 个文件          │
        │   tokenizer.json / tokenizer_config.json /                 │
        │   special_tokens_map.json / tokenizer_c_norm.json          │
        └────────────────────────────────────────────────────────────┘
                                  │  out_dir（HF fast 目录）
                                  ▼
        ┌────────────────────────────────────────────────────────────┐
        │ [4/4] verify   HF 侧：Tokenizer.from_file / AutoTokenizer    │
        │                C  侧：tok-cli encode --file 逐句 encode      │
        │                → C-sim 反证 → diff_kind 分级 → verdict       │
        └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
              <workdir>\pipeline_manifest.json（全链条留痕）
```

- **纯转换**只需要 `[3/4]`：`hf_export.py --json J --out OUT`。
- **纯自检**只需要 `[4/4]`：`hf_pipeline.py verify ...`（或 `hf_export.py --verify`）。
- `[1/4]`、`[2/4]` 是**可选段**：已有 `model.json` 时完全跳过，不碰训练。

### 2.2 目录与路径约定

| 角色 | 缺省值 | 说明 |
|---|---|---|
| `--json` | `<workdir>\model.json` | `tok-cli train --json` 的产物，转换输入 |
| `--model` | `<workdir>\model.tkm`，不存在时回落到 `<json 同目录同名>.tkm` | C 侧 parity 比对用 |
| `--out-dir` | `<workdir>\hf_tokenizer` | HF fast 目录（3 个 json 落这里） |
| `--workdir` | `<repo>\build\hf_pipeline` | 训练产物、自检临时目录、manifest 都落这里 |
| `--tmp-dir` | `<workdir>\verify_tmp` | 自检时的临时句子文件（**不写系统临时目录**） |
| `--tok-cli` | `<repo>\build\tok-cli.exe` | C 侧可执行文件 |
| manifest | `<workdir>\pipeline_manifest.json` | 每段耗时/命令/退出码/完整校验明细 |

### 2.3 把路径写成绝对路径理解（示例）

```
repo          E:\tokenizer_c
tok-cli       E:\tokenizer_c\build\tok-cli.exe
workdir       E:\tokenizer_c\build\hf_pipeline
json / model  E:\tokenizer_c\build\hf_pipeline\model.json | ...\model.tkm
out-dir       E:\tokenizer_c\build\hf_pipeline\hf_tokenizer
manifest      E:\tokenizer_c\build\hf_pipeline\pipeline_manifest.json
```

### 2.4 三条可直接复制的命令

在仓库根目录 `E:\tokenizer_c` 下执行（用仓库自带解释器，避免依赖缺失）：

```bat
:: ① 小语料跑通全流程（分钟级，含训练/转换/自检）
E:\tokenizer_c\.venv\Scripts\python.exe tools\hf_pipeline.py all --corpus tools/smoke_corpus.txt --vocab-size 2000 --iters 2 --workdir build/hf_pipeline

:: ② 已有 model_big2.json，只做转换 + 自检（目标目录已存在时须加 --force，默认拒绝覆盖）
E:\tokenizer_c\.venv\Scripts\python.exe tools\hf_pipeline.py all --json model_big2.json --model model_big2.tkm --out-dir build/hf_pipeline_big2/hf_tokenizer_cli --force

:: ③ 只做自检（转换已完成，重跑校验，不写盘）
E:\tokenizer_c\.venv\Scripts\python.exe tools\hf_pipeline.py verify --json model_big2.json --model model_big2.tkm --out-dir build/hf_pipeline_big2/hf_tokenizer_cli
```

> 解释器建议固定用 `E:\tokenizer_c\.venv\Scripts\python.exe`。实测环境：Python 3.11.8 + `tokenizers` 0.23.3 + `transformers` 5.19.0。若用其它 Python，需自行确保 `tokenizers`、`transformers` 可导入（校验段才需要）。
> 运行时会打印 `[transformers] PyTorch was not found...` —— 属正常提示（纯 tokenizer 场景不需要 torch）。

---

## 3. `hf_export.py` —— 转换器与校验器

无子命令，`--json` / `--out` 必填。

### 3.1 参数表

| 参数 | 必填 | 默认 | 含义 |
|---|---|---|---|
| `--json` | ✔ | — | 输入 `model.json`（`tok-cli train --json` 产物）；不存在直接 `exit 1` |
| `--out` | ✔ | — | 输出目录，写 `tokenizer.json` 等 4 个文件（含归一化 sidecar） |
| `--model-type` | ✘ | 取 json 里的 `model.type` | 覆盖 `model.type`，仅允许 `Unigram` / `BPE` / `WordPiece` / `WordLevel`，非法值 `exit 1` |
| `--norm` | ✘ | `nfc` | 归一化模式 `nfkc` / `nfc` / `reversible`（兼容别名 `legacy` / `norm-reversible` 等）；决定 normalizer 链、`tokenizer_c_norm.json` sidecar 与 C 侧比对口径 |
| `--norm-reversible` | ✘ | 关 | 兼容别名：等价于 `--norm=reversible` |
| `--dry-run` | ✘ | 关 | **只体检不写盘**，打印问题项与修复项后退出（安全试跑） |
| `--verify` | ✘ | 关 | 转换后接着做「HF 加载 + 与 tok-cli 的 piece 级 parity」 |
| `--model` | ✘ | json 同目录同名 `.tkm` | 校验用的 `.tkm` |
| `--tok-cli` | ✘ | `<repo>\build\tok-cli.exe` | C 侧可执行文件路径 |
| `--verify-file` | ✘ | 内置 `DEFAULT_SENTENCES`（7 条） | 校验句子文件，每行一句 |
| `--verify-tolerance` | ✘ | `0.02`（2%） | token 用量相对差的**提示**阈值；**只提示、不参与判定** |
| `--strict` | ✘ | 关 | 严格模式：任何非逐 piece 一致即判失败（CI 用） |
| `--tmp-dir` | ✘ | 系统临时目录 | 校验临时句子文件目录 |

**退出码**：`0` 成功（verdict 以 `PASS` 开头）；`1` 参数/输入/转换错误；`2` HF 加载失败；`3` 校验不通过（`FAIL_TEXT` / `FAIL_MISMATCH` / `FAIL_STRICT`）。

### 3.2 体检项与 7 处修复

转换前会做一次结构体检（`diagnose()`），每项给出 `id` / `severity` / `detail` / `fix`：

| # | finding id | severity | 触发条件 | 修复动作 |
|---|---|---|---|---|
| 1 | `nonstandard_top_keys` | hard | 顶层出现 `ALLOWED_TOP_KEYS` 之外的键 | 剥离，改写进 `tokenizer_config.json` / `special_tokens_map.json` |
| 2 | `decoder_missing_bytefallback` | hard | `decoder` 链中无 `ByteFallback` | `decoder = Sequence[ByteFallback, Metaspace(prepend_scheme=never)]` |
| 3 | `decoder_deprecated_add_prefix_space` | hard | `decoder` 使用已废弃的 `add_prefix_space` | 改用 `prepend_scheme=never` |
| 4 | `pretokenizer_metaspace` | semantic | `pre_tokenizer` 缺失或不匹配 | `Metaspace(replacement="▁", prepend_scheme="never", split=false)` |
| 5 | `normalizer_sequence` | semantic | `normalizer` 与所选 `--norm` 期望链不一致（默认 `nfc` 期望 `Sequence[NFC, Replace(\s→" "), Replace(零宽族→"")]`；`nfkc` 期望 `Sequence[NFKC, ...]`） | 按所选模式的期望链重建 |
| 6 | `null_scores` | hard | 有 N 条 piece 分数为 `null`/`NaN` | 改写为极小值 `-1e9`（`NAN_SCORE_REPLACEMENT`） |
| 7 | `missing_byte_pieces` | semantic | 词表缺 `<0xXX>` 字节 piece | 补齐 256 个，分数 `-10.0`（对应 C 端 `kByteFallbackScore`，`src\tokenizer.c`） |
| — | `byte_fallback_without_pieces` | semantic | `byte_fallback=true` 但字节 piece 数为 0（与 #7 互斥，属边界兜底） | 同上补 256 个，使两侧字节回退对称 |

关键常量（与附件脚本 / C 端源码对齐）：

```
ALLOWED_TOP_KEYS      = version, truncation, padding, added_tokens,
                        normalizer, pre_tokenizer, post_processor, decoder, model
STRIPPED_TOKEN_KEYS   = unk_token, bos_token, eos_token, pad_token
DEFAULT_SPECIALS      = {unk: "<unk>", bos: "<s>", eos: "</s>", pad: "<pad>"}
NAN_SCORE_REPLACEMENT = -1e9
BYTE_FALLBACK_SCORE   = -10.0            # src\tokenizer.c 的 kByteFallbackScore
ZERO_WIDTH_CLASS      = [\u200b-\u200f\u2028-\u202f\u2060\ufeff\u00ad]   # src\normalize.c 的剔除清单
```

> `null` → `-1e9` 是**刻意不做真删除**：任务红线要求保持 C 路径行为不变（C 端把 `null` 视为不可用、走字节回退；若在词表里真删掉这些 piece，反而会改变 HF 侧以外的语义）。这一取舍的后果见 §7「已知残留差异」。

### 3.3 历史遗留 json 兼容

`model.json` 若是老格式（顶层直接是 `{"vocab": [[piece, score], ...]}`，没有 `model` 骨架），`_normalize_input()` 会自动补出 `model = {type: "Unigram", byte_fallback: true, unk_id: 0, vocab: ...}`，保证老产物也能转换，无需手工改文件。

### 3.4 归一化模式（`--norm`）与默认值变更（2026-10-16）

| 模式 | normalizer 链 | 性质 | 与 C 侧比对 |
|---|---|---|---|
| `nfc`（**当前默认**） | `Sequence[NFC, Replace(\s→" "), Replace(零宽族→"")]` | 不做兼容折叠：全角标点/字母保持原样 | `tok-cli encode --norm=nfc` |
| `nfkc`（兼容模式） | `Sequence[NFKC, Replace(\s→" "), Replace(零宽族→"")]` | 有损：全角→半角；与 2026-10-15 及更早产出逐字节一致 | `tok-cli encode`（老二进制内建 NFKC 默认） |
| `reversible` | `Sequence[NFC, Replace([\t\n\r]→" ")]` | 可逆口径：保留全角/非 ASCII 空白与零宽字符，encode→decode 无损 | `tok-cli encode --norm=reversible` |

变更要点（仅导出侧，C 侧算法语义零改动）：

1. `tools\hf_export.py` 的 `DEFAULT_NORM_MODE` 由 `nfkc` 改为 `nfc`，`tools\hf_pipeline.py` 透传同一常量；`nfkc` / `reversible` 仍可经 `--norm` 显式指定（旧别名 `legacy` / `norm-reversible` 继续兼容）。
2. C 侧比对参数只在「非 nfkc」时附加：`--norm=nfc` / `--norm=reversible` 会显式传给 `tok-cli encode`（判定用常量 `LEGACY_CLI_DEFAULT_NORM = "nfkc"` 保留老二进制内建默认的兼容语义，不再拿 `DEFAULT_NORM_MODE` 判断）。因此默认模式下请使用**支持 `--norm` 的 tok-cli**（`scripts\build.ps1` 的重建产物，如 `build_big3\tok-cli.exe`）；否则 verify 会降级为 `PASS_LOAD_ONLY`。
3. `write_outputs()` 落盘的 `tokenizer_c_norm.json` sidecar 记录 `norm_mode` / `normalizer_types` / 完整 normalizer 与 `source_json`，供下游评测核对口径。
4. 既有产物（`hf_export_big3`、`hf_export_big3_rev`）保留不删；新增 `hf_export_big3_nfc` 作为默认口径产物，见 §10.4。

---

## 4. `hf_pipeline.py` —— 端到端编排器

### 4.1 子命令

| 子命令 | 执行内容 | 典型用途 |
|---|---|---|
| `all` | 按需 `[build] → [train] → export → verify` | 一站式跑通；默认入口 |
| `export` | 只做 `json → HF fast 目录` | 已有 json，只要产物 |
| `verify` | 只做自检（HF 加载 + piece 级 parity） | 换句表重验、CI 复检 |

### 4.2 参数表

| 参数 | 适用 | 默认 | 含义 |
|---|---|---|---|
| `--corpus` | `all` | — | 训练语料；`all` 且未给 `--json` 时必填，缺失即 `exit 1` |
| `--json` | 全部 | `<workdir>\model.json` | 输入 json；`all` 下给了它就**自动跳过训练段** |
| `--model` | 全部 | `<workdir>\model.tkm`，不存在则回落 json 同名 `.tkm` | `.tkm` 路径 |
| `--out-dir` | 全部 | `<workdir>\hf_tokenizer` | HF fast 目录 |
| `--workdir` | 全部 | `<repo>\build\hf_pipeline` | 工作目录（自动创建） |
| `--build` | `all` | 关 | 先跑 `scripts\build.ps1` 重建 `tok-cli.exe` |
| `--train` | `all` | 关 | 即使给了 `--json` 也**强制重训** |
| `--vocab-size` | `all` | — | 透传 `tok-cli train --vocab-size` |
| `--iters` | `all` | — | 透传 `--iters` |
| `--max-piece-len` | `all` | — | 透传 `--max-piece-len` |
| `--max-candidates` | `all` | — | 透传 `--max-candidates` |
| `--min-rounds` | `all` | — | 透传 `--min-rounds` |
| `--tok-cli` | 全部 | `<repo>\build\tok-cli.exe` | C 侧可执行文件 |
| `--model-type` | `all`/`export` | 取 json 值 | 覆盖 `model.type` |
| `--norm` | `all`/`export`/`verify` | `nfc` | 归一化模式 `nfkc` / `nfc` / `reversible`（语义同 §3.4）；导出与自检共用同一模式 |
| `--norm-reversible` | 全部 | 关 | 兼容别名：等价于 `--norm=reversible` |
| `--verify-file` | `all`/`verify` | 内置 7 句探针 | 句表文件（每行一句） |
| `--verify-tolerance` | `all`/`verify` | `0.02` | 用量偏差**提示**阈值，不参与判定 |
| `--tmp-dir` | `all`/`verify` | `<workdir>\verify_tmp` | 自检临时目录（不写系统 temp） |
| `--strict` | `all`/`verify` | 关 | 严格模式，任何非逐 piece 一致即失败 |
| `--force` | `all`/`export` | 关 | 允许覆盖已存在的输出目录 |

### 4.3 训练段的两道防覆盖保护

`step_train()` 在调用 `tok-cli` **之前**检查：

- 语料不存在 → `[FAIL] 语料不存在` → `exit 1`；
- `--model` 或 `--json` 目标文件**已存在** → `[FAIL] 目标文件已存在（避免覆盖）` → `exit 1`。

`step_export()` 同样保护：输出目录已含 `tokenizer.json` 且未加 `--force` → 拒绝写入（`exit 1`）。**默认不允许静默覆盖**。

### 4.4 退出码

| 码 | 含义 |
|---|---|
| `0` | 全部成功 |
| `1` | 参数错误 / 前置条件不满足（语料缺失、要覆盖已有产物、C 侧不可用） |
| `2` | 转换或 HF 加载失败（也含自检抛异常 → `verdict=FAIL_EXCEPTION`） |
| `3` | 校验不通过（`FAIL_TEXT` / `FAIL_MISMATCH` / `FAIL_STRICT`） |
| `4` | 训练失败，或训练结束但**未生成 json**（现装 `tok-cli` 不支持 `--json` 的典型症状，见 §9） |
| `5` | 构建失败 |

---

## 5. 产出物清单

| 文件 | 位置 | 内容 | 小语料实测 | model_big2 实测 |
|---|---|---|---|---|
| `tokenizer.json` | `--out-dir` | **配置主体**：`version` / `added_tokens` / `normalizer` / `pre_tokenizer` / `post_processor` / `decoder` / `model`（含修复后的 vocab） | 53,742 B | 5,826,740 B |
| `tokenizer_config.json` | `--out-dir` | `tokenizer_class=PreTrainedTokenizerFast`、四个 special token、`clean_up_tokenization_spaces=false` | 184 B | 184 B |
| `special_tokens_map.json` | `--out-dir` | `unk_token=<unk>` / `bos_token=<s>` / `eos_token=</s>` / `pad_token=<pad>` | 95 B | 95 B |
| `tokenizer_c_norm.json` | `--out-dir` | 归一化 sidecar：`generator` / `norm_mode` / `normalizer_types` / `normalizer` / `source_json` / `generated_at` / `note`（**不被 tokenizers / transformers 读取**，供下游评测核对口径） | — | 见 §10.4 |
| `pipeline_manifest.json` | `<workdir>` | 编排器留痕：`command` / `started` / `finished` / `python` / `python_version` / `repo` / `tok_cli` / `workdir` / `model` / `json` / `out_dir` / `verify_tmp_dir` / `train` / `export` / `verify` / `exit_code` | 13,258 B | 6,355 B |

`tokenizer.json` 顶层**只有** `ALLOWED_TOP_KEYS` 这 9 个键；`unk_token` 等 4 个键被搬到后两个文件——这正是 `AutoTokenizer.from_pretrained()` 期望的布局。

---

## 6. 自检判定口径

### 6.1 校验怎么做的（四步）

1. **HF 侧加载**：`Tokenizer.from_file(out_dir\tokenizer.json)` + `AutoTokenizer.from_pretrained(out_dir)`，记录 `tokenizers` / `transformers` 版本与 `hf_class`。任一失败即 `FAIL_LOAD`（`exit 2`）。
2. **C 侧逐句 encode**：把句子写进 `--tmp-dir` 下的临时文件，调 `tok-cli encode --model M --file F`，取 id 序列。C 端**负 id** 按 `-1-byte` 规则还原成 `<0xXX>` 后，与 HF 侧 tokens **逐位对照**（口径对齐核验报告）。
3. **C-sim 反证**：复制转换结果，把源 json 中 `score == null` 的剪枝 piece **从词表移除**，得到一个"C 端可选 piece 集合"的模拟器。用它重新切同一句：
   - 若切分与 `tok-cli` **逐 piece 一致** → 说明该句差异完全由「剪枝 piece 在 HF 端仍可匹配（`-1e9` 是有限分数）、C 端视其不可用而走字节回退」这一**已核实机制**导致。
   - 若仍不一致 → 再看"还原文本是否一致"与"两条路径总分是否相等"。
4. **文本可逆性检查**：`_pieces_to_text()` 把 piece 序列还原（`<0xXX>` 按 UTF-8 字节累积解码、`▁` 视作空格），比较两侧文本。文本不一致 = 信息损失。

### 6.2 `diff_kind` 五级

| diff_kind | 判定依据 | 性质 |
|---|---|---|
| `identical` | 两侧 piece 序列完全一致（`piece_diff=0`） | 通过 |
| `explained` | C-sim 复现了 tok-cli 的切分（`csim_reproduces_c=True`） | 通过（已知机制） |
| `tie_break` | C-sim 未复现，但还原文本一致 **且** C 路径总分与 C-sim 路径总分相等（`|Δ| ≤ 1e-3`） | 通过（Viterbi 同分不同切分） |
| `lost_text` | 还原文本不一致（存在信息损失） | **硬失败** |
| `unexplained` | 其余情况：既非剪枝机制、也非同分 | **硬失败** |

判定顺序：`identical` → 若不一致先试 C-sim（`explained`）→ 再看文本可逆性（`lost_text`）→ 再看同分（`tie_break`）→ 兜底 `unexplained`。

### 6.3 `verdict` 判定优先级

```
C 侧不可用（缺 .tkm 或 tok-cli）      → PASS_LOAD_ONLY   （只证明能加载，不证明等价）
逐 piece 全一致（all_piece_match）    → PASS_IDENTICAL
存在 lost_text                        → FAIL_TEXT        （exit 2/3）
存在 unexplained                      → FAIL_MISMATCH    （exit 2/3）
--strict 且存在任何非 identical       → FAIL_STRICT
其余（差异全部 explained/tie_break）  → PASS_KNOWN_DIFF  （exit 0）
加载/导入异常                         → FAIL_LOAD / FAIL_IMPORT
```

> 注意 `PASS_LOAD_ONLY`：`hf_export.py` 单独用时它以 `0` 退出（verdict 以 PASS 开头），但**它没有证明与 C 路径等价**；`hf_pipeline.py` 更严——`c_cli_available=false` 时直接 `exit 1` 并要求补 `--model` 或先构建 `tok-cli`。

### 6.4 `tolerance` 与 `strict`

- `--verify-tolerance`（默认 `0.02`）：只用来在报告里标出"第 N 条样本 token 用量偏差超阈值"，**不参与通过/失败判定**。这是刻意的设计——剪枝 piece 机制会让 C/HF 在分布外文本上天然差 15%~31%，用数值容差判 FAIL 必然误报。
- `--strict`：CI 场景用。任何非 `identical`（含 `explained` / `tie_break`）即 `FAIL_STRICT`。

---

## 7. 已知残留差异

### 7.1 三类差异与实测证据

| 类别 | 机制 | 表现 | 实测样本 | 性质 |
|---|---|---|---|---|
| **A. 剪枝 piece** | 源 json 中 `score=null` 的 piece 在转换后被写成 `-1e9`：HF 视其为可匹配，C 端视其为不可用（走字节回退，负 id） | C 侧 token 数明显多于 HF（`c_negative_ids` 大），piece 序列不同，但**还原文本一致**，且 C-sim 逐 piece 复现 | 小语料 7 句探针：5 句 | 已解释（`explained`） |
| **B. Viterbi 同分** | 字节代价 `-10/字` 与某 piece 分数数值接近，两条路径总分相同，取的切分位置不同 | piece 序列不同、**还原文本一致**、两侧路径总分差 `≤1e-3` | 小语料 7 句探针：1 句（`emoji 😀🎉 与零宽...混合`） | 已解释（`tie_break`） |
| **C. normalize 策略差异** | C 端 `src\normalize.c` 在可逆模式下**保留全角标点与零宽字符**（仅 ASCII 空白折成空格、零宽不剔除）；HF 侧 `normalizer` 由导出时的 `--norm` 决定：`nfkc`（2026-10-16 之前的默认）做兼容折叠并**剔除零宽族**，`nfc`（**当前默认**）不做兼容折叠但同样剔除零宽族，`reversible` 则与 C 端可逆口径对齐 | 两侧还原文本不同 → `lost_text`；或文本相同但切分不同 → `unexplained` | 见 7.2 逐码点探针（nfkc × 可逆模式对照） | **部分已解释、部分未解释（见下）** |

### 7.2 normalize 差异的逐码点探针（model_big2）

探针句表：全角标点 / 空白族 / 零宽 / 软连字符 / 变体选择符 / 组合符 / 制表换行。

**一致的部分（7 句 `identical`，可以放心）**：

| 输入特征 | C 与 HF |
|---|---|
| 全角字母数字与全角标点 `ＡＢＣ１２３，。！（）【】`、`：，。；！？——` | 完全一致（10/10、19/19 token，`piece_diff=0`） |
| ASCII 空格（含 4 连空格）、制表符与换行 | 完全一致 |
| 零宽四兄弟 `U+200B / U+200C / U+200D / U+FEFF` | 完全一致（16/16） |
| 变体选择符、组合符（`é` 与 `e`+组合音标） | 完全一致 |
| `U+2028 / U+2029`（行/段分隔符） | 两侧一致（都当作换行→空格：`a▁b` + `▁c`） |

**已确认的差异（`lost_text`，属设计取舍而非 bug）**：

| 码点 | C 端（可逆模式） | HF 端（NFKC） | 后果 |
|---|---|---|---|
| `U+00AD` 软连字符 | 保留：`['s\xad','oft']`（2 token，文本 `s\xadoft`） | 剔除：`['soft']`（1 token，文本 `soft`） | 该句 token **-11.11%**（9→8） |
| `U+200E / U+200F / U+2060` | 保留并触发字节回退（7 token、5 token） | 整体剔除（2 token、1 token） | 该句 token **-45.0%**（20→11） |

> 探针总判 `FAIL_TEXT`：`pruned=11210 identical=7 explained=0 tie_break=0 unexplained=1 lost_text=2`。也就是说 **C 端可逆 normalize 与 HF NFKC 在"零宽族/不可见字符是否保留"上根本不同**，这类句子在 HF 侧是**有损**的。若业务语料含大量软连字符、方向标记、word-joiner，需在 C 端切到全 NFKC 模式对齐，或接受该损失。

**目前仍未解释的一类（`unexplained`）**：

| 输入 | C 路径 | HF 路径 | 现象 |
|---|---|---|---|
| 空白族混排 `U+3000 / U+00A0 / U+202F / U+2000 / U+2009` 同句（如 `：` + 5 种空白 + `X`） | `[':▁','▁▁','▁▁','<0x58>']`（7 token，总分 `-49.7451`） | `[':','▁▁','▁▁','▁X']`（7 token，总分 `-50.0064`） | **token 数相同、还原文本一致**（`:     X`），但 `▁` 前缀归属不同；C-sim **未复现**，路径总分不等（差 0.26），故判 `unexplained` |

该条是**当前唯一未解释项**：不是剪枝机制、不是同分，也不损失文本，属 C/HF 在"多种 Unicode 空白折叠后 `▁` 词边界标记归属"上的实现差异。单字符逐个测试时（`U+3000` 单独、ASCII 空格）两侧一致，只有在**多空白混排**时才出现，用户侧不影响语义（token 数与文本都一致）。

> **口径说明（2026-10-16 起）**：§7.1 / §7.2 的对照实验是在导出默认 `nfkc`（`model_big2`）下做的。默认改为 `nfc` 后，「全角→半角」这类 `lost_text` 不再产生；`U+00AD / U+200E / U+200F / U+2060` 等零宽族在 `nfc` 下仍被剔除（与 `nfkc` 同，因为 `nfc` 链同样含「零宽族→空」替换），只有显式 `--norm=reversible` 才与 C 端可逆口径完全对齐。探针回归的实测对照见 §10.4。

### 7.3 处置建议

1. **常规语料**：`PASS_IDENTICAL` 或 `PASS_KNOWN_DIFF` 都可放行（差异不损失文本）。
2. **含不可见字符/软连字符的语料**：先跑 `probe` 句表确认是否命中 `lost_text`；命中则考虑 C 端 normalize 切 NFKC 模式后再训练，或在上游清洗阶段剥掉这些字符。
3. **CI 门禁**：用 `--strict` 会因 `explained`/`tie_break` 直接失败——若想在 CI 里放行"已知差异"，**不要加 `--strict`**，改为断言 `verdict in {PASS_IDENTICAL, PASS_KNOWN_DIFF}` 且 `sentences_lost_text == 0 and sentences_unexplained == 0`。

---

## 8. 与 Tokka-Bench / TokEval 的衔接

**核心结论：把 `--out-dir` 整个目录直接喂过去即可，不需要 `config.json`、不需要 `model_type`。**

| 工具 | 加载方式（已核验） | 衔接要点 |
|---|---|---|
| **Tokka-Bench**（`E:\tokka-bench`） | `AutoTokenizer.from_pretrained(<hf_dir>)` | 目录里 `tokenizer.json` + `tokenizer_config.json` 已足够；产物 `data/results/{name}.json` 里的 `bytes_per_token` 与 tokenizer_c 的 **BPT 同向**，可直接对拍（注意分母口径一致） |
| **TokEval**（`E:\tokenizer-analysis-suite`） | `tokenizer_utils.py` 对目录优先找 `tokenizer.json`，走 `Tokenizer.from_file()` / `AutoTokenizer.from_pretrained()` | 同样不需要 `config.json` 的 `model_type`；`basic` 五指标由 normalization 配置控制（与 §7 的 normalize 差异**直接相关**，对比时务必对齐 C 端的 normalize 模式） |

注意事项：

- 喂**目录**而不是单个 `tokenizer.json` 文件，这样 special tokens 与 `tokenizer_class` 才生效。
- TokEval 默认每语言采样 2000 条（`--samples-per-lang`），输出 JSON 的指标名是平铺的、slim 后只保留 `mean/std/median/count`。
- 两侧工具跑出的差异若落在 §7 的 A/B/C 三类里，属于**已知差异**，不要当成 tokenizer_c 的问题。

---

## 9. `tok-cli` 二进制与源码（重建提示）

现装 `E:\tokenizer_c\build\tok-cli.exe`（265,433 B，构建于 2026-10-14 11:07:39）实测能力：

```
tok-cli train  --corpus <file> --model <out.tkm> [options]
   options: --vocab-size N --max-piece-len N --max-candidates N --iters N
            --eps F --shrink PCT --min-rounds N --no-byte-fallback --fold-case
            --json <out.json> --ckpt <file> --sp-model <out.model> -v
tok-cli encode --model <file> (--text <text> | --file <in.txt>)
tok-cli decode --model <file> --ids <id,id,...>
```

- 本流水线的训练段**依赖 `train --json`**，现装二进制**支持**：实测用 20 行探针语料 `train --vocab-size 300 --json ...` → `exit=0`，产出 `probe.tkm`(3,925 B) + `probe.json`(8,258 B)。故 §4.4 的 `exit 4`（未生成 json）不会在本机触发。
- **二进制与源码的关系**：`build\tok-cli.exe` 由 `src\*.c` + `cli\main.c` 经 `scripts\build.ps1` 构建；该 exe 的构建时间**早于** `src\normalize.c`、`src\nfc.c`、`cli\main.c` 的最后修改（2026-10-14 13:39 / 14:35 / 14:37）。也就是说**现装 exe 不包含这几个文件的最新改动**（未观察到新增子命令；`--help` 显示的子命令集合见上）。
- 为确认源码可构建，本次验证另在会话中间产物目录下用 gcc 重新编译过一份（`temp\rebuild\tok-cli.exe`、`temp\rebuild2\tok-cli.exe`，313,777 B，**仅有告警、无错误**：`src\trainer.c` 的 `-Wmaybe-uninitialized` 等），**未覆盖仓库内 `build\tok-cli.exe`**。
- **重建命令**（如需让 C 侧与最新源码一致）：

```bat
:: CMake 路线（默认）
cd /d E:\tokenizer_c
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1

:: gcc / MinGW 路线
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1 -Gcc
```

重建后请**重跑一次自检**（§2.4 命令②/③）——normalize 行为若有变动，§7 的差异结论需要重新标定。若不想覆盖仓库二进制，可把新 exe 放在别处，用 `--tok-cli <path>` 指定。

---

## 10. 实测记录（2026-10-15）

### 10.1 小语料全流程 self-test

```
E:\tokenizer_c\.venv\Scripts\python.exe tools\hf_pipeline.py all ^
    --corpus tools\smoke_corpus.txt --vocab-size 2000 --iters 2 --workdir <workdir>
```

| 项 | 值 |
|---|---|
| 语料 | `tools\smoke_corpus.txt`（22,649 B） |
| 训练 | `tok-cli train` rc=0，产出 `model.tkm` + `model.json` |
| 转换 | `pieces=2109`、`null_fixed=1705`、`byte_added=109`、`model_type=Unigram`、`findings_count=7` |
| 写入 | `tokenizer.json` 53,742 B / `tokenizer_config.json` 184 B / `special_tokens_map.json` 95 B |
| 加载 | `ok=true`（`tokenizers` 0.23.3 / `transformers` 5.19.0 / `hf_class=Tokenizer`） |
| 剪枝 piece 数 | 1,705 |
| **verdict** | **`PASS_KNOWN_DIFF`** |
| 分级计数 | **identical=1 / explained=5 / tie_break=1 / unexplained=0 / lost_text=0** |
| 最大用量偏差 | 30.7692%（`Hello, world!`：C=13 vs HF=9），提示阈值 2%，**不参与判定** |
| 退出码 | 0 |

逐句：探针 1（中文长句）`identical`；2/3/4/5/7（中文、`Hello, world!`、Python 代码、全角标点、混合制表）`explained`（`csim_reproduces_c=True`）；6（emoji+零宽）`tie_break`。**全部句子 `text_equal=True`**。

### 10.2 `model_big2` 真机 verify

```
E:\tokenizer_c\.venv\Scripts\python.exe tools\hf_pipeline.py verify ^
    --json model_big2.json --model model_big2.tkm ^
    --out-dir build\hf_pipeline_big2\hf_tokenizer_cli
```

| 项 | 值 |
|---|---|
| 模型 | `E:\tokenizer_c\model_big2.json`(7,294,078 B) + `model_big2.tkm`(3,689,538 B) |
| 转换产物 | `tokenizer.json` 5,826,740 B |
| 剪枝 piece 数 | 11,210 |
| **verdict** | **`PASS_IDENTICAL`** |
| 句级一致 | **7/7 全 `identical`**（`piece_diff=0`），`all_piece_match=True` |
| token 用量差 | **`worst_delta=0.0000%`**（7 句全部 C 与 HF token 数相等） |
| 加载 | `ok=true`（同环境） |
| 退出码 | 0 |

> 大模型之所以比小语料更干净：`model_big2` 词表覆盖面足够，剪枝 piece 不再需要被"用上"，C/HF 切分自然完全收敛。小语料的 `explained` 差异正是词表小 + 剪枝多导致的放大效应。

### 10.3 全语料聚合对照（286 行，`agg_summary.json`）

| 指标 | 值 |
|---|---|
| 语料行数 | 286 |
| C token 总数（逐行求和） | 8,216 |
| C-sim（剔除剪枝 piece）token 总数 | 8,214 |
| HF token 总数 | 5,799 |
| **C vs C-sim 偏差** | **-0.0243%**（285/286 行逐行完全一致） |
| C vs HF 偏差 | -29.4182%（59/286 行一致） |
| 词表 / 剪枝 piece | 2,109 / 1,705 |

这组数字是"剪枝 piece 机制足以解释 C/HF 差异"的**聚合级证据**：把剪枝 piece 从 HF 词表移除后，两侧 token 数几乎完全贴合（差 2 个 token / 8,216，即 0.0243%），剩余 1 行差异即 §7.3 的 `unexplained` 空白折叠项。

### 10.4 默认归一化模式改为 NFC 后的重新导出与验证（2026-10-16）

- **变更范围**：`tools\hf_export.py` 的 `DEFAULT_NORM_MODE` 由 `nfkc` 改为 `nfc`，`tools\hf_pipeline.py` 透传同一常量；`nfkc` / `reversible` 仍可经 `--norm` 显式指定（§3.4）。C 侧算法与 `src\` 零改动。
- **改动前备份**：`temp\backup\orig_20261016_nfc\`（`hf_export.py` / `hf_pipeline.py` / `hf_export_pipeline.md`）。
- **重新导出**：

  ```powershell
  python tools\hf_export.py --json model_big3.json --out hf_export_big3_nfc --verify --tok-cli build_big3\tok-cli.exe
  ```

  产物 `hf_export_big3_nfc\`：`tokenizer.json`（5,822,085 B）、`tokenizer_config.json`、`special_tokens_map.json`、`tokenizer_c_norm.json`（806 B）。既有 `hf_export_big3`（nfkc）、`hf_export_big3_rev`（reversible）**保留未删**。
- **verify 结论**：`PASS_IDENTICAL`（7/7 identical，`worst_delta = 0.0000%`，`normalizer match = True`，传给 tok-cli 的模式参数为 `--norm=nfc`），报告见 `build\hf_export_big3_nfc_verify.json`。
- **探针回归**（19 子类 / 76 样本，`benchmark\probe_regression.py --tokenizer hf_export_big3_nfc --norm-mode nfc`）：fidelity 0.9605（73/76 通过）、micro_fertility 2.4568、micro_byte 0.3385、gini 0.1284、composite 54.87；3 条失败全部落在 `whitespace_control`（tab 折空格，与 `reversible` 同因）；与 `nfc` 基线槽 diff 为「**劣化 0 项**」。
- **模式对照（同一语料、同一模型 `model_big3`）**：

  | 导出物 | 模式 | fidelity | composite | 说明 |
  |---|---|---|---|---|
  | `hf_export_big3_nfc`（**新默认**） | nfc | **0.9605**（73/76） | 54.87 | 无「全角→半角」兼容折叠损失 |
  | `hf_export_big3`（旧默认） | nfkc | 0.9211（70/76） | 54.03 | 兼容折叠导致额外 3 条失败 |
  | `hf_export_big3_rev` | reversible | 0.9605（73/76） | 54.87 | 与 nfc 实测一致 |

- **门禁现状**：`fairness_gini ≤ 0.11`（实测 0.1284）与 `composite_score ≥ 60`（实测 54.87）两项仍 FAIL，与 `nfkc` / `reversible` 基线同源，属已知未达标项，非本次变更引入。
- **评测侧同步**：`benchmark\thresholds.json` 的 `calibrated_by.probe_tokenizers` 已补入 `hf_export_big3_nfc`，并由 `benchmark\gen_thresholds_doc.py` 重新生成 `docs\eval_thresholds.md`（`--check` 通过）；`benchmark\probe_baseline.json` 新增 `nfc` 槽位，原有 `nfkc` / `reversible` 槽位保留。

---

## 11. 现场保护与已知瑕疵

- **现场保护**：`build\hf_pipeline_smoke` / `_e2e` / `_e2e_final` / `_final` / `_final2` / `_big2` 及仓库根下 `hf_export\` 均为历史试跑痕迹，**原样保留、不要清理**（便于回溯当时的 manifest 与产物）。
- **已知瑕疵（未改动，遵守"不改既有文件"约束）**：`hf_export.py` 中 `--verify-file` 的帮助文本写作"缺省用内置 6 句"，实际内置 `DEFAULT_SENTENCES` 为 **7 条**。行为以代码常量为准。
- **依赖提示**：校验段依赖 `tokenizers` + `transformers`（实测 0.23.3 / 5.19.0）；转换段除标准库外不依赖第三方包（C-sim 反证才 import `tokenizers`）。运行期 `PyTorch was not found` 提示可忽略。
- **环境踩坑提醒（复现时注意）**：外部 shell 工具对 stderr / 超长 stdout 存在截断，`$LASTEXITCODE` 可能取不到；排障时**优先看脚本自写的日志文件与 `pipeline_manifest.json`**，不要只依赖终端回显。
*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*
