---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_119938c2c7a711f1b5a6525400393706
    ReservedCode1: /zaz3mbT0qbNukto24QCF+SAsRa0Uat84n1FxEHa3LlGFgKOhyaIA8zMuJIhPgsvPJL+MGFaTLphTH+0v69B0HR596S7z4AWNojMsRDVqLk66Kif/5nwobnZwmvyyB0mGCpd4Far7T6Yrjp8+NWq7M2xhzM8Zuhg29TYapFAQmk/tCmTs+Ct9144TPo=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_119938c2c7a711f1b5a6525400393706
    ReservedCode2: /zaz3mbT0qbNukto24QCF+SAsRa0Uat84n1FxEHa3LlGFgKOhyaIA8zMuJIhPgsvPJL+MGFaTLphTH+0v69B0HR596S7z4AWNojMsRDVqLk66Kif/5nwobnZwmvyyB0mGCpd4Far7T6Yrjp8+NWq7M2xhzM8Zuhg29TYapFAQmk/tCmTs+Ct9144TPo=
---

# Tokenizer 评测标准与参考依据

- 版本：v1.0
- 日期：2026-10-14
- 适用对象：`tokenizer_c` 项目（自研中文分词器 / 词元化器）的评测评级、横向对比与验收
- 文件定位：本项目 tokenizer 评测的**统一口径与参照基准**。任何评测结论必须先符合本文口径，方可作为优化决策依据。

---

## 0. 三条铁律

1. **方向先确认**。不同平台同名指标方向可能相反：平台的 `compression` = tokens/char（越低越好），本项目的 `BPT` = bytes/token（**越高越好**），二者互为倒数。跨口径比较前必须显式换算，禁止混用。
2. **只在同语言内比较**。跨语言指标不可比——UTF-8 字节宽度（ASCII 1 字节、西里尔/阿拉伯 2–3 字节、CJK 3–4 字节）与"语言单位"粒度均不同。
3. **内在指标不能替代下游验证**。任何改动最终必须用 BPB / 下游任务证伪，且必须先过 L2 保真闸门。

---

## 1. 指标定义、方向与阈值

| 指标 | 定义 / 公式 | 方向 | 参考阈值 | 出处 |
|---|---|---|---|---|
| fertility（subword fertility） | tokens / word。**语言感知单位**：字母文字按空格切词，CJK / 泰文按字符，藏文按音节 | 越低越好 | ≈1.0 理想；**≥4 = poor** | TokenizerBench；Tokka-Bench |
| compression ratio | tokens / character | 越低越好 | 无硬阈值 | TokenizerBench |
| byte compression | tokens / UTF-8 byte | 越低越好 | 跨语种相对可比（仍以同语言内对比为最可靠） | TokenizerBench |
| **BPT（本项目口径）** | UTF-8 字节数 / token 数 = 1 / byte-compression | **越高越好** | 与厂商横向对比时同时给出实际值与缺口百分比 | 项目内部 |
| fidelity / roundtrip | `decode(encode(x)) == x`（strip 后严格一致） | 必须 100% | 任何 failure 即视为缺陷 | TokenizerBench |
| 词表利用率 vocabulary utilization | 语料中出现过至少一次的词表条目占比 | 适中 | 过低 = 词表浪费；过高 = 词表不足 | TokEval |
| n-gram 熵（bigram / trigram entropy） | 条件熵；低值 = 序列高度可预测 | 适中 | 过低提示**过切分**（未在该语言上训练过的典型症状） | TokEval |
| 平均 token rank | 按频次降序排名后，语料中所有 token 出现的平均排名 | 高值更健康 | 低值 = 少数高频 token 承担多数出现，词表利用差 | TokEval |
| MorphScore | token–词素边界 precision / recall / F1，覆盖 70 语言 | 越高越好 | — | Arnett & Bergen 2025；已集成进 TokEval |
| TFG（Tokenizer Fairness Gini） | 跨语言 token 成本不公平度 | 越低越好 | — | TokEval |
| word-split rate | 词边界被拆开的比例 | 越低越好 | — | Tokka-Bench |
| 字符拆分率 / 操作符隔离 / digit 处理 | 本项目自建保真指标 | 越低 / 越严格越好 | 见 §3.2 | 项目内部 |
| NSL / bits-per-byte (BPB) | 归一化序列长度 / 每字节比特数 | BPB 越低越好 | 仅用于同参数量模型对比 | 下游验证 |

> **口径换算备忘**
> - `BPT = 1 / byte_compression_ratio`（注意 byte 口径，不是 char 口径）
> - 平台 `compression`（tokens/char）仅用于同语言内相对排序，**不要**直接与本项目 BPT 拼接成同一张表。

---

## 2. 可用平台与参照系

### 2.1 在线平台（开箱即用，用于快速横向定位）

| 平台 | 入口 | 覆盖规模 | 输出指标 | 关键限制 |
|---|---|---|---|---|
| **TokenizerBench** | https://kitefishai-tokenizerbench.hf.space/ | 固定 82 条：8 语种 × 5 + 编程 15 + 科学公式 16 + 边界 11（19 子类） | fertility、compression、byte compression、fidelity；per-subcategory 明细表 + 三张图（fertility 热力图、语言 fertility 柱图、fertility vs byte-compression 散点） | **按空白切词**，CJK fertility 严重虚高；无预置榜单；无文件导出（仅 `Copy table data`） |
| **Tokka-Bench** | https://tokka-bench.streamlit.app/ | 100 自然语言（30+ 文字系统）+ 20 编程语言，每语言 2 MB | bytes per token、unique token coverage、subword fertility、word-split rate、vocabulary composition | 单语言单样本、无方差估计；仅覆盖 BPE 系 |
| **Tokenizer Arena**（Cognitive Lab） | Hugging Face Space | 并排对比 Gemma / Mistral / Grok / GPT-4 / Claude / Phi-3 / Command R 等 | 同文本切分与 token 数对比 | 面向 Indic 场景，无量化指标 |
| **The Tokenizer Playground**（Xenova） | https://huggingface.co/spaces/Xenova/the-tokenizer-playground | 多模型即时对比 | 切分可视化 | 无指标，仅直观查看 |
| **Dutch Tokenizer Arena**（yhavinga） | https://huggingface.co/spaces/yhavinga/dutch-tokenizer-arena | 荷兰语 | 多口径压缩率表（bytes/token、chars/token、tokens/bytes） | 语种单一，但**口径定义值得借鉴** |
| mC4 60 语内在评测榜 | sota2.com 榜单页 | 60 语言不均衡 mC4 | Compression Rate、TTR、词表利用率 | 无 fidelity 校验 |

**本项目优先顺序：Tokka-Bench（中文按字符、语言感知）> TokenizerBench（体检）> 可视化工具（排查用）。**

### 2.2 开源评测框架（可换自建语料，评价最充分）

| 框架 | 来源 | 能力 |
|---|---|---|
| **TokEval** | arXiv 2608.18062 / https://github.com/AntonioLopardo/tokenizer-analysis-suite | 多 tokenizer 同时对比；压缩率、fertility（词与字符双口径）、token 长度、bigram/trigram 熵、词表利用率、平均 token rank；集成 MorphScore 与 TFG。**核心论点：不能只按压缩率选 tokenizer** |
| **Tokka-Bench** | arXiv 2610.08794 / https://github.com/bgub/tokka-bench | 五指标 × 120 语言，语言感知分词，可替换语料重跑 |
| **Qtok** | arXiv 2410.12989 | 语言覆盖度、token 完整性、跨语言与语言类别分布偏差；论文实测 58 个模型 / 13 个 tokenizer |
| **MorphScore** | Arnett & Bergen 2025 | token–词素边界对齐 F1，70 语言 |

### 2.3 参照 tokenizer（横向对比固定这四个）

| Tokenizer | 词表规模 | 年份 | 备注 |
|---|---|---|---|
| GPT-2 | 50,257 | 2019 | 历史基线，各语言普遍落后 |
| gpt-oss | 200K | — | 现代基线 |
| Gemma 3 | 262K | — | 大词表 |
| Qwen3 | — | — | CJK 表现强 |
| Kimi K2 | 164K | — | 中文 unique token 42K（GPT-2 仅 6.5K），词表集中于 CJK |

**已实证的关键结论（Tokka-Bench）**：词表**分配策略**比词表**绝对大小**更重要——Gemma 3（262K）与 gpt-oss（200K）平均效率相近（4.30 vs 4.25 bytes/token），而 Kimi K2（164K）靠集中分配达 3.19；编程语言效率在 2023 后模型间已基本收敛（GPT-2 落后 29–51%）。

### 2.4 上传要求（接入所有在线平台的前置条件）

在线平台接受的是 **tokenizer 文件本身，不是语料文档**：

| 上传文件 | 类型 |
|---|---|
| `tokenizer.json` | 任意 HuggingFace fast tokenizer（BERT / RoBERTa / GPT-2 / LLaMA…），可另附 `tokenizer_config.json`、`vocab.txt`、`merges.txt` |
| `vocab.json + merges.txt` | GPT-2 风格 BPE |
| `*.model` | SentencePiece（T5 / mT5 / XLM-R / mBERT） |

上传失败报错原文：`ValueError: Unrecognised file(s): …. Expected: tokenizer.json, *.model, or vocab.json + merges.txt`

> **本项目待办**：自研 C 实现需新增导出子命令，产出 `vocab.json + merges.txt` 或 `tokenizer.json`，否则无法接入上述任何平台。

---

## 3. 评测执行标准（三层）

### 3.1 L1 — 内在指标（快速定位，不可单独定论）

- **语料必须双口径**：① 训练 corpus 抽样；② 真实域 held-out 切片。禁止只用模板合成集。
- **聚合必须按语种宏平均**，禁止用混语种总体均值下结论。
- 输出：per-language 的 fertility / compression / BPT 矩阵 + 词表利用率 + n-gram 熵。

### 3.2 L2 — 保真与鲁棒闸门（不过闸不得进入 L3）

| 检查项 | 验收标准 |
|---|---|
| roundtrip 还原率 | **100%**，不允许任何失败行 |
| 归一化可逆性 | 采用 NFC + 自定义全角映射；**禁止 NFKC**（会折叠全角标点且不可逆） |
| byte fallback 正确性 | 全部正确，不得出现替换符吞字 |
| 字符拆分率 | 逐位对比，改动前后不得劣化 |
| 操作符隔离 / digit 处理 | 达标（此处为 tokenizers 后端不失真指标，跨厂商可比） |

**量化验收阈值**：

- `D_lang = BPT_eval / BPT_real`，各语种须落在 **[0.85, 1.15]**
- `dispersion = max(D) − min(D)` 须 **≤ 0.35**

### 3.3 L3 — 下游验证（唯一可证伪的一层）

- 同参数量小模型各训一版，比较验证集 **BPB / NSL**。
- 成本口径统一用 **tokens per 1000 汉字** 与 **chars per token**。
- 内在指标与下游表现不必然相关，L3 是结论的必要条件。

---

## 4. 结果判读规则（防误判清单）

| 陷阱 | 典型现象 | 正确做法 |
|---|---|---|
| 空白切词对 CJK 失真 | xlm-roberta 中文 fertility 7.6、日文 10.4、long_tokens 32，而英文仅 1.6 | 用语言感知 / 字符级口径，别据此判词表差 |
| overall 均值被拉高 | 混合样本算术平均，被 CJK 与边界样本主导 | 必须拆 per-subcategory / per-language |
| 模板合成评测集伪影 | 抽取短语高频重复（如"的角度出发"出现 16,105 次）、重复行率 66.5%，字符分布偏离真实语料 | 换真实 held-out 集，按语种宏平均 |
| 跨语言比较 | bytes-per-token 混淆 UTF-8 编码开销与切分质量 | 只在同语言内比较 |
| 样本量过小 | TokenizerBench 固定 82 条；Tokka-Bench 每语言单 2 MB 块、无置信区间 | 只能定位瓶颈，不能定论 |
| 数字类指标趋同 | digit 类指标跨词表趋同，不可信 | 只信 roundtrip / operator_isolation / 字符拆分率等不失真指标 |
| **方向读反** | 曾把 BPT 当"越低越好"，误判中文压缩领先厂商；实际 2.3746 vs grok 3.6904，**落后 35.7%** | 比较前先声明方向：BPT = bytes/token，越高越好 |
| 堆语料救压缩 | 语料扩到 15 GB（×4.26）后 real_zh BPT 仅 2.3726 → 2.3746，几乎无收益 | 优先改归一化可逆性与词表配额分配（原 262144 中 CJK 仅 1.09%） |

---

## 5. 标准执行流程（SOP）

1. **导出 HF 格式**：CLI 新增 dump 子命令 → `vocab.json + merges.txt`（BPE）或 `tokenizer.json`。
2. **TokenizerBench 体检**：82 条快速跑 → 记录 overall（fertility / compression / fidelity failures）与 per-subcategory 明细。
3. **Tokka-Bench 深评**：120 语言，记录 per-language bytes/token、unique token coverage、subword fertility、word-split rate。
4. **TokEval 全面对比**：多 tokenizer 同跑，取词表利用率、n-gram 熵、平均 token rank、MorphScore、TFG。
5. **自建双口径报表**：`D_lang` + `dispersion` + 保真指标（roundtrip / 字符拆分率 / 操作符隔离）。
6. **L3 下游验证**：小模型 BPB 对比。
7. **归档**：按 §6 目录规范落盘。

---

## 6. 归档与命名规范

```
benchmark/eval_<YYYYMMDD>_<model>/
├── report/     结论、图表（含热力图、散点图）
├── metrics/    原始指标（json / csv）
├── scripts/    复现脚本
├── logs/       运行日志
└── README.md   索引：模型 / 语料 / 命令 / 依赖 / 复现验证
```

- 不覆盖既有文件，重名自动加后缀。
- 横向对比矩阵中，每个 tokenizer 必须同时给出**厂商最佳值**与**缺口百分比**。

---

## 7. 引用来源

| 来源 | 链接 / 标识 |
|---|---|
| TokenizerBench | https://kitefishai-tokenizerbench.hf.space/ ；https://huggingface.co/spaces/KiteFishAI/TokenizerBench |
| Tokka-Bench | arXiv:2610.08794 ；https://github.com/bgub/tokka-bench ；https://tokka-bench.streamlit.app/ |
| TokEval | arXiv:2608.18062 ；https://github.com/AntonioLopardo/tokenizer-analysis-suite |
| Qtok | arXiv:2410.12989 |
| Beyond Text Compression: Evaluating Tokenizers Across Scales | ACL 2025（arXiv:2506.03101） |
| MorphScore | Arnett & Bergen, 2025 |
| Tokenizer unfairness（15× token premium） | Petrov et al., 2023 |
| The Tokenizer Playground | https://huggingface.co/spaces/Xenova/the-tokenizer-playground |
| Dutch Tokenizer Arena | https://huggingface.co/spaces/yhavinga/dutch-tokenizer-arena |

---

## 附：本文档尚未覆盖的空白

- 现有公开平台**均无跨语言归一化口径**，跨语言结论仍需自建。
- Tokka-Bench 明确列为 future work 的方向：内在指标与下游表现的相关性、多采样方差估计、WordPiece / Unigram 覆盖。
- 因此本项目的多厂商横向对比，仍以自建双口径报表为准，公开平台仅作定位与交叉验证。
*（内容由AI生成，仅供参考）*
