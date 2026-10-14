---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_5c8cdc7ec76811f1820a525400638852
    ReservedCode1: N02srsCUFR4TsgxUmOBjWWXRz0xRRwASVfyeInfvGLjXB5vpcLQq/bRjS3wvrho549Ty/QAZS6IkWJTfbHGIaE4JM464RBcWxi7hOjn5TyBPp2gv1xyoWiLRWUf+sZ2x55xFmUvJnIsuS4UTWKKSjI9dS1h+lA59HiQca3qnyBy9BZ0xKn08UaA0WxE=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_5c8cdc7ec76811f1820a525400638852
    ReservedCode2: N02srsCUFR4TsgxUmOBjWWXRz0xRRwASVfyeInfvGLjXB5vpcLQq/bRjS3wvrho549Ty/QAZS6IkWJTfbHGIaE4JM464RBcWxi7hOjn5TyBPp2gv1xyoWiLRWUf+sZ2x55xFmUvJnIsuS4UTWKKSjI9dS1h+lA59HiQca3qnyBy9BZ0xKn08UaA0WxE=
---

# 参考新论文的 tokenizer 优化方案

**文档日期**：2026-10-14　**文件位置**：`E:\tokenizer_c\2026-10-14-tokenizer-optimization-plan-papers.md`
**基线模型**：`E:\tokenizer_c\model_big2.tkm`（vocab 262,144，2026-10-13 训练完成）
**前置文档**：`E:\tokenizer_c\2026-10-13-improvement-plan.md`（优化诊断与路线图）
**统一验收闸门**：双口径比值 D（`D_lang = BPT_eval / BPT_real`）与离散度 dispersion（`max(D) - min(D)`），实现于 `E:\tokenizer_c\benchmark\eval_metrics.py` 协议鲁棒性段，判定阈值 0.35 / 1.0
**论文依据来源**：随附检索片段（SuperBPE、BoundlessBPE、STRR、Parity-Aware BPE、Bit-level BPE、Entropy-Driven Pre-Tokenization、BPE-knockout 等），著录见各章

本方案覆盖六项改造：词表分类别配额与跨语言公平（第 2 章）、CJK 感知预分词与种子（第 3 章）、piece 长度约束（第 3 章）、byte fallback / bit 级重构（第 4 章）、归一化可逆（第 4 章）、数字与操作符规则（第 5 章）；第 1 章定义闸门与论文地图，第 6 章给出消融矩阵、里程碑与风险。每项均按"论文依据（论文名 + 出处）→ 可落地改造点 → 验收闸门（D / dispersion + 专项指标）"三段式展开。

---

# 第 1 章 闸门与论文地图：先定义"过与不过"

## 1.1 验收闸门：双口径比值 D 与离散度 dispersion

本轮全局任务的验收闸门已实现于 `E:\tokenizer_c\benchmark\eval_metrics.py` 的"协议鲁棒性"段（`PROTOCOL_LANGS = ("en", "zh", "code")`），启用方式为显式传入 `--protocol-slices-dir`（可选配套 `--protocol-full-corpus-dir` / `--protocol-label` / `--protocol-out` / `--protocol-md`）：

- **双口径比值 D**：`D_lang = BPT_eval_lang / BPT_real_lang`。含义是"同一模型在评测语料与真实语料上的每 token 字节数之比"，D 越接近 1，说明两种语料给出的结论越一致。
- **离散度 dispersion**：`dispersion = max(D) - min(D)`，并附三语种标准差与三语种等权 macro 均值。
- **判定阈值**（`PROTOCOL_STABLE_MAX = 0.35`、`PROTOCOL_SENSITIVE_MAX = 1.0`）：

| dispersion | 判定 | 含义 |
|---|---|---|
| ≤ 0.35 | 稳定 | 评测语料与真实语料结论一致，可用于验收 |
| 0.35 < d ≤ 1.0 | 敏感 | 口径会影响排序，验收结论需附双口径明细 |
| > 1.0 | 口径依赖 | 该模型的任何"优化结论"不成立，先修口径 |

同时段还输出 `protocol_audit`（评测集质量审计：不同行数、重复行率等），这是判断"闸门本身是否可信"的前置。

**当前基线**（出处：`E:\tokenizer_c\benchmark\eval_2026-10-13_model_big2\metrics\protocol_robustness.json` 与 `report\protocol_robustness.md`，共 8 个模型）：

| 模型 | D_en | D_zh | D_code | dispersion | 判定 |
|---|---|---|---|---|---|
| model_big2 | 1.0065 | 1.0465 | 1.2134 | 0.2069 | 稳定 |
| model_256k | 未提供（本方案未取得逐语种值） | — | — | 0.3560 | 敏感 |

需要明确的判读前提：**当前 D 的"稳定"不能直接当作公平性证据**。现有 `eval_zh` 是模板合成集（3,362,472 汉字中仅 427 个不同汉字，`本文从`/`基于`/`的角度出发` 各出现 16,105 次），D_zh 会被模板长 piece 命中放大。因此本方案规定：**任何改造过闸前，必须先把中文评测切片换成真实 held-out 语料**（即路线图 P0-2），之后 D 与 dispersion 才是有效闸门。历史 256K 报告（`eval_zh` 8.1546 vs `real_zh` 1.7289）出自旧口径，不作为闸门基线。

## 1.2 闸门的另一半：真实域专项指标

dispersion 只回答"口径是否可信"，不回答"tokenizer 是否变好"。因此每档改造同时以 `real_*` 语料上的专项指标作为硬闸门（现有值均取自 `model_big2` 实测，出处：`benchmark\eval_2026-10-13_model_big2\`）：

| 专项指标 | model_big2 当前值 | 闸门目标 | 承担章节 |
|---|---|---|---|
| real_zh BPT | 2.3746 | ≥ 4.0（对齐 doubao 4.3739 / hunyuan 4.5382 区间下沿） | 第 2、3 章 |
| real_code BPT | 3.3796 | ≥ 3.7（追平 doubao 3.7229） | 第 2、3、5 章 |
| real_en BPT | 4.5598 | ≤ 4.60（不退化） | 第 2、3 章 |
| real_zh 单字节回退占比 neg_share | 42.88% | < 10% | 第 2、4 章 |
| single_cjk（整字 piece） | 968 | ≥ 3,000 | 第 2、3 章 |
| char_split_rate | 0.17249 | < 0.01 | 第 4 章 |
| utf8_completeness | 0.8793 | > 0.99 | 第 4 章 |
| roundtrip_ok（zh mismatch 行数） | false（85/200） | true（0） | 第 4 章 |
| digit_boundary_f1 / digit_split_variability | 0.400 / 1.660 | ≥ 0.90 / ≈ 0 | 第 5 章 |
| operator_isolation | 0.433（grok 0.967） | ≥ 0.95 | 第 5 章 |

**过闸定义（三件套同时成立）**：① `dispersion ≤ 0.35`；② 各语种 `D ∈ [0.85, 1.15]`；③ 本档涉及的专项指标达到目标值。

## 1.3 六项改造 ↔ 论文 ↔ 源码落点总览

| # | 改造项 | 论文依据（出处） | 主要源码落点 |
|---|---|---|---|
| 1 | 词表分类别配额与跨语言公平 | Parity-Aware BPE（Foroutan et al., arXiv:2508.04796）；STRR（Nayeem et al., NeurIPS 2025 WS, arXiv:2510.09947）；Crosslingual Tokenizer Inequities（arXiv:2510.21909）；BPE-knockout（Bauwens & Delobelle, NAACL 2024, pp. 5810–5832） | `src\trainer.c` 剪枝循环；`include\tokenizer.h` 训练选项 |
| 2 | CJK 感知预分词与种子 | Entropy-Driven Pre-Tokenization（Hu et al., Tokshop 2025, arXiv:2506.15889）；SentencePiece `auto_character_coverage.md`（引 Land & Arnett 2025, arXiv:2505.24689） | `src\trainer.c` 种子阶段与 `max_candidates`；新增预切分模块 |
| 3 | piece 长度约束 | SuperBPE（Liu et al., COLM 2025, arXiv:2503.13423）；BoundlessBPE（Schmidt et al., COLM 2025, arXiv:2504.00178） | `include\tokenizer.h`（`max_piece_len = 8`）；`src\tokenizer.c` encode 匹配 |
| 4 | byte fallback / bit 级重构 | Bit-level BPE（Moon, Hiraoka & Okazaki, arXiv:2506.07541）；SentencePiece byte fallback 无损承诺 | `src\trainer.c`（`is_bytefb_style` 跳过计数）；`src\tokenizer.c`（`kByteFallbackScore = -10.0f`） |
| 5 | 归一化可逆 | 无对口论文（工程正确性项）；依据 SentencePiece 文档对无损 round-trip 的承诺 | `src\normalize.c` L139；`src\nfkc_tables.inc` L3682；`src\tokenizer.c` L237–244 hook |
| 6 | 数字与操作符规则 | 无对口论文（工程规则项）；可援引 BoundlessBPE 对预分词"按空白与标点切分"的描述 | 种子约束 + 剪枝约束 + 编码前 pretokenizer |

## 1.4 依赖顺序：先可信，再变好

**论断：第 5 项（归一化可逆）与评测口径修订必须先行，否则其余五项的效果无法判定。** 三条依据：

1. 归一化不可逆是"文本等价性"缺陷：`normalize_text` 无条件 NFKC（`src\normalize.c` L139），`U+FF0C → U+002C` 无逆向项（`src\nfkc_tables.inc` L3682），roundtrip 必然失败（zh 85/200 行 mismatch）。此项不修，任何"BPT 提升"都可能只是归一化折叠幅度的函数。
2. 评测口径不修，优化方向会被带偏：`char_split_rate` 在两代模型间"逐位相同"（0.172494）说明当前指标对结构改动不敏感；而 `bpt_total` 按字节加权（英文文件最大 15.9 MB），会稳定掩盖中文崩溃。
3. 第 4 项（byte fallback）是第 2、3 项的兜底前提：若不先让 256 个字面 piece 常驻，配额与种子改造释放的长尾字符仍会走进"单字节回退 + 手写 `-10.0` 常数"的旧路径，效果会被稀释。

因此执行顺序固定为：**闸门与可逆性（A1）→ byte fallback（A2）→ 配额（A3）→ 种子与长度（A4）→ 规则（A5）→ 探索项（A6/A7）**，详见第 6 章。

---

# 第 2 章 词表分类别配额与跨语言公平

## 2.1 问题：频率驱动目标必然挤压 CJK

`model_big2` 的词表构成（自解析 `model_big2.tkm`）：CJK 相关 piece 合计 2,857 个（1.09%），其中整字 piece 仅 968 个；`non_ascii_non_cjk` 占 124,939 个（47.66%）；`ascii_*` 合计 134,344 个。后果是一条闭环：整字 piece 少 → 中文只能走字节路径 → `real_zh` 42.88% 的 token 是单字节回退 → EM 中中文的期望计数无法形成高价值 piece → 下一轮剪枝继续把中文挤出。绝对值对比：`real_zh BPT = 2.3746`（约 0.79 汉字/token），厂商为 3.69–4.54（约 1.23–1.51 汉字/token），落后 35.6%–47.7%。

论文对这一现象给出了直接解释与替代方案：

| 论文 | 结论 | 对本项目的直接含义 |
|---|---|---|
| Parity-Aware BPE（Foroutan et al., arXiv:2508.04796） | 标准频率目标偏向训练数据主导语言；改为按语言配额后，低资源语言 token 数更公平，整体压缩率与下游性能基本无损 | 把全局 `(score desc, len desc)` 换成"语言/字符类别内排序 + 配额"，不是新增语料能替代的 |
| Crosslingual Tokenizer Inequities（arXiv:2510.21909） | 用与英文内容对齐的平行语料训练，token premium 平均只降约 1%；按语言拟合幂律曲线（原文形如 `CTC = a × V^b`）确定各语言目标词表规模 | 配额比例应由"词表规模-压缩率曲线"反推，而非经验拍定；同时该文明确否证了"喂平行语料"这条捷径 |
| STRR（Nayeem et al., NeurIPS 2025 WS, arXiv:2510.09947） | fertility 只看平均值，掩盖词表分配偏差；STRR（type-level 单 token 保留率）揭示中文、阿拉伯语核心词在所有被测模型上都碎片化 | 新增 STRR 作为公平性辅助闸门：它能在 BPT 尚未改善时先暴露分配问题 |
| The Art of Breaking Words（arXiv:2508.06533） | 数据混合（mixture）优化在 16 语种、16K–128K 词表下取得最低 fertility，词表规模变化时分配结论偏差仅 1–2% | 配额 + 语料混合应联合设置；但论文只验证到 128K，本项目 262,144 需自行消融外推 |
| BPE-knockout（Bauwens & Delobelle, NAACL 2024, pp. 5810–5832） | 剪枝 "stepping stone" token 时若直接删除会引发 merge 级联失效；正确做法是只删 token 与其 merge，并把引用它的 merge 用 parents 重连（`(xy,z)→xyz` 变 `(x,y,z)→xyz`） | 回收 `non_ascii_non_cjk` 12.4 万槽位的安全操作规范 |

## 2.2 改造点：分类别配额剪枝

**动作 1：新增配额配置。** 在 `include\tokenizer.h` 的训练选项结构（现含 `max_piece_len = 8`、`shrink = 60`、`vocab = 262144`、`add_byte_fallback = 1`、`fold_case = 0`）中新增分类别预算字段，例如 `quota_cjk`、`quota_ascii_alpha`、`quota_ascii_digit`、`quota_ascii_punct`、`quota_code_symbol`、`quota_other_non_ascii`，并新增 `max_piece_chars_cjk` / `max_piece_bytes_latin`（见第 3 章）。

**动作 2：剪枝目标改造。** `src\trainer.c` 现有剪枝按全局分数与长度排序取 top-N。改为两段式：

1. 先把候选按字符类别打标签（CJK / ASCII 字母 / 数字 / 标点 / 代码符号 / 其他非 ASCII）；
2. 在每类配额内独立排序取 top-k，配额未用满的类别不允许自动被其他类别蚕食（避免回到全局竞争）。

建议初始配额（基于"要把 `real_zh` 拉到 ≥4.0 至少需 3,000+ 整字 piece + 3–6 万中文词/子词 piece"的估算）：

| 类别 | 建议预算占比 | 对应 piece 数（vocab 262,144） |
|---|---|---|
| CJK（整字 + 多字词/子词） | 15%–25% | 约 3.9 万 – 6.5 万 |
| ASCII 字母 / 混合 | 30%–40% | 约 7.9 万 – 10.5 万 |
| 数字 / 标点 / 操作符 | 3%–6% | 约 0.8 万 – 1.6 万 |
| 其他非 ASCII（回收后） | ≤ 10% | ≤ 2.6 万 |

**动作 3：安全回收 `non_ascii_non_cjk`。** 该类别占 47.66% 词表，却没有任何指标证明其价值，且 `utf8_completeness = 0.8793`、`boundary_crossing_rate = 0` 指向"词表里存在单个非 ASCII 字节 piece"。回收必须三步走：

1. 先做只读定性（路线图 §5-E2）：导出全部该类 piece，统计严格 UTF-8 可解码比例、长度分布、是否为单字节非 ASCII；
2. 判定为碎片的按 BPE-knockout 规范删除——只删 token 与其 merge 边，并把使用它的 merge 用 parents 重连为多 token merge，避免递归级联失效；
3. 回收出的槽位按动作 2 的配额重新分配，并保留一份回收前词表快照以便回滚。

**动作 4：新增 STRR 指标。** 在 `benchmark\eval_metrics.py` 增加 `strr_by_lang()`（type-level：统计某语种词表中"整词被单个 piece 完整覆盖"的比例），并入协议鲁棒性段落输出，与 D / dispersion 同表呈现。

## 2.3 闸门与失败处置

| 判据 | 目标 | 不达标时的处置 |
|---|---|---|
| `single_cjk` | ≥ 3,000 | 提高 `quota_cjk` 一档（15% → 20% → 25%）重跑 |
| `real_zh` 单字节回退占比 | < 10% | 检查 byte fallback 是否已按第 4 章修好，未修好则该闸门无效 |
| `real_zh BPT` | ≥ 4.0 | 见第 3 章种子与长度改造，两者需联合调 |
| `dispersion` | ≤ 0.35 | 若放宽，先复核 `--protocol-slices-dir` 的中文切片是否已换成真实语料 |
| `real_en BPT` | ≤ 4.60（退化 > 2% 即回退） | 回退配额比例至上一档，保护英文 |
| 三语 STRR 极差 | 未提供基线，首轮建立基线后收紧 | 首轮仅记录，不做硬性判定 |

**取舍交代**：配额制必然牺牲一部分英文/代码 piece 的边际收益，其正当性来自 Parity-Aware BPE 的实测结论（压缩率与下游性能基本无损）。但该结论在 262,144 规模、en/zh/code 三语混合语料上未经本项目验证，因此必须以路线图 E6 的三档消融（CJK 5% / 15% / 25%）确立 `real_zh` 与 `real_en`/`real_code` 的帕累托前沿，再定最终配额。

---

# 第 3 章 CJK 感知预分词、种子与 piece 长度

## 3.1 问题：种子按全局频次排序，长度按统一字节上限

两处结构性偏差：

1. **种子阶段对 CJK 系统性不利**。`src\trainer.c` 的种子构造是"UTF-8 边界字节 n-gram 频率统计 + 按 count/长度取 top `max_candidates`"，没有任何按字符类别的处理。英文子词的单体频次远高于单个汉字（9,449 个汉字摊薄），在"按 count 排序"下中文候选被大量挤出；同时 `max_candidates = 1,000,000` 的哈希表超限时**直接拒 key**，而语料按文件/块顺序读入（头部为 EN/CODE），后出现的 CJK n-gram 存在被系统性丢弃的风险（该假设尚待路线图 §5-E1 验证，未验证前不作为结论）。
2. **piece 长度上限 8 字节**。中文一字 3 字节，因此一个 piece 最多 2 个汉字（+2 字节余量），三字及以上中文词（"人工智能"、"社会主义市场经济"）在结构上不可能成为单个 token；英文则是 8 个字符的子词，尺度完全不对等。当前后果：968 个整字 piece 对 9,449 个不同汉字覆盖约 10.2%（对评测口径 4,113 个不同汉字覆盖 22.95%）。

论文侧的相关机制：

| 论文 | 机制 | 本项目的用法 |
|---|---|---|
| Entropy-Driven Pre-Tokenization（Hu et al., Tokshop 2025, arXiv:2506.15889） | 两种熵信息预切分策略；其中一种用逐点互信息（PMI）+ 左/右熵识别"连贯字符跨度"，指导 BPE 在无词边界语言上产出更有语言学意义的子词 | 用于生成中文候选白名单与边界约束：只有高 PMI、左右熵达阈值的字符跨度才进入种子候选池 |
| SentencePiece `auto_character_coverage.md`（引 Land & Arnett 2025, arXiv:2505.24689） | 固定 `character_coverage = 0.9995` 的启发式在多语语料下失效——语料含数万不同 Unicode 字符，无法全部登记；未被登记字符依赖 byte fallback 保证无损 round-trip | 支撑"把观测到的 9,449 个汉字全量纳入种子白名单"，同时把 byte fallback 修成真承诺（第 4 章） |
| SuperBPE（Liu et al., COLM 2025, arXiv:2503.13423） | 两阶段预分词课程：先学子词，再放开空白限制，把常见多词表达合并为 superword；同尺寸词表 token 数最多减少 33%，8B 模型下游平均 +4.0%（MMLU +8.2%） | 借其"分阶段放开约束"课程思想设计配额与候选（先类内、后跨类）；承载体本身列入探索项（见 3.3） |
| BoundlessBPE（Schmidt et al., COLM 2025, arXiv:2504.00178） | 取消预分词屏障、允许跨 pretoken 合并，每 token 压缩约 20%，token 分布更均匀，缓解大词表边际收益递减 | 同上，属"结构性替代路线"，不与配额改造同批上线 |

需要如实说明：Entropy-Driven 论文摘要提到"两种策略"，本次取得的片段完整描述了第一种（PMI + 左/右熵），第二种策略的构造细节未取得原文，实施前须回查 arXiv:2506.15889 全文。

## 3.2 改造点

| # | 动作 | 位置 | 说明 |
|---|---|---|---|
| 1 | 种子候选分层：按字符类别（CJK / 拉丁 / 数字 / 标点 / 符号 / 其他）分层，各类独立 top-k | `src\trainer.c` 种子阶段 | 取消全局 count 排序，与第 2 章的剪枝配额形成同构的两级约束 |
| 2 | 汉字白名单：把观测到的 9,449 个汉字全量纳入种子并优先保留 | `src\trainer.c` 种子阶段 | 依据 `auto_character_coverage` 对固定覆盖率启发式的批判 |
| 3 | 中文候选加熵过滤：对汉字 n-gram 计算 PMI 与左右熵，只保留高连贯跨度进入候选池 | 新增预切分模块（种子前） | 对应 Entropy-Driven 第一种策略；实现为独立候选过滤器，不改 encode 路径 |
| 4 | 解除拒键偏差：提高 `max_candidates` 或改分片哈希，并加计数器统计各类别拒键率 | `src\trainer.c` | 同时产出 E1 的验证数据（拒键候选的字符类别分布） |
| 5 | 长度上限按类别解耦：新增 `max_piece_chars_cjk`（建议 4–6 字符）与 `max_piece_bytes_latin`（建议 8–16 字节），原 `max_piece_len = 8` 保留为非 CJK 的字节硬上限 | `include\tokenizer.h`（`TOKENIZER_TRAIN_OPTIONS_DEFAULT`） | 直接消除"三字词不可能成 token"的结构性封锁 |
| 6 | 编码侧一致性检查：长度放开后校验 trie 匹配与分段逻辑的边界条件 | `src\tokenizer.c` L283–392 | 避免训练/推理长度语义不一致（训练允许 6 字符、推理提前截断） |

**先后关系**：动作 5 与动作 1、2 必须同批上线（否则种子放宽了、piece 仍装不下）；动作 3、4 可独立验证（分别产出语言学边界质量与拒键率数据）。

## 3.3 探索项：SuperBPE / BoundlessBPE 该不该上

| 维度 | 结论 |
|---|---|
| 论文收益 | 同尺寸词表下 token 数最多减少 33%（SuperBPE）；每 token 压缩约 20%（BoundlessBPE）；分布更均匀，缓解大词表边际递减 |
| 本项目风险 | ① 与 `max_piece_len` 短上限直接冲突：跨空白合并天然要求更长 piece；② 与第 5 章数字/操作符规则冲突（跨 pretoken 边界合并会破坏三位分组与操作符隔离）；③ 代码语料占行数 65.94%，跨空白合并对缩进/换行语义未必有利，且 8B 下游收益在本项目无对应验证条件 |
| 建议 | 列为 A6 独立探索项，**不与 A3/A4 同批上线**；前置条件是第 4 章 byte fallback 与长度解耦已完成。判断标准：在同一闸门下比较 `real_zh`/`real_code` BPT 与 dispersion，若无一致改善则放弃 |

## 3.4 闸门

| 判据 | 目标 | 说明 |
|---|---|---|
| 整字覆盖（distinct，对 9,449 汉字） | > 95% | 对应路线图 P1-2 的 22.95% → >95% |
| `single_cjk` | ≥ 3,000 | 与第 2 章共用 |
| `real_zh BPT` | ≥ 4.0 | 与配额改造联合判定，单改种子不一定达标 |
| `real_en` / `real_code` BPT | 退化 ≤ 1% | 长度解耦不得让拉丁/代码子词变差 |
| `dispersion` / 各语种 `D` | ≤ 0.35 / `D ∈ [0.85, 1.15]` | 三件套之一 |

---

# 第 4 章 byte fallback 重构、bit 级重编码与归一化可逆

## 4.1 byte fallback 的三处实现缺陷

`model_big2` 的元数据声明 `add_byte_fallback = true`，但实测三处缺陷使其形同虚设：

| # | 缺陷 | 证据 | 量化后果 |
|---|---|---|---|
| 1 | 初始化时搜索 256 个 `<0xXX>` 样式 piece，实际找到 0 个 | 路线图 §2.5 词表构成诊断 | 词表未持有字节字面 piece，长尾字符无兜底入口 |
| 2 | 训练 EM 阶段对 `is_bytefb_style` 的 piece 跳过计数 | `src\trainer.c` | 字节路径没有真实期望计数，永远不会被"学好" |
| 3 | 编码时字节路径边权是手写常数 `kByteFallbackScore = -10.0f` | `src\tokenizer.c` | 分词器在"低分但无损"与"高分但错误"之间必然选错，且该常数不可学习 |

三项合并后的可观察结果：`utf8_completeness = 0.8793`、`char_split_rate = 0.17249`、`roundtrip_ok = false`（zh 语料 200 行中 85 行 mismatch）。其中 `boundary_crossing_rate = 0` 与 `utf8_completeness < 1` 并存，指向"词表中存在单个非 ASCII 字节 piece，但解码端未能把它们拼回合法 UTF-8"。

**改造点（按序执行）**：

1. 训练结束时强制注入 256 个 `<0x00>`–`<0xFF>` 字面 piece，并将其标记为"不可剪枝"类别（`src\trainer.c`，与第 2 章配额机制联动：为字节 piece 单列一个小额保留预算，例如 0.1%）；
2. EM 阶段取消 `is_bytefb_style` 跳过，改为给字节路径真实计数（可用加权的负先验避免字节 piece 过度夺冠）；
3. 编码边权改用学习到的分数，删除 `kByteFallbackScore = -10.0f`；同时保留"无损优先"回退逻辑作为硬约束：允许字节解码的路径不得被判为不可用；
4. 同步修正 HF/`tokenizers` 导出，使外部生态看到与 `.tkm` 一致的 256 个字节 piece（否则跨生态 round-trip 结论不可比）。

**闸门**：`utf8_completeness > 0.99`、`char_split_rate < 0.01`、`roundtrip_ok = true`（zh mismatch 0）。这三项是"结构上不可失败"的承诺项，允许牺牲 BPT（±1% 内可接受）。

## 4.2 bit 级重编码：值得做，但不该现在做

Bit-level BPE（Moon, Hiraoka & Okazaki, arXiv:2506.07541）的出发点是：byte-level fallback 虽能避免 OOV，但把字符拆成单个字节会显著拉长 CJK、emoji 等长尾场景的序列长度，增加训练与推理计算，且字节序列的 Rényi 熵偏低（效率不高）。其做法是打破 8-bit 字节边界，把字节序列视为 bit 流，利用 CJK 三字节 UTF-8 的公共前缀冗余去重：对三字节序列 `b1 b2 b3` 重编码为 6-bit 前缀 piece 与两个 9-bit piece（`b̂1 = (b1 & 127) >> 2` 等按位运算），实现**无损**的序列压缩。

对本项目的取舍：

| 评估维度 | 判断 |
|---|---|
| 理论收益 | 中文走字节路径时序列更紧凑；`real_zh` 的 42.88% 单字节回退场景直接受益 |
| 前置依赖 | 必须先完成 4.1（byte fallback 修好）、第 3 章（整字覆盖 >95%）——否则测到的收益混杂了"兜底缺失"的影响 |
| 隐性代价 | piece 语义不再是"字节/字符"，需自行维护 bit 级编解码与跨生态映射；HF/`tokenizers` 无对应 piece 类型，导出与互操作需额外适配；论文未给出与配额/长度改造叠加时的收益数据（未提供） |
| 建议 | 列为 A7 独立探索项。判定原则：在 A2/A4 完成、中文单字节回退降到 <10% 之后，若 `real_zh BPT` 与 `char_split_rate` 仍有明显余量，再立项 |

## 4.3 归一化可逆：把"可逆"变成可承诺属性

**现状**：`normalize_text` 无条件执行 NFKC（`src\normalize.c` L139 的 `nfkc_normalize_cps`），经 `src\tokenizer.c` L237–244 的 normalizer hook 挂入 encode 路径；decode 端不做归一化，也没有逆向映射表。`src\nfkc_tables.inc` L3682 记录 `U+FF0C → U+002C`（全角逗号折叠为半角）。结果是 `roundtrip_ok = false`，且真实中文文本中的全角标点、全角字母在编码前就被改写，使"评测口径 — 真实口径"的差异（即 D 的分母与分子）混入了归一化折叠幅度这一混淆项。

**候选方案对比**：

| 方案 | 做法 | 可逆性 | 代价 |
|---|---|---|---|
| A 默认 NFC | `normalize.c` 改为 `nfc_normalize_cps`，不再折叠兼容字符 | 完全可逆 | 丢失 NFKC 的"同形合并"收益（对中文 BPT 影响预计在 ±1% 内，需实测） |
| B 保留 NFKC + 逆向映射 | 记录每个折叠点，decode 按段恢复 | 段落粒度可逆；一对多折叠（多个源字符映射到同一 NFKC 形式）无法完全还原 | 需持久化额外状态，改动面大 |
| C 外置 hook | 沿用 `tokenizer_set_normalizer_hook`，由调用方决定策略 | 取决于调用方 | 不解决默认行为，但为嵌入方提供兼容出口 |

**建议**：默认走方案 A（可逆性与正确性优先），同时把方案 B 做成可选增强模式（例如 `normalize_mode = strict | nfkc_inverse`），方案 C 保留为公开 API。**必须如实说明**：本项没有对口论文机制，依据是 SentencePiece 文档所承诺的"未登记字符经 byte fallback 仍保证无损 round-trip"——即"无损可逆"是 tokenizer 应承诺的基本属性，而不是精度优化项。

**闸门**：`roundtrip_ok = true`（zh 85 → 0）；`real_*` BPT 变化在 ±1% 内；`D_zh` 不劣化且 `dispersion ≤ 0.35`。若换成 NFC 后中文 BPT 反而下降超过 1%，说明当前语料的 NFKC 折叠确实在"刷分"，此时应如实报告"该收益以不可逆为代价"，而不是回退可逆性改造。

## 4.4 本章与 D / dispersion 的关系

本章三项改造共同决定闸门是否**可用**：

- byte fallback 修好后，`real_zh` 的 token 序列不再由"单字节回退"主导，`D_zh = BPT_eval / BPT_real` 的分子与分母才测量的是同一件事；
- 归一化改为可逆后，评测语料与真实语料在"标点是否被折叠"上不再有系统性差异，D 的"稳定"才具解释力；
- 因此 A1（可逆）+ A2（byte fallback）完成的判据不是"指标变好"，而是"闸门首次可信"——这也是第 1 章依赖顺序结论的落点。

---

# 第 5 章 数字与操作符规则

## 5.1 现状：数字分组与操作符隔离是"低价可修的确定性故障"

| 指标 | model_big2 | 参照（grok） | 含义 |
|---|---|---|---|
| `digit_boundary_f1` | 0.400 | 0.628 | 数字串的三位分组边界预测准确度 |
| `digit_split_variability` | 1.660 | 未提供 | 同一数字串在不同上下文中被切成不同分组的分歧度（越低越好） |
| `operator_isolation` | 0.433 | 0.967 | 操作符是否被隔离为独立 piece |

现状成因明确：种子与剪枝阶段**没有任何字符类别与位置约束**，数字与操作符自由参与 n-gram 合并，于是 `1234567` 可能被切成 `1234|567`、`12|34567`、`1234567` 之中的任意一种，取决于该片段在语料中的局部频次；操作符则与相邻标识符黏连成 piece。这是"加语料无法改善"的典型——路线图已实测语料 ×4.26 后 `bpt_total` 反而小幅恶化（3.5996 → 3.6279），因为频率目标本身不约束分组一致性。

论文依据的如实交代：本项**没有对口的论文机制**。可援引的只有两条弱依据：① BoundlessBPE（Schmidt et al., arXiv:2504.00178）在描述预分词时指出其"通常按空白与标点切分"（`typically splitting on whitespace and punctuation`），说明标点/操作符边界是通行预切分传统；② Tokenization 综述词条在评述 BLT 字节级路线时指出"非对称数字分组（asymmetric digit grouping）"是固定词表模型的已知痛点，属综述性资料而非论文实验证据。因此本项定位为**工程规则**，收益按"便宜分"评估，不作为学术创新点主张。

## 5.2 规则设计

**规则 1：三位分组右对齐。** 数字串从最低位起按 3 位分组，组内字符允许合并为 piece，**跨组禁止合并**；孤立的最左组（1–3 位）自有边界。例：`1234567` → `1|234|567`；`10000` → `10|000`。实现三层：

1. 种子阶段：为数字候选打"组内标记"，跨组候选不进入候选池；
2. 剪枝阶段：把跨越组边界的数字 piece 直接剔除，并在配额中为数字单列预算（第 2 章表中 3%–6% 的一部分）；
3. 编码前：用 pretokenizer 插入软边界，保证训练与推理的分组口径一致（避免"训练按 3 位、推理按频率"的双口径）。

**规则 2：数字不与相邻字母/标点黏连。** 版本号、坐标、时间戳等场景下（`v2.1.0`、`12:30`、`-3.5`）按 `.`、`:`、`-` 边界隔离数字组；符号保留其自身 piece 归属。

**规则 3：操作符隔离。** 把 `=`、`==`、`!=`、`<=`、`>=`、`->`、`+=`、`&&`、`||`、`**`、`//` 等常见操作符与其组合显式加入候选白名单（不依赖频率进入），并在剪枝时标记为"不可剪"。依据：`operator_isolation = 0.433` 说明当前 65.94%（按行）的代码语料并未把操作符推向独立 piece，白名单是成本最低的修正。

**规则 4：与跨空白合并（A6 探索项）的冲突处理。** 若将来采用 SuperBPE / BoundlessBPE 式跨空白合并，必须为数字与操作符设置"禁止跨空白"的例外，否则三位分组与操作符隔离会被重新打散。该条以约束形式写入 A6 的验收标准。

## 5.3 闸门与止损

| 判据 | 目标 | 止损/回退 |
|---|---|---|
| `digit_boundary_f1` | ≥ 0.90 | 未达标则先检查编码前 pretokenizer 是否与训练口径一致 |
| `digit_split_variability` | ≤ 0.05 | 未达标说明规则 2 与规则 1 冲突，优先保规则 1 |
| `operator_isolation` | ≥ 0.95 | 白名单覆盖不足，扩充操作符组合清单 |
| `real_code BPT` | ≤ 3.3796 × 1.01（不退化） | 若因隔离导致代码 BPT 恶化 >1%，放宽规则 2（保留三位分组与操作符隔离） |
| `real_en BPT` | 退化 ≤ 2% | 触发时放弃"数字不与相邻标点黏连"这一条，仅保留三位分组 |
| `D` / `dispersion` | `D ∈ [0.85, 1.15]` / ≤ 0.35 | 三件套之一，任一不达标则该档不认定通过 |

说明：规则类改造的收益主要体现在结构化文本指标与算术/代码场景的稳定性，**不承诺**显著改善 BPT。若某档改造后三项结构指标达标但 BPT 未改善，应按"通过"记录（结构正确性属承诺项），而不是因为 BPT 未涨就否定该档——这也是把结构指标与压缩指标分列的原因。

---

# 第 6 章 执行计划、消融矩阵与风险

## 6.1 消融矩阵（A0–A7）

每档只在前一档通过闸门后推进，避免多改造并发导致归因困难。

| 档位 | 包含改造 | 关键交付物 | 验收标准（三件套 + 专项） |
|---|---|---|---|
| A0 | 无（基线） | 闸门可信化：中文评测切片换成真实 held-out，`--protocol-slices-dir` 指向新的 `eval_*`/`real_*` | 输出 8 模型级 D/dispersion 表；`protocol_audit` 无重复行告警 |
| A1 | 归一化可逆（第 4.3 节） | 新的 `normalize.c` + 可选 `nfkc_inverse` 模式 | `roundtrip_ok = true`；BPT 变化 ±1% 内；`dispersion ≤ 0.35` |
| A2 | byte fallback 四修（第 4.1 节） | 256 字节 piece 常驻 + EM 计数 + 学习边权 + HF 一致 | `utf8_completeness > 0.99`；`char_split_rate < 0.01`；`roundtrip_ok = true` |
| A3 | 分类别配额剪枝（第 2 章） | 配额配置 + `non_ascii_non_cjk` 回收清单 | `single_cjk ≥ 3,000`；`real_zh` 单字节回退 < 10%；`real_en ≤ 4.60` |
| A4 | CJK 种子 + 长度解耦（第 3 章） | 分层种子 + 汉字白名单 + 熵过滤 + `max_piece_chars_cjk` | 整字覆盖 > 95%；`real_zh BPT ≥ 4.0`；`real_en`/`real_code` 退化 ≤ 1% |
| A5 | 数字与操作符规则（第 5 章） | pretokenizer 软边界 + 数字/操作符白名单 | `digit_boundary_f1 ≥ 0.90`；`operator_isolation ≥ 0.95`；`real_code ≤ 3.3796 × 1.01` |
| A6 | 探索：SuperBPE / BoundlessBPE 式跨空白合并 | 两阶段课程原型（先子词、后放开空白） | 在同一闸门下比较 `real_zh`/`real_code` BPT 与 dispersion，无一致改善则放弃 |
| A7 | 探索：Bit-level BPE 重编码 | bit 级编解码原型 + 跨生态映射说明 | 仅在 A2/A4 完成后评估；中文单字节回退已 < 10% 时若余量仍大再立项 |

按语种看 A3/A4/A5 的判定重点：`real_zh BPT` 由 A3+A4 联合负责，`real_en` 由 A3/A4 的保护阈值负责，`real_code BPT` 由 A3/A5 共同负责——三条语种线互不覆盖，必须分别报告，不能用 `bpt_total` 的字节加权值代替（该口径被英文文件主导，会掩盖中文崩溃）。

## 6.2 里程碑与执行计划

| 阶段 | 任务 | 关键交付物 | 负责人 | 依赖 | 验收标准 |
|---|---|---|---|---|---|
| M0 | 评测口径修订 + 闸门可信化 | 真实 held-out 中文切片、`protocol_robustness.json`（A0 基线） | 评测组（待指派） | 无 | `protocol_audit` 通过；D/dispersion 可复算 |
| M1 | 归一化可逆（A1） | 改动后的 `normalize.c`、roundtrip 报告 | 训练组（待指派） | M0 | zh roundtrip mismatch 0；BPT ±1% |
| M2 | byte fallback 四修（A2） | 词表字节 piece 清单、HF 导出校验 | 训练组（待指派） | M1 | `utf8_completeness > 0.99` |
| M3 | 配额 + 回收（A3） | 配额配置、回收清单与快照 | 训练组（待指派） | M2 | `single_cjk ≥ 3,000`；三语 BPT 不退化 |
| M4 | 种子 + 长度（A4） | 分层种子实现、汉字白名单 | 训练组（待指派） | M3 | 整字覆盖 > 95%；`real_zh ≥ 4.0` |
| M5 | 规则项（A5） | pretokenizer 规则 + 白名单 | 训练组 + 评测组 | M4 | 三项结构指标达标 |
| M6（可选） | 探索项 A6/A7 立项评审 | 原型与取舍结论 | 待指派 | M5 | 按 6.1 的探索档判据 |

时间与人力预算：**未提供**（项目现有排期与训练单轮耗时数据本次未取得），因此本表不填具体日期与工时；`shrink = 60`、`vocab = 262144`、EM 12 轮的既有训练配置与 15,006,950,149 B 语料（en 56.42% / zh 19.99% / code 23.59% 字节）可直接复用，每档需完整重训一次 `model_big2` 规格模型。

评估节奏：每档训练完成后跑一次全量评测（`benchmark\eval_metrics.py --protocol-slices-dir ... --protocol-full-corpus-dir ... --protocol-out ... --protocol-md ...`），产出 `metrics\protocol_robustness.json` 与 `report\protocol_robustness.md`；每档设止损点，未过闸不进下一档。

## 6.3 风险与止损

| 类别 | 风险 | 触发条件 | 影响 | 预防/止损 |
|---|---|---|---|---|
| 预算风险 | 配额挤压英文/代码 piece | `real_en BPT > 4.60`（退化 >2%）或 `real_code` 退化 >1% | 高 | 配额回退一档（如 CJK 25% → 15%）；E6 三档消融确定帕累托点 |
| 执行风险 | 多改造并发导致归因失败 | 同期上线 ≥2 档改造 | 高 | 强制 A1→A5 串行；每档保留词表与配置快照以支持回滚 |
| 效果风险 | 跨空白合并与长度/规则冲突；bit 级收益被前置改造吃掉 | A6/A7 上线后 `real_zh` 无改善或结构指标回退 | 中 | 明确不并入主序列，作为独立探索档；A7 以"单字节回退已 < 10%"为立项门槛 |
| 外部风险 | bit 级 piece 与自定义归一化在 HF/`tokenizers` 无对应表示 | 外部生态 round-trip 与 `.tkm` 不一致 | 中 | 导出层显式声明不支持项；跨生态结论单独报告，不混入闸门 |
| 效果风险 | 评测口径本身不可信（合成切片） | `protocol_audit` 报高重复行 / 低行数多样性 | 高 | M0 必须先完成切片替换，否则任何 D/dispersion 结论不得上报 |

## 6.4 明确不采用的机制

1. **平行语料对齐训练**：Crosslingual Tokenizer Inequities（arXiv:2510.21909）实测 token premium 平均只降约 1%，性价比过低，不列入路线。
2. **朴素删除 stepping-stone token**：会引发 merge 级联失效（Bauwens & Delobelle, NAACL 2024），回收槽位一律按 knockout 的 parents 重连规范执行。
3. **BLT 式取消 tokenizer 的字节级架构替换**：与本项目 C 语言 tokenizer + BPE 的既有工程栈不兼容，且论文级验证属 8B/4T 训练规模，本项目无对应算力条件（未提供），仅作为长期对照路线记录。

## 6.5 结论

六项改造中，**归一化可逆**（第 4.3 节）与 **byte fallback 四修**（第 4.1 节）是把闸门变可信的前置项，应先于一切精度优化；**分类别配额**（第 2 章）、**CJK 种子与长度解耦**（第 3 章）是 `real_zh BPT` 从 2.3746 提升至 ≥4.0 的主力；**数字与操作符规则**（第 5 章）是低成本的结构正确性补齐；**SuperBPE / BoundlessBPE** 与 **Bit-level BPE** 收益最大但代价与冲突也最大，应在 A2/A4 完成后作为独立探索项评估。全部档位以 `dispersion ≤ 0.35`、各语种 `D ∈ [0.85, 1.15]` 与专项指标三件套共同判定，任一不达标即不认定为通过。
*（内容由AI生成，仅供参考）*
