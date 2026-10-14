---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_09925452c6e611f1820a525400638852
    ReservedCode1: 6+uLjd3/rYwYNiPUmVZlGR0d1ZTrjyEfdm005lBdAwCxkcnbgQ/pILdeaSz1wD9XhQRb1Sz7lseo3SH202EeDFN9tQtWRCUGs2ggShg0AL7Dkh93lhpTTwb2hsyhg3eRTK8ONwGrcVJQ6cl2tce4ZdnIxAYFq+EJoIDIX3Jv0gDVkihrQiM9v/2F9n8=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_09925452c6e611f1820a525400638852
    ReservedCode2: 6+uLjd3/rYwYNiPUmVZlGR0d1ZTrjyEfdm005lBdAwCxkcnbgQ/pILdeaSz1wD9XhQRb1Sz7lseo3SH202EeDFN9tQtWRCUGs2ggShg0AL7Dkh93lhpTTwb2hsyhg3eRTK8ONwGrcVJQ6cl2tce4ZdnIxAYFq+EJoIDIX3Jv0gDVkihrQiM9v/2F9n8=
---

# model_big2 Tokenizer 优化诊断与路线图

日期：2026-10-13
受测模型：`E:\tokenizer_c\model_big2.tkm`（vocab 262,144，2026-10-13 11:26 训练完成）
证据来源：本轮实测取证 + `benchmark\eval_2026-10-13_model_big2\`（报告 / 指标 JSON / 源码位置）

---

## 0. 结论摘要

**核心判断：语料"量"不是瓶颈（已有反证）；瓶颈在词表预算分配、种子与剪枝目标、piece 长度设计、归一化不可逆这四处代码/结构问题，另有一层是评测口径本身失效。**

三句话版本：

1. **加语料已验证无效**：语料从 3.52 GB → 15.01 GB（×4.26）、EM 轮数 8 → 12，三语 BPT 变化仅 ±0.5%（real_zh 2.3726→2.3746，real_code 3.3676→3.3796），还原率错误数、`char_split_rate`、字节回退占比全部原地不动。**同质语料扩容的边际收益已归零。**
2. **真正的瓶颈是结构性的**：262,144 词表里 CJK 只占 1.09%（2,857 个），其中整字 piece 仅 968 个；而 `non_ascii_non_cjk` 占 124,939 个（47.7%）。结果是真实中文 42.88% 的 token 靠单字节回退，real_zh BPT 2.3746，落后厂商 35.6%–47.7%。
3. **评测层自身有硬缺陷**：中文评测语料是模板合成文（3.36 M 汉字仅 427 个不同汉字，模板句重复 16,105 次），en/code 评测集重复行率 51%/68%，吞吐指标含 80 ms 固定进程开销。历史 model_256k 在 eval_zh 上 8.1546、real_zh 上 1.7289（**同一模型相差 371%**），说明当前口径可以颠倒是非。**不先修口径，后续优化会朝错误方向调参。**

优先级：**P0 = 修口径 + 修归一化可逆性（低成本、当天见效）；P1 = 词表预算重分配 + CJK 感知种子 + piece 长度上限 + byte fallback 重构 + 语料去重配比（真正的主战场）；P2 = 解码性能与 HF 兼容工程化。**

---

## 1. 关键证据

### 1.1 语料（训练）

| 项 | 数值 |
|---|---|
| corpus_big2.txt | 15,006,950,149 B（13.98 GiB）/ 132,876,577 行 = 六份子语料字节之和（逐字节核对相等） |
| 字节占比 | en 56.42% / zh 19.99% / code 23.59% |
| 行数占比 | code 65.94% / en 25.10% / zh 8.95% |
| 汉字承载量 | zh 863,469,131 次（0xE4–0xE9 首字节计数）；**不同汉字约 9,449 个**（800 MB 抽样） |
| 去重/清洗 | **无任何实现**（`scripts\` 无 dedup/clean，`download_corpus.py` 无相关代码） |
| 内部重复行率 | **26.58%**（320 MB 抽样） |
| 来源 | en = allenai/c4；zh = wikipedia-cn + skypile（zh_head jsonl）；code = GitHub parquet |
| 已知浪费 | `TRAIN_MAX_LINE = 4096` 逐行截断，超长代码行被砍 |

### 1.2 词表构成（自解析 model_big2.tkm）

| 类别 | 数量 | 占比 |
|---|---|---|
| special | 4 | 0.00% |
| **single_cjk（单汉字整字）** | **968** | **0.37%** |
| multi_cjk_ge2（≥2 汉字） | 1,518 | 0.58% |
| cjk_mixed_with_other | 371 | 0.14% |
| ascii_alpha | 77,093 | 29.41% |
| ascii_digit | 6,106 | 2.33% |
| ascii_punct | 775 | 0.30% |
| ascii_mixed | 50,370 | 19.22% |
| **non_ascii_non_cjk** | **124,939** | **47.66%** |
| 合计 | 262,144 | 100% |
| 字面 `<0xXX>` byte fallback piece | **0** | 模型元数据却声明 `byte_fallback = true` |

- CJK 相关 piece 合计 2,857 个，占词表 **1.09%**；968 个整字 piece 对 9,449 个不同汉字覆盖约 10.2%（折算），对评测口径 4,113 个不同汉字的覆盖 22.95%（出现次数覆盖 79.31%）。
- `non_ascii_non_cjk` 高达 124,939 个，是本轮最可疑的一块预算占用（见 §2.6）。

### 1.3 指标（同脚本、同 37.1 MB 语料）

| 模型 | real_en | real_zh | real_code | char_split_rate | operator_iso | digit_f1 | utf8_completeness | roundtrip_ok |
|---|---|---|---|---|---|---|---|---|
| **model_big2** | 4.5598 | **2.3746** | 3.3796 | 0.17249 | 0.433 | 0.400 | 0.8793 | **false** |
| model_big | 4.5804 | 2.3726 | 3.3676 | 0.17249 | 0.367 | 0.467 | 0.8795 | false |
| qwen_1.5B | 4.6049 | 4.3911 | 4.2522 | — | — | — | — | true |
| grok | 4.6657 | 3.6904 | 4.2061 | — | 0.967 | — | — | true |
| doubao | 4.5132 | 4.3739 | 3.7229 | — | — | — | — | true |
| hunyuan | 4.6039 | 4.5382 | 4.2509 | — | — | — | — | true |

- 真实域内差距：**real_zh 落后 35.6%–47.7%**；real_code 落后 9.2%–20.5%；real_en 在厂商区间内（略优）。
- 中文 token 组成（`zh_composition.json`）：real_zh 883,092 token 中 **378,708 个（42.88%）是单字节回退**，vocab 用到 3.09%；eval_zh 仅用到 **0.26%**（distinct_ids = 692）。
- 还原率：zh 80,694 行中 36,419 行（45.1%）含全角变体；含变体组 **60/60 mismatch**，干净组 **0/60 mismatch**，首个差异恒为 `pos=6, U+FF0C → U+002C`。

### 1.4 反证：语料扩容无收益

| 项 | model_big | model_big2 | 变化 |
|---|---|---|---|
| 语料 | 3.52 GB / 3,007 万行 | 15.01 GB / 1.329 亿行 | ×4.26 |
| EM 轮数 | 8 | 12 | +4 |
| real_zh BPT | 2.3726 | 2.3746 | **-0.08%（略差）** |
| real_code BPT | 3.3676 | 3.3796 | **-0.36%（略差）** |
| real_en BPT | 4.5804 | 4.5598 | +0.45% |
| roundtrip zh mismatch | 85 | 85 | **不变** |
| char_split_rate | 0.172494 | 0.172494 | **逐位相同** |
| real_zh neg_share | 42.86% | 42.88% | 不变 |
| 单字 piece | 978 | 968 | **减少 10 个** |

**判读**：四个数量级以内的结构指标一位都没动，说明当前结果由"种子 + 剪枝目标 + 词表配额"在早期就锁死了，EM 与语料只是在这个骨架里微调概率。**继续加语料的边际收益 ≈ 0。**

---

## 2. 归因分析

### 2.1 主因一：词表预算分配严重失衡（影响最大）

一个明显要做中英双语 + 代码的分词器，在中文占 20% 字节、汉字出现 8.63 亿次的情况下，只给 CJK 留了 1.09% 的词表。后果是链式反应：

- 968 个整字 piece → 覆盖不到 1/4 的不同汉字 → 其余汉字只能被拆成字节；
- 字节路径用的是虚拟负 id（单字节一个 token）→ real_zh 42.88% 的 token 是单字节 → BPT 被压到 2.3746；
- 中文 piece 少 → EM 里中文的期望计数无法形成高价值 piece → 下一轮剪枝继续把中文挤掉，**形成负反馈闭环**。

对标：厂商 real_zh BPT 3.69–4.54（≈1.23–1.51 汉字/token），本项目 2.3746（≈0.79 汉字/token）。**要把 real_zh 拉到 4.2 以上，至少要保证 3,000+ 整字 piece（覆盖 ~99% 出现次数的常用字）加上 3–6 万中文词/子词 piece，即 CJK 预算需从 1.09% 提到 15%–25%。**

### 2.2 主因二：种子构造对 CJK 系统性不利

`src\trainer.c` 的种子阶段是"UTF-8 边界字节 n-gram 频率统计 + 按 count/长度取 top `max_candidates`"，没有按字符类别的任何特殊处理。两个结构性偏差：

- **频率天然偏英**：英文子词的单体频次远高于单个汉字（9,449 个汉字摊薄），在"按 count 排序"下中文候选被大量挤出；
- **候选哈希上限静默丢键**：`max_candidates = 1,000,000`，哈希表超限时**直接拒 key**，而语料是按文件/块顺序读入（头部为 EN/CODE）——后出现的 CJK n-gram 有被系统性丢弃的风险。这是一条**待验证的高价值假设**（见 §5-E1）。

### 2.3 主因三：piece 长度上限 8 字节，中文词天然成不了词

`max_piece_len` 默认 8 **字节**。中文一个字 3 字节，因此一个 piece 最多 2 个汉字（+2 字节余量），绝大多数三字及以上中文词（"人工智能"、"社会主义市场经济"）在结构上**不可能成为单个 token**。英文则是 8 个字符的子词，尺度完全不对等。这是"字节长度统一上限"在多字节书写系统上的经典缺陷。

### 2.4 主因四：byte fallback 的实现是半成品

三处问题叠加：

1. **字面 `<0xXX>` piece 被剪枝清零（实测 0 个）**，但元数据仍声明 `byte_fallback = true` → HF `tokenizers` 加载后行为未知，跨生态不可复现；
2. **byte fallback 在 EM 中不累积计数**（`is_bytefb_style` 跳过）→ 字节路径的概率从不参与优化，永远只能是兜底；
3. **编码时用固定常数 `kByteFallbackScore = -10.0f` 作为边权**（源码注明是 P1-2 修复）→ 与真实 log 概率不同量纲。任何 score < -10 的真实 piece 会被字节路径错误地击败或反之，**切分决策被一个手写常数污染**，这正是 `char_split_rate = 17.25%`（17% 的多字节字符永远不会作为完整 piece 出现）与 `utf8_completeness = 0.8793`（12.07% 的 token 出现无法严格 UTF-8 解码，且 `boundary_crossing_rate = 0` → 说明是**单个非 ASCII 字节 piece**在词表里）的合理来源。

### 2.5 主因五：NFKC 不可逆，直接吃掉还原率

- 调用链：`tokenizer_encode → normalize_for_encode → normalize_text`（`src\normalize.c` L139 无条件调 `nfkc_normalize_cps`），训练与推理共用同一路径；**decode 不做任何归一化**。
- 具体映射：`src\nfkc_tables.inc` L3682 `{0x0FF0C, 6794, 1}` → 全角逗号兼容分解为 U+002C，重组阶段无反向项，输出恒为半角。
- 结果：全角标点、全角字母数字、以及所有 NFKC 折叠类字符**在 encode 时被永久抹平**，roundtrip 必然失败（85/200 行）；厂商全部为 0 mismatch。
- 现成旁路：`tokenizer_set_normalizer_hook`（`src\tokenizer.c` L237–244）可在运行时替换归一化逻辑，不必改内核。

### 2.6 待验证：`non_ascii_non_cjk` 这 124,939 个 piece 是什么

它占了 47.66% 的词表，但没有任何一项指标证明它在发挥价值。结合 `utf8_completeness = 0.8793`，最可能的情况是其中混有**汉字/字符的半截字节片段**（不可能单独渲染成字符，只会把字符劈开）。若证实，这 12 万个槽位是纯粹的沉没成本——**把它们回收给 CJK 与代码符号，是性价比最高的单点改动**。

### 2.7 主因六：数字与操作符无任何约束

- 种子/剪枝全程没有数字原子性、三位分组或操作符隔离的规则，也没有 pre-tokenizer 规则；
- 实测 `digit_boundary_f1 = 0.400`、`digit_split_variability = 1.660`（厂商均为 0.000 变异性）、`operator_isolation = 0.433`（grok 0.967）。
- 这类指标的失分**几乎不依赖语料量**，纯靠规则/初始化设计即可拿回，属于"便宜分"。

### 2.8 测量层缺陷（先修，否则一切优化不可信）

| 缺陷 | 事实 | 后果 |
|---|---|---|
| 中文评测集是模板合成文 | 3,362,472 汉字 / 仅 **427 个不同汉字**；`本文从`、`基于`、`的角度出发` 各 16,105 次 | 长 piece 命中被极度放大；model_256k 在 eval_zh 8.1546 vs real_zh 1.7289（**相差 371%**），口径能颠倒排序 |
| en/code 评测集高重复 | 重复行率 en 51.24%、code 67.60% | BPT 被重复行主导，等价于变相加权少数文本 |
| 吞吐口径含固定开销 | 每次指标单开进程；1 字节 encode 也要 94.2 ms；decode K 从 10 到 4,000 耗时恒为 95–100 ms；`load_ms = 80.195` | `decode_tps = 53,764`（encode 的 5.05%）是**除以固定开销**的产物，不是解码内核性能 |
| 横向可比性 | 本项目用 C++ CLI 逐进程，厂商用 Python 绑定（`meta.note` 已声明） | 吞吐/加载数字不可直接横比；厂商侧 `char_split_rate / digit_f1` 受 tokenizers 后端 piece 字节失真影响，只能作参考 |
| 指标合成方式 | `bpt_total` 按语料字节加权，en 文件最大（15.9 MB） | "整体 BPT 最优"主要由英文撑起，掩盖中文崩溃 |

---

## 3. 语料该怎么改（不是加量，是改结构）

语料侧**唯一有效的干预是"结构与多样性"**，且必须与 §2 的结构改动配合，否则会重演"×4.26 语料 → 0 收益"：

| 干预 | 依据 | 预期作用 |
|---|---|---|
| **全语料去重**（精确 + MinHash 近重复） | 训练集重复行率 26.58% | 修正频率排名，让 CJK/代码子词的真实竞争力浮现；缩小语料但提升信息密度 |
| **中文配比提到 30%–40% 字节** | zh 仅 19.99% 字节，却要支撑中文指标 | 给 CJK 候选足够的期望计数，打破 §2.1 负反馈 |
| **换掉合成/套话中文** | zh_ext 为新闻体模板，`的角度出发` 等在训练集 320 MB 里 26 次、评测集 10.6 MB 里 16,105 次 | 合成文只贡献套话长串，挤占真实词表预算；应换真实、多样、非模板的中文 |
| **脚本多样性**（日文假名、韩文、西里尔、阿拉伯、emoji 各留最小配额） | vocab 的 47.7% 给了身份不明的非 ASCII piece | 让多语 piece 有序竞争，减少碎片化 |
| **代码语料去超长截断/去 minified** | 66% 行数来自代码，`TRAIN_MAX_LINE = 4096` 截断 | 提升代码子词质量，避免重复 JS 模板 |
| 评测集独立于训练集 | 评测集为训练语料的 0.247% 字节 | 需做近重复检测，确保 held-out 干净 |

**不建议做的**：单纯再堆几十 GB 同源语料；继续加 EM 轮数（×4.26 语料 + 4 轮已验证为 0 收益）；把 shrink 从 60% 调到极端值试运气。

---

## 4. 优先级路线图

### P0（当天—2 天，低成本，先让结果可信）

| # | 动作 | 位置 | 验收标准 |
|---|---|---|---|
| P0-1 | 归一化改为可逆：默认走 NFC（或提供 `normalize_mode` 开关 + 全角保留映射），encode/decode 对称 | `src\normalize.c` L139、`src\nfkc.c`、`src\normalize.h` L19 签名 | roundtrip_ok `false → true`，zh mismatch 85 → 0，厂商级持平；NFKC 保留为可选增强模式 |
| P0-2 | 评测协议修订：① 单进程批量 CLI（一次进程内跑完全部指标，剔除 `load_ms`）② 评测语料去重 ③ 中文换成真实 held-out 语料（合成模板集降级为单独的"模板鲁棒性"探针）④ 输出按语种 + macro 平均，不只给 `bpt_total` | `benchmark\eval_metrics.py`、`benchmark\corpus\` | decode/encode 吞吐不再被 80 ms 抵消；eval_zh 排序不再与 real_zh 反向；报告含逐语种与宏平均 |
| P0-3 | 诊断三连（只读） | 见 §5 | 产出 §2.2 / §2.6 的确认或否证结论 |

### P1（3—7 天，主战场）

| # | 动作 | 预期指标影响（估计，需实测） |
|---|---|---|
| P1-1 | **词表预算重分配**：给 CJK 预留 15%–25%（约 4–6.5 万），英文/代码/符号各自设配额上限，剪枝改为**分类别配额**而非全局 `(score desc, len desc)` | real_zh BPT 2.37 → **≥4.0**（目标对齐 doubao 4.37 / hunyuan 4.54 区间下沿）；neg_share 42.88% → <10% |
| P1-2 | **CJK 感知种子**：把观测到的 ~9,449 个汉字全部纳入种子并优先保留；中文 n-gram 候选不参与全局频率竞争；解除/提高 `max_candidates` 拒键影响 | 整字覆盖 22.95% → >95%（distinct），单字 piece ≥3,000 |
| P1-3 | **piece 长度按字符而非字节限制**（CJK 允许 4–6 字符，拉丁保持 8–16 字节） | 中文词可成 token；real_zh 进一步抬升，real_en/real_code 不退化 |
| P1-4 | **byte fallback 重构**：256 个字面 piece 常驻且不参与剪枝；EM 中给字节路径真实计数；删除 `-10.0` 常数改用学习到的分数 | utf8_completeness 0.879 → >0.99；char_split_rate 0.172 → <0.01；roundtrip 结构上不可再失败 |
| P1-5 | **语料结构改造**（§3 表） | 作为 P1-1~P1-4 的使能项，单独做无效（已有反证） |
| P1-6 | **数字/操作符规则**：右对齐三位分组数字规则 + 操作符隔离规则（pre-tokenizer 或种子约束） | digit_boundary_f1 0.400 → ≥0.90；digit_split_variability 1.660 → ≈0；operator_isolation 0.433 → ≥0.95 |

### P2（工程与生态）

| # | 动作 | 收益 |
|---|---|---|
| P2-1 | decode 内层循环优化：按 piece 预计算长度/是否为 `<0xXX>`/是否含 ▁ 的元数据，段拷贝替代逐字节循环，越界检查提到段级 | 真实解码吞吐（剔除固定开销后）有量级空间 |
| P2-2 | CLI 支持批量/常驻模式（stdin 流式或 server），消除每次 80 ms 加载 | 所有吞吐指标恢复物理意义 |
| P2-3 | HF 兼容：`.json` 导出真实 byte_fallback 字面 piece，用 `tokenizers` 加载验证 encode 一致性 | 跨生态可用性 |
| P2-4 | subword regularization（unigram sampling） | 提升下游模型鲁棒性，不影响本表指标但影响"tokenizer 好坏"的最终判据 |
| P2-5 | trie 构建/加载加速（当前 80 ms/进程，历史 256K 版 479.9 ms） | 训练迭代与评测效率 |

---

## 5. 下一步实验清单（可证伪）

- **E1（验证 §2.2）**：统计种子阶段哈希拒键的比例与拒键候选的字符类别分布，确认 CJK 是否被系统性丢弃。预期：CJK n-gram 拒键率显著高于英文。
- **E2（验证 §2.6）**：导出全部 `non_ascii_non_cjk` piece，统计严格 UTF-8 可解码比例、长度分布、是否为单字节非 ASCII。若碎片占比高 → P1-1 回收预算零风险。
- **E3（验证 §2.5）**：把归一化换成 NFC 后重跑 zh roundtrip 200 行，确认 mismatch 归零且 BPT 不掉（NFC 不折叠全角，BPT 应基本不变）。
- **E4（验证 §2.4）**：临时关闭字节路径（或把 `-10.0` 改为学习分数）后对比 `char_split_rate` 与 `utf8_completeness`，量化常数污染。
- **E5（验证口径）**：同一模型分别在"去重后真实中文集"与"模板合成集"上跑 BPT，报告两者差值作为评测集质量指标；差值 >20% 即判定评测集不合格。
- **E6（卫星实验）**：固定语料与轮数，仅改词表配额（CJK 5% / 15% / 25% 三档）做消融，确定 real_zh 与 real_en/code 的帕累托前沿。

---

## 6. 判读提醒

1. **不要再把 `bpt_total` 当唯一目标**：它按字节加权，英文文件最大，会稳定掩盖中文的崩溃。请以"逐语种 BPT + 宏平均 + 逐语种 token 组成"为主视图。
2. **厂商侧 TokEval 数字类指标（digit_boundary_f1 / operator_isolation / char_split_rate）在 tokenizers 后端下失真**，横向对比只作参考；本项目走 C CLI 路径，其值是可信的，因此**这几项是我们相对厂商的真实短板，而不是幻觉**。
3. **吞吐/加载数字与厂商不可直接横比**（harness 不同），修 P0-2 之后再谈性能排名。
4. **`eval_report_2026-10-10`（历史 256K）不可作为对外基线**：它在评测语料上 7.2–8.0 BPT、真实语料上骤降至 1.4–1.7，属同源语料记忆效应 + 模板集放大。
5. 本文件中所有"预期影响"为**基于当前证据的估计**，须按 §5 的实验逐条验证后再写入结论。

---

## 附：本次诊断用到的判据出处

| 判据 | 出处 |
|---|---|
| 语料字节/行数/重复率 | `corpus_big2.txt`、`corpus_big\` 六文件实测；320 MB / 800 MB 抽样 |
| 词表构成 | 自解析 `model_big2.tkm`（TKCZ 二进制） |
| 训练超参与算法 | `train_big2.py`、`src\trainer.c`、`include\tokenizer.h`（`TOKENIZER_TRAIN_OPTIONS_DEFAULT`） |
| 归一化链条与映射位置 | `src\normalize.c` L139、`src\nfkc.c` L106–179、`src\nfkc_tables.inc` L3682 |
| encode/decode 实现与常数 | `src\tokenizer.c` L237–244 / L253 / L283–392 / L393–464 |
| 指标定义与抽样 | `benchmark\eval_metrics.py` L50–60 / L175 / L269 / L296 / L455 / L469 / L522–569 / L628–641 |
| 中文与还原率证据 | `metrics\zh_composition.json`、`cjk_coverage_nfkc.json`、`roundtrip_rootcause.json`、`indomain_results2.json` |
| 厂商基线 | `benchmark\metrics_consolidated.json` 及 `benchmark\{qwen_1.5B,qwen_7B,grok,doubao,gpt6,hunyuan}_eval.json` |
*（内容由AI生成，仅供参考）*
