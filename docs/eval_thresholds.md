---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_aead3d3ec96411f18533525400638852
    ReservedCode1: MXe/QQXMoMpi1RwPuMipbfDCz0oHiUloZjZrXDh+KWBD5xFlZmkSvkh/XVgpJYg2LCDUKK5cutpBbu103UWkMFT26nZWUgIXQagc/SxE4pYTJ+Rc2uUM1Y7hwtjApDrlFX49zJUwjdNeGf+OjSMZN0WnxyZWXOElmKugXRaFvzAFa3wgLgV1vlRHqO4=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_aead3d3ec96411f18533525400638852
    ReservedCode2: MXe/QQXMoMpi1RwPuMipbfDCz0oHiUloZjZrXDh+KWBD5xFlZmkSvkh/XVgpJYg2LCDUKK5cutpBbu103UWkMFT26nZWUgIXQagc/SxE4pYTJ+Rc2uUM1Y7hwtjApDrlFX49zJUwjdNeGf+OjSMZN0WnxyZWXOElmKugXRaFvzAFa3wgLgV1vlRHqO4=
---

<!-- 本文件由 benchmark/gen_thresholds_doc.py 从 benchmark/thresholds.json 生成，请勿手改。 -->

# tokenizer_c 评测阈值与门禁（自动生成）

- 配置源：`E:\tokenizer_c\benchmark\thresholds.json`
- 配置版本：`1`（updated_at=2026-10-16，sha256_16=2b4d9289cf8bd94b）
- 生成方式：`python benchmark/gen_thresholds_doc.py`（`--check` 可校验文档与配置是否一致）

## 一、阈值单点化说明

- 本文件是 tokenizer_c 评测侧唯一阈值来源：eval_metrics.py（语料级主评测）与 probe_regression.py（19 子类探针回归）都从这里读门禁，不再各自硬编码。
- 文档 docs/eval_thresholds.md 由 benchmark/gen_thresholds_doc.py 从本文件生成，禁止手改该文档。
- 门禁初值来源：output/tokenizer_c-优化可能性分析报告.md §7.2 建议门禁表；composite_score 阈值由本机基线实跑校准（见 calibrated_by）。
- 改阈值只允许改本文件：改完请重跑 gen_thresholds_doc.py 让文档同步。

### 阈值标定出处

| 项 | 内容 |
|---|---|
| date | 2026-10-16 |
| model | E:\tokenizer_c\model_big3.tkm |
| cli | E:\tokenizer_c\build_big3\tok-cli.exe |
| method | 先跑基线得到实测值，再把 composite_score 阈值取实测值向下取整留出余量；其余阈值沿用报告 §7.2。 |
| probe_tokenizers | E:\tokenizer_c\hf_export_big3_nfc（nfc，2026-10-16 起为导出默认）；E:\tokenizer_c\hf_export_big3（nfkc 兼容模式）；E:\tokenizer_c\hf_export_big3_rev（reversible 模式） |

## 二、综合分（composite_score）口径

| 子项 | 权重 |
|---|---|
| fertility | 0.35 |
| byte_compression | 0.2 |
| fidelity_pass_rate | 0.25 |
| vocabulary_utilization | 0.1 |
| fairness | 0.1 |

归一化区间：fertility 1→4（好→差），byte_compression 0.1→0.4；fidelity/vocab_util/fairness 直接映射到 [0,1]。

- 公式来源：E:\tok_eval\app\metrics.py::DEFAULT_WEIGHTS / composite_score（同公式移植到 benchmark/eval_thresholds.py）
- 代码位置：`benchmark/eval_thresholds.py::composite_score`（主评测与探针共用）

### fertility 分级口径

| 分级 | 上界（tokens/word，越低越好） |
|---|---|
| excellent | < 1.5 |
| good | < 2 |
| fair | < 3 |
| poor | < 4 |

## 三、语料级主评测门禁（`eval_metrics.py`）

- 适用范围：语料级主评测 benchmark/eval_metrics.py（corpus: en/zh/code）

| # | 门禁指标 | 判定 | 阈值 | 生效条件 | 说明 |
|---|---|---|---|---|---|
| 1 | `fairness_gini` | <= | 0.11 | - | 跨语种 tokens/byte 基尼（语种等权），报告 §7.2 |
| 2 | `composite_score` | >= | 60 | - | 综合分（tok_eval 同公式，输入为语料级微平均 + 采样保真率）；基线校准值 |
| 3 | `stability` | >= | 0.85 | - | 扰动稳定性，报告 §7.2（README 记载 0.709 未达标） |
| 4 | `roundtrip_pass_rate` | >= | 0.95 | norm_mode = reversible | reversible 模式下每行 encode→decode 逐字节还原率 |
| 5 | `roundtrip_pass_rate` | >= | 0.7 | norm_mode ∈ {nfkc, nfc} | nfkc/nfc 兼容模式仅作不回归下限（有损归一化天然无法 100% 还原） |
| 6 | `utf8_completeness` | >= | 0.85 | - | TokEval UTF-8 完整性（不回归下限）。口径说明：该指标按『非特殊 token 字节能否严格解码』统计，单字节 byte fallback piece 天然判为不完整，因此只要模型存在 byte fallback，理论上限就 <1.0；报告 §7.2 的『100%』是理想目标值，实际门禁按 2026-10-16 基线实测值向下取整标定。 |
| 7 | `micro_byte_compression` | <= | 0.34 | - | 微平均 tokens/byte（= 1/bpt_total），报告 §7.2『不劣化』 |

## 四、19 子类探针回归门禁（`probe_regression.py`）

- 适用范围：19 子类探针回归 benchmark/probe_regression.py（tok_eval 内置语料 95 样本）
- tok_eval 根目录：`E:\tok_eval`
- 探针语料目录：`E:\tokenizer_c\benchmark\corpus_probe_clean\builtin`
- 基线文件：`E:\tokenizer_c\benchmark\probe_baseline.json`

| # | 门禁指标 | 判定 | 阈值 | 生效条件 | 说明 |
|---|---|---|---|---|---|
| 1 | `fairness_gini` | <= | 0.11 | - | 8 种人类语言 tokens/byte 基尼，报告 §7.2 |
| 2 | `composite_score` | >= | 60 | - | 探针综合分；基线校准值 |
| 3 | `micro_byte_compression` | <= | 0.35 | - | 探针微平均 tokens/byte，报告 §7.2『≤0.34 不劣化』留 0.01 采样余量 |
| 4 | `micro_fidelity_pass_rate` | >= | 0.95 | norm_mode = reversible | reversible 导出物在探针语料上的保真率下限 |

### 指标 diff 容差（劣化超容差 → REGRESSION）

| 指标 | 容差 |
|---|---|
| fertility | 0.05 |
| byte_compression | 0.005 |
| fidelity_pass_rate | 0 |
| subword_fertility | 0.05 |
| continued_word_rate | 0.02 |
| fairness_gini | 0.002 |
| composite_score | 0.5 |

- 探针回归 diff 容差：任一子类指标劣化超容差即标 REGRESSION（--strict 时退出码 1）

## 五、协议鲁棒性判定阈值（`eval_metrics.py`）

- 适用范围：协议鲁棒性判定 benchmark/eval_metrics.py（双口径比值 D / 模型内离散度 dispersion）

| 判定 | 条件（dispersion） | 阈值来源 |
|---|---|---|
| 稳定 | ≤ 0.35 | `thresholds.json::protocol.stable_max` |
| 敏感 | (0.35, 1] | `thresholds.json::protocol.sensitive_max` |
| 口径依赖 | > 1 | 同上 |

- dispersion ≤ stable_max → 稳定；stable_max < dispersion ≤ sensitive_max → 敏感；> sensitive_max → 口径依赖。该判据不再硬编码在脚本里，改阈值只改本文件。

## 六、失败样本归因词典（`eval_thresholds.py`）

| 归因标签 | 含义 |
|---|---|
| `nfkc_fullwidth_fold` | NFKC 兼容折叠（全角→半角等），原文信息不可逆丢失 |
| `whitespace_fold` | 空白折叠（多空白/制表符/换行→单空格，或首尾空白丢失） |
| `zero_width_removed` | 零宽字符被 normalizer 删除 |
| `edge_whitespace_strip` | 行首/行尾空白被丢弃 |
| `decode_empty` | 解码结果为空（token 丢失或全为特殊 token） |
| `invalid_utf8_input` | 输入或输出含非法 UTF-8（U+FFFD 替换） |
| `decode_error` | encode/decode 调用异常 |
| `other` | 其他（byte fallback 丢字、解码器拼接差异等，需人工复核） |

## 七、改动指引

1. 改阈值 → 只改 `benchmark/thresholds.json`，然后重跑本脚本让文档同步；
2. 跑主评测门禁 → `python benchmark/eval_metrics.py --model ... --corpus-dir ...`；
3. 跑探针回归门禁 → `python benchmark/probe_regression.py --tokenizer <tokenizer.json> --strict`；
4. 文档与配置一致性校验 → `python benchmark/gen_thresholds_doc.py --check`。
*（内容由AI生成，仅供参考）*
