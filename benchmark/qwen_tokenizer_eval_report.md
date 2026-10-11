---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_7b4a82f5c51e11f19063525400393706
    ReservedCode1: f45s7RueUnzeksDuMlJoBJ2iVmetB2BHV0CQ6i221wXHJgIxgF4ystMam9AZW0cfaXZuzz/1EbEQ5d9eBcENw0Guu2c721/qvOHoxMJetgxu4yAUxWcv1MS+DwNN/xuRpGRq1zlMHHZcyev05wdlpd1duo8wFHM+BjnIt8GV0SKmxrV4GH1fiA1ySxk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_7b4a82f5c51e11f19063525400393706
    ReservedCode2: f45s7RueUnzeksDuMlJoBJ2iVmetB2BHV0CQ6i221wXHJgIxgF4ystMam9AZW0cfaXZuzz/1EbEQ5d9eBcENw0Guu2c721/qvOHoxMJetgxu4yAUxWcv1MS+DwNN/xuRpGRq1zlMHHZcyev05wdlpd1duo8wFHM+BjnIt8GV0SKmxrV4GH1fiA1ySxk=
---

# Qwen2.5 系列模型 Tokenizer 评测报告

- 评测日期：2026-10-11
- 评测对象：`E:\qwen-distill\models\Qwen2.5-1.5B-Instruct` 与 `E:\qwen-distill\models\Qwen2.5-7B-Instruct`
- 评测体系：对齐 [yaoqianhao/tokenizer_c](https://github.com/yaoqianhao/tokenizer_c) 仓库 `benchmark/eval_metrics.py` 的完整指标集（BPT / 还原率 / 稳定性 / OOV / TokEval 六维 / 吞吐）
- 评测语料：仓库原版多语言语料 `benchmark/corpus`（en 15.9MB / zh 10.6MB / code 10.6MB，共 37.1MB、约 720 万 token）

## 1. 关键结论

**两个模型的 tokenizer 文件完全一致**（`tokenizer.json` / `vocab.json` / `merges.txt` 的 MD5 逐文件相同，均为 Qwen2.5 官方 BPE tokenizer，词表 151,643），因此除吞吐测量噪声外，全部质量指标相同，二者在 tokenizer 层面**无优劣之分**。差异仅在模型权重规模（1.5B vs 7B），与 tokenizer 评价无关。

## 2. 评测结果总览

| 指标 | Qwen2.5-1.5B | Qwen2.5-7B | 说明 |
|---|---|---|---|
| 词表大小 | 151,643 | 151,643 | 同一 tokenizer |
| overall BPT（字节/token） | 5.1523 | 5.1523 | 压缩效率 |
| 还原率（字节级，200 行/语种） | 100% OK | 100% OK | 无损往返 |
| 稳定性（Jaccard，100 句×3 扰动） | 0.8229 | 0.8229 | 扰动鲁棒性 |
| OOV 率 | 0.0000 | 0.0000 | 无未登录词 |
| 字节回退率 | 0.0000 | 0.0000 | 语料内无回退触发 |
| UTF-8 完整性 | 1.0000 | 1.0000 | 多字节字符完整覆盖 |
| 跨字符边界率 | 0.0000 | 0.0000 | 无跨边界切分 |
| 字符拆分率 | 0.9977 | 0.9977 | 约 0.23% 多字节字符被拆分 |
| 数字边界 F1 | 0.4033 | 0.4033 | code 语料 |
| 数字切分变异性 | 0.0000 | 0.0000 | BPE 确定性切分，方差为 0 |
| 操作符隔离度 | 0.1667 | 0.1667 | code/en 语料 |
| encode 吞吐（tokens/s） | 478,380 | 470,395 | 含测量噪声 |
| decode 吞吐（tokens/s） | 7,875,737 | 7,720,898 | 含测量噪声 |
| 模型加载 | <1 ms | <1 ms | Rust tokenizers 内存加载 |

## 3. 分语种明细

| 语种 | 字节数 | tokens | BPT | chars/token | 还原率 | 稳定性 | OOV% |
|---|---|---|---|---|---|---|---|
| code | 10,641,181 | 2,614,361 | 4.0703 | 4.0703 | OK | 0.7437 | 0 |
| en | 15,900,854 | 2,598,360 | 6.1196 | 6.1196 | OK | 0.7580 | 0 |
| zh | 10,566,465 | 1,989,620 | 5.3108 | 1.8243 | OK | 0.9671 | 0 |
| **total** | **37,108,500** | **7,202,341** | **5.1523** | — | — | 0.8229 | 0 |

### TokEval 维度（arXiv 2608.18062）

| 语种 | utf8_completeness | char_split_rate | boundary_crossing | digit_f1 | digit_variability | operator_isolation |
|---|---|---|---|---|---|---|
| code | 1.0000 | —（无多字节抽样） | 0.0000 | 0.4033 | 0.0000 | 0.3333 |
| en | 1.0000 | — | 0.0000 | —（无 ≥4 位数字串） | — | 0.0000 |
| zh | 1.0000 | 0.9977 | 0.0000 | — | — | —（无操作符） |
| **overall** | **1.0000** | **0.9977** | **0.0000** | **0.4033** | **0.0000** | **0.1667** |

## 4. 与仓库基准（tokenizer_c 256K Unigram）对比

| 指标 | Qwen2.5 BPE (151,643) | tokenizer_c 256K Unigram | SentencePiece 基线 (262,144) |
|---|---|---|---|
| overall BPT | **5.1523** | 7.4881 | 5.8653 |
| code BPT | **4.0703** | 7.2127 | 4.5740 |
| en BPT | 6.1196 | 7.3988 | **5.7108** |
| zh BPT | **5.3108** | 7.9345 | 8.6622 |
| 还原率 | **100%（zh 亦 100%）** | en/code 100%、zh 57.5%（NFKC 归一化所致） | en/code 100%、zh <100% |
| 稳定性 overall | 0.8229 | 0.7090 | ≈0.8617 |
| UTF-8 完整性 | **1.0000** | 1.0000 | 0.9957 |
| encode 吞吐 | 478K t/s | 90.3K t/s | 1,252K t/s |
| decode 吞吐 | 7,875K t/s | 7.3K t/s | 9.8K t/s |

对比要点：

1. **压缩效率全面领先**：Qwen2.5 在 overall、code、zh 三个维度 BPT 均优于 tokenizer_c 的 256K 模型，其中中文压缩（5.31 vs 8.66 的 SP 基线）优势显著；en 略高于 SP 基线但低于 tokenizer_c 差距比例。
2. **无损往返**：Qwen2.5 不做 NFKC 归一化，zh 还原率 100%，优于仓库模型 zh 的字节级非 100%（其偏差为归一化语义等价行为，非缺陷）。
3. **稳定性中等**：0.8229 介于 tokenizer_c（0.7090）与 SP 基线（0.8617）之间；短序列扰动敏感属 BPE 常见现象。
4. **操作符隔离较弱**：0.1667 明显低于仓库模型（0.9833），Qwen BPE 倾向把操作符与相邻 token 融合（如 `=`、`+` 与标识符粘连），对代码场景的细粒度切分不利。
5. **数字切分**：digit F1 0.4033 与仓库 256K（0.5）同量级，BPE 词表对长数字串覆盖有限；切分变异性为 0（确定性编码，优于仓库的 0.9658）。

## 5. 说明

- 评测脚本为对齐仓库指标体系的适配实现（`qwen_eval.py`），将仓库面向 TKCZ/.model 的 CLI 评测替换为 Rust `tokenizers` 对 HF BPE 的等指标计算，指标定义与抽样口径（200 行还原 / 100 句×3 扰动 / 30 数字串 / 30 操作符等）与原仓库一致。
- 两个模型复用同一 tokenizer 文件，故指标逐项相同；吞吐差异为进程测量噪声（同文件、同脚本两次运行，差约 1.7%）。
- 原始 JSON 报告：`output/qwen_1.5B_eval.json`、`output/qwen_7B_eval.json`。
*（内容由AI生成，仅供参考）*
