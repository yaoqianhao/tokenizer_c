# tokenizer_c 三模式归一化对比报告（nfkc / nfc / reversible）

- 生成时间：2026-10-14 14:48:48
- 模型：`E:\tokenizer_c\model_big2.tkm`（vocab_size=262144, md5=057072bb1cabc9249395e8818bca1255）
- CLI：`E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare\bin\tok-cli-norm.exe`（md5=b8faf2e8ec0d7df6b7aca9c9dce2e6a9）
- 语料（整语料口径）：`E:\tokenizer_c\benchmark\corpus`
- 切片（合成/真实两口径）：`E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare\corpus_slices`
- 统一开关：`--norm=nfkc|nfc|reversible`（`--norm-reversible` 为 reversible 别名）
- 编排脚本：`E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare\scripts\norm_mode_compare.py`
- eval_metrics.py：`E:\tokenizer_c\benchmark\eval_metrics.py`（md5=7af574cb7ff9312fc2e8a7b50cf19cd9）
- git：rev=24cac8aaca17a1f3ade54588743e6767addba6bb，未提交文件数=14

模式列顺序：`nfkc` / `nfc` / `reversible`；差值列 = nfc 相对 nfkc 的变化。

## BPT — 合成口径（eval_<lang>.txt 切片）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| en | 4.5896 | 4.5896 | 4.5896 | +0.0000 |
| zh | 2.4851 | 2.4021 | 2.4021 | -0.0831 |
| code | 4.1009 | 4.1009 | 4.1009 | +0.0000 |

## BPT — 真实口径（<lang>_real.txt 切片）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| en | 4.5598 | 4.5539 | 4.5534 | -0.0059 |
| zh | 2.3746 | 2.1573 | 2.1569 | -0.2174 |
| code | 3.3796 | 3.3779 | 3.3778 | -0.0018 |

## BPT — 整语料（corpus/<lang>/<lang>.txt，bytes/tokens）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| en | 4.5889 | 4.5889 | 4.5889 | +0.0000 |
| zh | 2.4828 | 2.4003 | 2.4003 | -0.0825 |
| code | 4.1097 | 4.1097 | 4.1097 | +0.0000 |
| byte-weighted total | 3.5992 | 3.5489 | 3.5489 | -0.0504 |
| macro（三语种等权） | 3.7271 | 3.6996 | 3.6996 | -0.0275 |
| 全语料合计 bytes | 37108500.0000 | 37108500.0000 | 37108500.0000 | +0.0000 |
| 全语料合计 tokens | 10310146.0000 | 10456447.0000 | 10456447.0000 | +146301.0000 |

## 协议鲁棒性：D = BPT_eval / BPT_real 与 dispersion（阈值 0.35 / 1.0）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| D(en) | 1.0065 | 1.0078 | 1.0080 | +0.0013 |
| D(zh) | 1.0465 | 1.1135 | 1.1137 | +0.0669 |
| D(code) | 1.2134 | 1.2140 | 1.2141 | +0.0006 |
| dispersion max-min | 0.2069 | 0.2062 | 0.2061 | -0.0007 |
| dispersion std | 0.0896 | 0.0842 | 0.0842 | -0.0054 |
| dispersion mean | 1.0888 | 1.1118 | 1.1119 | +0.0230 |
| verdict level | 稳定 | 稳定 | 稳定 | - |
| verdict text | 口径稳定：评测语料与真实语料结论一致（dispersion=0.2069 ≤ 0.35） | 口径稳定：评测语料与真实语料结论一致（dispersion=0.2062 ≤ 0.35） | 口径稳定：评测语料与真实语料结论一致（dispersion=0.2061 ≤ 0.35） | - |

## roundtrip 失败行数（越小越好，0 = 全还原）

失败行数 > 0 表示 decode 后字节与原输入不一致；reversible 模式设计上应接近 0。

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| en 失败/测试行 | 0/200 | 0/200 | 0/200 | - |
| zh 失败/测试行 | 85/200 | 0/200 | 0/200 | - |
| code 失败/测试行 | 0/200 | 0/200 | 0/200 | - |
| 失败合计 | 85.0000 | 0.0000 | 0.0000 | -85.0000 |

## TokEval 指标（三语种平均）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| utf8_completeness | 0.8793 | 0.8675 | 0.8675 | -0.0118 |
| char_split_rate | 0.1725 | 0.1725 | 0.1725 | +0.0000 |
| boundary_crossing_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| digit_boundary_f1 | 0.4000 | 0.4000 | 0.4000 | +0.0000 |
| digit_split_variability | 1.6603 | 1.6603 | 1.6603 | +0.0000 |
| operator_isolation | 0.4333 | 0.4333 | 0.4333 | +0.0000 |

## stability（1.0 = 完全稳定）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| en | 0.7920 | 0.7920 | 0.7920 | +0.0000 |
| zh | 0.9900 | 0.9902 | 0.9902 | +0.0002 |
| code | 0.7842 | 0.7842 | 0.7842 | +0.0000 |
| overall | 0.8554 | 0.8555 | 0.8555 | +0.0001 |

## 吞吐与加载（同一 CLI、同一模型）

| 指标 | `nfkc` | `nfc` | `reversible` | Δ(nfc-nfkc) |
|---|---|---|---|---|
| encode tokens/sec | 1033146.1560 | 1089587.2730 | 1129320.2910 | +56441.1170 |
| decode tokens/sec | 52679.4500 | 59875.1800 | 60969.0520 | +7195.7300 |
| load_ms | 68.8380 | 71.7740 | 75.7440 | +2.9360 |
| 单模式评测墙钟(s) | - | - | - | - |

## 关键差异摘要（相对 nfkc）

### `nfc` 相对 `nfkc`

- BPT 合成 en：4.5896 → 4.5896（+0.0000）
- BPT 合成 zh：2.4851 → 2.4021（-0.0831）
- BPT 合成 code：4.1009 → 4.1009（+0.0000）
- BPT 真实 en：4.5598 → 4.5539（-0.0059）
- BPT 真实 zh：2.3746 → 2.1573（-0.2174）
- BPT 真实 code：3.3796 → 3.3779（-0.0018）
- roundtrip 失败合计：85 → 0
- utf8_completeness：0.8793 → 0.8675（-0.0118）
- char_split_rate：0.1725 → 0.1725（+0.0000）
- digit_boundary_f1：0.4000 → 0.4000（+0.0000）
- digit_split_variability：1.6603 → 1.6603（+0.0000）
- operator_isolation：0.4333 → 0.4333（+0.0000）
- stability overall：0.8554 → 0.8555（+0.0001）
- dispersion(max-min)：0.2069 → 0.2062
- encode tps：1033146.2 → 1089587.3

### `reversible` 相对 `nfkc`

- BPT 合成 en：4.5896 → 4.5896（+0.0000）
- BPT 合成 zh：2.4851 → 2.4021（-0.0831）
- BPT 合成 code：4.1009 → 4.1009（+0.0000）
- BPT 真实 en：4.5598 → 4.5534（-0.0065）
- BPT 真实 zh：2.3746 → 2.1569（-0.2178）
- BPT 真实 code：3.3796 → 3.3778（-0.0018）
- roundtrip 失败合计：85 → 0
- utf8_completeness：0.8793 → 0.8675（-0.0118）
- char_split_rate：0.1725 → 0.1725（+0.0000）
- digit_boundary_f1：0.4000 → 0.4000（+0.0000）
- digit_split_variability：1.6603 → 1.6603（+0.0000）
- operator_isolation：0.4333 → 0.4333（+0.0000）
- stability overall：0.8554 → 0.8555（+0.0001）
- dispersion(max-min)：0.2069 → 0.2061
- encode tps：1033146.2 → 1129320.3

## 逐模式原始报告

- `nfkc`：metrics/eval_metrics_model_big2_nfkc.json；协议段：metrics/protocol_robustness_nfkc.json；协议 Markdown：reports/protocol_robustness_nfkc.md；stdout：logs/eval_nfkc_stdout.log
- `nfc`：metrics/eval_metrics_model_big2_nfc.json；协议段：metrics/protocol_robustness_nfc.json；协议 Markdown：reports/protocol_robustness_nfc.md；stdout：logs/eval_nfc_stdout.log
- `reversible`：metrics/eval_metrics_model_big2_reversible.json；协议段：metrics/protocol_robustness_reversible.json；协议 Markdown：reports/protocol_robustness_reversible.md；stdout：logs/eval_reversible_stdout.log
