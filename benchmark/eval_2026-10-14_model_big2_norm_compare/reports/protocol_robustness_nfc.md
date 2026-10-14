# 协议鲁棒性指标报告 — model_big2/nfc

- date: 2026-10-14
- 同口径切片目录: `E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare\corpus_slices`
- 整语料目录（聚合口径）: `E:\tokenizer_c\benchmark\corpus`
- 指标定义: D_lang = BPT_eval / BPT_real；dispersion = max(D) - min(D)
- 判定阈值: dispersion ≤ 0.35 稳定 / 0.35–1.0 敏感 / > 1.0 口径依赖

## 1. 双口径比值与模型内离散度

| model | D_en | D_zh | D_code | dispersion(max-min) | std(三语种) | verdict |
|---|---|---|---|---|---|---|
| model_big2/nfc | 1.0078 | 1.1135 | 1.2140 | **0.2062** | 0.0842 | 稳定 |

## 2. 聚合口径并列输出

| model | bpt_total（按字节加权） | BPT_macro（三语种等权） | BPT_en | BPT_zh | BPT_code | 口径来源 |
|---|---|---|---|---|---|---|
| model_big2/nfc | 3.5489 | 3.6996 | 4.5889 | 2.4003 | 4.1097 | corpus/<lang>/<lang>.txt 整语料实测（bytes/tokens） |

## 3. 评测集质量审计 protocol_audit（逐语料文件）

| tag | file | 行数 | 总字符数 | 不同字符数 | 不同汉字数 | 重复行率 | 重复行数 |
|---|---|---|---|---|---|---|---|
| eval_en | eval_en.txt | 22777 | 2097084 | 61 | 0 | 0.495500 | 11286 |
| eval_zh | eval_zh.txt | 15973 | 720327 | 431 | 427 | 0.002379 | 38 |
| eval_code | eval_code.txt | 30739 | 2097122 | 81 | 0 | 0.665409 | 20454 |
| en_real | en_real.txt | 8238 | 2087058 | 274 | 11 | 0.000000 | 0 |
| zh_real | zh_real.txt | 7111 | 739339 | 4508 | 4113 | 0.000000 | 0 |
| code_real | code_real.txt | 52739 | 2090366 | 760 | 533 | 0.334989 | 17667 |

## 4. 判定语

- **model_big2/nfc**（本机 CLI 实测（加速版 CLI，输出与 build/tok-cli.exe 逐字节一致））：口径稳定：评测语料与真实语料结论一致（dispersion=0.2062 ≤ 0.35）

## 5. 说明

- D_lang = BPT_eval / BPT_real（BPT = bytes/tokens，tokens 由 CLI 实测）。
- dispersion = max(D) - min(D)，并附三语种总体标准差；阈值 ≤0.35 稳定 / 0.35–1.0 敏感 / >1.0 口径依赖。
- 真实语料为 scripts/indomain_compare.py、indomain_compare2.py 所用同一语料来源，本次未引入新语料。
