---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_5bde4870c76811f1a08152540064ee0f
    ReservedCode1: 2T1hUGCyRhxyfBOWMWZWw2l5CNVWHTbBO7v21Q60R8HKh7qHpnzDrky/1yQUG3BSKOqKFJR0RgmthinBRRM4JE/ryozZgsaIAD/u6RspMPKGLnOnancAhzS1dAwXM6nBAlA415+LVxELV9/eRe4B6AcRau9H6trf/T+UUWbm2I1RdwniPZn8DncCoFQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_5bde4870c76811f1a08152540064ee0f
    ReservedCode2: 2T1hUGCyRhxyfBOWMWZWw2l5CNVWHTbBO7v21Q60R8HKh7qHpnzDrky/1yQUG3BSKOqKFJR0RgmthinBRRM4JE/ryozZgsaIAD/u6RspMPKGLnOnancAhzS1dAwXM6nBAlA415+LVxELV9/eRe4B6AcRau9H6trf/T+UUWbm2I1RdwniPZn8DncCoFQ=
---

# 协议鲁棒性指标报告（2026-10-14）

- date: 2026-10-14
- 同口径切片目录: `C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp\indomain`
- 整语料目录（聚合口径）: `E:\tokenizer_c\benchmark\corpus`
- 指标定义: D_lang = BPT_eval / BPT_real；dispersion = max(D) - min(D)
- 判定阈值: dispersion ≤ 0.35 稳定 / 0.35–1.0 敏感 / > 1.0 口径依赖

## 1. 双口径比值与模型内离散度

| model | D_en | D_zh | D_code | dispersion(max-min) | std(三语种) | verdict |
|---|---|---|---|---|---|---|
| model_big2 | 1.0065 | 1.0465 | 1.2134 | **0.2069** | 0.0896 | 稳定 |
| model_big | 1.0064 | 1.0451 | 1.2215 | **0.2151** | 0.0936 | 稳定 |
| model_256k | 4.9663 | 4.7167 | 5.0727 | **0.3560** | 0.1492 | 敏感 |
| qwen1.5B | 1.3284 | 1.2114 | 0.9557 | **0.3727** | 0.1556 | 敏感 |
| grok | 1.3113 | 1.1192 | 0.9677 | **0.3435** | 0.1406 | 稳定 |
| doubao | 1.2673 | 1.1822 | 1.0327 | **0.2346** | 0.0970 | 稳定 |
| gpt6 | 1.2827 | 1.1797 | 1.0024 | **0.2803** | 0.1158 | 稳定 |
| hunyuan | 1.3287 | 1.2117 | 0.9561 | **0.3726** | 0.1556 | 敏感 |

## 2. 聚合口径并列输出

| model | bpt_total（按字节加权） | BPT_macro（三语种等权） | BPT_en | BPT_zh | BPT_code | 口径来源 |
|---|---|---|---|---|---|---|
| model_big2 | 3.5992 | 3.7271 | 4.5889 | 2.4828 | 4.1097 | corpus/<lang>/<lang>.txt 整语料实测（bytes/tokens） |
| model_big | 3.6036 | 3.7358 | 4.6083 | 2.4771 | 4.1220 | corpus/<lang>/<lang>.txt 整语料实测（bytes/tokens） |
| model_256k | 7.5078 | 7.5414 | 7.3971 | 7.9955 | 7.2315 | corpus/<lang>/<lang>.txt 整语料实测（bytes/tokens） |
| qwen1.5B | 5.1523 | 5.1669 | 6.1196 | 5.3108 | 4.0703 | benchmark/qwen_1.5B_eval.json totals 与 langs（整语料） |
| grok | 4.7766 | 4.7745 | 6.1202 | 4.1268 | 4.0766 | benchmark/grok_eval.json totals 与 langs（整语料） |
| doubao | 4.8973 | 4.9175 | 5.7228 | 5.1639 | 3.8659 | benchmark/doubao_eval.json totals 与 langs（整语料） |
| gpt6 | 4.9335 | 4.9122 | 6.1211 | 4.3577 | 4.2577 | benchmark/gpt6_eval.json totals 与 langs（整语料） |
| hunyuan | 5.1990 | 5.2263 | 6.1196 | 5.4890 | 4.0705 | benchmark/hunyuan_eval.json totals 与 langs（整语料） |

## 3. 评测集质量审计 protocol_audit（逐语料文件）

| tag | file | 行数 | 总字符数 | 不同字符数 | 不同汉字数 | 重复行率 | 重复行数 |
|---|---|---|---|---|---|---|---|
| eval_en | eval_en.txt | 22777 | 2097084 | 61 | 0 | 0.495500 | 11286 |
| en_real | en_real.txt | 8238 | 2087058 | 274 | 11 | 0.000000 | 0 |
| corpus_en | en.txt | 172125 | 15900854 | 61 | 0 | 0.512424 | 88201 |
| eval_zh | eval_zh.txt | 15973 | 720327 | 431 | 427 | 0.002379 | 38 |
| zh_real | zh_real.txt | 7111 | 739339 | 4508 | 4113 | 0.000000 | 0 |
| corpus_zh | zh.txt | 80694 | 3629747 | 431 | 427 | 0.010013 | 808 |
| eval_code | eval_code.txt | 30739 | 2097122 | 81 | 0 | 0.665409 | 20454 |
| code_real | code_real.txt | 52739 | 2090366 | 760 | 533 | 0.334989 | 17667 |
| corpus_code | code.txt | 155408 | 10641181 | 81 | 0 | 0.676021 | 105059 |

## 4. 判定语

- **model_big2**（本机 CLI 实测（加速版 CLI，切片+整语料同口径））：口径稳定：评测语料与真实语料结论一致（dispersion=0.2069 ≤ 0.35）
- **model_big**（本机 CLI 实测（加速版 CLI，切片+整语料同口径））：口径稳定：评测语料与真实语料结论一致（dispersion=0.2151 ≤ 0.35）
- **model_256k**（本机 CLI 实测（加速版 CLI，切片+整语料同口径））：口径敏感：评测语料会放大或缩小真实表现，跨口径数值不可直接混用（0.35 < dispersion=0.3560 ≤ 1.00）
- **qwen1.5B**（indomain_results2.json 折算（厂商基线未重跑））：口径敏感：评测语料会放大或缩小真实表现，跨口径数值不可直接混用（0.35 < dispersion=0.3727 ≤ 1.00）
- **grok**（indomain_results2.json 折算（厂商基线未重跑））：口径稳定：评测语料与真实语料结论一致（dispersion=0.3435 ≤ 0.35）
- **doubao**（indomain_results2.json 折算（厂商基线未重跑））：口径稳定：评测语料与真实语料结论一致（dispersion=0.2346 ≤ 0.35）
- **gpt6**（indomain_results2.json 折算（厂商基线未重跑））：口径稳定：评测语料与真实语料结论一致（dispersion=0.2803 ≤ 0.35）
- **hunyuan**（indomain_results2.json 折算（厂商基线未重跑））：口径敏感：评测语料会放大或缩小真实表现，跨口径数值不可直接混用（0.35 < dispersion=0.3726 ≤ 1.00）

## 5. 数据质量观察（由 protocol_audit 派生）

- 评测语料重复行率高（模板合成痕迹）：eval_en=49.5%、eval_code=66.5%，整语料 corpus_en=51.2%、corpus_code=67.6%；而真实切片 en_real=0.0%、code_real=33.5%。
- 中文评测语料字符多样性极低：eval_zh 不同字符 431（不同汉字 427），真实中文切片 zh_real 达 4508（不同汉字 4113）；二者比 10 倍。即评测语料的「协议」覆盖度远低于真实域。
- 含义：D 稳定只说明「评测口径与真实口径的结论方向一致」，不代表评测语料真实；上述重复行率/字符多样性缺口会同时压低评测与真实两侧的 BPT，需与 protocol_audit 并读，不能单独用 D 或 bpt_total 评价 tokenizer。

## 6. 说明

- 项目侧 3 个模型（model_big2 / model_big / model_256k）用本机加速版 CLI 实测，切片与整语料同口径；加速版 CLI 与原版 build/tok-cli.exe 输出逐字节一致。
- 厂商侧 5 家（qwen1.5B / grok / doubao / gpt6 / hunyuan）不重跑，D 与 dispersion 直接由 metrics/indomain_results2.json 的 vendor eval_*/real_* 折算；聚合口径取自 benchmark/<name>_eval.json 的 totals。
- D 越高表示「评测语料低估压缩效率」（同一文本在评测语料上 token 更多）或「真实语料更易压缩」；dispersion 越大说明结论越依赖语料口径。
- protocol_audit 中 corpus_* 为完整评测语料（模板合成），eval_* 为同口径切片，*_real 为真实域内切片。
*（内容由AI生成，仅供参考）*
