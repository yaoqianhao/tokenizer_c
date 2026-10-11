---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_6a175964c52211f19063525400393706
    ReservedCode1: 6JImqqkx17N9DAYJL4bDuBMccxVLzl2tihmlpl3dOoEgbaasGO/7uQ+SGzoV4cFcqoiu9EUWdI2suBvI0ppbFfNYkQBfquJZdR+2qYZ9k+KM4i2EVQmbHpBEHS58E4ygRTeobdBdp+b4MO3N1MSAHajkD4Oc7WHCZsjtErxd+00JuXAYHpJq1Azk02g=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_6a175964c52211f19063525400393706
    ReservedCode2: 6JImqqkx17N9DAYJL4bDuBMccxVLzl2tihmlpl3dOoEgbaasGO/7uQ+SGzoV4cFcqoiu9EUWdI2suBvI0ppbFfNYkQBfquJZdR+2qYZ9k+KM4i2EVQmbHpBEHS58E4ygRTeobdBdp+b4MO3N1MSAHajkD4Oc7WHCZsjtErxd+00JuXAYHpJq1Azk02g=
---

# 多厂商 Tokenizer 横向评测报告

- 评测日期：2026-10-11
- 评测语料：tokenizer_c benchmark/corpus（en 15.9MB / zh 10.6MB / code 10.6MB，总计 37.1MB）
- 评测口径：与既有 Qwen2.5 评测完全一致（qwen_eval.py，对齐 tokenizer_c benchmark/eval_metrics.py）
- 来源说明：gpt6 无官方公开 tokenizer，使用 OpenAI gpt-oss-120b 的 tokenizer.json 替代；opus（Claude 3 Opus）官方 tokenizer 无公开下载源，按用户决定跳过；grok 使用 xai-org/grok-2 的 tokenizer.tok.json（tiktoken 加载，与 SGLang 同源）；doubao 使用 ByteDance-Seed/Seed-OSS-36B-Instruct；hunyuan 使用 tencent/Hunyuan-A13B-Instruct

## 一、总览

| tokenizer | vocab | 还原率 | 稳定性 | OOV | 字节回退 | overall BPT |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B/7B (基线) | 151643 | 100% | 0.8230 | 0% | 0% | 5.1523 |
| gpt6 (gpt-oss-120b) | 199998 | 100% | 0.8269 | 0% | 0% | 4.9335 |
| grok (grok-2) | 131072 | 100% | 0.8272 | 0% | 0% | 4.7766 |
| doubao (Seed-OSS-36B) | 155121 | 100% | 0.8283 | 0% | 0% | 4.8973 |
| hunyuan (Hunyuan-A13B) | 127957 | 100% | 0.8220 | 0% | 0% | 5.1990 |

## 二、分语种 BPT（越低越好）

| tokenizer | code | en | zh | 中文 chars/token |
|---|---|---|---|---|
| Qwen2.5-1.5B/7B (基线) | 4.0703 | 6.1196 | 5.3108 | 1.8243 |
| gpt6 (gpt-oss-120b) | 4.2577 | 6.1211 | 4.3577 | 1.4969 |
| grok (grok-2) | 4.0766 | 6.1202 | 4.1268 | 1.4176 |
| doubao (Seed-OSS-36B) | 3.8659 | 5.7228 | 5.1639 | 1.7739 |
| hunyuan (Hunyuan-A13B) | 4.0705 | 6.1196 | 5.4890 | 1.8856 |

## 三、TokEval 六维（overall）

| tokenizer | utf8完整 | 字符拆分率 | 跨边界 | 数字F1 | 数字变异性 | 操作符隔离 |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B/7B (基线) | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |
| gpt6 (gpt-oss-120b) | 1.0000 | 0.9977 | 0.0000 | 0.0000 | 0.0000 | 0.1667 |
| grok (grok-2) | 0.9924 | 0.0070 | 0.0038 | 0.4033 | 0.0000 | 0.9667 |
| doubao (Seed-OSS-36B) | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |
| hunyuan (Hunyuan-A13B) | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |

## 四、吞吐

| tokenizer | encode (tokens/sec) | decode (tokens/sec) | load (ms) |
|---|---|---|---|
| Qwen2.5-1.5B/7B (基线) | 478380 | 7875737 | 0.15 |
| gpt6 (gpt-oss-120b) | 512912 | 7797184 | 0.20 |
| grok (grok-2) | 10736449 | 47458900 | 0.05 |
| doubao (Seed-OSS-36B) | 495891 | 7737846 | 0.17 |
| hunyuan (Hunyuan-A13B) | 517973 | 7453284 | 0.18 |

## 五、结论

1. **整体编码效率（BPT）**：grok（4.7766）最优，其次 doubao（4.8973）、gpt6（4.9335），hunyuan（5.1990）与 Qwen 基线（5.1523）相近且垫底。
2. **分语种**：code 上 doubao（3.8659）最优；en 上 doubao（5.7228）最优，其余四家接近 6.1；zh 上 grok（4.1268）明显最优，gpt6（4.3577）次之，doubao/hunyuan 中文较贵（5.16/5.49）。
3. **还原率**：全部 100%，无 OOV、无字节回退。
4. **稳定性**：四家均 0.82~0.83，与 Qwen 基线（0.8230）基本持平，doubao 略高（0.8283）。
5. **TokEval**：grok 显著不同——中文字符拆分率 0.007（几乎不拆整字，UTF-8 完整 piece 覆盖 99%+ 中文字符），操作符隔离 0.9667（操作符独立成 token）；其余三家与 Qwen 趋同（字符拆分率 0.9977、操作符隔离 0.1667），说明 gpt-oss/doubao/hunyuan 与 Qwen 的中文/符号切分粒度接近、而 grok 更偏向整字与独立符号。grok 的 utf8 完整性 0.9924（zh 0.9771）略低于其余三家 1.0，存在少量跨 UTF-8 边界 piece，属字节级 BPE 常见现象。
6. **吞吐**：grok（tiktoken 后端）encode 约 1073 万 tokens/sec，远超其余 tokenizers 后端的约 50 万；但两者为不同实现，仅作参考。

## 六、局限说明

- gpt6 无官方 tokenizer，以 gpt-oss-120b 替代，不代表 OpenAI GPT-6 实际分词。
- Claude Opus 官方 tokenizer 无公开来源，本报告不含 opus 数据（用户确认跳过）。
- grok 使用 tiktoken 后端加载，其余使用 Rust tokenizers 后端，吞吐对比跨实现，非严格公平。
- Tokenizers 后端的 TokEval 数字类指标（digit_f1 等）在四家间趋同，区分度有限，建议以 BPT/稳定性/中文字符拆分率/操作符隔离为主判断。

## 附：数据文件

- C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_adc5bfde786d48539622bfb40a51c4b3\output\qwen_1.5B_eval.json
- C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_adc5bfde786d48539622bfb40a51c4b3\output\gpt6_eval.json
- C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_adc5bfde786d48539622bfb40a51c4b3\output\grok_eval.json
- C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_adc5bfde786d48539622bfb40a51c4b3\output\doubao_eval.json
- C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_adc5bfde786d48539622bfb40a51c4b3\output\hunyuan_eval.json
*（内容由AI生成，仅供参考）*
