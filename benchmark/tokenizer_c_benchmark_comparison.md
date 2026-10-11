---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_58e67286c52511f19063525400393706
    ReservedCode1: WCMiMTyuyQ/trUicE2GlNUEJBZ0aJuv4DsTrkSTelUKS7LeEqaqH//SuX7jy3Pb8L55FGwVtj5zSVKc/k3YF3DCUEe0Y7n6tx7hL98ve0nXAbzXWhwhr7Li5Q0G3+Rc6J30FL3iXt4bHYvs78YCCMEnpDZALHHTELle4byonnM12pFj3ueMTK7+Itjs=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_58e67286c52511f19063525400393706
    ReservedCode2: WCMiMTyuyQ/trUicE2GlNUEJBZ0aJuv4DsTrkSTelUKS7LeEqaqH//SuX7jy3Pb8L55FGwVtj5zSVKc/k3YF3DCUEe0Y7n6tx7hL98ve0nXAbzXWhwhr7Li5Q0G3+Rc6J30FL3iXt4bHYvs78YCCMEnpDZALHHTELle4byonnM12pFj3ueMTK7+Itjs=
---

# 多厂商 Tokenizer 与 tokenizer_c 基准对比报告

- 评测日期：2026-10-11
- 评测语料：tokenizer_c benchmark/corpus（en 15.9MB / zh 10.6MB / code 10.6MB）
- 评测口径：统一对齐 tokenizer_c benchmark/eval_metrics.py（BPT / 还原率 / 稳定性 / OOV / TokEval 六维 / 吞吐）
- 数据来源：
  - tokenizer_c 256K 基准：`temp/tokenizer_c/benchmark/eval_report_2026-10-10.md`（model_256k.tkm，Unigram，vocab 262144）
  - SP 基线：tokenizer_c README §13.6（官方 SentencePiece 0.2.2，同训练语料 262144 Unigram，NFKC）
  - 五家厂商：`output/*_eval.json` + `output/tokenizer_comparison_report.md`

## 一、总体对比

| tokenizer | 类型/词表 | overall BPT | 还原率 | 稳定性 | OOV | UTF-8 完整性 | 操作符隔离 |
|---|---|---|---|---|---|---|---|
| tokenizer_c 256K | Unigram 262144 | 7.4881 | en/code 100%、zh 57.5% | 0.7090 | - | 1.0000 | 0.9833 |
| SentencePiece 基线 | Unigram 262144 | 5.8653 | en/code 100%、zh <100% | ≈0.8617 | 0% | 0.9957 | - |
| Qwen2.5 (基线) | BPE 151643 | 5.1523 | 100% | 0.8230 | 0% | 1.0000 | 0.1667 |
| gpt6 (gpt-oss-120b) | BPE 199998 | 4.9335 | 100% | 0.8269 | 0% | 1.0000 | 0.1667 |
| grok (grok-2) | BPE 131072 | **4.7766** | 100% | 0.8272 | 0% | 0.9924 | 0.9667 |
| doubao (Seed-OSS-36B) | BPE 155121 | 4.8973 | 100% | 0.8283 | 0% | 1.0000 | 0.1667 |
| hunyuan (Hunyuan-A13B) | BPE 127957 | 5.1990 | 100% | 0.8220 | 0% | 1.0000 | 0.1667 |

overall BPT 排名（低=优）：
grok (4.7766) < doubao (4.8973) < gpt6 (4.9335) < Qwen (5.1523) < hunyuan (5.1990) < SP (5.8653) < tokenizer_c 256K (7.4881)

结论：五家厂商 tokenizer 全部优于 tokenizer_c 256K（降幅 30.6%~36.2%）和 SP 基线（降幅 11.4%~18.6%）；hunyuan 虽为五家中最高，仍比 SP 低 11.4%。

## 二、分语种 BPT（越低越好）

| tokenizer | code | en | zh | zh chars/token |
|---|---|---|---|---|
| tokenizer_c 256K | 7.2127 | 7.3988 | 7.9345 | 2.6855 |
| SentencePiece 基线 | 4.5740 | 5.7108 | 8.6622 | - |
| Qwen2.5 | 4.0703 | 6.1196 | 5.3108 | 1.8243 |
| gpt6 (gpt-oss-120b) | 4.2577 | 6.1211 | 4.3577 | 1.4969 |
| grok (grok-2) | 4.0766 | 6.1202 | **4.1268** | 1.4176 |
| doubao (Seed-OSS-36B) | **3.8659** | **5.7228** | 5.1639 | 1.7739 |
| hunyuan (Hunyuan-A13B) | 4.0705 | 6.1196 | 5.4890 | 1.8856 |

要点：
1. **code**：五家全部优于 256K 与 SP；doubao (3.8659) 最优，比 SP (4.5740) 低 15.5%。
2. **en**：doubao (5.7228) 与 SP (5.7108) 基本持平（略高 0.2%），其余四家 ~6.12 高于 SP 约 7%；五家均远优于 256K (7.3988)。
3. **zh**：五家全部大幅优于 256K (7.9345) 与 SP (8.6622)；grok (4.1268) 最优，比 SP 低 52.4%，gpt6 (4.3577) 次之。这是本轮对比中最显著的差距。

## 三、TokEval 六维（overall）

| tokenizer | utf8完整 | 字符拆分率 | 跨边界 | 数字F1 | 数字变异性 | 操作符隔离 |
|---|---|---|---|---|---|---|
| tokenizer_c 256K | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9658 | 0.9833 |
| Qwen2.5 | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |
| gpt6 (gpt-oss-120b) | 1.0000 | 0.9977 | 0.0000 | 0.0000 | 0.0000 | 0.1667 |
| grok (grok-2) | 0.9924 | 0.0070 | 0.0038 | 0.4033 | 0.0000 | 0.9667 |
| doubao (Seed-OSS-36B) | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |
| hunyuan (Hunyuan-A13B) | 1.0000 | 0.9977 | 0.0000 | 0.4033 | 0.0000 | 0.1667 |

要点：
1. **字符拆分率**：tokenizer_c 256K 为 0.0000（Unigram 整字覆盖，中文字符不被拆分）；grok (0.0070) 几乎不拆整字，与 256K 同属"整字优先"；Qwen/gpt6/doubao/hunyuan (0.9977) 属字节级 BPE，绝大多数多字节字符被拆成字节 piece——这是算法路线差异（Unigram/整词 vs 字节级 BPE），非缺陷。
2. **操作符隔离**：tokenizer_c 256K (0.9833) 与 grok (0.9667) 将操作符独立成 token；其余四家 (0.1667) 倾向融合。代码场景下 grok/256K 的细粒度符号切分更友好。
3. **数字**：256K 的 digit F1 (0.5) 与五家 (0.4033，gpt6 为 0) 同量级；但 256K 数字切分变异性 0.9658（同一数字串不同切分），五家均为 0（确定性编码），后者在可复现性上更优。

## 四、吞吐（跨实现，仅参考）

| tokenizer | encode (tokens/s) | decode (tokens/s) | load (ms) |
|---|---|---|---|
| tokenizer_c 256K (C CLI) | 90,311 | 7,293 | 479.9 |
| SentencePiece 基线 (C++/SP) | 1,252,417 | 9,830 | - |
| Qwen2.5 (Rust) | 478,380 | 7,875,737 | 0.15 |
| gpt6 (Rust) | 512,912 | 7,797,184 | 0.20 |
| grok (tiktoken) | 10,736,449 | 47,458,900 | 0.05 |
| doubao (Rust) | 495,891 | 7,737,846 | 0.17 |
| hunyuan (Rust) | 517,973 | 7,453,284 | 0.18 |

注意：tokenizer_c 256K 与 SP 的吞吐为其原生实现（C CLI / C++），五家为 tokenizers/tiktoken 库加载 HF 配置，实现不同、优化程度不同，吞吐对比仅供量级参考，不能作为实现优劣结论。

## 五、结论

1. **压缩效率：五家全部碾压 tokenizer_c 256K**。overall BPT 7.4881 → 4.78~5.20（降 30%~36%），尤其中文从 7.93 降至 4.13~5.49。tokenizer_c 256K 相对最弱的分项恰是五家最强的部分。
2. **相对 SP 基线：除 en 外全面占优**。zh 领先 36%~52%，code 领先 11%~15%；en 仅 doubao 追平 SP，其余四家落后约 7%（英文 BPE 词表对 SP 的优势不明显）。
3. **五家内部排序与是否引入 tokenizer_c 无关**：grok > doubao > gpt6 > Qwen > hunyuan；tokenizer_c 引入后只是把对比下限大幅拉低，不改变五家间相对位次。
4. **grok 在结构与符号上最接近 tokenizer_c 的设计取向**（整字优先、操作符隔离 0.9667），但压缩率远优；其余四家走字节级 BPE 路线，中文拆分为字节 piece 换来更高压缩率。
5. **tokenizer_c 256K 的独特优势**：中文整字覆盖（字符拆分率 0）、操作符隔离 0.9833、UTF-8 无损，适合"字符语义可解释"的应用；代价是 BPT 明显偏高（多语言语料上 7.4881）。

## 六、局限

- tokenizer_c 256K / SP 基线的稳定性与 TokEval 分项仅仓库报告口径，部分指标（如 SP 操作符隔离）仓库未提供，未列入。
- 吞吐跨实现（C CLI / C++ SP / Rust / tiktoken），不可直接比较。
- gpt6 为 gpt-oss-120b 替代源；opus 无公开源未纳入。
*（内容由AI生成，仅供参考）*
