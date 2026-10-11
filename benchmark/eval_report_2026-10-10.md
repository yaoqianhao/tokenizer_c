---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_2fd138fac50911f1820a525400638852
    ReservedCode1: Uq9JhSrgjxVetJHqd7yXMuGC5jr0pJHXYHa66+1DWuIs6DOx/FaAHy2VTNp/1fA5j5gayvYbLg7xwyLWd/LskIKixI8Fxh6d0N0R4IQUR42H62cpZfUU2rB0OfhDoT4VWu8V4PgmITjI9hzJ5SINXYM90/fGJhbgDOKcOYomLmkQoAU479YUHsuRkl4=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_2fd138fac50911f1820a525400638852
    ReservedCode2: Uq9JhSrgjxVetJHqd7yXMuGC5jr0pJHXYHa66+1DWuIs6DOx/FaAHy2VTNp/1fA5j5gayvYbLg7xwyLWd/LskIKixI8Fxh6d0N0R4IQUR42H62cpZfUU2rB0OfhDoT4VWu8V4PgmITjI9hzJ5SINXYM90/fGJhbgDOKcOYomLmkQoAU479YUHsuRkl4=
---

# Tokenizer Evaluation Report
- model: C:\Users\neroy\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_a043481fd20241bca905dfda23dc693d\output\tokenizer_c\benchmark\model_256k.tkm
- vocab_size: 262144
- date: 2026-10-10

| lang | files | bytes | tokens | BPT | chars/token | roundtrip | stability | oov% | bytefb% |
|---|---|---|---|---|---|---|---|---|---|
| code | 1 | 10485773 | 1453791 | 7.2127 | 7.2127 | OK | 0.6284 | - | - |
| en | 1 | 15728729 | 2125836 | 7.3988 | 7.3988 | OK | 0.6632 | - | - |
| zh | 1 | 10485771 | 1321548 | 7.9345 | 2.6855 | FAIL(85) | 0.8354 | - | - |
| **total** | | 36700273 | 4901175 | 7.4881 | | | | | |

## Throughput
- encode: 90310.8 tokens/sec
- decode: 7293.1 tokens/sec
- load: 479.9 ms

## TokEval metrics (arXiv 2608.18062)
| lang | utf8_completeness | char_split_rate | boundary_crossing | digit_f1 | digit_variability | operator_isolation |
|---|---|---|---|---|---|---|
| code | 1.0000 | - | 0.0000 | 0.5000 | 0.9658 | 1.0000 |
| en | 1.0000 | - | 0.0000 | - | - | 0.9667 |
| zh | 1.0000 | 0.0000 | 0.0000 | - | - | - |
| **overall** | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9658 | 0.9833 |

Report JSON: benchmark\eval_report_2026-10-10.json
*（内容由AI生成，仅供参考）*
