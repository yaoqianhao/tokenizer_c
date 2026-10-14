---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_46ef3fe9c79c11f18569525400638852
    ReservedCode1: 1VMFOMDUJjvzub/enOtLdH3JgE+6z4h9BiSzaNAVzrhNeHVKUZIjpeTEo0eMA0cLBrzd2pYYrnXkJ/eqhQsUQJrnxp3MQ6XbmniH33tLQl64jdYsHMH2FinLEMfbCnBfPu68Mj5ehHpRbpOdqTo0b391Dogp85+crpdt4e3LywlBDP46so8xQdOyB7o=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_46ef3fe9c79c11f18569525400638852
    ReservedCode2: 1VMFOMDUJjvzub/enOtLdH3JgE+6z4h9BiSzaNAVzrhNeHVKUZIjpeTEo0eMA0cLBrzd2pYYrnXkJ/eqhQsUQJrnxp3MQ6XbmniH33tLQl64jdYsHMH2FinLEMfbCnBfPu68Mj5ehHpRbpOdqTo0b391Dogp85+crpdt4e3LywlBDP46so8xQdOyB7o=
---

# model_big2 三模式归一化对比归档（nfkc / nfc / reversible）

- 归档日期：2026-10-14
- 模型：`E:\tokenizer_c\model_big2.tkm`（vocab_size=262144，md5=`057072bb1cabc9249395e8818bca1255`）
- CLI：`bin\tok-cli-norm.exe`（md5=`b8faf2e8ec0d7df6b7aca9c9dce2e6a9`）
- 统一模式开关：`--norm=nfkc|nfc|reversible`，`--norm-reversible` 为 `reversible` 的兼容别名
- 归档红线：本目录为**新建**目录，全过程未覆盖、未删除 `E:\tokenizer_c` 下任何既有文件；`build_norm/` 亦为新增构建目录，原 `build/`（tok-cli.exe / test_tokenizer.exe）未被触碰

## 1. 目录结构

| 路径 | 内容 |
|---|---|
| `reports\norm_mode_compare.md` | **主报告**：三模式并排指标（BPT 双口径 + 整语料、roundtrip、TokEval、stability、D/dispersion、吞吐/load_ms） |
| `reports\protocol_robustness_<mode>.md` | 逐模式协议鲁棒性报告 |
| `metrics\norm_mode_compare_summary.json` | 三模式并排汇总（机器可读） |
| `metrics\eval_metrics_model_big2_<mode>.json` | 逐模式原始评测结果 |
| `metrics\protocol_robustness_<mode>.json` | 逐模式协议段结果 |
| `metrics\wall_clock.json` | 逐模式评测墙钟（62.1 / 61.5 / 59.6 s） |
| `scripts\norm_mode_compare.py` | 三模式对比编排脚本（一键对比核心） |
| `scripts\run_compare_norm.ps1` | **一键复现入口**：编译 → 单测 → 三模式评测 → 并排报告 |
| `scripts\build_norm.ps1` | 独立构建脚本（产物落在 `E:\tokenizer_c\build_norm`） |
| `corpus_slices\eval_<lang>.txt` / `<lang>_real.txt` | 合成口径 / 真实口径切片（en / zh / code） |
| `logs\` | 构建、单测、逐模式运行、二进制等价性验证日志 |
| `bin\` | 本次评测所用 CLI 与单测可执行文件 |

## 2. 一键复现

```powershell
# 完整流程：编译 + 单测 + 三模式实跑 + 生成并排报告
powershell -ExecutionPolicy Bypass -File E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare\scripts\run_compare_norm.ps1

# 仅重渲染报告（复用已有 JSON，不重跑评测）
... \scripts\run_compare_norm.ps1 -SkipBuild -SkipRun

# 只对比部分模式
... \scripts\run_compare_norm.ps1 -Modes nfkc,nfc
```

底层等价调用（单模式）：

```
tok-cli-norm.exe encode --model E:\tokenizer_c\model_big2.tkm --file <corpus.txt> --norm=nfc
tok-cli-norm.exe decode --model E:\tokenizer_c\model_big2.tkm --ids <ids>
tok-cli-norm.exe bench  --model E:\tokenizer_c\model_big2.tkm --file <corpus.txt> --norm=reversible
```

## 3. 关键结论（同一语料、同一模型，仅归一化模式不同）

| 指标 | `nfkc`（默认，原路径） | `nfc`（新增） | `reversible`（原有） |
|---|---|---|---|
| zh roundtrip 失败行 | **85 / 200** | **0 / 200** | **0 / 200** |
| BPT 合成（zh） | 2.4851 | 2.4021 | 2.4021 |
| BPT 真实（zh） | 2.3746 | 2.1573 | 2.1569 |
| BPT 整语料 byte-weighted total | 3.5992 | 3.5489 | 3.5489 |
| D(zh) = BPT_eval / BPT_real | 1.0465 | 1.1135 | 1.1137 |
| dispersion(max-min) / verdict | 0.2069 / 稳定 | 0.2062 / 稳定 | 0.2061 / 稳定 |
| utf8_completeness | 0.8793 | 0.8675 | 0.8675 |
| char_split_rate | 0.1725 | 0.1725 | 0.1725 |
| digit_boundary_f1 | 0.4000 | 0.4000 | 0.4000 |
| digit_split_variability | 1.6603 | 1.6603 | 1.6603 |
| operator_isolation | 0.4333 | 0.4333 | 0.4333 |
| stability overall | 0.8554 | 0.8555 | 0.8555 |
| encode tokens/s | 1 033 146 | 1 089 587 | 1 129 320 |
| decode tokens/s | 52 679 | 59 875 | 60 969 |
| load_ms | 68.84 | 71.77 | 75.74 |

要点：

1. **可逆性**：NFKC 全角/兼容折叠不可逆，中文 roundtrip 85/200 行失败；NFC 与 reversible 均为 0/200。
2. **无损代价**：改用 NFC 后中文 BPT 略降（合成 -0.083、真实 -0.217），整语料 byte-weighted total 3.5992 → 3.5489（-0.0504），即"还原能力换少量压缩率"；en / code 基本无变化。
3. **统计其它指标不受影响**：char_split_rate、digit_boundary_f1、digit_split_variability、operator_isolation 三模式完全一致；stability 与 D/dispersion 均判"稳定"。
4. **吞吐反而提升**：nfc/reversible 的 encode 吞吐较 nfkc 高约 5.6 万 ~ 9.6 万 tokens/s（省去兼容折叠开销），load_ms 差异在毫秒级。
5. **nfc 与 reversible 的主要差异在"规范性单例与零宽字符"**：NFC 会把 U+212B 合成到 U+00C5 并剥离零宽字符；reversible 保留原码点，roundtrip 字节级最保真（详见 `logs\binary_equivalence_verify_2026-10-14.txt`，同长度 325/325 字节）。

## 4. 指标口径

- **BPT 双口径**：合成口径用 `corpus_slices\eval_<lang>.txt`，真实口径用 `corpus_slices\<lang>_real.txt`；另有整语料口径 `benchmark\corpus\<lang>\<lang>.txt`（bytes/tokens 实测）。
- **D = BPT_eval / BPT_real**，用于衡量"评测语料与真实语料的结论一致性"，dispersion 阈值 0.35（稳定）/ 1.0（敏感）。
- **roundtrip 失败行数**：decode 后字节与原输入不一致的行数，0 表示全还原。
- **throughput / load_ms**：由 CLI `bench` 同进程测得（encode 与 decode 同模型同 CLI）。

## 5. 质量证据

| 项 | 结果 | 证据 |
|---|---|---|
| 编译 | 通过（仅 trainer.c 既有警告，无 error） | `logs\build_norm_2026-10-14.log` |
| 单测回归 | `OK: all 186 checks passed`（原 96 项 + 模式相关新增项） | `logs\unit_test_2026-10-14.log` |
| 三模式实跑 | 三模式 rc=0，共 183.2 s | `logs\compare_run_*.log`、`metrics\wall_clock.json` |
| 归档 CLI 与源码重建一致性 | 三模式 encode ids 逐字节 md5 相同（archive = rebuild） | `logs\binary_equivalence_verify_2026-10-14.txt` |
| 默认行为不变 | `--norm` 不传时仍走原 NFKC 路径（nfkc 列即默认路径实测） | 报告 nfkc 列 |

> 说明：`bin\tok-cli-norm.exe`（b8faf2e8…）与 `build_norm\tok-cli-norm.exe`（574284db…）文件 md5 不同，原因是 PE 时间戳/构建路径差异；三模式实际编码输出 md5 完全一致，功能等价。

## 6. 代码侧改动位置（供后续提交参考）

- `cli\main.c`：`--norm=nfkc|nfc|reversible` 统一开关解析；`--norm-reversible` 保留为 reversible 别名（encode / bench / decode 路径）
- `src\nfc.c`：NFC 规范化实现（新增）
- `src\normalize.c`：模式分支（NFKC 原路径保留为默认；reversible 旁路）
- `src\tokenizer.c`、`include\tokenizer.h`：模式设置接口与编码入口接线
- 工作区改动尚未提交（生成报告时 git 未提交文件数=14），提交前请复核 diff。
*（内容由AI生成，仅供参考）*
