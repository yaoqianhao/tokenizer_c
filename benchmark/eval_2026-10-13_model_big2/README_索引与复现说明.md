---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_0341528dc6e211f1820a525400638852
    ReservedCode1: yPcnVGSvGsQxEcUEKRLIFKEL3lwIZFeeNnka5mzJyYTdMp39s2SSe35ycsPDeopU/vsiMjAcT0HDeNMUwG1sF2B6GSkw93yROEdIhI6dbh7K63S5DP2L6ZkTQ2reoMu0G2U+Lm2BOev3EcZMO1G8w3tetcvxsxdMiUQFsI/67Co59jZptE/Akt89CwQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_0341528dc6e211f1820a525400638852
    ReservedCode2: yPcnVGSvGsQxEcUEKRLIFKEL3lwIZFeeNnka5mzJyYTdMp39s2SSe35ycsPDeopU/vsiMjAcT0HDeNMUwG1sF2B6GSkw93yROEdIhI6dbh7K63S5DP2L6ZkTQ2reoMu0G2U+Lm2BOev3EcZMO1G8w3tetcvxsxdMiUQFsI/67Co59jZptE/Akt89CwQ=
---

# model_big2 Tokenizer 评测归档索引（2026-10-13）

## 0. 归档说明

- 本目录为 **2026-10-13 对 `model_big2` tokenizer 完整指标评测**的归档，目录名沿用项目既有日期式命名惯例（参考 `benchmark\eval_report_2026-10-10.md`）。
- 归档根目录：`E:\tokenizer_c\benchmark\eval_2026-10-13_model_big2`
- **非破坏性**：本次仅在 `benchmark` 下新增本目录，全部文件为复制件，未修改、未覆盖、未删除 `E:\tokenizer_c` 内任何已有文件；无重名冲突（无文件被改名）。
- 目录内共 **43 个文件**：本轮评测的脚本、加速版 CLI 及其补丁、构建日志、运行日志、全部原始指标 JSON 与评测报告。
- 未纳入归档的中间产物（可重新生成的语料切片、piece 表导出、encode 输出对照等，共 183 个文件约 132 MiB）见 §7。

## 1. 目录结构

```
eval_2026-10-13_model_big2/
├─ README_索引与复现说明.md      本文件（索引 / 运行命令 / 依赖 / 复现与验证记录）
├─ report/                       评测报告
├─ metrics/                      全部原始指标 JSON（报告数字的来源）
├─ scripts/                      本次评测与诊断脚本（run_eval_fast.ps1 为主评测脚本）
│  └─ probe_inputs/              CLI 探测/计时/诊断用的极小输入样本
├─ cli-fast/                     加速版 CLI（含补丁、改动后源码、构建脚本与构建日志、等价性验证记录）
└─ logs/                         三次评测的标准输出/错误与运行摘要、加载耗时原始输出
```

## 2. 文件清单

| 归档相对路径 | 大小 | 来源 | 说明 |
| --- | --- | --- | --- |
| `cli-fast/build_fast.err` | 1.8 KB | 会话 temp/ | 构建标准错误日志（含 gcc 告警） |
| `cli-fast/build_fast.log` | 183 | 会话 temp/ | 构建标准输出日志 |
| `cli-fast/build_fast_from_patch.ps1` | 2.9 KB | 本次归档时生成 | 可复现构建脚本：拷贝 E:\tokenizer_c 源码 + 覆盖 changed_src/ 后 gcc 编译 |
| `cli-fast/build_log_partial.txt` | 184 | 会话 temp/tkfast/ | 构建日志片段 |
| `cli-fast/changed_src/tokenizer.c` | 16.5 KB | 会话 temp/tkfast/src/ | 改动后完整源文件（唯一改动文件之一） |
| `cli-fast/changed_src/tokenizer_internal.h` | 3.0 KB | 会话 temp/tkfast/src/ | 改动后完整源文件（新增 cap_slots 字段） |
| `cli-fast/patch_tokenizer_fast_2026-10-13.patch` | 1.9 KB | 本次归档时生成 | 加速补丁 diff（相对项目原 src/，供审阅改动范围） |
| `cli-fast/tok-cli-fast.exe` | 254.8 KB | 会话 temp/tkfast/build/ | 本次评测实际使用的加速版 CLI（260,929 B，加载约 0.09 s） |
| `logs/eval_256k_stderr.log` | 0 | 会话 temp/ | 256k 评测标准错误（空文件） |
| `logs/eval_256k_stdout.log` | 2.4 KB | 会话 temp/ | 256k 评测标准输出 |
| `logs/eval_big2_stderr.log` | 0 | 会话 temp/ | big2 评测标准错误（空文件，无错误输出） |
| `logs/eval_big2_stdout.log` | 2.4 KB | 会话 temp/ | big2 评测标准输出 |
| `logs/eval_big_stderr.log` | 0 | 会话 temp/ | big 评测标准错误（空文件） |
| `logs/eval_big_stdout.log` | 2.4 KB | 会话 temp/ | big 评测标准输出 |
| `logs/eval_run_summary.txt` | 1.0 KB | 会话 temp/ | 运行摘要：三次评测 RC=0 与耗时（53 / 52.2 / 68.8 s）、原版与加速版加载耗时 |
| `logs/fastload_out.txt` | 14 | 会话 temp/ | 加速版 CLI 加载 model_big2 计时原始输出 |
| `logs/origload_out.txt` | 14 | 会话 temp/ | 原版 CLI 加载 model_big2 计时原始输出 |
| `metrics/cjk_coverage_nfkc.json` | 1.3 KB | 会话 temp/ | CJK 覆盖率与 NFKC 归一化影响统计 |
| `metrics/eval_metrics_model_256k.json` | 3.4 KB | 会话 output/ | benchmark\model_256k.tkm 原始指标 JSON（对照；报告已说明其数值不可信） |
| `metrics/eval_metrics_model_big.json` | 3.4 KB | 会话 output/ | 第一代 model_big 同口径原始指标 JSON（对照） |
| `metrics/eval_metrics_model_big2.json` | 3.4 KB | 会话 output/ | model_big2 原始指标 JSON（eval_metrics.py --output 直接产物） |
| `metrics/indomain_results2.json` | 2.7 KB | 会话 temp/ | 真实域内语料（en/zh/code_real）复核指标 |
| `metrics/metrics_consolidated.json` | 10.7 KB | 会话 output/ | 本项目 3 模型 + 5 家厂商基线 + 2026-10-10 历史基线拉平汇总（报告数字的来源） |
| `metrics/model_big2_评测补充证据.json` | 11.3 KB | 会话 output/ | 补充证据：逐线注入 / NFKC / ^ / Ġ / BOS / bos 等实验矩阵 |
| `metrics/roundtrip_rootcause.json` | 630 | 会话 temp/ | 中文还原率失败根因证据：NFKC 变体行数 36419/80694；A 组 60 行全败（全角逗号 U+FF0C→U+002C），B 组 60 行 0 失败 |
| `metrics/zh_composition.json` | 5.9 KB | 会话 temp/ | 中文语料 token 组成分解（CJK 单字/多字词/ASCII/字节回退等占比） |
| `report/model_big2_tokenizer_评测报告.md` | 20.0 KB | 会话 output/ | 本次评测主报告（含核心发现、全指标对比、根因分析） |
| `scripts/build_fast.ps1` | 1.2 KB | 会话 temp/ | 加速版 CLI 构建脚本（本次实际使用者，路径指向会话 temp） |
| `scripts/consolidate.py` | 4.9 KB | 会话 temp/ | 把本项目 eval JSON、厂商基线 JSON、历史报告统一拉平成一张表 |
| `scripts/diag3.py` | 4.2 KB | 会话 temp/ | 诊断脚本 3（token 组成 / 覆盖率分析） |
| `scripts/diag4.py` | 4.3 KB | 会话 temp/ | 诊断脚本 4（还原率差异定位） |
| `scripts/diag5.py` | 2.7 KB | 会话 temp/ | 诊断脚本 5（还原率根因：逐行 encode->decode 对照，产出 roundtrip_rootcause.json） |
| `scripts/indomain_compare.py` | 5.5 KB | 会话 temp/ | 域内语料复核（第一版，产出 indomain_results.json） |
| `scripts/indomain_compare2.py` | 5.9 KB | 会话 temp/ | 域内语料复核（第二版，产出 indomain_results2.json，报告采用） |
| `scripts/probe_inputs/emoji.txt` | 23 | 会话 temp/ | CLI 探测输入：emoji |
| `scripts/probe_inputs/invalid_bytes.bin` | 4 | 会话 temp/ | CLI 健壮性探测输入：非法 UTF-8 字节 |
| `scripts/probe_inputs/line.txt` | 240 | 会话 temp/clitmp/ | 还原率根因实验用单行样本 |
| `scripts/probe_inputs/probe_small.txt` | 52 | 会话 temp/ | CLI 单次调用耗时计时用极小输入 |
| `scripts/probe_inputs/rare.txt` | 13 | 会话 temp/ | CLI 探测输入：生僻字 |
| `scripts/run_eval.ps1` | 1.3 KB | 会话 temp/ | 加速前的原始评测尝试脚本（原版 CLI，用于对照） |
| `scripts/run_eval_fast.ps1` | 2.3 KB | 会话 temp/ | 评测主脚本：用加速版 CLI 依次跑 big2/big/256k 三个模型，并记录原版与加速版加载耗时 |

## 3. 受测模型、对照模型与语料

| 项 | 内容 |
| --- | --- |
| 受测模型 | `E:\tokenizer_c\model_big2.tkm`（3,689,538 B，mtime 2026-10-13 11:26:00）；配套 `model_big2.json`（7,294,078 B，同时间）、`model_big2.ckpt`（3,689,578 B，2026-10-13 11:24:50） |
| 模型规格 | vocab 262,144；Unigram + byte_fallback + NFKC + Metaspace（见评测报告首部） |
| 训练来源 | `E:\tokenizer_c\train_big2.py`（mtime 2026-10-12 21:09:39），训练语料 `E:\tokenizer_c\corpus_big\corpus_big2.txt`（15,006,950,149 B ≈ 14.0 GiB，mtime 2026-10-12 11:11:12；报告正文按其口径表述为 14.3 GB） |
| 对照模型 | 第一代 `E:\tokenizer_c\model_big.tkm`；`E:\tokenizer_c\benchmark\model_256k.tkm`（其评测数值报告已说明不可信） |
| 厂商基线 | `benchmark\{qwen_1.5B,qwen_7B,grok,doubao,gpt6,hunyuan}_eval.json`（项目已有，未复制；汇总结果见 `metrics/metrics_consolidated.json`） |
| 评测语料（合成） | `E:\tokenizer_c\benchmark\corpus\{en,zh,code}`，各 1 个文件：15,900,854 B / 10,566,465 B / 10,641,181 B |
| 域内复核语料 | 会话临时目录下 `en_real` / `zh_real` / `code_real`（各 2 MB 切片，**未归档**，见 §7）；复核结果已落盘为 `metrics/indomain_results2.json` |
| 评测脚本 | `E:\tokenizer_c\benchmark\eval_metrics.py`（项目原有，mtime 2026-10-11 10:39:35，38,201 B；未复制，属项目既有文件） |

## 4. 运行命令（复现步骤）

```powershell
# 0) 前置：项目已构建出原版 CLI（E:\tokenizer_c\build\tok-cli.exe，未打补丁）

# 1) 构建评测用加速版 CLI（可复现脚本，见 cli-fast\build_fast_from_patch.ps1）
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\cli-fast\build_fast_from_patch.ps1 -ProjectRoot "E:\tokenizer_c" -OutDir "<临时目录>"
#    产出 <临时目录>\build\tok-cli-fast.exe

# 2) 跑三个模型的全套指标（scripts\run_eval_fast.ps1 内的实际执行形态）
cd E:\tokenizer_c
python benchmark\eval_metrics.py --model model_big2.tkm `
       --corpus-dir benchmark\corpus --cli "<...>\tok-cli-fast.exe" --output "<...>\eval_metrics_model_big2.json"
#    对照：--model model_big.tkm / --model benchmark\model_256k.tkm
#    实测耗时：53 s / 52.2 s / 68.8 s（RC=0，见 logs\eval_run_summary.txt）

# 3) 三模型 + 厂商基线拉平汇总（产出 metrics_consolidated.json）
python .\scripts\consolidate.py

# 4) 辅助诊断（可选；输入样本在 scripts\probe_inputs\）
python .\scripts\indomain_compare2.py     # 域内语料复核 -> indomain_results2.json
python .\scripts\diag5.py                 # 还原率根因定位 -> roundtrip_rootcause.json
```

> 注意：`scripts\` 与 `cli-fast\build_fast.ps1` 中的绝对路径来自本次会话临时目录，复现时需替换为本机路径；`cli-fast\build_fast_from_patch.ps1` 已设计为可直接指定 `-ProjectRoot` / `-OutDir`，无需改脚本。

## 5. 依赖与环境

| 依赖 | 版本 / 说明 |
| --- | --- |
| Python | 3.11.8（`benchmark\eval_metrics.py` 仅用标准库；`sentencepiece` 仅在评测 `.model` 格式统计 OOV 时需要，本次为 `.tkm` 不依赖） |
| PowerShell | 5.1.26100.9444 |
| gcc | 16.2.0（MinGW-W64 x86_64-ucrt-posix-seh，r1），另需 `ar` |
| CLI | `tok-cli-fast.exe`（260,929 B，本目录 `cli-fast\`）或项目原版 `build\tok-cli.exe` |
| 随机性 | `eval_metrics.py` 抽样由固定随机种子驱动，抽样行可复现 |

## 6. 可复现性验证记录

详见 `cli-fast\等价性验证_2026-10-13.txt`，要点：

1. **加速版语义等价**（评测前完成）：对 `benchmark\corpus\{en,zh,code}` 各 1.5 MB 切片，原版 `tok-cli.exe` 与加速版 `tok-cli-fast.exe` 的 encode 输出逐字节一致。
2. **归档可复现**（归档时执行）：按本目录 `cli-fast\build_fast_from_patch.ps1` 从项目现源码 + `changed_src\` 重新编译，重建 exe 与归档 exe 尺寸一致（260,929 B），对同一三语切片 encode 输出逐字节一致。
3. 加速补丁**仅**改 piece 表扩容策略（倍增 + `cap_slots`），encode/decode 算法与 `.tkm` 解析逻辑未动，改动全文见 `cli-fast\patch_tokenizer_fast_2026-10-13.patch`。
4. **加载耗时对照**：原版 65.3 / 66.3 s（model_big2）、71.1 / 72.1 s（model_big）、66.9 / 73.4 s（model_256k）；加速版 0.093 / 0.089 / 0.107 s。

## 7. 未纳入归档的中间产物（可重新生成，故不复制）

位于本次会话临时目录，共 183 个文件、约 132 MiB：

- `slc\`、`indomain\`、`diff\`：语料切片（1.5–2 MB × 9）与加速版/原版 encode 输出对照（约 34 MiB）——切片可由 `benchmark\corpus` 前 N 字节重建；
- `v_*.json`、`bisect_*.json`、`f_*.json`、`w_*.json`、`t_*.json`、`u_*.json`、`syn*_*.json`、`canonical.json` 等：piece 表导出与逐轮实验中间态（约 70 MiB）；
- `tkfast\`（加速版完整工作副本，仅保留 `tok-cli-fast.exe`、改动后源码与构建日志）、`evallogs\`、`clitmp\`、`.python-scripts\`、`.tool-results\`：构建/执行环境临时文件；
- `rebuild_check\`：本次归档时的复现性验证构建目录（一次性）。

## 8. 关键结论摘要（完整内容见 report/）

1. 二代 `model_big2` 与一代 `model_big` 相比指标基本原地踏步：整体 BPT 3.5992 vs 3.6036（-0.12%），中文还原率同样 85/200 行失败。
2. 在合成语料 `benchmark\corpus` 上整体 BPT（3.599）优于 5 家厂商基线（4.777–5.199），但优势主要来自英文。
3. 换真实域内语料后优势缩小甚至反转：英文基本持平，代码落后 9.2%–20.5%，**中文落后 35.6%–47.7%**（2.375 vs 3.690–4.538）。
4. 中文还原率失败根因是 **NFKC 归一化**：原始文本中 42.9% 相关行含 NFKC 变体（36,419/80,694），全角逗号 `U+FF0C` 被归一化为 `U+002C`（证据：`metrics/roundtrip_rootcause.json`）。
5. `model_256k` 的同口径评测数值不可信（报告已说明），引用其历史数据时需注意。

## 9. 注意事项

- `report\model_big2_tokenizer_评测报告.md` 文件头带有 AIGC 合规标注块（Label/ContentProducer 等），为生成时的合规元信息。
- 本目录所有 JSON 均为评测脚本真实产出，未人工修改；`metrics_consolidated.json` 为报告数字的唯一汇总口径。
*（内容由AI生成，仅供参考）*
