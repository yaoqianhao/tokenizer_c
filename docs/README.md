# 文档索引（docs/）

本目录存放 tokenizer_c 的规格、评测口径与流水线文档，随代码一起纳入版本控制；文内相对链接均指向仓库内已存在的文件。

## 本目录文档

| 文档 | 主题 | 对应代码 / 产物 |
| --- | --- | --- |
| [ttbpe_spec.md](ttbpe_spec.md) | tiktoken 兼容 byte-level BPE 模块规格：词表格式（`.tok.json` / `.tiktoken`）、rank 即 id、字节级 BPE、预切分近似、special token、解码 `errors="replace"`、CLI `tt-*` 子命令与回归用例 | [../include/tiktoken_bpe.h](../include/tiktoken_bpe.h)、[../src/tiktoken_bpe.c](../src/tiktoken_bpe.c)、[../tests/test_tiktoken_bpe.c](../tests/test_tiktoken_bpe.c)、[../tests/test_cli_tt_raw.c](../tests/test_cli_tt_raw.c) |
| [tokenizer_evaluation_standard.md](tokenizer_evaluation_standard.md) | 评测口径与门槛标准：BPT、还原率、稳定性、OOV、吞吐等指标的定义与达标线 | [../benchmark/eval_metrics.py](../benchmark/eval_metrics.py) |
| [eval_thresholds.md](eval_thresholds.md) | 阈值门禁表与归因词典（由 `benchmark/gen_thresholds_doc.py` 从 `benchmark/thresholds.json` 生成，请勿手改） | [../benchmark/thresholds.json](../benchmark/thresholds.json)、[../benchmark/gen_thresholds_doc.py](../benchmark/gen_thresholds_doc.py)、[../benchmark/probe_regression.py](../benchmark/probe_regression.py) |
| [hf_export_pipeline.md](hf_export_pipeline.md) | HuggingFace 导出流水线：产物目录结构、校验方式与上传流程 | [../tools/hf_export.py](../tools/hf_export.py)、[../tools/hf_pipeline.py](../tools/hf_pipeline.py) |

## 仓库其它位置的文档

- [../README.md](../README.md) — 项目总览、构建方式、CLI 用法、评测与验收门槛。
- [../2026-10-09-tokenizer-design.md](../2026-10-09-tokenizer-design.md) — 核心设计文档（需求、架构、算法、接口、里程碑）。
- [../2026-10-11-code-review-fix-plan.md](../2026-10-11-code-review-fix-plan.md) — 代码审阅结论与修复计划（P0–P2 问题清单）。
- [../2026-10-13-improvement-plan.md](../2026-10-13-improvement-plan.md) — 改进计划（算法与评测方向）。
- [../2026-10-14-tokenizer-optimization-plan-papers.md](../2026-10-14-tokenizer-optimization-plan-papers.md) — 优化方案与论文梳理。
- [../2026-10-14-tokenizer-beginner-guide.md](../2026-10-14-tokenizer-beginner-guide.md) — 新人导览。
- [../fix_patches.md](../fix_patches.md) — 历史补丁记录。
- `../benchmark/*.md`、`../benchmark/eval_*/` — 各轮评测报告与复现说明。

## 维护约定

- 新增文档：放在 `docs/` 并在上表登记；文内引用一律使用仓库内相对路径，避免指向仓库外或不存在的文件。
- 生成型文档：`eval_thresholds.md` 由脚本生成，修改阈值请改 `benchmark/thresholds.json` 后重新生成。
