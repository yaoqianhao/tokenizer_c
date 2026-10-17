#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_thresholds_doc.py —— 从 benchmark/thresholds.json 生成 docs/eval_thresholds.md。

单一事实来源原则：阈值只在 thresholds.json 里写一次，本脚本负责把「门禁表 + 归因词典
+ 综合分权重」渲染成人类可读文档。禁止手改生成出来的 md（脚本会在文件头写明）。

用法：
  python gen_thresholds_doc.py [--thresholds PATH] [--out PATH] [--check]
    --check  只比对不写入；文档与配置不一致时退出码 1（可进 CI）
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eval_thresholds as ETH  # noqa: E402


def _fmt(v):
    if isinstance(v, float):
        return ("%g" % v)
    return str(v)


def _cond_text(cond):
    if not cond:
        return "-"
    parts = []
    if "norm_mode" in cond:
        parts.append("norm_mode = %s" % cond["norm_mode"])
    if "norm_mode_in" in cond:
        parts.append("norm_mode ∈ {%s}" % ", ".join(cond["norm_mode_in"]))
    if "cli_extra_contains" in cond:
        need = cond["cli_extra_contains"]
        parts.append("cli-extra 含 %s" % (need if isinstance(need, str) else " + ".join(need)))
    return "；".join(parts) or "-"


def _gate_tables(section):
    out = ["| # | 门禁指标 | 判定 | 阈值 | 生效条件 | 说明 |", "|---|---|---|---|---|---|"]
    for i, g in enumerate(section.get("gates") or [], 1):
        out.append("| %d | `%s` | %s | %s | %s | %s |" % (
            i, g.get("key"), g.get("op"), _fmt(g.get("value")),
            _cond_text(g.get("applies_when")), g.get("note") or ""))
    if not section.get("gates"):
        out.append("| - | （无） | - | - | - | - |")
    return out


def render(th):
    out = []
    out.append("<!-- 本文件由 benchmark/gen_thresholds_doc.py 从 benchmark/thresholds.json 生成，请勿手改。 -->")
    out.append("")
    out.append("# tokenizer_c 评测阈值与门禁（自动生成）")
    out.append("")
    out.append("- 配置源：`%s`" % (th.get("_path") or "-"))
    out.append("- 配置版本：`%s`（updated_at=%s，sha256_16=%s）" % (
        th.get("version"), th.get("updated_at"), th.get("_sha256_16")))
    out.append("- 生成方式：`python benchmark/gen_thresholds_doc.py`（`--check` 可校验文档与配置是否一致）")
    out.append("")
    out.append("## 一、阈值单点化说明")
    out.append("")
    for n in th.get("notes") or []:
        out.append("- %s" % n)
    out.append("")
    cb = th.get("calibrated_by") or {}
    if cb:
        out.append("### 阈值标定出处")
        out.append("")
        out.append("| 项 | 内容 |")
        out.append("|---|---|")
        for k in ("date", "model", "cli", "method"):
            if cb.get(k):
                out.append("| %s | %s |" % (k, cb[k]))
        if cb.get("probe_tokenizers"):
            out.append("| probe_tokenizers | %s |" % "；".join(cb["probe_tokenizers"]))
        out.append("")

    out.append("## 二、综合分（composite_score）口径")
    out.append("")
    comp = th.get("composite") or {}
    w = comp.get("weights") or ETH.TOK_EVAL_COMPOSITE_WEIGHTS
    norm = comp.get("norm") or ETH.TOK_EVAL_COMPOSITE_NORM
    out.append("| 子项 | 权重 |")
    out.append("|---|---|")
    for k, v in w.items():
        out.append("| %s | %s |" % (k, _fmt(v)))
    out.append("")
    out.append("归一化区间：fertility %s→%s（好→差），byte_compression %s→%s；"
               "fidelity/vocab_util/fairness 直接映射到 [0,1]。"
               % (_fmt(norm.get("fertility_good")), _fmt(norm.get("fertility_bad")),
                  _fmt(norm.get("byte_good")), _fmt(norm.get("byte_bad"))))
    out.append("")
    out.append("- 公式来源：%s" % (comp.get("source") or "tok_eval app/metrics.py::composite_score"))
    out.append("- 代码位置：`benchmark/eval_thresholds.py::composite_score`（主评测与探针共用）")
    out.append("")

    g = th.get("grades") or {}
    if g:
        out.append("### fertility 分级口径")
        out.append("")
        out.append("| 分级 | 上界（tokens/word，越低越好） |")
        out.append("|---|---|")
        for k in ("excellent", "good", "fair", "poor"):
            if k in g:
                out.append("| %s | < %s |" % (k, _fmt(g[k])))
        out.append("")

    out.append("## 三、语料级主评测门禁（`eval_metrics.py`）")
    out.append("")
    ev = th.get("eval") or {}
    out.append("- 适用范围：%s" % (ev.get("scope") or "-"))
    out.append("")
    out.extend(_gate_tables(ev))
    out.append("")

    out.append("## 四、19 子类探针回归门禁（`probe_regression.py`）")
    out.append("")
    pr = th.get("probe") or {}
    out.append("- 适用范围：%s" % (pr.get("scope") or "-"))
    out.append("- tok_eval 根目录：`%s`" % (pr.get("tok_eval_root") or "-"))
    out.append("- 探针语料目录：`%s`" % (pr.get("corpus_dir") or "-"))
    out.append("- 基线文件：`%s`" % (pr.get("baseline") or "-"))
    out.append("")
    out.extend(_gate_tables(pr))
    out.append("")
    tol = pr.get("diff_tolerance") or {}
    if tol:
        out.append("### 指标 diff 容差（劣化超容差 → REGRESSION）")
        out.append("")
        out.append("| 指标 | 容差 |")
        out.append("|---|---|")
        for k, v in tol.items():
            if k == "note":
                continue
            out.append("| %s | %s |" % (k, _fmt(v)))
        out.append("")
        if tol.get("note"):
            out.append("- %s" % tol["note"])
            out.append("")

    out.append("## 五、协议鲁棒性判定阈值（`eval_metrics.py`）")
    out.append("")
    pth = th.get("protocol") or {}
    if pth:
        out.append("- 适用范围：%s" % (pth.get("scope") or "-"))
        out.append("")
        out.append("| 判定 | 条件（dispersion） | 阈值来源 |")
        out.append("|---|---|---|")
        out.append("| 稳定 | ≤ %s | `thresholds.json::protocol.stable_max` |" % _fmt(pth.get("stable_max")))
        out.append("| 敏感 | (%s, %s] | `thresholds.json::protocol.sensitive_max` |"
                   % (_fmt(pth.get("stable_max")), _fmt(pth.get("sensitive_max"))))
        out.append("| 口径依赖 | > %s | 同上 |" % _fmt(pth.get("sensitive_max")))
        out.append("")
        if pth.get("note"):
            out.append("- %s" % pth["note"])
            out.append("")
    else:
        out.append("- （配置中未声明 protocol 段，脚本回退内置默认值 0.35 / 1.0）")
        out.append("")

    out.append("## 六、失败样本归因词典（`eval_thresholds.py`）")
    out.append("")
    out.append("| 归因标签 | 含义 |")
    out.append("|---|---|")
    for k, v in ETH._ATTR_DESC.items():
        out.append("| `%s` | %s |" % (k, v))
    out.append("")
    out.append("## 七、改动指引")
    out.append("")
    out.append("1. 改阈值 → 只改 `benchmark/thresholds.json`，然后重跑本脚本让文档同步；")
    out.append("2. 跑主评测门禁 → `python benchmark/eval_metrics.py --model ... --corpus-dir ...`；")
    out.append("3. 跑探针回归门禁 → `python benchmark/probe_regression.py --tokenizer <tokenizer.json> --strict`；")
    out.append("4. 文档与配置一致性校验 → `python benchmark/gen_thresholds_doc.py --check`。")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="", help="thresholds.json 路径")
    ap.add_argument("--out", default="", help="输出 md 路径（默认 <repo>/docs/eval_thresholds.md）")
    ap.add_argument("--check", action="store_true", help="只校验不写入；不一致则退出码 1")
    args = ap.parse_args(argv)

    th = ETH.load_thresholds(args.thresholds or None)
    text = render(th)
    out_path = os.path.abspath(args.out or os.path.join(os.path.dirname(_HERE), "docs",
                                                        "eval_thresholds.md"))
    if args.check:
        old = ""
        if os.path.isfile(out_path):
            with io.open(out_path, "r", encoding="utf-8") as fh:
                old = fh.read()
        if old.strip() == text.strip():
            print("[doc] OK 文档与配置一致：%s" % out_path)
            return 0
        print("[doc] MISMATCH 文档与 thresholds.json 不一致：%s" % out_path)
        print("[doc] 修复：python benchmark/gen_thresholds_doc.py")
        return 1
    d = os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("[doc] 已生成 %s（源：%s，版本 %s）" % (out_path, th.get("_path"), th.get("version")))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
