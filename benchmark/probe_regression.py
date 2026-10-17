#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_regression.py —— tok_eval 19 子类探针语料的秒级分层回归（含指标 diff）。

定位：把 E:\\tok_eval 的探针语料与指标口径接到 tokenizer_c 的评测侧，回答
「这次改动到底改好了没有、有没有把别的子类搞坏」。

设计约束：
  * 口径不复制、只 import：per-subcategory 行由 tok_eval `app/metrics.py::evaluate_samples`
    计算（与 tok_eval 面板完全同源）；fairness_gini / composite_score 走
    benchmark/eval_thresholds.py（与 tok_eval 同公式，且权重/归一化区间可被
    thresholds.json 覆盖）。
  * 阈值与 diff 容差唯一来源是 benchmark/thresholds.json 的 `probe` 段。
  * 失败样本落盘补齐 tok_eval 的缺口：原文（全量）、解码结果、token 序列、token id、
    offset，并按根因输出归因占比。
  * 不改动被测 tokenizer 与 E:\\tok_eval 任何文件（只读）。

用法：
  python probe_regression.py --tokenizer E:\\tokenizer_c\\hf_export_big3_rev\\tokenizer.json \\
      [--label big3_rev] [--corpus-dir DIR] [--out-dir DIR] \\
      [--baseline benchmark/probe_baseline.json] [--update-baseline] [--strict]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eval_thresholds as ETH  # noqa: E402

EVAL_COLUMNS = [
    "category", "subcategory", "name", "sample_count", "samples_evaluated",
    "avg_tokens", "fertility", "fertility_grade", "compression_ratio", "byte_compression",
    "fidelity_pass_rate", "fidelity_fails", "avg_token_bytes", "avg_token_chars",
    "type_token_ratio", "vocabulary_utilization", "unique_tokens", "vocab_size",
    "subword_fertility", "continued_word_rate", "shannon_entropy",
    "renyi_entropy_alpha_0_5", "renyi_entropy_alpha_1", "renyi_entropy_alpha_2",
]


def _clean(m):
    return {k: v for k, v in m.items() if k != "failed_samples"}


def _rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(EVAL_COLUMNS), extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in EVAL_COLUMNS})
    return buf.getvalue()


def _load_tok_eval(root):
    """import tok_eval 的 corpus / metrics / tokenizer_loader 三个模块。"""
    if root not in sys.path:
        sys.path.insert(0, root)
    from app import corpus as corpus_mod
    from app import metrics as metrics_mod
    from app.tokenizer_loader import load_tokenizer_from_dir
    return corpus_mod, metrics_mod, load_tokenizer_from_dir


def _override_corpus_dir(corpus_mod, corpus_dir):
    hit = []
    for attr in ("BUILTIN_DIR", "BUILTIN_CORPUS_DIR", "CORPUS_DIR"):
        if hasattr(corpus_mod, attr):
            setattr(corpus_mod, attr, Path(corpus_dir))
            hit.append(attr)
    return hit


def collect_failed_samples(handle, metrics_mod, entries):
    """逐样本判定保真（口径用 tok_eval 的 _fidelity_ok），失败样本落全量证据。"""
    failures = []
    for e in entries:
        for idx, text in enumerate(e["samples"]):
            if text is None or not text.strip():
                continue
            try:
                enc = handle.encode_with_offsets(text, add_special_tokens=False)
                ids = list(enc["ids"])
                tokens = list(enc["tokens"])
                offsets = [list(o) for o in enc["offsets"]]
                decoded = handle.decode(ids)
                err = None
            except Exception as exc:  # noqa: BLE001
                ids, tokens, offsets, decoded, err = [], [], [], "", "%r" % (exc,)
            ok = (err is None) and metrics_mod._fidelity_ok(text, decoded, handle, ids)
            if ok:
                continue
            label, desc = ETH.classify_roundtrip_failure(text, decoded, err)
            failures.append({
                "category": e["category"], "subcategory": e["subcategory"],
                "sample_index": idx, "text": text, "decoded": decoded,
                "token_ids": ids, "tokens": tokens, "offsets": offsets,
                "text_bytes": len(text.encode("utf-8")), "n_tokens": len(ids),
                "attribution": label, "attribution_desc": desc,
            })
    return failures


def failed_samples_markdown(payload, limit=40):
    out = ["## 失败样本落盘", ""]
    out.append("- 落盘目录：`%s`" % payload["failed_samples_dir"])
    out.append("- 失败样本数：%d（明文 TXT 全量、JSON 全量、此处展示前 %d 条）"
               % (payload["failure_total"], min(limit, payload["failure_total"])))
    out.append("")
    out.append("### 归因占比")
    out.append("")
    out.append("| 归因 | 条数 | 占比 | 说明 |")
    out.append("|---|---|---|---|")
    for k, v in (payload["attribution"].get("counts") or {}).items():
        out.append("| %s | %d | %.1f%% | %s |" % (
            k, v, 100.0 * (payload["attribution"]["shares"].get(k) or 0.0),
            payload["attribution"]["descriptions"].get(k, "")))
    out.append("")
    out.append("### 失败样本明细（前 %d 条）" % min(limit, payload["failure_total"]))
    out.append("")
    for f in payload["failures"][:limit]:
        out.append("- **%s / %s #%d**（归因：%s，%d tokens）" % (
            f["category"], f["subcategory"], f["sample_index"], f["attribution"], f["n_tokens"]))
        out.append("  - 原文：`%s`" % _one_line(f["text"]))
        out.append("  - 解码：`%s`" % _one_line(f["decoded"]))
        out.append("  - tokens：`%s`" % _one_line(" | ".join(f["tokens"][:40])))
        out.append("  - offsets：`%s`" % _one_line(json.dumps(f["offsets"][:40])))
    return "\n".join(out)


def _one_line(s):
    return (s or "").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t").replace("`", "'")


def diff_against_baseline(rows, agg, baseline, tol):
    """逐子类 + 聚合指标 diff；劣化超容差 → REGRESSION。"""
    items = []
    base_rows = {r["subcategory"]: r for r in (baseline or {}).get("rows") or []}
    higher_better = {"fidelity_pass_rate"}
    for r in rows:
        b = base_rows.get(r["subcategory"])
        if not b:
            continue
        for metric, t in (tol or {}).items():
            if metric not in r or metric not in b:
                continue
            try:
                cur_v, old_v = float(r[metric]), float(b[metric])
            except (TypeError, ValueError):
                continue
            delta = cur_v - old_v
            regressed = (delta < -float(t)) if metric in higher_better else (delta > float(t))
            items.append({
                "scope": "subcategory", "subcategory": r["subcategory"], "metric": metric,
                "baseline": old_v, "current": cur_v, "delta": round(delta, 6),
                "tolerance": float(t), "regressed": bool(regressed),
                "direction": "higher_better" if metric in higher_better else "lower_better",
            })
    base_agg = (baseline or {}).get("agg") or {}
    for metric, t in (tol or {}).items():
        if metric not in agg or metric not in base_agg:
            continue
        cur_v, old_v = float(agg[metric]), float(base_agg[metric])
        delta = cur_v - old_v
        regressed = (delta < -float(t)) if metric in higher_better else (delta > float(t))
        items.append({
            "scope": "aggregate", "subcategory": "-", "metric": metric,
            "baseline": old_v, "current": cur_v, "delta": round(delta, 6),
            "tolerance": float(t), "regressed": bool(regressed),
            "direction": "higher_better" if metric in higher_better else "lower_better",
        })
    regressions = [i for i in items if i["regressed"]]
    return {"items": items, "regressions": regressions,
            "baseline_found": bool(baseline and (baseline.get("rows") or baseline.get("agg")))}


def render_diff_markdown(diff, limit=30, baseline_mode=None):
    out = ["## 与基线 diff", ""]
    if not diff["baseline_found"]:
        out.append("- 未找到可用基线：本次指标即为首版基线（用 `--update-baseline` 固化）。")
        return "\n".join(out)
    out.append("- 基线槽位（归一化模式）：`%s`" % (baseline_mode or "-"))
    out.append("- 判定：**%s**（劣化超容差 %d 项，共比对 %d 项）" % (
        "REGRESSION" if diff["regressions"] else "OK",
        len(diff["regressions"]), len(diff["items"])))
    out.append("")
    if diff["regressions"]:
        out.append("| 范围 | 子类 | 指标 | 基线 | 当前 | 变化 | 容差 |")
        out.append("|---|---|---|---|---|---|---|")
        for i in diff["regressions"][:limit]:
            out.append("| %s | %s | %s | %s | %s | %+g | %g |" % (
                i["scope"], i["subcategory"], i["metric"], i["baseline"], i["current"],
                i["delta"], i["tolerance"]))
    else:
        out.append("- 无劣化项。")
    return "\n".join(out)


def render_report(payload):
    s = payload["summary"]
    agg = payload["agg"]
    out = ["# 19 子类探针回归报告", ""]
    out.append("- 对象：`%s`" % payload["tokenizer"])
    out.append("- 语料：`%s`（%d 子类 / %d 样本）" % (
        payload["corpus_dir"], agg["subcategories"], agg["samples_evaluated"]))
    out.append("- 归一化模式：`%s`" % payload["norm_mode"])
    out.append("- 阈值来源：`%s`（版本 %s，sha256_16=%s）" % (
        payload["thresholds"]["path"], payload["thresholds"]["version"],
        payload["thresholds"]["sha256_16"]))
    out.append("- 耗时：%.2f s" % payload["elapsed_s"])
    out.append("")
    out.append("## 聚合指标")
    out.append("")
    out.append("| 指标 | 实测 |")
    out.append("|---|---|")
    out.append("| micro fidelity_pass_rate | %.4f |" % agg["micro_fidelity_pass_rate"])
    out.append("| micro fertility | %.4f |" % agg["micro_fertility"])
    out.append("| micro byte_compression | %.4f |" % agg["micro_byte_compression"])
    out.append("| macro fertility | %.4f |" % agg["macro_fertility"])
    out.append("| fairness_gini（8 人类语言 tokens/byte） | %.4f |" % s["fairness_gini"])
    out.append("| composite_score | %.2f |" % s["composite_score"])
    out.append("")
    out.append("## 分层明细（19 子类）")
    out.append("")
    out.append("| 大类 | 子类 | tokens | fertility | 分级 | byte_comp | BPT | 保真 | subword_fert | cont_word_rate |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in payload["rows"]:
        bc = float(r["byte_compression"] or 0.0)
        out.append("| %s | %s | %.1f | %.2f | %s | %.4f | %s | %.0f%% | %.2f | %.3f |" % (
            r["category"], r["subcategory"], float(r["avg_tokens"] or 0.0),
            float(r["fertility"] or 0.0), r["fertility_grade"], bc,
            ("%.2f" % (1.0 / bc)) if bc else "-",
            100.0 * float(r["fidelity_pass_rate"] or 0.0),
            float(r["subword_fertility"] or 0.0), float(r["continued_word_rate"] or 0.0)))
    out.append("")
    out.append(ETH.render_gates_markdown(payload["gates"], title="门禁判定（thresholds.json::probe）"))
    out.append("")
    out.append(render_diff_markdown(payload["diff"], baseline_mode=payload.get("baseline_mode")))
    out.append("")
    out.append(failed_samples_markdown(payload))
    out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="HF fast tokenizer.json 路径或 HF fast 目录")
    ap.add_argument("--label", default="")
    ap.add_argument("--corpus-dir", default="", help="覆盖探针语料目录（默认取 thresholds.json::probe.corpus_dir）")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 benchmark/probe_runs/<时间戳>-<label>）")
    ap.add_argument("--thresholds", default="", help="thresholds.json 路径")
    ap.add_argument("--norm-mode", default="",
                    help="被测导出物的归一化模式（nfkc|reversible），默认按路径/标签自动推断")
    ap.add_argument("--baseline", default="", help="基线 JSON 路径（默认 thresholds.json::probe.baseline）")
    ap.add_argument("--update-baseline", action="store_true", help="把本次结果写为基线")
    ap.add_argument("--strict", action="store_true",
                    help="门禁 FAIL 或出现 REGRESSION 时退出码 1（可直接进 CI）")
    ap.add_argument("--show-failures", type=int, default=10, help="stdout 展示的失败样本条数")
    args = ap.parse_args(argv)

    th = ETH.load_thresholds(args.thresholds or None)
    probe_cfg = th.get("probe") or {}
    tok_eval_root = probe_cfg.get("tok_eval_root") or r"E:\tok_eval"
    corpus_dir = args.corpus_dir or probe_cfg.get("corpus_dir") or ""
    baseline_path = os.path.abspath(args.baseline or probe_cfg.get("baseline")
                                    or os.path.join(_HERE, "probe_baseline.json"))
    label = args.label or os.path.splitext(os.path.basename(os.path.abspath(args.tokenizer)))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(args.out_dir or os.path.join(_HERE, "probe_runs", "%s-%s" % (stamp, label)))
    os.makedirs(out_dir, exist_ok=True)

    corpus_mod, metrics_mod, load_tokenizer_from_dir = _load_tok_eval(tok_eval_root)
    if corpus_dir:
        hit = _override_corpus_dir(corpus_mod, corpus_dir)
        if not hit:
            print("[FAIL] tok_eval corpus 模块未找到可覆盖的内置语料常量")
            return 2
        print("[probe] corpus dir -> %s" % corpus_dir)

    tok_path = os.path.abspath(args.tokenizer)
    if os.path.isfile(tok_path):
        # tok_eval 的 loader 以「目录」为单位加载（目录内需有 tokenizer.json）
        entry = Path(tok_path)
        if entry.name != "tokenizer.json":
            print("[probe] 提示：%s 不是 tokenizer.json，loader 将按其所在目录加载 tokenizer.json"
                  % entry.name)
        load_dir = entry.parent
    else:
        load_dir = Path(tok_path)
    handle = load_tokenizer_from_dir(load_dir)

    entries = corpus_mod.load_builtin_corpus()
    t0 = time.time()
    rows, all_samples = [], []
    for e in entries:
        all_samples.extend(e["samples"])
        m = _clean(metrics_mod.evaluate_samples(handle, e["samples"]))
        rows.append({"category": e["category"], "subcategory": e["subcategory"],
                     "name": e["name"], "sample_count": e["sample_count"], **m})

    pooled = _clean(metrics_mod.evaluate_samples(handle, all_samples))
    lang_cost = {r["subcategory"]: float(r["byte_compression"]) for r in rows
                 if r["category"] == "human_languages" and float(r["byte_compression"] or 0) > 0}
    gini = ETH.fairness_gini(lang_cost)
    composite = ETH.composite_score({
        "fertility": pooled.get("fertility", 0.0),
        "byte_compression": pooled.get("byte_compression", 0.0),
        "fidelity_pass_rate": pooled.get("fidelity_pass_rate", 0.0),
        "vocabulary_utilization": pooled.get("vocabulary_utilization", 0.0),
        "fairness_gini": gini,
    }, th.get("composite"))

    failures = collect_failed_samples(handle, metrics_mod, entries)
    attribution = ETH.summarize_attribution(failures)
    elapsed = time.time() - t0

    n_eval = sum(r["samples_evaluated"] for r in rows)
    n_pass = sum(r["samples_evaluated"] - r["fidelity_fails"] for r in rows)
    tt = sum(r["total_tokens"] for r in rows)
    tb = sum(r["total_bytes"] for r in rows)
    tw = sum(r["total_words"] for r in rows)
    agg = {
        "samples_evaluated": n_eval,
        "fidelity_pass": n_pass,
        "micro_fidelity_pass_rate": (n_pass / n_eval) if n_eval else 0.0,
        "micro_fertility": (tt / tw) if tw else 0.0,
        "micro_byte_compression": (tt / tb) if tb else 0.0,
        "macro_fertility": (sum(float(r["fertility"] or 0.0) for r in rows) / len(rows)) if rows else 0.0,
        "fairness_gini": gini,
        "composite_score": (composite or {}).get("score"),
        "subcategories": len(rows),
    }

    gate_metrics = dict(agg)
    gate_metrics["fairness_gini"] = gini
    gate_metrics["composite_score"] = (composite or {}).get("score")
    norm_mode = args.norm_mode or ("reversible" if "rev" in (label + tok_path).lower() else "nfkc")
    gates = ETH.evaluate_gates(gate_metrics, probe_cfg.get("gates") or [],
                              {"norm_mode": norm_mode})

    # 基线按归一化模式分槽存放：不同 norm 模式的保真度不可直接比较
    baseline_raw = {}
    if os.path.isfile(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as fh:
                baseline_raw = json.load(fh) or {}
        except (OSError, ValueError):
            baseline_raw = {}
    if isinstance(baseline_raw, dict) and isinstance(baseline_raw.get("modes"), dict):
        baseline = baseline_raw["modes"].get(norm_mode) or {}
        baseline_mode = norm_mode
    elif baseline_raw:
        # 兼容旧版单模式基线文件
        baseline = baseline_raw
        baseline_mode = baseline_raw.get("norm_mode") or "legacy"
    else:
        baseline, baseline_mode = {}, None
    diff = diff_against_baseline(rows, agg, baseline, probe_cfg.get("diff_tolerance"))

    payload = {
        "label": label, "tokenizer": tok_path, "corpus_dir": corpus_dir,
        "tok_eval_root": tok_eval_root, "out_dir": out_dir,
        "thresholds": ETH.thresholds_provenance(th),
        "info": handle.info(label),
        "summary": {**pooled, "fairness_gini": gini,
                    "composite_score": (composite or {}).get("score"),
                    "per_language_token_cost": lang_cost},
        "composite_parts": (composite or {}).get("parts"),
        "rows": rows, "agg": agg, "gates": gates, "diff": diff,
        "norm_mode": norm_mode, "baseline_mode": baseline_mode,
        "failure_total": len(failures), "failures": failures,
        "attribution": attribution, "failed_samples_dir": out_dir,
        "elapsed_s": round(elapsed, 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    io.open(os.path.join(out_dir, "probe_rows.csv"), "w", encoding="utf-8", newline="\n").write(
        _rows_to_csv(rows))
    io.open(os.path.join(out_dir, "probe_summary.json"), "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=2))
    report = render_report(payload)
    io.open(os.path.join(out_dir, "probe_report.md"), "w", encoding="utf-8", newline="\n").write(report)
    io.open(os.path.join(out_dir, "failed_samples.json"), "w", encoding="utf-8", newline="\n").write(
        json.dumps({"total": len(failures), "attribution": attribution, "failures": failures},
                   ensure_ascii=False, indent=2))
    with io.open(os.path.join(out_dir, "failed_samples.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for f in failures:
            fh.write("# [%s/%s #%d] %s\n" % (f["category"], f["subcategory"],
                                             f["sample_index"], f["attribution"]))
            fh.write("原文     : %s\n" % f["text"])
            fh.write("解码     : %s\n" % f["decoded"])
            fh.write("token id : %s\n" % f["token_ids"])
            fh.write("tokens   : %s\n" % f["tokens"])
            fh.write("offsets  : %s\n\n" % f["offsets"])

    print("[probe] label=%s samples=%d pass=%d fidelity=%.4f micro_fertility=%.4f "
          "micro_byte=%.4f gini=%.4f composite=%.2f failures=%d elapsed=%.2fs" % (
              label, n_eval, n_pass, agg["micro_fidelity_pass_rate"], agg["micro_fertility"],
              agg["micro_byte_compression"], gini, agg["composite_score"] or 0.0,
              len(failures), elapsed))
    print("[probe] %s" % ETH.render_gates_markdown(gates, title="门禁").replace("\n", "\n[probe] "))
    if diff["baseline_found"]:
        print("[probe] diff: %s（劣化 %d 项）" % ("REGRESSION" if diff["regressions"] else "OK",
                                                  len(diff["regressions"])))
    else:
        print("[probe] diff: 无基线（--update-baseline 可固化）")
    bad = [r["subcategory"] for r in rows if float(r["fidelity_pass_rate"] or 0) < 1.0]
    print("[probe] 未满保真子类: %s" % (", ".join(bad) or "无"))
    for f in failures[:max(0, args.show_failures)]:
        print("[probe][fail] %s#%d %s | 原文=%r 解码=%r" % (
            f["subcategory"], f["sample_index"], f["attribution"], f["text"][:60], f["decoded"][:60]))
    print("[probe] out=%s" % out_dir)

    if args.update_baseline:
        slot = {"label": label, "tokenizer": tok_path, "corpus_dir": corpus_dir,
                "norm_mode": norm_mode,
                "agg": agg, "rows": [{k: r[k] for k in (
                    "category", "subcategory", "fertility", "byte_compression",
                    "fidelity_pass_rate", "subword_fertility",
                    "continued_word_rate") if k in r} for r in rows],
                "generated_at": payload["generated_at"]}
        store = dict(baseline_raw) if isinstance(baseline_raw, dict) else {}
        modes = dict(store.get("modes") or {})
        if not modes and store:  # 旧版单模式文件 → 迁移为分槽结构
            modes[store.get("norm_mode") or "legacy"] = store
        modes[norm_mode] = slot
        new_payload = {"schema": "probe_baseline/2", "modes": modes,
                       "updated_at": payload["generated_at"]}
        d = os.path.dirname(baseline_path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        io.open(baseline_path, "w", encoding="utf-8", newline="\n").write(
            json.dumps(new_payload, ensure_ascii=False, indent=2))
        print("[probe] baseline=%s（槽位 %s，累计 %d 个模式）" % (
            baseline_path, norm_mode, len(modes)))

    if args.strict and (not gates.get("passed") or diff["regressions"]):
        print("[probe] strict: 门禁/回归未通过 → 退出码 1")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        print("[probe][ERROR] 未捕获异常：")
        print(traceback.format_exc())
        sys.exit(2)
