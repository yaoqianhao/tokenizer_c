#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三模式归一化一键对比：nfkc / nfc / reversible。

同一语料、同一模型（model_big2.tkm）、同一套指标，并排输出：
  - BPT 两口径（合成切片 eval_<lang>.txt / 真实切片 <lang>_real.txt）
  - 整语料 byte-weighted bpt_total 与三语种等权 macro
  - D = BPT_eval / BPT_real 与 dispersion（max-min）、verdict
  - roundtrip 失败行数
  - TokEval：utf8_completeness / char_split_rate / boundary_crossing /
    digit_boundary_f1 / digit_split_variability / operator_isolation
  - stability 与 encode/decode 吞吐、load_ms

实现方式：对每个模式以子进程调用 benchmark/eval_metrics.py，通过
--cli-extra "--norm=<mode>" 把统一模式开关注入 encode/bench 调用，
再由本脚本聚合三份 JSON 生成并排 Markdown 报告与汇总 JSON。

零参数即可运行（一键复现）：
    python norm_mode_compare.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

LANGS = ("en", "zh", "code")
MODES = ("nfkc", "nfc", "reversible")

# 归档目录：本脚本位于 <archive>/scripts/ 下
ARCHIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = r"E:\tokenizer_c"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _f(v, nd=4):
    """数值格式化，None -> '-'。"""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return "%.*f" % (nd, v)
    return str(v)


def _i(v):
    return "-" if v is None else str(v)


def _d(vb, va, nd=4):
    """va 相对 vb 的变化量（va - vb）。"""
    if isinstance(vb, (int, float)) and isinstance(va, (int, float)):
        return "%+.*f" % (nd, va - vb)
    return "-"


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def md5_of(path, chunk=1 << 20):
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def git_info(root):
    info = {}
    try:
        info["rev"] = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
        st = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        info["dirty_files"] = len([l for l in st.splitlines() if l.strip()])
    except Exception:
        info["rev"] = None
        info["dirty_files"] = None
    return info


# ---------------------------------------------------------------------------
# 单模式评测（调用 eval_metrics.py）
# ---------------------------------------------------------------------------

def run_one_mode(args, mode, env):
    """对单个模式跑 eval_metrics.py，返回 (json_report, stdout_text, elapsed_s)。"""
    em = os.path.join(args.root, "benchmark", "eval_metrics.py")
    out_json = os.path.join(args.outdir, "metrics", "eval_metrics_model_big2_%s.json" % mode)
    proto_json = os.path.join(args.outdir, "metrics", "protocol_robustness_%s.json" % mode)
    proto_md = os.path.join(args.outdir, "reports", "protocol_robustness_%s.md" % mode)
    log_out = os.path.join(args.outdir, "logs", "eval_%s_stdout.log" % mode)
    log_err = os.path.join(args.outdir, "logs", "eval_%s_stderr.log" % mode)

    cmd = [
        args.python, em,
        "--model", args.model,
        "--corpus-dir", args.corpus_dir,
        "--cli", args.cli,
        "--cli-extra=--norm=%s" % mode,
        "--output", out_json,
        "--protocol-slices-dir", args.slices_dir,
        "--protocol-full-corpus-dir", args.corpus_dir,
        "--protocol-label", "%s/%s" % (args.label, mode),
        "--protocol-out", proto_json,
        "--protocol-md", proto_md,
    ]
    print("[RUN] %s" % " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=args.root, env=env, capture_output=True)
    elapsed = time.perf_counter() - t0
    out_text = proc.stdout.decode("utf-8", "replace")
    err_text = proc.stderr.decode("utf-8", "replace")
    with open(log_out, "w", encoding="utf-8") as fh:
        fh.write(out_text)
    with open(log_err, "w", encoding="utf-8") as fh:
        fh.write(err_text)
    print("[DONE] mode=%s rc=%d elapsed=%.1fs -> %s" % (mode, proc.returncode, elapsed, out_json))
    if proc.returncode != 0:
        raise RuntimeError("eval_metrics.py 失败（mode=%s, rc=%d）：\n%s" % (mode, proc.returncode, err_text[-2000:]))

    with open(out_json, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    report["_elapsed_s"] = round(elapsed, 1)
    return report


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def summarize(mode, rpt):
    """从单模式报告抽出对比所需字段（全部为真实实测值，缺失记 None）。"""
    langs = sorted(get(rpt, "langs", default={}).keys())
    proto = get(rpt, "protocol_robustness", default={}) or {}
    per_lang = get(proto, "per_lang", default={}) or {}
    agg = get(proto, "aggregate", default={}) or {}

    bpt_synth = {l: get(per_lang, l, "eval", "bpt") for l in LANGS}
    bpt_real = {l: get(per_lang, l, "real", "bpt") for l in LANGS}
    rt = {}
    for l in langs:
        rt[l] = {
            "ok": get(rpt, "langs", l, "roundtrip_ok"),
            "mismatches": get(rpt, "langs", l, "roundtrip_mismatches"),
            "lines": get(rpt, "langs", l, "roundtrip_lines"),
        }
    rt_fail_total = sum(v["mismatches"] or 0 for v in rt.values()) if rt else None

    return {
        "mode": mode,
        "cli_extra": get(rpt, "cli_extra", default=[]),
        "vocab_size": rpt.get("vocab_size"),
        "elapsed_s": rpt.get("_elapsed_s"),
        "totals": rpt.get("totals"),
        "bpt_synth_per_lang": bpt_synth,
        "bpt_real_per_lang": bpt_real,
        "bpt_corpus_per_lang": agg.get("per_lang_bpt"),
        "bpt_total_byte_weighted": agg.get("bpt_total_byte_weighted"),
        "bpt_macro": agg.get("bpt_macro"),
        "aggregate_source": agg.get("source"),
        "D": get(proto, "D"),
        "dispersion": get(proto, "dispersion"),
        "verdict": get(proto, "verdict"),
        "roundtrip": rt,
        "roundtrip_fail_total": rt_fail_total,
        "tokeval": rpt.get("tokeval"),
        "stability": rpt.get("stability"),
        "throughput": rpt.get("throughput"),
        "langs": langs,
    }


def render_markdown(sums, meta):
    """把三模式摘要渲染成并排 Markdown。"""
    order = [m for m in MODES if m in sums]
    base = order[0] if order else None
    L = []
    L.append("# tokenizer_c 三模式归一化对比报告（nfkc / nfc / reversible）")
    L.append("")
    L.append("- 生成时间：%s" % meta["generated_at"])
    L.append("- 模型：`%s`（vocab_size=%s, md5=%s）" % (
        meta["model"], meta["vocab_size"], meta["model_md5"]))
    L.append("- CLI：`%s`（md5=%s）" % (meta["cli"], meta["cli_md5"]))
    L.append("- 语料（整语料口径）：`%s`" % meta["corpus_dir"])
    L.append("- 切片（合成/真实两口径）：`%s`" % meta["slices_dir"])
    L.append("- 统一开关：`--norm=nfkc|nfc|reversible`（`--norm-reversible` 为 reversible 别名）")
    L.append("- 编排脚本：`%s`" % meta["script"])
    L.append("- eval_metrics.py：`%s`（md5=%s）" % (meta["eval_metrics"], meta["eval_metrics_md5"]))
    L.append("- git：rev=%s，未提交文件数=%s" % (meta["git"]["rev"], meta["git"]["dirty_files"]))
    L.append("")
    L.append("模式列顺序：%s；差值列 = %s 相对 %s 的变化。" % (
        " / ".join("`%s`" % m for m in order), order[1] if len(order) > 1 else "-", base))
    L.append("")

    def table(title, rows, note=None):
        L.append("## %s" % title)
        if note:
            L.append("")
            L.append(note)
        L.append("")
        head = ["指标"] + ["`%s`" % m for m in order]
        if len(order) > 1:
            head.append("Δ(%s-%s)" % (order[1], base))
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for name, getter in rows:
            vals = [getter(sums[m]) for m in order]
            cells = [_f(v) if not isinstance(v, str) else v for v in vals]
            if len(order) > 1:
                cells.append(_d(vals[0], vals[1]))
            L.append("| %s | %s |" % (name, " | ".join(cells)))
        L.append("")

    # 1) BPT — 合成口径
    table("BPT — 合成口径（eval_<lang>.txt 切片）", [
        ("en", lambda s: s["bpt_synth_per_lang"].get("en")),
        ("zh", lambda s: s["bpt_synth_per_lang"].get("zh")),
        ("code", lambda s: s["bpt_synth_per_lang"].get("code")),
    ])

    # 2) BPT — 真实口径
    table("BPT — 真实口径（<lang>_real.txt 切片）", [
        ("en", lambda s: s["bpt_real_per_lang"].get("en")),
        ("zh", lambda s: s["bpt_real_per_lang"].get("zh")),
        ("code", lambda s: s["bpt_real_per_lang"].get("code")),
    ])

    # 3) BPT — 整语料
    table("BPT — 整语料（corpus/<lang>/<lang>.txt，bytes/tokens）", [
        ("en", lambda s: get(s["bpt_corpus_per_lang"] or {}, "en")),
        ("zh", lambda s: get(s["bpt_corpus_per_lang"] or {}, "zh")),
        ("code", lambda s: get(s["bpt_corpus_per_lang"] or {}, "code")),
        ("byte-weighted total", lambda s: s["bpt_total_byte_weighted"]),
        ("macro（三语种等权）", lambda s: s["bpt_macro"]),
        ("全语料合计 bytes", lambda s: get(s["totals"] or {}, "bytes")),
        ("全语料合计 tokens", lambda s: get(s["totals"] or {}, "tokens")),
    ])

    # 4) D / dispersion
    table("协议鲁棒性：D = BPT_eval / BPT_real 与 dispersion（阈值 0.35 / 1.0）", [
        ("D(en)", lambda s: get(s["D"] or {}, "en")),
        ("D(zh)", lambda s: get(s["D"] or {}, "zh")),
        ("D(code)", lambda s: get(s["D"] or {}, "code")),
        ("dispersion max-min", lambda s: get(s["dispersion"] or {}, "max_minus_min")),
        ("dispersion std", lambda s: get(s["dispersion"] or {}, "std")),
        ("dispersion mean", lambda s: get(s["dispersion"] or {}, "mean")),
        ("verdict level", lambda s: get(s["verdict"] or {}, "level") or "-"),
        ("verdict text", lambda s: (get(s["verdict"] or {}, "text") or "-")[:120]),
    ])

    # 5) roundtrip
    table("roundtrip 失败行数（越小越好，0 = 全还原）", [
        ("en 失败/测试行", lambda s: "%s/%s" % (_i(get(s["roundtrip"], "en", "mismatches")),
                                              _i(get(s["roundtrip"], "en", "lines")))),
        ("zh 失败/测试行", lambda s: "%s/%s" % (_i(get(s["roundtrip"], "zh", "mismatches")),
                                              _i(get(s["roundtrip"], "zh", "lines")))),
        ("code 失败/测试行", lambda s: "%s/%s" % (_i(get(s["roundtrip"], "code", "mismatches")),
                                                _i(get(s["roundtrip"], "code", "lines")))),
        ("失败合计", lambda s: s["roundtrip_fail_total"]),
    ], note="失败行数 > 0 表示 decode 后字节与原输入不一致；reversible 模式设计上应接近 0。")

    # 6) TokEval
    table("TokEval 指标（三语种平均）", [
        ("utf8_completeness", lambda s: get(s["tokeval"] or {}, "utf8_completeness")),
        ("char_split_rate", lambda s: get(s["tokeval"] or {}, "char_split_rate")),
        ("boundary_crossing_rate", lambda s: get(s["tokeval"] or {}, "boundary_crossing_rate")),
        ("digit_boundary_f1", lambda s: get(s["tokeval"] or {}, "digit_boundary_f1")),
        ("digit_split_variability", lambda s: get(s["tokeval"] or {}, "digit_split_variability")),
        ("operator_isolation", lambda s: get(s["tokeval"] or {}, "operator_isolation")),
    ])

    # 7) stability
    table("stability（1.0 = 完全稳定）", [
        ("en", lambda s: get(s["stability"] or {}, "per_lang", "en")),
        ("zh", lambda s: get(s["stability"] or {}, "per_lang", "zh")),
        ("code", lambda s: get(s["stability"] or {}, "per_lang", "code")),
        ("overall", lambda s: get(s["stability"] or {}, "overall")),
    ])

    # 8) 吞吐
    table("吞吐与加载（同一 CLI、同一模型）", [
        ("encode tokens/sec", lambda s: get(s["throughput"] or {}, "encode_tps")),
        ("decode tokens/sec", lambda s: get(s["throughput"] or {}, "decode_tps")),
        ("load_ms", lambda s: get(s["throughput"] or {}, "load_ms")),
        ("单模式评测墙钟(s)", lambda s: s["elapsed_s"]),
    ])

    # 9) 自动差异摘要
    L.append("## 关键差异摘要（相对 %s）" % base)
    L.append("")
    if len(order) > 1:
        b = sums[base]
        for m in order[1:]:
            s = sums[m]
            L.append("### `%s` 相对 `%s`" % (m, base))
            L.append("")
            items = []
            for lang in LANGS:
                items.append("- BPT 合成 %s：%s → %s（%s）" % (
                    lang, _f(b["bpt_synth_per_lang"].get(lang)), _f(s["bpt_synth_per_lang"].get(lang)),
                    _d(b["bpt_synth_per_lang"].get(lang), s["bpt_synth_per_lang"].get(lang))))
            for lang in LANGS:
                items.append("- BPT 真实 %s：%s → %s（%s）" % (
                    lang, _f(b["bpt_real_per_lang"].get(lang)), _f(s["bpt_real_per_lang"].get(lang)),
                    _d(b["bpt_real_per_lang"].get(lang), s["bpt_real_per_lang"].get(lang))))
            items.append("- roundtrip 失败合计：%s → %s" % (
                _i(b["roundtrip_fail_total"]), _i(s["roundtrip_fail_total"])))
            for k in ("utf8_completeness", "char_split_rate", "digit_boundary_f1",
                      "digit_split_variability", "operator_isolation"):
                items.append("- %s：%s → %s（%s）" % (
                    k, _f(get(b["tokeval"] or {}, k)), _f(get(s["tokeval"] or {}, k)),
                    _d(get(b["tokeval"] or {}, k), get(s["tokeval"] or {}, k))))
            items.append("- stability overall：%s → %s（%s）" % (
                _f(get(b["stability"] or {}, "overall")), _f(get(s["stability"] or {}, "overall")),
                _d(get(b["stability"] or {}, "overall"), get(s["stability"] or {}, "overall"))))
            items.append("- dispersion(max-min)：%s → %s" % (
                _f(get(b["dispersion"] or {}, "max_minus_min")),
                _f(get(s["dispersion"] or {}, "max_minus_min"))))
            items.append("- encode tps：%s → %s" % (
                _f(get(b["throughput"] or {}, "encode_tps"), 1),
                _f(get(s["throughput"] or {}, "encode_tps"), 1)))
            L.extend(items)
            L.append("")
    else:
        L.append("- 仅一个模式，无差异可比。")
        L.append("")

    L.append("## 逐模式原始报告")
    L.append("")
    for m in order:
        L.append("- `%s`：metrics/eval_metrics_model_big2_%s.json；协议段：metrics/protocol_robustness_%s.json；"
                 "协议 Markdown：reports/protocol_robustness_%s.md；stdout：logs/eval_%s_stdout.log"
                 % (m, m, m, m, m))
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="nfkc/nfc/reversible 三模式一键对比")
    ap.add_argument("--root", default=ROOT, help="tokenizer_c 项目根")
    ap.add_argument("--model", default="", help="模型路径（默认 <root>/model_big2.tkm）")
    ap.add_argument("--cli", default="", help="CLI 路径（默认 <archive>/bin/tok-cli-norm.exe）")
    ap.add_argument("--corpus-dir", default="", help="整语料目录（默认 <root>/benchmark/corpus）")
    ap.add_argument("--slices-dir", default="", help="两口径切片目录（默认 <archive>/corpus_slices）")
    ap.add_argument("--outdir", default=ARCHIVE, help="归档目录（默认脚本上级目录）")
    ap.add_argument("--modes", default=",".join(MODES), help="逗号分隔的模式列表")
    ap.add_argument("--label", default="model_big2", help="报告中的模型显示名")
    ap.add_argument("--python", default=sys.executable, help="python 解释器")
    ap.add_argument("--skip-run", action="store_true",
                    help="跳过评测执行，直接复用 metrics/ 下已有 JSON 重新聚合")
    args = ap.parse_args()

    args.root = os.path.abspath(args.root)
    args.outdir = os.path.abspath(args.outdir)
    args.model = args.model or os.path.join(args.root, "model_big2.tkm")
    if not args.cli:
        cand = [os.path.join(args.outdir, "bin", "tok-cli-norm.exe"),
                os.path.join(args.root, "build_norm", "tok-cli-norm.exe")]
        args.cli = next((c for c in cand if os.path.isfile(c)), cand[0])
    if not args.slices_dir:
        cand = [os.path.join(args.outdir, "corpus_slices"),
                os.path.join(os.environ.get("TEMP", ""), "indomain")]
        args.slices_dir = next((c for c in cand if os.path.isdir(c)), cand[0])

    args.corpus_dir = args.corpus_dir or os.path.join(args.root, "benchmark", "corpus")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for sub in ("metrics", "reports", "logs"):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    for p, name in ((args.model, "模型"), (args.cli, "CLI"),
                    (args.corpus_dir, "语料目录"), (args.slices_dir, "切片目录")):
        if not os.path.exists(p):
            sys.exit("[ERROR] %s不存在: %s" % (name, p))
    for m in modes:
        if m not in MODES:
            sys.exit("[ERROR] 未知模式: %s（可选 %s）" % (m, "/".join(MODES)))

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    em = os.path.join(args.root, "benchmark", "eval_metrics.py")
    root_meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "model_md5": md5_of(args.model),
        "cli": args.cli,
        "cli_md5": md5_of(args.cli),
        "corpus_dir": args.corpus_dir,
        "slices_dir": args.slices_dir,
        "script": os.path.abspath(__file__),
        "eval_metrics": em,
        "eval_metrics_md5": md5_of(em),
        "git": git_info(args.root),
    }

    sums = {}
    for m in modes:
        if args.skip_run:
            jp = os.path.join(args.outdir, "metrics", "eval_metrics_model_big2_%s.json" % m)
            if not os.path.isfile(jp):
                sys.exit("[ERROR] --skip-run 但缺少 %s" % jp)
            with open(jp, "r", encoding="utf-8") as fh:
                rpt = json.load(fh)
        else:
            rpt = run_one_mode(args, m, env)
        sums[m] = summarize(m, rpt)
    root_meta["vocab_size"] = sums[modes[0]]["vocab_size"]

    md = render_markdown(sums, root_meta)
    md_path = os.path.join(args.outdir, "reports", "norm_mode_compare.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    json_path = os.path.join(args.outdir, "metrics", "norm_mode_compare_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": root_meta, "modes": sums}, fh, ensure_ascii=False, indent=2)

    print("")
    print(md)
    print("")
    print("[OK] Markdown -> %s" % md_path)
    print("[OK] Summary  -> %s" % json_path)


if __name__ == "__main__":
    main()
