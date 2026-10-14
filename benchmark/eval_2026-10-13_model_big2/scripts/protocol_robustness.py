# -*- coding: utf-8 -*-
"""协议鲁棒性指标运行器（新增文件，不修改/覆盖任何既有脚本或数据）。

指标实现位于 benchmark/eval_metrics.py 的新增函数：
    compute_protocol_robustness / protocol_audit_file / protocol_audit_files /
    protocol_dispersion / protocol_verdict / protocol_macro / render_protocol_markdown
本脚本只做编排与出报告。

口径：
    D_lang     = BPT_eval / BPT_real（en/zh/code 各一），BPT = bytes/tokens
    dispersion = max(D) - min(D)，并附三语种总体标准差
    判定        = ≤0.35 稳定 / 0.35–1.0 敏感 / >1.0 口径依赖
    聚合口径    = 并列输出「按字节加权 bpt_total」与「三语种等权 macro 平均」

数据来源（不引入新语料、不重跑厂商基线）：
    项目 3 模型  本机加速版 CLI 实测，切片目录 = temp/indomain（indomain_compare.py /
                 indomain_compare2.py 生成 eval_*/real_* 的同一语料来源）
    厂商 5 家    直接读 metrics/indomain_results2.json 的 vendor eval_*/real_* 折算；
                 整语料聚合口径读 benchmark/<pub>_eval.json 的 totals

产物：
    metrics/protocol_robustness.json
    report/protocol_robustness.md（若已存在则自动加日期后缀）
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime

ROOT = r"E:\tokenizer_c"
BENCH = os.path.join(ROOT, "benchmark")
ARCH = os.path.join(BENCH, "eval_2026-10-13_model_big2")
CONV_TEMP = (r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User"
             r"\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace"
             r"\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp")

SLICES = os.path.join(CONV_TEMP, "indomain")
FULL_CORPUS = os.path.join(BENCH, "corpus")
CLI = os.path.join(CONV_TEMP, "tkfast", "build", "tok-cli-fast.exe")
INDOMAIN2 = os.path.join(ARCH, "metrics", "indomain_results2.json")

OUT_JSON = os.path.join(ARCH, "metrics", "protocol_robustness.json")
OUT_MD = os.path.join(ARCH, "report", "protocol_robustness.md")
LOGF = os.path.join(CONV_TEMP, "protocol_robustness_run.log")

PROJ = [("model_big2", os.path.join(ROOT, "model_big2.tkm")),
        ("model_big", os.path.join(ROOT, "model_big.tkm")),
        ("model_256k", os.path.join(BENCH, "model_256k.tkm"))]
VEND = [("qwen1.5B", "qwen_1.5B"), ("grok", "grok"), ("doubao", "doubao"),
        ("gpt6", "gpt6"), ("hunyuan", "hunyuan")]
LANGS = ("en", "zh", "code")

LOG = []


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    LOG.append(line)
    print(line)
    sys.stdout.flush()


def load_em():
    spec = importlib.util.spec_from_file_location(
        "em", os.path.join(BENCH, "eval_metrics.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    return em


def safe_path(path, date_str):
    """已存在则加日期后缀（不覆盖任何既有文件）。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    cand = "%s_%s%s" % (base, date_str, ext)
    i = 2
    while os.path.exists(cand):
        cand = "%s_%s_%d%s" % (base, date_str, i, ext)
        i += 1
    return cand


def data_quality_findings(audit):
    """由 protocol_audit 派生数据质量观察结论（纯函数，便于复现）。"""
    f = []
    ee, ec = audit.get("eval_en", {}), audit.get("eval_code", {})
    ce, cc = audit.get("corpus_en", {}), audit.get("corpus_code", {})
    if ee.get("dup_line_rate") is not None:
        f.append("评测语料重复行率高（模板合成痕迹）：eval_en=%.1f%%、eval_code=%.1f%%，"
                 "整语料 corpus_en=%.1f%%、corpus_code=%.1f%%；而真实切片 en_real=%.1f%%、"
                 "code_real=%.1f%%。" % (
                     100 * ee["dup_line_rate"],
                     100 * (ec.get("dup_line_rate") or 0),
                     100 * (ce.get("dup_line_rate") or 0),
                     100 * (cc.get("dup_line_rate") or 0),
                     100 * (audit.get("en_real", {}).get("dup_line_rate") or 0),
                     100 * (audit.get("code_real", {}).get("dup_line_rate") or 0)))
    ez, zr = audit.get("eval_zh", {}), audit.get("zh_real", {})
    if ez.get("distinct_chars") and zr.get("distinct_chars"):
        f.append("中文评测语料字符多样性极低：eval_zh 不同字符 %d（不同汉字 %d），"
                 "真实中文切片 zh_real 达 %d（不同汉字 %d）；二者比 %.0f 倍。"
                 "即评测语料的「协议」覆盖度远低于真实域。" % (
                     ez["distinct_chars"], ez.get("distinct_cjk_chars", 0),
                     zr["distinct_chars"], zr.get("distinct_cjk_chars", 0),
                     zr["distinct_chars"] / float(ez["distinct_chars"])))
    f.append("含义：D 稳定只说明「评测口径与真实口径的结论方向一致」，不代表评测语料真实；"
             "上述重复行率/字符多样性缺口会同时压低评测与真实两侧的 BPT，需与 protocol_audit "
             "并读，不能单独用 D 或 bpt_total 评价 tokenizer。")
    return f


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    em = load_em()
    log("eval_metrics.py 已加载: %s" % os.path.join(BENCH, "eval_metrics.py"))

    # 前置检查
    for p in (SLICES, os.path.join(SLICES, "eval_en.txt"), os.path.join(SLICES, "en_real.txt"),
              FULL_CORPUS, CLI, INDOMAIN2):
        if not os.path.exists(p):
            raise SystemExit("[ERROR] 依赖缺失: %s" % p)

    # ---------- 语料质量审计（切片 6 个 + 整语料 3 个）----------
    audit_targets = {}
    for lang in LANGS:
        audit_targets["eval_%s" % lang] = os.path.join(SLICES, "eval_%s.txt" % lang)
        audit_targets["%s_real" % lang] = os.path.join(SLICES, "%s_real.txt" % lang)
        audit_targets["corpus_%s" % lang] = os.path.join(FULL_CORPUS, lang, "%s.txt" % lang)
    audit = em.protocol_audit_files(audit_targets)
    log("protocol_audit 完成，覆盖 %d 个语料文件" % len(audit))

    models = {}
    tmp_root = tempfile.mkdtemp(prefix="proto_rob_", dir=CONV_TEMP)

    # ---------- 项目侧：本机 CLI 实测 ----------
    for name, tkm in PROJ:
        if not os.path.isfile(tkm):
            log("[SKIP] %s 模型不存在: %s" % (name, tkm))
            continue
        t0 = time.perf_counter()
        work = os.path.join(tmp_root, name)
        os.makedirs(work, exist_ok=True)
        sec = em.compute_protocol_robustness(CLI, tkm, SLICES, work,
                                             full_corpus_dir=FULL_CORPUS, langs=LANGS)
        sec["source"] = "本机 CLI 实测（加速版 CLI，切片+整语料同口径）"
        sec["corpus"] = {"slices_dir": SLICES, "full_corpus_dir": FULL_CORPUS}
        sec["corpus_audit"] = audit
        sec["label"] = name
        models[name] = sec
        log("%-11s D(en/zh/code)=%s  dispersion=%.4f (%s)  %.1fs" % (
            name, [sec["D"][l] for l in LANGS],
            sec["dispersion"]["max_minus_min"] or -1,
            sec["verdict"]["level"], time.perf_counter() - t0))

    # ---------- 厂商侧：读 indomain_results2.json 折算 ----------
    ind = json.load(open(INDOMAIN2, encoding="utf-8"))
    for name, pubname in VEND:
        v = (ind.get("vendor") or {}).get(name)
        if not v:
            log("[SKIP] vendor %s 无 indomain_results2.json 数据" % name)
            continue
        per_lang = {}
        d_map = {}
        for lang in LANGS:
            ev, re_ = v.get("eval_%s" % lang), v.get("real_%s" % lang)
            d = (ev / re_) if (ev and re_) else None
            per_lang[lang] = {"eval": {"bpt": ev, "tokens": None, "bytes": None},
                              "real": {"bpt": re_, "tokens": None, "bytes": None},
                              "D": round(d, 6) if d is not None else None}
            d_map[lang] = per_lang[lang]["D"]
        disp = em.protocol_dispersion([d_map[l] for l in LANGS])
        verdict = em.protocol_verdict(disp["max_minus_min"])
        agg = {"bpt_total_byte_weighted": None, "bpt_macro": None,
               "per_lang_bpt": {}, "source": None}
        pubj = os.path.join(BENCH, "%s_eval.json" % pubname)
        sec = {"model": None, "langs": list(LANGS), "per_lang": per_lang, "D": d_map,
               "dispersion": disp, "verdict": verdict, "aggregate": agg,
               "thresholds": {"stable_max": em.PROTOCOL_STABLE_MAX,
                              "sensitive_max": em.PROTOCOL_SENSITIVE_MAX},
               "label": name,
               "source": "indomain_results2.json 折算（厂商基线未重跑）"}
        if os.path.isfile(pubj):
            pj = json.load(open(pubj, encoding="utf-8"))
            tot = pj.get("totals") or {}
            agg["bpt_total_byte_weighted"] = round(
                tot["bytes"] / float(tot["tokens"]), 6) if tot.get("tokens") else None
            agg["per_lang_bpt"] = {l: (pj.get("langs", {}).get(l) or {}).get("bpt") for l in LANGS}
            agg["bpt_macro"] = em.protocol_macro([agg["per_lang_bpt"][l] for l in LANGS])
            agg["source"] = "benchmark/%s_eval.json totals 与 langs（整语料）" % pubname
        else:
            agg["per_lang_bpt"] = {l: per_lang[l]["eval"]["bpt"] for l in LANGS}
            agg["bpt_macro"] = em.protocol_macro([agg["per_lang_bpt"][l] for l in LANGS])
            agg["source"] = "indomain_results2.json eval_* 切片折算（发布整语料报告缺失）"
        models[name] = sec
        log("%-11s D(en/zh/code)=%s  dispersion=%.4f (%s)  [json 折算]" % (
            name, [d_map[l] for l in LANGS], disp["max_minus_min"] or -1, verdict["level"]))

    # ---------- 组装 payload ----------
    payload = {
        "date": date_str,
        "title": "协议鲁棒性指标报告（%s）" % date_str,
        "generated_by": "benchmark/eval_metrics.py 新增函数 compute_protocol_robustness / "
                        "protocol_audit_files / protocol_dispersion / protocol_verdict / "
                        "protocol_macro / render_protocol_markdown；由 "
                        "scripts/protocol_robustness.py 编排",
        "definitions": {
            "D_lang": "BPT_eval / BPT_real，逐语种（en/zh/code），BPT = bytes / tokens",
            "dispersion": "max(D) - min(D)，并附三语种总体标准差 std 与等权均值 mean",
            "protocol_audit": "逐语料文件：行数 lines / 总字符数 chars / 不同字符数 "
                              "distinct_chars / 不同汉字数 distinct_cjk_chars / 重复行率 dup_line_rate",
            "aggregate": "并列输出按字节加权 bpt_total 与三语种等权 macro 平均",
        },
        "thresholds": {"stable_max": em.PROTOCOL_STABLE_MAX,
                       "sensitive_max": em.PROTOCOL_SENSITIVE_MAX,
                       "rule": "dispersion ≤ 0.35 稳定 / 0.35 < dispersion ≤ 1.0 敏感 / > 1.0 口径依赖"},
        "corpus": {
            "slices_dir": SLICES,
            "full_corpus_dir": FULL_CORPUS,
            "files": {t: audit_targets[t] for t in audit_targets},
            "note": "real_* 复用 scripts/indomain_compare.py、indomain_compare2.py 生成 "
                    "real_* 的同一语料来源，本次未引入新语料；厂商基线未重跑。",
            "protocol_audit": audit,
        },
        "models": models,
    }

    notes = [
        "项目侧 3 个模型（model_big2 / model_big / model_256k）用本机加速版 CLI 实测，"
        "切片与整语料同口径；加速版 CLI 与原版 build/tok-cli.exe 输出逐字节一致。",
        "厂商侧 5 家（qwen1.5B / grok / doubao / gpt6 / hunyuan）不重跑，D 与 dispersion 直接由 "
        "metrics/indomain_results2.json 的 vendor eval_*/real_* 折算；聚合口径取自 "
        "benchmark/<name>_eval.json 的 totals。",
        "D 越高表示「评测语料低估压缩效率」（同一文本在评测语料上 token 更多）或「真实语料更易压缩」；"
        "dispersion 越大说明结论越依赖语料口径。",
        "protocol_audit 中 corpus_* 为完整评测语料（模板合成），eval_* 为同口径切片，*_real 为真实域内切片。",
    ]
    findings = data_quality_findings(audit)
    payload["data_quality_findings"] = findings
    payload["notes"] = notes

    # ---------- 落盘 ----------
    jp = safe_path(OUT_JSON, date_str)
    d = os.path.dirname(jp)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    log("JSON 已写入: %s" % jp)

    mp = safe_path(OUT_MD, date_str)
    d = os.path.dirname(mp)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write(em.render_protocol_markdown(payload, title=payload["title"]))
    log("Markdown 已写入: %s" % mp)

    with open(LOGF, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    print("\n=== 数值表 ===")
    print("| model | D_en | D_zh | D_code | dispersion | std | verdict |")
    for n, s in models.items():
        dd = s["D"]
        print("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %s |" % (
            n, dd["en"], dd["zh"], dd["code"], s["dispersion"]["max_minus_min"],
            s["dispersion"]["std"], s["verdict"]["level"]))
    print("\nJSON: %s" % jp)
    print("MD:   %s" % mp)


if __name__ == "__main__":
    main()
