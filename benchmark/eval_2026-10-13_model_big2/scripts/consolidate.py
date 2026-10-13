# -*- coding: utf-8 -*-
"""汇总本次评测与厂商基线的同口径指标 -> output/metrics_consolidated.json"""
import os, json

OUT = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\output"
TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
B = r"E:\tokenizer_c\benchmark"

SRC = [
    ("model_big2",     os.path.join(OUT, "eval_metrics_model_big2.json"),  "本项目-二代", "tok-cli-fast(C++)"),
    ("model_big",      os.path.join(OUT, "eval_metrics_model_big.json"),   "本项目-一代", "tok-cli-fast(C++)"),
    ("model_256k",     os.path.join(OUT, "eval_metrics_model_256k.json"),  "本项目-256K历史版", "tok-cli-fast(C++)"),
    ("qwen_1.5B",      os.path.join(B, "qwen_1.5B_eval.json"),  "厂商基线", "原生Python绑定"),
    ("qwen_7B",        os.path.join(B, "qwen_7B_eval.json"),    "厂商基线", "原生Python绑定"),
    ("grok",           os.path.join(B, "grok_eval.json"),       "厂商基线", "原生Python绑定"),
    ("doubao",         os.path.join(B, "doubao_eval.json"),     "厂商基线", "原生Python绑定"),
    ("gpt6",           os.path.join(B, "gpt6_eval.json"),       "厂商基线", "原生Python绑定"),
    ("hunyuan",        os.path.join(B, "hunyuan_eval.json"),    "厂商基线", "原生Python绑定"),
    ("baseline_2026-10-10", os.path.join(B, "eval_report_2026-10-10.json"), "历史报告(256K)", "原生Python绑定"),
]

def g(d, *ks, default=None):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

rows = {}
missing = []
for tag, path, kind, harness in SRC:
    if not os.path.exists(path):
        missing.append(path); continue
    d = json.load(open(path, encoding="utf-8"))
    L = g(d, "langs", default={}) or {}
    row = {
        "kind": kind, "harness": harness, "source_file": path,
        "vocab_size": g(d, "vocab_size"),
        "bpt": {k: (L.get(k, {}) or {}).get("bpt") for k in ("en", "zh", "code")},
        "tokens": {k: (L.get(k, {}) or {}).get("tokens") for k in ("en", "zh", "code")},
        "chars_per_token": {k: (L.get(k, {}) or {}).get("chars_per_token") for k in ("en", "zh", "code")},
        "bytes_total": g(d, "totals", "bytes"), "tokens_total": g(d, "totals", "tokens"),
        "bpt_total": g(d, "totals", "bpt"),
        "roundtrip_ok": g(d, "roundtrip_ok"),
        "roundtrip_mismatch_zh": g(d, "langs", "zh", "roundtrip_mismatches"),
        "stability_overall": g(d, "stability", "overall"),
        "stability_zh": g(d, "stability", "per_lang", "zh"),
        "utf8_completeness": g(d, "tokeval", "utf8_completeness"),
        "char_split_rate": g(d, "tokeval", "char_split_rate"),
        "boundary_crossing_rate": g(d, "tokeval", "boundary_crossing_rate"),
        "operator_isolation": g(d, "tokeval", "operator_isolation"),
        "digit_boundary_f1": g(d, "tokeval", "digit_boundary_f1"),
        "digit_split_variability": g(d, "tokeval", "digit_split_variability"),
        "oov_rate": {k: (L.get(k, {}) or {}).get("oov_rate") for k in ("en", "zh", "code")},
        "encode_tps": g(d, "throughput", "encode_tps"),
        "decode_tps": g(d, "throughput", "decode_tps"),
        "load_ms": g(d, "throughput", "load_ms"),
    }
    rows[tag] = row

out = {"meta": {"date": "2026-10-13", "corpus": "benchmark/corpus/{en,zh,code} (合计 37.1MB)",
                "note": "本项目三款模型经 tok-cli-fast(C++ CLI) 跑测；厂商基线为原生 Python 绑定跑测，吞吐/加载不可直接横比，其余指标同脚本同语料同口径",
                "tool": "benchmark/eval_metrics.py"},
       "models": rows, "missing_files": missing}
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "metrics_consolidated.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved:", dst, "| missing:", missing)

print("\n%-22s %8s %8s %8s %8s %8s %9s %8s %8s" % ("model", "vocab", "bpt_en", "bpt_zh", "bpt_code", "bpt_tot", "roundtrip", "stab", "utf8c"))
for tag, r in rows.items():
    f = lambda v, p="%.3f": ("-" if v is None else (p % v))
    print("%-22s %8s %8s %8s %8s %8s %9s %8s %8s" % (
        tag, r["vocab_size"], f(r["bpt"]["en"]), f(r["bpt"]["zh"]), f(r["bpt"]["code"]),
        f(r["bpt_total"]), str(r["roundtrip_ok"]), f(r["stability_overall"]), f(r["utf8_completeness"])))
print("\n%-22s %10s %10s %10s %10s %10s %10s" % ("model", "char_split", "bd_cross", "op_iso", "digit_f1", "enc_tps", "dec_tps"))
for tag, r in rows.items():
    f = lambda v, p="%.4f": ("-" if v is None else (p % v))
    print("%-22s %10s %10s %10s %10s %10s %10s" % (
        tag, f(r["char_split_rate"]), f(r["boundary_crossing_rate"]), f(r["operator_isolation"]),
        f(r["digit_boundary_f1"]), f(r["encode_tps"], "%.0f"), f(r["decode_tps"], "%.0f")))
