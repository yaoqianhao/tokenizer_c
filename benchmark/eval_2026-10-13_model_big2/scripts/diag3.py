# -*- coding: utf-8 -*-
"""补充实验 v3: 评测 JSON 汇总 + 中文编码的成分分析（字节回退占比 / 词表利用率 / 高频 token）"""
import os, json, struct, importlib.util, collections

TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
OUT  = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\output"
IND  = os.path.join(TEMP, "indomain")
CLI  = os.path.join(TEMP, "tkfast", "build", "tok-cli-fast.exe")
TMPD = os.path.join(TEMP, "clitmp"); os.makedirs(TMPD, exist_ok=True)
spec = importlib.util.spec_from_file_location("em", r"E:\tokenizer_c\benchmark\eval_metrics.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

def pieces_of(path):
    d = open(path,'rb').read(); vocab = struct.unpack_from('<I', d, 8)[0]
    o=32; ps=[]
    for i in range(vocab):
        pl = struct.unpack_from('<I', d, o)[0]; o+=4
        ps.append(d[o:o+pl]); o+=pl+4
    return ps

MODELS = {"model_big2": r"E:\tokenizer_c\model_big2.tkm", "model_big": r"E:\tokenizer_c\model_big.tkm",
          "model_256k": r"E:\tokenizer_c\benchmark\model_256k.tkm"}
LANG = ("en","zh","code")

print("=== 1. 三个模型 eval JSON 关键指标（tokenizer_c benchmark/eval_metrics.py 产出）===")
for tag in ("big2","big","256k"):
    p = os.path.join(OUT, "eval_metrics_model_%s.json" % tag)
    j = json.load(open(p, encoding="utf-8"))
    t = j.get("totals", {})
    print("\n-- %s | vocab=%s | 综合 BPT=%.4f | 综合稳定性=%s | roundtrip_ok=%s" %
          (tag, j.get("vocab_size"), t.get("bpt", -1), j.get("stability", {}).get("overall"), j.get("roundtrip_ok")))
    def f(x, nd=4):
        return "  -  " if x is None else ("%." + str(nd) + "f") % x
    print("   %-5s %10s %10s %8s %10s %9s %9s %9s %9s %9s %9s" % ("lang","bytes","tokens","BPT","chars/tok","rt_mis","stab","utf8","charsplit","bound","op_iso"))
    for l in LANG:
        d = j["langs"][l]; te = d.get("tokeval", {})
        print("   %-5s %10d %10d %8s %10s %9s %9s %9s %9s %9s %9s" % (
            l, d["bytes"], d["tokens"], f(d["bpt"]), f(d.get("chars_per_token")),
            "%s/%s" % (d.get("roundtrip_mismatches"), d.get("roundtrip_lines")), f(d.get("stability")),
            f(te.get("utf8",{}).get("utf8_completeness")), f(te.get("utf8",{}).get("char_split_rate")),
            f(te.get("utf8",{}).get("boundary_crossing_rate")), f(te.get("operator",{}).get("operator_isolation"))))
        print("        oov=%s byte_fallback=%s digit_f1=%s digit_var=%s | files=%s" % (
            d.get("oov_rate"), d.get("byte_fallback_rate"), te.get("digit",{}).get("digit_boundary_f1"),
            te.get("digit",{}).get("digit_split_variability"), d.get("files")))
    print("   throughput:", j.get("throughput"))

print("\n=== 2. 中文编码成分分析（真实中文 vs 合成中文评测语料）===")
res = {}
for tag, mp in MODELS.items():
    ps = pieces_of(mp)
    res[tag] = {}
    for label, fp in (("real_zh", os.path.join(IND,"zh_real.txt")), ("eval_zh", os.path.join(IND,"eval_zh.txt"))):
        ids = em.encode_file(CLI, mp, fp, TMPD)
        cnt = collections.Counter(ids)
        neg = sum(1 for i in ids if i < 0)
        distinct = len(cnt)
        top = []
        for i, c in cnt.most_common(12):
            if i < 0: t = "<byte %d>" % (-i-1)
            else:
                try: t = ps[i].decode("utf-8")
                except Exception: t = repr(ps[i])
            top.append((i, c, round(100.0*c/len(ids),2), t))
        res[tag][label] = {"tokens": len(ids), "distinct_ids": distinct, "neg_bytes": neg,
                           "neg_share_pct": round(100.0*neg/len(ids),2), "top": top,
                           "vocab_used_pct": round(100.0*distinct/len(ps),2)}
        print("\n  [%s / %s] tokens=%d 独立token数=%d(占词表%.1f%%) 字节回退token=%d(%.1f%%)" %
              (tag, label, len(ids), distinct, 100.0*distinct/len(ps), neg, 100.0*neg/len(ids)))
        print("     高频 token:", ", ".join("%r x%d(%.1f%%)" % (t[3], t[1], t[2]) for t in top[:8]))

json.dump(res, open(os.path.join(TEMP,"zh_composition.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved:", os.path.join(TEMP,"zh_composition.json"))
