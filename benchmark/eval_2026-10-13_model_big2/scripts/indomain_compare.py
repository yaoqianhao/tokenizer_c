# -*- coding: utf-8 -*-
"""补充实验：域内(真实语料) vs 评测语料(合成) 的同口径压缩率对比
- 项目侧: 通过 benchmark/eval_metrics.py 的 encode_file() 调用加速版 CLI（语义与 build/tok-cli.exe 逐字节一致）
- 厂商侧: 直接用 tokenizers 库加载 E:\\qwen-distill\\models\\* 的 tokenizer.json
"""
import os, sys, json, glob, unicodedata, importlib.util

TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
IND  = os.path.join(TEMP, "indomain")
OUT  = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\output"
CLI  = os.path.join(TEMP, "tkfast", "build", "tok-cli-fast.exe")
TMPD = os.path.join(TEMP, "clitmp"); os.makedirs(TMPD, exist_ok=True)

spec = importlib.util.spec_from_file_location("em", r"E:\tokenizer_c\benchmark\eval_metrics.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

FILES = [
    ("eval_en",   "en",   os.path.join(IND, "eval_en.txt")),
    ("eval_zh",   "zh",   os.path.join(IND, "eval_zh.txt")),
    ("eval_code", "code", os.path.join(IND, "eval_code.txt")),
    ("real_en",   "en",   os.path.join(IND, "en_real.txt")),
    ("real_zh",   "zh",   os.path.join(IND, "zh_real.txt")),
    ("real_code", "code", os.path.join(IND, "code_real.txt")),
]

PROJ = [("model_big2", r"E:\tokenizer_c\model_big2.tkm"),
        ("model_big",  r"E:\tokenizer_c\model_big.tkm"),
        ("model_256k", r"E:\tokenizer_c\benchmark\model_256k.tkm")]
VEND = [("qwen1.5B", r"E:\qwen-distill\models\Qwen2.5-1.5B-Instruct"),
        ("grok", r"E:\qwen-distill\models\grok"),
        ("doubao", r"E:\qwen-distill\models\doubao"),
        ("gpt6", r"E:\qwen-distill\models\gpt6"),
        ("hunyuan", r"E:\qwen-distill\models\hunyuan")]

res = {"project": {}, "vendor": {}, "vendor_fullcorpus": {}}

def bpt(path, ntok):
    return round(os.path.getsize(path) / ntok, 4) if ntok else None

print("=== 项目三个 .tkm（加速版 CLI，与原始 build/tok-cli.exe 输出逐字节一致）===")
for name, mp in PROJ:
    res["project"][name] = {}
    row = []
    for tag, lang, fp in FILES:
        ids = em.encode_file(CLI, mp, fp, TMPD)
        b = bpt(fp, len(ids))
        res["project"][name][tag] = {"tokens": len(ids), "bytes": os.path.getsize(fp), "bpt": b}
        row.append("%s=%s" % (tag, b))
    print("  %-11s %s" % (name, "  ".join(row)))

print("\n=== 厂商 tokenizer（tokenizers 库，2MB 切片）===")
from tokenizers import Tokenizer
for name, d in VEND:
    cands = [os.path.join(d, "tokenizer.json")] + glob.glob(os.path.join(d, "*.json"))
    tj = next((c for c in cands if os.path.exists(c)), None)
    tk = Tokenizer.from_file(tj)
    res["vendor"][name] = {}
    row = []
    for tag, lang, fp in FILES:
        text = open(fp, encoding="utf-8", errors="replace").read()
        n = len(tk.encode(text).ids)
        b = bpt(fp, n)
        res["vendor"][name][tag] = {"tokens": n, "bytes": os.path.getsize(fp), "bpt": b}
        row.append("%s=%s" % (tag, b))
    print("  %-11s %s   [%s]" % (name, "  ".join(row), os.path.basename(tj)))

print("\n=== 校验：厂商 harness 复现（整个评测语料文件，对比已发布 *_eval.json）===")
pub = {}
for f in glob.glob(r"E:\tokenizer_c\benchmark\*_eval.json"):
    d = json.load(open(f, encoding="utf-8"))
    pub[os.path.basename(f).replace("_eval.json", "")] = d
pubname = {"qwen1.5B": "qwen_1.5B", "grok": "grok", "doubao": "doubao", "gpt6": "gpt6", "hunyuan": "hunyuan"}
for name, d in VEND:
    tk = Tokenizer.from_file(next(c for c in [os.path.join(d, "tokenizer.json")] + glob.glob(os.path.join(d, "*.json")) if os.path.exists(c)))
    p = pub.get(pubname[name])
    res["vendor_fullcorpus"][name] = {}
    parts = []
    for lang in ("en", "zh", "code"):
        fp = rf"E:\tokenizer_c\benchmark\corpus\{lang}\{lang}.txt"
        text = open(fp, encoding="utf-8", errors="replace").read()
        n = len(tk.encode(text).ids)
        mine = bpt(fp, n)
        theirs = p["langs"][lang]["tokens"] if p else None
        res["vendor_fullcorpus"][name][lang] = {"tokens": n, "bytes": os.path.getsize(fp), "bpt": mine, "published_tokens": theirs}
        parts.append("%s: 我=%d tok/%s BPT vs 发布=%s tok/%s BPT" % (lang, n, mine, theirs, p["langs"][lang]["bpt"] if p else "-"))
    print("  %-11s %s" % (name, " | ".join(parts)))

# ---- NFKC 还原率根因验证 ----
print("\n=== zh 还原率 FAIL 根因验证（模型 model_big2）===")
lines = open(r"E:\tokenizer_c\benchmark\corpus\zh\zh.txt", encoding="utf-8", errors="replace").read().splitlines()
cand = next(l for l in lines if unicodedata.normalize("NFKC", l) != l)
raw = cand.encode("utf-8"); nfkc = unicodedata.normalize("NFKC", cand).encode("utf-8")
ids = em.encode_text(CLI, r"E:\tokenizer_c\model_big2.tkm", raw, TMPD)
back = em.decode_ids(CLI, r"E:\tokenizer_c\model_big2.tkm", ids, TMPD)
print("  原行        :", cand[:50])
print("  NFKC 归一化 :", unicodedata.normalize("NFKC", cand)[:50])
print("  encode->decode 输出:", back.decode("utf-8", "replace")[:50])
print("  与原文逐字节相同? ", back == raw, " | 与 NFKC 归一化相同? ", back == nfkc)
res["nfkc_check"] = {"original": cand, "nfkc": unicodedata.normalize("NFKC", cand),
                     "decoded": back.decode("utf-8", "replace"), "equal_raw": back == raw, "equal_nfkc": back == nfkc}

json.dump(res, open(os.path.join(TEMP, "indomain_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved:", os.path.join(TEMP, "indomain_results.json"))
