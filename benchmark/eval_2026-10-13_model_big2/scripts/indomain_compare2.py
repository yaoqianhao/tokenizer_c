# -*- coding: utf-8 -*-
"""补充实验 v2
A. 校验: 用 tokenizers/tiktoken 直接加载厂商 tokenizer，在完整评测语料上复现已发布 *_eval.json 的 tokens
B. 同口径对比: 项目 3 个 .tkm + 5 家厂商 tokenizer，在「合成评测语料」与「真实域内语料」上的 bytes/token
"""
import os, sys, json, glob, unicodedata, importlib.util
from tokenizers import Tokenizer

TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
IND  = os.path.join(TEMP, "indomain")
CLI  = os.path.join(TEMP, "tkfast", "build", "tok-cli-fast.exe")
TMPD = os.path.join(TEMP, "clitmp"); os.makedirs(TMPD, exist_ok=True)
spec = importlib.util.spec_from_file_location("em", r"E:\tokenizer_c\benchmark\eval_metrics.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

FILES = [("eval_en","en",os.path.join(IND,"eval_en.txt")), ("eval_zh","zh",os.path.join(IND,"eval_zh.txt")),
         ("eval_code","code",os.path.join(IND,"eval_code.txt")), ("real_en","en",os.path.join(IND,"en_real.txt")),
         ("real_zh","zh",os.path.join(IND,"zh_real.txt")), ("real_code","code",os.path.join(IND,"code_real.txt"))]
PROJ = [("model_big2", r"E:\tokenizer_c\model_big2.tkm"), ("model_big", r"E:\tokenizer_c\model_big.tkm"),
        ("model_256k", r"E:\tokenizer_c\benchmark\model_256k.tkm")]
VEND = [("qwen1.5B", r"E:\qwen-distill\models\Qwen2.5-1.5B-Instruct"), ("grok", r"E:\qwen-distill\models\grok"),
        ("doubao", r"E:\qwen-distill\models\doubao"), ("gpt6", r"E:\qwen-distill\models\gpt6"),
        ("hunyuan", r"E:\qwen-distill\models\hunyuan")]
PUBNAME = {"qwen1.5B":"qwen_1.5B","grok":"grok","doubao":"doubao","gpt6":"gpt6","hunyuan":"hunyuan"}

def load_grok(d):
    import tiktoken
    raw = json.load(open(os.path.join(d,"tokenizer.tok.json"), encoding="utf-8"))
    ranks = {bytes(t["bytes"]): t["token"] for t in raw["regular_tokens"]}
    sp = raw.get("special_tokens") or []
    specials = {}
    for s in sp:
        if isinstance(s, dict):
            nm = s.get("name") or s.get("text") or s.get("token_str")
            tid = s.get("token", s.get("id"))
            if nm is not None and tid is not None: specials[str(nm)] = int(tid)
    if not specials:
        specials = {"<|reserved_%d|>"%i: i for i in range(int(raw.get("reserved_tokens",128)))}
    PATS = {
      "cl100k": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
      "o200k": r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    }
    zh = open(os.path.join(IND,"eval_zh.txt"), encoding="utf-8", errors="replace").read()
    best = None
    for nm, pat in PATS.items():
        try:
            e = tiktoken.Encoding("grok_"+nm, pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)
            n = len(e.encode(zh, disallowed_special=()))
            bpt = os.path.getsize(os.path.join(IND,"eval_zh.txt"))/n
            print("    [grok pattern %s] zh 切片 tokens=%d bpt=%.4f (已发布 zh bpt=4.1268)" % (nm, n, bpt))
            if best is None or abs(bpt-4.1268) < abs(best[1]-4.1268): best = (e, bpt, nm)
        except Exception as ex:
            print("    [grok pattern %s] 失败: %s" % (nm, ex))
    if best is None: raise RuntimeError("grok tiktoken 加载失败")
    print("    -> 采用 pattern:", best[2])
    return lambda t: len(best[0].encode(t, disallowed_special=())), best[2]

print("=== 加载厂商 tokenizers ===")
vend = {}
for name, d in VEND:
    if name == "grok":
        fn, pat = load_grok(d); vend[name] = (fn, "tiktoken/"+pat)
    else:
        tj = next(c for c in [os.path.join(d,"tokenizer.json")]+glob.glob(os.path.join(d,"*.json")) if os.path.exists(c))
        tk = Tokenizer.from_file(tj)
        vend[name] = (lambda t, tk=tk: len(tk.encode(t).ids), "tokenizers/"+os.path.basename(tj))
    print("  %-9s <- %s" % (name, vend[name][1]))

res = {"project":{}, "vendor":{}, "validate":{}}
print("\n=== A. 校验: 完整评测语料上复现已发布 tokens ===")
for name,_ in VEND:
    fn = vend[name][0]
    pub = json.load(open(rf"E:\tokenizer_c\benchmark\{PUBNAME[name]}_eval.json", encoding="utf-8"))
    res["validate"][name] = {}
    parts=[]
    for lang in ("en","zh","code"):
        fp = rf"E:\tokenizer_c\benchmark\corpus\{lang}\{lang}.txt"
        t = open(fp, encoding="utf-8", errors="replace").read()
        n = fn(t); their = pub["langs"][lang]["tokens"]
        dev = 100.0*(n-their)/their
        res["validate"][name][lang] = {"mine": n, "published": their, "dev_pct": round(dev,3)}
        parts.append("%s 我=%d 发布=%d (%+.2f%%)" % (lang, n, their, dev))
    print("  %-9s %s" % (name, " | ".join(parts)))

print("\n=== B. 同口径对比: bytes/token（越高=压缩越好，同文本 token 更少）===")
print("  项目侧用加速版 CLI（输出与 build/tok-cli.exe 逐字节一致）")
for name, mp in PROJ:
    res["project"][name] = {}
    row=[]
    for tag, lang, fp in FILES:
        n = len(em.encode_file(CLI, mp, fp, TMPD)); b = round(os.path.getsize(fp)/n,4)
        res["project"][name][tag] = b; row.append("%s=%.3f"%(tag,b))
    print("  %-11s %s" % (name, "  ".join(row)))
for name,_ in VEND:
    fn = vend[name][0]; res["vendor"][name] = {}
    row=[]
    for tag, lang, fp in FILES:
        n = fn(open(fp, encoding="utf-8", errors="replace").read()); b = round(os.path.getsize(fp)/n,4)
        res["vendor"][name][tag] = b; row.append("%s=%.3f"%(tag,b))
    print("  %-11s %s" % (name, "  ".join(row)))

json.dump(res, open(os.path.join(TEMP,"indomain_results2.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved:", os.path.join(TEMP,"indomain_results2.json"))
