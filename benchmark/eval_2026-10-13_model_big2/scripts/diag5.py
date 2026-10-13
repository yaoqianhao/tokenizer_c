# -*- coding: utf-8 -*-
"""还原率失败根因对照实验(逐行, 无分块伪影): 含/不含 NFKC 可变字符的行"""
import os, json, importlib.util, unicodedata
TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
IND = os.path.join(TEMP, "indomain")
CLI = os.path.join(TEMP, "tkfast", "build", "tok-cli-fast.exe")
TMPD = os.path.join(TEMP, "clitmp")
M = r"E:\tokenizer_c\model_big2.tkm"
spec = importlib.util.spec_from_file_location("em", r"E:\tokenizer_c\benchmark\eval_metrics.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

lines = [l for l in open(r"E:\tokenizer_c\benchmark\corpus\zh\zh.txt", encoding="utf-8", errors="replace").read().split("\n") if l]
def var(s): return any(unicodedata.normalize("NFKC", c) != c for c in s)
A = [l for l in lines if var(l)][:60]
B = [l for l in lines if not var(l)][:60]
allc = [ch for l in lines for ch in l if unicodedata.normalize("NFKC", ch) != ch]
print("zh.txt 总行数 %d; 含 NFKC 可变字符 %d 行 (%.1f%%); 可变字符样例: %s" %
      (len(lines), sum(1 for l in lines if var(l)), 100.0*sum(1 for l in lines if var(l))/len(lines),
       " ".join("%s(U+%04X)->%s" % (c, ord(c), unicodedata.normalize("NFKC", c)) for c in list(dict.fromkeys(allc))[:8])))

def probe(sub, tag):
    bad = 0; ex = None
    for i, ln in enumerate(sub):
        ids = em.encode_text(CLI, M, ln.encode("utf-8"), TMPD)
        back = em.decode_ids(CLI, M, ids, TMPD)
        if back != ln.encode("utf-8"):
            bad += 1
            if ex is None:
                b = back.decode("utf-8", "replace")
                d = [(k, x, y) for k, (x, y) in enumerate(zip(ln, b)) if x != y][:4]
                ex = {"orig": ln[:100], "decoded": b[:100],
                      "diff": [{"pos": t[0], "orig": t[1], "orig_u": "U+%04X" % ord(t[1]),
                                "dec": t[2], "dec_u": "U+%04X" % ord(t[2])} for t in d]}
    return bad, ex

bA, exA = probe(A, "var"); bB, exB = probe(B, "clean")
print("A组 含 NFKC 可变字符: %d/%d 行还原失败" % (bA, len(A)))
if exA:
    print("   例:", exA["orig"][:60]); print("   回:", exA["decoded"][:60]); print("   差异:", exA["diff"])
print("B组 不含 NFKC 可变字符: %d/%d 行还原失败" % (bB, len(B)))
if exB: print("   例:", exB)
out = {"zh_lines": len(lines), "nfkc_var_lines": sum(1 for l in lines if var(l)),
       "A_var_lines": len(A), "A_mismatch": bA, "B_clean_lines": len(B), "B_mismatch": bB,
       "example_A": exA, "example_B": exB}
json.dump(out, open(os.path.join(TEMP, "roundtrip_rootcause.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved:", os.path.join(TEMP, "roundtrip_rootcause.json"))
