# -*- coding: utf-8 -*-
"""补充实验 v4: 中文单字覆盖率 + 还原率失败根因(NFKC)验证"""
import os, json, struct, collections, importlib.util, unicodedata

TEMP = r"C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
IND  = os.path.join(TEMP, "indomain")
CLI  = os.path.join(TEMP, "tkfast", "build", "tok-cli-fast.exe")
TMPD = os.path.join(TEMP, "clitmp"); os.makedirs(TMPD, exist_ok=True)
spec = importlib.util.spec_from_file_location("em", r"E:\tokenizer_c\benchmark\eval_metrics.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

def pieces_of(path):
    d = open(path,'rb').read(); vocab = struct.unpack_from('<I', d, 8)[0]
    o=32; ps=[]
    for i in range(vocab):
        pl = struct.unpack_from('<I', d, o)[0]; o+=4; ps.append(d[o:o+pl]); o+=pl+4
    return ps

MODELS = {"model_big2": r"E:\tokenizer_c\model_big2.tkm", "model_big": r"E:\tokenizer_c\model_big.tkm",
          "model_256k": r"E:\tokenizer_c\benchmark\model_256k.tkm"}
rpt = {}

print("=== 真实中文语料的『汉字单字覆盖率』===")
txt = open(os.path.join(IND,"zh_real.txt"), encoding="utf-8", errors="replace").read()
cnt = collections.Counter(ch for ch in txt if '\u4e00' <= ch <= '\u9fff')
tot_occ = sum(cnt.values()); tot_dist = len(cnt)
print("  语料: %d 个汉字出现, 共 %d 个不同汉字" % (tot_occ, tot_dist))
for tag, mp in MODELS.items():
    ps = pieces_of(mp)
    single = set()
    for b in ps:
        try: s = b.decode("utf-8")
        except Exception: continue
        if len(s) == 1 and '\u4e00' <= s <= '\u9fff': single.add(s)
    cov_dist = sum(1 for c in cnt if c in single)
    cov_occ  = sum(v for c, v in cnt.items() if c in single)
    rpt[tag] = {"distinct_cjk": tot_dist, "single_char_pieces": len(single),
                "distinct_coverage_pct": round(100.0*cov_dist/tot_dist, 2),
                "occurrence_coverage_pct": round(100.0*cov_occ/tot_occ, 2)}
    print("  %-11s 汉字整字 piece=%4d | 不同汉字覆盖 %5.2f%% | 按出现次数加权覆盖 %5.2f%%" %
          (tag, len(single), 100.0*cov_dist/tot_dist, 100.0*cov_occ/tot_occ))

print("\n=== 还原率(roundtrip)失败根因验证: NFKC 归一化 ===")
lines = open(r"E:\tokenizer_c\benchmark\corpus\zh\zh.txt", encoding="utf-8", errors="replace").read().split("\n")
def nfkc_var(s):
    return any(unicodedata.normalize("NFKC", ch) != ch for ch in s)
hit = [l for l in lines if l and nfkc_var(l)][:200]
print("  zh.txt 共 %d 行，含 NFKC 可变字符的行取样 %d 行做 encode->decode 逐字节比对" % (len(lines), len(hit)))
def probe(sub, tag):
    bad = 0; example = None
    for i in range(0, len(sub), 5):
        chunk = sub[i:i+5]
        cfp = os.path.join(IND, "chunk_%s.txt" % tag)
        open(cfp, "w", encoding="utf-8", newline="").write("\n".join(chunk))
        cids = em.encode_file(CLI, MODELS["model_big2"], cfp, TMPD)
        cback = em.decode_ids(CLI, MODELS["model_big2"], cids, TMPD).decode("utf-8", "replace").split("\n")
        for j, (a, b) in enumerate(zip(chunk, cback)):
            if a != b:
                bad += 1
                if example is None:
                    diff = [(k, x, y) for k, (x, y) in enumerate(zip(a, b)) if x != y][:6]
                    example = {"orig": a[:120], "decoded": b[:120],
                               "first_diffs": [{"pos": d[0], "orig": d[1], "decoded": d[2],
                                                "orig_u": "U+%04X" % ord(d[1]), "dec_u": "U+%04X" % ord(d[2])} for d in diff]}
    return bad, example

bad, example = probe(hit, "var")
print("  含 NFKC 可变字符: 失配行数 %d / %d" % (bad, len(hit)))
if example:
    print("  示例: 原行:", example["orig"][:70]); print("        还原:", example["decoded"][:70])
    print("        差异:", example["first_diffs"])
hit2 = [l for l in lines if l and not nfkc_var(l)][:200]
bad2, _ = probe(hit2, "clean")
print("  对照(不含 NFKC 可变字符): 失配行数 %d / %d" % (bad2, len(hit2)))
rpt["nfkc"] = {"var_lines_tested": len(hit), "var_mismatch": bad, "clean_lines_tested": len(hit2),
               "clean_mismatch": bad2, "example": example}
json.dump(rpt, open(os.path.join(TEMP,"cjk_coverage_nfkc.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved:", os.path.join(TEMP,"cjk_coverage_nfkc.json"))
