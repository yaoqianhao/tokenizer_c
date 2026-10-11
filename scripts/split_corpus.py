#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 corpus_64mb.txt 按行拆分为评测语料目录 corpus/<lang>/*.txt。

语言分类（按优先级，corpus_64mb.txt 仅含中/英/代码三类）:
  zh            : 行内含 CJK 统一表意文字
  code          : 无 CJK 但含明显代码特征（花括号/分号/关键字等）
  en            : 其余（ASCII 为主的自然语言）

每语种按「先到者」限制规模：EN 上限 15MB，其余 10MB；每语种行数上限 200000。
输出 corpus/<lang>/<lang>.txt（UTF-8）。
"""
import os
import sys
import re

SRC = sys.argv[1] if len(sys.argv) > 1 else "benchmark/corpus_64mb.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmark/corpus"

BYTE_CAPS = {"en": 15 * 1024 * 1024, "zh": 10 * 1024 * 1024,
             "code": 10 * 1024 * 1024}
LINE_CAPS = 200000

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_NONASCII = re.compile(r"[^\x00-\x7f]")
_CODE_HINTS = re.compile(r"\{|\}|[;:]\)|\b(def|class|function|var|const|let|import|export|return|if|else|for|while|public|private|static|void|int|char|struct|include|printf|echo|SELECT|FROM|WHERE|BEGIN|END)\b", re.I)

langs = ["en", "zh", "code"]
counters = {k: 0 for k in langs}
bytes_used = {k: 0 for k in langs}

print("reading %s ..." % SRC)
with open(SRC, "rb") as fh:
    raw = fh.read()
print("  total bytes: %d" % len(raw))

lines = raw.split(b"\n")
print("  total lines: %d" % len(lines))

# 先开文件句柄
outfiles = {}
for lang in langs:
    d = os.path.join(OUT, lang)
    os.makedirs(d, exist_ok=True)
    outfiles[lang] = open(os.path.join(d, lang + ".txt"), "wb")

stats = {k: [0, 0] for k in langs}  # lines, bytes
try:
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            text = s.decode("utf-8", "ignore")
        except Exception:
            continue
        if _CJK.search(text):
            lang = "zh"
        elif _CODE_HINTS.search(text):
            lang = "code"
        else:
            lang = "en"

        # 规模限制
        if stats[lang][0] >= LINE_CAPS or stats[lang][1] >= BYTE_CAPS[lang]:
            continue
        outfiles[lang].write(s + b"\n")
        stats[lang][0] += 1
        stats[lang][1] += len(s) + 1
finally:
    for f in outfiles.values():
        f.close()

print("== result ==")
for lang in langs:
    print("  %-13s lines=%6d bytes=%10.2f MB" % (lang, stats[lang][0], stats[lang][1] / 1048576.0))
print("output dir: %s" % OUT)
