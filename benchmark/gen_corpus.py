#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_corpus.py - deterministic multi-language corpus generator.

Produces a UTF-8 plain-text corpus mixing Chinese / English / code snippets
(one "document" per line, lines <= 4096 bytes to match the trainer's
per-sequence cap). Usage:

    python gen_corpus.py --out corpus.txt --size-mb 64 [--seed 2026]

The generator is seeded so runs are reproducible.
"""
import argparse
import random
import sys

# ---------------------------------------------------------------------------
# Chinese material: common hanzi pool + phrase templates (news/encyclopedia
# style). The pool is intentionally large so n-grams have rich variety.
# ---------------------------------------------------------------------------
HANZI = ("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别真力权须圆书式术内际海受证什清牛体文列南号或失定争向物安感电球研质便平明断走眼示设东能种对年发线面等光命学改数看又华车世南华信甚好国分计比别目选门般光认术海较比即权度验步广确北单球引知品界构还务教据料员专效示受话油路场文进证段权见四传度证准矿示史离写反空批回高验场识务路示调平层识带况技数点历程议定具里华切断示教料员集专验响计格图" )
COMMON_WORDS = ["人工智能", "深度学习", "大语言模型", "神经网络", "自然语言处理", "机器学习",
    "数据挖掘", "计算机视觉", "语音识别", "推荐系统", "云计算", "大数据", "区块链", "物联网",
    "自动驾驶", "机器人", "芯片", "半导体", "操作系统", "数据库", "分布式系统", "编译器",
    "程序员", "软件工程", "开源社区", "算法", "数据结构", "网络安全", "隐私保护", "数字化转型",
    "数字经济", "基础设施", "产业链", "供应链", "市场", "用户", "产品", "服务", "创新",
    "发展", "建设", "推进", "提升", "优化", "治理", "监管", "政策", "改革", "开放",
    "教育", "医疗", "交通", "能源", "环境", "文化", "科技", "金融", "制造", "农业",
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "天津",
    "中国", "美国", "欧洲", "亚洲", "全球", "国际", "合作", "交流", "会议", "报告"]

def gen_chinese(rng):
    style = rng.random()
    if style < 0.55:
        n = rng.randint(8, 30)
        parts = []
        for _ in range(n):
            r = rng.random()
            if r < 0.35:
                parts.append(rng.choice(COMMON_WORDS))
            elif r < 0.65:
                parts.append(rng.choice(HANZI))
            else:
                parts.append(rng.choice(COMMON_WORDS) + rng.choice(HANZI))
        return "".join(parts)
    elif style < 0.8:
        # news-style sentence
        tmpl = rng.choice([
            "{}表示，{}是{}的重要基础，未来将在{}方面发挥更大作用。",
            "据报道，{}近日发布了{}的最新进展，{}对此表示高度关注。",
            "专家指出，{}的发展离不开{}，需要进一步加强{}的建设。",
            "在{}大会上，与会代表就{}展开了深入讨论，达成了多项共识。",
            "随着{}的快速发展，{}已经成为{}领域的焦点话题。",
            "{}相关负责人介绍，{}项目已取得阶段性成果，预计{}完成。",
        ])
        return tmpl.format(rng.choice(COMMON_WORDS), rng.choice(COMMON_WORDS),
                           rng.choice(COMMON_WORDS), rng.choice(COMMON_WORDS))
    else:
        # technical paragraph
        return "本文从{}的角度出发，分析了{}在{}中的应用，并提出了基于{}的解决方案。".format(
            rng.choice(COMMON_WORDS), rng.choice(COMMON_WORDS),
            rng.choice(COMMON_WORDS), rng.choice(COMMON_WORDS))

# ---------------------------------------------------------------------------
# English material: word pool + sentence templates.
# ---------------------------------------------------------------------------
EN_WORDS = ("the of and to in a is that for it as was with be by on not he are at from or have "
    "an they this which his has been but one we can all your when there more would will about no "
    "if so what up out their who get which only other new some could time these two may then do "
    "first any my now like over know people after such years make than them its just last were "
    "also many way own see part well through work country great should even world because system "
    "program data model language network learning machine algorithm neural processing natural "
    "intelligence computer vision speech recognition recommendation cloud computing blockchain "
    "internet things autonomous driving robot semiconductor database operating system compiler "
    "software engineering open source community development research science technology industry "
    "market user product service innovation digital economy infrastructure supply chain privacy "
    "security transformation education healthcare transportation energy environment culture finance "
    "manufacturing agriculture international cooperation conference analysis report summary").split()

def gen_english(rng):
    n = rng.randint(10, 45)
    words = [rng.choice(EN_WORDS) for _ in range(n)]
    s = " ".join(words)
    s = s.capitalize()
    if rng.random() < 0.8:
        s += "."
    else:
        s += "?"
    return s

# ---------------------------------------------------------------------------
# Code material: python / json / c snippets.
# ---------------------------------------------------------------------------
PY_FN = ["train", "evaluate", "predict", "load_data", "preprocess", "tokenize",
         "normalize", "build_model", "infer", "run_pipeline", "merge_chunks",
         "compute_metrics", "save_checkpoint", "parse_config", "validate_input"]
PY_VAR = ["data", "model", "result", "config", "text", "tokens", "vocab", "ids",
          "scores", "batch", "epoch", "loss", "accuracy", "learning_rate", "param"]
PY_LIBS = ["os", "sys", "json", "re", "math", "time", "numpy", "torch", "pandas",
           "sklearn", "transformers", "datasets", "logging", "collections", "random"]

def gen_python(rng):
    fn = rng.choice(PY_FN)
    var = rng.choice(PY_VAR)
    lib = rng.choice(PY_LIBS)
    r = rng.random()
    if r < 0.3:
        return "import {lib}\nimport json\n\n{var} = {lib}.load(\"{var}.txt\")\n{fn}({var})".format(
            lib=lib, var=var, fn=fn)
    elif r < 0.6:
        return "def {fn}({var}, param=0.01):\n    # {var} pipeline\n    out = []\n    for x in {var}:\n        out.append(x)\n    return out".format(
            fn=fn, var=var)
    else:
        return "if __name__ == \"__main__\":\n    {fn}()".format(fn=fn)

def gen_json(rng):
    k = rng.randint(3, 10)
    entries = []
    for i in range(k):
        key = rng.choice(["id", "name", "value", "type", "data", "label", "score", "count"])
        r = rng.random()
        if r < 0.3:
            val = str(rng.randint(0, 100000))
        elif r < 0.6:
            val = "\"" + rng.choice(EN_WORDS) + rng.choice(EN_WORDS) + "\""
        else:
            val = "true" if rng.random() < 0.5 else "false"
        entries.append("\"" + key + str(i) + "\": " + val)
    return "{ " + ", ".join(entries) + " }"

def gen_c(rng):
    fn = rng.choice(["init", "process", "free", "load", "save", "update", "compute", "parse"])
    var = rng.choice(["buf", "ptr", "data", "size", "ctx", "node", "list", "str"])
    r = rng.random()
    if r < 0.5:
        return "static int {fn}(char *{var}, size_t n) {{\n    if (!{var} || n == 0) return -1;\n    return 0;\n}}".format(fn=fn, var=var)
    else:
        return "#include <stdio.h>\nint main(void) {{\n    printf(\"{fn}_{var}\\n\");\n    return 0;\n}}".format(fn=fn, var=var)

def gen_line(rng, lang):
    if lang == "zh":
        return gen_chinese(rng)
    if lang == "en":
        return gen_english(rng)
    if lang == "py":
        return gen_python(rng)
    if lang == "json":
        return gen_json(rng)
    if lang == "c":
        return gen_c(rng)
    raise ValueError(lang)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-mb", type=float, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    target = int(args.size_mb * 1024 * 1024)
    # language mix: chinese 40%, english 35%, code 25%
    langs = ["zh"] * 40 + ["en"] * 35 + ["py"] * 12 + ["json"] * 8 + ["c"] * 5

    written = 0
    lines = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        while written < target:
            lang = rng.choice(langs)
            line = gen_line(rng, lang)
            b = line.encode("utf-8")
            if len(b) > 4090:
                continue
            f.write(line)
            f.write("\n")
            written += len(b) + 1
            lines += 1
            if lines % 20000 == 0:
                sys.stderr.write("written %.1f MB (%d lines)\n" % (written / 1048576.0, lines))
    sys.stderr.write("done: %.1f MB, %d lines\n" % (written / 1048576.0, lines))

if __name__ == "__main__":
    main()
