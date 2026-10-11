#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tokenizer-c 评测语料下载 / 生成脚本。

生成 corpus/<lang>/*.txt（UTF-8），配比 en 30% / zh 20% / code 20% / multilingual 20% / noise 10%。

数据源（huggingface datasets，streaming 模式，按需下载）:
    - en:           wikitext (WikiText-103 test) 或 allenai/c4 抽样
    - zh:           pleisto/wikipedia-cn-20230720-filtered 或 csebuetnlp/xwikis 抽样
    - code:         codeparrot/github-code（py/js/java）或 bigcode/the-stack 抽样
    - multilingual: facebook/flores（devtest 多语种）或 Helsinki-NLP/opus-100
    - noise:        程序化生成（数字串 / emoji / URL / 混合语言 / 化学式等）

预处理: UTF-8 写入、\\n 换行、过滤空行、单行超 4000 字节截断（匹配 trainer 4096 上限）。
幂等: 已存在且 sha256 匹配的文件跳过（依据 out-dir/manifest.json），不足部分继续追加。
缺 datasets 库时提示: pip install datasets

用法:
    python download_corpus.py --out-dir corpus --target-mb 50
"""

import argparse
import hashlib
import json
import os
import random
import sys

MAX_LINE_BYTES = 4000          # 匹配 trainer 单行 4096 字节上限
CHUNK_BYTES = 4 * 1024 * 1024  # 每个 txt 的目标分片大小（约 4MB）

RATIOS = {
    "en": 0.30,
    "zh": 0.20,
    "code": 0.20,
    "multilingual": 0.20,
    "noise": 0.10,
}

EMOJIS = ["\U0001F600", "\U0001F602", "\U0001F60D", "\U0001F389", "\U0001F680",
          "\U0001F30D", "\U0001F34E", "\U0001F4BB", "\U0001F525", "\U00002B50"]
ZH_WORDS = ["你好", "世界", "今天", "天气", "很好", "编程", "测试", "数据", "模型", "系统",
            "学习", "工作", "项目", "代码", "网络", "用户", "开发", "训练", "语料", "评测"]
EN_WORDS = ["hello", "world", "today", "weather", "fine", "programming", "test", "data",
            "model", "system", "study", "work", "project", "code", "network", "user",
            "development", "training", "corpus", "eval"]
TLDS = ["com", "org", "net", "io", "dev"]


def sha256_file(path):
    """计算文件 sha256。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate_utf8(b, max_bytes):
    """按字节数截断 UTF-8 字节串，不破坏多字节字符边界。"""
    if len(b) <= max_bytes:
        return b
    s = b.decode("utf-8", "ignore")
    while len(s.encode("utf-8")) > max_bytes:
        s = s[:-1]
    return s.encode("utf-8")


def process_text(text):
    """预处理：\\n 换行、过滤空行、单行超 4000 字节截断。返回写入行的 bytes 列表。"""
    out = []
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        b = raw.encode("utf-8")
        if len(b) > MAX_LINE_BYTES:
            b = truncate_utf8(b, MAX_LINE_BYTES)
        out.append(b)
    return out


def load_manifest(out_dir):
    path = os.path.join(out_dir, "manifest.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_manifest(out_dir, manifest):
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 数据源迭代器（yield 原始文本）
# ---------------------------------------------------------------------------

def iter_en():
    try:
        ds = __import__("datasets").load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
        return (ex["text"] for ex in ds), {"source": "wikitext/wikitext-103-raw-v1", "split": "test"}
    except Exception as exc:
        print("[warn] wikitext 加载失败，改用 allenai/c4: %s" % exc)
        ds = __import__("datasets").load_dataset("allenai/c4", "en", split="train", streaming=True)
        return (ex["text"] for ex in ds), {"source": "allenai/c4", "split": "train"}


def iter_zh():
    try:
        ds = __import__("datasets").load_dataset(
            "pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
        return (ex.get("text") or "" for ex in ds), {"source": "pleisto/wikipedia-cn-20230720-filtered", "split": "train"}
    except Exception as exc:
        print("[warn] wikipedia-cn 加载失败，改用 csebuetnlp/xwikis: %s" % exc)
        ds = __import__("datasets").load_dataset("csebuetnlp/xwikis", split="train", streaming=True)
        return (ex.get("text") or "" for ex in ds), {"source": "csebuetnlp/xwikis", "split": "train"}


def iter_code():
    code_langs = {"Python", "JavaScript", "Java"}

    def gen_github_code():
        ds = __import__("datasets").load_dataset(
            "codeparrot/github-code", split="train", streaming=True)
        for ex in ds:
            if ex.get("language") in code_langs:
                yield ex.get("code") or ex.get("content") or ""

    try:
        # 预取一条验证数据集可加载
        it = gen_github_code()
        first = next(it)
        return (lambda: (lambda fst, itr: (y for y in [fst] + list(itr)))(first, it))(), \
            {"source": "codeparrot/github-code", "split": "train"}
    except Exception as exc:
        print("[warn] github-code 加载失败，尝试 bigcode/the-stack: %s" % exc)

        def gen_stack():
            ds = __import__("datasets").load_dataset(
                "bigcode/the-stack", split="train", streaming=True)
            for ex in ds:
                yield ex.get("content") or ""

        it = gen_stack()
        first = next(it)
        return (lambda: (y for y in [first] + list(it)))(), \
            {"source": "bigcode/the-stack", "split": "train"}


def iter_multilingual():
    langs = ["deu_Latn", "jpn_Jpan", "kor_Hang", "fra_Latn", "spa_Latn", "ara_Arab", "rus_Cyrl"]

    def gen():
        dsmod = __import__("datasets")
        for lg in langs:
            try:
                ds = dsmod.load_dataset("facebook/flores", lg, split="devtest", streaming=True)
                for ex in ds:
                    yield ex.get("sentence") or ""
            except Exception as exc:
                print("[warn] flores %s 加载失败: %s" % (lg, exc))
        # 备用：opus-100 en-de
        try:
            ds = dsmod.load_dataset("Helsinki-NLP/opus-100", "en-de", split="train", streaming=True)
            for ex in ds:
                tr = ex.get("translation") or {}
                yield tr.get("de") or tr.get("en") or ""
        except Exception as exc:
            print("[warn] opus-100 en-de 加载失败: %s" % exc)

    return gen(), {"source": "facebook/flores + Helsinki-NLP/opus-100", "split": "devtest/train"}


def gen_noise_line(rng):
    """程序化生成一行噪声（数字串 / emoji / URL / 混合语言 / 化学式 / 符号串）。"""
    kind = rng.randrange(6)
    if kind == 0:  # 数字串
        n = rng.randrange(8, 60)
        return "".join(str(rng.randrange(10)) for _ in range(n))
    if kind == 1:  # emoji
        n = rng.randrange(5, 40)
        return "".join(rng.choice(EMOJIS) for _ in range(n))
    if kind == 2:  # URL
        host = rng.choice(["example", "news", "blog", "docs", "api", "cdn"])
        path = rng.choice(["a", "b", "path", "download", "x", "index"])
        return "https://%s.%s/%s?id=%d&q=%s" % (
            host, rng.choice(TLDS), path, rng.randrange(100000),
            rng.choice(["token", "model", "corpus", "eval"]))
    if kind == 3:  # 混合语言
        parts = []
        for _ in range(rng.randrange(3, 8)):
            parts.append(rng.choice(ZH_WORDS) if rng.random() < 0.5 else rng.choice(EN_WORDS))
        return " ".join(parts)
    if kind == 4:  # 化学式/科学记号
        return "C%dH%dO%dN%d S%d" % (
            rng.randrange(1, 30), rng.randrange(1, 60),
            rng.randrange(1, 20), rng.randrange(1, 10), rng.randrange(1, 5))
    # kind == 5 符号/标点混合
    return "".join(rng.choice("!@#$%^&*()_+-=[]{};:,.<>?/\\|~`")
                   for _ in range(rng.randrange(10, 50)))


def iter_noise(seed):
    rng = random.Random(seed)

    def gen():
        while True:
            yield gen_noise_line(rng)

    return gen(), {"source": "synthetic/noise-generator", "split": "gen"}


def iter_for_lang(lang, seed):
    if lang == "noise":
        return iter_noise(seed)
    if lang == "en":
        return iter_en()
    if lang == "zh":
        return iter_zh()
    if lang == "code":
        return iter_code()
    if lang == "multilingual":
        return iter_multilingual()
    raise ValueError("未知语种: %s" % lang)


# ---------------------------------------------------------------------------
# 语种构建
# ---------------------------------------------------------------------------

def build_lang(lang, target_bytes, out_dir, manifest, seed):
    lang_dir = os.path.join(out_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    existing = manifest.get("langs", {}).get(lang, {})
    existing_bytes = 0
    existing_samples = 0
    files = []
    for f in existing.get("files", []):
        p = os.path.join(lang_dir, f["name"])
        if os.path.isfile(p) and sha256_file(p) == f.get("sha256"):
            existing_bytes += f.get("bytes", 0)
            existing_samples += f.get("samples", 0)
            files.append(f)

    if existing_bytes >= target_bytes:
        print("[skip] %s: 已达标（%d >= %d 字节）" % (lang, existing_bytes, target_bytes))
        return {
            "source": existing.get("source"),
            "split": existing.get("split"),
            "total_bytes": existing_bytes,
            "samples": existing_samples,
            "files": files,
        }

    print("[gen] %s: 目标 %d 字节，已有 %d 字节，继续采样..." % (lang, target_bytes, existing_bytes))
    it, meta = iter_for_lang(lang, seed)

    acc = existing_bytes
    total_samples = existing_samples
    file_idx = len(files) + 1
    cur_path = os.path.join(lang_dir, "%s_%04d.txt" % (lang, file_idx))
    tmp_path = cur_path + ".tmp"
    fh = open(tmp_path, "wb")
    cur_bytes = 0
    cur_samples = 0
    written = 0

    for text in it:
        if not text:
            continue
        b_lines = process_text(text)
        if not b_lines:
            continue
        for b in b_lines:
            fh.write(b + b"\n")
            cur_bytes += len(b) + 1
            acc += len(b) + 1
            cur_samples += 1
            total_samples += 1
            written += 1
            if cur_bytes >= CHUNK_BYTES and acc < target_bytes:
                fh.close()
                os.replace(tmp_path, cur_path)
                files.append({"name": os.path.basename(cur_path), "bytes": cur_bytes,
                              "sha256": sha256_file(cur_path), "samples": cur_samples})
                file_idx += 1
                cur_path = os.path.join(lang_dir, "%s_%04d.txt" % (lang, file_idx))
                tmp_path = cur_path + ".tmp"
                fh = open(tmp_path, "wb")
                cur_bytes = 0
                cur_samples = 0
            if acc >= target_bytes:
                break
    if cur_bytes > 0:
        fh.close()
        os.replace(tmp_path, cur_path)
        files.append({"name": os.path.basename(cur_path), "bytes": cur_bytes,
                      "sha256": sha256_file(cur_path), "samples": cur_samples})
    elif written == 0:
        fh.close()
        raise RuntimeError("[ERROR] 语种 %s 未获得任何有效样本（数据源不可用或全部被过滤）" % lang)
    else:
        fh.close()

    print("[ok] %s: 累计 %d 字节 / %d 行，文件数 %d" % (lang, acc, total_samples, len(files)))
    return {
        "source": meta["source"],
        "split": meta["split"],
        "total_bytes": acc,
        "samples": total_samples,
        "files": files,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="tokenizer-c 评测语料下载/生成")
    ap.add_argument("--out-dir", default="corpus", help="输出目录（默认 corpus）")
    ap.add_argument("--target-mb", type=float, default=50.0,
                    help="目标语料总量 MB（默认 50）")
    ap.add_argument("--seed", type=int, default=1234, help="噪声生成随机种子")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    total_bytes = int(args.target_mb * 1024 * 1024)

    print("将下载/生成约 %.1f MB 评测语料（含网络下载，耗时取决于网络与带宽）。" % args.target_mb)
    if args.target_mb >= 100:
        print("[提示] 目标较大（>=100MB），请确保磁盘空间与网络条件允许。")

    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("缺少 huggingface datasets 库，请先安装: pip install datasets")

    random.seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest(out_dir)
    manifest.setdefault("target_mb", args.target_mb)
    manifest.setdefault("langs", {})

    for lang, ratio in RATIOS.items():
        target = int(total_bytes * ratio)
        rec = build_lang(lang, target, out_dir, manifest, args.seed)
        manifest["langs"][lang] = rec
        save_manifest(out_dir, manifest)

    print("\n完成。manifest: %s" % os.path.join(out_dir, "manifest.json"))
    for lang, rec in manifest["langs"].items():
        print("  %-14s %10d bytes / %8d 行 / %d 文件  [%s]" % (
            lang, rec.get("total_bytes", 0), rec.get("samples", 0),
            len(rec.get("files", [])), rec.get("source", "-")))


if __name__ == "__main__":
    main()
