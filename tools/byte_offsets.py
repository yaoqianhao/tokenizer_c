#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""byte_offsets.py —— byte fallback / byte-level token 的 offset span 标准化与审计

背景（tokenizer_c 优化报告 P0-②：byte fallback offset 语义）：
    big3 的 HF 导出物对不在词表的多字节字符（如“气”，UTF-8 3 字节）编码出 3 个
    字节级 token `<0xE6> <0xB0> <0x94>`，而在 `tokenizers` 侧这 3 个 token 的
    offset 全部等于源字符的整段 span `(1, 2)`（**共享 span**）。后果：

      * offsets 类“续接 / 切碎”统计自相矛盾：tok_eval 的 `subword_fertility`
        （按 offset 判定）为 0，而 `continued_word_rate`（按 token 形态判定）为 12.2%；
      * 字符长度类指标（avg_token_chars / avg_token_bytes / type_token_ratio /
        unique_tokens）被 6 字符的 `<0xE6>` 串系统性抬高；
      * 下游 alignment / NER / span 抽取按 offsets 做标签时会得到错误标注。

    根因与落点：
      * C 侧（src/tokenizer.c）只发射 id 序列、**不输出 char offset**，故 C 侧无可改；
      * HF 侧 `tokenizers` 的 Unigram + ByteFallback 在 **字符单位** 上对齐，一个多字节
        字符的 k 个字节 token 只能共享该字符的 1 个字符位 → 在字符单位下**不可再分**。
        因此契约把 span 单位统一到 **UTF-8 字节**：一个字节 token 恰好拥有 1 个字节位。

契约（`contract_version = "byte-span-1"`）：
    1. 所有 token 的 span 单位统一为 **UTF-8 字节**（`byte_spans`）；
    2. 每个字节 token 拥有**独立且长度恰为 1**的字节 span，互不重叠（消除共享 span）；
    3. 字节 token 的字节值序列必须与源文本在该字节区间内的 UTF-8 字节**逐字节一致**
       （防止“分组跨越多个字符”“字节数与区间宽度不一致”这类静默错位）；
    4. 非字节 token 的字节 span = 其字符 span 的字节投影；
    5. 同时保留 HF 原生字符单位 span（`hf_char_spans`）与字节组所属字符 span，显式标注
       共享事实，便于复现旧结论与做残差披露。

用法：
    python tools/byte_offsets.py --tokenizer <hf_dir|tokenizer.json> \\
        [--text "天气很好"] [--text-file <one_per_line.txt>] [--json <out.json>] \\
        [--assert-independent] [--quiet]

退出码：0 成功且满足契约；3 不满足契约（--assert-independent）；1 参数/加载错误。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

CONTRACT_VERSION = "byte-span-1"


def _is_byte_token(tok: str) -> bool:
    """<0xXX> 形式的字节 piece（与 C 端 byte fallback 发射形态一致）。"""
    return len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">")


def _byte_value(tok: str) -> int:
    return int(tok[3:5], 16)


def char_to_byte_offsets(text: str) -> list:
    """字符索引 -> UTF-8 字节偏移映射（长度 len(text)+1）。"""
    out = [0]
    acc = 0
    for ch in text:
        acc += len(ch.encode("utf-8"))
        out.append(acc)
    return out


def _load_tokenizer(path: str):
    from tokenizers import Tokenizer
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    return Tokenizer.from_file(path), path


def encode_with_byte_spans(tokenizer, text: str, add_special_tokens: bool = False) -> dict:
    """编码一句文本并按 byte-span-1 契约给出 span。

    返回字段：
      tokens / ids
      hf_char_spans : HF 原生返回的 span（字符单位；字节组共享一个字符 span）
      byte_spans    : 契约 span（字节单位；每字节 token 恰好 1 字节）
      byte_groups   : 字节组的明细（源字符 span、字节区间、逐字节 span、校验结果）
      checks        : 契约校验结果
    """
    enc = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    toks, ids, offs = list(enc.tokens), list(enc.ids), [tuple(o) for o in enc.offsets]
    n = len(toks)
    c2b = char_to_byte_offsets(text)
    raw = text.encode("utf-8")

    hf_char_spans = [list(o) for o in offs]
    byte_spans = [None] * n
    groups = []

    i = 0
    while i < n:
        if not _is_byte_token(toks[i]):
            s, e = offs[i]
            byte_spans[i] = [c2b[min(s, len(c2b) - 1)], c2b[min(e, len(c2b) - 1)]]
            i += 1
            continue
        j = i
        while j < n and _is_byte_token(toks[j]) and offs[j] == offs[i]:
            j += 1
        c0, c1 = offs[i]
        k = j - i
        bs, be = c2b[min(c0, len(c2b) - 1)], c2b[min(c1, len(c2b) - 1)]
        vals = [_byte_value(t) for t in toks[i:j]]
        expected = list(raw[bs:be])
        if k == be - bs and vals == expected:
            spans = [[bs + t, bs + t + 1] for t in range(k)]
            strict = True
        else:                                    # 退化：按比例分配（可能非 1 字节宽）
            width = max(be - bs, 0)
            spans = []
            for t in range(k):
                a = bs + (width * t) // k
                b = bs + (width * (t + 1)) // k
                if b <= a:
                    a, b = min(bs + t, max(be - 1, bs)), min(bs + t + 1, max(be, bs + 1))
                spans.append([a, b])
            strict = False
        for idx, sp in enumerate(spans):
            byte_spans[i + idx] = sp
        groups.append({
            "start_index": i, "count": k, "byte_tokens": toks[i:j],
            "source_char_span": [c0, c1], "source_byte_span": [bs, be],
            "byte_values": vals, "expected_bytes": expected,
            "hf_char_spans": [[c0, c1] for _ in range(k)],
            "byte_spans": [list(s) for s in spans],
            "shared_in_hf": k > 1,
            "independent": len({tuple(s) for s in spans}) == k and all(
                s[1] - s[0] == 1 for s in spans),
            "byte_value_match": vals == expected,
            "strict": strict,
        })
        i = j

    byte_collisions = 0
    seen = {}
    for idx, t in enumerate(toks):
        if not _is_byte_token(t):
            continue
        key = tuple(byte_spans[idx])
        seen[key] = seen.get(key, 0) + 1
    byte_collisions = sum(c - 1 for c in seen.values() if c > 1)
    bad_width = sum(1 for idx, t in enumerate(toks)
                    if _is_byte_token(t) and (byte_spans[idx][1] - byte_spans[idx][0]) != 1)
    mismatch = sum(1 for g in groups if not g["byte_value_match"])

    # 字符单位下的共享情况（HF 现状缺陷的量化口径）
    char_shared = sum(1 for g in groups if g["shared_in_hf"])

    return {
        "text": text,
        "tokens": toks,
        "ids": ids,
        "hf_char_spans": hf_char_spans,
        "byte_spans": byte_spans,
        "byte_groups": groups,
        "checks": {
            "byte_span_collisions": byte_collisions,
            "byte_span_width_violations": bad_width,
            "byte_value_mismatches": mismatch,
            "hf_char_shared_groups": char_shared,
            "contract_ok": (byte_collisions == 0 and bad_width == 0 and mismatch == 0),
            "hf_native_independent": char_shared == 0,
        },
    }


def audit(tokenizer, texts: list, tokenizer_path: str = "") -> dict:
    """审计 offset 契约：统计 HF 共享 span 组、契约化后是否逐字节独立且字节一致。"""
    aggs = {
        "byte_tokens": 0, "byte_groups": 0, "hf_shared_span_groups": 0,
        "byte_span_collisions": 0, "byte_span_width_violations": 0,
        "byte_value_mismatches": 0,
    }
    examples = []
    per_text = []
    for text in texts:
        r = encode_with_byte_spans(tokenizer, text)
        c = r["checks"]
        aggs["byte_tokens"] += sum(g["count"] for g in r["byte_groups"])
        aggs["byte_groups"] += len(r["byte_groups"])
        aggs["hf_shared_span_groups"] += c["hf_char_shared_groups"]
        aggs["byte_span_collisions"] += c["byte_span_collisions"]
        aggs["byte_span_width_violations"] += c["byte_span_width_violations"]
        aggs["byte_value_mismatches"] += c["byte_value_mismatches"]
        if r["byte_groups"] and len(examples) < 3:
            examples.append({
                "text": text,
                "tokens": r["tokens"][:24],
                "hf_char_spans": r["hf_char_spans"][:24],
                "byte_spans": r["byte_spans"][:24],
                "groups": r["byte_groups"][:2],
            })
        per_text.append({
            "text": text, "tokens": len(r["tokens"]),
            "byte_tokens": sum(g["count"] for g in r["byte_groups"]),
            "hf_shared_span_groups": c["hf_char_shared_groups"],
            "byte_span_collisions": c["byte_span_collisions"],
            "contract_ok": c["contract_ok"],
        })

    contract_ok = (aggs["byte_span_collisions"] == 0
                   and aggs["byte_span_width_violations"] == 0
                   and aggs["byte_value_mismatches"] == 0)
    return {
        "contract_version": CONTRACT_VERSION,
        "span_unit": "utf-8-byte",
        "tokenizer": tokenizer_path,
        "texts": len(texts),
        "byte_tokens": aggs["byte_tokens"],
        "byte_groups": aggs["byte_groups"],
        # 兼容字段：hf_shared_span_groups == 旧命名的 shared_span_groups
        "shared_span_groups": aggs["hf_shared_span_groups"],
        "hf_shared_span_groups": aggs["hf_shared_span_groups"],
        "byte_span_collisions": aggs["byte_span_collisions"],
        "byte_span_width_violations": aggs["byte_span_width_violations"],
        "byte_value_mismatches": aggs["byte_value_mismatches"],
        "contract_ok": contract_ok,
        "hf_native_independent": aggs["hf_shared_span_groups"] == 0,
        "raw_contract_ok": aggs["hf_shared_span_groups"] == 0,
        "per_text": per_text,
        "examples": examples,
        "note": "hf_shared_span_groups>0 = HF 原生（字符单位）下“一个多字节字符的 k 个字节 token "
                "共享同一 span”的现状缺陷；contract_ok=True 表示按 byte-span-1 契约（字节单位）"
                "标准化后每个字节 token 独立占 1 字节且字节值与源文本逐字节一致。",
    }


def audit_file(tokenizer_path: str, texts: list) -> dict:
    tk, path = _load_tokenizer(tokenizer_path)
    return audit(tk, texts, path)


DEFAULT_TEXTS = ["天气很好，模型质量", "混合型气质散乱", "（内容由AI生成，仅供参考）"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="byte_offsets.py")
    ap.add_argument("--tokenizer", required=True, help="HF 导出目录或 tokenizer.json 路径")
    ap.add_argument("--text", action="append", default=[], help="待探测文本（可重复）")
    ap.add_argument("--text-file", default=None, help="每行一句的文本文件")
    ap.add_argument("--json", default=None, help="审计结果 JSON 落盘路径")
    ap.add_argument("--assert-independent", action="store_true",
                    help="若不满足 byte-span-1 契约则退出码 3")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总行")
    args = ap.parse_args(argv)

    texts = list(args.text)
    if args.text_file:
        lines = io.open(args.text_file, encoding="utf-8", errors="replace").read().split("\n")
        texts += [ln for ln in lines if ln.strip()]
    if not texts:
        texts = list(DEFAULT_TEXTS)

    try:
        tk, path = _load_tokenizer(args.tokenizer)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] 加载失败 %s：%s: %s" % (args.tokenizer, type(exc).__name__, exc))
        return 1

    res = audit(tk, texts, path)
    if not args.quiet:
        for t in texts:
            r = encode_with_byte_spans(tk, t)
            if not r["byte_groups"]:
                continue
            print("text=%r" % t)
            print("  HF 原生 char_spans =%s" % (r["hf_char_spans"][:14],))
            print("  契约 byte_spans    =%s" % (r["byte_spans"][:14],))
            for g in r["byte_groups"][:2]:
                print("  group#%d %s 源字符 span=%s 源字节区间=%s -> 逐字节 span=%s 字节一致=%s"
                      % (g["start_index"], g["byte_tokens"], g["source_char_span"],
                         g["source_byte_span"], g["byte_spans"], g["byte_value_match"]))
    print("[audit] contract=%s unit=%s byte_tokens=%d byte_groups=%d "
          "hf_shared_span_groups=%d byte_span_collisions=%d width_violations=%d "
          "byte_value_mismatches=%d contract_ok=%s"
          % (res["contract_version"], res["span_unit"], res["byte_tokens"], res["byte_groups"],
             res["hf_shared_span_groups"], res["byte_span_collisions"],
             res["byte_span_width_violations"], res["byte_value_mismatches"], res["contract_ok"]))
    if args.json:
        with io.open(args.json, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=2)
        print("[audit] json=%s" % args.json)
    if args.assert_independent and not res["contract_ok"]:
        print("[FAIL] 不满足 %s 契约" % CONTRACT_VERSION)
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    sys.exit(main())
