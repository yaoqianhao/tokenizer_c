#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_thresholds.py —— tokenizer_c 评测侧「阈值单点化 + 指标口径单点化」共享模块。

被 benchmark/eval_metrics.py（语料级主评测）与 benchmark/probe_regression.py
（19 子类探针回归）共同 import，保证：

  * 阈值唯一来源是 benchmark/thresholds.json，代码与文档不再各存一份；
  * fairness_gini / composite_score 与 E:\\tok_eval\\app\\metrics.py 同公式；
  * 失败样本归因分类器在两处共用，避免口径漂移。

本模块只做只读计算，不写任何文件（文档由 benchmark/gen_thresholds_doc.py 生成）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "thresholds.json")

#: 综合分权重与归一化区间：与 E:\tok_eval\app\metrics.py 完全一致（thresholds.json 可覆盖）
TOK_EVAL_COMPOSITE_WEIGHTS: Dict[str, float] = {
    "fertility": 0.35,
    "byte_compression": 0.20,
    "fidelity_pass_rate": 0.25,
    "vocabulary_utilization": 0.10,
    "fairness": 0.10,
}
TOK_EVAL_COMPOSITE_NORM: Dict[str, float] = {
    "fertility_good": 1.0,
    "fertility_bad": 4.0,
    "byte_good": 0.10,
    "byte_bad": 0.40,
}


# ---------------------------------------------------------------------------
# thresholds.json 读取
# ---------------------------------------------------------------------------

def load_thresholds(path: Optional[str] = None) -> Dict[str, Any]:
    """读取阈值配置；文件缺失时返回带默认值的空骨架（不抛异常）。"""
    p = os.path.abspath(path or DEFAULT_THRESHOLDS_PATH)
    data: Dict[str, Any] = {}
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    data["_path"] = p
    data["_sha256_16"] = (hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
                          if os.path.isfile(p) else None)
    return data


def thresholds_provenance(th: Dict[str, Any]) -> Dict[str, Any]:
    """给报告用的阈值出处信息（路径 + 版本 + 摘要）。"""
    return {"path": th.get("_path"), "version": th.get("version"),
            "updated_at": th.get("updated_at"), "sha256_16": th.get("_sha256_16")}


# ---------------------------------------------------------------------------
# tok_eval 口径：公平性 / 综合分 / 词切分
# ---------------------------------------------------------------------------

def gini_coefficient(values: Sequence[float]) -> float:
    """基尼系数（0 = 完全均等）。与 tok_eval app/metrics.py::gini_coefficient 同公式。"""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def fairness_gini(per_language_cost: Dict[str, float]) -> float:
    """跨语种 token 成本（tokens / 原始 UTF-8 字节）的基尼系数，语种等权。

    与 tok_eval app/metrics.py::fairness_gini 同公式。
    """
    if not per_language_cost:
        return 0.0
    return gini_coefficient(list(per_language_cost.values()))


_CJK_WORD_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1100, 0x11FF),      # Hangul Jamo
    (0x3040, 0x30FF),      # Hiragana + Katakana
    (0x3400, 0x4DBF),      # CJK ext A
    (0x4E00, 0x9FFF),      # CJK unified ideographs
    (0xAC00, 0xD7AF),      # Hangul syllables
    (0xF900, 0xFAFF),      # CJK compatibility ideographs
    (0x20000, 0x2FA1F),    # CJK ext B-F
)
_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)


def is_cjk_word_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _CJK_WORD_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def word_spans(text: str) -> List[Tuple[int, int]]:
    """tok_eval 词切分口径：CJK/谚文/假名逐字成词，其余按空白切，纯标点丢弃。"""
    spans: List[Tuple[int, int]] = []
    buf_start = None
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace():
            if buf_start is not None:
                spans.append((buf_start, i))
                buf_start = None
            i += 1
            continue
        if is_cjk_word_char(ch):
            if buf_start is not None:
                spans.append((buf_start, i))
                buf_start = None
            spans.append((i, i + 1))
            i += 1
            continue
        if buf_start is None:
            buf_start = i
        i += 1
    if buf_start is not None:
        spans.append((buf_start, n))
    out: List[Tuple[int, int]] = []
    for start, end in spans:
        frag = text[start:end]
        if any(is_cjk_word_char(c) for c in frag) or _HAS_WORD_CHAR.search(frag):
            out.append((start, end))
    return out


def count_words(text: str) -> int:
    """tok_eval `fertility` 分母：词数（口径见 word_spans）。"""
    return len(word_spans(text))


def grade_fertility(fertility: float) -> str:
    """与 tok_eval app/metrics.py::grade_fertility 同分级。"""
    if fertility <= 0:
        return "n/a"
    if fertility < 1.5:
        return "excellent"
    if fertility < 2.0:
        return "good"
    if fertility < 3.0:
        return "fair"
    if fertility < 4.0:
        return "poor"
    return "bad"


def composite_score(summary: Dict[str, Any],
                    composite_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """加权综合分（0-100，越高越好）；公式与 tok_eval composite_score 一致。

    summary 需提供 fertility / byte_compression / fidelity_pass_rate /
    vocabulary_utilization / fairness_gini 五个输入。
    """
    cfg = composite_cfg or {}
    weights = dict(TOK_EVAL_COMPOSITE_WEIGHTS)
    weights.update(cfg.get("weights") or {})
    norm = dict(TOK_EVAL_COMPOSITE_NORM)
    norm.update(cfg.get("norm") or {})

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    f = float(summary.get("fertility") or 0.0)
    b = float(summary.get("byte_compression") or 0.0)
    f_lo, f_hi = float(norm["fertility_good"]), float(norm["fertility_bad"])
    b_lo, b_hi = float(norm["byte_good"]), float(norm["byte_bad"])
    parts = {
        "fertility": clamp((f_hi - f) / (f_hi - f_lo)) if (f > 0 and f_hi > f_lo) else 0.0,
        "byte_compression": clamp((b_hi - b) / (b_hi - b_lo)) if (b > 0 and b_hi > b_lo) else 0.0,
        "fidelity_pass_rate": clamp(float(summary.get("fidelity_pass_rate") or 0.0)),
        "vocabulary_utilization": clamp(float(summary.get("vocabulary_utilization") or 0.0)),
        "fairness": 1.0 - clamp(float(summary.get("fairness_gini") or 0.0)),
    }
    total = sum(parts[k] * float(weights.get(k, 0.0)) for k in parts)
    return {
        "score": round(100.0 * total, 2),
        "parts": {k: round(v, 6) for k, v in parts.items()},
        "weights": weights,
        "norm": norm,
        "inputs": {"fertility": f, "byte_compression": b,
                   "fidelity_pass_rate": summary.get("fidelity_pass_rate"),
                   "vocabulary_utilization": summary.get("vocabulary_utilization"),
                   "fairness_gini": summary.get("fairness_gini")},
        "source": "tok_eval:app/metrics.py::composite_score(默认权重) + thresholds.json 覆盖",
    }


# ---------------------------------------------------------------------------
# 失败样本归因（tok_eval analyze_tok_eval_csv 口径）
# ---------------------------------------------------------------------------

_ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff", "\u2060")

_ATTR_DESC = {
    "nfkc_fullwidth_fold": "NFKC 兼容折叠（全角→半角等），原文信息不可逆丢失",
    "whitespace_fold": "空白折叠（多空白/制表符/换行→单空格，或首尾空白丢失）",
    "zero_width_removed": "零宽字符被 normalizer 删除",
    "edge_whitespace_strip": "行首/行尾空白被丢弃",
    "decode_empty": "解码结果为空（token 丢失或全为特殊 token）",
    "invalid_utf8_input": "输入或输出含非法 UTF-8（U+FFFD 替换）",
    "decode_error": "encode/decode 调用异常",
    "other": "其他（byte fallback 丢字、解码器拼接差异等，需人工复核）",
}


def attribution_desc(label: str) -> str:
    return _ATTR_DESC.get(label, "")


def classify_roundtrip_failure(text: str, decoded: str,
                               error: Optional[str] = None) -> Tuple[str, str]:
    """把 encode→decode 还原失败按根因归类，返回 (标签, 说明)。"""
    if error:
        return "decode_error", str(error)
    if text == decoded:
        return "match", ""
    if decoded == "":
        return "decode_empty", _ATTR_DESC["decode_empty"]
    try:
        if unicodedata.normalize("NFKC", text) == decoded:
            return "nfkc_fullwidth_fold", _ATTR_DESC["nfkc_fullwidth_fold"]
    except Exception:
        pass
    if re.sub(r"\s+", " ", text).strip() == decoded:
        return "whitespace_fold", _ATTR_DESC["whitespace_fold"]
    stripped = text
    for z in _ZERO_WIDTH:
        stripped = stripped.replace(z, "")
    if stripped == decoded:
        return "zero_width_removed", _ATTR_DESC["zero_width_removed"]
    if text.strip() == decoded.strip() or text.rstrip() == decoded:
        return "edge_whitespace_strip", _ATTR_DESC["edge_whitespace_strip"]
    if "\ufffd" in text or "\ufffd" in decoded:
        return "invalid_utf8_input", _ATTR_DESC["invalid_utf8_input"]
    return "other", _ATTR_DESC["other"]


def summarize_attribution(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按归因标签聚合计数与占比；records 每项需含 `attribution` 字段。"""
    counts: Dict[str, int] = {}
    for r in records:
        lab = r.get("attribution") or "other"
        counts[lab] = counts.get(lab, 0) + 1
    total = sum(counts.values())
    shares = {k: round(v / float(total), 6) for k, v in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0]))} if total else {}
    return {"total_failures": total, "counts": dict(sorted(counts.items(),
                                                           key=lambda kv: (-kv[1], kv[0]))),
            "shares": shares,
            "descriptions": {k: attribution_desc(k) for k in counts}}


# ---------------------------------------------------------------------------
# 门禁求值
# ---------------------------------------------------------------------------

_OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "==": lambda a, b: a == b,
}


def norm_mode_from_cli_extra(cli_extra: Optional[Sequence[str]]) -> str:
    """从 --cli-extra 推断归一化模式（tok-cli 默认 legacy NFKC）。"""
    joined = " ".join(cli_extra or [])
    if "--norm=reversible" in joined or "--norm-reversible" in joined:
        return "reversible"
    if "--norm=nfc" in joined:
        return "nfc"
    return "nfkc"  # 显式 --norm=nfkc 或 CLI 默认 legacy NFKC


def resolve_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """解析门禁 key：优先扁平键，其次点号路径。返回 None 表示不可计算。"""
    if not isinstance(metrics, dict) or not key:
        return None
    if key in metrics and not isinstance(metrics[key], (dict, list)):
        return metrics[key]
    cur: Any = metrics
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    if isinstance(cur, (dict, list)):
        return None
    return cur


def build_gate_metrics(fairness: Optional[Dict[str, Any]],
                       composite: Optional[Dict[str, Any]],
                       stability_overall: Optional[float],
                       pooled_pass_rate: Optional[float],
                       totals_bpt: Optional[float],
                       utf8_completeness: Optional[float],
                       lang_results: Optional[Dict[str, Any]] = None,
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把报告各段落折算成扁平门禁指标（含 langs.<lang>.* 派生键）。"""
    m: Dict[str, Any] = {
        "fairness_gini": (fairness or {}).get("gini"),
        "composite_score": (composite or {}).get("score"),
        "stability": stability_overall,
        "roundtrip_pass_rate": pooled_pass_rate,
        "bpt_total": totals_bpt,
        "micro_byte_compression": (1.0 / totals_bpt) if totals_bpt else None,
        "utf8_completeness": utf8_completeness,
    }
    for lang, v in (lang_results or {}).items():
        tested = v.get("roundtrip_lines") or 0
        mism = v.get("roundtrip_mismatches") or 0
        bpt = v.get("bpt")
        m["langs.%s.roundtrip_pass_rate" % lang] = ((tested - mism) / float(tested)) if tested else None
        m["langs.%s.bpt" % lang] = bpt
        m["langs.%s.byte_compression" % lang] = (1.0 / bpt) if bpt else None
        m["langs.%s.stability" % lang] = v.get("stability")
        m["langs.%s.oov_rate" % lang] = v.get("oov_rate")
    if extra:
        m.update(extra)
    return m


def _cond_ok(cond: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    if not cond:
        return True
    if "norm_mode" in cond and ctx.get("norm_mode") != cond["norm_mode"]:
        return False
    if "norm_mode_in" in cond and ctx.get("norm_mode") not in list(cond["norm_mode_in"]):
        return False
    if "cli_extra_contains" in cond:
        need = cond["cli_extra_contains"]
        needles = [need] if isinstance(need, str) else list(need)
        joined = " ".join(ctx.get("cli_extra") or [])
        if not all(n in joined for n in needles):
            return False
    return True


def evaluate_gates(metrics: Dict[str, Any], gates: Sequence[Dict[str, Any]],
                   ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """逐条判定门禁；指标为 None（不可计算）或 applies_when 未命中的条目记入 skipped。"""
    ctx = dict(ctx or {})
    ctx.setdefault("norm_mode", norm_mode_from_cli_extra(ctx.get("cli_extra")))
    checked: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for g in gates or []:
        cond = g.get("applies_when") or {}
        if not _cond_ok(cond, ctx):
            skipped.append({"key": g.get("key"), "op": g.get("op"), "threshold": g.get("value"),
                            "reason": "applies_when 未命中：%s" % json.dumps(cond, ensure_ascii=False)})
            continue
        val = resolve_metric(metrics, g.get("key"))
        if val is None:
            skipped.append({"key": g.get("key"), "op": g.get("op"), "threshold": g.get("value"),
                            "reason": "指标不可计算（null）"})
            continue
        op = g.get("op", "<=")
        if op not in _OPS:
            skipped.append({"key": g.get("key"), "reason": "未知比较符：%s" % op})
            continue
        value = float(val)
        passed = bool(_OPS[op](value, float(g["value"])))
        checked.append({"key": g["key"], "op": op, "threshold": float(g["value"]),
                        "value": value, "passed": passed, "note": g.get("note", ""),
                        "applies_when": cond or None})
    return {"passed": all(r["passed"] for r in checked) if checked else True,
            "checked": checked, "skipped": skipped,
            "norm_mode": ctx.get("norm_mode"),
            "counts": {"checked": len(checked),
                       "passed": sum(1 for r in checked if r["passed"]),
                       "failed": sum(1 for r in checked if not r["passed"]),
                       "skipped": len(skipped)}}


def render_gates_markdown(gates: Dict[str, Any], title: str = "门禁判定") -> str:
    """门禁结果渲染为 Markdown（stdout 与报告文件共用）。"""
    out = ["## %s" % title]
    c = gates.get("counts") or {}
    out.append("- 结论：**%s**（检查 %d 项，通过 %d，未通过 %d，跳过 %d；归一化模式判定：%s）" % (
        "PASS" if gates.get("passed") else "FAIL",
        c.get("checked", 0), c.get("passed", 0), c.get("failed", 0), c.get("skipped", 0),
        gates.get("norm_mode") or "-"))
    out.append("")
    out.append("| 指标 | 判定 | 阈值 | 实测 | 结论 | 备注 |")
    out.append("|---|---|---|---|---|---|")
    for r in gates.get("checked") or []:
        out.append("| " + " | ".join([
            str(r["key"]), str(r["op"]), _fmt(r["threshold"]), _fmt(r["value"]),
            "PASS" if r["passed"] else "FAIL", str(r.get("note") or "")]) + " |")
    for s in gates.get("skipped") or []:
        out.append("| " + " | ".join([
            str(s.get("key")), str(s.get("op") or "-"), _fmt(s.get("threshold")), "-",
            "SKIP", str(s.get("reason") or "")]) + " |")
    if not (gates.get("checked") or gates.get("skipped")):
        out.append("| - | - | - | - | - | 无门禁条目 |")
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) >= 100:
        return "%d" % int(f)
    return ("%.6f" % f).rstrip("0").rstrip(".") if abs(f) < 1 else ("%.4f" % f)
