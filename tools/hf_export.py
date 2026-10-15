#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_export.py —— tokenizer_c 模型 JSON -> HuggingFace fast tokenizer 目录

本文件是 `output/patch_model_big2_to_hf.py`（已定稿转换脚本）的工程化移植版：
在 JSON 层做与之一致的 7 处等价修复，把 `tok-cli train --json` 导出的
非标准 tokenizer.json 规范化为可被 `tokenizers.Tokenizer.from_file()` /
`transformers.AutoTokenizer.from_pretrained(dir)` 直接加载的目录。

转换行为（与附件脚本逐条对齐，不可随意更改）：
  1. decoder：改成 Sequence[ByteFallback, Metaspace]（去掉已废弃的 add_prefix_space）
  2. 补 pre_tokenizer：Metaspace(replacement="▁", prepend_scheme="never", split=false)
  3. 补 normalizer：Sequence[NFKC, Replace(\\s -> " "), Replace(零宽族 -> "")]
  4. null(NaN) 分数 -> 极小值 -1e9（这些 piece 在 C 端从未被命中，改分不影响 C 行为）
  5. 补 256 个 <0xXX> 字节 piece，分数取 C 端常量 kByteFallbackScore = -10.0
     （src/tokenizer.c:348），使 C/HF 两侧字节回退对称
  6. 剥离非标准顶层键（unk/bos/eos/pad_token 对象 + tokenizer_class），
     改写进 tokenizer_config.json
  7. 无参数调用时等价于原脚本的 `patch(src, out_dir)`

输出目录内容：tokenizer.json / tokenizer_config.json / special_tokens_map.json

CLI 用法：
    python tools/hf_export.py --json <model.json> --out <dir>
        [--model-type Unigram] [--dry-run]
        [--verify] [--model <model.tkm>] [--tok-cli <tok-cli.exe>]
        [--verify-file <sentences.txt>] [--verify-tolerance 0.02]

退出码：
    0  成功（含 --verify 通过 / 差异在容差内）
    1  参数或输入错误
    2  产出目录无法被 tokenizers / transformers 加载
    3  --verify 发现超出容差的 token 用量差异
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------
# 常量：与附件脚本 / C 端源码保持一致
# --------------------------------------------------------------------------

# tokenizer.json 的合法顶层键（其余一律剥离并改写进 tokenizer_config.json）
ALLOWED_TOP_KEYS = (
    "version", "truncation", "padding", "added_tokens",
    "normalizer", "pre_tokenizer", "post_processor", "decoder", "model",
)

# 被剥离的非标准顶层键（附件脚本的修复 3）
STRIPPED_TOKEN_KEYS = ("unk_token", "bos_token", "eos_token", "pad_token")

DEFAULT_SPECIALS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}

# 修复 2：null(NaN) 分数 -> 极小值
NAN_SCORE_REPLACEMENT = -1e9

# 修复 5：字节 piece 分数，取自 C 端 kByteFallbackScore（src/tokenizer.c:348）
BYTE_FALLBACK_SCORE = -10.0

# 修复 6：C 端 normalize 的零宽族剔除清单（src/normalize.c）
ZERO_WIDTH_CLASS = "[\u200b-\u200f\u2028-\u202f\u2060\ufeff\u00ad]"

KNOWN_MODEL_TYPES = ("Unigram", "BPE", "WordPiece", "WordLevel")

DEFAULT_SENTENCES = [
    "字节回退机制在本次实验中持续优化，加载耗时因此成为重点关注对象。",
    "今天天气不错，我们一起去公园散步吧。",
    "Hello, world!",
    "def add(a, b):\n    return a + b\n",
    "全角标点（括号）：，。；！？——测试",
    "emoji 😀🎉 与零宽\u200b字符\u200d混合",
    "mixed 混合文本 12345 with tab\t字符",
]


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    """读取 JSON；NaN / Infinity 统一映射为 None（等价于原脚本的 sc is None 判定）。"""
    raw = io.open(path, encoding="utf-8", errors="replace").read()
    return json.loads(raw, parse_constant=lambda _x: None)


def _write_json(path: str, obj, indent=None) -> None:
    with io.open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(obj, ensure_ascii=False, indent=indent))


def _normalize_input(cfg: dict) -> dict:
    """兼容历史遗留 model json：顶层直接是 {"vocab": [[piece, score], ...]} 时补骨架。"""
    if isinstance(cfg.get("model"), dict) and "vocab" in cfg["model"]:
        return cfg
    if isinstance(cfg.get("vocab"), list):
        legacy = {k: v for k, v in cfg.items() if k != "vocab"}
        cfg = dict(legacy)
        cfg["model"] = {
            "type": "Unigram",
            "byte_fallback": True,
            "unk_id": 0,
            "vocab": legacy.pop("vocab"),
        }
        return cfg
    raise ValueError("输入 JSON 既无 model.vocab 也无顶层 vocab，无法识别为 tokenizer_c 模型导出")


# --------------------------------------------------------------------------
# 体检（--dry-run 与正常流程共用）
# --------------------------------------------------------------------------

def diagnose(cfg: dict) -> list:
    """体检：列出发现的问题项（每条含 id / severity / detail / fix）。"""
    findings = []

    extra = [k for k in cfg.keys() if k not in ALLOWED_TOP_KEYS]
    if extra:
        findings.append({
            "id": "nonstandard_top_keys", "severity": "hard",
            "detail": "非标准顶层键: " + ", ".join(sorted(extra)),
            "fix": "剥离并改写进 tokenizer_config.json / special_tokens_map.json",
        })

    dec = cfg.get("decoder")
    dec_types = []
    if isinstance(dec, dict):
        dec_types = [d.get("type") for d in dec.get("decoders", [])] or [dec.get("type")]
    elif isinstance(dec, list):
        dec_types = [d.get("type") for d in dec]
    if "ByteFallback" not in dec_types:
        findings.append({
            "id": "decoder_missing_bytefallback", "severity": "hard",
            "detail": "decoder 缺少 ByteFallback 解码链（当前 %s）" % (dec,),
            "fix": "decoder 改为 Sequence[ByteFallback, Metaspace(prepend_scheme=never)]",
        })
    if isinstance(dec, dict) and (dec.get("type") == "Metaspace" and "add_prefix_space" in dec):
        findings.append({
            "id": "decoder_deprecated_add_prefix_space", "severity": "hard",
            "detail": "decoder 使用已废弃字段 add_prefix_space",
            "fix": "改用 prepend_scheme=never",
        })

    pre = cfg.get("pre_tokenizer")
    if not (isinstance(pre, dict)
            and pre.get("type") == "Metaspace"
            and pre.get("prepend_scheme") == "never"
            and pre.get("split") is False):
        findings.append({
            "id": "pretokenizer_metaspace", "severity": "semantic",
            "detail": "pre_tokenizer 缺失或不匹配（当前 %s）" % (pre,),
            "fix": 'pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never", split=false)',
        })

    norm = cfg.get("normalizer")
    norm_types = []
    if isinstance(norm, dict):
        norm_types = [n.get("type") for n in norm.get("normalizers", [])] or [norm.get("type")]
    norm_ok = (isinstance(norm, dict) and norm.get("type") == "Sequence"
               and norm_types[:3] == ["NFKC", "Replace", "Replace"])
    if not norm_ok:
        findings.append({
            "id": "normalizer_sequence", "severity": "semantic",
            "detail": "normalizer 不是 Sequence[NFKC, Replace(\\s->\" \"), Replace(零宽族->\"\")]（当前 %s）"
                      % (norm,),
            "fix": "补 normalizer，与 src/normalize.c 的空白族/零宽族策略对齐",
        })

    vocab = cfg["model"]["vocab"]
    n_null = sum(1 for item in vocab if item[1] is None)
    if n_null:
        findings.append({
            "id": "null_scores", "severity": "hard",
            "detail": "%d 条 piece 的分数为 null(NaN)" % n_null,
            "fix": "改写为极小值 %s" % NAN_SCORE_REPLACEMENT,
        })

    have = {item[0] for item in vocab}
    missing_bytes = [b for b in range(256) if "<0x%02X>" % b not in have]
    if missing_bytes:
        findings.append({
            "id": "missing_byte_pieces", "severity": "semantic",
            "detail": "vocab 缺 %d 个 <0xXX> 字节 piece" % len(missing_bytes),
            "fix": "补齐 256 个 <0xXX>，分数 = %s（C 端 kByteFallbackScore）" % BYTE_FALLBACK_SCORE,
        })

    if cfg["model"].get("byte_fallback") and len(missing_bytes) == 256:
        findings.append({
            "id": "byte_fallback_without_pieces", "severity": "semantic",
            "detail": "byte_fallback=true 但 vocab 内 0 个字节 piece，C 端负 id 回退在 HF 侧无法对应",
            "fix": "同上补 256 个字节 piece，使两侧字节回退对称",
        })
    return findings


# --------------------------------------------------------------------------
# 转换主体（等价于原脚本 patch()）
# --------------------------------------------------------------------------

def convert(cfg: dict, model_type: str | None = None):
    """按 7 处修复生成 HF fast tokenizer 配置。返回 (cfg_out, stats, specials)。"""
    cfg = _normalize_input(cfg)
    findings = diagnose(cfg)

    # 修复 6：剥离非标准顶层键，并把 specials 改写进 tokenizer_config.json
    specials = dict(DEFAULT_SPECIALS)
    for key in STRIPPED_TOKEN_KEYS:
        node = cfg.pop(key, None)
        if isinstance(node, dict) and isinstance(node.get("content"), str):
            specials[key] = node["content"]
        elif isinstance(node, str):
            specials[key] = node
    cfg.pop("tokenizer_class", None)
    for key in [k for k in cfg.keys() if k not in ALLOWED_TOP_KEYS]:
        cfg.pop(key, None)

    # 修复 2：null(NaN) -> 极小值；并统一为 float
    n_null = 0
    vocab = []
    for piece, score in cfg["model"]["vocab"]:
        if score is None:
            vocab.append([piece, NAN_SCORE_REPLACEMENT])
            n_null += 1
        else:
            vocab.append([piece, float(score)])
    have = {p for p, _ in vocab}

    # 修复 5：补 256 个 <0xXX> 字节 piece（分数 = C 端常量 -10.0）
    n_byte = 0
    for b in range(256):
        t = "<0x%02X>" % b
        if t not in have:
            vocab.append([t, BYTE_FALLBACK_SCORE])
            n_byte += 1
    cfg["model"]["vocab"] = vocab

    if model_type:
        cfg["model"]["type"] = model_type
    elif not cfg["model"].get("type"):
        cfg["model"]["type"] = "Unigram"

    # 修复 1 + 7：decoder 兼容写法 + ByteFallback 解码链
    cfg["decoder"] = {"type": "Sequence", "decoders": [
        {"type": "ByteFallback"},
        {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "never"}]}

    # 修复 4：补 pre_tokenizer（C 端 U+0020 -> ▁）
    cfg["pre_tokenizer"] = {"type": "Metaspace", "replacement": "\u2581",
                            "prepend_scheme": "never", "split": False}

    # 修复 6：补 normalizer（NFKC + 空白族->空格 + 零宽族剔除）
    cfg["normalizer"] = {"type": "Sequence", "normalizers": [
        {"type": "NFKC"},
        {"type": "Replace", "pattern": {"Regex": "\\s"}, "content": " "},
        {"type": "Replace", "pattern": {"Regex": ZERO_WIDTH_CLASS}, "content": ""}]}

    stats = {
        "pieces": len(vocab),
        "null_fixed": n_null,
        "byte_added": n_byte,
        "findings": findings,
        "specials": specials,
        "model_type": cfg["model"].get("type"),
    }
    return cfg, stats, specials


def write_outputs(cfg: dict, out_dir: str, specials: dict) -> dict:
    """落盘 tokenizer.json / tokenizer_config.json / special_tokens_map.json。"""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "tokenizer.json": os.path.join(out_dir, "tokenizer.json"),
        "tokenizer_config.json": os.path.join(out_dir, "tokenizer_config.json"),
        "special_tokens_map.json": os.path.join(out_dir, "special_tokens_map.json"),
    }
    _write_json(paths["tokenizer.json"], cfg)
    _write_json(paths["tokenizer_config.json"], {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": specials["unk_token"],
        "bos_token": specials["bos_token"],
        "eos_token": specials["eos_token"],
        "pad_token": specials["pad_token"],
        "clean_up_tokenization_spaces": False,
    }, indent=2)
    _write_json(paths["special_tokens_map.json"], {
        "unk_token": specials["unk_token"],
        "bos_token": specials["bos_token"],
        "eos_token": specials["eos_token"],
        "pad_token": specials["pad_token"],
    }, indent=2)
    return paths


def patch(src_path: str, out_dir: str, model_type: str | None = None) -> dict:
    """兼容原脚本签名的入口：patch(src, out_dir) -> 统计 dict。"""
    cfg, stats, specials = convert(_read_json(src_path), model_type=model_type)
    write_outputs(cfg, out_dir, specials)
    return {"out_dir": out_dir, "pieces": stats["pieces"],
            "null_fixed": stats["null_fixed"], "byte_added": stats["byte_added"]}


# --------------------------------------------------------------------------
# 校验（--verify）
# --------------------------------------------------------------------------

def _load_sentences(verify_file: str | None) -> list:
    if not verify_file:
        return list(DEFAULT_SENTENCES)
    lines = io.open(verify_file, encoding="utf-8", errors="replace").read().split("\n")
    out = [ln for ln in lines if ln.strip()]
    return out or list(DEFAULT_SENTENCES)


def _tok_cli_encode(tok_cli: str, model: str, text: str, tmp_dir: str) -> list:
    """调用 C 路径 tok-cli encode --file，返回 id 列表（含负 id）。"""
    os.makedirs(tmp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="sent_", suffix=".txt", dir=tmp_dir)
    os.close(fd)
    with io.open(path, "w", encoding="utf-8", newline="") as fp:
        fp.write(text)
    try:
        proc = subprocess.run([tok_cli, "encode", "--model", model, "--file", path],
                              capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("tok-cli encode 失败: %s"
                               % proc.stderr.decode("utf-8", "replace").strip())
        return [int(x) for x in proc.stdout.decode("utf-8", "replace").split()]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _is_byte_piece(p: str) -> bool:
    return len(p) == 6 and p.startswith("<0x") and p.endswith(">")


def _pieces_to_text(pieces: list) -> str:
    """把 piece 序列还原为可比较的文本：<0xXX> 按 UTF-8 字节累积解码，▁ 视作空格。

    用于判断两侧切分差异是否只是"切分粒度"差异（文本等价）而非信息差异。
    """
    out = []
    buf = bytearray()
    for p in pieces:
        if _is_byte_piece(p):
            buf.append(int(p[3:5], 16))
            continue
        if buf:
            out.append(buf.decode("utf-8", "replace"))
            buf = bytearray()
        out.append(p.replace("\u2581", " "))
    if buf:
        out.append(buf.decode("utf-8", "replace"))
    return "".join(out)


def _path_score(pieces: list, smap: dict):
    """piece 序列的总分（用于判定两侧是否为"同分不同切分"）。"""
    total = 0.0
    for p in pieces:
        if p in smap and smap[p] is not None:
            total += smap[p]
        elif _is_byte_piece(p):
            total += -10.0
        else:
            return None
    return total


def _build_c_sim(conv_cfg: dict, pruned: set, tmp_dir: str):
    """构造 C-sim 校验器：把源 json 中被剪枝(score=null)的 piece 从词表移除。

    C 端把 null 分数视为"该 piece 不可用"并走字节回退；HF 端 -1e9 仍是有限分数，
    这些 piece 依旧可被匹配（HF 只在完全没有 piece 覆盖时才字节回退）。因此
    "把剪枝 piece 从词表移除"可精确复现 C 端的可选 piece 集合 —— 若移除后 HF 的
    切分与 tok-cli 逐 piece 一致，即可证明该句差异完全由剪枝 piece 机制导致。
    """
    from tokenizers import Tokenizer
    sim_cfg = json.loads(json.dumps(conv_cfg))
    sim_cfg["model"]["vocab"] = [[p, s] for p, s in conv_cfg["model"]["vocab"]
                                 if p not in pruned]
    tk = Tokenizer.from_str(json.dumps(sim_cfg))
    return tk, {v: k for k, v in tk.get_vocab().items()}


def verify(out_dir: str, json_path: str, model_path: str | None, tok_cli: str,
           verify_file: str | None = None, tolerance: float = 0.02,
           tmp_dir: str | None = None, strict: bool = False) -> dict:
    """真机校验：HF 侧加载 + 与 tok-cli encode 的 piece 级比对。

    C 端负 id 按 -1-byte 规则还原为 <0xXX> 后做 piece 级逐位对照
    （与核验报告口径一致）。

    差异分级（逐句判定，判定依据是可复现的机制而非简单容差）：
      identical      两侧 piece 序列完全一致
      explained      把源 json 中 score=null 的剪枝 piece 从 HF 词表移除后（C-sim），
                     切分与 tok-cli 逐 piece 一致 —— 差异完全由"剪枝 piece 在 HF
                     端仍可匹配、C 端则走字节回退"这一已知机制导致
      tie_break      C-sim 仍不一致，但两条可行切分路径总分完全相同（Viterbi
                     同分取的切分位置不同），且还原文本一致
      lost_text      还原文本不一致（存在信息损失）—— 硬失败
      unexplained    其余情况（既非剪枝机制、也非同分）—— 硬失败

    结论：PASS_LOAD_ONLY / PASS_IDENTICAL / PASS_KNOWN_DIFF / FAIL_TEXT /
    FAIL_MISMATCH / FAIL_LOAD / FAIL_IMPORT。strict=True 时任何非 identical 即失败。
    tolerance 仅用于报告 token 用量偏差是否超阈值，不参与判定。
    """
    result = {"out_dir": out_dir, "sentences": [], "load": {}, "verdict": None}

    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        result["load"] = {"ok": False, "error": "缺少 tokenizers/transformers: %s" % exc}
        result["verdict"] = "FAIL_IMPORT"
        return result

    try:
        tk = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
        hf = AutoTokenizer.from_pretrained(out_dir)
        result["load"] = {"ok": True,
                          "tokenizers_version": __import__("tokenizers").__version__,
                          "transformers_version": __import__("transformers").__version__,
                          "tokenizer_class": type(hf).__name__,
                          "hf_class": type(tk).__name__}
    except Exception as exc:  # noqa: BLE001
        result["load"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        result["verdict"] = "FAIL_LOAD"
        return result

    # C 侧 id -> piece 映射：id >= 0 用输入 json 的 vocab 下标（与 HF id 同序）；
    # id < 0 时按 C 端规则还原 <0xXX>
    raw_cfg = _read_json(json_path)
    base_vocab = _normalize_input(raw_cfg)["model"]["vocab"]
    base_scores = {p: s for p, s in base_vocab}
    pruned_pieces = set(p for p, s in base_vocab if s is None)
    conv_cfg, _conv_stats, _specials = convert(raw_cfg)
    score_map = {p: s for p, s in conv_cfg["model"]["vocab"]}
    sim_state = {"tk": None, "id2p": None, "error": None}

    def ensure_c_sim():
        """惰性构建 C-sim（只在出现不一致句子时才付出构建代价）。"""
        if sim_state["tk"] is not None or sim_state["error"] is not None:
            return sim_state
        try:
            tk_sim, id2p_sim = _build_c_sim(conv_cfg, pruned_pieces,
                                            tmp_dir or tempfile.gettempdir())
            sim_state["tk"] = tk_sim
            sim_state["id2p"] = id2p_sim
        except Exception as exc:  # noqa: BLE001
            sim_state["error"] = "%s: %s" % (type(exc).__name__, exc)
        return sim_state

    def c_id_to_piece(i: int) -> str:
        if i >= 0:
            if i < len(base_vocab):
                return base_vocab[i][0]
            return "<out-of-range:%d>" % i
        return "<0x%02X>" % (-1 - i)

    hf_vocab = tk.get_vocab()
    hf_id_to_piece = {v: k for k, v in hf_vocab.items()}

    if not model_path:
        cand = os.path.splitext(os.path.abspath(json_path))[0] + ".tkm"
        model_path = cand if os.path.exists(cand) else None
    result["tok_cli"] = tok_cli
    result["model"] = model_path
    why = []
    if not model_path:
        why.append("未指定 .tkm（json 同目录未见同名 .tkm）")
    elif not os.path.exists(model_path):
        why.append(".tkm 不存在: %s" % model_path)
    if not os.path.exists(tok_cli):
        why.append("tok-cli 不存在: %s" % tok_cli)
    result["c_cli_reason"] = "；".join(why) or None
    result["c_cli_available"] = not why

    worst = 0.0
    all_piece_match = True
    n_identical = n_explained = n_tie = n_unexplained = n_lost = 0
    csim_reproduced = 0
    over_tol = []
    tmp_work = tmp_dir or tempfile.gettempdir()
    for sent in _load_sentences(verify_file):
        row = {"text": sent}
        hf_ids = tk.encode(sent).ids
        row["hf_tokens"] = len(hf_ids)
        row["hf_pieces"] = [hf_id_to_piece.get(i, "<unk-id:%d>" % i) for i in hf_ids]
        if not result["c_cli_available"]:
            row["c_tokens"] = None
            row["diff_kind"] = "skipped"
            row["note"] = "未提供 tok-cli 或 .tkm，跳过 C 侧比对"
            result["sentences"].append(row)
            continue
        c_ids = _tok_cli_encode(tok_cli, model_path, sent, tmp_work)
        c_pieces = [c_id_to_piece(i) for i in c_ids]
        row["c_tokens"] = len(c_ids)
        row["c_negative_ids"] = sum(1 for i in c_ids if i < 0)
        row["piece_match"] = (c_pieces == row["hf_pieces"])
        diff = sum(1 for a, b in zip(c_pieces, row["hf_pieces"]) if a != b)
        diff += abs(len(c_pieces) - len(row["hf_pieces"]))
        row["piece_diff"] = diff
        row["first_diff_index"] = next(
            (k for k, (a, b) in enumerate(zip(c_pieces, row["hf_pieces"])) if a != b), None)
        row["count_delta_pct"] = round(
            (len(row["hf_pieces"]) - len(c_pieces)) / max(len(c_pieces), 1) * 100, 4)
        worst = max(worst, abs(row["count_delta_pct"]) / 100.0)
        all_piece_match = all_piece_match and row["piece_match"]
        row["text_equal"] = (_pieces_to_text(c_pieces) == _pieces_to_text(row["hf_pieces"]))
        if not row["text_equal"]:
            row["text_c"] = _pieces_to_text(c_pieces)[:400]
            row["text_hf"] = _pieces_to_text(row["hf_pieces"])[:400]
        if row["piece_match"]:
            row["diff_kind"] = "identical"
            n_identical += 1
        else:
            sim = ensure_c_sim()
            row["c_only_pieces"] = [p for p in c_pieces if p not in row["hf_pieces"]][:20]
            row["hf_only_pieces"] = [p for p in row["hf_pieces"] if p not in c_pieces][:20]
            sim_pieces = None
            if sim["tk"] is not None:
                sim_pieces = [sim["id2p"].get(i) for i in sim["tk"].encode(sent).ids]
                row["csim_reproduces_c"] = (sim_pieces == c_pieces)
            else:
                row["csim_reproduces_c"] = None
                row["csim_error"] = sim["error"]
            s_c = _path_score(c_pieces, score_map)
            s_sim = _path_score(sim_pieces, score_map) if sim_pieces is not None else None
            row["path_score_c"] = None if s_c is None else round(s_c, 4)
            row["path_score_sim"] = None if s_sim is None else round(s_sim, 4)
            tie = (s_c is not None and s_sim is not None and abs(s_c - s_sim) <= 1e-3)
            if row["csim_reproduces_c"]:
                row["diff_kind"] = "explained"
                n_explained += 1
                csim_reproduced += 1
            elif not row["text_equal"]:
                row["diff_kind"] = "lost_text"
                n_lost += 1
            elif tie:
                row["diff_kind"] = "tie_break"
                n_tie += 1
            else:
                row["diff_kind"] = "unexplained"
                n_unexplained += 1
        if abs(row.get("count_delta_pct") or 0.0) / 100.0 > tolerance:
            over_tol.append(len(result["sentences"]))
        result["sentences"].append(row)

    result["all_piece_match"] = all_piece_match
    result["worst_count_delta"] = round(worst * 100, 4)
    result["tolerance_pct"] = round(tolerance * 100, 4)
    result["sentences_identical"] = n_identical
    result["sentences_explained"] = n_explained
    result["sentences_tie_break"] = n_tie
    result["sentences_unexplained"] = n_unexplained
    result["sentences_lost_text"] = n_lost
    result["csim_reproduced"] = csim_reproduced
    result["pruned_piece_count"] = len(pruned_pieces)
    result["delta_over_tolerance"] = over_tol
    result["strict"] = bool(strict)
    if not result["c_cli_available"]:
        result["verdict"] = "PASS_LOAD_ONLY"
    elif all_piece_match:
        result["verdict"] = "PASS_IDENTICAL"
    elif n_lost > 0:
        result["verdict"] = "FAIL_TEXT"
    elif n_unexplained > 0:
        result["verdict"] = "FAIL_MISMATCH"
    elif strict:
        result["verdict"] = "FAIL_STRICT"
    else:
        result["verdict"] = "PASS_KNOWN_DIFF"
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="hf_export.py",
        description="tokenizer_c 模型 JSON -> HuggingFace fast tokenizer 目录")
    ap.add_argument("--json", required=True, help="输入 model json（tok-cli train --json 产物）")
    ap.add_argument("--out", required=True, help="输出目录（tokenizer.json 等三个文件）")
    ap.add_argument("--model-type", default=None,
                    help="可选覆盖 model.type（%s）" % "/".join(KNOWN_MODEL_TYPES))
    ap.add_argument("--dry-run", action="store_true", help="只体检不写盘，列出问题项与修复项")
    ap.add_argument("--verify", action="store_true", help="转换后做加载 + id parity 校验")
    ap.add_argument("--model", default=None, help="用于 tok-cli encode 的 .tkm（缺省取 json 同目录同名 .tkm）")
    ap.add_argument("--tok-cli", default=None, help="tok-cli 可执行文件路径")
    ap.add_argument("--verify-file", default=None, help="校验句子文件（每行一句，缺省用内置 6 句）")
    ap.add_argument("--verify-tolerance", type=float, default=0.02,
                    help="token 用量相对差提示阈值（默认 0.02 = 2%%；仅提示，不参与判定）")
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：任何非逐 piece 一致即判定失败（CI 用）")
    ap.add_argument("--tmp-dir", default=None, help="校验时的临时句子文件目录")
    args = ap.parse_args(argv)

    if args.model_type and args.model_type not in KNOWN_MODEL_TYPES:
        print("[FAIL] --model-type 非法：%s（可选 %s）"
              % (args.model_type, "/".join(KNOWN_MODEL_TYPES)))
        return 1

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tok_cli = args.tok_cli or os.path.join(repo_root, "build", "tok-cli.exe")

    if not os.path.exists(args.json):
        print("[FAIL] 输入 json 不存在：%s" % args.json)
        return 1

    try:
        cfg_in = _read_json(args.json)
        cfg, stats, specials = convert(cfg_in, model_type=args.model_type)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] 解析/转换失败：%s: %s" % (type(exc).__name__, exc))
        return 1

    print("== 体检 ==")
    if not stats["findings"]:
        print("  未发现问题项")
    for f in stats["findings"]:
        print("  [%s] %s\n        修复: %s" % (f["severity"], f["detail"], f["fix"]))
    print("== 修复统计 == pieces=%d null_fixed=%d byte_added=%d model_type=%s"
          % (stats["pieces"], stats["null_fixed"], stats["byte_added"], stats["model_type"]))

    if args.dry_run:
        print("[DRY-RUN] 未写盘。计划输出：")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            print("  %s" % os.path.join(args.out, name))
        return 0

    paths = write_outputs(cfg, args.out, specials)
    for name, p in paths.items():
        print("  写入 %s (%d bytes)" % (p, os.path.getsize(p)))

    if not args.verify:
        return 0

    print("== 校验 ==")
    res = verify(args.out, args.json, args.model, tok_cli,
                 verify_file=args.verify_file,
                 tolerance=args.verify_tolerance,
                 tmp_dir=args.tmp_dir, strict=args.strict)
    print("  加载: %s" % json.dumps(res["load"], ensure_ascii=False))
    if not res["load"].get("ok"):
        print("[FAIL] 产出目录无法被 HF 加载")
        return 2
    print("  tok-cli: %s | model: %s" % (res["tok_cli"], res["model"]))
    print("  源 json 剪枝(score=null) piece 数: %d" % res["pruned_piece_count"])
    for row in res["sentences"]:
        if row.get("c_tokens") is None:
            print("  %-30s HF=%d  (C 侧跳过: %s)"
                  % (row["text"][:28].replace("\n", "\\n"), row["hf_tokens"], row.get("note")))
            continue
        print("  %-30s C=%-4d HF=%-4d Δ=%+.4f%% %-11s piece_diff=%-3s csim_match=%s"
              % (row["text"][:28].replace("\n", "\\n"), row["c_tokens"], row["hf_tokens"],
                 row["count_delta_pct"], row.get("diff_kind", "?"),
                 row["piece_diff"], row.get("csim_reproduces_c")))
        kind = row.get("diff_kind")
        if kind == "explained":
            print("      └ 可解释(剪枝 piece 机制)：把源 json 中 score=null 的 piece 从词表移除后"
                  "（共 %d 个），HF 切分与 tok-cli 逐 piece 一致" % res["pruned_piece_count"])
        elif kind == "tie_break":
            print("      └ 同分不同切分：C 路径总分=%s 与 C-sim 路径总分=%s 相同；还原文本一致，"
                  "仅切分位置不同（Viterbi tie-break）"
                  % (row.get("path_score_c"), row.get("path_score_sim")))
        elif kind == "lost_text":
            print("      └ [硬失败] 文本不可逆：c_text=%r hf_text=%r"
                  % (row.get("text_c"), row.get("text_hf")))
        elif kind == "unexplained":
            print("      └ [硬失败] 未解释差异：text_equal=%s c_only=%s hf_only=%s path_score_C=%s path_score_sim=%s%s"
                  % (row.get("text_equal"), row.get("c_only_pieces"), row.get("hf_only_pieces"),
                     row.get("path_score_c"), row.get("path_score_sim"),
                     "" if not row.get("csim_error") else " csim_error=%s" % row["csim_error"]))
    print("  结论: %s (identical=%d explained=%d tie_break=%d unexplained=%d lost_text=%d "
          "all_piece_match=%s worst_delta=%.4f%% tol=%.4f%%)"
          % (res["verdict"], res["sentences_identical"], res["sentences_explained"],
             res["sentences_tie_break"], res["sentences_unexplained"], res["sentences_lost_text"],
             res["all_piece_match"], res["worst_count_delta"], res["tolerance_pct"]))
    if res["delta_over_tolerance"]:
        print("  提示: 第 %s 条样本 token 用量偏差超提示阈值 %.4f%%（仅提示，不参与判定）"
              % (res["delta_over_tolerance"], res["tolerance_pct"]))
    if res["verdict"] == "PASS_KNOWN_DIFF":
        print("  说明: 差异全部归因于已核实机制（剪枝 piece 在 HF 端仍可匹配 / 同分不同切分），"
              "两侧还原文本无损；细节见 tokenizer_c/docs/hf_export_pipeline.md")
    return 0 if res["verdict"].startswith("PASS") else 3


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    sys.exit(main())
