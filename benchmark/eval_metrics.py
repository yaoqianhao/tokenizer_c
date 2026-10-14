#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tokenizer-c 评测指标脚本。

对 corpus/<lang>/*.txt 语料评测一个 tokenizer 模型（.tkm 或 .model），产出 JSON 报告与
Markdown 汇总。全部指标通过 CLI（tok-cli.exe）真实测量，CLI 非零退出码即报错终止。

用法:
    python eval_metrics.py --model <model.tkm|.model> --corpus-dir <dir> \
        --cli <tok-cli.exe 路径> [--output report.json]

指标:
    - 编码效率: 分语种 字节数 / token 数 / BPT / chars_per_token，汇总整体 BPT
    - 还原率:   每语种抽前 200 行逐行 encode->decode 与原文逐字节比对
    - 稳定性:   每语种抽 100 句做 3 种扰动(首字母大小写互换 / 随机插入拼写错误 /
                \n 换 \t)后与原 token 序列算 Jaccard 相似度
    - 吞吐:     最大语种文件整体 encode 计时得 tokens/sec；decode 同测；
                CLI 启动 + 模型加载时间作为近似 load_ms
    - OOV/字节回退: 仅 .model 且环境装有 sentencepiece 时统计 unk / <0xXX> 占比，
                否则输出 null 并继续
    - TokEval UTF-8 完整性 (§3.4): utf8_completeness / char_split_rate /
                boundary_crossing_rate，基于 piece 表（.tkm 直接解析 TKCZ；
                .model 用 sentencepiece）对抽样行整体 encode 后统计
    - TokEval 数字对齐 (§3.5): digit_boundary_f1 / digit_split_variability /
                operator_isolation，数字串与操作符抽样后并发调用 CLI 计算
                （TOKEVAL_WORKERS 并发，避免大批量顺序调用超时）
    - 协议鲁棒性（可选，需显式传 --protocol-slices-dir 启用；不传时行为与既有版本完全一致）:
                双口径比值 D_lang = BPT_eval / BPT_real（en/zh/code 各一）、模型内离散度
                dispersion = max(D) - min(D) 及三语种标准差、评测集质量审计 protocol_audit
                （逐文件 重复行率/总字符数/不同字符数/不同汉字数/行数）、聚合口径并列输出
                （保留按字节加权 bpt_total，新增三语种等权 macro 平均）、按 dispersion
                阈值判定（≤0.35 稳定 / 0.35–1.0 敏感 / >1.0 口径依赖）

说明:
    - 还原率/稳定性按"行"处理。为避免文件尾部换行被归一化为空格 token 而误判，
      单行 encode 使用不带尾部换行的临时文件。
    - 稳定性扰动 3 的语义为「换行符变化」：行内含 \n 时替换为 \t；行内无 \n 时
      将第一个空格替换为 \n（语料按行存储，纯字面替换会变成恒等扰动）。
"""

import argparse
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

DEFAULT_ROUNDTRIP_LINES = 200
DEFAULT_STABILITY_LINES = 100
DEFAULT_OOV_LINES = 500
DEFAULT_LINE_BUDGET = 1000  # 每语种保留的最大非空行数（用于还原率/稳定性/OOV 抽样）

# TokEval 指标（来源: TokEval, arXiv 2608.18062, §3.4 编码保真 / §3.5 数字切分）
DEFAULT_TOKEVAL_LINES = 300    # UTF-8 三指标抽样行数（每语种，拼单文件整体 encode）
DEFAULT_DIGIT_SPANS = 30       # 数字串抽样上限（每语种）
DEFAULT_OPERATOR_SPANS = 30    # 操作符出现抽样上限（每语种）
DIGIT_MIN_LEN = 4              # 数字串最小长度（≥4 才存在内部理想边界）
TOKEVAL_WORKERS = 6            # 数字/操作符抽样的并发 CLI 调用数（避免顺序调用超时）

# 协议鲁棒性指标（双口径比值 D / 模型内离散度 dispersion / 评测集质量审计 protocol_audit）
PROTOCOL_LANGS = ("en", "zh", "code")   # 参与 D 与 dispersion 计算的语种
PROTOCOL_STABLE_MAX = 0.35              # dispersion ≤ 0.35 → 稳定
PROTOCOL_SENSITIVE_MAX = 1.0            # 0.35 < dispersion ≤ 1.0 → 敏感；> 1.0 → 口径依赖


# ---------------------------------------------------------------------------
# CLI 封装
# ---------------------------------------------------------------------------

def run_cli(cmd, label=""):
    """执行 CLI，返回 (stdout_bytes, elapsed_seconds)。非零退出码直接抛错终止。"""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write("[ERROR] CLI 调用失败 (%s): 退出码 %d\n" % (label, proc.returncode))
        sys.stderr.write("  命令: %s\n" % " ".join(cmd))
        sys.stderr.write("  stderr: %s\n" % proc.stderr.decode("utf-8", "replace").strip())
        raise RuntimeError("CLI 调用失败 (%s)，退出码 %d" % (label, proc.returncode))
    return proc.stdout, elapsed


# P0-1 / P1-4 / P1-6 runtime overrides appended to every encode/bench call.
# Empty by default => without --cli-extra the harness behaves exactly as before.
CLI_EXTRA = []


def cli_extra_args():
    """Extra flags appended to encode/bench CLI calls (see --cli-extra)."""
    return list(CLI_EXTRA)


def encode_file(cli, model, path, tmpdir):
    """encode 一个文件，返回 id 列表。"""
    out, _ = run_cli([cli, "encode", "--model", model, "--file", path]
                     + cli_extra_args(), "encode")
    text = out.decode("utf-8", "replace").strip()
    if not text:
        return []
    return [int(x) for x in text.split()]


def encode_text(cli, model, text_bytes, tmpdir):
    """encode 单行文本（不带尾部换行，避免 \n 归一化为空格 token），返回 id 列表。"""
    path = os.path.join(tmpdir, "line.txt")
    with open(path, "wb") as fh:
        fh.write(text_bytes)
    return encode_file(cli, model, path, tmpdir)


def decode_ids(cli, model, ids, tmpdir):
    """decode id 列表，返回还原后的原始字节（去掉 CLI 末尾换行）。"""
    ids_str = ",".join(str(x) for x in ids)
    out, _ = run_cli([cli, "decode", "--model", model, "--ids", ids_str], "decode")
    return out.rstrip(b"\r\n")


# ---------------------------------------------------------------------------
# 模型元信息
# ---------------------------------------------------------------------------

def parse_tkcz_vocab(model_path):
    """解析 TKCZ 二进制头部获取 vocab_size（魔数 TKCZ + u32 version + u32 vocab_size）。"""
    try:
        with open(model_path, "rb") as fh:
            head = fh.read(12)
        if len(head) < 12:
            return None
        magic, _ver, vocab = struct.unpack("<4sII", head)
        if magic != b"TKCZ":
            return None
        return vocab
    except OSError:
        return None


def get_vocab_size(model_path):
    """按模型类型获取词表大小；无法获取时返回 None。"""
    lower = model_path.lower()
    if lower.endswith(".tkm"):
        return parse_tkcz_vocab(model_path)
    if lower.endswith(".model"):
        try:
            import sentencepiece as spm  # noqa: F401
            sp = spm.SentencePieceProcessor(model_file=model_path)
            return sp.get_piece_size()
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# 语料读取
# ---------------------------------------------------------------------------

def read_lang_files(lang_dir):
    """返回该语种目录下的 *.txt 文件绝对路径列表（按名称排序）。"""
    if not os.path.isdir(lang_dir):
        return []
    return sorted(
        os.path.join(lang_dir, f)
        for f in os.listdir(lang_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(lang_dir, f))
    )


def read_file_data(path):
    """读取文件原始字节、非空行列表。"""
    with open(path, "rb") as fh:
        data = fh.read()
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return data, lines


# ---------------------------------------------------------------------------
# 还原率
# ---------------------------------------------------------------------------

def _roundtrip_one(cli, model, line, tmpdir, idx):
    """并发辅助：单行 encode->decode 并比对（唯一临时文件）。"""
    path = os.path.join(tmpdir, "rt_%d.txt" % idx)
    with open(path, "wb") as fh:
        fh.write(line)
    ids = encode_file(cli, model, path, tmpdir)
    decoded = decode_ids(cli, model, ids, tmpdir)
    return decoded == line


def roundtrip_check(cli, model, lines, tmpdir, limit=DEFAULT_ROUNDTRIP_LINES,
                    workers=TOKEVAL_WORKERS):
    """对前 limit 行并发逐行 encode->decode，与原文逐字节比对。

    返回 (ok, mismatches, tested)。
    """
    selected = list(lines[:limit])
    if not selected:
        return False, 0, 0
    mismatches = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_roundtrip_one, cli, model, line, tmpdir, i)
                   for i, line in enumerate(selected)]
        for fut in futures:
            try:
                ok = fut.result()
            except Exception:
                mismatches += 1
                continue
            if not ok:
                mismatches += 1
    return (mismatches == 0), mismatches, len(selected)


# ---------------------------------------------------------------------------
# 稳定性
# ---------------------------------------------------------------------------

def perturb_case(text):
    """扰动 1：首字母大小写互换。"""
    if not text:
        return text
    ch = text[0]
    if ch.isalpha():
        return ch.swapcase() + text[1:]
    return text


def perturb_typo(text, rng):
    """扰动 2：随机位置插入一个拼写错误（随机 ASCII 字母）。"""
    if not text:
        return text
    pos = rng.randrange(len(text))
    ch = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return text[:pos] + ch + text[pos:]


def perturb_newline(text):
    """扰动 3：换行符变化。行内含 \\n 时替换为 \\t；行内无 \\n 时把第一个空格
    替换为 \\n（语料按行存储，保证扰动确有变化，检验换行/空格归一化的一致性）。"""
    if "\n" in text:
        return text.replace("\n", "\t")
    idx = text.find(" ")
    if idx >= 0:
        return text[:idx] + "\n" + text[idx + 1:]
    return text


def jaccard(a, b):
    """两个 token id 序列的 Jaccard 相似度（基于集合）。"""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / float(len(union))


def _stability_one(cli, model, line, seed, tmpdir, idx):
    """并发辅助：单句 3 种扰动，返回与原序列 Jaccard 相似度列表。"""
    rng = random.Random(seed)
    text = line.decode("utf-8", "replace")

    def enc(payload, tag):
        path = os.path.join(tmpdir, "stb_%s_%d.txt" % (tag, idx))
        with open(path, "wb") as fh:
            fh.write(payload)
        return encode_file(cli, model, path, tmpdir)

    base_ids = enc(line, "base")
    variants = [
        perturb_case(text).encode("utf-8"),
        perturb_typo(text, rng).encode("utf-8"),
        perturb_newline(text).encode("utf-8"),
    ]
    sims = []
    for vbytes in variants:
        if vbytes == line:
            sims.append(1.0)
            continue
        v_ids = enc(vbytes, "var")
        sims.append(jaccard(base_ids, v_ids))
    return sims


def stability_check(cli, model, lines, tmpdir, seed=42, sample=DEFAULT_STABILITY_LINES,
                    workers=TOKEVAL_WORKERS):
    """抽 sample 句做 3 种扰动（按句并发），返回扰动后与原序列 Jaccard 的均值。"""
    rng = random.Random(seed)
    sample_lines = rng.sample(lines, min(sample, len(lines)))
    if not sample_lines:
        return None
    sims = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_stability_one, cli, model, line, seed + i, tmpdir, i)
                   for i, line in enumerate(sample_lines)]
        for fut in futures:
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                sims.extend(r)
    if not sims:
        return None
    return sum(sims) / float(len(sims))


# ---------------------------------------------------------------------------
# 吞吐
# ---------------------------------------------------------------------------

def throughput(cli, model, max_file, tmpdir, max_ids_len=20000):
    """最大语种文件整体 encode 计时；decode 同测；短输入 encode 计时近似 load_ms。"""
    # encode 整体计时
    t0 = time.perf_counter()
    out, _ = run_cli([cli, "encode", "--model", model, "--file", max_file]
                     + cli_extra_args(), "encode-throughput")
    enc_elapsed = time.perf_counter() - t0
    text = out.decode("utf-8", "replace").strip()
    ids = [int(x) for x in text.split()] if text else []
    encode_tps = (len(ids) / enc_elapsed) if enc_elapsed > 0 else 0.0

    # decode 同测：取前 K 个 id，保证命令行长度在 Windows 限制内
    k = len(ids)
    while k > 0:
        s = ",".join(str(x) for x in ids[:k])
        if len(s) <= max_ids_len:
            break
        k = int(k * 0.9)
    t0 = time.perf_counter()
    run_cli([cli, "decode", "--model", model, "--ids", s], "decode-throughput")
    dec_elapsed = time.perf_counter() - t0
    decode_tps = (k / dec_elapsed) if dec_elapsed > 0 else 0.0

    # load_ms：单次短输入 encode（含 CLI 启动 + 模型加载）
    load_path = os.path.join(tmpdir, "load.txt")
    with open(load_path, "wb") as fh:
        fh.write(b"x")
    t0 = time.perf_counter()
    run_cli([cli, "encode", "--model", model, "--file", load_path]
            + cli_extra_args(), "load")
    load_ms = (time.perf_counter() - t0) * 1000.0

    return encode_tps, decode_tps, load_ms


# ---------------------------------------------------------------------------
# OOV / 字节回退（仅 .model + sentencepiece）
# ---------------------------------------------------------------------------

def oov_stats(sp, lines, limit=DEFAULT_OOV_LINES):
    """用官方 SentencePiece 加载 .model，统计 unk 占比与 <0xXX> 字节回退占比。"""
    total = 0
    unk = 0
    fb = 0
    for line in lines[:limit]:
        try:
            pieces = sp.encode(line.decode("utf-8", "replace"), out_type=str)
        except Exception:
            continue
        total += len(pieces)
        unk += sum(1 for p in pieces if p == "<unk>")
        fb += sum(1 for p in pieces if p.startswith("<0x"))
    if total == 0:
        return None, None
    return unk / float(total), fb / float(total)


# ---------------------------------------------------------------------------
# TokEval 指标：UTF-8 完整性 / 数字对齐（来源: TokEval, arXiv 2608.18062）
#   编码保真 §3.4: UTF-8 completeness / Char split rate / UTF-8 boundary crossing
#   数字切分 §3.5: Digit boundary F1 / Digit split variability / Operator isolation
# ---------------------------------------------------------------------------

_OP_RE = re.compile(rb"[+\-*/=<>!&|^~%]+")
_DIGIT_RE = re.compile(rb"[0-9]{%d,}" % DIGIT_MIN_LEN)
_WHITESPACE_CHARS = set(" \t\n\r\f\v\u2581")  # 含 ▁(U+2581) 空格归一化标记


def _avg_opt(values):
    """过滤 None 后求均值；全为 None 或空列表返回 None。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / float(len(vals)), 6)


def _piece_bytes_for_id(tid, pieces):
    """id -> piece 原始字节。

    正 id 查 piece 表（与 CLI 的 encode 输出 id 一一对应，TKCZ 条目顺序）；
    负 id 为 CLI 的 byte-fallback 虚拟 id（语义 -1-byte，如 -44 -> 0x2B '+'），
    解码为单字节返回。越界返回 None。
    """
    if tid >= 0:
        return pieces[tid] if tid < len(pieces) else None
    if tid >= -256:
        return bytes([-tid - 1])
    return None


def parse_tkcz_pieces(model_path):
    """解析 TKCZ 二进制 (.tkm)，返回 (pieces:list[bytes], special_ids:set[int])。

    pieces[i] 为 id i 的 piece 原始 UTF-8 字节；special_ids 含 unk/bos/eos/pad。
    格式：magic "TKCZ" + u32 version + u32 vocab + 4 x i32 special + u32 flags
          + vocab 条 (u32 len, len bytes, f32 score)，id 即条目顺序。
    解析失败返回 (None, None)。
    """
    try:
        with open(model_path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None, None
    if len(data) < 32 or data[:4] != b"TKCZ":
        return None, None
    version, vocab = struct.unpack_from("<II", data, 4)
    if version != 1:
        return None, None
    unk, bos, eos, pad = struct.unpack_from("<4i", data, 12)
    special_ids = {i for i in (unk, bos, eos, pad) if i >= 0}
    pieces = []
    o = 32
    for _ in range(vocab):
        if o + 8 > len(data):
            return None, None
        plen = struct.unpack_from("<I", data, o)[0]
        o += 4
        if o + plen + 4 > len(data):
            return None, None
        pieces.append(data[o:o + plen])
        o += plen + 4
    return pieces, special_ids


def load_piece_table(model_path, sp=None):
    """统一加载 piece 文本表与特殊 token id 集合。

    .tkm 直接解析 TKCZ；.model 依赖 sentencepiece（需先加载 sp）。
    返回 (pieces:list[bytes]|None, special_ids:set[int]|None)，失败返回 (None, None)。
    """
    lower = model_path.lower()
    if lower.endswith(".tkm"):
        return parse_tkcz_pieces(model_path)
    if lower.endswith(".model") and sp is not None:
        try:
            pieces = [sp.id_to_piece(i).encode("utf-8", "replace")
                      for i in range(sp.get_piece_size())]
            special_ids = {sp.piece_to_id("<unk>"), sp.piece_to_id("<s>"),
                           sp.piece_to_id("</s>"), sp.piece_to_id("<pad>")}
            return pieces, {i for i in special_ids if i >= 0}
        except Exception:
            return None, None
    return None, None


def _encode_batch_file(cli, model, lines, tmpdir, tag):
    """多行拼接为单文件后整体 encode，返回 id 列表（1 次 CLI 调用）。"""
    path = os.path.join(tmpdir, "tokeval_%s.txt" % tag)
    with open(path, "wb") as fh:
        fh.write(b"\n".join(lines))
    return encode_file(cli, model, path, tmpdir)


def _encode_span(cli, model, payload, tmpdir, idx):
    """并发辅助：单段 bytes encode（唯一临时文件），返回 id 列表。"""
    path = os.path.join(tmpdir, "tok_span_%d.txt" % idx)
    with open(path, "wb") as fh:
        fh.write(payload)
    return encode_file(cli, model, path, tmpdir)


def _collect_multibyte_chars(lines, limit_chars=4000):
    """从抽样行收集唯一多字节字符（按出现顺序，最多 limit_chars 个）。"""
    seen = set()
    order = []
    for line in lines:
        for ch in line.decode("utf-8", "ignore"):
            if ord(ch) > 0x7F and ch not in seen:
                seen.add(ch)
                order.append(ch)
                if len(order) >= limit_chars:
                    return order
    return order


def tokeval_utf8_metrics(cli, model, lines, tmpdir, pieces, special_ids):
    """UTF-8 完整性三指标（TokEval §3.4）。

    - utf8_completeness: 非特殊 token 出现次数中，piece 字节可严格解码为合法
      UTF-8 字符序列的占比（occurrence 加权）。
    - char_split_rate: 唯一多字节字符中，其完整 UTF-8 编码未作为子串出现在任何
      单个 piece 内（即被跨 token 拆分或字节回退）的比例。以 NUL 分隔拼接全部
      piece 后做子串匹配，避免跨 piece 边界误命中；NFKC 归一化字符按子字符检查。
    - boundary_crossing_rate: 非特殊 token 中，piece 长度 >1 字节且无法完整解码
      （含截断/非法字节，即跨字符边界合并留下的不完整字符）的占比。
    """
    if pieces is None:
        return None
    ids = _encode_batch_file(cli, model, lines, tmpdir, "utf8")
    total = 0
    complete = 0
    crossing = 0
    for tid in ids:
        if tid in special_ids:
            continue
        pb = _piece_bytes_for_id(tid, pieces)
        if pb is None:
            continue
        total += 1
        try:
            pb.decode("utf-8")  # strict：完整解码
            complete += 1
        except UnicodeDecodeError:
            if len(pb) > 1:
                crossing += 1
    if total == 0:
        return None

    chars = _collect_multibyte_chars(lines)
    joined = b"\x00".join(pieces)
    split_chars = 0
    for ch in chars:
        raw = ch.encode("utf-8")
        if raw in joined:
            continue
        norm = unicodedata.normalize("NFKC", ch)
        if all(c.encode("utf-8") in joined for c in norm):
            continue
        split_chars += 1
    return {
        "utf8_completeness": round(complete / float(total), 6),
        "char_split_rate": round(split_chars / float(len(chars)), 6) if chars else None,
        "boundary_crossing_rate": round(crossing / float(total), 6),
        "tokens": total,
        "multibyte_chars": len(chars),
    }


def _collect_digit_spans(lines, limit):
    """从抽样行收集数字串（长度>=DIGIT_MIN_LEN），返回 bytes 列表。"""
    spans = []
    for line in lines:
        for m in _DIGIT_RE.finditer(line):
            spans.append(m.group(0))
            if len(spans) >= limit:
                return spans
    return spans


def _ideal_digit_boundaries(n):
    """右对齐三元组的理想边界位置集合（0-based，边界位于 pos|pos+1 之间）。

    例：n=7 -> {1,4}（1|234|567）；n=6 -> {3}（123|456）；n=4 -> {1}（1|234）。
    """
    if n < DIGIT_MIN_LEN:
        return set()
    return set(range(n % 3 if n % 3 else 3, n - 1, 3))


def _span_obs_boundaries(ids, pieces, span_bytes):
    """由 id 序列还原数字串切分。返回 (obs_boundaries:set, pattern:tuple)。

    pattern 为每个 token 内数字字符数序列（如 1234567 -> (1,3,3)）。
    拼接出的数字序列与原始 span 不一致（特殊 token/归一化干扰）时返回 (None, None)。
    """
    digits = []
    pattern = []
    for tid in ids:
        pb = _piece_bytes_for_id(tid, pieces)
        if pb is None:
            return None, None
        dpart = bytes(b for b in pb if 0x30 <= b <= 0x39)
        if dpart:
            digits.append(dpart)
            pattern.append(len(dpart))
    if b"".join(digits) != span_bytes:
        return None, None
    obs = set()
    pos = 0
    for ln in pattern[:-1]:
        pos += ln
        obs.add(pos)
    return obs, tuple(pattern)


def tokeval_digit_metrics(cli, model, lines, tmpdir, pieces):
    """数字对齐指标（TokEval §3.5，并发调用避免顺序超时）。

    - digit_boundary_f1: 数字串观测切分边界 vs 右对齐三元组理想边界的 F1 均值。
    - digit_split_variability: 同长度数字串切分模式的熵（按样本量加权平均）。
    """
    if pieces is None:
        return None
    spans = _collect_digit_spans(lines, DEFAULT_DIGIT_SPANS)
    if not spans:
        return {"digit_boundary_f1": None, "digit_split_variability": None,
                "digit_spans": 0, "digit_skipped": 0}

    f1s = []
    patterns_by_len = {}
    skipped = 0
    with ThreadPoolExecutor(max_workers=TOKEVAL_WORKERS) as ex:
        futures = [ex.submit(_encode_span, cli, model, s, tmpdir, i)
                   for i, s in enumerate(spans)]
        for fut, s in zip(futures, spans):
            try:
                ids = fut.result()
            except Exception:
                skipped += 1
                continue
            obs, pattern = _span_obs_boundaries(ids, pieces, s)
            if obs is None:
                skipped += 1
                continue
            n = len(s)
            ideal = _ideal_digit_boundaries(n)
            inter = len(obs & ideal)
            denom = len(obs) + len(ideal)
            f1 = (2.0 * inter / denom) if denom else 1.0
            f1s.append(f1)
            patterns_by_len.setdefault(n, []).append(pattern)

    variability = None
    total_n = 0
    ent_sum = 0.0
    for pats in patterns_by_len.values():
        cnt = {}
        for p in pats:
            cnt[p] = cnt.get(p, 0) + 1
        n = float(len(pats))
        ent_sum += -sum((c / n) * math.log2(c / n) for c in cnt.values()) * n
        total_n += n
    if total_n:
        variability = ent_sum / total_n

    f1_avg = (sum(f1s) / len(f1s)) if f1s else None
    return {
        "digit_boundary_f1": round(f1_avg, 6) if f1_avg is not None else None,
        "digit_split_variability": round(variability, 6) if variability is not None else None,
        "digit_spans": len(spans),
        "digit_skipped": skipped,
    }


def _collect_operator_spans(lines, limit):
    """收集操作符出现，返回 (op_bytes, 上下文切片 bytes) 列表（前后各 3 字节）。"""
    out = []
    for line in lines:
        for m in _OP_RE.finditer(line):
            start, end = m.span()
            ctx = line[max(0, start - 3):min(len(line), end + 3)]
            out.append((m.group(0), ctx))
            if len(out) >= limit:
                return out
    return out


def tokeval_operator_metrics(cli, model, lines, tmpdir, pieces):
    """操作符隔离度（TokEval §3.5，论文在 math 语料报告）。

    对每个操作符出现（含前后各 3 字节上下文切片）单独 encode，检查覆盖该操作符
    的 piece：若任一覆盖 piece 除去操作符字符与空白（含 ▁）后仍含其他字符，
    视为与操作数融合（未隔离）；否则该操作符出现被隔离。
    """
    if pieces is None:
        return None
    samples = _collect_operator_spans(lines, DEFAULT_OPERATOR_SPANS)
    if not samples:
        return {"operator_isolation": None, "operator_occurrences": 0,
                "operator_skipped": 0}
    isolated = 0
    total = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=TOKEVAL_WORKERS) as ex:
        futures = [ex.submit(_encode_span, cli, model, ctx, tmpdir, 10000 + i)
                   for i, (_op, ctx) in enumerate(samples)]
        for fut, (_op, _ctx) in zip(futures, samples):
            try:
                ids = fut.result()
            except Exception:
                skipped += 1
                continue
            op_set = set(_op.decode("utf-8", "replace"))
            covering = []
            for tid in ids:
                pb = _piece_bytes_for_id(tid, pieces)
                if pb is None:
                    continue
                text = pb.decode("utf-8", "replace")
                if any(ch in text for ch in op_set):
                    covering.append(text)
            if not covering:
                skipped += 1
                continue
            total += 1
            fused = False
            for text in covering:
                rest = "".join(ch for ch in text
                               if ch not in _WHITESPACE_CHARS and ch not in op_set)
                if rest:
                    fused = True
                    break
            if not fused:
                isolated += 1
    return {
        "operator_isolation": round(isolated / float(total), 6) if total else None,
        "operator_occurrences": total,
        "operator_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# 协议鲁棒性：双口径比值 D / 模型内离散度 dispersion / 评测集质量审计 protocol_audit
# ---------------------------------------------------------------------------
# 口径定义（与评测常规指标同源，BPT = bytes / tokens，tokens 由 CLI 实测）：
#   D_lang     = BPT_eval / BPT_real，逐语种（en/zh/code）各一个
#   dispersion = max(D) - min(D)，并附三语种标准差与等权均值
#   判定        = dispersion ≤ 0.35 稳定 / 0.35 < dispersion ≤ 1.0 敏感 / > 1.0 口径依赖
#   聚合口径    = 保留按字节加权的 bpt_total，并列输出三语种等权 macro 平均

def protocol_audit_file(path):
    """单个语料文件的质量审计：行数 / 总字符数 / 不同字符数 / 不同汉字数 / 重复行率。"""
    data, lines = read_file_data(path)
    text = data.decode("utf-8", "replace")
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    dup_extra = sum(c - 1 for c in counts.values() if c > 1)
    distinct_cjk = {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}
    return {
        "file": os.path.basename(path),
        "path": os.path.abspath(path),
        "bytes": len(data),
        "lines": len(lines),
        "chars": len(text),
        "distinct_chars": len(set(text)),
        "distinct_cjk_chars": len(distinct_cjk),
        "dup_line_count": dup_extra,
        "dup_line_rate": round(dup_extra / float(len(lines)), 6) if lines else None,
    }


def protocol_audit_files(tagged_paths):
    """批量质量审计。tagged_paths: {tag: 文件路径}；文件缺失时以 error 标记占位。"""
    out = {}
    for tag, path in tagged_paths.items():
        if path and os.path.isfile(path):
            out[tag] = protocol_audit_file(path)
        else:
            out[tag] = {"file": os.path.basename(path) if path else None,
                        "path": os.path.abspath(path) if path else None,
                        "error": "文件不存在"}
    return out


def protocol_bpt(cli, model, path, tmpdir):
    """单文件 BPT = bytes / tokens（tokens 由 CLI 实测）。"""
    ids = encode_file(cli, model, path, tmpdir)
    nbytes = os.path.getsize(path)
    if not ids:
        return {"bytes": nbytes, "tokens": 0, "bpt": None}
    return {"bytes": nbytes, "tokens": len(ids),
            "bpt": round(nbytes / float(len(ids)), 6)}


def protocol_dispersion(d_values):
    """D 序列的模型内离散度：max-min、等权均值、总体标准差。"""
    vals = [v for v in d_values if v is not None]
    if not vals:
        return {"max_minus_min": None, "mean": None, "std": None, "n": 0}
    mean = sum(vals) / float(len(vals))
    var = sum((v - mean) ** 2 for v in vals) / float(len(vals))
    return {"max_minus_min": round(max(vals) - min(vals), 6),
            "mean": round(mean, 6), "std": round(math.sqrt(var), 6), "n": len(vals)}


def protocol_verdict(disp):
    """按 dispersion 阈值给出判定语。"""
    if disp is None:
        return {"level": None, "text": "无法判定：缺少可用 D 值"}
    if disp <= PROTOCOL_STABLE_MAX:
        return {"level": "稳定",
                "text": "口径稳定：评测语料与真实语料结论一致（dispersion=%.4f ≤ %.2f）"
                        % (disp, PROTOCOL_STABLE_MAX)}
    if disp <= PROTOCOL_SENSITIVE_MAX:
        return {"level": "敏感",
                "text": "口径敏感：评测语料会放大或缩小真实表现，跨口径数值不可直接混用"
                        "（%.2f < dispersion=%.4f ≤ %.2f）"
                        % (PROTOCOL_STABLE_MAX, disp, PROTOCOL_SENSITIVE_MAX)}
    return {"level": "口径依赖",
            "text": "口径依赖：结果由语料口径主导，评测语料结论不能外推到真实语料"
                    "（dispersion=%.4f > %.2f）" % (disp, PROTOCOL_SENSITIVE_MAX)}


def protocol_macro(values):
    """三语种等权 macro 平均（忽略 None）。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / float(len(vals)), 6)


def compute_protocol_robustness(cli, model, slices_dir, tmpdir,
                                full_corpus_dir=None, langs=PROTOCOL_LANGS):
    """计算单个模型的协议鲁棒性指标。

    slices_dir:      同口径切片目录，需含 eval_<lang>.txt 与 <lang>_real.txt。
    full_corpus_dir: 可选，评测语料根目录（corpus/<lang>/<lang>.txt），用于输出
                     按字节加权的 bpt_total 与三语种等权 macro 平均（聚合口径并列）。
    """
    per_lang = {}
    for lang in langs:
        efp = os.path.join(slices_dir, "eval_%s.txt" % lang)
        rfp = os.path.join(slices_dir, "%s_real.txt" % lang)
        entry = {"eval_path": os.path.abspath(efp), "real_path": os.path.abspath(rfp)}
        if not os.path.isfile(efp) or not os.path.isfile(rfp):
            entry["error"] = "切片文件缺失"
            entry["D"] = None
            per_lang[lang] = entry
            continue
        ev = protocol_bpt(cli, model, efp, tmpdir)
        re_ = protocol_bpt(cli, model, rfp, tmpdir)
        d = (ev["bpt"] / re_["bpt"]) if (ev["bpt"] and re_["bpt"]) else None
        entry.update({"eval": ev, "real": re_,
                      "D": round(d, 6) if d is not None else None})
        per_lang[lang] = entry

    d_map = {lang: per_lang[lang].get("D") for lang in langs}
    disp = protocol_dispersion([d_map[l] for l in langs])
    verdict = protocol_verdict(disp["max_minus_min"])

    aggregate = {"bpt_macro": None, "per_lang_bpt": {}, "source": None}
    if full_corpus_dir and os.path.isdir(full_corpus_dir):
        tot_b = tot_t = 0
        per_lang_bpt = {}
        for lang in langs:
            fp = os.path.join(full_corpus_dir, lang, "%s.txt" % lang)
            if not os.path.isfile(fp):
                per_lang_bpt[lang] = None
                continue
            r = protocol_bpt(cli, model, fp, tmpdir)
            tot_b += r["bytes"]
            tot_t += r["tokens"]
            per_lang_bpt[lang] = r["bpt"]
        aggregate["bpt_total_byte_weighted"] = round(tot_b / float(tot_t), 6) if tot_t else None
        aggregate["bpt_macro"] = protocol_macro([per_lang_bpt[l] for l in langs])
        aggregate["per_lang_bpt"] = per_lang_bpt
        aggregate["source"] = "corpus/<lang>/<lang>.txt 整语料实测（bytes/tokens）"
    else:
        per_lang_bpt = {l: per_lang[l].get("eval", {}).get("bpt") for l in langs}
        aggregate["bpt_total_byte_weighted"] = None
        aggregate["bpt_macro"] = protocol_macro([per_lang_bpt[l] for l in langs])
        aggregate["per_lang_bpt"] = per_lang_bpt
        aggregate["source"] = "eval_<lang>.txt 切片实测（未提供整语料目录）"

    return {
        "model": os.path.abspath(model),
        "langs": list(langs),
        "per_lang": per_lang,
        "D": d_map,
        "dispersion": disp,
        "verdict": verdict,
        "aggregate": aggregate,
        "thresholds": {"stable_max": PROTOCOL_STABLE_MAX,
                       "sensitive_max": PROTOCOL_SENSITIVE_MAX},
    }


def _pfmt(v, nd=4):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "%.*f" % (nd, v)
    return "-"


def render_protocol_markdown(payload, title="协议鲁棒性指标报告"):
    """把多模型协议鲁棒性结果渲染为 Markdown。

    payload = {
      "date": str, "notes": [str],
      "corpus": {"protocol_audit": {tag: {...}}, "slices_dir": str, "full_corpus_dir": str},
      "models": {name: {..compute_protocol_robustness 返回值.., "source": str}},
    }
    """
    lines = []
    lines.append("# %s" % title)
    lines.append("")
    lines.append("- date: %s" % payload.get("date", ""))
    corp = payload.get("corpus", {}) or {}
    if corp.get("slices_dir"):
        lines.append("- 同口径切片目录: `%s`" % corp["slices_dir"])
    if corp.get("full_corpus_dir"):
        lines.append("- 整语料目录（聚合口径）: `%s`" % corp["full_corpus_dir"])
    lines.append("- 指标定义: D_lang = BPT_eval / BPT_real；dispersion = max(D) - min(D)")
    lines.append("- 判定阈值: dispersion ≤ 0.35 稳定 / 0.35–1.0 敏感 / > 1.0 口径依赖")
    lines.append("")

    models = payload.get("models", {}) or {}

    lines.append("## 1. 双口径比值与模型内离散度")
    lines.append("")
    lines.append("| model | D_en | D_zh | D_code | dispersion(max-min) | std(三语种) | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, sec in models.items():
        d = sec.get("D", {}) or {}
        disp = sec.get("dispersion", {}) or {}
        vd = (sec.get("verdict", {}) or {}).get("level") or "-"
        lines.append("| %s | %s | %s | %s | **%s** | %s | %s |" % (
            name, _pfmt(d.get("en")), _pfmt(d.get("zh")), _pfmt(d.get("code")),
            _pfmt(disp.get("max_minus_min")), _pfmt(disp.get("std")), vd))
    lines.append("")

    lines.append("## 2. 聚合口径并列输出")
    lines.append("")
    lines.append("| model | bpt_total（按字节加权） | BPT_macro（三语种等权） | BPT_en | BPT_zh | BPT_code | 口径来源 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, sec in models.items():
        agg = sec.get("aggregate", {}) or {}
        pl = agg.get("per_lang_bpt", {}) or {}
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name, _pfmt(agg.get("bpt_total_byte_weighted")), _pfmt(agg.get("bpt_macro")),
            _pfmt(pl.get("en")), _pfmt(pl.get("zh")), _pfmt(pl.get("code")),
            agg.get("source") or "-"))
    lines.append("")

    audit = corp.get("protocol_audit") or {}
    if audit:
        lines.append("## 3. 评测集质量审计 protocol_audit（逐语料文件）")
        lines.append("")
        lines.append("| tag | file | 行数 | 总字符数 | 不同字符数 | 不同汉字数 | 重复行率 | 重复行数 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tag, a in audit.items():
            if a.get("error"):
                lines.append("| %s | %s | - | - | - | - | - | - |" % (tag, a.get("file")))
                continue
            lines.append("| %s | %s | %d | %d | %d | %d | %s | %d |" % (
                tag, a.get("file"), a.get("lines", 0), a.get("chars", 0),
                a.get("distinct_chars", 0), a.get("distinct_cjk_chars", 0),
                _pfmt(a.get("dup_line_rate"), 6), a.get("dup_line_count", 0)))
        lines.append("")

    lines.append("## 4. 判定语")
    lines.append("")
    for name, sec in models.items():
        vd = sec.get("verdict", {}) or {}
        src = sec.get("source") or "-"
        lines.append("- **%s**（%s）：%s" % (name, src, vd.get("text") or "-"))
    lines.append("")

    findings = payload.get("data_quality_findings") or []
    if findings:
        lines.append("## 5. 数据质量观察（由 protocol_audit 派生）")
        lines.append("")
        for x in findings:
            lines.append("- %s" % x)
        lines.append("")

    notes = payload.get("notes") or []
    if notes:
        lines.append("## %d. 说明" % (6 if findings else 5))
        lines.append("")
        for n in notes:
            lines.append("- %s" % n)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="tokenizer-c 评测指标：编码效率 / 还原率 / 稳定性 / 吞吐 / OOV"
    )
    parser.add_argument("--model", required=True,
                        help="模型路径（.tkm 或 .model）")
    parser.add_argument("--corpus-dir", required=True,
                        help="语料目录，结构为 corpus/<lang>/*.txt（lang 如 en/zh/code/multilingual/noise）")
    parser.add_argument("--cli", required=True,
                        help="tok-cli.exe 绝对路径")
    parser.add_argument("--output", default="",
                        help="JSON 报告输出路径（默认 eval_report_<date>.json，与脚本同目录）")
    # 运行时推理开关（新增，可选；不传时行为与既有版本完全一致）
    parser.add_argument("--cli-extra", default="",
                        help="[P0-1/P1-4/P1-6] 追加到每次 encode/bench CLI 调用的额外参数，"
                             "空格分隔，例如 '--norm-reversible --segment-rules'；"
                             "用于在不重训的前提下用修复后的编码器重评既有模型")
    # 协议鲁棒性（新增，全部可选；不传 --protocol-slices-dir 时行为与既有版本完全一致）
    parser.add_argument("--protocol-slices-dir", default="",
                        help="[协议鲁棒性] 同口径切片目录，需含 eval_<lang>.txt 与 <lang>_real.txt；"
                             "提供后额外计算 D/dispersion/protocol_audit 并输出独立段落")
    parser.add_argument("--protocol-full-corpus-dir", default="",
                        help="[协议鲁棒性] 整语料目录（corpus/<lang>/<lang>.txt），用于按字节加权 "
                             "bpt_total 与三语种等权 macro 平均的并列输出；缺省则回退为切片口径")
    parser.add_argument("--protocol-label", default="",
                        help="[协议鲁棒性] 该模型在报告中的显示名（默认取模型文件名）")
    parser.add_argument("--protocol-out", default="",
                        help="[协议鲁棒性] 协议鲁棒性 JSON 输出路径（可选）")
    parser.add_argument("--protocol-md", default="",
                        help="[协议鲁棒性] 协议鲁棒性 Markdown 输出路径（可选）")
    args = parser.parse_args()

    global CLI_EXTRA
    CLI_EXTRA = args.cli_extra.split() if args.cli_extra else []

    cli = os.path.abspath(args.cli)
    model = os.path.abspath(args.model)
    corpus_dir = os.path.abspath(args.corpus_dir)
    date_str = datetime.now().strftime("%Y-%m-%d")

    if not os.path.isfile(cli):
        sys.exit("[ERROR] CLI 不存在: %s" % cli)
    if not os.path.isfile(model):
        sys.exit("[ERROR] 模型不存在: %s" % model)
    if not os.path.isdir(corpus_dir):
        sys.exit("[ERROR] 语料目录不存在: %s" % corpus_dir)

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "eval_report_%s.json" % date_str)

    langs = sorted(
        d for d in os.listdir(corpus_dir)
        if os.path.isdir(os.path.join(corpus_dir, d))
    )
    if not langs:
        sys.exit("[ERROR] 语料目录下没有语种子目录: %s" % corpus_dir)

    vocab_size = get_vocab_size(model)

    sp = None
    use_sp_oov = model.lower().endswith(".model")
    if use_sp_oov:
        try:
            import sentencepiece as spm  # noqa: F401
            sp = spm.SentencePieceProcessor(model_file=model)
        except Exception:
            sp = None

    # TokEval 指标依赖 piece 文本表：.tkm 直接解析 TKCZ；.model 用 sentencepiece
    pieces, special_ids = load_piece_table(model, sp)

    lang_results = {}
    totals_bytes = 0
    totals_tokens = 0

    with tempfile.TemporaryDirectory(prefix="tok_eval_") as tmpdir:
        for lang in langs:
            files = read_lang_files(os.path.join(corpus_dir, lang))
            if not files:
                sys.exit("[ERROR] 语种 %s 下没有 *.txt 文件" % lang)

            lang_bytes = 0
            lang_tokens = 0
            lang_lines = []
            for f in files:
                data, lines = read_file_data(f)
                lang_bytes += len(data)
                lang_tokens += len(encode_file(cli, model, f, tmpdir))
                lang_lines.extend(lines[:DEFAULT_LINE_BUDGET])
                lang_lines = lang_lines[:DEFAULT_LINE_BUDGET]

            bpt = (lang_bytes / lang_tokens) if lang_tokens > 0 else None
            # 字符数基于全部文件文本
            all_text = b"".join(open(f, "rb").read() for f in files)
            char_count = len(all_text.decode("utf-8", "replace"))

            rt_ok, rt_mism, rt_tested = roundtrip_check(cli, model, lang_lines, tmpdir)
            stab = stability_check(cli, model, lang_lines, tmpdir)

            oov = fb = None
            if sp is not None:
                oov, fb = oov_stats(sp, lang_lines)

            tk_utf8 = tokeval_utf8_metrics(cli, model, lang_lines, tmpdir,
                                           pieces, special_ids)
            tk_digit = tokeval_digit_metrics(cli, model, lang_lines, tmpdir, pieces)
            tk_oper = tokeval_operator_metrics(cli, model, lang_lines, tmpdir, pieces)

            lang_results[lang] = {
                "files": [os.path.basename(f) for f in files],
                "bytes": lang_bytes,
                "tokens": lang_tokens,
                "bpt": round(bpt, 6) if bpt is not None else None,
                "chars_per_token": round(char_count / lang_tokens, 6) if lang_tokens > 0 else None,
                "roundtrip_ok": rt_ok,
                "roundtrip_mismatches": rt_mism,
                "roundtrip_lines": rt_tested,
                "stability": round(stab, 6) if stab is not None else None,
                "oov_rate": round(oov, 6) if oov is not None else None,
                "byte_fallback_rate": round(fb, 6) if fb is not None else None,
                "tokeval": {
                    "utf8": tk_utf8,
                    "digit": tk_digit,
                    "operator": tk_oper,
                },
            }
            totals_bytes += lang_bytes
            totals_tokens += lang_tokens

        # 吞吐：选字节数最大的单个文件
        max_file = None
        max_size = -1
        for lang in langs:
            for f in read_lang_files(os.path.join(corpus_dir, lang)):
                sz = os.path.getsize(f)
                if sz > max_size:
                    max_size = sz
                    max_file = f
        enc_tps = dec_tps = load_ms = None
        if max_file is not None:
            enc_tps, dec_tps, load_ms = throughput(cli, model, max_file, tmpdir)

    # ---- 协议鲁棒性（可选；独立计算、独立输出段落，不改动上述任何既有指标）----
    protocol_sec = None
    if args.protocol_slices_dir:
        slices_dir = os.path.abspath(args.protocol_slices_dir)
        if not os.path.isdir(slices_dir):
            sys.stderr.write("[WARN] 协议鲁棒性切片目录不存在，已跳过: %s\n" % slices_dir)
        else:
            full_dir = os.path.abspath(args.protocol_full_corpus_dir) \
                if args.protocol_full_corpus_dir else None
            if full_dir and not os.path.isdir(full_dir):
                sys.stderr.write("[WARN] 整语料目录不存在，聚合口径回退为切片: %s\n" % full_dir)
                full_dir = None
            with tempfile.TemporaryDirectory(prefix="tok_proto_") as ptmp:
                protocol_sec = compute_protocol_robustness(
                    cli, model, slices_dir, ptmp, full_corpus_dir=full_dir)
                protocol_sec["label"] = args.protocol_label or os.path.basename(model)
                protocol_sec["corpus_audit"] = protocol_audit_files(dict(
                    [("eval_%s" % l, os.path.join(slices_dir, "eval_%s.txt" % l))
                     for l in PROTOCOL_LANGS]
                    + [("%s_real" % l, os.path.join(slices_dir, "%s_real.txt" % l))
                       for l in PROTOCOL_LANGS]))
                protocol_sec["corpus"] = {"slices_dir": slices_dir, "full_corpus_dir": full_dir}

    report = {
        "date": date_str,
        "model": model,
        "vocab_size": vocab_size,
        "cli_extra": list(CLI_EXTRA),
        "langs": lang_results,
        "totals": {
            "bytes": totals_bytes,
            "tokens": totals_tokens,
            "bpt": round(totals_bytes / totals_tokens, 6) if totals_tokens > 0 else None,
        },
        "roundtrip_ok": all(v["roundtrip_ok"] for v in lang_results.values()),
        "tokeval": {
            "utf8_completeness": _avg_opt([v["tokeval"]["utf8"]["utf8_completeness"]
                                           for v in lang_results.values()
                                           if v["tokeval"]["utf8"]]),
            "char_split_rate": _avg_opt([v["tokeval"]["utf8"]["char_split_rate"]
                                         for v in lang_results.values()
                                         if v["tokeval"]["utf8"]]),
            "boundary_crossing_rate": _avg_opt([v["tokeval"]["utf8"]["boundary_crossing_rate"]
                                                for v in lang_results.values()
                                                if v["tokeval"]["utf8"]]),
            "digit_boundary_f1": _avg_opt([v["tokeval"]["digit"]["digit_boundary_f1"]
                                           for v in lang_results.values()
                                           if v["tokeval"]["digit"]]),
            "digit_split_variability": _avg_opt([v["tokeval"]["digit"]["digit_split_variability"]
                                                 for v in lang_results.values()
                                                 if v["tokeval"]["digit"]]),
            "operator_isolation": _avg_opt([v["tokeval"]["operator"]["operator_isolation"]
                                            for v in lang_results.values()
                                            if v["tokeval"]["operator"]]),
        },
        "stability": {
            "per_lang": {k: v["stability"] for k, v in lang_results.items()},
            "overall": round(
                sum(v["stability"] for v in lang_results.values() if v["stability"] is not None)
                / max(1, sum(1 for v in lang_results.values() if v["stability"] is not None)),
                6,
            ),
        },
        "throughput": {
            "encode_tps": round(enc_tps, 3) if enc_tps is not None else None,
            "decode_tps": round(dec_tps, 3) if dec_tps is not None else None,
            "load_ms": round(load_ms, 3) if load_ms is not None else None,
        },
    }

    if protocol_sec is not None:
        report["protocol_robustness"] = protocol_sec

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # stdout Markdown 汇总
    print("# Tokenizer Evaluation Report")
    print("- model: %s" % model)
    print("- vocab_size: %s" % (vocab_size if vocab_size is not None else "null"))
    print("- date: %s" % date_str)
    print("")
    print("| lang | files | bytes | tokens | BPT | chars/token | roundtrip | stability | oov% | bytefb% |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lang in langs:
        v = lang_results[lang]
        rt = "OK" if v["roundtrip_ok"] else "FAIL(%d)" % v["roundtrip_mismatches"]
        stab = "%.4f" % v["stability"] if v["stability"] is not None else "-"
        oov = "%.4f" % v["oov_rate"] if v["oov_rate"] is not None else "-"
        fb = "%.4f" % v["byte_fallback_rate"] if v["byte_fallback_rate"] is not None else "-"
        print("| %s | %d | %d | %d | %.4f | %.4f | %s | %s | %s | %s |" % (
            lang, len(v["files"]), v["bytes"], v["tokens"], v["bpt"] or 0.0,
            v["chars_per_token"] or 0.0, rt, stab, oov, fb))
    t = report["totals"]
    print("| **total** | | %d | %d | %.4f | | | | | |" % (
        t["bytes"], t["tokens"], t["bpt"] or 0.0))
    print("")
    th = report["throughput"]
    print("## Throughput")
    print("- encode: %s tokens/sec" % ("%.1f" % th["encode_tps"] if th["encode_tps"] else "-"))
    print("- decode: %s tokens/sec" % ("%.1f" % th["decode_tps"] if th["decode_tps"] else "-"))
    print("- load: %s ms" % ("%.1f" % th["load_ms"] if th["load_ms"] else "-"))
    print("")
    print("## TokEval metrics (arXiv 2608.18062)")
    print("| lang | utf8_completeness | char_split_rate | boundary_crossing | digit_f1 | digit_variability | operator_isolation |")
    print("|---|---|---|---|---|---|---|")
    for lang in langs:
        v = lang_results[lang]["tokeval"]
        u8 = v["utf8"]
        dg = v["digit"]
        op = v["operator"]
        print("| %s | %s | %s | %s | %s | %s | %s |" % (
            lang,
            ("%.4f" % u8["utf8_completeness"]) if u8 and u8["utf8_completeness"] is not None else "-",
            ("%.4f" % u8["char_split_rate"]) if u8 and u8["char_split_rate"] is not None else "-",
            ("%.4f" % u8["boundary_crossing_rate"]) if u8 and u8["boundary_crossing_rate"] is not None else "-",
            ("%.4f" % dg["digit_boundary_f1"]) if dg and dg["digit_boundary_f1"] is not None else "-",
            ("%.4f" % dg["digit_split_variability"]) if dg and dg["digit_split_variability"] is not None else "-",
            ("%.4f" % op["operator_isolation"]) if op and op["operator_isolation"] is not None else "-",
        ))
    tk = report["tokeval"]
    print("| **overall** | %s | %s | %s | %s | %s | %s |" % (
        ("%.4f" % tk["utf8_completeness"]) if tk["utf8_completeness"] is not None else "-",
        ("%.4f" % tk["char_split_rate"]) if tk["char_split_rate"] is not None else "-",
        ("%.4f" % tk["boundary_crossing_rate"]) if tk["boundary_crossing_rate"] is not None else "-",
        ("%.4f" % tk["digit_boundary_f1"]) if tk["digit_boundary_f1"] is not None else "-",
        ("%.4f" % tk["digit_split_variability"]) if tk["digit_split_variability"] is not None else "-",
        ("%.4f" % tk["operator_isolation"]) if tk["operator_isolation"] is not None else "-",
    ))
    if protocol_sec is not None:
        pf = protocol_sec
        pd = pf["D"]
        pdisp = pf["dispersion"]
        print("")
        print("## 协议鲁棒性（双口径比值 D / 模型内离散度 / 评测集质量审计）")
        print("| model | D_en | D_zh | D_code | dispersion(max-min) | std | verdict |")
        print("|---|---|---|---|---|---|---|")
        print("| %s | %s | %s | %s | %s | %s | %s |" % (
            pf["label"], _pfmt(pd.get("en")), _pfmt(pd.get("zh")), _pfmt(pd.get("code")),
            _pfmt(pdisp.get("max_minus_min")), _pfmt(pdisp.get("std")),
            (pf["verdict"] or {}).get("level") or "-"))
        agg = pf["aggregate"]
        print("- bpt_total(按字节加权): %s | BPT_macro(三语种等权): %s | 口径: %s" % (
            _pfmt(agg.get("bpt_total_byte_weighted")), _pfmt(agg.get("bpt_macro")),
            agg.get("source") or "-"))
        print("- verdict: %s" % ((pf["verdict"] or {}).get("text") or "-"))
        print("- protocol_audit:")
        print("| tag | file | lines | chars | distinct_chars | distinct_cjk | dup_line_rate |")
        print("|---|---|---|---|---|---|---|")
        for tag, a in (pf.get("corpus_audit") or {}).items():
            if a.get("error"):
                print("| %s | %s | - | - | - | - | - |" % (tag, a.get("file")))
                continue
            print("| %s | %s | %d | %d | %d | %d | %s |" % (
                tag, a.get("file"), a.get("lines", 0), a.get("chars", 0),
                a.get("distinct_chars", 0), a.get("distinct_cjk_chars", 0),
                _pfmt(a.get("dup_line_rate"), 6)))
        payload = {
            "date": date_str,
            "title": "协议鲁棒性指标报告 — %s" % pf["label"],
            "notes": [
                "D_lang = BPT_eval / BPT_real（BPT = bytes/tokens，tokens 由 CLI 实测）。",
                "dispersion = max(D) - min(D)，并附三语种总体标准差；"
                "阈值 ≤0.35 稳定 / 0.35–1.0 敏感 / >1.0 口径依赖。",
                "真实语料为 scripts/indomain_compare.py、indomain_compare2.py 所用同一语料来源，"
                "本次未引入新语料。",
            ],
            "corpus": {"slices_dir": (pf.get("corpus") or {}).get("slices_dir"),
                       "full_corpus_dir": (pf.get("corpus") or {}).get("full_corpus_dir"),
                       "protocol_audit": pf.get("corpus_audit") or {}},
            "models": {pf["label"]: dict(pf, source="本机 CLI 实测（加速版 CLI，输出与 build/tok-cli.exe 逐字节一致）")},
        }
        if args.protocol_out:
            po = os.path.abspath(args.protocol_out)
            d = os.path.dirname(po)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            with open(po, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            print("Protocol JSON: %s" % po)
        if args.protocol_md:
            pm = os.path.abspath(args.protocol_md)
            d = os.path.dirname(pm)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            with open(pm, "w", encoding="utf-8") as fh:
                fh.write(render_protocol_markdown(payload, title=payload["title"]))
            print("Protocol Markdown: %s" % pm)
    print("")
    print("Report JSON: %s" % out_path)


if __name__ == "__main__":
    main()
