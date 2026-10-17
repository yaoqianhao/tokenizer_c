#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_pipeline.py —— tokenizer_c 训练 → HF fast 导出 → 自检 端到端流水线

四段串联（构建 / 训练为可选段）：
    [build] -> [train] -> export -> verify

导出与自检的归一化模式由 --norm=nfkc|nfc|reversible 决定（默认 nfc；nfkc 为兼容模式）；
导出物 normalizer、tokenizer_c_norm.json sidecar、tok-cli encode --norm 三者同模式。

子命令：
    all      按需执行 构建/训练 → 转换 → 自检（默认入口）
    export   只做 转换（json -> HF fast 目录）
    verify   只做 自检（HF 加载 + 与 tok-cli encode 的 id parity）

任一环节失败或校验不通过，立即以非零退出码结束，并在 stdout 打印明确原因。

示例：
    # 用仓库内小语料跑通全流程（分钟级）
    python tools/hf_pipeline.py all --corpus tools/smoke_corpus.txt ^
        --vocab-size 2000 --iters 2 --workdir build/hf_pipeline

    # 已有 model.json，只转换 + 校验
    python tools/hf_pipeline.py all --json model_big2.json --model model_big2.tkm ^
        --out-dir build/hf_pipeline/hf_tokenizer

    # 只校验
    python tools/hf_pipeline.py verify --json build/hf_pipeline/model.json ^
        --model build/hf_pipeline/model.tkm --out-dir build/hf_pipeline/hf_tokenizer

退出码：0 全部成功；2 转换/加载失败；3 校验不通过；4 训练失败；5 构建失败；1 参数错误。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_export  # noqa: E402  （同目录模块）

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOK_CLI = os.path.join(REPO_ROOT, "build", "tok-cli.exe")
DEFAULT_WORKDIR = os.path.join(REPO_ROOT, "build", "hf_pipeline")


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list, step: str, allow_fail: bool = False):
    log("  $ " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True)
    dt = time.time() - t0
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    tail = (out.strip().splitlines()[-3:] if out.strip() else [])
    for ln in tail:
        log("    | " + ln)
    if proc.returncode != 0 and not allow_fail:
        log("[FAIL] %s 失败（exit=%d, %.1fs）" % (step, proc.returncode, dt))
        if err.strip():
            log("  stderr: " + err.strip()[-800:])
        raise SystemExit({"build": 5, "train": 4}.get(step, 2))
    return {"cmd": cmd, "returncode": proc.returncode, "seconds": round(dt, 2),
            "stdout_tail": tail, "stderr_tail": err.strip()[-800:]}


# --------------------------------------------------------------------------
# 四段
# --------------------------------------------------------------------------

def step_build(manifest: dict) -> None:
    log("[1/4] 构建（可选段）")
    ps1 = os.path.join(REPO_ROOT, "scripts", "build.ps1")
    manifest["build"] = run_cmd(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
        "build")


def step_train(corpus: str, model: str, json_out: str, tok_cli: str,
               opts: dict, manifest: dict) -> None:
    log("[2/4] 训练")
    if not os.path.exists(corpus):
        log("[FAIL] 语料不存在：%s" % corpus)
        raise SystemExit(1)
    for p in (model, json_out):
        if os.path.exists(p):
            log("[FAIL] 目标文件已存在（避免覆盖）：%s" % p)
            raise SystemExit(1)
    cmd = [tok_cli, "train", "--corpus", corpus, "--model", model, "--json", json_out]
    for k, v in opts.items():
        if v is not None:
            cmd += ["--%s" % k, str(v)]
    manifest["train"] = run_cmd(cmd, "train")
    if not os.path.exists(json_out):
        log("[FAIL] 训练结束但未生成 json：%s（现装 tok-cli 可能不支持 --json，见交付说明重建命令）" % json_out)
        raise SystemExit(4)


def step_export(json_in: str, out_dir: str, model_type: str | None,
                force: bool, manifest: dict,
                norm_mode: str = hf_export.DEFAULT_NORM_MODE) -> dict:
    log("[3/4] 转换 -> HF fast 目录（--norm=%s）" % norm_mode)
    if not os.path.exists(json_in):
        log("[FAIL] 输入 json 不存在：%s" % json_in)
        raise SystemExit(1)
    if os.path.exists(os.path.join(out_dir, "tokenizer.json")) and not force:
        log("[FAIL] 输出目录已含 tokenizer.json（避免覆盖）：%s\n"
            "       如确认要重建，请加 --force" % out_dir)
        raise SystemExit(1)
    cfg, stats, specials = hf_export.convert(hf_export._read_json(json_in),
                                             model_type=model_type,
                                             norm_mode=norm_mode)
    log("  归一化模式: --norm=%s -> normalizer = Sequence[%s]"
        % (stats["norm_mode"], ", ".join(stats["normalizer_types"])))
    log("  体检: %d 项发现" % len(stats["findings"]))
    for f in stats["findings"]:
        log("    [%s] %s" % (f["severity"], f["detail"]))
    paths = hf_export.write_outputs(cfg, out_dir, specials,
                                    norm_mode=norm_mode, source_json=json_in)
    for name, p in paths.items():
        log("  写入 %s (%d bytes)" % (p, os.path.getsize(p)))
    manifest["export"] = {"in": json_in, "out_dir": out_dir,
                          "pieces": stats["pieces"],
                          "null_fixed": stats["null_fixed"],
                          "byte_added": stats["byte_added"],
                          "model_type": stats["model_type"],
                          "norm_mode": stats["norm_mode"],
                          "normalizer_types": stats["normalizer_types"],
                          "findings_count": len(stats["findings"]),
                          "files": {k: os.path.getsize(v) for k, v in paths.items()}}
    return stats


def step_verify(out_dir: str, json_in: str, model: str | None, tok_cli: str,
                verify_file: str | None, tolerance: float, manifest: dict,
                strict: bool = False, tmp_dir: str | None = None,
                norm_mode: str = hf_export.DEFAULT_NORM_MODE) -> int:
    log("[4/4] 自检（HF 加载 + 与 tok-cli encode 的 piece 级 parity，--norm=%s）" % norm_mode)
    try:
        res = hf_export.verify(out_dir, json_in, model, tok_cli,
                               verify_file=verify_file, tolerance=tolerance, strict=strict,
                               tmp_dir=tmp_dir, norm_mode=norm_mode)
    except Exception as exc:  # noqa: BLE001
        log("[FAIL] 自检异常（%s: %s）" % (type(exc).__name__, exc))
        manifest["verify"] = {"verdict": "FAIL_EXCEPTION",
                              "error": "%s: %s" % (type(exc).__name__, exc)}
        return 2
    manifest["verify"] = res
    log("  加载: " + json.dumps(res["load"], ensure_ascii=False))
    if not res["load"].get("ok"):
        log("[FAIL] 产出目录无法被 tokenizers/transformers 加载")
        return 2
    if res.get("verdict") == "FAIL_NORMALIZER_MODE":
        log("[FAIL] 导出物 normalizer 与 --norm=%s 不一致：实际 Sequence[%s]，期望 Sequence[%s]"
            % (norm_mode, ", ".join(res.get("normalizer_types") or []),
               ", ".join(res.get("normalizer_expected") or [])))
        return 3
    if not res.get("c_cli_available"):
        log("[WARN] C 侧 piece 级 parity 跳过：%s" % res.get("c_cli_reason"))
        log("       请用 --model 指定 .tkm，或用支持该 --norm 的 tok-cli（scripts/build.ps1）重建")
        return 0 if res.get("verdict") == "PASS_LOAD_ONLY" else 1
    log("  源 json 剪枝(score=null) piece 数: %d" % res["pruned_piece_count"])
    for row in res["sentences"]:
        if row.get("c_tokens") is None:
            log("    %-30s HF=%-6d (C 侧跳过)" % (row["text"][:28].replace("\n", "\\n"),
                                                 row["hf_tokens"]))
            continue
        log("    %-30s C=%-6d HF=%-6d Δ=%+.4f%% %-11s piece_diff=%-3s csim_match=%s"
            % (row["text"][:28].replace("\n", "\\n"), row["c_tokens"], row["hf_tokens"],
               row["count_delta_pct"], row.get("diff_kind", "?"), row["piece_diff"],
               row.get("csim_reproduces_c")))
        kind = row.get("diff_kind")
        if kind == "explained":
            log("        └ 剪枝 piece 机制：移除 score=null 的 piece 后 HF 与 tok-cli 逐 piece 一致")
        elif kind == "tie_break":
            log("        └ 同分不同切分：路径总分 C=%s / C-sim=%s，文本无损"
                % (row.get("path_score_c"), row.get("path_score_sim")))
        elif kind == "lost_text":
            log("        └ [硬失败] 文本不可逆：c_text=%r hf_text=%r"
                % (row.get("text_c"), row.get("text_hf")))
        elif kind == "unexplained":
            log("        └ [硬失败] 未解释差异：text_equal=%s c_only=%s hf_only=%s"
                % (row.get("text_equal"), row.get("c_only_pieces"), row.get("hf_only_pieces")))
    log("  结论: %s (identical=%d explained=%d tie_break=%d unexplained=%d lost_text=%d "
        "all_piece_match=%s worst_delta=%.4f%%, tol=%.4f%%)"
        % (res["verdict"], res["sentences_identical"], res["sentences_explained"],
           res["sentences_tie_break"], res["sentences_unexplained"], res["sentences_lost_text"],
           res["all_piece_match"], res["worst_count_delta"], res["tolerance_pct"]))
    if res["verdict"] == "FAIL_LOAD":
        return 2
    if res["verdict"] in ("FAIL_TEXT", "FAIL_MISMATCH", "FAIL_STRICT"):
        log("[FAIL] 校验不通过（verdict=%s）：存在信息不可逆或无法解释的 piece 级差异，"
            "详见上文逐句明细" % res["verdict"])
        return 3
    if res["verdict"] == "PASS_KNOWN_DIFF":
        log("  说明: 差异全部归因于已核实机制（剪枝 piece 在 HF 端仍可匹配 / 同分不同切分），"
            "两侧还原文本无损；详见 docs/hf_export_pipeline.md")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="hf_pipeline.py",
        description="tokenizer_c: 构建/训练 → HF fast 导出 → 自检 端到端流水线")
    ap.add_argument("command", choices=["all", "export", "verify"])
    ap.add_argument("--corpus", default=None, help="训练语料（train 段需要）")
    ap.add_argument("--model", default=None, help=".tkm 模型路径（缺省 <workdir>/model.tkm）")
    ap.add_argument("--json", default=None, help="model json 路径（缺省 <workdir>/model.json）")
    ap.add_argument("--out-dir", default=None, help="HF fast 目录（缺省 <workdir>/hf_tokenizer）")
    ap.add_argument("--workdir", default=None, help="工作目录（缺省 <repo>/build/hf_pipeline）")
    ap.add_argument("--build", action="store_true", help="all 段先执行 scripts/build.ps1")
    ap.add_argument("--train", action="store_true", help="强制执行训练段（即使已给 --json）")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--max-piece-len", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--min-rounds", type=int, default=None)
    ap.add_argument("--tok-cli", default=DEFAULT_TOK_CLI)
    ap.add_argument("--model-type", default=None)
    ap.add_argument("--norm", default=hf_export.DEFAULT_NORM_MODE,
                    help="导出/自检的归一化模式 %s（默认 %s；nfkc 为兼容模式，可复现历史产出）"
                         % ("|".join(hf_export.NORM_MODES), hf_export.DEFAULT_NORM_MODE))
    ap.add_argument("--norm-reversible", action="store_true",
                    help="兼容别名：等价于 --norm=reversible")
    ap.add_argument("--verify-file", default=None)
    ap.add_argument("--verify-tolerance", type=float, default=0.02,
                    help="token 用量相对差提示阈值（默认 0.02；仅提示，不参与判定）")
    ap.add_argument("--tmp-dir", default=None,
                    help="自检时临时句子文件目录（缺省 <workdir>/verify_tmp，不写系统临时目录）")
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：自检阶段任何非逐 piece 一致即判定失败")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的输出目录")
    args = ap.parse_args(argv)

    try:
        norm_mode = hf_export.normalize_norm_mode(
            "reversible" if args.norm_reversible else args.norm)
    except ValueError as exc:
        log("[FAIL] %s" % exc)
        return 1

    workdir = os.path.abspath(args.workdir or DEFAULT_WORKDIR)
    json_path = os.path.abspath(args.json) if args.json else os.path.join(workdir, "model.json")
    if args.model:
        model = os.path.abspath(args.model)
    else:
        cand = os.path.join(workdir, "model.tkm")
        if not os.path.exists(cand):
            # 回落到 --json 同目录同名 .tkm（与 hf_export.py 的缺省规则一致）
            alt = os.path.splitext(json_path)[0] + ".tkm"
            if os.path.exists(alt):
                cand = alt
        model = cand
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(workdir, "hf_tokenizer")
    tok_cli = os.path.abspath(args.tok_cli)
    tmp_dir = os.path.abspath(args.tmp_dir) if args.tmp_dir else os.path.join(workdir, "verify_tmp")

    os.makedirs(workdir, exist_ok=True)

    log("== hf_pipeline: %s ==" % args.command)
    log("  repo      : %s" % REPO_ROOT)
    log("  python    : %s (%s)" % (sys.executable, sys.version.split()[0]))
    log("  tok-cli   : %s" % tok_cli)
    log("  workdir   : %s" % workdir)
    log("  model/json: %s | %s" % (model, json_path))
    log("  out-dir   : %s" % out_dir)
    log("  norm      : %s（normalizer = Sequence[%s]）"
        % (norm_mode, ", ".join(hf_export.expected_normalizer_types(norm_mode))))

    manifest = {"command": args.command, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python": sys.executable, "python_version": sys.version.split()[0],
                "repo": REPO_ROOT, "tok_cli": tok_cli, "workdir": workdir,
                "model": model, "json": json_path, "out_dir": out_dir,
                "norm_mode": norm_mode, "verify_tmp_dir": tmp_dir}

    need_train = args.command == "all" and (args.train or not args.json)

    if args.command == "all":
        if args.build:
            step_build(manifest)
        else:
            log("[1/4] 构建 —— 跳过（未指定 --build）")
        if need_train:
            if not args.corpus:
                log("[FAIL] 未提供 --corpus 且未提供 --json，训练段无法执行")
                return 1
            step_train(args.corpus, model, json_path, tok_cli,
                       {"vocab-size": args.vocab_size, "iters": args.iters,
                        "max-piece-len": args.max_piece_len,
                        "max-candidates": args.max_candidates,
                        "min-rounds": args.min_rounds}, manifest)
        else:
            log("[2/4] 训练 —— 跳过（使用既有 --json）")

    if args.command in ("all", "export"):
        step_export(json_path, out_dir, args.model_type, args.force, manifest,
                    norm_mode=norm_mode)

    if args.command in ("all", "verify"):
        rc = step_verify(out_dir, json_path, model, tok_cli,
                         args.verify_file, args.verify_tolerance, manifest,
                         strict=args.strict, tmp_dir=tmp_dir, norm_mode=norm_mode)
    else:
        rc = 0

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["exit_code"] = rc
    mf = os.path.join(workdir, "pipeline_manifest.json")
    with io.open(mf, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    log("  manifest  : %s" % mf)

    if args.command == "export":
        log("== export 完成 ==")
    else:
        log("== %s ==" % ("全部通过" if rc == 0 else "存在失败项（exit=%d）" % rc))
    return rc


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    sys.exit(main())
