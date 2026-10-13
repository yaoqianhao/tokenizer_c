# -*- coding: utf-8 -*-
"""
tokenizer_c 15.0GB 加工语料训练脚本（Python 版，断点续行）
基于 train_big.py 改造：语料、模型、日志、checkpoint 与旧训练完全隔离。

用法：python train_big2.py [iters] [shrink] [vocab_size]
默认：iters=12  shrink=60  vocab_size=262144

语料：corpus_big2.txt = 旧(en/zh/code) + 新(en_ext/zh_ext/code_ext)
      六份语料块级混匀合并（100 万行/块轮流写入），15.0GB / 1.33 亿行。

断点续训：每轮 EM 结束自动保存 checkpoint。若 model_big2.ckpt 已存在，
重跑本脚本会自动检测并跳过 seed、从上一次轮次继续训练。
注意：本轮为全新训练，严禁复用旧训练 model_big.ckpt（语料分布已变，
且 tok-cli 原生不支持跨语料 resume）。
"""

import os
import shutil
import subprocess
import sys

BASE = r"E:\tokenizer_c"
CLI = os.path.join(BASE, "build3", "tok-cli.exe")
DLL = os.path.join(BASE, "build3", "libwinpthread-1.dll")
DLL_SRC = r"E:\WXQ\WXQ_Data\Plugins\x86_64\libwinpthread-1.dll"
CORPUS = os.path.join(BASE, "corpus_big", "corpus_big2.txt")
MODEL = os.path.join(BASE, "model_big2.tkm")
JSON = os.path.join(BASE, "model_big2.json")
LOG = os.path.join(BASE, "train_big2.log")
CKPT = os.path.join(BASE, "model_big2.ckpt")


def main():
    iters = sys.argv[1] if len(sys.argv) > 1 else "12"
    shrink = sys.argv[2] if len(sys.argv) > 2 else "60"
    vocab = sys.argv[3] if len(sys.argv) > 3 else "262144"

    # ---- 0. 运行库检查（libwinpthread-1.dll）----
    if not os.path.exists(DLL):
        print("[check] 复制 libwinpthread-1.dll 到 build3\\ ...")
        shutil.copyfile(DLL_SRC, DLL)
    if not os.path.exists(CLI):
        print(f"[err] 未找到 {CLI}")
        sys.exit(1)
    if not os.path.exists(CORPUS):
        print(f"[err] 未找到语料 {CORPUS}")
        sys.exit(1)

    # ---- 1. 断点检测提示 ----
    if os.path.exists(CKPT):
        print(f"[ckpt] 检测到 {CKPT}，将从断点续训（跳过 seed）")
    else:
        print("[ckpt] 未检测到 checkpoint，本次为全新训练")
    if os.path.exists(os.path.join(BASE, "model_big.ckpt")):
        print("[warn] 旧训练 model_big.ckpt 存在——请勿将其改名为 model_big2.ckpt 混入本轮！")

    # ---- 2. 启动训练（后台运行，日志写入 LOG）----
    print(f"[train] iters={iters} shrink={shrink} vocab={vocab}")
    print(f"[train] 日志: {LOG}")
    print("[train] 预估耗时: 15GB 语料 iters=8 约 400h+（种子显著长于旧版）")

    cmd = [
        CLI, "train",
        "--corpus", CORPUS,
        "--model", MODEL,
        "--vocab-size", vocab,
        "--json", JSON,
        "--iters", iters,
        "--shrink", shrink,
        "--ckpt", CKPT,
        "-v",
    ]

    log_f = open(LOG, "w", encoding="utf-8", errors="replace")
    err_f = open(LOG + ".err", "w", encoding="utf-8", errors="replace")

    # DETACHED_PROCESS: 脱离当前控制台独立后台运行，脚本退出后训练继续
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=err_f, creationflags=flags)
    log_f.close()
    err_f.close()

    print(f"[train] 已启动 PID={proc.pid}。查看进度： type \"{LOG}.err\" | findstr /i \"seed EM round\"")
    print('[train] 查看内存/进程： tasklist /fi "imagename eq tok-cli.exe"')


if __name__ == "__main__":
    main()
