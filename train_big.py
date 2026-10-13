# -*- coding: utf-8 -*-
"""
tokenizer_c 3.28GB 海量语料训练脚本（Python 版，断点续行）
替代 train_big.bat 使用。

用法：python train_big.py [iters] [shrink] [vocab_size]
默认：iters=8  shrink=60  vocab_size=262144

断点续训：训练命令带 --ckpt E:\tokenizer_c\model_big.ckpt，
每轮 EM 结束自动保存 checkpoint。若 model_big.ckpt 已存在，
重跑本脚本会自动检测并跳过 seed、从上一次轮次继续训练，
因此可直接重复运行本脚本实现断点续训。
"""

import os
import shutil
import subprocess
import sys

BASE = r"E:\tokenizer_c"
CLI = os.path.join(BASE, "build3", "tok-cli.exe")
DLL = os.path.join(BASE, "build3", "libwinpthread-1.dll")
DLL_SRC = r"E:\WXQ\WXQ_Data\Plugins\x86_64\libwinpthread-1.dll"
CORPUS = os.path.join(BASE, "corpus_big", "corpus_big.txt")
MODEL = os.path.join(BASE, "model_big.tkm")
JSON = os.path.join(BASE, "model_big.json")
LOG = os.path.join(BASE, "train_big.log")
CKPT = os.path.join(BASE, "model_big.ckpt")


def main():
    iters = sys.argv[1] if len(sys.argv) > 1 else "8"
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

    # ---- 2. 启动训练（后台运行，日志写入 LOG）----
    print(f"[train] iters={iters} shrink={shrink} vocab={vocab}")
    print(f"[train] 日志: {LOG}")
    print("[train] 预估耗时: iters=8 约 110~140 小时（种子约 3~4h + 每轮 EM 约 10~14h）")

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
