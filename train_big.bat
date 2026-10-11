@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
rem ============================================================
rem  tokenizer_c 3.28GB 海量语料训练脚本（断点续行版）
rem  参数：train_big.bat [iters] [shrink] [vocab_size]
rem  默认：iters=8  shrink=60  vocab_size=262144
rem
rem  断点续训：训练命令带 --ckpt E:\tokenizer_c\model_big.ckpt，
rem  每轮 EM 结束自动保存 checkpoint。若 model_big.ckpt 已存在，
rem  重跑本脚本会自动检测并跳过 seed、从上一次轮次继续训练，
rem  因此可直接重复运行本脚本实现断点续训。
rem ============================================================

set "BASE=E:\tokenizer_c"
set "CLI=%BASE%\build3\tok-cli.exe"
set "CORPUS=%BASE%\corpus_big\corpus_big.txt"
set "MODEL=%BASE%\model_big.tkm"
set "JSON=%BASE%\model_big.json"
set "LOG=%BASE%\train_big.log"

set "ITERS=%~1"
if "%ITERS%"=="" set "ITERS=8"
set "SHRINK=%~2"
if "%SHRINK%"=="" set "SHRINK=60"
set "VOCAB=%~3"
if "%VOCAB%"=="" set "VOCAB=262144"

rem ---- 0. 运行库检查（libwinpthread-1.dll）----
if not exist "%BASE%\build3\libwinpthread-1.dll" (
    echo [check] 复制 libwinpthread-1.dll 到 build3\ ...
    copy /Y "E:\WXQ\WXQ_Data\Plugins\x86_64\libwinpthread-1.dll" "%BASE%\build3\libwinpthread-1.dll" >nul
)
if not exist "%CLI%" ( echo [err] 未找到 %CLI% & exit /b 1 )
if not exist "%CORPUS%" ( echo [err] 未找到语料 %CORPUS% & exit /b 1 )

rem ---- 1. 断点检测提示 ----
if exist "%BASE%\model_big.ckpt" (
    echo [ckpt] 检测到 %BASE%\model_big.ckpt，将从断点续训（跳过 seed）
) else (
    echo [ckpt] 未检测到 checkpoint，本次为全新训练
)

rem ---- 2. 启动训练（后台运行，日志写入 %LOG%）----
echo [train] iters=%ITERS% shrink=%SHRINK% vocab=%VOCAB%
echo [train] 日志: %LOG%
echo [train] 预估耗时: iters=8 约 110~140 小时（种子约 3~4h + 每轮 EM 约 10~14h）

start "tokenizer-train" /min cmd /c ""%CLI%" train --corpus "%CORPUS%" --model "%MODEL%" --vocab-size %VOCAB% --json "%JSON%" --iters %ITERS% --shrink %SHRINK% --ckpt "%BASE%\model_big.ckpt" -v >"%LOG%" 2>"%LOG%.err""

echo [train] 已启动。查看进度： type "%LOG%.err" | findstr /i "seed EM round"
echo [train] 查看内存/进程： tasklist /fi "imagename eq tok-cli.exe"
endlocal
