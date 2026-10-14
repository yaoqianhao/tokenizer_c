# run_compare_norm.ps1 — 一键复现 nfkc / nfc / reversible 三模式并排对比
#
# 流程：编译（独立 build_norm 目录，不触碰既有 build/）-> 单测回归 -> 三模式评测 -> 并排报告
# 本脚本不删除、不移动任何既有文件；仅新增/覆盖本归档目录内由它生成的评测产物。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File run_compare_norm.ps1
#   powershell -ExecutionPolicy Bypass -File run_compare_norm.ps1 -SkipBuild        # 跳过编译，用已有 build_norm 产物
#   powershell -ExecutionPolicy Bypass -File run_compare_norm.ps1 -SkipRun          # 不重跑评测，仅用已有 JSON 重新渲染报告
#   powershell -ExecutionPolicy Bypass -File run_compare_norm.ps1 -Modes nfkc,nfc    # 只对比指定模式

param(
    [string]   $Root    = "E:\tokenizer_c",
    [string]   $Archive = "E:\tokenizer_c\benchmark\eval_2026-10-14_model_big2_norm_compare",
    [string]   $Model   = "E:\tokenizer_c\model_big2.tkm",
    [string[]] $Modes   = @("nfkc", "nfc", "reversible"),
    [switch]   $SkipBuild,
    [switch]   $SkipRun
)

$ErrorActionPreference = "Stop"
$buildDir  = Join-Path $Root "build_norm"
$scriptDir = Join-Path $Archive "scripts"
$cliRebuild = Join-Path $buildDir "tok-cli-norm.exe"
$cliArchive = Join-Path $Archive "bin\tok-cli-norm.exe"

Write-Host "=== [1/3] build (NFC-mode branch) ==="
if ($SkipBuild) {
    Write-Host "skip build (-SkipBuild)"
} else {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $scriptDir "build_norm.ps1")
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

Write-Host "=== [2/3] unit test regression ==="
$t = Join-Path $buildDir "test_tokenizer_norm.exe"
if (Test-Path $t) {
    & $t
    if ($LASTEXITCODE -ne 0) { throw "unit test failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "skip unit test (no $t)"
}

Write-Host "=== [3/3] three-mode comparison + report ==="
$cli = if (Test-Path $cliRebuild) { $cliRebuild } elseif (Test-Path $cliArchive) { $cliArchive } else { throw "no tok-cli-norm.exe found" }
Write-Host "cli   : $cli"
Write-Host "model : $Model"

$py = @(
    (Join-Path $scriptDir "norm_mode_compare.py"),
    "--root",   $Root,
    "--model",  $Model,
    "--cli",    $cli,
    "--outdir", $Archive,
    "--modes",  ($Modes -join ",")
)
if ($SkipRun) { $py += "--skip-run" }

& python @py
if ($LASTEXITCODE -ne 0) { throw "comparison failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "报告    : $(Join-Path $Archive 'reports\norm_mode_compare.md')"
Write-Host "汇总 JSON: $(Join-Path $Archive 'metrics\norm_mode_compare_summary.json')"
