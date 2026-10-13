# =====================================================================
# 复现评测用加速版 CLI（tok-cli-fast.exe）
#   生成环境：Windows 11 + gcc (MinGW-w64, c11)
#   做法：拷贝项目源码 -> 覆盖本次唯一改动的 2 个文件 -> gcc 编译
#   用法： powershell -NoProfile -ExecutionPolicy Bypass -File build_fast_from_patch.ps1 `
#            -ProjectRoot "E:\tokenizer_c" -OutDir "E:\temp_rebuild_fast"
# =====================================================================
param(
    [string]$ProjectRoot = "E:\tokenizer_c",
    [string]$OutDir      = (Join-Path $env:TEMP "rebuild_tok_cli_fast")
)
$ErrorActionPreference = "Continue"   # gcc 告警走 stderr，勿按错误中断；每步单独校验 $LASTEXITCODE
$src = $PSScriptRoot          # 本脚本所在目录（归档目录的 cli-fast/）

Write-Output "[1/3] 准备构建目录 $OutDir"
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Output "[2/3] 拷贝项目源码 src/include/cli/tests，并覆盖本次改动的文件"
foreach ($d in @("src", "include", "cli", "tests")) {
    Copy-Item -Recurse -Force (Join-Path $ProjectRoot $d) $OutDir
}
Copy-Item -Force (Join-Path $src "changed_src\tokenizer.c")          (Join-Path $OutDir "src\tokenizer.c")
Copy-Item -Force (Join-Path $src "changed_src\tokenizer_internal.h") (Join-Path $OutDir "src\tokenizer_internal.h")

Write-Output "[3/3] gcc 编译"
$build = Join-Path $OutDir "build"
$gccLog = Join-Path $OutDir "build_gcc_stderr.log"   # gcc 告警/提示统一落盘，避免被当作 PowerShell 错误中断脚本
New-Item -ItemType Directory -Force -Path $build | Out-Null
$objs = @()
foreach ($c in @("utf8.c", "normalize.c", "nfkc.c", "trie.c", "tokenizer.c", "model.c", "sentencepiece.c", "trainer.c")) {
    $obj = Join-Path $build ($c -replace '\.c$', '.o')
    & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $OutDir "include") -I (Join-Path $OutDir "src") -c (Join-Path $OutDir "src\$c") -o $obj 2>> $gccLog
    if ($LASTEXITCODE -ne 0) { Write-Output "COMPILE_FAIL $c"; exit 1 }
    Write-Output "compiled $c"
    $objs += $obj
}
& ar rcs (Join-Path $build "libtokenizer.a") @objs 2>> $gccLog
if ($LASTEXITCODE -ne 0) { Write-Output "AR_FAIL"; exit 1 }
& gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $OutDir "include") (Join-Path $OutDir "cli\main.c") (Join-Path $build "libtokenizer.a") -lm -lshell32 -o (Join-Path $build "tok-cli-fast.exe") 2>> $gccLog
if ($LASTEXITCODE -ne 0) { Write-Output "CLI_LINK_FAIL"; exit 1 }
& gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $OutDir "include") (Join-Path $OutDir "tests\test_tokenizer.c") (Join-Path $build "libtokenizer.a") -lm -o (Join-Path $build "test_tokenizer.exe") 2>> $gccLog
if ($LASTEXITCODE -ne 0) { Write-Output "TEST_LINK_FAIL"; exit 1 }

$exePath = Join-Path $build "tok-cli-fast.exe"
Write-Output "BUILD_DONE -> $exePath"
