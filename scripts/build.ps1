# Build script (Windows PowerShell)
# Usage:
#   .\scripts\build.ps1            # CMake build to build/
#   .\scripts\build.ps1 -Gcc      # plain gcc build (no CMake needed)
# Outputs: build\tok-cli.exe, build\test_tokenizer.exe, build\test_tiktoken_bpe.exe,
#          build\test_cli_tt_raw.exe, build\libtokenizer.a

param(
    [switch]$Gcc
)

$ErrorActionPreference = "Stop"
# PSScriptRoot may be empty in some hosts that invoke the script directly
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$Root = Split-Path -Parent $ScriptDir   # project root (parent of scripts/)
$Build = Join-Path $Root "build"

if ($Gcc) {
    # gcc writes warnings to stderr; with $ErrorActionPreference = "Stop", PowerShell 5.1 turns
    # any stderr write by a native command into a terminating error and aborts the build even on
    # a harmless warning.  Real failures are still detected by the $LASTEXITCODE checks below.
    $ErrorActionPreference = "Continue"
    New-Item -ItemType Directory -Force -Path $Build | Out-Null
    $objs = @()
    foreach ($src in @("utf8.c","normalize.c","nfkc.c","nfc.c","trie.c","tokenizer.c","model.c","segment_rules.c","sentencepiece.c","trainer.c","tiktoken_bpe.c")) {
        $obj = Join-Path $Build ($src -replace '\.c$', '.o')
        & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $Root "include") -I (Join-Path $Root "src") -c (Join-Path $Root ("src\" + $src)) -o $obj
        if ($LASTEXITCODE -ne 0) { throw "compile failed: $src" }
        $objs += $obj
    }
    & ar rcs (Join-Path $Build "libtokenizer.a") @objs
    if ($LASTEXITCODE -ne 0) { throw "ar failed" }
    & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $Root "include") (Join-Path $Root "cli\main.c") (Join-Path $Build "libtokenizer.a") -lm -lshell32 -o (Join-Path $Build "tok-cli.exe")
    if ($LASTEXITCODE -ne 0) { throw "cli link failed" }
    & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $Root "include") (Join-Path $Root "tests\test_tokenizer.c") (Join-Path $Build "libtokenizer.a") -lm -o (Join-Path $Build "test_tokenizer.exe")
    if ($LASTEXITCODE -ne 0) { throw "test link failed" }
    & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $Root "include") (Join-Path $Root "tests\test_tiktoken_bpe.c") (Join-Path $Build "libtokenizer.a") -lm -o (Join-Path $Build "test_tiktoken_bpe.exe")
    if ($LASTEXITCODE -ne 0) { throw "tiktoken_bpe test link failed" }
    # CLI level regression test for `tt-decode --raw` (needs the CLI binary only)
    & gcc -std=c11 -Wall -Wextra -O2 (Join-Path $Root "tests\test_cli_tt_raw.c") -o (Join-Path $Build "test_cli_tt_raw.exe")
    if ($LASTEXITCODE -ne 0) { throw "cli raw test link failed" }
    Write-Host "gcc build OK: $Build"
    return
}

# default CMake build
New-Item -ItemType Directory -Force -Path $Build | Out-Null
& cmake -S $Root -B $Build -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
& cmake --build $Build --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
Write-Host "cmake build OK: $Build"
