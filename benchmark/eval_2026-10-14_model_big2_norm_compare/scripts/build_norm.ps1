# 2026-10-14: build the NFC-mode branch into its own directory so the existing
# build/ artifacts (tok-cli.exe, test_tokenizer.exe) are left untouched.
$ErrorActionPreference = "Stop"
$Root = "E:\tokenizer_c"
$Build = Join-Path $Root "build_norm"
New-Item -ItemType Directory -Force -Path $Build | Out-Null

$srcs = @("utf8.c","normalize.c","nfkc.c","nfc.c","trie.c","tokenizer.c",
          "model.c","segment_rules.c","sentencepiece.c","trainer.c")
$objs = @()
foreach ($s in $srcs) {
    $obj = Join-Path $Build ($s -replace '\.c$', '.o')
    Write-Host "cc $s"
    & gcc -std=c11 -Wall -Wextra -O2 -I (Join-Path $Root "include") -I (Join-Path $Root "src") -c (Join-Path $Root ("src\" + $s)) -o $obj
    if ($LASTEXITCODE -ne 0) { throw "compile failed: $s" }
    $objs += $obj
}
& ar rcs (Join-Path $Build "libtokenizer.a") $objs
if ($LASTEXITCODE -ne 0) { throw "ar failed" }

Write-Host "link tok-cli-norm.exe"
& gcc -std=c11 -O2 -I (Join-Path $Root "include") -I (Join-Path $Root "src") (Join-Path $Root "cli\main.c") (Join-Path $Build "libtokenizer.a") -lm -o (Join-Path $Build "tok-cli-norm.exe")
if ($LASTEXITCODE -ne 0) { throw "cli link failed" }

Write-Host "link test_tokenizer_norm.exe"
& gcc -std=c11 -O2 -I (Join-Path $Root "include") -I (Join-Path $Root "src") (Join-Path $Root "tests\test_tokenizer.c") (Join-Path $Build "libtokenizer.a") -lm -o (Join-Path $Build "test_tokenizer_norm.exe")
if ($LASTEXITCODE -ne 0) { throw "test link failed" }

Write-Host "BUILD OK -> $Build"
