$ErrorActionPreference = "Stop"
$FAST = "C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp\tkfast"
New-Item -ItemType Directory -Force -Path "$FAST\build" | Out-Null
$objs = @()
foreach ($src in @("utf8.c", "normalize.c", "nfkc.c", "trie.c", "tokenizer.c", "model.c", "sentencepiece.c", "trainer.c")) {
    $obj = Join-Path "$FAST\build" ($src -replace '\.c$', '.o')
    & gcc -std=c11 -Wall -Wextra -O2 -I "$FAST\include" -I "$FAST\src" -c "$FAST\src\$src" -o $obj
    if ($LASTEXITCODE -ne 0) { Write-Output "COMPILE_FAIL $src"; exit 1 }
    Write-Output "compiled $src"
    $objs += $obj
}
& ar rcs "$FAST\build\libtokenizer.a" @objs
if ($LASTEXITCODE -ne 0) { Write-Output "AR_FAIL"; exit 1 }
& gcc -std=c11 -Wall -Wextra -O2 -I "$FAST\include" "$FAST\cli\main.c" "$FAST\build\libtokenizer.a" -lm -lshell32 -o "$FAST\build\tok-cli-fast.exe"
if ($LASTEXITCODE -ne 0) { Write-Output "CLI_LINK_FAIL"; exit 1 }
Write-Output "cli linked"
& gcc -std=c11 -Wall -Wextra -O2 -I "$FAST\include" "$FAST\tests\test_tokenizer.c" "$FAST\build\libtokenizer.a" -lm -o "$FAST\build\test_tokenizer.exe"
if ($LASTEXITCODE -ne 0) { Write-Output "TEST_LINK_FAIL"; exit 1 }
Write-Output "BUILD_DONE"
