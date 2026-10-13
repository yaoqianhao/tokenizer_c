$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$root  = "E:\tokenizer_c"
$temp  = "C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
$out   = "C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\output"
$fast  = Join-Path $temp "tkfast\build\tok-cli-fast.exe"
$orig  = Join-Path $root "build\tok-cli.exe"
$probe = Join-Path $temp "origload_probe.txt"
Set-Content -Path $probe -Value "x" -NoNewline -Encoding ascii
$summary = Join-Path $temp "eval_run_summary.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
Set-Location $root

$jobs = @(
  @{ tag = "big2"; model = "model_big2.tkm" },
  @{ tag = "big";  model = "model_big.tkm"  },
  @{ tag = "256k"; model = "benchmark\model_256k.tkm" }
)

foreach ($j in $jobs) {
  $tag = $j.tag
  $log = Join-Path $temp ("eval_$tag`_stdout.log")
  $err = Join-Path $temp ("eval_$tag`_stderr.log")
  $json = Join-Path $out ("eval_metrics_model_$tag.json")
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & python "benchmark\eval_metrics.py" --model $j.model --corpus-dir "benchmark\corpus" --cli $fast --output $json 1> $log 2> $err
  $rc = $LASTEXITCODE
  $sw.Stop()
  "EVAL TAG=$tag RC=$rc JSON=$json ELAPSED_S=$([math]::Round($sw.Elapsed.TotalSeconds,1))" | Out-File -FilePath $summary -Append -Encoding utf8
}

# 原版 CLI（未打补丁）单次启动+加载耗时，串行实测，每次间隔保证不受其它负载影响
foreach ($j in $jobs) {
  foreach ($round in 1..2) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $orig encode --model $j.model --file $probe 1> (Join-Path $temp "origload_out.txt") 2> $null
    $sw.Stop()
    "ORIGLOAD MODEL=$($j.model) ROUND=$round SEC=$([math]::Round($sw.Elapsed.TotalSeconds,3))" | Out-File -FilePath $summary -Append -Encoding utf8
  }
}

# 加速版 CLI 单次启动+加载耗时（对照）
foreach ($j in $jobs) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $fast encode --model $j.model --file $probe 1> (Join-Path $temp "fastload_out.txt") 2> $null
  $sw.Stop()
  "FASTLOAD MODEL=$($j.model) SEC=$([math]::Round($sw.Elapsed.TotalSeconds,3))" | Out-File -FilePath $summary -Append -Encoding utf8
}

"ALL_DONE" | Out-File -FilePath $summary -Append -Encoding utf8
