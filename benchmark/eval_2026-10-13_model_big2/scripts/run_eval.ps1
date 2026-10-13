$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$root = "E:\tokenizer_c"
$temp = "C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\temp"
$out  = "C:\Users\Nero\AppData\Roaming\Tencent\Marvis\User\oAN1i2dpCCuZq-FgvYnflZ6SBnQA\workspace\conv_dbd7c0fe35e3451ba9e8b1bc07331f46\output"
Set-Location $root

$jobs = @(
  @{ tag = "big2"; model = "model_big2.tkm" },
  @{ tag = "big";  model = "model_big.tkm"  },
  @{ tag = "m256k"; model = "benchmark\model_256k.tkm" }
)

foreach ($j in $jobs) {
  $tag = $j.tag
  $log = Join-Path $temp ("eval_{0}_stdout.log" -f $tag)
  $err = Join-Path $temp ("eval_{0}_stderr.log" -f $tag)
  $json = Join-Path $out ("eval_metrics_{0}.json" -f $tag)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & python "benchmark\eval_metrics.py" --model $j.model --corpus-dir "benchmark\corpus" --cli "build\tok-cli.exe" --output $json 1> $log 2> $err
  $rc = $LASTEXITCODE
  $sw.Stop()
  "TAG=$tag RC=$rc ELAPSED_S=$([math]::Round($sw.Elapsed.TotalSeconds,1)) JSON=$json" | Out-File -FilePath (Join-Path $temp "eval_run_summary.txt") -Append -Encoding utf8
}
"ALL_DONE" | Out-File -FilePath (Join-Path $temp "eval_run_summary.txt") -Append -Encoding utf8
