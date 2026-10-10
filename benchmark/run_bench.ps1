# run_bench.ps1 - drive a tok-cli train run while capturing:
#   * wall-clock timeline for each verbose stderr stage (polling the
#     redirected stderr file, ~150 ms resolution - enough for stage times)
#   * peak working set / private bytes (MB)
#   * final exit code and total wall time
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_bench.ps1 `
#       -Exe <tok-cli.exe> -Corpus <corpus.txt> -Model <out.tkm> `
#       [-Vocab 262144] [-ExtraArgs "-v --max-candidates 1000000"] `
#       [-Timeline bench_timeline.txt] [-Log bench_stdout.txt]
param(
    [string]$Exe,
    [string]$Corpus,
    [string]$Model,
    [int]$Vocab = 262144,
    [string]$ExtraArgs = "-v --max-candidates 1000000 --max-piece-len 24",
    [string]$Timeline = "bench_timeline.txt",
    [string]$Log = "bench_stdout.txt",
    [string]$ErrLog = "bench_stderr.txt",
    [string]$WorkDir = "."
)
$ErrorActionPreference = "Stop"
Set-Location $WorkDir

$argLine = "train --corpus `"$Corpus`" --model `"$Model`" --vocab-size $Vocab $ExtraArgs"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe -ArgumentList $argLine -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError $ErrLog `
        -WindowStyle Hidden
$swStart = [DateTime]::Now

$peakWS = [long]0
$peakPriv = [long]0
$samples = 0
$readPos = 0
$tlList = New-Object 'System.Collections.Generic.List[string]'

# seed timeline with process start marker
$tlList.Add(("{0:O} START pid={1}" -f $swStart, $p.Id))

while (-not $p.HasExited) {
    $p.Refresh()
    if ($p.WorkingSet64 -gt $peakWS) { $peakWS = $p.WorkingSet64 }
    if ($p.PrivateMemorySize64 -gt $peakPriv) { $peakPriv = $p.PrivateMemorySize64 }
    $samples++
    # read newly appended stderr lines with a timestamp
    if (Test-Path $ErrLog) {
        $fs = [System.IO.File]::Open($ErrLog, [System.IO.FileMode]::Open,
              [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            $fs.Seek($readPos, [System.IO.SeekOrigin]::Begin) | Out-Null
            $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
            while ($true) {
                $line = $sr.ReadLine()
                if ($null -eq $line) { break }
                $tlList.Add(("{0:O} {1}" -f [DateTime]::Now, $line))
            }
            $readPos = $fs.Position
            $sr.Dispose()
        } finally {
            $fs.Dispose()
        }
    }
    Start-Sleep -Milliseconds 150
}
$p.WaitForExit()
$p.Refresh()
$sw.Stop()
if ($p.WorkingSet64 -gt $peakWS) { $peakWS = $p.WorkingSet64 }
if ($p.PrivateMemorySize64 -gt $peakPriv) { $peakPriv = $p.PrivateMemorySize64 }

# final flush of stderr
if (Test-Path $ErrLog) {
    $fs = [System.IO.File]::Open($ErrLog, [System.IO.FileMode]::Open,
          [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        $fs.Seek($readPos, [System.IO.SeekOrigin]::Begin) | Out-Null
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
        while ($true) {
            $line = $sr.ReadLine()
            if ($null -eq $line) { break }
            $tlList.Add(("{0:O} {1}" -f [DateTime]::Now, $line))
        }
        $sr.Dispose()
    } finally {
        $fs.Dispose()
    }
}

$tlList | Out-File -FilePath $Timeline -Encoding utf8
$elapsed = $sw.Elapsed.TotalSeconds
"WALL_S={0:F3}" -f $elapsed
"PEAK_WS_MB={0:F1}" -f ($peakWS / 1MB)
"PEAK_PRIV_MB={0:F1}" -f ($peakPriv / 1MB)
"MEM_SAMPLES=$samples"
"EXIT=$($p.ExitCode)"
"TIMELINE=$Timeline"
"STDERR=$ErrLog"
