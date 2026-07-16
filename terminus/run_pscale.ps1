# Terminus PARAM-scale ladder -- local, RTX 5070. Varies MODEL SIZE at a FIXED 30M-token budget
# (the axis the data-scale ladder held fixed at 40M). Reuses data124M's validated grounding stream.
# Resumable (skips existing metrics files); logs to terminus\_pscale.out. Pins the 5070 (device 0).
# ps040 == the data-scale config (d=320,6L) -> its 30M numbers should reproduce metrics_lad_*_30M_s0
# (a built-in instrument/consistency anchor).
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "0"
$py   = "C:\Users\JT-DEV1\Desktop\development\ig-primon-t1\.venv\Scripts\python.exe"
$root = "C:\Users\JT-DEV1\Desktop\development"
Set-Location $root
$log  = "$root\terminus\_pscale.out"
function Note($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding utf8 }

Note "==== param-scale ladder START (pid $PID) ===="
$budget = 30000000
$sizes = @(
  @{ tag = "ps014"; d = 128; layers = 6;  heads = 4  },   # ~14M
  @{ tag = "ps040"; d = 320; layers = 6;  heads = 8  },   # ~40M (== data-scale config; consistency anchor)
  @{ tag = "ps090"; d = 576; layers = 8;  heads = 9  }    # ~90M (VRAM-safe on 12GB, fp32)
)
$arms = @("flat", "grounded", "grounded-random")
foreach ($s in $sizes) {
  foreach ($arm in $arms) {
    $mf = "$root\terminus\metrics_$($s.tag)_${arm}_30M_s0.json"
    if (Test-Path $mf) { Note "SKIP  $($s.tag) $arm (exists)"; continue }
    Note "RUN   $($s.tag) $arm  (d=$($s.d) L=$($s.layers) H=$($s.heads))"
    & $py "terminus/train.py" --arm $arm --budget $budget --data data124M --tag $s.tag `
        --d $s.d --layers $s.layers --heads $s.heads --seed 0 *>> $log
    $ec = $LASTEXITCODE
    if ($ec -ne 0) { Note "ERROR exit $ec on $($s.tag) $arm -- ABORT"; exit 1 }
    Note "DONE  $($s.tag) $arm"
  }
}
Note "==== all param points present -- aggregating ===="
& $py "terminus/aggregate_pscale.py" *>> $log
Note "==== DRIVER COMPLETE ===="
