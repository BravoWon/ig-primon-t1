# Terminus PARAM-scale HARDENING -- seeds 0,1,2 x 3 sizes x 3 arms (seed 0 already present -> skipped).
# Resumable (skips existing metrics); logs to terminus\_pscale.out; pins the 5070 (device 0).
# Seed-outer so seed 0 skips fast, then seed 1 fully, then seed 2. ~2.8 h for the 18 new runs.
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "0"
$py   = "C:\Users\JT-DEV1\Desktop\development\ig-primon-t1\.venv\Scripts\python.exe"
$root = "C:\Users\JT-DEV1\Desktop\development"
Set-Location $root
$log  = "$root\terminus\_pscale.out"
function Note($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding utf8 }

Note "==== param-scale HARDENING (seeds 0,1,2) START (pid $PID) ===="
$budget = 30000000
$sizes = @(
  @{ tag = "ps014"; d = 128; layers = 6; heads = 4 },
  @{ tag = "ps040"; d = 320; layers = 6; heads = 8 },
  @{ tag = "ps090"; d = 576; layers = 8; heads = 9 }
)
$arms = @("flat", "grounded", "grounded-random")
foreach ($seed in 0, 1, 2) {
  foreach ($s in $sizes) {
    foreach ($arm in $arms) {
      $mf = "$root\terminus\metrics_$($s.tag)_${arm}_30M_s${seed}.json"
      if (Test-Path $mf) { Note "SKIP  $($s.tag) $arm s$seed (exists)"; continue }
      Note "RUN   $($s.tag) $arm s$seed  (d=$($s.d) L=$($s.layers) H=$($s.heads))"
      & $py "terminus/train.py" --arm $arm --budget $budget --data data124M --tag $s.tag `
          --d $s.d --layers $s.layers --heads $s.heads --seed $seed *>> $log
      $ec = $LASTEXITCODE
      if ($ec -ne 0) { Note "ERROR exit $ec on $($s.tag) $arm s$seed -- ABORT"; exit 1 }
      Note "DONE  $($s.tag) $arm s$seed"
    }
  }
}
Note "==== all 27 present (3 seeds x 9 configs) -- aggregating ===="
& $py "terminus/aggregate_pscale.py" *>> $log
Note "==== HARDENING COMPLETE ===="
