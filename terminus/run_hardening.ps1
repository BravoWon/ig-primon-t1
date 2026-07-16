# Terminus hardening driver -- resumable multi-seed ladder (seeds 1,2 x 3 budgets x 3 arms).
# Skips any config whose metrics file already exists, so re-running resumes where it left off.
# Order: seed-outer (seed 1 fully first => a complete 2-seed read at the halfway mark, then seed 2).
# Logs every run to terminus\_hardening.out so a crash is diagnosable (the gap in the first attempt).
$ErrorActionPreference = "Continue"
$py   = "C:\Users\JT-DEV1\Desktop\development\ig-primon-t1\.venv\Scripts\python.exe"
$root = "C:\Users\JT-DEV1\Desktop\development"
Set-Location $root
$log  = "$root\terminus\_hardening.out"
function Note($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding utf8 }

Note "==== hardening driver START (pid $PID) ===="
$budgets = @(8000000, 30000000, 90000000)
$arms    = @("flat", "grounded", "grounded-random")
foreach ($seed in 1, 2) {
  foreach ($b in $budgets) {
    $bM = [int]($b / 1000000)
    foreach ($arm in $arms) {
      $mf = "$root\terminus\metrics_lad_${arm}_${bM}M_s${seed}.json"
      if (Test-Path $mf) { Note "SKIP  $arm ${bM}M s$seed (exists)"; continue }
      Note "RUN   $arm ${bM}M s$seed"
      & $py "terminus/train.py" --arm $arm --budget $b --data data124M --tag lad --seed $seed *>> $log
      $ec = $LASTEXITCODE
      if ($ec -ne 0) { Note "ERROR exit $ec on $arm ${bM}M s$seed -- ABORT"; exit 1 }
      Note "DONE  $arm ${bM}M s$seed"
    }
  }
}
Note "==== all 18 runs present -- aggregating ===="
& $py "terminus/aggregate.py" *>> $log
Note "==== DRIVER COMPLETE ===="
