# DeltaSheaf-v0.2 Phase-2 — ensemble screening: score 5 frozen models on MMLU (sequential, fits 12GB).
# Resumable (skips models whose data/raw/<tag>.jsonl exists); logs to _screen.out. Pins the 5070.
# Output feeds: 0-of-5 blind-spot (gate) selection + clean-train, then nomic-embed -> substrate -> arms.
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_OFFLINE = "1"        # all 5 models cached -> load from cache, zero network (the terminus regime)
$env:TRANSFORMERS_OFFLINE = "1"
$py   = "C:\Users\JT-DEV1\Desktop\development\ig-primon-t1\.venv\Scripts\python.exe"
$root = "C:\Users\JT-DEV1\Desktop\development"
Set-Location $root
$log  = "$root\deltasheaf-v02\_screen.out"
function Note($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) -Encoding utf8 }

Note "==== ensemble screen START (pid $PID) ===="
$models = @(
  @{ id = "Qwen/Qwen2.5-3B-Instruct";          tag = "qwen25_3b"  },
  @{ id = "microsoft/Phi-3.5-mini-instruct";   tag = "phi35_mini" },
  @{ id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"; tag = "smollm2_17b" },
  @{ id = "tiiuae/Falcon3-3B-Instruct";        tag = "falcon3_3b" },
  @{ id = "allenai/OLMo-2-0425-1B-Instruct";   tag = "olmo2_1b"   }
)
$N = 4200; $BATCH = 12    # expanded per H3 (204 gate items at 2500 < 300; ~8.2% rate -> ~4200 for >=300)
foreach ($m in $models) {
  $out = "deltasheaf-v02/data/raw/$($m.tag).jsonl"
  Note "RUN   $($m.tag)  [$($m.id)]  n=$N batch=$BATCH  (score.py resumes at item level)"
  & $py "deltasheaf-v02/score.py" --model $m.id --n $N --batch $BATCH --max_new 256 --out $out *>> $log
  if ($LASTEXITCODE -ne 0) { Note "ERROR exit $LASTEXITCODE on $($m.tag)"; }
  else { Note "DONE  $($m.tag)" }
}
Note "==== SCREEN COMPLETE ===="
