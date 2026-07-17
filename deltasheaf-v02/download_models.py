#!/usr/bin/env python
"""Pre-download the 3 ensemble models whose WEIGHTS aren't cached yet (AutoConfig fetched only configs).
Resumable (snapshot_download resumes partial downloads across kills). Then the offline screen scores them
kill-free. Run: python download_models.py"""
import sys
from huggingface_hub import snapshot_download
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

MODELS = ["HuggingFaceTB/SmolLM2-1.7B-Instruct", "tiiuae/Falcon3-3B-Instruct", "allenai/OLMo-2-0425-1B-Instruct"]
PATS = ["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"]
for m in MODELS:
    print(f"[download] {m}", flush=True)
    p = snapshot_download(m, allow_patterns=PATS)
    print(f"[done]     {m} -> {p}", flush=True)
print("ALL 3 WEIGHTS CACHED", flush=True)
