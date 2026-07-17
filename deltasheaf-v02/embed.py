#!/usr/bin/env python
"""DeltaSheaf-v0.2 — embed reply texts -> nomic ℝ⁵¹² stalks, and option texts -> option embeddings.
Output data/embedded.npz: stalks[n,5,512], opts[n,4,512], gold[n], correct[n,5], letters[n,5]."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

TAGS = ["qwen25_3b", "phi35_mini", "smollm2_17b", "falcon3_3b", "olmo2_1b"]
LET = {"A":0,"B":1,"C":2,"D":3}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
mdl = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(DEV).eval()

def embed(texts, bs=256):
    out = []
    for s in range(0, len(texts), bs):
        chunk = ["search_document: " + (t or "") for t in texts[s:s+bs]]
        enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        with torch.no_grad(): h = mdl(**enc)[0]
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h*m).sum(1)/m.sum(1)
        e = F.layer_norm(e, (e.shape[1],))[:, :512]
        out.append(F.normalize(e, dim=1).cpu().numpy())
        print(f"    embed {s+len(chunk)}/{len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float32)

pool = [json.loads(l) for l in open("data/mmlu_pool.jsonl", encoding="utf-8")]
raw = {t: [json.loads(l) for l in open(f"data/raw/{t}.jsonl", encoding="utf-8")] for t in TAGS}
n = min(len(v) for v in raw.values())
print(f"[embed] n={n}, {len(TAGS)} models")

# stalks: embed each model's reply text
stalks = np.zeros((n, len(TAGS), 512), np.float32)
for mi, t in enumerate(TAGS):
    print(f"  model {t}")
    stalks[:, mi] = embed([raw[t][i]["reply"] for i in range(n)])

# option embeddings (for cosine readout) + gold + per-model letters/correct
opts = np.zeros((n, 4, 512), np.float32)
flat = [c for i in range(n) for c in pool[i]["choices"]]
femb = embed(flat)
opts = femb.reshape(n, 4, 512)
gold = np.array([pool[i]["answer"] for i in range(n)], np.int64)
letters = np.array([[LET.get(raw[t][i]["letter"], -1) for t in TAGS] for i in range(n)], np.int64)
correct = np.array([[raw[t][i]["correct"] for t in TAGS] for i in range(n)], bool)

np.savez("data/embedded.npz", stalks=stalks, opts=opts, gold=gold, letters=letters, correct=correct)
print(f"[done] data/embedded.npz  stalks{stalks.shape} opts{opts.shape}")
