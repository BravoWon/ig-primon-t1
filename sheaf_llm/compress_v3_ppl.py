#!/usr/bin/env python
"""structure-vs-bits v3 — the DOWNSTREAM gate. Does the v2 proxy win survive real perplexity?

Apply each scheme to ALL mlp/proj linears at EQUAL avg ~4.0 b/param, measure WikiText-2 perplexity:
  fp16   : reference (no quant)
  u4     : uniform per-column 4-bit
  mix    : use-aware allocation (top-20% input channels by activation energy -> 8-bit, rest 3-bit)

Gate: mix ppl < u4 ppl  ->  the allocator brick is real downstream, not just on the weighted proxy.

    python compress_v3_ppl.py [hf_model_id]
"""
import sys
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("*.py"))
    blobs = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
            if len(t) > 200:
                blobs.append(t)
        except Exception:
            pass
    full = "\n\n".join(blobs)
    n = len(full)
    return full[:int(n * 0.45)], full[int(n * 0.50):]      # calib, held-out test

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rtn_percol(W, bits_col):
    Wq = W.clone()
    for b in bits_col.unique().tolist():
        mask = bits_col == b; qmax = 2 ** (int(b) - 1) - 1
        cols = W[:, mask]; s = cols.abs().amax(0, keepdim=True) / qmax + 1e-12
        Wq[:, mask] = torch.clamp(torch.round(cols / s), -qmax, qmax) * s
    return Wq


def main():
    print(f"[downstream ppl gate v3]  model={MODEL}  dev={DEV}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    calib, test = load_corpus()
    print(f"corpus: calib {len(calib)} chars, held-out test {len(test)} chars")

    targets = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 256
               and any(t in n for t in ("mlp", "proj"))}

    # --- calibration moments (on original fp16 weights) ---
    moments = {}
    def mk(nm):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            moments[nm] = moments.get(nm, torch.zeros(x.shape[-1], device=x.device)) + (x * x).sum(0)
        return hook
    handles = [m.register_forward_pre_hook(mk(n)) for n, m in targets.items()]
    ids_c = tok(calib, return_tensors="pt").input_ids[0]
    with torch.no_grad():
        for i in range(0, min(len(ids_c), 512 * 12), 512):
            model(ids_c[i:i + 512].unsqueeze(0).to(DEV))
    for h in handles:
        h.remove()

    orig = {n: m.weight.detach().clone().cpu() for n, m in targets.items()}

    def apply(scheme):
        for n, m in targets.items():
            W = orig[n].to(DEV); ncol = W.shape[1]
            if scheme == "fp16":
                Wq = W
            elif scheme == "u4":
                Wq = rtn_percol(W, torch.full((ncol,), 4, dtype=torch.long, device=DEV))
            else:  # mix
                d = moments[n].to(DEV); k = round(0.2 * ncol)
                bits = torch.full((ncol,), 3, dtype=torch.long, device=DEV)
                bits[torch.argsort(d, descending=True)[:k]] = 8
                Wq = rtn_percol(W, bits)
            m.weight.data = Wq.to(m.weight.dtype)

    ids = tok(test, return_tensors="pt").input_ids[0]
    def ppl():
        nll = ntok = 0; L = 512
        with torch.no_grad():
            for i in range(0, min(len(ids), L * 40), L):
                ch = ids[i:i + L].unsqueeze(0).to(DEV)
                if ch.shape[1] < 2:
                    break
                out = model(ch, labels=ch)
                nll += out.loss.item() * (ch.shape[1] - 1); ntok += ch.shape[1] - 1
        return float(np.exp(nll / ntok))

    res = {}
    for s in ("fp16", "u4", "mix"):
        apply(s); res[s] = ppl()
        print(f"  {s:5} perplexity = {res[s]:.3f}")
    print(f"\nGATE (equal ~4.0 b/param):  mix {res['mix']:.3f}  vs  u4 {res['u4']:.3f}  -> "
          f"{'ALLOCATOR BRICK HOLDS' if res['mix'] < res['u4'] else 'brick FAILS downstream'} "
          f"({(res['u4']-res['mix'])/res['u4']*100:+.1f}% ppl vs u4; fp16 ref {res['fp16']:.3f})")


if __name__ == "__main__":
    main()
