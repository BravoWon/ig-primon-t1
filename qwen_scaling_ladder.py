"""SCALING LADDER: does the cross-lingual manifold STRENGTHEN with model scale? (the PRH core claim)

7B loads here but lands on CPU (15GB fp16 > 12GB card -> dual-GPU split offloads to CPU -> impractical for batch
embedding). The reliable, GPU-resident scaling test is a within-family ladder: Qwen2.5-0.5B / 1.5B / 3B (all fit
the 5070 in fp16). For each: cross-lingual EN<->ZH and EN<->ES geometric alignment (Procrustes) across depth, and
cross-model OPT-2.7B-EN <-> Qwen-X-EN. The PRH predicts alignment GROWS with scale.

PRE-REG: cross-lingual peak Procrustes increases 0.5B -> 1.5B -> 3B; the upper-middle placement holds at all
sizes; cross-model alignment to OPT grows with Qwen scale. Falsifier: flat/decreasing peaks -> scale-robust but
convergence does not strengthen on this axis (on these small-to-mid sizes).
[V] Qwen2.5-{0.5B,1.5B,3B} fp16 + OPT-2.7B, mean-pooled, N=400 OPUS-100, depths near peak, fp32 metrics.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"
LADDER = [("0.5B", "Qwen/Qwen2.5-0.5B"), ("1.5B", "Qwen/Qwen2.5-1.5B"), ("3B", "Qwen/Qwen2.5-3B")]
N = 400; PCA_K = 32; MAXTOK = 48; DEPTHS = [0.3, 0.5, 0.7, 0.85, 1.0]


def load_pairs(lang, n):
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "-".join(sorted(["en", lang])), split="test")
    en, fo = [], []
    for r in ds:
        e = r["translation"]["en"].strip(); f = r["translation"][lang].strip()
        if 5 <= len(e.split()) <= 22 and 4 <= len(f) <= 70:
            en.append(e); fo.append(f)
        if len(en) >= n:
            break
    return en, fo


def embed_depths(path, sets, fp16):
    tok = AutoTokenizer.from_pretrained(path)
    dt = torch.float16 if fp16 else torch.float32
    m = AutoModelForCausalLM.from_pretrained(path).to(dt).to(DEV).eval()
    nl = m.config.num_hidden_layers; idxs = [int(round(d * nl)) for d in DEPTHS]
    outs = []
    for sents in sets:
        acc = {li: [] for li in idxs}
        for s in sents:
            ids = torch.tensor([tok.encode(s)[:MAXTOK]], device=DEV)
            hs = m(ids, output_hidden_states=True).hidden_states
            for li in idxs:
                acc[li].append(hs[li][0].float().mean(0).cpu())
        d2 = {}
        for d, li in zip(DEPTHS, idxs):
            X = torch.stack(acc[li]); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
            d2[d] = X
        outs.append(d2)
    del m; gc.collect(); torch.cuda.empty_cache()
    return outs


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); return Xc @ Vh[:k].t()


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr = np.arange(int(.8 * n)); te = np.arange(int(.8 * n), n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    return float(1 - ((B[te] - A[te] @ R) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[SCALING LADDER]  does the cross-lingual manifold strengthen with model scale?\n")
    en_zh, zh = load_pairs("zh", N); en_es, es = load_pairs("es", N)
    print(f"  N={N}; loading OPT-2.7B for cross-model arm ...")
    (opt_en,) = embed_depths(OPT, [en_zh], fp16=False)

    res = {}
    for tag, path in LADDER:
        print(f"  embedding Qwen2.5-{tag} ...")
        q_enzh, q_zh, q_enes, q_es = embed_depths(path, [en_zh, zh, en_es, es], fp16=True)
        zpk = max(procrustes_r2(q_enzh[d], q_zh[d]) for d in DEPTHS)
        epk = max(procrustes_r2(q_enes[d], q_es[d]) for d in DEPTHS)
        mpk = max(procrustes_r2(opt_en[d], q_enzh[d]) for d in DEPTHS)
        zcurve = [procrustes_r2(q_enzh[d], q_zh[d]) for d in DEPTHS]
        res[tag] = (zpk, epk, mpk, zcurve)

    print(f"\n  {'model':>8} {'EN<->ZH peak':>13} {'EN<->ES peak':>13} {'OPT<->Qwen(EN) peak':>20}")
    for tag, _ in LADDER:
        zpk, epk, mpk, _ = res[tag]
        print(f"  Qwen-{tag:>4} {zpk:>13.3f} {epk:>13.3f} {mpk:>20.3f}")
    print(f"\n  EN<->ZH Procrustes by depth {DEPTHS}:")
    for tag, _ in LADDER:
        print(f"    Qwen-{tag:>4}: " + " ".join(f"{v:>6.3f}" for v in res[tag][3]))

    print("\n[VERDICT -- PRH scaling]")
    zpks = [res[t][0] for t, _ in LADDER]; mpks = [res[t][2] for t, _ in LADDER]
    print(f"  cross-lingual EN<->ZH peak: {zpks[0]:.3f} -> {zpks[1]:.3f} -> {zpks[2]:.3f}  "
          f"({'GROWS with scale' if zpks[2] > zpks[0] + 0.02 else 'flat/scale-robust'})")
    print(f"  cross-model OPT<->Qwen peak: {mpks[0]:.3f} -> {mpks[1]:.3f} -> {mpks[2]:.3f}  "
          f"({'GROWS with scale (PRH)' if mpks[2] > mpks[0] + 0.02 else 'flat/scale-robust'})")
    print("\n[V] Qwen2.5-0.5/1.5/3B fp16 + OPT-2.7B, mean-pooled, N=400 OPUS-100, fp32 metrics.")


if __name__ == "__main__":
    run()
