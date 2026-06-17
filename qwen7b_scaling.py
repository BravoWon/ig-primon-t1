"""SCALE TEST: does the upper-middle cross-lingual manifold hold and STRENGTHEN at a standard 7B model?

Our cross-lingual finding (v0.2.19-21) rests on small models (OPT-2.7B, Qwen-1.5B). The PRH's central claim is
that convergence STRENGTHENS with scale. This re-runs the key alignment with Qwen2.5-7B (a standard 7B, loaded
fp16 across both GPUs) and compares to the 1.5B baseline:
  - Cross-lingual: Qwen-7B EN<->ZH and EN<->ES geometric alignment (Procrustes) across depth. Does the upper-
    middle peak hold? Is it STRONGER than Qwen-1.5B (zh peaked ~0.34 @0.9; es ~0.41 @0.9)?
  - Cross-model (PRH scaling): OPT-2.7B-EN <-> Qwen-7B-EN vs <-> Qwen-1.5B-EN -> does cross-model alignment grow
    with the partner's scale?

PRE-REG: PRH -> 7B shows the SAME upper-middle placement but HIGHER peak alignment than 1.5B (both cross-lingual
and cross-model). Falsifier: 7B same/weaker -> the manifold is scale-robust but convergence does not strengthen
with scale on this axis.
[V] Qwen2.5-7B (fp16, 2-GPU) + OPT-2.7B, mean-pooled, N=300 OPUS-100, depths near the peak, fp32 metrics.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"; QWEN7 = "Qwen/Qwen2.5-7B"
N = 300; PCA_K = 32; MAXTOK = 48; DEPTHS = [0.3, 0.5, 0.7, 0.85, 1.0]


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


def embed_depths(m, tok, sents, dev_in):
    nl = m.config.num_hidden_layers; idxs = [int(round(d * nl)) for d in DEPTHS]
    acc = {li: [] for li in idxs}
    for s in sents:
        ids = torch.tensor([tok.encode(s)[:MAXTOK]], device=dev_in)
        hs = m(ids, output_hidden_states=True).hidden_states
        for li in idxs:
            acc[li].append(hs[li][0].float().mean(0).cpu())
    out = {}
    for d, li in zip(DEPTHS, idxs):
        X = torch.stack(acc[li]); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        out[d] = X
    return out


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); return Xc @ Vh[:k].t()


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr = np.arange(int(.8 * n)); te = np.arange(int(.8 * n), n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    return float(1 - ((B[te] - A[te] @ R) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def p_at1(A, B):
    if A.shape[1] != B.shape[1]:
        A = _pca(A, PCA_K); B = _pca(B, PCA_K)
        M = A.t() @ B; U, S, Vh = torch.linalg.svd(M); A = A @ (U @ Vh)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9); B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return float((A @ B.t()).argmax(1).eq(torch.arange(A.shape[0])).float().mean())


def run():
    import time
    print("[SCALE TEST: Qwen2.5-7B]  does the cross-lingual manifold hold + strengthen at a standard 7B?\n")
    en_zh, zh = load_pairs("zh", N); en_es, es = load_pairs("es", N)
    print(f"  N={N} per language\n  loading Qwen2.5-7B (fp16, device_map auto)...")
    tok = AutoTokenizer.from_pretrained(QWEN7)
    m = AutoModelForCausalLM.from_pretrained(QWEN7, torch_dtype=torch.float16, device_map="auto")
    dev_in = m.get_input_embeddings().weight.device
    t0 = time.time()
    q_en_zh = embed_depths(m, tok, en_zh, dev_in); q_zh = embed_depths(m, tok, zh, dev_in)
    q_en_es = embed_depths(m, tok, en_es, dev_in); q_es = embed_depths(m, tok, es, dev_in)
    q_en_for_opt = q_en_zh                                          # reuse en-zh English set for the cross-model arm
    print(f"  Qwen-7B embedding done in {time.time()-t0:.0f}s")
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.empty_cache()

    print("  loading OPT-2.7B for the cross-model arm...")
    tok2 = AutoTokenizer.from_pretrained(OPT)
    mo = AutoModelForCausalLM.from_pretrained(OPT).to(torch.float32).to("cuda:0").eval()
    opt_en = embed_depths(mo, tok2, en_zh, "cuda:0"); del mo; gc.collect(); torch.cuda.empty_cache()

    print("\n  Procrustes-R2 by depth (Qwen-7B):")
    print(f"    {'depth':>6} {'EN<->ZH':>9} {'EN<->ES':>9} {'OPT<->Qw7B(EN)':>15}")
    for d in DEPTHS:
        print(f"    {d:>6.2f} {procrustes_r2(q_en_zh[d], q_zh[d]):>9.3f} {procrustes_r2(q_en_es[d], q_es[d]):>9.3f} "
              f"{procrustes_r2(opt_en[d], q_en_for_opt[d]):>15.3f}")
    print(f"\n  P@1 (Qwen-7B EN->ZH direct): " + ", ".join(f"{d}:{p_at1(q_en_zh[d], q_zh[d]):.3f}" for d in DEPTHS))

    print("\n[VERDICT vs 1.5B baseline]")
    z_peak = max(procrustes_r2(q_en_zh[d], q_zh[d]) for d in DEPTHS)
    e_peak = max(procrustes_r2(q_en_es[d], q_es[d]) for d in DEPTHS)
    print(f"  Qwen-7B EN<->ZH peak Procrustes = {z_peak:.3f}  (Qwen-1.5B was ~0.34)")
    print(f"  Qwen-7B EN<->ES peak Procrustes = {e_peak:.3f}  (Qwen-1.5B was ~0.41)")
    print("  -> if 7B > 1.5B, PRH scaling holds: convergence STRENGTHENS with scale. If ~equal, scale-robust but flat.")
    print("\n[V] Qwen2.5-7B fp16 2-GPU + OPT-2.7B, mean-pooled, N=300 OPUS-100, fp32 metrics.")


if __name__ == "__main__":
    run()
