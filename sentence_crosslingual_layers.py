"""DEPTH SWEEP of cross-lingual sentence alignment: WHERE in the network does the universal manifold live?

v0.2.19 showed contextualized sentence meaning aligns cross-language. This maps it across depth. Cross-lingual is
the clean probe: OPT & Qwen share NO vocab, so surface (token-embedding) alignment is impossible -- alignment can
only emerge where meaning abstracts away from surface tokens. Mean-pool parallel sentences at 8 relative depths;
track Qwen-EN<->Qwen-ZH (language invariance) and OPT-EN<->Qwen-EN (model invariance) per depth.

PRE-REG: cross-lingual alignment LOW at depth 0 (different vocab -> token embeddings can't align), PEAKS mid-to-
upper (~0.6-0.8, meaning abstracted), possibly dips at the last layer (specializes to next-token in each language).
This locates the universal manifold in the upper-middle. Falsifier: flat, or surface-peaked (would contradict the
meaning-abstraction picture).
[V] OPT-2.7B + Qwen2.5-1.5B, mean-pooled hidden states at 8 depths, N=700 OPUS-100 en-zh, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"; QWEN = "Qwen/Qwen2.5-1.5B"
N = 700; K = 10; PCA_K = 64; MAXTOK = 64; DEPTHS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def load_sentences(n):
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
    en, zh = [], []
    for r in ds:
        e = r["translation"]["en"].strip(); z = r["translation"]["zh"].strip()
        if 5 <= len(e.split()) <= 25 and 4 <= len(z) <= 40:
            en.append(e); zh.append(z)
        if len(en) >= n:
            break
    return en, zh


def embed_depths(path, sents):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    nl = m.config.num_hidden_layers; idxs = [int(round(d * nl)) for d in DEPTHS]
    acc = {li: [] for li in idxs}
    for s in sents:
        ids = torch.tensor([tok.encode(s)[:MAXTOK]], device=DEV)
        hs = m(ids, output_hidden_states=True).hidden_states
        for li in idxs:
            acc[li].append(hs[li][0].float().mean(0).cpu())
    out = {}
    for d, li in zip(DEPTHS, idxs):
        X = torch.stack(acc[li]); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        out[d] = X
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); return Xc @ Vh[:k].t()


def mutual_knn(A, B, k):
    sa = A @ A.t(); sb = B @ B.t(); n = A.shape[0]; sa.fill_diagonal_(-2); sb.fill_diagonal_(-2)
    na = sa.topk(k, 1).indices; nb = sb.topk(k, 1).indices
    return float(np.mean([len(set(na[i].tolist()) & set(nb[i].tolist())) / k for i in range(n)]))


def p_at1(A, B):                                                     # same-dim -> direct cosine; diff-dim -> PCA64+Procrustes
    if A.shape[1] != B.shape[1]:
        A = _pca(A, PCA_K); B = _pca(B, PCA_K)
        M = A.t() @ B; U, S, Vh = torch.linalg.svd(M); A = A @ (U @ Vh)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9); B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return float((A @ B.t()).argmax(1).eq(torch.arange(A.shape[0])).float().mean())


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr = np.arange(int(.8 * n)); te = np.arange(int(.8 * n), n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    return float(1 - ((B[te] - A[te] @ R) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[DEPTH SWEEP: cross-lingual sentence alignment]  where does the universal manifold live?\n")
    en, zh = load_sentences(N); print(f"  {len(en)} OPUS-100 en-zh sentence pairs; depths {DEPTHS}\n")
    print("  embedding OPT-2.7B (all depths) ..."); opt_en = embed_depths(OPT, en)
    print("  embedding Qwen2.5-1.5B EN ..."); qwen_en = embed_depths(QWEN, en)
    print("  embedding Qwen2.5-1.5B ZH ..."); qwen_zh = embed_depths(QWEN, zh)
    print("  embedded.\n")

    print("  Qwen-EN <-> Qwen-ZH (language invariance) across depth:")
    print(f"    {'depth':>6} {'P@1':>7} {'kNN':>7} {'procr-R2':>9}")
    for d in DEPTHS:
        A, B = qwen_en[d], qwen_zh[d]
        print(f"    {d:>6.2f} {p_at1(A,B):>7.3f} {mutual_knn(A,B,K):>7.3f} {procrustes_r2(A,B):>9.3f}")
    print("\n  OPT-EN <-> Qwen-EN (model invariance) across depth:")
    print(f"    {'depth':>6} {'P@1':>7} {'kNN':>7} {'procr-R2':>9}")
    for d in DEPTHS:
        A, B = opt_en[d], qwen_en[d]
        print(f"    {d:>6.2f} {p_at1(A,B):>7.3f} {mutual_knn(A,B,K):>7.3f} {procrustes_r2(A,B):>9.3f}")

    peak = max(DEPTHS, key=lambda d: p_at1(qwen_en[d], qwen_zh[d]))
    print(f"\n[VERDICT]  cross-lingual (Qwen EN<->ZH) P@1 peaks at relative depth {peak:.2f}")
    print("  -> if peak is mid-to-upper and depth-0 is low, the universal manifold lives where meaning abstracts")
    print("     from surface tokens -- NOT at the token surface (which can't align across different vocabularies).")
    print("\n[V] OPT-2.7B + Qwen2.5-1.5B, mean-pooled at 8 depths, N=700 OPUS-100 en-zh, fp32.")


if __name__ == "__main__":
    run()
