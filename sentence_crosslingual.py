"""SENTENCE-LEVEL cross-lingual alignment: does CONTEXTUALIZED meaning align globally where isolated words didn't?

The cross-lingual word result (v0.2.18) was local-only cross-language -- but single words are noisy and context-
free. This tests the sharpest remaining caveat: full PARALLEL SENTENCES (OPUS-100 en-zh), embedded as mean-pooled
contextual hidden states. If sentence-meaning aligns globally cross-language (positive Procrustes/linear map,
parallel sentences as mutual nearest neighbors), context rescues the universal manifold. If it stays local, the
cross-language boundary is fundamental for a general (non-alignment-trained) decoder.

Three arms (OPT GPT-2-BPE vs Qwen own tokenizer -> confound-free):
  (1) Qwen-EN <-> Qwen-ZH : language invariance of sentence meaning within one multilingual model.
  (2) OPT-EN  <-> Qwen-EN : model invariance (PRH) on sentences.
  (3) OPT-EN  <-> Qwen-ZH : full test.
Metric note: mutual-kNN at sentence level = is sentence i's translation among its cross-space nearest neighbors
(bitext-retrieval flavored). PCA-64, N=700 -> well-posed.

PRE-REG: cross-model same-language strong (PRH); cross-language -- open. LaBSE/LASER align sentences cross-lingual
but are TRAINED for it; general decoders may not. Honest either way.
[V] OPT-2.7B + Qwen2.5-1.5B, mean-pooled contextual ~2/3 depth, N=700 OPUS-100 en-zh sentence pairs, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"; QWEN = "Qwen/Qwen2.5-1.5B"
N = 700; K = 10; PCA_K = 64; MAXTOK = 64


def load_sentences(n):
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
    en, zh = [], []
    for r in ds:
        e = r["translation"]["en"].strip(); z = r["translation"]["zh"].strip()
        if 5 <= len(e.split()) <= 25 and 4 <= len(z) <= 40 and e and z:
            en.append(e); zh.append(z)
        if len(en) >= n:
            break
    return en, zh


def embed_all(path, sets):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    layer = int(round(0.66 * m.config.num_hidden_layers))
    out = []
    for sents in sets:
        X = []
        for s in sents:
            ids = torch.tensor([tok.encode(s)[:MAXTOK]], device=DEV)
            hs = m(ids, output_hidden_states=True).hidden_states[layer][0]   # (ntok,d)
            X.append(hs.float().mean(0).cpu())                              # mean-pool over the sentence
        X = torch.stack(X); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        out.append(X)
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:k].t()


def mutual_knn(A, B, k):
    sa = A @ A.t(); sb = B @ B.t(); n = A.shape[0]
    sa.fill_diagonal_(-2); sb.fill_diagonal_(-2)
    na = sa.topk(k, 1).indices; nb = sb.topk(k, 1).indices
    return float(np.mean([len(set(na[i].tolist()) & set(nb[i].tolist())) / k for i in range(n)]))


def retrieval_at1(A, B):
    """P@1: is B[i] the top match of A[i]? Same-dim -> direct cosine (no map). Diff-dim -> PCA-64 + Procrustes align."""
    if A.shape[1] != B.shape[1]:
        A = _pca(A, PCA_K); B = _pca(B, PCA_K)
        M = A.t() @ B; U, S, Vh = torch.linalg.svd(M); A = A @ (U @ Vh)   # orthogonal align A->B before retrieval
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9); B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    S = A @ B.t(); return float((S.argmax(1) == torch.arange(A.shape[0])).float().mean())


def _sp(n):
    return np.arange(int(.8 * n)), np.arange(int(.8 * n), n)


def linear_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _sp(n)
    A1 = torch.cat([A, torch.ones(n, 1)], 1); W = torch.linalg.lstsq(A1[tr], B[tr]).solution
    return float(1 - ((B[te] - A1[te] @ W) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _sp(n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    return float(1 - ((B[te] - A[te] @ R) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[SENTENCE-LEVEL CROSS-LINGUAL]  does contextualized meaning align globally cross-language?\n")
    en, zh = load_sentences(N)
    print(f"  {len(en)} OPUS-100 en-zh sentence pairs\n  sample EN: {en[0][:70]}\n  sample ZH: {zh[0][:30]}\n")
    print("  embedding OPT-2.7B ...");  (opt_en,) = embed_all(OPT, [en])
    print("  embedding Qwen2.5-1.5B ..."); qwen_en, qwen_zh = embed_all(QWEN, [en, zh])
    print("  embedded.\n")

    n = len(en); perm = torch.tensor(np.random.default_rng(0).permutation(n))
    arms = [("Qwen-EN <-> Qwen-ZH (language)", qwen_en, qwen_zh),
            ("OPT-EN  <-> Qwen-EN (model)", opt_en, qwen_en),
            ("OPT-EN  <-> Qwen-ZH (full)", opt_en, qwen_zh)]
    print(f"  {'arm':>32} {'kNN':>7} {'shuf':>6} {'P@1':>6} {'procr-R2':>9} {'lin-R2':>8}")
    rows = {}
    for name, A, B in arms:
        rows[name] = (mutual_knn(A, B, K), mutual_knn(A, B[perm], K), retrieval_at1(A, B), procrustes_r2(A, B), linear_r2(A, B))
        mk, sh, p1, pr, li = rows[name]
        print(f"  {name:>32} {mk:>7.3f} {sh:>6.3f} {p1:>6.3f} {pr:>9.3f} {li:>8.3f}")

    print("\n[VERDICT]")
    print(f"  chance kNN ~ {K/n:.4f}, chance P@1 ~ {1/n:.4f}")
    lang = rows["Qwen-EN <-> Qwen-ZH (language)"]
    if lang[3] > 0.1 or lang[2] > 0.3:
        print("  -> CONTEXT RESCUES IT: sentence-level cross-language aligns globally (positive Procrustes and/or high")
        print("     retrieval) where isolated words were local-only. Contextualized meaning is the universal manifold.")
    elif lang[0] > 3 * lang[1] or lang[2] > 5 / n:
        print("  -> STILL LOCAL (context helps retrieval but not a global map): parallel sentences are neighbors")
        print("     (P@1/kNN above chance) but cross-language Procrustes/linear stays weak. Boundary is fundamental.")
    else:
        print("  -> weak/mixed: report straight.")
    print("\n[V] OPT-2.7B + Qwen2.5-1.5B, mean-pooled contextual ~2/3 depth, N=700 OPUS-100 en-zh, PCA-64, fp32.")


if __name__ == "__main__":
    run()
