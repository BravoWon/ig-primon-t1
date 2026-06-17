"""CROSS-LINGUAL PROCRUSTES v2: powered up -- Qwen2.5-1.5B + N=700 MUSE pairs (well-posed map metrics).

v1 (128 hand pairs, Qwen-0.5B) left the map metrics underdetermined. This uses the standard MUSE en-zh bilingual
dictionary (frequency-ordered, verified) for N=700 clean 1:1 pairs and the larger Qwen2.5-1.5B, so PCA-64 maps
are well-posed (560 train >> 64) and Procrustes/linear-R^2 are now trustworthy alongside mutual-kNN.

Three arms decompose the invariances (OPT GPT-2-BPE vs Qwen's own tokenizer share NO vocab -> confound-free):
  (1) Qwen-EN <-> Qwen-ZH : language invariance within one multilingual model.
  (2) OPT-EN  <-> Qwen-EN : model invariance, same language, different tokenizers (confound-free PRH).
  (3) OPT-EN  <-> Qwen-ZH : the full universal-manifold test (model + language + tokenizer).

PRE-REG: with N powered up, if cross-language Procrustes is genuinely >0 (not just kNN), the strong "meaning is
invariant up to rotation across languages" claim survives; if Procrustes stays ~0 while kNN >> chance, it's
local-not-global (v1's read holds under power). Classic MUSE result: static embeddings rotate well cross-lingual;
contextual decoder single-word reps are the open question.
[V] OPT-2.7B + Qwen2.5-1.5B, contextual ~2/3 depth, N=700 MUSE en-zh pairs, PCA-64, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; OPT = "C:/Users/JT-DEV1/Documents/opt-2.7b"; QWEN = "Qwen/Qwen2.5-1.5B"
N = 700; K = 10; PCA_K = 64; SKIP = 800   # skip the top-SKIP frequency-ranked (function-word-heavy) entries -> content words


def load_pairs(n):
    pairs = []; seen_en = set(); seen_zh = set()
    for line in open("muse_en_zh.txt", encoding="utf-8"):
        p = line.split()
        if len(p) != 2:
            continue
        en, zh = p
        if not (en.isascii() and en.isalpha() and len(en) >= 4) or en in seen_en:
            continue
        if not all('一' <= c <= '鿿' for c in zh) or not (1 <= len(zh) <= 4) or zh in seen_zh:
            continue
        seen_en.add(en); seen_zh.add(zh); pairs.append((en, zh))
        if len(pairs) >= n + SKIP:
            break
    return pairs[SKIP:SKIP + n]                                      # content-word band (past the function-word head)


def embed_all(path, sets):
    """Load model once; embed each (items, leading_space) set -> centered, normalized matrix."""
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    layer = int(round(0.66 * m.config.num_hidden_layers))
    out = []
    for items, lead in sets:
        X = []
        for w in items:
            ids = torch.tensor([tok.encode((" " + w) if lead else w)], device=DEV)
            hs = m(ids, output_hidden_states=True).hidden_states[layer]
            X.append(hs[0, -1].float().cpu())
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
    print("[CROSS-LINGUAL PROCRUSTES v2]  powered up: Qwen2.5-1.5B + N=700 MUSE pairs\n")
    pairs = load_pairs(N); en = [e for e, _ in pairs]; zh = [z for _, z in pairs]
    print(f"  {len(pairs)} MUSE en-zh pairs (sample en: {en[:8]})")
    print("  embedding OPT-2.7B (local) ...")
    (opt_en,) = embed_all(OPT, [(en, True)])
    print("  embedding Qwen2.5-1.5B (downloading if needed) ...")
    qwen_en, qwen_zh = embed_all(QWEN, [(en, True), (zh, False)])
    print("  embedded.\n")

    n = len(pairs); perm = torch.tensor(np.random.default_rng(0).permutation(n))
    arms = [("Qwen-EN <-> Qwen-ZH  (language)", qwen_en, qwen_zh),
            ("OPT-EN  <-> Qwen-EN  (model)", opt_en, qwen_en),
            ("OPT-EN  <-> Qwen-ZH  (full)", opt_en, qwen_zh)]
    print(f"  {'arm':>34} {'mutual-kNN':>11} {'shuf':>6} {'procrustes-R2':>14} {'linear-R2':>10}")
    rows = {}
    for name, A, B in arms:
        rows[name] = (mutual_knn(A, B, K), mutual_knn(A, B[perm], K), procrustes_r2(A, B), linear_r2(A, B))
        mk, sh, pr, li = rows[name]
        print(f"  {name:>34} {mk:>11.3f} {sh:>6.3f} {pr:>14.3f} {li:>10.3f}")

    print("\n[VERDICT -- powered, map metrics now well-posed]")
    print(f"  chance kNN ~ {K/n:.4f}")
    lang = rows["Qwen-EN <-> Qwen-ZH  (language)"]; full = rows["OPT-EN  <-> Qwen-ZH  (full)"]
    if lang[2] > 0.1 and full[2] > 0.0:
        print("  -> STRONG: cross-language Procrustes positive -> meaning is invariant geometry up to ROTATION across")
        print("     languages. The universal-manifold strong claim survives under power.")
    elif lang[0] > 3 * lang[1]:
        print("  -> LOCAL-NOT-GLOBAL (confirmed under power): cross-language kNN >> chance but Procrustes ~0 -> shared")
        print("     neighbor structure transcends language, but it is NOT a clean global rotation. Same shape as the arc.")
    else:
        print("  -> weak/mixed: report straight.")
    print("\n[V] OPT-2.7B + Qwen2.5-1.5B, contextual ~2/3 depth, N=700 MUSE en-zh pairs, PCA-64, fp32.")


if __name__ == "__main__":
    run()
