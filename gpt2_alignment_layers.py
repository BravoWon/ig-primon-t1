"""DEEPEN THE GEM: per-layer alignment curve + orthogonal Procrustes (the strict 'same up to rotation' test).

v0.2.15 showed independently-trained models align (mutual-kNN 15-22x chance, affine R^2 ~0.5). This characterizes
it with resolution: WHERE does the shared structure peak (depth curve), and HOW rigid is the map (free-linear vs
orthogonal Procrustes -- if a pure rotation relates them, the spaces are isomorphic, the strongest claim)?

Embed a shared word set at 5 relative depths in each model (one forward, grab all hidden states). Per (depth,
model-pair): mutual-kNN, free-linear held-out R^2, orthogonal-Procrustes held-out R^2 (PCA-64). OPT and GPT-2
share the GPT-2 BPE vocab, so the depth-0 (embedding) comparison is meaningful too.

PRE-REG: alignment peaks mid-network (semantics), dips at token-surface (depth 0) and task-output (depth 1);
Procrustes R^2 < free-linear R^2 (rotation is stricter) but still well above chance -> 'isomorphic up to rotation'
substantially holds. Falsifier: flat/no peak, or Procrustes ~ 0 (map needs full linear freedom, not a rotation).
[V] gpt2-small/xl + opt-2.7b, contextual hidden states at 5 depths, shared word set, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"
MODELS = [("gpt2-small", "C:/Users/JT-DEV1/Documents/gpt2-sm"),
          ("gpt2-xl", "C:/Users/JT-DEV1/Documents/gpt2-xl"),
          ("opt-2.7b", "C:/Users/JT-DEV1/Documents/opt-2.7b")]
NW = 600; K = 10; DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]; PCA_K = 64


def word_list(n):
    tok = AutoTokenizer.from_pretrained(MODELS[0][1]); words = []
    for tid in range(tok.vocab_size):
        s = tok.decode([tid])
        if s.startswith(" ") and s[1:].isalpha() and len(s) >= 4:
            words.append(s[1:])
        if len(words) >= n:
            break
    return words


def embed_layers(path, words):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    nl = m.config.num_hidden_layers
    idxs = [max(0, min(nl, int(round(d * nl)))) for d in DEPTHS]
    acc = {li: [] for li in idxs}
    for w in words:
        ids = torch.tensor([tok.encode(" " + w)], device=DEV)
        hs = m(ids, output_hidden_states=True).hidden_states
        for li in idxs:
            acc[li].append(hs[li][0, -1].float().cpu())
    out = {}
    for d, li in zip(DEPTHS, idxs):
        X = torch.stack(acc[li]); X = X - X.mean(0); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        out[d] = X
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


def _split(n, seed=0):
    idx = np.random.default_rng(seed).permutation(n); return idx[:int(.8 * n)], idx[int(.8 * n):]


def linear_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _split(n)
    A1 = torch.cat([A, torch.ones(n, 1)], 1)
    W = torch.linalg.lstsq(A1[tr], B[tr]).solution; pred = A1[te] @ W
    return float(1 - ((B[te] - pred) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr, te = _split(n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh    # orthogonal map (rotation+reflection)
    pred = A[te] @ R
    return float(1 - ((B[te] - pred) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[DEEPEN THE GEM]  per-layer alignment + orthogonal Procrustes (strict isomorphism test)\n")
    words = word_list(NW)
    E = {}
    for name, path in MODELS:
        E[name] = embed_layers(path, words)
        print(f"  embedded {len(words)} words x {len(DEPTHS)} depths in {name}")
    print()

    names = [n for n, _ in MODELS]
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    for a, b in pairs:
        print(f"  [{a} <-> {b}]   {'depth':>6} {'mutual-kNN':>11} {'linear-R2':>10} {'procrustes-R2':>14}")
        for d in DEPTHS:
            A, B = E[a][d], E[b][d]
            print(f"  {'':>{len(a)+len(b)+10}} {d:>6.2f} {mutual_knn(A, B, K):>11.3f} {linear_r2(A, B):>10.3f} {procrustes_r2(A, B):>14.3f}")
        print()

    print("[READING]")
    print("  - depth of PEAK alignment = where the shared 'platonic' structure concentrates (expect mid-network).")
    print("  - procrustes-R2 vs linear-R2: if procrustes stays close to linear, the map is ~a ROTATION -> the")
    print("    spaces are isomorphic up to orthogonal transform (the strongest form). A big gap = the map needs")
    print("    real linear freedom (stretch/shear), weaker claim.")
    print("\n[V] gpt2-small/xl + opt-2.7b, contextual hidden states at 5 depths, shared word set, fp32.")


if __name__ == "__main__":
    run()
