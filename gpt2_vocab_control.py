"""VOCAB-CONFOUND CONTROL: is the deep cross-model alignment genuine SEMANTIC convergence, or shared-tokenizer?

v0.2.16 flagged that OPT & GPT-2 share the GPT-2 BPE vocab, so surface (depth-0) alignment is inflated by shared
token statistics. This isolates the genuine signal: from each model's DEEP representation, partial out the part
linearly predictable from (a) that model's OWN surface embeddings, and (b) token frequency/length features. Then
re-measure cross-model alignment on the RESIDUALS. If alignment survives partialling -> the deep layers converge
in MEANING beyond shared vocab. If it collapses to chance -> the alignment was just propagated input embeddings.

PRE-REG: alignment drops after partialling but stays ABOVE chance (real but modest semantic convergence).
Falsifier: residual alignment ~ shuffled chance -> deep alignment was a vocab artifact all the way down.
[V] gpt2-small/xl + opt-2.7b, depths {0.75 interior, 1.0 output}, partial-out surface + frequency, fp32.
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
NW = 600; K = 10; DEPTHS = [0.0, 0.75, 1.0]; PCA_K = 64


def word_data(n):
    tok = AutoTokenizer.from_pretrained(MODELS[0][1]); words = []; feats = []
    for tid in range(tok.vocab_size):
        s = tok.decode([tid])
        if s.startswith(" ") and s[1:].isalpha() and len(s) >= 4:
            ids = tok.encode(" " + s[1:])
            words.append(s[1:]); feats.append([np.log(ids[0] + 1.0), float(len(ids)), float(len(s) - 1)])
        if len(words) >= n:
            break
    return words, torch.tensor(feats)                              # [log(first_id), n_subtok, char_len]


def embed_layers(path, words):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path).to(torch.float32).to(DEV).eval()
    nl = m.config.num_hidden_layers; idxs = [int(round(d * nl)) for d in DEPTHS]
    acc = {li: [] for li in idxs}
    for w in words:
        ids = torch.tensor([tok.encode(" " + w)], device=DEV)
        hs = m(ids, output_hidden_states=True).hidden_states
        for li in idxs:
            acc[li].append(hs[li][0, -1].float().cpu())
    out = {d: torch.stack(acc[li]) for d, li in zip(DEPTHS, idxs)}
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:k].t()


def _norm(X):
    X = X - X.mean(0); return X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)


def partial_out(X, C):
    C = C.to(X.dtype)
    C1 = torch.cat([C, torch.ones(C.shape[0], 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(C1, X).solution
    return X - C1 @ W


def mutual_knn(A, B, k):
    A = _norm(A); B = _norm(B); sa = A @ A.t(); sb = B @ B.t(); n = A.shape[0]
    sa.fill_diagonal_(-2); sb.fill_diagonal_(-2)
    na = sa.topk(k, 1).indices; nb = sb.topk(k, 1).indices
    return float(np.mean([len(set(na[i].tolist()) & set(nb[i].tolist())) / k for i in range(n)]))


def linear_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]
    tr = np.arange(int(.8 * n)); te = np.arange(int(.8 * n), n)
    A1 = torch.cat([A, torch.ones(n, 1)], 1)
    W = torch.linalg.lstsq(A1[tr], B[tr]).solution; pred = A1[te] @ W
    return float(1 - ((B[te] - pred) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[VOCAB-CONFOUND CONTROL]  does deep cross-model alignment survive partialling out the surface?\n")
    words, feats = word_data(NW)
    E = {n: embed_layers(p, words) for n, p in MODELS}
    print(f"  embedded {len(words)} words in {len(MODELS)} models\n")
    rng = np.random.default_rng(0); perm = torch.tensor(rng.permutation(len(words)))

    for a, b in [("gpt2-xl", "opt-2.7b"), ("gpt2-small", "opt-2.7b")]:
        print(f"  [{a} <-> {b}]")
        for d in [0.75, 1.0]:
            A, B = E[a][d], E[b][d]
            # surface partial: remove each model's own depth-0 (PCA-64) component
            As = partial_out(A, _pca(E[a][0.0], PCA_K)); Bs = partial_out(B, _pca(E[b][0.0], PCA_K))
            # frequency partial: remove [logfreq, ntok, len]
            Af = partial_out(A, feats); Bf = partial_out(B, feats)
            print(f"    depth {d:>4}  {'metric':>10} {'raw':>8} {'freq-part':>10} {'surf-part':>10} {'shuffled':>9}")
            print(f"    {'':>10}  {'mutual-kNN':>10} {mutual_knn(A,B,K):>8.3f} {mutual_knn(Af,Bf,K):>10.3f} "
                  f"{mutual_knn(As,Bs,K):>10.3f} {mutual_knn(A,B[perm],K):>9.3f}")
            print(f"    {'':>10}  {'linear-R2':>10} {linear_r2(A,B):>8.3f} {linear_r2(Af,Bf):>10.3f} "
                  f"{linear_r2(As,Bs):>10.3f} {linear_r2(A,B[perm]):>9.3f}")
        print()

    print("[READING]")
    print("  surf-part = deep alignment with each model's OWN input-embedding subspace removed. If it stays well")
    print("  above shuffled, the deep layers converge in MEANING beyond shared tokenizer. If it collapses toward")
    print("  shuffled, the cross-model alignment was propagated vocab structure, not semantics.")
    print("\n[V] gpt2-small/xl + opt-2.7b, depths {0.75,1.0}, partial-out surface(PCA-64)+frequency, fp32.")


if __name__ == "__main__":
    run()
