"""MULTI-LANGUAGE depth sweep: is the upper-middle 'universal manifold' placement universal across families?

v0.2.20 placed cross-lingual (EN<->ZH) meaning in Qwen's upper-middle layers. This tests universality + typological
distance: EN paired with Spanish (IE, Latin, close), Russian (IE, Cyrillic), Chinese (Sino-Tibetan, logographic),
Arabic (Afro-Asiatic, RTL). One multilingual model (Qwen2.5-1.5B) embeds all; per language, the EN<->X geometric
alignment (Procrustes-R^2) and direct retrieval (P@1) are tracked across 8 depths.

PRE-REG: if the placement is a real property, the PEAK DEPTH is consistent (upper-middle) across all four. The
MAGNITUDE tests distance: classic typology predicts es>ru>zh~ar; but Qwen is Chinese-strong, so alignment may
track TRAINING REPRESENTATION (how much of each language Qwen saw) more than family distance -- genuinely open.
Falsifier for universality: peak depth varies wildly by language.
[V] Qwen2.5-1.5B, mean-pooled hidden states at 8 depths, ~400 OPUS-100 pairs per language, fp32.
"""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
DEV = "cuda:0"; QWEN = "Qwen/Qwen2.5-1.5B"
LANGS = ["es", "ru", "zh", "ar"]; N = 400; K = 10; PCA_K = 32; MAXTOK = 64
DEPTHS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def load_pairs(lang, n):
    from datasets import load_dataset
    cfg = "-".join(sorted(["en", lang]))                            # OPUS-100 orders pairs alphabetically (ar-en, en-es, ...)
    ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="test")
    en, fo = [], []
    for r in ds:
        e = r["translation"]["en"].strip(); f = r["translation"][lang].strip()
        if 5 <= len(e.split()) <= 25 and 4 <= len(f) <= 80:
            en.append(e); fo.append(f)
        if len(en) >= n:
            break
    return en, fo


def embed_depths(m, tok, sents):
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
    return out


def _pca(X, k):
    Xc = X - X.mean(0); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); return Xc @ Vh[:k].t()


def p_at1(A, B):
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9); B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return float((A @ B.t()).argmax(1).eq(torch.arange(A.shape[0])).float().mean())


def procrustes_r2(A, B):
    A = _pca(A, PCA_K); B = _pca(B, PCA_K); n = A.shape[0]; tr = np.arange(int(.8 * n)); te = np.arange(int(.8 * n), n)
    M = A[tr].t() @ B[tr]; U, S, Vh = torch.linalg.svd(M); R = U @ Vh
    return float(1 - ((B[te] - A[te] @ R) ** 2).sum() / ((B[te] - B[tr].mean(0)) ** 2).sum())


def run():
    print("[MULTI-LANGUAGE DEPTH SWEEP]  is the upper-middle manifold placement universal across families?\n")
    tok = AutoTokenizer.from_pretrained(QWEN)
    m = AutoModelForCausalLM.from_pretrained(QWEN).to(torch.float32).to(DEV).eval()
    data = {}
    for lg in LANGS:
        en, fo = load_pairs(lg, N)
        data[lg] = (embed_depths(m, tok, en), embed_depths(m, tok, fo), len(en))
        print(f"  embedded en-{lg}: {len(en)} pairs")
    del m; gc.collect(); torch.cuda.empty_cache()
    print()

    print("  Procrustes-R2 (EN <-> X geometric alignment) across depth:")
    print(f"    {'depth':>6} " + "".join(f"{lg:>9}" for lg in LANGS))
    pr = {lg: [] for lg in LANGS}
    for d in DEPTHS:
        row = []
        for lg in LANGS:
            en_e, fo_e, _ = data[lg]; v = procrustes_r2(en_e[d], fo_e[d]); pr[lg].append(v); row.append(v)
        print(f"    {d:>6.2f} " + "".join(f"{v:>9.3f}" for v in row))

    print("\n  P@1 (direct EN->X retrieval) across depth:")
    print(f"    {'depth':>6} " + "".join(f"{lg:>9}" for lg in LANGS))
    for d in DEPTHS:
        row = [p_at1(data[lg][0][d], data[lg][1][d]) for lg in LANGS]
        print(f"    {d:>6.2f} " + "".join(f"{v:>9.3f}" for v in row))

    print("\n[VERDICT]")
    print(f"    {'lang':>6} {'peak-depth':>11} {'peak-Procr':>11}")
    peaks = []
    for lg in LANGS:
        i = int(np.argmax(pr[lg])); peaks.append(DEPTHS[i])
        print(f"    {lg:>6} {DEPTHS[i]:>11.2f} {pr[lg][i]:>11.3f}")
    spread = max(peaks) - min(peaks)
    print(f"\n  peak-depth spread across languages = {spread:.2f}")
    print("  -> small spread => the upper-middle placement is UNIVERSAL across families (location is a real property).")
    print("  -> magnitude ordering reveals whether alignment tracks typological distance or training representation.")
    print("\n[V] Qwen2.5-1.5B, mean-pooled at 8 depths, ~400 OPUS-100 pairs/lang, PCA-32, fp32.")


if __name__ == "__main__":
    run()
