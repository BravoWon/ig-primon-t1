#!/usr/bin/env python
"""structure-vs-bits v0 — the first falsifiable test of "restriction maps not weights".

For real transformer weight matrices, compare reconstruction quality at EQUAL bytes for:
  * rtn4      : per-group symmetric 4-bit round-to-nearest (the "old way" baseline)
  * lowrank   : global SVD truncation (the simplest structure)
  * sheaf     : block / shared-transport factorization (restriction maps reused across blocks)

The sheaf compressor reshapes W into a grid of a*b blocks and represents every block as a
combination of K SHARED basis transport maps (atoms) -> "store the restriction maps + which
block uses them, not the dense weights". If `sheaf` beats `lowrank` at matched bits/param on
real weights, cross-block transport structure carries compressibility that per-weight bits and
flat low-rank miss -> first evidence for the 70B-in-6GB direction. If not, we learned it cheap.

    python compress_v0.py [hf_model_id]
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_mats(model, k=6, min_dim=512):
    mats = [(n, p.detach().float()) for n, p in model.named_parameters()
            if p.ndim == 2 and min(p.shape) >= min_dim
            and any(t in n for t in ("mlp", "fc", "proj"))]
    if not mats:
        mats = [(n, p.detach().float()) for n, p in model.named_parameters()
                if p.ndim == 2 and min(p.shape) >= min_dim]
    idx = np.linspace(0, len(mats) - 1, min(k, len(mats))).astype(int)
    return [mats[i] for i in idx]


def relerr(W, Wh):
    return (torch.norm(W - Wh) / torch.norm(W)).item()


def rtn4(W, group=128):
    m, n = W.shape
    Wh = W.clone(); qmax = 7
    g = min(group, n)
    for j0 in range(0, n, g):
        blk = W[:, j0:j0 + g]
        s = blk.abs().amax(dim=1, keepdim=True) / qmax + 1e-12
        Wh[:, j0:j0 + g] = torch.clamp(torch.round(blk / s), -qmax, qmax) * s
    return Wh, 4 + 16.0 / g                      # 4 bits + fp16 group scale per row


def lowrank(W, bpp):
    m, n = W.shape
    r = max(1, round(bpp * m * n / (16 * (m + n))))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(r, S.numel())
    Wh = (U[:, :r] * S[:r]) @ Vh[:r]
    return Wh, 16.0 * r * (m + n) / (m * n), r


def sheaf_block(W, bpp, a=64, b=64):
    m, n = W.shape
    a, b = min(a, m), min(b, n)
    p, q = m // a, n // b
    Wc = W[:p * a, :q * b]
    blocks = Wc.reshape(p, a, q, b).permute(0, 2, 1, 3).reshape(p * q, a * b)   # (pq, ab)
    K = max(1, round(bpp * (p * a) * (q * b) / (16 * (a * b + p * q))))
    U, S, Vh = torch.linalg.svd(blocks, full_matrices=False)
    K = min(K, S.numel())
    Bh = (U[:, :K] * S[:K]) @ Vh[:K]                                            # K shared atoms
    Wh = Bh.reshape(p, q, a, b).permute(0, 2, 1, 3).reshape(p * a, q * b)
    full = W.clone(); full[:p * a, :q * b] = Wh
    return full, 16.0 * K * (a * b + p * q) / ((p * a) * (q * b)), K


def main():
    print(f"[structure-vs-bits v0]  model={MODEL}  dev={DEV}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    mats = load_mats(model)
    print(f"sampled {len(mats)} weight matrices: "
          + ", ".join(f"{n.split('.')[-2]}{tuple(W.shape)}" for n, W in mats[:6]) + "\n")
    targets = [2.0, 3.0, 4.0]
    rows = {}
    for name, W in mats:
        W = W.to(DEV)
        wh, b4 = rtn4(W); rows.setdefault(("rtn4", round(b4, 2)), []).append(relerr(W, wh))
        for t in targets:
            wl, bl, r = lowrank(W, t)
            ws, bs, K = sheaf_block(W, t)
            rows.setdefault(("lowrank", t), []).append(relerr(W, wl))
            rows.setdefault(("sheaf", t), []).append(relerr(W, ws))

    print(f"{'method':9}{'bits/param':>11}{'rel-err':>10}   (mean over matrices, lower=better)")
    for key in sorted(rows, key=lambda k: (k[1], k[0])):
        meth, b = key
        print(f"{meth:9}{b:>11.2f}{np.mean(rows[key]):>10.4f}")

    print("\nGATE (at each matched bits/param):  does sheaf beat lowrank AND rtn4?")
    for t in targets:
        el = np.mean(rows[("lowrank", t)]); es = np.mean(rows[("sheaf", t)])
        win = "SHEAF WINS" if es < el else "lowrank wins"
        print(f"  {t:.0f} b/param:  sheaf {es:.4f}  vs  lowrank {el:.4f}   -> {win} "
              f"({(el-es)/el*100:+.1f}% vs lowrank)")
    b4key = next(k for k in rows if k[0] == "rtn4")
    print(f"  reference: rtn4 @ {b4key[1]:.2f} b/param = {np.mean(rows[b4key]):.4f}")


if __name__ == "__main__":
    main()
