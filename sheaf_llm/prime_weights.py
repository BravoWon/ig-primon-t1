#!/usr/bin/env python
"""v10 — PRIME weights {0,1,2} = Void/Identity/Prime (the Crystal encoding), native-trained.

3 levels = 1.58 bits, same as symmetric ternary; {0,1,2} is affine-equivalent to {-1,0,+1}
(offset folds into bias). So we expect prime ~= ternary ~= near-dense. The REAL number to extract is
the SKIPPABLE-state fraction (Void/Identity ~ zero-compute weights -> sparsity on top of 1.58 bits).

Variants (MLP linears, native straight-through): dense | ternary{-1,0,1} | prime{0,1,2}.
Report val (confirm near-lossless) + level distribution (the sparsity prize).

    python prime_weights.py
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK, H = 192, 6, 4, 128, 768
BATCH, ITERS, EVAL_EVERY, LR = 24, 2500, 500, 3e-4


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class TernLinear(nn.Module):                       # {-1,0,+1}: skip-state = 0
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)
    def _q(self):
        w = self.lin.weight; s = w.abs().mean() + 1e-5
        return torch.clamp(torch.round(w / s), -1, 1), s
    def forward(self, x):
        q, s = self._q(); w = self.lin.weight
        return F.linear(x, w + (s * q - w).detach(), self.lin.bias)
    def levels(self):
        q, _ = self._q()
        return {f"v={v:+d}": (q == v).float().mean().item() for v in (-1, 0, 1)}


class PrimeLinear(nn.Module):                      # {0,1,2} Void/Identity/Prime; skip-state = Identity(1)
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)
    def _q(self):
        w = self.lin.weight; s = w.abs().mean() + 1e-5
        return torch.clamp(torch.round(w / s) + 1, 0, 2), s        # {0,1,2}
    def forward(self, x):
        q, s = self._q(); w = self.lin.weight
        w_eff = s * (q - 1)                                        # recenter -> ternary-equivalent compute
        return F.linear(x, w + (w_eff - w).detach(), self.lin.bias)
    def levels(self):
        q, _ = self._q()
        return {"Void(0)": (q == 0).float().mean().item(), "Ident(1)": (q == 1).float().mean().item(),
                "Prime(2)": (q == 2).float().mean().item()}


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        L = {"dense": nn.Linear, "ternary": TernLinear, "prime": PrimeLinear}[kind]
        self.fc1, self.fc2 = L(D, H), L(H, D)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True); self.mlp = MLP(kind)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), 1))
    def forward(self, x):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=self.mask[:x.size(1), :x.size(1)], need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, kind):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block(kind) for _ in range(NL)])
        self.lnf, self.head = nn.LayerNorm(D), nn.Linear(D, V)
    def forward(self, idx, tgt=None):
        T = idx.size(1); x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None if tgt is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return logits, loss


def main():
    train, val, V = corpus(); train, val = train.to(DEV), val.to(DEV)
    print(f"[prime weights v10]  dev={DEV} vocab={V}  iters={ITERS}")

    def batch(data):
        ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
        return (torch.stack([data[i:i + BLOCK] for i in ix]).to(DEV),
                torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix]).to(DEV))

    @torch.no_grad()
    def evalloss(m):
        m.eval(); ls = []
        for _ in range(40):
            x, y = batch(val); ls.append(F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)).item())
        m.train(); return float(np.mean(ls))

    out = {}
    for kind in ("dense", "ternary", "prime"):
        torch.manual_seed(0)
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
            if it == ITERS:
                vl = evalloss(m)
        lv = None
        for b in m.blocks:
            if hasattr(b.mlp.fc1, "levels"):
                d = b.mlp.fc1.levels()
                lv = {k: lv[k] + d[k] if lv else d[k] for k in d} if lv else dict(d)
        if lv:
            lv = {k: v / NL for k, v in lv.items()}
        out[kind] = (vl, lv); print(f"  {kind:8} val {vl:.4f}   levels {lv}")

    dv = out["dense"][0]
    print(f"\nVERDICTS (vs dense val {dv:.4f}):")
    for k in ("ternary", "prime"):
        v, lv = out[k]
        skip = lv.get("v=+0", lv.get("Ident(1)", 0.0))
        print(f"  {k:8} {v:.4f} ({(v-dv)/dv*100:+.2f}% vs dense, ~1.58 b/w)  skippable-state {skip*100:.0f}%")
    pv, tv = out["prime"][0], out["ternary"][0]
    print(f"  prime vs ternary: {pv:.4f} vs {tv:.4f} ({(pv-tv)/tv*100:+.2f}%) -> "
          f"{'EQUIVALENT (as predicted: 3 levels = 3 levels)' if abs(pv-tv)/tv < 0.02 else 'differ'}")


if __name__ == "__main__":
    main()
