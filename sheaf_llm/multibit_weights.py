#!/usr/bin/env python
"""v11 — arbitrary native low-bit level sets. Tests the user's {-1,0,1,3} (2-bit, outlier-aware).

dense | ternary {-1,0,1} (1.58b) | quad13 {-1,0,1,3} (2b, asymmetric: keeps 0 + outlier level)
       | quadsym {-3,-1,1,3} (2b, symmetric, no zero/no sparsity)
Native straight-through. Report val + level USAGE (how often the outlier '3' fires) + skip-state %.

    python multibit_weights.py
"""
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK, H = 192, 6, 4, 128, 768
BATCH, ITERS, LR = 24, 2500, 3e-4
SETS = {"ternary": [-1, 0, 1], "quad13": [-1, 0, 1, 3], "quadsym": [-3, -1, 1, 3]}


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class LevelLinear(nn.Module):
    def __init__(self, i, o, levels):
        super().__init__(); self.lin = nn.Linear(i, o)
        self.register_buffer("L", torch.tensor(levels, dtype=torch.float))
    def _q(self):
        w = self.lin.weight; s = w.abs().mean() + 1e-5
        idx = ((w / s).unsqueeze(-1) - self.L).abs().argmin(-1)
        return self.L[idx], s
    def forward(self, x):
        q, s = self._q(); w = self.lin.weight
        return F.linear(x, w + (s * q - w).detach(), self.lin.bias)
    def usage(self):
        q, _ = self._q()
        return {float(v): round((q == v).float().mean().item(), 3) for v in self.L.tolist()}


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "dense":
            self.fc1, self.fc2 = nn.Linear(D, H), nn.Linear(H, D)
        else:
            self.fc1, self.fc2 = LevelLinear(D, H, SETS[kind]), LevelLinear(H, D, SETS[kind])
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
    print(f"[multibit v11]  dev={DEV} vocab={V}  iters={ITERS}")

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
    for kind in ("dense", "ternary", "quad13", "quadsym"):
        torch.manual_seed(0)
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
        vl = evalloss(m); us = m.blocks[0].mlp.fc1.usage() if kind != "dense" else None
        bits = 0 if kind == "dense" else math.log2(len(SETS[kind]))
        out[kind] = (vl, bits, us); print(f"  {kind:8} val {vl:.4f}  {bits:.2f} b/w  usage {us}")

    dv = out["dense"][0]
    print(f"\nVERDICTS (vs dense {dv:.4f}):")
    for k in ("ternary", "quad13", "quadsym"):
        v, b, us = out[k]
        skip = us.get(0.0, 0.0)
        print(f"  {k:8} {v:.4f} ({(v-dv)/dv*100:+.2f}%)  {b:.2f} b/w  skip-state {skip*100:.0f}%"
              + (f"  outlier(3) used {us.get(3.0,0)*100:.1f}%" if 3.0 in us else ""))
    print(f"  quad13 vs ternary: {(out['quad13'][0]-out['ternary'][0])/out['ternary'][0]*100:+.2f}%  "
          f"(2-bit headroom shows at SCALE, not toy where ternary is already lossless)")
    print(f"  asym quad13 vs sym quadsym: {(out['quad13'][0]-out['quadsym'][0])/out['quadsym'][0]*100:+.2f}% "
          f"(quad13 keeps 0=sparsity; quadsym has none)")


if __name__ == "__main__":
    main()
