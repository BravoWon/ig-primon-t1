#!/usr/bin/env python
"""v5 — the NATIVE-training gate (the only door left after the post-hoc quadrant closed).

Train tiny char-LMs FROM SCRATCH on the repo corpus, identical except the MLP block, at EQUAL params:
  dense       : Linear(d, 4d) -> GELU -> Linear(4d, d)
  lowrank-wide: 2x wider hidden (8d) but each projection is RANK-r transport, r set so params match
                -> "restriction maps not weights": trade dense capacity for structured wider capacity.

Verdict: at equal params, does structure-when-TRAINED-IN reach <= dense val loss? If yes, the strong
"restriction maps ARE the parameters" thesis lives natively (post-hoc was dead). If dense wins, it's
dead even native (for this structure family).

    python native_structure_test.py
"""
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK = 192, 6, 4, 128
BATCH, ITERS, EVAL_EVERY, LR = 24, 2500, 250, 3e-4


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class LowRankLinear(nn.Module):
    def __init__(self, i, o, r):
        super().__init__(); self.A = nn.Linear(i, r, bias=False); self.B = nn.Linear(r, o)
    def forward(self, x): return self.B(self.A(x))


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "dense":
            h = 4 * D; self.fc1, self.fc2 = nn.Linear(D, h), nn.Linear(h, D)
        else:
            h = 8 * D; r = round(D * (4 * D) / (D + h))            # match 2*D*4D params
            self.fc1, self.fc2 = LowRankLinear(D, h, r), LowRankLinear(h, D, r)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.mlp = MLP(kind)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), 1))
    def forward(self, x):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=self.mask[:x.size(1), :x.size(1)], need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, kind):
        super().__init__()
        self.tok = nn.Embedding(V, D); self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block(kind) for _ in range(NL)])
        self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, V)
    def forward(self, idx, tgt=None):
        T = idx.size(1); x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None if tgt is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return logits, loss


def main():
    train, val, V = corpus()
    train, val = train.to(DEV), val.to(DEV)
    print(f"[native structure gate v5]  dev={DEV}  vocab={V}  d={D} layers={NL}  iters={ITERS}")

    def batch(data):
        ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
        x = torch.stack([data[i:i + BLOCK] for i in ix]); y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
        return x.to(DEV), y.to(DEV)

    @torch.no_grad()
    def evalloss(m):
        m.eval(); ls = []
        for _ in range(40):
            x, y = batch(val); ls.append(m(x, y)[1].item())
        m.train(); return float(np.mean(ls))

    out = {}
    for kind in ("dense", "lowrank-wide"):
        m = GPT(V, kind).to(DEV)
        params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
            if it % EVAL_EVERY == 0 or it == ITERS:
                vl = evalloss(m)
                if it == ITERS or it % (EVAL_EVERY * 2) == 0:
                    print(f"  {kind:13} it{it:5} train {loss.item():.3f}  val {vl:.3f}")
        out[kind] = (params, evalloss(m))
        print(f"  {kind:13} PARAMS {params/1e6:.3f}M  FINAL val {out[kind][1]:.4f}")

    pd, ld = out["dense"], out["lowrank-wide"]
    print(f"\nVERDICT (equal params {pd[0]/1e6:.3f}M vs {ld[0]/1e6:.3f}M):")
    print(f"  dense val {pd[1]:.4f}  |  lowrank-wide val {ld[1]:.4f}  -> "
          f"{'STRUCTURE WINS (native thesis lives)' if ld[1] < pd[1] else 'dense wins (structure loses even native)'} "
          f"({(pd[1]-ld[1])/pd[1]*100:+.2f}% val loss)")


if __name__ == "__main__":
    main()
