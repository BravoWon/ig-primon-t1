#!/usr/bin/env python
"""v12 — weights as ADDRESSES into a LEARNED codebook (the user's "generalized address" idea).

Not fixed levels. A small LEARNED scalar codebook (the shared 'object'); each weight stores an ADDRESS
(nearest entry) + a per-tensor scale (the 'adjustment'). Codebook trained VQ-style (STE for the weight,
k-means codebook loss for the levels). Initialized from the user's {-1,0,1,3}, free to move.

  dense | quad13-fixed {-1,0,1,3} | learned4 (init {-1,0,1,3}, K=4, 2b) | learned8 (K=8, 3b)
Q: does LEARNING the address set beat the hand-picked one? does a bigger codebook (3b) help?
FIXED stable corpus (cbp_wolfcamp + TOSCO mds) so numbers don't drift.

    python native_codebook.py
"""
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK, H = 192, 6, 4, 128, 768
BATCH, ITERS, LR, CB = 24, 2500, 3e-4, 0.1


def corpus():                                   # PINNED to stable files (fixes cross-run drift)
    files = sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("TOSCO*.md"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class CodebookLinear(nn.Module):
    def __init__(self, i, o, init, learn):
        super().__init__(); self.lin = nn.Linear(i, o)
        lv = torch.tensor(init, dtype=torch.float)
        self.learn = learn
        if learn:
            self.levels = nn.Parameter(lv)
        else:
            self.register_buffer("levels", lv)          # moves with .to(DEV)
        self.cb = torch.zeros(())
    def forward(self, x):
        w = self.lin.weight; s = w.abs().mean() + 1e-5; L = self.levels
        idx = ((w / s).unsqueeze(-1) - L).abs().argmin(-1)
        recon = s * L[idx]
        if self.learn:
            self.cb = ((w.detach() - recon) ** 2).mean()       # k-means: pull levels toward weights
        return F.linear(x, w + (recon - w).detach(), self.lin.bias)   # STE for weights


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__()
        spec = {"dense": None, "quad13": ([-1, 0, 1, 3], False),
                "learned4": ([-1, 0, 1, 3], True), "learned8": ([-3, -2, -1, 0, 1, 2, 3, 4], True)}[kind]
        if spec is None:
            self.fc1, self.fc2 = nn.Linear(D, H), nn.Linear(H, D)
        else:
            self.fc1, self.fc2 = CodebookLinear(D, H, *spec), CodebookLinear(H, D, *spec)
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
        if tgt is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        cb = sum(m.cb for m in self.modules() if isinstance(m, CodebookLinear) and m.learn)
        return logits, loss + CB * cb


def main():
    train, val, V = corpus(); train, val = train.to(DEV), val.to(DEV)
    print(f"[learned codebook v12]  dev={DEV} vocab={V}  iters={ITERS}  (pinned corpus)")

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
    for kind in ("dense", "quad13", "learned4", "learned8"):
        torch.manual_seed(0)
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
        vl = evalloss(m)
        lv = None
        for mm in m.modules():
            if isinstance(mm, CodebookLinear):
                lv = [round(v, 2) for v in mm.levels.tolist()]; break
        bits = 0 if kind == "dense" else math.log2(len(MLP(kind).fc1.levels))
        out[kind] = (vl, bits, lv); print(f"  {kind:9} val {vl:.4f}  {bits:.2f} b/w  levels {lv}")

    dv = out["dense"][0]
    print(f"\nVERDICTS (vs dense {dv:.4f}, pinned corpus):")
    for k in ("quad13", "learned4", "learned8"):
        v, b, lv = out[k]
        print(f"  {k:9} {v:.4f} ({(v-dv)/dv*100:+.2f}%)  {b:.2f} b/w  learned-levels {lv}")
    qv, l4 = out["quad13"][0], out["learned4"][0]
    print(f"  learned4 vs fixed quad13: {(l4-qv)/qv*100:+.2f}%  -> "
          f"{'LEARNING the addresses helps' if l4 < qv else 'fixed {-1,0,1,3} already ~optimal (your set was good)'}")


if __name__ == "__main__":
    main()
