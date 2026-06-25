#!/usr/bin/env python
"""v8 — the LAST untested cells, to close (or flip) the native structure map.

Same tiny char-LM (fixed eval), MLP variants at EQUAL params, trained from scratch:
  dense     : Linear(d,4d)->GELU->Linear(4d,d)
  butterfly : BlockDiag(d->G*4d, G) -> GELU -> channel-shuffle -> BlockDiag(G*4d->d, G)
              (Monarch/butterfly family: block-diagonal + cross-group shuffle; STRONGER than low-rank,
               equal params).  Q: does structure beat dense at equal params?
  ternary   : dense MLP but BitLinear (weights -> {-1,0,1}*scale via straight-through, BitNet b1.58).
              Q: how much quality does NATIVE ternary cost? (its weights are ~1.58 b/w, ~10x fewer bits)

    python native_lastcells.py
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK = 192, 6, 4, 128
BATCH, ITERS, EVAL_EVERY, LR, G = 24, 2500, 500, 3e-4, 4


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class BlockDiag(nn.Module):
    def __init__(self, i, o, G):
        super().__init__(); self.G, self.o = G, o
        self.w = nn.Parameter(torch.randn(G, o // G, i // G) / (i // G) ** 0.5)
        self.b = nn.Parameter(torch.zeros(o))
    def forward(self, x):
        *pre, i = x.shape
        y = torch.einsum('ngj,goj->ngo', x.reshape(-1, self.G, i // self.G), self.w)
        return y.reshape(*pre, self.o) + self.b


def shuffle(x, G):
    *pre, n = x.shape
    return x.reshape(*pre, G, n // G).transpose(-1, -2).reshape(*pre, n)


class BitLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.lin = nn.Linear(i, o)
    def forward(self, x):
        w = self.lin.weight; scale = w.abs().mean() + 1e-5
        wq = torch.clamp(torch.round(w / scale), -1, 1) * scale
        return F.linear(x, w + (wq - w).detach(), self.lin.bias)        # straight-through ternary


class MLP(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind = kind
        if kind == "dense":
            self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        elif kind == "butterfly":
            hw = G * 4 * D; self.fc1, self.fc2 = BlockDiag(D, hw, G), BlockDiag(hw, D, G)
        else:  # ternary
            self.fc1, self.fc2 = BitLinear(D, 4 * D), BitLinear(4 * D, D)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        if self.kind == "butterfly":
            x = shuffle(x, G)
        return self.fc2(x)


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
    print(f"[last cells v8]  dev={DEV} vocab={V}  G={G}  iters={ITERS}")

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
    for kind in ("dense", "butterfly", "ternary"):
        torch.manual_seed(0)
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        mlp_p = sum(p.numel() for b in m.blocks for p in b.mlp.parameters())
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
            if it % (EVAL_EVERY * 2) == 0 or it == ITERS:
                print(f"  {kind:9} it{it:5} val {evalloss(m):.3f}")
        out[kind] = (mlp_p, evalloss(m)); print(f"  {kind:9} MLP-params {mlp_p/1e6:.3f}M  FINAL val {out[kind][1]:.4f}")

    d = out["dense"][1]
    print(f"\nVERDICTS (equal params, vs dense val {d:.4f}):")
    print(f"  butterfly (Monarch-family): {out['butterfly'][1]:.4f}  -> "
          f"{'STRUCTURE WINS' if out['butterfly'][1] < d else 'dense wins'} ({(d-out['butterfly'][1])/d*100:+.2f}%)")
    print(f"  ternary (BitNet ~1.58 b/w MLP): {out['ternary'][1]:.4f}  -> "
          f"cost {(out['ternary'][1]-d)/d*100:+.2f}% val for ~10x fewer MLP bits "
          f"{'(near-lossless!)' if (out['ternary'][1]-d)/d < 0.03 else '(real quality cost)'}")


if __name__ == "__main__":
    main()
