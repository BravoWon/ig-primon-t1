#!/usr/bin/env python
"""v9 — "the geometry of the weights is ONE object, branched" (the cell v0-v8 never tested).

Not factoring each matrix. Instead: ONE shared base MLP (the single object / global section) used by
ALL layers, plus a cheap PER-LAYER BRANCH (restriction map) so layers are views of the one object:
  dense       : NL independent MLPs (the usual "N separate objects")               [baseline]
  shared-film : 1 base MLP + per-layer FiLM (gamma,beta on hidden)  branch ~ 2h     [tiniest branch]
  shared-lora : 1 base MLP + per-layer rank-r delta on fc1 & fc2     branch ~ 2r(d+h)
  shared-affine: 1 base MLP + per-layer diagonal restriction maps in & out (D_out W D_in) ~ d+h

Q: how close does one-object-branched get to N-independent-dense, and at what FRACTION of the params?
If shared-* ~= dense at ~30% params, the weight-geometry-is-one-object thesis is real -> scale it.

    python native_shared_object.py
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK, H = 192, 6, 4, 128, 768
BATCH, ITERS, EVAL_EVERY, LR, RANK = 24, 2500, 500, 3e-4, 8


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class Base(nn.Module):                                   # the single object, shared across layers
    def __init__(self):
        super().__init__(); self.fc1, self.fc2 = nn.Linear(D, H), nn.Linear(H, D)


class Branch(nn.Module):                                 # per-layer restriction map off the one object
    def __init__(self, base, kind):
        super().__init__(); self.base, self.kind = base, kind
        if kind == "film":
            self.g, self.b = nn.Parameter(torch.ones(H)), nn.Parameter(torch.zeros(H))
        elif kind == "lora":
            self.A1 = nn.Parameter(torch.zeros(D, RANK)); self.B1 = nn.Parameter(torch.randn(RANK, H) * 0.02)
            self.A2 = nn.Parameter(torch.zeros(H, RANK)); self.B2 = nn.Parameter(torch.randn(RANK, D) * 0.02)
        elif kind == "affine":
            self.din, self.dh, self.dout = (nn.Parameter(torch.ones(D)), nn.Parameter(torch.ones(H)),
                                            nn.Parameter(torch.ones(D)))
    def forward(self, x):
        if self.kind == "affine":
            x = x * self.din
        h = self.base.fc1(x)
        if self.kind == "film":
            h = self.g * h + self.b
        elif self.kind == "lora":
            h = h + (x @ self.A1) @ self.B1
        elif self.kind == "affine":
            h = h * self.dh
        h = F.gelu(h)
        y = self.base.fc2(h)
        if self.kind == "lora":
            y = y + (h @ self.A2) @ self.B2
        elif self.kind == "affine":
            y = y * self.dout
        return y


class DenseMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1, self.fc2 = nn.Linear(D, H), nn.Linear(H, D)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, kind, base):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.mlp = DenseMLP() if kind == "dense" else Branch(base, kind.split("-")[-1])
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), 1))
    def forward(self, x):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=self.mask[:x.size(1), :x.size(1)], need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, kind):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(BLOCK, D)
        self.base = None if kind == "dense" else Base()      # ONE object, registered once
        self.blocks = nn.ModuleList([Block(kind, self.base) for _ in range(NL)])
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
    print(f"[one-object-branched v9]  dev={DEV} vocab={V}  d={D} layers={NL}  iters={ITERS}")

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

    def mlp_params(m):
        seen, tot = set(), 0
        for b in m.blocks:
            for p in b.mlp.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); tot += p.numel()
        return tot

    out = {}
    for kind in ("dense", "shared-film", "shared-affine", "shared-lora"):
        torch.manual_seed(0)
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
            if it == ITERS:
                vl = evalloss(m)
        out[kind] = (mlp_params(m), vl); print(f"  {kind:14} MLP-params {out[kind][0]/1e6:.3f}M  val {vl:.4f}")

    dp, dv = out["dense"]
    print(f"\nVERDICTS (vs dense: {dp/1e6:.3f}M MLP params, val {dv:.4f}):")
    for kind in ("shared-film", "shared-affine", "shared-lora"):
        p, v = out[kind]
        print(f"  {kind:14} {p/1e6:.3f}M ({p/dp*100:.0f}% of dense MLP)  val {v:.4f}  "
              f"({(v-dv)/dv*100:+.2f}% vs dense)  -> "
              f"{'ONE-OBJECT MATCHES DENSE at a fraction of params' if (v-dv)/dv < 0.02 else 'gap remains'}")


if __name__ == "__main__":
    main()
