#!/usr/bin/env python
"""v6 — the MoE gate (the honest "70B in 6 GB": equal ACTIVE compute, far more TOTAL params).

Same tiny char-LM as v5, but the MLP variants are:
  dense : Linear(d,4d)->GELU->Linear(4d,d)
  moe   : E experts each = dense MLP(d,4d), TOP-1 routing (+ switch load-balance aux).
          Active compute/token ~= ONE expert ~= dense.  Total params ~= E x dense.

If moe val < dense val at EQUAL ACTIVE compute, then "more stored capacity, same working set" pays
-> the sheaf-as-ROUTER / stream-cold-experts mechanism is the real door (compression was dead, 6 gates).

    python native_moe_test.py
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK = 192, 6, 4, 128
BATCH, ITERS, EVAL_EVERY, LR = 24, 2500, 250, 3e-4
E, AUX = 4, 0.01


def corpus():
    files = sorted(glob.glob("*.md")) + sorted(glob.glob("cbp_wolfcamp/*.md")) + sorted(glob.glob("sheaf_llm/*.py"))
    t = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)[:2_000_000]
    chars = sorted(set(t)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in t], dtype=torch.long); n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


class Dense(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.fc1, self.fc2 = nn.Linear(d, h), nn.Linear(h, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, d, h, E):
        super().__init__(); self.gate = nn.Linear(d, E); self.experts = nn.ModuleList([Dense(d, h) for _ in range(E)])
        self.E = E; self.aux = torch.zeros((), device=DEV)
    def forward(self, x):
        B, T, d = x.shape; xf = x.reshape(-1, d)
        probs = F.softmax(self.gate(xf), -1)                  # (N,E)
        idx = probs.argmax(-1); w = probs.gather(1, idx[:, None]).squeeze(1)
        f = torch.bincount(idx, minlength=self.E).float() / idx.numel()
        self.aux = self.E * (f * probs.mean(0)).sum()          # switch load-balance aux
        out = torch.zeros_like(xf)
        for e in range(self.E):
            m = idx == e
            if m.any():
                out[m] = self.experts[e](xf[m]) * w[m].unsqueeze(1)
        return out.reshape(B, T, d)


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.mlp = Dense(D, 4 * D) if kind == "dense" else MoE(D, 4 * D, E)
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
        aux = sum(b.mlp.aux for b in self.blocks if isinstance(b.mlp, MoE))
        return logits, loss + AUX * aux


def main():
    train, val, V = corpus(); train, val = train.to(DEV), val.to(DEV)
    print(f"[MoE gate v6]  dev={DEV} vocab={V} d={D} layers={NL} E={E} top-1  iters={ITERS}")

    def batch(data):
        ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
        return (torch.stack([data[i:i + BLOCK] for i in ix]).to(DEV),
                torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix]).to(DEV))

    @torch.no_grad()
    def evalloss(m):
        m.eval(); ls = []
        for _ in range(40):
            x, y = batch(val)                       # x and y from the SAME batch (v6 bug was 2 calls)
            ls.append(F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)).item())
        m.train(); return float(np.mean(ls))

    out = {}
    for kind in ("dense", "moe"):
        m = GPT(V, kind).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        mlp_params = sum(p.numel() for b in m.blocks for p in b.mlp.parameters())
        active = mlp_params if kind == "dense" else mlp_params // E      # ~ top-1 active
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
            if it % (EVAL_EVERY * 2) == 0 or it == ITERS:
                print(f"  {kind:6} it{it:5} val {evalloss(m):.3f}")
        out[kind] = (sum(p.numel() for p in m.parameters()), active, evalloss(m))
        print(f"  {kind:6} TOTAL {out[kind][0]/1e6:.3f}M  active-MLP ~{out[kind][1]/1e6:.3f}M  FINAL val {out[kind][2]:.4f}")

    dt, mt = out["dense"], out["moe"]
    print(f"\nVERDICT (equal ACTIVE-MLP ~{dt[1]/1e6:.3f}M; MoE total {mt[0]/1e6:.2f}M vs dense {dt[0]/1e6:.2f}M):")
    print(f"  dense val {dt[2]:.4f}  |  moe val {mt[2]:.4f}  -> "
          f"{'MoE WINS — store-many/activate-few pays (the real door)' if mt[2] < dt[2] else 'MoE does NOT beat dense here'} "
          f"({(dt[2]-mt[2])/dt[2]*100:+.2f}% val)")


if __name__ == "__main__":
    main()
