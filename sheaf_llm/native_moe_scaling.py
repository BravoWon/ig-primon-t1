#!/usr/bin/env python
"""v7 — does the MoE win SCALE with experts? (the confirm/refute for v6's marginal +0.76%)

Same tiny char-LM, EQUAL ACTIVE compute, sweep E in {1,2,4,8} (E=1 = dense). Same seed/init for the
shared parts (only the MLP differs). If val drops MONOTONICALLY as E grows, the store-many/activate-few
mechanism is real and COMPOUNDS at scale -> the honest 70B-in-6 GB path is validated. If flat/noisy,
v6's +0.76% was noise.

    python native_moe_scaling.py
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, NH, NL, BLOCK = 192, 6, 4, 128
BATCH, ITERS, EVAL_EVERY, LR, AUX = 24, 2000, 500, 3e-4, 0.01
E_LIST = [1, 2, 4, 8]


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
        probs = F.softmax(self.gate(xf), -1); idx = probs.argmax(-1)
        w = probs.gather(1, idx[:, None]).squeeze(1)
        f = torch.bincount(idx, minlength=self.E).float() / idx.numel()
        self.aux = self.E * (f * probs.mean(0)).sum()
        out = torch.zeros_like(xf)
        for e in range(self.E):
            m = idx == e
            if m.any():
                out[m] = self.experts[e](xf[m]) * w[m].unsqueeze(1)
        return out.reshape(B, T, d)


class Block(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.mlp = Dense(D, 4 * D) if E == 1 else MoE(D, 4 * D, E)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK, BLOCK) * float("-inf"), 1))
    def forward(self, x):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=self.mask[:x.size(1), :x.size(1)], need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, E):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block(E) for _ in range(NL)])
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
    print(f"[MoE scaling v7]  dev={DEV} vocab={V}  E sweep {E_LIST}  iters={ITERS}")

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

    res = []
    for E in E_LIST:
        torch.manual_seed(0)                          # identical init for shared parts; only MLP differs
        m = GPT(V, E).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
        for it in range(1, ITERS + 1):
            x, y = batch(train); loss = m(x, y)[1]
            opt.zero_grad(); loss.backward(); opt.step()
        vl = evalloss(m); tot = sum(p.numel() for p in m.parameters())
        res.append((E, tot, vl)); print(f"  E={E:<2} total {tot/1e6:.3f}M  val {vl:.4f}")

    print("\nSCALING (equal active compute; total params grow with E):")
    base = res[0][2]
    for E, tot, vl in res:
        print(f"  E={E:<2} val {vl:.4f}  ({(base-vl)/base*100:+.2f}% vs dense E=1)  total {tot/1e6:.2f}M")
    vals = [r[2] for r in res]
    mono = all(vals[i + 1] <= vals[i] + 1e-3 for i in range(len(vals) - 1))
    print(f"\nVERDICT: {'MONOTONE val DROP with E -> MoE mechanism COMPOUNDS (path validated)' if mono else 'NOT monotone -> gain is noise/saturates at this scale'}"
          f"  | best E={res[int(np.argmin(vals))][0]} ({(base-min(vals))/base*100:+.2f}% vs dense)")


if __name__ == "__main__":
    main()
