#!/usr/bin/env python
"""semantic gate v4 -- CLEAN diagnostic: does the HUMAN ROUTE (cross-task sharing) help discovery?

v3 confounded multi-task with a harder FiLM architecture. Here we use v2's EXACT clean architecture
(per-dimension, task-conditioned heads; shared codebook) and only vary single- vs multi-task, plus a
correctly-reformulated glue (codebook USAGE-ENTROPY = anti-collapse; stabilizes the dictionary without
strangling task-specificity -- the exact v3 failure mode, fixed). Same compositional held-out probe.

  flat        : opaque emb + task -> MLP                         (lower bound)
  st-VQ       : shared codebook, SINGLE task (the v2 reference)
  mt-VQ       : shared codebook, MULTI task  (the human route, cleanly isolated)
  mt-VQ-glue  : mt-VQ + usage-entropy glue   (stabilized dictionary)
  handed      : given (a,b), task-conditioned (upper bound)
Q: mt-VQ vs st-VQ -- does sharing across K tasks make the dictionary discoverable from FEWER pairs?

    python semantic_gate_v4_clean.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, D, HID, DT, STEPS, BS, LR = 8, 64, 128, 16, 4000, 256, 1e-3
G = D // 2
TASKS = [(1, 1), (1, 3), (2, 1), (1, 7)]
FRACS = [0.05, 0.08, 0.12, 0.20, 0.35]
RED, TEAL, AMBER, GREEN, NAVY = "#c0392b", "#2c6fa6", "#9a6a2f", "#1e7d34", "#15293f"


def expand(pl, tasks):
    ks, T1, T2, YA, YB = [], [], [], [], []
    for (t1, t2) in pl:
        a1, b1 = divmod(t1, N); a2, b2 = divmod(t2, N)
        for k, (al, be) in enumerate(tasks):
            ks.append(k); T1.append(t1); T2.append(t2)
            YA.append((al * a1 + be * a2) % N); YB.append((al * b1 + be * b2) % N)
    tt = lambda z: torch.tensor(z, device=DEV)
    return tt(ks), tt(T1), tt(T2), tt(YA), tt(YB)


def build(frac, train_tasks, seed=0):
    pairs = [(t1, t2) for t1 in range(N * N) for t2 in range(N * N)]
    idx = np.random.default_rng(seed).permutation(len(pairs))
    ntr = int(frac * len(pairs)); trp = [pairs[i] for i in idx[:ntr]]; tep = [pairs[i] for i in idx[ntr:]]
    return expand(trp, train_tasks), expand(tep, [TASKS[0]])      # ALL arms tested on the SAME (1,1) held-out


def quant(x, cb):
    idx = (x.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1).argmin(-1); q = cb[idx]
    vq = ((x.detach() - q) ** 2).mean() + 0.25 * ((x - q.detach()) ** 2).mean()
    return x + (q - x).detach(), vq, idx


class Flat(nn.Module):
    def __init__(self, nk):
        super().__init__(); self.et, self.ek = nn.Embedding(N * N, D), nn.Embedding(nk, DT)
        self.mlp = nn.Sequential(nn.Linear(2 * D + DT, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.ha, self.hb = nn.Linear(HID, N), nn.Linear(HID, N)
    def forward(self, k, t1, t2):
        h = self.mlp(torch.cat([self.et(t1), self.et(t2), self.ek(k)], -1)); return self.ha(h), self.hb(h), 0.0


class VQ(nn.Module):                                 # v2 clean architecture + task-conditioned heads
    def __init__(self, nk, glue=False):
        super().__init__(); self.glue = glue; self.et, self.ek = nn.Embedding(N * N, D), nn.Embedding(nk, DT)
        self.cb_a, self.cb_b = nn.Parameter(torch.randn(N, G) * 0.5), nn.Parameter(torch.randn(N, G) * 0.5)
        mk = lambda: nn.Sequential(nn.Linear(2 * G + DT, HID), nn.GELU(), nn.Linear(HID, N)); self.ha, self.hb = mk(), mk()
    def forward(self, k, t1, t2):
        e1, e2 = self.et(t1), self.et(t2); tk = self.ek(k)
        qa1, la1, ia1 = quant(e1[:, :G], self.cb_a); qa2, la2, ia2 = quant(e2[:, :G], self.cb_a)
        qb1, lb1, ib1 = quant(e1[:, G:], self.cb_b); qb2, lb2, ib2 = quant(e2[:, G:], self.cb_b)
        oa = self.ha(torch.cat([qa1, qa2, tk], -1)); ob = self.hb(torch.cat([qb1, qb2, tk], -1))
        aux = 0.1 * (la1 + la2 + lb1 + lb2)
        if self.glue:                                # usage-entropy: keep all codes alive (anti-collapse)
            def H(idx):
                p = torch.bincount(idx, minlength=N).float(); p = p / p.sum().clamp_min(1)
                return -(p * (p + 1e-9).log()).sum()
            aux = aux - 0.02 * (H(torch.cat([ia1, ia2])) + H(torch.cat([ib1, ib2])))
        return oa, ob, aux


class Handed(nn.Module):
    def __init__(self, nk):
        super().__init__(); self.ea, self.eb, self.ek = nn.Embedding(N, G), nn.Embedding(N, G), nn.Embedding(nk, DT)
        mk = lambda: nn.Sequential(nn.Linear(2 * G + DT, HID), nn.GELU(), nn.Linear(HID, N)); self.ha, self.hb = mk(), mk()
    def forward(self, k, t1, t2):
        a1, b1, a2, b2 = t1 // N, t1 % N, t2 // N, t2 % N; tk = self.ek(k)
        oa = self.ha(torch.cat([self.ea(a1), self.ea(a2), tk], -1))
        ob = self.hb(torch.cat([self.eb(b1), self.eb(b2), tk], -1)); return oa, ob, 0.0


def acc(m, d):
    k, t1, t2, ya, yb = d
    with torch.no_grad():
        oa, ob, _ = m(k, t1, t2); return ((oa.argmax(-1) == ya) & (ob.argmax(-1) == yb)).float().mean().item()


def train(factory, tr, te):
    torch.manual_seed(0); m = factory().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
    k, t1, t2, ya, yb = tr
    for _ in range(STEPS):
        i = torch.randint(0, len(k), (BS,), device=DEV)
        oa, ob, aux = m(k[i], t1[i], t2[i])
        (F.cross_entropy(oa, ya[i]) + F.cross_entropy(ob, yb[i]) + aux).backward()
        opt.step(); opt.zero_grad()
    return acc(m, te)


def main():
    print(f"[clean diagnostic v4]  dev={DEV}  single vs multi-task discovery  (chance={1/(N*N):.3f})")
    ST = [TASKS[0]]                                   # single-task = just the (1,1) addition (v2)
    arms = [("flat", lambda: Flat(len(TASKS)), TASKS, RED),
            ("st-VQ", lambda: VQ(1), ST, TEAL),
            ("mt-VQ", lambda: VQ(len(TASKS)), TASKS, AMBER),
            ("mt-VQ-glue", lambda: VQ(len(TASKS), glue=True), TASKS, GREEN),
            ("handed", lambda: Handed(len(TASKS)), TASKS, NAVY)]
    curves = {n: [] for n, _, _, _ in arms}
    for frac in FRACS:
        line = f"  frac {frac:.2f} ({int(frac*4096):4d} pairs):"
        for n, fac, tasks, _ in arms:
            tr, te = build(frac, tasks); a = train(fac, tr, te); curves[n].append(a); line += f"  {n}={a:.3f}"
        print(line)
    print("\nVERDICT (OOS by pairs-fraction; metric is per-pair compositional generalization):")
    for n, _, _, _ in arms:
        print(f"  {n:11} " + " ".join(f"{a:.2f}" for a in curves[n]))
    st, mt = curves["st-VQ"], curves["mt-VQ"]
    print(f"  HUMAN ROUTE (mt-VQ vs st-VQ): at frac=0.12 {mt[2]:.2f} vs {st[2]:.2f}; max {max(mt):.2f} vs {max(st):.2f} -> "
          f"{'multi-task HELPS discovery' if max(mt) > max(st) + 0.05 else ('multi-task HURTS' if max(mt) < max(st)-0.05 else 'no clean difference')}")

    fig, ax = plt.subplots(figsize=(7.8, 5))
    for n, _, _, c in arms:
        ax.plot([fr * 4096 for fr in FRACS], curves[n], "-o", color=c, lw=2, label=n)
    ax.axhline(1 / (N * N), color="#999", ls=":", lw=1, label="chance")
    ax.set_xlabel("training PAIRS (held-out combos novel)"); ax.set_ylabel("OOS acc on novel combinations")
    ax.set_title("Clean diagnostic: does cross-task sharing (human route) make the dictionary discoverable?",
                 color=NAVY, fontsize=10)
    ax.legend(frameon=False); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("sheaf_llm/semantic_gate_v4.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/semantic_gate_v4.png")


if __name__ == "__main__":
    main()
