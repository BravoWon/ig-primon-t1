#!/usr/bin/env python
"""semantic gate v1 — does DIMENSIONAL STRUCTURE + RELATIONS generalize where flat statistics fail?

Tests the human-trace thesis on the cleanest controlled probe. A "word" = token with two TIERS of
dimensional meaning: t=(a,b), a,b in [0,N). The answer is a COMPOSITIONAL function of the dimensions:
  (t1,t2) -> ((a1+a2) mod N, (b1+b2) mod N).
Split: train on a random subset of PAIRS, test on HELD-OUT pairs (every token seen; the COMBINATION
novel) = compositional generalization. Three representations of the SAME meaning:
  flat       : token = one opaque embedding (statistics only)
  factored   : token = (embed(a), embed(b))         -- the dimensional dictionary
  relational : factored + PER-DIMENSION relation     -- the meaning's structure made explicit (sheaf)
Sweep training-data fraction: structure should generalize from FEW examples (the human trait); flat
should sit at chance until it has memorized enough to grok. If flat keeps pace, the thesis is wrong.

    python semantic_gate.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, D, HID, STEPS, BS, LR = 8, 64, 128, 3000, 256, 1e-3
FRACS = [0.05, 0.08, 0.12, 0.20, 0.35, 0.55]
NAVY, RED, GREEN = "#15293f", "#c0392b", "#1e7d34"


def build_data(frac, seed=0):
    """PR#13 review: the 'every token seen' split invariant is now ENFORCED (resample until
    cov == 1.0), not merely reported."""
    for s in range(seed, seed + 200):
        out = _build_once(frac, s)
        if out[2] == 1.0:
            return out
    raise SystemExit(f"nm: no split with full token coverage at frac={frac} in 200 seeds")


def _build_once(frac, seed):
    pairs = [(t1, t2) for t1 in range(N * N) for t2 in range(N * N)]
    def tgt(t1, t2):
        a1, b1 = divmod(t1, N); a2, b2 = divmod(t2, N)
        return (a1 + a2) % N, (b1 + b2) % N
    idx = np.random.default_rng(seed).permutation(len(pairs))
    ntr = int(frac * len(pairs)); tr = [pairs[i] for i in idx[:ntr]]; te = [pairs[i] for i in idx[ntr:]]
    cov = len({t for p in tr for t in p}) / (N * N)
    def tens(pl):
        return (torch.tensor([p[0] for p in pl], device=DEV), torch.tensor([p[1] for p in pl], device=DEV),
                torch.tensor([tgt(*p) for p in pl], device=DEV))
    return tens(tr), tens(te), cov


class Flat(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(N * N, D)
        self.mlp = nn.Sequential(nn.Linear(2 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.ha, self.hb = nn.Linear(HID, N), nn.Linear(HID, N)
    def forward(self, t1, t2):
        h = self.mlp(torch.cat([self.emb(t1), self.emb(t2)], -1)); return self.ha(h), self.hb(h)


class Factored(nn.Module):
    def __init__(self):
        super().__init__(); self.ea, self.eb = nn.Embedding(N, D), nn.Embedding(N, D)
        self.mlp = nn.Sequential(nn.Linear(4 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.ha, self.hb = nn.Linear(HID, N), nn.Linear(HID, N)
    def forward(self, t1, t2):
        a1, b1 = t1 // N, t1 % N; a2, b2 = t2 // N, t2 % N
        h = self.mlp(torch.cat([self.ea(a1), self.eb(b1), self.ea(a2), self.eb(b2)], -1))
        return self.ha(h), self.hb(h)


class Relational(nn.Module):
    def __init__(self):
        super().__init__(); self.ea, self.eb = nn.Embedding(N, D), nn.Embedding(N, D)
        mk = lambda: nn.Sequential(nn.Linear(2 * D, HID), nn.GELU(), nn.Linear(HID, N))
        self.ra, self.rb = mk(), mk()
    def forward(self, t1, t2):
        a1, b1 = t1 // N, t1 % N; a2, b2 = t2 // N, t2 % N
        return (self.ra(torch.cat([self.ea(a1), self.ea(a2)], -1)),
                self.rb(torch.cat([self.eb(b1), self.eb(b2)], -1)))


def acc(m, t1, t2, y):
    with torch.no_grad():
        la, lb = m(t1, t2)
        return ((la.argmax(-1) == y[:, 0]) & (lb.argmax(-1) == y[:, 1])).float().mean().item()


def train_one(Cls, tr, te):
    torch.manual_seed(0); m = Cls().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
    tr1, tr2, try_ = tr
    for _ in range(STEPS):
        i = torch.randint(0, len(tr1), (BS,), device=DEV)
        la, lb = m(tr1[i], tr2[i])
        loss = F.cross_entropy(la, try_[i, 0]) + F.cross_entropy(lb, try_[i, 1])
        opt.zero_grad(); loss.backward(); opt.step()
    return sum(p.numel() for p in m.parameters()), acc(m, *te)


def main():
    print(f"[semantic gate v1]  dev={DEV}  N={N}  data-efficiency sweep  (chance={1/(N*N):.3f})")
    models = [("flat", Flat, RED), ("factored", Factored, "#9a6a2f"), ("relational", Relational, GREEN)]
    curves = {n: [] for n, _, _ in models}; params = {}
    for frac in FRACS:
        tr, te, cov = build_data(frac); ntr = len(tr[0])
        line = f"  frac {frac:.2f}  ({ntr:4d} train, cov {cov:.2f}):"
        for n, C, _ in models:
            p, a = train_one(C, tr, te); curves[n].append(a); params[n] = p
            line += f"  {n}={a:.3f}"
        print(line)

    print("\nVERDICT (OOS acc on NOVEL combinations):")
    for n, _, _ in models:
        print(f"  {n:10} params {params[n]/1e3:5.1f}k   OOS by frac: " + " ".join(f"{a:.2f}" for a in curves[n]))
    f0 = curves["flat"]; r0 = curves["relational"]
    # PR#13 review: verdict text now CONDITIONAL on the measured curves (it previously asserted
    # the thesis unconditionally), compared at matched fractions, and labeled exploratory.
    lowi, hii = 0, len(FRACS) - 1
    wins = r0[lowi] > f0[lowi] + 0.05 and r0[hii] >= f0[hii] - 0.02
    print(f"  at frac={FRACS[lowi]:.2f}: relational {r0[lowi]:.2f} vs flat {f0[lowi]:.2f}; "
          f"at frac={FRACS[hii]:.2f}: {r0[hii]:.2f} vs {f0[hii]:.2f} -> "
          f"{'structure generalizes from few examples here' if wins else 'no structural advantage at this budget'}")
    print("  NOTE (exploratory): single seed per cell; models NOT capacity-matched "
          f"(params {', '.join(f'{n}={params[n]/1e3:.1f}k' for n, _, _ in models)}) -- "
          "architectural comparison, not isolated evidence for structure.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for n, _, c in models:
        ax.plot([f * 4096 for f in FRACS], curves[n], "-o", color=c, lw=2, label=f"{n} ({params[n]/1e3:.0f}k params)")
    ax.axhline(1 / (N * N), color="#999", ls=":", lw=1, label="chance")
    ax.set_xlabel("training examples (of 4096 possible combinations)")
    ax.set_ylabel("accuracy on NOVEL combinations (OOS)")
    ax.set_title("Meaning-structure generalizes from few examples; flat statistics memorize and fail",
                 color=NAVY, fontsize=11)
    ax.legend(frameon=False); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("sheaf_llm/semantic_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/semantic_gate.png")


if __name__ == "__main__":
    main()
