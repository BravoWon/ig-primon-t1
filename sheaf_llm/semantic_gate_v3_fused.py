#!/usr/bin/env python
"""semantic gate v3 (FUSED) -- shared cross-task dictionary under a sheaf-gluing loss.

Fuses the "human route" (#3: words recur across contexts -> a shared, stable dictionary) with "pure
sheaf pressure" (#2/#4: discover the structure, punish representations that fail to GLUE). Multi-task
compositional probe: tokens t=(a,b); K tasks each a per-dimension relation out=(al*x1+be*x2)%N that ALL
require the same clean (a,b) factors; held-out = NOVEL (t1,t2) combinations (shared split across tasks).

Locked baselines + the one new arm (gluing isolated):
  flat        : opaque token emb + task emb -> MLP                              (lower bound)
  dict-VQ     : SHARED per-dim codebook (dictionary) + per-task FiLM modulation (NO gluing loss)
  sheaf-glued : IDENTICAL to dict-VQ  +  the GLUING LOSS                        (isolates sheaf pressure)
  handed      : given (a,b) + same FiLM machinery                              (upper bound)

Faithful: shared cross-task dictionary; FiLM modulation of shared base restriction maps; gluing loss =
sheaf-Dirichlet "don't let task modulation drag a token's transported meaning off its shared anchor".
Simplified (flagged): deterministic inferred codebook (not a variational sheaf-Laplacian latent);
gluing anchored to cross-task consistency (not attention edge-diffs -- trivial on a 2-token edge).

Q: does shared-dictionary + sheaf-gluing close the discovery gap toward HANDED in one shot?

    python semantic_gate_v3_fused.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, D, HID, STEPS, BS, LR = 8, 64, 128, 4000, 384, 1e-3
G = D // 2
TASKS = [(1, 1), (1, 3), (2, 1), (1, 7)]            # per-dim relations (7 == -1 mod 8); all need (a,b)
K = len(TASKS)
VQ_W, GLUE_W = 0.1, 0.05
FRACS = [0.05, 0.08, 0.12, 0.20, 0.35]
RED, AMBER, GREEN, NAVY = "#c0392b", "#9a6a2f", "#1e7d34", "#15293f"


def build_data(frac, seed=0):
    pairs = [(t1, t2) for t1 in range(N * N) for t2 in range(N * N)]
    idx = np.random.default_rng(seed).permutation(len(pairs))
    ntr = int(frac * len(pairs)); trp = [pairs[i] for i in idx[:ntr]]; tep = [pairs[i] for i in idx[ntr:]]

    def expand(pl):
        ks, T1, T2, YA, YB = [], [], [], [], []
        for (t1, t2) in pl:
            a1, b1 = divmod(t1, N); a2, b2 = divmod(t2, N)
            for k, (al, be) in enumerate(TASKS):
                ks.append(k); T1.append(t1); T2.append(t2)
                YA.append((al * a1 + be * a2) % N); YB.append((al * b1 + be * b2) % N)
        t = lambda z: torch.tensor(z, device=DEV)
        return t(ks), t(T1), t(T2), t(YA), t(YB)
    return expand(trp), expand(tep)


def quant(x, cb):                                   # snap to nearest code (an ADDRESS); STE + codebook loss
    idx = (x.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1).argmin(-1); q = cb[idx]
    return x + (q - x).detach(), ((x.detach() - q) ** 2).mean() + 0.25 * ((x - q.detach()) ** 2).mean()


class Flat(nn.Module):
    def __init__(self):
        super().__init__(); self.et, self.ek = nn.Embedding(N * N, D), nn.Embedding(K, D)
        self.mlp = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.ha, self.hb = nn.Linear(HID, N), nn.Linear(HID, N)
    def forward(self, k, t1, t2):
        h = self.mlp(torch.cat([self.et(t1), self.et(t2), self.ek(k)], -1)); return self.ha(h), self.hb(h), 0.0


class SharedDict(nn.Module):                        # shared codebook + per-task FiLM; gluing optional
    def __init__(self, glue):
        super().__init__(); self.glue = glue; self.et = nn.Embedding(N * N, D)
        self.cb_a, self.cb_b = nn.Parameter(torch.randn(N, G) * 0.5), nn.Parameter(torch.randn(N, G) * 0.5)
        self.Ra, self.Rb = nn.Linear(G, G), nn.Linear(G, G); self.film = nn.Embedding(K, 4 * G)
        mk = lambda: nn.Sequential(nn.Linear(2 * G, HID), nn.GELU(), nn.Linear(HID, N)); self.ha, self.hb = mk(), mk()
    def forward(self, k, t1, t2):
        e1, e2 = self.et(t1), self.et(t2)
        qa1, la1 = quant(e1[:, :G], self.cb_a); qa2, la2 = quant(e2[:, :G], self.cb_a)
        qb1, lb1 = quant(e1[:, G:], self.cb_b); qb2, lb2 = quant(e2[:, G:], self.cb_b)
        ga, ba, gb, bb = self.film(k).split(G, dim=-1)
        ra1, ra2, rb1, rb2 = self.Ra(qa1), self.Ra(qa2), self.Rb(qb1), self.Rb(qb2)
        za1, za2 = ra1 * (1 + ga) + ba, ra2 * (1 + ga) + ba
        zb1, zb2 = rb1 * (1 + gb) + bb, rb2 * (1 + gb) + bb
        oa, ob = self.ha(torch.cat([za1, za2], -1)), self.hb(torch.cat([zb1, zb2], -1))
        aux = VQ_W * (la1 + la2 + lb1 + lb2)
        if self.glue:                               # sheaf-Dirichlet: task modulation must not drift off the anchor
            aux = aux + GLUE_W * (((za1 - ra1) ** 2).mean() + ((za2 - ra2) ** 2).mean()
                                  + ((zb1 - rb1) ** 2).mean() + ((zb2 - rb2) ** 2).mean())
        return oa, ob, aux


class Handed(nn.Module):                            # given factors + same FiLM machinery (upper bound)
    def __init__(self):
        super().__init__(); self.ea, self.eb = nn.Embedding(N, G), nn.Embedding(N, G)
        self.Ra, self.Rb = nn.Linear(G, G), nn.Linear(G, G); self.film = nn.Embedding(K, 4 * G)
        mk = lambda: nn.Sequential(nn.Linear(2 * G, HID), nn.GELU(), nn.Linear(HID, N)); self.ha, self.hb = mk(), mk()
    def forward(self, k, t1, t2):
        a1, b1, a2, b2 = t1 // N, t1 % N, t2 // N, t2 % N
        qa1, qa2, qb1, qb2 = self.ea(a1), self.ea(a2), self.eb(b1), self.eb(b2)
        ga, ba, gb, bb = self.film(k).split(G, dim=-1)
        za1, za2 = self.Ra(qa1) * (1 + ga) + ba, self.Ra(qa2) * (1 + ga) + ba
        zb1, zb2 = self.Rb(qb1) * (1 + gb) + bb, self.Rb(qb2) * (1 + gb) + bb
        return self.ha(torch.cat([za1, za2], -1)), self.hb(torch.cat([zb1, zb2], -1)), 0.0


def acc(m, d):
    k, t1, t2, ya, yb = d
    with torch.no_grad():
        oa, ob, _ = m(k, t1, t2)
        return ((oa.argmax(-1) == ya) & (ob.argmax(-1) == yb)).float().mean().item()


def train_one(factory, tr, te):
    torch.manual_seed(0); m = factory().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR)
    k, t1, t2, ya, yb = tr
    for _ in range(STEPS):
        i = torch.randint(0, len(k), (BS,), device=DEV)
        oa, ob, aux = m(k[i], t1[i], t2[i])
        loss = F.cross_entropy(oa, ya[i]) + F.cross_entropy(ob, yb[i]) + aux
        opt.zero_grad(); loss.backward(); opt.step()
    return sum(p.numel() for p in m.parameters()), acc(m, te)


def main():
    print(f"[fused gate v3]  dev={DEV}  N={N}  K={K} tasks  (chance={1/(N*N):.3f})  shared dict + sheaf-glue")
    models = [("flat", Flat, RED), ("dict-VQ", lambda: SharedDict(False), AMBER),
              ("sheaf-glued", lambda: SharedDict(True), GREEN), ("handed", Handed, NAVY)]
    curves = {n: [] for n, _, _ in models}; params = {}
    for frac in FRACS:
        tr, te = build_data(frac); line = f"  frac {frac:.2f} ({len(tr[0]):5d} ex):"
        for n, C, _ in models:
            p, a = train_one(C, tr, te); curves[n].append(a); params[n] = p; line += f"  {n}={a:.3f}"
        print(line)

    def ex_to(curve, thr=0.9):
        for fr, a in zip(FRACS, curve):
            if a >= thr:
                return int(fr * 4096 * K)
        return None
    print("\nVERDICT -- data-efficiency (examples to reach 0.9 OOS on novel combinations):")
    for n, _, _ in models:
        e = ex_to(curves[n]); print(f"  {n:12} {(str(e)+' ex') if e else '>max (never)':>16}   "
                                    + " ".join(f"{a:.2f}" for a in curves[n]))
    dv, sg, h = curves["dict-VQ"], curves["sheaf-glued"], curves["handed"]
    egap = (ex_to(sg) or 10 ** 9) - (ex_to(h) or 0)
    print(f"  GLUING effect (sheaf-glued vs dict-VQ): max {max(sg):.2f} vs {max(dv):.2f}; "
          f"to-0.9 {ex_to(sg)} vs {ex_to(dv)}.")
    print(f"  gap to handed: sheaf-glued reaches 0.9 at {ex_to(sg)} vs handed {ex_to(h)} "
          f"-> {'CLOSED (human route + sheaf pressure discovers ~ as cheaply as handed)' if (ex_to(sg) and ex_to(h) and ex_to(sg) <= 2*ex_to(h)) else 'narrowed but NOT closed -- discovery still costs'}")

    fig, ax = plt.subplots(figsize=(7.8, 5))
    for n, _, c in models:
        ax.plot([fr * 4096 * K for fr in FRACS], curves[n], "-o", color=c, lw=2, label=f"{n} ({params[n]/1e3:.0f}k)")
    ax.axhline(1 / (N * N), color="#999", ls=":", lw=1, label="chance")
    ax.set_xlabel(f"training examples (pairs x {K} tasks)"); ax.set_ylabel("acc on NOVEL combinations (OOS)")
    ax.set_title("Fused gate: shared cross-task dictionary + sheaf-gluing -- does it close the discovery gap?",
                 color=NAVY, fontsize=10)
    ax.legend(frameon=False); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("sheaf_llm/semantic_gate_v3.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/semantic_gate_v3.png")


if __name__ == "__main__":
    main()
