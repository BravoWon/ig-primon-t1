#!/usr/bin/env python
"""adversarial gate -- HARDEN brick 2. Does grounding's zero-shot win survive a target that is NOT a
clean function of the inherited dictionary?

Brick 2 got grounded-sheaf = 1.00/1.00, but partly BY CONSTRUCTION: its target was exactly a modular
function of the two inherited WordNet tiers (supersense, depth). Real word-meaning is not. So here the
role-sensitive target has TWO components:

  y_dict  = (ss_subj  + 2*ss_obj  + verb) % S     <- shareable: a function of the INHERITED tiers
  y_resid = (r_subj   + 2*r_obj)          % R     <- idiosyncratic: a hidden per-word latent r, assigned
                                                     RANDOMLY and INDEPENDENT of (supersense, depth)
                                                     => meaning the dictionary cannot see

Three arms:
  flat            : per-word token embedding only (standard LLM); UNK on unseen words.
  grounded-pure   : inherited (ss,depth) + role-specific restriction maps; NO per-word capacity.
  grounded+token  : hybrid -- inherited tiers AND a per-word token embedding (augment, not replace).

PRE-REGISTERED.
  Hypothesis: on y_dict, grounded arms generalize ZERO-SHOT to unseen words (~1.0) where flat is at
              chance; on y_resid, NOBODY generalizes to unseen words (idiosyncratic by construction),
              and grounded-PURE is at chance even on SEEN words (no per-word capacity) -- the honest
              cost of pure grounding. So NO arm gets 1.00/1.00; the 1.00/1.00 of brick 2 was a
              zero-residual target. The hybrid is the honest architecture (seen residual + zero-shot dict).
  Control   : flat must still ace BOTH components on SEEN words (it can memorize r) -- proves the
              residual head is learnable in principle, so grounded-pure's chance there is a real blind
              spot, not an unlearnable task.
  Falsifier : if grounded-pure does NOT beat flat on y_dict for UNSEEN words, brick 2's win was rigged.

    python adversarial_gate.py
"""
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SS = ["noun.animal", "noun.artifact", "noun.food", "noun.plant",
      "noun.body", "noun.location", "noun.person", "noun.substance"]
S, B, R, NV = len(SS), 4, 5, 8          # S supersense, B depth, R residual levels (prime), NV verbs
D, HID, STEPS, BS, LR = 64, 128, 5000, 256, 1e-3
RED, AMBER, GREEN, NAVY = "#c0392b", "#9a6a2f", "#1e7d34", "#15293f"


def build_vocab(per_ss=34):
    cand = defaultdict(set)
    for syn in wn.all_synsets("n"):
        if syn.lexname() in SS:
            d = syn.min_depth()
            for lem in syn.lemmas():
                w = lem.name()
                if w.isalpha() and len(w) > 2:
                    cand[w].add((SS.index(syn.lexname()), d))
    vocab, by = {}, defaultdict(list)
    for w, st in cand.items():
        if len({x[0] for x in st}) == 1:
            ss = next(iter(st))[0]; vocab[w] = (ss, int(np.median([x[1] for x in st]))); by[ss].append(w)
    rng = np.random.default_rng(0); chosen = {}
    for ss, ws in by.items():
        ws = sorted(ws); rng.shuffle(ws)
        for w in ws[:per_ss]:
            chosen[w] = vocab[w]
    qs = np.quantile([d for _, d in chosen.values()], [0.25, 0.5, 0.75])
    return {w: (ss, int(sum(d > q for q in qs))) for w, (ss, d) in chosen.items()}


def main():
    feat = build_vocab(); words = sorted(feat)
    rng = np.random.default_rng(0); rng.shuffle(words)
    # hidden per-word residual latent r, INDEPENDENT of (ss, depth) -- the meaning the dictionary can't see
    resid = {w: int(rng.integers(0, R)) for w in words}
    ns = int(0.8 * len(words)); seen, unseen = words[:ns], words[ns:]; seenset = set(seen)
    wid = {w: i for i, w in enumerate(words)}; UNK = len(words)
    print(f"[adversarial gate]  dev={DEV}  vocab={len(words)} nouns (seen {len(seen)}/unseen {len(unseen)})  "
          f"S={S} R={R} verbs={NV}  (chance_dict={1/S:.3f}  chance_resid={1/R:.3f})")

    def make(pool, n):
        si = rng.integers(0, len(pool), n); oi = rng.integers(0, len(pool), n); v = rng.integers(0, NV, n)
        subj = [pool[i] for i in si]; obj = [pool[j] for j in oi]
        yd = [(feat[s][0] + 2 * feat[o][0] + vv) % S for s, o, vv in zip(subj, obj, v)]
        yr = [(resid[s] + 2 * resid[o]) % R for s, o in zip(subj, obj)]
        return (subj, obj, torch.tensor(v, device=DEV),
                torch.tensor(yd, device=DEV), torch.tensor(yr, device=DEV))

    tr = make(seen, 40000); te_seen = make(seen, 5000); te_unseen = make(unseen, 5000)

    def t_flat(subj, obj, train):
        f = lambda ws: torch.tensor([wid[w] if (train or w in seenset) else UNK for w in ws], device=DEV)
        return f(subj), f(obj)

    def t_grnd(ws):
        return (torch.tensor([feat[w][0] for w in ws], device=DEV),
                torch.tensor([feat[w][1] for w in ws], device=DEV))

    def t_tok(ws, train):
        return torch.tensor([wid[w] if (train or w in seenset) else UNK for w in ws], device=DEV)

    class Flat(nn.Module):
        def __init__(self):
            super().__init__(); self.et = nn.Embedding(len(words) + 1, D); self.ev = nn.Embedding(NV, D)
            self.mlp = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
            self.hd, self.hr = nn.Linear(HID, S), nn.Linear(HID, R)
        def forward(self, fs, fo, v):
            h = self.mlp(torch.cat([self.et(fs), self.ev(v), self.et(fo)], -1)); return self.hd(h), self.hr(h)

    class Grounded(nn.Module):
        def __init__(self, hybrid):
            super().__init__(); self.hybrid = hybrid
            self.ess, self.ed, self.ev = nn.Embedding(S, D), nn.Embedding(B, D), nn.Embedding(NV, D)
            nin = 3 * D if hybrid else 2 * D                 # +token emb per noun if hybrid
            if hybrid:
                self.et = nn.Embedding(len(words) + 1, D)
            self.Rs, self.Ro = nn.Linear(nin, D), nn.Linear(nin, D)   # role-specific restriction maps
            self.mlp = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
            self.hd, self.hr = nn.Linear(HID, S), nn.Linear(HID, R)
        def noun(self, g, tok):
            parts = [self.ess(g[0]), self.ed(g[1])]
            if self.hybrid:
                parts.append(self.et(tok))
            return torch.cat(parts, -1)
        def forward(self, gs, go, v, ts, to):
            zs, zo = self.Rs(self.noun(gs, ts)), self.Ro(self.noun(go, to))
            h = self.mlp(torch.cat([zs, self.ev(v), zo], -1)); return self.hd(h), self.hr(h)

    def run(kind):
        torch.manual_seed(0)
        m = (Flat() if kind == "flat" else Grounded(kind == "grounded+token")).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        subj, obj, v, yd, yr = tr
        if kind == "flat":
            Xs, Xo = t_flat(subj, obj, True)
        else:
            Gs, Go = t_grnd(subj), t_grnd(obj)
            Ts, To = t_tok(subj, True), t_tok(obj, True)
        for _ in range(STEPS):
            i = torch.randint(0, len(subj), (BS,), device=DEV)
            if kind == "flat":
                ld, lr = m(Xs[i], Xo[i], v[i])
            else:
                ld, lr = m((Gs[0][i], Gs[1][i]), (Go[0][i], Go[1][i]), v[i], Ts[i], To[i])
            (F.cross_entropy(ld, yd[i]) + F.cross_entropy(lr, yr[i])).backward(); opt.step(); opt.zero_grad()

        def acc(te):
            s2, o2, v2, yd2, yr2 = te
            with torch.no_grad():
                if kind == "flat":
                    fs, fo = t_flat(s2, o2, False); ld, lr = m(fs, fo, v2)
                else:
                    ld, lr = m(t_grnd(s2), t_grnd(o2), v2, t_tok(s2, False), t_tok(o2, False))
                ad = (ld.argmax(-1) == yd2).float().mean().item()
                ar = (lr.argmax(-1) == yr2).float().mean().item()
                return ad, ar
        p = sum(x.numel() for x in m.parameters())
        return p, acc(te_seen), acc(te_unseen)

    print(f"\n{'model':16}{'params':>9}    {'SEEN: dict / resid':>20}    {'UNSEEN: dict / resid':>22}")
    res = {}
    for kind in ("flat", "grounded-pure", "grounded+token"):
        p, (sd, sr), (ud, ur) = run(kind); res[kind] = (sd, sr, ud, ur)
        print(f"{kind:16}{p/1e3:>8.1f}k    {sd:>8.2f} / {sr:<8.2f}    {ud:>9.2f} / {ur:<9.2f}")

    print(f"\nVERDICT (chance_dict={1/S:.2f}  chance_resid={1/R:.2f}):")
    fd, _, fud, fur = res["flat"]; gpd_s, gpr_s, gpd_u, gpr_u = res["grounded-pure"]
    hyd_s, hyr_s, hyd_u, hyr_u = res["grounded+token"]
    print(f"  dict zero-shot: grounded-pure UNSEEN={gpd_u:.2f} vs flat UNSEEN={fud:.2f}  -> "
          f"{'GROUNDING WIN SURVIVES (zero-shot dict is real)' if gpd_u > fud + 0.3 else 'RIGGED -- no zero-shot lift with a residual present'}")
    print(f"  resid blind spot: grounded-pure SEEN resid={gpr_s:.2f} (~chance) while flat SEEN resid={res['flat'][1]:.2f}  -> "
          f"pure grounding {'CANNOT see idiosyncratic meaning (honest cost)' if gpr_s < 0.45 else 'unexpectedly recovers residual'}")
    print(f"  honest ceiling: best UNSEEN both-correct ~ dict*resid; NO arm gets 1.00/1.00 "
          f"(idiosyncratic meaning is irreducible zero-shot).")
    print(f"  hybrid: SEEN dict/resid={hyd_s:.2f}/{hyr_s:.2f}  UNSEEN dict/resid={hyd_u:.2f}/{hyr_u:.2f}  -> "
          f"{'augment beats replace (seen residual + zero-shot dict)' if (hyr_s > 0.8 and hyd_u > fud + 0.3) else 'no hybrid advantage'}")

    # plot: dict vs resid, seen vs unseen, three arms
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    arms = [("flat", RED), ("grounded-pure", AMBER), ("grounded+token", GREEN)]
    for ax, comp, ci, chance, ttl in [
            (axes[0], "dict", (0, 2), 1 / S, "y_dict  (shareable: in the WordNet dictionary)"),
            (axes[1], "resid", (1, 3), 1 / R, "y_resid  (idiosyncratic: NOT in the dictionary)")]:
        x = np.arange(2); w = 0.26
        for i, (k, c) in enumerate(arms):
            ax.bar(x + (i - 1) * w, [res[k][ci[0]], res[k][ci[1]]], w, color=c, label=k)
        ax.axhline(chance, color="#999", ls=":", lw=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(["SEEN words", "UNSEEN words\n(zero-shot)"])
        ax.set_ylim(0, 1.05); ax.set_title(ttl, color=NAVY, fontsize=9.5)
    axes[0].set_ylabel("component accuracy"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Adversarial gate: grounding recovers the SHAREABLE fraction zero-shot, not idiosyncratic meaning",
                 color=NAVY, fontsize=10.5)
    fig.tight_layout(); fig.savefig("sheaf_llm/adversarial_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/adversarial_gate.png")


if __name__ == "__main__":
    main()
