#!/usr/bin/env python
"""sentence gate -- the natural extension: grounded dictionary + SHEAF RELATIONS over sentence graphs.

The word-gate proved grounding gives zero-shot to unseen WORDS. But language is STRUCTURED -- roles
matter. So: short SVO sentences (subj, verb, obj) from real WordNet-grounded nouns; the target is a
ROLE-SENSITIVE compositional function out = (ss(subj) + 2*ss(obj) + verb) % S -- subject and object
count differently, so you MUST respect structure, not just bag meanings. Two held-out tests: novel
(subj,obj) combinations of SEEN words, and pairs with UNSEEN words (zero-shot).

  flat          : learned token embeddings, positional (roles via order) -- but UNK on unseen words
  grounded-bag  : inherited (supersense,depth), SUMMED over the two nouns (no role distinction)
  grounded-sheaf: inherited features + ROLE-SPECIFIC restriction maps (R_subj != R_obj) over the graph

Prediction: flat dies on UNSEEN words; grounded-bag dies on ROLES; grounded-sheaf gets BOTH -> the
inherited dictionary and the sheaf relations are both necessary and they COMPOSE.

    python sentence_gate.py
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
S, B, NV, D, HID, STEPS, BS, LR = len(SS), 4, 8, 64, 128, 5000, 256, 1e-3
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
    ns = int(0.8 * len(words)); seen, unseen = words[:ns], words[ns:]; seenset = set(seen)
    wid = {w: i for i, w in enumerate(words)}; UNK = len(words)
    print(f"[sentence gate]  dev={DEV}  vocab={len(words)} nouns (seen {len(seen)}/unseen {len(unseen)})  "
          f"S={S} verbs={NV}  role-sensitive target  (chance={1/S:.3f})")

    def make(pool, n):
        si = rng.integers(0, len(pool), n); oi = rng.integers(0, len(pool), n); v = rng.integers(0, NV, n)
        subj = [pool[i] for i in si]; obj = [pool[j] for j in oi]
        y = [(feat[s][0] + 2 * feat[o][0] + vv) % S for s, o, vv in zip(subj, obj, v)]
        return subj, obj, torch.tensor(v, device=DEV), torch.tensor(y, device=DEV)

    tr = make(seen, 40000); te_seen = make(seen, 5000); te_unseen = make(unseen, 5000)

    def t_flat(subj, obj, train):
        f = lambda ws: torch.tensor([wid[w] if (train or w in seenset) else UNK for w in ws], device=DEV)
        return f(subj), f(obj)

    def t_grnd(ws):
        return (torch.tensor([feat[w][0] for w in ws], device=DEV),
                torch.tensor([feat[w][1] for w in ws], device=DEV))

    class Flat(nn.Module):
        def __init__(self):
            super().__init__(); self.et = nn.Embedding(len(words) + 1, D); self.ev = nn.Embedding(NV, D)
            self.mlp = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, S))
        def forward(self, fs, fo, v):
            return self.mlp(torch.cat([self.et(fs), self.ev(v), self.et(fo)], -1))

    class Grounded(nn.Module):
        def __init__(self, sheaf):
            super().__init__(); self.sheaf = sheaf
            self.ess, self.ed, self.ev = nn.Embedding(S, D), nn.Embedding(B, D), nn.Embedding(NV, D)
            if sheaf:
                self.Rs, self.Ro = nn.Linear(2 * D, D), nn.Linear(2 * D, D)      # role-specific restriction maps
            self.mlp = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, S))
        def forward(self, gs, go, v):
            s = torch.cat([self.ess(gs[0]), self.ed(gs[1])], -1); o = torch.cat([self.ess(go[0]), self.ed(go[1])], -1)
            if self.sheaf:
                zs, zo = self.Rs(s), self.Ro(o)                                  # subj/obj transported by DIFFERENT maps
            else:
                zs = zo = (s + o)[:, :D]                                         # bag: symmetric sum, role-blind
            return self.mlp(torch.cat([zs, self.ev(v), zo], -1))

    def run(kind):
        torch.manual_seed(0)
        m = (Flat() if kind == "flat" else Grounded(kind == "grounded-sheaf")).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        subj, obj, v, y = tr
        if kind == "flat":
            Xs, Xo = t_flat(subj, obj, True)
        else:
            Xs, Xo = t_grnd(subj), t_grnd(obj)
        for _ in range(STEPS):
            i = torch.randint(0, len(subj), (BS,), device=DEV)
            if kind == "flat":
                lo = m(Xs[i], Xo[i], v[i])
            else:
                lo = m((Xs[0][i], Xs[1][i]), (Xo[0][i], Xo[1][i]), v[i])
            F.cross_entropy(lo, y[i]).backward(); opt.step(); opt.zero_grad()

        def acc(te):
            s2, o2, v2, y2 = te
            with torch.no_grad():
                if kind == "flat":
                    fs, fo = t_flat(s2, o2, False); lo = m(fs, fo, v2)
                else:
                    lo = m(t_grnd(s2), t_grnd(o2), v2)
                return (lo.argmax(-1) == y2).float().mean().item()
        return sum(p.numel() for p in m.parameters()), acc(te_seen), acc(te_unseen)

    print(f"\n{'model':16}{'params':>9}{'novel-combo (seen)':>20}{'UNSEEN words':>15}")
    res = {}
    for kind in ("flat", "grounded-bag", "grounded-sheaf"):
        p, sa, ua = run(kind); res[kind] = (sa, ua)
        print(f"{kind:16}{p/1e3:>8.1f}k{sa:>20.3f}{ua:>15.3f}")
    print(f"\nVERDICT (chance {1/S:.3f}) -- only the FUSION should get BOTH:")
    for kind in res:
        sa, ua = res[kind]
        tag = ("roles AND words" if sa > 0.6 and ua > 0.6 else
               ("dies on UNSEEN words" if sa > 0.6 else ("dies on ROLES" if ua < 0.6 else "partial")))
        print(f"  {kind:16} seen {sa:.2f} / unseen {ua:.2f}  -> {tag}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8)); x = np.arange(2); w = 0.26
    for i, (k, c) in enumerate([("flat", RED), ("grounded-bag", AMBER), ("grounded-sheaf", GREEN)]):
        ax.bar(x + (i - 1) * w, [res[k][0], res[k][1]], w, color=c, label=k)
    ax.axhline(1 / S, color="#999", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["novel combos\n(seen words)", "UNSEEN words\n(zero-shot)"])
    ax.set_ylabel("role-sensitive compositional acc"); ax.set_ylim(0, 1.05)
    ax.set_title("Sentence gate: only inherited dictionary + sheaf relations gets BOTH roles and new words",
                 color=NAVY, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig("sheaf_llm/sentence_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/sentence_gate.png")


if __name__ == "__main__":
    main()
