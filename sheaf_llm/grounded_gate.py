#!/usr/bin/env python
"""grounded gate -- the inherited dictionary on REAL words, tested ZERO-SHOT on unseen words.

The 4 toy gates said: handing dimensional structure is a crushing win; discovering it from scratch is
hard. Humans don't discover it -- they INHERIT a dictionary. So: ground real words in a REAL sense
inventory (WordNet) and test the human move -- understanding a word you've NEVER trained on, by its
dictionary entry.

Real nouns -> two inherited TIERS from WordNet: supersense (noun.animal/artifact/food/...) + hypernym
depth bucket. Task = compositional function of the tiers: (t1,t2) -> ((ss1+ss2)%S, (d1+d2)%B). Split
HOLDS OUT WHOLE WORDS (never seen in training); test on pairs of UNSEEN words = zero-shot.
  flat     : per-word learned embedding (the standard LLM approach); unseen word -> UNK
  grounded : word -> (supersense, depth) from WordNet -> per-tier relational heads (the inherited dict)
Q: does the grounded model generalize to words it never trained on, where flat collapses to chance?

    python grounded_gate.py
"""
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SS = ["noun.animal", "noun.artifact", "noun.food", "noun.plant",
      "noun.body", "noun.location", "noun.person", "noun.substance"]
S, B, D, HID, STEPS, BS, LR = len(SS), 4, 64, 128, 4000, 256, 1e-3
RED, NAVY, GREEN = "#c0392b", "#15293f", "#1e7d34"


def build_vocab(per_ss=34):
    cand = defaultdict(set)
    for syn in wn.all_synsets("n"):
        ln = syn.lexname()
        if ln in SS:
            d = syn.min_depth()
            for lem in syn.lemmas():
                w = lem.name()
                if w.isalpha() and len(w) > 2:
                    cand[w].add((SS.index(ln), d))
    vocab, by = {}, defaultdict(list)
    for w, st in cand.items():
        if len({x[0] for x in st}) == 1:                     # unambiguous supersense only
            ss = next(iter(st))[0]; depth = int(np.median([x[1] for x in st]))
            vocab[w] = (ss, depth); by[ss].append(w)
    rng = np.random.default_rng(0); chosen = {}
    for ss, ws in by.items():
        ws = sorted(ws); rng.shuffle(ws)
        for w in ws[:per_ss]:
            chosen[w] = vocab[w]
    qs = np.quantile([d for _, d in chosen.values()], [0.25, 0.5, 0.75])
    return {w: (ss, int(sum(d > q for q in qs))) for w, (ss, d) in chosen.items()}


def main():
    feat = build_vocab()
    words = sorted(feat); rng = np.random.default_rng(0); rng.shuffle(words)
    nseen = int(0.8 * len(words)); seen, unseen = words[:nseen], words[nseen:]
    wid = {w: i for i, w in enumerate(words)}; UNK = len(words)
    print(f"[grounded gate]  dev={DEV}  vocab={len(words)} real nouns  seen={len(seen)} unseen={len(unseen)}  "
          f"S={S} supersenses x B={B} depth  (chance={1/(S*B):.3f})")

    def pairs(pool, n):
        a = rng.integers(0, len(pool), n); b = rng.integers(0, len(pool), n)
        w1 = [pool[i] for i in a]; w2 = [pool[j] for j in b]
        ya = [(feat[x][0] + feat[y][0]) % S for x, y in zip(w1, w2)]
        yb = [(feat[x][1] + feat[y][1]) % B for x, y in zip(w1, w2)]
        return w1, w2, torch.tensor(ya, device=DEV), torch.tensor(yb, device=DEV)

    tr = pairs(seen, 20000)
    te_seen = pairs(seen, 4000)                              # held-out pairs of SEEN words
    te_unseen = pairs(unseen, 4000)                          # pairs of UNSEEN words = zero-shot

    def to_flat(w1, w2, train):                             # word ids; unseen -> UNK
        f1 = torch.tensor([wid[w] if (train or w in set(seen)) else UNK for w in w1], device=DEV)
        f2 = torch.tensor([wid[w] if (train or w in set(seen)) else UNK for w in w2], device=DEV)
        return f1, f2

    def to_grnd(w1, w2):                                    # inherited (supersense, depth) tiers
        g = lambda ws, k: torch.tensor([feat[w][k] for w in ws], device=DEV)
        return (g(w1, 0), g(w1, 1)), (g(w2, 0), g(w2, 1))

    class Flat(nn.Module):
        def __init__(self):
            super().__init__(); self.emb = nn.Embedding(len(words) + 1, D)
            self.mlp = nn.Sequential(nn.Linear(2 * D, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
            self.ha, self.hb = nn.Linear(HID, S), nn.Linear(HID, B)
        def forward(self, f1, f2):
            h = self.mlp(torch.cat([self.emb(f1), self.emb(f2)], -1)); return self.ha(h), self.hb(h)

    class Grounded(nn.Module):
        def __init__(self):
            super().__init__(); self.ess, self.ed = nn.Embedding(S, D), nn.Embedding(B, D)
            mk = lambda o: nn.Sequential(nn.Linear(2 * D, HID), nn.GELU(), nn.Linear(HID, o))
            self.ha, self.hb = mk(S), mk(B)
        def forward(self, g1, g2):
            (s1, d1), (s2, d2) = g1, g2
            return (self.ha(torch.cat([self.ess(s1), self.ess(s2)], -1)),
                    self.hb(torch.cat([self.ed(d1), self.ed(d2)], -1)))

    seenset = set(seen)
    def run(kind):
        torch.manual_seed(0); m = (Flat if kind == "flat" else Grounded)().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        tw1, tw2, tya, tyb = tr
        prep = (lambda w1, w2, tr: to_flat(w1, w2, tr)) if kind == "flat" else (lambda w1, w2, tr: to_grnd(w1, w2))
        X = prep(tw1, tw2, True)
        for _ in range(STEPS):
            i = torch.randint(0, len(tw1), (BS,), device=DEV)
            if kind == "flat":
                la, lb = m(X[0][i], X[1][i])
            else:
                la, lb = m((X[0][0][i], X[0][1][i]), (X[1][0][i], X[1][1][i]))
            (F.cross_entropy(la, tya[i]) + F.cross_entropy(lb, tyb[i])).backward(); opt.step(); opt.zero_grad()

        def acc(te):
            w1, w2, ya, yb = te
            with torch.no_grad():
                if kind == "flat":
                    f1, f2 = to_flat(w1, w2, False); la, lb = m(f1, f2)
                else:
                    g1, g2 = to_grnd(w1, w2); la, lb = m(g1, g2)
                return ((la.argmax(-1) == ya) & (lb.argmax(-1) == yb)).float().mean().item()
        return sum(p.numel() for p in m.parameters()), acc(te_seen), acc(te_unseen)

    print(f"\n{'model':10}{'params':>9}{'seen-pair acc':>15}{'UNSEEN-word acc (zero-shot)':>30}")
    res = {}
    for kind in ("flat", "grounded"):
        p, sa, ua = run(kind); res[kind] = (sa, ua)
        print(f"{kind:10}{p/1e3:>8.1f}k{sa:>15.3f}{ua:>30.3f}")
    print(f"\nVERDICT (chance {1/(S*B):.3f}):")
    print(f"  flat: seen {res['flat'][0]:.2f} but UNSEEN {res['flat'][1]:.2f} -> "
          f"{'CANNOT generalize to new words (UNK)' if res['flat'][1] < 0.15 else 'partial'}")
    print(f"  grounded: seen {res['grounded'][0]:.2f}, UNSEEN {res['grounded'][1]:.2f} -> "
          f"{'INHERITED DICTIONARY generalizes ZERO-SHOT to unseen words' if res['grounded'][1] > res['flat'][1] + 0.3 else 'no zero-shot advantage'}")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [res["flat"][0], res["flat"][1]], w, color=RED, label="flat (learned per-word emb)")
    ax.bar(x + w / 2, [res["grounded"][0], res["grounded"][1]], w, color=GREEN, label="grounded (WordNet dict)")
    ax.axhline(1 / (S * B), color="#999", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["SEEN-word pairs", "UNSEEN-word pairs\n(zero-shot)"])
    ax.set_ylabel("compositional accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Inherited dictionary (WordNet) generalizes to words never trained on; flat cannot",
                 color=NAVY, fontsize=10.5)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("sheaf_llm/grounded_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/grounded_gate.png")


if __name__ == "__main__":
    main()
