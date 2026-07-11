#!/usr/bin/env python
"""brick 3 -- recursion/depth. Do grounded sheaf restriction maps COMPOSE across nested structure and
EXTRAPOLATE to depth beyond training? Brick 2 showed roles compose at ONE level; this is the property
no gate has tested and the one a real grounded LM needs.

Nested clauses:  (S V O) -> (S V (S V O)) -> (S V (S V (S V O))) ...
Recursive role-weighted target (MUST respect nesting):
    val(leaf obj) = ss(obj);   val(clause) = (ss(subj) + 2*val(child) + verb) % S;   target = val(root)

Train on shallow depths {1,2,3}; test on DEEPER depths {4,5,6} (extrapolation) AND on unseen words.
Arms (fold-order sequences, so the GRU baseline CAN in principle run the recurrence -- a fair test):
  flat-GRU        : token ids, generic recurrence;            unseen word -> UNK
  grounded-GRU    : grounded (supersense,depth) features, generic recurrence
  grounded-sheaf  : typed, WEIGHT-TIED restriction maps (R_subj,R_verb,R_child) folded along the tree

PRE-REGISTERED.
  Hypothesis: grounded-sheaf extrapolates to unseen DEPTH (tied maps = depth-invariant) AND unseen WORDS
              (dictionary); the GRUs fit in-distribution but decay on deeper trees; flat also dies on
              unseen words.
  Falsifier : grounded-GRU extrapolates to depth as well as grounded-sheaf -> the typed tied fold adds
              nothing; generic recurrence suffices.

    python recursion_gate.py
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
try:
    wn.ensure_loaded()
except LookupError:
    import nltk
    nltk.download("wordnet", quiet=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SS = ["noun.animal", "noun.artifact", "noun.food", "noun.plant",
      "noun.body", "noun.location", "noun.person", "noun.substance"]
S, B, NV, D, HID = len(SS), 4, 8, 64, 128
TRAIN_DEPTHS, TEST_DEPTHS, STEPS, BS, LR, SEED = [1, 2, 3], [1, 2, 3, 4, 5, 6], 7000, 128, 2e-3, 0
RED, AMBER, GREEN, NAVY = "#c0392b", "#9a6a2f", "#1e7d34", "#15293f"


def build_vocab(per_ss=40):
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
    rng = np.random.default_rng(SEED); chosen = {}
    for ss, ws in by.items():
        ws = sorted(ws); rng.shuffle(ws)
        for w in ws[:per_ss]:
            chosen[w] = vocab[w]
    qs = np.quantile([d for _, d in chosen.values()], [0.25, 0.5, 0.75])
    return {w: (ss, int(sum(d > q for q in qs))) for w, (ss, d) in chosen.items()}


def main():
    feat = build_vocab(); words = sorted(feat)
    rng = np.random.default_rng(SEED); rng.shuffle(words)
    ns = int(0.8 * len(words)); seen, unseen = words[:ns], words[ns:]
    wid = {w: i for i, w in enumerate(words)}; UNK = len(words)
    ss_of = {w: feat[w][0] for w in words}; dp_of = {w: feat[w][1] for w in words}
    print(f"[recursion gate]  dev={DEV}  vocab={len(words)} nouns (seen {len(seen)}/unseen {len(unseen)})  "
          f"S={S} verbs={NV}  train depths {TRAIN_DEPTHS} -> test {TEST_DEPTHS}  (chance={1/S:.3f})")

    def gen(pool, depth, n):
        # each example: d subjects, d verbs, 1 obj; recursive fold target
        subj = [[pool[i] for i in rng.integers(0, len(pool), n)] for _ in range(depth)]
        verb = rng.integers(0, NV, (depth, n))
        obj = [pool[i] for i in rng.integers(0, len(pool), n)]
        val = np.array([ss_of[w] for w in obj])
        for i in reversed(range(depth)):                       # fold inner->outer
            sv = np.array([ss_of[w] for w in subj[i]])
            val = (sv + 2 * val + verb[i]) % S
        return subj, verb, obj, torch.tensor(val, device=DEV)

    def sheaf_inputs(subj, verb, obj):
        T = lambda arr: torch.tensor(arr, device=DEV)
        s_ss = [T([ss_of[w] for w in lvl]) for lvl in subj]
        s_dp = [T([dp_of[w] for w in lvl]) for lvl in subj]
        vb = [T(v) for v in verb]
        return s_ss, s_dp, vb, T([ss_of[w] for w in obj]), T([dp_of[w] for w in obj])

    def seq_inputs(subj, verb, obj, train):
        # fold order: obj, then (verb_i, subj_i) inner->outer. token ids + grounded feats + type(0 noun,1 verb)
        depth = len(subj); n = len(obj)
        noun_tok, vb_tok, types, ss_seq, dp_seq, is_noun = [], [], [], [], [], []
        cols = [("noun", obj)]
        for i in reversed(range(depth)):
            cols.append(("verb", verb[i])); cols.append(("noun", subj[i]))
        tok = torch.full((n, len(cols)), 0, device=DEV); typ = torch.zeros((n, len(cols)), dtype=torch.long, device=DEV)
        ssq = torch.zeros((n, len(cols)), dtype=torch.long, device=DEV); dpq = torch.zeros_like(ssq)
        nounmask = torch.zeros((n, len(cols)), dtype=torch.bool, device=DEV)
        for c, (kind, vals) in enumerate(cols):
            if kind == "noun":
                ids = torch.tensor([wid[w] if (train or w in set(seen)) else UNK for w in vals], device=DEV)
                tok[:, c] = ids; typ[:, c] = 0; nounmask[:, c] = True
                ssq[:, c] = torch.tensor([ss_of[w] for w in vals], device=DEV)
                dpq[:, c] = torch.tensor([dp_of[w] for w in vals], device=DEV)
            else:
                tok[:, c] = torch.tensor(vals, device=DEV); typ[:, c] = 1
        return tok, typ, ssq, dpq, nounmask

    class FlatGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.noun = nn.Embedding(len(words) + 1, D); self.verb = nn.Embedding(NV, D)
            self.typ = nn.Embedding(2, D); self.gru = nn.GRU(D, HID, batch_first=True); self.head = nn.Linear(HID, S)
        def forward(self, tok, typ, ssq, dpq, nm):
            e = torch.where(nm[..., None], self.noun(tok), self.verb(tok.clamp(max=NV - 1))) + self.typ(typ)
            return self.head(self.gru(e)[0][:, -1])

    class GroundedGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.ess, self.edp, self.verb = nn.Embedding(S, D), nn.Embedding(B, D), nn.Embedding(NV, D)
            self.proj = nn.Linear(2 * D, D); self.typ = nn.Embedding(2, D)
            self.gru = nn.GRU(D, HID, batch_first=True); self.head = nn.Linear(HID, S)
        def forward(self, tok, typ, ssq, dpq, nm):
            noun_e = self.proj(torch.cat([self.ess(ssq), self.edp(dpq)], -1))
            e = torch.where(nm[..., None], noun_e, self.verb(tok.clamp(max=NV - 1))) + self.typ(typ)
            return self.head(self.gru(e)[0][:, -1])

    class GroundedSheaf(nn.Module):
        def __init__(self):
            super().__init__()
            self.ess, self.edp, self.everb = nn.Embedding(S, D), nn.Embedding(B, D), nn.Embedding(NV, D)
            self.leaf = nn.Sequential(nn.Linear(2 * D, D), nn.GELU())
            self.Rs, self.Rv, self.Rc = nn.Linear(2 * D, D), nn.Linear(D, D), nn.Linear(D, D)   # tied across depth
            self.compose = nn.Sequential(nn.Linear(3 * D, HID), nn.GELU(), nn.Linear(HID, D))
            self.ln = nn.LayerNorm(D)              # normalize the recursive value -> depth-consistent manifold
            self.head = nn.Linear(D, S)
        def forward(self, s_ss, s_dp, vb, o_ss, o_dp):
            val = self.leaf(torch.cat([self.ess(o_ss), self.edp(o_dp)], -1))
            for i in reversed(range(len(vb))):
                s = torch.cat([self.ess(s_ss[i]), self.edp(s_dp[i])], -1)
                val = self.ln(self.compose(torch.cat([self.Rs(s), self.Rv(self.everb(vb[i])), self.Rc(val)], -1)))
            return self.head(val)

    def run(kind):
        nonlocal rng
        rng = np.random.default_rng(SEED + 1)                    # PR#13 review: every arm gets the
        torch.manual_seed(SEED)                                  # SAME generated train/test stream
        m = {"flat": FlatGRU, "grounded-gru": GroundedGRU, "grounded-sheaf": GroundedSheaf}[kind]().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        run_loss = []
        for step in range(STEPS):
            d = TRAIN_DEPTHS[rng.integers(0, len(TRAIN_DEPTHS))]
            subj, verb, obj, y = gen(seen, d, BS)
            if kind == "grounded-sheaf":
                logits = m(*sheaf_inputs(subj, verb, obj))
            else:
                logits = m(*seq_inputs(subj, verb, obj, True))
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step(); opt.zero_grad(); run_loss.append(loss.item())
            if (step + 1) % 1750 == 0:
                print(f"    [{kind} step {step+1}] loss={np.mean(run_loss[-500:]):.3f}")

        def acc(pool, depth):
            subj, verb, obj, y = gen(pool, depth, 2000)
            with torch.no_grad():
                if kind == "grounded-sheaf":
                    logits = m(*sheaf_inputs(subj, verb, obj))
                else:
                    logits = m(*seq_inputs(subj, verb, obj, False))
                return (logits.argmax(-1) == y).float().mean().item()
        return {d: (acc(seen, d), acc(unseen, d)) for d in TEST_DEPTHS}, sum(p.numel() for p in m.parameters())

    arms = ["flat", "grounded-gru", "grounded-sheaf"]
    res = {}
    hdr = "  ".join(f"d{d}" for d in TEST_DEPTHS)
    print(f"\n{'model':16}{'params':>9}   SEEN words by depth ({hdr})        UNSEEN words by depth")
    for kind in arms:
        r, p = run(kind); res[kind] = r
        seen_s = " ".join(f"{r[d][0]:.2f}" for d in TEST_DEPTHS)
        uns_s = " ".join(f"{r[d][1]:.2f}" for d in TEST_DEPTHS)
        print(f"{kind:16}{p/1e3:>7.1f}k   {seen_s}     {uns_s}")

    tr_max = max(TRAIN_DEPTHS)
    def extrap(kind, col):
        return np.mean([res[kind][d][col] for d in TEST_DEPTHS if d > tr_max])
    print(f"\nVERDICT (train depths <= {tr_max}; extrapolation = depths {[d for d in TEST_DEPTHS if d>tr_max]}; chance {1/S:.2f}):")
    for kind in arms:
        print(f"  {kind:16} extrapolation acc  seen={extrap(kind,0):.2f}  unseen={extrap(kind,1):.2f}")
    sh, gg, fl = extrap("grounded-sheaf", 0), extrap("grounded-gru", 0), extrap("flat", 0)
    print(f"  DEPTH: grounded-sheaf {sh:.2f} vs grounded-gru {gg:.2f}  -> "
          f"{'typed tied fold EXTRAPOLATES where generic recurrence decays' if sh > gg + 0.15 else 'generic recurrence keeps up (tied fold not the lever)'}")
    print(f"  WORDS: grounded-sheaf unseen {extrap('grounded-sheaf',1):.2f} vs flat unseen {extrap('flat',1):.2f}  -> "
          f"{'dictionary still gives zero-shot words at depth' if extrap('grounded-sheaf',1) > extrap('flat',1)+0.15 else 'no word advantage'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, ttl in [(axes[0], 0, "SEEN words"), (axes[1], 1, "UNSEEN words (zero-shot)")]:
        for kind, c in [("flat", RED), ("grounded-gru", AMBER), ("grounded-sheaf", GREEN)]:
            ax.plot(TEST_DEPTHS, [res[kind][d][col] for d in TEST_DEPTHS], "o-", color=c, label=kind)
        ax.axvspan(tr_max + 0.5, TEST_DEPTHS[-1] + 0.3, color="#f0e9da", alpha=0.6)
        ax.axhline(1 / S, ls=":", color="#999", lw=1)
        ax.set_xlabel("nesting depth"); ax.set_title(ttl, color=NAVY, fontsize=10); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("recursive-fold accuracy"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].text(tr_max + 1.2, 0.92, "extrapolation\n(unseen depth)", fontsize=8, color="#9a6a2f")
    fig.suptitle("Recursion gate: does grounded sheaf composition extrapolate to depth beyond training?",
                 color=NAVY, fontsize=11)
    fig.tight_layout(); fig.savefig("sheaf_llm/recursion_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/recursion_gate.png")


if __name__ == "__main__":
    main()
