#!/usr/bin/env python
"""brick 3, honest rescue -- NON-UNIFORM (non-commutative) composition. The recursion gate fired the
falsifier on typed sheaf composition, but only on a UNIFORM (abelian mod-add) target that a generic
recurrence rolls out trivially. Its stated caveat: "a non-uniform compositional task is the only place
typed composition could still separate." This gate builds that target and runs the abelian one as an
IN-SCRIPT CONTROL, changing nothing else -- same arms, same architectures, same trees, same data stream.

Nested clauses:  (S V O) -> (S V (S V O)) -> ...   leaf val = ss(obj);   fold inner->outer:
    abelian    (CONTROL, commutative twin):   val <- (val + ss(subj) + verb) % S           # recursion-gate-style abelian fold
    nonabelian (PAYOFF, this gate):           val <- P[verb][ (val + ss(subj)) % S ]        # order/type matter
                                              P = NV fixed random permutations of {0..S-1}
Permutations don't commute => the folded value depends on the ORDERED sequence of (verb-type, subj-shift)
edges; a generic untyped recurrence cannot collapse it to one effective operator, while weight-tied
per-role restriction maps match the structure by construction.

Arms (identical to recursion_gate; only the TARGET's commutativity is flipped):
  flat-GRU        : token ids, generic recurrence;            unseen word -> UNK
  grounded-GRU    : grounded (supersense,depth) features, generic recurrence
  grounded-sheaf  : typed, WEIGHT-TIED restriction maps (R_subj,R_verb,R_child) folded along the tree

PRE-REGISTERED (see PREREG_nonuniform_composition_gate.md).
  H1 PAYOFF   : on nonabelian, grounded-sheaf extrapolates to unseen depth (4-6) on SEEN words better than
                grounded-GRU (margin > 0.15) -- typed tied fold finally the lever.
  H2 CONTROL  : on abelian, grounded-sheaf ~= grounded-GRU (reproduce the recursion-gate null) -- isolates
                non-commutativity as the cause of any H1 separation.
  Falsifier   : grounded-GRU keeps up on nonabelian too -> typed fold inert even when the task requires
                ordered typed composition -> the final nail. Live, acceptable, thread-prior-favored.

    python nonuniform_gate.py
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
TRAIN_DEPTHS, TEST_DEPTHS, STEPS, BS, LR, SEED = [1, 2, 3], [1, 2, 3, 4, 5, 6], 9000, 128, 2e-3, 0
PERM_SEED = 12345                       # verb permutations: fixed, identical across all arms and tasks
RED, AMBER, GREEN, NAVY = "#c0392b", "#9a6a2f", "#1e7d34", "#15293f"

# NV fixed random permutations of {0..S-1}, one per verb -- the non-abelian instruction set.
PERM = np.stack([np.random.default_rng(PERM_SEED + v).permutation(S) for v in range(NV)])   # (NV, S)


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
    print(f"[nonuniform gate]  dev={DEV}  vocab={len(words)} nouns (seen {len(seen)}/unseen {len(unseen)})  "
          f"S={S} verbs={NV}  train depths {TRAIN_DEPTHS} -> test {TEST_DEPTHS}  (chance={1/S:.3f})")

    def gen(pool, depth, n, task):
        # each example: d subjects, d verbs, 1 obj; recursive fold target (task-dependent operator)
        subj = [[pool[i] for i in rng.integers(0, len(pool), n)] for _ in range(depth)]
        verb = rng.integers(0, NV, (depth, n))
        obj = [pool[i] for i in rng.integers(0, len(pool), n)]
        val = np.array([ss_of[w] for w in obj])
        for i in reversed(range(depth)):                       # fold inner->outer
            sv = np.array([ss_of[w] for w in subj[i]])
            if task == "abelian":
                val = (val + sv + verb[i]) % S                 # commutative twin: verb enters as additive shift
            else:
                val = PERM[verb[i], (val + sv) % S]            # non-commutative: verb enters as a permutation
            # ^ the ONLY difference between tasks is add-vs-permute (commutativity); subj-shift identical
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

    def run(kind, task):
        nonlocal rng
        rng = np.random.default_rng(SEED + 1)                    # every arm gets the SAME train/test stream
        torch.manual_seed(SEED)
        m = {"flat": FlatGRU, "grounded-gru": GroundedGRU, "grounded-sheaf": GroundedSheaf}[kind]().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        run_loss = []
        for step in range(STEPS):
            d = TRAIN_DEPTHS[rng.integers(0, len(TRAIN_DEPTHS))]
            subj, verb, obj, y = gen(seen, d, BS, task)
            logits = m(*sheaf_inputs(subj, verb, obj)) if kind == "grounded-sheaf" else m(*seq_inputs(subj, verb, obj, True))
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step(); opt.zero_grad(); run_loss.append(loss.item())
            if (step + 1) % 3000 == 0:
                print(f"    [{task}/{kind} step {step+1}] loss={np.mean(run_loss[-500:]):.3f}")

        def acc(pool, depth):
            subj, verb, obj, y = gen(pool, depth, 2000, task)
            with torch.no_grad():
                logits = m(*sheaf_inputs(subj, verb, obj)) if kind == "grounded-sheaf" else m(*seq_inputs(subj, verb, obj, False))
                return (logits.argmax(-1) == y).float().mean().item()
        return {d: (acc(seen, d), acc(unseen, d)) for d in TEST_DEPTHS}, sum(p.numel() for p in m.parameters())

    arms = ["flat", "grounded-gru", "grounded-sheaf"]
    tr_max = max(TRAIN_DEPTHS)
    extrap_depths = [d for d in TEST_DEPTHS if d > tr_max]
    results = {}
    for task in ["abelian", "nonabelian"]:
        res, params = {}, {}
        hdr = "  ".join(f"d{d}" for d in TEST_DEPTHS)
        tag = "CONTROL (commutative, = recursion_gate)" if task == "abelian" else "PAYOFF (non-commutative)"
        print(f"\n===== TASK: {task}  {tag} =====")
        print(f"{'model':16}{'params':>9}   SEEN words by depth ({hdr})        UNSEEN words by depth")
        for kind in arms:
            r, p = run(kind, task); res[kind] = r; params[kind] = p
            seen_s = " ".join(f"{r[d][0]:.2f}" for d in TEST_DEPTHS)
            uns_s = " ".join(f"{r[d][1]:.2f}" for d in TEST_DEPTHS)
            print(f"{kind:16}{p/1e3:>7.1f}k   {seen_s}     {uns_s}")
        results[task] = (res, params)

    def extrap(task, kind, col):
        res = results[task][0]
        return float(np.mean([res[kind][d][col] for d in extrap_depths]))
    def indist(task, kind, col):
        res = results[task][0]
        return float(np.mean([res[kind][d][col] for d in TRAIN_DEPTHS]))

    print(f"\n================  VERDICT  (train depths <= {tr_max}; extrapolation = depths {extrap_depths}; chance {1/S:.3f})  ================")
    for task in ["abelian", "nonabelian"]:
        print(f"\n[{task}]  in-distribution (d1-3, seen) fit:  " +
              "  ".join(f"{k}={indist(task,k,0):.2f}" for k in arms))
        print(f"          extrapolation (d4-6) seen / unseen:")
        for k in arms:
            print(f"            {k:16} seen={extrap(task,k,0):.2f}  unseen={extrap(task,k,1):.2f}  (params {results[task][1][k]/1e3:.1f}k)")

    ab_sh, ab_gg = extrap("abelian", "grounded-sheaf", 0), extrap("abelian", "grounded-gru", 0)
    na_sh, na_gg = extrap("nonabelian", "grounded-sheaf", 0), extrap("nonabelian", "grounded-gru", 0)
    na_learn = max(indist("nonabelian", k, 0) for k in arms)
    print(f"\n  H2 CONTROL   abelian:   sheaf {ab_sh:.2f} vs gru {ab_gg:.2f}  ->  "
          f"{'reproduces the null (|d|<=0.15), architecture isolated' if abs(ab_sh-ab_gg)<=0.15 else 'arms DIFFER on abelian -- H1 not cleanly attributable'}")
    if na_learn < 0.30:
        print(f"  H1 PAYOFF    nonabelian: UNINFORMATIVE -- no arm fits trained depths (best in-dist {na_learn:.2f} ~ chance); task too hard at this scale")
    else:
        print(f"  H1 PAYOFF    nonabelian: sheaf {na_sh:.2f} vs gru {na_gg:.2f}  (margin {na_sh-na_gg:+.2f})  ->  "
              f"{'TYPED FOLD IS THE LEVER on non-uniform composition (first structural-sheaf win)' if na_sh > na_gg + 0.15 else 'FALSIFIER: generic recurrence keeps up even here -- typed sheaf inert (final nail)'}")
    print(f"  H3 WORDS     nonabelian unseen: sheaf {extrap('nonabelian','grounded-sheaf',1):.2f} vs flat {extrap('nonabelian','flat',1):.2f}  ->  "
          f"{'dictionary still gives zero-shot words' if extrap('nonabelian','grounded-sheaf',1) > extrap('nonabelian','flat',1)+0.15 else 'no word advantage'}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    rows = [("abelian", "CONTROL: commutative (= recursion_gate)"), ("nonabelian", "PAYOFF: non-commutative (typed composition needed)")]
    for ri, (task, rttl) in enumerate(rows):
        res = results[task][0]
        for ci, (col, ttl) in enumerate([(0, "SEEN words"), (1, "UNSEEN words (zero-shot)")]):
            ax = axes[ri][ci]
            for kind, c in [("flat", RED), ("grounded-gru", AMBER), ("grounded-sheaf", GREEN)]:
                ax.plot(TEST_DEPTHS, [res[kind][d][col] for d in TEST_DEPTHS], "o-", color=c, label=kind)
            ax.axvspan(tr_max + 0.5, TEST_DEPTHS[-1] + 0.3, color="#f0e9da", alpha=0.6)
            ax.axhline(1 / S, ls=":", color="#999", lw=1)
            ax.set_ylim(0, 1.05)
            if ri == 1:
                ax.set_xlabel("nesting depth")
            ax.set_title(f"{rttl.split(':')[0]} - {ttl}", color=NAVY, fontsize=9)
    axes[0][0].set_ylabel("fold accuracy"); axes[1][0].set_ylabel("fold accuracy")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Non-uniform composition gate: does typed sheaf composition separate from generic recurrence\n"
                 "when the target is NON-commutative (top row = commutative control)?", color=NAVY, fontsize=11)
    fig.tight_layout(); fig.savefig("sheaf_llm/nonuniform_gate.png", dpi=160); plt.close(fig)
    print("\n  wrote sheaf_llm/nonuniform_gate.png")


if __name__ == "__main__":
    main()
