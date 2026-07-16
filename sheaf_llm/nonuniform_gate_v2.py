#!/usr/bin/env python
"""brick 3, honest rescue -- v2. Recurse on v1's failing pairs (control was invalid + sheaf undertrained).

v1 found: on a non-commutative fold, generic grounded-GRU extrapolated to depth (0.99) while the typed
sheaf fold decayed (0.80). BUT two instrument faults made it unsafe to conclude:
  (1) the commutative CONTROL was an UNWEIGHTED modular sum = a generalized-parity task GD-RNNs can't
      optimize -> both GRUs sat at chance, so it never produced the "both learn, architectures tie" null.
  (2) the sheaf was UNDERTRAINED at the comparison point (nonabelian loss 0.055 vs GRU 0.001); calling it
      "worse at depth" while it was still converging is an instrument error, not a finding.

v2 fixes both, plus the two verdict-reporting gaps the round-trip verifier named. THREE tasks now:
  abelian_wt   (CONTROL, learnable): val = (2*val + ss(subj) + verb) % S     # recursion_gate's exact fold;
                                                                             # WEIGHTED -> GRU-learnable -> valid H2 null
  abelian_sym  (clean twin, parity): val = (val + ss(subj) + verb) % S       # differs from nonabelian in EXACTLY
                                                                             # the operator, but is parity-hard (documents the cliff)
  nonabelian   (PAYOFF):             val = P[verb][ (val + ss(subj)) % S ]    # non-commutative; leaf val = ss(obj)

Arms identical to recursion_gate (flat-GRU / grounded-GRU / grounded-sheaf). Train depths {1,2,3};
test {1..6}; extrapolation = {4,5,6} x {seen,unseen} words. 2 SEEDS, mean reported. Converged (16k steps);
final training loss printed per arm so undertraining is visible.

PRE-REGISTERED (PREREG_nonuniform_composition_gate.md).
  H1 PAYOFF  : nonabelian, sheaf extrapolates to depth 4-6 (seen) better than gru (margin > 0.15) -- typed
               fold the lever. FALSIFIER: gru keeps up/beats -> typed sheaf not the lever (thread prior).
  H2 CONTROL : on abelian_wt (the LEARNABLE commutative anchor), sheaf ~= gru (|d|<=0.15) -- valid null.
               GUARDED: only asserted if the arms actually LEARN (best in-dist >= 0.30).
  H3 WORDS   : both grounded arms > flat on UNSEEN words, on every task, at extrapolation depth.

    python nonuniform_gate_v2.py
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
TRAIN_DEPTHS, TEST_DEPTHS, STEPS, BS, LR = [1, 2, 3], [1, 2, 3, 4, 5, 6], 16000, 128, 2e-3
SEEDS = [0, 1]
PERM_SEED = 12345
TASKS = ["abelian_wt", "abelian_sym", "nonabelian"]
TASK_TAG = {"abelian_wt": "CONTROL learnable (weighted, = recursion_gate)",
            "abelian_sym": "clean twin (unweighted -> parity-hard)",
            "nonabelian": "PAYOFF (non-commutative)"}
RED, AMBER, GREEN, NAVY = "#c0392b", "#9a6a2f", "#1e7d34", "#15293f"
PERM = np.stack([np.random.default_rng(PERM_SEED + v).permutation(S) for v in range(NV)])   # (NV,S) fixed


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
    rng = np.random.default_rng(0); chosen = {}
    for ss, ws in by.items():
        ws = sorted(ws); rng.shuffle(ws)
        for w in ws[:per_ss]:
            chosen[w] = vocab[w]
    qs = np.quantile([d for _, d in chosen.values()], [0.25, 0.5, 0.75])
    return {w: (ss, int(sum(d > q for q in qs))) for w, (ss, d) in chosen.items()}


def fold(val, sv, vb_i, task):
    if task == "abelian_wt":
        return (2 * val + sv + vb_i) % S      # weighted commutative -> GRU-learnable (recursion_gate)
    if task == "abelian_sym":
        return (val + sv + vb_i) % S          # unweighted commutative -> parity-hard
    return PERM[vb_i, (val + sv) % S]         # non-commutative permutation


def main():
    feat = build_vocab(); words = sorted(feat)
    split_rng = np.random.default_rng(0); split_rng.shuffle(words)
    ns = int(0.8 * len(words)); seen, unseen = words[:ns], words[ns:]
    wid = {w: i for i, w in enumerate(words)}; UNK = len(words)
    ss_of = {w: feat[w][0] for w in words}; dp_of = {w: feat[w][1] for w in words}
    seen_set = set(seen)
    print(f"[nonuniform gate v2]  dev={DEV}  vocab={len(words)} (seen {len(seen)}/unseen {len(unseen)})  "
          f"S={S} verbs={NV}  train {TRAIN_DEPTHS}->test {TEST_DEPTHS}  steps={STEPS} seeds={SEEDS}  chance={1/S:.3f}")

    class RNGState:
        rng = None
    st = RNGState()

    def gen(pool, depth, n, task):
        subj = [[pool[i] for i in st.rng.integers(0, len(pool), n)] for _ in range(depth)]
        verb = st.rng.integers(0, NV, (depth, n))
        obj = [pool[i] for i in st.rng.integers(0, len(pool), n)]
        val = np.array([ss_of[w] for w in obj])
        for i in reversed(range(depth)):
            sv = np.array([ss_of[w] for w in subj[i]])
            val = fold(val, sv, verb[i], task)
        return subj, verb, obj, torch.tensor(val, device=DEV)

    def sheaf_inputs(subj, verb, obj):
        T = lambda a: torch.tensor(a, device=DEV)
        return ([T([ss_of[w] for w in lvl]) for lvl in subj],
                [T([dp_of[w] for w in lvl]) for lvl in subj],
                [T(v) for v in verb], T([ss_of[w] for w in obj]), T([dp_of[w] for w in obj]))

    def seq_inputs(subj, verb, obj, train):
        depth = len(subj); n = len(obj)
        cols = [("noun", obj)]
        for i in reversed(range(depth)):
            cols.append(("verb", verb[i])); cols.append(("noun", subj[i]))
        tok = torch.zeros((n, len(cols)), dtype=torch.long, device=DEV); typ = torch.zeros_like(tok)
        ssq = torch.zeros_like(tok); dpq = torch.zeros_like(tok)
        nm = torch.zeros((n, len(cols)), dtype=torch.bool, device=DEV)
        for c, (kind, vals) in enumerate(cols):
            if kind == "noun":
                tok[:, c] = torch.tensor([wid[w] if (train or w in seen_set) else UNK for w in vals], device=DEV)
                typ[:, c] = 0; nm[:, c] = True
                ssq[:, c] = torch.tensor([ss_of[w] for w in vals], device=DEV)
                dpq[:, c] = torch.tensor([dp_of[w] for w in vals], device=DEV)
            else:
                tok[:, c] = torch.tensor(vals, device=DEV); typ[:, c] = 1
        return tok, typ, ssq, dpq, nm

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
            self.ln = nn.LayerNorm(D)
            self.head = nn.Linear(D, S)
        def forward(self, s_ss, s_dp, vb, o_ss, o_dp):
            val = self.leaf(torch.cat([self.ess(o_ss), self.edp(o_dp)], -1))
            for i in reversed(range(len(vb))):
                s = torch.cat([self.ess(s_ss[i]), self.edp(s_dp[i])], -1)
                val = self.ln(self.compose(torch.cat([self.Rs(s), self.Rv(self.everb(vb[i])), self.Rc(val)], -1)))
            return self.head(val)

    CLS = {"flat": FlatGRU, "grounded-gru": GroundedGRU, "grounded-sheaf": GroundedSheaf}

    def run(kind, task, seed):
        st.rng = np.random.default_rng(seed + 1)                 # same stream per arm within a seed
        torch.manual_seed(seed)
        m = CLS[kind]().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        losses = []
        for step in range(STEPS):
            d = TRAIN_DEPTHS[st.rng.integers(0, len(TRAIN_DEPTHS))]
            subj, verb, obj, y = gen(seen, d, BS, task)
            logits = m(*sheaf_inputs(subj, verb, obj)) if kind == "grounded-sheaf" else m(*seq_inputs(subj, verb, obj, True))
            loss = F.cross_entropy(logits, y); loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())

        def acc(pool, depth):
            subj, verb, obj, y = gen(pool, depth, 2000, task)
            with torch.no_grad():
                logits = m(*sheaf_inputs(subj, verb, obj)) if kind == "grounded-sheaf" else m(*seq_inputs(subj, verb, obj, False))
                return (logits.argmax(-1) == y).float().mean().item()
        depths = {d: (acc(seen, d), acc(unseen, d)) for d in TEST_DEPTHS}
        return depths, sum(p.numel() for p in m.parameters()), float(np.mean(losses[-300:]))

    arms = ["flat", "grounded-gru", "grounded-sheaf"]
    tr_max = max(TRAIN_DEPTHS); extrap_ds = [d for d in TEST_DEPTHS if d > tr_max]
    # agg[task][arm][depth] = [(seen,unseen) per seed];  loss/params tracked too
    agg = {t: {k: {"depths": defaultdict(list), "loss": [], "params": 0} for k in arms} for t in TASKS}
    for task in TASKS:
        print(f"\n===== TASK: {task}  ({TASK_TAG[task]})  ({len(SEEDS)} seeds) =====")
        hdr = "  ".join(f"d{d}" for d in TEST_DEPTHS)
        print(f"{'model':16}{'params':>8}{'trainloss':>11}   SEEN by depth ({hdr})       UNSEEN by depth")
        for kind in arms:
            per_seed = [run(kind, task, s) for s in SEEDS]
            for depths, p, lo in per_seed:
                for d in TEST_DEPTHS:
                    agg[task][kind]["depths"][d].append(depths[d])
                agg[task][kind]["loss"].append(lo); agg[task][kind]["params"] = p
            sm = {d: float(np.mean([x[0] for x in agg[task][kind]["depths"][d]])) for d in TEST_DEPTHS}
            um = {d: float(np.mean([x[1] for x in agg[task][kind]["depths"][d]])) for d in TEST_DEPTHS}
            lo = float(np.mean(agg[task][kind]["loss"]))
            print(f"{kind:16}{p/1e3:>6.1f}k{lo:>11.3f}   " +
                  " ".join(f"{sm[d]:.2f}" for d in TEST_DEPTHS) + "     " +
                  " ".join(f"{um[d]:.2f}" for d in TEST_DEPTHS))

    def E(task, kind, col):   # extrapolation mean (seed-mean) over depths 4-6
        return float(np.mean([np.mean([x[col] for x in agg[task][kind]["depths"][d]]) for d in extrap_ds]))
    def I(task, kind, col):   # in-distribution mean over depths 1-3
        return float(np.mean([np.mean([x[col] for x in agg[task][kind]["depths"][d]]) for d in TRAIN_DEPTHS]))
    def L(task, kind):
        return float(np.mean(agg[task][kind]["loss"]))

    print(f"\n================  VERDICT  (train<= {tr_max}; extrapolation depths {extrap_ds}; chance {1/S:.3f})  ================")
    for task in TASKS:
        print(f"\n[{task}]  final train loss:  " + "  ".join(f"{k}={L(task,k):.3f}" for k in arms))
        print(f"          in-dist (d1-3 seen):  " + "  ".join(f"{k}={I(task,k,0):.2f}" for k in arms))
        for k in arms:
            print(f"            {k:16} extrap seen={E(task,k,0):.2f}  unseen={E(task,k,1):.2f}  (params {agg[task][k]['params']/1e3:.1f}k)")

    # H2 -- valid learnable commutative anchor, GUARDED
    wt_learn = max(I("abelian_wt", k, 0) for k in arms)
    wt_sh, wt_gg = E("abelian_wt", "grounded-sheaf", 0), E("abelian_wt", "grounded-gru", 0)
    print(f"\n  H2 CONTROL  abelian_wt (learnable): best in-dist {wt_learn:.2f}  ->  " + (
        "UNINFORMATIVE (arms did not learn)" if wt_learn < 0.30 else
        (f"sheaf {wt_sh:.2f} vs gru {wt_gg:.2f}: VALID NULL reproduced (|d|<=0.15) -- architectures tie on a learnable commutative fold"
         if abs(wt_sh - wt_gg) <= 0.15 else
         f"sheaf {wt_sh:.2f} vs gru {wt_gg:.2f}: arms DIFFER even on the learnable commutative anchor")))
    # abelian_sym cliff, documented
    print(f"  (twin)      abelian_sym (parity): best in-dist {max(I('abelian_sym',k,0) for k in arms):.2f}"
          f"  gru={I('abelian_sym','grounded-gru',0):.2f} sheaf={I('abelian_sym','grounded-sheaf',0):.2f}"
          f"  -> documents the commutative-twin parity cliff")
    # H1 payoff, with convergence shown
    na_sh, na_gg = E("nonabelian", "grounded-sheaf", 0), E("nonabelian", "grounded-gru", 0)
    na_learn = max(I("nonabelian", k, 0) for k in arms)
    conv = f"[converged? sheaf loss {L('nonabelian','grounded-sheaf'):.3f} / gru {L('nonabelian','grounded-gru'):.3f}, sheaf in-dist {I('nonabelian','grounded-sheaf',0):.2f}]"
    if na_learn < 0.30:
        print(f"  H1 PAYOFF   nonabelian: UNINFORMATIVE (best in-dist {na_learn:.2f}) {conv}")
    else:
        print(f"  H1 PAYOFF   nonabelian: sheaf {na_sh:.2f} vs gru {na_gg:.2f} (margin {na_sh-na_gg:+.2f}) {conv}  ->  " +
              ("TYPED FOLD IS THE LEVER on non-uniform composition (first structural-sheaf win)"
               if na_sh > na_gg + 0.15 else
               "FALSIFIER: generic recurrence >= typed sheaf even here -- typed composition not the lever"))
    # H3 -- full: both grounded arms vs flat, every task, unseen extrapolation
    print(f"  H3 WORDS    (grounded arms vs flat, UNSEEN extrapolation):")
    for task in TASKS:
        fl = E(task, "flat", 1); gg = E(task, "grounded-gru", 1); sh = E(task, "grounded-sheaf", 1)
        ok = (gg > fl + 0.15) and (sh > fl + 0.15)
        print(f"            {task:12} flat={fl:.2f}  gru={gg:.2f}  sheaf={sh:.2f}  -> "
              f"{'both grounded > flat (dictionary zero-shot win)' if ok else 'word-win NOT clean here'}")

    fig, axes = plt.subplots(len(TASKS), 2, figsize=(11, 12), sharex=True, sharey=True)
    for ri, task in enumerate(TASKS):
        for ci, (col, ttl) in enumerate([(0, "SEEN"), (1, "UNSEEN (zero-shot)")]):
            ax = axes[ri][ci]
            for kind, c in [("flat", RED), ("grounded-gru", AMBER), ("grounded-sheaf", GREEN)]:
                ys = [np.mean([x[col] for x in agg[task][kind]["depths"][d]]) for d in TEST_DEPTHS]
                ax.plot(TEST_DEPTHS, ys, "o-", color=c, label=kind)
            ax.axvspan(tr_max + 0.5, TEST_DEPTHS[-1] + 0.3, color="#f0e9da", alpha=0.6)
            ax.axhline(1 / S, ls=":", color="#999", lw=1); ax.set_ylim(0, 1.05)
            ax.set_title(f"{task} - {ttl}", color=NAVY, fontsize=9)
            if ri == len(TASKS) - 1:
                ax.set_xlabel("nesting depth")
    axes[0][0].legend(frameon=False, fontsize=8)
    for ri in range(len(TASKS)):
        axes[ri][0].set_ylabel("fold accuracy")
    fig.suptitle("Non-uniform composition gate v2 (2-seed, converged): does typed sheaf composition ever\n"
                 "beat generic recurrence at depth extrapolation?  (rows: learnable-abelian / parity-twin / non-abelian)",
                 color=NAVY, fontsize=11)
    fig.tight_layout(); fig.savefig("sheaf_llm/nonuniform_gate_v2.png", dpi=150); plt.close(fig)
    print("\n  wrote sheaf_llm/nonuniform_gate_v2.png")


if __name__ == "__main__":
    main()
