#!/usr/bin/env python
"""H1 routing-gate -- does a global-gluing (Cech H1-style) obstruction beat a PAIRWISE-consistency
baseline at flagging where a model is in trouble? Per PREREG_h1_routing_gate.md. [GATE], not design.

The obstruction we compute is the GLOBAL-SECTION / strong-contextuality obstruction: local sections agree
pairwise but cannot all glue into one global section (the H1-neq-0 regime). The decisive question (H2) is
whether this HIGHER-ORDER signal predicts difficulty BEYOND the cheap first-order pairwise baseline --
exactly the shape of the subword gate (rigorous higher-order structure can be real yet add ~nothing).

  Stage 1  anchors: PR box (proven contextual) -> no global section though pairwise-OK; classical -> glues.
           Validates the math AND shows H1 strictly beats pairwise ON CONTEXTUAL SYSTEMS (by construction).
  Stage 3  model-in-the-loop (the real, non-circular test): GPT-2 on Brown. For each target token, K
           left-context windows give possibilistic supports (top-p next-token sets). pairwise = mean
           support disagreement (first-order). obstruction = no token common to ALL windows / contextual
           fraction (higher-order). LABEL = true-token surprisal under full context (independent). H2:
           does obstruction add over pairwise at predicting high surprisal?
  Stage 2  garden-path illustration (n small, labeled): same machinery on classic garden-paths vs controls.

  Falsifier (H2): AUROC(pairwise+obstruction) ~= AUROC(pairwise) -> obstruction operationally inert; H1
                  joins entropy-reg/natural-gradient as "real, known, not the lever". Does NOT enter design.

    python h1_gate.py
"""
import itertools, math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import brown

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPT2 = "C:/Users/JT-DEV1/Documents/gpt2-sm"
LS = [4, 8, 16, 32]          # context-window lengths = the "contexts" of the cover
LFULL, TOPP, CAP = 48, 0.9, 40
NAVY = "#15293f"


# ---------- obstruction primitives (possibilistic sheaf / global section) ----------
def restrict(assign, names, sub):
    return tuple(assign[names.index(v)] for v in sub)

def global_sections(contexts, supports, var_domains):
    names = sorted({v for c in contexts for v in c})
    doms = [var_domains[v] for v in names]
    out = []
    for assign in itertools.product(*doms):
        if all(restrict(assign, names, c) in supports[i] for i, c in enumerate(contexts)):
            out.append(assign)
    return names, out

def pairwise_consistent(contexts, supports):
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            o = [v for v in contexts[i] if v in contexts[j]]
            if not o:
                continue
            pi = {restrict(s, contexts[i], o) for s in supports[i]}
            pj = {restrict(s, contexts[j], o) for s in supports[j]}
            if pi.isdisjoint(pj):
                return False
    return True


def stage1_anchors():
    print("=== Stage 1: anchors (validate the obstruction; H1 vs pairwise on KNOWN systems) ===")
    dom = {v: (0, 1) for v in ["A0", "A1", "B0", "B1"]}
    ctx = [("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")]
    # PR box: a XOR b == x AND y
    pr = []
    for (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        pr.append({(a, b) for a in (0, 1) for b in (0, 1) if (a ^ b) == (x & y)})
    # classical deterministic model (a=b=0 everywhere) -> single global section
    cl = [{(0, 0)} for _ in ctx]
    for name, sup in [("PR box (proven contextual)", pr), ("classical deterministic", cl)]:
        _, gs = global_sections(ctx, sup, dom); pw = pairwise_consistent(ctx, sup)
        contextual = pw and not gs
        print(f"  {name:28}  global_sections={len(gs)}  pairwise_consistent={pw}  "
              f"H1-obstruction(contextual)={contextual}")
    print("  -> PR: pairwise says OK but NO global section (obstruction fires). classical: glues. "
          "Higher-order strictly beats pairwise HERE by construction.\n")


# ---------- model-in-the-loop ----------
def topp_sets(logits, p=TOPP, cap=CAP):
    probs = F.softmax(logits.float(), -1)
    vals, idx = probs.topk(cap, dim=-1)
    cum = vals.cumsum(-1)
    keep = (cum - vals) < p
    keep[:, 0] = True
    idx, keep = idx.cpu().numpy(), keep.cpu().numpy()
    return [frozenset(idx[r][keep[r]].tolist()) for r in range(idx.shape[0])]


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def features(support_lists):
    # support_lists: list over K windows, each a list-over-targets of frozensets
    K, N = len(support_lists), len(support_lists[0])
    pair, ctxfrac, noglob = np.zeros(N), np.zeros(N), np.zeros(N)
    for t in range(N):
        S = [support_lists[k][t] for k in range(K)]
        js = [jaccard(S[i], S[j]) for i in range(K) for j in range(i + 1, K)]
        pair[t] = 1.0 - float(np.mean(js))                      # first-order: pairwise disagreement
        cnt = Counter()
        for s in S:
            cnt.update(s)
        max_share = max(cnt.values()) if cnt else 0
        ctxfrac[t] = 1.0 - max_share / K                        # higher-order: cannot all glue
        noglob[t] = 1.0 if max_share < K else 0.0               # binary H1-style veto
    return pair, ctxfrac, noglob


def get_logits(model, ids_batch):
    with torch.no_grad():
        return model(ids_batch.to(DEV)).logits[:, -1, :]


def stage3_model_gate(model, seq, n_targets=1500, stride=3):
    cand = list(range(LFULL, len(seq) - 1, stride))
    rng = np.random.default_rng(0); rng.shuffle(cand)
    T = sorted(cand[:n_targets])
    true_next = torch.tensor([seq[t] for t in T])
    supports = {L: [] for L in LS}; surpr = []
    CH = 256
    for c0 in range(0, len(T), CH):
        chunk = T[c0:c0 + CH]
        for L in LS:
            batch = torch.tensor([seq[t - L:t] for t in chunk])
            supports[L].extend(topp_sets(get_logits(model, batch)))
        fb = torch.tensor([seq[t - LFULL:t] for t in chunk])
        lp = F.log_softmax(get_logits(model, fb).float(), -1).cpu()
        surpr.extend((-lp[range(len(chunk)), [seq[t] for t in chunk]]).tolist())
    surpr = np.array(surpr)
    pair, ctxfrac, noglob = features([supports[L] for L in LS])
    hard = (surpr >= np.quantile(surpr, 0.75)).astype(int)       # independent label: top-quartile surprisal

    def auc(x):
        return roc_auc_score(hard, x)
    print("=== Stage 3: model-in-the-loop (GPT-2 / Brown) -- the non-circular H2 test ===")
    print(f"  N={len(T)} targets  K={len(LS)} context windows {LS}  label=top-quartile true-token surprisal")
    print(f"  Spearman-ish AUROC at predicting HIGH surprisal:")
    print(f"    pairwise-disagreement (first-order baseline): {auc(pair):.3f}")
    print(f"    contextual-fraction   (higher-order/H1):      {auc(ctxfrac):.3f}")
    print(f"    no-global-section veto (binary H1):           {auc(noglob):.3f}")
    # H2: does higher-order ADD over pairwise? train/test logistic, test-AUROC
    Xp = pair[:, None]; Xb = np.column_stack([pair, ctxfrac])
    n = len(T); idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]
    def fit_auc(X):
        m = LogisticRegression(max_iter=500).fit(X[tr], hard[tr])
        return roc_auc_score(hard[te], m.predict_proba(X[te])[:, 1])
    a_p, a_b = fit_auc(Xp), fit_auc(Xb)
    print(f"\n  H2 (held-out test AUROC):  pairwise-only {a_p:.3f}  ->  pairwise+H1 {a_b:.3f}   "
          f"(increment {a_b - a_p:+.3f})")
    verdict = ("H1 ADDS over pairwise -> earns a place in routing (prong 3 alive)" if a_b - a_p > 0.01
               else "H1 INERT over pairwise -> joins entropy-reg/nat-grad as real-but-not-the-lever (FALSIFIED as a veto)")
    print(f"  VERDICT: {verdict}")
    # base rates of the binary veto
    fire = noglob.astype(bool)
    print(f"  binary veto fires on {fire.mean()*100:.0f}% of positions; P(hard|veto)={hard[fire].mean():.2f} "
          f"vs P(hard|no-veto)={hard[~fire].mean():.2f} vs base {hard.mean():.2f}")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    names = ["pairwise\n(first-order\nbaseline)", "contextual-frac\n(H1 higher-order)", "no-global veto\n(binary H1)"]
    ax.bar(names, [auc(pair), auc(ctxfrac), auc(noglob)], color=["#2c6fbb", "#c0392b", "#c0392b"])
    ax.axhline(0.5, ls=":", color="#999", label="chance")
    ax.set_ylim(0.45, 0.65); ax.set_ylabel("AUROC: predict high-surprisal positions")
    ax.set_title("H1 routing gate: the higher-order obstruction is INERT vs the pairwise baseline",
                 color=NAVY, fontsize=9.5)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("sheaf_llm/h1_gate.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/h1_gate.png")
    return a_p, a_b


def stage2_gardenpath(model, tok):
    print("\n=== Stage 2: garden-path illustration (n small -- ILLUSTRATIVE, not evidence) ===")
    gp = ["The horse raced past the barn fell.",
          "The old man the boats.",
          "The complex houses married and single soldiers.",
          "The cotton clothing is made of grows in Mississippi.",
          "Until the police arrest the drug dealers control the street."]
    ctl = ["The horse raced past the barn quickly.",
           "The old sailor sailed the boats.",
           "The tall building houses many married soldiers.",
           "The cotton fabric that clothing is made of feels soft.",
           "Until the police arrived the drug dealers ran from the street."]
    PRE = tok("In the report that follows, please read the next sentence slowly and consider what each "
              "word is doing in its context before deciding what comes next: ", return_tensors="pt").input_ids[0].tolist()
    def mean_obstruction(sent):
        ids = PRE + tok(sent, return_tensors="pt").input_ids[0].tolist()   # fixed preamble so windows always fit
        Ts = list(range(max(len(PRE), LS[-1]), len(ids)))                 # obstruction over sentence region; windows fit
        if not Ts:
            return float("nan"), float("nan")
        sup = {L: [] for L in LS}
        for L in LS:
            b = torch.tensor([ids[t - L:t] for t in Ts])
            sup[L].extend(topp_sets(get_logits(model, b)))
        _, ctxfrac, _ = features([sup[L] for L in LS])
        return float(ctxfrac.max()), float(ctxfrac.mean())     # peak + mean cross-window irreconcilability
    gpa = np.array([mean_obstruction(s) for s in gp]); cta = np.array([mean_obstruction(s) for s in ctl])
    print(f"  garden-path  peak/mean contextual-fraction: {gpa[:,0].mean():.2f} / {gpa[:,1].mean():.2f}")
    print(f"  control      peak/mean contextual-fraction: {cta[:,0].mean():.2f} / {cta[:,1].mean():.2f}")
    print(f"  -> {'garden-paths show higher peak irreconcilability (suggestive)' if gpa[:,0].mean() > cta[:,0].mean() else 'no separation'}"
          f"  [n=5 each; illustrative only -- the modeling/encoding does the work, see PREREG limits]")


def main():
    stage1_anchors()
    nltk.download("brown", quiet=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GPT2)
    model = AutoModelForCausalLM.from_pretrained(GPT2).to(DEV).eval()
    text = " ".join(brown.words()[200000:260000])
    seq = tok(text, return_tensors="pt").input_ids[0].tolist()
    stage3_model_gate(model, seq)
    stage2_gardenpath(model, tok)


if __name__ == "__main__":
    main()
