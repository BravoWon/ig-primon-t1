#!/usr/bin/env python
"""DeltaSheaf-v0.2 — Phase-1 INSTRUMENT SMOKE (synthetic, no real models).

Validate the gate BEFORE spending hours on 5-model inference (SPEC principle: validate the instrument
before the finding). Scaled-down (d=128 vs 512) but the logic is identical. Four synthetic regimes; the
gate MUST return the right verdict on each, or the instrument is broken:

  R_signal     gold planted in the CYCLE residuals, edges swamped by per-node gauge (phi) that CANCELS in
               the coboundary -> C_cycle recovers gold, beats B_edge/ctrl-mag/ctrl-blind/ctrl-shuffle -> PASS
  R_firstorder gold recoverable directly from edges (small gauge) -> C_cycle ~= B_edge -> FALSIFIED
               (the gate must DECLINE to credit topology when first-order already explains it)
  R_noise      no gold in the deltas -> every arm ~ chance -> FALSIFIED (no false positive)
  R_leak       no real signal, but the gold-option distribution is shifted -> a blind/constant predictor
               beats chance -> H4 leak sentinel fires -> VOID

Instrument finding surfaced by this smoke and reported up: on a blind-spot set (0-of-N correct BY
CONSTRUCTION), majority A = 0% always, so the frozen SPEC's 'beat A' and 'ctrl-blind <= A + eps' thresholds
are DEGENERATE. The sound verdict is CHANCE-relative (chance = 1/n_options). Implemented that way here.
"""
import itertools, math, sys
import numpy as np
import torch
import torch.nn as nn

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DEV = "cpu"
N, D, NOPT, DSIG = 5, 128, 4, 128
N_CLEAN, N_GATE = 800, 300
SEEDS = [0, 1, 2, 3, 4]
TEMP = 0.1
CHANCE = 1.0 / NOPT
DMIN, EPS = 0.10, 0.02
TRIANGLES = list(itertools.combinations(range(N), 3))   # C(5,3)=10
EDGES = list(itertools.combinations(range(N), 2))        # C(5,2)=10  (i<j)
# fixed edge scalars with a nonzero coboundary (so cycle residual carries a gold term)
TT = {(i, j): float((i + 1) * (j + 2) % 7 + 1) for (i, j) in EDGES}
# fixed per-POSITION identity embeddings: real answer-priors ride a consistent axis (option letter/position),
# which is what lets a blind/constant predictor exploit a shifted gold distribution (the H4 leak).
POS = (np.random.default_rng(777).standard_normal((NOPT, D)).astype(np.float32))

def rng_of(seed): return np.random.default_rng(seed)

def make_regime(regime, n, seed):
    """Return deltas[n,N,N,D] (i<j filled), options[n,NOPT,D], gold_idx[n], is_gate flag handled by caller."""
    g = rng_of(seed)
    phi_scale = {"R_signal": 6.0, "R_firstorder": 0.15, "R_noise": 0.15, "R_leak": 0.15}[regime]
    signal    = {"R_signal": 1.0, "R_firstorder": 1.0,  "R_noise": 0.0,  "R_leak": 0.0}[regime]
    # options: random content + fixed per-position identity, then unit-normalized
    opt = g.standard_normal((n, NOPT, D)).astype(np.float32)
    opt /= np.linalg.norm(opt, axis=2, keepdims=True) + 1e-8
    opt = opt + POS[None]                                        # consistent position axis (leak substrate)
    opt /= np.linalg.norm(opt, axis=2, keepdims=True) + 1e-8
    # gold index: uniform, except R_leak shifts it toward option 0 (an exploitable prior)
    if regime == "R_leak":
        p = np.array([0.70, 0.10, 0.10, 0.10])
        gold_idx = g.choice(NOPT, size=n, p=p)
    else:
        gold_idx = g.integers(0, NOPT, size=n)
    gold_emb = opt[np.arange(n), gold_idx]                       # [n,D]
    phi = (g.standard_normal((n, N, D)) * phi_scale).astype(np.float32)
    noise = 0.3
    delta = np.zeros((n, N, N, D), np.float32)
    for (i, j) in EDGES:
        d = phi[:, i] - phi[:, j] + TT[(i, j)] * signal * gold_emb
        d += noise * g.standard_normal((n, D)).astype(np.float32)
        delta[:, i, j] = d
    return delta, opt.astype(np.float32), gold_idx.astype(np.int64), gold_emb.astype(np.float32)

def model_answers(opt, gold_idx, seed, gate):
    """Which option each of N models emits. Gate items: 0-of-N correct (all wrong), >=2 distinct."""
    g = rng_of(seed + 999); n = len(gold_idx); ans = np.zeros((n, N), np.int64)
    for r in range(n):
        if gate:
            wrong = [o for o in range(NOPT) if o != gold_idx[r]]
            a = g.choice(wrong, size=N)
            if len(set(a.tolist())) < 2: a[0] = wrong[0]; a[1] = wrong[1]   # enforce >=2 distinct
        else:
            a = g.integers(0, NOPT, size=N)
        ans[r] = a
    return ans

def cyc_and_edges(delta):
    n = delta.shape[0]
    cyc = np.stack([delta[:, i, j] + delta[:, j, k] - delta[:, i, k] for (i, j, k) in TRIANGLES], 1)  # [n,10,D]
    edg = np.stack([delta[:, i, j] for (i, j) in EDGES], 1)                                            # [n,10,D]
    return cyc.reshape(n, -1), edg.reshape(n, -1)

# fixed seeded JL projections + shuffle
_JL = rng_of(20260716)
P_CYC = (_JL.standard_normal((len(TRIANGLES) * D, DSIG)) / math.sqrt(DSIG)).astype(np.float32)
P_EDG = (_JL.standard_normal((len(EDGES) * D, DSIG)) / math.sqrt(DSIG)).astype(np.float32)
PERM = _JL.permutation(DSIG)

def arm_inputs(delta):
    cyc, edg = cyc_and_edges(delta)
    H = cyc @ P_CYC                       # cycle-residual Hole Signature
    E = edg @ P_EDG                       # edge (first-order) control
    return {
        "C_cycle":     H,
        "B_edge":      E,
        "ctrl_mag":    np.linalg.norm(H, axis=1, keepdims=True).astype(np.float32),  # volume only
        "ctrl_blind":  np.ones((len(H), DSIG), np.float32),                          # item-independent
        "ctrl_shuffle": H[:, PERM],
    }

class Dec(nn.Module):
    def __init__(s, din): super().__init__(); s.w = nn.Linear(din, D)
    def forward(s, x): return s.w(x)

def train_eval(Xtr, gtr, Otr, Xte, gte, Ote, seed):
    torch.manual_seed(seed)
    # 80/20 train/val on clean
    ntr = len(Xtr); nval = max(20, ntr // 5); idx = torch.randperm(ntr)
    tr, va = idx[nval:], idx[:nval]
    Xtr_t = torch.tensor(Xtr); gtr_t = torch.tensor(gtr); Otr_t = torch.tensor(Otr)
    m = Dec(Xtr.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-3)
    best_va, best_state, patience = 1e9, None, 0
    for ep in range(300):
        m.train()
        out = m(Xtr_t[tr])
        logits = torch.bmm(Otr_t[tr], out.unsqueeze(2)).squeeze(2) / TEMP   # cosine-ish (opts unit) [b,NOPT]
        loss = nn.functional.cross_entropy(logits, gtr_t[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vo = m(Xtr_t[va]); vl = torch.bmm(Otr_t[va], vo.unsqueeze(2)).squeeze(2) / TEMP
            vloss = nn.functional.cross_entropy(vl, gtr_t[va]).item()
        if vloss < best_va - 1e-4: best_va, best_state, patience = vloss, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 30: break
    m.load_state_dict(best_state)
    with torch.no_grad():
        te_out = m(torch.tensor(Xte)); te_logits = torch.bmm(torch.tensor(Ote), te_out.unsqueeze(2)).squeeze(2)
        pred = te_logits.argmax(1).numpy()
    return float((pred == gte).mean())

def run_regime(regime):
    # clean (train) and gate (eval) sets
    dc, oc, gc, _ = make_regime(regime, N_CLEAN, seed=1)
    dg, og, gg, _ = make_regime(regime, N_GATE, seed=2)
    ans_g = model_answers(og, gg, seed=2, gate=True)
    # unit-normalize option embeddings for cosine
    def unit(o): return o / (np.linalg.norm(o, axis=2, keepdims=True) + 1e-8)
    oc, og = unit(oc), unit(og)
    Ac = arm_inputs(dc); Ag = arm_inputs(dg)
    # ctrl_shuffle: break the signature<->item pairing (ROW permutation). A column/feature permutation is
    # absorbed by a linear decoder; only breaking the H<->label correspondence destroys the signal.
    gsh = rng_of(4242)
    Ac["ctrl_shuffle"] = Ac["C_cycle"][gsh.permutation(N_CLEAN)]
    Ag["ctrl_shuffle"] = Ag["C_cycle"][gsh.permutation(N_GATE)]
    # majority baseline on gate (0-of-N by construction -> 0%)
    maj = np.array([np.bincount(a, minlength=NOPT).argmax() for a in ans_g])
    acc_A = float((maj == gg).mean())
    res = {"A": (acc_A, 0.0)}
    per_seed = {}
    for arm in ["ctrl_blind", "ctrl_mag", "B_edge", "ctrl_shuffle", "C_cycle"]:
        accs = [train_eval(Ac[arm], gc, oc, Ag[arm], gg, og, s) for s in SEEDS]
        res[arm] = (float(np.mean(accs)), float(np.std(accs))); per_seed[arm] = accs
    # ---- CHANCE-relative verdict (majority A is degenerate=0 on a 0-of-N blind-spot set) ----
    cyc_m = res["C_cycle"][0]
    leak = res["ctrl_blind"][0] > CHANCE + EPS
    beats_edge = cyc_m - res["B_edge"][0] >= DMIN
    beats_mag  = cyc_m - res["ctrl_mag"][0] >= DMIN
    beats_shuf = cyc_m > res["ctrl_shuffle"][0]
    above_chance = cyc_m > CHANCE + EPS
    if leak: verdict = "VOID(leak)"
    elif beats_edge and beats_mag and beats_shuf and above_chance: verdict = "PASS"
    else: verdict = "FALSIFIED"
    return res, verdict

def main():
    print(f"[DeltaSheaf-v0.2 instrument smoke]  N={N} D={D} opts={NOPT} d_sig={DSIG}  "
          f"clean={N_CLEAN} gate={N_GATE}  seeds={len(SEEDS)}  chance={CHANCE:.2f}  dmin={DMIN} eps={EPS}")
    print("  NOTE: majority A=0 on the 0-of-N blind-spot set -> verdict is CHANCE-relative (SPEC fix).\n")
    expect = {"R_signal": "PASS", "R_firstorder": "FALSIFIED", "R_noise": "FALSIFIED", "R_leak": "VOID(leak)"}
    hdr = f"  {'regime':13}{'A':>6}{'blind':>8}{'mag':>7}{'edge':>7}{'shuffle':>9}{'CYCLE':>8}{'verdict':>13}{'expect':>13}  ok"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    allok = True
    for reg in ["R_signal", "R_firstorder", "R_noise", "R_leak"]:
        res, verdict = run_regime(reg)
        ok = (verdict == expect[reg]); allok &= ok
        def c(a): return f"{res[a][0]:.2f}"
        print(f"  {reg:13}{c('A'):>6}{c('ctrl_blind'):>8}{c('ctrl_mag'):>7}{c('B_edge'):>7}"
              f"{c('ctrl_shuffle'):>9}{c('C_cycle'):>8}{verdict:>13}{expect[reg]:>13}  {'OK' if ok else 'FAIL'}")
    print()
    print(f"  INSTRUMENT {'VALIDATED — gate returns the correct verdict in all 4 regimes.' if allok else 'BROKEN — a regime returned the wrong verdict (see FAIL).'}")
    sys.exit(0 if allok else 1)

if __name__ == "__main__":
    main()
