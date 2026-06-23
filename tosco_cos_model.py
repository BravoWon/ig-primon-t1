#!/usr/bin/env python
"""Tosco Branch 4 RR — quantitative chance-of-success (CoS) model with cross-domain V&V.

Three independent estimators, then triangulate (validation = do they converge?):
  OUTSIDE VIEW  reference-class base rate (Red River dev/extension success) + this-well logit
                adjustments. The empirical anchor.
  INSIDE VIEW   sequential stage-gate model with SHARED latent factors (execution quality,
                geology quality) inducing realistic correlation between gates -> the dynamics.
  COMBINED      pooled posterior. + tornado sensitivity + the Bayesian risk-retirement ladder.

Inputs are structured EXPERT-JUDGMENT priors (not measured frequencies) except the reference class,
which is the one empirical anchor (NDGS Red River: Camel Hump 6/7, step-out 14/20). Value = the
structure (decomposition, correlation, triangulation, sensitivity, dynamics), not false precision.

    python tosco_cos_model.py
"""
import numpy as np
rng = np.random.default_rng(7)
N = 400_000
logit = lambda p: np.log(p / (1 - p))
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
pct = lambda a: tuple(np.percentile(a, [10, 50, 90]).round(3)) + (round(a.mean(), 3),)


def banner(s): print(f"\n{'='*68}\n{s}\n{'='*68}")


# ---------- OUTSIDE VIEW: reference class + this-well adjustments ----------
banner("OUTSIDE VIEW  (reference class + this-well logit adjustments)")
# reference class: SW-ND Red River development/extension wells (NDGS): 6/7 + 14/20 = 20/27
base = rng.beta(20 + 1, 7 + 1, N)
ADJ = {  # (mean, sd) shift in LOGIT units; + raises CoS, - lowers it
    "re-entry: reservoir already logged":        (+0.30, 0.15),
    "adjacent Branch 5 discovery PRODUCING":     (+0.40, 0.18),
    "continuous Red River 'B' fairway":          (+0.15, 0.10),
    "new pool -> virgin pressure (no depletion)": (+0.20, 0.12),
    "gentle 25-30deg build + modern tooling":    (+0.10, 0.08),
    "SHORT-RADIUS RE-ENTRY mechanical risk":     (-0.55, 0.25),
    "target CONFIDENTIAL (could be Madison)":     (-0.20, 0.15),
    "pool limits undefined (edge risk)":         (-0.18, 0.13),
}
shift = np.zeros(N)
draws = {}
for k, (m, s) in ADJ.items():
    d = rng.normal(m, s, N); draws[k] = d; shift += d
p_out = sig(logit(base) + shift)
print(f"  reference-class base rate         P10/P50/P90/mean = {pct(base)}")
print(f"  net logit shift (this well)       mean = {shift.mean():+.2f}")
print(f"  OUTSIDE-VIEW CoS                  P10/P50/P90/mean = {pct(p_out)}")

# ---------- INSIDE VIEW: correlated stage-gate (the dynamics) ----------
banner("INSIDE VIEW  (sequential stage gates, shared latent factors -> correlation)")
qe = rng.normal(0, 1, N)   # latent EXECUTION quality (operator/wellbore competence)
qg = rng.normal(0, 1, N)   # latent GEOLOGY quality (local reservoir/structure)
GATES = [   # (name, base prob, loading on qe, loading on qg)
    ("G1 reservoir present & charged",   0.90, 0.0, 0.55),
    ("G2 old casing/cement integrity",   0.85, 0.65, 0.0),
    ("G3 window + short-radius curve",   0.88, 0.60, 0.0),
    ("G4 lateral placed in oil (OWC)",   0.85, 0.35, 0.40),
    ("G5 commercial deliverability",     0.90, 0.20, 0.35),
]
gate_p = {}
p_in = np.ones(N)
for nm, b, le, lg in GATES:
    g = sig(logit(b) + le * qe + lg * qg + rng.normal(0, 0.18, N))
    gate_p[nm] = g; p_in *= g
    print(f"  {nm:34} marginal P10/P50/P90/mean = {pct(g)}")
print(f"  INSIDE-VIEW CoS (product)        P10/P50/P90/mean = {pct(p_in)}")
print(f"  (note: naive product of gate base-probs = {np.prod([b for _,b,_,_ in GATES]):.3f}; "
      f"per-gate uncertainty (Jensen) lowers each gate's mean, only partly offset by correlation, "
      f"-> simulated mean {p_in.mean():.3f} sits just BELOW the naive product)")

# ---------- COMBINED / triangulation ----------
banner("TRIANGULATION  (pool the two independent views)")
p_comb = np.concatenate([p_out, p_in])
print(f"  OUTSIDE  P50={np.median(p_out):.2f}   INSIDE  P50={np.median(p_in):.2f}   "
      f"divergence={abs(np.median(p_out)-np.median(p_in)):.2f}")
print(f"  COMBINED CoS                     P10/P50/P90/mean = {pct(p_comb)}")
print(f"  -> headline P(successful producer) ~ {np.median(p_comb):.0%}  "
      f"(80% band {np.percentile(p_comb,10):.0%}-{np.percentile(p_comb,90):.0%})")

# ---------- SENSITIVITY (tornado on outside-view adjustments) ----------
banner("SENSITIVITY  (tornado: swing each factor +/-1sd, measure dP50 of CoS)")
base_logit = logit(base) + shift
rows = []
for k, (m, s) in ADJ.items():
    lo = sig(base_logit - draws[k] + (m - s))   # this factor at its pessimistic 1sd
    hi = sig(base_logit - draws[k] + (m + s))   # ... optimistic 1sd
    rows.append((k, np.median(lo), np.median(hi), abs(np.median(hi) - np.median(lo))))
for k, lo, hi, sw in sorted(rows, key=lambda r: -r[3]):
    print(f"  {k:38} P50 {lo:.2f} <-> {hi:.2f}   swing {sw:.3f}")

# ---------- DYNAMICS: Bayesian risk-retirement ladder ----------
banner("DYNAMICS  (P(success) updates as each stage GATE is passed -- risk retires)")
running = 1.0
order = [(nm, b) for nm, b, _, _ in GATES]
print(f"  {'stage observed PASS':36} {'gate P':>7} {'P(success | reached here)':>26}")
remaining = [b for _, b in order]
for i, (nm, b) in enumerate(order):
    p_reach_success = np.prod(remaining[i:])   # product of this + all later gate base-probs
    print(f"  {('-- start --' if i==0 else nm):36} {b:>7.2f} {p_reach_success:>26.2f}")
print("  (each passed gate removes that risk; conditional CoS climbs toward 1 as drilling/"
      "completion succeed)")
