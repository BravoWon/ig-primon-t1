#!/usr/bin/env python
"""C1c fitter correction -- implements the PRE-REGISTERED window rule faithfully.
PREREG_C1_choptuik.md verdict 1: "largest window with locally stable slope" (contiguous, per the
rule text also inlined in gate_C1_choptuik.py: 'largest contiguous window with locally stable
slope (drop tail where local slope deviates > tol from the window median)').
The in-run fitwin() had two bugs, both provable by inspection (Law #1, fitter not physics):
  (a) median taken over ALL local slopes including the mass-floor plateau (~0), poisoning the rule;
  (b) keep-set used non-contiguously via min/max, so the 'window' could span the noise band;
  (c) decade span reported with sign (eps descending -> negative decades).
This script re-fits the SAME raw arrays from gate_C1c_results.json with the registered rule:
largest contiguous slope-run whose local slopes all lie within tol of that run's own median;
ties broken by log-eps span. Same tolerances as in-run: 0.05 (A), 0.1 (B). No data is touched.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))


def refit(eps, y, tol):
    """Largest contiguous window with (a) local slopes within tol of the window median AND
    (b) window fit R^2 > 0.99 -- the FULL registered conjunction ('>=2-decade linear window,
    R^2>0.99, largest with locally stable slope'). The mass-floor plateau is slope-stable but
    R^2~0.3 (flat + noise), so it self-excludes under (b); no post-hoc exclusion needed."""
    lx, ly = np.log(np.asarray(eps)), np.log(np.asarray(y))
    sl = np.diff(ly) / np.diff(lx)
    best = None                                     # (span_decades, i0, i1) over slope indices
    n = len(sl)
    for i in range(n):
        for j in range(i + 1, n):                   # window = slopes i..j -> points i..j+1 (>=3 pts)
            w = sl[i:j + 1]
            if np.max(np.abs(w - np.median(w))) >= tol:
                continue
            xs, ys = lx[i:j + 2], ly[i:j + 2]
            g, b = np.polyfit(xs, ys, 1)
            r2 = 1 - np.var(ys - (g * xs + b)) / np.var(ys)
            if r2 <= 0.99:
                continue
            span = abs(lx[j + 1] - lx[i]) / math.log(10)
            if best is None or span > best[0]:
                best = (span, i, j, float(g), float(r2))
    if best is None:                                             # PR#13 review: crash -> clear nm
        raise SystemExit("refit nm: no contiguous window satisfies tol AND R^2>0.99 "
                         "(the registered conjunction admits no window on this data)")
    span, i, j, g, r2 = best
    return g, r2, float(span), float(eps[i]), float(eps[j + 1])


gA, r2A, decA, eA_hi, eA_lo = refit(res["epsA"], res["MA"], 0.05)
sB, r2B, decB, eB_hi, eB_lo = refit(res["epsB"], res["RB"], 0.1)
gB = -sB / 2

print(f"[C1c refit] pre-registered contiguous-window rule, same raw arrays, same tolerances")
print(f"  Route A: gamma_A = {gA:.4f}  R^2={r2A:.5f}  window eps {eA_hi:.2e}..{eA_lo:.2e} ({decA:.2f} decades)")
print(f"  Route B: gamma_B = {gB:.4f}  (slope {sB:.4f})  R^2={r2B:.5f}  window eps {eB_hi:.2e}..{eB_lo:.2e} ({decB:.2f} decades)")

v1 = r2A > 0.99 and decA >= 2
v2 = abs(gA - gB) <= 0.03
v3 = abs(gA - 0.374) <= 0.02 and abs(gB - 0.374) <= 0.02
print(f"\n  VERDICTS (prereg, corrected fitter):")
print(f"    (1) >=2-decade window, R^2>0.99: {'PASS' if v1 else 'FAIL'}")
print(f"    (2) |gamma_A - gamma_B| = {abs(gA - gB):.4f} (<=0.03): {'PASS' if v2 else 'FAIL'}")
print(f"    (3) both within 0.374 +- 0.02: {'PASS' if v3 else 'FAIL'}  (A off by {abs(gA-0.374):.4f}, B off by {abs(gB-0.374):.4f})")

res["refit"] = {"gamma_A": gA, "r2A": r2A, "decA": decA, "winA": [eA_hi, eA_lo],
                "gamma_B": gB, "r2B": r2B, "decB": decB, "winB": [eB_hi, eB_lo],
                "verdicts": [bool(v1), bool(v2), bool(v3)],
                "note": "pre-registered contiguous-window rule; in-run fitwin disowned (plateau-poisoned median, non-contiguous keep, signed decades)"}
json.dump(res, open(os.path.join(HERE, "gate_C1c_results.json"), "w"), indent=1)
print("  refit block appended to cosmos/gate_C1c_results.json")
