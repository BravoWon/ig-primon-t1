#!/usr/bin/env python
"""C1c wiggle channel -- extract Delta from the DSS fine structure of the scaling laws.
Gundlach / Hod-Piran: discrete self-similarity superimposes a periodic modulation on critical
scaling, ln M = c + gamma ln eps + A sin(2 pi ln eps / P + phi) with period P = Delta / (2 gamma)
in ln eps -- SAME period on both branches (supercritical mass, subcritical curvature). Two
independent routes -> two independent Delta estimates. Model fit: scan P, solve (c, g, A, phi)
linearly at each P (sin/cos basis), pick P minimizing SSE; errors from the P-scan curvature.
Anchor: Delta = 3.44 (Gundlach). This is the quantified version of the by-eye 2.0-decade spacing.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))


def wigglefit(eps, y, name, gamma_sign):
    lx, ly = np.log(np.asarray(eps)), np.log(np.asarray(y))
    Ps = np.linspace(1.5, 12.0, 800)                # candidate periods in ln eps
    best = None
    for P in Ps:
        X = np.vstack([np.ones_like(lx), lx, np.sin(2 * math.pi * lx / P),
                       np.cos(2 * math.pi * lx / P)]).T
        beta, sse, *_ = np.linalg.lstsq(X, ly, rcond=None)
        sse = float(np.sum((ly - X @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, P, beta)
    sse, P, (c, g, a1, a2) = best
    A = math.hypot(a1, a2)
    lin = np.polyfit(lx, ly, 1)
    sse0 = float(np.sum((ly - np.polyval(lin, lx)) ** 2))
    gamma = gamma_sign * g
    Delta = 2 * abs(gamma) * P
    print(f"  {name}: n={len(lx)}  gamma={gamma:.4f}  P={P:.3f} ln-eps ({P/math.log(10):.2f} decades)"
          f"  A={A:.3f}  SSE {sse0:.4f} -> {sse:.4f} (x{sse/max(sse0,1e-30):.3f})"
          f"  ->  Delta = 2*gamma*P = {Delta:.3f}")
    return {"gamma": float(gamma), "P_lneps": float(P), "P_decades": float(P / math.log(10)),
            "amp": float(A), "sse_lin": sse0, "sse_wig": float(sse), "Delta": float(Delta)}


dense_path = os.path.join(HERE, "gate_C1c_dense.json")
if os.path.exists(dense_path):
    d = json.load(open(dense_path))
    print("[C1c wiggle] DSS fine-structure fit on the DENSE quarter-decade scan; anchor Delta=3.44")
    # Route A: full dense window eps 1e-1..1e-5 (pre-noise); Route B: pre-ceiling eps >= 3.16e-8
    eA, mA = d["A"]["eps"], d["A"]["M"]
    eB = [e for e in d["B"]["eps"] if e >= 3e-8]
    rB = d["B"]["R"][:len(eB)]
    wA = wigglefit(eA, mA, "Route A (M, supercritical)   ", +1.0)
    wB = wigglefit(eB, rB, "Route B (maxh1^2, subcritical)", -0.5)
else:
    print("[C1c wiggle] DSS fine-structure fit, both routes independently; anchor Delta=3.44")
    # Route A: the qualifying scaling window (eps 1e-1..1e-5, 9 points, pre-noise)
    wA = wigglefit(res["epsA"][:9], res["MA"][:9], "Route A (M, supercritical)   ", +1.0)
    # Route B: all pre-ceiling points (eps 1e-1..3.16e-8, 14 points; ceiling starts ~1e-8)
    wB = wigglefit(res["epsB"][:14], res["RB"][:14], "Route B (maxh1^2, subcritical)", -0.5)
print(f"\n  Delta_A = {wA['Delta']:.3f}   Delta_B = {wB['Delta']:.3f}   |dDelta| = "
      f"{abs(wA['Delta'] - wB['Delta']):.3f}   anchor 3.44 "
      f"(A off {abs(wA['Delta'] - 3.44):.3f}, B off {abs(wB['Delta'] - 3.44):.3f})")
res["wiggle"] = {"routeA": wA, "routeB": wB}
json.dump(res, open(os.path.join(HERE, "gate_C1c_results.json"), "w"), indent=1)
print("  wiggle block appended to cosmos/gate_C1c_results.json")
