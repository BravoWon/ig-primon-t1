#!/usr/bin/env python
"""C1c echo stretch -- extract the DSS echoing period Delta from the h1(u) trace at p*(1-1e-12).
DSS: zero crossings of the axis field accumulate at u* geometrically, u*-u_i = C e^{-i Delta/2}
(the field oscillates once per Delta in tau = -ln(u*-u); two sign flips per cycle). Then the
ratio of CONSECUTIVE crossing intervals r_i = (u_{i+2}-u_{i+1})/(u_{i+1}-u_i) = e^{-Delta/2},
so Delta = -2 ln r_i -- no estimate of u* required. Anchor: Choptuik/Gundlach Delta = 3.44.
Cross-check channel: the Route-B wiggle period (2.0 decades in eps -> Delta = 2*gamma*ln(10)*2.0).
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
tr = np.asarray(res["echo_trace"])                  # (u, h1) rows, last 4000 steps of the run
u, h1 = tr[:, 0], tr[:, 1]
print(f"[C1c echo] trace: {len(u)} steps, u in [{u[0]:.4f}, {u[-1]:.6f}], |h1| in "
      f"[{np.abs(h1).min():.3e}, {np.abs(h1).max():.3e}]")

s = np.sign(h1)
flips = np.where(s[1:] * s[:-1] < 0)[0]             # crossing between step i and i+1
uc = 0.5 * (u[flips] + u[flips + 1])                # crossing times
print(f"  zero crossings found: {len(uc)}")
if len(uc) >= 4:
    dv = np.diff(uc)                                # intervals between crossings
    ratios = dv[1:] / dv[:-1]
    good = (ratios > 0.01) & (ratios < 1.0)         # accumulating means shrinking intervals
    D = -2 * np.log(ratios[good])
    for i, (r, d) in enumerate(zip(ratios, -2 * np.log(np.maximum(ratios, 1e-12)))):
        tag = "" if (0.01 < r < 1.0) else "   (excluded: not shrinking)"
        print(f"    crossing pair {i}: interval ratio {r:.4f} -> Delta = {d:.3f}{tag}")
    if len(D):
        print(f"\n  Delta (median of {len(D)} interval ratios) = {np.median(D):.3f}"
              f"   [mean {np.mean(D):.3f} +- {np.std(D):.3f}]   anchor 3.44")
        res["echo"] = {"n_crossings": int(len(uc)), "Delta_median": float(np.median(D)),
                       "Delta_mean": float(np.mean(D)), "Delta_std": float(np.std(D)),
                       "crossing_u": [float(x) for x in uc]}
        json.dump(res, open(os.path.join(HERE, "gate_C1c_results.json"), "w"), indent=1)
        print("  echo block appended to cosmos/gate_C1c_results.json")
else:
    print("  too few crossings for the ratio method -- echo nm (declared possible in prereg)")
