#!/usr/bin/env python
"""Gate P4.1 -- parity-split fractal wall, out-of-sample. Per PREREG_P41_parity_wall.md.

    python phys/gate_P41_parity.py
"""
import math, json
from math import gcd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_P4_butterfly import hof, bands_and_states, chern_multiband, diophantine

P4_WALL = {3: 1.26795, 4: 1.64575, 5: 0.15718, 6: 0.76813, 7: 0.02025, 8: 0.13670,
           9: 0.00231, 10: 0.06573, 11: 0.00025, 12: 0.06462}
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def min_gap_q(q, N=96):
    """Min indirect gap over all reduced p and internal gaps r; returns (gap, p, r)."""
    best = (1e9, None, None)
    for p in range(1, q):
        if gcd(p, q) != 1:
            continue
        k1s = np.linspace(0, 2 * math.pi / q, N, endpoint=False)
        k2s = np.linspace(0, 2 * math.pi, N, endpoint=False)
        E = np.empty((N, N, q))
        for i, k1 in enumerate(k1s):
            for j, k2 in enumerate(k2s):
                E[i, j] = np.linalg.eigvalsh(hof(k1, k2, p, q))
        for r in range(1, q):
            if q % 2 == 0 and r == q // 2:
                continue                                          # gapless center, nm
            g = float(E[:, :, r].min() - E[:, :, r - 1].max())
            if 1e-12 < g < best[0]:
                best = (g, p, r)
    return best


def main():
    print("[P4.1] parity-split wall, out-of-sample; prereg phys/PREREG_P41_parity_wall.md")
    # frozen in-sample fit (odd q<=11, P4 values)
    qs_in = np.array([5, 7, 9, 11], float)
    ys_in = np.log([P4_WALL[5], P4_WALL[7], P4_WALL[9], P4_WALL[11]])
    c0, a0 = np.polyfit(qs_in, ys_in, 1)
    print(f"  frozen odd law (q<=11): min-gap = exp({a0:.3f}) * e^({c0:.3f} q)   [c0 = {-c0:.3f}]")

    print("\n  Pass 1: the wall, q = 3..21 (N=96):")
    wall = {}
    for q in range(3, 22):
        g, p, r = min_gap_q(q)
        wall[q] = (g, p, r)
        anchor = ""
        if q in P4_WALL:
            dev = abs(g - P4_WALL[q]) / P4_WALL[q]
            anchor = f"  [P4: {P4_WALL[q]:.5f}, dev {dev*100:.1f}%]"
        print(f"    q={q:2d}: min gap = {g:.3e}  at p={p}, r={r}{anchor}", flush=True)

    # verdict 1: out-of-sample odd prediction
    print("\n  verdict 1 -- out-of-sample odd law:")
    errs = []
    for q in (13, 15, 17, 19, 21):
        pred = math.exp(a0 + c0 * q)
        meas = wall[q][0]
        e = abs(math.log(pred / meas))
        errs.append(e)
        print(f"    q={q}: predicted {pred:.2e}  measured {meas:.2e}  |log-err| = {e:.2f}")
    qs_all = np.array([q for q in range(5, 22, 2)], float)
    ys_all = np.log([wall[q][0] for q in range(5, 22, 2)])
    c1, a1 = np.polyfit(qs_all, ys_all, 1)
    r2 = 1 - np.var(ys_all - (c1 * qs_all + a1)) / np.var(ys_all)
    v1 = all(e < 0.5 for e in errs) and (0.92 <= -c1 <= 1.22) and r2 > 0.99
    print(f"    combined odd fit q=5..21: c = {-c1:.3f} (band [0.92,1.22]), R^2 = {r2:.4f}  -> {'PASS' if v1 else 'FAIL/replace'}")

    # verdict 2: parity persistence
    print("\n  verdict 2 -- even-q gaps vs odd-law prediction:")
    v2 = True
    for q in (14, 16, 18, 20):
        pred_odd = math.exp(a0 + c0 * q)
        ratio = wall[q][0] / pred_odd
        v2 &= ratio > 10
        print(f"    q={q}: even min-gap {wall[q][0]:.3e} = {ratio:.0f}x the odd-law prediction")

    # verdict 3: location of the odd minimal gap
    locs = {q: wall[q][2] for q in range(5, 22, 2)}
    v3 = all(locs[q] in ((q - 1) // 2, (q + 1) // 2) for q in locs)
    print(f"\n  verdict 3 -- odd min-gap location r vs center (q-+1)/2: " +
          " ".join(f"q{q}:r{locs[q]}" for q in locs) + f"  -> {'PASS' if v3 else 'FAIL'}")

    # verdict 4: receipts extension q=13..16, gaps > 1e-4
    print("\n  verdict 4 -- two-route receipts, q = 13..16 (gaps > 1e-4):")
    rec = ag = nm = 0
    for q in range(13, 17):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            E, U = bands_and_states(p, q, 24)
            for r in range(1, q):
                t = diophantine(p, q, r)
                gap = float(E[:, :, r].min() - E[:, :, r - 1].max())
                if t is None or gap < 1e-4:
                    nm += 1; continue
                rec += 1
                ag += (chern_multiband(U, r) == t)
        print(f"    q={q}: cumulative receipts {rec}, agree {ag}, nm {nm}", flush=True)
    v4 = ag == rec
    print(f"    -> {ag}/{rec} AGREE  -> {'PASS' if v4 else 'FAIL'}")

    print(f"\n  PRE-REGISTERED VERDICTS: (1) {'PASS' if v1 else 'FAIL'}  (2) {'PASS' if v2 else 'FAIL'}"
          f"  (3) {'PASS' if v3 else 'FAIL'}  (4) {'PASS' if v4 else 'FAIL'}")
    json.dump({"wall": {str(q): list(wall[q]) for q in wall}, "c_frozen": -c0, "c_combined": float(-c1),
               "r2": float(r2), "oos_errs": errs, "receipts": [rec, ag, nm]},
              open(__file__.replace("gate_P41_parity.py", "gate_P41_results.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    qo = list(range(5, 22, 2)); qe = list(range(4, 21, 2))
    ax.semilogy(qo, [wall[q][0] for q in qo], "o", color=RED, label="odd q (measured)")
    ax.semilogy(qe, [wall[q][0] for q in qe], "s", color=BLUE, label="even q (measured)")
    qq = np.linspace(4, 22, 50)
    ax.semilogy(qq, np.exp(a0 + c0 * qq), "--", color=AMBER, label=f"frozen q<=11 fit: e^(-{-c0:.2f}q)")
    ax.axvline(12.5, ls=":", color="#999", lw=1); ax.text(12.7, 1e-1, "out-of-sample ->", fontsize=8, color="#666")
    ax.set_xlabel("denominator q"); ax.set_ylabel("minimal gap")
    ax.set_title("P4.1: the parity-split fractal wall -- frozen odd law vs virgin denominators",
                 fontsize=9.5, color=NAVY)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(__file__.replace("gate_P41_parity.py", "gate_P41.png"), dpi=160)
    print("  wrote phys/gate_P41.png, phys/gate_P41_results.json")


if __name__ == "__main__":
    main()
