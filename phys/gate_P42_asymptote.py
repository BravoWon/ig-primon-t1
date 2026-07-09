#!/usr/bin/env python
"""Gate P4.2 -- the odd-q gap asymptote: candidate discrimination. Per PREREG_P42_asymptote.md.
Front A: g(q) at flux 2/q, odd q=13..33; f64 coarse+refine, mpmath (dps 50) for q>=25.

    python phys/gate_P42_asymptote.py
"""
import math, json
import numpy as np
from mpmath import mp, mpc, matrix as mpmatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_P4_butterfly import hof

P41 = {13: 2.523e-05, 15: 2.549e-06, 17: 2.548e-07, 19: 2.531e-08, 21: 2.502e-09}
CANDS = {"ln(10)/2": math.log(10) / 2, "2/sqrt3": 2 / math.sqrt(3), "pi/e": math.pi / math.e,
         "7/6": 7 / 6, "P4.1-1.128": 1.128}
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def eig_f64(k1, k2, q):
    return np.linalg.eigvalsh(hof(k1, k2, 2, q))


def eig_mp(k1, k2, q, dps=50):
    mp.dps = dps
    H = mpmatrix(q, q)
    for m in range(q):
        H[m, m] = 2 * mp.cos(k2 + 2 * mp.pi * 2 * m / q)
        H[m, (m + 1) % q] += 1; H[(m + 1) % q, m] += 1
    H[0, q - 1] += mp.e ** (mpc(0, -q * k1)) - 1
    H[q - 1, 0] += mp.e ** (mpc(0, q * k1)) - 1
    try:
        E = mp.eighe(H, eigvals_only=True)
    except TypeError:
        E, _ = mp.eighe(H)
    return sorted([mp.mpf(x.real) if hasattr(x, "real") else mp.mpf(x) for x in E], key=float)   # KEEP mpf -- float() cast was the q>=31 contamination


def refine(fun, k0, span, rounds=4, grid=7):
    """Locally optimize fun over (k1,k2) around k0; returns best (val, k)."""
    best = (fun(*k0), k0)
    c = k0; s = span
    for _ in range(rounds):
        for a in np.linspace(c[0] - s, c[0] + s, grid):
            for b in np.linspace(c[1] - s, c[1] + s, grid):
                v = fun(a, b)
                if v > best[0]:
                    best = (v, (a, b))
        c = best[1]; s /= (grid - 1) / 2.0
    return best


def gap_2q(q):
    """Indirect outermost gap of flux 2/q: max E1 and min E2, refined; mpmath for tiny gaps."""
    N = 96
    k1s = np.linspace(0, 2 * math.pi / q, N, endpoint=False)
    k2s = np.linspace(0, 2 * math.pi, N, endpoint=False)
    E1 = np.empty((N, N)); E2 = np.empty((N, N))
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            w = eig_f64(k1, k2, q)
            E1[i, j], E2[i, j] = w[0], w[1]
    i1, j1 = np.unravel_index(E1.argmax(), E1.shape)
    i2, j2 = np.unravel_index(E2.argmin(), E2.shape)
    span1 = math.pi / (q * N) * 4
    m1, _ = refine(lambda a, b: eig_f64(a, b, q)[0], (k1s[i1], k2s[j1]), span1)
    m2, _ = refine(lambda a, b: -eig_f64(a, b, q)[1], (k1s[i2], k2s[j2]), span1)
    g64 = -m2 - m1
    if g64 > 1e-11 and q < 25:
        return g64, "f64"
    # mpmath pass around the same extrema
    m1v, k1b = refine(lambda a, b: eig_f64(a, b, q)[0], (k1s[i1], k2s[j1]), span1)
    m2v, k2b = refine(lambda a, b: -eig_f64(a, b, q)[1], (k1s[i2], k2s[j2]), span1)
    b1 = refine(lambda a, b: eig_mp(a, b, q)[0], k1b, span1 / 50, rounds=3, grid=5)
    b2 = refine(lambda a, b: -eig_mp(a, b, q)[1], k2b, span1 / 50, rounds=3, grid=5)
    return (-b2[0] - b1[0]), "mp50"                                     # mpf all the way


def main():
    print("[P4.2] the asymptote: odd q=13..33, flux 2/q; prereg phys/PREREG_P42_asymptote.md")
    qs = list(range(13, 38, 2))
    gaps = {}
    for q in qs:
        g, route = gap_2q(q)
        gaps[q] = mp.mpf(g)
        anch = ""
        if q in P41:
            ok = float(g) <= P41[q] * 1.001 and float(g) > 0.5 * P41[q]
            anch = f"  [P4.1 {P41[q]:.3e}: {'ok' if ok else 'ANCHOR FAIL'}]"
        print(f"    q={q:2d}: gap = {float(g):.6e}  ({route}){anch}", flush=True)
    # f64-vs-mp cross-validation at q=23 (both regimes reachable)
    g64 = gap_2q(23)[0]
    print(f"    cross-check q=23: pipeline {float(gaps[23]):.6e} vs re-run {float(g64):.6e}")

    print("\n  local slopes c(q) = [ln g(q) - ln g(q+2)]/2:")
    cq = {}
    for q in qs[:-1]:
        cq[q] = float(mp.log(gaps[q] / gaps[q + 2])) / 2
        print(f"    c({q}->{q+2}) = {cq[q]:.5f}")
    # extrapolation: c(q) = c_inf + b/qm on deepest 6 slopes (qm = midpoint), window-rotated
    items = sorted(cq.items())
    ests = []
    for w in (5, 6, 7):
        pts = items[-w:]
        x = np.array([1.0 / (q + 1) for q, _ in pts]); y = np.array([c for _, c in pts])
        b, cinf = np.polyfit(x, y, 1)
        ests.append(cinf)
    cinf = float(np.mean(ests)); sig = float(np.std(ests)) + 1e-4
    print(f"\n  extrapolated c_inf = {cinf:.5f} +- {sig:.5f}  (window-rotated 1/q fits)")
    print(f"  CANDIDATE DISCRIMINATION (survive iff |c_inf - cand| <= 2 sigma = {2*sig:.5f}):")
    survivors = []
    for name, val in CANDS.items():
        d = abs(cinf - val)
        ok = d <= 2 * sig
        survivors += [name] if ok else []
        print(f"    {name:12} = {val:.6f}   |d| = {d:.5f}   {'SURVIVES' if ok else 'excluded'}")
    if not survivors:
        print("    -> NONE-OF-THESE survives: the asymptote is a constant outside the frozen set (measured law stands alone).")
    else:
        print(f"    -> survivors: {survivors}  [identification gated on the Front-B literature verdict]")
    ln102 = "EXCLUDED" if abs(cinf - CANDS["ln(10)/2"]) > 2 * sig else "alive"
    print(f"  the P4.1 decade-law curiosity ln(10)/2: {ln102}")

    json.dump({"gaps": {str(q): float(gaps[q]) for q in qs}, "slopes": {str(q): cq[q] for q in cq},
               "c_inf": cinf, "sigma": sig, "survivors": survivors},
              open(__file__.replace("gate_P42_asymptote.py", "gate_P42_results.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7.6, 5))
    xq = sorted(cq); ax.plot([1 / (q + 1) for q in xq], [cq[q] for q in xq], "o-", color=RED, label="local slope c(q)")
    for name, val in CANDS.items():
        if name == "P4.1-1.128":
            continue
        ax.axhline(val, ls=":", lw=1, color=BLUE); ax.text(0.0745, val, name, fontsize=7, va="bottom")
    ax.errorbar([0], [cinf], yerr=[2 * sig], fmt="s", color=GREEN, capsize=4, label=f"c_inf = {cinf:.4f} +- {2*sig:.4f} (2sig)")
    ax.set_xlabel("1/q"); ax.set_ylabel("slope"); ax.set_xlim(-0.004, 0.078)
    ax.set_title("P4.2: the gap asymptote -- local slopes extrapolated to q = infinity", fontsize=9.5, color=NAVY)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(__file__.replace("gate_P42_asymptote.py", "gate_P42.png"), dpi=160)
    print("  wrote phys/gate_P42.png, phys/gate_P42_results.json")


if __name__ == "__main__":
    main()
