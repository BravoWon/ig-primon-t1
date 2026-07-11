#!/usr/bin/env python
"""Gate P4.3 -- flux 3/q outermost gap: the one-instanton out-of-sample test. Per PREREG_P43_one_instanton.md.

    python phys/gate_P43_one_instanton.py
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_P4_butterfly import hof
from gate_P42_asymptote import refine
from mpmath import mp, mpc, matrix as mpmatrix


def eig_mp3(k1, k2, q, dps=40):
    mp.dps = dps
    H = mpmatrix(q, q)
    for m in range(q):
        H[m, m] = 2 * mp.cos(k2 + 2 * mp.pi * 3 * m / q)          # p = 3 (P4.2's eig_mp hardcodes p=2)
        H[m, (m + 1) % q] += 1; H[(m + 1) % q, m] += 1
    H[0, q - 1] += mp.e ** (mpc(0, -q * k1)) - 1
    H[q - 1, 0] += mp.e ** (mpc(0, q * k1)) - 1
    try:
        E = mp.eighe(H, eigvals_only=True)
    except TypeError:
        E, _ = mp.eighe(H)
    return sorted([mp.mpf(x.real) if hasattr(x, "real") else mp.mpf(x) for x in E], key=float)

C = 0.915965594177219015
C3 = 4 * C / (3 * math.pi)
C2 = 4 * C / math.pi
QS = [13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61, 65, 67, 71, 73]
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def gap_3q(q, N=96):
    k1s = np.linspace(0, 2 * math.pi / q, N, endpoint=False)
    k2s = np.linspace(0, 2 * math.pi, N, endpoint=False)
    E1 = np.empty((N, N)); E2 = np.empty((N, N))
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            w = np.linalg.eigvalsh(hof(k1, k2, 3, q))
            E1[i, j], E2[i, j] = w[0], w[1]
    i1, j1 = np.unravel_index(E1.argmax(), E1.shape)
    i2, j2 = np.unravel_index(E2.argmin(), E2.shape)
    span = math.pi / (q * N) * 4
    f1 = lambda a, b: np.linalg.eigvalsh(hof(a, b, 3, q))[0]
    f2 = lambda a, b: -np.linalg.eigvalsh(hof(a, b, 3, q))[1]
    m1, k1b = refine(f1, (k1s[i1], k2s[j1]), span)
    m2, k2b = refine(f2, (k1s[i2], k2s[j2]), span)
    return -m2 - m1, k1b, k2b


def main():
    print(f"[P4.3] flux 3/q, r=1 gap; prediction c3 = 4C/(3pi) = {C3:.7f}; prereg PREREG_P43_one_instanton.md")
    gaps = {}
    for q in QS:
        g, k1b, k2b = gap_3q(q)
        gaps[q] = g
        print(f"    q={q:2d}: gap = {g:.6e}", flush=True)
    # mp spot-check at q=49
    g49, k1b, k2b = gap_3q(49)
    b1 = refine(lambda a, b: eig_mp3(a, b, 49)[0], k1b, math.pi / (49 * 96) / 10, rounds=2, grid=5)
    b2 = refine(lambda a, b: -eig_mp3(a, b, 49)[1], k2b, math.pi / (49 * 96) / 10, rounds=2, grid=5)
    gmp = float(-b2[0] - b1[0])
    dev = abs(gmp - gaps[49]) / gaps[49]
    print(f"    cross-validation q=49: f64 {gaps[49]:.8e} vs mp40 {gmp:.8e}  (dev {dev*100:.4f}%)")

    print("\n  local slopes c3(q) between consecutive coprime q:")
    cq = {}
    for qa, qb in zip(QS, QS[1:]):
        cq[(qa + qb) / 2] = (math.log(gaps[qa]) - math.log(gaps[qb])) / (qb - qa)
    for m in sorted(cq):
        print(f"    c3(mid {m:.1f}) = {cq[m]:.6f}")
    items = sorted(cq.items())
    ests = []
    for w in (6, 8, 10):
        pts = items[-w:]
        x = np.array([1.0 / m for m, _ in pts]); y = np.array([c for _, c in pts])
        ests.append(np.polyfit(x, y, 1)[1])
    a = [e for e in ests]
    cinf = float(np.mean(ests)); sig = float(np.std(ests)) + 5e-4
    print(f"\n  extrapolated c3_inf = {cinf:.5f} +- {sig:.5f}   (target 4C/(3pi) = {C3:.5f})")
    band = (0.98 * C3, 1.02 * C3)
    v1 = band[0] <= cinf <= band[1]
    ratio = C2 / cinf
    v2 = abs(ratio - 3) <= 0.06
    resid = [math.log(gaps[q]) + C3 * q for q in QS]
    X = np.array([[1.0, math.log(q)] for q in QS]); y = np.array(resid)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - X @ coef) ** 2)))
    v3 = rms < 0.05
    v4 = dev < 0.001
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) c3_inf in [{band[0]:.4f}, {band[1]:.4f}]: {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) exponent ratio c2/c3 = {ratio:.4f} (target 3 +- 0.06): {'PASS' if v2 else 'FAIL'}")
    print(f"    (3) fixed-exponent fit: alpha = {coef[1]:.3f}, rms = {rms*100:.2f}% (<5%): {'PASS' if v3 else 'FAIL'}")
    print(f"    (4) f64-vs-mp cross-validation: {'PASS' if v4 else 'FAIL'}")
    if v1 and v2 and v3 and v4:
        print("  ==> THE MECHANISM HOLDS OUT-OF-SAMPLE: one instanton at p=3, two at p=2 -- the exact")
        print("      factor of three, measured. P4.2's parent claim survives its child's bite.")
    else:
        print("  ==> miss located -- P4.2's mechanism section flagged per prereg; the table is the finding.")

    json.dump({"gaps": {str(q): gaps[q] for q in QS}, "slopes": {str(k): v for k, v in cq.items()},
               "c3_inf": cinf, "sigma": sig, "target": C3, "ratio": ratio, "alpha": float(coef[1]),
               "rms": rms, "mp_dev": dev},
              open(__file__.replace("gate_P43_one_instanton.py", "gate_P43_results.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7.8, 5))
    ms = sorted(cq)
    ax.plot([1 / m for m in ms], [cq[m] for m in ms], "o-", color=RED, ms=4, label="local slope c3(q), flux 3/q")
    ax.axhline(C3, ls="--", lw=1.3, color=GREEN)
    ax.text(0.056, C3 + 0.0006, "4C/(3pi) -- one-instanton prediction", fontsize=8, color=GREEN)
    ax.errorbar([0], [cinf], yerr=[2 * sig], fmt="s", color=NAVY, capsize=4,
                label=f"c3_inf = {cinf:.4f} +- {2*sig:.4f} (2sig)")
    ax.set_xlabel("1/q (slope midpoint)"); ax.set_ylabel("slope")
    ax.set_title("P4.3: the loaded gun -- flux 3/q decays at exactly one third of p=2's rate?",
                 fontsize=9.5, color=NAVY)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(__file__.replace("gate_P43_one_instanton.py", "gate_P43.png"), dpi=160)
    print("  wrote phys/gate_P43.png, phys/gate_P43_results.json")


if __name__ == "__main__":
    main()
