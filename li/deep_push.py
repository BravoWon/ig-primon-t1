#!/usr/bin/env python
"""L3 Phase 3 -- deep push to n=1000, priced by the measured wall law. Per PREREG_L3p3_deep.md.
Reuses the Phase-2 machinery (taylor_Z / series_log / sigmas / lambdas / route_A) with deep parameters.

    python li/deep_push.py [--smoke]
"""
import sys, json, math
import numpy as np
from mpmath import mp, mpf, euler, pi, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "li")
from push_wall import taylor_Z, series_log, sigmas_from_g, lambdas_from_sigmas, route_A

SMOKE = "--smoke" in sys.argv
if SMOKE:
    NMAX, DPS_MAIN, DPS_CHECK, M = 40, 60, 45, 512
    TABLE_N = [20, 40]
else:
    NMAX, DPS_MAIN, DPS_CHECK, M = 1000, 360, 310, 2048
    TABLE_N = [400, 500, 600, 700, 800, 900, 1000]
R = 3.0
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def pipeline(dps):
    a = taylor_Z(NMAX, dps, R, M)
    g = series_log(a, NMAX)
    sig = sigmas_from_g(g, NMAX)
    return lambdas_from_sigmas(sig, NMAX), sig


def main():
    print(f"[L3p3] deep push n<={NMAX}; dps {DPS_MAIN}/{DPS_CHECK}; M={M}; R={R}  {'(SMOKE)' if SMOKE else ''}")
    mp.dps = DPS_MAIN
    lam1_closed = 1 + euler / 2 - log(4 * pi) / 2

    print("  Route B @ main dps ...", flush=True)
    lamB, sig = pipeline(DPS_MAIN)
    print(f"    anchors: lambda_1 err {mp.nstr(abs(lamB[1]-lam1_closed),3)}  "
          f"sigma_2 {mp.nstr(sig[2],17)}  sigma_3 {mp.nstr(sig[3],17)}")
    print(f"             lambda_40 = {mp.nstr(lamB[40],15)}  (cross-run: 30.4773754237807)")

    print("  Route B @ check dps (wall-law out-of-sample test) ...", flush=True)
    lamB2, _ = pipeline(DPS_CHECK)
    ns_w, lost = [], []
    step = max(NMAX // 40, 2)
    for n in range(step, NMAX + 1, step):
        d = abs(lamB[n] - lamB2[n]) / abs(lamB[n])
        eff = float(-mp.log(d, 10)) if d > 0 else DPS_CHECK
        ns_w.append(n); lost.append(DPS_CHECK - eff)
    A = np.polyfit(ns_w, lost, 1)
    r2 = 1 - np.var(np.array(lost) - np.polyval(A, ns_w)) / max(np.var(lost), 1e-30)
    ok_wall = abs(A[0] - 0.2947) / 0.2947 < 0.15 and r2 > 0.99
    print(f"    WALL: measured slope {A[0]:.4f}/n (R^2={r2:.4f}) vs Phase-2 law 0.2947/n -> "
          f"{'CONFIRMED out-of-sample' if ok_wall else 'DEVIATES (report as-is)'}")
    eff_1000 = DPS_CHECK - np.polyval(A, NMAX)
    print(f"    effective agreeing digits at n={NMAX}: ~{eff_1000:.0f} (check-dps side)")

    print("  Route A: 100k zeros + semi-analytic tail ...", flush=True)
    gammas = np.loadtxt("li/odlyzko_zeros1.txt")
    lamA = route_A(TABLE_N, gammas)

    print(f"\n  {'n':>5} {'lambda_B':>18} {'A+tail':>18} {'|diff|':>10} {'budget':>10} {'ratio':>7}")
    worst, allpos, rows = 0.0, True, []
    for n in TABLE_N:
        Aval, tail, budget = lamA[n]
        d = abs(float(lamB[n]) - Aval); ratio = d / budget
        worst = max(worst, ratio); allpos &= lamB[n] > 0
        rows.append((n, float(lamB[n]), Aval, d, budget, ratio))
        print(f"  {n:>5} {float(lamB[n]):>18.8f} {Aval:>18.8f} {d:>10.2e} {budget:>10.2e} {ratio:>7.2f}")

    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) wall-slope reproduction : {'PASS' if ok_wall else 'FAIL -- measured law updated'}")
    print(f"    (2) two-route agree (worst ratio {worst:.2f} vs 3.0): {'PASS' if worst<=3 else 'FAIL'}")
    print(f"    (3) all lambda_n > 0 to n={NMAX}: {allpos}  (RH-consistent; occupied territory)")

    if not SMOKE:
        json.dump({"table": rows, "wall_slope": float(A[0]), "r2": float(r2)},
                  open("li/deep_push_results.json", "w"), indent=1)
        nn = np.array([r[0] for r in rows]); lb = np.array([r[1] for r in rows])
        trend = nn / 2 * (np.log(nn) - math.log(2 * math.pi) + float(euler) - 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        allns = np.arange(1, NMAX + 1); lall = np.array([float(lamB[n]) for n in allns])
        tr_all = allns / 2 * (np.log(allns) - math.log(2 * math.pi) + float(euler) - 1)
        axes[0].plot(allns, lall, "-", color=GREEN, lw=1, label="lambda_n (zero-free, dps 360)")
        axes[0].plot(nn, [r[2] for r in rows], "x", color=BLUE, ms=8, label="route A + tail (100k zeros)")
        axes[0].plot(allns, tr_all, "--", color=AMBER, lw=1, label="RH trend")
        axes[0].set_title(f"Li coefficients to n={NMAX}: two routes", fontsize=9.5, color=NAVY)
        axes[0].legend(fontsize=7.5, frameon=False); axes[0].set_xlabel("n")
        axes[1].plot(allns, lall - tr_all, "-", color=GREEN, lw=0.9)
        axes[1].axhline(0, color="#999", lw=0.8)
        axes[1].set_title("residue off the RH trend (bounded <-> zeros on line, in range)", fontsize=9.5, color=NAVY)
        axes[1].set_xlabel("n")
        fig.tight_layout(); fig.savefig("li/deep_push.png", dpi=160); plt.close(fig)
        print("  wrote li/deep_push.png, li/deep_push_results.json")


if __name__ == "__main__":
    main()
