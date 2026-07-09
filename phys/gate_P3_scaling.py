#!/usr/bin/env python
"""Gate P3 -- finite-size scaling at the disorder-driven topological transition. Per PREREG_P3_criticality.md.

    python phys/gate_P3_scaling.py
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_P1b_disorder import build, T2
from gate_P2a_lcm import lcm

M2 = 5.5 * T2
VS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8]
LS = [12, 16, 24, 32]
NSEED = 16
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def bott_relaxed(L, V, seed, EF=0.0, gate=0.005):
    H, xy = build(L, L, M2, V, seed, torus=True)
    w, v = np.linalg.eigh(H)
    if np.min(np.abs(w - EF)) < gate:
        return None
    P = v[:, w < EF]
    Ux = np.exp(2j * math.pi * xy[:, 0] / L); Uy = np.exp(2j * math.pi * xy[:, 1] / L)
    U = P.conj().T @ (Ux[:, None] * P); Vm = P.conj().T @ (Uy[:, None] * P)
    ev = np.linalg.eigvals(Vm @ U @ np.linalg.inv(Vm) @ np.linalg.inv(U))
    B = float(np.sum(np.angle(ev))) / (2 * math.pi)
    return int(round(B)) if abs(B - round(B)) < 0.1 else None


def crossing(Va, ya, yb):
    """First sign change of (ya-yb) over the V grid, linear interp."""
    d = np.asarray(ya) - np.asarray(yb)
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            f = d[i] / (d[i] - d[i + 1])
            return Va[i] + f * (Va[i + 1] - Va[i])
    return None


def collapse_obj(tbl, Ls, Vs, Vc, nu, nbin=8):
    xs, ys = [], []
    for L in Ls:
        for j, V in enumerate(Vs):
            xs.append((V - Vc) * L ** (1 / nu)); ys.append(tbl[L][j])
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    bins = np.array_split(np.arange(len(xs)), nbin)
    return float(np.mean([np.var(ys[b]) for b in bins if len(b) > 1]))


def main():
    print(f"[P3] finite-size scaling at the TAI transition; prereg phys/PREREG_P3_criticality.md")
    print("  per-L calibration at the clean Haldane point:")
    CAL = {}
    for L in LS:
        CAL[L] = lcm(L, 0.0, 0.0, 0, margin=L // 4) / (-1.0)
        print(f"    L={L}: CAL = {CAL[L]:+.4f}")

    print("\n  LCM order parameter C(V, L), 16 seeds (CRN across V):")
    mean_tbl, std_tbl, raw = {}, {}, {}
    for L in LS:
        mrow, srow, rrow = [], [], []
        for V in VS:
            vals = np.array([lcm(L, M2, V, s, margin=L // 4) / CAL[L] for s in range(NSEED)])
            mrow.append(float(vals.mean())); srow.append(float(vals.std())); rrow.append(vals.tolist())
        mean_tbl[L], std_tbl[L], raw[L] = mrow, srow, rrow
        print(f"    L={L:2d}: C(V) = " + " ".join(f"{m:+.2f}" for m in mrow))
    print("  sample-std (criticality signature peaks near V_c):")
    for L in LS:
        print(f"    L={L:2d}: std  = " + " ".join(f"{s:.2f}" for s in std_tbl[L]))

    print("\n  V_c route 1 (LCM crossings of successive L-pairs):")
    xs_lcm = []
    for La, Lb in zip(LS, LS[1:]):
        xc = crossing(VS, mean_tbl[La], mean_tbl[Lb])
        xs_lcm.append(xc); print(f"    ({La},{Lb}): V_x = {xc}")
    print("  V_c route 2 (Bott-fraction crossings, torus, relaxed gate; nm reported):")
    fb, nm_tot = {}, 0
    for L in (12, 16, 24):
        row = []
        for V in VS:
            bs = [bott_relaxed(L, V, s) for s in range(NSEED)]
            meas = [b for b in bs if b is not None]; nm_tot += NSEED - len(meas)
            row.append(sum(1 for b in meas if b == -1) / max(len(meas), 1))
        fb[L] = row
        print(f"    L={L:2d}: f(V) = " + " ".join(f"{f:.2f}" for f in row))
    xs_bott = []
    for La, Lb in zip((12, 16), (16, 24)):
        xc = crossing(VS, fb[La], fb[Lb])
        xs_bott.append(xc); print(f"    ({La},{Lb}): V_x = {xc}   (total nm: {nm_tot})")

    lcm_xs = [x for x in xs_lcm if x]; bott_xs = [x for x in xs_bott if x]
    Vc_lcm = float(np.mean(lcm_xs)) if lcm_xs else None
    Vc_bott = float(np.mean(bott_xs)) if bott_xs else None
    print(f"\n  V_c: LCM {Vc_lcm} (band {min(lcm_xs) if lcm_xs else '-'}..{max(lcm_xs) if lcm_xs else '-'})  "
          f"Bott {Vc_bott} (band {min(bott_xs) if bott_xs else '-'}..{max(bott_xs) if bott_xs else '-'})")

    # exponent by collapse grid-search + bootstrap over seeds
    if Vc_lcm:
        nus = np.arange(0.8, 4.05, 0.05)
        Vcs = np.arange(Vc_lcm - 0.3, Vc_lcm + 0.31, 0.05)
        best = min(((collapse_obj(mean_tbl, LS, VS, vc, nu), nu, vc) for nu in nus for vc in Vcs))
        S0, nu0, vc0 = best
        # null: V-shuffled collapse objective (does collapse beat chance?)
        rng = np.random.default_rng(0)
        Snull = np.mean([collapse_obj({L: list(rng.permutation(mean_tbl[L])) for L in LS}, LS, VS, vc0, nu0)
                         for _ in range(20)])
        boots = []
        for _ in range(200):
            pick = rng.integers(0, NSEED, NSEED)
            tb = {L: [float(np.mean([raw[L][j][p] for p in pick])) for j in range(len(VS))] for L in LS}
            b = min(((collapse_obj(tb, LS, VS, vc, nu), nu, vc) for nu in nus[::2] for vc in Vcs[::2]))
            boots.append(b[1])
        lo, hi = np.percentile(boots, [16, 84])
        print(f"\n  COLLAPSE: nu = {nu0:.2f}  (bootstrap 68% CI [{lo:.2f}, {hi:.2f}])  V_c* = {vc0:.2f}")
        print(f"            objective {S0:.4f} vs V-shuffled null {Snull:.4f}  -> "
              f"{'collapse REAL' if S0 < 0.3 * Snull else 'collapse NOT better than null -> not measured at these sizes'}")
        inband = lo <= 2.6 and hi >= 2.3
        print(f"\n  PRE-REGISTERED VERDICTS:")
        print(f"    (1) two-route V_c: LCM {Vc_lcm:.2f} vs Bott {Vc_bott if Vc_bott else '-'}  -> "
              f"{'AGREE' if (Vc_bott and abs(Vc_lcm - Vc_bott) < 0.4) else 'tension/nm -- report'}")
        print(f"    (2) nu = {nu0:.2f} CI [{lo:.2f},{hi:.2f}]; [E] IQH band 2.3-2.6 -> "
              f"{'CONSISTENT with plateau-transition class (occupied; claimed as consistency)' if inband else 'outside band at these sizes -> escalation named, no claim'}")
        peak_ok = all(np.argmax(std_tbl[L]) in (2, 3, 4) for L in LS[1:])
        print(f"    (3) marker-std peaks near V_c: {peak_ok}")
        json.dump({"CAL": CAL, "mean": mean_tbl, "std": std_tbl, "fb": fb, "Vc_lcm": Vc_lcm,
                   "Vc_bott": Vc_bott, "nu": nu0, "ci": [float(lo), float(hi)], "S0": S0, "Snull": float(Snull)},
                  open(__file__.replace("gate_P3_scaling.py", "gate_P3_results.json"), "w"), indent=1)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for L, c in zip(LS, (AMBER, BLUE, GREEN, RED)):
            axes[0].plot(VS, mean_tbl[L], "o-", color=c, ms=4, label=f"L={L}")
        axes[0].axhline(-0.5, ls=":", color="#999"); axes[0].set_xlabel("V"); axes[0].set_ylabel("C(V,L)")
        axes[0].set_title("LCM order parameter: crossings locate V_c", fontsize=9, color=NAVY)
        axes[0].legend(fontsize=7.5, frameon=False)
        for L, c in zip(LS, (AMBER, BLUE, GREEN, RED)):
            x = [(V - vc0) * L ** (1 / nu0) for V in VS]
            axes[1].plot(x, mean_tbl[L], "o", color=c, ms=4)
        axes[1].set_xlabel(f"(V-{vc0:.2f}) L^(1/{nu0:.2f})")
        axes[1].set_title(f"data collapse: nu={nu0:.2f} [{lo:.2f},{hi:.2f}]", fontsize=9, color=NAVY)
        for L, c in zip(LS, (AMBER, BLUE, GREEN, RED)):
            axes[2].plot(VS, std_tbl[L], "s-", color=c, ms=4, label=f"L={L}")
        axes[2].set_xlabel("V"); axes[2].set_ylabel("std of C over seeds")
        axes[2].set_title("criticality signature: marker fluctuations peak", fontsize=9, color=NAVY)
        fig.tight_layout(); fig.savefig(__file__.replace("gate_P3_scaling.py", "gate_P3.png"), dpi=160)
        print("  wrote phys/gate_P3.png, phys/gate_P3_results.json")


if __name__ == "__main__":
    main()
