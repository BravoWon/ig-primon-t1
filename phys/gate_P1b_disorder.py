#!/usr/bin/env python
"""Gate P1b -- disordered Chern survival, two-route in real space. Per phys/PREREG_P1b_disorder.md.
Route A: Bott index on L x L torus. Route B: Laughlin flux pump on L x W cylinder.
Real-space Haldane in cell gauge (lattice vectors R1, R2; A-B bonds within-cell, (0,-1), (+1,-1);
NNN A->A +R1, +R2-R1, -R2 with e^{+i phi}; B with e^{-i phi}; +M on A, -M on B).

    python phys/gate_P1b_disorder.py
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T1, T2, PHI = 1.0, 0.2, math.pi / 2
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"


def build(L1, L2, M, V, seed, torus=True, theta=0.0):
    """Real-space Haldane. Sites indexed (n1, n2, s); s=0 A, s=1 B. Returns H, (x,y) cell coords."""
    n = 2 * L1 * L2
    idx = lambda n1, n2, s: 2 * ((n1 % L1) * L2 + (n2 % L2)) + s
    rng = np.random.default_rng(seed)
    dis = V * (rng.random(n) - 0.5)
    off = np.zeros((n, n), complex)
    diag = np.zeros(n)
    ephi, emhi = np.exp(1j * PHI), np.exp(-1j * PHI)
    for n1 in range(L1):
        for n2 in range(L2):
            a, b = idx(n1, n2, 0), idx(n1, n2, 1)
            diag[a] += M + dis[a]; diag[b] += -M + dis[b]
            # seam phase for bonds wrapping the n1-cycle
            ph = lambda d1: np.exp(1j * theta) if (n1 + d1) >= L1 else (np.exp(-1j * theta) if (n1 + d1) < 0 else 1.0)
            # NN: A(n) - B(n), B(n1, n2-1), B(n1+1, n2-1)
            off[a, b] += T1
            if torus or n2 - 1 >= 0:
                off[a, idx(n1, n2 - 1, 1)] += T1
                off[a, idx(n1 + 1, n2 - 1, 1)] += T1 * ph(1)
            # NNN A->A: +R1, +(R2-R1), -R2  (phase e^{+i phi}); B: e^{-i phi}
            off[idx(n1 + 1, n2, 0), a] += T2 * ephi * ph(1)
            off[idx(n1 + 1, n2, 1), b] += T2 * emhi * ph(1)
            if torus or n2 + 1 < L2:
                off[idx(n1 - 1, n2 + 1, 0), a] += T2 * ephi * ph(-1)
                off[idx(n1 - 1, n2 + 1, 1), b] += T2 * emhi * ph(-1)
            if torus or n2 - 1 >= 0:
                off[idx(n1, n2 - 1, 0), a] += T2 * ephi
                off[idx(n1, n2 - 1, 1), b] += T2 * emhi
    H = off + off.conj().T
    H[np.arange(n), np.arange(n)] = diag
    xy = np.array([[(i // 2) // L2, (i // 2) % L2] for i in range(n)], float)
    return H, xy


def bott(L, M, V, seed, EF):
    H, xy = build(L, L, M, V, seed, torus=True)
    w, v = np.linalg.eigh(H)
    if np.min(np.abs(w - EF)) < 0.02 * T1:
        return None
    P = v[:, w < EF]
    Ux = np.exp(2j * math.pi * xy[:, 0] / L)
    Uy = np.exp(2j * math.pi * xy[:, 1] / L)
    U = P.conj().T @ (Ux[:, None] * P)
    Vm = P.conj().T @ (Uy[:, None] * P)
    ev = np.linalg.eigvals(Vm @ U @ np.linalg.inv(Vm) @ np.linalg.inv(U))
    B = float(np.sum(np.angle(ev))) / (2 * math.pi)
    return int(round(B)) if abs(B - round(B)) < 0.1 else None


def pump(L, W, M, V, seed, EF, nth=48):
    """Laughlin charge pump: Q_top(theta) = charge in the top half of the cylinder over one flux
    quantum. States crossing EF give unit jumps; the SMOOTH drift between jumps is the pumped
    charge = Chern number. No branch tracing, no polarization thresholds. Half-step grid offset
    avoids the exact edge-degeneracy at theta=0. Ambiguous jumps (0.3<|dQ|<0.7) -> None (nm)."""
    dth = 2 * math.pi / nth
    thetas = (np.arange(nth) + 0.5) * dth
    Qs = []
    for th in thetas:
        H, xy = build(L, W, M, V, seed, torus=False, theta=th)
        w, v = np.linalg.eigh(H)
        occ = v[:, w < EF]
        top = xy[:, 1] >= W / 2
        Qs.append(float(np.sum(np.abs(occ[top, :]) ** 2)))
    smooth = 0.0
    for q0, q1 in zip(Qs, Qs[1:] + [Qs[0]]):
        d = q1 - q0
        if 0.3 < abs(d) < 0.7:
            return None                                          # ambiguous partial jump -> not measured
        if abs(d) <= 0.3:
            smooth += d
    return int(round(smooth))                                    # sign convention fixed by the V=0 anchor


def main():
    print(f"[P1b] disordered Chern, two-route real space; prereg phys/PREREG_P1b_disorder.md")
    EF = 0.0                                                     # clean mid-gap at Haldane point (d0=0 at phi=pi/2)
    print("\n  anchors (V=0):")
    b0 = bott(12, 0.0, 0.0, 0, EF); p0 = pump(12, 16, 0.0, 0.0, 0, EF)
    bt = bott(12, 6 * T2, 0.0, 0, EF)
    print(f"    Bott(Haldane pt) = {b0}   pump = {p0}   Bott(trivial M=6t2) = {bt}   [expect -1, -1, 0]")
    if not (b0 == p0 == -1 and bt == 0):
        print("    ANCHOR FAIL -- stop."); return

    print("\n  D1: survival at the Haldane point, V-sweep x 5 seeds, L in (8,12,16):")
    Vs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    seeds = range(5)
    agree = disagree = nm = 0
    surv = {}
    for L in (8, 12, 16):
        fr = []
        for V in Vs:
            ok1 = 0; meas = 0
            for s in seeds:
                B = bott(L, 0.0, V, s, EF)
                if B is None:
                    nm += 1; continue
                if L == 12:                                       # pump route at the mid rotation
                    P = pump(L, 16, 0.0, V, s, EF)
                    if P == B:
                        agree += 1
                    else:
                        disagree += 1
                        print(f"      DISAGREE L={L} V={V} seed={s}: Bott {B} pump {P}")
                meas += 1; ok1 += (B == -1)
            fr.append(ok1 / max(meas, 1))
        surv[L] = fr
        print(f"    L={L:2d}: f(V) = " + " ".join(f"{f:.1f}" for f in fr))
    # V_c: 50% crossing at L=16
    f16 = surv[16]; Vc = next((Vs[i] for i in range(len(f16)) if f16[i] < 0.5), None)
    print(f"    -> V_c (first f<0.5, L=16) ~ {Vc}   | two-route: agree {agree}, disagree {disagree}, nm {nm}")

    print("\n  D2: TAI probe just OUTSIDE the phase (M/t2=5.5), V in (0.5..3):")
    M2 = 5.5 * T2
    induced = []
    for V in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        cnt = meas = 0
        for s in seeds:
            B = bott(16, M2, V, s, EF)
            if B is None:
                continue
            meas += 1; cnt += (B == -1)
        induced.append((V, cnt, meas))
        print(f"    V={V}: induced C=-1 in {cnt}/{meas} realizations")
    tai = any(c / max(m, 1) >= 0.6 for _, c, m in induced)

    print(f"\n  PRE-REGISTERED VERDICTS:")
    v1 = disagree == 0 and agree > 0
    print(f"    (1) two-route agreement on measured points ({agree} checks): {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) anchors: PASS")
    mono = all(surv[16][i] >= surv[16][i + 1] - 0.21 for i in range(len(Vs) - 1))
    print(f"    (3) D1: V_c ~ {Vc} (f(V) ~monotone: {mono}; L-stability visible in table above)")
    print(f"        D2: disorder-induced topology at M/t2=5.5: {'YES (TAI-type observed)' if tai else 'NO at these parameters'} -- the answer is the finding")

    json.dump({"survival": {str(k): v for k, v in surv.items()}, "Vs": Vs, "Vc": Vc,
               "tai": induced, "agree": agree, "disagree": disagree, "nm": nm},
              open("phys/gate_P1b_results.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for L, c in ((8, AMBER), (12, BLUE), (16, GREEN)):
        axes[0].plot(Vs, surv[L], "o-", color=c, label=f"L={L}")
    axes[0].axhline(0.5, ls=":", color="#999"); axes[0].set_xlabel("disorder V"); axes[0].set_ylabel("f(C=-1)")
    axes[0].set_title("D1: Chern survival under disorder (Haldane pt)", fontsize=9.5, color=NAVY)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot([v for v, _, _ in induced], [c / max(m, 1) for _, c, m in induced], "s-", color=RED)
    axes[1].set_xlabel("disorder V"); axes[1].set_ylabel("f(C=-1) from TRIVIAL side")
    axes[1].set_title("D2: disorder-INDUCED topology probe (M/t2=5.5)", fontsize=9.5, color=NAVY)
    fig.tight_layout(); fig.savefig("phys/gate_P1b.png", dpi=160); plt.close(fig)
    print("  wrote phys/gate_P1b.png, phys/gate_P1b_results.json")


if __name__ == "__main__":
    main()
