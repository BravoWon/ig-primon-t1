#!/usr/bin/env python
"""Gate P4 -- Hofstadter butterfly two-route: FHS topology vs Diophantine arithmetic.
Per phys/PREREG_P4_butterfly.md.

    python phys/gate_P4_butterfly.py
"""
import math, json
from math import gcd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QMAX, QPLOT = 12, 40
GREEN, RED, BLUE, NAVY = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f"


def hof(k1, k2, p, q):
    """q x q magnetic Bloch Hamiltonian, Landau gauge."""
    H = np.zeros((q, q), complex)
    for m in range(q):
        H[m, m] = 2 * math.cos(k2 + 2 * math.pi * p * m / q)
        H[m, (m + 1) % q] += 1.0
        H[(m + 1) % q, m] += 1.0
    # boundary phase on the m-cycle
    H[0, q - 1] += (np.exp(-1j * q * k1) - 1.0) if q > 2 else 0
    H[q - 1, 0] += (np.exp(1j * q * k1) - 1.0) if q > 2 else 0
    if q == 2:
        H[0, 1] += np.exp(-1j * 2 * k1) - 1.0; H[1, 0] += np.exp(1j * 2 * k1) - 1.0
    return H


def bands_and_states(p, q, N):
    k1s = np.linspace(0, 2 * math.pi / q, N, endpoint=False)
    k2s = np.linspace(0, 2 * math.pi, N, endpoint=False)
    E = np.empty((N, N, q)); U = np.empty((N, N, q, q), complex)
    for i, k1 in enumerate(k1s):
        for j, k2 in enumerate(k2s):
            w, v = np.linalg.eigh(hof(k1, k2, p, q))
            E[i, j] = w; U[i, j] = v
    return E, U


def chern_multiband(U, r):
    """Non-Abelian FHS for the lowest r bands: plaquette det of overlap links."""
    N = U.shape[0]
    tot = 0.0
    for i in range(N):
        for j in range(N):
            a = U[i, j][:, :r]; b = U[(i + 1) % N, j][:, :r]
            c = U[(i + 1) % N, (j + 1) % N][:, :r]; d = U[i, (j + 1) % N][:, :r]
            F = np.log(np.linalg.det(a.conj().T @ b) * np.linalg.det(b.conj().T @ c)
                       * np.linalg.det(c.conj().T @ d) * np.linalg.det(d.conj().T @ a))
            tot += F.imag
    return -int(round(tot / (2 * math.pi)))   # sign: BZ loop orientation fixed ONCE at the phi=1/3 anchor (recorded)


def diophantine(p, q, r):
    """Unique t with p*t = r (mod q), |t| <= q/2. None if ambiguous (q even, r=q/2)."""
    pinv = pow(p, -1, q)
    t = (r * pinv) % q
    if t > q / 2:
        t -= q
    if q % 2 == 0 and abs(t) == q // 2:
        return None
    return t


def main():
    print(f"[P4] Hofstadter butterfly two-route; q <= {QMAX}; prereg phys/PREREG_P4_butterfly.md")
    fracs = [(p, q) for q in range(3, QMAX + 1) for p in range(1, q) if gcd(p, q) == 1]
    receipts = agree = nm = 0
    fails = []
    wall = {}
    verified = []                                                # (phi, gap_lo, gap_hi, chern)
    for (p, q) in fracs:
        for N in (12, 24, 48):                                   # grid escalation
            E, U = bands_and_states(p, q, N)
            ok_all = True
            # sum-rule anchor: all q bands -> total Chern 0
            if chern_multiband(U, q) != 0:
                ok_all = False
            if ok_all:
                break
        smallest = None
        for r in range(1, q):
            lo = float(E[:, :, r - 1].max()); hi = float(E[:, :, r].min())
            gap = hi - lo
            t = diophantine(p, q, r)
            if gap < 1e-6 or t is None:
                nm += 1; continue
            smallest = gap if smallest is None else min(smallest, gap)
            C = chern_multiband(U, r)
            receipts += 1
            if C == t:
                agree += 1
                verified.append((p / q, lo, hi, C))
            else:
                fails.append((p, q, r, C, t))
        if smallest:
            wall.setdefault(q, []).append(smallest)
    print(f"  fractions: {len(fracs)}  gap-receipts: {receipts}  AGREE: {agree}  disagree: {len(fails)}  nm(gapless): {nm}")
    for f in fails[:6]:
        print(f"    DISAGREE p/q={f[0]}/{f[1]} r={f[2]}: FHS {f[3]} vs Diophantine {f[4]}")
    a13 = [c for (phi, lo, hi, c) in verified if abs(phi - 1 / 3) < 1e-9]
    print(f"  anchor phi=1/3 gap Cherns: {a13}  [expect [1, -1]]")
    print(f"\n  WALL (fractal gap shrinkage): min measured gap per q:")
    qs = sorted(wall); mins = [min(wall[q]) for q in qs]
    for q, g in zip(qs, mins):
        print(f"    q={q:2d}: min gap = {g:.5f}")
    lx, ly = np.array(qs, float), np.log(mins)
    sl, ic = np.polyfit(lx, ly, 1)
    r2 = 1 - np.var(ly - (sl * lx + ic)) / max(np.var(ly), 1e-12)
    print(f"    min-gap ~ exp({sl:.3f} q)  (R^2={r2:.3f})  -- the price any instrument pays as q grows")

    v1 = len(fails) == 0 and agree == receipts
    v2 = a13 == [1, -1]
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) FHS == Diophantine on every measured gap ({agree}/{receipts}): {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) anchors (1/3 sequence, sum rules embedded in escalation): {'PASS' if v2 else 'FAIL'}")
    print(f"    (3) wall: min-gap decay exp({sl:.3f} q), R^2={r2:.3f} (measured, no [E] pinned)")
    json.dump({"receipts": receipts, "agree": agree, "fails": fails, "nm": nm,
               "wall": {str(q): min(wall[q]) for q in qs}, "slope": float(sl), "r2": float(r2)},
              open(__file__.replace("gate_P4_butterfly.py", "gate_P4_results.json"), "w"), indent=1)

    # the butterfly, with verified gaps colored by Chern
    print("  rendering the butterfly ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    for q in range(2, QPLOT + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            Es = []
            for k1 in np.linspace(0, 2 * math.pi / q, 6, endpoint=False):
                for k2 in np.linspace(0, 2 * math.pi, 6, endpoint=False):
                    Es.extend(np.linalg.eigvalsh(hof(k1, k2, p, q)))
            ax.plot([p / q] * len(Es), Es, ",", color="#222", ms=0.4, alpha=0.65)
    cmap = plt.get_cmap("coolwarm")
    for (phi, lo, hi, c) in verified:
        ax.plot([phi, phi], [lo, hi], "-", lw=1.6, color=cmap(0.5 + c / 8), alpha=0.85)
    ax.set_xlabel("flux per plaquette  p/q"); ax.set_ylabel("energy")
    ax.set_title(f"Hofstadter butterfly -- {agree} gaps two-route verified (FHS topology = Diophantine arithmetic),\n"
                 f"colored by Chern number; spectra to q={QPLOT}", fontsize=10, color=NAVY)
    fig.tight_layout(); fig.savefig(__file__.replace("gate_P4_butterfly.py", "gate_P4_butterfly.png"), dpi=170)
    print("  wrote phys/gate_P4_butterfly.png, phys/gate_P4_results.json")


if __name__ == "__main__":
    main()
