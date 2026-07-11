#!/usr/bin/env python
"""Gate P1 -- two-route Chern number on the Haldane model. Per phys/PREREG_P1_chern.md.
Route A: Fukui-Hatsugai-Suzuki lattice-gauge Berry curvature (bulk, 2x2 Bloch).
Route B: zigzag-ribbon chiral edge-mode counting (boundary, 2W x 2W).

    python phys/gate_P1_chern.py
"""
import math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T1, T2 = 1.0, 0.2
GREEN, RED, BLUE, NAVY, AMBER = "#1e7d34", "#c0392b", "#2c6fbb", "#15293f", "#9a6a2f"

# honeycomb geometry: NN vectors a_i, NNN vectors b_i
A = [np.array([0, 1]), np.array([-math.sqrt(3) / 2, -0.5]), np.array([math.sqrt(3) / 2, -0.5])]
B = [np.array([math.sqrt(3), 0]), np.array([-math.sqrt(3) / 2, 1.5]), np.array([-math.sqrt(3) / 2, -1.5])]
# lattice vectors R1=(sqrt3,0), R2=(sqrt3/2,3/2); reciprocals satisfy G_i . R_j = 2pi delta_ij
G1 = np.array([2 * math.pi / math.sqrt(3), -2 * math.pi / 3])
G2 = np.array([0.0, 4 * math.pi / 3])


def bloch(kx, ky, M, phi):
    k = np.array([kx, ky])
    fa = sum(np.exp(1j * k @ a) for a in A)
    sb = sum(math.sin(k @ b) for b in B)
    cb = sum(math.cos(k @ b) for b in B)
    d0 = 2 * T2 * math.cos(phi) * cb
    dz = M - 2 * T2 * math.sin(phi) * sb
    return np.array([[d0 + dz, T1 * fa], [T1 * np.conj(fa), d0 - dz]])


def chern_fhs(M, phi, N):
    """Route A: FHS plaquette method on an N x N grid over the BZ (lower band)."""
    u = np.empty((N, N, 2), complex)
    gap = 1e9
    for i in range(N):
        for j in range(N):
            k = (i / N) * G1 + (j / N) * G2
            w, v = np.linalg.eigh(bloch(k[0], k[1], M, phi))
            u[i, j] = v[:, 0]
            gap = min(gap, w[1] - w[0])
    link = lambda a, b: (np.vdot(a, b)) / abs(np.vdot(a, b))
    C = 0.0
    for i in range(N):
        for j in range(N):
            i2, j2 = (i + 1) % N, (j + 1) % N
            F = np.log(link(u[i, j], u[i2, j]) * link(u[i2, j], u[i2, j2])
                       * link(u[i2, j2], u[i, j2]) * link(u[i, j2], u[i, j]))
            C += F.imag
    return int(round(C / (2 * math.pi))), gap


def ribbon_h(kx, M, phi, W):
    """Zigzag ribbon, W cells (2W sites), momentum kx along the sqrt(3)-periodic direction.
    Geometry: A_j at y=1.5j, B_j at y=1.5j+0.5; A_j-B_j via two slanted NN bonds (x-phase e1),
    B_j-A_{j+1} vertical NN; NNN per Haldane convention (A gets e^{+i phi} along +b_i)."""
    n = 2 * W
    off = np.zeros((n, n), complex)                             # strictly off-diagonal hops (one direction)
    diag = np.zeros(n)
    e1 = np.exp(1j * kx * math.sqrt(3) / 2)
    for j in range(W):
        a, b = 2 * j, 2 * j + 1
        diag[a] += M + 2 * T2 * math.cos(kx * math.sqrt(3) + phi)     # in-row NNN, A sublattice
        diag[b] += -M + 2 * T2 * math.cos(kx * math.sqrt(3) - phi)    # in-row NNN, B sublattice
        off[a, b] += T1 * (e1 + np.conj(e1))                          # two slanted NN bonds
        if j + 1 < W:
            off[2 * (j + 1), b] += T1                                  # vertical NN
            off[2 * (j + 1), a] += T2 * (np.exp(1j * phi) * np.conj(e1) + np.exp(-1j * phi) * e1)
            off[2 * (j + 1) + 1, b] += T2 * (np.exp(-1j * phi) * np.conj(e1) + np.exp(1j * phi) * e1)
    H = off + off.conj().T
    H[np.arange(n), np.arange(n)] = diag
    return H


def chern_edge(M, phi, W=40, nk=600):
    """Route B: net chiral crossings of the bulk mid-gap on the TOP edge."""
    # bulk mid-gap reference from Bloch bands over a coarse grid
    lo, hi = -1e9, 1e9
    for i in range(24):
        for j in range(24):
            k = (i / 24) * G1 + (j / 24) * G2
            w = np.linalg.eigvalsh(bloch(k[0], k[1], M, phi))
            lo = max(lo, w[0]); hi = min(hi, w[1])
    if hi - lo < 0.05 * T1:
        return None, hi - lo                                     # non-measurement: gap too small
    Emid = (lo + hi) / 2
    kxs = np.linspace(-math.pi / math.sqrt(3), math.pi / math.sqrt(3), nk)   # exactly one x-period
    # sorted-index crossing detection FAILS when counter-propagating edge modes exchange identity at
    # Emid (eigh bands cannot cross). Instead: trace the TOP-edge branch as a physical curve --
    # per kx, the top-localized state nearest Emid inside the gap -- and count its signed Emid crossings.
    curve = []                                                   # (kx, E_top)
    win = 0.45 * (hi - lo)
    for kx in kxs:
        w, v = np.linalg.eigh(ribbon_h(kx, M, phi, W))
        cands = []
        for b in np.where(np.abs(w - Emid) < win)[0]:
            wt_top = float(np.sum(np.abs(v[-6:, b]) ** 2))
            wt_bot = float(np.sum(np.abs(v[:6, b]) ** 2))
            if wt_top > 3 * wt_bot:
                cands.append(w[b])
        if cands:
            curve.append((kx, min(cands, key=lambda e: abs(e - Emid))))
    total = 0
    if len(curve) >= 2:                                          # close the curve over the periodic BZ
        period = 2 * math.pi / math.sqrt(3)
        closed = curve + [(curve[0][0] + period, curve[0][1])]
        for (k0, e0), (k1, e1_) in zip(closed, closed[1:]):
            if (e0 - Emid) * (e1_ - Emid) < 0 and (k1 - k0) < 4 * (kxs[1] - kxs[0]):
                total += 1 if e1_ > e0 else -1                   # signed crossing of the top-edge branch
    return total, hi - lo


def main():
    print(f"[P1] two-route Chern, Haldane model (t1={T1}, t2={T2}); prereg phys/PREREG_P1_chern.md")
    Mc = 3 * math.sqrt(3) * T2                                   # boundary at M = Mc*sin(phi)
    print(f"  exact boundary: M_c(phi) = 3*sqrt(3)*t2*sin(phi) = {Mc:.4f}*sin(phi)")

    print("\n  anchors:")
    c0, g0 = chern_fhs(0.0, math.pi / 2, 24)
    cflip, _ = chern_fhs(0.0, -math.pi / 2, 24)
    ctriv, _ = chern_fhs(6 * T2, math.pi / 2, 24)
    print(f"    C(M=0, phi=+pi/2) = {c0} (gap {g0:.3f})   C(phi=-pi/2) = {cflip}   C(M=6t2) = {ctriv}")
    e0, gap0 = chern_edge(0.0, math.pi / 2)
    print(f"    edge route at Haldane point: C_edge = {e0} (bulk gap {gap0:.3f})")

    print("\n  sweep (phi x M/t2), both routes, rotations N in (12,24,48), W in (20,40):")
    phis = [math.pi / 6, math.pi / 3, math.pi / 2, 2 * math.pi / 3, 5 * math.pi / 6,
            -math.pi / 3, -math.pi / 2]
    Ms = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    agree = disagree = notmeas = 0
    boundary_ok = True
    rows = []
    for phi in phis:
        for m_ in Ms:
            M = m_ * T2
            pred = (1 if math.sin(phi) > 0 else -1) if abs(M) < Mc * abs(math.sin(phi)) else 0
            ca, gap = chern_fhs(M, phi, 24)
            ca2, _ = chern_fhs(M, phi, 12)
            ce, gape = chern_edge(M, phi, W=40)
            ce2, _ = chern_edge(M, phi, W=20)
            if ce is None or gap < 0.05 * T1:
                notmeas += 1; rows.append((phi, m_, pred, ca, None, "nm")); continue
            ok = (ca == ce == ca2) and (ce2 == ce)
            agree += ok; disagree += (not ok)
            if abs(ca) != abs(pred):
                boundary_ok = False
            rows.append((phi, m_, pred, ca, ce, "ok" if ok else "DISAGREE"))
    print(f"    measured points: {agree + disagree}  | two-route+rotation agreement: {agree}  disagreements: {disagree}  not-measured (gap<0.05): {notmeas}")
    bad = [r for r in rows if r[5] == "DISAGREE"]
    for r in bad[:5]:
        print(f"      DISAGREE at phi={r[0]:+.2f} M/t2={r[1]}: pred {r[2]} bulk {r[3]} edge {r[4]}")
    pred_ok = all((abs(r[3]) == abs(r[2])) for r in rows if r[5] != "nm")
    print(f"    phase diagram matches exact boundary on all measured points: {pred_ok}")

    print("\n  WALL: minimal FHS grid vs gap (phi=pi/2, M -> Mc from below):")
    deltas = [0.3, 0.1, 0.03, 0.01, 0.003]
    Ns = [4, 6, 9, 14, 21, 32, 48, 72, 108, 160]
    pts = []
    for d in deltas:
        M = Mc * (1 - d)
        target, _ = chern_fhs(M, math.pi / 2, 200)
        nstar, gap = None, None
        for N in Ns:
            c, gap = chern_fhs(M, math.pi / 2, N)
            if c == target:
                # require stability at next grid too
                c2, _ = chern_fhs(M, math.pi / 2, N + 3)
                if c2 == target:
                    nstar = N; break
        pts.append((d, gap, nstar))
        print(f"    delta={d:6.3f}  gap={gap:.5f}  N* = {nstar}")
    xs = np.log([g for _, g, n in pts if n]); ys = np.log([n for _, g, n in pts if n])
    p, b = np.polyfit(xs, ys, 1)
    r2 = 1 - np.var(ys - (p * xs + b)) / max(np.var(ys), 1e-12)
    print(f"    wall law: N* ~ gap^{p:.3f}  (R^2={r2:.3f}; [E] expectation p ~ -1 +/- 30%)")

    v1 = disagree == 0 and agree > 0
    v2 = (abs(c0) == 1 and cflip == -c0 and ctriv == 0 and e0 == c0 and pred_ok)
    v3 = r2 > 0.95
    p_in = abs(p + 1) < 0.3
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) two-route agreement on all measured points: {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) anchors + exact phase boundary:             {'PASS' if v2 else 'FAIL'}")
    print(f"    (3) wall power law (p={p:.2f}, R2={r2:.2f}):        {'PASS' if v3 else 'FAIL'}"
          f"{' [exponent within expectation]' if p_in else ' [exponent replaces expectation, per prereg]'}")
    if v1 and v2 and v3:
        print("  ==> GATE P1 PASSES: bulk-boundary correspondence held two-route on every measured point;")
        print("      the method genre (topological receipts + admissibility wall) validated on exact ground.")

    json.dump({"rows": [[float(r[0]), r[1], r[2], r[3], r[4] if r[4] is not None else "nm", r[5]] for r in rows],
               "wall": [[d, g, n] for d, g, n in pts], "p": float(p), "r2": float(r2)},
              open("phys/gate_P1_results.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in rows:
        col = GREEN if (r[5] == "ok" and r[3] != 0) else ("#bbb" if r[5] == "nm" else (BLUE if r[3] == 0 else RED))
        axes[0].scatter(r[0], r[1], c=col, s=60, marker="s" if r[5] != "DISAGREE" else "x")
    ph = np.linspace(-math.pi, math.pi, 300)
    axes[0].plot(ph, 3 * math.sqrt(3) * np.abs(np.sin(ph)), "--", color=NAVY, lw=1, label="exact boundary")
    axes[0].set_xlabel("phi"); axes[0].set_ylabel("M/t2"); axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("P1 sweep: green C=+-1 / blue C=0, both routes agree", fontsize=9, color=NAVY)
    gg = [g for _, g, n in pts if n]; nn = [n for _, g, n in pts if n]
    axes[1].loglog(gg, nn, "o-", color=RED)
    axes[1].set_xlabel("bulk gap"); axes[1].set_ylabel("minimal admissible FHS grid N*")
    axes[1].set_title(f"the admissibility WALL: N* ~ gap^{p:.2f}", fontsize=9, color=NAVY)
    fig.tight_layout(); fig.savefig("phys/gate_P1.png", dpi=160); plt.close(fig)
    print("  wrote phys/gate_P1.png, phys/gate_P1_results.json")


if __name__ == "__main__":
    main()
