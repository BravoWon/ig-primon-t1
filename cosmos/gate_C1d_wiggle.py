#!/usr/bin/env python
"""Gate C1d -- wiggle-native Route B. Per cosmos/PREREG_C1d_wiggle.md (registered first).
Model: ln y = c + s ln eps + sum_{m=1,2} [a_m sin(2 pi m ln_eps / P) + b_m cos(...)], P scanned.
Route B (max h1^2, s=-2*gamma) PRIMARY; Route A (M, s=+gamma) conditional second channel.
Detection = permutation-surrogate SSE gate + interior P + half-split phase coherence.
Anchors: gamma=0.374, Delta=3.4453 (Gundlach). Data: banked C1c dense quarter-decade points
(even eighth-k) + NEW odd eighth-k points, same instrument, banked p*.

    python cosmos/gate_C1d_wiggle.py [--smoke]     (smoke = synthetic calibration, no evolves)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_C1c_garfinkle as G

SMOKE = "--smoke" in sys.argv
SCAN = np.linspace(2.0, 9.0, 400)                   # period scan in ln-eps (pre-registered)
NSURR, SEED = 200, 20260709
ANCH_G, ANCH_D = 0.374, 3.4453


def design(lx, P):
    w = 2 * math.pi / P
    return np.vstack([np.ones_like(lx), lx, np.sin(w * lx), np.cos(w * lx),
                      np.sin(2 * w * lx), np.cos(2 * w * lx)]).T


def fitP(lx, ly):
    best = None
    for P in SCAN:
        X = design(lx, P)
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        sse = float(np.sum((ly - X @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(P), beta)
    return best


def pipeline(eps, y, sign, do_surr=True):
    """sign: gamma = sign * slope (+1 route A, -0.5 route B). Returns full diagnostics dict."""
    lx, ly = np.log(np.asarray(eps)), np.log(np.asarray(y))
    lin = np.polyfit(lx, ly, 1)
    sse0 = float(np.sum((ly - np.polyval(lin, lx)) ** 2))
    sse, P, beta = fitP(lx, ly)
    gamma = float(sign * beta[1])
    Delta = 2 * abs(gamma) * P
    A1 = math.hypot(beta[2], beta[3])
    red_real = sse / max(sse0, 1e-300)
    out = {"n": len(lx), "gamma": gamma, "P": P, "Delta": float(Delta), "amp1": float(A1),
           "sse_lin": sse0, "sse_wig": sse, "reduction": float(red_real)}
    if not do_surr:
        return out
    # permutation-surrogate null (pre-registered: same scan, residuals of the LINEAR fit permuted)
    rng = np.random.default_rng(SEED)
    base, resid = np.polyval(lin, lx), ly - np.polyval(lin, lx)
    reds = []
    for _ in range(NSURR):
        ys = base + rng.permutation(resid)
        l2 = np.polyfit(lx, ys, 1)
        s0 = float(np.sum((ys - np.polyval(l2, lx)) ** 2))
        ss, _, _ = fitP(lx, ys)
        reds.append(ss / max(s0, 1e-300))
    pval = (1 + sum(r <= red_real for r in reds)) / (NSURR + 1)
    # half-split phase coherence at fixed global P (fundamental phase per half)
    mid = 0.5 * (lx.min() + lx.max())
    phis = []
    for m in (lx <= mid, lx > mid):
        Xh = design(lx[m], P)
        bh, *_ = np.linalg.lstsq(Xh, ly[m], rcond=None)
        phis.append(math.atan2(bh[3], bh[2]))       # a sin + b cos = C sin(. + atan2(b, a))
    dphi = abs((phis[1] - phis[0] + math.pi) % (2 * math.pi) - math.pi)
    detect = (pval < 0.05) and (2.2 < P < 8.5) and (dphi <= math.pi / 3)
    out.update({"pval": float(pval), "dphi": float(dphi), "detect": bool(detect),
                "surr_red_p5": float(np.percentile(reds, 5))})
    return out


def eps_grid(k_lo, k_hi):
    return [10 ** (-k / 8.0) for k in range(k_lo, k_hi + 1)]


def smoke():
    print("[C1d smoke] synthetic calibration on the registered grids (prereg gate: must pass)")
    rng = np.random.default_rng(7)
    ok = True
    for name, eps in (("B-grid", eps_grid(16, 60)), ("A-grid", eps_grid(8, 40))):
        lx = np.log(np.asarray(eps))
        P0 = ANCH_D / (2 * ANCH_G)
        # sawtooth (2 harmonics of a real sawtooth), amplitude 0.26, ln-noise 0.05
        saw = 0.26 * (np.sin(2 * math.pi * lx / P0) + 0.5 * np.sin(4 * math.pi * lx / P0 + 0.7))
        ly = 3.0 - 2 * ANCH_G * lx + saw + rng.normal(0, 0.05, len(lx))
        r = pipeline(eps, np.exp(ly), -0.5)
        g_ok, d_ok = abs(r["gamma"] - ANCH_G) <= 0.01, abs(r["Delta"] - ANCH_D) <= 0.15
        print(f"  inject {name}: gamma={r['gamma']:.4f} (|d|={abs(r['gamma']-ANCH_G):.4f})  "
              f"Delta={r['Delta']:.3f} (|d|={abs(r['Delta']-ANCH_D):.3f})  p={r['pval']:.4f}  "
              f"dphi={r['dphi']:.2f}  detect={r['detect']}")
        ok &= g_ok and d_ok and r["detect"]
        ly0 = 3.0 - 2 * ANCH_G * lx + rng.normal(0, 0.05, len(lx))
        r0 = pipeline(eps, np.exp(ly0), -0.5)
        print(f"  null   {name}: p={r0['pval']:.4f}  detect={r0['detect']}  (must be False)")
        ok &= not r0["detect"]
    print(f"  CALIBRATION: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if SMOKE:
        smoke()
        return
    res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
    dense = json.load(open(os.path.join(HERE, "gate_C1c_dense.json")))
    conv = json.load(open(os.path.join(HERE, "gate_C1c_conv1600.json")))
    ps, ps16 = res["pstar"], conv["pstar1600"]
    print(f"[C1d] wiggle-native gate; prereg cosmos/PREREG_C1d_wiggle.md; p*={ps:.14f}")

    banked = {"A": dict(zip([round(-8 * math.log10(e)) for e in dense["A"]["eps"]], dense["A"]["M"])),
              "B": dict(zip([round(-8 * math.log10(e)) for e in dense["B"]["eps"]], dense["B"]["R"]))}
    data = {"A": {}, "B": {}}
    dropped = []
    for route, klo, khi, sup in (("A", 8, 40, True), ("B", 16, 60, False)):
        for k in range(klo, khi + 1):
            e = 10 ** (-k / 8.0)
            if k in banked[route]:
                data[route][k] = (e, banked[route][k], "banked")
                continue
            out, v, _ = G.evolve(ps * (1 + e) if sup else ps * (1 - e))
            want = "bh" if sup else "disp"
            if out == want and v > 0:
                data[route][k] = (e, v, "new")
            else:
                dropped.append((route, e, out))
            print(f"  {route} eps={e:.4e}: {out}  {'M' if sup else 'maxh1^2'}={v:.5e}", flush=True)
    if dropped:
        print(f"  dropped (wrong phase, listed per prereg): {dropped}")

    epsB = [data['B'][k][0] for k in sorted(data['B'])]
    yB = [data['B'][k][1] for k in sorted(data['B'])]
    epsA = [data['A'][k][0] for k in sorted(data['A'])]
    yA = [data['A'][k][1] for k in sorted(data['A'])]
    print(f"\n  Route B pipeline (PRIMARY, n={len(epsB)}):", flush=True)
    rB = pipeline(epsB, yB, -0.5)
    print(f"    gamma_B={rB['gamma']:.4f}  P={rB['P']:.3f}  Delta_B={rB['Delta']:.3f}  amp={rB['amp1']:.3f}"
          f"  SSE x{rB['reduction']:.3f}  p={rB['pval']:.4f}  dphi={rB['dphi']:.2f}  detect={rB['detect']}")
    print(f"  Route A pipeline (conditional, n={len(epsA)}):", flush=True)
    rA = pipeline(epsA, yA, +1.0)
    print(f"    gamma_A={rA['gamma']:.4f}  P={rA['P']:.3f}  Delta_A={rA['Delta']:.3f}  amp={rA['amp1']:.3f}"
          f"  SSE x{rA['reduction']:.3f}  p={rA['pval']:.4f}  dphi={rA['dphi']:.2f}  detect={rA['detect']}")

    print(f"\n  V5 control: N=1600, banked p*(1600)={ps16:.12f}, quarter-decade k'=8..30:", flush=True)
    e16, y16 = [], []
    for kq in range(8, 31):
        e = 10 ** (-kq / 4.0)
        out, v, _ = G.evolve(ps16 * (1 - e), 1600)
        if out == "disp" and v > 0:
            e16.append(e); y16.append(v)
        print(f"    eps={e:.4e}: {out}  maxh1^2={v:.5e}", flush=True)
    r16 = pipeline(e16, y16, -0.5, do_surr=False)
    print(f"    gamma_B(1600)={r16['gamma']:.4f}  Delta_B(1600)={r16['Delta']:.3f}")

    v1 = rB["detect"]
    v2 = v1 and abs(rB["gamma"] - ANCH_G) <= 0.02 and abs(rA["gamma"] - rB["gamma"]) <= 0.03
    v3 = v1 and abs(rB["Delta"] - ANCH_D) <= 0.25
    v4 = ("nm (Route A detection failed -- advisory)" if not rA["detect"]
          else ("PASS" if abs(rA["Delta"] - rB["Delta"]) <= 0.4 else "FAIL"))
    v5 = v1 and abs(r16["gamma"] - rB["gamma"]) <= 0.02 and abs(r16["Delta"] - rB["Delta"]) <= 0.3
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) detection (surrogate p<0.05, interior P, |dphi|<=pi/3): {'PASS' if v1 else 'FAIL'}")
    print(f"    (2) |gamma_B-0.374|<=0.02 AND |gamma_A-gamma_B|<=0.03: {'PASS' if v2 else 'FAIL'}"
          f"  (gB off {abs(rB['gamma']-ANCH_G):.4f}, |dg|={abs(rA['gamma']-rB['gamma']):.4f})")
    print(f"    (3) Delta_B within 3.4453 +- 0.25: {'PASS' if v3 else 'FAIL'} (off {abs(rB['Delta']-ANCH_D):.3f})")
    print(f"    (4) two-channel |Delta_A-Delta_B|<=0.4: {v4}")
    print(f"    (5) N=1600 control (|dgamma|<=0.02, |dDelta|<=0.3): {'PASS' if v5 else 'FAIL'}"
          f"  (dg={abs(r16['gamma']-rB['gamma']):.4f}, dD={abs(r16['Delta']-rB['Delta']):.3f})")
    json.dump({"pstar": ps, "routeB": rB, "routeA": rA, "control1600": r16,
               "epsB": epsB, "yB": yB, "epsA": epsA, "yA": yA, "eps16": e16, "y16": y16,
               "dropped": dropped, "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3),
                                                "v4": str(v4), "v5": bool(v5)}},
              open(os.path.join(HERE, "gate_C1d_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1d_results.json")


if __name__ == "__main__":
    main()
