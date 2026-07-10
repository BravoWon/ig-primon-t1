#!/usr/bin/env python
"""Gate C1f -- kill the max(): echo TIMING (-> Delta) + echo COUNTING (-> 2 gamma / Delta).
Per cosmos/PREREG_C1f_echoes.md (registered first). No extremal operator, no amplitude, anywhere:
  Route 1: zero crossings of h1(u) accumulate as u_i = u* - C e^{-i Delta/2}; per-run (u*,C,Delta)
           fit on the registered shrinking-interval prefix -> Delta directly, gamma-free.
  Route 2: crossing count N(eps) = (2 gamma/Delta) ln(1/eps) + const (integer statistics) ->
           gamma_B = (Delta_timing/2) * slope_N.
Crossings counted for u > 1.0 (kinematic trim: inner pulse edge r0-3sigma=0.5 reaches the axis at
u ~ 2*0.5 with rdot=-1/2; earlier h1 flicker is field-free axis noise, eps-independent).

    python cosmos/gate_C1f_echoes.py --smoke     (synthetic calibration a+b, per prereg)
    python cosmos/gate_C1f_echoes.py --pilot     (disclosed pilot: trace machinery at 2 eps)
    python cosmos/gate_C1f_echoes.py             (full gate)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_C1c_garfinkle as G

SMOKE, PILOT = "--smoke" in sys.argv, "--pilot" in sys.argv
UTRIM = 1.0
ANCH_G, ANCH_D = 0.374, 3.4453


def crossings(trace):
    """Interpolated zero-crossing times of h1(u), u > UTRIM."""
    tr = np.asarray(trace)
    if len(tr) < 2:
        return np.array([])
    u, h = tr[:, 0], tr[:, 1]
    m = u > UTRIM
    u, h = u[m], h[m]
    s = np.sign(h)
    j = np.where(s[1:] * s[:-1] < 0)[0]
    return u[j] + (u[j + 1] - u[j]) * np.abs(h[j]) / (np.abs(h[j]) + np.abs(h[j + 1]))


def prefix(uc):
    """Amended rule (A3): drop first crossing; maximal alternation-tolerant shrinking prefix
    (d_i < d_{i-2}; DSS phases alternate within a period, so d_i < d_{i-1} truncates early)."""
    if len(uc) < 4:
        return uc[:0]
    uc = uc[1:]
    d = np.diff(uc)
    n = 2
    while n < len(d) and d[n] < d[n - 2]:
        n += 1
    return uc[:n + 1]


def events(trace):
    """A5-A7 event extraction: inbound crossings cut at the turnaround (global min interval),
    plus one inter-crossing extremum TIME per segment; drop first+last event per family."""
    tr = np.asarray(trace)
    if len(tr) < 4:
        return None
    u, h = tr[:, 0], tr[:, 1]
    m = u > UTRIM
    u, h = u[m], h[m]
    s = np.sign(h)
    j = np.where(s[1:] * s[:-1] < 0)[0]
    if len(j) < 5:
        return None
    uc = u[j] + (u[j + 1] - u[j]) * np.abs(h[j]) / (np.abs(h[j]) + np.abs(h[j + 1]))
    ti = int(np.argmin(np.diff(uc)))                             # A5: turnaround
    ucin, jin = uc[:ti + 2], j[:ti + 2]
    pks = []
    for a, b in zip(jin[:-1], jin[1:]):                          # A7: one extremum per segment
        if b > a + 1:
            pks.append(u[a + 1 + int(np.argmax(np.abs(h[a + 1:b + 1])))])
    F0, F1 = ucin[1:-1], np.asarray(pks)[1:-1]                   # A6: trim first+last per family
    if len(F0) < 3 or len(F1) < 2 or len(F0) + len(F1) < 6:
        return None
    return F0, F1


def fit_timing2(F0, F1):
    """A7 joint 2-family ladder fit: scan u*; per family y=ln(u*-t) vs order index, SHARED slope
    (closed form: pooled cov/var), per-family intercepts. Delta = -2*slope."""
    tmax = max(F0[-1], F1[-1])
    dlast = F0[-1] - F0[-2]
    best = None
    for us in np.linspace(tmax + 1e-6 * dlast, tmax + 2 * dlast, 2000):
        num = den = 0.0
        ys = []
        for F in (F0, F1):
            x = np.arange(len(F))
            y = np.log(us - F)
            num += float(np.sum((x - x.mean()) * (y - y.mean())))
            den += float(np.sum((x - x.mean()) ** 2))
            ys.append((x, y))
        sl = num / den
        sse = sum(float(np.sum((y - y.mean() - sl * (x - x.mean())) ** 2)) for x, y in ys)
        if best is None or sse < best[0]:
            best = (sse, us, sl)
    sse, us, sl = best
    D0 = -2 * np.polyfit(np.arange(len(F0)), np.log(us - F0), 1)[0]
    D1 = -2 * np.polyfit(np.arange(len(F1)), np.log(us - F1), 1)[0]
    return -2 * sl, us, sse, D0, D1


def smoke():
    print("[C1f smoke] synthetic calibration (prereg: must pass before launch)")
    rng = np.random.default_rng(11)
    ok = True
    # (a) timing (A7 two-family, A8 criteria): alternating phase +-10% + 1% jitter, qualifying
    # ladders (>=4+3 events); recovery within the VERDICT band +-0.25 (alternation-adversarial)
    us_true = 5.0
    for n0, n1 in ((4, 3), (5, 4), (6, 5)):
        def fam(n, C, ph):
            tau = np.arange(n) * ANCH_D / 2 + ph + np.where(np.arange(n) % 2 == 0, 0.05, -0.05) * ANCH_D
            t = us_true - C * np.exp(-tau)
            t += rng.normal(0, 0.01, n) * np.gradient(t)
            return np.sort(t)
        F0, F1 = fam(n0, 2.0, 0.0), fam(n1, 2.0, 0.8)
        D, us, sse, D0, D1 = fit_timing2(F0, F1)
        good = abs(D - ANCH_D) <= 0.25
        ok &= good
        print(f"  timing {n0}+{n1} events: Delta={D:.3f} (|d|={abs(D-ANCH_D):.3f})  u*={us:.4f} "
              f"(true {us_true})  fams {D0:.2f}/{D1:.2f}  {'PASS' if good else 'FAIL'}")
    # (b) counting (A8): the real N(eps) is a DETERMINISTIC staircase (pure quantization), with
    # occasional +-1 skip errors (15% of points); 20-draw mean recovery within 5%
    eps = np.array([10 ** (-k / 8.0) for k in range(16, 61)])
    a_true = 1 / 4.606 * 2                                       # A1 doubling: 4 gamma/Delta
    sls = []
    for _ in range(20):
        N = np.floor(a_true * np.log(1 / eps) + 2.0)
        skips = rng.random(len(eps)) < 0.15
        N += skips * rng.choice([-1, 1], len(eps))
        sls.append(np.polyfit(np.log(1 / eps), N, 1)[0])
    mrec = float(np.mean(sls))
    good = abs(mrec - a_true) / a_true <= 0.05
    ok &= good
    print(f"  counting: 20-draw mean slope={mrec:.4f} vs {a_true:.4f} "
          f"(rel {abs(mrec-a_true)/a_true:.3f}, sd {np.std(sls):.4f})  {'PASS' if good else 'FAIL'}")
    print(f"  CALIBRATION: {'PASS' if ok else 'FAIL'}")


def pilot():
    pst = json.load(open(os.path.join(HERE, "gate_C1f_pstars.json")))
    ps01 = pst["pstar_cfl01"]
    print("[C1f pilot] A5-A7 event machinery at the Route-1 instrument (CFL=0.1), disclosed")
    G.CFL = 0.1
    for e in (1e-10, 1e-11, 1e-12):
        out, _, tr = G.evolve(ps01 * (1 - e), collect=True)
        ev = events(tr)
        if ev is None:
            print(f"  eps={e:.0e}: {out}  events -> nm")
            continue
        F0, F1 = ev
        D, us, sse, D0, D1 = fit_timing2(F0, F1)
        print(f"  eps={e:.0e}: {out}  events {len(F0)}+{len(F1)}  Delta={D:.3f}  "
              f"u*={us:.6f}  fams {D0:.2f}/{D1:.2f}")
    G.CFL = 0.4


def main():
    if SMOKE:
        smoke(); return
    if PILOT:
        pilot(); return
    res = json.load(open(os.path.join(HERE, "gate_C1c_results.json")))
    conv = json.load(open(os.path.join(HERE, "gate_C1c_conv1600.json")))
    pst = json.load(open(os.path.join(HERE, "gate_C1f_pstars.json")))
    ps, ps16 = res["pstar"], conv["pstar1600"]
    ps01, ps005 = pst["pstar_cfl01"], pst["pstar_cfl005"]
    gA = res["refit"]["gamma_A"]
    print(f"[C1f] echo timing+counting; prereg cosmos/PREREG_C1f_echoes.md (amendments A1-A4)")
    print(f"  p*(banked)={ps:.14f}  p*(CFL=0.1)={ps01:.14f}  p*(CFL=0.05)={ps005:.14f}")

    print("\n  Route 1 (timing -> Delta), CFL=0.1, A5-A7 two-family events, 6 deep runs:", flush=True)
    G.CFL = 0.1
    Ds, tdiag, nm1 = [], [], []
    for e in (1e-10, 3.162e-11, 1e-11, 3.162e-12, 1e-12, 3.162e-13):
        out, _, tr = G.evolve(ps01 * (1 - e), collect=True)
        ev = events(tr) if out == "disp" else None
        if ev is None:
            nm1.append((e, out))
            print(f"    eps={e:.3e}: {out}  events -> nm (listed)", flush=True)
            continue
        F0, F1 = ev
        D, us, sse, D0, D1 = fit_timing2(F0, F1)
        Ds.append(D)
        tdiag.append({"eps": e, "Delta": float(D), "ustar": float(us),
                      "n_events": [int(len(F0)), int(len(F1))], "fam": [float(D0), float(D1)],
                      "F0": [float(x) for x in F0], "F1": [float(x) for x in F1]})
        print(f"    eps={e:.3e}: events {len(F0)}+{len(F1)}  Delta={D:.3f}  u*={us:.6f}  "
              f"fams {D0:.2f}/{D1:.2f}", flush=True)
    Dmed = float(np.median(Ds)) if len(Ds) >= 4 else float("nan")
    iqr = float(np.subtract(*np.percentile(Ds, [75, 25]))) if len(Ds) >= 4 else float("nan")
    print(f"  Delta_timing = {Dmed:.3f}  (median of {len(Ds)}, IQR {iqr:.3f})")

    print("\n  Route 2 (counting -> 4 gamma/Delta per A1), banked instrument, eighth-decade grid:",
          flush=True)
    G.CFL = 0.4
    lE, Ns = [], []
    for k in range(16, 61):
        e = 10 ** (-k / 8.0)
        out, _, tr = G.evolve(ps * (1 - e), collect=True)
        n = len(crossings(tr))
        if out == "disp":
            lE.append(math.log(1 / e)); Ns.append(n)
        print(f"    eps={e:.4e}: {out}  N={n}", flush=True)
    slope, b = np.polyfit(lE, Ns, 1)
    r2 = 1 - np.var(Ns - (slope * np.asarray(lE) + b)) / np.var(Ns)
    print(f"  slope_N = {slope:.4f}  (R^2={r2:.4f}; 2/slope = {2/slope:.3f} vs banked P 4.28)")
    gB = 0.25 * Dmed * slope                                     # A1: slope = 4 gamma / Delta

    print("\n  V4 controls: timing at CFL=0.05 (A4), counting at N=1600:", flush=True)
    G.CFL = 0.05
    D05 = []
    for e in (1e-11, 1e-12):
        out, _, tr = G.evolve(ps005 * (1 - e), collect=True)
        ev = events(tr) if out == "disp" else None
        if ev is not None:
            D05.append(fit_timing2(*ev)[0])
            print(f"    timing eps={e:.0e}: events {len(ev[0])}+{len(ev[1])}  "
                  f"Delta={D05[-1]:.3f}", flush=True)
        else:
            print(f"    timing eps={e:.0e}: {out}  -> nm", flush=True)
    G.CFL = 0.4
    lE6, Ns6 = [], []
    for kq in range(8, 31):
        e = 10 ** (-kq / 4.0)
        out, _, tr = G.evolve(ps16 * (1 - e), 1600, collect=True)
        n = len(crossings(tr))
        if out == "disp":
            lE6.append(math.log(1 / e)); Ns6.append(n)
        print(f"    eps={e:.4e}: {out}  N={n}", flush=True)
    slope6, _ = np.polyfit(lE6, Ns6, 1)
    D05m = float(np.median(D05)) if D05 else float("nan")
    print(f"    Delta(CFL=0.05)={D05m:.3f} [drift {D05m-Dmed:+.3f}]  "
          f"slope_N(1600)={slope6:.4f} [drift {slope6-slope:+.4f}]")

    v1 = len(Ds) >= 4 and abs(Dmed - ANCH_D) <= 0.25 and iqr <= 1.0
    v2 = v1 and r2 >= 0.95 and abs(gB - ANCH_G) <= 0.02 and abs(gB - gA) <= 0.03
    v3 = r2 >= 0.95 and abs(2 / slope - 4.28) <= 0.5
    v4 = (not math.isnan(D05m)) and abs(D05m - Dmed) <= 0.15 and abs(slope6 - slope) <= 0.05 * slope
    print(f"\n  PRE-REGISTERED VERDICTS (as amended pre-execution):")
    print(f"    (1) Delta_timing in 3.4453+-0.25, IQR<=1.0: {'PASS' if v1 else 'FAIL'}"
          f"  (off {abs(Dmed-ANCH_D):.3f}, IQR {iqr:.3f})")
    print(f"    (2) gamma_B=(D/4)*slope in 0.374+-0.02, |gB-gA|<=0.03: {'PASS' if v2 else 'FAIL'}"
          f"  (gB={gB:.4f}, off {abs(gB-ANCH_G):.4f}, |dgA|={abs(gB-gA):.4f})")
    print(f"    (3) 2/slope_N vs banked P: {'PASS' if v3 else 'FAIL'}  (|d|={abs(2/slope-4.28):.3f})")
    print(f"    (4) controls (|dD|<=0.15 sampling, |dslope|<=5% grid): {'PASS' if v4 else 'FAIL'}"
          f"  (dD={D05m-Dmed:+.3f}, dslope={(slope6-slope)/slope:+.2%})")
    json.dump({"pstar_banked": ps, "pstar_cfl01": ps01, "pstar_cfl005": ps005,
               "Delta_timing": Dmed, "iqr": iqr, "runs_timing": tdiag, "nm1": nm1,
               "slope_N": float(slope), "r2_N": float(r2), "counts": Ns, "lE": lE,
               "gamma_B": float(gB), "gamma_A_used": gA,
               "control": {"slope6": float(slope6), "D05": D05m, "counts6": Ns6},
               "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3), "v4": bool(v4)}},
              open(os.path.join(HERE, "gate_C1f_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1f_results.json")


if __name__ == "__main__":
    main()
