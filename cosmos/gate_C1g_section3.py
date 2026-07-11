#!/usr/bin/env python
"""Gate C1g -- Garfinkle Section III: the outermost gridpoint rides the marginal ray.
Per cosmos/PREREG_C1g_section3.md (registered first). Physics = gate_C1c_garfinkle bars/rates,
untouched. Instrument = truncated grid (0, v_out], v_out iterated onto the marginal ray per the
paper; p* re-bisected to the f64 floor per configuration; CFL=0.1 (C1f A2). Event extraction
INBOUND-ONLY (A5 re-registered); 2-family ladder fit imported from C1f (calibration banked there).

    python cosmos/gate_C1g_section3.py            (full protocol: iterate -> verdicts)
"""
import sys, math, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_C1c_garfinkle as G
from gate_C1f_echoes import fit_timing2, UTRIM

ANCH_D, PBAR, GA = 3.4453, 4.29, 0.3681


def init_trunc(p, vout, n0=800):
    r = np.linspace(vout / n0, vout, n0)
    z = (r - G.R0) / G.SIG
    h = p * np.exp(-z * z) * (3 * r * r - 2 * r ** 3 * z / G.SIG)
    return r, h


def evolve_g(p, vout, cfl=0.1, n0=800, collect=False):
    """C1c scheme on the truncated grid; trace = (u, h1, r_outer).
    Termination amended for truncation (disclosed in prereg): the C1c m_out early-exit fires
    DURING the echo phase here (bounce flux crosses the near-in outer boundary at u=4.857 <
    u* -- measured), clipping the ladder and mislabeling near-critical runs. Replacement:
    dispersal = no h1 sign change for 1.5 u-units after the last one (crossing-timing based,
    amplitude-free); the m_out exit survives only for runs with NO crossings (nothing to clip)."""
    r, h = init_trunc(p, vout, n0)
    u, m_init, trace = 0.0, None, []
    u_lc, s_prev, quiet, steps, mots_min = None, 0.0, 0, 0, 1.0
    while u < G.UEND and len(r) > 8:
        steps += 1
        if steps > 2_000_000:                                    # terminal backstop (amendment 8):
            return ("bh" if mots_min < 2 * G.MOTS_THRESH         # classify by closest approach --
                    else "disp"), trace                          # at 1e-13 brackets either label is
                                                                 # within the threshold's f64 meaning
        if not np.isfinite(h).all():                             # overflow cascade: the g-clip bounds
            return "bh", trace                                   # g but not hdot's product; non-finite
                                                                 # h = collapse-side violence (receipt:
                                                                 # trisect worker LinAlgError, 7b)
        hd1, rd1, g, gbar, h0, h1 = G.rates(r, h)
        mots = gbar / g
        j = int(np.argmin(mots))
        mots_min = min(mots_min, float(mots[j]))
        if mots[j] < G.MOTS_THRESH and r[j] > 5e-4:              # r-floor: a "horizon" at r~1e-8 on a
            return "bh", trace                                   # drained endgame grid is fit noise
                                                                 # (banked mass floor is 1e-3-scale)
        if u > 1.0:                                              # same trim as the event rules
            s = math.copysign(1.0, h1) if h1 != 0 else 0.0
            if s_prev != 0.0 and s != 0.0 and s != s_prev:
                u_lc, quiet = u, 0
            else:
                quiet += 1
            s_prev = s
        m_out = 0.5 * r[-1] * (1 - mots[-1])
        if m_init is None:
            m_init = max(m_out, 1e-30)
        elif u_lc is None and m_out < 1e-3 * m_init:
            return "disp", trace
        elif u_lc is not None and u > u_lc + 1.5:
            return "disp", trace
        du = cfl * float(np.quantile(np.diff(r), 0.05))
        r2 = r + du * rd1
        h2 = h + du * hd1
        keep2 = r2 > 1e-9
        if keep2.sum() < 8:
            break
        hd2 = np.zeros_like(h); rd2 = np.zeros_like(r)
        hd2[keep2], rd2[keep2], *_ = G.rates(r2[keep2], h2[keep2])
        hd2[~keep2] = hd1[~keep2]; rd2[~keep2] = rd1[~keep2]
        r = r + 0.5 * du * (rd1 + rd2)
        h = h + 0.5 * du * (hd1 + hd2)
        keep = r > 1e-9
        r, h = r[keep], h[keep]
        u += du
        if collect and len(r) > 8:
            trace.append((u, float(G.bars(r, h)[1]), float(r[-1])))
        # post-echo regrid FREEZE (termination amendments 2+3+5, disclosed): stop refining when
        # the run has gone quiet, so the grid drains in O(N) steps instead of stalling on a
        # geometrically shrinking du. Amendment 5 RETRACTED the u-based term (u > u_lc + 0.2):
        # the ladder's own inter-crossing gaps reach 1.04 in u, so it froze the regrid INSIDE
        # the cascade and degraded resolution exactly where the next echo needed it (measured
        # at it1: crossing 3 unresolved). The step-based quiet trigger is the sole criterion --
        # it adapts to du naturally (early gaps ~3.5k steps: no fire; post-cascade drain: tens
        # of thousands of tiny steps: fires). MOTS detection continues on the frozen grid.
        frozen = (quiet > 20000 and mots[j] > 1.05 * G.MOTS_THRESH)   # amendment 8: 1.5x left a
                                                                 # hover shell (0.02, 0.03) -- one probe
                                                                 # spun 11.8 CPU-h there; 20k quiet
                                                                 # steps is the real criterion (a final
                                                                 # plunge takes hundreds, not 20k)
        if not frozen and 8 < len(r) < n0 // 2:
            rm = 0.5 * (r[1:] + r[:-1]); hm = 0.5 * (h[1:] + h[:-1])
            rn = np.empty(2 * len(r) - 1); hn = np.empty_like(rn)
            rn[0::2] = r; rn[1::2] = rm
            hn[0::2] = h; hn[1::2] = hm
            r, h = rn, hn
    return "disp", trace


def _probe(args):
    p, vout, cfl = args
    try:
        return evolve_g(p, vout, cfl)[0]
    except Exception:                                            # belt-and-suspenders for the pool:
        return "bh"                                              # any numeric blowup is collapse-side


def bisect_g(vout, cfl=0.1, lo=0.005, hi=0.1, pool=None):
    while evolve_g(hi, vout, cfl)[0] != "bh":                    # registered bracket widening
        hi *= 2
        if hi > 0.4:
            return None
    while evolve_g(lo, vout, cfl)[0] == "bh":                    # lower-side escape guard (PR#13
        lo /= 2                                                  # CodeRabbit): lo must DISPERSE,
        if lo < 1e-4:                                            # else bisection returns silent
            return None                                          # garbage near lo
    lvl = 0
    while (hi - lo) / hi >= 3e-14 and lvl < 26:                  # 3-worker trisection: interval
        ms = [lo + (hi - lo) * k / 4 for k in (1, 2, 3)]         # shrinks 4x per level (~20 levels
        args = [(m_, vout, cfl) for m_ in ms]                    # vs 46 sequential bisections)
        outs = pool.map(_probe, args) if pool else [_probe(a) for a in args]
        newlo, newhi = lo, hi
        for m_, o in zip(ms, outs):                              # monotone: first bh caps hi
            if o == "bh":
                newhi = m_
                break
            newlo = m_
        lo, hi = newlo, newhi
        lvl += 1
        print(f"      trisect[{lvl:02d}] ({lo:.14f}, {hi:.14f}) rel={(hi - lo) / hi:.2e}",
              flush=True)
    return 0.5 * (lo + hi)


def events_inbound(trace):
    """A5 re-registered: strictly inbound events; turnaround = min crossing interval; events end
    BEFORE its second endpoint. Per family drop first + last (A6). Returns (F0, F1, ut, r_out_ut)."""
    tr = np.asarray(trace)
    if len(tr) < 4:
        return None
    u, h, rout = tr[:, 0], tr[:, 1], tr[:, 2]
    m = (u > UTRIM) & (rout > 1e-2)
    # A5b flicker rejection (disclosed, two parts): (i) GRID-EXTENT FLOOR r_outer > 1e-2 —
    # the endgame flicker (hundreds of h1-fit sign flips spaced ~1e-9 in u, which hijacked the
    # turnaround argmin) lives exclusively on the drained grid (r_out ~ 1e-8); real echoes run
    # at r_out ~ 0.1-1. Amplitude-free, instrument-capacity-based. (ii) sign-runs >= 5 steps on
    # each side of a counted crossing (step-length alone failed: at drained-endgame du ~ 1e-9,
    # noise flips persist for many steps — measured before the r_out floor was added).
    u, h, rout = u[m], h[m], rout[m]
    if len(u) < 4:
        return None
    s = np.sign(h)
    runs = []                                                    # (sign, start, length)
    for i in range(len(s)):
        if runs and s[i] == runs[-1][0]:
            runs[-1][2] += 1
        else:
            runs.append([s[i], i, 1])
    j = []
    for a, b in zip(runs[:-1], runs[1:]):
        if a[0] != 0 and b[0] != 0 and a[0] != b[0] and a[2] >= 5 and b[2] >= 5:
            j.append(b[1] - 1)                                   # last index of the left run
    j = np.asarray(j, dtype=int)
    if len(j) < 5:
        return None
    uc = u[j] + (u[j + 1] - u[j]) * np.abs(h[j]) / (np.abs(h[j]) + np.abs(h[j + 1]))
    # A5c geometric band-gating (disclosed): the accepted ladder telescopes -- the next crossing's
    # gap from the last ACCEPTED one must fall in the DSS shrink band [0.05, 0.8] x previous gap.
    # Sub-band gaps are turnaround jitter (measured at v_out=4.5: 9 noise crossings at gaps
    # 1e-5..2e-4 against a real continuation of 1.5e-3, diluting Delta to 0.736); above-band gaps
    # are outbound/growth. Skipped, not terminal: the scan recovers real crossings buried past
    # jitter fog. Turnaround = last accepted crossing.
    acc = [0, 1]
    gprev = uc[1] - uc[0]
    for k in range(2, len(uc)):
        g = uc[k] - uc[acc[-1]]
        if 0.05 * gprev <= g <= 0.8 * gprev:
            acc.append(k)
            gprev = g
    if len(acc) < 4:
        return None
    ut = float(uc[acc[-1]])                                      # turnaround = deepest accepted
    r_ut = float(np.interp(ut, u, rout))
    ucin, jin = uc[acc], j[acc]
    pks = []
    for a, b in zip(jin[:-1], jin[1:]):
        if b > a + 1:
            pks.append(u[a + 1 + int(np.argmax(np.abs(h[a + 1:b + 1])))])
    F0, F1 = ucin[1:-1], np.asarray(pks)[1:-1]
    if len(F0) < 4 or len(F1) < 3:
        return None, ut, r_ut
    return (F0, F1), ut, r_ut


def deep_delta(ps, vout, cfl, eps):
    out, tr = evolve_g(ps * (1 - eps), vout, cfl, collect=True)
    arr = float(tr[-1][0]) if len(tr) else float("nan")          # arrival = grid-drain time
    if out != "disp":
        return None, out, 0, float("nan"), float("nan"), arr
    got = events_inbound(tr)
    if got is None or got[0] is None:
        n = 0 if (got is None) else -1
        ut, r_ut = (float("nan"), float("nan")) if got is None else (got[1], got[2])
        return None, "few-events", n, ut, r_ut, arr
    (F0, F1), ut, r_ut = got
    D, us, sse, D0, D1 = fit_timing2(F0, F1)
    return {"Delta": float(D), "ustar": float(us), "n": [len(F0), len(F1)],
            "fam": [float(D0), float(D1)],
            "F0": [float(x) for x in F0], "F1": [float(x) for x in F1]}, "ok", \
        len(F0) + len(F1), ut, r_ut, arr


KNOWN = {}                                                       # seeds dropped: amendment 8 changes
                                                                 # hover-probe dynamics; re-bisect all


def main():
    import multiprocessing
    pool = multiprocessing.Pool(3)
    print("[C1g] Section III; prereg cosmos/PREREG_C1g_section3.md")
    vout, configs, ups, v_good = 4.5, [], 0, None                # amendment 6: the pile-up receipts
                                                                 # (arrivals 4.859/4.931/4.955 at
                                                                 # 3.0/3.18/3.37, gains collapsing 3x
                                                                 # per rung) show d(arrival)/dv -> 0
                                                                 # below the marginal ray -- JUMP the
                                                                 # pile-up to a safe rung and descend
                                                                 # from above via measured r_ut
    for it in range(8):
        ps = KNOWN.get((round(vout, 4), 0.1)) or bisect_g(vout, pool=pool)
        if ps is None:
            print(f"  it{it}: v_out={vout:.4f} -- bracket failed, nm", flush=True)
            break
        for e in (1e-12, 3.162e-12, 1e-11):                      # registered eps-backoff: bh at the
            fit, status, nev, ut, r_ut, arr = deep_delta(ps, vout, 0.1, e)  # floor = fog/false-MOTS
            if fit is not None or status == "few-events":
                break
            print(f"    (deep eps={e:.3e}: {status} -> backoff)", flush=True)
        D = fit["Delta"] if fit else float("nan")
        print(f"  it{it}: v_out={vout:.4f}  p*={ps:.14f}  [{status}]  events={nev}  Delta={D:.3f}  "
              f"u_t={ut:.4f}  r_out(u_t)={r_ut:.4f}  arrival={arr:.4f}", flush=True)
        configs.append({"vout": vout, "pstar": ps, "n_events": nev, "ut": ut, "r_ut": r_ut,
                        "Delta": D, "arrival": arr})
        if fit is None:                                          # ladder incomplete at this rung
            ups += 1
            if ups > 4:
                print("  protocol nm: ladder not resolved within the ascent budget", flush=True)
                break
            if v_good is None:                                   # ascent: jump the pile-up
                vout = 6.0 if vout >= 4.5 - 1e-9 else 4.5
                print(f"    (ladder incomplete: safe-rung jump -> {vout:.4f})", flush=True)
            else:                                                # descent undershot: split on v
                vout = 0.5 * (vout + v_good)
                print(f"    (descent undershot: splitting -> {vout:.4f})", flush=True)
            continue
        v_good = vout
        if not math.isnan(r_ut) and r_ut < 0.05:
            break
        vout = vout - r_ut + max(0.1 * r_ut, 0.02)
    usable = [c for c in configs if not math.isnan(c["Delta"])]
    if not usable:
        print("  protocol nm: no configuration resolved the ladder"); return
    fin = usable[-1]
    pen = usable[-2] if len(usable) >= 2 else None

    print(f"\n  FINAL config v_out={fin['vout']:.4f}: verdict runs eps in 1e-11/1e-12/3.16e-13:",
          flush=True)
    Ds, diag = [], []
    for e in (1e-11, 1e-12, 3.162e-13):
        fit, status, nev, ut, r_ut, arr = deep_delta(fin["pstar"], fin["vout"], 0.1, e)
        if fit:
            Ds.append(fit["Delta"]); diag.append({"eps": e, **fit})
            print(f"    eps={e:.3e}: events {fit['n'][0]}+{fit['n'][1]}  Delta={fit['Delta']:.3f}"
                  f"  fams {fit['fam'][0]:.2f}/{fit['fam'][1]:.2f}", flush=True)
        else:
            print(f"    eps={e:.3e}: {status} -> nm", flush=True)
    Dmed = float(np.median(Ds)) if Ds else float("nan")
    nev_12 = next((d["n"][0] + d["n"][1] for d in diag if d["eps"] == 1e-12), 0)

    print(f"\n  V3 controls: CFL=0.05 (own bisection) + penultimate "
          f"v_out={pen['vout'] if pen else float('nan'):.4f}:", flush=True)
    ps05 = bisect_g(fin["vout"], cfl=0.05, pool=pool)
    D05 = float("nan")
    if ps05:
        fit05, *_ = deep_delta(ps05, fin["vout"], 0.05, 1e-12)
        if fit05:
            D05 = fit05["Delta"]
            print(f"    CFL=0.05: p*={ps05:.14f}  Delta={D05:.3f}  "
                  f"events {fit05['n'][0]}+{fit05['n'][1]}", flush=True)
    Dpen = pen["Delta"] if pen else float("nan")
    print(f"    penultimate: Delta={Dpen:.3f}  (from iteration record; nan = nm, no second "
          f"resolved config)")

    gB = Dmed / (2 * PBAR)
    v1 = nev_12 >= 10
    v2 = len(Ds) >= 2 and abs(Dmed - ANCH_D) <= 0.25
    v3 = (not math.isnan(D05)) and abs(D05 - Dmed) <= 0.15 and \
         (not math.isnan(Dpen)) and abs(Dpen - Dmed) <= 0.15
    v4 = v2 and v3 and abs(gB - 0.374) <= 0.03 and abs(gB - GA) <= 0.03
    print(f"\n  PRE-REGISTERED VERDICTS:")
    print(f"    (1) ladder >=10 inbound events at eps=1e-12: {'PASS' if v1 else 'FAIL'} ({nev_12})")
    print(f"    (2) Delta={Dmed:.3f} in 3.4453+-0.25: {'PASS' if v2 else 'FAIL'} "
          f"(off {abs(Dmed-ANCH_D):.3f})")
    print(f"    (3) controls |dD|<=0.15 both: {'PASS' if v3 else 'FAIL'} "
          f"(sampling {D05-Dmed:+.3f}, ray-alloc {Dpen-Dmed:+.3f})")
    print(f"    (4) implied gamma_B=Delta/(2*4.29)={gB:.4f} in 0.374+-0.03 and near gamma_A: "
          f"{'PASS' if v4 else 'FAIL'} (off {abs(gB-0.374):.4f}, |dgA|={abs(gB-GA):.4f})")
    if v2 and not v3:
        print("    NOTE: verdict 2 VOIDED by verdict-3 failure (registered rule)")
    json.dump({"configs": configs, "final": fin, "Delta": Dmed, "runs": diag,
               "D05": D05, "pstar05": ps05, "Dpen": Dpen, "gamma_B_implied": float(gB),
               "verdicts": {"v1": bool(v1), "v2": bool(v2), "v3": bool(v3), "v4": bool(v4)}},
              open(os.path.join(HERE, "gate_C1g_results.json"), "w"), indent=1)
    print("  wrote cosmos/gate_C1g_results.json")


if __name__ == "__main__":
    main()
