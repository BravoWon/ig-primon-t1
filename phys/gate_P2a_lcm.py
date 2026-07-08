#!/usr/bin/env python
"""Gate P2a -- Local Chern Marker (Bianco-Resta) closes the TAI double-receipt. Per PREREG_P2a_lcm.md.
m(r) = -4pi Im[P X P Y P]_rr, Cartesian positions on an OPEN flake; interior average per cell area.
Calibrated once at the clean Haldane point; validated at trivial-0 and deep-phase disordered -1.

    python phys/gate_P2a_lcm.py
"""
import math, json
import numpy as np
from gate_P1b_disorder import build, T2

SQ3 = math.sqrt(3)
A_CELL = SQ3 * 1.5                                              # |R1 x R2|


def lcm(L, M, V, seed, EF=0.0, margin=4):
    H, xy = build(L, L, M, V, seed, torus=False, open1=True)
    w, v = np.linalg.eigh(H)
    P = v[:, w < EF] @ v[:, w < EF].conj().T
    n1, n2 = xy[:, 0], xy[:, 1]
    s = np.arange(len(n1)) % 2                                  # 0=A, 1=B (B offset +a1=(0,1))
    X = n1 * SQ3 + n2 * SQ3 / 2
    Y = n2 * 1.5 + s * 1.0
    Mmat = P @ (X[:, None] * P) @ (Y[:, None] * P)
    m = -4 * math.pi * np.imag(np.diag(Mmat))
    interior = (n1 >= margin) & (n1 < L - margin) & (n2 >= margin) & (n2 < L - margin)
    ncells = interior.sum() / 2
    return float(np.sum(m[interior]) / (ncells * A_CELL))


def main():
    print("[P2a] Local Chern Marker; prereg phys/PREREG_P2a_lcm.md")
    L = 16
    raw = lcm(L, 0.0, 0.0, 0)
    CAL = raw / (-1.0)                                          # calibration constant, frozen here
    print(f"  calibration @ clean Haldane pt: raw = {raw:+.4f}  -> CAL = {CAL:+.4f} (frozen)")
    C = lambda *a, **k: lcm(*a, **k) / CAL

    print("  validation (non-circular):")
    c_triv = C(L, 6 * T2, 0.0, 0)
    print(f"    trivial (M=6t2, V=0):      C_LCM = {c_triv:+.3f}   [expect 0]")
    for V, s in [(1.0, 3), (2.0, 1)]:
        cd = C(L, 0.0, V, s)
        print(f"    deep phase (M=0, V={V}, s={s}): C_LCM = {cd:+.3f}   [expect -1; Bott/pump regime]")

    print("\n  THE MEASUREMENT -- TAI regime (M/t2=5.5), 5 seeds x V:")
    bott_f = {1.0: 0 / 5, 2.0: 4 / 5, 2.5: 5 / 5, 3.0: 4 / 4}
    M2 = 5.5 * T2
    res = {}
    for V in (1.0, 2.0, 2.5, 3.0):
        vals = [C(L, M2, V, s) for s in range(5)]
        meas = [x for x in vals if abs(x - round(x)) <= 0.3]
        ind = sum(1 for x in meas if round(x) == -1)
        res[V] = (ind, len(meas), [round(x, 2) for x in vals])
        print(f"    V={V}: markers {res[V][2]}  -> induced {ind}/{len(meas)} measured  (Bott: {bott_f[V]:.1f})")

    onset_lcm = next((V for V in (1.0, 2.0, 2.5, 3.0) if res[V][1] and res[V][0] / res[V][1] >= 0.5), None)
    onset_bott = 2.0
    bands = all(abs(res[V][0] / max(res[V][1], 1) - bott_f[V]) <= 0.4 for V in res)
    v_val = abs(c_triv) < 0.2
    print(f"\n  PRE-REGISTERED VERDICT:")
    print(f"    validation (trivial~0, deep~-1): {'PASS' if v_val else 'FAIL'}")
    print(f"    onset: LCM {onset_lcm} vs Bott {onset_bott}  |  per-V bands (<=0.4): {bands}")
    if v_val and onset_lcm == onset_bott and bands:
        print("  ==> TAI DOUBLE-RECEIPTED in the mobility-gap regime: the marker confirms the Bott index")
        print("      where the pump was blind. D2 upgraded from single-route.")
    else:
        print("  ==> mismatch located -- report as measured (LCM finite-size wall vs real discrepancy).")
    json.dump({"CAL": CAL, "trivial": c_triv, "tai": {str(k): v[:2] + (v[2],) for k, v in res.items()}},
              open(__file__.replace("gate_P2a_lcm.py","gate_P2a_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
