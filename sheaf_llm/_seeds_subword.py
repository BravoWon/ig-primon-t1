#!/usr/bin/env python
"""Decisive robustness: is char+grounded < char ROBUST, or is the -3.5% seed noise? Varies H set + init.
Only the two decisive arms (the +18% grounded-vs-char gap is large enough to trust single-seed)."""
import numpy as np
import subword_gate as sg

rows = []
for s in (0, 1, 2):
    sg.SEED = s
    d = sg.build()
    out = {}
    for mode in ("char", "char+grounded"):
        _, ov, bk = sg.run(mode, d)
        out[mode] = (ov, bk[2])                  # overall, post-H
    rows.append((s, out["char"][0], out["char+grounded"][0], out["char"][1], out["char+grounded"][1]))
    co, go, cH, gH = rows[-1][1:]
    print(f"  seed {s}: overall char {co:.1f} / char+grnd {go:.1f}   post-H char {cH:.1f} / char+grnd {gH:.1f}"
          f"   (post-H delta {(gH-cH)/cH*100:+.1f}%)")

a = np.array([r[1:] for r in rows])
mco, mgo, mcH, mgH = a.mean(0)
dH = [(r[4] - r[3]) / r[3] * 100 for r in rows]
dO = [(r[2] - r[1]) / r[1] * 100 for r in rows]
print(f"\n=== mean over seeds (char -> char+grounded) ===")
print(f"  overall  {mco:6.1f} -> {mgo:6.1f}   {(mgo-mco)/mco*100:+5.1f}%   per-seed {[round(x,1) for x in dO]}")
print(f"  post-H   {mcH:6.1f} -> {mgH:6.1f}   {(mgH-mcH)/mcH*100:+5.1f}%   per-seed {[round(x,1) for x in dH]}")
print(f"  -> grounding-on-top-of-subword is "
      f"{'ROBUST (every seed negative)' if all(x < 0 for x in dH) else 'NOT robust (sign flips) = within noise'}")
