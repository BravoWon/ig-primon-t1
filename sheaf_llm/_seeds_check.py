#!/usr/bin/env python
"""Robustness: re-run brick 3' over seeds (varies the held-out H set AND init). Confirms the post-H
payoff is stable, not single-seed noise. STEPS reduced for speed; both arms always matched per seed."""
import numpy as np
import generative_gate as gg

gg.STEPS = 3000
rows = []
for s in (0, 1, 2):
    gg.SEED = s
    d = gg.build()
    out = {}
    for kind in ("flat", "grounded"):
        p, ov, bk, _ = gg.run(kind, d)
        out[kind] = (ov, bk[1], bk[2])          # overall, post-seen-noun, post-H-noun
    fo, fs, fH = out["flat"]; go, gs, gH = out["grounded"]
    rows.append((s, fo, go, fs, gs, fH, gH))
    print(f"  seed {s}: overall {fo:.1f}->{go:.1f}  post-seen {fs:.1f}->{gs:.1f}  post-H {fH:.1f}->{gH:.1f}")

a = np.array([r[1:] for r in rows])
mfo, mgo, mfs, mgs, mfH, mgH = a.mean(0)
print("\n=== mean over seeds (flat -> grounded), %delta ===")
print(f"  overall      {mfo:6.1f} -> {mgo:6.1f}   {(mgo-mfo)/mfo*100:+5.1f}%")
print(f"  post-seen    {mfs:6.1f} -> {mgs:6.1f}   {(mgs-mfs)/mfs*100:+5.1f}%   (control: ~flat)")
print(f"  post-H       {mfH:6.1f} -> {mgH:6.1f}   {(mgH-mfH)/mfH*100:+5.1f}%   (payoff)")
dH = [(r[6]-r[5])/r[5]*100 for r in rows]
print(f"  post-H %delta per seed: {[round(x,1) for x in dH]}  -> "
      f"{'STABLE payoff across seeds' if all(x < -10 for x in dH) else 'unstable'}")
