"""Scope the null: is it the resource-play population? Split single-well-lease verticals into
OLD (producing at PDQ start 199301 = pre-1993, conventional-leaning) vs NEW (post-1993 completion,
Wolfberry/resource era), and test structure->production separately. The 1993 truncation means OLD wells'
cum_oil is only their post-93 tail -- a real limitation, reported honestly."""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_test import CTY, DATA, design2d, load, residual_field, spearman


def run():
    coords, subsea, link, prod = load()
    verts = set(link["verticals"])
    coef = residual_field(verts, coords, subsea)
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    rows = []
    for a in verts:
        if a not in coords or a not in subsea or a not in api_oil:
            continue
        for k in api_oil[a]:
            if nwells.get("}".join(k), 9) != 1 or k not in prod:
                continue
            pr = prod[k]; cum = float(pr["cum_oil"]); b12 = float(pr["best12_oil"])
            if cum <= 0:
                continue
            la, lo = coords[a]; resid = subsea[a] - (np.r_[1, design2d(np.array([lo]), np.array([la]))[0]] @ coef)
            rows.append((float(resid), cum, b12, pr["first_ym"]))
            break

    def report(name, sub):
        if len(sub) < 30:
            print("  %-32s n=%d (too few)" % (name, len(sub))); return
        res = np.array([r[0] for r in sub]); cum = np.array([r[1] for r in sub]); b12 = np.array([r[2] for r in sub])
        hi = cum[res > 150]; lo = cum[res < -150]
        rr = (np.median(hi) / max(np.median(lo), 1)) if (len(hi) > 5 and len(lo) > 5) else float("nan")
        print("  %-32s n=%-4d  Spearman(resid,logCum)=%+.3f  Spearman(resid,logBest12)=%+.3f  cumHi/cumLo=%.2fx"
              % (name, len(sub), spearman(res, np.log1p(cum)), spearman(res, np.log1p(b12)), rr))

    old = [r for r in rows if r[3] == "199301"]; new = [r for r in rows if r[3] > "199301"]
    print("[Vintage split — single-well-lease vertical Wolfcamp, Andrews]\n")
    print("  total single-well-lease verticals with production: %d" % len(rows))
    print("  OLD (producing at 1993 PDQ start, pre-93/conventional-leaning): %d" % len(old))
    print("  NEW (first prod >1993, Wolfberry/resource era):                 %d\n" % len(new))
    report("ALL single-well verticals", rows)
    report("OLD (pre-93, conventional-leaning)", old)
    report("NEW (post-93, resource era)", new)
    print("\n  NOTE: OLD wells' cum_oil is only their POST-1993 TAIL (PDQ starts 1993) -> their true productivity")
    print("  is undermeasured; a clean conventional-trap test needs pre-1993 production not in public PDQ.")


if __name__ == "__main__":
    run()
