"""Formation-parameterized structure->value test. Re-points the Wolfcamp machinery at any formation
(default SAN ANDRES -- conventional carbonate where structural/strat traps SHOULD control pooling).

Conventional pools: cumulative oil reflects structural drainage better than early IP, so PRIMARY=cum_oil.
Same honest test: spatial-block CV, structure-blind baseline, calibrated probability. 1993-truncation
caveat still applies to old conventional wells -- reported, both metrics shown.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_test import DATA, auc, design2d, logit_fit, logit_pred, residual_field, spearman

FORMATION = (sys.argv[1] if len(sys.argv) > 1 else "SAN ANDRES").upper()


def run():
    print("[Value test — %s (conventional structural-trap test)]\n" % FORMATION)
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    sub = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "andrews_tops.csv"), encoding="utf-8")):
        nm = r["formation_name"].upper()
        if FORMATION not in nm or "DISPOSAL" in nm:
            continue
        try:
            d = float(r["depth_md"]); s = float(r["subsea"])
        except ValueError:
            continue
        if 1000 <= d <= 12000 and -8000 <= s <= 1500:
            sub[r["api8"]].append(s)
    subsea = {a: float(np.mean(v)) for a, v in sub.items()}
    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}
    verts = set(subsea) - hz
    link = json.load(open(os.path.join(DATA, "andrews_linkage_all.json")))
    prod = {(r["ogc"], r["dist"], r["lease"]): r for r in csv.DictReader(open(os.path.join(DATA, "andrews_lease_production_all.csv")))}
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    print("  %s tops: %d wells | verticals (minus horiz): %d" % (FORMATION, len(subsea), len(verts)))

    coef = residual_field(verts, coords, subsea)
    rows = []
    for a in verts:
        if a not in coords or a not in api_oil:
            continue
        for k in api_oil[a]:
            if nwells.get("}".join(k), 9) != 1 or k not in prod:
                continue
            pr = prod[k]; cum = float(pr["cum_oil"]); b12 = float(pr["best12_oil"])
            if cum <= 0:
                continue
            la, lo = coords[a]; resid = subsea[a] - (np.r_[1, design2d(np.array([lo]), np.array([la]))[0]] @ coef)
            rows.append((a, la, lo, subsea[a], float(resid), cum, b12, pr["first_ym"]))
            break
    print("  clean single-well-lease verticals with production: %d" % len(rows))
    if len(rows) < 80:
        print("  sample too small; stopping"); return

    la = np.array([r[1] for r in rows]); lo = np.array([r[2] for r in rows]); ss = np.array([r[3] for r in rows])
    res = np.array([r[4] for r in rows]); cum = np.array([r[5] for r in rows]); b12 = np.array([r[6] for r in rows])
    old = sum(1 for r in rows if r[7] == "199301")
    print("  (old/pre-93-leaning: %d | newer: %d)\n" % (old, len(rows) - old))

    print("  [HEADLINE associations]")
    print("    Spearman(resid, log CUM oil)  = %+.3f" % spearman(res, np.log1p(cum)))
    print("    Spearman(resid, log best12)   = %+.3f" % spearman(res, np.log1p(b12)))
    hi = cum[res > 150]; loo = cum[res < -150]
    if len(hi) > 5 and len(loo) > 5:
        print("    median CUM on highs(>+150ft): %.0f (n=%d) vs lows(<-150ft): %.0f (n=%d) -> %.2fx"
              % (np.median(hi), len(hi), np.median(loo), len(loo), np.median(hi) / max(np.median(loo), 1)))

    metric = cum; y = (metric >= np.quantile(metric, 0.75)).astype(float)
    Xs = np.c_[design2d(lo, la), ss]; Xf = np.c_[Xs, res]
    blocks = [(int(la[i] / 0.1), int(lo[i] / 0.1)) for i in range(len(rows))]; ub = list(set(blocks))

    def cv(X):
        oof = np.full(len(y), np.nan)
        for b in ub:
            te = np.array([blocks[i] == b for i in range(len(blocks))]); tr = ~te
            if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
                continue
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
            w = logit_fit((X[tr] - mu) / sd, y[tr]); oof[te] = logit_pred(w, (X[te] - mu) / sd)
        return oof

    ob, of = cv(Xs), cv(Xf); m = ~np.isnan(ob) & ~np.isnan(of)
    ab, af = auc(y[m], ob[m]), auc(y[m], of[m])
    print("\n  [SPATIAL-BLOCK CV — %d blocks, %d scored]" % (len(ub), m.sum()))
    print("    out-of-fold AUC: baseline %.3f -> +structure %.3f  (lift %+.3f)" % (ab, af, af - ab))
    sidx = np.argsort(-of[m]); cap = y[m][sidx][:int(0.2 * m.sum())].sum() / max(y[m].sum(), 1)
    print("    top-quintile by P(struct) captures %.0f%% of top-quartile producers (base 25%%)" % (100 * cap))
    sp = spearman(res, np.log1p(cum))
    verdict = "STRUCTURE PAYS" if (af - ab) > 0.02 and sp > 0.05 else "NULL / weak"
    print("\n  [VERDICT] %s  (%s)" % (verdict, FORMATION))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(res, cum, s=10, alpha=.5); ax[0].set_yscale("log")
        ax[0].set_xlabel("structural residual (ft)"); ax[0].set_ylabel("cumulative oil (bbl)")
        ax[0].set_title("%s: value vs structure (Spearman %+.2f)" % (FORMATION, sp))
        order = np.argsort(-of[m]); cumy = np.cumsum(y[m][order]) / y[m].sum(); frac = np.arange(1, m.sum() + 1) / m.sum()
        ax[1].plot(frac, cumy); ax[1].plot([0, 1], [0, 1], "k--")
        ax[1].set_xlabel("frac wells (ranked by P)"); ax[1].set_ylabel("frac top producers captured")
        ax[1].set_title("Lift curve (AUC %.2f)" % af)
        png = os.path.join(DATA, "%s_value_test.png" % FORMATION.replace(" ", "_")); plt.tight_layout(); plt.savefig(png, dpi=110)
        print("  wrote %s" % png)
    except Exception as e:                                       # noqa: BLE001
        print("  (plot skipped: %r)" % e)

    json.dump({"formation": FORMATION, "n": len(rows), "spearman_cum": sp,
               "auc_base": ab, "auc_struct": af, "auc_lift": af - ab, "verdict": verdict},
              open(os.path.join(DATA, "%s_value_summary.json" % FORMATION.replace(" ", "_")), "w"), indent=2)


if __name__ == "__main__":
    run()
