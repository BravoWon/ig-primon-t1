"""PERF-INDEXED + clean-decline structure->value test. Restricts to wells actually PERFORATED in San
Andres (not merely having a SA top), with cleaned type-curve IP (toss first 2 months ramp; cap rework
spikes via robust MAD clip). Runs CONTAMINATED (all SA-top) vs CLEAN (perf-in-SA) side by side.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_test import DATA, auc, design2d, logit_fit, logit_pred, spearman
from value_test_tensor import detrended_residuals


def clean_ip(ser):
    oil = np.array([v for _, v in sorted(ser.items())], float)
    oil = oil[2:] if len(oil) > 14 else oil                      # toss first 2 months (ramp/cleanup)
    pos = oil[oil > 0]
    if len(pos) < 8:
        return None
    med = np.median(pos); mad = np.median(np.abs(pos - med)) + 1e-9
    capped = np.minimum(oil, med + 5 * 1.4826 * mad)             # cap rework spikes (robust)
    best12 = max((capped[i:i + 12].sum() for i in range(max(1, len(capped) - 11))), default=0.0)
    return float(best12)


def build(sample_set, SA, coords, api_oil, nwells, series):
    rows = []
    for a in sample_set:
        if a not in coords or a not in api_oil or a not in SA:
            continue
        for k in api_oil[a]:
            ks = "}".join(k)
            if nwells.get(ks, 9) != 1 or ks not in series:
                continue
            ip = clean_ip(series[ks])
            if ip is None or ip <= 0:
                continue
            la, lo = coords[a]
            rows.append((la, lo, SA[a], ip)); break
    return rows


def evaluate(name, rows):
    if len(rows) < 60:
        print("  %-28s n=%d (too few)" % (name, len(rows))); return
    la = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
    res = np.array([r[2] for r in rows]); ip = np.array([r[3] for r in rows])
    y = (ip >= np.quantile(ip, 0.75)).astype(float)
    base = design2d(lo, la); full = np.c_[base, res]
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
    ob, of = cv(base), cv(full); m = ~np.isnan(ob) & ~np.isnan(of)
    ab, af = auc(y[m], ob[m]), auc(y[m], of[m])
    hi = ip[res > 150]; lo_ = ip[res < -150]
    ratio = np.median(hi) / max(np.median(lo_), 1) if (len(hi) > 5 and len(lo_) > 5) else float("nan")
    print("  %-28s n=%-4d  Spearman %+.3f  IP hi/lo %.2fx  CV-AUC base %.3f -> +struct %.3f (lift %+.3f)"
          % (name, len(rows), spearman(res, np.log1p(ip)), ratio, ab, af, af - ab))


def run():
    print("[PERF-INDEXED structure->value test — San Andres]\n")
    coords = {r["api8"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(open(os.path.join(DATA, "andrews_coords.csv")))}
    SA = detrended_residuals(coords).get("SAN ANDRES", {})
    pt = json.load(open(os.path.join(DATA, "andrews_perfs_tops.json")))
    perf_sa = {a for a, v in pt.items() if "SAN ANDRES" in v["perf_forms"]}
    series = json.load(open(os.path.join(DATA, "andrews_lease_series.json")))
    link = json.load(open(os.path.join(DATA, "andrews_linkage_all.json")))
    api_oil = {a: [tuple(k.split("}")) for k in ks] for a, ks in link["api_oil"].items()}
    nwells = link["lease_nwells"]
    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}

    all_sa = set(SA) - hz
    clean = (set(SA) & perf_sa) - hz
    print("  comparison (clean type-curve IP, single-well oil leases):\n")
    evaluate("ALL SA-top (contaminated)", build(all_sa, SA, coords, api_oil, nwells, series))
    evaluate("PERF-INDEXED (perfed in SA)", build(clean, SA, coords, api_oil, nwells, series))
    print("\n  -> if perf-indexing raises the lift, the earlier null was partly wrong-zone contamination.")


if __name__ == "__main__":
    run()
