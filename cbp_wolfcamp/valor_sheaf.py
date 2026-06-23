#!/usr/bin/env python
"""Point the cellular-sheaf value model at the Valor acreage. Two plays, honest per-play gate:
  MORROW   Finney+Gray (sandstone, channel-controlled -> public structure is the WRONG dimension;
           expect the sheaf to be weak: the value driver, channel sand, is sub-resolution).
  ARBUCKLE Butler (carbonate, structure-controlled -> should transfer better).

Same machinery as `sheaf_value_model.py` (geology-only manifold, restriction-map harmonic extension
anchored to a kNN prior, buffered spatial holdout, provenance-clean pre-drill labels). Then high-grade
the ACQUIRED FOOTPRINTS by held-out sheaf score.

    python valor_sheaf.py
"""
from __future__ import annotations
import csv, re
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import sheaf_value_model as sv

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
TOPS = r"C:\Users\JT-DEV1\kgs_public\tops\ks_tops.txt"
MARK = ["LANSING", "KANSAS CITY", "MARMATON", "CHEROKEE", "MISSISSIPPIAN", "MORROW", "ARBUCKLE"]
PAIRS = [("LANSING", "KANSAS CITY"), ("KANSAS CITY", "MARMATON"), ("KANSAS CITY", "MISSISSIPPIAN"),
         ("MISSISSIPPIAN", "MORROW"), ("MISSISSIPPIAN", "ARBUCKLE"), ("LANSING", "MISSISSIPPIAN")]
PLAYS = {
    "MORROW (Finney/Gray)":  dict(cty={"FINNEY", "GRAY"}, coord="subsea_MISS",
                                  foot=dict(lat=(37.86, 38.12), lon=(-100.80, -100.42))),
    "ARBUCKLE (Butler)":     dict(cty={"BUTLER"}, coord="subsea_MISS",
                                  foot=dict(lat=(37.52, 37.70), lon=(-96.98, -96.66))),
}


def a10(s):
    d = re.sub(r"\D", "", s or "")
    return d[:10] if len(d) >= 10 else ""


def load_play(counties):
    W = {}
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        if r["county"].upper() not in counties:
            continue
        a = next((a10(t) for t in r["apis"].split(",") if a10(t)), "")
        if not a:
            continue
        try:
            b12 = float(r["best12_oil"] or 0)
            if b12 <= 0:
                continue
            la, lo = float(r["lat"]), float(r["lon"])
        except (ValueError, KeyError):
            continue
        ym = re.sub(r"\D", "", r.get("first_ym") or "")
        W[a] = dict(b12=b12, la=la, lo=lo, cty=r["county"].upper(),
                    vint=float(ym[:4]) if len(ym) >= 4 else np.nan)
    T = set(W)
    for r in csv.DictReader(open(TOPS, encoding="latin-1")):
        a = a10(r.get("API_NUM_NODASH") or r.get("API_NUMBER"))
        if a not in T:
            continue
        form = (r.get("FORMATION", "") + " " + r.get("OLD_FORMATION", "")).upper()
        try:
            top = float(r["TOP"])
        except (ValueError, KeyError):
            continue
        w = W[a]
        try:
            w.setdefault("elev", float(r["ELEVATION"]))
        except (ValueError, KeyError):
            pass
        for m in MARK:
            if m in form:
                w.setdefault(m, top)
    return [w for w in W.values() if sum(m in w for m in MARK) >= 2]  # >=2 markers (elev imputed if absent)


def featurize(W):
    names = ["elev", "depth_KC", "lat", "lon", "vint", "n_tops"]
    names += ["subsea_" + m[:4] for m in MARK]
    names += ["thk_%s_%s" % (a[:4], b[:4]) for a, b in PAIRS]
    X = []
    for w in W:
        el = w.get("elev", np.nan)                       # may be absent (>=2-marker filter); imputed later
        row = [el, w.get("KANSAS CITY", np.nan), w["la"], w["lo"], w["vint"],
               sum(1 for m in MARK if m in w)]
        row += [el - w[m] if m in w else np.nan for m in MARK]
        row += [w[b] - w[a] if (a in w and b in w) else np.nan for a, b in PAIRS]
        X.append(row)
    return np.array(X, float), names


def run_play(label, cfg):
    W = load_play(cfg["cty"])
    X, names = featurize(W)
    y = np.log(np.array([w["b12"] for w in W]))
    n = len(W)
    if n < 60:
        print(f"\n=== {label}:  n={n} -- too few wells with tops for a model. ==="); return
    lat = X[:, names.index("lat")]; lon = X[:, names.index("lon")]
    Xs = sv.standardize_impute(X)
    gidx = [j for j, nm in enumerate(names) if nm not in ("lat", "lon")]
    Xg = Xs[:, gidx]
    Gc = Xg - Xg.mean(0)
    _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
    g = Gc @ Vt[:sv.K].T; g = g / (g.std(0) + 1e-9)
    phi = np.c_[np.ones(n), g]
    tree = cKDTree(Xg); _, idx = tree.query(Xg, k=min(sv.KNN + 1, n))
    edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(n) for j in idx[i, 1:] if i != int(j)})

    GRID = 2 if n < 300 else (3 if n < 800 else 4)
    BUF = 0.08 if n < 300 else sv.BUF                    # smaller buffer for compact small plays
    la0, la1, lo0, lo1 = lat.min(), lat.max(), lon.min(), lon.max()
    bi = np.minimum(((lat - la0) / (la1 - la0 + 1e-9) * GRID).astype(int), GRID - 1)
    bj = np.minimum(((lon - lo0) / (lo1 - lo0 + 1e-9) * GRID).astype(int), GRID - 1)
    block = bi * GRID + bj
    geo = np.c_[lat, lon * np.cos(np.radians(lat.mean()))]
    coordj = names.index(cfg["coord"])
    p_sheaf = np.full(n, np.nan); p_knn = np.full(n, np.nan); p_coord = Xs[:, coordj]
    for b in sorted(set(block.tolist())):
        te = block == b
        if te.sum() < 8:
            continue
        dmin, _ = cKDTree(geo[te]).query(geo, k=1)
        tr = (~te) & (dmin > BUF)
        if tr.sum() < 40:
            continue
        known = np.where(tr)[0]
        kt = cKDTree(Xg[known]); _, jall = kt.query(Xg, k=min(sv.KNN, len(known)))
        prior = np.median(y[known][jall], axis=1)
        p_knn[np.where(te)[0]] = prior[te]
        p_sheaf[np.where(te)[0]] = sv.solve_sheaf(phi, edges, known, y, n, prior, sv.RHO)[te]

    m = np.isfinite(p_sheaf) & np.isfinite(p_knn)
    print(f"\n=== {label}:  n={n} oil leases with tops; scored {m.sum()} held-out (GRID {GRID}x{GRID}) ===")
    if m.sum() < 30:
        print("   too few held-out wells for a stable gate."); return
    ics = {k: spearmanr(v[m], y[m])[0] for k, v in
           [("single coord (" + cfg["coord"] + ")", p_coord), ("manifold-kNN", p_knn),
            ("CELLULAR SHEAF", p_sheaf)]}
    for k, v in ics.items():
        print(f"   {k:28} OOS rank-IC = {v:+.3f}")
    sh, kn = ics["CELLULAR SHEAF"], ics["manifold-kNN"]
    print(f"   -> sheaf vs kNN {sh-kn:+.3f}; sheaf vs coord {sh-ics['single coord ('+cfg['coord']+')']:+.3f}"
          f"   [{'sheaf high-grades this play' if sh>0.12 else 'WEAK -- public geology not predictive here'}]")

    # high-grade the acquired footprint by held-out sheaf score
    fb = cfg["foot"]
    infoot = (lat >= fb["lat"][0]) & (lat <= fb["lat"][1]) & (lon >= fb["lon"][0]) & (lon <= fb["lon"][1])
    fp = np.where(infoot & m)[0]
    if len(fp) >= 5:
        order = fp[np.argsort(-p_sheaf[fp])]
        print(f"   FOOTPRINT high-grade ({len(fp)} scored wells): top sheaf-ranked locations")
        print("     lat       lon      sheaf   actual best12")
        for i in order[:6]:
            print(f"     {lat[i]:7.3f}  {lon[i]:8.3f}  {p_sheaf[i]:+.2f}   {int(np.exp(y[i])):>6}")
        ic_fp = spearmanr(p_sheaf[fp], y[fp])[0] if len(fp) >= 8 else np.nan
        print(f"   footprint sheaf rank-IC = {ic_fp:+.3f} (n={len(fp)})")
    else:
        print(f"   footprint: only {int((infoot&m).sum())} scored wells -- too few to high-grade.")


def main():
    print("[Valor sheaf]  cellular-sheaf high-grading on the acquired plays (public KGS data)")
    for label, cfg in PLAYS.items():
        run_play(label, cfg)
    print("\nNOTE: Morrow = sandstone (channel sand sub-resolution -> public-geology ceiling is LOW by")
    print("design); Arbuckle = carbonate (structure-controlled). The seller's seismic resolves what")
    print("public tops cannot. This is the cheap public high-grade screen, not the valuation.")


if __name__ == "__main__":
    main()
