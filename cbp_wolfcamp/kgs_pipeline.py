#!/usr/bin/env python
"""KGS powered public study pipeline (PUBLIC data; logs+production+structure all free).

Stages (run with a county code, default Barton=9):
  1. list  : page KGS SelectWells -> per-well (kid, api, las_url, operator, depth_start/stop, loc)
  2. join  : link api10 -> reduced lease production (cum_oil, max_wells, zone) [kgs_prod_agg output]
  3. report: joinable powered-n (wells with LAS AND single-well oil-lease production)
  4. logs  : (if 'pull') download each joinable LAS from Azure, compute GR rock-quality features
  5. write : county_master.csv

    python kgs_pipeline.py [county_code] [pull]
"""
from __future__ import annotations
import csv, os, re, subprocess, sys, warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

PROD = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"
LASDIR = r"C:\Users\JT-DEV1\kgs_public\las"
OUTDIR = r"C:\Users\JT-DEV1\kgs_public"
BASE = "https://chasm.kgs.ku.edu/ords/las.lasd5.SelectWells"


def curl(url, data=None):
    cmd = ["curl", "-skL", "-m", "40", url]
    if data:
        cmd = ["curl", "-skL", "-m", "40", "-X", "POST", url, "--data", data]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="latin-1").stdout


def api10(s):
    d = re.sub(r"\D", "", s)
    return d[:10] if len(d) >= 10 else ""


def get_county_wells(cc, state=15):
    wells = {}
    page = 1
    while True:
        url = f"{BASE}?f_st={state}&f_c={cc}&f_t=&f_r=&ew=&f_s=&f_l=&f_op=&f_api=&sort_by=&f_pg={page}"
        html = curl(url)
        if not html:
            break
        chunks = html.split("DisplayWell?f_kid=")
        found = 0
        for ch in chunks[1:]:
            km = re.match(r"(\d+)", ch)
            if not km:
                continue
            kid = km.group(1)
            apim = re.search(r"15-\d{3}-\d{5}", ch)
            lasm = re.search(r"https://kgsimages\.blob\.core\.windows\.net/\S+?\.las", ch)
            depths = re.findall(r"<td[^>]*>\s*(\d{3,6})\s*</td>", ch)
            loc = re.search(r"T\d+[NS] R\d+[EW], Sec\. ?\d+", ch)
            rec = wells.setdefault(kid, dict(kid=kid, api="", las="", operator="",
                                             depth_start="", depth_stop="", loc=""))
            if apim and not rec["api"]:
                rec["api"] = apim.group(0)
            if lasm and not rec["las"]:
                rec["las"] = lasm.group(0)
            if loc and not rec["loc"]:
                rec["loc"] = loc.group(0)
            if depths and not rec["depth_start"]:
                rec["depth_start"] = depths[0]
                if len(depths) > 1:
                    rec["depth_stop"] = depths[1]
            found += 1
        # stop when no "Next" link / no new page
        if f"f_pg={page+1}" not in html or page > 60:
            break
        page += 1
    return list(wells.values()), page


def load_production():
    idx = {}                       # api10 -> production row
    for r in csv.DictReader(open(PROD, encoding="utf-8")):
        for a in r["apis"].split(","):
            a10 = api10(a)
            if a10:
                # keep richest (highest cum) if api maps to multiple leases
                if a10 not in idx or int(r["cum_oil"]) > int(idx[a10]["cum_oil"]):
                    idx[a10] = r
    return idx


def gr_features(path):
    import numpy as np, lasio
    try:
        las = lasio.read(path, ignore_header_errors=True)
    except Exception:
        return None
    gr = None
    for nm in ("GR", "GRGC", "GRD", "SGR", "CGR", "GRR"):
        try:
            a = np.asarray(las[nm], dtype=float); a[a <= -999] = np.nan
            if np.isfinite(a).sum() > 50:
                gr = a; break
        except Exception:
            pass
    elev = ""
    for k in ("EKB", "KB", "ELEV", "EREF", "EGL"):
        try:
            v = float(las.well[k].value); elev = v; break
        except Exception:
            pass
    if gr is None:
        return dict(has_gr=0, elev=elev)
    g = gr[np.isfinite(gr)]
    return dict(has_gr=1, elev=elev, gr_p50=round(float(np.median(g)), 1),
                gr_clean_frac=round(float(np.mean(g < 50)), 3),    # clean carbonate fraction
                net_clean_ft=round(float(np.mean(g < 50) * (las.index[-1]-las.index[0])), 0),
                n_gr=int(g.size))


def main():
    cc = sys.argv[1] if len(sys.argv) > 1 else "9"
    do_pull = len(sys.argv) > 2 and sys.argv[2] == "pull"
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[1] listing county {cc} wells from KGS ...", flush=True)
    wells, pages = get_county_wells(cc)
    with_las = [w for w in wells if w["las"]]
    with_api = [w for w in wells if w["api"]]
    print(f"    wells={len(wells)} ({pages} pages)  with_LAS={len(with_las)}  with_API={len(with_api)}")

    print("[2] joining to lease production ...", flush=True)
    prod = load_production()
    joined = 0; single = 0
    for w in wells:
        p = prod.get(api10(w["api"]))
        if p:
            joined += 1
            w["cum_oil"] = p["cum_oil"]; w["best12_oil"] = p["best12_oil"]
            w["max_wells"] = p["max_wells"]; w["zone"] = p["zone"]; w["lease"] = p["lease"]
            w["county"] = p["county"]
            if p["max_wells"] == "1":
                single += 1
        else:
            w["cum_oil"] = w["best12_oil"] = w["max_wells"] = w["zone"] = w["lease"] = w["county"] = ""
    joinable = [w for w in wells if w["las"] and w["cum_oil"] != ""]
    joinable_single = [w for w in joinable if w["max_wells"] == "1"]
    print(f"    wells joined to production: {joined}")
    print(f"    POWERED joinable (LAS + production): {len(joinable)}  "
          f"of which single-well leases: {len(joinable_single)}")

    if do_pull:
        import time
        ld = os.path.join(LASDIR, cc); os.makedirs(ld, exist_ok=True)
        print(f"[4] downloading {len(joinable)} LAS + computing GR rock quality ...", flush=True)
        for i, w in enumerate(joinable):
            fp = os.path.join(ld, w["kid"] + ".las")
            if not os.path.exists(fp):
                subprocess.run(["curl", "-skL", "-m", "30", "-o", fp, w["las"]],
                               capture_output=True)
            feat = gr_features(fp) or {}
            w.update(feat)
            if (i + 1) % 50 == 0:
                print(f"      ...{i+1}/{len(joinable)}", flush=True)

    cols = ["kid","api","operator","loc","county","lease","zone","max_wells","depth_start",
            "depth_stop","cum_oil","best12_oil","has_gr","elev","gr_p50","gr_clean_frac",
            "net_clean_ft","n_gr","las"]
    out = os.path.join(OUTDIR, f"county_{cc}_master.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in wells:
            w.writerow(row)
    print(f"[5] wrote {out}")


if __name__ == "__main__":
    main()
