"""Stage C: stream OG_LEASE_CYCLE (12.7 GB) filtered to Andrews vertical-Wolfcamp oil leases ->
per-lease monthly oil -> productivity metrics (IP = first/best-12-month, cumulative, peak, vintage).

Outputs andrews_lease_production.csv + linkage maps (json) for the value test.
"""
import csv
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "PDQ_DSV.zip")
CTY = "003"


def dsv(member):
    fh = io.TextIOWrapper(zipfile.ZipFile(ZIP).open(member), encoding="latin-1", newline="")
    header = next(fh).rstrip("\n").split("}")
    return fh, {c.strip(): i for i, c in enumerate(header)}


def linkage():
    wolf = set()
    for r in csv.DictReader(open(os.path.join(DATA, "wolfcamp_tops.csv"), encoding="utf-8")):
        if r["county"] == CTY and "WOLFCAMP" in r["formation_name"].upper() and "DISPOSAL" not in r["formation_name"].upper():
            wolf.add(r["api8"])
    hz = {r["api8"] for r in csv.DictReader(open(os.path.join(DATA, "andrews_horiz.csv")))}
    vert = wolf - hz
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    api_oil = defaultdict(set); lease_wells = defaultdict(set)
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix["API_UNIQUE_NO"] or f[ix["API_COUNTY_CODE"]].strip().zfill(3) != CTY:
            continue
        api8 = CTY + f[ix["API_UNIQUE_NO"]].strip().zfill(5)
        key = (f[ix["OIL_GAS_CODE"]].strip(), f[ix["DISTRICT_NO"]].strip().zfill(2), f[ix["LEASE_NO"]].strip().zfill(6))
        lease_wells[key].add(api8)
        if f[ix["OIL_GAS_CODE"]].strip() == "O":
            api_oil[api8].add(key)
    target = {k for a in vert if a in api_oil for k in api_oil[a]}
    return vert, api_oil, lease_wells, target


def run():
    print("[PDQ Stage C: stream 12.7 GB monthly -> Andrews Wolfcamp-vertical lease production]\n")
    vert, api_oil, lease_wells, target = linkage()
    print("  target oil leases to extract: %d" % len(target))
    fh, ix = dsv("OG_LEASE_CYCLE_DATA_TABLE.dsv")
    oi, gi, ymi = ix["LEASE_OIL_PROD_VOL"], ix["LEASE_GAS_PROD_VOL"], ix["CYCLE_YEAR_MONTH"]
    oc, dc, lc = ix["OIL_GAS_CODE"], ix["DISTRICT_NO"], ix["LEASE_NO"]
    monthly = defaultdict(dict); n = 0; t0 = time.time()
    for line in fh:
        n += 1
        if n % 10_000_000 == 0:
            print("    %dM rows, %.0fs" % (n // 1_000_000, time.time() - t0)); sys.stdout.flush()
        f = line.rstrip("\n").split("}")
        if len(f) <= oi:
            continue
        key = (f[oc].strip(), f[dc].strip().zfill(2), f[lc].strip().zfill(6))
        if key not in target:
            continue
        try:
            oil = float(f[oi] or 0)
        except ValueError:
            oil = 0.0
        monthly[key][f[ymi].strip()] = monthly[key].get(f[ymi].strip(), 0.0) + oil
    print("  scanned %d rows in %.0fs; leases with production: %d" % (n, time.time() - t0, len(monthly)))

    out = open(os.path.join(DATA, "andrews_lease_production.csv"), "w", newline="")
    w = csv.writer(out); w.writerow(["ogc", "dist", "lease", "n_wells", "cum_oil", "n_months",
                                     "first_ym", "last_ym", "first12_oil", "best12_oil", "peak_oil"])
    for key, ms in monthly.items():
        yms = sorted(ms)
        vals = [ms[y] for y in yms]
        cum = sum(vals); peak = max(vals) if vals else 0
        first12 = sum(vals[:12])
        best12 = max((sum(vals[i:i + 12]) for i in range(max(1, len(vals) - 11))), default=0)
        w.writerow([key[0], key[1], key[2], len(lease_wells[key]), int(cum), len(yms),
                    yms[0] if yms else "", yms[-1] if yms else "", int(first12), int(best12), int(peak)])
    out.close()
    json.dump({"api_oil": {a: ["}".join(k) for k in ks] for a, ks in api_oil.items()},
               "lease_nwells": {"}".join(k): len(s) for k, s in lease_wells.items()},
               "verticals": sorted(vert)},
              open(os.path.join(DATA, "andrews_linkage.json"), "w"))
    print("  wrote andrews_lease_production.csv (%d leases) + andrews_linkage.json" % len(monthly))


if __name__ == "__main__":
    run()
