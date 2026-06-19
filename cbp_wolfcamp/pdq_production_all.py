"""Re-extract PDQ production for ALL Andrews oil leases (formation-agnostic), so any formation pivot
(San Andres, Grayburg, ...) is covered. Streams OG_LEASE_CYCLE once."""
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdq_production import CTY, DATA, dsv


def all_oil():
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    target = set(); lease_wells = defaultdict(set); api_oil = defaultdict(set)
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix["API_UNIQUE_NO"] or f[ix["API_COUNTY_CODE"]].strip().zfill(3) != CTY:
            continue
        api8 = CTY + f[ix["API_UNIQUE_NO"]].strip().zfill(5)
        key = (f[ix["OIL_GAS_CODE"]].strip(), f[ix["DISTRICT_NO"]].strip().zfill(2), f[ix["LEASE_NO"]].strip().zfill(6))
        lease_wells[key].add(api8)
        if key[0] == "O":
            target.add(key); api_oil[api8].add(key)
    return target, api_oil, lease_wells


def run():
    print("[PDQ re-extract: ALL Andrews oil leases]\n")
    target, api_oil, lease_wells = all_oil()
    print("  all Andrews oil leases:", len(target))
    fh, ix = dsv("OG_LEASE_CYCLE_DATA_TABLE.dsv")
    oi, ymi = ix["LEASE_OIL_PROD_VOL"], ix["CYCLE_YEAR_MONTH"]
    oc, dc, lc = ix["OIL_GAS_CODE"], ix["DISTRICT_NO"], ix["LEASE_NO"]
    monthly = defaultdict(dict); n = 0; t0 = time.time()
    for line in fh:
        n += 1
        if n % 10_000_000 == 0:
            print("    %dM rows %.0fs" % (n // 1_000_000, time.time() - t0)); sys.stdout.flush()
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

    out = open(os.path.join(DATA, "andrews_lease_production_all.csv"), "w", newline="")
    w = csv.writer(out); w.writerow(["ogc", "dist", "lease", "n_wells", "cum_oil", "n_months",
                                     "first_ym", "last_ym", "first12_oil", "best12_oil", "peak_oil"])
    for key, ms in monthly.items():
        yms = sorted(ms); vals = [ms[y] for y in yms]; cum = sum(vals); peak = max(vals) if vals else 0
        first12 = sum(vals[:12]); best12 = max((sum(vals[i:i + 12]) for i in range(max(1, len(vals) - 11))), default=0)
        w.writerow([key[0], key[1], key[2], len(lease_wells[key]), int(cum), len(yms),
                    yms[0] if yms else "", yms[-1] if yms else "", int(first12), int(best12), int(peak)])
    out.close()
    json.dump({"api_oil": {a: ["}".join(k) for k in ks] for a, ks in api_oil.items()},
               "lease_nwells": {"}".join(k): len(s) for k, s in lease_wells.items()}},
              open(os.path.join(DATA, "andrews_linkage_all.json"), "w"))
    print("  wrote andrews_lease_production_all.csv (%d leases) + andrews_linkage_all.json" % len(monthly))


if __name__ == "__main__":
    run()
