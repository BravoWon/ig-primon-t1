"""Extract the per-lease monthly oil SERIES (not just scalars) for Andrews oil leases -> decline-curve
dynamics. Re-streams OG_LEASE_CYCLE once. Saves andrews_lease_series.json = {ogc}dist}lease: {YYYYMM: oil}}.
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdq_production import DATA, dsv
from pdq_production_all import all_oil


def run():
    print("[PDQ series extract: monthly oil per Andrews oil lease]\n")
    target, api_oil, lease_wells = all_oil()
    print("  target oil leases:", len(target))
    fh, ix = dsv("OG_LEASE_CYCLE_DATA_TABLE.dsv")
    oi, ymi = ix["LEASE_OIL_PROD_VOL"], ix["CYCLE_YEAR_MONTH"]
    oc, dc, lc = ix["OIL_GAS_CODE"], ix["DISTRICT_NO"], ix["LEASE_NO"]
    series = defaultdict(dict); n = 0; t0 = time.time()
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
        k = "}".join(key); ym = f[ymi].strip()
        series[k][ym] = series[k].get(ym, 0.0) + oil
    json.dump(series, open(os.path.join(DATA, "andrews_lease_series.json"), "w"))
    print("  scanned %d rows in %.0fs; wrote series for %d leases" % (n, time.time() - t0, len(series)))


if __name__ == "__main__":
    run()
