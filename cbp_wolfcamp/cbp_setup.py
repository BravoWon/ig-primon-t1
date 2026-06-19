"""Powered expansion Stage 1: derive the CBP/San-Andres county codes (from county NAMES in the data) and
build the API->oil-lease linkage for ALL those counties. Foundation for the multi-county powered test."""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdq_production import DATA, dsv

TARGET = {"ANDREWS", "ECTOR", "CRANE", "GAINES", "YOAKUM", "TERRY", "COCHRAN", "DAWSON", "MARTIN",
          "MIDLAND", "CROCKETT", "PECOS", "WARD", "WINKLER", "UPTON", "GLASSCOCK", "HOWARD",
          "BORDEN", "SCURRY", "MITCHELL"}


def run():
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    code2name = {}; counts = defaultdict(int)
    api_oil = defaultdict(set); lease_wells = defaultdict(set); target = set()
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix["COUNTY_NAME"]:
            continue
        cc = f[ix["API_COUNTY_CODE"]].strip().zfill(3); nm = f[ix["COUNTY_NAME"]].strip().upper()
        code2name[cc] = nm
        if nm in TARGET:
            target.add(cc); counts[cc] += 1
            api8 = cc + f[ix["API_UNIQUE_NO"]].strip().zfill(5)
            key = (f[ix["OIL_GAS_CODE"]].strip(), f[ix["DISTRICT_NO"]].strip().zfill(2), f[ix["LEASE_NO"]].strip().zfill(6))
            lease_wells[key].add(api8)
            if key[0] == "O":
                api_oil[api8].add(key)
    oil_leases = [k for k in lease_wells if k[0] == "O"]
    json.dump({"target_codes": sorted(target), "code2name": {c: code2name[c] for c in target},
               "api_oil": {a: ["}".join(k) for k in ks] for a, ks in api_oil.items()},
               "lease_nwells": {"}".join(k): len(s) for k, s in lease_wells.items()},
               "oil_leases": ["}".join(k) for k in oil_leases]},
              open(os.path.join(DATA, "cbp_setup.json"), "w"))
    print("target CBP/San-Andres counties: %d" % len(target))
    for c in sorted(target, key=lambda c: -counts[c]):
        print("  %s %-11s %6d completions" % (c, code2name[c], counts[c]))
    print("CBP oil leases: %d | wells with oil completion: %d" % (len(oil_leases), len(api_oil)))


if __name__ == "__main__":
    run()
