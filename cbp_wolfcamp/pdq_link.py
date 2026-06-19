"""Stage A+B: API->lease linkage + clean-sample sizing BEFORE the 12.7 GB monthly stream.

Builds api8 -> oil lease(s) from OG_WELL_COMPLETION (Andrews), counts wells/lease (single-well = clean),
filters to vertical Wolfcamp wells (tops minus GIS layer-9 horizontals), and pulls first-production month
from OG_SUMMARY_ONSHORE_LEASE (1993 truncation / vintage). Reports the clean sample size for the value test.
"""
import csv
import io
import os
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "PDQ_DSV.zip")
CTY = "003"


def dsv(member):
    z = zipfile.ZipFile(ZIP); fh = io.TextIOWrapper(z.open(member), encoding="latin-1", newline="")
    header = next(fh).rstrip("\n").split("}")
    idx = {c.strip(): i for i, c in enumerate(header)}
    return fh, idx


def andrews_wolfcamp_verticals():
    wolf = set()
    for r in csv.DictReader(open(os.path.join(DATA, "wolfcamp_tops.csv"), encoding="utf-8")):
        if r["county"] == CTY and "WOLFCAMP" in r["formation_name"].upper() and "DISPOSAL" not in r["formation_name"].upper():
            wolf.add(r["api8"])
    hzcache = os.path.join(DATA, "andrews_horiz.csv")
    if os.path.exists(hzcache):
        hz = {r["api8"] for r in csv.DictReader(open(hzcache))}
    else:
        rows = G.fetch_where("API LIKE '%s%%'" % CTY, "API", layer=9)
        hz = {(a.get("API") or "").strip()[:8] for a in rows if len((a.get("API") or "").strip()) >= 8}
        with open(hzcache, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["api8"])
            for a in sorted(hz):
                w.writerow([a])
    return wolf, hz


def run():
    print("[PDQ Stage A+B: linkage + clean-sample sizing]\n")
    wolf, hz = andrews_wolfcamp_verticals()
    vert = wolf - hz
    print("  Andrews Wolfcamp wells: %d | horizontals(layer9): %d removed | verticals: %d" % (len(wolf), len(wolf & hz), len(vert)))

    # crosswalk
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    api_oil_lease = {}; lease_wells = defaultdict(set); n = 0
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix["API_UNIQUE_NO"]:
            continue
        if f[ix["API_COUNTY_CODE"]].strip().zfill(3) != CTY:
            continue
        api8 = CTY + f[ix["API_UNIQUE_NO"]].strip().zfill(5)
        ogc = f[ix["OIL_GAS_CODE"]].strip()
        key = (ogc, f[ix["DISTRICT_NO"]].strip().zfill(2), f[ix["LEASE_NO"]].strip().zfill(6))
        lease_wells[key].add(api8)
        if ogc == "O":
            api_oil_lease.setdefault(api8, set()).add(key)
        n += 1
    print("  Andrews well-completion rows: %d | distinct Andrews leases: %d" % (n, len(lease_wells)))

    vert_with_oil = [a for a in vert if a in api_oil_lease]
    singlewell = {k for k, s in lease_wells.items() if len(s) == 1}
    vert_single = [a for a in vert_with_oil if any(k in singlewell for k in api_oil_lease[a])]
    print("\n  vertical Wolfcamp wells linked to an OIL lease: %d" % len(vert_with_oil))
    print("  ...on a SINGLE-WELL oil lease (clean per-well outcome): %d" % len(vert_single))

    # first-production month per lease (vintage / 1993 truncation)
    fh2, ix2 = dsv("OG_SUMMARY_ONSHORE_LEASE_DATA_TABLE.dsv")
    lease_first = {}
    our_leases = set(api_oil_lease[a].__iter__().__next__() for a in vert_single) if vert_single else set()
    our_leases = {k for a in vert_single for k in api_oil_lease[a] if k in singlewell}
    for line in fh2:
        f = line.rstrip("\n").split("}")
        if len(f) <= ix2["CYCLE_YEAR_MONTH_MIN"]:
            continue
        key = (f[ix2["OIL_GAS_CODE"]].strip(), f[ix2["DISTRICT_NO"]].strip().zfill(2), f[ix2["LEASE_NO"]].strip().zfill(6))
        if key in our_leases:
            lease_first[key] = f[ix2["CYCLE_YEAR_MONTH_MIN"]].strip()
    post93 = [a for a in vert_single for k in api_oil_lease[a]
              if k in singlewell and lease_first.get(k, "000000") >= "199301"]
    print("  ...with first production in PDQ era (>=1993, clean cumulative/IP): %d" % len(set(post93)))

    print("\n  [SAMPLE VERDICT] clean single-well-lease verticals: %d  (post-93: %d)" % (len(vert_single), len(set(post93))))
    print("  -> if healthy (>~150), stream OG_LEASE_CYCLE for IP on these leases; else widen to vintage-normalized cumulative on all %d oil-linked verticals." % len(vert_with_oil))
    # cache the lease set to extract from the 12.7GB monthly
    import json
    json.dump({"oil_leases_to_extract": ["}".join(k) for a in vert_with_oil for k in api_oil_lease[a]],
               "vert_with_oil": len(vert_with_oil), "vert_single": len(vert_single)},
              open(os.path.join(DATA, "pdq_link_summary.json"), "w"))
    return vert, api_oil_lease, lease_wells


if __name__ == "__main__":
    run()
