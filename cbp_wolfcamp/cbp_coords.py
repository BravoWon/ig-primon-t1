"""Powered Stage 3: GIS coords + horizontal flags for all CBP counties (attribute query per county)."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G
from pdq_production import DATA

CODES = json.load(open(os.path.join(DATA, "cbp_setup.json")))["target_codes"]


def run():
    coords = {}; horiz = set()
    for cc in CODES:
        rows = G.fetch_where("API LIKE '%s%%'" % cc, "API,GIS_LAT83,GIS_LONG83", layer=1)
        for a in rows:
            api = (a.get("API") or "").strip()
            if len(api) >= 8:
                try:
                    coords[api[:8]] = (float(a["GIS_LAT83"]), float(a["GIS_LONG83"]))
                except (TypeError, ValueError):
                    pass
        hz = G.fetch_where("API LIKE '%s%%'" % cc, "API", layer=9)
        for a in hz:
            api = (a.get("API") or "").strip()
            if len(api) >= 8:
                horiz.add(api[:8])
        print("  county %s: cum coords %d, horiz %d" % (cc, len(coords), len(horiz))); sys.stdout.flush()
    with open(os.path.join(DATA, "cbp_coords.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["api8", "lat", "lon"])
        for a, (la, lo) in coords.items():
            w.writerow([a, la, lo])
    json.dump(sorted(horiz), open(os.path.join(DATA, "cbp_horiz.json"), "w"))
    print("TOTAL: coords %d, horizontals %d" % (len(coords), len(horiz)))


if __name__ == "__main__":
    run()
