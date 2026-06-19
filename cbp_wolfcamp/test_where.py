"""Test attribute-query pull (API LIKE '003%') vs bbox -- should fix the join rate."""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
CTY = "003"
L1 = G.RRC + "/1/query"


def _get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=G.UA), timeout=90).read().decode("utf-8", "replace"))


def fetch_where(where, fields):
    rows, off = [], 0
    while True:
        p = {"where": where, "outFields": fields, "returnGeometry": "false", "f": "json",
             "resultOffset": off, "resultRecordCount": G.PAGE}
        d = _get(L1 + "?" + urllib.parse.urlencode(p))
        f = d.get("features", [])
        rows += [x["attributes"] for x in f]
        if not f or not d.get("exceededTransferLimit"):
            break
        off += len(f)
    return rows


cnt = _get(L1 + "?" + urllib.parse.urlencode({"where": "API LIKE '003%'", "returnCountOnly": "true", "f": "json"}))
print("wells where API LIKE '003%':", cnt.get("count"))
t0 = time.time()
rows = fetch_where("API LIKE '003%'", "API,GIS_API5,GIS_LAT83,GIS_LONG83")
print("pulled %d Andrews wells in %.0fs" % (len(rows), time.time() - t0))

coord = {}
for a in rows:
    api = (a.get("API") or "").strip()
    if len(api) >= 8:
        try:
            coord[api[:8]] = (float(a["GIS_LAT83"]), float(a["GIS_LONG83"]))
        except (TypeError, ValueError):
            pass
print("usable Andrews coords:", len(coord))

tops = set()
for r in csv.DictReader(open(os.path.join(DATA, "wolfcamp_tops.csv"), encoding="utf-8")):
    if r["county"] == CTY and "WOLFCAMP" in r["formation_name"].upper() and "DISPOSAL" not in r["formation_name"].upper():
        tops.add(r["api8"])
ov = len(tops & set(coord))
print("Andrews wolfcamp tops: %d | joined: %d (%.0f%%)" % (len(tops), ov, 100 * ov / len(tops)))

# cache for reuse by the structure script
with open(os.path.join(DATA, "andrews_coords.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["api8", "lat", "lon"])
    for api, (la, lo) in coord.items():
        w.writerow([api, la, lo])
print("cached -> andrews_coords.csv")
