"""Diagnose the GIS<->tops API join (only 32% joined). Find the correct join key; cache the GIS pull."""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrc_gis as G

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
CTY = "003"; BOX = G.CBP_SUBAREAS["andrews_co"]
raw = os.path.join(DATA, "andrews_gis_raw.csv")

if not os.path.exists(raw):
    rows, pg = G.fetch(1, BOX, "API,GIS_API5,GIS_LAT83,GIS_LONG83")
    with open(raw, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["API", "GIS_API5", "lat", "lon"])
        for a in rows:
            w.writerow([a.get("API"), a.get("GIS_API5"), a.get("GIS_LAT83"), a.get("GIS_LONG83")])
    print("pulled + cached %d gis rows (%d pages)" % (len(rows), pg))
gis = list(csv.DictReader(open(raw)))
print("gis rows:", len(gis))

tops = set()
for r in csv.DictReader(open(os.path.join(DATA, "wolfcamp_tops.csv"), encoding="utf-8")):
    if r["county"] == CTY and "WOLFCAMP" in r["formation_name"].upper() and "DISPOSAL" not in r["formation_name"].upper():
        tops.add(r["api8"])
print("andrews wolfcamp tops api8:", len(tops))
print("  sample tops api8 :", sorted(tops)[:8])
print("  sample gis (API, GIS_API5):", [(g["API"], g["GIS_API5"]) for g in gis[:8]])
print("  GIS API len dist :", collections.Counter(len((g["API"] or "").strip()) for g in gis))
print("  GIS API5 len dist:", collections.Counter(len((g["GIS_API5"] or "").strip()) for g in gis))


def keyset(fn):
    return {k for k in (fn(g) for g in gis) if k}


cands = {
    "API[:8]":        lambda g: (g["API"] or "").strip()[:8] if len((g["API"] or "").strip()) >= 8 else None,
    "API full":       lambda g: (g["API"] or "").strip() or None,
    "API last8":      lambda g: (g["API"] or "").strip()[-8:] if len((g["API"] or "").strip()) >= 8 else None,
    "CTY+API5":       lambda g: CTY + (g["GIS_API5"] or "").strip().zfill(5) if (g["GIS_API5"] or "").strip() else None,
    "API5 only(vs unique)": lambda g: (g["GIS_API5"] or "").strip().zfill(5) if (g["GIS_API5"] or "").strip() else None,
}
print("\njoin overlap of each candidate key vs %d tops:" % len(tops))
tops_uniq = {t[3:] for t in tops}                                   # 5-digit unique part
for nm, fn in cands.items():
    ks = keyset(fn)
    ov = len(tops & ks) if "only" not in nm else len(tops_uniq & ks)
    print("  %-22s : %d unique keys, overlap %d (%.0f%%)" % (nm, len(ks), ov, 100 * ov / len(tops)))
