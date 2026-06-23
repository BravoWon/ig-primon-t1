#!/usr/bin/env python
"""KGS Kansas lease oil production -> per-lease outcomes (PUBLIC data, local).

Streams the bulk KGS oil-lease monthly files (2000-present) and reduces each LEASE_KID
to cum_oil / best12 / peak / n_months / max_wells, with its API list, producing zone,
county, lat/long. This is the public-data analogue of the TX RRC PDQ reduction -- the
outcome side of a fully-public structure/rock-quality -> production study.

    python kgs_prod_agg.py
"""
from __future__ import annotations
import csv, glob, os
from collections import defaultdict

PROD_DIR = r"C:\Users\JT-DEV1\kgs_public\prod"
OUT = r"C:\Users\JT-DEV1\kgs_public\lease_prod.csv"


def main():
    files = sorted(glob.glob(os.path.join(PROD_DIR, "oil_leases_*.txt")))
    print(f"reducing {len(files)} oil production files:", [os.path.basename(f) for f in files])
    monthly = defaultdict(list)             # lease_kid -> [(ym_int, vol)]
    meta = {}                               # lease_kid -> dict
    rown = 0
    for f in files:
        with open(f, encoding="latin-1", newline="") as fh:
            r = csv.DictReader(fh)
            for row in r:
                rown += 1
                if rown % 5_000_000 == 0:
                    print(f"  ...{rown//1_000_000}M rows  leases={len(monthly)}", flush=True)
                if row.get("PRODUCT") != "O":
                    continue
                lk = row["LEASE_KID"]
                try:
                    vol = float(row["PRODUCTION"] or 0)
                except ValueError:
                    continue
                my = row["MONTH-YEAR"]              # "9-2006"
                try:
                    mm, yy = my.split("-"); ym = int(yy) * 100 + int(mm)
                except Exception:
                    continue
                monthly[lk].append((ym, vol))
                if lk not in meta:
                    meta[lk] = dict(
                        lease=row.get("LEASE", "").strip(),
                        apis=row.get("API_NUMBER", "").strip(),
                        zone=row.get("PRODUCING_ZONE", "").strip(),
                        county=row.get("COUNTY", "").strip(),
                        lat=row.get("LATITUDE", "").strip(),
                        lon=row.get("LONGITUDE", "").strip(),
                        max_wells=0)
                try:
                    meta[lk]["max_wells"] = max(meta[lk]["max_wells"], int(row.get("WELLS") or 0))
                except ValueError:
                    pass

    print(f"  total rows {rown:,}; distinct oil leases {len(monthly):,}")

    def reduce(series):
        series.sort()
        vols = [v for _, v in series]
        cum = sum(vols); peak = max(vols) if vols else 0
        best12 = max((sum(vols[i:i+12]) for i in range(max(1, len(vols) - 11))), default=0)
        first12 = sum(vols[:12])
        return cum, first12, best12, peak, len(series), series[0][0], series[-1][0]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["lease_kid","lease","county","zone","max_wells","apis","lat","lon",
            "cum_oil","first12_oil","best12_oil","peak_oil","n_months","first_ym","last_ym"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for lk, series in monthly.items():
            cum, f12, b12, pk, n, fy, ly = reduce(series)
            m = meta[lk]
            w.writerow([lk, m["lease"], m["county"], m["zone"], m["max_wells"], m["apis"],
                        m["lat"], m["lon"], int(cum), int(f12), int(b12), int(pk), n, fy, ly])

    # ---- scope report -----------------------------------------------------------
    from collections import Counter
    leases = list(monthly)
    single = [lk for lk in leases if meta[lk]["max_wells"] == 1]
    print(f"\n=== KGS LEASE PRODUCTION REDUCED -> {OUT} ===")
    print(f"  oil leases: {len(leases):,}   single-well (clean per-well): {len(single):,}")
    # Central Kansas Uplift structural-carbonate counties
    CKU = {"BARTON","RUSSELL","ELLIS","RUSH","NESS","ROOKS","STAFFORD","PAWNEE","TREGO",
           "RICE","BARBER","COMANCHE","KIOWA","STAFFORD","ELLSWORTH","LINCOLN","OSBORNE"}
    cc = Counter(meta[lk]["county"].upper() for lk in leases)
    print("  Central-Kansas-Uplift carbonate counties (oil leases | single-well):")
    for c in sorted(CKU):
        n = cc.get(c, 0)
        if n:
            ns = sum(1 for lk in single if meta[lk]["county"].upper() == c)
            print(f"    {c:12} {n:5} | {ns:4}")


if __name__ == "__main__":
    main()
