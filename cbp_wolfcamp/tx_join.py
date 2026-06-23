#!/usr/bin/env python
"""TX authoritative production join (LOCAL, no fetching): API -> lease -> RRC PDQ outcome.

Reuses the local 12.7 GB statewide PDQ already on disk. For every TX well in
master_features.csv:
  1) completion table  -> api8 -> (oil_gas_code, district, lease) + wells-per-lease
  2) lease-cycle stream -> cum_oil, cum_gas, n_months, first/last ym, first12/best12/peak oil
  3) join to rock-quality features -> tx_master.csv  (lease-level; n_wells flags clean per-well)

    python tx_join.py
"""
from __future__ import annotations
import csv, io, os, sys, zipfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "PDQ_DSV.zip")
FEAT = r"C:\Users\JT-DEV1\mwd_local_out\master_features.csv"
OUT  = r"C:\Users\JT-DEV1\mwd_local_out\tx_master.csv"


def dsv(member):
    z = zipfile.ZipFile(ZIP)
    fh = io.TextIOWrapper(z.open(member), encoding="latin-1", newline="")
    header = next(fh).rstrip("\n").split("}")
    return fh, {c.strip(): i for i, c in enumerate(header)}


def main():
    rows = list(csv.DictReader(open(FEAT, encoding="utf-8")))
    tx = {r["api10"][2:]: r for r in rows if r["state"] == "TX" and r["api10"]}  # api8 -> feature row
    target_api8 = set(tx)
    print(f"TX wells to link: {len(target_api8)}")

    # --- stage 1: completion table -> api8 -> lease, and wells-per-lease -----------
    fh, ix = dsv("OG_WELL_COMPLETION_DATA_TABLE.dsv")
    iC, iU = ix["API_COUNTY_CODE"], ix["API_UNIQUE_NO"]
    iO, iD, iL = ix["OIL_GAS_CODE"], ix["DISTRICT_NO"], ix["LEASE_NO"]
    api8_lease = {}                         # api8 -> set of (ogc,dist,lease)
    lease_wells = defaultdict(set)          # (ogc,dist,lease) -> set of api8
    for line in fh:
        f = line.rstrip("\n").split("}")
        if len(f) <= max(iU, iL):
            continue
        a8 = f[iC].strip().zfill(3) + f[iU].strip().zfill(5)
        key = (f[iO].strip(), f[iD].strip().zfill(2), f[iL].strip().zfill(6))
        lease_wells[key].add(a8)            # count ALL wells on lease (clean-ness)
        if a8 in target_api8:
            api8_lease.setdefault(a8, set()).add(key)
    target_leases = {k for v in api8_lease.values() for k in v}
    print(f"  linked api8: {len(api8_lease)}/{len(target_api8)}  target leases: {len(target_leases)}")

    # --- stage 2: stream 12.7GB lease cycle, accumulate outcome per target lease ---
    prefixes = {f"{o}}}{d}}}{l}}}" for (o, d, l) in target_leases}   # fast startswith filter
    fh, ix = dsv("OG_LEASE_CYCLE_DATA_TABLE.dsv")
    iO, iD, iL = ix["OIL_GAS_CODE"], ix["DISTRICT_NO"], ix["LEASE_NO"]
    iYM, iOIL = ix["CYCLE_YEAR_MONTH"], ix["LEASE_OIL_PROD_VOL"]
    iGAS, iCSGD = ix["LEASE_GAS_PROD_VOL"], ix["LEASE_CSGD_PROD_VOL"]
    monthly = defaultdict(list)             # lease -> list of (ym, oil, gas)
    scanned = 0
    for line in fh:
        scanned += 1
        if scanned % 20_000_000 == 0:
            print(f"    ...scanned {scanned//1_000_000}M lines", flush=True)
        # cheap prefix gate before full split
        p = line[:24]
        if "}" not in p:
            continue
        f = line.rstrip("\n").split("}")
        if len(f) <= iCSGD:
            continue
        key = (f[iO], f[iD].zfill(2), f[iL].zfill(6))
        if key not in target_leases:
            continue
        try:
            oil = int(f[iOIL] or 0); gas = int(f[iGAS] or 0); csgd = int(f[iCSGD] or 0)
            ym = int(f[iYM])
        except ValueError:
            continue
        monthly[key].append((ym, oil, gas + csgd))

    # --- stage 3: reduce to per-lease outcomes ------------------------------------
    def outcomes(series):
        series.sort()
        oils = [o for _, o, _ in series]
        cum_oil = sum(oils); cum_gas = sum(g for _, _, g in series)
        first12 = sum(oils[:12]); peak = max(oils) if oils else 0
        best12 = max((sum(oils[i:i+12]) for i in range(max(1, len(oils)-11))), default=0)
        return dict(cum_oil=cum_oil, cum_gas=cum_gas, n_months=len(series),
                    first_ym=series[0][0], last_ym=series[-1][0],
                    first12_oil=first12, best12_oil=best12, peak_oil=peak)

    lease_out = {k: outcomes(v) for k, v in monthly.items()}

    # --- stage 4: join back to wells ----------------------------------------------
    out_rows = []
    for a8, frow in tx.items():
        leases = api8_lease.get(a8, set())
        # prefer the OIL lease with most cumulative oil
        best = None
        for k in leases:
            o = lease_out.get(k)
            if o and (best is None or o["cum_oil"] > best[1]["cum_oil"]):
                best = (k, o)
        rec = dict(well=frow["well"], api10=frow["api10"], basin=frow["basin"],
                   operator=frow["operator"], gr_p50=frow["gr_p50"],
                   gr_clean_frac=frow["gr_clean_frac"], gr_shale_frac=frow["gr_shale_frac"],
                   mse_p50_ksi=frow["mse_p50_ksi"], lat_proxy_ft=frow["lat_proxy_ft"])
        if best:
            (o, d, l), out = best
            rec.update(lease=f"{o}-{d}-{l}", n_wells_on_lease=len(lease_wells[(o, d, l)]),
                       **out)
        out_rows.append(rec)

    cols = ["well","api10","basin","operator","lease","n_wells_on_lease","gr_p50","gr_clean_frac",
            "gr_shale_frac","mse_p50_ksi","lat_proxy_ft","cum_oil","cum_gas","first12_oil",
            "best12_oil","peak_oil","n_months","first_ym","last_ym"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh2:
        w = csv.DictWriter(fh2, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    matched = [r for r in out_rows if r.get("cum_oil") is not None and "lease" in r]
    print(f"\n=== TX MASTER: {len(out_rows)} wells, {len(matched)} with lease production -> {OUT} ===")
    for r in sorted(matched, key=lambda x: -(x.get("best12_oil") or 0)):
        print(f"  {r['well'][:30]:31} {r.get('lease',''):14} n={r.get('n_wells_on_lease','?'):>2} "
              f"GRp50={r['gr_p50'] or '-':>5} clean={r['gr_clean_frac'] or '-':>5} "
              f"best12={r.get('best12_oil',0):>9,} cum={r.get('cum_oil',0):>10,}")


if __name__ == "__main__":
    main()
