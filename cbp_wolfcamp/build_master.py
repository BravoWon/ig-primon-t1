#!/usr/bin/env python
"""Master de-confounding table, step 1: LOCAL rock-quality features + public URLs.

For every well (grouped by public API across all its runs) compute log-derived rock
quality -- GR distribution (always) and torque-normalized MSE (mechanics subset) --
plus a trajectory proxy, then construct the public-production lookup URL. Curves never
leave the machine; only the public API/identity is emitted for the join.

Output: master_features.csv (LOCAL, off-repo). Production columns are left blank for
step 2 to fill.

    python build_master.py
"""
from __future__ import annotations
import csv, sys, warnings, logging, re
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("lasio").setLevel(logging.CRITICAL)
import numpy as np, lasio
sys.path.insert(0, str(Path(__file__).parent))
from las_inspect import walk_las
from well_deepdive import col, first, pick_torque, teale_mse

ROOTS = [r"E:\JONESMWDSTK", r"E:\Data", r"E:\MWD-USB"]
OUT   = r"C:\Users\JT-DEV1\mwd_local_out\master_features.csv"
REF_BIT = 8.75
API_STATE = {"42": "TX", "30": "NM", "03": "AR", "35": "OK", "05": "CO", "17": "LA"}
STATE_SLUG = {"TX": "texas", "NM": "new-mexico", "AR": "arkansas", "OK": "oklahoma",
              "LA": "louisiana", "CO": "colorado"}


def wkey(n): return "".join(c for c in n.upper() if c.isalnum())
def clean_api(*vals):
    for v in vals:
        d = re.sub(r"\D", "", str(v))
        if len(d) >= 10: return d[:10]
    return ""
def hdr(las, *keys):
    for k in keys:
        if k in las.well:
            v = str(las.well[k].value).strip()
            if v and v != "-999.25": return v
    return ""


def basin(state, county, field):
    c, f = county.upper(), field.upper()
    if state == "AR" or "B-43" in f: return "Fayetteville-AR"
    if state == "OK": return "Anadarko-OK"
    if state == "LA" or "LOGANSPORT" in f: return "Haynesville-LA"
    if state == "CO" or "SUSSEX" in f: return "DJ-CO"
    if state == "NM": return "Delaware-NM"
    if state == "TX":
        if any(x in c for x in ("REEVES","PECOS","CULBERSON","LOVING","WARD","WINKLER")): return "Delaware-TX"
        if any(x in c for x in ("MIDLAND","REAGAN","UPTON","GLASSCOCK","HOWARD","MARTIN")): return "Midland-TX"
        if any(x in c for x in ("LA SALLE","MCMULLEN","DIMMIT","KARNES")): return "EagleFord-TX"
        return "TX-other"
    return "?"


def depth_axis(las):
    idx = np.asarray(las.index, dtype=float)
    d = np.nanmedian(np.diff(idx))
    if idx.size >= 50 and np.isfinite(idx[0]) and 0 <= idx[0] < 40000 and 0.02 < abs(d) < 5:
        return idx
    dept = col(las, "DEPT")
    if dept is not None:
        fin = dept[np.isfinite(dept)]
        if fin.size >= 50 and 0 <= np.min(fin) < 40000 and (np.max(fin)-np.min(fin)) > 100:
            return dept
    return None


def run_features(las):
    """Return per-run (gr_array_valid, mse_p50_ksi_or_nan, md_max, tvd_max, n_gr)."""
    md = depth_axis(las)
    if md is None: return None
    grc, _ = first(las, "GRC", "GRM1", "GR", "GR_MWD")
    tvd, _ = first(las, "MTTVD", "TVD", "TVDE")
    wob, _ = first(las, "SWOB", "WOBX")
    rpm = col(las, "RPM"); rop = col(las, "ROP")
    md_max = float(np.nanmax(md[np.isfinite(md)])) if np.isfinite(md).any() else np.nan
    tvd_max = float(np.nanmax(tvd[np.isfinite(tvd)])) if (tvd is not None and np.isfinite(tvd).any()) else np.nan
    gr_valid = grc[np.isfinite(grc)] if grc is not None else np.array([])
    mse_p50 = np.nan
    if all(x is not None for x in (wob, rpm, rop)):
        drilling = (np.isfinite(rop)&(rop>5)&np.isfinite(wob)&(wob>1)&np.isfinite(rpm)&(rpm>10))
        if drilling.sum() >= 200:
            tq, _, _ = pick_torque(las, drilling)
            if tq is not None and np.isfinite(tq).any():
                mse = teale_mse(wob, rpm, tq, rop, REF_BIT)
                m = drilling & np.isfinite(mse) & (mse > 0)
                if m.any(): mse_p50 = float(np.median(mse[m]) / 1e3)
    return gr_valid, mse_p50, md_max, tvd_max, len(gr_valid)


def main():
    out = Path(OUT); out.parent.mkdir(parents=True, exist_ok=True)
    files, seen = [], set()
    for rt in (Path(r) for r in ROOTS):
        if rt.exists():
            for p in walk_las(rt):
                k = str(p).lower()
                if k not in seen: seen.add(k); files.append(p)

    # group runs by well key (API if present else normalized name)
    runs = defaultdict(list)          # key -> list of (features, identity)
    for f in files:
        try:
            las = lasio.read(str(f), ignore_header_errors=True)
        except Exception:
            continue
        name = hdr(las, "WELL")
        if not name: continue
        api = clean_api(hdr(las, "API"), hdr(las, "UWI"), hdr(las, "APIN"))
        key = api or wkey(name)
        feat = run_features(las)
        if feat is None: continue
        state = (API_STATE.get(api[:2], "") if api else "") or \
            {"TEXAS":"TX","NEW MEXICO":"NM","ARKANSAS":"AR","OKLAHOMA":"OK","LOUISIANA":"LA","COLORADO":"CO"}\
            .get(hdr(las,"STAT","STATE").upper(), hdr(las,"STAT","STATE")[:2].upper())
        ident = dict(well=name, api10=api, state=state, county=hdr(las,"CNTY","COUN"),
                     operator=hdr(las,"COMP","OPER"), field=hdr(las,"FLD"))
        runs[key].append((feat, ident))

    rows = []
    for key, lst in runs.items():
        feats = [f for f, _ in lst]
        # richest identity: prefer one with api + county
        ident = max((i for _, i in lst), key=lambda i: (bool(i["api10"]), bool(i["county"]), len(i["well"])))
        # best GR run = most valid GR samples
        best_gr = max(feats, key=lambda x: x[4])
        gr = best_gr[0]
        mse_vals = [f[1] for f in feats if np.isfinite(f[1])]
        md_max = max((f[2] for f in feats if np.isfinite(f[2])), default=np.nan)
        tvd_max = max((f[3] for f in feats if np.isfinite(f[3])), default=np.nan)
        bsn = basin(ident["state"], ident["county"], ident["field"])
        lat_proxy = (md_max - tvd_max) if (np.isfinite(md_max) and np.isfinite(tvd_max)) else np.nan
        # construct public URL (drillingedge) where we have state+county+api
        url = ""
        if ident["api10"] and ident["state"] in STATE_SLUG and ident["county"]:
            cty = re.sub(r"\bCO\.?$", "", ident["county"].strip(), flags=re.I).strip()
            cty = re.sub(r"[^a-z0-9]+", "-", cty.lower()).strip("-")
            wl = re.sub(r"[^a-z0-9]+", "-", ident["well"].lower()).strip("-")
            a = ident["api10"]; ad = f"{a[:2]}-{a[2:5]}-{a[5:]}"
            url = f"https://www.drillingedge.com/{STATE_SLUG[ident['state']]}/{cty}-county/wells/{wl}/{ad}"
        rows.append(dict(
            well=ident["well"], api10=ident["api10"], state=ident["state"], county=ident["county"],
            basin=bsn, operator=ident["operator"], field=ident["field"],
            gr_n=len(gr),
            gr_p10=round(float(np.percentile(gr,10)),1) if gr.size>20 else "",
            gr_p50=round(float(np.median(gr)),1) if gr.size>20 else "",
            gr_p90=round(float(np.percentile(gr,90)),1) if gr.size>20 else "",
            gr_clean_frac=round(float(np.mean(gr<60)),3) if gr.size>20 else "",
            gr_shale_frac=round(float(np.mean(gr>90)),3) if gr.size>20 else "",
            mse_p50_ksi=round(float(np.median(mse_vals)),1) if mse_vals else "",
            md_max=round(md_max,0) if np.isfinite(md_max) else "",
            tvd_max=round(tvd_max,0) if np.isfinite(tvd_max) else "",
            lat_proxy_ft=round(lat_proxy,0) if np.isfinite(lat_proxy) else "",
            url=url,
            cum_oil_bbl="", cum_gas_mcf="", first_prod="", recent_oil="", recent_gas="",
            status="", well_type="",
        ))

    cols = ["well","api10","state","county","basin","operator","field","gr_n","gr_p10","gr_p50",
            "gr_p90","gr_clean_frac","gr_shale_frac","mse_p50_ksi","md_max","tvd_max","lat_proxy_ft",
            "url","cum_oil_bbl","cum_gas_mcf","first_prod","recent_oil","recent_gas","status","well_type"]
    rows.sort(key=lambda r: (r["basin"], r["state"], r["well"]))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    joinable = [r for r in rows if r["api10"] and (r["gr_p50"] != "" or r["mse_p50_ksi"] != "")]
    have_url = [r for r in rows if r["url"]]
    print(f"=== MASTER FEATURES: {len(rows)} wells -> {out} ===")
    print(f"  joinable (API + rock-quality): {len(joinable)}   with constructed URL: {len(have_url)}")
    from collections import Counter
    for b, c in Counter(r["basin"] for r in rows).most_common():
        sub = [r for r in rows if r["basin"] == b]
        print(f"    {b:16} wells {len(sub):2} | api {sum(1 for r in sub if r['api10']):2} | "
              f"GR {sum(1 for r in sub if r['gr_p50']!=''):2} | MSE {sum(1 for r in sub if r['mse_p50_ksi']!=''):2} | "
              f"url {sum(1 for r in sub if r['url']):2}")


if __name__ == "__main__":
    main()
