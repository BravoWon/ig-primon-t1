#!/usr/bin/env python
"""Well -> public-identifier manifest (LOCAL ONLY for the curves; APIs are public record).

Turns the proprietary log set into a JOIN MANIFEST against public databases: one row per
distinct well with a cleaned API-10, state, basin/play, operator, lat/long, and which
log features it carries. The API number is the public key into RRC/OCD/FracFocus etc.
Curves never leave the machine; only public identifiers are emitted.

    python well_manifest.py [out_csv]
"""
from __future__ import annotations
import csv, sys, warnings, logging, re
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("lasio").setLevel(logging.CRITICAL)
import numpy as np, lasio
sys.path.insert(0, str(Path(__file__).parent))
from las_inspect import walk_las

ROOTS = [r"E:\JONESMWDSTK", r"E:\Data", r"E:\MWD-USB"]
OUT   = r"C:\Users\JT-DEV1\mwd_local_out\well_manifest.csv"
MECH  = r"C:\Users\JT-DEV1\mwd_local_out\cross_well_table.csv"

API_STATE = {"42": "TX", "30": "NM", "03": "AR", "35": "OK", "05": "CO", "17": "LA"}


def wkey(n):
    return "".join(c for c in n.upper() if c.isalnum())


def clean_api(*vals):
    for v in vals:
        d = re.sub(r"\D", "", str(v))
        if len(d) >= 10:
            return d[:10]                      # API-10: state(2)+county(3)+well(5)
    return ""


def hdr(las, *keys):
    for k in keys:
        if k in las.well:
            v = str(las.well[k].value).strip()
            if v and v != "-999.25":
                return v
    return ""


def basin(state, county, field):
    c, f = county.upper(), field.upper()
    if state == "AR" or "B-43" in f:
        return "Fayetteville (AR gas)"
    if state == "OK":
        return "Anadarko/Ardmore (OK)"
    if state == "LA" or "LOGANSPORT" in f:
        return "Haynesville (LA)"
    if state == "CO" or "SUSSEX" in f:
        return "DJ (CO)"
    if state == "NM":
        return "Delaware-NM (Permian)"
    if state == "TX":
        if any(x in c for x in ("REEVES", "PECOS", "CULBERSON", "LOVING", "WARD", "WINKLER")):
            return "Delaware-TX (Permian)"
        if any(x in c for x in ("MIDLAND", "REAGAN", "UPTON", "GLASSCOCK", "HOWARD", "MARTIN")):
            return "Midland (Permian)"
        if any(x in c for x in ("LA SALLE", "MCMULLEN", "DIMMIT", "KARNES")):
            return "Eagle Ford (S-TX)"
        return "TX-other"
    return "?"


def channels(las):
    def has(*names):
        for n in names:
            try:
                a = np.asarray(las[n], dtype=float); a[a == -999.25] = np.nan
                if np.isfinite(a).any():
                    return True
            except Exception:
                pass
        return False
    return ("M" if has("SWOB", "WOBX") and has("TQA", "TQX") and has("RPM") else "-",
            "G" if has("GRC", "GRM1", "GR", "GR_MWD") else "-")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    mech_wells = set()
    if Path(MECH).exists():
        for r in csv.DictReader(open(MECH, encoding="utf-8")):
            mech_wells.add(wkey(r["well"]))

    files, seen = [], set()
    for rt in (Path(r) for r in ROOTS):
        if rt.exists():
            for p in walk_las(rt):
                k = str(p).lower()
                if k not in seen:
                    seen.add(k); files.append(p)

    wells = {}                                  # wkey -> richest row
    for f in files:
        try:
            las = lasio.read(str(f), ignore_header_errors=True)
        except Exception:
            continue
        name = hdr(las, "WELL")
        if not name:
            continue
        wk = wkey(name)
        api = clean_api(hdr(las, "API"), hdr(las, "UWI"), hdr(las, "APIN"))
        state = (API_STATE.get(api[:2], "") if api else "") or \
                {"TEXAS": "TX", "NEW MEXICO": "NM", "ARKANSAS": "AR", "OKLAHOMA": "OK",
                 "LOUISIANA": "LA", "COLORADO": "CO"}.get(hdr(las, "STAT", "STATE").upper(),
                                                          hdr(las, "STAT", "STATE")[:2].upper())
        county = hdr(las, "CNTY", "COUN")
        field = hdr(las, "FLD")
        m, g = channels(las)
        # richness: prefer the copy that actually resolved an API + location
        score = (2 if api else 0) + (1 if county else 0) + (1 if m == "M" else 0)
        row = {"well": name, "api10": api, "state": state, "county": county,
               "operator": hdr(las, "COMP", "OPER"), "field": field,
               "basin": basin(state, county, field),
               "mech": m, "gamma": g, "in_mech_table": "Y" if wk in mech_wells else "",
               "_score": score}
        if wk not in wells or score > wells[wk]["_score"]:
            wells[wk] = row

    rows = sorted(wells.values(), key=lambda r: (r["basin"], r["state"], r["well"]))
    cols = ["well", "api10", "state", "county", "operator", "field", "basin",
            "mech", "gamma", "in_mech_table"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ---- summary ---------------------------------------------------------------
    from collections import Counter
    print(f"=== WELL MANIFEST: {len(rows)} distinct wells  (-> {out}) ===")
    n_api = sum(1 for r in rows if r["api10"])
    n_mech = sum(1 for r in rows if r["mech"] == "M")
    n_gam = sum(1 for r in rows if r["gamma"] == "G")
    print(f"  with public API-10: {n_api}/{len(rows)}   with mechanics: {n_mech}   with gamma: {n_gam}")
    print("\n  by basin/play (wells | with-API | mechanics):")
    for b, _ in Counter(r["basin"] for r in rows).most_common():
        sub = [r for r in rows if r["basin"] == b]
        print(f"    {b:26} {len(sub):3} | api {sum(1 for r in sub if r['api10']):3} "
              f"| mech {sum(1 for r in sub if r['mech']=='M'):2}")
    print("\n  by state:")
    for s, c in Counter(r["state"] for r in rows).most_common():
        print(f"    {s or '?':4} {c}")


if __name__ == "__main__":
    main()
