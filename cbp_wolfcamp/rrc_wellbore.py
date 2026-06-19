"""RRC Wellbore bulk -> Wolfcamp formation tops (statewide), fully automated.

Source-verified 2026-06-18 (CBP_WOLFCAMP_STRUCTURAL_EDGE_prereg, v0.2):
  Wellbore DB = ONE hierarchical ASCII file dbf900.txt (1.97 GB), gzipped dbf900.txt.gz (367 MB),
  behind the RRC GoDrive JSF portal (download cracked below; no manual step).
  Record = one line, 2-char leading segment code (RRC-TAPE-RECORD-ID). Child segments belong to the
  most recent '01' root. Field positions confirmed against the WBA091 copybook:
    '01' root  : WB-API-CNTY cols 3-5 + WB-API-UNIQUE 6-10  -> API8 = line[2:10]
    '03' WBDATE: WB-ELEVATION cols 86-89, WB-ELEVATION-CODE 90-91 (DF/GR/GL); recurring, take first non-zero
    '09' form  : WB-FORMATION-NAME cols 6-37, WB-FORMATION-DEPTH(MD) 38-42
  (NB: WB-ELEVATION is in WBDATE '03', NOT completion '02' -- '02' carries no elevation. Verified at source.)
  subsea_top = WB-ELEVATION - WB-FORMATION-DEPTH  (ft; Wolfcamp top depth is operator-reported MD).

PHASE 1: verticals (MD~=TVD). TVD-via-survey expansion is Phase 2 (task #17). This module just extracts
every Wolfcamp pick statewide with its elevation; the CBP/vertical filtering happens downstream at join.
"""
import gzip
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GZ = os.path.join(DATA, "dbf900.txt.gz")
OUT_CSV = os.path.join(DATA, "wolfcamp_tops.csv")
OUT_SUM = os.path.join(DATA, "wolfcamp_tops_summary.json")
LINK = "https://mft.rrc.texas.gov/link/b070ce28-5c58-4fe2-9eb7-8b70befb7af9"
PORTAL = "https://mft.rrc.texas.gov/webclient/godrive/PublicGoDrive.xhtml"
UA = {"User-Agent": "Mozilla/5.0 (research; RRC public data)"}
GZ_ROWKEY = "fileTable:3:j_id_2f"   # dbf900.txt.gz row in the GoDrive file list


def download(dest=GZ):
    """Automated GoDrive JSF download of dbf900.txt.gz -> dest (streamed, resumable-skip if present)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 380_000_000:
        print("  gz already present (%.1f MB), skipping download" % (os.path.getsize(dest) / 1e6)); return dest
    os.makedirs(DATA, exist_ok=True)
    cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    page = op.open(urllib.request.Request(LINK, headers=UA), timeout=60).read().decode("utf-8", "replace")
    vs = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', page).group(1)
    body = urllib.parse.urlencode({"fileList_SUBMIT": "1", "javax.faces.ViewState": vs,
                                   "fileTable_selection": "", GZ_ROWKEY: GZ_ROWKEY}).encode()
    req = urllib.request.Request(PORTAL, data=body, headers={**UA,
                                 "Content-Type": "application/x-www-form-urlencoded", "Referer": LINK})
    r = op.open(req, timeout=180); total = int(r.headers.get("Content-Length", 0)); got = 0; t0 = time.time()
    print("  downloading dbf900.txt.gz (%.0f MB)..." % (total / 1e6))
    with open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
            if got % (50 << 20) < (1 << 20):
                print("    %.0f / %.0f MB  (%.0f s)" % (got / 1e6, total / 1e6, time.time() - t0)); sys.stdout.flush()
    os.replace(dest + ".part", dest)
    print("  done: %.1f MB in %.0f s" % (got / 1e6, time.time() - t0)); return dest


def _i(s):
    s = s.strip()
    return int(s) if s.isdigit() else None


def parse(gz=GZ):
    """Stream the gz, buffer per wellbore, emit Wolfcamp tops. Memory-bounded."""
    n_well = n_form = n_wolf = n_wolf_elev = 0
    names = Counter(); counties = Counter()
    cur_api = None; cur_elev = None; cur_elev_code = None; cur_wolf = []
    t0 = time.time()
    fout = open(OUT_CSV, "w", encoding="utf-8")
    fout.write("api8,county,elevation,elev_code,formation_name,depth_md,subsea_top\n")

    def flush():
        nonlocal n_wolf, n_wolf_elev
        for (nm, dep) in cur_wolf:
            n_wolf += 1
            subsea = (cur_elev - dep) if (cur_elev is not None and dep is not None) else ""
            if subsea != "":
                n_wolf_elev += 1
            fout.write("%s,%s,%s,%s,%s,%s,%s\n" % (cur_api, (cur_api[:3] if cur_api else ""),
                       cur_elev if cur_elev is not None else "", (cur_elev_code or "").strip(),
                       nm.replace(",", " "), dep if dep is not None else "", subsea))

    with gzip.open(gz, "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            code = line[:2]
            if code == "01":
                if cur_api is not None:
                    flush()
                n_well += 1
                cur_api = line[2:10]; cur_elev = None; cur_elev_code = None; cur_wolf = []
                counties[line[2:5]] += 1
                if n_well % 200000 == 0:
                    print("    %dk wellbores, %d wolfcamp picks, %.0fs" % (n_well // 1000, n_wolf, time.time() - t0)); sys.stdout.flush()
            elif code == "03":                                   # WBDATE (recurring) carries elevation
                if cur_elev is None:
                    e = _i(line[85:89])
                    if e:
                        cur_elev = e; cur_elev_code = line[89:91]
            elif code == "09":
                n_form += 1
                nm = line[5:37].strip(); dep = _i(line[37:42])
                if "WOLF" in nm.upper():
                    names[nm.upper()] += 1; cur_wolf.append((nm, dep))
    if cur_api is not None:
        flush()
    fout.close()

    summary = {
        "wellbores": n_well, "formation_records": n_form,
        "wolfcamp_picks": n_wolf, "wolfcamp_picks_with_elevation": n_wolf_elev,
        "distinct_wolfcamp_names_top20": names.most_common(20),
        "county_code_distribution_top20": counties.most_common(20),
        "seconds": round(time.time() - t0, 1), "out_csv": OUT_CSV,
    }
    json.dump(summary, open(OUT_SUM, "w"), indent=2)
    print("\n[WELLBORE -> WOLFCAMP TOPS]")
    print("  wellbores parsed:        %d" % n_well)
    print("  formation records:       %d" % n_form)
    print("  WOLFCAMP picks:          %d  (with elevation: %d)" % (n_wolf, n_wolf_elev))
    print("  distinct wolfcamp names: %s" % names.most_common(8))
    print("  wrote %s" % OUT_CSV)
    return summary


if __name__ == "__main__":
    print("[RRC Wellbore -> Wolfcamp tops]  fully automated, source-verified\n")
    download()
    parse()
