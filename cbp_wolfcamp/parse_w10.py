"""Parse the Oil Detail Well (W-10) EBCDIC -> per-well oil TEST RATE (bbl/d). RECL=100, cp037.
DW layout: DST 1-3, FIELD 4-11, OPER 12-17, LSE 18-22, WELLNO 24-29, CO 30-32, TST-DT 43-50, DW-OIL 51-55;
the MOST RECENT test repeats +24 (DW-OIL2 at 75-79). Per-WELL metric -> breaks the single-well-lease n-wall.
"""
import http.cookiejar
import os
import re
import urllib.parse
import urllib.request
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
LINK = "https://mft.rrc.texas.gov/link/f5ac3552-50ce-4959-844d-5079de0f1f62"
UA = {"User-Agent": "Mozilla/5.0"}


def download():
    p = os.path.join(DATA, "oltdw.ebc")
    if os.path.exists(p):
        return open(p, "rb").read()
    cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    page = op.open(urllib.request.Request(LINK, headers=UA), timeout=60).read().decode("utf-8", "replace")
    vs = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', page).group(1)
    src = re.search(r'id="(fileTable:\d+:[^"]+)"[^>]*onclick="PrimeFaces\.addSubmitParam[^>]*>oltdw\.ebc\.gz', page).group(1)
    body = urllib.parse.urlencode({"fileList_SUBMIT": "1", "javax.faces.ViewState": vs, "fileTable_selection": "", src: src}).encode()
    r = op.open(urllib.request.Request("https://mft.rrc.texas.gov/webclient/godrive/PublicGoDrive.xhtml", data=body,
                headers={**UA, "Content-Type": "application/x-www-form-urlencoded", "Referer": LINK}), timeout=120)
    raw = zlib.decompress(r.read(), 16 + zlib.MAX_WBITS); open(p, "wb").write(raw); return raw


def _num(s):
    s = s.strip()
    return int(s) / 10.0 if s.isdigit() else None       # DW-OIL is 9(4)V9 -> 1 implied decimal


def run():
    raw = download(); text = raw.decode("cp037", "replace"); recs = [text[i:i + 100] for i in range(0, len(raw), 100)]
    print("W-10 records: %d" % len(recs))
    import csv
    out = open(os.path.join(DATA, "w10_well_rates.csv"), "w", newline="")
    w = csv.writer(out); w.writerow(["dst", "lease", "wellno", "county", "test_date", "oil_test_bbld"])
    rates = []; n = 0
    for r in recs:
        dst = r[0:3].strip(); lse = r[17:22].strip(); wno = r[23:29].strip(); co = r[29:32].strip()
        oil2 = _num(r[74:79]); td2 = r[66:74].strip()       # most recent test
        oil1 = _num(r[50:55]); td1 = r[42:50].strip()       # previous test
        oil = oil2 if oil2 else oil1; td = td2 if oil2 else td1
        if oil is not None:
            w.writerow([dst, lse, wno, co, td, oil]); n += 1
            if oil > 0:
                rates.append(oil)
    out.close()
    rates = np.array(rates)
    print("  per-well oil test rates extracted: %d (nonzero: %d)" % (n, len(rates)))
    if len(rates):
        print("  oil test rate (bbl/d): median %.1f | p10 %.1f | p90 %.1f | max %.0f"
              % (np.median(rates), np.percentile(rates, 10), np.percentile(rates, 90), rates.max()))
    print("  wrote w10_well_rates.csv  (key: district+lease+well -> join to API via OG_WELL_COMPLETION)")


if __name__ == "__main__":
    run()
