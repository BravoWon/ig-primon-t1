"""Download the RRC PDQ production dump (PDQ_DSV.zip, 3.37 GB) via the cracked GoDrive JSF portal.

Lease-level production (1993-present). API<->lease crosswalk via OG_WELL_COMPLETION. We stream tables
straight out of the zip later (no full extraction). Nuances carried downstream:
  - oil reported by LEASE (multi-well) -> prefer single-well leases / document allocation
  - PDQ starts 1993 -> old wells' pre-1993 peak is missing -> use post-1993 first-prod or vintage-normalize
"""
import http.cookiejar
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research; RRC public data)"}
PDQ_LINK = "https://mft.rrc.texas.gov/link/1f5ddb8d-329a-4459-b7f8-177b4f5ee60d"
PORTAL = "https://mft.rrc.texas.gov/webclient/godrive/PublicGoDrive.xhtml"
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "PDQ_DSV.zip")


def download(dest=ZIP):
    if os.path.exists(dest) and os.path.getsize(dest) > 3_000_000_000:
        print("  PDQ zip already present (%.2f GB)" % (os.path.getsize(dest) / 1e9)); return dest
    os.makedirs(DATA, exist_ok=True)
    cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    page = op.open(urllib.request.Request(PDQ_LINK, headers=UA), timeout=60).read().decode("utf-8", "replace")
    vs = re.search(r'name="javax\.faces\.ViewState"[^>]*value="([^"]+)"', page).group(1)
    m = re.search(r'id="(fileTable:\d+:[^"]+)"[^>]*onclick="PrimeFaces\.addSubmitParam[^>]*>PDQ_DSV\.zip', page)
    src = m.group(1)
    body = urllib.parse.urlencode({"fileList_SUBMIT": "1", "javax.faces.ViewState": vs,
                                   "fileTable_selection": "", src: src}).encode()
    req = urllib.request.Request(PORTAL, data=body, headers={**UA,
                                 "Content-Type": "application/x-www-form-urlencoded", "Referer": PDQ_LINK})
    r = op.open(req, timeout=300); total = int(r.headers.get("Content-Length", 0)); got = 0; t0 = time.time()
    print("  downloading PDQ_DSV.zip (%.2f GB)..." % (total / 1e9))
    with open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
            if got % (250 << 20) < (1 << 20):
                print("    %.0f / %.0f MB  (%.0fs)" % (got / 1e6, total / 1e6, time.time() - t0)); sys.stdout.flush()
    os.replace(dest + ".part", dest)
    print("  done: %.2f GB in %.0fs" % (got / 1e9, time.time() - t0)); return dest


if __name__ == "__main__":
    download()
    import zipfile
    with zipfile.ZipFile(ZIP) as z:
        print("\n[PDQ_DSV.zip members:]")
        for i in z.infolist():
            print("  %-40s %12.1f MB" % (i.filename, i.file_size / 1e6))
