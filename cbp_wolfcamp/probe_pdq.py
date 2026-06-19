"""Verify the PDQ production source at source: enumerate the package files, and pull the record layout
from the PDQ dump manual -- specifically how production links to API / lease (the lease-allocation nuance)."""
import html as H
import os
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research)"}
PDQ_LINK = "https://mft.rrc.texas.gov/link/1f5ddb8d-329a-4459-b7f8-177b4f5ee60d"
MANUAL = "https://www.rrc.texas.gov/media/50ypu2cg/pdq-dump-user-manual.pdf"
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def enumerate_files():
    page = urllib.request.urlopen(urllib.request.Request(PDQ_LINK, headers=UA), timeout=60).read().decode("utf-8", "replace")
    rows = re.findall(r'data-ri="\d+"\s+data-rk="(\d+)"(.*?)</tr>', page, re.S)
    print("[PDQ package files: %d]" % len(rows))
    print("  %-8s %-34s %s" % ("rowkey", "filename", "size"))
    for rk, body in rows:
        nm = re.search(r'>([^<>]+\.(?:txt|gz|zip|dat|csv|exe|pdf))</a>', body, re.I)
        sz = re.search(r'class="SizeColumn">([^<]+)</td>', body)
        print("  %-8s %-34s %s" % (rk, H.unescape(nm.group(1)) if nm else "?", sz.group(1).strip() if sz else "?"))


def manual_schema():
    p = os.path.join(DATA, "pdq_manual.pdf")
    if not os.path.exists(p):
        os.makedirs(DATA, exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(MANUAL, headers=UA), timeout=90).read()
        open(p, "wb").write(data)
    from pypdf import PdfReader
    full = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
    print("\n[PDQ manual: %d chars]" % len(full))
    # field/column names of interest
    print("\n--- lines mentioning API / LEASE / WELL / DISTRICT / FIELD / OIL / GAS / DATE / linkage ---")
    seen = 0
    for ln in full.splitlines():
        L = ln.strip()
        if re.search(r'\b(API|LEASE|WELL[ _-]?(NO|NBR|ID)|DISTRICT|FIELD[ _-]?(NO|NBR)|GAS[ _-]?WELL|OIL[ _-]?GAS[ _-]?CODE|MONTH|YEAR|PRODUCTION|VOLUME|CYCLE)\b', L, re.I) and 6 < len(L) < 95:
            print("   ", L); seen += 1
            if seen > 45:
                break


if __name__ == "__main__":
    enumerate_files()
    manual_schema()
