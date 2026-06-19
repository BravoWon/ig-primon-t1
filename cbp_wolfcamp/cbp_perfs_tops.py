"""Powered Stage 2: parse tops + perfs + elevation for ALL CBP/San-Andres counties (one gz re-stream),
with perf->formation matching. Multi-county generalization of andrews_perfs_tops."""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from andrews_perfs_tops import DATA, GZ, marker_of, perf_formations

CODES = set(json.load(open(os.path.join(DATA, "cbp_setup.json")))["target_codes"])


def run():
    out = {}; api = elev = ecode = cc = None; keep = False; tops = []; perfs = []

    def flush():
        if keep and (tops or perfs):
            out[api] = {"cc": cc, "elev": elev, "tops": tops, "perfs": perfs,
                        "perf_forms": sorted(perf_formations(tops, perfs))}

    with gzip.open(GZ, "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            c = line[:2]
            if c == "01":
                flush()
                api = line[2:10]; cc = line[2:5]; keep = cc in CODES
                elev = None; ecode = None; tops = []; perfs = []
            elif keep:
                if c == "03" and elev is None:
                    e = line[85:89].strip()
                    if e.isdigit() and int(e) > 0:
                        elev = int(e); ecode = line[89:91].strip()
                elif c == "09":
                    nm = line[5:37].strip(); d = line[37:42].strip()
                    if d.isdigit() and 1000 < int(d) < 30000:
                        tops.append([nm, int(d)])
                elif c == "07":
                    fr = line[5:10].strip(); to = line[10:15].strip()
                    if fr.isdigit() and to.isdigit() and int(fr) > 0:
                        perfs.append([int(fr), int(to)])
        flush()
    json.dump(out, open(os.path.join(DATA, "cbp_perfs_tops.json"), "w"))
    sa = sum(1 for v in out.values() if "SAN ANDRES" in v["perf_forms"])
    print("CBP wells parsed: %d | perforated in SAN ANDRES: %d" % (len(out), sa))


if __name__ == "__main__":
    run()
