"""Index production to the PERFORATED INTERVAL: parse perfs ('07': WB-FROM-PERF cols 6-10, WB-TO-PERF 11-15)
+ formation tops ('09') + elevation ('03') for Andrews wells, then match each perf to the formation it
penetrates. Answers: which wells actually PRODUCE FROM San Andres (vs merely having a SA top above a deeper
completion)?  This is the 'two topologies rematricized' fix: discrete completion events on continuous geology.
"""
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
GZ = os.path.join(DATA, "dbf900.txt.gz"); CTY = "003"
MARKERS = ["SAN ANDRES", "GRAYBURG", "GLORIETA", "CLEAR FORK", "TUBB", "DRINKARD", "ABO", "SPRABERRY",
           "DEAN", "WOLFCAMP", "CISCO", "CANYON", "STRAWN", "ATOKA", "MISSISSIPPIAN", "DEVONIAN",
           "FUSSELMAN", "MONTOYA", "ELLENBURGER"]


def marker_of(name):
    u = name.upper()
    return next((m for m in MARKERS if m in u), None)


def perf_formations(tops, perfs):
    """tops: [[name, md]]; perfs: [[from, to]] -> set of markers the perfs penetrate."""
    mk = sorted([(d, marker_of(n)) for n, d in tops if marker_of(n)], key=lambda x: x[0])
    if not mk:
        return set()
    out = set()
    for fr, to in perfs:
        mid = (fr + to) / 2
        cand = [m for d, m in mk if d <= mid]
        if cand:
            out.add(cand[-1])                         # deepest top at/above the perf midpoint
    return out


def run():
    out = {}; api = elev = ecode = None; is3 = False; tops = []; perfs = []

    def flush():
        if is3 and (tops or perfs):
            out[api] = {"elev": elev, "code": ecode, "tops": tops, "perfs": perfs,
                        "perf_forms": sorted(perf_formations(tops, perfs))}

    with gzip.open(GZ, "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            c = line[:2]
            if c == "01":
                flush()
                api = line[2:10]; is3 = (line[2:5] == CTY); elev = None; ecode = None; tops = []; perfs = []
            elif is3:
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
    json.dump(out, open(os.path.join(DATA, "andrews_perfs_tops.json"), "w"))

    from collections import Counter
    nperf = sum(1 for v in out.values() if v["perfs"])
    has_sa_top = [v for v in out.values() if any(marker_of(n) == "SAN ANDRES" for n, _ in v["tops"])]
    perf_in_sa = [v for v in has_sa_top if "SAN ANDRES" in v["perf_forms"]]
    pf = Counter(m for v in out.values() for m in v["perf_forms"])
    print("Andrews wells parsed: %d | with perforations: %d" % (len(out), nperf))
    print("\nperf-formation distribution (wells completed in each):")
    for m, n in pf.most_common(14):
        print("   %-14s %d" % (m, n))
    print("\n[THE CONFOUND, quantified]")
    print("  wells with a SAN ANDRES top:           %d" % len(has_sa_top))
    print("  ...actually PERFORATED in San Andres:  %d (%.0f%%)" % (len(perf_in_sa), 100 * len(perf_in_sa) / max(len(has_sa_top), 1)))
    print("  -> the rest had a SA top but completed in another zone = noise in the earlier test.")


if __name__ == "__main__":
    run()
