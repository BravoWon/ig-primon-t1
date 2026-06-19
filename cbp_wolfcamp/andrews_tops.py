"""Parse ALL Andrews formation tops (with elevation->subsea) once, so any formation pivot is free.
Re-streams the cached wellbore gz, keeps county-003 wellbores only."""
import csv
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
GZ = os.path.join(DATA, "dbf900.txt.gz"); CTY = "003"


def run():
    out = open(os.path.join(DATA, "andrews_tops.csv"), "w", newline="")
    w = csv.writer(out); w.writerow(["api8", "formation_name", "depth_md", "elevation", "elev_code", "subsea"])
    api = elev = elev_code = None; is3 = False; tops = []; nrows = 0

    def flush():
        nonlocal nrows
        if is3 and elev:
            for nm, dep in tops:
                w.writerow([api, nm, dep, elev, elev_code, elev - dep]); nrows += 1

    with gzip.open(GZ, "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            c = line[:2]
            if c == "01":
                flush()
                api = line[2:10]; is3 = (line[2:5] == CTY); elev = None; elev_code = None; tops = []
            elif c == "03" and is3 and elev is None:
                e = line[85:89].strip()
                if e.isdigit() and int(e) > 0:
                    elev = int(e); elev_code = line[89:91].strip()
            elif c == "09" and is3:
                nm = line[5:37].strip(); dep = line[37:42].strip()
                if dep.isdigit() and 1000 < int(dep) < 30000:
                    tops.append((nm, int(dep)))
        flush()
    out.close()
    print("wrote andrews_tops.csv (%d formation-top rows)" % nrows)


if __name__ == "__main__":
    run()
