"""Peek at PDQ .dsv table headers + sample rows (streamed from the zip, no extraction) to learn
the delimiter and which columns carry API/lease linkage and cumulative production."""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "PDQ_DSV.zip")
PEEK = ["OG_WELL_COMPLETION_DATA_TABLE.dsv", "OG_SUMMARY_ONSHORE_LEASE_DATA_TABLE.dsv",
        "OG_REGULATORY_LEASE_DW_DATA_TABLE.dsv", "OG_LEASE_CYCLE_DATA_TABLE.dsv",
        "OG_FIELD_DW_DATA_TABLE.dsv"]


def detect_delim(line):
    for d in ["}", "|", "\t", "~", ",", ";"]:
        if line.count(d) >= 3:
            return d
    return None


with zipfile.ZipFile(ZIP) as z:
    for name in PEEK:
        print("=" * 28, name)
        with z.open(name) as fh:
            raw = fh.read(4000).decode("latin-1", "replace")
        lines = raw.splitlines()
        delim = detect_delim(lines[0]) if lines else None
        print("  delimiter:", repr(delim))
        if lines:
            hdr = lines[0].split(delim) if delim else [lines[0]]
            print("  columns (%d):" % len(hdr))
            for i, c in enumerate(hdr):
                print("    [%2d] %s" % (i, c.strip()))
            for r in lines[1:3]:
                cells = r.split(delim) if delim else [r]
                print("  sample:", cells[:14])
        print()
