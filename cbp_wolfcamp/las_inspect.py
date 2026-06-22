#!/usr/bin/env python
"""LAS inventory + structure inspector (local-only; for proprietary MWD/LWD logs).

Generic: takes a directory, parses one representative file in full (header + curves +
depth range), then catalogs curve mnemonics / well count / depth coverage across the set.
Prints STRUCTURE and AGGREGATE stats only -- no bulk dump of proprietary curve values.

    python las_inspect.py <dir> [sample.las]
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
import lasio


def walk_las(root: Path):
    seen = set()
    for p in root.rglob("*"):
        if p.suffix.lower() == ".las" and "trash" not in str(p).lower():
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                yield p


def inspect_one(path: Path):
    las = lasio.read(str(path), ignore_header_errors=True)
    print(f"\n=== SAMPLE: {path.name} ===")
    # well header (a few standard keys)
    for k in ("WELL", "COMP", "FLD", "STRT", "STOP", "STEP", "NULL"):
        try:
            it = las.well[k]
            print(f"  {k:5} = {it.value}   ({it.descr})")
        except Exception:
            pass
    print(f"  curves ({len(las.curves)}):")
    for c in las.curves:
        n = 0
        try:
            import numpy as np
            arr = c.data
            n = int(np.sum(~np.isnan(arr)))
        except Exception:
            pass
        print(f"    {c.mnemonic:12} [{c.unit or '-':6}]  {('valid=%d' % n):12} {c.descr[:40]}")
    try:
        d = las.index
        print(f"  index span: {d[0]:.1f} .. {d[-1]:.1f}  ({len(d)} samples)")
    except Exception:
        pass


def catalog(root: Path, limit=None):
    files = list(walk_las(root))
    if limit:
        files = files[:limit]
    curve_count = Counter()
    wells = set()
    depth_logs = time_logs = parse_fail = 0
    spans = []
    for i, f in enumerate(files):
        try:
            las = lasio.read(str(f), ignore_header_errors=True)
            mnem = [c.mnemonic.upper() for c in las.curves]
            curve_count.update(set(mnem))
            try:
                wells.add(str(las.well["WELL"].value).strip())
            except Exception:
                pass
            idx0 = las.curves[0].mnemonic.upper() if las.curves else ""
            if "TIME" in idx0 or "TIME" in f.name.upper():
                time_logs += 1
            else:
                depth_logs += 1
            try:
                spans.append(float(las.index[-1]) - float(las.index[0]))
            except Exception:
                pass
        except Exception:
            parse_fail += 1
        if (i + 1) % 100 == 0:
            print(f"  ...scanned {i+1}/{len(files)}", flush=True)
    print(f"\n=== CATALOG: {root} ===")
    print(f"  files parsed: {len(files)-parse_fail}/{len(files)} (parse_fail={parse_fail})")
    print(f"  distinct wells (by WELL header): {len(wells)}")
    print(f"  depth-indexed: {depth_logs}   time-indexed: {time_logs}")
    if spans:
        import numpy as np
        sp = np.array(spans)
        print(f"  index span: median {np.median(sp):.0f}, max {sp.max():.0f}")
    print(f"  most common curve mnemonics (data types present):")
    for mn, c in curve_count.most_common(30):
        print(f"    {mn:14} in {c:4} files")


def main():
    root = Path(sys.argv[1])
    if len(sys.argv) > 2:
        inspect_one(Path(sys.argv[2]))
    else:
        # auto-pick a depth-module file for the sample
        sample = next((f for f in walk_las(root)
                       if "depth" in f.name.lower() and "trash" not in str(f).lower()), None)
        if sample:
            inspect_one(sample)
    catalog(root)


if __name__ == "__main__":
    main()
