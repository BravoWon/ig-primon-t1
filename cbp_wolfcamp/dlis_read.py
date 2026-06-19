"""Read PROPRIETARY LWD logs (DLIS) on-machine, stays local: enumerate channels, pull petrophysical curves
(GR, resistivity, density/neutron if present), compute a porosity where density exists. This is the
reservoir-quality dimension the PUBLIC tensor lacked -- the term that physically couples to carbonate flow.
Demonstration of the proprietary-petrophysics ingestion (the moat). NOTE: these are scattered horizontal
LWD wells, NOT the CBP San Andres vertical population -- capability proof, not a model plug-in.
"""
import os

import numpy as np
from dlisio import dlis

CANDIDATES = [
    r"E:\MWD-USB\EOW_QC\EOW_QC\03029114360000_Desalvo Tony 8-14 2-6H36_DLIS_FINAL.dlis",
    r"E:\MWD-USB\Desalvo Tony 8-14 2-6H31\Client Deliverables\03029114360000_Desalvo Tony 8-14 2-6H36_DLIS_FINAL.dlis",
    r"E:\MWD-USB\MISC\HorizonJobArchivePr__17Dec15_160816.Dlis",
]
PETRO = {"GR": ["GR", "GAM", "ECGR", "GRMA"], "RES": ["RES", "RT", "A40", "P40", "RESISTIVITY", "AT", "RLA"],
         "RHOB": ["RHOB", "DEN", "ROBB", "DENS"], "NPHI": ["NPHI", "TNPH", "NEU", "BPHI", "PORO"],
         "PEF": ["PEF", "PE"]}


def classify(name):
    u = name.upper()
    for k, pats in PETRO.items():
        if any(p in u for p in pats):
            return k
    return None


def run():
    path = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if not path:
        print("no DLIS found at expected paths"); return
    print("reading: %s (%.1f MB)\n" % (os.path.basename(path), os.path.getsize(path) / 1e6))
    with dlis.load(path) as physical:
        for li, logical in enumerate(physical):
            chans = logical.channels
            print("logical file %d: %d channels" % (li, len(chans)))
            found = {}
            for c in chans:
                k = classify(c.name)
                if k and k not in found:
                    found[k] = c
            print("  petrophysical channels found: %s" % {k: (v.name, v.units) for k, v in found.items()})
            for fr in logical.frames:
                try:
                    cur = fr.curves()
                except Exception as e:                          # noqa: BLE001
                    print("  frame %s: curves() failed %r" % (fr.name, e)); continue
                names = list(cur.dtype.names)
                idx = fr.index or names[0]
                depth = cur[idx] if idx in names else cur[names[0]]
                print("  frame %s: %d samples, index %s [%.0f..%.0f]" % (fr.name, len(depth), idx,
                      np.nanmin(depth), np.nanmax(depth)))
                for k, ch in found.items():
                    if ch.name in names:
                        v = cur[ch.name].astype(float); vv = v[np.isfinite(v)]
                        if len(vv):
                            print("    %-5s (%s): mean %.2f  p10 %.2f  p90 %.2f  %s"
                                  % (k, ch.name, vv.mean(), np.percentile(vv, 10), np.percentile(vv, 90), ch.units))
                # density-porosity if RHOB present (limestone matrix 2.71, fluid 1.0)
                if "RHOB" in found and found["RHOB"].name in names:
                    rb = cur[found["RHOB"].name].astype(float); rb = rb[np.isfinite(rb)]
                    rb = rb[(rb > 1.5) & (rb < 3.2)]
                    if len(rb):
                        phi = (2.71 - rb) / (2.71 - 1.0)
                        print("    -> DENSITY POROSITY (limestone): mean %.1f%%  p10 %.1f%%  p90 %.1f%%  (the missing reservoir-quality dimension)"
                              % (100 * phi.mean(), 100 * np.percentile(phi, 10), 100 * np.percentile(phi, 90)))
                break                                            # first frame is enough for the demo
            break


if __name__ == "__main__":
    run()
