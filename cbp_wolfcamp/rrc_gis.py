"""RRC GIS ingestion — pull well locations by polygon from the AUTHORITATIVE RRC ArcGIS server.

Source-verified 2026-06-18 (see CBP_WOLFCAMP_STRUCTURAL_EDGE_prereg_v0_1.md, v0.2):
  gis.rrc.texas.gov/server/.../rrc_public/RRC_Public_Viewer_Srvs/MapServer
  - Layer 1 "Well Locations" = 1,394,934 wells statewide, SR EPSG:4326, maxRecordCount 1000.
  - Layer 9 "Horiz/Dir Surface Locations" = horizontal/directional discriminator.
  API is the join key to the Wellbore (tops) and PDQ (production) bulk datasets.

NOT the gis.hctx.net mirror (only 12,796 wells, partial, non-CBP) — that trap is documented in the pre-reg.
Fully automated HTTP; zero manual downloads.
"""
import json
import time
import urllib.parse
import urllib.request

RRC = "https://gis.rrc.texas.gov/server/rest/services/rrc_public/RRC_Public_Viewer_Srvs/MapServer"
UA = {"User-Agent": "Mozilla/5.0 (research; RRC public data)"}
PAGE = 1000  # = layer maxRecordCount


def _get(url, timeout=90, retries=3):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:                                  # noqa: BLE001
            last = e; time.sleep(1.5 * (k + 1))
    raise last


def count(layer, bbox):
    p = {"where": "1=1", "geometry": "%f,%f,%f,%f" % bbox, "geometryType": "esriGeometryEnvelope",
         "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "returnCountOnly": "true", "f": "json"}
    return _get("%s/%d/query?%s" % (RRC, layer, urllib.parse.urlencode(p))).get("count")


def fetch_where(where, out_fields, layer=1, max_pages=None):
    """Paginated ATTRIBUTE-query pull (e.g. where=\"API LIKE '003%'\"). Avoids bbox clipping of counties."""
    base = "%s/%d/query" % (RRC, layer)
    common = {"where": where, "outFields": out_fields, "returnGeometry": "false", "f": "json"}
    rows, off, pages = [], 0, 0
    while True:
        p = dict(common); p["resultOffset"] = off; p["resultRecordCount"] = PAGE
        d = _get("%s?%s" % (base, urllib.parse.urlencode(p)))
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]; pages += 1
        if not feats or not d.get("exceededTransferLimit"):
            break
        off += len(feats)
        if max_pages and pages >= max_pages:
            break
    return rows


def fetch(layer, bbox, out_fields, max_pages=None):
    """Paginated polygon pull. Returns list of attribute dicts (NAD83 lat/long in EPSG:4326)."""
    base = "%s/%d/query" % (RRC, layer)
    common = {"where": "1=1", "geometry": "%f,%f,%f,%f" % bbox, "geometryType": "esriGeometryEnvelope",
              "inSR": "4326", "outSR": "4326", "spatialRel": "esriSpatialRelIntersects",
              "outFields": out_fields, "returnGeometry": "false", "f": "json"}
    rows, offset, pages = [], 0, 0
    while True:
        p = dict(common); p["resultOffset"] = offset; p["resultRecordCount"] = PAGE
        d = _get("%s?%s" % (base, urllib.parse.urlencode(p)))
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        pages += 1
        if not feats or not d.get("exceededTransferLimit"):
            break
        offset += len(feats)
        if max_pages and pages >= max_pages:
            break
    return rows, pages


# Candidate Central Basin Platform sub-areas (lon_min, lat_min, lon_max, lat_max), WGS84.
CBP_SUBAREAS = {
    "andrews_co":  (-103.16, 31.98, -102.56, 32.52),   # Andrews County (core CBP)
    "ector_co":    (-102.90, 31.58, -102.30, 32.09),   # Ector County
    "crane_co":    (-102.55, 31.13, -102.07, 31.67),   # Crane County
    "demo_small":  (-103.02, 32.20, -102.86, 32.34),   # small Andrews sub-box for fast proof-pull
}


def demo():
    print("[RRC GIS ingestion — authoritative source, source-verified]\n")
    box = CBP_SUBAREAS["demo_small"]
    print("demo sub-box (Andrews Co. core):", box)
    c1 = count(1, box); c9 = count(9, box)
    print("  layer1 Well Locations:        %6d" % c1)
    print("  layer9 Horiz/Dir surface:     %6d  (horizontal/directional)" % c9)
    print("  => vertical-ish (1 - 9):      %6d  (%.1f%% vertical)\n" % (c1 - c9, 100 * (c1 - c9) / max(c1, 1)))

    rows, pages = fetch(1, box, "API,GIS_API5,GIS_LAT83,GIS_LONG83,GIS_SYMBOL_DESCRIPTION")
    print("  pulled %d well records in %d page(s)" % (len(rows), pages))
    hz, _ = fetch(9, box, "API")
    hzset = {r.get("API") for r in hz}
    vert = [r for r in rows if r.get("API") not in hzset]
    print("  vertical after layer-9 join:  %6d" % len(vert))
    print("\n  sample vertical wells:")
    print("    %-16s %-10s %-11s %s" % ("API", "LAT83", "LONG83", "SYMBOL"))
    for r in vert[:5]:
        print("    %-16s %-10s %-11s %s" % (r.get("API"), r.get("GIS_LAT83"), r.get("GIS_LONG83"),
                                            r.get("GIS_SYMBOL_DESCRIPTION")))
    sym = {}
    for r in rows:
        s = r.get("GIS_SYMBOL_DESCRIPTION"); sym[s] = sym.get(s, 0) + 1
    print("\n  GIS_SYMBOL_DESCRIPTION distribution (well status/type):")
    for s, n in sorted(sym.items(), key=lambda kv: -kv[1])[:12]:
        print("    %5d  %s" % (n, s))
    return rows, vert


if __name__ == "__main__":
    demo()
