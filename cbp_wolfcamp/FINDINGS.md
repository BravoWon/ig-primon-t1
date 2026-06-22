# CBP / KGS Carbonate-Productivity Expedition — Consolidated Findings

*What predicts oil productivity in Central-Kansas-Uplift carbonate wells, what doesn't, and the one
drillable output that survived every out-of-sample test. Every claim below is gated by a leave-one-
county-out (LOCO) rank-IC; scripts in this directory reproduce each number.*

## Bottom line (one paragraph)
Rock quality and rock geometry at the wellbore **do not** predict carbonate well productivity in the
CKU public set — three independent nulls plus one refuted artifact. The **only** signal that
generalizes out of sample is **structural/basin position**, a coordinate you read off **free public
structure maps** — the proprietary MWD/log adds nothing it doesn't already give. That edge is weak
(LOCO IC ≈ +0.2–0.4), **concentrated on the northern shelf**, and best deployed not as a predictor
but as a **search-priority ranker for open step-out acreage**. The expensive elaborations (gradient
boosting, topology, regime-routing, pure active learning) were all honestly **held** by the
out-of-sample gate; the durable product is a single coordinate plus a UCB siting rule.

## What does NOT predict productivity (the nulls)
| Lead | Test | Result | Script |
|---|---|---|---|
| Structure (tight Wolfcamp, TX) | RRC join | real geology, doesn't pay (*structure-real ≠ structure-pays*) | (prior) |
| Wellbore rock quality φ·h | KGS powered, n=246 | **ρ = −0.02** (null) | (prior) |
| Lansing-KC isopach "thinner pays" | wringer, n=894 cleaned | **artifact** — combined-marker pick (median 0 ft); cleaned −0.04, p=0.21 | `kgs_isopach_wringer.py` |

The isopach lead is the cautionary tale: a raw ρ=−0.25 *** that was entirely a degenerate-pick
artifact. Always wring a tops correlation (clean degenerate picks; test all intervals for a generic
depth confound) before believing it.

## What DOES predict it (the one survivor)
**Structural / basin-accommodation position → productivity.** Quantified in `kgs_basin_position.py`:
- Per-county-detrended section thickness → log best12: ρ = **+0.243 ***
- **LOCO out-of-sample: mean rank-IC +0.22, 7/7 counties positive**; random holdout +0.27.
- Collinear axis: thickness ↔ structure ↔ depth (|r| 0.45–0.77) — one structural surface; the data
  cannot say which proxy is "the" cause. Mechanism is *map position*, not wellbore rock.
- Reconciles (doesn't break) the structure null: this is the deeper **conventional** Mississippian-
  bearing subset (median KC 3,937 ft) where structural trapping operates.

**It is regime-specific** (`regime_route.py`): an unsupervised 2-regime split =
- **North shelf** (NESS/TREGO/STAFFORD; KC 3,821 ft; med best12 **2,966 bbl**) — edge **IC +0.41**.
- **South flank** (BARBER/KIOWA/COMANCHE; KC 4,342 ft; med best12 **2,046 bbl**) — edge **IC +0.16**.

High-grade aggressively on the shelf; the flank is marginal.

## The method (value-geometry kernel) and its discipline
`value_geometry.py` implements the IGVF value-geometry kernel (see `../IGVF_VALUE_ALGORITHM.md`):
embed → fit value field → warp metric by value → persist robust structure → find nodal boundaries →
rank → **VERIFY out-of-sample**. The recurring result is the point: **every elaboration was HELD by
the VERIFY gate.**
- 23-feature gradient-boosted value field: OOS IC +0.198 < single coordinate +0.21–0.30 → **HOLD**.
- Regime-routed models: pooled +0.246 = routed-soft +0.246 > per-regime +0.195 → **HOLD** (routing
  is *diagnostic*, not predictive).
- Pure info-gain active learning: ties random siting (field too 1-D to learn) → no edge.

The kernel's real gift is the gate that keeps refusing to let complexity masquerade as value.

## The one drillable output
`slice_acquisition.py` (SLICE prime, `driftwave-verifier` PASS 0.98, leakage-free backtest):
**UCB siting (`v̂ + σ`) on the shelf discovers +53% more productive wells than random** (4,500 vs
2,947 bbl mean). `slice_sanity.py` confirms the recommendations are **genuine step-outs** (only 2/10
in saturated acreage; the σ term steers toward open cells). Top site: **38.989°N, −100.007°W (NESS
county)** — `v̂≈8,000 bbl` model estimate, 1 well <3 km, nearest producer 4 km.
**Caveat:** `v̂` is a weak one-axis field and UCB reports an optimistic mean+σ — this ranks *where to
look first*, not guaranteed barrels.

## Reproduce
```
python kgs_isopach_wringer.py     # the refuted artifact + the wringer discipline
python kgs_basin_position.py      # the survivor, confound-controlled + LOCO holdout
python regime_route.py            # the shelf/flank 2-regime split
python value_geometry.py          # full kernel; VERIFY gate -> HOLD
python slice_acquisition.py       # SLICE/UCB next-well siting; VERIFY gate -> SHIP
python slice_sanity.py            # drillability vs real 23,903-well density field
```
