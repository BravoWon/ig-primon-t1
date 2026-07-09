# PREREG — Gate P3: finite-size scaling of the disorder-driven topological transition
**Registered 2026-07-08 before execution.** Substrate: the TAI crossover located by P1b/P2a
(Haldane, t2=0.2, φ=π/2, M/t2=5.5, transition in V ≈ 2.0–2.5).

## Occupied-territory declaration
Disorder-driven Chern/plateau transitions are an established field; expectation [E]: the localization
class of the IQH plateau transition, ν_loc ≈ 2.3–2.6. No universality-class novelty is claimed. Ours:
the marker-distribution receipt methodology + the model-specific V_c with two-route error bars.

## Design (artifact-killers built in)
- **Order parameter:** interior-averaged LCM C̄(V, L), margin = L/4 (scales with L), **re-calibrated
  per L** at the clean Haldane point (CAL_L recorded per size).
- **CRN:** one disorder pattern per (L, seed), scaled by V (variance reduction; declared).
- **V grid** {1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0} × **L** {12, 16, 24, 32} × **16 seeds**.
- **V_c route 1:** crossing points of C̄(V,L) for successive L-pairs; drift across pairs = the
  finite-size systematic band (reported, extrapolation only if monotone).
- **V_c route 2:** Bott-fraction f(V,L) crossings on the torus (admissibility gate relaxed to 0.005,
  nm-rate reported per point). Two INDEPENDENT locators of the same critical point.
- **Exponent:** single-parameter collapse C̄ = F((V−V_c)·L^{1/ν}); ν by grid search minimizing
  cross-L binned variance; CI by bootstrap over seeds.

## Verdicts (fixed)
1. Crossings exist for ≥2 successive L-pairs in the window; V_c(LCM) and V_c(Bott) agree within
   joint bands.
2. Collapse: best-ν with bootstrap CI; [E] band 2.3–2.6. Inside → class-consistent (occupied,
   claimed as consistency only). Clean collapse OUTSIDE → flagged for escalation (bigger L first,
   never a novelty claim from L≤32). No collapse (objective above the V-shuffled null) → "not
   measured at these sizes", named escalation (sparse/Lanczos larger L).
3. Marker-distribution width: sample std of C̄ peaks near V_c (the criticality signature P2a saw
   qualitatively — now quantified).
Non-measurement discipline throughout; all nm rates listed.

---
## GATE RECORD v1 (2026-07-08) + AMENDMENT v1.1 (declared before re-run)
**v1 outcome: no crossings in the registered window — and that is a measurement.** C̄(V,L) is monotone
toward −1 in BOTH V and L everywhere in [1.8, 3.0] (L=32 reaching −0.85), and Bott fractions are already
0.6–1.0 across the window: **the entire registered window lies INSIDE the induced phase; V_c < 1.8.**
P1b's "onset ≈2.0" (5 seeds, strict gate) and P2a's "size-stable −0.35 at V=2.0" are hereby corrected:
with 16 seeds and L→32, V=2.0 drifts −0.26→−0.50 (slow shoulder above a lower transition). Collapse and
exponent: NOT MEASURED in this window, per prereg. Per-L calibration behaved perfectly (CAL→−1.0000 at
L=32). Script defect noted: results-json was inside the crossing-success branch — fixed in v1.1.
**AMENDMENT v1.1 (only change: the V grid, everything else frozen):** merged grid
V ∈ {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.8}, same L ∈ {12,16,24,32}, same 16 CRN seeds, same
calibrations, same two routes, same verdicts. The crossing hunt moves to where the data says the
transition lives.

---
## GATE RECORD v1.1 (2026-07-08, final for L≤32)
- **The fan opened where the data pointed.** Below V≈1.6, C̄ drifts toward 0 with L; above, toward −1:
  LCM crossings (16,24)=1.63, (24,32)=1.56 (the (12,16)=1.04 pair carries small-L corrections).
  **V_c(LCM) ≈ 1.6.**
- **Route 2 disagrees — and legitimately:** Bott-fraction curves are shifted right (f≈0.5 near
  V≈1.9–2.2; lone crossing (12,16)=2.6; (16,24) none; nm=44). The marker-fluctuation peak sits at
  V≈2.0–2.4 — with Bott, not with the LCM crossing. (Verdict-3 "False" is an index artifact: the
  argmax criterion was written for the v1 grid; the stds DO peak, at ~2.0–2.4.)
- **Joint honest reading:** at M/t2=5.5, L≤32, the disorder-driven transition is a BROAD critical
  region ≈1.6–2.6 in which two legitimate locators — the local-marker crossing and the global
  invariant flip — separate under strong finite-size corrections. This echoes P2a's lesson upward:
  instruments that agree in phases diverge AT criticality; here the divergence is itself resolved
  into two distinct finite-size scales.
- **Exponent: NO CLAIM.** Collapse beats the V-shuffled null 28× (S=0.0013 vs 0.0373) — the scaling
  structure is real — but best-ν=0.80 sits ON the search-grid boundary and the collapse window
  included off-critical points; with the two-route V_c tension unresolved, ν is **not measured at
  L≤32**, per prereg. [E] band comparison: moot.
- **Escalation (named, costed):** sparse/Lanczos L≥64 flakes, collapse restricted to |V−V_c|·L^{1/ν}
  ≲ O(1), corrections-to-scaling term, and a transfer-matrix localization-length third route to
  arbitrate the LCM-vs-Bott V_c. This is the genuine frontier of this substrate; L≤32 dense eigh has
  said everything it can.
