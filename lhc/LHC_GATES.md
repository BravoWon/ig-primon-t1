# LHC gates — "chaos / max-entropy / meaning / primes-zeta" made falsifiable. Executed 2026-07-02.

*The worldview under test (user's): chaos = language fully randomized (max entropy); meaning = what
observation assigns; the encoding = primes / zeta zeros / the critical line. Distilled per this program's
standing protocol: quarantine the metaphysics, keep the operational kernel, pre-register the falsifiers.*

**Data:** CERN ColliderML-Release-1 (`ttbar_pu0` truth particles) — **SIMULATED** 14 TeV pp
(MadGraph→Pythia→Geant4), flagged as such; 250 events, hard-scatter primaries (pt>0.5, |η|<4), median
215 particles/event. Cached in `events_ttbar_pu0.npz`; first 2000 true zeta zeros via mpmath in
`zeta_zeros.npy`.

## Gate L1 — "collision chaos is max entropy": FALSIFIED (as expected; the value is the ladder)
| measure | REAL | angular-scramble | pooled-marginal |
|---|---|---|---|
| mean nn-ΔR (clustering/jets) | **0.123** | 0.200 | 0.234 |
| zlib ratio (compressibility) | **0.792** | 0.810 | 0.816 |
| pT-balance (conservation) | 0.152 | 0.134 | 0.155 |

- **Clustering is the dominant measurable "meaning"** — jets pull nn-ΔR ~40% below chaos. Compressibility
  concurs. "Observation assigns value" survives only in its operational form: *meaning = the structure an
  observer's model compresses away from the max-ent baseline* — measured, no consciousness premise needed.
- **Honest casualty:** the conservation rung is INVALID under our selection (pt/η cuts + neutrinos carry
  the balance); conservation must be tested on the uncut final state. Design error, flagged.

## Gate L2 — the spectral/zeta gate: instrument arc + verdicts
- **v1 (`gate_L1_L2.py`): H-bulk was a NON-measurement.** Polynomial unfolding choked on the kernel
  spectra's near-zero eigenvalue pile (tell: mean unfolded spacing 1.564 ≠ 1.000). The KS distances and
  number-variance from v1 are unfolding artifacts — recorded, not interpreted. The **sanity anchor inside
  v1 passed**: true zeta zeros vs GUE, KS = 0.045 (Montgomery–Odlyzko reproduced) — the machinery is
  sound where the spectrum is well-behaved.
- **v2 (`gate_L2_v2_rstat.py`): repaired via r-statistics** (adjacent-spacing ratios; unfolding-free;
  exact references Poisson .3863 / GOE .5307 / GUE .5996; anchors reproduced to ±0.003; zeta → 0.617 ≈ GUE).

| ensemble | ⟨r⟩ | class |
|---|---|---|
| surrogate (scrambled chaos) | 0.518 | **GOE (β=1)** |
| REAL collider bulk | 0.4925 | GOE pulled **Poisson-ward** |
| zeta zeros | 0.617 | **GUE (β=2)** |

**Verdicts (pre-registered):**
1. **H-bulk PASS (repaired):** randomized collider "chaos" is RMT-universal — but **GOE-class, not the
   zeta/GUE class**. Time-reversal-symmetric real kernels were expected to be β=1; they are.
2. **H-meaning (reframed from H-spike):** the REAL bulk deviates from GOE *toward Poisson* — jet
   clustering creates independent localized modes with weaker level repulsion. **The meaning is literally
   the measured deviation from universality.** (v1's spike count was kernel-bandwidth-limited; the
   deviation shows up in the bulk statistic instead.)
3. **H-zeta: doubly NULL.** (i) wrong symmetry class (β=1 vs zeta's β=2); (ii) no prime-specific
   long-range signature. "Collision data is encoded by the primes/zeta zeros" is dead by its
   pre-registered falsifier. What survives is real but generic: *chaos in both collider kernels and the
   critical line belongs to random-matrix universality* — kinship of class, not encoding by primes.

## The honest residue of the worldview
- "Chaos = max entropy" → falsified in detail, and *that's the productive part*: the structure is
  localized, rankable, measurable (jets ≫ compressibility; conservation needs uncut data).
- "Meaning = assigned by observation" → survives as compression-gain / deviation-from-universality.
  Quarantined: any consciousness premise (can't fail ⇒ can't carry weight).
- "Encoded by primes/zeta" → **null twice**, with the instrument validated on the true zeros in the same
  run. The genuine zeta connection available for development is the **critical-line thread already in
  this repo** (Keiper–Li / H13a-b, `New folder (2)`) — Gate L3, where the primes actually live.

## Reproduce
`gate_L1_L2.py` (data pull + L1 + L2-v1 incl. the failed unfolding, kept for the record) →
`gate_L2_v2_rstat.py` (the repaired H-bulk). Figure: `gate_L1_L2.png`.
