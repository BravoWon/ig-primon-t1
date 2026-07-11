# PREREG — Gate P4: the Hofstadter butterfly, two-route (FHS topology vs Diophantine arithmetic)
**Registered 2026-07-08 before execution.**

## Substrate
Hofstadter model (square lattice, flux φ = p/q per plaquette, Landau gauge; q×q magnetic Bloch
Hamiltonian). For every internal gap r (r bands below), TKNN: the Hall/Chern number is the unique
integer t_r with **p·t_r ≡ r (mod q), |t_r| ≤ q/2**.

## Two routes (nothing shared)
- **Route A (topology):** non-Abelian FHS — multiband plaquette link determinants over the magnetic BZ
  for the lowest r bands → integer C_r.
- **Route B (arithmetic):** the Diophantine congruence, solved exactly. No spectrum, no wavefunctions —
  modular arithmetic only.

## Kernel slots
- **SWEEP:** all reduced fractions p/q with q ≤ 12 (Farey), all internal gaps r = 1..q−1.
- **ANCHORS:** φ=1/3 gap sequence (t = +1, −1); **sum rule** Σ all-band Chern = 0 at every fraction
  (free anchor per point); q-even central touching (r = q/2) → gapless, nm by construction.
- **NM:** indirect gap < 1e−6 → "not measured" (listed); FHS grid escalation 12→24→48 before nm.
- **WALL:** the fractal gap structure itself — smallest measured gap vs q (the butterfly's
  self-similar shrinkage = the resolution wall any instrument must pay); report min-gap(q) and the
  FHS grid N*(gap) behavior against P1's wall-free expectation.
- **VERDICTS:** (1) C_r(FHS) = t_r(Diophantine) for EVERY measured gap — one receipt per gap;
  (2) anchors + sum rules all pass; (3) wall: min-gap(q) decay measured (fit reported, no [E] pinned).
- **FIGURE:** the butterfly (q ≤ 40 spectra) with two-route-verified gaps marked by Chern color
  (verification only claimed for q ≤ 12; beyond = Diophantine-colored, flagged single-route).

## Honest scope
Hofstadter 1976 / TKNN 1982 / colored butterfly (Osadchy–Avron 2001) — fully occupied territory. The
deliverable is the receipt genre completing its circle: **topology computed numerically, verified
against a number-theoretic congruence, gap by gap** — the same two-route discipline that began with
zeta zeros vs cumulants, now closing on the fractal spectrum where arithmetic IS the physics.

---
## GATE RECORD (2026-07-08, appended post-execution)
- **Verdict 1 PASS — 312/312.** Every measured gap across all 44 reduced fractions q≤12: FHS multiband
  topology = Diophantine congruence exactly. Zero disagreements; 16 gapless points nm by construction.
  Sign convention (BZ loop orientation) fixed once at the φ=1/3 anchor and recorded — after which the
  entire ±{...} pattern structure (e.g. (−2,1,−1,2) at 2/5) matched with no further freedom.
- **Verdict 2 PASS** — anchor sequence [1,−1] at 1/3; sum rules embedded in grid escalation.
- **Verdict 3 — the wall is PARITY-SPLIT (measured; the single-exponential fit R²=0.506 is the wrong
  model and the residual structure is the finding):** odd-q minimal gaps shrink geometrically,
  min-gap ≈ e^{−1.07 q} (q=5,7,9,11: 0.157/0.020/0.0023/0.00025 — near-perfect), while even-q retains
  wide gaps (central touching reorganizes the hierarchy). The fractal's resolution price curve is
  itself number-theoretic. [Exploratory post-hoc observation, flagged as such — a pre-registered
  parity-split wall fit is the natural P4.1 if pursued.]
