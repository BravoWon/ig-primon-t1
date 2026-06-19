# MCP Server — Operating Manual & Defensibility Filter

*Distilled from the IG-PRIMON-T1 session. The research finding — the quantization stack is
prior art (rotation = QuaRot, clipping/GPTQ/SmoothQuant published); the value was the
verification discipline, not the algorithm — generalized into a business model. This is the
operating manual for it.*

## The one-line thesis
Rent the commodity, own the defensible, connect them with a model-agnostic socket.

- **Commodity (rent, never try to own):** the frontier model, the MCP protocol itself, generic/canonical domain knowledge.
- **Defensible (own):** proprietary data, field relationships, regulatory/liability trust, the verification discipline.
- **MCP's role:** the standard adapter that sells the defensible layer to whoever owns the commodity layer *this year* — so a model vendor can never rug-pull you.

---

## Part 1 — The Defensibility Filter
Run every candidate server (drilling-ops/MPD, oilfield-legal/MSA, calc-engine, …) through all four. Score each PASS/FAIL.

**T1 — Next-Quarter Test.** Would this survive a frontier lab deciding to build it next quarter?
PASS only if backed by data they can't acquire, relationships they can't buy, or liability they won't underwrite. FAIL = thin wrapper; prior art; rug-pullable.

**T2 — Membrane Test.** Where does the value live — in the wrapper code, or in what the wrapper *gates*?
PASS = value is behind the socket (data / judgment). FAIL = value is the wrapper (commodity).

**T3 — Sparse-Content Test.** Is the truth this server serves in the public canon?
PASS = proprietary, or *divergent from* the canon (your wells ≠ the textbook). FAIL = canonical; a frontier model reproduces it for free.

**T4 — Insurability / HITL Test.** In a high-consequence call, is there a human-in-the-loop gate that makes a *wrong* tool-call survivable?
PASS = gated and auditable. FAIL = no gate → not shippable in any liability-sensitive domain, regardless of T1–T3.

**Verdict rule:** ship only if **T1 ∧ (T2 ∨ T3) ∧ T4**.
A server that passes only on its wrapper is commodity; a server with no HITL gate is uninsurable. Both are no-go even if otherwise clever.

---

## Part 2 — The Operating Manual (how the partnership runs)

- **Division of labor.** The AI brings *form* (methods, derivation shapes, argument/code structure) and *dense-canon synthesis*. You bring *sparse proprietary content* and *validation*. Don't pay the AI to mint novel algorithm IP — there's none to be had; it will re-derive the legible frontier, as it did all session.

- **Novelty lives in the loop, not in either party.** Validated-new truth is a property of the *coupling* — a proposal meeting ground contact fast enough that wrong-but-elegant dies fast. Slow loop → no novelty, just fluent consensus. Keep the loop tight.

- **Aim validation at the danger zone.** Not where the AI is obviously ignorant (it flags that — you're safe there). The costly failure is **confident genericness over a domain with proprietary exceptions**: the textbook answer, fluent, *wrong for your wells*, passing casual review *because* it's the consensus answer. Concentrate review there.

- **No epistemic exemption.** The AI's read on its own edges is itself a synthesis output, weighted toward the plausible-sounding. Validate its self-assessment like any other output.

- **The forward bet to watch.** On your proprietary domain, the AI's *confidence* and its *correctness* will decouple. If they ever start tracking — if it hedges reliably right where it's about to be wrong — something better than pattern-synthesis is happening, and this manual needs an update.

---

---

## Part 3 — Field test (what one full run actually proved)

The thesis above stopped being theory. We ran a complete drilling-analytics expedition on **public** RRC data
(GIS 1.39M wells, EBCDIC wellbore + W-10, 12.7 GB PDQ production, dlisio LWD) → a reconstruction-validated
"future mapper," plus a frontier-method transposition (IG-PRIMON ↔ singular learning theory). **12 pre-registered
verdicts (v0.1–v1.2).** The thesis came back **confirmed, with receipts.**

**Confirmed RULED OUT (empirically, not by opinion):**
- **Output is commodity.** The validated future mapper (Spearman **+0.21**, offset-driven) is exactly what
  Enverus / Novi / TGS already sell — and we *measured* that the ceiling is the rock's irreducible variance,
  which they already sit on. Public-data analytics **FAILS** the Next-Quarter test (T1).
- **No algorithm moat.** Structure-mapping, decline curves, kriging cross-validation, even the SLT/RLCT
  machinery — all prior art. Re-derived, never owned, exactly as predicted.
- **Abstraction ≠ resolution.** Tested a dozen ways (tensor SVD, CCA, dimensional stacking, sheaf framing):
  sophistication added **no** out-of-sample lift over the simple offset map. Abstraction buys SPEED + FLEXIBILITY,
  not precision beyond ground truth. Measured, not asserted.

**Confirmed DEFENSIBLE:**
- **The verified loop is the asset** — zero to a reconstruction-validated pipeline in an afternoon, that
  *accepts its own nulls and ships honest error bars*. The trustworthy "no" is the rare product.
- **Frontier-transposition velocity is the sharpest version** — took a 2025 SLT result (Watanabe's RLCT) into a
  verified, computed tool in three scripts, solo, in minutes. Most operators can't turn an arXiv paper into a
  validated tool on their data in a quarter. This is the capability that doesn't commoditize fast.
- **The moat is the coupling** — your proprietary data × the rigorous loop, in a trust-prizing domain. The
  drilling-ops / MWD wedge **PASSES** the filter (proprietary data + relationships + HITL + insurability)
  exactly where public-data analytics fails it.

**Updated operating rules:**
1. Sell the **verified loop**, not the map. The map is the demo; the loop + the honest error bar is the product.
2. Wedge where you hold the data AND trust *is* the product (drilling/MWD instrumentation — the TTDownhole world).
3. The **research is the credential.** A frontier-transposition paper (IG-PRIMON ↔ SLT) is the proof-of-capability
   that sells the velocity pitch — the demonstration of rigor *is* the sales asset.
4. Guardrails: never sell "resolution from abstraction" (disproved); never fight Enverus on public data (they win);
   the moat is the hard-to-**assemble** combination — domain expert + proprietary data + verify-everything loop +
   willingness to ship the null — defensible because almost no one has all four.

*Receipts: `CBP_WOLFCAMP_STRUCTURAL_EDGE_prereg_v0_1.md` (v0.1–v1.2), `cbp_wolfcamp/` pipeline, `primon_*_bridge.py`.*

---

*House rule, ported: "agreement is not verification" (research) = "the protocol is not the product" (business).*
