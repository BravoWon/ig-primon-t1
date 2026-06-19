# MCP Server Spec v0.1 — Oilfield-Legal / MSA Liability
*Second candidate from `MCP_OPERATING_MANUAL.md`, specced through the defensibility filter — to show the filter generalizes.*

## The wedge
A model-agnostic MCP server over a **proprietary corpus of negotiated oilfield Master Service Agreements
(MSAs) + their dispute / payout outcomes**, exposing clause-risk and liability-exposure tools (knock-for-knock
indemnity, gross-negligence carve-outs, insurance triggers) — with a **licensed attorney in the loop.** The
moat isn't contract-reading (every legal-AI does that); it's *which clauses actually cost money*, learned from
outcomes only this firm/operator holds.

## Defensibility filter (T1–T4)
- **T1 Next-Quarter — PASS (conditional).** A frontier lab reproduces generic contract analysis for free.
  It cannot reproduce *your negotiated positions + dispute outcomes* — what bit, where, for how much. PASS **only**
  when bound to the proprietary corpus *with outcome data*; a generic clause-summarizer FAILS.
- **T2 Membrane — PASS.** Value is behind the socket: the corpus + the outcome ledger + counsel judgment. The
  wrapper (LLM, MCP plumbing) is commodity. ✓
- **T3 Sparse-Content — PASS.** It serves what *diverges from* the public canon — this operator's risk allocation
  and dispute history, not "what is knock-for-knock." ✓
- **T4 Insurability / HITL — PASS, and here it's *mandatory*.** A licensed attorney signs every output; the server
  drafts/flags, never advises. HITL is both the insurability gate (malpractice) **and** a regulatory requirement
  (unauthorized practice of law). ✓
- **Verdict: SHIP** — *iff* proprietary corpus **with dispute-outcome data** + counsel-in-the-loop. Without the
  outcome ledger it's a commodity summarizer in an oilfield costume — **FAIL**.

## Tools the server exposes
1. `clause_risk(clause)` → risk rating + **the analog prior disputes it resembles + their outcomes** (the moat).
2. `indemnity_check(MSA)` → knock-for-knock / carve-out / insurance-trigger flags **vs the operator's negotiated baseline.**
3. `exposure_estimate(MSA, scenario)` → liability **$ range with CI**, from analog payout history.
4. `redline_suggest(draft)` → positions drawn from the proprietary corpus, ranked by outcome track record.
Every output carries: the call, its evidence (the analog matters), a confidence band, and an
**"attorney must sign"** flag. No output is legal advice; every output is decision support *for* counsel.

## Rented vs owned
- **RENT:** the frontier model, the MCP protocol, generic contract/NLP capability.
- **OWN (the moat):** the negotiated-MSA corpus, the **dispute-outcome ledger**, counsel relationships, the
  verification harness, the regulatory/malpractice posture.

## Honest scope — the product is calibrated precedent + a licensed signature
Generic legal-AI is commodity and getting cheaper. The differentiator is the **outcome-grounded** read —
"this carve-out cost $X across N analog disputes" — plus a licensed human who stakes their name. A raw LLM
hallucinates an indemnity reading; this server returns the outcome-grounded estimate with its evidence, or
flags that it has no analog. **That, gated by counsel, is the insurable product.**

## Build path
Same shape as drilling-ops: thin MCP wrapper over the verify-everything loop, pointed at the firm's proprietary
MSA + outcome corpus. The gate (attorney sign-off) is not friction — it is the feature that makes it shippable.
