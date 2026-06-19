# MCP Server Spec v0.1 — Drilling-Ops / MWD Advisory
*The first candidate from `MCP_OPERATING_MANUAL.md`, specced end-to-end through the defensibility filter.*

## The wedge
A model-agnostic MCP server that exposes an operator's **own** proprietary drilling history — MWD/LWD telemetry,
offset-well performance, completions, production — as **verified, calibrated advisory tools**, behind a
human-in-the-loop gate. Commodity model rented; the operator's data + the verified loop owned behind the socket.

## Defensibility filter (T1–T4)
- **T1 Next-Quarter — PASS (conditional).** A frontier lab can't build *this operator's* offset map, telemetry
  calibration, or completion outcomes — it lacks the data and the field relationships. PASS **only** when bound to
  proprietary data; a generic-public version FAILS (that is Enverus, and they win it).
- **T2 Membrane — PASS.** Value lives behind the socket: the operator's data + the reconstruction-validated loop +
  the calibrated error bars. The wrapper (MCP plumbing, the LLM) is commodity. ✓
- **T3 Sparse-Content — PASS.** It serves what *diverges from* the public canon — this operator's wells, not the
  textbook. ✓
- **T4 Insurability / HITL — PASS by design.** Every output is an advisory carrying a calibrated probability + CI +
  the evidence it stands on; a licensed driller/geologist signs; no autonomous control of the bit. The honest error
  bar + audit trail is what makes a wrong call survivable → insurable. ✓
- **Verdict: SHIP** — `T1 ∧ (T2∨T3) ∧ T4` all pass — *iff* bound to proprietary data and HITL-gated.

## Tools the server exposes (each = the verified loop on proprietary data)
1. `offset_analog(location, target_zone)` → predicted productivity (the reconstruction-validated future mapper;
   ~+0.21-class on public data, sharper on the operator's denser proprietary set) **+ CI + the offset wells used.**
2. `decline_forecast(well | lease)` → Arps type-curve + EUR **with uncertainty**, from clean (ramp- and
   rework-stripped) series.
3. `mwd_advisory(realtime_telemetry)` → interpretation (formation, dogleg, vibration risk) with **calibrated
   confidence**; explicitly flags the *danger zone* — where confident-genericness would mislead.
4. `completion_qc(perf_plan, logs)` → net-pay / placement check against the operator's analog outcomes.

Every tool returns four things: the number, its **CI**, the **evidence** it stands on, and an explicit
**"needs human sign-off"** flag. No tool returns a bare point estimate.

## Rented vs owned
- **RENT (never own):** the frontier model, the MCP protocol, the standard methods (offset analysis, decline curves, kriging).
- **OWN (the moat):** the operator's proprietary telemetry / logs / completions / production; the verify-everything
  loop; the calibrated error bars; the field relationships and the regulatory/liability posture.

## Honest scope — the product is the trust, not the oracle
Public-data skill tops at the irreducible floor (measured: ~+0.21 reconstruction). Proprietary density raises it —
not to omniscience. The differentiator is **not** superhuman resolution; it is that the advisory **ships its own
uncertainty and accepts its own nulls**, in a domain where a confident-wrong number is a blowout or a dry hole.
A frontier model hallucinates a casing pressure; this server returns the verified number with its CI — or says it
doesn't know. **That** is the product, and it is exactly what an Enverus dashboard and a raw LLM both refuse to do.

## Build path
Thin MCP wrapper (days) over the **already-built** verified pipeline (`cbp_wolfcamp/`), re-pointed from public RRC
to the operator's data. The hard part is done; the wrapper is commodity; the moat is the data × the loop. Ship the
trust, gate the human, never bet past the error bar.
