# DeltaSheaf — Conclusion (PROVISIONAL — v0.3 retrieval test under repair), 2026-07-17

**Thesis tested:** "the delta is the map" / "the boundary layer maps the manifold holes" — inter-model
disagreement geometry carries recoverable information about the gold answer on items where an ensemble is
jointly wrong. Refined law under test: *import from a complementary system recovers the hole*.

## What is banked (trustworthy)

**1. v0.2 — sheaf geometry over an LLM ensemble: FALSIFIED (solid).** 5 families (26M–7B), 322 blind-spot
items (0-of-5), ridge-LS restriction maps, cosine readout, 5 seeds. All arms at chance — **map** (`B_edge`),
**displacement** (`C_cycle`), **area** (`C_area`), **volume** (`ctrl_mag`); under MLP, `C_cycle` loses to
its own shuffle. Instrument validated on synthetic (detects a planted map); round-trip: pipeline recovers on
clean (0.37) but not blind-spot (0.21). A genuine null: intra-ensemble disagreement geometry carries no
recoverable pointer to the missed answer.

**2. Strong-model premise: scale-invariant blind spots (solid).** Qwen2.5-7B (~70% full MMLU) on the same
322 = **25.8% = chance**. A bigger *same-kind* model shares the hole; scaling the LLM imports nothing.

## What is NOT yet decided (the v0.3 retrieval-import test — under repair)

**3. Retrieval-import test: INCONCLUSIVE — the reader was broken.** Swapped in a Wikipedia passage (73%
non-empty). Sheaf embedding-cosine readout: null (degenerate — a topic vector can't pick the gold option).
Open-book readout: appeared to *hurt* (closed 20.2% → open 17.4%; passage-present 17.9%→12.2%).
**But the driftwave round-trip VOIDED that null:** open-book hurts **EASY** items too (77.3% → 70.0%, Δ −7.3%)
— a *larger* drop than on the hard items — so the effect is the **open-book prompt degrading the 3B reader
generally** (small model distracted by long, partially-relevant context), NOT "retrieval can't fill the
hole." **Recurse (fix the reader):** a use-context-only-if-relevant prompt did NOT fix the 3B's
context-degradation (−6.0% on easy); the 3B fundamentally can't hold context without losing accuracy. The
valid reader (Qwen-7B) is **prohibitively slow here** — ~15 s/item under CPU offload ≈ 4 h, exceeding the
~1 h background-window cap, needing a checkpointed multi-window run. **Verdict: the retrieval-import test is
INCONCLUSIVE — reader-instrument-limited (no fast valid reader in this environment), NOT negative.**
Retrieval answer-relevance also unresolved (exact-substring proxy ~2/322, but it excludes short answers and
is unreliable). A clean verdict needs a checkpointed 7B run (or a context-robust reader that fits the GPU)
and a real relevance measure.

**Retracted:** the earlier "reasoning-holes aren't importable" synthesis — it rested on the void open-book
null. Do not cite it. The import question is **undecided**, not answered.

## Instrument discipline (FOUR near-misses caught before believing a false result)
1. Degenerate substrate (R_ij=identity → zero residuals) → used ridge-LS maps.
2. Empty retrieval (throttled to 1%) → caught by non-empty diagnostic; fixed to 73%.
3. Degenerate cosine reader → moved to open-book.
4. **Degrading open-book reader (−7.3% on EASY items) → voided the v0.3 null.** This one would have banked a
   confident *wrong story* ("import can't fill reasoning-holes"). The round-trip control caught it.

## Standing position
Banked: intra-ensemble geometry and bigger-same-kind models are null (geometry/scale is not the lever).
The one survivor remains **grounding (information-hole import)**. Whether *retrieval* import recovers these
blind spots is **undecided** — the honest verdict awaits the fixed reader. **Do not bank a DeltaSheaf verdict
on the import question until then.**

*(Reproduce: `build.py`[+mlp] · ensemble `score/embed` · `retrieve.py`/`build_v3.py`/`openbook.py` ·
round-trip control `easy_check.py`. Results: `RESULTS*.md`.)*
