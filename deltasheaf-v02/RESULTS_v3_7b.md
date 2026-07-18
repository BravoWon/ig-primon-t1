# v0.3 RESOLUTION — 7B reader, checkpointed, control-first (2026-07-18)

**Protocol:** pre-registered control-first: easy items closed→open, abort if Δ<−3%. First pass n=150 fired
the abort (Δ−4.7%) but McNemar p=0.19 → pre-registered extension to n=400 (rule fixed before relaunch).

**Control (n=400): PASSED — reader VALID.** closed 88.0% → open 86.0% (Δ−2.0%; 23 harmed / 15 helped;
McNemar p=0.256). The 7B does not meaningfully degrade on context; the 3B's −7.3% degradation does not
persist at 7B; the n=150 abort was noise.

**Gate (n=322 blind spots, chance 25%): PERFECT NULL.** closed 25.8% → open 25.8% (Δ+0.0%; 29 harmed /
29 helped; McNemar p=1.000). Split: passage-present (196) open 23.0%; passage-absent (126) 30.2% — the
passages do not help even when present.

**VERDICT: retrieval-at-inference does NOT recover the scale-invariant blind spots — the import law is
REFUTED on this instrument (validated reader, symmetric null).** Named caveat: passage RELEVANCE was never
well measured (gold-substring proxy ~2/322), so oracle-relevant import remains untested; what is refuted is
the real pipeline (Wikipedia top-hit retrieval → context). The program survivor GROUNDING is untouched —
import pays when baked into representations at training time (terminus), not when appended to context.

*(Files: openbook7b.py; data/raw/qwen7b_easy_closed.jsonl, qwen7b_easy_open.jsonl, qwen7b_gate_open.jsonl,
qwen7b_gate.jsonl [closed]. ~2.3h total on the RTX 5070 under CPU offload, per-item checkpointed.)*
