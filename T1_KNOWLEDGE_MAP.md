# IG-PRIMON-T1 — Standing Knowledge Map

*Resume from truth, not in-session enthusiasm. Every claim here traces to a pre-registered amendment in
`T1_precision_map_v0_2.md` (v0.2.1–v0.2.17) and a banked receipt (`results_*.txt`). Numbers are held-out,
PPL-based unless noted. This is a synthesis; the amendments are the source of record.*

---

## Two arcs

1. **Precision/quantization arc** (the program's core) — *Does a certified reference buy deployment quality?*
   **No.** The thesis was refuted, the lever relocated to use-aware quantization, and that lever delivered
   genuine value (near-lossless 4-bit). v0.2.1–v0.2.12.
2. **Horizon arc** (the exploratory thread) — a chain of intuitions (eigenweights, primes, Hangul, "what's in
   between") run to ground. Most were dressing that fell off under test; one survived and was measured to its
   true (modest) size. v0.2.13–v0.2.17.

---

## Precision arc — established facts

- **Certification is operationally refuted** `[V-hw]` (v0.2.7, v0.2.10). Certified-reference *allocation*
  minimizes KL-from-fp32 — the wrong objective for deployment; at deployable budgets it is no better, often worse,
  than a cheap bf16-reference allocation. The **certified vs cheap Hessian** question (the one genuinely-novel
  program question) is a **clean null**: −0.51% PPL, 95% CI [−1.30%, +0.30%], predicted by a control showing the
  Hessian shifts only ~1e-4 under bf16 rounding. Certification's only defensible home is *faithful replication of
  a reference*, not "deploy a better model."
- **The lever is use-aware quantization** (the activation Hessian / GPTQ). Validated layer-level (GPTQ ~7× RTN)
  and end-to-end. Best configs:
  - **FP4-g128 + act-order = +0.70% PPL (near-lossless 4-bit), 3.88× smaller than fp16.** Best weight-only.
  - **W4A8 (FP4 weights + per-token FP8 acts) = +1.01%** — deployable, both axes compressed.
- **Two significant refinements** (v0.2.11): **act-order −1.44% PPL**; **FP4 E2M1 + block scaling beats INT4 by
  −1.32% PPL** (nonuniform grid denser near zero — the program's original FP4 direction vindicated as *superior*,
  not merely viable; block scaling was the missing piece). The old +58% FP4-perchan was a rank-deficient Hessian.
- **The activation cliff** (v0.2.11–v0.2.12): FP8 acts ~free; **naive 4-bit acts collapse (+37,332%)** on OPT's
  per-channel outliers; **SmoothQuant is necessary-but-insufficient** for W4A4 (recovery negligible). The 4-bit
  activation frontier needs **rotation** (QuaRot/SpinQuant) — named, not built. A8-smoothing gain was a *certified
  null* (v0.2.12, paired test refuted a cross-file splice the round-trip verifier caught).
- **Hardware value, measured** (v0.2.11): FP8 `_scaled_mm` **2.0–2.24×** vs bf16; 4-bit weights **3.88×** memory.
  Native INT4/FP4 *rowwise GEMM* decode speedup is **unmeasured** — no kernel on this Blackwell/Windows build.

## Horizon arc — established facts

- **Eigenweights** (v0.2.13): weak form **alive** (directions are special: top-k ≫ random-k ~65×; ASVD beats
  plain SVD per-matrix), strong form **dead** (quantization dominates low-rank ~150× at matched storage; uniform
  all-layer low-rank is catastrophic). **Sequential ASVD falsified** the rescue — sequential reconstruction needs
  *small per-step error* (GPTQ clears it, low-rank truncation doesn't).
- **Prime-value grid** (v0.2.14): **null.** {0,1,2,3,5,7,11,13} ≈ FP4 at 4-bit; wins for the generic
  (nonuniform) reason, not primality.
- **Use-aware bit allocation** (v0.2.14): **inert as a gainer.** Sensitivity signal is real and directional
  (reverse worse) but uncashable — a sharp bit-cliff means trading bits across layers only falls off it. Closes
  the allocation thread (use-blind refuted, use-aware inert).
- **Semantic-prime projection** (v0.2.14): spanning **refuted** — NSM primes are not privileged over random words
  (robust across token *and* contextual representations; the wte "55% of optimal" was largely a function-word
  tokenization artifact, contextual is 92% but still tied-below random). Composition (composites→prime atoms)
  above chance but **fragile** (10× wte, 5× contextual, individual words flip).
- **Representation alignment / "the shape in between"** (v0.2.15–v0.2.17) — *the one survivable gem.*
  Independently-trained models (GPT-2-sm/xl, OPT-2.7B) align, but measured to true size: **most of the raw
  alignment was the shared GPT-2 BPE tokenizer**; the genuine semantic convergence beyond vocab is **local
  neighbor-agreement at ~6–10× chance, strongest at the output layer, scaling with capability — not a global
  isomorphism** (residual linear-R² ~0.05–0.1, ≈0 for the small model). U-shaped over depth (shared at
  vocab-surface + task-output, private in the middle); rotation-rigid only at the surface.

---

## Cross-cutting patterns (the real lessons)

1. **Weak-form-alive / strong-form-dead** (≥5 instances: eigenweights, low-rank, semantic primes, the alignment
   gem). Structure is consistently *real and readable* but *not a small efficient/compressible basis*. Meaning is
   distributed. "Project" wins decodability/interpretability; "float" wins efficiency — complementary, not XOR.
2. **Sharp cliffs gate optimization.** The bit-cliff (4→3 bit on GPT-2) makes allocation impossible; below it,
   everything dies the same way. Quantization is the lever precisely because a *good* quantizer keeps you above
   the cliff with no room left for allocation.
3. **Sequential reconstruction needs small per-step error.** It rescued GPTQ (+150%→+58%→+2%) but *not* low-rank
   — because low-rank's per-layer error is catastrophic, so propagation carries garbage.
4. **Representation/metric choice masquerades as a result.** wte→contextual moved prime spanning 55%→92%;
   KL-over-vocab and 128-chunk PPL were hypersensitive; forward-KL ranked 8-bit *worse* than 4-bit. Always check
   the metric/representation before believing the geometry.
5. **Inversions are diagnostics, not fixes.** When a metric points the wrong way (certified→deployment loss,
   KL→PPL), its operational complement *is* the target. When an inversion is a bug (collapsed dict), inverting
   would launder it. Diagnose the cause first.

## Methodology — the meta-asset

The program **refuted its own founding thesis, relocated the lever, validated the method** — and the discipline
caught, symmetrically: false positives (certified edge, A8-smoothing splice), a false negative (+150% parallel
GPTQ), overclaims (a fresh-context round-trip verifier at drift 0.96 flagged a cross-file splice, refuted by a
paired experiment), and its own bugs (collapsed dict 48→4 via non-unique keys, a budget-leaking allocation tuple,
N<d Procrustes overfit, a KV-cache OOM, dtype mismatches). Standing rules that earned their keep: **agreement is
not verification; verify at source; control before scan; derive before numerics; PPL not KL; accept negatives;
read the traceback before theorizing; pre-register hypothesis/control/falsifier; commits/pushes only on explicit
word.**

## Hardware / environment

RTX 5070 (cuda:0, sm_120 Blackwell — FP8 `_scaled_mm` works; FP4 `float4_e2m1fn_x2` dtype present, rowwise GEMM
unsupported) + GTX 1660 Ti (cuda:1, sm_75 — fp32/fp16 annex, no bf16/FP8). `gptqmodel`/`auto-gptq` won't build
(`pypcre`). Models under `C:/Users/JT-DEV1/Documents/`: gpt2-sm, gpt2-xl, opt-2.7b (all GPT-2 BPE vocab). torch
2.11+cu128, transformers 5.12, python 3.11. `expandable_segments` unsupported on this platform.

## Open frontiers (named, not built) — and what each needs

- **Rotation-based W4A4** (QuaRot/SpinQuant Hadamard) — the activation frontier; motivated by the spread
  spectrum (spread outliers across the necessary bulk). Buildable on the current harness.
- **A real INT4/FP4 GEMM kernel** (Marlin/MXFP4) — to convert the 3.88× memory + fidelity into measured decode
  speedup. Needs a kernel this build lacks.
- **Cross-lingual alignment** (the legitimate residue of the Korean idea) — align an English and a multilingual
  model on translation pairs; measure the in-between *without* the shared-vocab confound. Needs a multilingual
  model on disk.
- **A real PRH convergence curve** — more model sizes/families to test whether semantic convergence climbs
  monotonically with capability. Needs more models on disk.

---
*15 commits (df9e417…42f475f) local at time of writing; nothing pushed. This document is a map, not a
registration — amendments in `T1_precision_map_v0_2.md` remain the source of record.*
