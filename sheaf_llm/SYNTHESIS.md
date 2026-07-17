# The Sheaf, Honestly — what fourteen gates established

The "Geometric Determinism / Prime Crystal" documents claim sheaf-theoretic geometry transforms LLMs.
We tested that claim **gate by gate, falsifiably, on a 6 GB GPU**, stripping the mythology and keeping
only what survived a curve that moved the right way. The verdict is one sentence:

> **Structure pays for MEANING, not for BYTES. The sheaf is a geometry of relationships/value/meaning —
> never a compressor — and the right way to get the dimensional structure is to GROUND it (inherit a
> dictionary, as humans do), not to discover it from scratch.**

Two arcs, fourteen gates, every claim with a number behind it.

---

## Arc 1 — BYTES (compression). 10 gates. Sheaf-as-compressor: REFUTED.

| gate | test | result |
|---|---|---|
| v0–v1 | factor pretrained weights (low-rank / block / residual) | **dead** — 4-bit beats it ~3–5×; weights are full-rank |
| v2–v4 | use-aware bit *allocation*, then λ₁/sheaf saliency | allocation only beat *weak* RTN; on strong group-wise 4-bit it **loses**; sheaf-spectral λ₁ ≈ AWQ diagonal (**no gain**) |
| v5 | native low-rank structure (trained from scratch) | **loses** to dense at equal params |
| v8 | Monarch/butterfly | **ties** dense (equivalent reparam, no free lunch) |
| v8/v10/v11 | native ternary / prime {0,1,2} / {−1,0,1,3} | **near-lossless** at ~1.58–2 b/w — *native low-bit is the lever, not the sheaf* |
| v9 | one-object-branched (shared base + restriction-map branches) | **win**: dense quality at ~25% of MLP params (sharing, not sheaf-magic) |
| v6/v7 | MoE | real at scale; **unprovable** on a toy (E-sweep showed the toy win was noise) |

**Buildable small-footprint design** (`DESIGN.md`): one-object-branched backbone × native prime/ternary
low-bit (dial: 1.58 b ternary ↔ 2 b `{−1,0,1,3}`) × MoE-at-scale. **The sheaf's role here is
router/allocator, not compressor.** The "70B in 6 GB by re-representing weights" dream is closed.

---

## Arc 2 — MEANING (compositional generalization). 4 gates. Sheaf-as-meaning: WINS (when grounded).

| gate | test | result |
|---|---|---|
| v1 | dimensional dictionary + relational sheaf vs flat embeddings | **decisive win**: generalizes to novel combinations from ~200 examples; flat sits at *chance* until it memorizes half the space |
| v2 | *discover* the dictionary (single-task codebook) | works, but costs **~4× the data** (~819 vs ~204) and is unstable |
| v3 | FUSE shared cross-task dictionary + sheaf-gluing loss | **did not close the gap**; confounded by a harder architecture |
| v4 | clean isolation: does multi-task sharing help discovery? | **refuted** — multi-task *hurts* (task-conditioning is an escape hatch from clean factorization); the gluing loss was **inert** |

**The reframe the refutation forced:** humans don't discover the dimensional dictionary from scratch —
**they inherit it.** No child rederives the sense inventory of a language; they're handed a dictionary
and recombine over it. So "handed structure" in v1 was never a cheat — **it is the human condition.**
*"Semantically trace how a human does it"* does **not** mean discover meaning tabula rasa; it means
**stand on an inherited dimensional dictionary and compose sheaf relations over it** — which is exactly
the gate that won, overwhelmingly.

---

## The unified takeaway

Point the sheaf at **bytes** → ten negatives. Point it at **meaning** → a decisive win. This is
consistent across the whole repo: the one prior reproduced success was the **value/verification model**
(`sheaf_value_model.py`, +0.314 OOS) — relationships → value, not compression. The sheaf is a geometry
of *relationship, value, and meaning.* And the dimensional structure should be **grounded (inherited),
not discovered** — the discovery-from-scratch detour (v2–v4) was worth running precisely because it
told us not to take it.

---

## The grounded build — brick 1 DONE (confirmed on real language)

**`grounded_gate.py`**: 272 real English nouns, two inherited WordNet tiers (supersense + hypernym
depth), compositional task, **held-out whole words (zero-shot)**. Flat per-word embedding = 1.00 on
seen words but **0.05 (chance) on UNSEEN words**; the WordNet-grounded model = **1.00 on both, at fewer
params.** Inherited dimensional structure generalizes **zero-shot to words never trained on** — the
human move, which flat token-embeddings fundamentally cannot do (the UNK problem). **The "ground, don't
discover" reframe is real on real language.** **Brick 2 (`sentence_gate.py`) — grounding + sheaf relations COMPOSE.** Structured SVO sentences from
real WordNet nouns; a **role-sensitive** target (subject and object count differently → structure must
be respected). `flat` (positional) aces roles but **dies on unseen words (0.12 = chance)**;
`grounded-bag` aces unseen words but **dies on roles (capped ~0.56** — a symmetric bag can't compute an
asymmetric target); **`grounded-sheaf` (inherited dictionary + role-specific restriction maps) gets
BOTH, 1.00 / 1.00.** The dictionary and the sheaf relations are **each necessary and they stack** —
exactly the human-trace architecture (inherit meaning, compose it structurally).
*(Baseline repaired 2026-07-10 after PR review: the original bag truncated away the depth feature,
confounding role-blindness with information loss. Re-run with an information-preserving symmetric
map (`Rb(s+o)`): the cap is unchanged — 0.55 seen / 0.57 unseen — so the role-blind ceiling, not the
missing feature, was binding. The brick-2 conclusion survives its corrected control.)*
*(Deep-sweep audit 2026-07-11, PR #13: recursion_gate re-run with per-arm shared data streams —
conclusion unchanged (flat unseen 0.11≈chance, both grounded arms 1.00 at extrapolation depths);
semantic_gate re-run with the every-token-seen invariant ENFORCED and a data-conditional verdict —
conclusion unchanged and strengthened: relational wins 0.93-vs-0.01 OOS at 5% data with FEWER
parameters than flat (36.1k vs 39.2k), so the capacity confound ran against the win. Three gates
now re-verified under corrected controls; three conclusions intact.)*

**Brick 2 HARDENED (`adversarial_gate.py`) — the 1.00/1.00 was partly by construction; the win still
survives.** Brick 2's target was a clean function of *exactly* the two inherited tiers, so a skeptic
rightly says it was rigged in grounding's favor. The adversarial gate splits the role-sensitive target
into **`y_dict`** (a function of the inherited WordNet tiers — *shareable* meaning) **plus `y_resid`**
(a hidden per-word latent assigned **randomly, independent of (supersense, depth)** — *idiosyncratic*
meaning the dictionary cannot see). Pre-registered hypothesis/control/falsifier. Result (chance 0.12 /
0.20):

| arm | SEEN dict / resid | UNSEEN dict / resid |
|---|---|---|
| flat (token-emb only) | 0.97 / **1.00** | 0.12 / 0.20 |
| grounded-pure (tiers + role maps) | **1.00** / 0.20 | **1.00** / 0.20 |
| grounded+token (hybrid) | **1.00** / **1.00** | **1.00** / 0.20 |

- **The zero-shot win is REAL, not rigged** (falsifier *not* triggered): on the shareable component
  grounded arms hit 1.00 on UNSEEN words where flat is at chance (0.12). The control held — flat aces
  both on SEEN (0.97/1.00), so the residual head is genuinely learnable and grounded-pure's chance on it
  is a true blind spot, not an unlearnable task.
- **But 1.00/1.00 was an artifact of a zero-residual target.** Idiosyncratic per-word meaning is
  **irreducible zero-shot for everyone** (all arms ~chance on UNSEEN resid). No architecture escapes it.
- **Design correction — AUGMENT, don't replace.** Pure grounding is at chance on the residual even for
  *seen* words (it discarded per-word capacity). The honest architecture is **grounded+token**: it
  recovers everything flat does on seen words (1.00/1.00) *and* adds zero-shot dictionary transfer on
  unseen words (dict 1.00 vs flat 0.12) — strictly dominating both. A real grounded LM keeps token
  embeddings for the idiosyncratic residual *on top of* inherited dimensional structure.
- **Refined claim (supersedes "1.00/1.00"):** grounding recovers exactly the *shareable* fraction of
  meaning zero-shot — which is the ceiling — not idiosyncratic meaning. Sharper and more honest.

**Brick 3′ (`generative_gate.py`) — first NUMBER on the grounded-vs-fluent tension: at small scale there
is no tension.** A ~3 M-param causal Transformer LM on the **Brown corpus** (real running English), two
arms at matched (grounded is slightly *leaner*) params: `flat` token embedding vs `grounded+token`
(`concat(token-emb, WordNet supersense-emb, depth-emb) == d_model`, no projection). **Embedding-starved
protocol** (generative analog of brick-1 zero-shot): a held-out noun set `H` is forced to `UNK` on the
*input* for **both** arms — neither learns its token embedding — but the grounded arm still receives
`H`-words' `(supersense, depth)`. 3-seed mean (`H` set + init varied per seed), flat → grounded:

| bucket | flat → grounded | Δ |
|---|---|---|
| overall PPL | 91.4 → 86.2 | **−5.7%** |
| post-seen-noun (control) | 82.1 → 79.9 | −2.7% |
| **post-H-noun (starved, payoff)** | 143.3 → 104.8 | **−26.9%** (per-seed −28 / −34 / −19%, stable) |

- **Fluency preserved** — grounding slightly *helps* overall, with *fewer* params. The tension does not
  bite at this scale.
- **Payoff −27%, concentrated** at embedding-starved positions: where the token embedding is blank,
  inherited category alone predicts the continuation. The control (post-seen-noun, −2.7%) confirms the
  win is concentrated, not a global capacity edge — ruling out the param confound (grounded is leaner).
- **HONEST caveats (why the terminus stays open):** (1) **scale** — 3 M params / 1 M tokens; at billions
  of params rare-word embeddings are better learned so the starved-position payoff may shrink, and
  whether routing *every* token through grounding constrains a large *fluent* model is untested. (2) the
  `H`-protocol collapses *all* starved words to one `UNK`, so flat gets **zero** signal there; a real
  subword/BPE model gets *partial* signal — so −27% is the gap vs `UNK`-collapse, **not** vs BPE backoff.
  That baseline gap is closed by the next gate ↓.

**Brick 3′ HARDENED (`subword_gate.py`) — the subword-backoff baseline deflates the headline but grounding
survives, small.** Four arms, same backbone, differing only in the held-out backoff: `flat`(UNK) ·
`char`(fastText hashed char-n-grams = a real subword LM) · `grounded`(UNK+WordNet) · `char+grounded`.
At held-out (starved) positions:

| arm | post-H PPL | vs `char` |
|---|---|---|
| flat (UNK floor) | ~133 | — |
| **char (real subword)** | **~81** | baseline |
| grounded (WordNet only) | ~95 | **+18% (worse)** |
| char+grounded | ~78 | **−2.3%** (3-seed mean; every seed negative) |

- **Brick 3′'s −27% was mostly beating a broken `UNK` baseline.** A real char/subword backoff alone
  recovers *more* of the gap (−39% vs flat) than grounding did. **Spelling > coarse WordNet category** as
  a standalone backoff — morphology carries number/tense/derivation and often pins the exact word, while
  26 supersenses are coarse — so `grounded`-only is **+18% worse than `char`**.
- **Grounding still earns a place, but a small one:** on top of subword it is a **robust −2.3%** at
  starved positions (−1.5% overall), every seed negative — real semantic signal beyond spelling, but a
  *minor complement, not a major lever*. Falsifier ("no gain over `char`") not triggered.
- **Corrected architecture:** a real grounded LM = **subword/BPE backbone (the strong backoff) + token
  embeddings + a thin grounding top-up**. This is the program's recurring shape — the unfair part was the
  *baseline* (`UNK`), exactly as earlier arcs were misled by the wrong *metric*.

**Brick 3 (`recursion_gate.py`) — depth extrapolation works, but GROUNDING carries it, not typed sheaf
composition.** Nested clauses `(S V O)→(S V (S V O))→…` with a recursive role-weighted target; train
depths {1,2,3}, test {4,5,6} (extrapolation) + unseen words. flat-GRU vs grounded-GRU (generic recurrence
on grounded features) vs grounded-sheaf (typed weight-tied restriction maps `R_subj/R_verb/R_child` folded
along the tree).
- **Depth extrapolation is "easy" here** — *every* recurrent arm rolls the uniform recurrence out to depths
  4–6 untrained, including plain flat-GRU on seen words (1.00). Depth per se is not a discriminating axis.
- **The dictionary is the lever (words):** both grounded arms = **1.00 on UNSEEN words at every depth**;
  flat = chance (0.10). Grounding's zero-shot-words win HOLDS recursively (brick 1/2 confirmed at depth).
- **Typed sheaf composition is NOT a distinct lever:** grounded-sheaf ≡ grounded-GRU on everything
  (1.00/1.00) at comparable params — a generic grounded recurrence matches the typed restriction-map fold.
  **Falsifier fired.** (Methodology: the sheaf cell needed LayerNorm to keep the recursive value
  depth-consistent — an unnormalized fold collapsed to ln2/0.50; the discipline caught a false negative.)
- **Consistent with the whole thread:** the sheaf's value is MEANING/grounding (the dictionary); the
  specifically-sheaf STRUCTURE (eigenweights, λ₁, gluing/H¹, typed composition) keeps not being the lever.
  Caveat: the target is a *uniform* recurrence (easy to roll out); a non-uniform compositional task is the
  only place typed composition could still separate — low prior given the thread, but the honest rescue.

**Brick 3 rescue RESOLVED-NEGATIVE (`nonuniform_gate_v2.py`, `PREREG_nonuniform_composition_gate.md`) — the
non-uniform task closes the last door.** Same three arms, same architectures; the *only* change is the fold
operator's commutativity — a verb-typed **permutation** `val ← P[verb][(val+ss(subj))%S]` (non-commutative:
order and type now matter, so a generic recurrence cannot collapse the fold to one effective operator, and
typed weight-tied restriction maps have their fairest possible advantage). Converged (16k steps, both grounded
arms in-dist ≈ 1.00), 2-seed, with a *valid* commutative control (`abelian_wt`, the weighted fold GRUs can
learn) reproducing a clean architectures-tie null. **Falsifier fired again, and it holds after convergence:**
on the non-commutative target the generic grounded-GRU extrapolates to depth **better** than the typed sheaf
(**0.91 vs 0.79** seen, d4–6; margin −0.13) — typed composition is not just inert but modestly *worse*.
Grounding's zero-shot-word win holds (both grounded arms 0.91/0.78 vs flat 0.13). *(Two instrument faults
were caught and fixed before believing it — the clean commutative twin is parity-hard so GRUs sat at chance
[forcing the weighted control], and the sheaf was undertrained in the first pass [inflating the gap to −0.19];
the round-trip verifier + convergence discipline caught both. Same lesson as the whole thread: validate the
instrument before the finding.)* **Verdict: specifically-sheaf typed composition never separates from generic
grounded recurrence — uniform OR non-uniform. Grounding is the lever; sheaf geometry is not.** The one door
the recursion gate left open is now shut.

**The terminus (RUN on BOTH scaling axes — POSITIVE and 3-seed SEED-ROBUST on data AND params; Chinchilla-scaled + billions still open):**
the goal is a **fluent generative LM** routing every token through a grounded sheaf at billions of parameters.
Brick 3′'s one open question — *does the grounding edge survive scale, or is it a small-data crutch that
better-learned rare embeddings erase?* — was gated (`PREREG_terminus_scale_gate.md`, `terminus/`). A ~40 M-param
GPT (GPT-2 BPE, real **FineWeb-Edu** with a validated WordNet↔BPE grounding stream), arms flat / grounded /
grounded-**random** placebo, swept over token budgets {8M, 30M, 90M}, **3 seeds**. Result: the overall
grounded-vs-flat edge **holds ~7–9% across an 11× data increase (does not decay)**, and — the load-bearing
control — grounded beats its matched-capacity **random-supersense placebo at every seed × every budget (9/9)**
with a margin that **grows with scale (+3.1 → +10.4%)**. So the edge is **semantic** (real inherited meaning,
not the grounding pathway's extra capacity) **and seed-robust**, not a small-data artifact. This is the program's
**first structural positive** — sheaf composition, Möbius geometry, and SHA "certification" all died at their
gates; grounding is the one inherited-structure idea that survives *and strengthens* under scale. **A second,
independent ladder confirms it on the other axis:** holding data fixed (30 M tokens) and varying **model size**
14 M→90 M (3 seeds, `RESULT 3`), grounding again beats its placebo **9/9 across every size and seed**, margin
growing +1.3→+13.5% — the "better-learned embeddings erase it" falsifier did not fire on the param axis either.
The two ladders **weld exactly** at their shared 40 M × 30 M point (bit-for-bit per seed). **Honest limits:** each
ladder is *single-variable* (vary tokens at fixed params, or params at fixed tokens); the fixed-token param ladder
is undertrained-favoring, so the clean claim is *survives / does not decay*, not *grows because of params*. A
**Chinchilla-scaled** ladder (tokens ∝ params) and **true billions** (real GPUs / cloud) remain the sharper open
tests; one corpus, WordNet nouns only. Everything up to here — words (brick 1), structured composition (brick 2,
hardened), a small generative LM (brick 3′, hardened against subword), and now the terminus **hardened on both
scaling axes (3-seed data + 3-seed params)** — is built and confirmed on real language.

---

## Routing logic — de-mythologized, and the H¹ gate (another rigorous-but-inert negative)

A "coherence transport / dimensions of merit" proposal arrived (thermodynamic entropy, geodesic flow,
topos/Lax pairs, categorical point-replacement). Run through the program's gate filter: #4 quarantined;
**#1 reduces to entropy-regularized routing `[E]`** and **#2 to natural-gradient/Fisher optimization `[E]`**
— both standard, real, not novel; **#3 — the Čech H¹ global-consistency obstruction — was the only one
with a falsifiable in-program core** (continuous with the sheaf-as-verifier win), so it got a pre-registration
(`PREREG_h1_routing_gate.md`) and a gate (`h1_gate.py`), *not* a slot in the design.

**The gate fired the falsifier.** Stage 1 validated the obstruction (PR box: pairwise-consistent yet no
global section; classical glues). Stage 3 (GPT-2/Brown, N=1500, label = true-token surprisal): pairwise
baseline AUROC **0.601**, the H¹ higher-order signal **0.503 ≈ chance**, held-out increment over pairwise
**+0.000**. Garden-path illustration: contextual-fraction 0.00 on both — under the natural nested-window
cover, genuine multi-way contextuality barely arises. **The theorem is rigorous; the operational payoff is
zero.** H¹ does not enter the routing logic. (Honest limit: one cover; a non-nested cover is the only
rescue path, but the burden was on H¹ and it didn't clear it.) Net routing logic: entropy-reg + natural
gradient, both standard; **no structural veto is supported by evidence.** Same lesson as the subword gate —
rigorous higher-order structure, real, and *not the lever* — now at the routing layer too.

## DeltaSheaf — the obstruction idea re-tested on real LLM ensembles (`../../deltasheaf-v02/`, 2026-07-17)
A follow-on to the H¹ gate: does inter-model disagreement geometry (edge deltas, cycle residuals, the "area
between maps") **decode-recover** the gold answer on items where an ensemble is jointly wrong? Frozen SPEC +
controls (`CONCLUSION.md`). **Banked negative, twice:** (a) the sheaf geometry — map / displacement / area /
volume — is all at chance on 322 blind-spot items (0-of-5 across 5 families 26M–7B), instrument-validated on
synthetic and round-tripped (recovers on clean 0.37, null on blind-spot 0.21); (b) a bigger *same-kind* reader
(Qwen-7B) is also at chance (25.8%) — the blind spots are **scale-invariant**. The one open branch — whether a
genuinely *complementary* system (Wikipedia retrieval) recovers them — is honestly **UNDECIDED**: the only
valid reader (7B open-book) is too slow for this environment, and the small-model reader degrades on context
(a driftwave round-trip **VOIDED** an apparent "retrieval hurts" null — it was reader degradation, −7% on
*easy* items too). **Meta-lesson (the program's spine, sharpest instance yet): FOUR instrument artifacts caught
before believing a false finding** — degenerate substrate, throttled retrieval, degenerate cosine reader, and
the dangerous degrading open-book reader. Geometry/obstruction is *still* not the lever; grounding
(information-hole import) remains the survivor; whether retrieval *import* recovers reasoning-hard holes is open.

## Quarantine — what we never needed

constants-from-topology · RH "resolution" · "deterministic → prevents all hallucination" · Waypoint
Grub · the variational-sheaf-Laplacian-as-cosmic-operator framing. **The buildable architecture — both
the small-footprint design and the grounded-meaning substrate — lives entirely without the mythology.**
What's real is real because it survived a gate; what's grandiose stayed at the door.
