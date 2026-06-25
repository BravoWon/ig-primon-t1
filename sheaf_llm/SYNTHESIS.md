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

**The terminus (open — with an honest, modest data point at the bottom of the ladder):** a **fluent
generative LM** routing every token through a grounded sheaf at billions of parameters. Measured against
a *real* subword baseline (not the `UNK` strawman), grounding is a **small but robust generative positive
(~2% PPL)** at 3 M params — not the large lever the `UNK` comparison suggested. Whether even that thin
edge survives scale (better-learned rare embeddings) and whether grounding-everywhere constrains a large
fluent model remain open and need real GPUs. Everything up to here — words (brick 1), structured
composition (brick 2, hardened), and a small generative LM (brick 3′, hardened against subword) — is
built and confirmed on real language.

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

## Quarantine — what we never needed

constants-from-topology · RH "resolution" · "deterministic → prevents all hallucination" · Waypoint
Grub · the variational-sheaf-Laplacian-as-cosmic-operator framing. **The buildable architecture — both
the small-footprint design and the grounded-meaning substrate — lives entirely without the mythology.**
What's real is real because it survived a gate; what's grandiose stayed at the door.
