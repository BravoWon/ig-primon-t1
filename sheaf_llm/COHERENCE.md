# Codebase Design↔Evolution Coherence — what five gates established

Applying the sheaf program's one proven lane (value / coherence / verification) to **real repositories**:
does a codebase's commit evolution *glue* to its stated design (`.md` docs)? Tested on **14 real repos**
— 10 Claude Code plugins + this repo (`isoZ`) + AAA/mixed (`bevy`, `react`, `godot`, `obs-studio`);
cloned to `development/_coherence_repos/`. Every version validated against a baseline. **The conclusion
reversed twice as the instrument sharpened — and that is the real result.**

## The metric
Design `.md` sections = the inherited intent dictionary (the "global section"); commits (message +
changed paths) = local realizations; `glue(commit) = max cosine to any design section`. A **cross-repo
baseline** (each commit's glue to its *own* design vs *other* repos' pooled design) is carried **inside
every sweep**, so nothing is ever plotted blind — the lesson banked from the H¹ routing gate.

## The arc (two reversals)
| ver | method | verdict |
|---|---|---|
| **v1** | glue vs cross-repo baseline | weak-but-real: mean own−baseline **+0.042**, 7/10 positive — *beats* baseline (the H¹ gate was flat zero). Absolute glue 0.4–0.6 dominated by domain-generic similarity. |
| **v2** | de-boilerplate (common-component removal) + orphan-drift | didn't raise the mean, **sharpened the ranking**. Coherence is an **archetype** property crossing CC-vs-AAA (focused+documented glue tight; directories / sparse-doc churn). AAA ≈ CC (+0.028 vs +0.029). |
| **v3** | split **coverage** (documented?) vs **contradiction** (fights the doc?), churn-vocabulary proxy | 7 COHERENT / 7 UNDER-DOC, **0 CONTESTED** → "repos fail by omission, not commission." |
| **v4** | replace proxy with an **LLM judge** (Qwen-3B, message-only) | **REVERSED v3**: mean contradiction 7%→24%, CONTESTED appears (superpowers 50%, daymade 60%, claude-hud 75%). Looked like v3's null was a proxy artifact. |
| **v5 AUDIT** | full-diff re-judge + self-consistency + resampling + printed evidence | **v4's reversal does NOT replicate.** |

### Why v5 broke v4
- **Not reproducible:** message-only rates swing on resampling (superpowers 50%→0%, daymade 60%→29%, claude-hud 75%→33%) — v4's labels were **small-N noise** (N≤14).
- **Collapses under full diff:** reading the actual code, the "CONTESTED" repos drop to **~0%** — the message-only judge hallucinated contradiction from terse subjects.
- **False-positives on a control:** bevy (COHERENT) scores 25%, flagging *"make the Tracy layer optional"* as contradicting *"Bevy has tracing"* — a normal toggle, not a contradiction. Self-agreement 83–100% ⇒ the judge is **confidently miscalibrated**, not flaky.

## The honest bottom
- **Coverage is robustly measurable** and archetype-crossing: focused, documented projects (small
  plugins, bevy, react) glue tight; directories/marketplaces and sparse-doc-high-churn projects
  (`anthropics-official`, `obs-studio`, `godot`, `claude-hud`) are **under-documented**. Survives every version.
- **Contradiction is NOT reliably measurable by cheap means.** Three instruments → three answers
  (regex 7%, LLM-message 24%, LLM-diff ~0–33% with control-level false positives). A trustworthy
  contradiction axis needs **full diffs + a bias-calibrated judge + large N + human-validated labels** —
  none of which a cheap sweep provides. The lone plausible real flag (`claude-hud` "local-only by design"
  vs an "external usage fallback" commit) survives as a *candidate*, unconfirmed.

## The meta-lesson (the session's spine)
**The measurement method determined the conclusion — three times.** Same pattern as the LLM-research
thread: brick 3′'s −27% was a baseline artifact (≈ +2% vs a real subword baseline); the Čech H¹
obstruction was a rigorous theorem with **zero** operational payoff; and here "design contradiction" was
whatever the instrument said. The robust posture, earned: **carry the baseline inside the sweep; validate
the instrument before the finding; audit at source (read the actual flagged commits) before believing a
reversal.** The deliverable is not "repos are (in)coherent" — it is a *validated map of what is measurable
(coverage) and what is not (contradiction, cheaply)*, with every claim's instrument-dependence exposed.

## Reproduce
`coherence_v1.py … v4.py`, `coherence_v5_audit.py` (CPU+GPU; MiniLM embeddings, Qwen2.5-3B-Instruct judge;
repos under `development/_coherence_repos/`). Each prints its table + writes its figure(s).
