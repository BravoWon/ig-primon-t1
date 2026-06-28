#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v2 -- de-boilerplated + orphan-drift, CC plugins + AAA/mixed.

v1 worked but was small: absolute glue (0.4-0.6) was dominated by domain-generic similarity (everything
is "CC plugin" text). v2 fixes that and sharpens the drift signal:

  1. DE-BOILERPLATE (common-component removal, Arora's "all-but-the-top"): pool ALL embeddings, remove
     the mean + top-K principal directions (the generic "this-is-a-software-repo" subspace), renormalize.
  2. PER-COMMIT de-baselined signal: signal(c) = glue_own(c) - glue_otherpool(c). coherence = mean signal.
  3. ORPHAN-DRIFT (primary): orphan = signal(c) <= 0; orphan% = fraction; drift trace = rolling signal.
  4. WIDEN: AAA/mixed repos (godot, bevy, react, obs) added to break the CC-plugin monoculture.

Shared plumbing (embed/git/design_sections/commits/deboilerplate) lives in coherence_lib.

    python coherence_v2.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coherence_lib import (DEV, AAA, REPOS, KREMOVE, NAVY, GREEN, BLUE,
                           embed, design_sections, commits, deboilerplate)
import os


def main():
    print(f"[coherence v2]  dev={DEV}  de-boilerplate(remove mean+top{KREMOVE})  orphan-drift  CC+AAA")
    raw = {}
    for repo in REPOS:
        name = os.path.basename(repo)
        secs = design_sections(repo)
        coms = [t for _, _, t in commits(repo)]                      # v2 uses the text string only
        if len(secs) < 3 or len(coms) < 8:
            print(f"  skip {name}: secs={len(secs)} commits={len(coms)}"); continue
        raw[name] = dict(secs=secs, coms=coms)
    names = list(raw)
    alltexts, spans = [], {}
    for n in names:
        s0 = len(alltexts); alltexts += raw[n]["secs"]
        c0 = len(alltexts); alltexts += raw[n]["coms"]
        spans[n] = (s0, c0, len(alltexts))
    V = deboilerplate(embed(alltexts), KREMOVE)
    D = {n: V[spans[n][0]:spans[n][1]] for n in names}
    C = {n: V[spans[n][1]:spans[n][2]] for n in names}

    rng = np.random.default_rng(0); rows = {}
    for n in names:
        others = np.concatenate([D[m] for m in names if m != n])
        idx = rng.choice(len(others), min(600, len(others)), replace=False)
        glue_own = (C[n] @ D[n].T).max(1)
        glue_oth = (C[n] @ others[idx].T).max(1)
        sig = glue_own - glue_oth
        rows[n] = dict(sig=sig, coh=float(sig.mean()), orphan=float((sig <= 0).mean()),
                       own=float(glue_own.mean()), nc=len(C[n]), aaa=(n in AAA))

    order = sorted(names, key=lambda n: -rows[n]["coh"])
    print(f"\n{'repo':40}{'kind':>5}{'commits':>8}{'coh(signal)':>12}{'orphan%':>9}")
    for n in order:
        r = rows[n]
        print(f"{n[:40]:40}{'AAA' if r['aaa'] else 'CC':>5}{r['nc']:>8}{r['coh']:>+12.3f}{r['orphan']*100:>8.0f}%")
    cc = [rows[n]["coh"] for n in names if not rows[n]["aaa"]]
    aaa = [rows[n]["coh"] for n in names if rows[n]["aaa"]]
    print(f"\n  mean coherence-signal: CC plugins {np.mean(cc):+.3f}   AAA/mixed {np.mean(aaa):+.3f}" if aaa else
          f"\n  mean coherence-signal CC {np.mean(cc):+.3f}")
    allsig = [rows[n]["coh"] for n in names]
    print(f"  de-boilerplated signal: {sum(s>0.02 for s in allsig)}/{len(allsig)} repos > +0.02  "
          f"(v1 was +0.042 mean; v2 mean {np.mean(allsig):+.3f})")
    print(f"  -> {'sharper & holds across CC+AAA' if np.mean(allsig)>0.04 else 'modest'};"
          f" orphan-drift now ranks repos by design-lag (high orphan% = code outran docs).")

    fig, ax = plt.subplots(figsize=(10, 6)); y = np.arange(len(order))
    ax.barh(y, [rows[n]["coh"] for n in order], color=[BLUE if rows[n]["aaa"] else GREEN for n in order])
    for i, n in enumerate(order):
        ax.text(rows[n]["coh"] + 0.002, i, f"orphan {rows[n]['orphan']*100:.0f}%", va="center", fontsize=6, color="#555")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n[:32] for n in order], fontsize=7)
    ax.set_xlabel("de-boilerplated coherence signal (mean per-commit own-minus-baseline)")
    ax.set_title("Design<->evolution coherence v2 (de-boilerplated): green=CC plugin, blue=AAA/mixed",
                 color=NAVY, fontsize=10); fig.tight_layout()
    fig.savefig("sheaf_llm/coherence_v2_bars.png", dpi=160); plt.close(fig)

    rich = sorted(names, key=lambda n: -rows[n]["nc"])[:9]
    cols = 3; rws = int(np.ceil(len(rich) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(12, 2.6 * rws), squeeze=False)
    for k, n in enumerate(rich):
        ax = axes[k // cols][k % cols]; s = rows[n]["sig"]
        win = max(3, len(s) // 25); sm = np.convolve(s, np.ones(win)/win, mode="valid")
        ax.plot(sm, color=BLUE if rows[n]["aaa"] else GREEN, lw=1.3)
        ax.axhline(0, ls=":", color="#c0392b", lw=0.8)
        ax.set_title(f"{n[:30]} ({'AAA' if rows[n]['aaa'] else 'CC'}, orphan {rows[n]['orphan']*100:.0f}%)",
                     fontsize=7.5, color=NAVY)
        ax.tick_params(labelsize=6)
    for k in range(len(rich), rws * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Orphan-drift traces: rolling per-commit coherence signal (below red 0 = drift / orphan commits)",
                 color=NAVY, fontsize=11); fig.tight_layout()
    fig.savefig("sheaf_llm/coherence_v2_traces.png", dpi=150); plt.close(fig)
    print("  wrote sheaf_llm/coherence_v2_bars.png, sheaf_llm/coherence_v2_traces.png")


if __name__ == "__main__":
    main()
