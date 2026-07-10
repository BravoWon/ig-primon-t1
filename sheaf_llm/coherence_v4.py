#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v4 -- LLM-JUDGE contradiction (the test embeddings can't do).

v3's contradiction axis was a churn-VOCABULARY proxy that cannot see semantic contradiction. v4 replaces
it: an instruct LLM reads each COVERED commit against its COVERING design section and rates ALIGN /
CONTRADICT / UNRELATED. Tests whether v3's CONTESTED=0 is true or a proxy artifact.

Honest scope: judges commit MESSAGE+FILES vs design (not the full diff -- that's v5). Bounded sample/repo.
Shared plumbing in coherence_lib; CHURN + the message-only judge prompt are v4's, kept local.

    python coherence_v4.py
"""
import os, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coherence_lib import (DEV, AAA, REPOS, KREMOVE, NAVY, GREEN, BLUE,
                           embed, design_sections, commits, deboilerplate, load_judge)

NJUDGE = 14
CHURN = re.compile(r"\b(revert|rollback|back ?out|undo|breaking change|no longer|deprecat|remove|delete|"
                   r"drop support|rewrite|overhaul|revamp|hack|workaround|regression|broke|kludge)\b", re.I)
_jtok = _jmod = None                                     # lazy: loaded on first judge() call


@torch.no_grad()
def judge(section, change):
    global _jtok, _jmod
    if _jmod is None:
        _jtok, _jmod = load_judge()
    msgs = [{"role": "system", "content": "You audit whether a code change is consistent with a project's "
             "stated design. Reply with exactly one word: ALIGN, CONTRADICT, or UNRELATED."},
            {"role": "user", "content": f"DESIGN STATEMENT:\n{section[:600]}\n\nCODE CHANGE (commit):\n"
             f"{change[:400]}\n\nIs the change ALIGN, CONTRADICT, or UNRELATED to the design statement? One word:"}]
    enc = _jtok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True).to(DEV)
    out = _jmod.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=_jtok.eos_token_id)
    txt = _jtok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).upper()
    m = re.search(r"CONTRADICT|UNRELATED|ALIGN", txt)
    return m.group(0) if m else "UNRELATED"


def main():
    raw = {}
    for repo in REPOS:
        name = os.path.basename(repo)
        secs = design_sections(repo)
        coms = [(m, t) for _, m, t in commits(repo)]
        if len(secs) < 3 or len(coms) < 8:
            continue
        raw[name] = dict(secs=secs, msgs=[m for m, _ in coms], texts=[t for _, t in coms])
    names = list(raw)
    alltexts, spans = [], {}
    for n in names:
        s0 = len(alltexts); alltexts += raw[n]["secs"]
        c0 = len(alltexts); alltexts += raw[n]["texts"]
        spans[n] = (s0, c0, len(alltexts))
    rawV = embed(alltexts); V = deboilerplate(rawV, KREMOVE)
    D = {n: V[spans[n][0]:spans[n][1]] for n in names}
    C = {n: V[spans[n][1]:spans[n][2]] for n in names}
    Draw = {n: rawV[spans[n][0]:spans[n][1]] for n in names}
    Craw = {n: rawV[spans[n][1]:spans[n][2]] for n in names}

    rng = np.random.default_rng(0); R = {}
    print(f"  judging up to {NJUDGE} covered commits/repo (ALIGN/CONTRADICT/UNRELATED)...")
    for n in names:
        others = np.concatenate([D[m] for m in names if m != n])
        idx = rng.choice(len(others), min(600, len(others)), replace=False)
        sig = (C[n] @ D[n].T).max(1) - (C[n] @ others[idx].T).max(1)
        covered = np.where(sig > 0)[0]
        churn = np.array([bool(CHURN.search(raw[n]["msgs"][i])) for i in range(len(sig))])
        sample = rng.choice(covered, min(NJUDGE, len(covered)), replace=False) if len(covered) else []
        verds = []
        for i in sample:
            cover_idx = int((Craw[n][i] @ Draw[n].T).argmax())
            verds.append(judge(raw[n]["secs"][cover_idx], raw[n]["texts"][i]))
        dec = [v for v in verds if v in ("ALIGN", "CONTRADICT")]
        llm_contra = (sum(v == "CONTRADICT" for v in dec) / len(dec)) if dec else 0.0
        churn_contra = float(churn[covered].mean()) if len(covered) else 0.0
        R[n] = dict(cov=float((sig > 0).mean()), churn=churn_contra, llm=llm_contra,
                    align=1 - llm_contra, nj=len(dec), aaa=(n in AAA))

    def quad(r):
        return ("COHERENT" if r["cov"] >= 0.5 and r["align"] >= 0.6 else
                "CONTESTED" if r["cov"] >= 0.5 else
                "UNDER-DOC" if r["align"] >= 0.6 else "CHAOTIC")

    order = sorted(names, key=lambda n: -(R[n]["cov"] + R[n]["align"]))
    print(f"\n{'repo':38}{'kind':>5}{'cover%':>8}{'churn-c%':>9}{'LLM-c%':>8}{'class':>11}")
    for n in order:
        r = R[n]
        print(f"{n[:38]:38}{'AAA' if r['aaa'] else 'CC':>5}{r['cov']*100:>7.0f}%{r['churn']*100:>8.0f}%"
              f"{r['llm']*100:>7.0f}%{quad(r):>11}")
    from collections import Counter
    cnt = Counter(quad(R[n]) for n in names)
    mc = np.mean([R[n]["llm"] for n in names]); mk = np.mean([R[n]["churn"] for n in names])
    print(f"\n  quadrants: " + "  ".join(f"{k}={v}" for k, v in cnt.items()))
    print(f"  mean contradiction among covered: churn-proxy {mk*100:.0f}%  vs  LLM-judge {mc*100:.0f}%")
    print(f"  -> {'LLM finds MORE contradiction than churn-regex (proxy under-counted)' if mc > mk + 0.05 else 'LLM CONFIRMS contradiction is rare even under semantic judging' if mc < mk + 0.05 else ''};"
          f" {'CONTESTED now appears' if cnt.get('CONTESTED',0) else 'CONTESTED still empty -> omission-not-commission holds under a real judge'}.")

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axhline(0.6, color="#bbb", lw=1); ax.axvline(0.5, color="#bbb", lw=1)
    ax.add_patch(plt.Rectangle((0.5, 0.6), 0.55, 0.5, color="#e7f3ea", zorder=0))
    ax.add_patch(plt.Rectangle((0.5, 0), 0.55, 0.6, color="#f7e3e0", zorder=0))
    for n in names:
        r = R[n]
        ax.scatter(r["cov"], r["align"], s=60, color=BLUE if r["aaa"] else GREEN, edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(f"{n[:24]} (n={r['nj']})", (r["cov"], r["align"]), fontsize=6.3,
                    xytext=(4, 3), textcoords="offset points", color=NAVY)
    ax.text(0.97, 1.04, "COHERENT", ha="right", fontsize=9, color=GREEN, weight="bold")
    ax.text(0.97, 0.02, "CONTESTED (LLM: code contradicts doc)", ha="right", fontsize=9, color="#c0392b", weight="bold")
    ax.text(0.02, 1.04, "UNDER-DOCUMENTED", fontsize=9, color="#9a6a2f", weight="bold")
    ax.set_xlabel("coverage rate"); ax.set_ylabel("alignment among covered (1 - LLM-judged contradiction)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.1)
    ax.set_title("v4 coherence map: contradiction by LLM JUDGE (green=CC plugin, blue=AAA/mixed)",
                 color=NAVY, fontsize=11)
    fig.tight_layout(); fig.savefig("sheaf_llm/coherence_v4_map.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/coherence_v4_map.png")


if __name__ == "__main__":
    main()
