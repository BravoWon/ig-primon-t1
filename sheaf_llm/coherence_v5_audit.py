#!/usr/bin/env python
"""coherence v5 -- AUDIT the v4 judge before trusting its reversal. (verify at source.)

v4's headline (contradiction is real, CONTESTED appears) rests on a 3B judge reading commit MESSAGE-only
over N<=14. Before consolidating that, validate the instrument:
  1. FULL DIFF: re-judge the flagged repos' covered commits using the actual code diff, not just the message.
  2. SELF-CONSISTENCY: judge each commit 3x (sampled) -> majority + agreement rate (judge reliability).
  3. EVIDENCE: print the commits judged CONTRADICT (design statement + commit + diff) so a human can check.
Focus = repos v4 called CONTESTED/CHAOTIC + a COHERENT control (the judge must give it LOW contradiction).

Shared plumbing in coherence_lib; the full-diff judge prompt + get_diff are v5's, kept local.

    python coherence_v5_audit.py
"""
import re
import numpy as np
import torch
from coherence_lib import (DEV, REPOS, KREMOVE, embed, git, design_sections, commits,
                           deboilerplate, load_judge)

FOCUS = {"obra_superpowers-marketplace": "CONTESTED?", "daymade_claude-code-skills": "CONTESTED?",
         "jarrodwatts_claude-hud": "CHAOTIC?", "team-attention_plugins-for-claude-natives": "control(COHERENT)",
         "bevyengine_bevy": "control(COHERENT)"}
NJ = 16
_jtok = _jmod = None                                     # lazy: loaded on first judge() call


@torch.no_grad()
def judge(section, change, sample=False):
    global _jtok, _jmod
    if _jmod is None:
        _jtok, _jmod = load_judge()
    msgs = [{"role": "system", "content": "You audit whether a code change is consistent with a project's "
             "stated design. Reply with exactly one word: ALIGN, CONTRADICT, or UNRELATED."},
            {"role": "user", "content": f"DESIGN STATEMENT:\n{section[:600]}\n\nCODE CHANGE (commit message "
             f"and diff):\n{change[:1400]}\n\nDoes the change ALIGN with, CONTRADICT, or is it UNRELATED to "
             f"the design statement? One word:"}]
    enc = _jtok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEV)
    g = dict(max_new_tokens=4, pad_token_id=_jtok.eos_token_id)
    g.update(dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False))
    out = _jmod.generate(**enc, **g)
    txt = _jtok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).upper()
    m = re.search(r"CONTRADICT|UNRELATED|ALIGN", txt)
    return m.group(0) if m else "UNRELATED"


def get_diff(repo, h):
    d = git(repo, "show", h, "--no-color", "--format=", "--", ".")
    return re.sub(r"\n{3,}", "\n", d)[:1400]


def main():
    import os
    raw = {}
    for repo in REPOS:
        n = os.path.basename(repo)
        secs, coms = design_sections(repo), commits(repo)
        if len(secs) >= 3 and len(coms) >= 8:
            raw[n] = dict(repo=repo, secs=secs, coms=coms)
    names = list(raw)
    alltexts, spans = [], {}
    for n in names:
        s0 = len(alltexts); alltexts += raw[n]["secs"]
        c0 = len(alltexts); alltexts += [t for _, _, t in raw[n]["coms"]]
        spans[n] = (s0, c0, len(alltexts))
    rawV = embed(alltexts); V = deboilerplate(rawV, KREMOVE)
    D = {n: V[spans[n][0]:spans[n][1]] for n in names}; C = {n: V[spans[n][1]:spans[n][2]] for n in names}
    Draw = {n: rawV[spans[n][0]:spans[n][1]] for n in names}; Craw = {n: rawV[spans[n][1]:spans[n][2]] for n in names}

    rng = np.random.default_rng(0)
    print(f"\n{'repo':40}{'expect':>16}{'msg-only(v4)':>14}{'FULL-DIFF':>11}{'self-agree':>12}")
    evidence = []
    for n in [n for n in names if n in FOCUS]:
        others = np.concatenate([D[m] for m in names if m != n])
        idx = rng.choice(len(others), min(600, len(others)), replace=False)
        sig = (C[n] @ D[n].T).max(1) - (C[n] @ others[idx].T).max(1)
        covered = np.where(sig > 0)[0]
        samp = rng.choice(covered, min(NJ, len(covered)), replace=False)
        v_msg, v_diff, agree = [], [], []
        for i in samp:
            cover = int((Craw[n][i] @ Draw[n].T).argmax()); sec = raw[n]["secs"][cover]
            h, msg, text = raw[n]["coms"][i]
            vm = judge(sec, text)                                       # v4-style: message+files only
            diff = get_diff(raw[n]["repo"], h)
            change = f"{msg}\nDIFF:\n{diff}"
            vd = judge(sec, change)                                     # full diff
            s3 = [judge(sec, change, sample=True) for _ in range(3)]    # self-consistency
            agree.append(s3.count(max(set(s3), key=s3.count)) / 3)
            v_msg.append(vm); v_diff.append(vd)
            if vd == "CONTRADICT":
                evidence.append((n, msg[:90], sec[:160], diff[:240]))
        dm = [v for v in v_msg if v in ("ALIGN", "CONTRADICT")]; dd = [v for v in v_diff if v in ("ALIGN", "CONTRADICT")]
        cm = sum(v == "CONTRADICT" for v in dm) / max(len(dm), 1)
        cd = sum(v == "CONTRADICT" for v in dd) / max(len(dd), 1)
        print(f"{n[:40]:40}{FOCUS[n]:>16}{cm*100:>12.0f}%{cd*100:>10.0f}%{np.mean(agree)*100:>11.0f}%")

    print(f"\n--- EVIDENCE: commits judged CONTRADICT under full diff (read & sanity-check) ---")
    for n, msg, sec, diff in evidence[:8]:
        print(f"\n[{n}]\n  design : {sec}\n  commit : {msg}\n  diff   : {re.sub(chr(10),' ',diff)[:200]}")
    print(f"\n  -> if these are REAL contradictions and the control stayed low + self-agree high,"
          f" v4's reversal HOLDS. If they're judge noise, the reversal is an artifact.")


if __name__ == "__main__":
    main()
