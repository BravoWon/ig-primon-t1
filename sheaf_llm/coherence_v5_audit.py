#!/usr/bin/env python
"""coherence v5 -- AUDIT the v4 judge before trusting its reversal. (verify at source.)

v4's headline (contradiction is real, CONTESTED appears) rests on a 3B judge reading commit MESSAGE-only
over N<=14. Before consolidating that, validate the instrument:
  1. FULL DIFF: re-judge the flagged repos' covered commits using the actual code diff, not just the message.
  2. SELF-CONSISTENCY: judge each commit 3x (sampled) -> majority + agreement rate (judge reliability).
  3. EVIDENCE: print the commits judged CONTRADICT (design statement + commit + diff) so a human can check.
Focus = repos v4 called CONTESTED/CHAOTIC + a COHERENT control (the judge must give it LOW contradiction).

  reversal holds IF: flagged repos keep high contradiction under full-diff + the printed examples are real
  contradictions + the control stays low + self-agreement is high. Else v4's reversal is itself an artifact.

    python coherence_v5_audit.py
"""
import os, re, subprocess, glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/JT-DEV1/Desktop/development/_coherence_repos"
REPOS = [d for d in sorted(glob.glob(ROOT + "/*")) if os.path.isdir(d + "/.git")]
REPOS += ["C:/Users/JT-DEV1/Desktop/development/proj-0/isoZ"]
FOCUS = {"obra_superpowers-marketplace": "CONTESTED?", "daymade_claude-code-skills": "CONTESTED?",
         "jarrodwatts_claude-hud": "CHAOTIC?", "team-attention_plugins-for-claude-natives": "control(COHERENT)",
         "bevyengine_bevy": "control(COHERENT)"}
MAXC, MAXSEC, KREMOVE, NJ = 500, 150, 3, 16
DESIGN_PAT = re.compile(r"(readme|architect|design|contribut|overview|docs/|adr|spec|roadmap|manifesto)", re.I)
JUDGE_ID = "Qwen/Qwen2.5-3B-Instruct"

_etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
_jtok = AutoTokenizer.from_pretrained(JUDGE_ID)
_jmod = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def embed(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = _etok([t[:512] for t in texts[i:i + bs]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
        h = _emod(**enc).last_hidden_state; m = enc.attention_mask[..., None].float()
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384))


@torch.no_grad()
def judge(section, change, sample=False):
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


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="ignore", timeout=120).stdout


def design_sections(repo):
    files = sorted([f for f in git(repo, "ls-files", "*.md").splitlines() if DESIGN_PAT.search(f)],
                   key=lambda f: (f.count("/"), len(f)))[:60]
    secs = []
    for f in files:
        try:
            txt = open(os.path.join(repo, f), encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for s in re.split(r"\n#{1,3}\s", txt):
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 80:
                secs.append(s[:600])
    return [secs[i] for i in np.linspace(0, len(secs) - 1, MAXSEC).astype(int)] if len(secs) > MAXSEC else secs


def commits(repo):
    raw = git(repo, "log", f"-n{MAXC}", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s %b%x1f")
    out = []
    for rec in raw.split("\x1e"):
        p = rec.split("\x1f")
        if len(p) < 3:
            continue
        msg = re.sub(r"\s+", " ", p[1]).strip()[:300]
        files = [l.strip() for l in p[2].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        out.append((p[0].strip(), msg, f"{msg}  || files: {dirs}"))
    return list(reversed(out))


def deboil(v, k):
    mu = v.mean(0); X = v - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    X = X - (X @ Vt[:k].T) @ Vt[:k]
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def get_diff(repo, h):
    d = git(repo, "show", h, "--no-color", "--format=", "--", ".")
    return re.sub(r"\n{3,}", "\n", d)[:1400]


def main():
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
    rawV = embed(alltexts); V = deboil(rawV, KREMOVE)
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
