#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v4 -- LLM-JUDGE contradiction (the test embeddings can't do).

v3 found 0 CONTESTED repos, but its contradiction axis was a churn-VOCABULARY proxy that cannot see
semantic contradiction. v4 replaces it with a real judge: an instruct LLM reads each COVERED commit
against its COVERING design section and rates ALIGN / CONTRADICT / UNRELATED. Tests whether CONTESTED=0
is true or a proxy artifact -- and whether the churn-regex over/under-counts real contradiction.

  coverage      : same de-baselined gluing as v3 (does the change match its OWN design?)
  judge(commit) : Qwen2.5-3B-Instruct on (covering design section, commit message+files) -> verdict
  LLM-contra%   : CONTRADICT / (ALIGN+CONTRADICT) among judged covered commits
  compare to v3 churn-proxy; re-draw the coverage-vs-alignment map with the SEMANTIC axis.

Honest scope: judges commit MESSAGE+FILES vs design (semantic, far better than keyword-matching), not
the full diff (that's v5; avoids 100s of lazy blob fetches). Bounded sample per repo.

    python coherence_v4.py
"""
import os, re, subprocess, glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/JT-DEV1/Desktop/development/_coherence_repos"
AAA = {"godotengine_godot", "bevyengine_bevy", "microsoft_TypeScript", "facebook_react", "obsproject_obs-studio"}
REPOS = [d for d in sorted(glob.glob(ROOT + "/*")) if os.path.isdir(d + "/.git")]
REPOS += ["C:/Users/JT-DEV1/Desktop/development/proj-0/isoZ"]
MAXC, MAXSEC, KREMOVE, NJUDGE = 500, 150, 3, 14
DESIGN_PAT = re.compile(r"(readme|architect|design|contribut|overview|docs/|adr|spec|roadmap|manifesto)", re.I)
CHURN = re.compile(r"\b(revert|rollback|back ?out|undo|breaking change|no longer|deprecat|remove|delete|"
                   r"drop support|rewrite|overhaul|revamp|hack|workaround|regression|broke|kludge)\b", re.I)
JUDGE_ID = "Qwen/Qwen2.5-3B-Instruct"
NAVY, GREEN, BLUE = "#15293f", "#1e7d34", "#2c6fbb"

_etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
print(f"[coherence v4]  loading judge {JUDGE_ID} ...")
_jtok = AutoTokenizer.from_pretrained(JUDGE_ID)
_jmod = AutoModelForCausalLM.from_pretrained(JUDGE_ID, torch_dtype=torch.float16).to(DEV).eval()


@torch.no_grad()
def embed(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = _etok([t[:512] for t in texts[i:i + bs]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
        h = _emod(**enc).last_hidden_state
        m = enc.attention_mask[..., None].float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384))


@torch.no_grad()
def judge(section, change):
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


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="ignore", timeout=180).stdout


def design_sections(repo):
    files = [f for f in git(repo, "ls-files", "*.md").splitlines() if DESIGN_PAT.search(f)]
    files = sorted(files, key=lambda f: (f.count("/"), len(f)))[:60]
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
    if len(secs) > MAXSEC:
        secs = [secs[i] for i in np.linspace(0, len(secs) - 1, MAXSEC).astype(int)]
    return secs


def commits(repo):
    raw = git(repo, "log", f"-n{MAXC}", "--no-merges", "--name-only", "--format=%x1e%s %b%x1f")
    out = []
    for rec in raw.split("\x1e"):
        parts = rec.split("\x1f")
        if len(parts) < 2:
            continue
        msg = re.sub(r"\s+", " ", parts[0]).strip()[:300]
        files = [l.strip() for l in parts[1].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        out.append((msg, f"{msg}  || files: {dirs}"))
    return list(reversed(out))


def deboilerplate(vecs, k):
    mu = vecs.mean(0); X = vecs - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:k]; X = X - (X @ U.T) @ U
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def main():
    raw = {}
    for repo in REPOS:
        name = os.path.basename(repo)
        secs, coms = design_sections(repo), commits(repo)
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
        # LLM-judge a sample of covered commits
        sample = rng.choice(covered, min(NJUDGE, len(covered)), replace=False) if len(covered) else []
        verds = []
        for i in sample:
            cover_idx = int((Craw[n][i] @ Draw[n].T).argmax())       # covering section (raw similarity)
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
