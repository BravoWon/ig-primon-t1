#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v3 -- separate COVERAGE from CONTRADICTION.

v2's orphan% conflated two failures: (a) a commit's area has NO design doc to glue to (under-doc), and
(b) a commit's area IS covered but the change FIGHTS the design (true drift). Only (b) is strong
incoherence. v3 splits them into a 2D map.

  COVERAGE(commit)      = de-baselined glue > 0  (does this change distinctively match its OWN design?)
  coverage_rate(repo)   = fraction covered                       -> the 'is it documented' axis
  CONTRADICTION(commit) = churn/anti-design vocabulary in msg (revert|rewrite|deprecate|breaking|
                          remove|workaround|hack|regression|...) -> 'fighting the design' (a heuristic proxy)
  contradiction_rate    = fraction of COVERED commits that are churn-type
  alignment_rate        = 1 - contradiction_rate  (among covered)

2D map (per repo):  x = coverage_rate,  y = alignment_rate(among covered)
  top-right  COHERENT      (documented AND realized)
  bottom-right CONTESTED   (documented but code fights it -- the STRONG incoherence)
  top-left   UNDER-DOC     (sparse docs, but what exists isn't fought)
  bottom-left CHAOTIC

    python coherence_v3.py
"""
import os, re, subprocess, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/JT-DEV1/Desktop/development/_coherence_repos"
AAA = {"godotengine_godot", "bevyengine_bevy", "microsoft_TypeScript", "facebook_react", "obsproject_obs-studio"}
REPOS = [d for d in sorted(glob.glob(ROOT + "/*")) if os.path.isdir(d + "/.git")]
REPOS += ["C:/Users/JT-DEV1/Desktop/development/proj-0/isoZ"]
MAXC, MAXSEC, KREMOVE = 500, 150, 3
DESIGN_PAT = re.compile(r"(readme|architect|design|contribut|overview|docs/|adr|spec|roadmap|manifesto)", re.I)
CHURN = re.compile(r"\b(revert|rollback|roll back|back ?out|undo|breaking change|no longer|deprecat|"
                   r"remove|delete|drop support|rewrite|re-write|overhaul|revamp|hack|workaround|"
                   r"work around|temporary|band-?aid|regression|broke|broken|messy|cleanup|kludge)\b", re.I)
NAVY, GREEN, BLUE = "#15293f", "#1e7d34", "#2c6fbb"

_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()


@torch.no_grad()
def embed(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = _tok([t[:512] for t in texts[i:i + bs]], padding=True, truncation=True,
                   max_length=128, return_tensors="pt").to(DEV)
        h = _mod(**enc).last_hidden_state
        m = enc.attention_mask[..., None].float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384))


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
    raw = git(repo, "log", f"-n{MAXC}", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s %b%x1f")
    out = []
    for rec in raw.split("\x1e"):
        parts = rec.split("\x1f")
        if len(parts) < 3:
            continue
        msg = re.sub(r"\s+", " ", parts[1]).strip()[:300]
        files = [l.strip() for l in parts[2].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        out.append((msg, f"{msg}  || files: {dirs}"))
    return list(reversed(out))


def deboilerplate(vecs, k):
    mu = vecs.mean(0); X = vecs - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:k]; X = X - (X @ U.T) @ U
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def main():
    print(f"[coherence v3]  dev={DEV}  COVERAGE vs CONTRADICTION split  CC+AAA")
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
    V = deboilerplate(embed(alltexts), KREMOVE)
    D = {n: V[spans[n][0]:spans[n][1]] for n in names}
    C = {n: V[spans[n][1]:spans[n][2]] for n in names}

    rng = np.random.default_rng(0); R = {}
    for n in names:
        others = np.concatenate([D[m] for m in names if m != n])
        idx = rng.choice(len(others), min(600, len(others)), replace=False)
        sig = (C[n] @ D[n].T).max(1) - (C[n] @ others[idx].T).max(1)
        covered = sig > 0
        churn = np.array([bool(CHURN.search(m)) for m in raw[n]["msgs"]])
        cov_rate = float(covered.mean())
        contra = float(churn[covered].mean()) if covered.sum() else 0.0   # contradiction AMONG covered
        R[n] = dict(cov=cov_rate, contra=contra, align=1 - contra, churn_all=float(churn.mean()),
                    nc=len(covered), aaa=(n in AAA))

    def quad(r):
        hi_cov, hi_align = r["cov"] >= 0.5, r["align"] >= 0.6
        return ("COHERENT" if hi_cov and hi_align else "CONTESTED" if hi_cov and not hi_align
                else "UNDER-DOC" if not hi_cov and hi_align else "CHAOTIC")

    order = sorted(names, key=lambda n: -(R[n]["cov"] + R[n]["align"]))
    print(f"\n{'repo':40}{'kind':>5}{'coverage%':>10}{'contra%(cov)':>13}{'class':>12}")
    for n in order:
        r = R[n]
        print(f"{n[:40]:40}{'AAA' if r['aaa'] else 'CC':>5}{r['cov']*100:>9.0f}%{r['contra']*100:>12.0f}%{quad(r):>12}")
    from collections import Counter
    cnt = Counter(quad(R[n]) for n in names)
    print(f"\n  quadrants: " + "  ".join(f"{k}={v}" for k, v in cnt.items()))
    print(f"  -> COVERAGE (is it documented) and CONTRADICTION (is the doc fought) are now SEPARATE axes;")
    print(f"     CONTESTED = high coverage + low alignment = the strong design-drift (doc exists, code fights it).")
    print(f"     UNDER-DOC = sparse docs but not fought (an honest 'undocumented' verdict, not 'incoherent').")

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axhline(0.6, color="#bbb", lw=1); ax.axvline(0.5, color="#bbb", lw=1)
    ax.add_patch(plt.Rectangle((0.5, 0.6), 0.55, 0.5, color="#e7f3ea", zorder=0))
    ax.add_patch(plt.Rectangle((0.5, 0), 0.55, 0.6, color="#f7e3e0", zorder=0))
    for n in names:
        r = R[n]
        ax.scatter(r["cov"], r["align"], s=40 + r["nc"] / 3, color=BLUE if r["aaa"] else GREEN,
                   edgecolor="k", lw=0.5, zorder=3, alpha=0.85)
        ax.annotate(n[:26], (r["cov"], r["align"]), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points", color=NAVY)
    ax.text(0.97, 1.04, "COHERENT", ha="right", fontsize=9, color=GREEN, weight="bold", transform=ax.transData)
    ax.text(0.97, 0.02, "CONTESTED (doc exists, code fights it)", ha="right", fontsize=9, color="#c0392b", weight="bold")
    ax.text(0.02, 1.04, "UNDER-DOCUMENTED", fontsize=9, color="#9a6a2f", weight="bold")
    ax.text(0.02, 0.02, "CHAOTIC", fontsize=9, color="#555", weight="bold")
    ax.set_xlabel("coverage rate  (commits whose change matches its OWN design)")
    ax.set_ylabel("alignment rate among covered  (1 - churn/anti-design vocabulary)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.1)
    ax.set_title("v3 coherence map: COVERAGE vs CONTRADICTION (green=CC plugin, blue=AAA/mixed)",
                 color=NAVY, fontsize=11)
    fig.tight_layout(); fig.savefig("sheaf_llm/coherence_v3_map.png", dpi=160); plt.close(fig)
    print("  wrote sheaf_llm/coherence_v3_map.png")


if __name__ == "__main__":
    main()
