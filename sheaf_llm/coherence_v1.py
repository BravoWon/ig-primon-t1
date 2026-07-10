#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v1 -- the sheaf-gluing idea on REAL repos.

Thesis (user's): a repo encodes intent at two scales -- DESIGN (.md docs = stated intent dictionary,
the 'global section') and EVOLUTION (commits = local realizations). Coherence = how well the commits
GLUE to the stated design. This is the design<->evolution gluing metric.

  design dictionary  = embedded sections of README/ARCHITECTURE/DESIGN/docs .md  (the inherited vocabulary)
  commit embeddings  = embed(message + changed file paths)
  glue(commit)       = max cosine similarity to any design section
  coherence(repo)    = mean_commits glue
  drift trace        = glue(commit) over commit time

BASELINE (carried INSIDE the sweep -- the H1 lesson: never plot an unvalidated 'coherence'):
  cross-repo control = each commit's glue to its OWN design vs to OTHER repos' pooled design.
  SIGNAL = coherence_own - coherence_other. If ~0, the metric is just generic text similarity
  (commits match any docs equally) -> FALSIFIED. If > 0, commits genuinely align to THEIR design.

Embeds with sentence-transformers/all-MiniLM-L6-v2 via transformers (mean pooling; no extra install).

    python coherence_v1.py
"""
import os, re, subprocess, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_DEV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("COHERENCE_REPOS", os.path.join(_DEV_DIR, "_coherence_repos"))
REPOS = [d for d in sorted(glob.glob(ROOT + "/*")) if os.path.isdir(d + "/.git")]
_ISOZ = os.environ.get("COHERENCE_SELF_REPO", os.path.join(_DEV_DIR, "proj-0", "isoZ"))
if os.path.isdir(os.path.join(_ISOZ, ".git")):
    REPOS += [_ISOZ]                                     # local self-anchor
MAXC, MAXSEC = 500, 150
DESIGN_PAT = re.compile(r"(readme|architect|design|contribut|overview|docs/|adr|spec|roadmap|manifesto)", re.I)
NAVY, GREEN, RED = "#15293f", "#1e7d34", "#c0392b"

_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()


@torch.no_grad()
def embed(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        b = [t[:512] for t in texts[i:i + bs]]
        enc = _tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        h = _mod(**enc).last_hidden_state
        m = enc.attention_mask[..., None].float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384))


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          errors="ignore", timeout=120).stdout


def design_sections(repo):
    files = [f for f in git(repo, "ls-files", "*.md").splitlines() if DESIGN_PAT.search(f)]
    files = sorted(files, key=lambda f: (f.count("/"), len(f)))[:60]      # prefer shallow/design-y
    secs = []
    for f in files:
        p = os.path.join(repo, f)
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        parts = re.split(r"\n#{1,3}\s", txt)
        for s in parts:
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 80:
                secs.append(s[:600])
    if len(secs) > MAXSEC:
        idx = np.linspace(0, len(secs) - 1, MAXSEC).astype(int)
        secs = [secs[i] for i in idx]
    return secs


def commits(repo):
    raw = git(repo, "log", f"-n{MAXC}", "--no-merges", "--name-only",
              "--format=%x1e%H%x1f%s %b%x1f")
    recs = []
    for rec in raw.split("\x1e"):
        if "\x1f" not in rec:
            continue
        parts = rec.split("\x1f")
        if len(parts) < 3:
            continue
        msg = re.sub(r"\s+", " ", parts[1]).strip()[:300]
        files = [l.strip() for l in parts[2].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        recs.append(f"{msg}  || files: {dirs}")
    return list(reversed(recs))                                          # chronological


def main():
    print(f"[coherence v1]  dev={DEV}  repos={len(REPOS)}  (design<->evolution gluing + cross-repo baseline)")
    data = {}
    for repo in REPOS:
        name = os.path.basename(repo)
        secs, coms = design_sections(repo), commits(repo)
        if len(secs) < 3 or len(coms) < 5:
            print(f"  skip {name}: secs={len(secs)} commits={len(coms)}"); continue
        D, C = embed(secs), embed(coms)
        glue = (C @ D.T).max(1)                                          # per-commit best design match
        data[name] = dict(D=D, C=C, glue=glue, n=len(coms), nsec=len(secs))
        print(f"  {name[:42]:42}  commits={len(coms):4}  design_secs={len(secs):3}  coherence_own={glue.mean():.3f}")

    names = list(data)
    allD = {n: data[n]["D"] for n in names}
    rows = []
    for n in names:
        C = data[n]["C"]
        others = np.concatenate([allD[m] for m in names if m != n])
        idx = np.random.default_rng(0).choice(len(others), min(400, len(others)), replace=False)
        glue_other = (C @ others[idx].T).max(1)
        own = data[n]["glue"].mean(); oth = glue_other.mean()
        orphan = float((data[n]["glue"] < 0.30).mean())
        rows.append((n, data[n]["n"], own, oth, own - oth, orphan))

    rows.sort(key=lambda r: -r[4])
    print(f"\n{'repo':42}{'commits':>8}{'coh_own':>9}{'baseline':>9}{'SIGNAL':>8}{'orphan%':>8}")
    for n, nc, own, oth, sig, orp in rows:
        print(f"{n[:42]:42}{nc:>8}{own:>9.3f}{oth:>9.3f}{sig:>+8.3f}{orp*100:>7.0f}%")
    sigs = np.array([r[4] for r in rows])
    print(f"\nVERDICT: mean SIGNAL (own-baseline) = {sigs.mean():+.3f}; "
          f"{(sigs>0.02).sum()}/{len(sigs)} repos show real design<->evolution gluing above generic similarity.")
    print(f"  -> {'METRIC CAPTURES REAL COHERENCE (beats cross-repo baseline)' if sigs.mean() > 0.02 else 'METRIC ~= generic text similarity (FALSIFIED, like H1)'}")

    # plot 1: own vs baseline, sorted by signal
    fig, ax = plt.subplots(figsize=(10, 5)); y = np.arange(len(rows)); w = 0.4
    ax.barh(y + w/2, [r[2] for r in rows], w, color=GREEN, label="coherence (own design)")
    ax.barh(y - w/2, [r[3] for r in rows], w, color="#9aa7b2", label="baseline (other repos' design)")
    ax.set_yticks(y); ax.set_yticklabels([r[0][:34] for r in rows], fontsize=7)
    ax.set_xlabel("mean glue (max cosine commit->design)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Design<->evolution coherence vs cross-repo baseline (gap = real gluing signal)",
                 color=NAVY, fontsize=10); fig.tight_layout()
    fig.savefig("sheaf_llm/coherence_v1_bars.png", dpi=160); plt.close(fig)

    # plot 2: drift traces (smoothed glue over commit time) for repos with >=40 commits
    rich = [n for n in names if data[n]["n"] >= 40][:9]
    if rich:
        cols = 3; rws = int(np.ceil(len(rich) / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(12, 2.6 * rws), squeeze=False)
        for k, n in enumerate(rich):
            ax = axes[k // cols][k % cols]; g = data[n]["glue"]
            win = max(3, len(g) // 30); sm = np.convolve(g, np.ones(win)/win, mode="valid")
            ax.plot(sm, color=GREEN, lw=1.3); ax.axhline(g.mean(), ls=":", color="#999", lw=0.8)
            ax.set_title(f"{n[:30]}  (n={data[n]['n']})", fontsize=7.5, color=NAVY)
            ax.set_ylim(0.1, 0.7); ax.tick_params(labelsize=6)
        for k in range(len(rich), rws*cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle("Design-coherence DRIFT traces over commit time (smoothed glue; dips = drift)",
                     color=NAVY, fontsize=11); fig.tight_layout()
        fig.savefig("sheaf_llm/coherence_v1_traces.png", dpi=150); plt.close(fig)
    print("  wrote sheaf_llm/coherence_v1_bars.png, sheaf_llm/coherence_v1_traces.png")


if __name__ == "__main__":
    main()
