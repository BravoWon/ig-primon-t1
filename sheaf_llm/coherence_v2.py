#!/usr/bin/env python
"""codebase design<->evolution COHERENCE, v2 -- de-boilerplated + orphan-drift, CC plugins + AAA/mixed.

v1 worked but was small: absolute glue (0.4-0.6) was dominated by domain-generic similarity (everything
is "CC plugin" text). v2 fixes that and sharpens the drift signal:

  1. DE-BOILERPLATE (common-component removal, Arora's "all-but-the-top"): pool ALL embeddings, remove
     the mean + top-K principal directions (the generic "this-is-a-software-repo" subspace), renormalize.
     What's left is repo-distinctive content -> the own-vs-baseline gap should widen.
  2. PER-COMMIT de-baselined signal: signal(c) = glue_own(c) - glue_otherpool(c). coherence = mean signal.
  3. ORPHAN-DRIFT (the sharper metric, promoted to primary): orphan = signal(c) <= 0 (a commit that
     matches some OTHER repo's design as well as its own -> not distinctively tied to its stated design).
     orphan% = fraction orphaned; drift trace = rolling signal over commit time.
  4. WIDEN: AAA/mixed repos (godot, bevy, TypeScript, react, obs) added to break the CC-plugin monoculture
     and test generality / lower the baseline.

    python coherence_v2.py
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
    recs = []
    for rec in raw.split("\x1e"):
        parts = rec.split("\x1f")
        if len(parts) < 3:
            continue
        msg = re.sub(r"\s+", " ", parts[1]).strip()[:300]
        files = [l.strip() for l in parts[2].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        recs.append(f"{msg}  || files: {dirs}")
    return list(reversed(recs))


def deboilerplate(vecs, k):
    mu = vecs.mean(0)
    X = vecs - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:k]                                            # top-k generic directions
    X = X - (X @ U.T) @ U
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-8, None)


def main():
    print(f"[coherence v2]  dev={DEV}  de-boilerplate(remove mean+top{KREMOVE})  orphan-drift  CC+AAA")
    raw = {}
    for repo in REPOS:
        name = os.path.basename(repo)
        secs, coms = design_sections(repo), commits(repo)
        if len(secs) < 3 or len(coms) < 8:
            print(f"  skip {name}: secs={len(secs)} commits={len(coms)}"); continue
        raw[name] = dict(secs=secs, coms=coms)
    # embed everything, then de-boilerplate jointly
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

    # plot 1: coherence-signal bar, CC vs AAA colored, + orphan% annotation
    fig, ax = plt.subplots(figsize=(10, 6)); y = np.arange(len(order))
    ax.barh(y, [rows[n]["coh"] for n in order],
            color=[BLUE if rows[n]["aaa"] else GREEN for n in order])
    for i, n in enumerate(order):
        ax.text(rows[n]["coh"] + 0.002, i, f"orphan {rows[n]['orphan']*100:.0f}%", va="center", fontsize=6, color="#555")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n[:32] for n in order], fontsize=7)
    ax.set_xlabel("de-boilerplated coherence signal (mean per-commit own-minus-baseline)")
    ax.set_title("Design<->evolution coherence v2 (de-boilerplated): green=CC plugin, blue=AAA/mixed",
                 color=NAVY, fontsize=10); fig.tight_layout()
    fig.savefig("sheaf_llm/coherence_v2_bars.png", dpi=160); plt.close(fig)

    # plot 2: orphan-drift traces (rolling signal over commit time) for the most-commits repos
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
