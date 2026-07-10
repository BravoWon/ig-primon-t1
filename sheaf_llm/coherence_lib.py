#!/usr/bin/env python
"""Shared plumbing for the design<->evolution coherence sweeps (coherence_v2..v5).

Extracts the identical helpers that were copy-pasted across versions: repo discovery, the MiniLM
embedder, git access, design-section extraction, commit parsing, common-component de-boilerplating, and
the (lazy) LLM-judge loader. Each `coherence_vN.py` imports from here and keeps only its OWN metric +
plot + (for v4/v5) its experiment-specific judge prompt. The standalone gate receipts are deliberately
NOT refactored to share code -- their self-containedness is a program feature.

commits() returns (hash, msg, text) -- the superset; callers take the fields they need.
"""
import os, re, subprocess, glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_DEV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("COHERENCE_REPOS", os.path.join(_DEV_DIR, "_coherence_repos"))
ISOZ = os.environ.get("COHERENCE_SELF_REPO", os.path.join(_DEV_DIR, "proj-0", "isoZ"))
AAA = {"godotengine_godot", "bevyengine_bevy", "microsoft_TypeScript", "facebook_react", "obsproject_obs-studio"}
MAXC, MAXSEC, KREMOVE = 500, 150, 3
DESIGN_PAT = re.compile(r"(readme|architect|design|contribut|overview|docs/|adr|spec|roadmap|manifesto)", re.I)
NAVY, GREEN, BLUE, RED, AMBER = "#15293f", "#1e7d34", "#2c6fbb", "#c0392b", "#9a6a2f"


def discover_repos():
    repos = [d for d in sorted(glob.glob(ROOT + "/*")) if os.path.isdir(d + "/.git")]
    return repos + ([ISOZ] if os.path.isdir(os.path.join(ISOZ, ".git")) else [])


REPOS = discover_repos()

_etok = _emod = None                                     # lazy: importing this lib must stay cheap


def _ensure_embedder():
    global _etok, _emod
    if _emod is None:
        _etok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _emod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()


@torch.no_grad()
def embed(texts, bs=128):
    _ensure_embedder()
    out = []
    for i in range(0, len(texts), bs):
        enc = _etok([t[:512] for t in texts[i:i + bs]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEV)
        h = _emod(**enc).last_hidden_state
        m = enc.attention_mask[..., None].float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 384))


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True,
                          errors="ignore", timeout=180).stdout


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
    """Return [(hash, msg, text), ...] in chronological order. text = 'msg  || files: <dirs>'."""
    raw = git(repo, "log", f"-n{MAXC}", "--no-merges", "--name-only", "--format=%x1e%H%x1f%s %b%x1f")
    out = []
    for rec in raw.split("\x1e"):
        parts = rec.split("\x1f")
        if len(parts) < 3:
            continue
        msg = re.sub(r"\s+", " ", parts[1]).strip()[:300]
        files = [l.strip() for l in parts[2].splitlines() if l.strip()][:20]
        dirs = " ".join(sorted({("/".join(f.split("/")[:2]) if "/" in f else f) for f in files}))
        out.append((parts[0].strip(), msg, f"{msg}  || files: {dirs}"))
    return list(reversed(out))


def deboilerplate(vecs, k):
    mu = vecs.mean(0)
    X = vecs - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[:k]                                           # top-k generic directions
    X = X - (X @ U.T) @ U
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def load_judge(judge_id="Qwen/Qwen2.5-3B-Instruct"):
    """Lazy: only v4/v5 need the LLM judge, so it is NOT loaded at import time."""
    print(f"[coherence] loading judge {judge_id} ...")
    tok = AutoTokenizer.from_pretrained(judge_id)
    mod = AutoModelForCausalLM.from_pretrained(
        judge_id, torch_dtype=(torch.float16 if DEV == "cuda" else torch.float32)).to(DEV).eval()
    return tok, mod
