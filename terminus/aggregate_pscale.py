#!/usr/bin/env python
"""Aggregate the terminus PARAM-scale ladder across SEEDS -> does grounding's edge (and its semantic
vs-random margin) survive PARAMETER scaling, robustly?  x-axis = params, tokens FIXED at 30M.
Companion to aggregate.py (data-scale). Mirrors its per-seed grounded-vs-random SEED-ROBUST logic.

Honest confound (printed): at FIXED tokens a bigger model is more undertrained, which FAVORS grounding --
so read the load-bearing claim as 'survives / does not decay', not 'grows because of params'."""
import sys, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BUDGET = 30                                   # fixed token budget (M) across all param points
SIZES = ["ps014", "ps040", "ps090"]           # small -> large (tag encodes size)
ARMS = ["flat", "grounded", "grounded-random"]
SEEDS = [0, 1, 2]

def load(tag, arm, s):
    try: d = json.load(open(f"terminus/metrics_{tag}_{arm}_{BUDGET}M_s{s}.json"))
    except FileNotFoundError: return None
    d["seed"] = s                             # authoritative from filename
    return d

def edge(nll_a, nll_b):                        # % PPL reduction of b vs a (positive = b better)
    return 100 * (1 - math.exp(nll_b - nll_a))

def main():
    runs = {}
    for tag in SIZES:
        for arm in ARMS:
            ms = [m for m in (load(tag, arm, s) for s in SEEDS) if m]
            if ms: runs[(tag, arm)] = ms
    if not runs:
        print("no param-scale metrics found yet."); return
    nseed = max(len(v) for v in runs.values())
    print(f"[terminus PARAM-scale ladder, {nseed}-seed, {BUDGET}M tokens fixed]  "
          f"does grounding's edge survive PARAM scale, and beat its placebo, robustly?\n")

    def mnll(tag, arm, bucket): return float(np.mean([m["nll"][bucket] for m in runs[(tag, arm)]]))
    def mppl(tag, arm, bucket): return math.exp(min(mnll(tag, arm, bucket), 20))

    print(f"  {'params':>8} {'arm':16} {'PPL all':>9} {'g_rare':>11} {'g_common':>10} {'seeds':>6}")
    for tag in SIZES:
        if (tag, "flat") not in runs: continue
        npar = runs[(tag, "flat")][0]["nparams"] / 1e6
        for arm in ARMS:
            if (tag, arm) not in runs: continue
            print(f"  {runs[(tag,arm)][0]['nparams']/1e6:>6.0f}M  {arm:16} {mppl(tag,arm,'all'):>9.1f} "
                  f"{mppl(tag,arm,'g_rare'):>11.1f} {mppl(tag,arm,'g_common'):>10.1f} {len(runs[(tag,arm)]):>6}")

    print(f"\n  ---- edges (seed-mean), positive = grounding helps ----")
    print(f"  {'params':>8} {'overall':>9} {'rare':>8} {'grounded-vs-RANDOM (rare, per-seed)':>40}")
    trend = []
    for tag in SIZES:
        if (tag, "grounded") not in runs or (tag, "flat") not in runs: continue
        npar = runs[(tag, "flat")][0]["nparams"] / 1e6
        d_all  = edge(mnll(tag, "flat", "all"),    mnll(tag, "grounded", "all"))
        d_rare = edge(mnll(tag, "flat", "g_rare"), mnll(tag, "grounded", "g_rare"))
        per = []
        if (tag, "grounded-random") in runs:
            gs = {m["seed"]: m for m in runs[(tag, "grounded")]}
            rs = {m["seed"]: m for m in runs[(tag, "grounded-random")]}
            for s in sorted(set(gs) & set(rs)):
                per.append(edge(rs[s]["nll"]["g_rare"], gs[s]["nll"]["g_rare"]))
        vs_rand = float(np.mean(per)) if per else float("nan")
        perstr = " ".join(f"{v:+.1f}" for v in per) + (f"  (mean {vs_rand:+.1f}%)" if per else "  (no placebo)")
        trend.append((npar, d_all, d_rare, vs_rand, per))
        print(f"  {npar:>6.0f}M  {d_all:>+8.1f}% {d_rare:>+7.1f}% {perstr:>40}")

    if trend:
        ov = [t[1] for t in trend]; pm = [t[3] for t in trend if not math.isnan(t[3])]
        allper = [v for t in trend for v in t[4]]
        margin_decayed = len(pm) >= 2 and pm[-1] < 0.5 * max(pm)
        consistent = len(allper) > 0 and all(v > 0 for v in allper)
        span = trend[-1][0] / trend[0][0]
        print(f"\n  VERDICT ({nseed}-seed, {span:.0f}x params, {BUDGET}M tokens fixed):")
        print(f"    overall edge (grounded vs flat): {ov[0]:+.1f}% -> {ov[-1]:+.1f}%")
        if pm:
            print(f"    SEMANTIC margin (grounded vs RANDOM, rare): {pm[0]:+.1f}% -> {pm[-1]:+.1f}%  -> "
                  f"{'DECAYS (FALSIFIER: better-learned embeddings erase it)' if margin_decayed else 'HOLDS/grows (survives param scale)'}")
            print(f"    seed consistency: every grounded-vs-random margin "
                  f"{'POSITIVE (' + str(len(allper)) + '/' + str(len(allper)) + ' runs)' if consistent else 'NOT all positive -- fragile'}")
        print(f"    => {'GROUNDING SURVIVES PARAM SCALE + SEMANTIC + SEED-ROBUST' if (not margin_decayed and consistent) else 'see rows'}.")
        print(f"    (confound: fixed tokens => bigger models more undertrained, which favors grounding;")
        print(f"     the clean claim is 'survives/does not decay', not 'grows because of params'.)")

        xp = [t[0] for t in trend]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xp, [t[1] for t in trend], "o-", color="#1e7d34", lw=2, label="Δ overall PPL (grounded vs flat)")
        ax.plot(xp, [t[2] for t in trend], "s-", color="#c0392b", lw=2, label="Δ rare-noun PPL")
        ax.plot(xp, [t[3] for t in trend], "^-", color="#9a6a2f", lw=2, label="grounded vs RANDOM placebo (rare)")
        for t in trend:
            ax.scatter([t[0]] * len(t[4]), t[4], color="#9a6a2f", s=18, alpha=0.5, zorder=3)
        ax.axhline(0, ls=":", color="#999"); ax.set_xscale("log")
        ax.set_xlabel("model parameters (M, log)"); ax.set_ylabel("grounding improvement (%)")
        ax.set_title(f"Terminus PARAM-scale ({nseed}-seed, {BUDGET}M tokens fixed): grounding survives parameter scaling\n"
                     "(amber = semantic signal vs placebo; dots = individual seeds)", color="#15293f", fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.set_xticks(xp); ax.set_xticklabels([f"{p:.0f}M" for p in xp])
        fig.tight_layout(); fig.savefig("terminus/terminus_pscale_multiseed.png", dpi=150); plt.close(fig)
        print("  wrote terminus/terminus_pscale_multiseed.png")

if __name__ == "__main__":
    main()
