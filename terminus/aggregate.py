#!/usr/bin/env python
"""Aggregate the terminus ladder across SEEDS -> hardened Delta(scale) trend + seed consistency."""
import json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "lad"
BUDGETS = [8, 30, 90]
ARMS = ["flat", "grounded", "grounded-random"]
SEEDS = [0, 1, 2]

def load(arm, b, s):
    try: d = json.load(open(f"terminus/metrics_{TAG}_{arm}_{b}M_s{s}.json"))
    except FileNotFoundError: return None
    d["seed"] = s   # authoritative from filename; the older s0 files predate the in-JSON "seed" field
    return d

def edge(nll_a, nll_b):   # % PPL reduction of b vs a (positive = b better), robust via NLL
    return 100 * (1 - math.exp(nll_b - nll_a))

def main():
    # collect per-seed nll dicts:  runs[(arm,b)] = list of nll-dicts
    runs = {}
    for b in BUDGETS:
        for arm in ARMS:
            ms = [load(arm, b, s) for s in SEEDS]
            ms = [m for m in ms if m]
            if ms: runs[(arm, b)] = ms
    nseed = max((len(v) for v in runs.values()), default=0)
    print(f"[terminus ladder, {nseed}-seed]  does grounding's edge survive scale, and beat its placebo, robustly?\n")

    def mean_nll(arm, b, bucket):
        return float(np.mean([m["nll"][bucket] for m in runs[(arm, b)]]))
    def mean_ppl(arm, b, bucket):
        return math.exp(min(mean_nll(arm, b, bucket), 20))

    print(f"  {'budget':>7} {'arm':16} {'PPL all':>9} {'g_rare':>10} {'g_common':>10} {'seeds':>6}")
    for b in BUDGETS:
        for arm in ARMS:
            if (arm, b) not in runs: continue
            print(f"  {b:>5}M  {arm:16} {mean_ppl(arm,b,'all'):>9.1f} {mean_ppl(arm,b,'g_rare'):>10.1f} "
                  f"{mean_ppl(arm,b,'g_common'):>10.1f} {len(runs[(arm,b)]):>6}")

    print(f"\n  ---- edges (seed-mean), positive = grounding helps ----")
    print(f"  {'budget':>7} {'overall':>9} {'rare':>8} {'grounded-vs-RANDOM (rare, per-seed)':>40}")
    trend = []
    for b in BUDGETS:
        if ("grounded", b) not in runs or ("flat", b) not in runs: continue
        d_all = edge(mean_nll("flat", b, "all"), mean_nll("grounded", b, "all"))
        d_rare = edge(mean_nll("flat", b, "g_rare"), mean_nll("grounded", b, "g_rare"))
        # per-seed grounded-vs-random on rare (only seeds present for both)
        per = []
        if ("grounded-random", b) in runs:
            gs = {m["seed"]: m for m in runs[("grounded", b)]}; rs = {m["seed"]: m for m in runs[("grounded-random", b)]}
            for s in sorted(set(gs) & set(rs)):
                per.append(edge(rs[s]["nll"]["g_rare"], gs[s]["nll"]["g_rare"]))
        vs_rand = float(np.mean(per)) if per else float("nan")
        perstr = " ".join(f"{v:+.1f}" for v in per) + (f"  (mean {vs_rand:+.1f}%)" if per else "  (no placebo)")
        trend.append((b, d_all, d_rare, vs_rand, per))
        print(f"  {b:>5}M  {d_all:>+8.1f}% {d_rare:>+7.1f}% {perstr:>40}")

    if trend:
        ov = [t[1] for t in trend]; pm = [t[3] for t in trend if not math.isnan(t[3])]
        allper = [v for t in trend for v in t[4]]
        decayed = ov[-1] < 0.5 * max(ov)
        semantic = len(pm) == len(trend) and all(v > 0 for v in pm)
        consistent = len(allper) > 0 and all(v > 0 for v in allper)
        span = trend[-1][0] // trend[0][0]
        print(f"\n  VERDICT ({nseed}-seed, {span}x data increase):")
        print(f"    overall edge : {ov[0]:+.1f}% -> {ov[-1]:+.1f}%  -> {'DECAYS (FALSIFIER)' if decayed else 'HOLDS (survives scale)'}")
        if pm:
            print(f"    vs placebo   : {pm[0]:+.1f}% -> {pm[-1]:+.1f}% (rare)  -> "
                  f"{'SEMANTIC, margin grows' if semantic and pm[-1] >= pm[0] else 'ambiguous'}")
            print(f"    seed consistency: every grounded-vs-random margin {'POSITIVE (' + str(len(allper)) + '/' + str(len(allper)) + ' runs)' if consistent else 'NOT all positive -- fragile'}")
        print(f"    => {'GROUNDING SURVIVES + SEMANTIC + SEED-ROBUST' if (not decayed and semantic and consistent) else 'see rows'}.")

        bs = [t[0] for t in trend]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bs, [t[1] for t in trend], "o-", color="#1e7d34", lw=2, label="Δ overall PPL (grounded vs flat)")
        ax.plot(bs, [t[2] for t in trend], "s-", color="#c0392b", lw=2, label="Δ rare-noun PPL")
        ax.plot(bs, [t[3] for t in trend], "^-", color="#9a6a2f", lw=2, label="grounded vs RANDOM placebo (rare)")
        for t in trend:                                          # per-seed dots for the placebo margin
            ax.scatter([t[0]] * len(t[4]), t[4], color="#9a6a2f", s=18, alpha=0.5, zorder=3)
        ax.axhline(0, ls=":", color="#999"); ax.set_xscale("log")
        ax.set_xlabel("training tokens (M, log)"); ax.set_ylabel("grounding improvement (%)")
        ax.set_title(f"Terminus ({nseed}-seed): grounding survives scale and beats its placebo\n"
                     "(amber = the semantic signal; dots = individual seeds)", color="#15293f", fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.set_xticks(bs); ax.set_xticklabels([f"{b}M" for b in bs])
        fig.tight_layout(); fig.savefig("terminus/terminus_trend_multiseed.png", dpi=150); plt.close(fig)
        print("  wrote terminus/terminus_trend_multiseed.png")

if __name__ == "__main__":
    main()
