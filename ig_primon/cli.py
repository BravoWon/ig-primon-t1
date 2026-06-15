"""``igprimon`` — the operational command line over the IG-PRIMON-T1 receipts.

    igprimon verify [--group G ...] [--quick] [--json]   run the anchor suite (default: all)
    igprimon list                                         list anchors, groups, and receipts
    igprimon run <receipt>                                run a top-level receipt's full output
    igprimon hwscan                                       scan the device; show the Tier-C/E map
    igprimon firewall [--kappa K] [--budget B] [--backend auto|cuda|cpu]
                                                          run the Precision-Certification Firewall
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# friendly name -> top-level module to execute with `python -m`
RECEIPTS = {
    "module-e": "module_e_radius_finding",
    "audit": "audit_independent",
    "ridge": "module_L_ridge_curvature",
    "sk": "module_L_SK_converse",
    "perceptron-replica": "module_L_perceptron_replica",
    "perceptron-replicon": "module_L_perceptron_replicon",
    "perceptron-curvature": "module_L_perceptron_curvature",
    "perceptron-finiteT": "module_L_perceptron_finiteT",
    "hw-firewall": "module_hw_firewall",
}


def _cmd_verify(args):
    from .harness import run_and_report
    return run_and_report(groups=args.group or None,
                          include_slow=not args.quick,
                          as_json=args.json)


def _cmd_list(args):
    from .anchors import ANCHORS, all_groups
    print("groups :", ", ".join(all_groups()))
    print("\nanchors:")
    for a in ANCHORS:
        flag = " (slow)" if a.slow else ""
        print(f"  {a.status:>3} {a.group:<14} {a.id:<22} {a.desc}{flag}")
    print("\nreceipts (igprimon run <name>):")
    for name, mod in RECEIPTS.items():
        print(f"  {name:<22} -> {mod}.py")
    return 0


def _cmd_run(args):
    mod = RECEIPTS.get(args.receipt)
    if mod is None:
        print(f"unknown receipt '{args.receipt}'. choices: {', '.join(RECEIPTS)}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, "-m", mod])


def _cmd_hwscan(args):
    from .hardware import scan, format_scan
    print(format_scan(scan()))
    return 0


def _cmd_firewall(args):
    from .firewall import run_firewall, format_firewall
    res = run_firewall(kappa=args.kappa, explore_budget=args.budget, backend=args.backend)
    print(format_firewall(res))
    return 0 if res.firewall_intact else 1


def _cmd_precision_matrix(args):
    from .precision import OPS, build_matrix, format_matrix, sweep_reduction, format_sweep
    ops = list(OPS) if args.op == "all" else [args.op]
    cells = build_matrix(ops, size=args.size, budget=args.budget, iters=args.iters)
    if args.json:
        import json
        from dataclasses import asdict
        print(json.dumps({"size": args.size, "budget": args.budget,
                          "cells": [asdict(c) for c in cells]}, indent=2, default=str))
        return 0
    print(format_matrix(cells, args.size, args.budget))
    if args.sweep:
        for op in ops:
            if OPS[op].reduces:
                be, rows, cross = sweep_reduction(op, [256, 512, 1024, 2048, 4096, 8192],
                                                  budget=args.budget)
                print(format_sweep(be, rows, cross, op, args.budget))
    return 0


def _cmd_precision_bf16(args):
    from . import torch_precision as T
    if not T.available():
        print("bf16 pass needs torch+CUDA. Install: "
              "pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 2
    cells = T.bf16_matrix(size=args.size, budget=args.budget)
    print(T.format_bf16_matrix(cells, args.size, args.budget))
    print(T.format_range_check(T.layernorm_range_check()))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="igprimon",
                                description="Operational layer over the IG-PRIMON-T1 research receipts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="run the anchor verification suite (Tier-C / CPU)")
    v.add_argument("--group", action="append", help="restrict to a group (repeatable)")
    v.add_argument("--quick", action="store_true", help="skip slow high-precision anchors")
    v.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    v.set_defaults(func=_cmd_verify)

    l = sub.add_parser("list", help="list anchors, groups, and receipts")
    l.set_defaults(func=_cmd_list)

    r = sub.add_parser("run", help="run a top-level receipt's full certification output")
    r.add_argument("receipt", choices=list(RECEIPTS))
    r.set_defaults(func=_cmd_run)

    h = sub.add_parser("hwscan", help="scan the device and show the Tier-C / Tier-E map")
    h.set_defaults(func=_cmd_hwscan)

    f = sub.add_parser("firewall", help="run the Precision-Certification Firewall (Tier-E vs Tier-C)")
    f.add_argument("--kappa", type=float, default=0.0)
    f.add_argument("--budget", type=float, default=1e-5,
                   help="loose FP32-grade exploration tolerance (Tier-C certify budget is the float32 noise floor)")
    f.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="auto")
    f.set_defaults(func=_cmd_firewall)

    pm = sub.add_parser("precision-matrix",
                        help="certify inference primitives (GEMM/softmax/norm/attention) across {device}x{precision}")
    pm.add_argument("--op", choices=["gemm", "softmax", "layernorm", "attention", "all"], default="all")
    pm.add_argument("--size", type=int, default=1024)
    pm.add_argument("--budget", type=float, default=1e-3, help="inference safety budget (relative error)")
    pm.add_argument("--iters", type=int, default=10)
    pm.add_argument("--sweep", action="store_true", help="reduction-width dependence for reduction ops")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=_cmd_precision_matrix)

    bf = sub.add_parser("precision-bf16",
                        help="bf16 pass (torch): real-inference precision map + the LayerNorm range inversion")
    bf.add_argument("--size", type=int, default=1024)
    bf.add_argument("--budget", type=float, default=1e-3)
    bf.set_defaults(func=_cmd_precision_bf16)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
