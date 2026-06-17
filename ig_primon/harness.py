"""Harness — run the anchors and report drift. The engine behind ``igprimon verify``."""

from __future__ import annotations

import json
import time

from .anchors import AnchorResult, get_anchors, all_groups

# Note: no hardcoded groups needed; precision-depth (and others) are discovered
# dynamically via anchors.py ANCHORS list (filter in get_anchors). Per plan Task 4,
# no edit required for the new group -- "if needed" does not apply.


def run(groups=None, include_slow=True):
    """Run the selected anchors, return a list of AnchorResult."""
    specs = get_anchors(groups=groups, include_slow=include_slow)
    results = []
    for spec in specs:
        results.append(spec.run())
    return results


def summarize(results):
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    return {"total": total, "passed": passed, "failed": len(failed),
            "all_ok": len(failed) == 0}


def format_table(results, elapsed=None):
    lines = []
    lines.append("=" * 84)
    lines.append("IG-PRIMON-T1 - anchor verification  (Tier-C: CPU / mpmath - the sole [V] authority)")
    lines.append("=" * 84)
    last_group = None
    for r in results:
        if r.group != last_group:
            lines.append(f"\n[{r.group}]")
            last_group = r.group
        mark = "PASS" if r.ok else "FAIL"
        flag = " (slow)" if r.slow else ""
        lines.append(f"  [{mark}] {r.status:>3} {r.id:<20} {r.desc}{flag}")
        if r.value or r.expected:
            lines.append(f"         {r.value}   |   target: {r.expected}")
        if r.detail:
            lines.append(f"         {r.detail}")
        if r.error:
            lines.append(f"         ERROR: {r.error}")
    s = summarize(results)
    lines.append("\n" + "-" * 84)
    tail = f"{s['passed']}/{s['total']} anchors reproduced"
    if s["failed"]:
        tail += f"  --  {s['failed']} FAILED"
    if elapsed is not None:
        tail += f"   ({elapsed:.1f}s)"
    lines.append(tail)
    lines.append("=" * 84)
    return "\n".join(lines)


def to_json(results, elapsed=None):
    s = summarize(results)
    return json.dumps({
        "summary": {**s, "elapsed_s": elapsed},
        "anchors": [
            {"id": r.id, "group": r.group, "status": r.status, "ok": r.ok,
             "value": r.value, "expected": r.expected, "detail": r.detail,
             "slow": r.slow, "error": r.error}
            for r in results
        ],
    }, indent=2)


def run_and_report(groups=None, include_slow=True, as_json=False):
    """Run anchors, print a report, return process exit code (0 = all reproduced)."""
    t0 = time.perf_counter()
    results = run(groups=groups, include_slow=include_slow)
    elapsed = time.perf_counter() - t0
    if as_json:
        print(to_json(results, elapsed))
    else:
        print(format_table(results, elapsed))
    return 0 if summarize(results)["all_ok"] else 1
