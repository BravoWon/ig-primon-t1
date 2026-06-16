#!/usr/bin/env python3
"""
module_hw_firewall.py  --  IG-PRIMON-T1 hardware-execution track, H1 flagship receipt.

Realizes the Precision-Certification Firewall named in T1_hw_optimization_doctrine, around its
actual thesis -- agreement is not verification. A fast FP32 Tier-E explorer proposes candidates;
the slow exact Tier-C (mpmath, dps>=50) authority adjudicates. The load-bearing case is a
NEAR-MISS kernel wrong only ~3x the float32 epsilon: it passes an FP32-grade tolerance and would
fool an FP32 'certify-by-agreement' reference (whose own error is ~1 eps), but Tier-C REJECTS it
because its deviation exceeds the FP32 noise floor (the precision teeth). On this machine Tier-E
is the NVIDIA RTX 5070 (Blackwell, sm_120) via CuPy FP32
-- the doctrine was written for a Snapdragon Adreno/Hexagon device; the firewall is
hardware-agnostic and is detected at runtime (see ig_primon.hardware).

Target invariant: the Gardner capacity alpha_c(kappa) = 1 / E_t[(t+kappa)^2 ; t>=-kappa],
with alpha_c(0) = 2 exactly -- the same anchor module_L_perceptron_replica.py pins on the CPU.

[E-hw] discipline: nothing printed here is [V]. A Tier-E number is exploratory by
construction; only the Tier-C reproduction (within budget) would license a [V] tag.

Run:  python module_hw_firewall.py     (or:  igprimon firewall)
"""

from ig_primon.firewall import run_firewall, format_firewall
from ig_primon.hardware import scan, format_scan

if __name__ == "__main__":
    print(format_scan(scan()))
    print()
    res = run_firewall(kappa=0.0)
    print(format_firewall(res))
    raise SystemExit(0 if res.firewall_intact else 1)
