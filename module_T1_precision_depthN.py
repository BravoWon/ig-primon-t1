"""T1_precision_map_v0_2 depth-N error composition receipt.

Implements the locked pre-registration: certified error curves through depth
on small decoder-only models using the Precision-Certification Firewall.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ig_primon.firewall import run_firewall
from ig_primon.hardware import scan

def compute_block_error(
    prev_error: np.ndarray,
    block_jacobian_approx: np.ndarray,
    local_delta: np.ndarray,
) -> np.ndarray:
    """Implements ε_{l+1} = (I + J_f) ε_l + δ_l per the locked pre-reg."""
    return prev_error + block_jacobian_approx @ prev_error + local_delta

def run_depth_error_map(model_name: str = "gpt2-small", prec: str = "bf16") -> dict:
    """Stub for depth error map using firewall/hardware per plan Task 3."""
    dm = scan()
    res = run_firewall(kappa=0.0, backend=dm.tier_e_backend)
    return {"firewall": res, "device": dm}

def main() -> int:
    """Entry point when run as `python -m module_T1_precision_depthN`."""
    print("T1_precision_map_v0_2 receipt skeleton")
    # TODO: implement per plan tasks
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
