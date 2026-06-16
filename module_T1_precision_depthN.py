"""T1_precision_map_v0_2 depth-N error composition receipt.

Implements the locked pre-registration: certified error curves through depth
on small decoder-only models using the Precision-Certification Firewall.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from mpmath import mp


def compute_block_error(
    prev_error: np.ndarray,
    block_jacobian_approx: np.ndarray,
    local_delta: np.ndarray,
    subsample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Implements ε_{l+1} = (I + J_f) ε_l + δ_l per the locked pre-reg.

    Supports optional subsampling of vector elements (tokens/positions/dims)
    for later pre-registered subsampled certification (mitigates mpmath cost).
    When subsample_indices provided, selects the subset for the recursion
    (and corresponding sub-Jacobian if J is square matching the full vector length);
    returns the (subsampled) result. Default (None) preserves full-vector behavior.
    """
    e = prev_error
    J = block_jacobian_approx
    d = local_delta
    if subsample_indices is not None:
        idx = subsample_indices
        e = e[idx]
        d = d[idx]
        # Subsample square Jacobian submatrix if shapes align for full vector
        if J.ndim == 2 and J.shape[0] == J.shape[1] == len(prev_error):
            J = J[np.ix_(idx, idx)]
    return e + J @ e + d


def main() -> int:
    """Entry point when run as `python -m module_T1_precision_depthN`."""
    print("T1_precision_map_v0_2 receipt skeleton")
    # TODO: implement per plan tasks
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
