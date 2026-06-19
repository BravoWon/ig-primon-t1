"""Paper test #5: RLCT of a real 2-layer LINEAR network. f(x)=(sum_h a_h b_h) x, true function 0,
loss K=(sum_{h=1}^H a_h b_h)^2 in 2H parameters (H hidden units). H=1 is the normal-crossing toy
(lambda=1/2, m=2) -> validates the estimator. Prediction: lambda ~ 1/2 (codim-1 vanishing) for all H;
multiplicity m runs 2 (H=1 crossing) -> 1 as the singular set smooths into a hypersurface.
This is the IG-PRIMON theme on the ML side: leading (lambda) universal, MULTIPLICITY carries the structure.
"""
import numpy as np
rng = np.random.default_rng(0)
N = 12_000_000
eps = np.array([10.0 ** -k for k in range(2, 8)])
print("Paper test #5 -- RLCT of K=(sum_h a_h b_h)^2, 2-layer linear net, H hidden units:\n")
print(f"  {'H':>3} {'params':>7} {'RLCT lambda':>12} {'mult. m':>9}   (anchor H=1: lambda=0.5, m=2)")
for H in [1, 2, 3, 5]:
    a = rng.standard_normal((N, H)); b = rng.standard_normal((N, H))
    K = (a * b).sum(1) ** 2
    V = np.array([np.mean(K < e) for e in eps])
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(eps)), np.log(eps), np.log(-np.log(eps))], np.log(V), rcond=None)
    print(f"  {H:>3} {2*H:>7} {coef[1]:>12.3f} {coef[2]+1:>9.2f}")
    del a, b, K
print("\n  lambda ~ 0.5 across widths (codim-1, universal); multiplicity falls 2 -> ~1 as the crossing smooths.")
print("  Same signature as the number-theory side: pole LOCATION universal, pole ORDER carries the structure.")
