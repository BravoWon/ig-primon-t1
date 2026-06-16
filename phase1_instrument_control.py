"""Phase-1 INSTRUMENT control (control-before-scan, before any posterior chain).

Protocol: PHASE1_protocol_genuine_branch_placement_v0_1.md, sections 1 and 3.

Phase 0 validated the DETECTOR (FSS on chi) on SK. It did NOT test the two real-net risks:
a permutation-invariant FUNCTION-SPACE overlap, and the connectivity cross-check. This script
validates THOSE on cases whose answer is known, before a single real-net chain runs.

Two instruments, each checked against a known answer:

  (1) Function-space overlap q  (protocol s.1)
      - q(A,A) = 1 exactly
      - PERMUTATION GUARD (the "weights forbidden" rule, executable): permuting a net's hidden
        units is an exact function-preserving symmetry, so q(A, perm(A)) MUST be 1 to machine
        precision. If function-space purity ever breaks, this screams.
      - null/limit: two independent nets give a baseline q; a net vs a tiny perturbation gives q->1.

  (2) Connectivity barrier Delta  (protocol s.3)
      - known-connected control: A' = perm(A) is the SAME function. Weight-matching alignment must
        recover the permutation, so pi(A') = A and the interpolation barrier Delta = 0.
      - alignment must MATTER: the same pair interpolated WITHOUT alignment has Delta > 0.

CPU-only numpy/scipy. [E-hw] exploratory instrument, NOT a [V] receipt.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---- a 1-hidden-layer MLP: y = W2 @ tanh(W1 x + b1) + b2 (scalar output) ----
def init_net(d_in, H, rng, scale=1.0):
    return dict(W1=rng.standard_normal((H, d_in)) * scale / np.sqrt(d_in),
                b1=rng.standard_normal(H) * 0.1,
                W2=rng.standard_normal(H) * scale / np.sqrt(H),
                b2=0.0)


def forward(net, X):
    h = np.tanh(X @ net["W1"].T + net["b1"])      # (n, H)
    return h @ net["W2"] + net["b2"]               # (n,)


def mse(net, X, y):
    return float(np.mean((forward(net, X) - y) ** 2))


def train(net, X, y, steps=3000, lr=0.05):
    net = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in net.items()}
    n = X.shape[0]
    for _ in range(steps):
        h = np.tanh(X @ net["W1"].T + net["b1"])
        pred = h @ net["W2"] + net["b2"]
        g = (2.0 / n) * (pred - y)                 # dL/dpred
        gW2 = h.T @ g
        gb2 = g.sum()
        gh = np.outer(g, net["W2"]) * (1 - h ** 2)  # (n,H)
        gW1 = gh.T @ X
        gb1 = gh.sum(0)
        net["W1"] -= lr * gW1; net["b1"] -= lr * gb1
        net["W2"] -= lr * gW2; net["b2"] -= lr * gb2
    return net


# ---- function-space overlap (protocol s.1) ----
def phi(net, D):
    """Centered, unit-normalized output vector on the probe set D (function space; weights never enter)."""
    f = forward(net, D)
    f = f - f.mean()
    nrm = np.linalg.norm(f)
    return f / nrm if nrm > 0 else f


def overlap(a, b, D):
    return float(np.dot(phi(a, D), phi(b, D)))


# ---- hidden-unit permutation: an exact function-preserving symmetry ----
def permute(net, P):
    return dict(W1=net["W1"][P], b1=net["b1"][P], W2=net["W2"][P], b2=net["b2"])


# ---- weight-matching alignment of b onto a (Ainsworth-style, by W1 rows) ----
def align(a, b):
    """Permutation of b's hidden units maximizing match to a's (cosine of W1 rows). Returns permuted b."""
    Wa, Wb = a["W1"], b["W1"]
    Wan = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    Wbn = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    cost = -(Wan @ Wbn.T)                          # maximize similarity = minimize -sim
    _, col = linear_sum_assignment(cost)            # a-row i <- b-row col[i]
    return permute(b, col)                          # b reordered to a's unit order


# ---- interpolation barrier (protocol s.3) ----
def barrier(a, b, X, y, n_t=21):
    la, lb = mse(a, X, y), mse(b, X, y)
    keys = ("W1", "b1", "W2", "b2")
    ls = []
    for t in np.linspace(0, 1, n_t):
        mid = {k: (1 - t) * a[k] + t * b[k] for k in keys}
        ls.append(mse(mid, X, y))
    return max(ls) - max(la, lb), la, lb


def run(seed=0):
    rng = np.random.default_rng(seed)
    d_in, H = 8, 32
    Xtr = rng.standard_normal((512, d_in)); D = rng.standard_normal((256, d_in))
    teacher = init_net(d_in, H, rng, scale=1.5)
    ytr = forward(teacher, Xtr)

    A = train(init_net(d_in, H, rng), Xtr, ytr)
    print(f"trained student A: MSE = {mse(A, Xtr, ytr):.2e}")
    fails = []

    # ---------- (1) function-space overlap ----------
    print("\n[1] function-space overlap q")
    q_self = overlap(A, A, D)
    print(f"  q(A,A)              = {q_self:.12f}   (must be 1)")
    if abs(q_self - 1) > 1e-9: fails.append("q(A,A)!=1")

    P = rng.permutation(H)
    q_perm = overlap(A, permute(A, P), D)
    print(f"  q(A, perm(A))       = {q_perm:.12f}   PERMUTATION GUARD (must be 1 to machine precision)")
    if abs(q_perm - 1) > 1e-9: fails.append("permutation guard")

    R1 = init_net(d_in, H, rng); R2 = init_net(d_in, H, rng)
    q_null = overlap(R1, R2, D)
    print(f"  q(rand1, rand2)     = {q_null:+.4f}   (null baseline; must be < 1)")
    if not (q_null < 0.99): fails.append("null not < 1")

    Anoise = {k: (v + 1e-4 * rng.standard_normal(np.shape(v)) if isinstance(v, np.ndarray) else v)
              for k, v in A.items()}
    q_lim = overlap(A, Anoise, D)
    print(f"  q(A, A+tiny noise)  = {q_lim:.6f}   (limit; must be ~1)")
    if q_lim < 0.999: fails.append("perturbation limit")

    # ---------- (2) connectivity barrier ----------
    print("\n[2] connectivity barrier Delta  (known-connected control: A' = perm(A))")
    Aprime = permute(A, rng.permutation(H))
    d_un, _, _ = barrier(A, Aprime, Xtr, ytr)
    A_aligned = align(A, Aprime)
    d_al, _, _ = barrier(A, A_aligned, Xtr, ytr)
    print(f"  Delta UNaligned     = {d_un:.4e}   (mismatched units -> barrier > 0)")
    print(f"  Delta ALIGNED       = {d_al:.4e}   (alignment recovers perm -> must be ~0)")
    if not (d_al < 1e-6): fails.append("aligned barrier not ~0")
    if not (d_un > d_al): fails.append("alignment did not matter")

    # bonus: two INDEPENDENTLY trained solutions -- the real phenomenon (not a hard assert)
    B = train(init_net(d_in, H, rng), Xtr, ytr)
    C = train(init_net(d_in, H, rng), Xtr, ytr)
    d_bc_un, _, _ = barrier(B, C, Xtr, ytr)
    d_bc_al, _, _ = barrier(B, align(B, C), Xtr, ytr)
    print(f"\n  [demo] two independent minima: barrier {d_bc_un:.3e} (unaligned) -> "
          f"{d_bc_al:.3e} (aligned)   alignment {'reduces' if d_bc_al < d_bc_un else 'does NOT reduce'} it")

    print("\n" + ("INSTRUMENT-CONTROL: PASS" if not fails else f"INSTRUMENT-CONTROL: FAIL {fails}"))
    return not fails


if __name__ == "__main__":
    run()
