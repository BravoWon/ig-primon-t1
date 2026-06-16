"""T1_precision_map v0.2 -- Stage 1: the certified-reference depth-error harness + controls.

Builds E_cert(L): the relative round-off error of a low-precision (float32 here; bf16/fp8/fp4 in Stage 2)
L-block forward pass vs a high-precision reference, where the reference's right to be trusted is itself
CERTIFIED with mpmath (dps>=50) -- the Firewall move, not FP32-agreement.

Controls (control-before-scan; the trained-weight scan + F1-F3 is Stage 2, NOT here):
  C1  harness identity: ref-vs-ref forward -> E_cert == 0 to machine precision.
  C2  HARD GATE: on RANDOM-weight blocks (Budzinskiy regime, d=n=20, L=40), the MEAN E_cert(L) must grow
      ~exponentially in L with MEDIAN << MEAN. If C2 fails, the harness is wrong and no trained scan runs.
  C4  composition: depth-L error must exceed L x (single-block error) -- composition does more than per-op.
  mpmath spot-cert: float64 reference agrees with mpmath dps>=50 within the float64 noise floor at a
      sampled position -> float64 licensed as the working reference (float32 round-off >> float64's own).

NB the exponential is a WORST-CASE / mean phenomenon (Budzinskiy); median<<mean is the typical-case slack
that P1 (Stage 2) will measure on trained weights. CPU, numpy + mpmath.
"""
import numpy as np

EPS64 = np.finfo(np.float64).eps


def block(x, W, dtype):
    """Residual single-head self-attention block (Budzinskiy regime), every op cast to `dtype`."""
    Wq, Wk, Wv, Wo = (w.astype(dtype) for w in W)
    x = x.astype(dtype)
    q = (x @ Wq).astype(dtype); k = (x @ Wk).astype(dtype); v = (x @ Wv).astype(dtype)
    s = (q @ k.T / np.sqrt(np.float64(x.shape[1])).astype(dtype)).astype(dtype)
    s = (s - s.max(1, keepdims=True)).astype(dtype)
    e = np.exp(s).astype(dtype)
    a = (e / e.sum(1, keepdims=True)).astype(dtype)
    o = ((a @ v) @ Wo).astype(dtype)
    return (x + o).astype(dtype)


def forward(x0, Ws, dtype):
    x = x0.astype(dtype); traj = [x.astype(np.float64).copy()]
    for W in Ws:
        x = block(x, W, dtype); traj.append(x.astype(np.float64).copy())
    return traj


def mk_weights(d, L, gen, scale):
    return [tuple(scale * gen.standard_normal((d, d)) / np.sqrt(d) for _ in range(4)) for _ in range(L)]


def e_curve(x0, Ws, low, ref=np.float64):
    tl, tr = forward(x0, Ws, low), forward(x0, Ws, ref)
    return np.array([np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300) for a, b in zip(tl, tr)])


def mpmath_spot_cert(x0, W, d):
    """One block: does float64 agree with mpmath dps=50? Licenses float64 as the reference."""
    from mpmath import mp, mpf, matrix
    mp.dps = 50
    f64 = block(x0, W, np.float64)
    Xm = matrix(x0.tolist())
    Wm = [matrix(w.tolist()) for w in W]
    Q, K, V = Xm * Wm[0], Xm * Wm[1], Xm * Wm[2]
    sd = mpf(1) / mp.sqrt(d)
    S = (Q * K.T) * sd
    A = matrix(d, d)
    for i in range(d):
        row = [S[i, j] for j in range(d)]; m = max(row)
        ex = [mp.e ** (r - m) for r in row]; z = sum(ex)
        for j in range(d):
            A[i, j] = ex[j] / z
    O = (A * V) * Wm[3]
    mpout = Xm + O
    num = mp.sqrt(sum((mpf(float(f64[i, j])) - mpout[i, j]) ** 2 for i in range(d) for j in range(d)))
    den = mp.sqrt(sum(mpout[i, j] ** 2 for i in range(d) for j in range(d)))
    return float(num / den)


def run():
    d, L, n_init, scale = 20, 40, 300, 1.0
    g = np.random.default_rng(0)
    print(f"[Stage-1 harness]  Budzinskiy regime d={d} L={L}, {n_init} random inits, float32 vs float64\n")
    fails = []

    # C1 -- harness identity
    Ws = mk_weights(d, 4, g, scale); x0 = g.standard_normal((d, d))
    c1 = e_curve(x0, Ws, np.float64).max()
    print(f"C1 identity (ref vs ref): max E_cert = {c1:.1e}  (must be 0)")
    if c1 > 1e-15: fails.append("C1 identity")

    # mpmath spot-cert -- license float64 as the reference
    spot = mpmath_spot_cert(g.standard_normal((d, d)), mk_weights(d, 1, g, scale)[0], d)
    print(f"mpmath dps=50 spot-cert (1 block): float64 vs mpmath rel err = {spot:.1e}  "
          f"(<= ~{20*EPS64:.1e} float64 floor -> float64 licensed)")
    if spot > 1e-13: fails.append("mpmath spot-cert")

    # C2 -- HARD GATE: exponential mean, median << mean on random weights
    Ls = [1, 5, 10, 20, 30, 40]
    curves = np.array([e_curve(g.standard_normal((d, d)), mk_weights(d, L, g, scale), np.float32)
                       for _ in range(n_init)])                      # (n_init, L+1)
    mean_c = curves.mean(0); med_c = np.median(curves, 0)
    print("\nC2 (HARD GATE) -- E_cert(L) on random weights:")
    print(f"  {'L':>3} {'mean':>10} {'median':>10} {'mean/median':>12}")
    for Lq in Ls:
        print(f"  {Lq:>3} {mean_c[Lq]:>10.2e} {med_c[Lq]:>10.2e} {mean_c[Lq]/(med_c[Lq]+1e-300):>12.1f}")
    # exponential <=> positive slope of log(mean E) vs L
    LL = np.arange(5, L + 1)
    slope = np.polyfit(LL, np.log(mean_c[5:] + 1e-300), 1)[0]
    mm_ratio = mean_c[L] / (med_c[L] + 1e-300)
    print(f"  log(mean E) vs L slope = {slope:+.3f} per layer  ->  {'EXPONENTIAL' if slope > 0.02 else 'NOT exp'}")
    print(f"  median << mean at L={L}: mean/median = {mm_ratio:.1f}x  ->  {'heavy-tailed (Budzinskiy)' if mm_ratio > 3 else 'NOT'}")
    if not (slope > 0.02): fails.append("C2 not exponential")
    if not (mm_ratio > 3): fails.append("C2 median not << mean")

    # C4 -- composition beyond per-op
    single = mean_c[1]                                                # one-block error
    depthL = mean_c[L]
    print(f"\nC4 composition: E_cert(1)={single:.2e}, E_cert({L})={depthL:.2e}; "
          f"ratio={depthL/(single+1e-300):.1f}x  vs linear {L}x  -> "
          f"{'COMPOSITION (super-linear)' if depthL > 2*L*single else 'linear/per-op only'}")
    if not (depthL > 2 * L * single): fails.append("C4 no composition effect")

    print("\nSTAGE-1 HARNESS:", "C1+C2+C4 PASS -- certified harness ready; trained-weight scan (Stage 2) admissible"
          if not fails else f"FAIL {fails}")
    return not fails


if __name__ == "__main__":
    run()
