#!/usr/bin/env python
"""Vectorized double-double (DD) arithmetic over numpy — ~31 significant digits at ~20x f64 cost.
Each DD value is an (hi, lo) pair of float64 arrays with |lo| <= ulp(hi)/2. Error-free
transforms: TwoSum (Knuth), Dekker split / TwoProd (no FMA on this toolchain). Prefix sum via
Hillis-Steele doubling: log2(N) vectorized DD adds (the enabling trick for the C1h evolution --
a sequential compensated loop would be ~5e9 Python iterations). exp via ln2 range reduction +
r/2^10 scaling + degree-9 Taylor + 10 squarings (|r'|<=3.4e-4 => truncation ~6e-32). Division
by Newton refinement of the f64 quotient. Per PREREG_C1h_mpfloat.md; validated against mpmath
dps=40 by dd_selftest().
"""
import numpy as np

_SPLIT = 134217729.0                                             # 2^27 + 1 (Dekker)
LN2_HI = 6.931471805599453e-01
LN2_LO = 2.3190468138462996e-17


def two_sum(a, b):
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def quick_two_sum(a, b):                                         # requires |a| >= |b|
    s = a + b
    return s, b - (s - a)


def _split(a):
    t = _SPLIT * a
    hi = t - (t - a)
    return hi, a - hi


def two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def dd(hi, lo=None):
    hi = np.asarray(hi, dtype=np.float64)
    return (hi, np.zeros_like(hi) if lo is None else np.asarray(lo, dtype=np.float64))


def add(x, y):
    s, e = two_sum(x[0], y[0])
    e = e + x[1] + y[1]
    return quick_two_sum(s, e)


def sub(x, y):
    return add(x, (-y[0], -y[1]))


def neg(x):
    return (-x[0], -x[1])


def mul(x, y):
    p, e = two_prod(x[0], y[0])
    e = e + x[0] * y[1] + x[1] * y[0]
    return quick_two_sum(p, e)


def div(x, y):
    q0 = x[0] / y[0]
    r = sub(x, mul((q0, np.zeros_like(q0)), y))
    q1 = r[0] / y[0]
    r = sub(r, mul((q1, np.zeros_like(q1)), y))
    q2 = r[0] / y[0]
    s, e = quick_two_sum(q0, q1)
    return quick_two_sum(s, e + q2)


def scale(x, c):                                                 # c: exact f64 scalar (e.g. 2^k)
    return (x[0] * c, x[1] * c)


def dd_sum_prefix(x):
    """Inclusive prefix sum along the last axis, Hillis-Steele doubling in DD."""
    hi, lo = x[0].copy(), x[1].copy()
    n, sh = hi.shape[-1], 1
    while sh < n:
        ph = np.zeros_like(hi); pl = np.zeros_like(lo)
        ph[..., sh:], pl[..., sh:] = hi[..., :-sh], lo[..., :-sh]
        hi, lo = add((hi, lo), (ph, pl))
        sh <<= 1
    return hi, lo


def _dd_recip(n):                                                # 1/n to full DD precision,
    one = dd(np.array(1.0))                                      # via the kernel's own Newton
    q = div(one, dd(np.array(float(n))))                         # division (the hand-rolled
    return float(q[0]), float(q[1])                              # residual version degenerated
                                                                 # to lo=0 -- caught in calib)


_EXP_COEFS = [_dd_recip(f) for f in
              (40320, 5040, 720, 120, 24, 6, 2, 1, 1)]           # 1/8! .. 1/0!


def exp(x):
    """DD exp, vectorized: k = round(x/ln2); r = x - k*ln2; r' = r/1024; Taylor deg 9; square x10.
    Coefficients are DD-exact (f64-literal coefficients cost ~1e-17 each, x1024 by the squaring
    chain = the 3.6e-25 bug caught in calibration). Callers pre-clip args to [0, 600]."""
    k = np.round(x[0] / LN2_HI)
    r = sub(x, mul((k, np.zeros_like(k)), (np.full_like(k, LN2_HI), np.full_like(k, LN2_LO))))
    rp = scale(r, 1.0 / 1024.0)
    acc = (np.full_like(k, 1.0 / 362880.0), np.full_like(k, 0.0))    # 1/9! (exact enough at deg 9)
    for ch, cl in _EXP_COEFS:
        acc = add(mul(acc, rp), (np.full_like(k, ch), np.full_like(k, cl)))
    for _ in range(10):                                          # undo the /1024 by squaring
        acc = mul(acc, acc)
    p2k = np.exp2(k)                                             # 2^k exact in f64 (k <= 866 here)
    return (acc[0] * p2k, acc[1] * p2k)


def diff(x):
    return sub((x[0][..., 1:], x[1][..., 1:]), (x[0][..., :-1], x[1][..., :-1]))


def dd_selftest(n=10000, seed=7):
    """Validate against mpmath dps=40. Returns dict of max relative errors."""
    import mpmath as mp
    mp.mp.dps = 40
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, n) * 10.0 ** rng.integers(-30, 30, n)
    b = rng.uniform(-1, 1, n) * 10.0 ** rng.integers(-30, 30, n)
    al = a * 1e-17 * rng.uniform(-1, 1, n)                       # plausible lo parts
    bl = b * 1e-17 * rng.uniform(-1, 1, n)
    A, B = (a, al), (b, bl)
    out = {}

    def check(name, got, exact_fn):
        errs = []
        for i in range(0, n, max(1, n // 200)):                  # sample 200 for mpmath speed
            ex = exact_fn(mp.mpf(a[i]) + mp.mpf(al[i]), mp.mpf(b[i]) + mp.mpf(bl[i]))
            gv = mp.mpf(got[0][i]) + mp.mpf(got[1][i])
            if ex != 0:
                errs.append(abs((gv - ex) / ex))
        out[name] = float(max(errs))

    check("add", add(A, B), lambda x, y: x + y)
    check("sub", sub(A, B), lambda x, y: x - y)
    check("mul", mul(A, B), lambda x, y: x * y)
    check("div", div(A, B), lambda x, y: x / y)
    e_arg = (rng.uniform(0, 600, n), rng.uniform(-1e-14, 1e-14, n))
    got_e = exp(e_arg)
    errs = []
    for i in range(0, n, max(1, n // 200)):
        ex = mp.exp(mp.mpf(e_arg[0][i]) + mp.mpf(e_arg[1][i]))
        gv = mp.mpf(got_e[0][i]) + mp.mpf(got_e[1][i])
        errs.append(abs((gv - ex) / ex))
    out["exp"] = float(max(errs))
    xs = (rng.uniform(-1, 1, 800), rng.uniform(-1e-17, 1e-17, 800))
    ps = dd_sum_prefix(xs)
    ex_acc, errs = mp.mpf(0), []
    for i in range(800):
        ex_acc += mp.mpf(xs[0][i]) + mp.mpf(xs[1][i])
        if i % 40 == 0 and ex_acc != 0:
            gv = mp.mpf(ps[0][i]) + mp.mpf(ps[1][i])
            errs.append(abs((gv - ex_acc) / abs(ex_acc)))
    out["prefix"] = float(max(errs))
    return out


if __name__ == "__main__":
    for k, v in dd_selftest().items():
        print(f"  {k:8s} max rel err = {v:.3e}")
