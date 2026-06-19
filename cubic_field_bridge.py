"""Paper test #4: extend the Hagedorn-RLCT bridge to a CUBIC field (unit rank 2).
Cyclic cubic of conductor 7 = unique cubic subfield of Q(zeta_7): zeta_K = zeta * |L(.,chi)|^2 with chi a
cubic character mod 7. Prediction: leading (b-1)^2 I -> 1 (RLCT pole order, universal), subleading is its own
field value -- now with a rank-2 / two-L-factor structure. Computed alongside the quadratics for contrast.
"""
import mpmath as mp
mp.mp.dps = 40
w = mp.e ** (2j * mp.pi / 3)                       # primitive cube root of unity
ind = {1: 0, 3: 1, 2: 2, 6: 3, 4: 4, 5: 5}          # discrete log base g=3 mod 7
chi = {a: w ** (ind[a] % 3) for a in ind}           # cubic character mod 7

def Lval(b):
    return 7 ** (-b) * sum(chi[a] * mp.zeta(b, mp.mpf(a) / 7) for a in chi)

def zk_cubic(b):                                    # zeta_K = zeta * L * conj(L) = zeta * |L|^2 (real, >0)
    L = Lval(b)
    return mp.zeta(b) * (mp.re(L) ** 2 + mp.im(L) ** 2)

b = mp.mpf('1.02')
I = mp.diff(lambda x: mp.log(zk_cubic(x)), b, 2)
print("Paper test #4 -- cyclic cubic field Q(zeta_7)^+  (totally real, unit rank 2):")
print("  (b-1)^2 * I  @ b=1.02 = %s   (predict 1.000, RLCT pole order universal)" % mp.nstr((b - 1) ** 2 * I, 8))
print("  subleading I - 1/(b-1)^2 = %s   (field-specific; rank-2 / two L-factors)" % mp.nstr(I - 1 / (b - 1) ** 2, 8))
print("\n  context (from primon_field_bridge): Q -0.185 | Q(sqrt-3) -0.478 | Q(sqrt5) -1.223 | this cubic ->")
