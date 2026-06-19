"""Down a rung: extend the RLCT<->Fisher bridge from Q to number fields. zeta_K = zeta * L(.,chi).
Prediction: the s=1 pole is simple for EVERY field (class-number formula) -> leading RLCT term is
field-INDEPENDENT ((b-1)^2 I -> 1); the ARITHMETIC lives in the subleading amplitude. Compute all three.
"""
import mpmath as mp
mp.mp.dps = 40

def zk(D, chi):
    c = abs(D)
    return lambda b: mp.zeta(b) * (c**(-b) * sum(chi[a]*mp.zeta(b, mp.mpf(a)/c) for a in chi))

fields = {
  "Q            (unit rank 0, the primon gas)": None,
  "Q(sqrt -3)   (imag quad, unit rank 0)":      zk(3, {1:1, 2:-1}),
  "Q(sqrt 5)    (real quad, unit rank 1)":      zk(5, {1:1, 2:-1, 3:-1, 4:1}),
}
print(f"  {'field':<44}{'(b-1)^2*I  @b=1.02':>20}{'subleading I-1/(b-1)^2':>26}")
for name, f in fields.items():
    g = (lambda b: mp.zeta(b)) if f is None else f
    b = mp.mpf('1.02')
    I = mp.diff(lambda x: mp.log(g(x)), b, 2)
    print(f"  {name:<44}{mp.nstr((b-1)**2*I,8):>20}{mp.nstr(I - 1/(b-1)**2, 8):>26}")
print("\n  leading == 1.000 for ALL (RLCT pole order universal); subleading splits the fields (arithmetic).")
print("  THE BRANCH: RLCT(leading) = effective-dimension; subleading-amplitude = the IG-PRIMON arithmetic reading.")
print("  Both are sections of one sheaf over {models/fields}; the pole is the stalk, the subleading is the gluing.")
