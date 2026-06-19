"""Transposition (leading-edge -> discrete IG methodology), COMPUTED not described:
SLT's RLCT = largest pole of the learning zeta function; IG-PRIMON reads geometry off the singularity of
the primon-gas zeta. Same machinery. Concrete bridge: the primon-gas Fisher info I(b)=Var_b(log n)=
d^2/db^2 log zeta(b) diverges at the Hagedorn point b=1 at a rate set by the ORDER of the pole of zeta at
s=1 -- which is exactly the quantity SLT names (pole of the zeta -> asymptotics). Compute it.
"""
import mpmath as mp
mp.mp.dps = 30

print("Primon gas: I(b) = Var_b(log n) = d^2/db^2 log zeta(b)  (IG-PRIMON Fisher info)")
print("Pole of zeta at s=1 has order 1 -> RLCT/SLT prediction: I(b) ~ ORDER / (b-1)^2 = 1/(b-1)^2\n")
print(f"  {'beta':>7} {'I(beta)':>18} {'(b-1)^2 * I(b)':>18}")
for b in [mp.mpf(x) for x in ['1.5','1.3','1.2','1.1','1.05','1.02','1.01','1.005']]:
    z  = mp.zeta(b)
    z1 = mp.zeta(b, derivative=1)
    z2 = mp.zeta(b, derivative=2)
    I  = z2/z - (z1/z)**2
    print(f"  {mp.nstr(b,5):>7} {mp.nstr(I,10):>18} {mp.nstr((b-1)**2 * I,10):>18}")

print("\n  -> (b-1)^2 * I(b) -> 1.000 : the curvature-blowup exponent IS the pole order of the learning zeta.")
print("     IG-PRIMON 'radius/amplitude near singularity' == SLT 'location/order of the zeta pole' (RLCT).")
print("     Field generalization: ord_{s=0} zeta_K = unit rank rho  <->  RLCT multiplicity (effective dim).")
