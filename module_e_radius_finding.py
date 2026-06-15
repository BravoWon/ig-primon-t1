"""
IG-PRIMON-T1 Module E (HE.2, [GATE]) — radius of the primon-gas Fisher expansion
for quadratic number fields, tested against the registered prediction.

Registered claim (v0.4): imaginary quadratic -> radius 2; real quadratic -> radius 3.
Finding: imaginary quadratic = 2 (correct); real quadratic = 1 (the registered 3 is WRONG).
Mechanism: zeta_K has an order-(r1+r2-1) zero at s=0 (Dirichlet unit rank). For real
quadratic that is a simple zero at s=0, distance 1 from s=1 -- it binds before the
s=-2,-4,... even-character trivial zeros HE.2 was tracking. Imaginary quadratic has
rank 0, no s=0 zero, so its radius is set by the L-trivial zero at s=-1 (distance 2).

zeta_K = zeta(s) L(s, chi_D);  L(s,chi) = q^{-s} sum_a chi(a) zeta(s, a/q) (Hurwitz).
"""
from mpmath import mp, mpf, mpc, zeta, log, fabs

mp.dps = 40

def Lfun(q, chi):
    def L(s):
        s = mpc(s)
        return sum(ch*zeta(s, mpf(a)/q) for a, ch in chi.items() if ch) * q**(-s)
    return L

FIELDS = {
    "Q(sqrt5)  real quad  (even chi mod 5, unit rank 1)": (Lfun(5, {1:1,2:-1,3:-1,4:1}), "1.0", 3),
    "Q(sqrt-3) imag quad  (odd chi mod 3, unit rank 0)":  (Lfun(3, {1:1,2:-1}),          "0.5", 2),
}

print("=== rigorous: order of vanishing of zeta_K at s=0  (= r1+r2-1) ===")
print(f"  zeta(0) = {mp.nstr(zeta(0),6)}")
for name, (L, _, _) in FIELDS.items():
    zk0 = zeta(0)*L(0)
    print(f"  {name[:20]}:  L(0)={mp.nstr(L(0),5):>16}   zeta_K(0)={mp.nstr(zk0,5):>16}"
          f"   {'ZERO at s=0 -> radius<=1' if abs(zk0)<mpf('1e-20') else 'nonzero at s=0'}")

def acoeffs(L, N=16, M=96, r=mpf('0.5')):
    # h(eps)=eps*zeta(1+eps)*L(1+eps) is entire; Taylor coeffs by DFT on |eps|=r
    h = lambda eps: (lambda s: mpc(eps)*zeta(s)*L(s))(1+mpc(eps))
    smp = [h(r*mp.expjpi(2*mpf(j)/M)) for j in range(M)]
    return [sum(smp[j]*mp.expjpi(-2*mpf(k)*j/M) for j in range(M))/(M*r**k) for k in range(N+1)]

def logcoeffs(a):
    b = [log(a[0])]
    for k in range(1, len(a)):
        b.append((k*a[k] - sum(j*b[j]*a[k-j] for j in range(1, k)))/(k*a[0]))
    return b

# radius of {b_k} (log-series coeffs) = dist to nearest zero of (s-1)zeta_K; |b_(k+1)/b_k| -> 1/radius
for name, (L, pred, reg) in FIELDS.items():
    b = logcoeffs(acoeffs(L))
    print(f"\n== {name} ==   |b_(k+1)/b_k| -> {pred} (=1/radius);  registered radius {reg}")
    for k in range(6, 14):
        # the residual from the limit is the k/(k+1) signature of a logarithmic singularity
        print(f"   k={k:2d}   |b_(k+1)/b_k| = {mp.nstr(fabs(b[k+1]/b[k]),7)}"
              f"    [log-sing model 1/radius * k/(k+1) = {mp.nstr(mpf(pred)*k/(k+1),7)}]")

print("\nCorrected statement: radius reads the Dirichlet unit rank r1+r2-1.")
print("  rank 0  <=>  radius >= 2   (Q -> 3 via zeta trivial zero s=-2; imag quad -> 2 via L zero s=-1)")
print("  rank >=1            radius  = 1   (zeta_K zero at s=0; real quad and all higher-unit fields)")
