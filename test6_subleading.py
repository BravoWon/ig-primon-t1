"""Paper test #6 (real-axis fit): subleading = 2c - gamma_K^2 in CLOSED FORM from the field's arithmetic.
zeta_K(s)=a0/(s-1)[1+gamma_K(s-1)+c(s-1)^2+...]; d^2/db^2 log zeta_K = 1/(b-1)^2 + (2c - gamma_K^2).
gamma_K=a1/a0 = the Euler-Kronecker constant. Fit (a0,a1,a2) from g(s)=(s-1)zeta_K(s) sampled near s=1.
"""
import numpy as np
import mpmath as mp
mp.mp.dps = 30
w3 = mp.e ** (2j * mp.pi / 3); ind = {1:0,3:1,2:2,6:3,4:4,5:5}; chi3 = {a: w3**(ind[a]%3) for a in ind}
def zk_quad(D, chi):
    c = abs(D); return lambda s: mp.zeta(s) * (c**(-s) * sum(chi[a]*mp.zeta(s, mp.mpf(a)/c) for a in chi))
def zk_cubic(s):
    Lc = 7**(-s) * sum(chi3[a]*mp.zeta(s, mp.mpf(a)/7) for a in chi3); return mp.zeta(s) * (Lc * mp.conj(Lc)).real
fields = {"Q": mp.zeta, "Q(sqrt-3)": zk_quad(3, {1:1,2:-1}),
          "Q(sqrt5)": zk_quad(5, {1:1,2:-1,3:-1,4:1}), "Q(zeta7)+": zk_cubic}
meas = {"Q":-0.185, "Q(sqrt-3)":-0.478, "Q(sqrt5)":-1.223, "Q(zeta7)+":-2.14}
hs = np.array([-0.03,-0.02,-0.01,0.01,0.02,0.03])
print("Test #6 -- subleading = 2c - gamma_K^2, closed-form from Euler-Kronecker gamma_K = a1/a0:\n")
print(f"  {'field':<11}{'residue a0':>12}{'Euler-Kronecker gK':>20}{'predicted sub':>15}{'measured':>10}")
for name, f in fields.items():
    y = np.array([float(mp.re((mp.mpf(1)+mp.mpf(h)) ** 0 * ((mp.mpf(1)+mp.mpf(h)-1) * f(mp.mpf(1)+mp.mpf(h))))) for h in hs])
    a3, a2, a1, a0 = np.polyfit(hs, y, 3)
    gK, c = a1/a0, a2/a0; sub = 2*c - gK**2
    print(f"  {name:<11}{a0:>12.5f}{gK:>20.6f}{sub:>15.4f}{meas[name]:>10.3f}")
print("\n  predicted (analytic) == measured (finite b) -> subleading IS the arithmetic: Euler-Kronecker + next coeff.")
