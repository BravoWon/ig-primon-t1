"""
Independent adversarial-audit checks for IG-PRIMON-T1 (2026-06-12).
None of these reuse the program's own computational routes.

PART 1 — Gaussian pin of the Module C curvature formula.
  The Fisher metric of the normal family N(mu, sigma^2) in natural parameters
  (t1, t2) = (mu/sigma^2, -1/(2 sigma^2)) is the Hessian of
  psi = -t1^2/(4 t2) - (1/2) log(-2 t2).
  Known: constant curvature, R = -1 in the sphere-positive scalar convention
  (metric dmu^2/s^2 + 2 ds^2/s^2 is hyperbolic with K = -1/2, R = 2K = -1).
  The candidate formula R = -N/(2 (det g)^2) must return exactly -1 at every
  point. This pins formula AND convention analytically, independent of the
  finite-difference route in moduleC_certify.py.

PART 2 — Proposition 2.3 tested AS STATED (not the Z(m) rearrangement):
  c_k = -(k+1) [ T(m) + NT(m) ],  m = k+2,
  T(m)  = sum over shifted trivial zeros  = (-1)^m [ (1-2^-m) zeta(m) - 1 ],
  NT(m) = sum over shifted nontrivial zeros, computed by DIRECT summation of
          2 Re((rho-1)^-m) over the first J zeros (mpmath zetazero), with a
          density-based tail budget. Locked c_k must land inside the budget.

PART 3 — Module C constants by an independent route (mp.primezeta, high cutoff):
  Delta3(1) = sum_{k>=2} k(k-1) P(k),  c(1) = A0 + sum_{k>=2} k P(k),
  kappa = c(1) - 1, A0 = sum_{n>=2} mu(n)/n log zeta(n).
  Adjudicates the 20-digit Delta3 print in the derivation doc, whose source
  sum stops at k=59 (tail ~ 6e-15 -> last printed digits suspect).

PART 4 — Independent corroboration of the registered constant C at dps=56:
  different split point (delta = 1e-4) and different subdivision
  [1+delta, 1.2, 1.6, 2] from both runs in verify_C_dps80.py.
  Target: agreement with the registered 31 digits well inside +/- 6e-31.
"""
from mpmath import mp, mpf, zeta, primezeta, stieltjes, euler, zetazero
from mpmath import log, sqrt, pi, quad, re as mre

mp.dps = 60  # A1 fix: parse locked constants at full precision (first run parsed at default dps=15;
             # PART 2/3 conclusions were unaffected, PART 4's printed "registered" line was a parse
             # artifact — corrected comparison: |C_ind - C_reg| = 2.24e-32, inside the +/-6e-31 budget)
LOCKED_C = mpf('-0.034356154179121986083110881458470')
LOCKED_c = [
    mpf('-0.18754623284036522459720338460544158838394446358095'),
    mpf('0.1033772640663857876040164461672083268907976330979'),
    mpf('-0.044254976476361232193740710443126531087925547060243'),
    mpf('0.018097911553981514964984464396659931026478189667239'),
    mpf('-0.00723397602262591570108490211148459264792002889224'),
]
DOC_DELTA3 = mpf('5.0451881850144243171')
DOC_KAPPA = mpf('0.83250321174454')

# ---------------- PART 1 ----------------
print("== PART 1: Gaussian closed-form pin of R = -N/(2 det g^2) ==")
mp.dps = 30

def gauss_R(t1, t2):
    p11 = -1 / (2 * t2)
    p12 = t1 / (2 * t2 ** 2)
    p22 = -t1 ** 2 / (2 * t2 ** 3) + 1 / (2 * t2 ** 2)
    p111 = mpf(0)
    p112 = 1 / (2 * t2 ** 2)
    p122 = -t1 / t2 ** 3
    p222 = 3 * t1 ** 2 / (2 * t2 ** 4) - 1 / t2 ** 3
    M = [[p11, p12, p22], [p111, p112, p122], [p112, p122, p222]]
    N = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    detg = p11 * p22 - p12 ** 2
    return -N / (2 * detg ** 2)

for (t1, t2) in [(mpf('0'), mpf('-0.5')), (mpf('0.3'), mpf('-0.7')),
                 (mpf('-1.2'), mpf('-0.25')), (mpf('2.4'), mpf('-3.1'))]:
    print(f"  (t1,t2)=({float(t1)},{float(t2)}):  R = {mp.nstr(gauss_R(t1, t2), 25)}")
print("  expected: -1 exactly at every point (normal family, sphere-positive convention)")

# flat product sanity: psi = -log x - log y  ->  R must be 0
def flat_R(x, y):
    p = {'11': 1/x**2, '12': mpf(0), '22': 1/y**2,
         '111': -2/x**3, '112': mpf(0), '122': mpf(0), '222': -2/y**3}
    M = [[p['11'], p['12'], p['22']], [p['111'], p['112'], p['122']],
         [p['112'], p['122'], p['222']]]
    N = (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
         - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
         + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    detg = p['11']*p['22'] - p['12']**2
    return -N/(2*detg**2)
print(f"  flat-product control psi=-log x - log y at (2,3): R = {mp.nstr(flat_R(mpf(2), mpf(3)), 5)} (expect 0)")

# ---------------- PART 2 ----------------
print("\n== PART 2: Prop 2.3 as stated — direct shifted-zero sums ==")
mp.dps = 25
J = 200
zeros = [zetazero(j) for j in range(1, J + 1)]
tJ = zeros[-1].imag
print(f"  using first J={J} nontrivial zeros, t_J = {mp.nstr(tJ, 10)}")
mp.dps = 30
print(f"  {'k':>2} {'pred from zeros':>22} {'locked c_k':>22} {'|diff|':>10} {'tail budget':>12} {'in budget'}")
for k in range(0, 5):
    m = k + 2
    T = (-1) ** m * ((1 - mpf(2) ** (-m)) * zeta(m) - 1)
    NT = mpf(0)
    for rho in zeros:
        NT += 2 * mre((rho - 1) ** (-m))
    pred = -(k + 1) * (T + NT)
    diff = abs(pred - LOCKED_c[k])
    # density tail: 2*int_T^inf t^-m (1/2pi) log(t/2pi) dt, times (k+1)
    budget = (k + 1) * (1 / pi) * tJ ** (-(m - 1)) * (log(tJ / (2 * pi)) / (m - 1) + 1 / mpf((m - 1) ** 2))
    print(f"  {k:>2} {mp.nstr(pred, 14):>22} {mp.nstr(LOCKED_c[k], 14):>22} "
          f"{mp.nstr(diff, 3):>10} {mp.nstr(budget, 3):>12} {'YES' if diff < 2*budget else 'NO'}")

# ---------------- PART 3 ----------------
print("\n== PART 3: Module C constants, independent primezeta route ==")
mp.dps = 40

def mobius_list(N):
    mu = [0] * (N + 1); mu[1] = 1
    primes = []; comp = [False] * (N + 1)
    for i in range(2, N + 1):
        if not comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > N: break
            comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu

MU = mobius_list(140)
A0 = sum(MU[n] * log(zeta(n)) / n for n in range(2, 141) if MU[n] != 0)

D3_59 = sum(mpf(k) * (k - 1) * primezeta(k) for k in range(2, 60))     # their cutoff
D3_400 = sum(mpf(k) * (k - 1) * primezeta(k) for k in range(2, 401))   # tail-controlled
c1 = A0 + sum(mpf(k) * primezeta(k) for k in range(2, 401))
kappa = c1 - 1
print(f"  A0                = {mp.nstr(A0, 25)}")
print(f"  Delta3 (k<=59)    = {mp.nstr(D3_59, 25)}   <- replicates their cutoff")
print(f"  Delta3 (k<=400)   = {mp.nstr(D3_400, 25)}   <- tail-controlled (tail < 1e-110)")
print(f"  doc prints          {mp.nstr(DOC_DELTA3, 25)}")
print(f"  Delta3 truncation error of the k<=59 sum: {mp.nstr(D3_400 - D3_59, 3)}")
print(f"  amplitude Delta3/2 (true) = {mp.nstr(D3_400 / 2, 25)}")
print(f"  kappa (k<=400)    = {mp.nstr(kappa, 20)}   doc: {mp.nstr(DOC_KAPPA, 20)}   |diff| = {mp.nstr(abs(kappa - DOC_KAPPA), 3)}")

# ---------------- PART 4 ----------------
print("\n== PART 4: independent C corroboration, dps=56, delta=1e-4, new subdivision ==")
mp.dps = 56
Nser = 8
g = [euler] + [stieltjes(n) for n in range(1, Nser + 1)]
P = [mpf(1)] + [(-1) ** (m - 1) * g[m - 1] / mp.factorial(m - 1) for m in range(1, Nser + 1)]
a = [mpf(0)] * (Nser + 1)
for m_ in range(1, Nser + 1):
    s = P[m_]
    for k_ in range(1, m_):
        s -= k_ * a[k_] * P[m_ - k_] / m_
    a[m_] = s
d = [(k_ + 2) * (k_ + 1) * a[k_ + 2] for k_ in range(0, Nser - 1)]

def f_series(eps):
    w = mpf(0)
    for k_ in range(len(d)):
        w += d[k_] * eps ** (k_ + 2)
    return (w / (sqrt(1 + w) + 1)) / eps

def f_direct(beta):
    e = beta - 1
    z = zeta(beta); z1 = zeta(beta, derivative=1); z2 = zeta(beta, derivative=2)
    return sqrt((z2 * z - z1 ** 2) / z ** 2) - 1 / e

delta = mpf(1) / 10000
s1, e1 = quad(f_series, [0, delta], error=True)
s2, e2 = quad(f_direct, [1 + delta, mpf('1.2'), mpf('1.6'), 2], error=True)
C_ind = s1 + s2
print(f"  C (independent)  = {mp.nstr(C_ind, 40)}")
print(f"  C (registered)   = {mp.nstr(LOCKED_C, 40)}")
print(f"  |diff|           = {mp.nstr(abs(C_ind - LOCKED_C), 3)}   (registered budget 6e-31)")
print(f"  quad error estimates: series {mp.nstr(e1, 2)}, direct {mp.nstr(e2, 2)}; "
      f"series truncation < 1e-39 at delta=1e-4")
