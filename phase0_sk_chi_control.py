"""Phase-0 POSITIVE/NEGATIVE control for the branch-placement experiment.

Preregistration: PREREGISTRATION_genuine_branch_placement_v0_1.md, section 3.

Before any real-net sampler, we validate the instrument -- "sample two replicas, measure the overlap
susceptibility chi = N*Var(q), read finite-size scaling" -- on a system whose answer is KNOWN:
the Sherrington-Kirkpatrick spin glass, which has a continuous spin-glass transition at T_c = 1.

  POSITIVE control (must FIRE): chi_SG must GROW as T -> T_c=1 from above, and the growth must
                                STRENGTHEN with N (finite-size scaling toward divergence). This is
                                branch 1 (finite-T RSB). If the instrument cannot see this known
                                divergence, it is blind and the program stops (prereg s.2-3).
  NEGATIVE control (must SATURATE): at high T (paramagnet, T >> T_c) chi_SG must stay small / flat.

Equilibration guard (prereg s.2): two independent replicas run from independent random starts with
the SAME disorder J; the overlap is measured between them. We also report a relaxation check (chi from
the first vs second measurement half) -- if they disagree, the chain is not equilibrated and the point
is VACUOUS, not a reading.

CPU-only numpy. This is an [E-hw] exploratory instrument, NOT a [V] receipt.
"""
import numpy as np


def sk_couplings(N, rng):
    """Symmetric SK couplings J_ij ~ N(0, 1/N), zero diagonal."""
    J = rng.standard_normal((N, N)) / np.sqrt(N)
    J = np.triu(J, 1)
    return J + J.T


def metropolis_sweep(s, J, beta, rng):
    """One sequential Metropolis sweep (single-spin flips) over all N spins."""
    N = s.size
    order = rng.permutation(N)
    for i in order:
        # local field h_i = sum_j J_ij s_j ; dE = 2 s_i h_i for a flip
        hi = J[i] @ s
        dE = 2.0 * s[i] * hi
        if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
            s[i] = -s[i]
    return s


def chi_sg_two_replicas(N, beta, rng, n_equil=2000, n_meas=4000, meas_every=10):
    """Sample two independent replicas (same J), return chi = N*Var(q) split into
    first-half and second-half estimates (equilibration guard)."""
    J = sk_couplings(N, rng)
    sa = rng.choice([-1.0, 1.0], size=N)
    sb = rng.choice([-1.0, 1.0], size=N)
    for _ in range(n_equil):
        metropolis_sweep(sa, J, beta, rng)
        metropolis_sweep(sb, J, beta, rng)
    qs = []
    for t in range(n_meas):
        metropolis_sweep(sa, J, beta, rng)
        metropolis_sweep(sb, J, beta, rng)
        if t % meas_every == 0:
            qs.append(float(np.dot(sa, sb) / N))
    qs = np.array(qs)
    half = len(qs) // 2
    chi = lambda q: N * (np.mean(q**2) - np.mean(q)**2)
    return chi(qs), chi(qs[:half]), chi(qs[half:])


def run(Ns=(32, 64), Ts=(0.6, 0.8, 1.0, 1.3, 2.0), n_disorder=8, seed=0):
    rng = np.random.default_rng(seed)
    print("Phase-0 control: SK spin glass, T_c = 1.0 (known continuous SG transition)")
    print("chi_SG = N*Var(q) from two replicas; disorder-averaged over", n_disorder, "realizations")
    print(f"{'N':>4} {'T':>5} {'chi_SG':>9} {'chi_1sthalf':>12} {'chi_2ndhalf':>12} {'equil?':>7}")
    table = {}
    for N in Ns:
        for T in Ts:
            beta = 1.0 / T
            cs, c1, c2 = [], [], []
            for _ in range(n_disorder):
                c, ca, cb = chi_sg_two_replicas(N, beta, rng)
                cs.append(c); c1.append(ca); c2.append(cb)
            chi, h1, h2 = np.mean(cs), np.mean(c1), np.mean(c2)
            # equilibration guard: the two measurement halves must agree to ~20%
            equil = "ok" if abs(h1 - h2) <= 0.20 * max(chi, 1e-9) else "DRIFT"
            table[(N, T)] = chi
            print(f"{N:>4} {T:>5.2f} {chi:>9.3f} {h1:>12.3f} {h2:>12.3f} {equil:>7}")
    print()
    # POSITIVE control verdict: chi at T_c grows with N
    print("Positive-control read (chi at/near T_c=1 should grow with N):")
    for T in (1.0,):
        vals = [(N, table[(N, T)]) for N in Ns]
        grew = vals[-1][1] > vals[0][1]
        print(f"  T={T}: " + " -> ".join(f"N{N}:{c:.2f}" for N, c in vals)
              + f"   {'GROWS with N (FIRES)' if grew else 'flat/no-FSS'}")
    # NEGATIVE control read: chi at high T small and ~N-independent
    print("Negative-control read (chi at high T should stay small / flat):")
    for T in (2.0,):
        vals = [(N, table[(N, T)]) for N in Ns]
        print(f"  T={T}: " + " -> ".join(f"N{N}:{c:.2f}" for N, c in vals))
    return table


if __name__ == "__main__":
    run()
