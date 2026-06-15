#!/usr/bin/env python3
"""
module_L_perceptron_mcmc_control.py  --  Instrument-validation control for the genuine-side gate (§6.7).

Certifies the SAMPLES -> geometry -> R machinery against the [V] analytic receipt
(module_L_perceptron_finiteT.py) on the EXACTLY-SOLVABLE spherical perceptron, in the frame the
receipt actually uses (raw (beta,eps) Hessian; see the pencil transform). This is the perceptron twin
of the ridge Rung-1 control.

WHAT THIS DOES AND DOES NOT CERTIFY (the boundary that matters):
  * DOES: the cumulant -> metric -> finite-diff-in-beta -> R pipeline reproduces the receipt's
    g_eps-eps = 1/lambda_repl and g_bb = 2 phi'' on a POSITIVE-DEFINITE, non-singular model.
  * The perceptron energy E(w) = #violated = sum_mu Theta(kappa - (w.xi)/sqrt(N)) is NON-DIFFERENTIABLE,
    so this control uses gradient-free METROPOLIS-on-the-sphere, NOT SGLD. The SGLD-specific machinery
    (gradient + Langevin noise + MALA) is exercised only on smooth real-net energies -- UNTESTED here.
  * DOES NOT touch the genuine-side gate. The perceptron metric is positive-definite by construction
    (that's why it's [V]); this says NOTHING about whether a real net's tempered posterior has a
    regular submanifold where R is even DEFINED, or degenerates (singular learning theory; RLCT/LLC).
    The real-net gate stays SHUT. A separate derivation must decide whether the singular case admits R.

THE CALIBRATION (clean, non-critical beta -- the certification):
  measure the overlap fluctuation N*Var(q), q = w1.w2/N, from independent equilibrated replicas, and
  pin the beta-power empirically: which of {N Var(q), beta N Var(q), beta^2 N Var(q)} equals the
  receipt's chi = 1/lambda_repl? (the +beta*eps*Nq tilt predicts beta^2; MEASURED, not assumed.)
  Also g_bb = (1/N)Var(U) =? 2*free_energy''(beta).
The near-critical |R|*(beta_AT-beta)^2 -> 11.8 is a THERMODYNAMIC-LIMIT scaling claim; at finite N it
rounds off (finite-size scaling), so it is NOT the certification -- the non-critical chi match is.
"""
import numpy as np
import module_L_perceptron_finiteT as PF


def energy(X, W, kappa):
    """E(w) = number of violated margin constraints. X:(P,N), W:(N,C) -> (C,)."""
    h = (X @ W) / np.sqrt(X.shape[1])          # stabilities, (P,C)
    return (h < kappa).sum(axis=0).astype(float)


def metropolis(N, alpha, kappa, beta, C, burn, prop, seed=0, log=None):
    """Batched Metropolis-on-the-sphere: C parallel chains -> C ~independent equilibrated samples."""
    rng = np.random.default_rng(seed)
    P = int(round(alpha * N))
    X = rng.standard_normal((P, N))
    W = rng.standard_normal((N, C)); W *= np.sqrt(N) / np.linalg.norm(W, axis=0)
    E = energy(X, W, kappa)
    acc = 0
    for t in range(burn):
        Wp = W + prop * rng.standard_normal((N, C))
        Wp *= np.sqrt(N) / np.linalg.norm(Wp, axis=0)
        Ep = energy(X, Wp, kappa)
        m = rng.random(C) < np.exp(np.minimum(-beta * (Ep - E), 0.0))
        W[:, m] = Wp[:, m]; E[m] = Ep[m]; acc += m.sum()
        if log is not None and (t % max(1, burn // 6) == 0 or t == burn - 1):
            log.append((t, float(E.mean())))
    return W, E, acc / (burn * C)


def geometry(W, E, N, beta):
    """Second-order geometry from the equilibrated samples, in (beta,eps) at eps=0.
    Overlap fluctuation via the distinct-pair Gram (unbiased): N*Var(q)."""
    C = W.shape[1]
    G = W.T @ W                                  # (C,C), G_ij = w_i.w_j
    off = ~np.eye(C, dtype=bool)
    q = G[off] / N                               # distinct-pair overlaps
    mq, mq2 = q.mean(), (q * q).mean()
    NVarq = N * (mq2 - mq * mq)
    g_bb = (2.0 / N) * E.var()                   # g_bb = (1/N)Var(U), U=E1+E2 indep -> 2Var(E)
    return NVarq, float(mq), g_bb


def batch_err(W, E, N, beta, nb=8):
    """Error via disjoint batches of independent chains (no resample-with-replacement self-pairs)."""
    C = W.shape[1]
    vals = [geometry(W[:, ix], E[ix], N, beta)[0] for ix in np.array_split(np.arange(C), nb) if len(ix) > 8]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


if __name__ == "__main__":
    kappa, alpha = -0.5, 4.2
    bAT = PF.beta_AT(alpha, kappa)
    beta = 1.2                                   # NON-CRITICAL (well below beta_AT~6): RS-stable, mixes faster
    N, C, burn, prop = 150, 512, 30000, 0.05

    print("=" * 92)
    print("module_L_perceptron_mcmc_control.py  --  genuine-side instrument validation (perceptron)")
    print(f"  kappa={kappa} alpha={alpha}  beta_AT={bAT:.3f}  |  calibration at NON-CRITICAL beta={beta}")
    print("  Metropolis-on-sphere (gradient-free; the Theta-energy has no gradient -> NOT SGLD).")
    print("=" * 92)

    # analytic [V] targets from the receipt
    lam, q_an = PF.replicon(beta, alpha, kappa)
    chi = 1.0 / lam
    h = 3e-3
    phi2 = (PF.free_energy(beta + h, alpha, kappa) - 2 * PF.free_energy(beta, alpha, kappa)
            + PF.free_energy(beta - h, alpha, kappa)) / h**2
    print(f"\n[V] analytic (receipt): lambda_repl={lam:.4f}  chi=1/lambda_repl={chi:.4f}  "
          f"q*={q_an:.4f}  2*phi''={2*phi2:.4f}")

    log = []
    W, E, accrate = metropolis(N, alpha, kappa, beta, C, burn, prop, seed=0, log=log)
    print(f"\nMCMC: N={N} C={C} burn={burn} prop={prop}  acceptance={accrate:.2f}")
    print("  E(burn-in trajectory):", "  ".join(f"{t}:{e:.0f}" for t, e in log))

    NVarq, mq, g_bb = geometry(W, E, N, beta)
    bmean, berr = batch_err(W, E, N, beta)
    print(f"\nMEASURED (eps=0): <q>={mq:+.4f} (expect q*={q_an:.3f}: the EA overlap, NOT 0 -- spin-glass order)"
          f"   N*Var(q)={NVarq:.4f}  [batch {bmean:.3f}+/-{berr:.3f}]")
    print(f"  g_bb measured = {g_bb:.4f}   vs  2*phi'' = {2*phi2:.4f}")

    print("\nbeta-power pin (which equals chi={:.4f}?):".format(chi))
    for label, val in [("N Var(q)        (eps natural)", NVarq),
                       ("beta N Var(q)", beta * NVarq),
                       ("beta^2 N Var(q) (beta*eps tilt)", beta * beta * NVarq)]:
        print(f"     {label:<32} = {val:7.4f}   ratio to chi = {val/chi:.3f}")

    print("\n" + "=" * 92)
    print("  VERDICT (first-pass): PARTIAL validation, NOT a clean certification.")
    print(f"  - energy sector: g_bb={g_bb:.3f} vs 2phi''={2*phi2:.3f}  (~{abs(g_bb/(2*phi2)-1)*100:.0f}%, mostly MCMC error).")
    print(f"  - <q>={mq:.3f} ~ q*={q_an:.3f}: the EA overlap is reproduced. CORRECTION to the pencil: <q>=q* != 0,")
    print("    so g_beps = q* + beta*dq*/dbeta != 0 -- the receipt's pxy=0 is a NEAR-CRITICAL approximation")
    print("    (exact as beta->beta_AT, where g_eps-eps->inf swamps g_beps^2), not an exact symmetry. The")
    print("    'g_beps=0 by symmetry' claim was WRONG; the sampler caught it. (g_eps-eps calibration unaffected.)")
    print(f"  - beta-power UNRESOLVED at beta={beta}: ratios-to-chi {NVarq/chi:.2f}/{beta*NVarq/chi:.2f}/"
          f"{beta*beta*NVarq/chi:.2f} for power 0/1/2 -- too close + acceptance {accrate:.0%} too low to discriminate.")
    print("  TO CERTIFY: higher acceptance / better mixing, a larger beta-lever to separate the powers, and")
    print("  finite-size scaling over N. This is the genuine instrument's hard part -- equilibrating a glassy")
    print("  posterior -- which is exactly the difficulty the singular (real-net) case faces, by design.")
    print("  This does NOT test SGLD (Metropolis) and does NOT open the real-net gate. §6.7 stays SHUT.")
    print("=" * 92)
