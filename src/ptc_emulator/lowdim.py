"""Low-dimensional Floquet engine for time-periodic second-order systems.

Equation (all frequencies in units of a reference frequency, dimensionless):

    q'' + 2*gamma*q' + omega0^2 * (1 + m*(1 + cos(omega_m*t))) * q = 0

with ' = d/dtau, tau = omega_ref * t. The modulation form matches the
Kerr-background convention  Delta_eps(t)/eps_inf = 1 + m*(1 + cos(omega_m t)):
mean 1 + m, AC amplitude m, each +/-omega_m sideband carrying m/2.

Conventions
-----------
- Phasor/time convention exp(-i*omega*t): growth <=> Im(omega) > 0
  <=> Floquet exponent lambda = ln|mu|/T > 0 <=> |mu| > 1.
- Instability criterion is max|mu| (equivalently Re lambda). For the real
  2x2 monodromy of this equation with 0 < d = det M < 1, the strict Jury
  condition |tr M| < 1 + d is equivalent; |tr M| > 2 is only the d = 1
  (lossless) special case and is NOT a valid damped test.
- Harmonic index n in Z, ladder omega + n*omega_m; J = 2N + 1 counts the
  retained ladder points (N is the half-width).

Methods
-------
- monodromy: M over one drive period by direct integration (DOP853).
- hill_matrix / hill_candidates / hill_monodromy: harmonic-balance
  eigenproblem with differential-residual and harmonic-tail screening
  before any pair is used.
- time_domain_growth: renormalised long-run direct integration; the log of
  per-period maxima of |q| is exactly linear for a Floquet mode with slope
  Re(lambda).

Every Hill candidate carries a differential residual and a tail ratio. A
truncated eigenvector satisfies the truncated system exactly, so its
residual against the full equation lives entirely one harmonic outside the
window: the cosine couples N to N+1, leaving |W0^2 (m/2) c_N| unbalanced
(and likewise at -N). That measured residual, not wishful convergence, is
what qualifies a candidate.

This module performs no file I/O and has no import side effects.
Provenance: adapted from an independently reviewed private engine whose
audit record (gates, counterexamples) is retained with the project evidence.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "monodromy",
    "growth_and_stability",
    "hill_matrix",
    "hill_candidates",
    "hill_monodromy",
    "cond2_normalised",
    "time_domain_growth",
    "MODULATION_FORM",
]

MODULATION_FORM = "1 + m*(1 + cos(omega_m*t))"


def _rhs_factory(omega0, gamma, m, omega_m):
    def rhs(tau, y):
        q, qd = y
        drive = 1.0 + m * (1.0 + np.cos(omega_m * tau))
        return [qd, -2.0 * gamma * qd - omega0 * omega0 * drive * q]
    return rhs


def monodromy(omega0, gamma, m, omega_m, rtol=1e-13, atol=1e-15):
    """Monodromy over one drive period on the (q, dq/dtau) state."""
    T = 2.0 * np.pi / omega_m
    rhs = _rhs_factory(omega0, gamma, m, omega_m)
    M = np.zeros((2, 2))
    for j, y0 in enumerate(([1.0, 0.0], [0.0, 1.0])):
        sol = solve_ivp(rhs, [0.0, T], y0, rtol=rtol, atol=atol,
                        method="DOP853", dense_output=False)
        if not sol.success:
            raise RuntimeError("integration failed: {}".format(sol.message))
        M[:, j] = sol.y[:, -1]
    return M, T


def growth_and_stability(M, T):
    """Growth rate (two routes, cross-checked) with a DOMAIN-GUARDED Jury test.

    The multiplier modulus is authoritative everywhere: growth is read from
    max|mu| (cross-checked against max Re log(mu)/T; they are equal in exact
    arithmetic, so disagreement flags an eigenvalue problem, not physics).

    The Jury form |tr M| < 1 + det M is the Schur condition for a REAL 2x2
    monodromy with 0 < det <= 1 only; inside that domain "STABLE" means
    BOUNDED / NON-GROWING (all |mu| <= 1), NOT strictly decaying — strict
    decay additionally requires det < 1, and det == 1 is the lossless
    marginal boundary (e.g. a pure rotation: zero growth, IN_DOMAIN,
    STABLE). Outside the domain the form is not applied: a complex-dtyped M
    (Hill pairings build complex combinations) must not have its trace/det
    imaginary parts silently discarded, det > 1 already implies
    max|mu| > 1 (product of moduli = |det|), and det <= 0 or non-finite
    input is outside the passive real domain entirely. `jury_status`
    reports why; `verdict_by_jury`/`unstable_by_jury` are None unless
    jury_status == "IN_DOMAIN".
    """
    M = np.asarray(M)
    if M.shape != (2, 2):
        raise ValueError("M must be 2x2, got shape %s" % (M.shape,))
    # layout-independent finite check: real/imag parts exist for any dtype
    # (a .view(float) reinterpretation breaks on Fortran/non-contiguous
    # complex input, which is a valid matrix in every other respect)
    if not (np.all(np.isfinite(np.real(M))) and np.all(np.isfinite(np.imag(M)))):
        raise ValueError("M contains non-finite entries")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive, got %r" % (T,))

    mu = np.linalg.eigvals(M)
    lam = np.log(mu.astype(complex)) / T
    growth_from_log = float(max(x.real for x in lam))
    max_mod = float(max(abs(x) for x in mu))
    growth_from_modulus = float(np.log(max_mod) / T)

    tr_c = complex(np.trace(M))
    d_c = complex(np.linalg.det(M))
    tr = float(tr_c.real)
    d = float(d_c.real)
    # measured, never assumed: the largest imaginary part that WOULD be
    # discarded by reading trace/det as real
    imaginary_part_discarded = float(max(abs(tr_c.imag), abs(d_c.imag)))
    imag_tolerance = 1e-10 * (1.0 + abs(tr) + abs(d))

    if imaginary_part_discarded > imag_tolerance:
        jury_status = "NONREAL_MATRIX"
    elif d <= 0.0:
        jury_status = "NONPOSITIVE_DET"
    elif d > 1.0:
        jury_status = "DET_ABOVE_ONE"
    else:
        jury_status = "IN_DOMAIN"

    if jury_status == "IN_DOMAIN":
        boundary = 1.0 + d
        if abs(tr) < boundary:
            verdict = "STABLE"
        elif abs(tr) == boundary:
            verdict = "BOUNDARY"
        else:
            verdict = "UNSTABLE"
    else:
        boundary = None
        verdict = None

    discriminant = tr * tr - 4.0 * d
    return {
        "growth_from_log": growth_from_log,
        "growth_from_modulus": growth_from_modulus,
        "growth_route_difference": abs(growth_from_log - growth_from_modulus),
        "max_modulus": max_mod,
        "unstable_by_modulus": bool(max_mod > 1.0),
        "trace": tr,
        "abs_trace": abs(tr),
        "det": d,
        "imaginary_part_discarded_from_trace_and_det": imaginary_part_discarded,
        "jury_status": jury_status,
        "jury_boundary_1_plus_d": boundary,
        "verdict_by_jury": verdict,
        "unstable_by_jury": (verdict == "UNSTABLE") if verdict is not None else None,
        "discriminant": discriminant,
        "multipliers_are_real": bool(discriminant >= 0.0),
    }


def hill_matrix(omega0, gamma, m, omega_m, N):
    """Companion matrix of the Hill problem truncated to harmonics -N..N.

    Substituting q = exp(Lam tau) sum_n c_n exp(i n w tau) gives the
    quadratic eigenproblem Lam^2 I + Lam C + K = 0 with
        C = diag(2 (i n w + G))
        K = diag(-n^2 w^2 + 2 i G n w + W0^2 (1+m)) + W0^2 (m/2) (shift+1 + shift-1)
    linearised as [[0, I], [-K, -C]]; its eigenvalues are the Floquet
    exponents.
    """
    w = omega_m
    k = np.arange(-N, N + 1)
    n = len(k)
    K = np.zeros((n, n), dtype=complex)
    C = np.zeros((n, n), dtype=complex)
    idx = {int(kk): i for i, kk in enumerate(k)}
    for i, kk in enumerate(k):
        K[i, i] = (-(kk * w) ** 2 + 2j * gamma * kk * w + omega0 ** 2 * (1.0 + m))
        C[i, i] = 2.0 * (1j * kk * w + gamma)
        for dlt in (+1, -1):
            j = idx.get(int(kk) + dlt)
            if j is not None:
                K[i, j] += omega0 ** 2 * m / 2.0
    I = np.eye(n, dtype=complex)
    A = np.block([[np.zeros((n, n), dtype=complex), I], [-K, -C]])
    return A, k


def hill_candidates(omega0, gamma, m, omega_m, N):
    """Harmonic eigenvectors as initial/period-end states, with screening.

    For each eigenpair (Lam, c):  q(0) = sum c_n, q'(0) = sum (Lam + i n w) c_n,
    and the same with an extra exp(Lam T) at tau = T. Each candidate carries
    diff_residual (residual of the truncated solution against the FULL
    equation, normalised by the term norms) and tail_ratio (outermost
    coefficients over all coefficients -- amplitude at the window edge means
    truncated, not resolved).
    """
    A, k = hill_matrix(omega0, gamma, m, omega_m, N)
    n = len(k)
    evals, V = np.linalg.eig(A)
    T = 2.0 * np.pi / omega_m
    wm = float(omega_m)
    kk_vec = k.astype(float)

    out = []
    for i in range(len(evals)):
        Lam = evals[i]
        c = V[:n, i]
        resid = np.zeros(n, dtype=complex)
        for a, kk in enumerate(k):
            if kk == k[0] or kk == k[-1]:
                resid[a] = omega0 ** 2 * m / 2.0 * c[a]
        norm_resid = float(np.linalg.norm(resid))
        term_inertia = float(np.linalg.norm((Lam + 1j * wm * kk_vec) ** 2 * c))
        term_damp = float(np.linalg.norm(2.0 * gamma * (Lam + 1j * wm * kk_vec) * c))
        term_stiff = float(np.linalg.norm(omega0 ** 2 * (1.0 + m) * c))
        denom = term_inertia + term_damp + term_stiff
        diff_resid = float(norm_resid / denom) if denom > 0 else float("inf")
        tail = float((abs(c[0]) ** 2 + abs(c[-1]) ** 2) ** 0.5
                     / max(np.linalg.norm(c), 1e-300))
        q0 = complex(c.sum())
        qd0 = complex(((Lam + 1j * wm * kk_vec) * c).sum())
        e = np.exp(Lam * T)
        out.append({
            "index": int(i),
            "Lambda": complex(Lam),
            "coeffs": np.array(c, dtype=complex, copy=True),
            "harmonics": np.array(k, dtype=int, copy=True),
            "diff_residual": diff_resid,
            "tail_ratio": tail,
            "state0": np.array([q0, qd0], dtype=complex),
            "stateT": np.array([e * q0, e * qd0], dtype=complex),
        })
    return out, k, evals


def cond2_normalised(Y):
    """cond_2 of a 2x2 state matrix after column normalisation.

    max|det Y| is not scale invariant and is not a condition number; column
    normalisation removes the scaling freedom first.
    """
    Yn = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    return float(np.linalg.cond(Yn))


def hill_monodromy(omega0, gamma, m, omega_m, N,
                   max_diff_residual=1e-8, max_tail_ratio=1e-6, screen=True):
    """Recover the monodromy from screened Hill eigenvector pairs.

    Screening runs first; if fewer than two candidates qualify the result is
    UNRESOLVED and the function does NOT fall back to whichever pair happens
    to match a known monodromy. RESOLVED additionally requires a runner-up
    pair (n_pairs >= 2): a single pair, however good, cannot show
    self-consistency. screen=False (select by conditioning alone from ALL
    candidates) exists only to demonstrate that conditioning alone is
    insufficient -- a well-conditioned spurious pair is reproducible -- and
    must never back a reported result; pool_name travels in the output.
    """
    cands, k, w = hill_candidates(omega0, gamma, m, omega_m, N)
    if screen:
        pool = [c for c in cands
                if c["diff_residual"] <= max_diff_residual
                and c["tail_ratio"] <= max_tail_ratio]
        pool_name = "screened"
    else:
        pool = list(cands)
        pool_name = "unscreened_conditioning_only"

    pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            Y = np.array([pool[i]["state0"], pool[j]["state0"]]).T
            cn = cond2_normalised(Y)
            if not np.isfinite(cn) or cn < 1e-12:
                continue
            M = np.array([pool[i]["stateT"], pool[j]["stateT"]]).T @ np.linalg.inv(Y)
            pairs.append({"i": pool[i]["index"], "j": pool[j]["index"],
                          "cond2": cn, "M": M})
    pairs.sort(key=lambda p: p["cond2"])

    return {
        "pool_name": pool_name,
        "n_candidates": len(cands),
        "n_pool": len(pool),
        "n_qualified": len(pool) if screen else None,
        "thresholds": {"max_diff_residual": max_diff_residual,
                       "max_tail_ratio": max_tail_ratio},
        "best_pair": pairs[0] if pairs else None,
        "runner_up_pair": pairs[1] if len(pairs) > 1 else None,
        "n_pairs": len(pairs),
        "status": "RESOLVED" if len(pairs) >= 2 else "UNRESOLVED",
        "min_diff_residual": float(min(c["diff_residual"] for c in cands)),
        "min_tail_ratio": float(min(c["tail_ratio"] for c in cands)),
    }


def time_domain_growth(omega0, gamma, m, omega_m, n_periods=90, discard=12,
                       rtol=1e-12, atol=1e-14):
    """Growth rate from a renormalised direct time-domain integration.

    Integrates period by period, rescaling the state each period (nothing
    overflows), and records the UNnormalised per-period max of |q|. For a
    Floquet mode q = exp(lam tau) p(tau), p periodic, the log of those maxima
    is exactly linear with slope Re(lam); least squares over periods >=
    discard (after transients) gives the growth, with a linearity residual
    that measures the transient contamination rather than hiding it.
    """
    T = 2.0 * np.pi / omega_m
    rhs = _rhs_factory(omega0, gamma, m, omega_m)
    y = np.array([1.0, 0.3])
    log_scale = 0.0
    samples = []
    for k in range(n_periods):
        sol = solve_ivp(rhs, [0.0, T], y, rtol=rtol, atol=atol,
                        method="DOP853", dense_output=True)
        if not sol.success:
            raise RuntimeError("time-domain integration failed: %s" % sol.message)
        tau_grid = np.linspace(0.0, T, 241)
        q_abs = np.abs(sol.sol(tau_grid)[0])
        unmax = float(np.log(np.max(q_abs)) + log_scale)
        samples.append((k, (k + 1) * T, unmax))
        y = sol.y[:, -1]
        nrm = float(np.linalg.norm(y))
        y = y / nrm
        log_scale += np.log(nrm)
    fit = [(t, lg) for (k, t, lg) in samples if k >= discard]
    ts = np.array([f[0] for f in fit])
    lgs = np.array([f[1] for f in fit])
    A = np.vstack([ts, np.ones_like(ts)]).T
    coeff, *_ = np.linalg.lstsq(A, lgs, rcond=None)
    slope = float(coeff[0])
    resid = float(np.max(np.abs(A @ coeff - lgs)))
    return {
        "growth_fit": slope,
        "fit_linearity_residual": resid,
        "n_periods": n_periods,
        "discard": discard,
    }
