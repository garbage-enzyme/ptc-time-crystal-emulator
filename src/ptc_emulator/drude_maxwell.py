"""Augmented Maxwell-Drude system at fixed real k (dispersive time-varying
medium).

Normalized working equations (callers fix scales; c0 = eps0 = 1 in the
declared units): transverse plane wave exp(ikz), complex amplitudes,
state y = (D, B, S, P) with

    Ddot = i k B,   Bdot = i k E,   E = (D - P)/eps_inf,
    Sdot = E - gamma(t) S,          Pdot = wp2(t) S,

i.e. the accepted causal time-varying Drude memory states (see
``ptc_emulator.drude``) coupled to Maxwell at a single real wavenumber.

Energy identity (exact, for the single-k complex-amplitude
representation): with

    w = 0.5 eps_inf |E|^2 + 0.5 |B|^2 + 0.5 wp2(t) |S|^2

(field plus carrier kinetic energy; P is bookkeeping and carries no
independent energy) the Poynting flux terms cancel in the |.|^2
bookkeeping and

    dw/dt = 0.5 wp2'(t)|S|^2 - gamma(t) wp2(t)|S|^2,

so constant positive gamma with no modulation is passive, while
modulation work can grow the total field even for gamma > 0.

Static anchors: the constant-coefficient dispersion is the Drude cubic

    eps_inf w^3 + i eps_inf gamma w^2 - (wp2 + k^2) w - i gamma k^2 = 0

(eigenvalues of the constant augmented matrix are lambda = -i w), and for
periodic coefficients with period T the one-period monodromy M satisfies
|det M| = exp(-int_0^T gamma dt).

The harmonic (Floquet/Hill) solver is derived independently of the ODE
integration: with y = e^{lambda t} sum_n c_n e^{-i n Omega t} and
A(t) = A0 + sum_p A_p e^{i p Omega t}, the blocks satisfy

    lambda c_n = (A0 - i n Omega I) c_n + sum_p A_p c_{n+p}.

All ladder members of one Floquet exponent share the same multiplier
exp(lambda T) (Omega T = 2 pi), so multipliers are recovered by reducing
the truncated spectrum modulo Omega in the imaginary direction and
clustering.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "MODULATION_NOTE",
    "augmented_rhs",
    "direct_time_growth",
    "drude_dispersion_roots",
    "harmonic_matrix",
    "match_sets",
    "monodromy_augmented",
    "multipliers_from_harmonic",
    "neutral_direction",
    "project_out_neutral",
    "solve_augmented",
    "static_matrix",
]

MODULATION_NOTE = (
    "augmented Maxwell-Drude states at fixed real k; energy identity "
    "dw/dt = 0.5*wp2'(t)|S|^2 - gamma(t)*wp2(t)*|S|^2 (modulation work "
    "minus dissipation)")


def augmented_rhs(k, eps_inf, wp2_fn, gamma_fn):
    """RHS of the augmented 4-state system for ``solve_ivp`` (complex y)."""

    def rhs(t, y):
        E = (y[0] - y[3]) / eps_inf
        return [1j * k * y[1],
                1j * k * E,
                E - gamma_fn(t) * y[2],
                wp2_fn(t) * y[2]]

    return rhs


def solve_augmented(t_eval, k, eps_inf, wp2_fn, gamma_fn, y0,
                    rtol=1e-12, atol=1e-14):
    """Integrate the augmented system with DOP853; returns y on t_eval."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(augmented_rhs(k, eps_inf, wp2_fn, gamma_fn),
                    (t_eval[0], t_eval[-1]), list(y0), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("augmented integration failed: %s" % sol.message)
    return sol.y


def monodromy_augmented(T, k, eps_inf, wp2_fn, gamma_fn,
                        rtol=1e-12, atol=1e-14):
    """One-period monodromy matrix (4x4 complex) from 4 identity ICs."""
    M = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        y0 = np.zeros(4, dtype=complex)
        y0[j] = 1.0
        M[:, j] = solve_augmented(np.array([0.0, T]), k, eps_inf,
                                  wp2_fn, gamma_fn, y0,
                                  rtol=rtol, atol=atol)[:, -1]
    return M


def static_matrix(k, eps_inf, wp2, gamma):
    """Constant-coefficient augmented matrix A (ydot = A y)."""
    ik = 1j * k
    return np.array([
        [0.0, ik, 0.0, 0.0],
        [ik / eps_inf, 0.0, 0.0, -ik / eps_inf],
        [1.0 / eps_inf, 0.0, -gamma, -1.0 / eps_inf],
        [0.0, 0.0, wp2, 0.0]], dtype=complex)


def drude_dispersion_roots(k, eps_inf, wp2, gamma):
    """Roots of the static Drude dispersion cubic (ascending none;
    unordered). Frequencies w satisfy lambda = -i w."""
    return np.roots([eps_inf, 1j * eps_inf * gamma,
                     -(wp2 + k * k), -1j * gamma * k * k])


def match_sets(a, b):
    """Permutation min-max distance between two small unordered sets."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.size != b.size:
        return np.inf
    n = a.size
    best = np.inf
    idx = np.arange(n)
    from itertools import permutations
    for perm in permutations(range(n)):
        d = np.max(np.abs(a[idx] - b[list(perm)])) if n else 0.0
        best = min(best, float(d))
    return best


def harmonic_matrix(k, eps_inf, wp2_dc, wp2_p1, gamma_dc, gamma_p1,
                    Omega, N):
    """Block-tridiagonal Floquet matrix for single-harmonic (cos)
    coefficients wp2(t) = wp2_dc + 2 Re[wp2_p1 e^{i Omega t}] (pass the
    complex half-amplitude wp2_p1; likewise gamma). Truncation |n| <= N.
    Eigenvalues approximate the Floquet-exponent ladder lambda - i n Omega.
    """
    A0 = static_matrix(k, eps_inf, wp2_dc, gamma_dc)
    Ap = np.zeros_like(A0)         # coefficient of e^{+i Omega t}
    Ap[2, 2] = -gamma_p1          # Sdot row: -gamma(t) S
    Ap[3, 2] = wp2_p1             # Pdot row: wp2(t) S
    Am = np.zeros_like(A0)         # coefficient of e^{-i Omega t} (conj)
    Am[2, 2] = -np.conj(gamma_p1)
    Am[3, 2] = np.conj(wp2_p1)
    n_blocks = 2 * N + 1
    size = 4 * n_blocks
    H = np.zeros((size, size), dtype=complex)
    for bi, n in enumerate(range(-N, N + 1)):
        d = bi * 4
        # lambda c_n = (A0 + i n Omega) c_n + A+ c_{n+1} + A- c_{n-1}
        # (from d/dt[e^{lam t} c_n e^{-i n Om t}] = (lam - i n Om),
        #  moved to the right-hand side with +i n Om)
        H[d:d + 4, d:d + 4] = A0 + 1j * n * Omega * np.eye(4)
        if bi + 1 < n_blocks:
            H[d:d + 4, d + 4:d + 8] = Ap
        if bi - 1 >= 0:
            H[d:d + 4, d - 4:d] = Am
    return H


def neutral_direction():
    """The exactly neutral polarization direction (D, 0, 0, P = D):
    E = B = S = 0 leaves it invariant (lambda = 0 for any k > 0).
    The static dispersion cubic has three physical roots; the augmented
    system carries this fourth neutral mode, which physical zero-history
    initial data never excites."""
    n = np.zeros(4, dtype=complex)
    n[0] = n[3] = 1.0
    return n / np.linalg.norm(n)


def project_out_neutral(y0):
    """Remove the neutral polarization component from initial data."""
    n = neutral_direction()
    return np.asarray(y0, dtype=complex) \
        - np.vdot(n, np.asarray(y0, dtype=complex)) * n


def multipliers_from_harmonic(H, Omega, T, n_states=4):
    """Recover Floquet multipliers from the truncated harmonic spectrum.

    Ladder copies of one Floquet exponent differ by exactly i*Omega, and
    copies near the truncation edge are polluted while central copies are
    accurate, so the physical set is the CENTRAL STRIP |Im lambda| <
    Omega/2 (one copy per exponent, farthest from the edges). Cluster
    averaging is deliberately avoided: it mixes in edge copies and biases
    the result at the 1e-5*m level. Returns (multipliers,
    floquet_exponents). Raises if the strip does not contain exactly
    n_states eigenvalues."""
    eig = np.linalg.eigvals(H)
    central = eig[np.abs(np.imag(eig)) < 0.5 * Omega]
    if central.size != n_states:
        raise ValueError("central strip holds %d eigenvalues (need %d)"
                         % (central.size, n_states))
    mult = np.exp(central * T)
    return mult, central


def direct_time_growth(T, n_periods, k, eps_inf, wp2_fn, gamma_fn,
                       discard=12, rtol=1e-12, atol=1e-14):
    """Direct-time growth from the leading monodromy eigenvector with
    per-period renormalization (production protocol: n_periods samples,
    first `discard` dropped). Each sample records the log norm grown in
    one period from a unit-norm start, so the growth rate is the mean
    kept increment divided by T. Returns (growth_rate, samples) with
    samples = [(j, t_j, log_norm)]."""
    M = monodromy_augmented(T, k, eps_inf, wp2_fn, gamma_fn,
                            rtol=rtol, atol=atol)
    vals, vecs = np.linalg.eig(M)
    lead = int(np.argmax(np.abs(vals)))
    v = vecs[:, lead]
    v = v / np.linalg.norm(v)
    samples = []
    for j in range(n_periods):
        y = solve_augmented(np.array([0.0, T]), k, eps_inf,
                            wp2_fn, gamma_fn, v, rtol=rtol, atol=atol)
        log_norm = float(np.log(np.linalg.norm(y[:, -1])))
        samples.append((j, (j + 1) * T, log_norm))
        v = y[:, -1] / np.linalg.norm(y[:, -1])
    kept = samples[discard:]
    rate = float(np.mean([s[2] for s in kept]) / T)
    return rate, samples
