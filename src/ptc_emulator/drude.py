"""Causal time-varying Drude response: memory-state form and kernel quadrature.

Normalized working equations (callers fix their own scales; eps0 may be
folded into the polarization scale):

    Sdot = E(t) - gamma(t) * S        # S carries the causal field memory
    Pdot = wp2(t) * S                  # polarization;  J := Pdot
    D    = eps_inf * E + P             # displacement (eps_inf instantaneous part)

``S`` is proportional to the carrier momentum per unit charge of the
momentum-form Drude equation ``d/dt[m*(t)v] + m*(t)gamma(t)v = qE`` (which
is Markovian in the momentum); the velocity/current form is what becomes
non-Markovian when ``m*`` or the density vary.  The equivalent two-time
causal kernel is

    chi(t, t') = Theta(t - t') * int_{t'}^{t} wp2(s)
                 * exp(-int_{t'}^{s} gamma(u) du) ds,

with polarization ``P(t) = int_{t0}^{t} chi(t, t') E(t') dt'`` for zero
initial history (field and states identically zero before ``t0``).  Because
the inner integrand factorizes as ``wp2(s)e^{-G(s)} * e^{G(t')}`` with
``G = int gamma``, the double integral equals ``B(t)A(t) - D(t)`` with

    B = int wp2 e^{-G},   A = int e^{G} E,   D = int e^{G} B E,

each computed here by high-order composite Gauss-Legendre quadrature on a
master grid (``Cumulative`` supports vectorized queries between nodes via
local panel Gauss).  ``kernel_chi_literal`` evaluates the defining inner
integral directly on its own local grid for independent verification.

For smooth, strictly positive ``wp2`` the current form

    Jdot = [dlog(wp2)/dt - gamma] J + wp2 * E

is equivalent to the S/P system (``solve_j_log_form``).  The *stationary*
auxiliary-differential-equation update ``Jdot = -gamma*J + wp2*E`` (no
``dlog(wp2)/dt`` term, ``stationary_ade_rhs``) is a DIFFERENT model
whenever ``wp2`` varies in time; it is provided only so tests can
demonstrate that difference.

All profile functions must accept numpy arrays or scalars (vectorized
evaluation is used on quadrature nodes).
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "MODULATION_NOTE",
    "Cumulative",
    "drude_history",
    "kernel_chi_literal",
    "memory_state_rhs",
    "pulse_envelope",
    "solve_j_log_form",
    "solve_memory_state",
    "stationary_ade_rhs",
    "stationary_chi",
]

MODULATION_NOTE = (
    "time-varying Drude memory-state realization; equivalent to the causal "
    "two-time kernel with zero initial history"
)


@functools.lru_cache(maxsize=8)
def _leggauss(order):
    return np.polynomial.legendre.leggauss(order)


class Cumulative:
    """Cumulative integral of a vectorized integrand from the grid start.

    Panel integrals between consecutive grid nodes use composite
    Gauss-Legendre quadrature of the given order; queries between nodes
    add a local Gauss panel, so the result is a high-accuracy antiderivative
    valid anywhere in ``[x[0], x[-1]]``."""

    def __init__(self, x, integrand, order=6):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("x must be a strictly increasing 1-D grid")
        self.x = x
        self.fn = integrand
        self.order = int(order)
        lo, hi = x[:-1], x[1:]
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        nodes, weights = _leggauss(self.order)
        s = mid[:, None] + half[:, None] * nodes[None, :]
        fv = np.asarray(integrand(s), dtype=float) \
            * (half[:, None] * weights[None, :])
        self._panel = fv.sum(axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._panel)])

    def at_nodes(self):
        return self._cum.copy()

    def __call__(self, s):
        arr = np.asarray(s, dtype=float)
        shape = arr.shape
        s1 = arr.ravel()
        nodes, weights = _leggauss(self.order)
        idx = np.clip(np.searchsorted(self.x, s1, side="right") - 1,
                      0, self.x.size - 2)
        x0 = self.x[idx]
        half = 0.5 * (s1 - x0)
        mid = 0.5 * (s1 + x0)
        sn = mid[:, None] + half[:, None] * nodes[None, :]
        fv = np.asarray(self.fn(sn), dtype=float) \
            * (half[:, None] * weights[None, :])
        vals = self._cum[idx] + fv.sum(axis=1)
        inside = (s1 >= self.x[0]) & (s1 <= self.x[-1])
        vals = np.where(inside, vals, np.nan)
        vals = vals.reshape(shape)
        return float(vals) if arr.ndim == 0 else vals


def pulse_envelope(t, tau_l, tau_d, normalize=True):
    """Asymmetric smooth envelope ``exp(-t/tau_d)*(1 + tanh(t/tau_l))``.

    With ``normalize=True`` the profile is divided by its analytic
    supremum (stationary point of the log envelope,
    ``tanh(t*/tau_l) = 1 - tau_l/tau_d``) so its peak is exactly 1 for
    any valid tau ratio; requires ``0 < tau_l < 2*tau_d`` for a bounded
    envelope.  Vectorized in ``t``."""
    t = np.asarray(t, dtype=float)

    def raw(u):
        # log-space form avoids overflow at large |u| (exp(-u/tau_d) alone
        # overflows to inf while (1 + tanh) underflows to 0 -> inf*0 = nan)
        u = np.asarray(u, dtype=float)
        return np.exp(-u / tau_d + np.log1p(np.tanh(u / tau_l)))

    if not normalize:
        return raw(t)
    if not (0.0 < tau_l < 2.0 * tau_d):
        raise ValueError("bounded envelope requires 0 < tau_l < 2*tau_d")
    x_star = float(np.arctanh(1.0 - tau_l / tau_d))
    t_star = tau_l * x_star
    peak = float(np.exp(-t_star / tau_d
                        + np.log1p(np.tanh(x_star))))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("envelope normalization failed")
    return raw(t) / peak


def drude_history(t_grid, wp2_fn, gamma_fn, E_fn, order=6):
    """Causal-kernel (two-time) polarization by factorized quadrature.

    Returns the cumulative arrays ``G = int gamma``, ``B = int wp2 e^{-G}``,
    ``A = int e^{G} E``, ``Dint = int e^{G} B E`` at the grid nodes and the
    kernel-quadrature polarization ``P = B*A - Dint`` (zero initial history:
    the grid must start at or before the field onset)."""
    t_grid = np.asarray(t_grid, dtype=float)
    Gam = Cumulative(t_grid, gamma_fn, order)

    def _b_integrand(s):
        return wp2_fn(s) * np.exp(-Gam(s))

    Bc = Cumulative(t_grid, _b_integrand, order)

    def _a_integrand(s):
        return np.exp(Gam(s)) * E_fn(s)

    Ac = Cumulative(t_grid, _a_integrand, order)

    def _d_integrand(s):
        return np.exp(Gam(s)) * Bc(s) * E_fn(s)

    Dc = Cumulative(t_grid, _d_integrand, order)
    G = Gam.at_nodes()
    B = Bc.at_nodes()
    A = Ac.at_nodes()
    Dint = Dc.at_nodes()
    return {"G": G, "B": B, "A": A, "Dint": Dint, "P": B * A - Dint,
            "gamma_cum": Gam, "B_cum": Bc, "A_cum": Ac}


def kernel_chi_literal(t, t_prime, wp2_fn, gamma_fn, n_sub=400, order=8):
    """Direct evaluation of the causal kernel ``chi(t, t')`` on its own
    local grid (composite Gauss), independent of any master grid.  For
    ``t < t_prime`` returns 0 (causality)."""
    t = float(t)
    tp = float(t_prime)
    if t < tp:
        return 0.0
    xs = np.linspace(tp, t, int(n_sub) + 1)
    Gam = Cumulative(xs, gamma_fn, order)
    # Gam starts at t_prime, so exp(-(Gam(s) - Gam(t_prime))) = exp(-Gam(s))
    integ = Cumulative(xs, lambda s: wp2_fn(s) * np.exp(-(Gam(s))), order)
    return float(integ.at_nodes()[-1])


def memory_state_rhs(wp2_fn, gamma_fn, E_fn):
    """RHS of the S/P memory-state system for ``solve_ivp``."""

    def rhs(t, y):
        s, _p = y
        e_t = float(np.asarray(E_fn(t)))
        return [e_t - float(np.asarray(gamma_fn(t))) * s,
                float(np.asarray(wp2_fn(t))) * s]

    return rhs


def solve_memory_state(t_eval, wp2_fn, gamma_fn, E_fn, y0=(0.0, 0.0),
                       rtol=1e-12, atol=1e-14):
    """Integrate the S/P memory-state system with DOP853.

    Returns ``(S, P)`` evaluated on ``t_eval`` (zero-history initial state
    by default)."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(memory_state_rhs(wp2_fn, gamma_fn, E_fn),
                    (t_eval[0], t_eval[-1]), list(y0), method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError("memory-state integration failed: %s"
                           % sol.message)
    return sol.y[0], sol.y[1]


def j_log_rhs(wp2_fn, dlog_wp2_fn, gamma_fn, E_fn):
    """RHS of the equivalent current form Jdot = [dlog(wp2) - gamma] J
    + wp2 E (smooth, strictly positive wp2 only)."""

    def rhs(t, y):
        _j, _p = y
        return [float(np.asarray(dlog_wp2_fn(t)))
                * y[0] - float(np.asarray(gamma_fn(t))) * y[0]
                + float(np.asarray(wp2_fn(t)))
                * float(np.asarray(E_fn(t))), y[0]]

    return rhs


def solve_j_log_form(t_eval, wp2_fn, dlog_wp2_fn, gamma_fn, E_fn,
                     rtol=1e-12, atol=1e-14):
    """Integrate the J/P current form; returns ``(J, P)`` on ``t_eval``."""
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(j_log_rhs(wp2_fn, dlog_wp2_fn, gamma_fn, E_fn),
                    (t_eval[0], t_eval[-1]), [0.0, 0.0], method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("J-form integration failed: %s" % sol.message)
    return sol.y[0], sol.y[1]


def stationary_ade_rhs(wp2_fn, gamma_fn, E_fn):
    """RHS of the STATIONARY ADE update Jdot = -gamma J + wp2 E.

    This omits the dlog(wp2)/dt term, so it matches the causal kernel only
    for time-independent wp2; under modulation it is a different model
    (kept for demonstrating that difference)."""

    def rhs(t, y):
        _j, _p = y
        return [-float(np.asarray(gamma_fn(t))) * y[0]
                + float(np.asarray(wp2_fn(t)))
                * float(np.asarray(E_fn(t))), y[0]]

    return rhs


def stationary_chi(omega, wp2, gamma):
    """Static Drude susceptibility for the Fourier convention
    ``E(t) = int dω/(2π) E(ω) e^{-iωt}``:
    ``chi(ω) = -wp2 / [ω (ω + i gamma)]``."""
    omega = np.asarray(omega, dtype=complex)
    return -wp2 / (omega * (omega + 1j * gamma))
