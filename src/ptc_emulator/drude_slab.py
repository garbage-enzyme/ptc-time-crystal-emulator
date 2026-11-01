"""Static dispersive Drude slab scattering at normal incidence.

Frequency-domain baseline for the optional finite-slab branch, derived
from Maxwell under the exp(-i w t) convention accepted in the project:
outside the slab (background eps_b = 1 in normalized vacuum, c0 = eps0
= 1) and inside a Drude medium with

    eps(w) = eps_inf - wp2 / [w (w + i gamma)],   Im chi > 0 for w > 0,

k(w) = w sqrt(eps(w)) on the principal branch (Im k >= 0 for w > 0,
decaying into the slab), k(-w) = -conj(k(w)). The admittance Y = k/w is
the REMOVABLE form of sqrt(eps): Y_s(w) = sqrt(eps(w)) directly, which
stays finite where k -> 0 and diverges analytically at w = 0. The DC
channel (w -> 0) is the RESISTIVE-SHEET limit — the slab is always
electrically thin (k d ~ sqrt(w) -> 0) while Y_s ~ w^{-1/2} diverges,
competing through kappa = d wp2/gamma — with complete flux bookkeeping

    r(0) = -kappa/(2+kappa),  t(0) = 2/(2+kappa),
    E_inside(0) = 2/(2+kappa) (constant),  A(0) = 4 kappa/(2+kappa)^2,

of which r -> -1, t -> 0, A -> 0 is only the kappa -> infinity
perfect-conductor sub-case (taken exactly by the lossless plasma), and
kappa = 0 the matched no-sheet case. Tangential E and B are continuous
at both interfaces with B = Y E for a forward wave.

Two independent derivations are provided:

* :func:`slab_admittance` — input-admittance formulation (r from the
  slab input admittance backed by the background; internal amplitudes
  from the z = 0 continuity);
* :func:`slab_boundary_solve` — direct linear solve for the four
  amplitudes (r, a+, a-, t) from the four boundary equations (linear
  solves, no explicit inverses), returning condition numbers.

Flux bookkeeping: R = |r|^2, T = |t|^2 (same medium on both sides),
A = 1 - R - T, and an independent volumetric absorption from the
analytic integral of the time-averaged dissipation density
(w/2) Im(chi) |E(z)|^2 over the slab. The interference term of
|E(z)|^2 carries a pure phase e^{2 i Re(k) z} (no decay).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MODULATION_NOTE",
    "absorption_volumetric",
    "dc_limit",
    "dc_sheet",
    "drude_chi",
    "drude_epsilon",
    "fresnel_interface_reflection",
    "slab_admittance",
    "slab_boundary_solve",
    "slab_k",
    "slab_admittance_y",
]

MODULATION_NOTE = (
    "static dispersive slab baseline; a slab scattering result is NOT an "
    "infinite-medium fixed-k bandgap claim")


def drude_chi(omega, wp2, gamma):
    """chi(w) = -wp2/[w(w + i gamma)] (exp(-i w t); Im chi > 0 for w>0)."""
    omega = np.asarray(omega, dtype=complex)
    return -wp2 / (omega * (omega + 1j * gamma))


def drude_epsilon(omega, wp2, gamma, eps_inf=1.0):
    return eps_inf + drude_chi(omega, wp2, gamma)


def slab_k(omega, wp2, gamma, eps_inf=1.0):
    """k(w) = w sqrt(eps(w)) on the principal branch; Im k >= 0 for
    w > 0 (Im eps > 0 strictly there), k(-w) = -conj(k(w)); k(0) = 0."""
    om = np.asarray(omega, dtype=complex)
    out = om * np.sqrt(drude_epsilon(om, wp2, gamma, eps_inf))
    return np.where(np.asarray(omega) == 0.0, 0.0 + 0.0j, out)


def slab_admittance_y(omega, wp2, gamma, eps_inf=1.0):
    """Slab admittance Y_s = sqrt(eps(w)) (the removable form of k/w);
    diverges analytically at w = 0 (returned as complex inf there)."""
    om = np.asarray(omega, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sqrt(drude_epsilon(om, wp2, gamma, eps_inf))
    return y


def dc_limit(d, wp2, gamma):
    """Analytic DC scattering amplitudes (CONTRACT: eps_inf = eps_b = 1;
    general backgrounds are not covered by this sheet formula). The
    resistive-sheet limit r(0) = -kappa/(2+kappa), t(0) = 2/(2+kappa)
    with kappa = d wp2 / gamma. Degeneracy precedence is by KAPPA: an
    exactly vanishing sheet (d = 0 or wp2 = 0, any gamma) is the
    matched/no-sheet case (0, 1) BEFORE the lossless-plasma branch;
    d wp2 > 0 with gamma = 0 takes the kappa -> infinity
    perfect-conductor sub-case (-1, 0)."""
    if d == 0.0 or wp2 == 0.0:
        return 0.0 + 0.0j, 1.0 + 0.0j
    if gamma == 0.0:
        return -1.0 + 0.0j, 0.0 + 0.0j
    kappa = d * wp2 / gamma
    return -kappa / (2.0 + kappa) + 0.0j, 2.0 / (2.0 + kappa) + 0.0j


def dc_sheet(d, wp2, gamma):
    """COMPLETE analytic DC channel of the resistive-sheet limit
    (CONTRACT: eps_inf = eps_b = 1): (r, t, E_inside, A) with
    E_inside(0) = 2/(2+kappa) constant across the electrically-thin slab
    and the DC dissipation fraction A(0) = 4 kappa/(2+kappa)^2 (equal to
    1 - R - T of the sheet scattering). Degeneracy precedence is by
    KAPPA: d = 0 or wp2 = 0 (any gamma) gives the matched/no-sheet
    (0, 1, 1, 0) BEFORE the lossless branch; d wp2 > 0 with gamma = 0
    (lossless plasma, kappa -> inf) gives (-1, 0, 0, 0)."""
    if d == 0.0 or wp2 == 0.0:
        return (0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0)
    if gamma == 0.0:
        return (-1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0)
    kappa = d * wp2 / gamma
    r = -kappa / (2.0 + kappa)
    t = 2.0 / (2.0 + kappa)
    e_in = 2.0 / (2.0 + kappa)
    a = 4.0 * kappa / (2.0 + kappa) ** 2
    return (r + 0.0j, t + 0.0j, e_in + 0.0j, a)


def slab_admittance(omega, d, wp2, gamma, eps_b=1.0, eps_inf=1.0):
    """Method A: input-admittance formulation, in the STABLE exponential
    form. Returns (r, t) complex arrays for a unit incident forward wave
    from z < 0. For q d >> 1 (q = Im k) the formulation underflows
    gracefully to the half-space limit Y_in -> Y_s instead of
    cancelling catastrophically in a sin/cos arrangement."""
    om = np.asarray(omega, dtype=float)
    Y_s = slab_admittance_y(om, wp2, gamma, eps_inf)
    Y_b = np.sqrt(eps_b + 0j) * np.ones_like(Y_s)
    k = slab_k(om, wp2, gamma, eps_inf)
    delta = k * d
    beta = Y_b / Y_s
    e2 = np.exp(2j * delta)          # decays like e^{-2 q d}
    with np.errstate(divide="ignore", invalid="ignore"):
        Y_in = Y_s * ((1.0 + beta) - (1.0 - beta) * e2) \
            / ((1.0 + beta) + (1.0 - beta) * e2)
        r = (Y_b - Y_in) / (Y_b + Y_in)
        # internal forward amplitude from the z = 0 continuity and the
        # stable load relation  t = 2 a+ e^{i kd} / (1 + beta)
        a_p = 0.5 * ((1.0 + r) + beta * (1.0 - r))
        t = 2.0 * a_p * np.exp(1j * delta) / (1.0 + beta)
    zero = (om == 0.0)
    if np.any(zero):
        r0, t0 = dc_limit(d, wp2, gamma)
        r = np.where(zero, r0, r)
        t = np.where(zero, t0, t)
    return r, t


def slab_boundary_solve(omega, d, wp2, gamma, eps_b=1.0, eps_inf=1.0):
    """Method B: four boundary equations in SCALED unknowns
    (r, x = a+, y = a- e^{-i kd}, t) so that only the DECAYING factor
    e^{i kd} appears (no catastrophic cancellation for q d >> 1):

        1 + r = x + y e^{i kd}                 (E at z = 0)
        Y_b (1 - r) = Y_s (x - y e^{i kd})     (B at z = 0)
        x e^{i kd} + y = t                     (E at z = d)
        Y_s (x e^{i kd} - y) = Y_b t           (B at z = d)

    Solved by linear solves (no explicit inverses). Returns
    (r, t, a_plus, a_minus, cond) with a_minus = y e^{i kd}. The DC
    point takes the analytic sheet limit; the a+/- values there are
    CANONICAL THIN-LIMIT REPRESENTATIVES (a+ = a- = E_inside/2 — the
    split is undetermined in the electrically-thin limit), NOT solved
    field values; use dc_sheet for the DC channel semantics."""
    om = np.asarray(omega, dtype=float)
    Y_s = slab_admittance_y(om, wp2, gamma, eps_inf)
    Y_b = np.sqrt(eps_b + 0j) * np.ones_like(Y_s)
    k = slab_k(om, wp2, gamma, eps_inf)
    r = np.empty(om.shape, dtype=complex)
    t = np.empty(om.shape, dtype=complex)
    a_p = np.empty(om.shape, dtype=complex)
    a_m = np.empty(om.shape, dtype=complex)
    cond = np.empty(om.shape, dtype=float)
    for i in range(om.size):
        if om[i] == 0.0:
            r0, t0, e_in, _a0 = dc_sheet(d, wp2, gamma)
            r[i], t[i] = r0, t0
            a_p[i] = a_m[i] = 0.5 * e_in   # canonical thin-limit halves
            cond[i] = 1.0
            continue
        e_d = np.exp(1j * k[i] * d)
        A = np.array([
            [-1.0, 1.0, e_d, 0.0],
            [-Y_b[i], -Y_s[i], Y_s[i] * e_d, 0.0],
            [0.0, e_d, 1.0, -1.0],
            [0.0, Y_s[i] * e_d, -Y_s[i], -Y_b[i]]],
            dtype=complex)
        b = np.array([1.0, -Y_b[i], 0.0, 0.0], dtype=complex)
        x = np.linalg.solve(A, b)
        r[i], a_p[i], y, t[i] = x
        a_m[i] = y * e_d
        cond[i] = float(np.linalg.cond(A))
    return r, t, a_p, a_m, cond


def absorption_volumetric(omega, a_p, a_m, k, d, wp2, gamma,
                          eps_b=1.0, eps_inf=1.0):
    """Analytic integral of (w/2) Im(chi) |E(z)|^2 dz over the slab,
    divided by the incident flux Re(Y_b)/2.

    |a+ e^{i k z} + a- e^{-i k z}|^2 with k = p + i q integrates to
    |a+|^2 (1 - e^{-2qd})/(2q) + |a-|^2 (e^{2qd} - 1)/(2q)
    + 2 Re[ a+ conj(a-) (e^{2 i p d} - 1)/(2 i p) ]  (pure-phase term)."""
    om = np.asarray(omega, dtype=float)
    chi_im = np.imag(drude_chi(np.where(om == 0.0, np.nan, om),
                               wp2, gamma))
    # at w = 0 exactly, w Im(chi) -> wp2/gamma and the internal field is
    # the constant sheet value: the integral reduces to the analytic DC
    # dissipation fraction A(0) = 4 kappa/(2+kappa)^2
    dc_a = np.array([dc_sheet(d, wp2, gamma)[3]
                     if w == 0.0 else 0.0 for w in np.atleast_1d(om)])
    p = np.real(k)
    q = np.imag(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        def _eint(sgn):
            # int_0^d e^{sgn*2qz} dz = (e^{sgn 2 q d} - 1)/(sgn 2 q),
            # with the q -> 0 limit d (lossless propagating band, q = 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(q) < 1e-14, d,
                                (np.exp(sgn * 2.0 * q * d) - 1.0)
                                / (sgn * 2.0 * q))
        term1 = (np.abs(a_p) ** 2) * _eint(-1.0)
        term2 = (np.abs(a_m) ** 2) * _eint(+1.0)
        c = a_p * np.conj(a_m)
        # p = 0 exactly (lossless evanescent band): e^{2ipz} = 1 and the
        # cross-term integral is the limit 2 Re(c) * d
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(p) < 1e-14, d,
                             (np.exp(2j * p * d) - 1.0) / (2j * p))
        term3 = 2.0 * np.real(c * ratio)
        vol = chi_im * om * 0.5 * (term1 + term2 + term3)
    flux = 0.5 * float(np.real(np.sqrt(eps_b + 0j)))
    vol = np.where(om == 0.0, dc_a * flux, np.real(vol))
    return vol / flux


def fresnel_interface_reflection(omega, wp2, gamma, eps_b=1.0,
                                 eps_inf=1.0):
    """Single-interface power reflection |(Y_b - Y_s)/(Y_b + Y_s)|^2."""
    om = np.asarray(omega, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y_s = slab_admittance_y(om, wp2, gamma, eps_inf)
        Y_b = np.sqrt(eps_b + 0j) * np.ones_like(Y_s)
        gam = (Y_b - Y_s) / (Y_b + Y_s)
    return np.where(om == 0.0, 1.0, np.abs(gam) ** 2)
