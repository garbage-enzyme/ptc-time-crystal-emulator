"""Gates for the causal time-varying Drude memory-state module.

Every check that backs a reported quantity: memory-state ODE vs
independent causal-kernel quadrature, literal kernel vs factorized
quadrature, analytic stationary response, J-form equivalence, the
stationary-ADE model distinction, causality and residual-DC semantics.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ptc_emulator import drude

W = 2.0 * np.pi * 4.0  # probe window length (well past transients)
OMEGA = 2.0            # probe angular frequency for CW-style checks


def _burst(t, t0=0.0, width=20.0, omega=OMEGA):
    """Smooth C1 probe: sin^2 envelope times cos(omega t), zero before t0."""
    t = np.asarray(t, dtype=float)
    tau = t - t0
    env = np.where((tau > 0.0) & (tau < width),
                   np.sin(np.pi * tau / width) ** 2, 0.0)
    return env * np.cos(omega * tau)


def _plasma_only(t, d1=0.4, tau_l=1.0, tau_d=10.0):
    f = drude.pulse_envelope(t, tau_l, tau_d)
    return 1.0 + d1 * f


def _gamma_const(t):
    return 0.1 + 0.0 * np.asarray(t, dtype=float)


def _gamma_pulse(t, g0=0.1, dg=0.041, tau_l=1.0, tau_d=10.0):
    f = drude.pulse_envelope(t, tau_l, tau_d)
    return g0 + dg * f


def _grid(t0=0.0, t1=120.0, n=12001):
    return np.linspace(t0, t1, n)


def test_stationary_cw_matches_analytic_chi():
    """After the transient e^{-gamma t} dies out, the memory-state
    response to cos(omega t) must equal Re[chi(omega) e^{-i omega t}]
    with chi = -wp2/[omega(omega + i gamma)]."""
    wp2, gamma, omega = 1.0, 0.1, OMEGA
    t = _grid(0.0, 400.0, 8001)
    E = np.cos(omega * t)
    S, P = drude.solve_memory_state(t, lambda s: wp2 + 0 * s,
                                    lambda s: gamma + 0 * s,
                                    lambda s: np.cos(omega * s))
    chi = drude.stationary_chi(omega, wp2, gamma)
    # E(t) = Re[e^{-i omega t}]; steady P(t) = Re[chi e^{-i omega t}]
    P_theory = np.real(chi * np.exp(-1j * omega * t))
    tail = t > 150.0  # >> 1/gamma
    err = np.max(np.abs(P[tail] - P_theory[tail]))
    assert err <= 1e-6


def test_kernel_quadrature_matches_memory_state_plasma_only():
    wp2_fn = _plasma_only
    gam_fn = _gamma_const
    E_fn = _burst
    t = _grid()
    hist = drude.drude_history(t, wp2_fn, gam_fn, E_fn, order=6)
    S, P = drude.solve_memory_state(t, wp2_fn, gam_fn, E_fn)
    assert np.max(np.abs(P - hist["P"])) <= 1e-6


def test_kernel_quadrature_matches_memory_state_loss_only():
    wp2_fn = lambda s: 1.0 + 0.0 * np.asarray(s, dtype=float)
    gam_fn = _gamma_pulse
    E_fn = _burst
    t = _grid()
    hist = drude.drude_history(t, wp2_fn, gam_fn, E_fn, order=6)
    S, P = drude.solve_memory_state(t, wp2_fn, gam_fn, E_fn)
    assert np.max(np.abs(P - hist["P"])) <= 1e-6


def test_kernel_quadrature_matches_memory_state_combined():
    wp2_fn = _plasma_only
    gam_fn = _gamma_pulse
    E_fn = _burst
    t = _grid()
    hist = drude.drude_history(t, wp2_fn, gam_fn, E_fn, order=6)
    S, P = drude.solve_memory_state(t, wp2_fn, gam_fn, E_fn)
    assert np.max(np.abs(P - hist["P"])) <= 1e-6


def test_literal_kernel_agrees_with_factorized_quadrature():
    wp2_fn, gam_fn = _plasma_only, _gamma_pulse
    t = _grid(0.0, 60.0, 6001)
    hist = drude.drude_history(t, wp2_fn, gam_fn, _burst, order=6)
    Gam, Bc = hist["gamma_cum"], hist["B_cum"]
    rng = np.random.default_rng(20261030)
    for _ in range(12):
        tp = float(rng.uniform(0.0, 50.0))
        tt = float(rng.uniform(tp, 60.0))
        chi_fac = np.exp(Gam(tp)) * (Bc(tt) - Bc(tp))
        chi_lit = drude.kernel_chi_literal(tt, tp, wp2_fn, gam_fn)
        assert abs(chi_fac - chi_lit) <= 1e-10 * max(1.0, abs(chi_lit))


def test_j_log_form_equivalent_for_smooth_wp2():
    t = _grid(0.0, 60.0, 6001)
    E_fn = _burst
    J, P_j = drude.solve_j_log_form(t, _plasma_only, _dlog_wp2,
                                    _gamma_const, E_fn)
    S, P = drude.solve_memory_state(t, _plasma_only, _gamma_const, E_fn)
    assert np.max(np.abs(P_j - P)) <= 1e-8


def _dlog_wp2(s):
    eps = 1e-6
    wp = _plasma_only
    return (wp(s + eps) - wp(s - eps)) / (2.0 * eps * wp(s))


def test_stationary_ade_diverges_under_modulation():
    """The stationary ADE (no dlog term) must NOT match the causal kernel
    when wp2 varies: this is the model-distinction gate."""
    wp2_fn, gam_fn, E_fn = _plasma_only, _gamma_const, _burst
    t = _grid()
    sol = solve_ivp(drude.stationary_ade_rhs(wp2_fn, gam_fn, E_fn),
                    (t[0], t[-1]), [0.0, 0.0], method="DOP853",
                    t_eval=t, rtol=1e-12, atol=1e-14)
    hist = drude.drude_history(t, wp2_fn, gam_fn, E_fn, order=6)
    assert np.max(np.abs(sol.y[1] - hist["P"])) > 1e-3


def test_causality_zero_field_before_onset():
    t = np.linspace(-20.0, 40.0, 6001)
    pre = t < 0.0
    S, P = drude.solve_memory_state(t, _plasma_only, _gamma_pulse,
                                    lambda s: _burst(s))
    assert np.max(np.abs(P[pre])) == 0.0


def test_zero_field_gives_zero_states():
    t = _grid()
    S, P = drude.solve_memory_state(t, _plasma_only, _gamma_pulse,
                                    lambda s: 0.0 * np.asarray(s))
    assert np.max(np.abs(S)) == 0.0 and np.max(np.abs(P)) == 0.0


def test_residual_polarization_is_bounded_dc_not_growth():
    """After the pulse and probe end, J -> 0 and P settles to a finite
    offset; dlog|P|/dt must not be positive in the tail."""
    wp2_fn = lambda s: 1.0 + 0.4 * drude.pulse_envelope(s, 1.0, 10.0)
    t = np.linspace(0.0, 400.0, 8001)
    E_fn = lambda s: _burst(s, width=20.0)
    S, P = drude.solve_memory_state(t, wp2_fn, _gamma_const, E_fn)
    tail = t > 200.0
    assert np.max(np.abs(S[tail])) <= 1e-8          # J = wp2*S -> 0
    assert np.all(np.isfinite(P))
    d = np.diff(np.log(np.abs(P[tail]) + 1e-300))
    assert np.max(d) <= 1e-6                        # no exponential growth
    assert np.abs(P[-1]) > 1e-12                    # finite residual offset


def test_envelope_peak_is_one():
    for tau_l, tau_d in ((0.1, 0.1), (1.0, 10.0), (3.0, 100.0)):
        # exact supremum at the analytic stationary point
        t_star = tau_l * float(np.arctanh(1.0 - tau_l / tau_d))
        assert abs(drude.pulse_envelope(t_star, tau_l, tau_d) - 1.0) \
            <= 1e-12
        # on any grid the profile never exceeds its normalized supremum
        probe = np.linspace(-200.0, 800.0, 200001)
        f = drude.pulse_envelope(probe, tau_l, tau_d)
        assert np.max(f) <= 1.0 + 1e-12
        assert np.min(f) >= 0.0
        assert np.max(f) >= 1.0 - 1e-5  # grid resolves the peak
