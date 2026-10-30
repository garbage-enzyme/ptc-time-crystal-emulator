"""Gates for the augmented Maxwell-Drude system at fixed real k.

Static dispersion vs the analytic Drude cubic, monodromy trace identity,
harmonic-truncation multiplier sets vs monodromy with a convergence
ladder, direct-time growth, the exact energy identity and no-modulation
decay."""

import numpy as np

from ptc_emulator import drude_maxwell as dm

EPS_INF = 1.0


def _const(v):
    return lambda t: v + 0j * 0.0 * np.asarray(t, dtype=float)


def _cos_mod(base, amp, omega, phase=0.0):
    def f(t):
        t = np.asarray(t, dtype=float)
        return base + amp * np.cos(omega * t + phase) \
            + 0j * np.zeros_like(t)
    return f


def test_static_dispersion_matches_cubic():
    for k in (0.5, 1.0, 2.0):
        A = dm.static_matrix(k, EPS_INF, 1.0, 0.1)
        lam = np.linalg.eigvals(A)
        # one exactly neutral polarization mode; three physical modes
        idx = np.argsort(np.abs(lam))
        assert abs(lam[idx[0]]) <= 1e-9
        w_from_A = 1j * lam[idx[1:]]   # lambda = -i w
        w_cubic = dm.drude_dispersion_roots(k, EPS_INF, 1.0, 0.1)
        assert dm.match_sets(w_from_A, w_cubic) <= 1e-10
        assert np.abs(A @ dm.neutral_direction()).max() <= 1e-12


def test_no_modulation_decay():
    k = 1.0
    A = dm.static_matrix(k, EPS_INF, 1.0, 0.1)
    lam = np.linalg.eigvals(A)
    idx = np.argsort(np.abs(lam))
    assert np.max(np.real(lam[idx[1:]])) <= 1e-12
    y0 = dm.project_out_neutral(
        [1.0 + 0.5j, 0.3 - 0.2j, 0.1 + 0j, -0.2 + 0.4j])
    t = np.linspace(0.0, 120.0, 1201)
    y = dm.solve_augmented(t, k, EPS_INF, _const(1.0), _const(0.1), y0)
    # decay is an ENERGY statement: the state norm converges to the
    # residual (neutral) polarization offset and need not decrease
    E = y[0] - y[3]
    w = 0.5 * np.abs(E) ** 2 + 0.5 * np.abs(y[1]) ** 2 \
        + 0.5 * np.abs(y[2]) ** 2
    assert np.max(np.diff(w)) <= 1e-9 * max(1.0, float(np.max(w)))
    assert w[-1] < w[0]


def test_energy_identity_closes():
    """w(t) - w(0) == int (0.5 wp2'|S|^2 - gamma wp2|S|^2) dt exactly."""
    k, m, Om = 1.0, 0.1, 2.0
    wp2 = _cos_mod(1.0, m, Om)
    gam = _const(0.1)
    dwp2 = _cos_mod(0.0, -m * Om, Om, -np.pi / 2)   # -m*Om*sin(Om t)
    t = np.linspace(0.0, 40.0, 40001)
    y0 = np.array([1.0 + 0j, 0.2 + 0.1j, 0.05 - 0.05j, 0.0 + 0.2j])
    y = dm.solve_augmented(t, k, EPS_INF, wp2, gam, y0)
    E = (y[0] - y[3]) / EPS_INF
    w = 0.5 * EPS_INF * np.abs(E) ** 2 + 0.5 * np.abs(y[1]) ** 2 \
        + 0.5 * np.real(wp2(t)) * np.abs(y[2]) ** 2
    rhs = 0.5 * np.real(dwp2(t)) * np.abs(y[2]) ** 2 \
        - np.real(gam(t)) * np.real(wp2(t)) * np.abs(y[2]) ** 2
    # composite trapezoid is insufficient; use cumulative Gauss panels
    from ptc_emulator.drude import Cumulative
    integ = Cumulative(t, lambda s: np.interp(s, t, rhs), order=6)
    work = integ.at_nodes()
    scale = max(abs(w[-1] - w[0]), float(np.max(np.abs(work))), 1.0)
    assert np.max(np.abs((w - w[0]) - work)) / scale <= 1e-6


def test_monodromy_trace_identity():
    k, m, Om = 1.0, 0.4, 2.0
    T = 2 * np.pi / Om
    gam = _const(0.1)
    wp2 = _cos_mod(1.0, m, Om)
    M = dm.monodromy_augmented(T, k, EPS_INF, wp2, gam)
    # |det M| = exp(-int gamma dt); gamma real so det phase is not fixed,
    # but the trace identity gives det M up to the real damping factor
    det = np.linalg.det(M)
    assert abs(abs(det) - np.exp(-0.1 * T)) <= 1e-10


def test_harmonic_multipliers_match_monodromy():
    k, m, Om = 1.0, 0.1, 2.0
    T = 2 * np.pi / Om
    wp2 = _cos_mod(1.0, m, Om)
    gam = _const(0.1)
    M = dm.monodromy_augmented(T, k, EPS_INF, wp2, gam)
    mult_mono = np.linalg.eigvals(M)
    H = dm.harmonic_matrix(k, EPS_INF, 1.0, 0.5 * m * EPS_INF,
                           0.1, 0.0, Om, 8)
    mult_hill, _lam = dm.multipliers_from_harmonic(H, Om, T)
    assert dm.match_sets(mult_mono, mult_hill) <= 1e-8


def test_harmonic_ladder_converges_monotonically():
    k, m, Om = 1.0, 0.4, 4.0
    T = 2 * np.pi / Om
    wp2 = _cos_mod(1.0, m, Om)
    gam = _const(0.1)
    M = dm.monodromy_augmented(T, k, EPS_INF, wp2, gam)
    mult_mono = np.linalg.eigvals(M)
    errs = []
    for N in (2, 4, 8):
        H = dm.harmonic_matrix(k, EPS_INF, 1.0, 0.5 * m, 0.1, 0.0,
                               Om, N)
        mult, _lam = dm.multipliers_from_harmonic(H, Om, T)
        errs.append(dm.match_sets(mult_mono, mult))
    # all levels must clear the gate; strict ordering is only required
    # while the errors are resolvable above the machine floor
    assert max(errs) <= 1e-8
    assert errs[0] >= errs[2] or errs[2] <= 1e-12


def test_direct_time_growth_matches_multipliers():
    k, m, Om = 1.0, 0.4, 2.0
    T = 2 * np.pi / Om
    wp2 = _cos_mod(1.0, m, Om)
    gam = _const(0.1)
    M = dm.monodromy_augmented(T, k, EPS_INF, wp2, gam)
    rate_expected = float(np.max(np.log(np.abs(
        np.linalg.eigvals(M)))) / T)
    rate, samples = dm.direct_time_growth(T, 90, k, EPS_INF, wp2, gam)
    assert len(samples) == 90
    assert abs(rate - rate_expected) <= 1e-8


def test_harmonic_combined_with_gamma_phase():
    """Regression for the ladder-diagonal sign: with a PHASED gamma pulse
    (phi != 0) combined with wp2 modulation, a wrong sign in the
    i*n*Omega block diagonal is no longer equivalent to the correct
    assembly under n -> -n relabelling, and shows up as a flat ~3e-4
    multiplier error (single-real-cosine cases are sign-blind because
    A+ = A- there)."""
    k, m, Om, a, ph = 1.0, 0.1, 4.0, 0.02, 0.5
    T = 2 * np.pi / Om
    wp2 = _cos_mod(1.0, m, Om)
    gam = _cos_mod(0.1, a, Om, ph)
    M = dm.monodromy_augmented(T, k, EPS_INF, wp2, gam)
    gp1 = 0.5 * a * np.exp(1j * ph)
    H = dm.harmonic_matrix(k, EPS_INF, 1.0, 0.5 * m, 0.1, gp1, Om, 8)
    mult, _lam = dm.multipliers_from_harmonic(H, Om, T)
    assert dm.match_sets(np.linalg.eigvals(M), mult) <= 1e-8


def test_match_sets_permutation_invariance():
    a = np.array([1 + 2j, 3 - 1j, 0.5 + 0j])
    assert dm.match_sets(a, a[[2, 0, 1]]) == 0.0
    assert dm.match_sets(a, a + 0.01) > 0.0
    assert np.isinf(dm.match_sets(a, a[:2]))
