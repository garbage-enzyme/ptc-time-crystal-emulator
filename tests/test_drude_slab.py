"""Gates for the static dispersive Drude slab baseline.

Two independent derivations agree; matched, zero-thickness,
semi-infinite, lossless, DC and conjugacy limits; volumetric absorption
vs flux balance; passivity bounds; conditioning."""

import numpy as np

from ptc_emulator import drude_slab as ds

WP2, GAMMA = 1.0, 0.1


def _grid():
    return np.linspace(-8.0, 8.0, 321)


def test_matched_medium_transmits():
    om = _grid()
    r, t = ds.slab_admittance(om, 1.0, 0.0, GAMMA)          # wp2 = 0
    assert np.max(np.abs(r)) <= 1e-14
    assert np.max(np.abs(np.abs(t) - 1.0)) <= 1e-14


def test_two_methods_agree():
    om = _grid()
    for d in (0.5, 1.0, 5.0):
        r_a, t_a = ds.slab_admittance(om, d, WP2, GAMMA)
        r_b, t_b, ap, am, cond = ds.slab_boundary_solve(
            om, d, WP2, GAMMA)
        assert np.max(np.abs(r_a - r_b)) <= 1e-12
        assert np.max(np.abs(t_a - t_b)) <= 1e-12


def test_zero_thickness_limit():
    # a conductive sheet PHYSICALLY absorbs ~ d/omega^2 at low omega,
    # so the geometric zero-thickness limit is gated on the propagating
    # band (same band as the lossless gate)
    om = np.linspace(1.5, 8.0, 200)
    r, t = ds.slab_admittance(om, 1e-10, WP2, GAMMA)
    R, T = np.abs(r) ** 2, np.abs(t) ** 2
    assert np.max(R) <= 1e-12
    assert np.max(np.abs(T - 1.0)) <= 1e-10


def test_semi_infinite_matches_fresnel():
    om = np.linspace(0.05, 8.0, 200)
    d = 50.0
    r, t = ds.slab_admittance(om, d, WP2, GAMMA)
    k = ds.slab_k(om, WP2, GAMMA)
    mask = np.imag(k) * d >= 30.0
    assert np.any(mask)
    assert np.max(np.abs(np.abs(r[mask]) ** 2
                         - ds.fresnel_interface_reflection(
                             om[mask], WP2, GAMMA))) <= 1e-9
    assert np.max(np.abs(t[mask]) ** 2) <= 1e-12


def test_lossless_unit_sum():
    om = np.linspace(1.5, 8.0, 200)      # propagating band w > wp2
    for d in (0.5, 1.0, 5.0):
        r, t = ds.slab_admittance(om, d, WP2, 0.0)
        assert np.max(np.abs(np.abs(r) ** 2 + np.abs(t) ** 2 - 1.0)) \
            <= 1e-12


def test_volumetric_absorption_matches_flux_balance():
    om = np.linspace(0.05, 8.0, 200)
    for d in (0.5, 1.0, 5.0):
        r, t, ap, am, cond = ds.slab_boundary_solve(om, d, WP2, GAMMA)
        k = ds.slab_k(om, WP2, GAMMA)
        a_vol = ds.absorption_volumetric(om, ap, am, k, d, WP2, GAMMA)
        bal = 1.0 - np.abs(r) ** 2 - np.abs(t) ** 2
        assert np.max(np.abs(a_vol - bal)) <= 1e-8


def test_passivity_bounds():
    om = _grid()
    for d in (0.5, 1.0, 5.0):
        r, t = ds.slab_admittance(om, d, WP2, GAMMA)
        R, T = np.abs(r) ** 2, np.abs(t) ** 2
        A = 1.0 - R - T
        assert np.min(R) >= -1e-12 and np.max(R) <= 1.0 + 1e-12
        assert np.min(T) >= -1e-12 and np.max(T) <= 1.0 + 1e-12
        assert np.min(A) >= -1e-12 and np.max(A) <= 1.0 + 1e-12
        assert np.max(np.abs(R + T + A - 1.0)) <= 1e-12


def test_negative_frequency_conjugacy():
    om = _grid()
    r, t = ds.slab_admittance(om, 1.0, WP2, GAMMA)
    # grid is symmetric: pair omega and -omega by index
    r_c = r[::-1].conj()
    t_c = t[::-1].conj()
    assert np.max(np.abs(r - r_c)) <= 1e-13
    assert np.max(np.abs(t - t_c)) <= 1e-13


def test_dc_channel_resistive_sheet():
    """The DC limit is the resistive-sheet formula r = -k/(2+k),
    t = 2/(2+k) with k = d wp2/gamma (the slab is electrically thin as
    w -> 0: k d ~ sqrt(w) -> 0 while Y_s ~ w^{-1/2})."""
    for d in (0.1, 0.5, 1.0, 5.0):
        om = np.array([0.0])
        r, t = ds.slab_admittance(om, d, WP2, GAMMA)
        kappa = d * WP2 / GAMMA
        assert abs(r[0] - (-kappa / (2.0 + kappa))) <= 1e-12
        assert abs(t[0] - 2.0 / (2.0 + kappa)) <= 1e-12
        R, T = abs(r[0]) ** 2, abs(t[0]) ** 2
        assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0
        assert R + T <= 1.0 + 1e-12            # absorption positive
    # thick/good-conductor sub-case: kappa -> inf recovers r -> -1
    r5, _ = ds.slab_admittance(np.array([0.0]), 5.0, WP2, GAMMA)
    assert abs(r5[0] + 5.0 * WP2 / GAMMA / (2.0 + 5.0 * WP2 / GAMMA)) \
        <= 1e-12
    # matched medium at DC: kappa = 0 -> r = 0, t = 1
    r0, t0 = ds.slab_admittance(np.array([0.0]), 1.0, 0.0, GAMMA)
    assert abs(r0[0]) <= 1e-15 and abs(t0[0] - 1.0) <= 1e-15


def test_conditioning_bounded():
    om = np.linspace(0.05, 8.0, 200)
    for d in (0.5, 1.0, 5.0, 50.0):
        _r, _t, _ap, _am, cond = ds.slab_boundary_solve(om, d, WP2, GAMMA)
        assert np.max(cond) < 1e8
