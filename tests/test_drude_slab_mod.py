"""Gates for the modulated Drude slab pilot.

Static regression (m = 0 reduces EXACTLY to the accepted static slab),
the steady-state energy identity O - 1 = 2W (modulation work minus
dissipation at unit incident flux), harmonic-truncation convergence,
and the independent time-domain route agreeing within the declared
discretization tolerance."""

import numpy as np

from ptc_emulator import drude_slab as dsl
from ptc_emulator import drude_slab_mod as dsm

W0, OM_A, OM_B, D, GAMMA = 2.0, 0.7, 1.9, 1.0, 0.1


def test_static_regression_exact():
    sol = dsm.modulated_slab_channels(W0, OM_A, 0.0, D, GAMMA, 4)
    r_st, t_st = dsl.slab_admittance(np.array([W0]), D, 1.0, GAMMA)
    i0 = int(np.argmin(np.abs(sol["n"])))
    assert abs(sol["r"][i0] - r_st[0]) <= 1e-12
    assert abs(sol["t"][i0] - t_st[0]) <= 1e-12
    others = max(abs(v) for j, v in enumerate(sol["r"]) if j != i0)
    assert others == 0.0


def test_energy_identity_closes():
    for m, om in ((0.05, OM_A), (0.2, OM_A), (0.05, OM_B), (0.2, OM_B)):
        en = dsm.modulated_energy_terms(W0, om, m, D, GAMMA, 8)
        assert abs(en["closure"]) <= 1e-9
        assert en["Diss"] > 0.0          # lossy slab always dissipates


def test_harmonic_ladder_converges():
    for m, om in ((0.2, OM_A), (0.2, OM_B)):
        s4 = dsm.modulated_slab_channels(W0, om, m, D, GAMMA, 4)
        s8 = dsm.modulated_slab_channels(W0, om, m, D, GAMMA, 8)
        # compare on the shared channel range n in [-4, 4]
        n_shared = 9
        dr = np.max(np.abs(np.abs(s4["r"][:n_shared])
                           - np.abs(s8["r"][4:4 + n_shared])))
        dt = np.max(np.abs(np.abs(s4["t"][:n_shared])
                           - np.abs(s8["t"][4:4 + n_shared])))
        assert max(dr, dt) <= 1e-8


def test_signed_negative_frequency_channels():
    """Negative-frequency sidebands are genuine output channels: at
    finite m they are nonzero and the flux bookkeeping (which includes
    them) is consistent with the energy identity."""
    sol = dsm.modulated_slab_channels(W0, OM_A, 0.2, D, GAMMA, 8)
    neg = sol["w"] < 0.0
    assert np.any(np.abs(sol["r"][neg]) > 1e-6)
    assert np.any(np.abs(sol["t"][neg]) > 1e-8)


def test_time_domain_static_regression():
    td = dsm.time_domain_slab(W0, OM_A, 0.0, D, GAMMA, t_end=360.0)
    r_st, t_st = dsl.slab_admittance(np.array([W0]), D, 1.0, GAMMA)
    i0 = int(np.argmin(np.abs(td["n"])))
    assert abs(abs(td["r"][i0]) - abs(r_st[0])) <= 5e-3
    assert abs(abs(td["t"][i0]) - abs(t_st[0])) <= 5e-3


def test_time_domain_matches_harmonic():
    td = dsm.time_domain_slab(W0, OM_A, 0.05, D, GAMMA, t_end=360.0)
    sol = dsm.modulated_slab_channels(W0, OM_A, 0.05, D, GAMMA, 8)
    band = np.abs(sol["w"]) <= 4.0
    dr = np.max(np.abs(np.abs(sol["r"][band]) - np.abs(td["r"][band])))
    dt = np.max(np.abs(np.abs(sol["t"][band]) - np.abs(td["t"][band])))
    assert max(dr, dt) <= 5e-3
