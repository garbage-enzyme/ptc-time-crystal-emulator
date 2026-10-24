"""Gate tests for the low-dimensional Floquet engine (stdlib unittest).

Run with:  python -m unittest discover -s tests -v
(PYTHONPATH must include src/; or install the package.)

Each test encodes one numerical gate. A failure means a reported quantity
from this engine is not trustworthy; it is never waived by re-running.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptc_emulator import (  # noqa: E402
    growth_and_stability,
    hill_candidates,
    hill_monodromy,
    monodromy,
    time_domain_growth,
)

W0, G = 1.0, 0.002  # omega0, gamma in units of omega_ref


class TestAnchors(unittest.TestCase):
    def test_undriven_growth_is_minus_gamma(self):
        M, T = monodromy(W0, G, 0.0, 0.97 * W0)
        g = growth_and_stability(M, T)
        self.assertLessEqual(abs(g["growth_from_modulus"] - (-G)), 1e-10)

    def test_zero_modulation_monodromy_is_expm(self):
        M, T = monodromy(W0, G, 0.0, 0.97 * W0)
        A0 = np.array([[0.0, 1.0], [-W0 * W0, -2.0 * G]])
        self.assertLessEqual(float(np.abs(M - expm(A0 * T)).max()), 1e-8)

    def test_two_growth_routes_agree(self):
        worst = 0.0
        for m in (0.0, 0.10, 0.30):
            for ratio in (0.97, 2.18):
                M, T = monodromy(W0, G, m, ratio * W0)
                g = growth_and_stability(M, T)
                worst = max(worst, g["growth_route_difference"])
        self.assertLessEqual(worst, 1e-10)

    def test_jury_counterexample_breaks_abs_trace_gt_2(self):
        # |tr M| = 1.98836... < 2 while 1+d = 1.98836... and max|mu| > 1:
        # the old |tr M|>2 test calls this stable; it is unstable.
        M, T = monodromy(W0, G, 0.30, 2.1471399172267067 * W0)
        g = growth_and_stability(M, T)
        self.assertLessEqual(g["abs_trace"], 2.0)
        self.assertTrue(g["unstable_by_modulus"])
        self.assertGreater(g["abs_trace"], g["jury_boundary_1_plus_d"])
        self.assertEqual(g["jury_status"], "IN_DOMAIN")
        self.assertEqual(g["verdict_by_jury"], "UNSTABLE")


class TestJuryDomainGuards(unittest.TestCase):
    """R1 regressions: the Jury form must never report STABLE outside its
    passive real domain; max|mu| stays authoritative everywhere."""

    def test_det_above_one_is_unstable_and_jury_withheld(self):
        # Anti-damped oscillator (gamma < 0): det = exp(-2*gamma*T) > 1.
        # Product of moduli > 1 forces max|mu| > 1; |tr| < 1 + det would
        # wrongly read STABLE if the determinant restriction were dropped.
        M, T = monodromy(1.0, -0.02, 0.0, 2.0)
        g = growth_and_stability(M, T)
        self.assertGreater(g["det"], 1.0)
        self.assertTrue(g["unstable_by_modulus"])
        self.assertGreater(g["growth_from_modulus"], 0.0)
        self.assertEqual(g["jury_status"], "DET_ABOVE_ONE")
        self.assertIsNone(g["verdict_by_jury"])
        self.assertIsNone(g["unstable_by_jury"])

    def test_complex_matrix_is_not_silently_stable(self):
        # Complex monodromy: trace 0.6j must not be discarded to 0.
        M = np.diag([1.1j, -0.5j])
        g = growth_and_stability(M, T=2.0 * np.pi / 1.5)
        self.assertAlmostEqual(g["max_modulus"], 1.1, places=12)
        self.assertTrue(g["unstable_by_modulus"])
        self.assertGreater(
            g["imaginary_part_discarded_from_trace_and_det"], 0.5)
        self.assertEqual(g["jury_status"], "NONREAL_MATRIX")
        self.assertIsNone(g["verdict_by_jury"])

    def test_nonpositive_det_withholds_jury(self):
        M2 = np.diag([2.0, -0.5])                 # det = -1 < 0, max|mu| = 2
        g = growth_and_stability(M2, T=1.0)
        self.assertTrue(g["unstable_by_modulus"])
        self.assertEqual(g["jury_status"], "NONPOSITIVE_DET")
        self.assertIsNone(g["verdict_by_jury"])

    def test_input_guards(self):
        with self.assertRaises(ValueError):
            growth_and_stability(np.eye(3), 1.0)
        with self.assertRaises(ValueError):
            growth_and_stability(np.array([[np.nan, 0.0], [0.0, 1.0]]), 1.0)
        with self.assertRaises(ValueError):
            growth_and_stability(np.eye(2), 0.0)
        with self.assertRaises(ValueError):
            growth_and_stability(np.eye(2), -1.0)


class TestHill(unittest.TestCase):
    def test_resolved_pairs_reproduce_monodromy(self):
        for (m, ratio, N) in ((0.30, 2.20, 4), (0.05, 2.18, 4), (0.0, 0.97, 2)):
            M, _ = monodromy(W0, G, m, ratio * W0)
            h = hill_monodromy(W0, G, m, ratio * W0, N)
            self.assertIsNotNone(
                h["best_pair"],
                "no pair at m=%s ratio=%s N=%s" % (m, ratio, N))
            err = float(np.abs(h["best_pair"]["M"] - M).max())
            self.assertLessEqual(err, 1e-6,
                                 "matrix err %g at m=%s ratio=%s N=%s"
                                 % (err, m, ratio, N))

    def test_screening_rejects_small_truncations(self):
        # At strong modulation and J=3 no candidate passes the screens; the
        # engine must report UNRESOLVED rather than return a spurious pair.
        h = hill_monodromy(W0, G, 0.30, 2.20, 1)
        self.assertEqual(h["n_qualified"], 0)
        self.assertEqual(h["status"], "UNRESOLVED")
        self.assertIsNone(h["best_pair"])


class TestTimeDomain(unittest.TestCase):
    def test_undriven_decay_rate(self):
        M, _ = monodromy(W0, G, 0.0, 0.97 * W0)
        g = growth_and_stability(M, T=2 * np.pi / (0.97 * W0))
        td = time_domain_growth(W0, G, 0.0, 0.97 * W0,
                                n_periods=90, discard=12)
        self.assertLessEqual(abs(td["growth_fit"] - g["growth_from_modulus"]),
                             1e-4)

    def test_tongue_growth_rate(self):
        ratio = 2.18
        M, T = monodromy(W0, G, 0.30, ratio * W0)
        g = growth_and_stability(M, T)
        td = time_domain_growth(W0, G, 0.30, ratio * W0,
                                n_periods=90, discard=12)
        self.assertLessEqual(abs(td["growth_fit"] - g["growth_from_modulus"]),
                             1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRawCandidateFields(unittest.TestCase):
    """R4 evidence contract: candidates expose their harmonic coefficients."""

    def test_coeffs_and_harmonics_present(self):
        cands, k, evals = hill_candidates(W0, G, 0.05, 2.18 * W0, 4)
        self.assertEqual(len(cands), 2 * (2 * 4 + 1))
        for c in cands:
            self.assertEqual(c["coeffs"].shape, (9,))
            self.assertTrue(np.all(np.isfinite(c["coeffs"].view(float))))
            self.assertEqual(list(c["harmonics"]), list(range(-4, 5)))
            # consistency: state0 reconstructed from coeffs + Lambda
            q0 = c["coeffs"].sum()
            qd0 = ((c["Lambda"] + 1j * (2.18 * W0) * c["harmonics"])
                   * c["coeffs"]).sum()
            self.assertAlmostEqual(abs(q0 - c["state0"][0]), 0.0, places=12)
            self.assertAlmostEqual(abs(qd0 - c["state0"][1]), 0.0, places=12)


class TestLayoutIndependentFiniteCheck(unittest.TestCase):
    """B2 regressions: valid complex matrices must be accepted regardless of
    memory layout; non-finite input stays rejected."""

    def test_fortran_contiguous_complex_accepted(self):
        M = np.asfortranarray(np.eye(2, dtype=complex) * 0.9)
        g = growth_and_stability(M, T=1.7)
        self.assertAlmostEqual(g["max_modulus"], 0.9, places=12)
        self.assertTrue(g["unstable_by_modulus"] is False)

    def test_noncontiguous_complex_accepted(self):
        base = np.zeros((3, 3), dtype=complex)
        base[0, 0] = 0.9
        base[2, 2] = 0.9
        M = base[::2, ::2]  # strided on both axes, equals 0.9*I
        self.assertFalse(M.flags["C_CONTIGUOUS"])
        g = growth_and_stability(M, T=1.7)
        self.assertAlmostEqual(g["max_modulus"], 0.9, places=12)

    def test_c_contiguous_complex_equivalent(self):
        g1 = growth_and_stability(np.eye(2, dtype=complex) * 0.9, T=1.7)
        g2 = growth_and_stability(
            np.asfortranarray(np.eye(2, dtype=complex) * 0.9), T=1.7)
        self.assertEqual(g1["max_modulus"], g2["max_modulus"])

    def test_nonfinite_still_rejected(self):
        with self.assertRaises(ValueError):
            growth_and_stability(np.array([[complex(np.nan, 0), 0],
                                           [0, 1.0]]), 1.0)

    def test_lossless_rotation_d1_documented_boundary(self):
        # det == 1 marginal boundary: bounded, zero growth, IN_DOMAIN,
        # STABLE means non-growing (documented, not strict decay).
        M = np.array([[0.0, -1.0], [1.0, 0.0]])  # pure rotation, det = 1
        g = growth_and_stability(M, T=2.0 * np.pi)
        self.assertAlmostEqual(g["max_modulus"], 1.0, places=12)
        self.assertFalse(g["unstable_by_modulus"])
        self.assertEqual(g["jury_status"], "IN_DOMAIN")
        self.assertEqual(g["verdict_by_jury"], "STABLE")
