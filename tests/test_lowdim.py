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
        self.assertEqual(g["verdict_by_jury"], "UNSTABLE")


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
