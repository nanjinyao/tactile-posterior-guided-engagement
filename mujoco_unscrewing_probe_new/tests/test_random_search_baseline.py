from __future__ import annotations

import unittest

import numpy as np

from src.render_m8_search_comparison import random_waypoint_path


class TestRandomSearchBaseline(unittest.TestCase):
    def test_seeded_path_is_deterministic_bounded_and_budget_matched(self) -> None:
        center = np.array([0.003, -0.001])
        radius = 0.004
        budget = 0.120
        arc_a, path_a = random_waypoint_path(center, radius, budget, 1234)
        arc_b, path_b = random_waypoint_path(center, radius, budget, 1234)
        np.testing.assert_allclose(arc_a, arc_b)
        np.testing.assert_allclose(path_a, path_b)
        self.assertAlmostEqual(float(arc_a[-1]), budget)
        self.assertTrue(np.all(np.diff(arc_a) > 0.0))
        self.assertLessEqual(float(np.max(np.linalg.norm(path_a - center, axis=1))), radius + 1.0e-12)

    def test_new_seed_changes_the_stochastic_path(self) -> None:
        _, path_a = random_waypoint_path(np.zeros(2), 0.004, 0.020, 10)
        _, path_b = random_waypoint_path(np.zeros(2), 0.004, 0.020, 11)
        self.assertFalse(np.allclose(path_a[: min(len(path_a), len(path_b))], path_b[: min(len(path_a), len(path_b))]))


if __name__ == "__main__":
    unittest.main()
