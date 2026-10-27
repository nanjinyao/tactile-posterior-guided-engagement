from __future__ import annotations

import unittest

import numpy as np

from src.posterior_search_plan import plan_posterior_guided_search


class TestPosteriorSearchPlan(unittest.TestCase):
    def test_transport_uses_conservative_projected_distance(self) -> None:
        rng = np.random.default_rng(4)
        samples = rng.normal(loc=(0.0030, 0.0), scale=(0.00025, 0.00020), size=(2000, 2))
        plan = plan_posterior_guided_search(
            np.zeros(2),
            samples,
            np.array([1.0, 0.0]),
            0.00025,
        )
        self.assertGreater(plan.transport_distance_m, 0.0)
        self.assertLessEqual(plan.transport_distance_m, plan.transport_distance_cap_m)
        self.assertTrue(np.allclose(plan.transport_target_xy[1], 0.0))
        self.assertEqual(plan.transport_mode, "continuous_risk_bounded")

    def test_transport_changes_continuously_with_direction_quality(self) -> None:
        rng = np.random.default_rng(9)
        samples = rng.normal(loc=(0.0030, 0.0), scale=(0.00045, 0.00025), size=(4000, 2))
        low = plan_posterior_guided_search(
            np.zeros(2), samples, np.array([1.0, 0.0]), 0.00025,
            direction_quality=0.20,
        )
        high = plan_posterior_guided_search(
            np.zeros(2), samples, np.array([1.0, 0.0]), 0.00025,
            direction_quality=0.90,
        )
        self.assertGreater(high.transport_distance_m, low.transport_distance_m)
        self.assertGreater(high.transport_quantile, low.transport_quantile)

    def test_ambiguous_direction_can_collapse_transport_to_zero(self) -> None:
        samples = np.array([[-0.001, 0.0], [0.001, 0.0], [-0.0005, 0.0002], [0.0005, -0.0002]])
        plan = plan_posterior_guided_search(
            np.zeros(2),
            samples,
            np.array([1.0, 0.0]),
            0.00025,
        )
        self.assertEqual(plan.transport_distance_m, 0.0)

    def test_pitch_respects_capture_coverage_bound(self) -> None:
        samples = np.array([[0.001, 0.0], [0.0012, 0.0001], [0.0009, -0.0001]])
        capture_radius = 0.00025
        plan = plan_posterior_guided_search(
            np.zeros(2),
            samples,
            np.array([1.0, 0.0]),
            capture_radius,
        )
        self.assertLessEqual(plan.spiral_pitch_m_per_rev, 2.0 * capture_radius)

    def test_max_transport_cap_is_respected(self) -> None:
        samples = np.array([[0.004, 0.0], [0.0042, 0.0001], [0.0038, -0.0001]])
        plan = plan_posterior_guided_search(
            np.zeros(2),
            samples,
            np.array([1.0, 0.0]),
            0.00025,
            max_transport_m=0.0008,
        )
        self.assertLessEqual(plan.transport_distance_m, 0.0008)


if __name__ == "__main__":
    unittest.main()
