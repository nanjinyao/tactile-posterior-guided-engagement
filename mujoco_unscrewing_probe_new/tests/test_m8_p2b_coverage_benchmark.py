import unittest

import numpy as np

from src.render_m8_search_comparison import (
    CadRadialGeometry,
    cad_state_likelihood,
    center_posterior_from_direction,
)
from src.run_m8_p2b_coverage_benchmark import (
    ACTIVE_GUARD_BUDGET_S,
    BASELINE_MAX_RADIUS_M,
    EFFECTIVE_CAPTURE_RADIUS_M,
    METHODS,
    VISUAL_BOUND_M,
    build_tasks,
    coverage_path_length_m,
)
import src.train_m8_direction_models as training


class TestM8P2bCoverageBenchmark(unittest.TestCase):
    def test_design_has_five_in_domain_radii_and_six_sector_triplets(self) -> None:
        tasks = build_tasks()
        self.assertEqual(len(tasks), 90)
        pose_ids = {task.pose_id for task in tasks}
        self.assertEqual(len(pose_ids), 30)
        self.assertEqual(len({round(task.radius_m, 12) for task in tasks}), 5)
        self.assertEqual({task.sector_index for task in tasks}, set(range(6)))
        self.assertTrue(all(task.radius_m <= VISUAL_BOUND_M for task in tasks))
        for pose_id in pose_ids:
            self.assertEqual(
                {task.method for task in tasks if task.pose_id == pose_id}, set(METHODS)
            )

    def test_coverage_and_guard_are_derived_from_geometry(self) -> None:
        self.assertAlmostEqual(
            BASELINE_MAX_RADIUS_M,
            VISUAL_BOUND_M - EFFECTIVE_CAPTURE_RADIUS_M,
        )
        nominal_coverage_time = coverage_path_length_m() / 0.00225
        self.assertGreater(nominal_coverage_time, 70.0)
        self.assertGreater(ACTIVE_GUARD_BUDGET_S, 2.0 * nominal_coverage_time + 5.8)

    def test_truncated_visual_posterior_stays_inside_acceptance_set(self) -> None:
        probability = np.full(training.DIRECTION_CLASSES, 1.0e-4)
        probability[training._direction_class(3.75)] = 1.0
        probability /= probability.sum()
        center = np.array([0.002, -0.001])
        samples = center_posterior_from_direction(
            center,
            probability,
            0.002,
            np.random.default_rng(7),
            visual_bound_m=VISUAL_BOUND_M,
        )
        self.assertLessEqual(float(np.max(np.linalg.norm(samples - center, axis=1))), VISUAL_BOUND_M)

    def test_ft_state_and_soft_cad_annulus_add_radial_information(self) -> None:
        probability = np.full(training.DIRECTION_CLASSES, 1.0e-5)
        probability[training._direction_class(183.75)] = 1.0
        probability /= probability.sum()
        state_probability = np.zeros(len(training.STATE_NAMES), dtype=float)
        state_probability[training.STATE_TO_INDEX["double_overhang"]] = 1.0
        geometry = CadRadialGeometry(
            full_outer_support_radius_m=0.003036,
            inner_opening_overlap_limit_m=0.006714,
            transition_sigma_m=0.00035,
        )
        center = np.array([0.004, 0.0])
        direction_only = center_posterior_from_direction(
            center,
            probability,
            0.002,
            np.random.default_rng(11),
            visual_bound_m=VISUAL_BOUND_M,
            candidates=65536,
        )
        fused = center_posterior_from_direction(
            center,
            probability,
            0.002,
            np.random.default_rng(11),
            visual_bound_m=VISUAL_BOUND_M,
            state_probabilities=state_probability,
            cad_geometry=geometry,
            candidates=65536,
        )
        direction_q10 = float(
            np.quantile(np.linalg.norm(direction_only - center, axis=1), 0.10)
        )
        fused_q10 = float(np.quantile(np.linalg.norm(fused - center, axis=1), 0.10))
        self.assertGreater(fused_q10, direction_q10 + 0.0015)
        self.assertLessEqual(float(np.max(np.linalg.norm(fused - center, axis=1))), VISUAL_BOUND_M)

    def test_cad_state_likelihood_is_soft_near_boundaries(self) -> None:
        geometry = CadRadialGeometry(0.003036, 0.006714, 0.00035)
        state_probability = np.zeros(len(training.STATE_NAMES), dtype=float)
        state_probability[training.STATE_TO_INDEX["double_overhang"]] = 1.0
        radii = np.array([0.0015, 0.0032, 0.0042, 0.0065, 0.0080])
        likelihood = cad_state_likelihood(radii, state_probability, geometry)
        self.assertGreater(likelihood[2], likelihood[0])
        self.assertGreater(likelihood[2], likelihood[-1])
        self.assertTrue(np.all(likelihood > 0.0))


if __name__ == "__main__":
    unittest.main()
