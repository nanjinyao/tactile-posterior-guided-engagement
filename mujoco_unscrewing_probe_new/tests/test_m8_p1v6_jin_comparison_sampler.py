import argparse
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from src.sample_m8_conical_pilot import (
    DEFAULT_P1V6_JIN_COMPARISON_OUT,
    build_final_untouched_direction_audit_tasks,
    build_jin_comparison_direction_audit_tasks,
)
from src.sample_m8_jin_comparison_direction_audit import run


ROOT = Path(__file__).resolve().parents[1]


class TestM8P1v6JinComparisonSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(
            (ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8")
        )

    def test_frozen_design_has_180_paired_trials(self) -> None:
        tasks = build_jin_comparison_direction_audit_tasks(self.spec)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 60)
        self.assertEqual(len({task.symmetry_orbit_id for task in tasks}), 30)
        self.assertEqual(
            sorted({task.radius_m for task in tasks}),
            [0.00132, 0.00212, 0.00295, 0.00388, 0.00472],
        )
        self.assertEqual({task.base_angle_deg for task in tasks}, {17.0, 53.0})
        self.assertEqual({task.rotation_index for task in tasks}, set(range(6)))
        self.assertLess(max(task.radius_m for task in tasks), 0.004895)
        self.assertEqual(
            {task.search_state for task in tasks if task.radial_level <= 2},
            {"supported_search"},
        )
        self.assertEqual(
            {task.search_state for task in tasks if task.radial_level >= 3},
            {"double_overhang"},
        )

    def test_sobol_nuisance_is_seeded_and_shared_within_each_pairing_block(self) -> None:
        tasks = build_jin_comparison_direction_audit_tasks(self.spec)
        samples = qmc.Sobol(d=4, scramble=True, seed=20260732).random_base2(4)[:15]
        for radial_level in range(5):
            for repeat_index in range(3):
                block_index = 3 * radial_level + repeat_index
                block = [
                    task
                    for task in tasks
                    if task.radial_level == radial_level
                    and task.nuisance_id == f"p1v6_sobol_{radial_level:02d}_{repeat_index:02d}"
                ]
                self.assertEqual(len(block), 12)
                self.assertEqual({task.nuisance_seed for task in block}, {60000 + block_index})
                sample = samples[block_index]
                expected = (
                    0.55 + 0.40 * float(sample[0]),
                    0.040 + 0.030 * float(sample[1]),
                    2.5 + float(sample[2]),
                    math.radians(-1.5 + 3.0 * float(sample[3])),
                )
                actual = (
                    block[0].friction_slide,
                    block[0].contact_timeconst_s,
                    block[0].target_down_force_n,
                    block[0].yaw_rad,
                )
                np.testing.assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)
                self.assertEqual(len({task.symmetry_orbit_id for task in block}), 2)
                for orbit_id in {task.symmetry_orbit_id for task in block}:
                    orbit = [task for task in block if task.symmetry_orbit_id == orbit_id]
                    self.assertEqual({task.rotation_index for task in orbit}, set(range(6)))

    def test_p1v6_is_disjoint_from_and_does_not_replace_p1v5(self) -> None:
        p1v5 = build_final_untouched_direction_audit_tasks(self.spec)
        p1v6 = build_jin_comparison_direction_audit_tasks(self.spec)
        self.assertEqual(len(p1v5), 180)
        self.assertEqual({task.base_angle_deg for task in p1v5}, {7.0, 41.0})
        self.assertFalse({task.radius_m for task in p1v5} & {task.radius_m for task in p1v6})
        self.assertFalse({task.base_angle_deg for task in p1v5} & {task.base_angle_deg for task in p1v6})
        self.assertTrue(all(task.trial_id.startswith(("supported_search_jin_compare", "double_overhang_jin_compare")) for task in p1v6))
        self.assertTrue(all(task.nuisance_id.startswith("p1v6_sobol_") for task in p1v6))

    @patch("src.sample_m8_jin_comparison_direction_audit.run_sampler", return_value=0)
    def test_wrapper_freezes_both_estimators_out_of_training(self, mocked_sampler) -> None:
        args = argparse.Namespace(
            output=str(DEFAULT_P1V6_JIN_COMPARISON_OUT),
            workers=1,
            phase_bins=360,
            limit=None,
            resume=True,
        )
        self.assertEqual(run(args), 0)
        mocked_sampler.assert_called_once_with(args)
        self.assertIs(args.task_builder, build_jin_comparison_direction_audit_tasks)
        self.assertEqual(args.training_use, "forbidden_for_both")
        self.assertEqual(args.stage_name, "P1v6_M8_Jin_style_vs_improved_direction_audit")
        self.assertIn("forbidden", args.claim_boundary.lower())
        self.assertIn("either estimator", args.claim_boundary)


if __name__ == "__main__":
    unittest.main()
