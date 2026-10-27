from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.finetune_cross_size_s4_state_direction_heads import (
    configure_trainable_heads,
    probability_metrics,
)
from src.sample_cross_size_s4_adaptation import (
    ROLE_PHASES_DEG,
    SECTORS,
    build_dataset_contract,
    build_role_tasks,
    deployment_geometry,
    deployment_rings,
    run,
)
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]


class CrossSizeS4DatasetContractTests(unittest.TestCase):
    def _spec(self, size: str) -> dict:
        return json.loads(
            (ROOT / "configs" / f"allen_{size}_source.json").read_text(
                encoding="utf-8"
            )
        )

    def test_deployment_rings_match_frozen_annulus(self) -> None:
        expected_mm = {
            "m10": [1.6340, 2.9050, 3.9865, 4.8156, 5.6447],
            "m12": [1.5644, 2.7812, 4.1550, 5.7382, 7.3215],
        }
        for size in ("m10", "m12"):
            spec = self._spec(size)
            geometry = deployment_geometry(spec)
            rings = deployment_rings(spec)
            actual = [1000.0 * float(item["radius_m"]) for item in rings]
            np.testing.assert_allclose(actual, expected_mm[size], atol=5.0e-5)
            self.assertEqual(
                [item["search_state"] for item in rings],
                [
                    "supported_search",
                    "supported_search",
                    "double_overhang",
                    "double_overhang",
                    "double_overhang",
                ],
            )
            self.assertTrue(
                all(
                    geometry["radius_min_m"]
                    <= item["radius_m"]
                    <= geometry["visual_bound_m"]
                    for item in rings
                )
            )

    def test_complete_c6_roles_are_disjoint_and_sensor_matched(self) -> None:
        expected_counts = {"train": 60, "validation": 30, "calibration": 30}
        for size, seed in (("m10", 20261031), ("m12", 20261032)):
            tasks = build_role_tasks(self._spec(size), size, seed)
            orbit_sets = {}
            for role, role_tasks in tasks.items():
                self.assertEqual(len(role_tasks), expected_counts[role])
                orbit_sets[role] = {task.symmetry_orbit_id for task in role_tasks}
                self.assertEqual(
                    {task.base_angle_deg for task in role_tasks},
                    set(ROLE_PHASES_DEG[role]),
                )
                by_orbit = {}
                for task in role_tasks:
                    by_orbit.setdefault(task.symmetry_orbit_id, []).append(task)
                    self.assertIsNotNone(task.sensor_noise_seed)
                    self.assertGreaterEqual(task.sensor_delay_s, 0.002)
                    self.assertLessEqual(task.sensor_delay_s, 0.008)
                    self.assertEqual(task.yaw_rad, 0.0)
                    self.assertAlmostEqual(
                        math.hypot(task.x_m, task.y_m), task.radius_m, places=12
                    )
                for orbit in by_orbit.values():
                    self.assertEqual(
                        {task.rotation_index for task in orbit}, set(SECTORS)
                    )
                    self.assertEqual(len(orbit), 6)
                    self.assertEqual(len({task.sensor_noise_seed for task in orbit}), 1)
                    self.assertEqual(len({task.sensor_delay_s for task in orbit}), 1)
            self.assertFalse(orbit_sets["train"] & orbit_sets["validation"])
            self.assertFalse(orbit_sets["train"] & orbit_sets["calibration"])
            self.assertFalse(orbit_sets["validation"] & orbit_sets["calibration"])

    def test_contract_only_run_never_launches_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "s4"
            result = run(
                argparse.Namespace(
                    output=str(output), sizes=["m10"], sample=False, workers=1
                )
            )
            self.assertEqual(result, 0)
            root_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            size_manifest = json.loads(
                (output / "m10" / "dataset_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            contract = json.loads(
                (output / "m10" / "dataset_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(root_manifest["status"], "contract_only")
            self.assertFalse(root_manifest["sampling_explicitly_requested"])
            self.assertEqual(size_manifest["status"], "contract_only")
            self.assertEqual(size_manifest["trial_files"], [])
            self.assertFalse((output / "m10" / "trials").exists())
            self.assertEqual(contract["role_counts"]["train"]["scans"], 60)
            self.assertIn(
                "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3",
                contract["excluded_artifacts"],
            )

    def test_contract_records_per_state_counts_and_zero_leakage(self) -> None:
        contract = build_dataset_contract(self._spec("m12"), "m12", 20261032)
        self.assertEqual(
            contract["role_counts"]["train"]["state_scans"],
            {"supported_search": 24, "double_overhang": 36},
        )
        self.assertTrue(
            all(
                not intersection
                for intersection in contract["pairwise_orbit_intersections"].values()
            )
        )
        self.assertEqual(
            contract["role_semantics"]["validation"],
            "early stopping and model selection only",
        )


class CrossSizeS4HeadMetricTests(unittest.TestCase):
    def test_only_small_state_and_direction_heads_are_trainable(self) -> None:
        model = training.CircularDirectionCNN()
        names = configure_trainable_heads(model)
        self.assertEqual(
            names,
            (
                "state_head.weight",
                "state_head.bias",
                "direction_head.weight",
                "direction_head.bias",
            ),
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            6579,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith(("encoder.", "shared."))
            )
        )

    def test_control_direction_metric_exposes_multimodal_mean_failure(self) -> None:
        states = np.asarray([0, 1], dtype=np.int64)
        directions = np.asarray([0, 8], dtype=np.int64)
        angles = np.asarray(
            training._class_angle_deg(directions), dtype=np.float32
        )
        state_probability = np.zeros((2, len(training.STATE_NAMES)))
        state_probability[np.arange(2), states] = 1.0
        direction_probability = np.zeros((2, training.DIRECTION_CLASSES))
        # MAP is correct for sample 0, but the aggregate opposite mode carries
        # more vector mass and flips the circular control direction.
        direction_probability[0, 0] = 0.40
        direction_probability[0, 23] = 0.25
        direction_probability[0, 24] = 0.35
        direction_probability[1, 8] = 1.0
        examples = {
            "state": states,
            "direction": directions,
            "angle_deg": angles,
        }
        metrics = probability_metrics(
            state_probability,
            direction_probability,
            examples,
            member_direction_probabilities=np.stack(
                (direction_probability, direction_probability)
            ),
        )
        self.assertEqual(metrics["top1_error_gt90_count"], 0)
        self.assertEqual(metrics["control_direction_error_gt90_count"], 1)
        self.assertEqual(
            metrics["per_state"]["supported_search"][
                "control_direction_error_gt90_count"
            ],
            1,
        )
        self.assertGreater(
            metrics["worst_state_control_direction_error_p90_deg"], 90.0
        )
        self.assertEqual(metrics["state_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
