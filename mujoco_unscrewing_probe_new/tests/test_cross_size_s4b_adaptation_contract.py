from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.sample_cross_size_s4b_adaptation import (
    DEVELOPMENT_V2_REPORT,
    INTENDED_MODEL_OUTPUT,
    ROLE_PHASES_DEG,
    S4_DATA_ROOT,
    SECTORS,
    SEEDS,
    build_dataset_contract,
    build_role_tasks,
    deployment_geometry,
    deployment_rings,
    run,
)


ROOT = Path(__file__).resolve().parents[1]


class CrossSizeS4bDatasetContractTests(unittest.TestCase):
    def _spec(self, size: str) -> dict:
        return json.loads(
            (ROOT / "configs" / f"allen_{size}_source.json").read_text(
                encoding="utf-8"
            )
        )

    def test_sixth_ring_is_exactly_generic_visual_bound(self) -> None:
        expected_mm = {
            "m10": [1.6340, 2.9050, 3.9865, 4.8156, 5.6447, 6.0000],
            "m12": [1.5644, 2.7812, 4.1550, 5.7382, 7.3215, 8.0000],
        }
        for size in ("m10", "m12"):
            spec = self._spec(size)
            geometry = deployment_geometry(spec)
            rings = deployment_rings(spec)
            self.assertEqual(len(rings), 6)
            np.testing.assert_allclose(
                [1000.0 * float(ring["radius_m"]) for ring in rings],
                expected_mm[size],
                atol=5.0e-5,
            )
            outer = rings[-1]
            self.assertEqual(outer["ring_index"], 5)
            self.assertEqual(outer["search_state"], "double_overhang")
            self.assertEqual(
                outer["coverage_role"], "generic_full_loop_visual_boundary"
            )
            self.assertEqual(outer["normalized_visual_radius"], 1.0)
            self.assertAlmostEqual(
                outer["radius_m"], geometry["visual_bound_m"], places=15
            )
            self.assertLess(
                outer["radius_m"], geometry["inner_overlap_radius_m"]
            )

    def test_complete_c6_roles_are_phase_and_pose_disjoint(self) -> None:
        expected_counts = {"train": 72, "validation": 36, "calibration": 36}
        expected_orbits = {"train": 12, "validation": 6, "calibration": 6}
        for size in ("m10", "m12"):
            tasks = build_role_tasks(self._spec(size), size, SEEDS[size])
            orbit_sets = {}
            pose_sets = {}
            for role, role_tasks in tasks.items():
                self.assertEqual(len(role_tasks), expected_counts[role])
                self.assertEqual(
                    {task.base_angle_deg for task in role_tasks},
                    set(ROLE_PHASES_DEG[role]),
                )
                orbit_sets[role] = {task.symmetry_orbit_id for task in role_tasks}
                self.assertEqual(len(orbit_sets[role]), expected_orbits[role])
                pose_sets[role] = {
                    (round(task.x_m, 12), round(task.y_m, 12))
                    for task in role_tasks
                }
                by_orbit = {}
                for task in role_tasks:
                    by_orbit.setdefault(task.symmetry_orbit_id, []).append(task)
                    self.assertTrue(task.trial_id.startswith("s4b"))
                    self.assertTrue(task.nuisance_id.startswith("s4b_"))
                    self.assertEqual(task.yaw_rad, 0.0)
                    self.assertIsNotNone(task.sensor_noise_seed)
                    self.assertGreaterEqual(task.sensor_delay_s, 0.002)
                    self.assertLessEqual(task.sensor_delay_s, 0.008)
                    self.assertAlmostEqual(
                        math.hypot(task.x_m, task.y_m), task.radius_m, places=12
                    )
                for orbit in by_orbit.values():
                    self.assertEqual(len(orbit), len(SECTORS))
                    self.assertEqual(
                        {task.rotation_index for task in orbit}, set(SECTORS)
                    )
                    self.assertEqual(len({task.nuisance_id for task in orbit}), 1)
                    self.assertEqual(
                        len({task.sensor_noise_seed for task in orbit}), 1
                    )
            roles = tuple(ROLE_PHASES_DEG)
            for index, left in enumerate(roles):
                for right in roles[index + 1 :]:
                    self.assertFalse(orbit_sets[left] & orbit_sets[right])
                    self.assertFalse(pose_sets[left] & pose_sets[right])

    def test_contract_records_development_use_and_forbids_data_reuse(self) -> None:
        contract = build_dataset_contract(self._spec("m12"), "m12", SEEDS["m12"])
        self.assertEqual(contract["contract_version"], "cross_size_s4b_v1")
        self.assertEqual(
            contract["role_counts"]["train"],
            {
                "orbits": 12,
                "scans": 72,
                "state_scans": {
                    "supported_search": 24,
                    "double_overhang": 48,
                },
            },
        )
        for role in ("validation", "calibration"):
            self.assertEqual(
                contract["role_counts"][role],
                {
                    "orbits": 6,
                    "scans": 36,
                    "state_scans": {
                        "supported_search": 12,
                        "double_overhang": 24,
                    },
                },
            )
        trigger = contract["development_trigger"]
        self.assertEqual(trigger["finding"], "aggregate outer-boundary observability gap")
        self.assertEqual(trigger["old_task_pose_records_reused"], 0)
        self.assertEqual(trigger["old_nuisance_records_reused"], 0)
        self.assertEqual(trigger["old_raw_fingerprints_reused"], 0)
        self.assertEqual(trigger["development_trial_fields_loaded_by_generator"], [])
        self.assertEqual(
            trigger["aggregate_report"]["path"],
            str(DEVELOPMENT_V2_REPORT.relative_to(ROOT)),
        )
        independence = contract["independence_contract"]
        self.assertTrue(independence["tasks_generated_ab_initio"])
        self.assertFalse(independence["reads_existing_S4_dataset"])
        self.assertFalse(independence["writes_existing_S4_dataset"])
        self.assertEqual(
            independence["intended_model_output"],
            str(INTENDED_MODEL_OUTPUT.relative_to(ROOT)),
        )
        self.assertTrue(
            all(
                not intersection
                for intersection in contract["pairwise_orbit_intersections"].values()
            )
        )
        self.assertTrue(
            all(
                not intersection
                for intersection in contract[
                    "pairwise_physical_pose_intersections"
                ].values()
            )
        )

    def test_contract_only_run_never_samples_or_touches_s4_root(self) -> None:
        old_s4_files = {
            path.relative_to(S4_DATA_ROOT): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in S4_DATA_ROOT.rglob("*")
            if path.is_file()
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "s4b"
            result = run(
                argparse.Namespace(
                    output=str(output),
                    sizes=["m10", "m12"],
                    sample=False,
                    workers=1,
                )
            )
            self.assertEqual(result, 0)
            root_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(root_manifest["status"], "contract_only")
            self.assertFalse(root_manifest["sampling_explicitly_requested"])
            self.assertFalse(root_manifest["existing_S4_dataset_written"])
            for size in ("m10", "m12"):
                size_manifest = json.loads(
                    (output / size / "dataset_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(size_manifest["status"], "contract_only")
                self.assertEqual(size_manifest["trial_files"], [])
                self.assertFalse((output / size / "trials").exists())
        new_s4_files = {
            path.relative_to(S4_DATA_ROOT): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in S4_DATA_ROOT.rglob("*")
            if path.is_file()
        }
        self.assertEqual(old_s4_files, new_s4_files)

    def test_s4_output_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses to overwrite"):
            run(
                argparse.Namespace(
                    output=str(S4_DATA_ROOT),
                    sizes=["m10"],
                    sample=False,
                    workers=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
