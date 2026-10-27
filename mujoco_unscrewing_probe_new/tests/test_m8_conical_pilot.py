import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.sample_m8_conical_pilot import (
    PHASE_COLUMNS,
    RAW_COLUMNS,
    _aligned_to_reference,
    _load_trial,
    _rotate_xy,
    audit_contact_continuity,
    audit_dataset_contact_quarantine,
    build_collision_tasks,
    build_independent_direction_audit_tasks,
    build_observability_tasks,
    build_sector_anchor_tasks,
    build_symmetry_tasks,
    build_targeted_direction_training_tasks,
    build_untouched_direction_audit_tasks,
    derive_direction_supervision_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_raw_scan(samples: int = 12) -> np.ndarray:
    raw = np.zeros((samples, len(RAW_COLUMNS)), dtype=np.float32)
    raw[:, RAW_COLUMNS.index("time_s")] = (
        np.arange(1, samples + 1, dtype=float) * 0.002
    )
    raw[:, RAW_COLUMNS.index("phase_rad")] = np.linspace(
        0.0, 2.0 * math.pi, samples, endpoint=False
    )
    raw[:, RAW_COLUMNS.index("Fz_N")] = 3.0
    raw[:, RAW_COLUMNS.index("tip_z_m")] = 0.0003
    raw[:, RAW_COLUMNS.index("contact_count")] = 2.0
    return raw


def _write_legacy_trial(path: Path, raw: np.ndarray, trial_id: str) -> None:
    metadata = {
        "trial_id": trial_id,
        "status": "ok",
        "target_down_force_n": 3.0,
        "controller_dt_s": 0.002,
        "full_scan_complete": True,
        "early_capture": False,
        "capture_during_scan": False,
        "full_fingerprint_valid": True,
        "direction_supervision_valid": True,
        "capture_classifier": {"z_threshold_m": -0.0002},
    }
    sequence = np.zeros((len(PHASE_COLUMNS), 12), dtype=np.float32)
    np.savez_compressed(
        path,
        raw_scan=raw,
        raw_columns=np.asarray(RAW_COLUMNS),
        phase_sequence=sequence,
        phase_columns=np.asarray(PHASE_COLUMNS),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


class TestM8ConicalPilot(unittest.TestCase):
    def test_symmetry_plan_has_twelve_base_poses_and_six_rotations(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_symmetry_tasks(spec)
        self.assertEqual(len(tasks), 72)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 12)
        for base_pose_id in {task.base_pose_id for task in tasks}:
            rotations = {task.rotation_index for task in tasks if task.base_pose_id == base_pose_id}
            self.assertEqual(rotations, set(range(6)))

    def test_signed_distances_match_state_partition(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_symmetry_tasks(spec)
        for task in tasks:
            if task.search_state == "supported_search":
                self.assertGreater(task.outer_signed_distance_m, 0.0)
                self.assertLess(task.inner_signed_distance_m, 0.0)
            elif task.search_state == "double_overhang":
                self.assertLess(task.outer_signed_distance_m, 0.0)
                self.assertLess(task.inner_signed_distance_m, 0.0)
            elif task.search_state == "outer_partial":
                self.assertLess(task.outer_signed_distance_m, 0.0)
                self.assertGreater(task.inner_signed_distance_m, 0.0)

    def test_observability_plan_is_stratified_and_bounded(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_observability_tasks(spec)
        self.assertEqual(len(tasks), 360)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 120)
        for base_pose_id in {task.base_pose_id for task in tasks}:
            self.assertEqual(sum(task.base_pose_id == base_pose_id for task in tasks), 3)
        self.assertTrue(all(0.55 <= task.friction_slide <= 0.95 for task in tasks))
        self.assertTrue(all(0.040 <= task.contact_timeconst_s <= 0.070 for task in tasks))
        self.assertTrue(all(2.5 <= task.target_down_force_n <= 3.5 for task in tasks))
        self.assertTrue(all(abs(math.degrees(task.yaw_rad)) <= 1.5 for task in tasks))

    def test_collision_followup_only_targets_reported_hard_groups(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_collision_tasks(spec)
        self.assertEqual(len(tasks), 90)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 18)
        self.assertTrue(all(sum(other.base_pose_id == task.base_pose_id for other in tasks) == 5 for task in tasks))

    def test_sector_anchor_plan_pairs_nuisance_across_six_physical_sectors(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_sector_anchor_tasks(spec)
        self.assertEqual(len(tasks), 324)
        orbit_ids = {task.symmetry_orbit_id for task in tasks}
        self.assertEqual(len(orbit_ids), 54)
        for orbit_id in orbit_ids:
            orbit = [task for task in tasks if task.symmetry_orbit_id == orbit_id]
            self.assertEqual({task.rotation_index for task in orbit}, set(range(6)))
            self.assertEqual(len({task.friction_slide for task in orbit}), 1)
            self.assertEqual(len({task.target_down_force_n for task in orbit}), 1)

    def test_independent_audit_is_frozen_in_domain_and_nuisance_paired(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_independent_direction_audit_tasks(spec)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 60)
        self.assertEqual({task.base_angle_deg for task in tasks}, {15.0, 45.0})
        self.assertTrue(all(task.radius_m < 0.004895 for task in tasks))
        self.assertNotIn("outer_partial", {task.search_state for task in tasks})
        full_support = 0.5 * float(spec["screw_head_diameter_m"]) - float(
            spec["key_af_m"]
        ) / math.sqrt(3.0)
        radii = sorted({task.radius_m for task in tasks})
        self.assertEqual(sum(radius <= full_support for radius in radii), 3)
        self.assertEqual(sum(radius > full_support for radius in radii), 2)
        for radial_level in range(5):
            for repeat_index in range(3):
                paired = [
                    task
                    for task in tasks
                    if task.radial_level == radial_level
                    and task.nuisance_id.endswith(f"_{repeat_index:02d}")
                ]
                self.assertEqual(len(paired), 12)
                self.assertEqual(len({task.friction_slide for task in paired}), 1)
                self.assertEqual(len({task.target_down_force_n for task in paired}), 1)

    def test_targeted_training_covers_two_regimes_with_off_grid_angles(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_targeted_direction_training_tasks(spec)
        self.assertEqual(len(tasks), 288)
        self.assertEqual(len({task.base_pose_id for task in tasks}), 144)
        self.assertEqual(len({task.base_angle_deg for task in tasks}), 4)
        self.assertFalse({15.0, 45.0} & {task.base_angle_deg for task in tasks})
        self.assertEqual(
            {task.search_state for task in tasks}, {"supported_search", "double_overhang"}
        )
        for base_pose_id in {task.base_pose_id for task in tasks}:
            self.assertEqual(sum(task.base_pose_id == base_pose_id for task in tasks), 2)

    def test_second_audit_is_disjoint_from_targeted_angles_and_radii(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        training = build_targeted_direction_training_tasks(spec)
        audit = build_untouched_direction_audit_tasks(spec)
        self.assertEqual(len(audit), 180)
        self.assertFalse({task.radius_m for task in training} & {task.radius_m for task in audit})
        self.assertFalse(
            {task.base_angle_deg for task in training}
            & {task.base_angle_deg for task in audit}
        )

    def test_sector_anchor_plan_pairs_nuisance_across_six_physical_sectors(self) -> None:
        spec = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
        tasks = build_sector_anchor_tasks(spec)
        self.assertEqual(len(tasks), 324)
        self.assertEqual(len({task.symmetry_orbit_id for task in tasks}), 54)
        for orbit_id in {task.symmetry_orbit_id for task in tasks}:
            orbit = [task for task in tasks if task.symmetry_orbit_id == orbit_id]
            self.assertEqual({task.rotation_index for task in orbit}, set(range(6)))
            self.assertEqual(len({task.friction_slide for task in orbit}), 1)
            self.assertEqual(len({task.target_down_force_n for task in orbit}), 1)

    def test_sixty_degree_rotation_is_invertible(self) -> None:
        vector = np.array([0.004, -0.002])
        rotated = _rotate_xy(vector, math.pi / 3.0)
        restored = _rotate_xy(rotated, -math.pi / 3.0)
        np.testing.assert_allclose(restored, vector, atol=1.0e-12, rtol=0.0)

    def test_phase_and_vector_symmetry_alignment(self) -> None:
        columns = ("Fx_N", "Fy_N", "Tx_Nm", "Ty_Nm", "tip_x_m", "tip_y_m", "Fz_N")
        bins = 360
        phase = np.linspace(0.0, 2.0 * math.pi, bins, endpoint=False)
        reference = np.vstack(
            (
                np.cos(phase),
                np.sin(phase),
                0.1 * np.cos(2.0 * phase),
                0.1 * np.sin(2.0 * phase),
                0.003 + 0.0001 * np.cos(phase),
                -0.002 + 0.0001 * np.sin(phase),
                3.0 + 0.2 * np.cos(phase),
            )
        )
        angle = math.pi / 3.0
        transformed = reference.copy()
        for x_name, y_name in (("Fx_N", "Fy_N"), ("Tx_Nm", "Ty_Nm"), ("tip_x_m", "tip_y_m")):
            ix, iy = columns.index(x_name), columns.index(y_name)
            transformed[[ix, iy]] = _rotate_xy(transformed[[ix, iy]], angle)
        rotated_trace = np.roll(transformed, -bins // 6, axis=1)
        aligned = _aligned_to_reference(rotated_trace, columns, 1)
        np.testing.assert_allclose(aligned, reference, atol=1.0e-12, rtol=0.0)

    def test_contact_audit_keeps_brief_loss_but_records_it(self) -> None:
        raw = _synthetic_raw_scan()
        raw[4:6, RAW_COLUMNS.index("Fz_N")] = 0.0
        raw[4:6, RAW_COLUMNS.index("contact_count")] = 0.0
        audit = audit_contact_continuity(
            raw,
            RAW_COLUMNS,
            target_down_force_n=3.0,
            downward_progress_z_threshold_m=-0.0008,
            sample_dt_s=0.002,
        )
        ground_truth = audit["mujoco_offline_ground_truth"]
        observable = audit["sensor_observable_force"]
        self.assertEqual(len(ground_truth["contact_loss_runs"]), 1)
        self.assertAlmostEqual(
            ground_truth["contact_loss_runs"][0]["duration_s"], 0.004
        )
        self.assertFalse(audit["has_sustained_contact_loss"])
        self.assertEqual(len(observable["low_force_runs"]), 1)
        self.assertFalse(audit["has_sustained_low_force"])

    def test_sustained_loss_is_quarantined_with_separate_reasons(self) -> None:
        raw = _synthetic_raw_scan()
        raw[4:7, RAW_COLUMNS.index("Fz_N")] = 0.0
        raw[4:7, RAW_COLUMNS.index("contact_count")] = 0.0
        derived = derive_direction_supervision_metadata(
            raw,
            RAW_COLUMNS,
            status="ok",
            target_down_force_n=3.0,
            downward_progress_z_threshold_m=-0.0008,
            full_scan_complete=True,
            early_capture=False,
            capture_during_scan=False,
            sample_dt_s=0.002,
        )
        self.assertFalse(derived["direction_supervision_valid"])
        self.assertIn(
            "sustained_contact_loss_offline_ground_truth",
            derived["direction_quarantine_reasons"],
        )
        self.assertNotIn(
            "sustained_low_force_sensor_observable",
            derived["direction_quarantine_reasons"],
        )
        self.assertIn(
            "sustained_low_force_sensor_observable",
            derived["direction_diagnostic_flags"],
        )
        self.assertEqual(
            derived["capture_or_incomplete_reasons"], []
        )

    def test_low_force_with_continuous_contact_is_diagnostic_not_quarantine(
        self,
    ) -> None:
        raw = _synthetic_raw_scan()
        raw[3:8, RAW_COLUMNS.index("Fz_N")] = 0.5
        derived = derive_direction_supervision_metadata(
            raw,
            RAW_COLUMNS,
            status="ok",
            target_down_force_n=3.0,
            downward_progress_z_threshold_m=-0.0008,
            full_scan_complete=True,
            early_capture=False,
            capture_during_scan=False,
            sample_dt_s=0.002,
        )
        self.assertTrue(derived["direction_supervision_valid"])
        self.assertEqual(derived["direction_quarantine_reasons"], [])
        self.assertEqual(
            derived["direction_diagnostic_flags"],
            ["sustained_low_force_sensor_observable"],
        )

    def test_low_force_without_contact_ground_truth_is_quarantined(
        self,
    ) -> None:
        raw = _synthetic_raw_scan()
        raw[3:8, RAW_COLUMNS.index("Fz_N")] = 0.5
        contact_index = RAW_COLUMNS.index("contact_count")
        legacy_columns = RAW_COLUMNS[:contact_index]
        legacy_raw = raw[:, :contact_index]
        derived = derive_direction_supervision_metadata(
            legacy_raw,
            legacy_columns,
            status="ok",
            target_down_force_n=3.0,
            downward_progress_z_threshold_m=-0.0008,
            full_scan_complete=True,
            early_capture=False,
            capture_during_scan=False,
            sample_dt_s=0.002,
        )
        self.assertFalse(derived["direction_supervision_valid"])
        self.assertIn(
            "sustained_low_force_without_contact_ground_truth",
            derived["direction_quarantine_reasons"],
        )

    def test_within_scan_downward_progress_cannot_recover_into_training(self) -> None:
        raw = _synthetic_raw_scan()
        raw[3:7, RAW_COLUMNS.index("tip_z_m")] = -0.0009
        derived = derive_direction_supervision_metadata(
            raw,
            RAW_COLUMNS,
            status="ok",
            target_down_force_n=3.0,
            downward_progress_z_threshold_m=-0.0008,
            full_scan_complete=True,
            early_capture=False,
            # The endpoint classifier intentionally misses the recovered drop.
            capture_during_scan=False,
            sample_dt_s=0.002,
        )
        self.assertFalse(derived["direction_supervision_valid"])
        self.assertTrue(
            derived["contact_continuity_audit"][
                "has_sustained_downward_progress"
            ]
        )
        self.assertIn(
            "within_scan_sustained_downward_progress",
            derived["capture_or_incomplete_reasons"],
        )

    def test_legacy_loader_revalidates_stale_positive_label_from_raw(self) -> None:
        raw = _synthetic_raw_scan()
        raw[2:6, RAW_COLUMNS.index("Fz_N")] = 0.0
        raw[2:6, RAW_COLUMNS.index("contact_count")] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            _write_legacy_trial(path, raw, "legacy")
            metadata, _, _ = _load_trial(path)
        self.assertFalse(metadata["direction_supervision_valid"])
        self.assertTrue(
            metadata["direction_supervision_audit"][
                "revalidated_from_raw_on_load"
            ]
        )
        self.assertTrue(
            metadata["direction_supervision_audit"][
                "recorded_values_before_revalidation"
            ]["direction_supervision_valid"]
        )
        self.assertIn(
            "sustained_contact_loss_offline_ground_truth",
            metadata["direction_quarantine_reasons"],
        )

    def test_dataset_audit_is_read_only_and_reason_counted(self) -> None:
        clean = _synthetic_raw_scan()
        lost = _synthetic_raw_scan()
        lost[4:8, RAW_COLUMNS.index("Fz_N")] = 0.0
        lost[4:8, RAW_COLUMNS.index("contact_count")] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials"
            trials.mkdir()
            _write_legacy_trial(trials / "clean.npz", clean, "clean")
            _write_legacy_trial(trials / "lost.npz", lost, "lost")
            report_path = root / "audit.json"
            before = {
                path.name: path.read_bytes()
                for path in trials.glob("*.npz")
            }
            report = audit_dataset_contact_quarantine(
                root, output_path=report_path
            )
            after = {
                path.name: path.read_bytes()
                for path in trials.glob("*.npz")
            }
            written = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(before, after)
        self.assertFalse(report["source_trials_rewritten"])
        self.assertEqual(report["trial_count"], 2)
        self.assertEqual(report["admitted_count"], 1)
        self.assertEqual(report["quarantined_count"], 1)
        self.assertEqual(
            report["reason_counts"][
                "sustained_contact_loss_offline_ground_truth"
            ],
            1,
        )
        self.assertEqual(written["reason_counts"], report["reason_counts"])


if __name__ == "__main__":
    unittest.main()
