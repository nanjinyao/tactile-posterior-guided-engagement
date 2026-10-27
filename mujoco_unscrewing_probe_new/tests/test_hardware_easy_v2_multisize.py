from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import unittest

import numpy as np

from src.evaluate_m8_hardware_easy_v2_robustness import _evaluate
from src.evaluate_hardware_easy_v2_multisize import _radial_finite_sample_screen
from src.m8_direction_runtime import DirectionEstimate, M8DirectionEnsemble
from src.sample_hardware_easy_v2_multisize_direction import (
    SIZES,
    build_dense_tasks,
    build_easy_audit_tasks,
    build_m8_source_audit_tasks,
)
from src.sample_m8_conical_pilot import (
    PHASE_COLUMNS,
    capture_xy_shift_threshold_for_spec,
)
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
M8_BIT_RADIUS_M = 0.006 / math.sqrt(3.0)


def _spec(size: str, profile: str = "hardware_easy_v2") -> dict:
    path = ROOT / "configs" / f"allen_{size}_{profile}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _circular_error_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _synthetic_trial() -> training.LoadedTrial:
    phase = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    sequence = np.zeros((len(PHASE_COLUMNS), len(phase)), dtype=np.float32)
    columns = tuple(PHASE_COLUMNS)
    sequence[columns.index("Fx_N")] = 0.20 * np.cos(phase)
    sequence[columns.index("Fy_N")] = 0.20 * np.sin(phase)
    sequence[columns.index("Fz_N")] = 3.0 + 0.40 * np.cos(phase)
    sequence[columns.index("Tx_Nm")] = 0.010 + 0.002 * np.sin(phase)
    sequence[columns.index("Ty_Nm")] = -0.006 + 0.003 * np.cos(phase)
    sequence[columns.index("tip_x_m")] = 0.002 * np.cos(phase)
    sequence[columns.index("tip_y_m")] = 0.002 * np.sin(phase)
    sequence[columns.index("tip_z_m")] = 0.001 + 0.0002 * np.sin(phase)
    metadata = {
        "trial_id": "synthetic_cross_size",
        "base_pose_id": "synthetic_cross_size",
        "symmetry_orbit_id": "synthetic_cross_size_orbit",
        "search_state": "supported_search",
        "radial_level": 0,
        "radius_m": 0.003,
        "base_angle_deg": 0.0,
        "rotation_deg": 0.0,
        "rotation_index": 0,
        "target_down_force_n": 3.0,
        "wrench_observation": "ideal",
        "direction_supervision_valid": True,
    }
    return training.LoadedTrial(metadata, sequence, columns)


class _RecordingRuntime:
    def __init__(self) -> None:
        self.bit_radii: list[float] = []

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        self.bit_radii.append(float(bit_radius_m))
        state_probability = np.zeros(len(training.STATE_NAMES), dtype=float)
        state_probability[training.STATE_TO_INDEX["supported_search"]] = 1.0
        direction_probability = np.zeros(training.DIRECTION_CLASSES, dtype=float)
        true_class = training._direction_class(180.0)
        direction_probability[true_class] = 1.0
        return DirectionEstimate(
            state_probabilities=state_probability,
            direction_probabilities=direction_probability,
            top_state="supported_search",
            top_direction_deg=180.0,
            circular_mean_direction_deg=180.0,
            direction_xy=np.asarray([-1.0, 0.0]),
            confidence=1.0,
            aps_direction_classes=(true_class,),
            aps_direction_angles_deg=(180.0,),
            circular_concentration=1.0,
            ensemble_top1_span_deg=0.0,
            transport_allowed=True,
        )


class TestHardwareEasyV2Multisize(unittest.TestCase):
    def _assert_complete_c6(self, tasks: list, expected_orbits: int) -> None:
        by_orbit: dict[str, list] = {}
        for task in tasks:
            by_orbit.setdefault(task.symmetry_orbit_id, []).append(task)
        self.assertEqual(len(by_orbit), expected_orbits)
        for orbit in by_orbit.values():
            self.assertEqual(len(orbit), 6)
            self.assertEqual({task.rotation_index for task in orbit}, set(range(6)))
            self.assertEqual(
                {task.rotation_deg for task in orbit},
                {60.0 * index for index in range(6)},
            )
            invariant = {
                (
                    task.radius_m,
                    task.base_angle_deg,
                    task.friction_slide,
                    task.contact_timeconst_s,
                    task.target_down_force_n,
                    task.yaw_rad,
                    task.nuisance_seed,
                )
                for task in orbit
            }
            self.assertEqual(len(invariant), 1)

    def test_task_counts_c6_and_all_audit_generations_are_disjoint(self) -> None:
        all_ids: dict[str, set[str]] = {}
        for size in SIZES:
            spec = _spec(size)
            train = build_dense_tasks(spec, size)
            audits = [
                build_easy_audit_tasks(spec, size, generation)
                for generation in range(1, 10)
            ]
            with self.subTest(size=size):
                self.assertEqual(len(train), 768)
                self._assert_complete_c6(train, expected_orbits=128)
                collections = [train, *audits]
                for audit in audits:
                    self.assertEqual(len(audit), 180)
                    self._assert_complete_c6(audit, expected_orbits=30)
                for left_index, left_tasks in enumerate(collections):
                    for right_tasks in collections[left_index + 1 :]:
                        self.assertFalse(
                            {task.radius_m for task in left_tasks}
                            & {task.radius_m for task in right_tasks}
                        )
                        self.assertFalse(
                            {task.base_angle_deg for task in left_tasks}
                            & {task.base_angle_deg for task in right_tasks}
                        )
                        self.assertFalse(
                            {task.trial_id for task in left_tasks}
                            & {task.trial_id for task in right_tasks}
                        )
                        self.assertFalse(
                            {task.symmetry_orbit_id for task in left_tasks}
                            & {task.symmetry_orbit_id for task in right_tasks}
                        )
                all_ids[size] = {
                    task.trial_id for tasks in collections for task in tasks
                }
        for index, left in enumerate(SIZES):
            for right in SIZES[index + 1 :]:
                self.assertFalse(all_ids[left] & all_ids[right])

        source_generations = [
            build_m8_source_audit_tasks(_spec("m8", "source"), generation)
            for generation in range(1, 10)
        ]
        for source in source_generations:
            self.assertEqual(len(source), 180)
            self._assert_complete_c6(source, expected_orbits=30)
        for left_index, left_tasks in enumerate(source_generations):
            for right_tasks in source_generations[left_index + 1 :]:
                self.assertFalse(
                    {task.radius_m for task in left_tasks}
                    & {task.radius_m for task in right_tasks}
                )
                self.assertFalse(
                    {task.base_angle_deg for task in left_tasks}
                    & {task.base_angle_deg for task in right_tasks}
                )

    def test_capture_shift_threshold_is_size_aware_and_source_has_fallback(self) -> None:
        expected_easy = {
            "m8": 0.0002475,
            "m10": 0.0003105,
            "m12": 0.0003750,
        }
        for size, expected in expected_easy.items():
            threshold, formula = capture_xy_shift_threshold_for_spec(_spec(size))
            with self.subTest(profile="easy", size=size):
                self.assertAlmostEqual(threshold, expected, places=12)
                self.assertEqual(
                    formula, "0.15 * paired_flat_normal_guidance_margin_m"
                )

        for size in SIZES:
            spec = _spec(size, "source")
            threshold, formula = capture_xy_shift_threshold_for_spec(spec)
            expected = 0.00025 * float(spec["key_af_m"]) / 0.006
            with self.subTest(profile="source", size=size):
                self.assertAlmostEqual(threshold, expected, places=12)
                self.assertEqual(
                    formula, "0.25 mm * (bit_radius / M8_bit_radius)"
                )

    def _assert_task_geometry(self, task: object, spec: dict) -> None:
        bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
        head_radius = 0.5 * float(spec["screw_head_diameter_m"])
        mouth_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
        expected_outer = head_radius - (float(task.radius_m) + bit_radius)
        expected_inner = (
            float(task.radius_m) - bit_radius - mouth_apothem
        )
        self.assertAlmostEqual(task.outer_signed_distance_m, expected_outer, places=12)
        self.assertAlmostEqual(task.inner_signed_distance_m, expected_inner, places=12)
        expected_state = (
            "supported_search"
            if expected_outer > 0.0
            else "double_overhang"
            if expected_inner < 0.0
            else "outer_partial"
        )
        self.assertEqual(task.search_state, expected_state)
        self.assertAlmostEqual(math.hypot(task.x_m, task.y_m), task.radius_m, places=12)
        true_center_direction = math.degrees(
            math.atan2(-float(task.y_m), -float(task.x_m))
        ) % 360.0
        label_direction = training._direction_angle_deg(asdict(task), 0)
        self.assertLess(
            _circular_error_deg(label_direction, true_center_direction), 1.0e-10
        )

    def test_every_task_label_points_to_center_and_state_matches_its_spec(self) -> None:
        for size in SIZES:
            spec = _spec(size)
            tasks = build_dense_tasks(spec, size) + build_easy_audit_tasks(
                spec, size
            )
            with self.subTest(size=size):
                for task in tasks:
                    self._assert_task_geometry(task, spec)
        source_spec = _spec("m8", "source")
        for task in build_m8_source_audit_tasks(source_spec):
            self._assert_task_geometry(task, source_spec)

    def test_make_example_uses_the_requested_size_radius(self) -> None:
        trial = _synthetic_trial()
        columns = trial.columns
        raw = np.vstack(
            [trial.sequence[columns.index(name)] for name in training.INPUT_CHANNELS]
        )
        m8_example = training.make_example(
            trial, rotation_index=0, bit_radius_m=M8_BIT_RADIUS_M
        )
        for size in ("m10", "m12"):
            bit_radius = float(_spec(size)["key_af_m"]) / math.sqrt(3.0)
            example = training.make_example(
                trial, rotation_index=0, bit_radius_m=bit_radius
            )
            shape, context = M8DirectionEnsemble._network_inputs(
                raw, float(trial.metadata["target_down_force_n"]), bit_radius
            )
            with self.subTest(size=size):
                np.testing.assert_allclose(example["sequence"], shape, atol=1.0e-7)
                np.testing.assert_allclose(example["context"], context, atol=1.0e-7)
                np.testing.assert_allclose(
                    training.make_examples(
                        [trial], None, (0,), bit_radius_m=bit_radius
                    )[0]["context"],
                    context,
                    atol=1.0e-7,
                )
                self.assertFalse(
                    np.allclose(example["context"][4:], m8_example["context"][4:])
                )

    def test_evaluator_forwards_the_requested_size_radius(self) -> None:
        trial = _synthetic_trial()
        records = [(Path("synthetic.npz"), trial.metadata, trial.sequence, trial.columns)]
        for size in ("m10", "m12"):
            bit_radius = float(_spec(size)["key_af_m"]) / math.sqrt(3.0)
            runtime = _RecordingRuntime()
            result, rows = _evaluate(runtime, records, bit_radius_m=bit_radius)
            with self.subTest(size=size):
                self.assertEqual(runtime.bit_radii, [bit_radius])
                self.assertEqual(len(rows), 1)
                self.assertEqual(result["metrics"]["valid_full_fingerprint_samples"], 1)
                self.assertEqual(result["metrics"]["angle_error_median_deg"], 0.0)

    def test_radial_screen_uses_absolute_safety_not_unpaired_one_count_delta(self) -> None:
        def result(accepted: int, errors: int) -> dict:
            return {
                "strata": {
                    "radial_level": {
                        "0": {
                            "samples": 36,
                            "transport_accepted_count": accepted,
                            "transport_accepted_within_7p5_rate": (
                                (accepted - errors) / accepted
                            ),
                        }
                    }
                }
            }

        screen = _radial_finite_sample_screen(result(30, 2), result(28, 0))
        self.assertTrue(screen["passed"])
        detail = screen["radial_levels"]["0"]
        self.assertFalse(detail["relative_non_forgetting_diagnostic_only"])
        self.assertTrue(detail["absolute_tolerance_ok"])
        self.assertTrue(detail["passed"])

        self.assertFalse(_radial_finite_sample_screen(result(30, 3))["passed"])
        self.assertFalse(_radial_finite_sample_screen(result(3, 0))["passed"])

    def test_promoted_v4_and_final_fresh6_contracts_are_frozen(self) -> None:
        model_root = (
            ROOT
            / "outputs/current/models/hardware_easy_v2_multisize_robust_direction_v4"
        )
        audit_root = (
            ROOT
            / "outputs/current/evaluations/hardware_easy_v2_multisize_robustness_fresh6"
        )
        contract = json.loads(
            (model_root / "ensemble_interface_contract.json").read_text(
                encoding="utf-8"
            )
        )
        adaptation = json.loads(
            (model_root / "adaptation_report.json").read_text(encoding="utf-8")
        )
        report = json.loads((audit_root / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(contract["supported_sizes"], list(SIZES))
        self.assertEqual(len(contract["checkpoints"]), 3)
        self.assertEqual(contract["aps_miscoverage_target"], 0.05)
        self.assertEqual(
            contract["adaptation_contract"][
                "captured_or_incomplete_traces_used_for_direction_supervision"
            ],
            0,
        )
        self.assertEqual(
            contract["adaptation_contract"][
                "final_promotion_audit_samples_seen_during_fitting"
            ],
            0,
        )
        self.assertEqual(adaptation["test_samples_seen"], 0)
        self.assertFalse(adaptation["final_audit_run"])
        self.assertTrue(
            report[
                "promotion_ready_for_geometry_conditioned_multisize_simulation"
            ]
        )
        self.assertTrue(
            all(
                item["passed"]
                for item in report["size_promotion_gates"].values()
            )
        )
        self.assertTrue(
            report["models"]["joint_multisize"]["root"].endswith(
                "hardware_easy_v2_multisize_robust_direction_v4"
            )
        )
        for domain in ("m8_source", "m8_easy", "m10_easy", "m12_easy"):
            self.assertEqual(report["datasets"][domain]["training_use"], "forbidden")
            metrics = report["results"]["joint_multisize"][domain]["metrics"]
            self.assertEqual(metrics["total_task_outcomes"], 180)
            self.assertEqual(metrics["hard_stop_outcome_count"], 0)
            self.assertEqual(
                metrics["transport_accepted_catastrophic_gt90_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
