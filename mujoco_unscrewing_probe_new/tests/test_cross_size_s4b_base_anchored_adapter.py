from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import src.cross_size_s4b_base_anchored_adapter as adapter
from src.m8_direction_runtime import DirectionEstimate
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]


def _estimate(
    direction_probability: np.ndarray,
    *,
    mean_deg: float,
    concentration: float,
    span_deg: float,
    confidence: float,
    aps: tuple[int, ...],
    state_probability: np.ndarray | None = None,
    allowed: bool = True,
) -> DirectionEstimate:
    if state_probability is None:
        state_probability = np.asarray([0.6, 0.4], dtype=float)
    direction_rad = np.deg2rad(mean_deg)
    return DirectionEstimate(
        state_probabilities=np.asarray(state_probability, dtype=float),
        direction_probabilities=np.asarray(direction_probability, dtype=float),
        top_state=training.STATE_NAMES[int(np.argmax(state_probability))],
        top_direction_deg=float(
            training._class_angle_deg(int(np.argmax(direction_probability)))
        ),
        circular_mean_direction_deg=float(mean_deg),
        direction_xy=np.asarray(
            [np.cos(direction_rad), np.sin(direction_rad)], dtype=float
        ),
        confidence=float(confidence),
        aps_direction_classes=aps,
        aps_direction_angles_deg=tuple(
            map(float, training._class_angle_deg(np.asarray(aps, dtype=int)))
        ),
        circular_concentration=float(concentration),
        ensemble_top1_span_deg=float(span_deg),
        transport_allowed=allowed,
    )


class BaseAnchoredAdapterFormulaTests(unittest.TestCase):
    def test_residual_mixture_has_nonzero_parent_weights_and_vector_identity(self) -> None:
        base = np.zeros(training.DIRECTION_CLASSES, dtype=float)
        fine_tuned = np.zeros_like(base)
        base[0] = 1.0
        fine_tuned[4] = 1.0
        alpha = 0.92
        mixed = adapter.residual_probability_mix(base, fine_tuned, alpha)
        self.assertAlmostEqual(mixed[0], 0.08)
        self.assertAlmostEqual(mixed[4], 0.92)
        angles = np.deg2rad(
            training._class_angle_deg(
                np.arange(training.DIRECTION_CLASSES)
            )
        )
        z_base = np.sum(base * np.exp(1j * angles))
        z_fine = np.sum(fine_tuned * np.exp(1j * angles))
        z_mixed = np.sum(mixed * np.exp(1j * angles))
        self.assertAlmostEqual(
            abs(z_mixed - (z_base + alpha * (z_fine - z_base))), 0.0
        )
        with self.assertRaisesRegex(ValueError, "strictly between"):
            adapter.residual_probability_mix(base, fine_tuned, 0.0)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            adapter.residual_probability_mix(base, fine_tuned, 1.0)

    def test_confidence_envelope_and_aps_union_are_conservative(self) -> None:
        base_probability = np.zeros(training.DIRECTION_CLASSES, dtype=float)
        fine_probability = np.zeros_like(base_probability)
        base_probability[0] = 0.8
        base_probability[1] = 0.2
        fine_probability[0] = 0.1
        fine_probability[1] = 0.7
        fine_probability[2] = 0.2
        base = _estimate(
            base_probability,
            mean_deg=3.0,
            concentration=0.80,
            span_deg=7.5,
            confidence=0.80,
            aps=(0, 1),
        )
        fine = _estimate(
            fine_probability,
            mean_deg=10.0,
            concentration=0.70,
            span_deg=15.0,
            confidence=0.70,
            aps=(1, 2),
            state_probability=np.asarray([0.2, 0.8]),
        )
        estimate, telemetry = adapter.combine_direction_estimates(
            base,
            fine,
            alpha=0.92,
            minimum_fine_tuned_concentration=0.60,
        )
        self.assertTrue(telemetry.risk_gate_passed)
        self.assertEqual(set(estimate.aps_direction_classes), {0, 1, 2})
        self.assertLessEqual(estimate.circular_concentration, 0.70)
        self.assertLessEqual(estimate.confidence, 0.70)
        self.assertGreaterEqual(estimate.ensemble_top1_span_deg, 15.0)
        np.testing.assert_allclose(
            estimate.state_probabilities,
            0.08 * base.state_probabilities + 0.92 * fine.state_probabilities,
        )

    def test_low_ft_concentration_abstains_instead_of_correcting_label(self) -> None:
        probability = np.zeros(training.DIRECTION_CLASSES, dtype=float)
        probability[0] = 1.0
        base = _estimate(
            probability,
            mean_deg=0.0,
            concentration=0.99,
            span_deg=0.0,
            confidence=0.99,
            aps=(0,),
        )
        fine = _estimate(
            probability,
            mean_deg=5.0,
            concentration=0.59,
            span_deg=0.0,
            confidence=0.90,
            aps=(0,),
        )
        estimate, telemetry = adapter.combine_direction_estimates(
            base,
            fine,
            alpha=0.92,
            minimum_fine_tuned_concentration=0.60,
        )
        self.assertFalse(telemetry.risk_gate_passed)
        self.assertFalse(estimate.transport_allowed)
        self.assertEqual(estimate.circular_concentration, 0.0)
        self.assertEqual(estimate.ensemble_top1_span_deg, 180.0)
        self.assertIn(
            "fine_tuned_concentration_below_train_validation_selected_floor",
            telemetry.risk_gate_reasons,
        )


class BaseAnchoredAdapterSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "adapter"
        cls.roles_read: list[str] = []
        original = adapter._load_role_examples

        def audited_loader(data_root, size, contract, role):
            cls.roles_read.append(role)
            return original(data_root, size, contract, role)

        args = argparse.Namespace(
            data_root=str(adapter.DEFAULT_DATA_ROOT),
            base_root=str(adapter.DEFAULT_BASE_ROOT),
            fine_tuned_root=str(adapter.DEFAULT_FINE_TUNED_ROOT),
            output=str(cls.output),
            sizes=list(adapter.SIZES),
        )
        with mock.patch.object(
            adapter, "_load_role_examples", side_effect=audited_loader
        ):
            adapter.run(args)
        cls.report = json.loads(
            (cls.output / "selection_report.json").read_text(encoding="utf-8")
        )
        cls.contract = json.loads(
            (cls.output / "adapter_contract.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_selector_reads_only_train_and_validation_examples(self) -> None:
        self.assertEqual(
            sorted(set(self.roles_read)), ["train", "validation"]
        )
        access = self.report["data_access_contract"]
        self.assertEqual(access["example_roles_read"], ["train", "validation"])
        self.assertEqual(access["calibration_examples_read"], 0)
        self.assertEqual(access["calibration_labels_read"], 0)
        self.assertEqual(access["full_loop_rollouts_read"], 0)
        self.assertEqual(access["full_loop_pose_labels_read"], 0)
        self.assertFalse(self.contract["full_loop_execution_authorized"])
        self.assertFalse(self.contract["selector_may_launch_full_loop"])
        self.assertTrue(
            self.contract[
                "eligible_for_separately_frozen_full_loop_evaluation"
            ]
        )

    def test_joint_validation_rule_selects_a_true_residual_adapter(self) -> None:
        selected = self.report["selected_parameters"]
        self.assertEqual(selected["residual_alpha"], 0.92)
        self.assertAlmostEqual(selected["base_anchor_weight"], 0.08)
        self.assertEqual(selected["minimum_fine_tuned_concentration"], 0.60)
        self.assertEqual(
            selected["maximum_base_fine_tuned_disagreement_deg"], 90.0
        )
        self.assertGreater(selected["residual_alpha"], 0.0)
        self.assertLess(selected["residual_alpha"], 1.0)

    def test_m10_raw_outlier_is_reported_and_abstained(self) -> None:
        metrics = self.report["selected_metrics"]["m10"]["validation"]
        self.assertEqual(metrics["raw_control_gt90_count"], 1)
        self.assertEqual(metrics["accepted_control_gt90_count"], 0)
        self.assertAlmostEqual(metrics["gate_coverage"], 35.0 / 36.0)
        outlier = metrics["control_gt90_outliers"]
        self.assertEqual(len(outlier), 1)
        self.assertEqual(
            outlier[0]["trial_id"],
            "s4bva_m10_double_overhang_r5_p225_s0",
        )
        self.assertFalse(outlier[0]["accepted_for_directed_shortcut"])
        self.assertLess(outlier[0]["fine_tuned_concentration"], 0.60)

    def test_m12_validation_has_no_raw_or_accepted_catastrophe(self) -> None:
        metrics = self.report["selected_metrics"]["m12"]["validation"]
        self.assertEqual(metrics["raw_control_gt90_count"], 0)
        self.assertEqual(metrics["accepted_control_gt90_count"], 0)
        self.assertGreaterEqual(
            metrics["gate_coverage"], adapter.MIN_OVERALL_GATE_COVERAGE
        )
        self.assertAlmostEqual(
            metrics["raw_control_error_p90_deg"], 19.5755453584145
        )

    def test_deployable_runtime_replays_the_m10_abstention(self) -> None:
        trial_id = "s4bva_m10_double_overhang_r5_p225_s0"
        trial_path = (
            adapter.DEFAULT_DATA_ROOT
            / "m10"
            / "trials"
            / f"{trial_id}.npz"
        )
        with np.load(trial_path, allow_pickle=False) as data:
            sequence = np.asarray(data["phase_sequence"], dtype=float)
            columns = tuple(map(str, data["phase_columns"].tolist()))
            metadata = json.loads(str(data["metadata_json"]))
        raw = np.vstack(
            [
                sequence[columns.index(name)]
                for name in ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")
            ]
        )
        spec = json.loads(
            (ROOT / "configs/allen_m10_source.json").read_text(
                encoding="utf-8"
            )
        )
        runtime = adapter.BaseAnchoredS4bDirectionAdapter(
            "m10", self.output
        )
        estimate = runtime.infer_phase_sequence(
            raw,
            target_down_force_n=float(metadata["target_down_force_n"]),
            bit_radius_m=float(spec["key_af_m"]) / math.sqrt(3.0),
        )
        self.assertIsNotNone(runtime.last_telemetry)
        self.assertFalse(runtime.last_telemetry.risk_gate_passed)
        self.assertEqual(estimate.circular_concentration, 0.0)
        self.assertEqual(estimate.ensemble_top1_span_deg, 180.0)
        self.assertFalse(estimate.transport_allowed)
        self.assertAlmostEqual(
            runtime.last_telemetry.fine_tuned_concentration,
            0.5963206537736432,
        )

    def test_provenance_hashes_frozen_inputs(self) -> None:
        provenance = self.report["provenance"]
        for size in adapter.SIZES:
            self.assertIn(size, provenance["dataset_contracts"])
            self.assertIn(size, provenance["fine_tuned_model_contracts"])
            self.assertEqual(
                len(provenance["execution_files"][size]["base"]), 3
            )
            self.assertEqual(
                len(provenance["execution_files"][size]["fine_tuned"]), 3
            )
            for variant in ("base", "fine_tuned"):
                for record in provenance["execution_files"][size][variant]:
                    self.assertEqual(len(record["checkpoint_sha256"]), 64)
                    self.assertEqual(len(record["normalization_sha256"]), 64)
                    self.assertEqual(len(record["metrics_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
