from __future__ import annotations

import json
import unittest

from src.run_m10_global_fallback_gate_targeted import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    METHODS,
    POSE_ID,
    build_contract,
)


class TestM10GlobalFallbackGateTargeted(unittest.TestCase):
    def test_contract_is_only_pose010_development(self) -> None:
        contract, task = build_contract(DEFAULT_SOURCE)
        self.assertEqual(contract["phase"], "development")
        self.assertEqual(task.pose_id, POSE_ID)
        self.assertEqual(contract["methods"], list(METHODS))
        self.assertFalse(contract["untouched_tasks_or_seeds_read"])
        self.assertNotIn("untouched", json.dumps(contract["task"]).lower())

    def test_contract_uses_raw_ft_without_adapter_or_true_center(self) -> None:
        contract, _ = build_contract(DEFAULT_SOURCE)
        self.assertFalse(contract["base_anchored_adapter_used"])
        self.assertFalse(contract["true_center_available_to_controller"])
        self.assertIn("loaded directly", contract["raw_fine_tuned_runtime"])
        self.assertFalse(contract["raw_hard_wrench_thresholds_relaxed"])
        self.assertFalse(contract["soft_wrench_thresholds_changed"])
        self.assertFalse(contract["capture_thresholds_changed"])
        self.assertFalse(contract["terminal_verifier_thresholds_changed"])
        execution_json = json.dumps(contract["execution_files"])
        self.assertNotIn("base_anchored_adapter", execution_json)

    def test_source_cases_are_exactly_reverse_posterior_global_failures(self) -> None:
        contract, _ = build_contract(DEFAULT_SOURCE)
        self.assertIn("reverse learned posterior", contract["source_failure_mechanism"])
        self.assertIn("global outward fallback", contract["source_failure_mechanism"])

    def test_completed_targeted_audit_has_persistent_retirement_telemetry(self) -> None:
        report = json.loads(
            (DEFAULT_OUTPUT / "report.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["phase"], "development")
        self.assertTrue(
            report[
                "all_exercised_global_fallback_gates_retired_before_motion"
            ]
        )
        self.assertFalse(report["base_anchored_adapter_used"])
        self.assertFalse(report["true_center_available_to_controller"])
        self.assertFalse(report["untouched_tasks_or_seeds_read"])
        for method in METHODS:
            row = report["methods"][method]
            self.assertTrue(row["regression_strict_bottom_success"])
            self.assertEqual(row["regression_result"], "cad_bottom_verified")
            self.assertFalse(row["hard_stop"])
            self.assertTrue(row["global_outward_fallback_used"])
            self.assertTrue(row["gate_initially_enabled"])
            self.assertTrue(row["gate_retired"])
            self.assertFalse(row["gate_active_at_outcome"])
            self.assertTrue(row["gate_retired_before_or_at_fallback_start"])
            self.assertEqual(
                row["gate_retirement_reason"],
                "initial_learned_posterior_abandoned_before_global_outward_fallback",
            )
            telemetry = manifest["gate_retirement_telemetry"][method]
            self.assertTrue(telemetry["retired"])
            self.assertEqual(
                telemetry["active_time_s"],
                row["gate_retired_active_time_s"],
            )


if __name__ == "__main__":
    unittest.main()
