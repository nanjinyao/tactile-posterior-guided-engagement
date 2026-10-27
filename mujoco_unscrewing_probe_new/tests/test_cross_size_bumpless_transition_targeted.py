from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.run_cross_size_bumpless_transition_targeted import (
    DEFAULT_SOURCE,
    TARGET_MATRIX,
    build_contract,
)
from src.run_cross_size_s4_full_loop_evaluation import OutputExecutionLock


class TestCrossSizeBumplessTransitionTargeted(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract, cls.tasks = build_contract(DEFAULT_SOURCE)

    def test_matrix_is_exactly_the_predeclared_ten_development_arms(self) -> None:
        self.assertEqual(self.contract["phase"], "development")
        self.assertEqual(self.contract["target_arm_count"], 10)
        self.assertEqual(
            self.contract["pose_ids"],
            [pose_id for _, pose_id, _ in TARGET_MATRIX],
        )
        self.assertEqual(
            self.contract["methods_by_pose"]["m10_s3test_011"],
            ["s4_fine_tuned", "zero_shot", "cal_only"],
        )
        self.assertEqual(
            self.contract["methods_by_pose"]["m12_s3test_003"],
            [
                "s4_fine_tuned",
                "pure_outward",
                "privileged_straight_to_center",
            ],
        )
        self.assertEqual(
            self.contract["methods_by_pose"]["m12_s3test_010"],
            ["s4_fine_tuned", "cal_only", "privileged_straight_to_center"],
        )
        self.assertEqual(
            self.contract["methods_by_pose"]["m12_s3test_001"],
            ["privileged_straight_to_center"],
        )

    def test_contract_reads_no_untouched_tasks_or_seeds(self) -> None:
        self.assertFalse(self.contract["untouched_tasks_or_seeds_read"])
        self.assertFalse(self.contract["full_development_campaign_launched"])
        self.assertNotIn("untouched", json.dumps(self.contract["target_matrix"]).lower())
        self.assertEqual(set(self.tasks), set(self.contract["pose_ids"]))

    def test_no_safety_or_terminal_threshold_is_relaxed(self) -> None:
        safety = self.contract["fairness_and_safety"]
        self.assertFalse(safety["raw_hard_wrench_thresholds_relaxed"])
        self.assertFalse(safety["soft_wrench_thresholds_changed"])
        self.assertFalse(safety["capture_thresholds_changed"])
        self.assertFalse(safety["terminal_verifier_thresholds_changed"])
        self.assertFalse(safety["non_oracle_true_center_available"])
        self.assertFalse(safety["non_oracle_offline_contact_tags_available"])
        self.assertTrue(
            safety["same_transition_and_physical_funnel_for_all_exercised_methods"]
        )

    def test_trace_and_structured_transition_telemetry_are_mandatory(self) -> None:
        telemetry = self.contract["telemetry_contract"]
        self.assertTrue(telemetry["one_trace_csv_per_arm"])
        self.assertTrue(telemetry["trace_contains_measured_six_axis_wrench"])
        self.assertTrue(telemetry["trace_contains_raw_physical_six_axis_wrench"])
        self.assertTrue(telemetry["trace_contains_filtered_six_axis_wrench"])
        self.assertTrue(telemetry["transition_event_on_first_post_rebase_step"])
        self.assertTrue(telemetry["structured_transition_telemetry_in_result"])
        self.assertTrue(telemetry["exclusive_output_execution_lock_required"])
        self.assertTrue(telemetry["run_id_recorded_per_arm_and_report"])

    def test_targeted_execution_lock_rejects_a_second_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = OutputExecutionLock.acquire(
                output,
                phase="development",
                task_contract_sha256="task-hash",
                execution_bundle_sha256="bundle-hash",
                run_id="first-targeted-owner",
            )
            try:
                with self.assertRaises(RuntimeError):
                    OutputExecutionLock.acquire(
                        output,
                        phase="development",
                        task_contract_sha256="task-hash",
                        execution_bundle_sha256="bundle-hash",
                        run_id="second-targeted-owner",
                    )
            finally:
                first.release()

    def test_source_v4_contains_the_expected_failure_controls(self) -> None:
        by_arm = {
            (row["pose_id"], row["method"]): row["source_v4"]
            for row in self.contract["target_matrix"]
        }
        self.assertTrue(by_arm[("m10_s3test_011", "s4_fine_tuned")]["hard_stop"])
        self.assertFalse(by_arm[("m10_s3test_011", "zero_shot")]["hard_stop"])
        self.assertTrue(by_arm[("m12_s3test_003", "pure_outward")]["hard_stop"])
        self.assertTrue(by_arm[("m12_s3test_010", "s4_fine_tuned")]["hard_stop"])
        self.assertTrue(
            by_arm[("m12_s3test_001", "privileged_straight_to_center")][
                "hard_stop"
            ]
        )


if __name__ == "__main__":
    unittest.main()
