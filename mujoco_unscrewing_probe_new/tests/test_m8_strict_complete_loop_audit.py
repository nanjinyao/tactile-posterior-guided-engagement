from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop"


@unittest.skipUnless((OUT / "report.json").is_file(), "strict audit not generated")
class TestM8StrictCompleteLoopAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))

    def test_contract_is_full_loop_sensor_only_online_and_strict_budget(self) -> None:
        design = self.report["design"]
        self.assertEqual(self.report["experiment"], "M8_strict_24_pose_complete_closed_loop")
        self.assertEqual(design["poses"], 24)
        self.assertTrue(design["continue_to_cad_bottom"])
        self.assertTrue(
            design["budget_is_hard_accumulated_simulated_active_controller_time"]
        )
        self.assertEqual(design["active_time_budget_s"], 130.0)
        self.assertFalse(design["online_true_center_or_contact_tags"])
        self.assertTrue(design["conditioned_on_ambiguous_top_contact"])
        self.assertTrue(design["direct_retry_excluded_from_strict_table"])
        self.assertEqual(self.report["fingerprint_match_rate"], 1.0)

    def test_every_non_oracle_trial_obeys_budget_and_has_no_extension(self) -> None:
        trials = sorted((OUT / "trials").glob("sobol_*.json"))
        self.assertEqual(len(trials), 24)
        for path in trials:
            record = json.loads(path.read_text(encoding="utf-8"))
            for method, result in record["methods"].items():
                self.assertLessEqual(result["total_active_time_s"], 130.0 + 1.0e-9)
                self.assertAlmostEqual(result["fallback_budget_extension_s"], 0.0)
                if method != "oracle":
                    self.assertTrue(result["online_capture_sensor_only"])

    def test_report_separates_capture_from_cad_bottom(self) -> None:
        for method in self.report["methods"].values():
            self.assertIn("offline_capture_successes", method)
            self.assertIn("strict_bottom_successes", method)
            self.assertIn("sensor_handoff_false_positive_count", method)
            self.assertLessEqual(
                method["strict_bottom_successes"], method["offline_capture_successes"]
            )
            self.assertEqual(method["budget_contract_violations"], 0)
            self.assertEqual(method["fallback_budget_extension_nonzero"], 0)
            self.assertIn("role", method)
            self.assertIn("terminal_mode_counts", method)
            self.assertIn("preprobe_route_counts", method)
            self.assertIn("hard_stop_reason_counts", method)
            self.assertGreaterEqual(
                method["max_raw_physical_force_all_steps_p90_n"], 0.0
            )

    def test_reported_strict_successes_match_the_full_contract(self) -> None:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((OUT / "trials").glob("sobol_*.json"))
        ]
        for method_name, summary in self.report["methods"].items():
            counted = sum(
                _strict_bottom_success(record["methods"][method_name])
                for record in records
            )
            self.assertEqual(counted, summary["strict_bottom_successes"])


if __name__ == "__main__":
    unittest.main()
