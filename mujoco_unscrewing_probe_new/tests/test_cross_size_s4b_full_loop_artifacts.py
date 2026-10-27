from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4b_full_loop_development_v5"
)
UNTOUCHED = (
    ROOT
    / "outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1"
)


class TestCrossSizeS4bFullLoopArtifacts(unittest.TestCase):
    def _load_and_validate_run(self, path: Path) -> tuple[dict, dict, list[dict]]:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        provenance = json.loads(
            (path / "provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["run_id"], report["run_id"])
        self.assertEqual(provenance["run_id"], report["run_id"])
        self.assertEqual(
            manifest["task_contract_sha256"], report["task_contract_sha256"]
        )
        self.assertEqual(
            provenance["task_contract_sha256"], report["task_contract_sha256"]
        )
        self.assertEqual(
            manifest["execution_bundle_sha256"], report["execution_bundle_sha256"]
        )
        self.assertEqual(
            provenance["execution_bundle_sha256"], report["execution_bundle_sha256"]
        )
        self.assertFalse((path / ".s4_execution.lock.json").exists())

        trials: list[dict] = []
        for size in ("m10", "m12"):
            records = [
                json.loads(item.read_text(encoding="utf-8"))
                for item in sorted((path / size / "trials").glob("*.json"))
            ]
            self.assertEqual(len(records), 12)
            self.assertTrue(all("error" not in record for record in records))
            self.assertTrue(
                all(record["run_id"] == report["run_id"] for record in records)
            )
            self.assertTrue(
                all(
                    record["worker_init_execution_bundle_sha256"]
                    == report["execution_bundle_sha256"]
                    == record["worker_completion_execution_bundle_sha256"]
                    for record in records
                )
            )
            self.assertTrue(
                all(record["fairness"]["simulator_reset_count"] == 0 for record in records)
            )
            self.assertTrue(
                all(record["fairness"]["zero_budget_extension"] for record in records)
            )
            self.assertTrue(
                all(
                    record["fairness"]["learned_first_probe_hashes_match"]
                    for record in records
                )
            )
            trials.extend(records)
        return report, provenance, trials

    def test_development_passes_but_does_not_support_necessity(self) -> None:
        report, _, _ = self._load_and_validate_run(DEVELOPMENT)
        self.assertEqual(report["phase"], "development")
        self.assertTrue(report["all_S4_fine_tuned_size_gates_pass"])
        self.assertFalse(report["adaptation_necessity_supported_for_all_sizes"])
        expected = {"m10": (12, 0), "m12": (10, 0)}
        for size, (bottom, hard) in expected.items():
            method = report["sizes"][size]["methods"]["s4_fine_tuned"]
            self.assertEqual(method["strict_bottom_successes"], bottom)
            self.assertEqual(method["hard_stops"], hard)
            self.assertTrue(report["sizes"][size]["S4_fine_tuned_full_loop_gate_pass"])

    def test_single_read_untouched_has_one_real_m10_safety_failure(self) -> None:
        report, provenance, trials = self._load_and_validate_run(UNTOUCHED)
        self.assertEqual(report["phase"], "untouched")
        self.assertFalse(report["all_S4_fine_tuned_size_gates_pass"])
        self.assertFalse(report["adaptation_necessity_supported_for_all_sizes"])
        self.assertIn("forbidden", provenance["untouched_use_policy"])

        m10 = report["sizes"]["m10"]
        m12 = report["sizes"]["m12"]
        self.assertEqual(
            (m10["methods"]["s4_fine_tuned"]["strict_bottom_successes"],
             m10["methods"]["s4_fine_tuned"]["hard_stops"]),
            (10, 1),
        )
        self.assertFalse(m10["S4_fine_tuned_full_loop_gate_pass"])
        self.assertEqual(
            (m12["methods"]["s4_fine_tuned"]["strict_bottom_successes"],
             m12["methods"]["s4_fine_tuned"]["hard_stops"]),
            (10, 0),
        )
        self.assertTrue(m12["S4_fine_tuned_full_loop_gate_pass"])

        pose006 = next(
            record
            for record in trials
            if record["task"]["pose_id"] == "m10_s4_untouched_006"
        )
        outcome = pose006["methods"]["s4_fine_tuned"]
        self.assertEqual(outcome["result"], "hard_stop")
        self.assertTrue(outcome["hard_stop"])
        self.assertEqual(outcome["hard_stop_reason"], "raw_physical_wrench_limit:Ty")
        self.assertEqual(outcome["terminal_attempts"], 0)
        self.assertFalse(outcome["post_terminal_event_display_new_hard_stop"])


if __name__ == "__main__":
    unittest.main()
