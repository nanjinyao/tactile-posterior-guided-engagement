from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
METHODS = (
    "proposed",
    "proposed_plus",
    "outward",
    "improved_direct_retry3",
    "jin_direct_retry3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestM8FiveMethodRetry3Artifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(
            (AUDIT / "task_contract.json").read_text(encoding="utf-8")
        )
        self.report = json.loads((AUDIT / "report.json").read_text(encoding="utf-8"))

    def test_frozen_common_contract_and_counts(self) -> None:
        self.assertEqual(self.contract["methods"], list(METHODS))
        self.assertEqual(len(self.contract["tasks"]), 24)
        self.assertEqual(self.contract["active_time_budget_s"], 130.0)
        self.assertTrue(self.contract["zero_extension"])
        direct = self.contract["direct_contract"]
        self.assertEqual(direct["maximum_rays"], 3)
        self.assertEqual(direct["adjustment_m"], 0.00025)
        self.assertTrue(direct["same_rollout"])
        self.assertFalse(direct["simulator_reset"])
        funnel = self.contract["post_capture_contract"]
        self.assertTrue(funnel["enabled"])
        self.assertEqual(funnel["max_recoveries"], 3)
        self.assertTrue(funnel["retry_on_terminal_rejection"])

        expected_bottom = {
            "proposed": 20,
            "proposed_plus": 19,
            "outward": 19,
            "improved_direct_retry3": 17,
            "jin_direct_retry3": 16,
        }
        for method, expected in expected_bottom.items():
            with self.subTest(method=method):
                summary = self.report["methods"][method]
                self.assertEqual(summary["trials"], 24)
                self.assertEqual(summary["strict_bottom_successes"], expected)
                self.assertEqual(summary["budget_violations"], 0)
                self.assertEqual(summary["simulator_resets"], 0)
        self.assertTrue(self.report["all_first_probe_hashes_match"])
        self.assertTrue(self.report["all_budget_extensions_zero"])
        self.assertTrue(self.report["all_simulator_reset_counts_zero"])

    def test_direct_interface_comparison_is_mixed_not_uniformly_superior(self) -> None:
        improved = self.report["methods"]["improved_direct_retry3"]
        jin = self.report["methods"]["jin_direct_retry3"]
        self.assertEqual(improved["direct_first_ray_captures"], 18)
        self.assertEqual(jin["direct_first_ray_captures"], 16)
        self.assertEqual(improved["direct_any_ray_captures"], 19)
        self.assertEqual(jin["direct_any_ray_captures"], 19)
        improved_error = improved["first_probe_executed_direction_error_deg"]
        jin_error = jin["first_probe_executed_direction_error_deg"]
        self.assertLess(improved_error["median"], jin_error["median"])
        self.assertGreater(improved_error["p90"], jin_error["p90"])
        self.assertLess(improved_error["within_15_deg"], jin_error["within_15_deg"])
        self.assertEqual(
            self.report["paired"]["improved_vs_jin_direct_retry3"][
                "exact_mcnemar_p"
            ],
            1.0,
        )

    def test_all_trials_pay_retry_time_without_reset_and_share_first_probe(self) -> None:
        trials = sorted((AUDIT / "trials").glob("sobol_*.json"))
        self.assertEqual(len(trials), 24)
        for path in trials:
            with self.subTest(trial=path.stem):
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(set(record["methods"]), set(METHODS))
                fairness = record["fairness"]
                self.assertTrue(fairness["first_probe_fingerprint_sha_match"])
                self.assertTrue(fairness["direct_attempts_include_physical_adjustment_time"])
                self.assertEqual(fairness["simulator_reset_count"], 0)
                self.assertTrue(fairness["zero_budget_extension"])
                hashes = {
                    record["methods"][method]["fingerprint_sha256"]
                    for method in (
                        "proposed",
                        "proposed_plus",
                        "improved_direct_retry3",
                        "jin_direct_retry3",
                    )
                }
                self.assertEqual(len(hashes), 1)
                self.assertNotIn(None, hashes)
                for method in ("improved_direct_retry3", "jin_direct_retry3"):
                    result = record["methods"][method]
                    self.assertEqual(result["simulator_reset_count"], 0)
                    self.assertLessEqual(result["direct_attempts_completed"], 3)
                    self.assertLessEqual(result["total_active_time_s"], 130.0 + 1e-9)
                    attempts = result["direct_attempt_records"]
                    self.assertEqual(
                        [item["attempt"] for item in attempts],
                        list(range(1, len(attempts) + 1)),
                    )

    def test_manifest_hashes_current_report(self) -> None:
        manifest = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"], 24)
        self.assertEqual(manifest["report_sha256"], _sha256(AUDIT / "report.json"))
        self.assertEqual(manifest["summary_sha256"], _sha256(AUDIT / "summary.md"))


if __name__ == "__main__":
    unittest.main()
