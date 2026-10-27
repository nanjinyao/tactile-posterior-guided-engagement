from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/evaluations/m8_fair_policy_final_audit"


class TestM8FairPolicyFinalAudit(unittest.TestCase):
    def test_final_audit_is_complete_and_fair(self) -> None:
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["design"]["poses"], 36)
        self.assertEqual(report["probe_hash_match_rate"], 1.0)
        self.assertTrue(report["design"]["proposed_global_outward_fallback"])
        self.assertEqual(report["design"]["posterior_residual_coverage"], 0.99)
        self.assertTrue(report["design"]["capture_uses_privileged_mujoco_tags"])

    def test_final_result_keeps_the_claim_boundary(self) -> None:
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        proposed = report["methods"]["proposed_posterior"]
        outward = report["methods"]["baseline_a"]
        direct = report["methods"]["baseline_b"]
        self.assertEqual((proposed["successes"], outward["successes"], direct["successes"]), (34, 31, 34))
        self.assertEqual(proposed["global_fallback_count"], 1)
        self.assertEqual(report["paired_proposed_posterior_vs_baseline_b"]["left_only_success"], 2)
        self.assertEqual(report["paired_proposed_posterior_vs_baseline_b"]["right_only_success"], 2)
        self.assertEqual(
            report["paired_efficiency_proposed_vs_baseline_b"][
                "left_lower_time_to_inner_wall_s_rate"
            ],
            0.0,
        )

    def test_development_runs_are_not_in_final_evaluation(self) -> None:
        diagnostics = ROOT / "outputs/current/diagnostics"
        self.assertTrue((diagnostics / "m8_fair_policy_pre_confirmation_fix").is_dir())
        self.assertTrue((diagnostics / "m8_fair_policy_development_q95").is_dir())
        self.assertTrue((diagnostics / "m8_fair_policy_development_q99_no_fallback").is_dir())
        self.assertTrue((diagnostics / "m8_fair_policy_development_false_progress_gate").is_dir())


if __name__ == "__main__":
    unittest.main()
