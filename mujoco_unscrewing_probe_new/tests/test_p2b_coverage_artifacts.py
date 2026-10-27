import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "evaluations" / "p2b_m8_in_domain_coverage"


class TestP2bCoverageArtifacts(unittest.TestCase):
    def test_p2b_is_complete_in_domain_and_claim_bounded(self) -> None:
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["completed"], 90)
        self.assertEqual(manifest["worker_errors"], 0)
        self.assertEqual(len(list((OUT / "trials").glob("*.json"))), 90)
        self.assertEqual(report["design"]["initial_poses"], 30)
        self.assertEqual(len(report["design"]["radial_levels_mm"]), 5)
        self.assertGreater(
            report["design"]["program_guard_budget_s"],
            2.0 * report["design"]["baseline_nominal_coverage_time_s"],
        )
        self.assertEqual(report["methods"]["learned"]["successes"], 21)
        self.assertEqual(report["methods"]["baseline"]["successes"], 24)
        self.assertEqual(report["methods"]["oracle"]["successes"], 30)
        self.assertEqual(report["methods"]["learned"]["direction_within_7p5_rate"], 1.0)
        paired = report["paired"]
        self.assertEqual(paired["both_learned_and_baseline_success"], 21)
        self.assertEqual(paired["learned_shorter_path_rate_on_both_success"], 1.0)
        self.assertGreaterEqual(paired["learned_lower_work_rate_on_both_success"], 0.95)
        edge = report["by_radius"]["4.875_mm"]
        self.assertEqual(edge["learned"]["successes"], 0)
        self.assertEqual(edge["baseline"]["successes"], 0)
        self.assertEqual(edge["baseline"]["termination_counts"]["hard_stop"], 6)
        self.assertIn("privileged MuJoCo tags", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
