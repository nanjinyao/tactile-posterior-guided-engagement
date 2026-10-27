import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "evaluations" / "p2_m8_learned_search_benchmark"


class TestP2LearnedSearchArtifacts(unittest.TestCase):
    def test_p2_is_complete_paired_and_claim_bounded(self) -> None:
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["completed"], 54)
        self.assertEqual(manifest["worker_errors"], 0)
        self.assertEqual(len(list((OUT / "trials").glob("*.json"))), 54)
        self.assertEqual(report["design"]["common_active_budget_s"], 30.0)
        self.assertEqual(report["paired"]["complete_pose_triplets"], 18)
        self.assertEqual(report["methods"]["learned"]["successes"], 7)
        self.assertEqual(report["methods"]["baseline"]["successes"], 6)
        self.assertEqual(report["methods"]["oracle"]["successes"], 18)
        self.assertGreater(
            report["methods"]["learned"]["success_rate"],
            report["methods"]["baseline"]["success_rate"],
        )
        self.assertEqual(report["methods"]["learned"]["direction_within_7p5_rate"], 1.0)
        inside = report["strata"]["inside_synthetic_visual_95"]["methods"]
        outside = report["strata"]["outside_synthetic_visual_95"]["methods"]
        self.assertEqual(inside["learned"]["successes"], 7)
        self.assertEqual(inside["baseline"]["successes"], 6)
        self.assertLess(
            inside["learned"]["penalized_time_median_s"],
            inside["baseline"]["penalized_time_median_s"],
        )
        self.assertEqual(outside["learned"]["successes"], 0)
        self.assertEqual(outside["baseline"]["successes"], 0)
        self.assertIn("privileged MuJoCo contact tags", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
