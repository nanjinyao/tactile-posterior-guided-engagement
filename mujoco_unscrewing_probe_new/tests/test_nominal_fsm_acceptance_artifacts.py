import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "evaluations" / "p2b_nominal_fsm_acceptance"
DIAGNOSTIC = ROOT / "outputs" / "current" / "diagnostics" / "p2b_nominal_failure_diagnostic"


class TestNominalFsmAcceptanceArtifacts(unittest.TestCase):
    def test_nominal_regression_is_complete_safe_and_claim_bounded(self) -> None:
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["completed"], 36)
        self.assertEqual(progress["hard_stops"], 0)
        self.assertEqual(manifest["worker_errors"], 0)
        self.assertEqual(len(list((OUT / "trials").glob("*.json"))), 36)
        for method in ("learned", "baseline"):
            self.assertEqual(report["methods"][method]["successes"], 18)
            self.assertEqual(report["methods"][method]["hard_stops"], 0)
        self.assertEqual(report["paired"]["both_success"], 18)
        self.assertEqual(report["paired"]["learned_lower_path_mm_rate"], 1.0)
        self.assertGreaterEqual(report["paired"]["learned_lower_work_mj_rate"], 17 / 18)
        self.assertIn("not an independent test", report["claim_boundary"])

    def test_trace_driven_smoke_removes_the_nominal_hard_stop(self) -> None:
        summary = json.loads((DIAGNOSTIC / "summary.json").read_text(encoding="utf-8"))
        learned = summary["results"]["learned"]
        baseline = summary["results"]["baseline"]
        self.assertTrue(learned["inner_wall_confirmed"])
        self.assertFalse(learned["hard_stop"])
        self.assertEqual(learned["candidate_attempts"], 1)
        self.assertLess(learned["max_torque_mnm"], 60.0)
        self.assertTrue(baseline["inner_wall_confirmed"])
        self.assertFalse(baseline["hard_stop"])
        for name in ("learned_trace.csv", "baseline_trace.csv"):
            self.assertFalse((DIAGNOSTIC / name).exists())
        retention = json.loads(
            (ROOT / "outputs/current/DIAGNOSTIC_RETENTION.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(retention["archive"]["members"], 21)


if __name__ == "__main__":
    unittest.main()
