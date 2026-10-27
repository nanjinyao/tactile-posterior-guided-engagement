import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "current" / "evaluations" / "p1v2_m8_independent_direction_audit"


class TestP1v2IndependentDirectionAudit(unittest.TestCase):
    def test_frozen_audit_is_complete_and_excluded_from_training(self) -> None:
        progress = json.loads((AUDIT / "progress.json").read_text(encoding="utf-8"))
        contract = json.loads(
            (AUDIT / "sampling_contract.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (AUDIT / "independent_direction_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(progress["completed"], 180)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(len(list((AUDIT / "trials").glob("*.npz"))), 180)
        self.assertEqual(contract["training_use"], "forbidden")
        self.assertEqual(report["samples"], 180)
        self.assertIn("No fitting", report["claim_boundary"])

    def test_report_exposes_gate_failure_instead_of_hiding_it(self) -> None:
        report = json.loads(
            (AUDIT / "independent_direction_report.json").read_text(encoding="utf-8")
        )
        metrics = report["metrics"]
        self.assertGreaterEqual(metrics["aps_coverage"], 0.90)
        self.assertGreater(metrics["catastrophic_gt90_count"], 0)
        self.assertGreater(metrics["catastrophic_transport_allowed_count"], 0)
        self.assertIn("not real-robot", report["claim_boundary"].lower())


if __name__ == "__main__":
    unittest.main()
