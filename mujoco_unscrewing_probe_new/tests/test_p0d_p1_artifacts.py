import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
P0D = ROOT / "outputs" / "current" / "datasets" / "p0d_m8_sector_anchors"
P1 = ROOT / "outputs" / "current" / "models" / "p1_m8_direction_models"


class TestP0dP1Artifacts(unittest.TestCase):
    def test_p0d_is_complete_paired_and_claim_bounded(self) -> None:
        progress = json.loads((P0D / "progress.json").read_text(encoding="utf-8"))
        audit = json.loads((P0D / "sector_anchor_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["completed"], 324)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(len(list((P0D / "trials").glob("*.npz"))), 324)
        self.assertEqual(audit["complete_orbits"], 54)
        self.assertEqual(audit["comparisons"], 270)
        self.assertGreater(audit["mean_primary_shape_correlation"], 0.90)
        self.assertIn("not a trained posterior", audit["claim_boundary"])

    def test_p1_ensemble_has_fixed_split_evidence_and_safe_boundary(self) -> None:
        report = json.loads((P1 / "ensemble_metrics.json").read_text(encoding="utf-8"))
        contract = json.loads(
            (P1 / "ensemble_interface_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["seeds"], [20260710, 20260711, 20260712])
        for seed in report["seeds"]:
            self.assertTrue(
                (P1 / "seeds" / f"seed_{seed}" / "circular_cnn_best.pt").is_file()
            )
        self.assertGreaterEqual(report["metrics"]["p0d_test"]["angle_within_7p5_rate"], 0.95)
        self.assertGreaterEqual(
            report["metrics"]["p0a_full_angle"]["angle_within_7p5_rate"], 0.90
        )
        self.assertEqual(
            report["metrics"]["p0d_test"]["gate_catastrophic_gt90_rate"], 0.0
        )
        self.assertIn("diagnostic only", contract["state_role"])
        self.assertIn("not a real-robot posterior", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
