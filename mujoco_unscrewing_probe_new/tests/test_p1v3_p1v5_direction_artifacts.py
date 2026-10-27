import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TARGETED = ROOT / "outputs" / "current" / "datasets" / "p1v3_m8_targeted_direction_training"
MODEL = ROOT / "outputs" / "current" / "models" / "p1v3_m8_direction_models"
FINAL = ROOT / "outputs" / "current" / "evaluations" / "p1v5_m8_final_untouched_direction_audit"


class TestP1v3P1v5DirectionArtifacts(unittest.TestCase):
    def test_targeted_data_and_three_seed_model_are_complete(self) -> None:
        progress = json.loads((TARGETED / "progress.json").read_text(encoding="utf-8"))
        contract = json.loads(
            (TARGETED / "sampling_contract.json").read_text(encoding="utf-8")
        )
        interface = json.loads(
            (MODEL / "ensemble_interface_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(progress["completed"], 288)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(contract["training_use"], "allowed_after_grouped_split")
        self.assertEqual(len(interface["checkpoints"]), 3)
        self.assertEqual(interface["structured_transport_gate"]["max_aps_set_size"], 3)
        self.assertEqual(
            interface["structured_transport_gate"]["max_ensemble_top1_span_deg"], 7.5
        )
        self.assertNotIn("p1v2", " ".join(interface["checkpoints"]).lower())

    def test_final_untouched_audit_passes_narrow_direction_gate(self) -> None:
        progress = json.loads((FINAL / "progress.json").read_text(encoding="utf-8"))
        contract = json.loads((FINAL / "sampling_contract.json").read_text(encoding="utf-8"))
        report = json.loads(
            (FINAL / "independent_direction_report.json").read_text(encoding="utf-8")
        )
        metrics = report["metrics"]
        self.assertEqual(progress["completed"], 180)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(contract["training_use"], "forbidden")
        self.assertGreaterEqual(metrics["angle_within_7p5_rate"], 0.98)
        self.assertGreaterEqual(metrics["aps_coverage"], 0.98)
        self.assertEqual(metrics["catastrophic_gt90_count"], 0)
        self.assertGreaterEqual(metrics["transport_accepted_within_7p5_rate"], 0.99)
        self.assertIn("Forbidden for fitting", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
