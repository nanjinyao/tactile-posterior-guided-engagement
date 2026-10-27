from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/current/evaluations/m8_model_m10_m12_zero_shot/report.json"


class TestCrossSizeZeroShotArtifacts(unittest.TestCase):
    def test_frozen_m8_model_has_a_bounded_cross_size_direction_audit(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["stage"],
            "M8_frozen_direction_model_M8_reference_M10_M12_zero_shot",
        )
        self.assertFalse(report["online_true_center_or_contact_tags"])
        self.assertTrue(report["all_direction_gates_pass"])
        self.assertEqual(set(report["sizes"]), {"m8", "m10", "m12"})
        for size in ("m8", "m10", "m12"):
            with self.subTest(size=size):
                summary = report["sizes"][size]["summary"]
                self.assertEqual(summary["trials"], 72)
                self.assertEqual(summary["successful_scans"], 72)
                self.assertEqual(summary["hard_stops"], 0)
                self.assertTrue(summary["direction_interface_gate_pass"])
                self.assertEqual(
                    set(summary["by_state"]),
                    {"supported_search", "double_overhang", "outer_partial"},
                )

    def test_report_does_not_promote_direction_transfer_to_full_loop_transfer(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        boundary = report["claim_boundary"].lower()
        self.assertIn("does not prove", boundary)
        self.assertIn("complete m10/m12 insertion", boundary)
        self.assertGreater(
            report["sizes"]["m12"]["summary"]["context_z_gt5_rate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
