import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "datasets" / "p0b_m8_observability"


class TestP0bObservabilityArtifacts(unittest.TestCase):
    def test_p0b_is_complete_and_claim_bounded(self) -> None:
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((OUT / "observability_report.json").read_text(encoding="utf-8"))

        self.assertEqual(progress["total"], 360)
        self.assertEqual(progress["completed"], 360)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(manifest["tasks"], 360)
        self.assertEqual(len(list((OUT / "trials").glob("*.npz"))), 360)
        self.assertEqual(report["successful_trials"], 360)

        shape = report["shape"]
        combined = report["shape_plus_absolute"]
        self.assertEqual(shape["base_pose_groups"], 120)
        self.assertEqual(shape["fisher_interior_points"], 54)
        self.assertGreater(shape["cv_state_accuracy"], 0.85)
        self.assertLessEqual(shape["cv_angle_error_p90_deg"], 15.0)
        self.assertGreater(combined["cv_state_accuracy"], shape["cv_state_accuracy"])
        self.assertLess(combined["action_collision_rate"], shape["action_collision_rate"])
        self.assertIn("not a trained CNN", report["claim_boundary"])
        self.assertIn("not a", report["claim_boundary"])

        example = sorted((OUT / "trials").glob("*.npz"))[0]
        with np.load(example, allow_pickle=False) as data:
            self.assertEqual(data["raw_scan"].shape, (1733, 24))
            self.assertEqual(data["phase_sequence"].shape, (22, 360))


if __name__ == "__main__":
    unittest.main()
