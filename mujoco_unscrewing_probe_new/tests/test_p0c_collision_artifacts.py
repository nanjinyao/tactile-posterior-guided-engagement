import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "datasets" / "p0c_m8_collision_followup"


class TestP0cCollisionArtifacts(unittest.TestCase):
    def test_p0c_independent_hard_set_is_complete(self) -> None:
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((OUT / "collision_followup_report.json").read_text(encoding="utf-8"))

        self.assertEqual(progress["total"], 90)
        self.assertEqual(progress["completed"], 90)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(manifest["tasks"], 90)
        self.assertEqual(len(list((OUT / "trials").glob("*.npz"))), 90)

        combined = report["shape_plus_absolute"]
        self.assertEqual(combined["train_trials"], 360)
        self.assertEqual(combined["test_trials"], 90)
        self.assertEqual(combined["test_groups"], 18)
        self.assertGreater(combined["state_accuracy"], 0.90)
        self.assertGreater(combined["direction_acceptable_rate"], 0.98)
        self.assertLessEqual(combined["angle_error_p90_deg"], 7.5)
        self.assertIn("not a CNN result", report["claim_boundary"])

        example = sorted((OUT / "trials").glob("*.npz"))[0]
        with np.load(example, allow_pickle=False) as data:
            self.assertEqual(data["raw_scan"].shape, (1733, 24))
            self.assertEqual(data["phase_sequence"].shape, (22, 360))


if __name__ == "__main__":
    unittest.main()
