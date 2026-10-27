import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current" / "datasets" / "p0a_m8_symmetry"


class TestP0aSymmetryArtifacts(unittest.TestCase):
    def test_p0a_is_complete_traceable_and_bounded(self) -> None:
        progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((OUT / "symmetry_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(progress["total"], 72)
        self.assertEqual(progress["completed"], 72)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(manifest["tasks"], 72)
        self.assertEqual(len(list((OUT / "trials").glob("*.npz"))), 72)

        self.assertEqual(audit["complete_base_groups"], 12)
        self.assertEqual(audit["comparisons"], 60)
        self.assertGreater(audit["mean_primary_shape_correlation"], 0.90)
        self.assertGreater(audit["p05_primary_shape_correlation"], 0.80)
        self.assertIn("does not prove hardware symmetry", audit["decision_boundary"])

        example = sorted((OUT / "trials").glob("*.npz"))[0]
        with np.load(example, allow_pickle=False) as data:
            self.assertEqual(data["raw_scan"].shape, (1733, 24))
            self.assertEqual(data["phase_sequence"].shape, (22, 360))


if __name__ == "__main__":
    unittest.main()
