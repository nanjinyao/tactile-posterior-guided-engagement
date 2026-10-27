from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "outputs/current/evaluations/m8_entropy_grid_coverage_certificate/report.json"
)


@unittest.skipUnless(REPORT.is_file(), "coverage certificate not generated")
class TestM8EntropyGridCoverageCertificate(unittest.TestCase):
    def test_finite_disk_is_certified_without_online_truth(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        certificate = report["certificate"]
        geometry = report["geometry"]
        self.assertTrue(certificate["certified"])
        self.assertTrue(certificate["complete_required_lattice"])
        self.assertLessEqual(
            certificate["analytic_worst_case_gap_m"],
            0.001 * geometry["effective_capture_radius_mm"],
        )
        self.assertLessEqual(
            certificate["observed_worst_gap_m"],
            0.001 * geometry["effective_capture_radius_mm"],
        )
        self.assertGreater(
            geometry["action_nodes"], geometry["clipped_action_nodes_before_padding"]
        )
        self.assertFalse(
            report["online_truth_boundary"]["true_center_used_for_action_selection"]
        )


if __name__ == "__main__":
    unittest.main()
