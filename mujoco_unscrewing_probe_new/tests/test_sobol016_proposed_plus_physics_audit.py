from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "outputs/current/diagnostics/m8_sobol016_proposed_plus_physics_audit"
    / "report.json"
)


class Sobol016ProposedPlusPhysicsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_shortcut_is_inaccurate_but_short_and_radially_helpful(self) -> None:
        shortcut = self.report["shortcut"]
        self.assertGreater(shortcut["offline_control_direction_error_deg"], 40.0)
        self.assertLess(shortcut["first_executed_distance_mm"], 1.0)
        self.assertLess(
            shortcut["derived_endpoint_radius_mm"], shortcut["initial_radius_mm"]
        )
        self.assertFalse(shortcut["captured_during_shortcut"])

    def test_mechanism_is_coverage_not_entropy_convergence(self) -> None:
        coverage = self.report["task_aware_coverage"]
        self.assertFalse(coverage["entropy_converged"])
        self.assertGreater(
            coverage["entropy_final_nats"], coverage["entropy_initial_nats"]
        )
        self.assertGreater(
            coverage["credible_radius_final_mm"],
            coverage["credible_radius_initial_mm"],
        )
        self.assertLess(
            coverage["measured_tip_xy_path_mm"],
            coverage["commanded_nominal_segment_length_sum_mm"],
        )

    def test_capture_is_sensor_only_and_terminal_is_strict(self) -> None:
        capture = self.report["capture_and_terminal"]
        verdict = self.report["verdict"]
        self.assertTrue(capture["online_capture_sensor_only"])
        self.assertFalse(verdict["online_true_center_or_contact_tags_used"])
        self.assertEqual(capture["capture_probe"], 5)
        self.assertLess(capture["time_to_inner_wall_s"], capture["time_to_bottom_s"])
        self.assertFalse(capture["hard_stop"])

    def test_bilingual_figures_exist(self) -> None:
        for item in self.report["figures"].values():
            path = ROOT / item["path"]
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
