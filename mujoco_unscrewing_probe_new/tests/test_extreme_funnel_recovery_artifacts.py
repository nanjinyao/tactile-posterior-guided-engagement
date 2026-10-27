from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "outputs/current/active_comparison/funnel_extreme_recovery/manifest.json"
)


class TestExtremeFunnelRecoveryArtifacts(unittest.TestCase):
    def test_paired_severe_fault_has_truthful_terminal_outcomes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        scenario = manifest["scenario"]
        self.assertEqual(scenario["post_capture_tilt_disturbance_deg"], 8.0)
        self.assertEqual(scenario["post_capture_lateral_disturbance_m"], 0.0006)
        self.assertTrue(scenario["post_capture_disturbance_persistent"])

        legacy = manifest["results"]["no_physical_funnel"]
        guard = manifest["results"]["tip_rcc_funnel_recovery"]
        self.assertEqual(legacy["result"], "bottom_not_verified")
        self.assertFalse(legacy["hard_stop"])
        self.assertFalse(legacy["bottom_reached"])
        self.assertGreater(legacy["max_tilt_deg"], 8.0)
        self.assertFalse(legacy["terminal_outcome_hard_stop"])
        self.assertTrue(legacy["post_terminal_event_display_new_hard_stop"])
        self.assertGreater(legacy["max_raw_physical_force_n"], 20.0)
        self.assertGreater(legacy["max_raw_physical_torque_mnm"], 60.0)

        self.assertEqual(guard["result"], "funnel_safe_abort")
        self.assertFalse(guard["hard_stop"])
        self.assertFalse(guard["bottom_reached"])
        self.assertEqual(guard["funnel_abort_reason"], "cad_depth_overshoot")
        self.assertEqual(guard["funnel_recovery_count"], 0)
        self.assertFalse(guard["post_capture_disturbance_recovered"])
        self.assertEqual(
            guard["post_capture_disturbance_reason"], "cad_depth_overshoot"
        )
        self.assertLess(guard["max_raw_physical_force_n"], 20.0)
        self.assertLess(guard["max_raw_physical_torque_mnm"], 60.0)
        record = guard["post_release_requalification_records"][0]
        self.assertTrue(record["skipped_due_to_existing_terminal_outcome"])
        self.assertEqual(record["preexisting_terminal_reason"], "cad_depth_overshoot")
        self.assertIsNone(record["failure_action"])

    def test_video_is_active_and_claim_is_bounded(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertIn("deliberately severe synthetic residual pose error", manifest["claim_boundary"])
        self.assertIn("unrecoverable terminal", manifest["claim_boundary"])
        self.assertIn("inactive display step", manifest["claim_boundary"])
        self.assertIn("does not exercise the full", manifest["claim_boundary"])
        self.assertAlmostEqual(
            manifest["derived_geometry"]["deep_tilt_scale_deg"],
            1.432096,
            places=5,
        )
        self.assertTrue((ROOT / manifest["videos"]["comparison"]).is_file())
        self.assertTrue((ROOT / manifest["keyframe"]).is_file())
        self.assertFalse(
            (
                ROOT
                / "outputs/current/active_comparison/funnel_extreme_recovery/01_no_physical_funnel.mp4"
            ).exists()
        )
        self.assertFalse(
            (
                ROOT
                / "outputs/current/active_comparison/funnel_extreme_recovery/02_tip_rcc_funnel_recovery.mp4"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
