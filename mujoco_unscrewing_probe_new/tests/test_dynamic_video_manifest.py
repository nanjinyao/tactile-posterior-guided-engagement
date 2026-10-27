from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestDynamicVideoManifest(unittest.TestCase):
    def test_all_current_videos_are_dynamic_and_successful(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retention = json.loads(
            (root / "outputs/current/VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        manifest = json.loads((root / "outputs/current/reports/m8_sampling_video_manifest.json").read_text(encoding="utf-8"))
        records = {record["key"]: record for record in manifest["videos"]}
        self.assertEqual(set(records), {"V1", "V2", "V3", "V4", "V5", "V6a", "V6b", "V6c"})
        for key, record in records.items():
            with self.subTest(key=key):
                self.assertTrue(record["dynamic_physics"])
                self.assertTrue(record["admittance_controlled"])
                self.assertFalse(record["hard_stop"])
                self.assertFalse((root / record["path"]).exists())

        current_video_files = {
            path.relative_to(root).as_posix()
            for path in (root / "outputs/current").rglob("*.mp4")
        }
        active = set(retention["active_videos"])
        downgraded = set(retention["downgraded_videos"])
        retained = active | downgraded
        self.assertFalse(active & downgraded)
        self.assertEqual(current_video_files, retained)
        self.assertEqual(
            active,
            {
                "outputs/current/active_comparison/m8_retry3_selected_media/sobol_000/comparison_clear_retry3_2x2.mp4",
                "outputs/current/active_comparison/m8_retry3_selected_media/sobol_005/comparison_improved4_vs_jin3_retry3.mp4",
                "outputs/current/active_comparison/m8_retry3_selected_media/sobol_016/comparison_fuzzy_retry3_2x2.mp4",
                "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003/m8_strict_entropy_grid_topview_sobol_003.mp4",
                "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003_en/m8_strict_entropy_grid_topview_sobol_003_en.mp4",
            },
        )
        self.assertEqual(retention["historical_archive"]["members"], 43)
        for relative in retention["active_videos"]:
            path = root / relative
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 100_000)
        for relative in retention["downgraded_videos"]:
            path = root / relative
            self.assertTrue(path.is_file())
            self.assertNotIn(relative, active)

        evidence = retention["active_evidence"]
        for key in (
            "distribution_report",
            "physical_funnel_report",
            "entropy_topview_manifest",
            "entropy_topview_manifest_en",
            "claim_boundary",
        ):
            self.assertTrue((root / evidence[key]).is_file(), key)

        for key in ("V1", "V2", "V3"):
            self.assertTrue(records[key]["nominal_tip_xy_fixed"])
            self.assertLess(records[key]["max_tip_xy_deviation_m"], 0.00030)
            self.assertLess(records[key]["final_tip_xy_deviation_m"], 0.00010)
            self.assertLess(records[key]["max_orientation_tracking_error_deg"], 1.5)

        self.assertEqual(records["V4"]["result"], "inner_wall_capture_cue_latched")
        self.assertTrue(records["V4"]["upright_motion"])
        self.assertTrue(records["V4"]["inner_wall_captured"])
        self.assertEqual(records["V5"]["result"], "cad_bottom_verified")
        self.assertTrue(records["V5"]["bottom_reached"])
        self.assertTrue(records["V5"]["wall_gate_triggered"])
        for key in ("V6a", "V6b", "V6c"):
            self.assertEqual(records[key]["result"], "chamfer_depth_cue_latched")
            self.assertAlmostEqual(records[key]["outward_pitch_m_per_rev"], 0.0004)
            self.assertEqual(records[key]["spiral_phase_source"], "posterior_transport_direction_only")
            self.assertGreater(records[key]["min_tip_z_m"], -0.00060)


if __name__ == "__main__":
    unittest.main()
