from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestSearchComparisonArtifacts(unittest.TestCase):
    def test_same_start_comparison_is_traceable_and_bounded(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "outputs/current/reports/m8_outer_partial_search_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        fairness = payload["fairness_contract"]
        self.assertEqual(fairness["same_initial_xy_mm"], [4.0, 0.0])
        self.assertEqual(fairness["same_target_down_force_n"], 3.0)
        self.assertEqual(fairness["same_spiral_pitch_mm_per_rev"], 0.4)
        self.assertEqual(fairness["same_xy_speed_mm_s"], 2.25)
        self.assertEqual(fairness["same_active_time_budget_s"], 70.0)
        self.assertTrue(fairness["proposed_probe_cost_included"])
        self.assertIn("CAD capture region", fairness["same_endpoint"])

        proposed = payload["proposed"]
        baseline = payload["baseline"]
        self.assertTrue(proposed["inner_wall_confirmed"])
        self.assertEqual(proposed["result"], "inner_wall_gate_confirmed")
        self.assertFalse(proposed["hard_stop"])
        self.assertLess(proposed["time_to_inner_wall_s"], baseline["time_to_inner_wall_s"])
        self.assertLess(proposed["actual_xy_path_mm"], baseline["actual_xy_path_mm"])
        self.assertTrue(baseline["inner_wall_confirmed"])
        self.assertEqual(baseline["result"], "inner_wall_gate_confirmed")
        self.assertFalse(baseline["hard_stop"])
        self.assertLess(baseline["total_active_time_s"], 70.0)
        self.assertEqual(baseline["display_fast_forward_speed"], 8.0)
        self.assertIsNotNone(baseline["display_fast_forward_start_frame"])

        video_dir = root / "outputs/current/media/videos/comparisons"
        self.assertEqual(list(video_dir.glob("*.mp4")), [])

        for relative in payload["keyframes"].values():
            path = root / relative
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 50_000)

        boundary = payload["comparison"]["wear_comparison_boundary"]
        self.assertIn("same simulation capture evaluator", boundary)
        self.assertGreater(payload["comparison"]["xy_path_reduction_pct"], 0.0)
        self.assertEqual(
            payload["comparison"]["display_schedule"]["after_proposed_completion"],
            "baseline 8x; proposed result held",
        )
        self.assertIn(
            "Xz/Mx/My polar fingerprint",
            payload["comparison"]["display_schedule"]["proposed_upper_right"],
        )
        self.assertIn("not learned", payload["claim_boundary"])

    def test_frozen_p1v3_full_closed_loop_comparison_is_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "outputs/current/reports/m8_outer_partial_full_closed_loop_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            payload["model_root"], "outputs/current/models/p1v3_m8_direction_models"
        )
        self.assertEqual(len(payload["preserved_originals"]), 3)
        for relative in payload["preserved_originals"]:
            self.assertFalse((root / relative).exists())

        assumed = payload["methods"]["assumed"]
        learned = payload["methods"]["learned"]
        baseline = payload["methods"]["baseline"]
        self.assertEqual(assumed["direction_source"], "assumed_profile")
        self.assertEqual(learned["direction_source"], "three_seed_ensemble")
        self.assertEqual(baseline["direction_source"], "none")
        self.assertTrue(learned["transport_allowed"])
        self.assertLessEqual(learned["aps_set_size"], 3)
        self.assertLessEqual(learned["direction_error_deg"], 7.5)
        self.assertAlmostEqual(learned["control_direction_deg"], 180.0, delta=7.5)
        self.assertLess(learned["transport_distance_mm"], assumed["transport_distance_mm"])
        self.assertGreater(learned["spiral_max_radius_mm"], assumed["spiral_max_radius_mm"])
        for result in (assumed, learned, baseline):
            self.assertTrue(result["inner_wall_confirmed"])
            self.assertTrue(result["bottom_reached"])
            self.assertEqual(result["result"], "cad_bottom_verified")
            self.assertFalse(result["hard_stop"])
        self.assertLess(learned["time_to_bottom_s"], baseline["time_to_bottom_s"])
        self.assertLess(learned["actual_xy_path_mm"], baseline["actual_xy_path_mm"])

        for relative in payload["keyframes"].values():
            self.assertGreater((root / relative).stat().st_size, 50_000)
        self.assertIn("privileged MuJoCo contact tags", payload["claim_boundary"])

    def test_p2c_soft_cad_fusion_is_auditable_and_preserves_old_videos(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                root
                / "outputs/current/reports/m8_outer_partial_cad_fused_full_closed_loop_comparison.json"
            ).read_text(encoding="utf-8")
        )
        learned = payload["methods"]["learned"]
        baseline = payload["methods"]["baseline"]
        self.assertEqual(learned["top_state"], "double_overhang")
        self.assertGreater(learned["state_probabilities"]["double_overhang"], 0.99)
        self.assertEqual(
            learned["posterior_model"],
            "synthetic_visual_gaussian_x_ft_direction_and_state_x_soft_cad_radial_annuli",
        )
        self.assertAlmostEqual(
            learned["cad_radial_boundaries_mm"]["full_outer_support"],
            3.0358983848622447,
        )
        self.assertGreater(learned["posterior_radial_quantiles_mm"]["q10"], 2.0)
        self.assertAlmostEqual(learned["transport_distance_mm"], 2.880447076387683)
        self.assertEqual(learned["transport_policy"], "continuous_risk_bounded")
        self.assertGreater(learned["transport_direction_quality"], 0.99)
        self.assertEqual(learned["transport_reverse_probability"], 0.0)
        self.assertLess(learned["spiral_max_radius_mm"], 2.0)
        self.assertTrue(learned["bottom_reached"])
        self.assertFalse(learned["hard_stop"])
        self.assertLess(learned["time_to_bottom_s"], baseline["time_to_bottom_s"])
        self.assertLess(learned["actual_xy_path_mm"], baseline["actual_xy_path_mm"])
        self.assertLess(learned["max_force_n"], 10.0)
        self.assertLess(learned["max_filtered_force_n"], 5.0)
        self.assertFalse(learned["global_outward_fallback_used"])
        self.assertIn("does not use the true center", payload["claim_boundary"])
        for relative in payload["preserved_originals"]:
            self.assertFalse((root / relative).exists())
        for relative in payload["keyframes"].values():
            self.assertGreater((root / relative).stat().st_size, 50_000)

    def test_active_three_method_comparison_uses_shared_direction_evidence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "outputs/current/diagnostics/historical_comparisons/three_method_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        fairness = payload["fairness"]
        self.assertTrue(fairness["proposed_and_direct_same_probe"])
        self.assertEqual(
            fairness["proposed_and_direct_same_direction_estimator"],
            "outputs/current/models/p1v3_m8_direction_models",
        )
        self.assertFalse(fairness["direct_uses_true_center_or_distance"])
        proposed = payload["results"]["proposed"]
        direct = payload["results"]["direct"]
        outward = payload["results"]["outward"]
        self.assertAlmostEqual(
            proposed["control_direction_deg"], direct["control_direction_deg"]
        )
        self.assertEqual(
            direct["direction_source"], "shared_three_seed_ensemble_direction_only"
        )
        self.assertTrue(all(item["bottom_reached"] for item in (proposed, direct, outward)))
        self.assertLess(direct["time_to_bottom_s"], proposed["time_to_bottom_s"])
        self.assertLess(direct["actual_xy_path_mm"], proposed["actual_xy_path_mm"])
        self.assertLess(direct["max_force_n"], proposed["max_force_n"])
        self.assertIn("direct baseline wins this case", payload["claim_boundary"])
        for relative in payload["keyframes"].values():
            self.assertGreater((root / relative).stat().st_size, 50_000)
        diagnostic = root / payload["diagnostic_not_active_baseline"]
        self.assertTrue((diagnostic / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
