from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "outputs/current/evaluations/m8_four_method_sensor_only_audit"
CUE = ROOT / "outputs/current/evaluations/m8_entropy_contact_cue_calibration"


class TestM8ClosedLoopScientificAudit(unittest.TestCase):
    def test_four_method_audit_is_complete_fair_and_sensor_only_online(self) -> None:
        report = json.loads((POLICY / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["design"]["poses"], 36)
        self.assertFalse(
            report["design"]["online_capture_uses_privileged_mujoco_tags"]
        )
        self.assertTrue(report["design"]["offline_success_scoring_uses_mujoco_geometry"])
        self.assertEqual(report["probe_hash_match_rate"], 1.0)
        self.assertEqual(
            tuple(report["methods"][name]["successes"] for name in (
                "proposed_posterior",
                "baseline_a",
                "baseline_b",
                "baseline_c_random",
            )),
            (34, 30, 28, 19),
        )
        self.assertEqual(report["methods"]["baseline_b"]["first_attempt_successes"], 25)
        self.assertEqual(report["methods"]["proposed_posterior"]["global_fallback_count"], 4)
        random = report["methods"]["baseline_c_random"]
        self.assertEqual(random["hard_stops"], 1)
        self.assertGreater(random["contact_path_all_trials_median_mm"], 80.0)
        self.assertEqual(
            report["paired_proposed_posterior_vs_baseline_c_random"],
            {
                "both_failed": 1,
                "both_success": 18,
                "left_only_success": 16,
                "mcnemar_exact_two_sided_p": 0.000274658203125,
                "right_only_success": 1,
            },
        )

    def test_trial_contract_keeps_privileged_truth_offline(self) -> None:
        trials = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((POLICY / "trials").glob("sobol_*.json"))
        ]
        self.assertEqual(len(trials), 36)
        random_seeds = set()
        for trial in trials:
            self.assertTrue(trial["fairness"]["first_probe_hash_match"])
            proposed = trial["methods"]["proposed_posterior"]
            direct = trial["methods"]["baseline_b"]
            random = trial["methods"]["baseline_c_random"]
            self.assertTrue(proposed["online_capture_sensor_only"])
            self.assertTrue(random["online_capture_sensor_only"])
            self.assertEqual(
                proposed["fingerprint_sha256"], direct["first_fingerprint_sha256"]
            )
            self.assertTrue(
                all(item["online_capture_sensor_only"] for item in direct["attempt_results"])
            )
            self.assertIsInstance(proposed["offline_capture_validated"], bool)
            self.assertIsInstance(random["offline_capture_validated"], bool)
            self.assertIsNone(random["fingerprint_sha256"])
            self.assertGreater(random["random_search_path_budget_mm"], 0.0)
            self.assertGreater(random["random_search_waypoints"], 1)
            random_seeds.add(random["random_search_seed"])
        self.assertEqual(len(random_seeds), 36)

    def test_signed_torque_cue_resolves_inner_outer_edge_ambiguity(self) -> None:
        report = json.loads((CUE / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["all_inner_outer_magnitudes_ambiguous"])
        self.assertTrue(report["all_signed_categories_distinct"])
        self.assertEqual(report["hard_stops"], 0)
        self.assertEqual(len(report["records"]), 12)
        self.assertEqual(len(report["inner_outer_pairs"]), 4)

    def test_state_cad_is_soft_and_action_relevant_not_a_distance_head(self) -> None:
        report = json.loads(
            (
                ROOT
                / "outputs/current/evaluations/m8_state_cad_fusion_audit/report.json"
            ).read_text(encoding="utf-8")
        )
        direction_only = report["methods"]["direction_visual"]
        state_cad = report["methods"]["state_cad_soft"]
        self.assertGreaterEqual(state_cad["coverage_rate"], direction_only["coverage_rate"])
        self.assertLess(
            state_cad["post_transport_true_residual_median_mm"],
            direction_only["post_transport_true_residual_median_mm"],
        )
        self.assertEqual(state_cad["reverse_move_rate"], 0.0)
        self.assertIn("not dynamic insertion", report["claim_boundary"])

    def test_machine_readable_audit_preserves_task_boundary(self) -> None:
        payload = json.loads(
            (
                ROOT / "outputs/current/reports/closed_loop_scientific_audit.json"
            ).read_text(encoding="utf-8")
        )
        boundary = payload["online_offline_boundary"]
        self.assertFalse(boundary["online_controller_uses_true_center"])
        self.assertFalse(boundary["online_controller_uses_mujoco_contact_tags"])
        self.assertTrue(boundary["offline_scoring_uses_mujoco_geometry"])
        self.assertTrue(
            boundary["terminal_cad_depth_is_geometric_engagement_not_unscrewing"]
        )
        terminal = payload["terminal_interface"]
        self.assertFalse(terminal["online_true_center_or_contact_tags"])
        self.assertEqual(
            terminal["completion_channels"],
            ["bottom_force_dwell", "cad_depth_low_wrench_dwell"],
        )
        self.assertTrue(terminal["force_channel_requires_expected_axial_sign"])
        self.assertTrue(terminal["both_channels_bound_lateral_force_and_tilt_torque"])
        self.assertTrue(terminal["strict_successes_exercised_low_wrench_branch_only"])
        self.assertFalse(terminal["signed_axial_force_branch_batch_exercised"])
        strict = payload["strict_m8_complete_loop"]
        self.assertEqual(strict["design"]["poses"], 24)
        self.assertTrue(strict["design"]["strict_outcome_frozen_before_inactive_display_tail"])
        self.assertEqual(
            {
                name: values["strict_bottom_successes"]
                for name, values in strict["methods"].items()
            },
            {
                "direct_first": 15,
                "entropy_grid": 22,
                "oracle": 22,
                "outward": 19,
                "proposed": 19,
                "proposed_plus": 19,
                "random": 10,
            },
        )
        for counts in terminal["strict_terminal_mode_counts"].values():
            self.assertNotIn("bottom_force_dwell", counts)
            self.assertGreaterEqual(counts.get("cad_depth_low_wrench_dwell", 0), 0)
        online_offline = payload["online_offline_boundary"]
        self.assertTrue(online_offline["observation_model_advances_once_per_physics_step"])
        self.assertTrue(online_offline["raw_physical_wrench_drives_fast_hard_stop"])
        self.assertTrue(online_offline["terminal_outcome_frozen_before_inactive_display_tail"])
        self.assertIn("M10", payload["task"]["zero_shot_direction_interface_sizes"])
        self.assertIn("M12", payload["task"]["zero_shot_direction_interface_sizes"])
        self.assertIn("M10", payload["task"]["full_loop_not_validated_sizes"])
        self.assertIn("M12", payload["task"]["full_loop_not_validated_sizes"])
        self.assertTrue(payload["cross_size_direction_interface"]["all_direction_gates_pass"])
        self.assertFalse(payload["cross_size_full_loop_S2"]["all_size_gates_pass"])
        self.assertEqual(
            payload["task"]["cross_size_full_loop_gate"],
            "failed for both M10 and M12",
        )

    def test_four_method_video_manifest_is_explanatory_and_fair(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "outputs/current/active_comparison/four_methods/manifest.json"
            ).read_text(encoding="utf-8")
        )
        fairness = manifest["fairness"]
        self.assertTrue(fairness["same_initial_pose_contact_nuisance_and_sensor_prior"])
        self.assertTrue(fairness["proposed_and_b_identical_first_fingerprint"])
        self.assertTrue(fairness["baseline_b_all_attempts_visible"])
        self.assertTrue(fairness["shared_capture_guard_and_terminal_verifier"])
        self.assertFalse(fairness["online_true_center_or_contact_tags"])
        self.assertIsInstance(fairness["baseline_c_seed_fixed_before_rollout"], int)
        for result in manifest["results"].values():
            self.assertTrue(result["inner_wall_confirmed"])
            self.assertTrue(result["online_capture_sensor_only"])
        random = manifest["results"]["baseline_c_random"]
        self.assertIsNotNone(random["random_search_seed"])
        self.assertTrue(random["offline_capture_validated"])
        self.assertTrue(random["bottom_reached"])
        self.assertEqual(random["result"], "cad_bottom_verified")
        self.assertTrue((ROOT / manifest["keyframe"]).is_file())
        self.assertTrue((ROOT / manifest["comparison"]).is_file())
        self.assertIn("36-pose paired random-baseline audit", manifest["claim_boundary"])

        source = json.loads(
            (
                ROOT
                / "outputs/current/diagnostics/historical_comparisons/robustness_noise/manifest.json"
            ).read_text(encoding="utf-8")
        )
        attempts = source["baseline_b_attempt_results"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            [item["inner_wall_confirmed"] for item in attempts],
            [False, False, True],
        )
        self.assertEqual(
            [item["hard_stop"] for item in attempts],
            [False, False, False],
        )
        self.assertEqual(len(source["baseline_b_attempt_videos"]), 3)
        self.assertEqual(len(source["baseline_b_reset_videos"]), 2)

    def test_clear_fingerprint_video_is_frozen_complementary_case(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "outputs/current/active_comparison/four_methods_clear_fingerprint/manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["selected_audit_trial"], "sobol_019")
        self.assertEqual(manifest["full_closed_loop_preflight"]["sobol_019"]["decision"], "select")
        self.assertGreater(manifest["selection_diagnostics"]["proposed_transport_mm"], 2.8)
        self.assertLess(manifest["selection_diagnostics"]["direction_error_deg"], 3.0)
        self.assertGreater(manifest["selection_diagnostics"]["direct_ray_margin_mm"], 0.0)
        fairness = manifest["fairness"]
        self.assertTrue(fairness["proposed_and_b_identical_first_fingerprint"])
        self.assertTrue(fairness["shared_capture_guard_and_terminal_verifier"])
        self.assertFalse(fairness["online_true_center_or_contact_tags"])
        self.assertTrue(fairness["component_videos_removed_after_composition"])
        self.assertEqual(
            manifest["results"]["proposed"]["fingerprint_sha256"],
            manifest["results"]["direct"]["fingerprint_sha256"],
        )
        for key in ("proposed", "outward", "direct"):
            self.assertEqual(manifest["results"][key]["result"], "cad_bottom_verified")
            self.assertTrue(manifest["results"][key]["bottom_reached"])
            self.assertFalse(manifest["results"][key]["hard_stop"])
        random = manifest["results"]["random"]
        self.assertEqual(random["result"], "search_radius_exhausted")
        self.assertFalse(random["inner_wall_confirmed"])
        self.assertFalse(random["bottom_reached"])
        self.assertFalse(random["hard_stop"])
        self.assertTrue((ROOT / manifest["keyframe"]).is_file())
        self.assertTrue((ROOT / manifest["comparison"]).is_file())
        self.assertIn("not an additional independent benchmark", manifest["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
