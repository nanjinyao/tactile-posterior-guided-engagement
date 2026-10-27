from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestM8NoiseRobustnessArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_pre_hardware_noise_audit_is_paired_and_bounded(self) -> None:
        report = json.loads(
            (
                self.root
                / "outputs/current/evaluations/m8_pre_hardware_noise_audit/report.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["design"]["poses"], 36)
        self.assertEqual(report["probe_hash_match_rate"], 1.0)
        self.assertIn("engineering prior", report["claim_boundary"])
        self.assertEqual(report["methods"]["proposed_posterior"]["successes"], 36)
        self.assertEqual(report["methods"]["baseline_a"]["successes"], 35)
        self.assertEqual(report["methods"]["baseline_b"]["successes"], 29)
        paired = report["paired_proposed_posterior_vs_baseline_b"]
        self.assertEqual(
            (paired["both_success"], paired["left_only_success"], paired["right_only_success"], paired["both_failed"]),
            (29, 7, 0, 0),
        )
        self.assertAlmostEqual(paired["mcnemar_exact_two_sided_p"], 0.015625)
        proposed = report["methods"]["proposed_posterior"]
        self.assertGreater(proposed["transport_median_mm"], 1.0)
        self.assertLess(proposed["local_radius_median_mm"], 2.5)

    def test_failure_video_is_audit_selected_not_scripted(self) -> None:
        manifest_path = (
            self.root
            / "outputs/current/diagnostics/historical_comparisons/robustness_noise/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_audit_trial"], "sobol_002")
        self.assertIn("succeeds only after retry", manifest["selection_rule"])
        self.assertTrue(manifest["fairness"]["proposed_and_b_same_noisy_fingerprint"])
        self.assertIn("all three attempts", manifest["fairness"]["baseline_b_video_scope"])
        self.assertGreater(manifest["fairness"]["baseline_b_rendered_reset_duration_s"], 7.0)
        self.assertTrue(manifest["fairness"]["baseline_b_audit_reset_motion_time_excluded"])
        proposed = manifest["results"]["proposed"]
        direct = manifest["results"]["direct"]
        self.assertEqual(proposed["fingerprint_sha256"], direct["fingerprint_sha256"])
        self.assertTrue(proposed["bottom_reached"])
        self.assertTrue(direct["inner_wall_confirmed"])
        self.assertEqual(len(manifest["baseline_b_attempt_results"]), 3)
        self.assertEqual(
            [item["inner_wall_confirmed"] for item in manifest["baseline_b_attempt_results"]],
            [False, False, True],
        )
        self.assertTrue(proposed["transport_allowed"])
        self.assertFalse(proposed["legacy_transport_gate"])
        self.assertAlmostEqual(
            proposed["transport_distance_mm"],
            manifest["continuous_transport"]["distance_mm"],
        )
        self.assertGreater(proposed["transport_distance_mm"], 0.9)
        self.assertLess(proposed["transport_distance_mm"], 1.1)
        self.assertEqual(proposed["transport_policy"], "continuous_risk_bounded")
        self.assertLess(proposed["spiral_max_radius_mm"], 2.7)
        self.assertFalse(manifest["baseline_a_terminal_diagnostic"]["hard_stop"])
        self.assertTrue(manifest["results"]["outward"]["bottom_reached"])
        self.assertIn("not measurements", manifest["claim_boundary"])
        self.assertTrue((self.root / manifest["keyframe"]).is_file())
        archived_traces = [manifest["traces"]["proposed"], manifest["traces"]["baseline_a"]]
        archived_traces += manifest["traces"]["baseline_b_attempts"]
        for relative in archived_traces:
            self.assertFalse((self.root / relative).exists(), relative)
        archived_videos = list(manifest["videos"].values()) + [
            manifest["comparison_video"]
        ] + manifest["baseline_b_attempt_videos"] + manifest["baseline_b_reset_videos"]
        for relative in archived_videos:
            self.assertFalse((self.root / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
