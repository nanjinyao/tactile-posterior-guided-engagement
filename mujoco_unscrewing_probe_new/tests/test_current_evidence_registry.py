from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs/current"


class TestCurrentEvidenceRegistry(unittest.TestCase):
    def test_five_method_report_is_the_registered_distribution_source(self) -> None:
        retention = json.loads(
            (CURRENT / "VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        report_path = ROOT / retention["active_evidence"]["distribution_report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["experiment"], "M8_five_method_retry3_shared_R3_closed_loop")
        self.assertEqual(
            set(report["methods"]),
            {
                "proposed",
                "proposed_plus",
                "outward",
                "improved_direct_retry3",
                "jin_direct_retry3",
            },
        )
        self.assertEqual(report["methods"]["proposed"]["strict_bottom_successes"], 20)
        self.assertEqual(report["methods"]["proposed_plus"]["strict_bottom_successes"], 19)
        self.assertEqual(report["methods"]["outward"]["strict_bottom_successes"], 19)
        self.assertEqual(report["methods"]["improved_direct_retry3"]["strict_bottom_successes"], 17)
        self.assertEqual(report["methods"]["jin_direct_retry3"]["strict_bottom_successes"], 16)
        self.assertTrue(report["all_budget_extensions_zero"])
        self.assertTrue(report["all_simulator_reset_counts_zero"])

    def test_current_selected_media_use_retry3_r3_contract(self) -> None:
        base = CURRENT / "active_comparison/m8_retry3_selected_media"
        manifests = (
            base / "sobol_000/comparison_clear_retry3_2x2_manifest.json",
            base / "sobol_016/comparison_fuzzy_retry3_2x2_manifest.json",
            base / "sobol_005/comparison_improved4_vs_jin3_retry3_manifest.json",
        )
        for path in manifests:
            with self.subTest(path=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                contract = record["contract"]
                self.assertEqual(contract["direct_ray_attempts_max"], 3)
                self.assertTrue(contract["direct_retry_is_physical_same_rollout"])
                self.assertEqual(contract["shared_post_capture_recoveries_max"], 3)
                self.assertEqual(contract["simulator_resets"], 0)
                video = ROOT / record["artifacts"]["video"]["path"]
                self.assertTrue(video.is_file())

    def test_entropy_topview_and_funnel_boundaries_are_registered(self) -> None:
        retention = json.loads(
            (CURRENT / "VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        entropy = json.loads(
            (ROOT / retention["active_evidence"]["entropy_topview_manifest"])
            .read_text(encoding="utf-8")
        )
        self.assertEqual(entropy["finite_sets"]["selected_action_count"], 5)
        self.assertEqual(entropy["finite_sets"]["belief_hypotheses"], 1201)
        self.assertEqual(entropy["finite_sets"]["total_available_action_candidates"], 749)
        self.assertFalse(entropy["offline_explanation_boundary"]["true_center_used_online"])

        funnel = json.loads(
            (ROOT / retention["active_evidence"]["physical_funnel_report"])
            .read_text(encoding="utf-8")
        )
        self.assertEqual(funnel["methods"]["no_funnel"]["bottom_verified"], 2)
        self.assertEqual(funnel["methods"]["rcc_shield_r0"]["bottom_verified"], 0)
        self.assertEqual(funnel["methods"]["rcc_shield_r3"]["bottom_verified"], 9)
        self.assertEqual(funnel["methods"]["rcc_shield_r3"]["hard_stops"], 2)
        self.assertEqual(
            funnel["paired"]["r0_vs_r3"]["rcc_shield_r3_only_bottom"], 9
        )
        self.assertEqual(
            funnel["paired"]["r0_vs_r3"]["exact_two_sided_sign_p"],
            0.00390625,
        )
        self.assertIn("not a proof", funnel["claim_boundary"])

    def test_claim_docs_and_s4b_reports_are_synchronized(self) -> None:
        retention = json.loads(
            (CURRENT / "VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            retention["active_evidence"]["cross_size_s4b_development_report"],
            "outputs/current/diagnostics/cross_size_s4b_full_loop_development_v5/report.json",
        )
        self.assertEqual(
            retention["active_evidence"]["cross_size_s4b_untouched_report"],
            "outputs/current/evaluations/cross_size_s4b_full_loop_untouched_v1/report.json",
        )
        development = json.loads(
            (
                ROOT
                / retention["active_evidence"]["cross_size_s4b_development_report"]
            ).read_text(encoding="utf-8")
        )
        untouched = json.loads(
            (
                ROOT
                / retention["active_evidence"]["cross_size_s4b_untouched_report"]
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(development["all_S4_fine_tuned_size_gates_pass"])
        self.assertFalse(
            development["adaptation_necessity_supported_for_all_sizes"]
        )
        self.assertFalse(untouched["all_S4_fine_tuned_size_gates_pass"])
        self.assertFalse(untouched["adaptation_necessity_supported_for_all_sizes"])
        self.assertEqual(
            untouched["sizes"]["m10"]["methods"]["s4_fine_tuned"][
                "strict_bottom_successes"
            ],
            10,
        )
        self.assertEqual(
            untouched["sizes"]["m10"]["methods"]["s4_fine_tuned"]["hard_stops"],
            1,
        )
        self.assertEqual(
            untouched["sizes"]["m12"]["methods"]["s4_fine_tuned"][
                "strict_bottom_successes"
            ],
            10,
        )
        self.assertEqual(
            untouched["sizes"]["m12"]["methods"]["s4_fine_tuned"]["hard_stops"],
            0,
        )
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        boundary = (ROOT / "CLAIM_BOUNDARY.md").read_text(encoding="utf-8")
        current_readme = (CURRENT / "README.md").read_text(encoding="utf-8")
        retention_readme = (CURRENT / "VIDEO_RETENTION.md").read_text(encoding="utf-8")
        for text in (root_readme, boundary, current_readme, retention_readme):
            self.assertIn("m8_five_method_retry3_r3_closed_loop", text)
            self.assertNotIn("pending", text.lower())
        for text in (root_readme, boundary, current_readme):
            self.assertIn("m8_recoverable_funnel_r3_ablation", text)
            self.assertIn("cross_size_s4b_full_loop_untouched_v1", text)
        self.assertIn("mixed", boundary.lower())
        self.assertIn("or a guarantee", boundary.lower())
        self.assertIn("Fine-tuning necessity is also not identified", boundary)


if __name__ == "__main__":
    unittest.main()
