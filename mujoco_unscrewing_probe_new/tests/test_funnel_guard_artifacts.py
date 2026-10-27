from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs/current/evaluations/m8_funnel_guard_ablation"
MANIFEST = ROOT / "outputs/current/active_comparison/funnel_guard/manifest.json"


class TestFunnelGuardArtifacts(unittest.TestCase):
    def test_ablation_is_paired_and_post_capture_only(self) -> None:
        report = json.loads((EVALUATION / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["design"]["paired_tasks"], 12)
        self.assertIn("identical ordinary outward search", report["design"]["search_policy"])
        self.assertEqual(len(list((EVALUATION / "trials").glob("funnel_*.json"))), 12)
        self.assertIn("not hardware evidence", report["claim_boundary"])

    def test_guard_reports_safety_tradeoff_without_success_overclaim(self) -> None:
        report = json.loads((EVALUATION / "report.json").read_text(encoding="utf-8"))
        legacy = report["methods"]["legacy_insertion"]
        guarded = report["methods"]["funnel_guard"]
        self.assertLessEqual(guarded["hard_stops"], legacy["hard_stops"])
        self.assertLess(guarded["max_cad_depth_overrun_mm"], legacy["max_cad_depth_overrun_mm"])
        self.assertGreater(guarded["cad_depth_safe_aborts"], 0)
        self.assertLessEqual(guarded["bottom_verified"], legacy["bottom_verified"])

    def test_video_case_is_audit_selected_and_claim_bounded(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertIn("first sorted paired audit case", manifest["selection"])
        legacy = manifest["results"]["legacy_insertion"]
        guarded = manifest["results"]["funnel_guard"]
        self.assertEqual(legacy["result"], "hard_stop")
        self.assertTrue(legacy["hard_stop"])
        self.assertEqual(manifest["results"]["funnel_guard"]["result"], "funnel_safe_abort")
        self.assertFalse(guarded["hard_stop"])
        self.assertLess(guarded["max_force_n"], legacy["max_force_n"])
        self.assertIn("not a success-rate gain", manifest["claim_boundary"])
        self.assertTrue((ROOT / manifest["videos"]["comparison"]).is_file())
        self.assertEqual(manifest["archived_component_video_keys"], ["legacy", "guard"])
        visualization = manifest["visualization"]
        self.assertTrue(visualization["synchronized_trace_overlay"])
        self.assertEqual(visualization["force_hard_limit_n"], 20.0)
        self.assertEqual(visualization["torque_hard_limit_mnm"], 60.0)
        self.assertEqual(visualization["slow_motion_factor"], 0.5)
        self.assertTrue(visualization["keyframe_is_dynamic_divergence_frame"])
        self.assertTrue((ROOT / manifest["keyframe"]).is_file())


if __name__ == "__main__":
    unittest.main()
