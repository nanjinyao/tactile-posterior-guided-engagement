from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/current/evaluations/m8_fresh_24_task_aware_validation"


class TestTaskAwareFreshValidation(unittest.TestCase):
    def test_fresh_validation_is_complete_frozen_and_truth_bounded(self) -> None:
        report = json.loads((AUDIT / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["design"]["poses"], 24)
        self.assertEqual(report["design"]["source_seed"], 20260723)
        self.assertTrue(report["design"]["frozen_before_rollout"])
        self.assertFalse(report["design"]["online_true_center_or_contact_tags"])
        self.assertEqual(report["fingerprint_match_rate"], 1.0)
        self.assertEqual(len(list((AUDIT / "trials").glob("sobol_*.json"))), 24)

    def test_result_supports_a_tradeoff_not_superiority(self) -> None:
        report = json.loads((AUDIT / "report.json").read_text(encoding="utf-8"))
        methods = report["methods"]
        expected = {
            "proposed_plus_hybrid": 21,
            "proposed_posterior": 23,
            "baseline_a": 23,
            "baseline_b": 22,
            "baseline_c_entropy": 23,
        }
        self.assertEqual(
            {method: methods[method]["successes"] for method in expected}, expected
        )
        self.assertTrue(all(methods[method]["hard_stops"] == 0 for method in expected))
        pair = report["paired"]["proposed_plus_hybrid_vs_proposed_posterior"][
            "paired_efficiency"
        ]
        self.assertLess(pair["left_minus_right_actual_xy_path_mm_median"], 0.0)
        self.assertLess(pair["left_minus_right_wear_proxy_mj_median"], 0.0)
        self.assertGreater(pair["left_minus_right_time_to_inner_wall_s_median"], 0.0)
        self.assertGreater(
            methods["proposed_plus_hybrid"]["wear_proxy_median_mj"],
            methods["baseline_c_entropy"]["wear_proxy_median_mj"],
        )

    def test_proposed_plus_uses_direction_gate_and_no_local_spiral(self) -> None:
        for path in sorted((AUDIT / "trials").glob("sobol_*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))["methods"][
                "proposed_plus_hybrid"
            ]
            self.assertTrue(result["capture_direction_consistency_gate_enabled"])
            self.assertAlmostEqual(
                result["capture_reverse_projection_tolerance_mm"], 0.25
            )
            self.assertFalse(result["entropy_grid_used_spiral"])


if __name__ == "__main__":
    unittest.main()
