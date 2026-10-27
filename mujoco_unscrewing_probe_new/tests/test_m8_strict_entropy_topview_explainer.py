from __future__ import annotations

import json
from pathlib import Path
import unittest

import imageio.v2 as imageio
import numpy as np

from src.render_m8_strict_entropy_topview_explainer import (
    EXPECTED_BELIEF_HYPOTHESES,
    EXPECTED_PADDING_ACTIONS,
    EXPECTED_SELECTED_ACTIONS,
    EXPECTED_SUPPORT_ACTIONS,
    _capture_trace_index,
    _segments,
    load_trace,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003"
)


def _manifest() -> dict:
    return json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))


class TestM8StrictEntropyTopviewExplainer(unittest.TestCase):
    def test_strict_replay_matches_frozen_and_keeps_sets_separate(self) -> None:
        manifest = _manifest()
        verification = manifest["replay_contract"]["frozen_result_verification"]
        self.assertTrue(verification["exact_substantive_match"])
        self.assertEqual(verification["missing_fields"], [])
        self.assertEqual(verification["mismatches"], [])
        self.assertEqual(verification["frozen_field_count"], 134)
        self.assertEqual(verification["compared_field_count"], 132)

        finite = manifest["finite_sets"]
        self.assertEqual(finite["belief_hypotheses"], EXPECTED_BELIEF_HYPOTHESES)
        self.assertTrue(finite["belief_hypotheses_are_not_action_grid"])
        self.assertEqual(finite["actions_inside_visual_support"], EXPECTED_SUPPORT_ACTIONS)
        self.assertEqual(finite["boundary_padding_actions"], EXPECTED_PADDING_ACTIONS)
        self.assertEqual(finite["total_available_action_candidates"], 749)
        self.assertEqual(finite["selected_action_count"], EXPECTED_SELECTED_ACTIONS)
        self.assertFalse(finite["controller_traversed_all_action_candidates"])
        self.assertTrue(finite["coverage_certificate_offline_audit"]["certified"])

    def test_capture_sets_and_measured_path_are_auditable(self) -> None:
        manifest = _manifest()
        capture = manifest["capture_sets"]
        self.assertTrue(np.isclose(capture["direct_clearance_radius_mm"], 0.10))
        self.assertTrue(np.isclose(capture["chamfer_eligible_radius_mm"], 0.25))
        self.assertEqual(len(capture["nominal_direct_clearance_action_nodes_mm"]), 1)
        self.assertEqual(capture["nominal_chamfer_only_action_nodes_mm"], [])
        self.assertTrue(capture["measured_capture_inside_direct_clearance"])
        self.assertTrue(capture["measured_capture_inside_chamfer_eligible"])
        self.assertGreater(capture["nominal_selected_action_5_radius_mm"], 0.25)

        trace = load_trace(OUT / "trace.csv")
        result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))["result"]
        capture_index, nearest_distance = _capture_trace_index(trace, result)
        self.assertEqual(capture_index, manifest["trajectory"]["capture_trace_row"])
        self.assertLess(nearest_distance, 1.0e-9)
        styles = {is_lift for is_lift, _ in _segments(trace, capture_index)}
        self.assertEqual(styles, {False, True})
        self.assertEqual(result["result"], "cad_bottom_verified")
        self.assertTrue(np.isclose(result["time_to_inner_wall_s"], 10.034000000000772))
        self.assertTrue(np.isclose(result["time_to_bottom_s"], 12.745500000004085))

    def test_entropy_progression_and_rendered_artifacts(self) -> None:
        manifest = _manifest()
        progression = manifest["belief_progression"]
        self.assertEqual([row["probe"] for row in progression], [1, 2, 3, 4, 5])
        self.assertTrue(
            np.allclose(
                [row["posterior_entropy_nats"] for row in progression[:4]],
                [5.95326530438526, 5.149168481423901, 3.7099121809492126, 2.242832471909397],
            )
        )
        self.assertTrue(
            np.allclose(
                [row["posterior_credible_radius_95_mm"] for row in progression[:4]],
                [3.8393314209433544, 2.8818838910297346, 2.2605636975435703, 1.5919876582818269],
            )
        )
        self.assertEqual(progression[-1]["observation"], "direct_capture")

        for artifact in manifest["artifacts"].values():
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, artifact["bytes"])

        reader = imageio.get_reader(ROOT / manifest["artifacts"]["video"]["path"])
        try:
            self.assertEqual(reader.count_frames(), manifest["video"]["frame_count"])
            frame = reader.get_data(manifest["video"]["frame_count"] - 1)
        finally:
            reader.close()
        self.assertEqual(frame.shape[:2], (900, 1600))

    def test_offline_overlay_claim_boundary_is_explicit(self) -> None:
        boundary = _manifest()["offline_explanation_boundary"]
        self.assertTrue(boundary["mujoco_geometry_is_actual_allen_m8_source_model"])
        self.assertFalse(boundary["true_center_used_online"])
        self.assertFalse(boundary["contact_tags_used_online"])
        self.assertFalse(boundary["capture_disks_and_node_eligibility_used_online"])
        self.assertTrue(boundary["true_center_capture_disks_and_eligibility_are_offline_overlays"])
        self.assertTrue(boundary["selected_single_rollout_is_not_distribution_level_evidence"])


if __name__ == "__main__":
    unittest.main()
