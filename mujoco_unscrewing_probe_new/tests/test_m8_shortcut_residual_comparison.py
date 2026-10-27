from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.render_m8_shortcut_residual_comparison import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    METHODS,
    load_trace,
    path_segments,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/active_comparison/m8_shortcut_residual_sobol_008"


def _manifest() -> dict:
    return json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))


def _results() -> dict:
    return json.loads((OUT / "results.json").read_text(encoding="utf-8"))["methods"]


def test_all_three_measured_replays_match_the_frozen_five_method_trial() -> None:
    manifest = _manifest()
    verification = manifest["same_rollout_contract"]["frozen_result_verification"]
    assert verification["all_exact_substantive_match"] is True
    for method in METHODS:
        item = verification["methods"][method]
        assert item["exact_substantive_match"] is True
        assert item["missing_fields"] == []
        assert item["mismatches"] == []
    assert manifest["same_rollout_contract"][
        "same_first_probe_fingerprint_for_proposed_and_proposed_plus"
    ] is True
    funnel = manifest["same_rollout_contract"]["shared_post_capture_funnel"]
    assert funnel["enabled"] is True
    assert funnel["max_recoveries"] == 3
    assert funnel["retry_on_terminal_rejection"] is True


def test_selected_mechanism_paths_and_times_are_trace_backed() -> None:
    manifest = _manifest()
    results = _results()
    expected = {
        "proposed": (16.12450000000777, 17.488500000004592),
        "proposed_plus": (11.944000000003106, 16.208500000007575),
        "outward": (24.507499999988234, 26.124499999984465),
    }
    for method, (capture_s, bottom_s) in expected.items():
        summary = manifest["methods"][method]
        assert summary["strict_bottom_success"] is True
        assert np.isclose(summary["capture_time_s"], capture_s)
        assert np.isclose(summary["bottom_time_s"], bottom_s)
        trace = load_trace(OUT / f"{method}_trace.csv")
        segments = path_segments(trace, results[method], method)
        assert len(segments["residual"]) > 0
        if method != "outward":
            assert len(segments["shortcut"]) > 0
        assert summary["residual_measured_path_length_mm"] > 0.0


def test_task_aware_grid_is_labeled_nonmonotonic_and_capture_terminated() -> None:
    ordering = _manifest()["proposed_plus_task_aware_ordering"]
    assert ordering["selected_action_count"] == 5
    assert ordering["entropy_is_monotonic_decreasing"] is False
    assert np.allclose(
        ordering["posterior_entropy_after_non_capture_actions_nats"],
        [
            4.908344233854252,
            5.130629506641007,
            5.359193183994287,
            5.427062947969357,
        ],
    )
    assert ordering["fifth_action_outcome"] == "capture_during_smooth_segment"
    assert "capture probability" in ordering["action_objective"]
    assert "capture cue" in ordering["termination"]


def test_m8_geometry_and_offline_overlay_boundary_are_explicit() -> None:
    manifest = _manifest()
    geometry = manifest["m8_geometry_offline_overlay"]
    assert geometry["mujoco_geometry_is_actual_allen_m8_source_model"] is True
    assert np.isclose(geometry["direct_clearance_radius_mm"], 0.10)
    assert np.isclose(geometry["chamfer_capture_radius_mm"], 0.25)
    assert geometry["true_center_used_online"] is False
    assert geometry["capture_regions_used_online"] is False
    figure_contract = manifest["figure_contract"]
    assert figure_contract["paths_are_measured_tip_xy_from_replayed_mujoco_traces"] is True
    assert figure_contract["pure_outward_is_faint_same_start_reference_only"] is True
    assert figure_contract["selected_case_is_not_isolated_causal_ablation"] is True

    figure = ROOT / manifest["artifacts"]["figure"]["path"]
    with Image.open(figure) as image:
        assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    for artifact in manifest["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]


class TestM8ShortcutResidualComparison(unittest.TestCase):
    """Expose the artifact assertions to the repository's unittest runner."""

    def test_frozen_replays(self) -> None:
        test_all_three_measured_replays_match_the_frozen_five_method_trial()

    def test_trace_backed_paths_and_times(self) -> None:
        test_selected_mechanism_paths_and_times_are_trace_backed()

    def test_nonmonotonic_task_aware_grid(self) -> None:
        test_task_aware_grid_is_labeled_nonmonotonic_and_capture_terminated()

    def test_geometry_and_offline_boundary(self) -> None:
        test_m8_geometry_and_offline_overlay_boundary_are_explicit()


if __name__ == "__main__":
    unittest.main()
