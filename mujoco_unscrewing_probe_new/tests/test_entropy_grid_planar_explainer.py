from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.render_m8_entropy_grid_planar_explainer import DEFAULT_SOURCE, load_run


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/active_comparison/entropy_grid_planar_explainer"


def test_selected_entropy_grid_geometry_and_route_are_auditable() -> None:
    run = load_run(DEFAULT_SOURCE)
    assert run.pose_id == "sobol_002"
    assert len(run.action_nodes_mm) == 673
    assert int(np.count_nonzero(run.capture_eligible)) == 2
    assert np.allclose(run.initial_xy_mm, [1.4380550574, -2.6953096618])
    assert np.linalg.norm(run.capture_event_xy_mm) <= run.capture_radius_mm
    assert len(run.visited_xy_mm) == 6
    assert run.capture_time_s < run.bottom_time_s < run.active_budget_s


def test_entropy_grid_explainer_artifacts_match_manifest() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["online_true_center_used"] is False
    assert manifest["interpretation_only_true_center"] is True
    assert manifest["action_node_count"] == 673
    assert len(manifest["nominal_capture_eligible_nodes_mm"]) == 2
    for key in ("figure", "video", "poster"):
        assert (ROOT / manifest[key]).is_file()
