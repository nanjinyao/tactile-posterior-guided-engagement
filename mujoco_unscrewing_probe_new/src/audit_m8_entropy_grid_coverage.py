from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from src.entropy_grid_baseline import EntropyGridBelief, EntropyGridConfig, disk_grid


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/allen_m8_source.json"
OUTPUT = ROOT / "outputs/current/evaluations/m8_entropy_grid_coverage_certificate"
VISUAL_SIGMA_M = 0.0020
VISUAL_BOUND_M = math.sqrt(5.991) * VISUAL_SIGMA_M


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    effective_capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    action_spacing_m = min(
        0.00075,
        0.95 * math.sqrt(2.0) * effective_capture_radius_m,
    )
    belief = EntropyGridBelief(
        np.zeros(2, dtype=float),
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        support_inner_radius_m=0.5 * float(spec["derived"]["socket_top_af_m"]),
        support_outer_radius_m=0.5 * float(spec["screw_head_diameter_m"]),
        probe_radius_m=float(spec["key_af_m"]) / math.sqrt(3.0),
        config=EntropyGridConfig(
            belief_spacing_m=min(0.00025, effective_capture_radius_m),
            action_spacing_m=action_spacing_m,
            credible_radius_stop_m=effective_capture_radius_m,
        ),
        effective_capture_radius_m=effective_capture_radius_m,
    )
    certificate = belief.certify_action_coverage(
        dense_subdivisions_per_cell=8,
        boundary_sample_count=8192,
    )
    clipped_action_nodes = len(
        disk_grid(np.zeros(2, dtype=float), VISUAL_BOUND_M, action_spacing_m)
    )
    report = {
        "experiment": "M8_finite_disk_entropy_grid_coverage_certificate",
        "geometry": {
            "visual_support_radius_mm": 1000.0 * VISUAL_BOUND_M,
            "effective_capture_radius_mm": 1000.0 * effective_capture_radius_m,
            "belief_spacing_mm": 1000.0 * belief.config.belief_spacing_m,
            "action_spacing_mm": 1000.0 * belief.config.action_spacing_m,
            "configured_action_padding_mm": 1000.0 * belief.action_padding_m,
            "achieved_lattice_radial_padding_mm": (
                1000.0 * certificate.action_padding_m
            ),
            "candidate_center_nodes": int(len(belief.candidate_xy_m)),
            "clipped_action_nodes_before_padding": int(clipped_action_nodes),
            "action_nodes": int(len(belief.action_xy_m)),
        },
        "certificate": asdict(certificate),
        "online_truth_boundary": {
            "true_center_used_for_action_selection": False,
            "certificate_uses_only_declared_support_geometry": True,
            "capture_radius_is_CAD_derived_not_hardware_calibrated": True,
        },
        "claim_boundary": (
            "This certifies geometric node coverage of the declared finite M8 "
            "visual-support disk by the padded action lattice. It does not "
            "certify compliant tracking, capture-cue detection, finite-time "
            "completion, CAD-bottom insertion, hardware chamfer attraction, or "
            "success after unsafe/budget interruption; those are audited by the "
            "measured-path controller and strict complete-loop rollout."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_write(
        OUTPUT / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# M8 Finite-Disk Entropy-Grid Coverage Certificate",
        "",
        f"- certified: **{certificate.certified}**",
        f"- visual support radius: {1000.0 * VISUAL_BOUND_M:.4f} mm",
        f"- effective capture radius: {1000.0 * effective_capture_radius_m:.4f} mm",
        f"- action spacing / configured padding: {1000.0 * action_spacing_m:.4f} / {1000.0 * belief.action_padding_m:.4f} mm",
        f"- candidate centers: {len(belief.candidate_xy_m)}",
        f"- clipped/padded action nodes: {clipped_action_nodes} / {len(belief.action_xy_m)}",
        f"- analytic worst gap: {1000.0 * certificate.analytic_worst_case_gap_m:.4f} mm",
        f"- dense/boundary observed worst gap: {1000.0 * certificate.observed_worst_gap_m:.4f} mm",
        "",
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
    ]
    _atomic_write(OUTPUT / "summary.md", "\n".join(lines))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
