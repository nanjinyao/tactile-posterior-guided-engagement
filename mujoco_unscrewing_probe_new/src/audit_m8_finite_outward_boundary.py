from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.render_m8_search_comparison import SPIRAL_PITCH_M, archimedean_path
from src.run_m8_p2b_coverage_benchmark import (
    BASELINE_MAX_RADIUS_M,
    EFFECTIVE_CAPTURE_RADIUS_M,
    VISUAL_BOUND_M,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "allen_m8_source.json"
OUTPUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_finite_outward_boundary_audit"
)

PATH_SAMPLE_COUNT = 12_000
BOUNDARY_SAMPLE_COUNT = 65_536
BOUNDARY_CHUNK_SIZE = 512
NUMERICAL_CAPTURE_TOLERANCE_M = 1.0e-12


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _nearest_polyline_distances(
    query_xy_m: np.ndarray,
    path_xy_m: np.ndarray,
    *,
    first_segment_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact point-to-segment distances for a suffix of a polyline.

    The computation is chunked and uses only NumPy.  ``first_segment_index``
    lets the caller discard inner turns after establishing a radial lower
    bound that proves they cannot be nearest to any outer-boundary sample.
    """

    query = np.asarray(query_xy_m, dtype=float)
    path = np.asarray(path_xy_m, dtype=float)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("query_xy_m must have shape (N, 2)")
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_xy_m must have shape (M, 2), M >= 2")
    if not 0 <= first_segment_index < len(path) - 1:
        raise ValueError("first_segment_index is outside the path")

    segment_start = path[first_segment_index:-1]
    segment_vector = path[first_segment_index + 1 :] - segment_start
    segment_length_sq = np.einsum("ij,ij->i", segment_vector, segment_vector)
    if np.any(segment_length_sq <= 0.0):
        raise ValueError("path contains a zero-length segment")

    start_norm_sq = np.einsum("ij,ij->i", segment_start, segment_start)
    start_dot_vector = np.einsum(
        "ij,ij->i", segment_start, segment_vector
    )
    nearest_distance_sq = np.full(len(query), np.inf, dtype=float)
    nearest_segment = np.full(len(query), -1, dtype=np.int32)
    nearest_fraction = np.zeros(len(query), dtype=float)

    for lower in range(0, len(query), BOUNDARY_CHUNK_SIZE):
        upper = min(lower + BOUNDARY_CHUNK_SIZE, len(query))
        points = query[lower:upper]
        point_norm_sq = np.einsum("ij,ij->i", points, points)
        point_start_dot = points @ segment_start.T
        point_vector_dot = points @ segment_vector.T

        displacement_sq = (
            point_norm_sq[:, None]
            + start_norm_sq[None, :]
            - 2.0 * point_start_dot
        )
        projection_dot = point_vector_dot - start_dot_vector[None, :]
        fraction = np.clip(
            projection_dot / segment_length_sq[None, :], 0.0, 1.0
        )
        distance_sq = (
            displacement_sq
            - 2.0 * fraction * projection_dot
            + fraction * fraction * segment_length_sq[None, :]
        )
        local_segment = np.argmin(distance_sq, axis=1)
        row = np.arange(len(points))
        nearest_distance_sq[lower:upper] = distance_sq[row, local_segment]
        nearest_segment[lower:upper] = (
            local_segment + first_segment_index
        ).astype(np.int32)
        nearest_fraction[lower:upper] = fraction[row, local_segment]

    return (
        np.sqrt(np.maximum(nearest_distance_sq, 0.0)),
        nearest_segment,
        nearest_fraction,
    )


def build_report() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    spec_capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    if not math.isclose(
        spec_capture_radius_m,
        EFFECTIVE_CAPTURE_RADIUS_M,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("shared r_eff constant has drifted from the M8 spec")
    expected_max_radius_m = VISUAL_BOUND_M - EFFECTIVE_CAPTURE_RADIUS_M
    if not math.isclose(
        BASELINE_MAX_RADIUS_M,
        expected_max_radius_m,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("strict Outward radius is no longer R - r_eff")

    arc_m, path_xy_m = archimedean_path(
        np.zeros(2, dtype=float),
        SPIRAL_PITCH_M,
        BASELINE_MAX_RADIUS_M,
        0.0,
        samples=PATH_SAMPLE_COUNT,
    )
    path_radius_m = np.linalg.norm(path_xy_m, axis=1)
    endpoint_angle_rad = math.atan2(
        float(path_xy_m[-1, 1]), float(path_xy_m[-1, 0])
    )

    # Align sample zero with the finite spiral endpoint.  This makes the one
    # tangent boundary point explicit instead of losing it between samples.
    boundary_angles_rad = endpoint_angle_rad + (
        2.0 * math.pi * np.arange(BOUNDARY_SAMPLE_COUNT, dtype=float)
        / BOUNDARY_SAMPLE_COUNT
    )
    boundary_xy_m = VISUAL_BOUND_M * np.column_stack(
        (np.cos(boundary_angles_rad), np.sin(boundary_angles_rad))
    )

    last_turn_inner_radius_m = BASELINE_MAX_RADIUS_M - SPIRAL_PITCH_M
    crossing_vertex = int(
        np.searchsorted(path_radius_m, last_turn_inner_radius_m, side="left")
    )
    first_segment_index = max(0, crossing_vertex - 1)
    nearest_distance_m, nearest_segment, nearest_fraction = (
        _nearest_polyline_distances(
            boundary_xy_m,
            path_xy_m,
            first_segment_index=first_segment_index,
        )
    )

    # Every omitted chord lies inside the circle through this retained start
    # vertex.  Its distance to the R-boundary is therefore at least the radial
    # gap below.  Since that lower bound exceeds the observed outer-suffix
    # worst gap, the suffix computation is also globally valid for the entire
    # finite polyline.
    omitted_inner_radius_upper_bound_m = float(
        path_radius_m[first_segment_index]
    )
    omitted_inner_distance_lower_bound_m = (
        VISUAL_BOUND_M - omitted_inner_radius_upper_bound_m
    )
    worst_index = int(np.argmax(nearest_distance_m))
    worst_distance_m = float(nearest_distance_m[worst_index])
    globality_margin_m = (
        omitted_inner_distance_lower_bound_m - worst_distance_m
    )
    if globality_margin_m <= 0.0:
        raise AssertionError(
            "last-turn suffix did not prove global nearest-distance validity"
        )

    capture_threshold_m = (
        EFFECTIVE_CAPTURE_RADIUS_M + NUMERICAL_CAPTURE_TOLERANCE_M
    )
    beyond = nearest_distance_m > capture_threshold_m
    within = np.logical_not(beyond)
    endpoint_gap_m = float(nearest_distance_m[0])
    pitch_bound_m = 2.0 * EFFECTIVE_CAPTURE_RADIUS_M
    pitch_bound_satisfied = SPIRAL_PITCH_M <= pitch_bound_m + 1.0e-15
    full_outer_boundary_within_capture = bool(np.all(within))

    relative_worst_angle_rad = (
        boundary_angles_rad[worst_index] - endpoint_angle_rad
    ) % (2.0 * math.pi)
    report = {
        "schema_version": "m8-finite-outward-boundary-audit-v1",
        "experiment": "M8_finite_strict_Outward_outer_boundary_audit",
        "audit_type": "read_only_derived_geometry",
        "inputs": {
            "path_generator": (
                "src.render_m8_search_comparison.archimedean_path"
            ),
            "strict_constant_source": (
                "src.run_m8_p2b_coverage_benchmark"
            ),
            "m8_spec": str(SPEC_PATH.relative_to(ROOT)),
            "path_center_m": [0.0, 0.0],
            "path_phase_rad": 0.0,
            "path_sample_count": PATH_SAMPLE_COUNT,
            "boundary_sample_count": BOUNDARY_SAMPLE_COUNT,
            "boundary_phase_alignment": (
                "sample_0_aligned_with_finite_path_endpoint"
            ),
            "numerical_capture_tolerance_m": (
                NUMERICAL_CAPTURE_TOLERANCE_M
            ),
        },
        "geometry": {
            "visual_support_radius_m": VISUAL_BOUND_M,
            "visual_support_radius_mm": 1000.0 * VISUAL_BOUND_M,
            "effective_capture_radius_m": EFFECTIVE_CAPTURE_RADIUS_M,
            "effective_capture_radius_mm": (
                1000.0 * EFFECTIVE_CAPTURE_RADIUS_M
            ),
            "strict_outward_pitch_m_per_rev": SPIRAL_PITCH_M,
            "strict_outward_pitch_mm_per_rev": 1000.0 * SPIRAL_PITCH_M,
            "adjacent_turn_pitch_bound_m_per_rev": pitch_bound_m,
            "adjacent_turn_pitch_bound_mm_per_rev": 1000.0 * pitch_bound_m,
            "pitch_to_adjacent_turn_bound_ratio": (
                SPIRAL_PITCH_M / pitch_bound_m
            ),
            "finite_path_max_radius_m": BASELINE_MAX_RADIUS_M,
            "finite_path_max_radius_mm": 1000.0 * BASELINE_MAX_RADIUS_M,
            "finite_path_max_radius_definition": "R - r_eff",
            "last_turn_inner_radius_m": last_turn_inner_radius_m,
        },
        "finite_path": {
            "arc_length_m": float(arc_m[-1]),
            "revolutions": BASELINE_MAX_RADIUS_M / SPIRAL_PITCH_M,
            "endpoint_angle_rad_wrapped": endpoint_angle_rad,
            "segment_count": int(len(path_xy_m) - 1),
            "evaluated_suffix_first_segment_index": first_segment_index,
            "evaluated_suffix_segment_count": int(
                len(path_xy_m) - 1 - first_segment_index
            ),
            "omitted_inner_radius_upper_bound_m": (
                omitted_inner_radius_upper_bound_m
            ),
            "omitted_inner_boundary_distance_lower_bound_m": (
                omitted_inner_distance_lower_bound_m
            ),
            "suffix_globality_margin_m": globality_margin_m,
            "suffix_is_proven_global_for_boundary_samples": True,
        },
        "boundary_nearest_distance": {
            "minimum_m": float(np.min(nearest_distance_m)),
            "minimum_mm": 1000.0 * float(np.min(nearest_distance_m)),
            "endpoint_aligned_sample_distance_m": endpoint_gap_m,
            "worst_m": worst_distance_m,
            "worst_mm": 1000.0 * worst_distance_m,
            "median_m": float(np.median(nearest_distance_m)),
            "p90_m": float(np.percentile(nearest_distance_m, 90.0)),
            "p95_m": float(np.percentile(nearest_distance_m, 95.0)),
            "p99_m": float(np.percentile(nearest_distance_m, 99.0)),
            "worst_sample_index": worst_index,
            "worst_endpoint_relative_angle_rad": float(
                relative_worst_angle_rad
            ),
            "worst_endpoint_relative_angle_deg": math.degrees(
                relative_worst_angle_rad
            ),
            "worst_nearest_segment_index": int(
                nearest_segment[worst_index]
            ),
            "worst_nearest_segment_fraction": float(
                nearest_fraction[worst_index]
            ),
            "at_or_within_r_eff_count": int(np.count_nonzero(within)),
            "beyond_r_eff_count": int(np.count_nonzero(beyond)),
            "fraction_beyond_r_eff": float(np.mean(beyond)),
        },
        "conclusions": {
            "adjacent_turn_pitch_bound_satisfied": bool(
                pitch_bound_satisfied
            ),
            "full_outer_boundary_within_r_eff": (
                full_outer_boundary_within_capture
            ),
            "complete_visual_disk_coverage_certificate": False,
            "analytic_outer_boundary_contact_set": (
                "one endpoint-tangent point only; zero boundary-arc measure"
            ),
            "interpretation": (
                "The pitch bound controls spacing between adjacent turns, but "
                "the finite spiral has no padded outer closing turn. Its "
                "r_eff tube therefore does not cover the outer boundary of "
                "the declared visual-support disk."
            ),
        },
        "provenance_and_change_boundary": {
            "policy_changed": False,
            "trial_files_changed": False,
            "simulation_rerun": False,
            "derived_from_declared_geometry_and_current_path_generator": True,
        },
        "claim_boundary": (
            "The current strict Outward method remains an empirical search "
            "comparator. The adjacent-turn pitch bound is satisfied, but this "
            "finite-path audit is not a complete-domain coverage certificate "
            "and does not certify capture, insertion, finite-time success, or "
            "hardware behavior."
        ),
    }
    return report


def render_summary(report: dict) -> str:
    geometry = report["geometry"]
    distance = report["boundary_nearest_distance"]
    conclusions = report["conclusions"]
    provenance = report["provenance_and_change_boundary"]
    return "\n".join(
        [
            "# M8 Finite Outward Boundary Audit",
            "",
            "## Result",
            "",
            (
                "- Adjacent-turn pitch bound: **satisfied** "
                f"({geometry['strict_outward_pitch_mm_per_rev']:.4f} <= "
                f"{geometry['adjacent_turn_pitch_bound_mm_per_rev']:.4f} "
                "mm/rev)."
            ),
            (
                "- Finite path endpoint radius: "
                f"{geometry['finite_path_max_radius_mm']:.6f} mm = "
                "R - r_eff."
            ),
            (
                f"- Endpoint-aligned boundary samples beyond r_eff: "
                f"**{distance['beyond_r_eff_count']}/"
                f"{report['inputs']['boundary_sample_count']}** "
                f"({100.0 * distance['fraction_beyond_r_eff']:.6f}%)."
            ),
            (
                "- Boundary nearest-distance min / median / worst: "
                f"{distance['minimum_mm']:.6f} / "
                f"{1000.0 * distance['median_m']:.6f} / "
                f"{distance['worst_mm']:.6f} mm, versus "
                f"r_eff={geometry['effective_capture_radius_mm']:.6f} mm."
            ),
            (
                "- Full outer boundary within r_eff: "
                f"**{conclusions['full_outer_boundary_within_r_eff']}**."
            ),
            (
                "- Complete visual-disk coverage certificate: "
                f"**{conclusions['complete_visual_disk_coverage_certificate']}**."
            ),
            "",
            "## Interpretation",
            "",
            conclusions["interpretation"],
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
            "## Change boundary",
            "",
            (
                "This is a read-only derived audit: policy changed="
                f"{provenance['policy_changed']}, trial files changed="
                f"{provenance['trial_files_changed']}, simulation rerun="
                f"{provenance['simulation_rerun']}."
            ),
            "",
        ]
    )


def run() -> dict:
    report = build_report()
    _atomic_write(
        OUTPUT / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(OUTPUT / "summary.md", render_summary(report))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
