#!/usr/bin/env python3
"""Generate the deterministic multi-size conical-probe sampling plan.

This utility is deliberately offline.  It reads only frozen geometry/config
files, writes a plan/table/figure, and has no robot, sensor, or network
interface.  The generated points are CAD-seeded *local offsets*.  At execution
they must be rebased to the measured as-built socket centre and transformed by
a separately qualified fixture-to-WORLD transform.  They are never robot
WORLD poses or labels by themselves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle


REAL_ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = REAL_ROOT.parent
SIM_ROOT = GROUP_ROOT / "mujoco_unscrewing_probe_new"
TOOLING_PROFILE = SIM_ROOT / "configs" / "real_tooling_v1.json"

# Photo order is deliberately the execution/readout order.
SIZE_ORDER = ("m12", "m10", "m8")
EXPECTED_CENTRE_X_MM = {"m8": -45.0, "m10": 0.0, "m12": 45.0}

CLEAN_ANGLES_DEG = (3.75, 41.25, 86.25, 131.25, 176.25, 228.75, 273.75, 318.75)
HOLDOUT_ANGLES_DEG = {"RS": (16.0, 196.0), "RO": (46.0, 226.0)}
QUARANTINE_ANGLES_DEG = {"QI": (24.5, 204.5), "QO": (54.5, 234.5)}
SENTINEL_SPECS = (("RS", 3.75), ("RO", 176.25))

EXPECTED_VISUAL_BOUND_MM = {"m8": 5.0, "m10": 6.0, "m12": 8.0}

PARTITION_STYLE = {
    "clean_candidate": {
        "marker": "o",
        "color": "#2563eb",
        "training_use": "eligible_only_after_complete_trace_and_quality_gates",
    },
    "holdout": {
        "marker": "s",
        "color": "#f59e0b",
        "training_use": "never_train_reserved_for_final_real_alignment_audit",
    },
    "quarantine": {
        "marker": "X",
        "color": "#dc2626",
        "training_use": "forbidden_boundary_diagnostic_only",
    },
    "sentinel": {
        "marker": "*",
        "color": "#111827",
        "training_use": "repeatability_monitor_not_an_independent_training_sample",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def round_float(value: float, digits: int = 9) -> float:
    # Constrain textual noise while retaining sub-micrometre precision in mm.
    return round(float(value), digits)


def angle_token(angle_deg: float) -> str:
    return f"{angle_deg:06.2f}".replace(".", "p")


def add_point(
    points: list[dict[str, Any]],
    *,
    size: str,
    centre_x_mm: float,
    radius_mm: float,
    angle_deg: float,
    partition: str,
    ring: str,
    duplicate_of: str | None = None,
) -> None:
    theta = math.radians(angle_deg)
    local_x = radius_mm * math.cos(theta)
    local_y = radius_mm * math.sin(theta)
    fixture_x = centre_x_mm + local_x
    fixture_y = local_y
    sample_id = (
        f"{size.upper()}-{partition.upper()}-{ring}-A{angle_token(angle_deg)}"
    )
    style = PARTITION_STYLE[partition]
    point = {
        "attempt_index": len(points) + 1,
        "sample_id": sample_id,
        "partition": partition,
        "training_use": style["training_use"],
        "ring": ring,
        "radius_mm": round_float(radius_mm),
        "angle_fixture_deg": round_float(angle_deg),
        "local_offset_fixture_mm": [
            round_float(local_x),
            round_float(local_y),
        ],
        "fixture_xy_mm": [
            round_float(fixture_x),
            round_float(fixture_y),
        ],
        # The plotted top view uses ordinary mathematical display axes:
        # horizontal u grows right and vertical v_up grows upward.  The viewer
        # is on +Z_F looking along -Z_F.  Since +X_F is left in the reviewed
        # photograph and X_F x Y_F = Z_F, +Y_F must point downward on the
        # displayed top view.  Thus (u_right, v_up) = (-x_F, -y_F).
        # Conventional image pixels instead use v_down, for which
        # (u_right, v_down) = (-x_F, +y_F).
        "photo_uv_mm": [
            round_float(-fixture_x),
            round_float(-fixture_y),
        ],
        "true_direction_to_centre_fixture_deg": round_float(
            (angle_deg + 180.0) % 360.0
        ),
        "true_direction_to_centre_fixture_unit": [
            round_float(-math.cos(theta)),
            round_float(-math.sin(theta)),
        ],
        "duplicate_of": duplicate_of,
    }
    points.append(point)


def source_record(path: Path) -> dict[str, str]:
    return {
        "path_from_group_meeting": str(path.relative_to(GROUP_ROOT)),
        "sha256": sha256(path),
    }


def build_plan() -> dict[str, Any]:
    tooling = read_json(TOOLING_PROFILE)
    fixture = tooling["fixture"]
    source_files: dict[str, dict[str, str]] = {
        "real_tooling_profile": source_record(TOOLING_PROFILE)
    }
    sizes: list[dict[str, Any]] = []

    for size in SIZE_ORDER:
        spec_path = SIM_ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json"
        spec = read_json(spec_path)
        source_files[f"{size}_hardware_easy_v2"] = source_record(spec_path)

        centre_m = fixture["socket_poses"][size]["translation_m"]
        centre_x_mm = 1000.0 * float(centre_m[0])
        centre_y_mm = 1000.0 * float(centre_m[1])
        if not math.isclose(centre_x_mm, EXPECTED_CENTRE_X_MM[size], abs_tol=1e-12):
            raise ValueError(
                f"{size}: fixture centre x={centre_x_mm} mm, expected "
                f"{EXPECTED_CENTRE_X_MM[size]} mm"
            )
        if not math.isclose(centre_y_mm, 0.0, abs_tol=1e-12):
            raise ValueError(f"{size}: fixture centre y must be 0 mm")

        head_radius_mm = 500.0 * float(spec["screw_head_diameter_m"])
        bit_circumradius_mm = (
            1000.0 * float(spec["key_af_m"]) / math.sqrt(3.0)
        )
        support_radius_mm = head_radius_mm - bit_circumradius_mm

        # The reviewed 5/6/8 mm numerical envelope is the maximum configured
        # sweep, not visual_offset_default_m (currently 3/4/5 mm).
        sweep_mm = [1000.0 * float(v) for v in spec["visual_offset_sweep_m"]]
        visual_bound_mm = max(sweep_mm)
        expected_visual = EXPECTED_VISUAL_BOUND_MM[size]
        if not math.isclose(visual_bound_mm, expected_visual, abs_tol=1e-12):
            raise ValueError(
                f"{size}: max visual sweep={visual_bound_mm} mm, expected "
                f"{expected_visual} mm"
            )
        if visual_bound_mm <= support_radius_mm:
            raise ValueError(
                f"{size}: visual bound must exceed full-support radius"
            )

        radii = {
            "RS": 0.85 * support_radius_mm,
            "RO": support_radius_mm
            + 0.70 * (visual_bound_mm - support_radius_mm),
            "QI": support_radius_mm - 0.15,
            "QO": support_radius_mm
            + 0.30 * (visual_bound_mm - support_radius_mm),
        }

        points: list[dict[str, Any]] = []
        for ring in ("RS", "RO"):
            for angle in CLEAN_ANGLES_DEG:
                add_point(
                    points,
                    size=size,
                    centre_x_mm=centre_x_mm,
                    radius_mm=radii[ring],
                    angle_deg=angle,
                    partition="clean_candidate",
                    ring=ring,
                )
        for ring in ("RS", "RO"):
            for angle in HOLDOUT_ANGLES_DEG[ring]:
                add_point(
                    points,
                    size=size,
                    centre_x_mm=centre_x_mm,
                    radius_mm=radii[ring],
                    angle_deg=angle,
                    partition="holdout",
                    ring=ring,
                )
        for ring in ("QI", "QO"):
            for angle in QUARANTINE_ANGLES_DEG[ring]:
                add_point(
                    points,
                    size=size,
                    centre_x_mm=centre_x_mm,
                    radius_mm=radii[ring],
                    angle_deg=angle,
                    partition="quarantine",
                    ring=ring,
                )
        for ring, angle in SENTINEL_SPECS:
            duplicate_id = (
                f"{size.upper()}-CLEAN_CANDIDATE-{ring}-"
                f"A{angle_token(angle)}"
            )
            add_point(
                points,
                size=size,
                centre_x_mm=centre_x_mm,
                radius_mm=radii[ring],
                angle_deg=angle,
                partition="sentinel",
                ring=ring,
                duplicate_of=duplicate_id,
            )

        counts = {
            name: sum(point["partition"] == name for point in points)
            for name in PARTITION_STYLE
        }
        expected_counts = {
            "clean_candidate": 16,
            "holdout": 4,
            "quarantine": 4,
            "sentinel": 2,
        }
        if counts != expected_counts or len(points) != 26:
            raise AssertionError(
                f"{size}: invalid partition counts {counts}, n={len(points)}"
            )

        sizes.append(
            {
                "size": size,
                "photo_order_index": SIZE_ORDER.index(size) + 1,
                "fixture_centre_xy_mm": [
                    round_float(centre_x_mm),
                    round_float(centre_y_mm),
                ],
                "photo_centre_uv_mm": [
                    round_float(-centre_x_mm),
                    round_float(-centre_y_mm),
                ],
                "geometry_mm": {
                    "head_radius": round_float(head_radius_mm),
                    "bit_circumradius": round_float(bit_circumradius_mm),
                    "full_support_radius": round_float(support_radius_mm),
                    "socket_mouth_af": round_float(
                        1000.0 * float(spec["derived"]["socket_top_af_m"])
                    ),
                    "configured_visual_offset_default": round_float(
                        1000.0 * float(spec["visual_offset_default_m"])
                    ),
                    "sampling_visual_bound": round_float(visual_bound_mm),
                    "sampling_visual_bound_source": (
                        "max(visual_offset_sweep_m); the reviewed 5/6/8 mm "
                        "values are not visual_offset_default_m"
                    ),
                },
                "sampling_radii_mm": {
                    key: round_float(value) for key, value in radii.items()
                },
                "attempt_count": len(points),
                "partition_counts": counts,
                "points": points,
            }
        )

    plan = {
        "schema_version": 1,
        "name": "fixture_multisize_conical_sampling_v1",
        "status": "offline_cad_plan_requires_fixture_transform_teach_and_preflight",
        "claim_boundary": {
            "purpose": (
                "minimum structured conical-probe acquisition using CAD-seeded "
                "offsets rebased to measured M12/M10/M8 as-built centres"
            ),
            "not_a_robot_pose_plan": (
                "local offsets must first be rebased to the measured as-built "
                "socket centre, then transformed by a separately measured and "
                "qualified fixture-to-WORLD transform"
            ),
            "no_centre_attempt": True,
            "label_policy": (
                "CAD centres are initial guesses only; measured touchdown and "
                "measured as-built centre define labels; centre samples carry "
                "no direction label and are not included"
            ),
            "data_governance": (
                "only clean candidates may become training data after complete "
                "cone/force/frame/label quality gates; holdout, quarantine, and "
                "sentinel attempts are never independent training samples"
            ),
        },
        "coordinate_contract": {
            "fixture_frame": {
                "name": "F",
                "angle_definition": (
                    "degrees counter-clockwise from +X_F toward +Y_F"
                ),
                "socket_centres_mm": {
                    size: [EXPECTED_CENTRE_X_MM[size], 0.0]
                    for size in ("m8", "m10", "m12")
                },
            },
            "photo_top_view": {
                "left_to_right": list(SIZE_ORDER),
                "view_direction": (
                    "viewer is on +Z_F looking along -Z_F; +Z_F points out "
                    "of the fixture top plane toward the viewer"
                ),
                "mapping": (
                    "u_photo_mm = -x_fixture_mm; "
                    "v_photo_up_mm = -y_fixture_mm"
                ),
                "display_axis_mapping": (
                    "u_right_mm = -x_fixture_mm; "
                    "v_up_mm = -y_fixture_mm"
                ),
                "image_pixel_mapping": (
                    "u_pixel_right is proportional to -x_fixture; "
                    "v_pixel_down is proportional to +y_fixture"
                ),
                "plus_x_fixture_points": "left",
                "plus_y_fixture_points": "down",
                "qualification": (
                    "the displayed-axis mapping fixes handedness for the "
                    "schematic only; it is not a calibrated camera homography"
                ),
            },
        },
        "source_files": source_files,
        "sampling_design": {
            "attempts_per_size": 26,
            "total_attempts": 78,
            "clean_candidate": {
                "rings": ["RS", "RO"],
                "angles_deg": list(CLEAN_ANGLES_DEG),
                "count_per_size": 16,
            },
            "holdout": {
                "angles_deg_by_ring": {
                    key: list(value) for key, value in HOLDOUT_ANGLES_DEG.items()
                },
                "count_per_size": 4,
            },
            "quarantine": {
                "angles_deg_by_ring": {
                    key: list(value)
                    for key, value in QUARANTINE_ANGLES_DEG.items()
                },
                "count_per_size": 4,
            },
            "sentinel": {
                "ring_angle_pairs": [
                    {"ring": ring, "angle_deg": angle}
                    for ring, angle in SENTINEL_SPECS
                ],
                "count_per_size": 2,
                "semantics": (
                    "exact repeats of two clean-candidate coordinates for "
                    "drift/repeatability monitoring"
                ),
            },
            "radius_equations": {
                "R_support": "head_radius - bit_circumradius",
                "RS": "0.85 * R_support",
                "RO": "R_support + 0.70 * (visual_bound - R_support)",
                "QI": "R_support - 0.15 mm",
                "QO": "R_support + 0.30 * (visual_bound - R_support)",
            },
        },
        "sizes": sizes,
        "validation": {
            "fixture_centres_match_profile": True,
            "photo_order_is_m12_m10_m8": True,
            "per_size_counts_are_16_4_4_2": True,
            "total_attempts": sum(size["attempt_count"] for size in sizes),
            "centre_attempts": 0,
        },
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    sizes = plan["sizes"]
    if [entry["size"] for entry in sizes] != list(SIZE_ORDER):
        raise AssertionError("Physical photo order must be M12, M10, M8")
    if sum(entry["attempt_count"] for entry in sizes) != 78:
        raise AssertionError("Expected 78 total attempts")
    for entry in sizes:
        if entry["attempt_count"] != 26:
            raise AssertionError(f"{entry['size']}: expected 26 attempts")
        if any(
            math.isclose(point["radius_mm"], 0.0, abs_tol=1e-12)
            for point in entry["points"]
        ):
            raise AssertionError(f"{entry['size']}: centre samples are forbidden")
        for point in entry["points"]:
            cx, cy = entry["fixture_centre_xy_mm"]
            px, py = point["fixture_xy_mm"]
            photo_u, photo_v_up = point["photo_uv_mm"]
            if (
                not math.isclose(photo_u, -px, abs_tol=1e-9)
                or not math.isclose(photo_v_up, -py, abs_tol=1e-9)
            ):
                raise AssertionError(
                    f"{point['sample_id']}: displayed top-view mapping must "
                    "be (u_right, v_up)=(-x_F, -y_F)"
                )
            ux, uy = point["true_direction_to_centre_fixture_unit"]
            radial_x = px - cx
            radial_y = py - cy
            dot = radial_x * ux + radial_y * uy
            if not dot < -0.99 * point["radius_mm"]:
                raise AssertionError(
                    f"{point['sample_id']}: direction is not inward"
                )
        sentinels = [
            point for point in entry["points"] if point["partition"] == "sentinel"
        ]
        candidates = {
            point["sample_id"]: point
            for point in entry["points"]
            if point["partition"] == "clean_candidate"
        }
        for sentinel in sentinels:
            source = candidates.get(sentinel["duplicate_of"])
            if source is None:
                raise AssertionError(
                    f"{sentinel['sample_id']}: missing duplicate source"
                )
            if (
                source["fixture_xy_mm"] != sentinel["fixture_xy_mm"]
                or source["ring"] != sentinel["ring"]
            ):
                raise AssertionError(
                    f"{sentinel['sample_id']}: repeat coordinate mismatch"
                )


def serialize_plan(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, ensure_ascii=False) + "\n"


def serialize_csv(plan: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "size",
        "photo_order_index",
        "attempt_index",
        "sample_id",
        "partition",
        "training_use",
        "ring",
        "radius_mm",
        "angle_fixture_deg",
        "local_x_fixture_mm",
        "local_y_fixture_mm",
        "fixture_x_mm",
        "fixture_y_mm",
        "photo_u_mm",
        "photo_v_mm",
        "true_direction_to_centre_fixture_deg",
        "true_direction_unit_x_fixture",
        "true_direction_unit_y_fixture",
        "duplicate_of",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for size_entry in plan["sizes"]:
        for point in size_entry["points"]:
            writer.writerow(
                {
                    "size": size_entry["size"],
                    "photo_order_index": size_entry["photo_order_index"],
                    "attempt_index": point["attempt_index"],
                    "sample_id": point["sample_id"],
                    "partition": point["partition"],
                    "training_use": point["training_use"],
                    "ring": point["ring"],
                    "radius_mm": f"{point['radius_mm']:.9f}",
                    "angle_fixture_deg": f"{point['angle_fixture_deg']:.2f}",
                    "local_x_fixture_mm": (
                        f"{point['local_offset_fixture_mm'][0]:.9f}"
                    ),
                    "local_y_fixture_mm": (
                        f"{point['local_offset_fixture_mm'][1]:.9f}"
                    ),
                    "fixture_x_mm": f"{point['fixture_xy_mm'][0]:.9f}",
                    "fixture_y_mm": f"{point['fixture_xy_mm'][1]:.9f}",
                    "photo_u_mm": f"{point['photo_uv_mm'][0]:.9f}",
                    "photo_v_mm": f"{point['photo_uv_mm'][1]:.9f}",
                    "true_direction_to_centre_fixture_deg": (
                        f"{point['true_direction_to_centre_fixture_deg']:.2f}"
                    ),
                    "true_direction_unit_x_fixture": (
                        f"{point['true_direction_to_centre_fixture_unit'][0]:.9f}"
                    ),
                    "true_direction_unit_y_fixture": (
                        f"{point['true_direction_to_centre_fixture_unit'][1]:.9f}"
                    ),
                    "duplicate_of": point["duplicate_of"] or "",
                }
            )
    return buffer.getvalue()


def regular_hexagon_photo(
    centre_u: float, centre_v: float, across_flats_mm: float
) -> list[tuple[float, float]]:
    radius = across_flats_mm / math.sqrt(3.0)
    # Socket yaw is zero in F.  The top-view display is a 180-degree in-plane
    # rotation: (u_right, v_up)=(-x_F, -y_F).  It therefore preserves
    # handedness rather than reflecting the socket.
    return [
        (
            centre_u - radius * math.cos(math.radians(30.0 + 60.0 * index)),
            centre_v - radius * math.sin(math.radians(30.0 + 60.0 * index)),
        )
        for index in range(6)
    ]


def render_figure(plan: dict[str, Any], png_path: Path, svg_path: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "fixture_multisize_conical_sampling_v1",
        }
    )
    figure, axis = plt.subplots(figsize=(14.0, 7.2), constrained_layout=True)
    axis.add_patch(
        Rectangle(
            (-75.0, -35.0),
            150.0,
            70.0,
            facecolor="#f8fafc",
            edgecolor="#475569",
            linewidth=1.8,
            zorder=0,
        )
    )

    for size_entry in plan["sizes"]:
        centre_u, centre_v = size_entry["photo_centre_uv_mm"]
        geometry = size_entry["geometry_mm"]
        axis.add_patch(
            Circle(
                (centre_u, centre_v),
                geometry["head_radius"],
                facecolor="#e2e8f0",
                edgecolor="#64748b",
                linewidth=1.2,
                zorder=1,
            )
        )
        axis.add_patch(
            Polygon(
                regular_hexagon_photo(
                    centre_u, centre_v, geometry["socket_mouth_af"]
                ),
                closed=True,
                facecolor="white",
                edgecolor="#334155",
                linewidth=1.2,
                zorder=2,
            )
        )
        axis.text(
            centre_u,
            -18.0,
            size_entry["size"].upper(),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
        )

        # First plot inward directions, then points so arrow tails stay legible.
        for point in size_entry["points"]:
            point_u, point_v = point["photo_uv_mm"]
            delta_u = centre_u - point_u
            delta_v = centre_v - point_v
            style = PARTITION_STYLE[point["partition"]]
            axis.arrow(
                point_u,
                point_v,
                0.34 * delta_u,
                0.34 * delta_v,
                width=0.020,
                head_width=0.28,
                head_length=0.34,
                color=style["color"],
                alpha=0.52,
                length_includes_head=True,
                zorder=3,
            )
        for partition in ("clean_candidate", "holdout", "quarantine", "sentinel"):
            selected = [
                point
                for point in size_entry["points"]
                if point["partition"] == partition
            ]
            style = PARTITION_STYLE[partition]
            axis.scatter(
                [point["photo_uv_mm"][0] for point in selected],
                [point["photo_uv_mm"][1] for point in selected],
                marker=style["marker"],
                s=52 if partition != "sentinel" else 110,
                facecolor=(
                    "none"
                    if partition in ("holdout", "sentinel")
                    else style["color"]
                ),
                edgecolor=style["color"],
                linewidth=1.4,
                zorder=6 if partition == "sentinel" else 5,
            )

    axis.annotate(
        "+X$_F$ (left in photo)",
        xy=(39.0, 27.0),
        xytext=(69.0, 27.0),
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#0f172a"},
    )
    axis.annotate(
        "+Y$_F$",
        xy=(69.0, -21.0),
        xytext=(69.0, -7.0),
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#0f172a"},
    )
    axis.text(
        48.0,
        20.0,
        r"$\odot$ +Z$_F$ (out of table / toward viewer)",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=PARTITION_STYLE[key]["marker"],
            color="none",
            markerfacecolor=(
                "none" if key in ("holdout", "sentinel") else PARTITION_STYLE[key]["color"]
            ),
            markeredgecolor=PARTITION_STYLE[key]["color"],
            markersize=8 if key != "sentinel" else 11,
            label={
                "clean_candidate": "clean candidate (16/size)",
                "holdout": "holdout — never train (4/size)",
                "quarantine": "boundary quarantine (4/size)",
                "sentinel": "repeatability sentinel (2/size)",
            }[key],
        )
        for key in ("clean_candidate", "holdout", "quarantine", "sentinel")
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    axis.set_title(
        "CAD-seeded, metrology-rebased conical-probe plan — "
        "fixture top view from +Z$_F$\n"
        "display left → right: M12, M10, M8; arrows point toward each "
        "measured as-built centre at execution",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel("display rightward $u$ [mm], with $u=-x_F$")
    axis.set_ylabel("display upward $v_{up}$ [mm], with $v_{up}=-y_F$")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-79.0, 79.0)
    axis.set_ylim(-39.0, 39.0)
    axis.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.55)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.savefig(
        png_path,
        dpi=180,
        metadata={"Software": "generate_fixture_sampling_plan.py"},
    )
    figure.savefig(
        svg_path,
        metadata={
            "Date": None,
            "Creator": "generate_fixture_sampling_plan.py",
            "Title": "Fixture multi-size conical sampling top view",
        },
    )
    plt.close(figure)


def write_if_changed(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return
    path.write_bytes(encoded)


def check_outputs(plan_path: Path, csv_path: Path, png_path: Path, svg_path: Path) -> None:
    plan = build_plan()
    expected_plan = serialize_plan(plan)
    expected_csv = serialize_csv(plan)
    problems: list[str] = []
    if not plan_path.exists() or plan_path.read_text(encoding="utf-8") != expected_plan:
        problems.append(f"stale or missing {plan_path}")
    if not csv_path.exists() or csv_path.read_text(encoding="utf-8") != expected_csv:
        problems.append(f"stale or missing {csv_path}")
    for figure_path in (png_path, svg_path):
        if not figure_path.exists() or figure_path.stat().st_size == 0:
            problems.append(f"missing or empty {figure_path}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("PASS: plan/table are byte-current; figures exist; 78 attempts validated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that current JSON/CSV outputs match this generator",
    )
    args = parser.parse_args()

    plan_path = REAL_ROOT / "configs" / "fixture_multisize_conical_sampling_v1.json"
    csv_path = REAL_ROOT / "docs" / "fixture_multisize_sampling_points.csv"
    png_path = REAL_ROOT / "docs" / "fixture_multisize_sampling_top_view.png"
    svg_path = REAL_ROOT / "docs" / "fixture_multisize_sampling_top_view.svg"

    if args.check:
        check_outputs(plan_path, csv_path, png_path, svg_path)
        return

    plan = build_plan()
    write_if_changed(plan_path, serialize_plan(plan))
    write_if_changed(csv_path, serialize_csv(plan))
    render_figure(plan, png_path, svg_path)
    print(
        "Wrote deterministic offline plan: "
        f"{plan_path}, {csv_path}, {png_path}, {svg_path}"
    )
    print("Validated: 26 attempts/size, 78 total, 0 centre attempts.")


if __name__ == "__main__":
    main()
