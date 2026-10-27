#!/usr/bin/env python3
"""Generate a 26-point plan from the active printed CAD artifacts.

This is an offline-only utility.  It has no robot, ROS, KST, Bota, or network
interface.  The active STEP files and their generated CAD manifest are the
geometry authority; MuJoCo configurations are deliberately not inputs.

The contact-state label describes the upright touchdown footprint of the
selected printed tip face:

* supported_search: the complete tip footprint remains inside the head OD;
* double_overhang: it crosses the head OD and still overlaps the mouth;
* outer_partial_support: it crosses the head OD but clears the mouth while
  retaining a non-zero patch on the head top.

The 15-degree cone can deliberately traverse additional contact transitions.
Its complete printed chamfer/frustum is projected phase by phase
and recorded separately; those diagnostics never replace the upright
touchdown-state label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon


REAL_ROOT = Path(__file__).resolve().parents[1]
TOOLING_ROOT = REAL_ROOT / "model" / "parametric_tooling"
GENERATED_ROOT = TOOLING_ROOT / "generated"

DESIGN_PARAMETERS = TOOLING_ROOT / "design_parameters.json"
CAD_MANIFEST = GENERATED_ROOT / "manifest.json"
SOCKET_STEP = GENERATED_ROOT / "socket_base_hardware_easy_v2.step"

EXPECTED_REVISION = "v1.5-integrated-flange-tools-yaw-datum"
EXPECTED_SOCKET_STEP_SHA256 = (
    "f3b26582f409547e63296f7b8afb6f30d52d64f4f97f0dd5f74b4d1ffafab93f"
)

TARGET_SIZE = ""
TARGET_LABEL = ""
PLAN_NAME = ""
GENERATOR_NAME = ""
PEG_STEP = Path()
PLAN_PATH = Path()
CSV_PATH = Path()
PNG_PATH = Path()
SVG_PATH = Path()
OPERATOR_PNG_PATHS: tuple[Path, ...] = ()
EXPECTED_PEG_STEP_SHA256 = ""
EXPECTED_GEOMETRY: dict[str, float] = {}
NOMINAL_FIXTURE_CENTER_XY_MM = (0.0, 0.0)


def configure_target(size: str) -> None:
    """Select one printed tool size without changing the geometry algorithm."""

    global TARGET_SIZE, TARGET_LABEL, PLAN_NAME, GENERATOR_NAME
    global PEG_STEP, PLAN_PATH, CSV_PATH, PNG_PATH, SVG_PATH
    global OPERATOR_PNG_PATHS, EXPECTED_PEG_STEP_SHA256
    global EXPECTED_GEOMETRY, NOMINAL_FIXTURE_CENTER_XY_MM

    size = str(size).lower()
    if size == "m12":
        expected_peg_sha256 = (
            "362018e71e4bb80ea07640d6fb7bc0ddbe6178c25046f0283741e5a49e52a4fe"
        )
        expected_geometry = {
            "head_outer_diameter": 21.275,
            "socket_mouth_af": 13.0,
            "socket_deep_wall_af": 10.6,
            "socket_chamfer_width": 1.2,
            "socket_chamfer_depth": 1.2,
            "center_through_hole_diameter": 0.5,
            "peg_tip_face_af": 8.0,
            "peg_shaft_af": 10.0,
            "peg_chamfer_radial_setback": 1.0,
            "peg_chamfer_axial_length": 1.0,
        }
        nominal_center = (45.0, 0.0)
        operator_paths = (REAL_ROOT / "m12采样规划图.png",)
    elif size == "m10":
        expected_peg_sha256 = (
            "17381968212daa2b7758dcf3795df32b3a4e687b42c482d3550111e03ae8daa2"
        )
        expected_geometry = {
            "head_outer_diameter": 18.975,
            "socket_mouth_af": 10.54,
            "socket_deep_wall_af": 8.54,
            "socket_chamfer_width": 1.0,
            "socket_chamfer_depth": 1.0,
            "center_through_hole_diameter": 0.5,
            "peg_tip_face_af": 6.4,
            "peg_shaft_af": 8.0,
            "peg_chamfer_radial_setback": 0.8,
            "peg_chamfer_axial_length": 0.8,
        }
        nominal_center = (0.0, 0.0)
        operator_paths = (REAL_ROOT / "m10采样规划图.png",)
    else:
        raise ValueError(f"Unsupported printed sampling size {size}")

    TARGET_SIZE = size
    TARGET_LABEL = size.upper()
    PLAN_NAME = f"{size}_printed_cad_conical_sampling_v2"
    GENERATOR_NAME = f"generate_{size}_printed_cad_sampling_plan.py"
    PEG_STEP = GENERATED_ROOT / f"integrated_flange_peg_{size}.step"
    PLAN_PATH = REAL_ROOT / "configs" / f"{size}_printed_cad_conical_sampling_v2.json"
    CSV_PATH = REAL_ROOT / "docs" / f"{size}_printed_cad_sampling_points.csv"
    PNG_PATH = REAL_ROOT / "docs" / f"{size}_printed_cad_sampling_top_view.png"
    SVG_PATH = REAL_ROOT / "docs" / f"{size}_printed_cad_sampling_top_view.svg"
    OPERATOR_PNG_PATHS = operator_paths
    EXPECTED_PEG_STEP_SHA256 = expected_peg_sha256
    EXPECTED_GEOMETRY = expected_geometry
    NOMINAL_FIXTURE_CENTER_XY_MM = nominal_center


configure_target("m12")

PARTITION_STYLE = {
    "clean_candidate": {
        "marker": "o",
        "training_use": "eligible_only_after_complete_trace_and_offline_qc",
    },
    "holdout": {
        "marker": "s",
        "training_use": "never_train_reserved_for_final_real_alignment_audit",
    },
    "quarantine": {
        "marker": "X",
        "training_use": "forbidden_boundary_diagnostic_only",
    },
    "sentinel": {
        "marker": "*",
        "training_use": "repeatability_monitor_not_independent_training_data",
    },
}

STATE_STYLE = {
    "supported_search": {
        "color": "#16a34a",
        "title": "supported search (no outer overhang)",
    },
    "double_overhang": {
        "color": "#9333ea",
        "title": "double overhang",
    },
    "outer_partial_support": {
        "color": "#f97316",
        "title": "outer partial support",
    },
    "supported_double_boundary": {
        "color": "#dc2626",
        "title": "supported/double boundary",
    },
    "double_outer_boundary": {
        "color": "#be123c",
        "title": "double/outer boundary",
    },
}

# Clean candidates cover all three upright touchdown states around the actual
# sixfold printed tip.  The four outer-partial points deliberately contain two
# landing-mouth-clear points and two points where the complete tilted printed
# frustum also clears the mouth for every sampled cone phase.
CLEAN_SPECS = (
    (
        "supported_search",
        "landing_supported_cone_transition",
        (3.75, 63.75, 123.75, 183.75, 243.75, 303.75),
    ),
    (
        "double_overhang",
        "landing_double_overhang",
        (33.75, 93.75, 153.75, 213.75, 273.75, 333.75),
    ),
    (
        "outer_partial_support",
        "landing_mouth_clear_cone_transition",
        (18.75, 198.75),
    ),
    (
        "outer_partial_support",
        "full_frustum_mouth_clear",
        (108.75, 288.75),
    ),
)
HOLDOUT_SPECS = (
    ("supported_search", "landing_supported_cone_transition", 46.0),
    ("double_overhang", "landing_double_overhang", 136.0),
    (
        "outer_partial_support",
        "landing_mouth_clear_cone_transition",
        226.0,
    ),
    ("outer_partial_support", "full_frustum_mouth_clear", 316.0),
)
QUARANTINE_SPECS = (
    ("supported_double_boundary", "landing_boundary_diagnostic", 75.0),
    ("supported_double_boundary", "landing_boundary_diagnostic", 255.0),
    ("double_outer_boundary", "landing_boundary_diagnostic", 165.0),
    ("double_outer_boundary", "landing_boundary_diagnostic", 345.0),
)
SENTINEL_SPECS = (
    ("supported_search", "landing_supported_cone_transition", 3.75),
    (
        "outer_partial_support",
        "landing_mouth_clear_cone_transition",
        198.75,
    ),
)

TIP_YAW_IN_FIXTURE_DEG = 0.0
TIP_YAW_ACCEPTANCE_DEG = 1.0
CONE_TILT_DEG = 15.0
CONE_PHASE_BINS = 360
CONE_MOUTH_OVERLAP_EPSILON = 1e-6


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.relative_to(REAL_ROOT))


def source_record(path: Path) -> dict[str, str]:
    return {"path_from_real_device": relative_path(path), "sha256": sha256(path)}


def round_float(value: float, digits: int = 9) -> float:
    return round(float(value), digits)


def angle_token(angle_deg: float) -> str:
    return f"{angle_deg:06.2f}".replace(".", "p")


def signed_polygon_area(vertices: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
        - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
        for index in range(len(vertices))
    )


def polygon_area(vertices: Sequence[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    return abs(signed_polygon_area(vertices))


def counter_clockwise(
    vertices: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = list(vertices)
    if signed_polygon_area(output) < 0.0:
        output.reverse()
    return output


def convex_intersection(
    subject: Sequence[tuple[float, float]],
    clip: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman intersection for two convex polygons."""

    output = counter_clockwise(subject)
    clip_ccw = counter_clockwise(clip)
    tolerance = 1e-12

    for index, clip_start in enumerate(clip_ccw):
        clip_end = clip_ccw[(index + 1) % len(clip_ccw)]
        input_vertices = output
        output = []
        if not input_vertices:
            break

        def inside(point: tuple[float, float]) -> bool:
            return (
                (clip_end[0] - clip_start[0]) * (point[1] - clip_start[1])
                - (clip_end[1] - clip_start[1]) * (point[0] - clip_start[0])
                >= -tolerance
            )

        def intersection(
            start: tuple[float, float], end: tuple[float, float]
        ) -> tuple[float, float]:
            segment = (end[0] - start[0], end[1] - start[1])
            boundary = (
                clip_end[0] - clip_start[0],
                clip_end[1] - clip_start[1],
            )
            denominator = (
                segment[0] * boundary[1] - segment[1] * boundary[0]
            )
            if abs(denominator) < 1e-15:
                return end
            scale = (
                (clip_start[0] - start[0]) * boundary[1]
                - (clip_start[1] - start[1]) * boundary[0]
            ) / denominator
            return (
                start[0] + scale * segment[0],
                start[1] + scale * segment[1],
            )

        previous = input_vertices[-1]
        for current in input_vertices:
            current_inside = inside(current)
            previous_inside = inside(previous)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
    return output


def regular_hexagon(
    across_flats_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    flat_normal_yaw_deg: float = 0.0,
) -> list[tuple[float, float]]:
    radius = across_flats_mm / math.sqrt(3.0)
    return [
        (
            center_xy_mm[0]
            + radius
            * math.cos(math.radians(30.0 + flat_normal_yaw_deg + 60.0 * index)),
            center_xy_mm[1]
            + radius
            * math.sin(math.radians(30.0 + flat_normal_yaw_deg + 60.0 * index)),
        )
        for index in range(6)
    ]


def convex_hull(
    vertices: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the counter-clockwise convex hull without a repeated endpoint."""

    points = sorted(set(vertices))
    if len(points) <= 1:
        return points

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def rotate_cone_phase(
    point_xyz: tuple[float, float, float],
    phase_deg: float,
    tilt_deg: float = CONE_TILT_DEG,
) -> tuple[float, float, float]:
    """Apply Rz(-phase) * Ry(tilt) * Rz(phase) to a tip-frame point."""

    x_value, y_value, z_value = point_xyz
    phase = math.radians(phase_deg)
    tilt = math.radians(tilt_deg)

    first_x = math.cos(phase) * x_value - math.sin(phase) * y_value
    first_y = math.sin(phase) * x_value + math.cos(phase) * y_value
    first_z = z_value

    second_x = math.cos(tilt) * first_x + math.sin(tilt) * first_z
    second_y = first_y
    second_z = -math.sin(tilt) * first_x + math.cos(tilt) * first_z

    output_x = math.cos(phase) * second_x + math.sin(phase) * second_y
    output_y = -math.sin(phase) * second_x + math.cos(phase) * second_y
    return output_x, output_y, second_z


def circle_polygon(radius_mm: float, count: int = 720) -> list[tuple[float, float]]:
    return [
        (
            radius_mm * math.cos(2.0 * math.pi * index / count),
            radius_mm * math.sin(2.0 * math.pi * index / count),
        )
        for index in range(count)
    ]


class ContactGeometry:
    def __init__(self, geometry_mm: dict[str, float]):
        self.geometry = geometry_mm
        self.head = circle_polygon(0.5 * geometry_mm["head_outer_diameter"])
        self.mouth = regular_hexagon(geometry_mm["socket_mouth_af"])
        self.tip_area = polygon_area(
            regular_hexagon(geometry_mm["peg_tip_face_af"])
        )
        bottom = [
            (x_value, y_value, 0.0)
            for x_value, y_value in regular_hexagon(
                geometry_mm["peg_tip_face_af"]
            )
        ]
        top = [
            (
                x_value,
                y_value,
                geometry_mm["peg_chamfer_axial_length"],
            )
            for x_value, y_value in regular_hexagon(
                geometry_mm["peg_shaft_af"]
            )
        ]
        self.cone_phase_polygons = []
        for phase_index in range(CONE_PHASE_BINS):
            phase_deg = 360.0 * phase_index / CONE_PHASE_BINS
            projected = [
                rotate_cone_phase(vertex, phase_deg)[:2]
                for vertex in bottom + top
            ]
            self.cone_phase_polygons.append(convex_hull(projected))

    def metrics(
        self,
        radius_mm: float,
        angle_deg: float,
        peg_yaw_deg: float = TIP_YAW_IN_FIXTURE_DEG,
    ) -> dict[str, float]:
        theta = math.radians(angle_deg)
        center = (
            radius_mm * math.cos(theta),
            radius_mm * math.sin(theta),
        )
        tip = regular_hexagon(
            self.geometry["peg_tip_face_af"], center, peg_yaw_deg
        )
        inside_head = polygon_area(convex_intersection(tip, self.head))
        inside_mouth = polygon_area(convex_intersection(tip, self.mouth))
        outer = max(0.0, 1.0 - inside_head / self.tip_area)
        mouth = max(0.0, inside_mouth / self.tip_area)
        support = max(0.0, 1.0 - outer - mouth)
        return {
            "outer_overhang_fraction": outer,
            "mouth_overlap_fraction": mouth,
            "head_top_support_fraction": support,
        }

    def solve_radius(
        self,
        angle_deg: float,
        field: str,
        target: float,
        *,
        decreasing: bool = False,
    ) -> float:
        lower = 0.0
        upper = (
            self.geometry["head_outer_diameter"]
            + 2.0 * self.geometry["peg_tip_face_af"]
        )
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            value = self.metrics(middle, angle_deg)[field]
            if (not decreasing and value < target) or (
                decreasing and value > target
            ):
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def cone_phase_diagnostics(
        self, radius_mm: float, angle_deg: float
    ) -> dict[str, float | int]:
        theta = math.radians(angle_deg)
        center = (
            radius_mm * math.cos(theta),
            radius_mm * math.sin(theta),
        )
        mouth_overlap_fractions = []
        outer_overhang_flags = []
        outer_clearances = []
        projected_areas = []
        head_radius = 0.5 * self.geometry["head_outer_diameter"]
        for local_polygon in self.cone_phase_polygons:
            polygon = [
                (vertex[0] + center[0], vertex[1] + center[1])
                for vertex in local_polygon
            ]
            area = polygon_area(polygon)
            projected_areas.append(area)
            mouth_overlap_fractions.append(
                polygon_area(convex_intersection(polygon, self.mouth)) / area
            )
            clearance = min(
                head_radius - math.hypot(vertex[0], vertex[1])
                for vertex in polygon
            )
            outer_clearances.append(clearance)
            outer_overhang_flags.append(clearance < -1e-9)
        return {
            "phase_bins": CONE_PHASE_BINS,
            "maximum_mouth_overlap_fraction": max(mouth_overlap_fractions),
            "minimum_mouth_overlap_fraction": min(mouth_overlap_fractions),
            "mouth_overlap_phase_fraction": sum(
                value > CONE_MOUTH_OVERLAP_EPSILON
                for value in mouth_overlap_fractions
            )
            / CONE_PHASE_BINS,
            "outer_overhang_phase_fraction": sum(outer_overhang_flags)
            / CONE_PHASE_BINS,
            "minimum_outer_clearance_mm": min(outer_clearances),
            "minimum_projected_area_mm2": min(projected_areas),
            "maximum_projected_area_mm2": max(projected_areas),
        }

    def solve_frustum_outer_tangent(self, angle_deg: float) -> float:
        lower = 0.0
        upper = (
            self.geometry["head_outer_diameter"]
            + 2.0 * self.geometry["peg_shaft_af"]
        )
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            diagnostics = self.cone_phase_diagnostics(middle, angle_deg)
            if diagnostics["minimum_outer_clearance_mm"] >= 0.0:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def solve_frustum_mouth_clear(self, angle_deg: float) -> float:
        lower = 0.0
        upper = (
            self.geometry["head_outer_diameter"]
            + 2.0 * self.geometry["peg_shaft_af"]
        )
        for _ in range(52):
            middle = 0.5 * (lower + upper)
            diagnostics = self.cone_phase_diagnostics(middle, angle_deg)
            if (
                diagnostics["maximum_mouth_overlap_fraction"]
                > CONE_MOUTH_OVERLAP_EPSILON
            ):
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def radius_for_state(
        self, state: str, subtype: str, angle_deg: float
    ) -> tuple[float, str]:
        if state == "supported_search":
            landing_tangent = self.solve_radius(
                angle_deg, "outer_overhang_fraction", 1e-6
            )
            frustum_tangent = self.solve_frustum_outer_tangent(angle_deg)
            return (
                0.5 * (landing_tangent + frustum_tangent),
                "midpoint of exact full-frustum and tip-face landing outer tangencies",
            )
        if state == "double_overhang":
            landing_tangent = self.solve_radius(
                angle_deg, "outer_overhang_fraction", 1e-6
            )
            landing_mouth_clear = self.solve_radius(
                angle_deg,
                "mouth_overlap_fraction",
                1e-6,
                decreasing=True,
            )
            return (
                0.5 * (landing_tangent + landing_mouth_clear),
                "midpoint of exact tip-face outer tangency and mouth-clear boundaries",
            )
        if state == "outer_partial_support":
            landing_mouth_clear = self.solve_radius(
                angle_deg,
                "mouth_overlap_fraction",
                1e-6,
                decreasing=True,
            )
            if subtype == "landing_mouth_clear_cone_transition":
                return (
                    landing_mouth_clear + 0.60,
                    "tip-face landing mouth-clear boundary plus 0.60 mm",
                )
            if subtype == "full_frustum_mouth_clear":
                return (
                    self.solve_frustum_mouth_clear(angle_deg) + 0.35,
                    "complete 15deg printed-frustum mouth-clear boundary plus 0.35 mm",
                )
            raise ValueError(f"Unknown outer-partial subtype {subtype}")
        if state == "supported_double_boundary":
            return (
                self.solve_radius(
                    angle_deg, "outer_overhang_fraction", 0.02
                ),
                "2 percent tip area beyond head OD boundary diagnostic",
            )
        if state == "double_outer_boundary":
            return (
                self.solve_radius(
                    angle_deg,
                    "mouth_overlap_fraction",
                    0.02,
                    decreasing=True,
                ),
                "2 percent residual mouth overlap boundary diagnostic",
            )
        raise ValueError(f"Unknown contact state {state}")

    def yaw_envelope(
        self, radius_mm: float, angle_deg: float
    ) -> dict[str, list[float]]:
        samples = [
            self.metrics(radius_mm, angle_deg, yaw)
            for yaw in (
                -TIP_YAW_ACCEPTANCE_DEG,
                0.0,
                TIP_YAW_ACCEPTANCE_DEG,
            )
        ]
        return {
            key: [
                min(sample[key] for sample in samples),
                max(sample[key] for sample in samples),
            ]
            for key in samples[0]
        }


def load_printed_geometry() -> tuple[dict[str, Any], dict[str, float]]:
    design = read_json(DESIGN_PARAMETERS)
    manifest = read_json(CAD_MANIFEST)
    if design["design_revision"] != EXPECTED_REVISION:
        raise ValueError("The active printed-tool design revision changed")
    if manifest["design_revision"] != EXPECTED_REVISION:
        raise ValueError("The generated CAD manifest revision changed")
    if manifest["recommended_hardware_socket_profile"] != "hardware_easy_v2":
        raise ValueError("The active socket profile is no longer hardware_easy_v2")
    if manifest["recommended_hardware_peg_tip_profile"] != "hardware_easy_v2":
        raise ValueError("The active peg tip profile is no longer hardware_easy_v2")
    if sha256(PEG_STEP) != EXPECTED_PEG_STEP_SHA256:
        raise ValueError(f"The active {TARGET_LABEL} peg STEP hash changed")
    if sha256(SOCKET_STEP) != EXPECTED_SOCKET_STEP_SHA256:
        raise ValueError("The active socket STEP hash changed")

    socket = manifest["socket_profile_validation"][
        "socket_base_hardware_easy_v2"
    ]["sizes"][TARGET_SIZE]
    socket_target = socket["target_geometry_mm"]
    tip = manifest["integrated_tool_validation"][TARGET_SIZE][
        "target_tip_geometry_mm"
    ]
    through_hole = design["socket_base"]["center_through_hole_diameter_mm"]
    geometry = {
        "head_outer_diameter": float(socket_target["head_outer_diameter_mm"]),
        "socket_mouth_af": float(socket_target["top_af_mm"]),
        "socket_deep_wall_af": float(socket_target["wall_af_mm"]),
        "socket_chamfer_width": float(socket_target["entry_chamfer_width_mm"]),
        "socket_chamfer_depth": float(socket_target["entry_chamfer_depth_mm"]),
        "center_through_hole_diameter": float(through_hole),
        "peg_tip_face_af": float(tip["tip_face_af_mm"]),
        "peg_shaft_af": float(tip["shaft_af_mm"]),
        "peg_chamfer_radial_setback": float(
            tip["tip_chamfer_radial_setback_mm"]
        ),
        "peg_chamfer_axial_length": float(
            tip["tip_chamfer_axial_length_mm"]
        ),
        "peg_tip_face_flat_normal_fixture_deg": TIP_YAW_IN_FIXTURE_DEG,
        "peg_selected_flat_normal_in_flange_cad_deg": float(
            manifest["peg_flange_yaw_contract"][
                "selected_peg_flat_normal_in_flange_deg"
            ]
        ),
        "socket_selected_flat_normal_fixture_deg": 0.0,
    }
    for key, value in EXPECTED_GEOMETRY.items():
        if not math.isclose(geometry[key], value, abs_tol=1e-9):
            raise ValueError(f"Printed CAD {key}={geometry[key]}, expected {value}")

    sources = {
        "design_parameters": source_record(DESIGN_PARAMETERS),
        "generated_cad_manifest": source_record(CAD_MANIFEST),
        f"printed_{TARGET_SIZE}_integrated_peg_step": source_record(PEG_STEP),
        "printed_hardware_easy_v2_socket_step": source_record(SOCKET_STEP),
    }
    return sources, geometry


def classify_nominal(metrics: dict[str, float]) -> str:
    outer = metrics["outer_overhang_fraction"]
    mouth = metrics["mouth_overlap_fraction"]
    if outer <= 1e-5:
        return "supported_search"
    if mouth > 1e-5:
        return "double_overhang"
    return "outer_partial_support"


def add_point(
    points: list[dict[str, Any]],
    geometry: ContactGeometry,
    *,
    partition: str,
    state: str,
    cone_sweep_subtype: str,
    angle_deg: float,
    duplicate_of: str | None = None,
) -> None:
    radius_mm, radius_rule = geometry.radius_for_state(
        state, cone_sweep_subtype, angle_deg
    )
    theta = math.radians(angle_deg)
    local_x = radius_mm * math.cos(theta)
    local_y = radius_mm * math.sin(theta)
    fixture_x = NOMINAL_FIXTURE_CENTER_XY_MM[0] + local_x
    fixture_y = NOMINAL_FIXTURE_CENTER_XY_MM[1] + local_y
    nominal_metrics = geometry.metrics(radius_mm, angle_deg)
    envelope = geometry.yaw_envelope(radius_mm, angle_deg)
    cone_diagnostics = geometry.cone_phase_diagnostics(radius_mm, angle_deg)
    state_token = {
        "supported_search": "SUPPORTED",
        "double_overhang": "DOUBLE",
        "outer_partial_support": "OUTER",
        "supported_double_boundary": "BOUNDARY_SD",
        "double_outer_boundary": "BOUNDARY_DO",
    }[state]
    sample_id = (
        f"{TARGET_LABEL}-{partition.upper()}-{state_token}-A{angle_token(angle_deg)}"
    )
    point = {
        "attempt_index": len(points) + 1,
        "sample_id": sample_id,
        "partition": partition,
        "training_use": PARTITION_STYLE[partition]["training_use"],
        "contact_state": state,
        "cone_sweep_subtype": cone_sweep_subtype,
        "state_definition_phase": "UPRIGHT_TOUCHDOWN_BEFORE_CONICAL_SCAN",
        "radius_rule": radius_rule,
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
        "nominal_touchdown_geometry": {
            key: round_float(value) for key, value in nominal_metrics.items()
        },
        "touchdown_geometry_over_plus_minus_1deg_peg_yaw": {
            key: [round_float(value[0]), round_float(value[1])]
            for key, value in envelope.items()
        },
        "printed_frustum_cone_phase_diagnostics": {
            key: (
                int(value)
                if key == "phase_bins"
                else round_float(float(value))
            )
            for key, value in cone_diagnostics.items()
        },
        "duplicate_of": duplicate_of,
    }
    if partition != "quarantine":
        actual = classify_nominal(nominal_metrics)
        if actual != state:
            raise AssertionError(f"{sample_id}: expected {state}, computed {actual}")
    points.append(point)


def closed_route_distance(
    order: Sequence[int], points: Sequence[dict[str, Any]]
) -> float:
    coordinates = [(0.0, 0.0)] + [
        tuple(points[index - 1]["local_offset_fixture_mm"]) for index in order
    ] + [(0.0, 0.0)]
    return sum(
        math.dist(start, end) for start, end in zip(coordinates, coordinates[1:])
    )


def two_opt(
    order: list[int], points: Sequence[dict[str, Any]]
) -> list[int]:
    best = order[:]
    best_distance = closed_route_distance(best, points)
    improved = True
    while improved:
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + list(reversed(best[left:right])) + best[right:]
                distance = closed_route_distance(candidate, points)
                if distance < best_distance - 1e-9:
                    best = candidate
                    best_distance = distance
                    improved = True
                    break
            if improved:
                break
    return best


def sentinel_separations(
    order: Sequence[int], points: Sequence[dict[str, Any]]
) -> list[int]:
    positions = {attempt: index for index, attempt in enumerate(order, start=1)}
    by_id = {point["sample_id"]: point["attempt_index"] for point in points}
    output = []
    for point in points:
        if point["partition"] == "sentinel":
            output.append(
                abs(
                    positions[point["attempt_index"]]
                    - positions[by_id[point["duplicate_of"]]]
                )
            )
    return output


def optimize_route(points: Sequence[dict[str, Any]]) -> list[int]:
    base = [
        point["attempt_index"]
        for point in points
        if point["partition"] != "sentinel"
    ]
    sentinels = [
        point["attempt_index"]
        for point in points
        if point["partition"] == "sentinel"
    ]
    best_base: list[int] | None = None
    best_base_distance = math.inf
    coordinate = {
        point["attempt_index"]: tuple(point["local_offset_fixture_mm"])
        for point in points
    }
    for first in base:
        remaining = set(base)
        remaining.remove(first)
        route = [first]
        current = coordinate[first]
        while remaining:
            next_attempt = min(
                remaining,
                key=lambda attempt: (
                    math.dist(current, coordinate[attempt]),
                    attempt,
                ),
            )
            route.append(next_attempt)
            remaining.remove(next_attempt)
            current = coordinate[next_attempt]
        route = two_opt(route, points)
        distance = closed_route_distance(route, points)
        if distance < best_base_distance:
            best_base = route
            best_base_distance = distance
    assert best_base is not None

    best: list[int] | None = None
    best_distance = math.inf
    for first_position in range(len(best_base) + 1):
        one = best_base[:]
        one.insert(first_position, sentinels[0])
        for second_position in range(len(one) + 1):
            candidate = one[:]
            candidate.insert(second_position, sentinels[1])
            if min(sentinel_separations(candidate, points)) < 10:
                continue
            distance = closed_route_distance(candidate, points)
            if distance < best_distance:
                best = candidate
                best_distance = distance
    if best is None:
        raise AssertionError("No route satisfies the sentinel spacing contract")

    improved = True
    while improved:
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + list(reversed(best[left:right])) + best[right:]
                if min(sentinel_separations(candidate, points)) < 10:
                    continue
                distance = closed_route_distance(candidate, points)
                if distance < best_distance - 1e-9:
                    best = candidate
                    best_distance = distance
                    improved = True
                    break
            if improved:
                break

    # Escape the deterministic 2-opt local minimum without introducing a
    # runtime dependency.  The fixed seed makes checked-in outputs stable.
    rng = random.Random(20260727)
    current = best[:]
    current_distance = closed_route_distance(current, points)
    annealed_best = current[:]
    annealed_best_distance = current_distance
    move_count = 120_000
    for move_index in range(move_count):
        candidate = current[:]
        left, right = sorted(rng.sample(range(len(candidate)), 2))
        operation = move_index % 3
        if operation == 0:
            candidate[left], candidate[right] = (
                candidate[right],
                candidate[left],
            )
        elif operation == 1:
            candidate[left : right + 1] = reversed(
                candidate[left : right + 1]
            )
        else:
            moved = candidate.pop(left)
            candidate.insert(right, moved)
        if min(sentinel_separations(candidate, points)) < 10:
            continue
        candidate_distance = closed_route_distance(candidate, points)
        fraction = move_index / max(move_count - 1, 1)
        temperature = 3.0 * (0.002 / 3.0) ** fraction
        if candidate_distance < current_distance or rng.random() < math.exp(
            (current_distance - candidate_distance) / temperature
        ):
            current = candidate
            current_distance = candidate_distance
            if current_distance < annealed_best_distance:
                annealed_best = current[:]
                annealed_best_distance = current_distance
    return annealed_best


def build_plan() -> dict[str, Any]:
    sources, geometry_mm = load_printed_geometry()
    geometry = ContactGeometry(geometry_mm)
    points: list[dict[str, Any]] = []

    for state, cone_sweep_subtype, angles in CLEAN_SPECS:
        for angle in angles:
            add_point(
                points,
                geometry,
                partition="clean_candidate",
                state=state,
                cone_sweep_subtype=cone_sweep_subtype,
                angle_deg=angle,
            )
    for state, cone_sweep_subtype, angle in HOLDOUT_SPECS:
        add_point(
            points,
            geometry,
            partition="holdout",
            state=state,
            cone_sweep_subtype=cone_sweep_subtype,
            angle_deg=angle,
        )
    for state, cone_sweep_subtype, angle in QUARANTINE_SPECS:
        add_point(
            points,
            geometry,
            partition="quarantine",
            state=state,
            cone_sweep_subtype=cone_sweep_subtype,
            angle_deg=angle,
        )
    for state, cone_sweep_subtype, angle in SENTINEL_SPECS:
        source = next(
            point
            for point in points
            if point["partition"] == "clean_candidate"
            and point["contact_state"] == state
            and point["cone_sweep_subtype"] == cone_sweep_subtype
            and math.isclose(point["angle_fixture_deg"], angle, abs_tol=1e-12)
        )
        add_point(
            points,
            geometry,
            partition="sentinel",
            state=state,
            cone_sweep_subtype=cone_sweep_subtype,
            angle_deg=angle,
            duplicate_of=source["sample_id"],
        )

    order = optimize_route(points)
    route_distance = closed_route_distance(order, points)
    original_distance = closed_route_distance(
        [point["attempt_index"] for point in points], points
    )
    separations = sentinel_separations(order, points)
    max_radius = max(point["radius_mm"] for point in points)
    required_lateral = math.ceil((max_radius + 0.25 + 0.20 + 0.25) * 10.0) / 10.0
    cone_projection_radius = (
        geometry_mm["peg_shaft_af"] / math.sqrt(3.0)
        + geometry_mm["peg_chamfer_axial_length"]
        * math.sin(math.radians(CONE_TILT_DEG))
    )

    counts = {
        partition: sum(point["partition"] == partition for point in points)
        for partition in PARTITION_STYLE
    }
    clean_state_counts = {
        state: sum(
            point["partition"] == "clean_candidate"
            and point["contact_state"] == state
            for point in points
        )
        for state in (
            "supported_search",
            "double_overhang",
            "outer_partial_support",
        )
    }
    clean_cone_subtype_counts = {
        subtype: sum(
            point["partition"] == "clean_candidate"
            and point["cone_sweep_subtype"] == subtype
            for point in points
        )
        for subtype in (
            "landing_supported_cone_transition",
            "landing_double_overhang",
            "landing_mouth_clear_cone_transition",
            "full_frustum_mouth_clear",
        )
    }

    plan = {
        "schema_version": 2,
        "name": PLAN_NAME,
        "status": "PRINTED_CAD_DERIVED_REQUIRES_RUN_LOCAL_CENTER_AND_YAW_COMMISSIONING",
        "claim_boundary": {
            "purpose": (
                f"{TARGET_LABEL} real-device 26-point conical fingerprint "
                "acquisition"
            ),
            "no_centre_attempt": True,
            "cad_not_as_built_metrology": (
                "STEP geometry is the printed design authority; every physical "
                "print still requires instance IDs, dimensional inspection, "
                "run-local centre teaching, and peg-to-fixture yaw confirmation"
            ),
            "state_label_phase": "upright touchdown before the 15 degree cone",
            "cone_transition_policy": (
                "the cone may cross additional contact boundaries; the complete "
                "printed tip-to-shaft frustum is evaluated at 360 command phases "
                "and kept as separate per-point metadata"
            ),
            "data_governance": (
                "real raw Bota and measured touchdown dominate; clean remains "
                "pending offline QC; holdout, quarantine, and sentinel never train"
            ),
        },
        "coordinate_contract": {
            "fixture_frame": {
                "name": "F",
                f"nominal_{TARGET_SIZE}_center_xy_mm": list(
                    NOMINAL_FIXTURE_CENTER_XY_MM
                ),
                "angle_definition": "counter-clockwise from +X_F toward +Y_F",
                "execution_rebase": (
                    f"confirmed run-start TCP XY is the {TARGET_LABEL} centre; fixture XY "
                    "is aligned with KUKA WORLD XY for this campaign"
                ),
            },
            "photo_top_view": {
                "mapping": "u_photo_mm = -x_fixture_mm; v_photo_up_mm = -y_fixture_mm",
                "display_axis_mapping": "u_right_mm = -x_fixture_mm; v_up_mm = -y_fixture_mm",
                "plus_x_fixture_points": "left",
                "plus_y_fixture_points": "down",
                "plus_z_fixture_points": "toward viewer",
            },
            "yaw_contract": {
                "printed_peg_selected_flat_normal_in_flange_cad_deg": (
                    geometry_mm["peg_selected_flat_normal_in_flange_cad_deg"]
                ),
                "commissioned_peg_flat_normal_in_fixture_deg": (
                    TIP_YAW_IN_FIXTURE_DEG
                ),
                "socket_flat_normal_in_fixture_deg": 0.0,
                "acceptance_deg": TIP_YAW_ACCEPTANCE_DEG,
                "rule": (
                    "teach the printed flange +45 degree tangent datum parallel "
                    "to +X_F, making the selected peg flat normal 0 degrees in F"
                ),
            },
        },
        "source_files": sources,
        "geometry_mm": {
            key: round_float(value) for key, value in geometry_mm.items()
        },
        "contact_state_definition": {
            "footprint": (
                f"actual printed upright {TARGET_LABEL} tip face "
                f"AF{geometry_mm['peg_tip_face_af']:g} regular hexagon"
            ),
            "support_surface": (
                "head OD disk minus the "
                f"AF{geometry_mm['socket_mouth_af']:g} mouth"
            ),
            "supported_search": "outer_overhang_fraction <= 1e-5",
            "double_overhang": (
                "outer_overhang_fraction > 1e-5 and mouth_overlap_fraction > 1e-5"
            ),
            "outer_partial_support": (
                "outer_overhang_fraction > 1e-5, mouth_overlap_fraction <= 1e-5, "
                "and head_top_support_fraction > 0"
            ),
            "area_method": (
                "deterministic convex polygon clipping; head circle uses a "
                "720-edge inscribed polygon"
            ),
        },
        "cone_command_geometry": {
            "tilt_deg": CONE_TILT_DEG,
            "phase_bins": CONE_PHASE_BINS,
            "peg_shaft_af_mm": geometry_mm["peg_shaft_af"],
            "peg_chamfer_axial_length_mm": geometry_mm[
                "peg_chamfer_axial_length"
            ],
            "conservative_xy_projection_radius_bound_mm": round_float(
                cone_projection_radius
            ),
            "semantics": (
                "conservative drawing bound only; quantitative diagnostics use "
                "conv(xy(Rz(-phi) Ry(15deg) Rz(phi) "
                f"[AF{geometry_mm['peg_tip_face_af']:g}@z0, "
                f"AF{geometry_mm['peg_shaft_af']:g}@z1])) "
                "at every phase and are not touchdown-state labels"
            ),
            "rotation_convention": "R(phi)=Rz(-phi)*Ry(15deg)*Rz(phi)",
            "phase_grid": "360 uniform samples on [0,360deg), matching the recorded phase-bin contract",
            "diagnostic_area_normalization": (
                "mouth intersection area divided by that phase's projected "
                "printed-frustum convex-hull area"
            ),
            "mouth_overlap_epsilon": CONE_MOUTH_OVERLAP_EPSILON,
        },
        "sampling_design": {
            "attempt_count": 26,
            "partition_counts": counts,
            "clean_contact_state_counts": clean_state_counts,
            "clean_cone_sweep_subtype_counts": clean_cone_subtype_counts,
            "sentinel_semantics": (
                "exact repeats of one supported and one landing-mouth-clear "
                "outer-partial clean point"
            ),
            "boundary_quarantine_semantics": (
                "two supported/double and two double/outer near-boundary diagnostics"
            ),
        },
        "fixture_centre_xy_mm": list(NOMINAL_FIXTURE_CENTER_XY_MM),
        "attempt_count": 26,
        "points": points,
        "execution": {
            "attempt_indices": order,
            "closed_route_distance_mm": round_float(route_distance),
            "source_order_closed_route_distance_mm": round_float(original_distance),
            "sentinel_duplicate_execution_separations": separations,
            "route_solver": (
                "deterministic multistart nearest-neighbour plus 2-opt, "
                "best constrained sentinel insertion, constrained 2-opt, "
                "and fixed-seed constrained annealing"
            ),
            "start_and_finish": (
                f"run-local {TARGET_LABEL} centre at clear-air height"
            ),
        },
        "motion_contract": {
            "maximum_planned_radius_mm": round_float(max_radius),
            "measured_landing_tolerance_mm": 0.25,
            "air_tracking_tolerance_mm": 0.20,
            "additional_workspace_margin_mm": 0.25,
            "required_sampling_only_lateral_excursion_mm": required_lateral,
        },
        "validation": {
            "actual_printed_cad_hashes_verified": True,
            "no_simulation_geometry_input": True,
            "all_three_clean_contact_states_present": all(
                count > 0 for count in clean_state_counts.values()
            ),
            "outer_partial_clean_count": clean_state_counts[
                "outer_partial_support"
            ],
            "sentinel_minimum_execution_separation": min(separations),
            "route_improvement_fraction": round_float(
                1.0 - route_distance / original_distance
            ),
        },
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    points = plan["points"]
    if len(points) != 26:
        raise AssertionError(
            f"{TARGET_LABEL} plan must contain exactly 26 attempts"
        )
    if [point["attempt_index"] for point in points] != list(range(1, 27)):
        raise AssertionError("attempt_index must match point-array order")
    if len({point["sample_id"] for point in points}) != 26:
        raise AssertionError("sample IDs must be unique")
    order = plan["execution"]["attempt_indices"]
    if sorted(order) != list(range(1, 27)):
        raise AssertionError("execution order must be a permutation of 1:26")
    if min(plan["execution"]["sentinel_duplicate_execution_separations"]) < 10:
        raise AssertionError("sentinel repeats must be separated by >=10 attempts")
    if plan["sampling_design"]["partition_counts"] != {
        "clean_candidate": 16,
        "holdout": 4,
        "quarantine": 4,
        "sentinel": 2,
    }:
        raise AssertionError("partition counts changed")
    if plan["sampling_design"]["clean_contact_state_counts"] != {
        "supported_search": 6,
        "double_overhang": 6,
        "outer_partial_support": 4,
    }:
        raise AssertionError("clean contact-state counts changed")
    if plan["sampling_design"]["clean_cone_sweep_subtype_counts"] != {
        "landing_supported_cone_transition": 6,
        "landing_double_overhang": 6,
        "landing_mouth_clear_cone_transition": 2,
        "full_frustum_mouth_clear": 2,
    }:
        raise AssertionError("clean cone-sweep subtype counts changed")
    for point in points:
        radius = math.hypot(*point["local_offset_fixture_mm"])
        if not math.isclose(radius, point["radius_mm"], abs_tol=2e-8):
            raise AssertionError(f"{point['sample_id']}: radius mismatch")
        direction = point["true_direction_to_centre_fixture_unit"]
        radial = point["local_offset_fixture_mm"]
        if (
            radial[0] * direction[0] + radial[1] * direction[1]
            >= -0.999999 * radius
        ):
            raise AssertionError(f"{point['sample_id']}: inward label mismatch")
        nominal = point["nominal_touchdown_geometry"]
        cone = point["printed_frustum_cone_phase_diagnostics"]
        if cone["phase_bins"] != CONE_PHASE_BINS:
            raise AssertionError(
                f"{point['sample_id']}: cone phase count changed"
            )
        if point["contact_state"] == "outer_partial_support":
            if (
                nominal["mouth_overlap_fraction"] > 1e-5
                or nominal["outer_overhang_fraction"] <= 0.5
                or nominal["head_top_support_fraction"] <= 0.2
            ):
                raise AssertionError(
                    f"{point['sample_id']}: weak outer-partial margin"
                )
            if point["cone_sweep_subtype"] == "full_frustum_mouth_clear":
                if (
                    cone["maximum_mouth_overlap_fraction"]
                    > CONE_MOUTH_OVERLAP_EPSILON
                ):
                    raise AssertionError(
                        f"{point['sample_id']}: full frustum still overlaps mouth"
                    )
            elif (
                point["partition"] != "quarantine"
                and cone["maximum_mouth_overlap_fraction"]
                <= CONE_MOUTH_OVERLAP_EPSILON
            ):
                raise AssertionError(
                    f"{point['sample_id']}: transition subtype has no mouth phase"
                )
        if point["contact_state"] == "double_overhang":
            if (
                nominal["mouth_overlap_fraction"] <= 0.05
                or nominal["outer_overhang_fraction"] <= 0.10
                or nominal["head_top_support_fraction"] <= 0.20
            ):
                raise AssertionError(
                    f"{point['sample_id']}: weak double-overhang margin"
                )
        if point["contact_state"] == "supported_search":
            if nominal["outer_overhang_fraction"] > 1e-5:
                raise AssertionError(
                    f"{point['sample_id']}: supported point crosses OD"
                )


def serialize_csv(plan: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fields = [
        "attempt_index",
        "execution_index",
        "sample_id",
        "partition",
        "contact_state",
        "cone_sweep_subtype",
        "radius_mm",
        "angle_fixture_deg",
        "local_x_fixture_mm",
        "local_y_fixture_mm",
        "outer_overhang_fraction",
        "mouth_overlap_fraction",
        "head_top_support_fraction",
        "cone_maximum_mouth_overlap_fraction",
        "cone_mouth_overlap_phase_fraction",
        "cone_outer_overhang_phase_fraction",
        "cone_minimum_outer_clearance_mm",
        "duplicate_of",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    execution_position = {
        attempt: index
        for index, attempt in enumerate(
            plan["execution"]["attempt_indices"], start=1
        )
    }
    for point in plan["points"]:
        metrics = point["nominal_touchdown_geometry"]
        cone = point["printed_frustum_cone_phase_diagnostics"]
        writer.writerow(
            {
                "attempt_index": point["attempt_index"],
                "execution_index": execution_position[point["attempt_index"]],
                "sample_id": point["sample_id"],
                "partition": point["partition"],
                "contact_state": point["contact_state"],
                "cone_sweep_subtype": point["cone_sweep_subtype"],
                "radius_mm": f"{point['radius_mm']:.9f}",
                "angle_fixture_deg": f"{point['angle_fixture_deg']:.2f}",
                "local_x_fixture_mm": (
                    f"{point['local_offset_fixture_mm'][0]:.9f}"
                ),
                "local_y_fixture_mm": (
                    f"{point['local_offset_fixture_mm'][1]:.9f}"
                ),
                "outer_overhang_fraction": (
                    f"{metrics['outer_overhang_fraction']:.9f}"
                ),
                "mouth_overlap_fraction": (
                    f"{metrics['mouth_overlap_fraction']:.9f}"
                ),
                "head_top_support_fraction": (
                    f"{metrics['head_top_support_fraction']:.9f}"
                ),
                "cone_maximum_mouth_overlap_fraction": (
                    f"{cone['maximum_mouth_overlap_fraction']:.9f}"
                ),
                "cone_mouth_overlap_phase_fraction": (
                    f"{cone['mouth_overlap_phase_fraction']:.9f}"
                ),
                "cone_outer_overhang_phase_fraction": (
                    f"{cone['outer_overhang_phase_fraction']:.9f}"
                ),
                "cone_minimum_outer_clearance_mm": (
                    f"{cone['minimum_outer_clearance_mm']:.9f}"
                ),
                "duplicate_of": point["duplicate_of"] or "",
            }
        )
    return buffer.getvalue()


def display_xy(
    fixture_xy: Iterable[tuple[float, float]] | tuple[float, float]
) -> list[tuple[float, float]] | tuple[float, float]:
    if (
        isinstance(fixture_xy, tuple)
        and len(fixture_xy) == 2
        and isinstance(fixture_xy[0], (int, float))
    ):
        return (-float(fixture_xy[0]), -float(fixture_xy[1]))
    return [(-float(x), -float(y)) for x, y in fixture_xy]  # type: ignore[arg-type]


def add_hex_patch(
    axis: Any,
    af_mm: float,
    center: tuple[float, float] = (0.0, 0.0),
    yaw_deg: float = 0.0,
    **kwargs: Any,
) -> Polygon:
    vertices = display_xy(regular_hexagon(af_mm, center, yaw_deg))
    patch = Polygon(vertices, closed=True, **kwargs)
    axis.add_patch(patch)
    return patch


def draw_cad_background(axis: Any, geometry: dict[str, float]) -> None:
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            0.5 * geometry["head_outer_diameter"],
            facecolor="#f8fafc",
            edgecolor="#475569",
            linewidth=1.5,
            zorder=1,
        )
    )
    add_hex_patch(
        axis,
        geometry["socket_mouth_af"],
        facecolor="#fee2e2",
        edgecolor="#be123c",
        linewidth=1.4,
        alpha=0.34,
        zorder=2,
    )
    add_hex_patch(
        axis,
        geometry["socket_deep_wall_af"],
        facecolor="white",
        edgecolor="#7f1d1d",
        linewidth=1.1,
        linestyle="--",
        zorder=3,
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            0.5 * geometry["center_through_hole_diameter"],
            facecolor="#111827",
            edgecolor="#111827",
            zorder=4,
        )
    )


def representative_point(
    plan: dict[str, Any],
    state: str,
    preferred_subtype: str | None = None,
) -> dict[str, Any]:
    candidates = [
        point
        for point in plan["points"]
        if point["partition"] == "clean_candidate"
        and point["contact_state"] == state
    ]
    if preferred_subtype is not None:
        candidates = [
            point
            for point in candidates
            if point["cone_sweep_subtype"] == preferred_subtype
        ]
    return candidates[0]


def render_figure(plan: dict[str, Any]) -> tuple[bytes, str]:
    geometry = plan["geometry_mm"]
    contact_geometry = ContactGeometry(geometry)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": PLAN_NAME,
        }
    )
    figure = plt.figure(figsize=(16.0, 9.0))
    grid = figure.add_gridspec(3, 5)
    main = figure.add_subplot(grid[:, :3])
    state_axes = [figure.add_subplot(grid[index, 3:]) for index in range(3)]
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.17,
        top=0.835,
        wspace=0.58,
        hspace=0.62,
    )

    draw_cad_background(main, geometry)
    points = plan["points"]
    order = plan["execution"]["attempt_indices"]
    route = [(0.0, 0.0)] + [
        tuple(points[index - 1]["local_offset_fixture_mm"]) for index in order
    ] + [(0.0, 0.0)]
    route_uv = display_xy(route)
    main.plot(
        [point[0] for point in route_uv],
        [point[1] for point in route_uv],
        "--",
        color="#64748b",
        linewidth=0.85,
        alpha=0.75,
        zorder=5,
    )
    execution_position = {
        attempt: index for index, attempt in enumerate(order, start=1)
    }
    for point in points:
        local = tuple(point["local_offset_fixture_mm"])
        u, v = display_xy(local)
        state = point["contact_state"]
        color = STATE_STYLE[state]["color"]
        partition = point["partition"]
        marker = PARTITION_STYLE[partition]["marker"]
        face = "none" if partition in ("holdout", "sentinel") else color
        size = 120 if partition == "sentinel" else 58
        main.scatter(
            [u],
            [v],
            marker=marker,
            s=size,
            facecolor=face,
            edgecolor=color,
            linewidth=1.5,
            zorder=8,
        )
        main.text(
            u,
            v,
            str(execution_position[point["attempt_index"]]),
            fontsize=6.5,
            ha="center",
            va="center",
            color="#0f172a",
            zorder=9,
        )

    main.annotate(
        "+X$_F$",
        xy=(-13.5, 12.8),
        xytext=(-7.0, 12.8),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#111827"},
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    main.annotate(
        "+Y$_F$",
        xy=(13.0, -13.5),
        xytext=(13.0, -7.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#111827"},
        ha="left",
        va="center",
        fontweight="bold",
    )
    main.text(
        8.0,
        12.6,
        r"$\odot$ +Z$_F$ toward viewer",
        fontweight="bold",
        ha="left",
    )
    main.set_title(
        f"{TARGET_LABEL} actual printed-CAD 26-point route\n"
        "numbers are execution order; display (u,v)=(-x$_F$,-y$_F$); "
        f"closed distance={plan['execution']['closed_route_distance_mm']:.2f} mm",
        fontsize=14,
        fontweight="bold",
    )
    main.set_xlabel("display rightward u [mm]")
    main.set_ylabel("display upward v [mm]")
    main.set_aspect("equal", adjustable="box")
    main.set_xlim(-15.5, 15.5)
    main.set_ylim(-15.5, 15.5)
    main.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.7)

    for axis, state in zip(
        state_axes,
        (
            "supported_search",
            "double_overhang",
            "outer_partial_support",
        ),
    ):
        draw_cad_background(axis, geometry)
        preferred_subtype = (
            "full_frustum_mouth_clear"
            if state == "outer_partial_support"
            else None
        )
        point = representative_point(plan, state, preferred_subtype)
        center = tuple(point["local_offset_fixture_mm"])
        for phase_index in range(0, CONE_PHASE_BINS, 30):
            phase_polygon = [
                (vertex[0] + center[0], vertex[1] + center[1])
                for vertex in contact_geometry.cone_phase_polygons[phase_index]
            ]
            phase_uv = display_xy(phase_polygon)
            axis.add_patch(
                Polygon(
                    phase_uv,
                    closed=True,
                    facecolor="none",
                    edgecolor=STATE_STYLE[state]["color"],
                    linewidth=0.55,
                    alpha=0.24,
                    zorder=5,
                )
            )
        add_hex_patch(
            axis,
            geometry["peg_tip_face_af"],
            center,
            geometry["peg_tip_face_flat_normal_fixture_deg"],
            facecolor=STATE_STYLE[state]["color"],
            edgecolor="#111827",
            linewidth=1.2,
            alpha=0.34,
            zorder=6,
        )
        metrics = point["nominal_touchdown_geometry"]
        cone = point["printed_frustum_cone_phase_diagnostics"]
        axis.set_title(
            f"{STATE_STYLE[state]['title']}  |e|={point['radius_mm']:.2f} mm\n"
            f"outer={metrics['outer_overhang_fraction']:.2f}, "
            f"mouth={metrics['mouth_overlap_fraction']:.2f}, "
            f"solid={metrics['head_top_support_fraction']:.2f}; "
            f"cone mouth phases={cone['mouth_overlap_phase_fraction']:.2f}",
            fontsize=9.6,
            fontweight="bold",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-19.5, 19.5)
        axis.set_ylim(-15.5, 15.5)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(True, color="#e2e8f0", linewidth=0.4)

    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATE_STYLE[state]["color"],
            markeredgecolor=STATE_STYLE[state]["color"],
            label=STATE_STYLE[state]["title"],
        )
        for state in (
            "supported_search",
            "double_overhang",
            "outer_partial_support",
        )
    ]
    partition_handles = [
        Line2D(
            [0],
            [0],
            marker=PARTITION_STYLE[partition]["marker"],
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#111827",
            label={
                "clean_candidate": "clean",
                "holdout": "holdout",
                "quarantine": "boundary quarantine",
                "sentinel": "repeatability sentinel",
            }[partition],
        )
        for partition in PARTITION_STYLE
    ]
    figure.legend(
        handles=state_handles + partition_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.058),
        ncol=7,
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        f"{TARGET_LABEL} conical-fingerprint sampling derived from the printed "
        "v1.5 STEP solids\n"
        f"peg: AF{geometry['peg_tip_face_af']:g} tip / "
        f"{geometry['peg_chamfer_axial_length']:g} mm 45° chamfer / "
        f"AF{geometry['peg_shaft_af']:g} shaft; "
        f"socket: D{geometry['head_outer_diameter']:g} / "
        f"AF{geometry['socket_mouth_af']:g} mouth / "
        f"AF{geometry['socket_deep_wall_af']:g} wall / "
        f"D{geometry['center_through_hole_diameter']:g} bore",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        (
            f"closed route {plan['execution']['closed_route_distance_mm']:.2f} mm; "
            f"sampling-only XY envelope "
            f"{plan['motion_contract']['required_sampling_only_lateral_excursion_mm']:.1f} mm. "
            "Thin outlines in state panels are exact sampled projections of the "
            "printed tip-to-shaft frustum during the 15° cone; the filled "
            "tip-face hexagon is touchdown."
        ),
        ha="center",
        fontsize=9,
        color="#334155",
    )

    png_buffer = io.BytesIO()
    figure.savefig(
        png_buffer,
        format="png",
        dpi=180,
        metadata={"Software": GENERATOR_NAME},
    )
    svg_buffer = io.StringIO()
    figure.savefig(
        svg_buffer,
        format="svg",
        metadata={
            "Date": None,
            "Creator": GENERATOR_NAME,
        },
    )
    plt.close(figure)
    return png_buffer.getvalue(), svg_buffer.getvalue()


def generated_outputs(plan: dict[str, Any]) -> dict[Path, bytes]:
    png, svg = render_figure(plan)
    plan_bytes = (
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    csv_bytes = serialize_csv(plan).encode("utf-8")
    outputs = {
        PLAN_PATH: plan_bytes,
        CSV_PATH: csv_bytes,
        PNG_PATH: png,
        SVG_PATH: svg.encode("utf-8"),
    }
    outputs.update({path: png for path in OPERATOR_PNG_PATHS})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any checked-in output differs from deterministic generation",
    )
    arguments = parser.parse_args()
    plan = build_plan()
    outputs = generated_outputs(plan)
    if arguments.check:
        stale = [
            str(path)
            for path, expected in outputs.items()
            if not path.is_file() or path.read_bytes() != expected
        ]
        if stale:
            raise SystemExit(
                f"stale generated {TARGET_LABEL} sampling outputs:\n"
                + "\n".join(stale)
            )
        print(
            f"{TARGET_LABEL} printed-CAD sampling outputs are current; "
            f"route={plan['execution']['closed_route_distance_mm']:.3f} mm"
        )
        return 0

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(path)
    print(
        "route",
        plan["execution"]["attempt_indices"],
        f"{plan['execution']['closed_route_distance_mm']:.3f} mm",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
