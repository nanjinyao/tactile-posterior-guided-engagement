#!/usr/bin/env python3
"""Render lightweight PNG previews directly from the generated binary STL files."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "generated"


def cad_xy_to_tip_view(points: np.ndarray) -> np.ndarray:
    """Project flange/fixture XY into the view from the peg tip along +Z.

    Keeping +X to screen-right in this underside view forces +Y to point
    screen-down.  Consequently a positive right-handed yaw about +Z appears
    clockwise.  Keeping this reflection explicit avoids accidentally drawing
    a flange-side (+Z-looking-down) view and labelling it as a tip-side view.
    """

    projected = np.asarray(points, dtype=float).copy()
    if projected.shape[-1] != 2:
        raise ValueError("tip-view projection expects XY pairs")
    projected[..., 1] *= -1.0
    return projected


def unit_xy(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    return np.array([np.cos(yaw_rad), np.sin(yaw_rad)])


def draw_tip_view_axes(axis, origin: Tuple[float, float]) -> None:
    """Draw the handed coordinate triad used by both yaw-contract panels."""

    origin_xy = np.asarray(origin, dtype=float)
    axis.annotate(
        "",
        xy=origin_xy + np.array([7.0, 0.0]),
        xytext=origin_xy,
        arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 1.8},
    )
    axis.text(
        *(origin_xy + np.array([7.4, 0.0])),
        r"$+X_F$",
        ha="left",
        va="center",
        fontsize=9,
        weight="bold",
    )
    axis.annotate(
        "",
        xy=origin_xy + np.array([0.0, -7.0]),
        xytext=origin_xy,
        arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 1.8},
    )
    axis.text(
        *(origin_xy + np.array([0.0, -7.5])),
        r"$+Y_F$",
        ha="center",
        va="top",
        fontsize=9,
        weight="bold",
    )
    axis.text(
        *(origin_xy + np.array([4.0, -4.0])),
        r"$\otimes\ +Z_F$",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )


def read_binary_stl(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"invalid binary STL: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    expected = 84 + triangle_count * dtype.itemsize
    if len(payload) != expected:
        raise ValueError(f"unexpected STL size: {path}")
    return np.frombuffer(payload, dtype=dtype, count=triangle_count, offset=84)[
        "vertices"
    ].astype(float)


def draw_shapes(
    axis,
    shapes: Sequence[Tuple[np.ndarray, str]],
    title: str,
    elevation: float,
    azimuth: float,
) -> None:
    all_vertices = []
    for triangles, color in shapes:
        axis.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolor=color,
                edgecolor="#20242a",
                linewidth=0.045,
                alpha=0.96,
            )
        )
        all_vertices.append(triangles.reshape(-1, 3))
    vertices = np.concatenate(all_vertices, axis=0)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    half_span = 0.56 * float(np.max(upper - lower))
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(title, fontsize=13)
    axis.set_axis_off()


def main() -> None:
    params = json.loads((HERE / "design_parameters.json").read_text(encoding="utf-8"))
    tools = {
        size: read_binary_stl(OUTPUT_DIR / f"integrated_flange_peg_{size}.stl")
        for size in ("m8", "m10", "m12")
    }
    hardware_easy_base = read_binary_stl(
        OUTPUT_DIR / "socket_base_hardware_easy_v2.stl"
    )
    nominal_base = read_binary_stl(OUTPUT_DIR / "socket_base_nominal.stl")
    base = hardware_easy_base

    figure = plt.figure(figsize=(15.0, 10.0), dpi=180)
    draw_shapes(
        figure.add_subplot(2, 2, 1, projection="3d"),
        [(tools["m8"], "#2f7de1")],
        "One-piece M8 flange tool - self-supporting load path",
        10.0,
        -55.0,
    )
    draw_shapes(
        figure.add_subplot(2, 2, 2, projection="3d"),
        [
            (tools["m8"] + np.array([-100.0, 0.0, 0.0]), "#e9a13a"),
            (tools["m10"], "#dd8732"),
            (tools["m12"] + np.array([100.0, 0.0, 0.0]), "#c96f2c"),
        ],
        "Three complete flange tools - common 42 mm tip datum",
        10.0,
        -58.0,
    )
    draw_shapes(
        figure.add_subplot(2, 2, 3, projection="3d"),
        [(base, "#727d88")],
        "Recommended hardware-easy-v2 base - larger labels and capture",
        58.0,
        -88.0,
    )
    draw_shapes(
        figure.add_subplot(2, 2, 4, projection="3d"),
        [(nominal_base, "#b7bec8")],
        "Simulation-exact control base - not the default hardware target",
        58.0,
        -88.0,
    )
    figure.tight_layout()
    output_path = OUTPUT_DIR / "preview_overview.png"
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(output_path)

    detail_figure = plt.figure(figsize=(15.0, 7.0), dpi=200)
    side_axis = detail_figure.add_subplot(1, 2, 1, projection="3d")
    draw_shapes(
        side_axis,
        [(base, "#727d88")],
        "Front-side detail: 2 mm smooth bands and vertical ribs",
        12.0,
        -90.0,
    )
    side_axis.set_xlim(-62.0, 62.0)
    side_axis.set_ylim(-17.0, 17.0)
    side_axis.set_zlim(-14.0, 2.0)
    side_axis.set_box_aspect((124.0, 34.0, 16.0))

    underside_axis = detail_figure.add_subplot(1, 2, 2, projection="3d")
    draw_shapes(
        underside_axis,
        [(base, "#727d88")],
        "M10 underside close-up: representative 0.5 mm bore exit",
        -88.0,
        -90.0,
    )
    underside_axis.set_xlim(-10.0, 10.0)
    underside_axis.set_ylim(-10.0, 10.0)
    underside_axis.set_zlim(-31.0, -28.0)
    underside_axis.set_box_aspect((20.0, 20.0, 3.0))

    detail_figure.tight_layout()
    detail_path = OUTPUT_DIR / "preview_socket_fixture_details.png"
    detail_figure.savefig(detail_path, bbox_inches="tight", facecolor="white")
    print(detail_path)

    top_figure = plt.figure(figsize=(15.0, 7.2), dpi=220)
    top_axis = top_figure.add_subplot(1, 1, 1, projection="3d")
    draw_shapes(
        top_axis,
        [(base, "#8a96a3")],
        "Hardware-easy-v2 top view: 7 mm engraved labels and enlarged 45-degree mouths",
        89.9,
        -90.0,
    )
    top_axis.set_xlim(-78.0, 78.0)
    top_axis.set_ylim(-37.0, 37.0)
    top_axis.set_zlim(-31.0, 1.0)
    top_axis.set_box_aspect((156.0, 74.0, 16.0))
    top_path = OUTPUT_DIR / "preview_socket_base_top_labels.png"
    top_figure.savefig(top_path, bbox_inches="tight", facecolor="white")
    print(top_path)

    manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    validation = manifest["socket_profile_validation"]
    strict_rows = validation["socket_base_nominal"]["sizes"]
    easy_rows = validation["socket_base_hardware_easy_v2"]["sizes"]
    profile_figure, profile_axes = plt.subplots(1, 3, figsize=(15.0, 5.4), dpi=200)
    for axis, size in zip(profile_axes, ("m8", "m10", "m12")):
        strict_af = strict_rows[size]["target_geometry_mm"]["top_af_mm"]
        easy_geometry = easy_rows[size]["target_geometry_mm"]
        easy_af = easy_geometry["top_af_mm"]
        easy_head_diameter = easy_geometry["head_outer_diameter_mm"]

        def hex_vertices(across_flats: float) -> np.ndarray:
            radius = across_flats / np.sqrt(3.0)
            theta = np.deg2rad(30.0 + 60.0 * np.arange(6))
            return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

        axis.add_patch(
            Circle(
                (0.0, 0.0),
                0.5 * easy_head_diameter,
                facecolor="#d8dde3",
                edgecolor="#475569",
                linewidth=1.5,
            )
        )
        axis.add_patch(
            Polygon(
                hex_vertices(easy_af),
                closed=True,
                facecolor="#8ecae6",
                edgecolor="#0077b6",
                linewidth=2.2,
                alpha=0.85,
                label=f"hardware easy v2 {easy_af:.2f} mm",
            )
        )
        axis.add_patch(
            Polygon(
                hex_vertices(strict_af),
                closed=True,
                fill=False,
                edgecolor="#c2410c",
                linewidth=2.0,
                linestyle="--",
                label=f"simulation exact {strict_af:.2f} mm",
            )
        )
        radius = 0.5 * easy_head_diameter + 0.8
        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)
        axis.set_aspect("equal")
        axis.set_title(size.upper(), fontsize=15, weight="bold")
        axis.legend(loc="lower center", fontsize=8, frameon=True)
        axis.axis("off")
    profile_figure.suptitle(
        "Socket-mouth profiles: strict control vs recommended hardware capture",
        fontsize=16,
        weight="bold",
        y=0.98,
    )
    profile_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    profile_path = OUTPUT_DIR / "preview_socket_geometry_profiles.png"
    profile_figure.savefig(profile_path, bbox_inches="tight", facecolor="white")
    print(profile_path)

    tool_validation = manifest["integrated_tool_validation"]
    peg_figure, peg_axes = plt.subplots(1, 3, figsize=(15.0, 5.2), dpi=200)
    for axis, size in zip(peg_axes, ("m8", "m10", "m12")):
        geometry = tool_validation[size]["target_tip_geometry_mm"]
        shaft_af = float(geometry["shaft_af_mm"])
        easy_tip_af = float(geometry["tip_face_af_mm"])
        easy_height = float(geometry["tip_chamfer_axial_length_mm"])
        source_tip_af = float(geometry["source_tip_face_af_mm"])
        source_height = float(geometry["source_tip_chamfer_axial_length_mm"])
        easy_polygon = np.array(
            [
                [-0.5 * easy_tip_af, 0.0],
                [0.5 * easy_tip_af, 0.0],
                [0.5 * shaft_af, easy_height],
                [0.5 * shaft_af, 1.5],
                [-0.5 * shaft_af, 1.5],
                [-0.5 * shaft_af, easy_height],
            ]
        )
        source_outline = np.array(
            [
                [-0.5 * source_tip_af, 0.0],
                [0.5 * source_tip_af, 0.0],
                [0.5 * shaft_af, source_height],
                [0.5 * shaft_af, 1.5],
                [-0.5 * shaft_af, 1.5],
                [-0.5 * shaft_af, source_height],
                [-0.5 * source_tip_af, 0.0],
            ]
        )
        axis.add_patch(
            Polygon(
                easy_polygon,
                closed=True,
                facecolor="#8ecae6",
                edgecolor="#0077b6",
                linewidth=2.2,
                label=f"easy 45°: {easy_height:.2f} mm",
            )
        )
        axis.plot(
            source_outline[:, 0],
            source_outline[:, 1],
            color="#c2410c",
            linestyle="--",
            linewidth=2.0,
            label=f"source: {source_height:.2f} mm",
        )
        axis.set_xlim(-0.65 * shaft_af, 0.65 * shaft_af)
        axis.set_ylim(-0.08, 1.58)
        axis.set_aspect("equal")
        axis.set_title(
            f"{size.upper()}  tip AF {easy_tip_af:.2f} mm",
            fontsize=14,
            weight="bold",
        )
        axis.set_xlabel("flat-normal width [mm]")
        axis.set_ylabel("height from physical tip [mm]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper center", fontsize=8)
    peg_figure.suptitle(
        "Peg-tip profiles: visible hardware chamfer vs source-simulation taper",
        fontsize=16,
        weight="bold",
    )
    peg_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    peg_path = OUTPUT_DIR / "preview_peg_tip_profiles.png"
    peg_figure.savefig(peg_path, bbox_inches="tight", facecolor="white")
    print(peg_path)

    yaw_contract = manifest["peg_flange_yaw_contract"]
    peg_yaw_deg = float(yaw_contract["shape_rotation_about_positive_z_deg"])
    selected_flat_deg = float(
        yaw_contract["selected_peg_flat_normal_in_flange_deg"]
    )
    flange_reference_deg = float(
        yaw_contract["flange_reference_tangent_normal_deg"]
    )
    socket_flat_deg = float(yaw_contract["socket_selected_flat_normal_deg"])

    alignment_figure, (alignment_axis, installed_axis) = plt.subplots(
        1,
        2,
        figsize=(15.8, 8.2),
        dpi=220,
    )
    flange_radius = 18.0
    flange_circle = Circle(
        (0.0, 0.0),
        flange_radius,
        facecolor="#dbe3ec",
        edgecolor="#334155",
        linewidth=2.0,
        alpha=0.95,
    )
    alignment_axis.add_patch(flange_circle)

    hex_af = 10.0
    hex_radius = hex_af / np.sqrt(3.0)
    hex_angles = np.deg2rad(30.0 + peg_yaw_deg + 60.0 * np.arange(6))
    peg_hex_cad = np.column_stack(
        (hex_radius * np.cos(hex_angles), hex_radius * np.sin(hex_angles))
    )
    peg_hex = cad_xy_to_tip_view(peg_hex_cad)
    alignment_axis.add_patch(
        Polygon(
            peg_hex,
            closed=True,
            facecolor="#2f7de1",
            edgecolor="#0f172a",
            linewidth=2.0,
            alpha=0.9,
        )
    )

    normal = unit_xy(flange_reference_deg)
    tangent = np.array([-normal[1], normal[0]])
    for sign, label in ((1.0, "flange B/C tangent planes"), (-1.0, None)):
        center_cad = sign * flange_radius * normal
        endpoints = cad_xy_to_tip_view(
            np.vstack(
                (
                    center_cad - 5.0 * tangent,
                    center_cad + 5.0 * tangent,
                )
            )
        )
        alignment_axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            color="#dc2626",
            linewidth=4.0,
            solid_capstyle="round",
            label=label,
        )
        normal_start = cad_xy_to_tip_view(
            center_cad - 0.8 * sign * normal
        )
        normal_end = cad_xy_to_tip_view(
            center_cad + 4.2 * sign * normal
        )
        alignment_axis.annotate(
            "",
            xy=normal_end,
            xytext=normal_start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#dc2626",
                "lw": 2.2,
            },
        )
    positive_normal_label_xy = cad_xy_to_tip_view(17.0 * normal)
    alignment_axis.text(
        *positive_normal_label_xy,
        "B/C tangent NORMAL\n(CAD +45°)",
        color="#b91c1c",
        fontsize=8.5,
        ha="left",
        va="top",
        weight="bold",
    )

    selected_normal = unit_xy(selected_flat_deg)
    selected_tip = cad_xy_to_tip_view(13.0 * selected_normal)
    alignment_axis.arrow(
        0.0,
        0.0,
        selected_tip[0],
        selected_tip[1],
        width=0.18,
        head_width=1.25,
        head_length=1.5,
        length_includes_head=True,
        color="#16a34a",
        label="selected peg flat NORMAL (CAD +45°)",
    )

    legacy_deg = selected_flat_deg - peg_yaw_deg
    legacy_normal = unit_xy(legacy_deg)
    legacy_tip = cad_xy_to_tip_view(12.0 * legacy_normal)
    alignment_axis.plot(
        [0.0, legacy_tip[0]],
        [0.0, legacy_tip[1]],
        color="#f59e0b",
        linestyle="--",
        linewidth=2.0,
        label="legacy selected normal (CAD +60°)",
    )

    shape_arc_deg = np.linspace(legacy_deg, selected_flat_deg, 48)
    shape_arc_cad = 9.0 * np.column_stack(
        (np.cos(np.deg2rad(shape_arc_deg)), np.sin(np.deg2rad(shape_arc_deg)))
    )
    shape_arc = cad_xy_to_tip_view(shape_arc_cad)
    alignment_axis.plot(
        shape_arc[:, 0],
        shape_arc[:, 1],
        color="#f59e0b",
        linewidth=2.0,
    )
    alignment_axis.annotate(
        "",
        xy=shape_arc[-1],
        xytext=shape_arc[-5],
        arrowprops={"arrowstyle": "-|>", "color": "#f59e0b", "lw": 2.0},
    )
    alignment_axis.text(
        2.0,
        -10.4,
        f"CAD construction rotation {peg_yaw_deg:+.0f}°\n"
        "(appears counter-clockwise in this view)",
        color="#b45309",
        fontsize=8.5,
        ha="center",
        va="top",
    )

    positive_yaw_deg = np.linspace(0.0, 52.0, 64)
    positive_yaw_cad = 6.8 * np.column_stack(
        (
            np.cos(np.deg2rad(positive_yaw_deg)),
            np.sin(np.deg2rad(positive_yaw_deg)),
        )
    )
    positive_yaw_view = cad_xy_to_tip_view(positive_yaw_cad)
    alignment_axis.plot(
        positive_yaw_view[:, 0],
        positive_yaw_view[:, 1],
        color="#0f172a",
        linewidth=1.6,
    )
    alignment_axis.annotate(
        "",
        xy=positive_yaw_view[-1],
        xytext=positive_yaw_view[-6],
        arrowprops={"arrowstyle": "-|>", "color": "#0f172a", "lw": 1.6},
    )
    alignment_axis.text(
        7.0,
        -3.2,
        "right-handed +yaw\n(clockwise)",
        ha="left",
        va="center",
        fontsize=8,
        color="#0f172a",
    )
    alignment_axis.text(
        0.0,
        -21.5,
        (
            "Tip-side hand: +X right, +Y down, +Z into page; +yaw clockwise.\n"
            f"CAD: shape {peg_yaw_deg:+.0f}°; selected-flat NORMAL "
            f"{selected_flat_deg:.0f}° = B/C NORMAL "
            f"{flange_reference_deg:.0f}°; residual "
            f"{selected_flat_deg - flange_reference_deg:+.3f}°."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.94},
    )
    alignment_axis.set_xlim(-25.0, 25.0)
    alignment_axis.set_ylim(-25.0, 25.0)
    alignment_axis.set_aspect("equal")
    alignment_axis.set_title(
        "1. CAD-local geometry: selected flat NORMAL aligns B/C NORMAL",
        fontsize=12,
        weight="bold",
    )
    alignment_axis.set_xlabel(r"display right $=+X_{\mathrm{flange}}$ [mm]")
    alignment_axis.set_ylabel(r"display up $=-Y_{\mathrm{flange}}$ [mm]")
    alignment_axis.grid(alpha=0.18)
    alignment_axis.legend(loc="upper left", fontsize=9)
    draw_tip_view_axes(alignment_axis, (14.0, 22.0))

    installed_axis.add_patch(
        Circle(
            (0.0, 0.0),
            flange_radius,
            facecolor="#edf2f7",
            edgecolor="#334155",
            linewidth=2.0,
            alpha=0.95,
        )
    )
    installed_hex_angles = np.deg2rad(
        30.0 + socket_flat_deg + 60.0 * np.arange(6)
    )
    installed_hex = cad_xy_to_tip_view(
        np.column_stack(
            (
                hex_radius * np.cos(installed_hex_angles),
                hex_radius * np.sin(installed_hex_angles),
            )
        )
    )
    installed_axis.add_patch(
        Polygon(
            installed_hex,
            closed=True,
            facecolor="#bfdbfe",
            edgecolor="#1d4ed8",
            linewidth=2.2,
            alpha=0.8,
            label="installed peg: selected flat NORMAL = +X_F",
        )
    )
    installed_axis.add_patch(
        Polygon(
            1.35 * installed_hex,
            closed=True,
            fill=False,
            edgecolor="#7c3aed",
            linewidth=2.2,
            linestyle="--",
            label=f"socket: selected flat NORMAL = {socket_flat_deg:.0f}° = +X_F",
        )
    )
    installed_axis.plot(
        [flange_radius, flange_radius],
        [-5.0, 5.0],
        color="#dc2626",
        linewidth=4.0,
        solid_capstyle="round",
        label="installed B/C tangent plane",
    )
    for y_offset, color, text in (
        (4.0, "#dc2626", "B/C tangent NORMAL"),
        (0.0, "#16a34a", "peg selected-flat NORMAL"),
        (-4.0, "#7c3aed", "socket selected-flat NORMAL"),
    ):
        installed_axis.annotate(
            "",
            xy=(16.0, y_offset),
            xytext=(1.5, y_offset),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5},
        )
        installed_axis.text(
            0.7,
            y_offset,
            text,
            ha="right",
            va="center",
            fontsize=8.5,
            color=color,
            weight="bold",
        )
    installed_axis.text(
        0.0,
        11.8,
        r"ONE installation mapping: B/C tangent NORMAL $\rightarrow +X_F$",
        ha="center",
        va="center",
        fontsize=11,
        color="#991b1b",
        weight="bold",
    )
    installed_axis.text(
        0.0,
        -13.0,
        (
            f"CAD-local: shape {peg_yaw_deg:+.0f}°, selected flat "
            f"{selected_flat_deg:.0f}°\n"
            f"Installed fixture: peg flat 0°, socket {socket_flat_deg:.0f}°\n"
            "The CAD -15° is already built in: apply NO second yaw compensation."
        ),
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "#fff7ed", "edgecolor": "#ea580c", "alpha": 0.96},
    )
    installed_axis.set_xlim(-25.0, 25.0)
    installed_axis.set_ylim(-25.0, 25.0)
    installed_axis.set_aspect("equal")
    installed_axis.set_title(
        r"2. Installed mapping: three NORMALS coincide with $+X_F$",
        fontsize=12,
        weight="bold",
    )
    installed_axis.set_xlabel(r"display right $=+X_F$ [mm]")
    installed_axis.set_ylabel(r"display up $=-Y_F$ [mm]")
    installed_axis.grid(alpha=0.18)
    installed_axis.legend(loc="upper left", fontsize=8)
    draw_tip_view_axes(installed_axis, (14.0, 22.0))
    alignment_figure.suptitle(
        "v1.5 peg/flange yaw contract — tip-side view (looking along +Z)",
        fontsize=15,
        weight="bold",
    )
    alignment_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    alignment_path = OUTPUT_DIR / "preview_peg_flange_yaw_alignment.png"
    alignment_figure.savefig(
        alignment_path,
        bbox_inches="tight",
        facecolor="white",
    )
    print(alignment_path)


if __name__ == "__main__":
    main()
