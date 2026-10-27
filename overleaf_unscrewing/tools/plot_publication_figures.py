"""Generate publication figures directly from the frozen experiment records.

The script intentionally keeps scientific plots deterministic.  It reads the
24 paired MuJoCo trial records and timestamped M12 hardware development runs.
No values are imputed or smoothed.
"""

from __future__ import annotations

import json
import io
import math
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


PAPER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PAPER_ROOT.parent
FIG_DIR = PAPER_ROOT / "figures"
SIM_DIR = (
    WORKSPACE
    / "mujoco_unscrewing_probe_new"
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_five_method_retry3_r3_closed_loop"
)
HARDWARE_DIR = (
    WORKSPACE
    / "real_device"
    / "data"
    / "M12_proposed_plus_entropy_grid_20260729_215410_474"
)
HARDWARE_PHOTO = PAPER_ROOT / "figures" / "hardware_contact_original.png"
HARDWARE_PAIR_DIRS = [
    WORKSPACE / "real_device" / "data" / "M12_proposed_plus_entropy_grid_20260730_030400_362",
    WORKSPACE / "real_device" / "data" / "M12_baseline_improved_direct_x3_20260730_032603_301",
]
HARDWARE_FOUR_POLICY_DIRS = [
    WORKSPACE / "real_device" / "data" / "M12_proposed_plus_entropy_grid_20260729_215410_474",
    WORKSPACE / "real_device" / "data" / "M12_proposed_local_spiral_20260729_215905_875",
    WORKSPACE / "real_device" / "data" / "M12_baseline_improved_direct_x3_20260729_220513_919",
    WORKSPACE / "real_device" / "data" / "M12_baseline_pure_outward_spiral_20260729_225552_920",
]
SHORTCUT_HARDWARE_INSET = (
    PAPER_ROOT / "figures" / "fig4_m8_surface_contact_inset_rgba.png"
)
SHORTCUT_SIM_INSET = PAPER_ROOT / "figures" / "fig4_m8_mujoco_interface.png"
VIDEO_ROOT = (
    WORKSPACE
    / "mujoco_unscrewing_probe_new"
    / "outputs"
    / "current"
    / "active_comparison"
    / "m8_retry3_selected_media"
)
# Resolve the external decoder without depending on one author's machine.
# An explicit FFMPEG environment variable takes precedence over PATH.
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
OUTPUT_DIR = FIG_DIR


COLORS = {
    "ink": "#171717",
    "muted": "#5f6368",
    "grid": "#d7dce2",
    "blue": "#2676b8",
    "green": "#2a9d68",
    "orange": "#e69f00",
    "red": "#d64a3b",
    "purple": "#7a5195",
    "cyan": "#38a6c5",
}

METHODS = [
    ("proposed", "PGS-O", COLORS["blue"]),
    ("proposed_plus", "PGS-G", COLORS["green"]),
    ("outward", "Outward", COLORS["orange"]),
    ("improved_direct_retry3", "Direct-4C", COLORS["purple"]),
    ("jin_direct_retry3", "Direct-3C", COLORS["cyan"]),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.25,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def block(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    color: str,
    *,
    linewidth: float = 1.55,
    fontsize: float = 6.3,
) -> tuple[float, float, float, float]:
    x, y = xy
    width, height = size
    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        facecolor="white",
        edgecolor=color,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=fontsize,
        linespacing=1.15,
        zorder=4,
    )
    return x, y, width, height


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    connectionstyle: str = "arc3,rad=0",
    linewidth: float = 1.15,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )
    if label:
        lx, ly = label_xy if label_xy is not None else (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.12,
        )
        ax.text(lx, ly, label, ha="center", va="center", fontsize=6.7, color=color)


def poly_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    linewidth: float = 1.15,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=linewidth, zorder=1.5)
    arrow(ax, points[-2], points[-1], color=color, linewidth=linewidth)
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center", fontsize=6.7, color=color)


def plot_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.28))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # The dashed boundary makes the learned routing authority explicit.
    ax.plot([0.15, 15.8], [3.70, 3.70], color="#9aa0a6", lw=0.8, ls=(0, (4, 3)))
    ax.text(0.20, 7.88, "LEARNED TACTILE INFERENCE AND BELIEF-SPACE ROUTING",
            fontsize=6.4, color=COLORS["muted"], va="top")
    ax.text(0.20, 3.50, "PHYSICAL INTERACTION, RECOVERY, AND TERMINAL AUTHORITY",
            fontsize=6.4, color=COLORS["muted"], va="top")

    # Upper layer: the model proposes only high-level routing quantities.
    block(ax, (0.25, 5.20), (2.30, 0.96), "Phase-indexed\nobservation $y(\\phi)$", COLORS["blue"])
    block(ax, (3.05, 5.20), (2.55, 0.96), "Tactile ensemble\n$p(s\\mid y),\\;p(\\theta\\mid y)$", COLORS["blue"])
    block(ax, (6.15, 5.20), (2.55, 0.96), "Center belief\n$p(C\\mid y,G)$", COLORS["green"])
    block(ax, (9.30, 5.58), (2.35, 0.92), "Capped shortcut\n$L^*,\\hat d$", COLORS["green"])
    block(ax, (12.20, 4.42), (2.55, 0.92), "Posterior residual\nsearch $\\mathcal{R}_{\\rm res}$", COLORS["green"])
    block(ax, (6.15, 6.72), (2.55, 0.58), "Visual/CAD prior $p_0(C),G$", COLORS["orange"])

    arrow(ax, (2.55, 5.68), (3.05, 5.68), color=COLORS["blue"])
    arrow(ax, (5.60, 5.68), (6.15, 5.68), color=COLORS["blue"])
    arrow(ax, (8.70, 5.68), (9.30, 6.04), color=COLORS["green"])
    arrow(ax, (8.70, 5.50), (12.20, 4.93), color=COLORS["green"])
    arrow(ax, (7.43, 6.72), (7.43, 6.16), color=COLORS["orange"])
    arrow(ax, (11.65, 5.86), (12.20, 5.18), color=COLORS["green"],
          label="miss: update", label_xy=(12.10, 5.92))

    # Lower layer: one controller and one verifier are shared by all routes.
    block(ax, (0.25, 1.62), (2.30, 0.96), "Phase logic\nprobe | route\ncapture | insert", COLORS["ink"], fontsize=5.5)
    block(ax, (3.05, 1.62), (2.55, 0.96), "Admittance/RCC\nmotion control", COLORS["ink"])
    block(ax, (6.15, 1.62), (2.55, 0.96), "Robot + F/T sensor\n+ hex driver", COLORS["ink"])
    block(ax, (9.30, 1.62), (2.35, 0.96), "Socket /\nenvironment", COLORS["ink"])
    block(ax, (12.20, 0.48), (2.55, 1.08), "Physical guards\nrecover | verify | stop", COLORS["red"], fontsize=5.8)

    arrow(ax, (2.55, 2.10), (3.05, 2.10), label="$a_t$")
    arrow(ax, (5.60, 2.10), (6.15, 2.10), label="$u_t$")
    arrow(ax, (8.70, 2.10), (9.30, 2.10), label="$x_t$")
    arrow(ax, (9.30, 1.84), (8.70, 1.84), label="$f,\\tau$", label_xy=(9.0, 1.68))

    poly_arrow(
        ax,
        [(7.43, 2.58), (7.43, 4.14), (1.40, 4.14), (1.40, 5.20)],
        color=COLORS["blue"],
        label="$w(\\phi),z_{\\rm tip}(\\phi)$",
        label_xy=(4.75, 4.29),
    )
    poly_arrow(
        ax,
        [(10.47, 5.58), (10.47, 3.98), (1.40, 3.98), (1.40, 2.58)],
        color=COLORS["green"],
    )
    poly_arrow(
        ax,
        [(13.47, 4.42), (13.47, 3.82), (1.70, 3.82), (1.70, 2.58)],
        color=COLORS["green"],
    )

    arrow(ax, (8.05, 1.62), (12.70, 1.50), color=COLORS["red"],
          label="$w(t),x(t)$", label_xy=(10.25, 1.30))
    poly_arrow(
        ax,
        [(12.20, 0.96), (11.82, 0.96), (11.82, 0.18), (1.40, 0.18), (1.40, 1.62)],
        color=COLORS["red"],
        label="recover / success / abort",
        label_xy=(6.65, 0.34),
    )

    save(fig, "fig2_control_framework")


def plot_tactile_observation() -> None:
    audit_root = (
        WORKSPACE
        / "mujoco_unscrewing_probe_new"
        / "outputs"
        / "current"
        / "evaluations"
        / "p1v5_m8_final_untouched_direction_audit"
    )
    trial_id = "double_overhang_audit3_r04_l01_s04_n01"
    with np.load(audit_root / "trials" / f"{trial_id}.npz", allow_pickle=False) as payload:
        sequence = np.asarray(payload["phase_sequence"], dtype=float)
        columns = tuple(map(str, payload["phase_columns"].tolist()))
    prediction = pd.read_csv(audit_root / "predictions.csv")
    row = prediction.loc[prediction["trial_id"] == trial_id].iloc[0]

    signals = [
        (sequence[columns.index("Fz_N")], "$F_z$ [N]"),
        (1000.0 * sequence[columns.index("Tx_Nm")], r"$\tau_x\,[\mathrm{mN}\!\cdot\!\mathrm{m}]$"),
        (1000.0 * sequence[columns.index("Ty_Nm")], r"$\tau_y\,[\mathrm{mN}\!\cdot\!\mathrm{m}]$"),
        (1000.0 * sequence[columns.index("tip_z_m")], "$z_{\\mathrm{tip}}$ [mm]"),
    ]
    phase = np.linspace(0.0, 360.0, sequence.shape[1], endpoint=False)

    fig = plt.figure(figsize=(7.15, 3.15))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.12), hspace=0.25, wspace=0.34)
    trace_axes: list[plt.Axes] = []
    for index, (values, ylabel) in enumerate(signals):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        trace_axes.append(ax)
        ax.plot(phase, values, color=COLORS["blue"], lw=1.0)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylabel(ylabel)
        if index < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"Probe phase $\phi$ ($^\circ$)")
        finish_axis(ax, grid_axis="both")

    axp = fig.add_subplot(grid[:, 2], projection="polar")
    true_angle = math.radians(float(row["true_direction_deg"]))
    top_angle = math.radians(float(row["pred_direction_deg"]))
    mean_angle = math.radians(float(row["control_direction_deg"]))
    axp.plot([true_angle, true_angle], [0, 1.0], color=COLORS["red"], lw=2.2, label="True (evaluation)")
    axp.plot([top_angle, top_angle], [0, 0.90], color=COLORS["blue"], lw=3.4, alpha=0.72, label="Top bin")
    axp.plot([mean_angle, mean_angle], [0, 0.80], color=COLORS["green"], lw=2.2, label="Circular mean")
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_yticklabels([])
    axp.set_ylim(0, 1.05)
    axp.set_title(
        f"prediction-set size = {int(row['aps_size'])}; ensemble $M=3$\n"
        f"circular-mean error = {float(row['angle_error_deg']):.2f}$^\\circ$",
        fontsize=7.4,
        pad=9,
    )
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), frameon=False, ncol=1, fontsize=6.4, handlelength=1.4)

    fig.text(0.335, 0.005, "(a) Phase-indexed tactile fingerprint", ha="center", va="bottom", fontweight="bold")
    fig.text(0.835, 0.005, "(b) Circular direction summary", ha="center", va="bottom", fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.17)
    save(fig, "tactile_fingerprint_and_direction")


def plot_shortcut_geometry() -> None:
    capture_radius = 0.25
    case_specs = [
        ("sobol_000", "clear", COLORS["blue"]),
        ("sobol_016", "fuzzy", COLORS["orange"]),
        ("sobol_002", "reverse mass", COLORS["red"]),
    ]
    cases: list[tuple[str, str, str, dict]] = []
    for trial_id, label, color in case_specs:
        path = SIM_DIR / "trials" / f"{trial_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))["methods"]["proposed"]
        cases.append((trial_id, label, color, record))

    all_records: list[tuple[str, dict]] = []
    for path in sorted((SIM_DIR / "trials").glob("sobol_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))["methods"]["proposed"]
        all_records.append((path.stem, record))
    if len(all_records) != 24:
        raise RuntimeError("Expected 24 frozen M8 records for shortcut geometry")

    fig = plt.figure(figsize=(7.15, 3.30))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.34, 1.16, 0.78), wspace=0.40)
    ax_alloc = fig.add_subplot(grid[0, 0])
    ax_geom = fig.add_subplot(grid[0, 1])
    ax_batch = fig.add_subplot(grid[0, 2])

    # (a) Three frozen cases expose the continuous allocation rule and fallback.
    y_positions = np.asarray([2.0, 1.0, 0.0])
    for y, (trial_id, label, color, record) in zip(y_positions, cases):
        quantiles = record["transport_projected_quantiles_mm"]
        q10 = float(quantiles["q10"])
        q50 = float(quantiles["q50"])
        q90 = float(quantiles["q90"])
        shortcut = float(record["transport_distance_mm"])
        raw = float(record["transport_raw_distance_mm"])
        cap = float(record["transport_distance_cap_mm"])
        quality = float(record["transport_direction_quality"])
        reverse = float(record["transport_reverse_probability"])
        residual = float(record["spiral_max_radius_mm"])

        ax_alloc.hlines(y, q10, q90, color="#a9b4c3", lw=4.2, alpha=0.70, zorder=1)
        ax_alloc.vlines([q10, q90], y - 0.09, y + 0.09, color="#7b8794", lw=0.9)
        ax_alloc.scatter(q50, y, marker="o", s=20, facecolor="white", edgecolor=COLORS["purple"], lw=1.25, zorder=4)
        ax_alloc.vlines(cap, y - 0.19, y + 0.19, color=COLORS["green"], lw=1.4, ls="--", zorder=3)
        if raw > 0.0:
            selected_bound = raw + 0.25 * capture_radius
            ax_alloc.scatter(selected_bound, y, marker="D", s=24, color=COLORS["blue"], edgecolor="white", lw=0.45, zorder=5)
        if shortcut > 0.0:
            ax_alloc.annotate(
                "",
                xy=(shortcut, y - 0.23),
                xytext=(0.0, y - 0.23),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0, "mutation_scale": 10},
            )
        else:
            ax_alloc.scatter(0.0, y - 0.23, marker="x", s=30, color=COLORS["red"], lw=1.7, zorder=5)
        reverse_text = "$p_-<0.001$" if reverse < 0.001 else f"$p_-={reverse:.3f}$"
        ax_alloc.text(
            4.55,
            y + 0.18,
            f"$q={quality:.3f}$, {reverse_text}",
            ha="right",
            va="bottom",
            fontsize=6.25,
            color=COLORS["muted"],
        )
        action = "$L^*=0$ (fallback)" if shortcut == 0.0 else f"$L^*={shortcut:.2f}$ mm, $R_{{\\rm res}}={residual:.2f}$ mm"
        ax_alloc.text(4.55, y - 0.18, action, ha="right", va="top", fontsize=6.35, color=color)

    ax_alloc.axvline(0.0, color=COLORS["ink"], lw=0.9, ls=(0, (3, 2)))
    ax_alloc.text(0.03, -0.43, "$z=0$: contact plane", fontsize=5.9, color=COLORS["muted"], ha="left")
    ax_alloc.set_xlim(-2.15, 4.65)
    ax_alloc.set_ylim(-0.52, 2.52)
    ax_alloc.set_yticks(y_positions)
    ax_alloc.set_yticklabels(["clear\nset size 3", "ambiguous\nset size 8", "reversed\nset size 10"])
    ax_alloc.set_xlabel("Projected center distance $z$ [mm]")
    ax_alloc.set_title("(a) Adaptive travel allocation", loc="left", fontweight="bold", fontsize=8.0)
    finish_axis(ax_alloc, grid_axis="x")
    ax_alloc.legend(
        handles=[
            Line2D([0], [0], color="#a9b4c3", lw=4.0, label="$[Q_{0.10},Q_{0.90}]$"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["purple"], label="$Q_{0.50}$"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="white", label=r"$Q^w_{\alpha(q)}(z)$"),
            Line2D([0], [0], color=COLORS["green"], lw=1.3, ls="--", label="$L_{\\rm rad}$"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.99),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d5dbe3",
        framealpha=0.92,
        fontsize=5.55,
        columnspacing=0.65,
        handlelength=1.15,
        borderaxespad=0.0,
    )

    # (b) The radial bound is an envelope, not an isotropic posterior density.
    clear = cases[0][3]
    initial = np.asarray(clear["initial_xy_mm"], dtype=float)
    endpoint = np.asarray(clear["transport_target_xy_mm"], dtype=float)
    center = np.zeros(2)
    residual = float(clear["spiral_max_radius_mm"])
    pitch = 2.0 * capture_radius * 0.80
    theta = np.linspace(0.0, 2.0 * np.pi * residual / pitch, 900)
    radius = pitch * theta / (2.0 * np.pi)
    spiral = endpoint[:, None] + np.vstack((radius * np.cos(theta), radius * np.sin(theta)))

    ax_geom.add_patch(
        patches.Circle(
            endpoint,
            residual,
            facecolor="#dbeafe",
            edgecolor=COLORS["blue"],
            lw=1.15,
            ls=(0, (4, 2.3)),
            alpha=0.18,
            zorder=0,
        )
    )
    ax_geom.plot(spiral[0], spiral[1], color=COLORS["green"], lw=0.80, alpha=0.78, zorder=2)
    ax_geom.add_patch(
        patches.Circle(
            center,
            capture_radius,
            facecolor="#dcfce7",
            edgecolor=COLORS["green"],
            lw=1.2,
            alpha=0.92,
            zorder=4,
        )
    )
    ax_geom.annotate(
        "",
        xy=endpoint,
        xytext=initial,
        arrowprops={"arrowstyle": "-|>", "color": COLORS["blue"], "lw": 2.35, "mutation_scale": 12},
        zorder=5,
    )
    ax_geom.scatter(*initial, s=22, facecolor="white", edgecolor=COLORS["ink"], lw=0.8, zorder=6)
    ax_geom.scatter(*endpoint, s=30, color=COLORS["blue"], zorder=6)
    ax_geom.scatter(*center, marker="x", s=46, color=COLORS["red"], lw=1.7, zorder=7)
    ax_geom.annotate(
        "$x_0$ (touchdown)",
        initial,
        xytext=(-5, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    ax_geom.annotate("$x_s$", endpoint, xytext=(7, -10), textcoords="offset points", fontsize=6.2, color=COLORS["blue"])
    ax_geom.annotate("$C^*$", center, xytext=(8, 5), textcoords="offset points", fontsize=6.2, color=COLORS["red"])

    # Reserve the upper part of the axes for a compact key.  Keeping these
    # labels outside the data region prevents them from covering the endpoint,
    # support boundary, or route after reduction to IEEE page width.
    ax_geom.add_patch(
        patches.FancyBboxPatch(
            (0.425, 0.815),
            0.57,
            0.17,
            boxstyle="round,pad=0.008",
            transform=ax_geom.transAxes,
            facecolor="white",
            edgecolor="#d5dbe3",
            lw=0.45,
            alpha=0.94,
            zorder=9,
        )
    )
    ax_geom.plot([0.45, 0.505], [0.947, 0.947], transform=ax_geom.transAxes, color=COLORS["blue"], lw=1.7, zorder=10)
    ax_geom.plot([0.45, 0.505], [0.897, 0.897], transform=ax_geom.transAxes, color=COLORS["blue"], lw=1.1, ls=(0, (3, 2)), zorder=10)
    ax_geom.plot([0.45, 0.505], [0.847, 0.847], transform=ax_geom.transAxes, color=COLORS["green"], lw=1.2, zorder=10)
    ax_geom.text(
        0.52,
        0.955,
        "Shortcut $L^*$",
        transform=ax_geom.transAxes,
        ha="left",
        va="top",
        fontsize=5.55,
        color=COLORS["blue"],
        zorder=10,
    )
    ax_geom.text(
        0.52,
        0.905,
        f"99% support: $r_{{99}}={residual:.2f}$ mm",
        transform=ax_geom.transAxes,
        ha="left",
        va="top",
        fontsize=5.55,
        color=COLORS["blue"],
        zorder=10,
    )
    ax_geom.text(
        0.52,
        0.855,
        f"Residual spiral: $h={pitch:.2f}$ mm/rev",
        transform=ax_geom.transAxes,
        ha="left",
        va="top",
        fontsize=5.55,
        color=COLORS["green"],
        zorder=10,
    )
    # A deterministic top-view cue replaces the earlier synthetic hardware
    # crop, whose aperture was visibly off-centre.  The annulus and hex socket
    # are exactly concentric; this inset is explanatory and not metrically scaled.
    inset = ax_geom.inset_axes([0.025, 0.825, 0.18, 0.145], zorder=9)
    inset.set_xlim(-1.1, 1.1)
    inset.set_ylim(-1.1, 1.1)
    inset.set_aspect("equal")
    inset.add_patch(patches.Circle((0, 0), 0.94, facecolor="#f2a3a5", edgecolor="#be6668", lw=0.8))
    inset.add_patch(patches.Circle((0, 0), 0.50, facecolor="#f7c9c9", edgecolor="#be6668", lw=0.75))
    inset.add_patch(patches.RegularPolygon((0, 0), 6, radius=0.35, orientation=np.pi / 6, facecolor="#353a40", edgecolor="white", lw=0.65))
    inset.plot([0], [0], marker="+", color="white", ms=4.0, mew=0.7)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#cad2dc")
        spine.set_linewidth(0.65)
    if SHORTCUT_SIM_INSET.exists():
        inset = ax_geom.inset_axes([0.215, 0.825, 0.18, 0.145], zorder=9)
        inset.imshow(plt.imread(SHORTCUT_SIM_INSET))
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("#cad2dc")
            spine.set_linewidth(0.65)
        inset.text(
            0.5,
            0.97,
            "MuJoCo",
            transform=inset.transAxes,
            ha="center",
            va="top",
            fontsize=4.6,
            color="white",
            bbox={"facecolor": "#22262c", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        )
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_xlim(-2.25, 2.25)
    ax_geom.set_ylim(-3.72, 2.05)
    ax_geom.set_xlabel("$x$ [mm]")
    ax_geom.set_ylabel("$y$ [mm]")
    ax_geom.set_title("(b) Residual geometry and coverage", loc="left", fontweight="bold", fontsize=8.0)
    finish_axis(ax_geom, grid_axis="both")
    # (c) The 24 paired records quantify contraction of the conservative disk.
    highlighted = {trial_id: color for trial_id, _, color, _ in cases}
    before_values: list[float] = []
    after_values: list[float] = []
    area_reductions: list[float] = []
    contracted = 0
    for trial_id, record in all_records:
        quality = float(record["transport_direction_quality"])
        radial_fraction = 0.35 + 0.30 * quality
        before = float(record["transport_distance_cap_mm"]) / radial_fraction - capture_radius
        after = float(record["spiral_max_radius_mm"])
        before_values.append(before)
        after_values.append(after)
        area_reductions.append(1.0 - (after / before) ** 2)
        contracted += int(after < before - 1.0e-9)
        color = highlighted.get(trial_id, "#aeb7c2")
        alpha = 0.95 if trial_id in highlighted else 0.42
        width = 1.55 if trial_id in highlighted else 0.72
        ax_batch.plot([0, 1], [before, after], color=color, lw=width, alpha=alpha, zorder=2)
        ax_batch.scatter([0, 1], [before, after], s=12 if trial_id in highlighted else 7, color=color, alpha=alpha, zorder=3, edgecolor="white", linewidth=0.25)
    median_area_reduction = 100.0 * float(np.median(area_reductions))
    ax_batch.set_xlim(-0.25, 1.25)
    ax_batch.set_ylim(0.0, max(before_values) + 0.34)
    ax_batch.set_xticks([0, 1])
    ax_batch.set_xticklabels(["Initial", "After shortcut"])
    ax_batch.set_ylabel("Conservative search radius [mm]")
    ax_batch.set_title("(c) Frozen M8 batch ($n=24$)", loc="left", fontweight="bold", fontsize=8.0)
    ax_batch.text(
        0.04,
        0.055,
        f"{contracted}/24 contracted\nmedian area reduction\n{median_area_reduction:.1f}%",
        transform=ax_batch.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.35,
        color=COLORS["green"],
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#effaf4", "edgecolor": "#abd9c2", "alpha": 0.95},
    )
    finish_axis(ax_batch, grid_axis="y")
    ax_batch.text(0.50, 0.985, "24 individual tasks", transform=ax_batch.transAxes, ha="center", va="top", fontsize=5.8, color=COLORS["muted"])

    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.16)
    save(fig, "posterior_shortcut_geometry")


def load_trials() -> dict[str, list[dict]]:
    records = {key: [] for key, _, _ in METHODS}
    for path in sorted((SIM_DIR / "trials").glob("sobol_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in records:
            record = dict(payload["methods"][key])
            record["trial_id"] = path.stem
            records[key].append(record)
    if any(len(values) != 24 for values in records.values()):
        raise RuntimeError("Expected 24 paired M8 trials for every method")
    return records


def is_success(record: dict) -> bool:
    return record.get("result") == "cad_bottom_verified"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half


def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.55, alpha=0.72)
    ax.set_axisbelow(True)


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode one audited MP4 frame without modifying the source video."""

    if not FFMPEG or not Path(FFMPEG).is_file():
        raise FileNotFoundError("Install ffmpeg on PATH or set FFMPEG to its executable path.")
    command = [
        str(FFMPEG),
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    payload = subprocess.run(command, check=True, capture_output=True).stdout
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))


def show_keyframe(
    ax: plt.Axes,
    frame: np.ndarray,
    crop: tuple[int, int, int, int],
    color: str,
) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(frame[y0:y1, x0:x1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(0.85)


def direct_probe_error(record: dict) -> float:
    attempts = record.get("direct_attempt_records") or []
    if not attempts:
        raise RuntimeError(f"Missing direct-attempt record for {record.get('trial_id')}")
    return float(attempts[0]["offline_control_error_deg"])


def plot_representation_ablation(records: dict[str, list[dict]]) -> None:
    """Pair aggregate inference evidence with an audited same-probe sequence."""

    keys = ["improved_direct_retry3", "jin_direct_retry3"]
    labels = ["Direct-4C", "Direct-3C proxy"]
    colors = [COLORS["purple"], COLORS["cyan"]]
    by_id = {key: {row["trial_id"]: row for row in records[key]} for key in keys}
    trial_ids = sorted(by_id[keys[0]])
    errors = {
        key: np.asarray([direct_probe_error(by_id[key][trial_id]) for trial_id in trial_ids])
        for key in keys
    }

    fig = plt.figure(figsize=(7.15, 3.52))
    outer = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.27, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.995,
        top=0.90,
        bottom=0.09,
        wspace=0.08,
        hspace=0.13,
    )
    left = outer[:, 0].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.72)

    ax = fig.add_subplot(left[0, 0])
    for key, label, color in zip(keys, labels, colors):
        ordered = np.sort(errors[key])
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.step(ordered, cdf, where="post", color=color, lw=1.55, label=label)
        median = float(np.median(ordered))
        p90 = float(np.quantile(ordered, 0.9))
        ax.scatter([median, p90], [0.5, 0.9], s=19, color=color, edgecolor="white", lw=0.45, zorder=4)
    ax.axvline(15.0, color=COLORS["muted"], ls="--", lw=0.8)
    ax.text(17.5, 0.04, "$15^\\circ$ capture tolerance", rotation=90, va="bottom", ha="left", fontsize=5.9, color=COLORS["muted"])
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 15, 45, 90, 135])
    ax.set_xlabel(r"Direction error of first executed ray $|\hat\theta-\theta^*|$ ($^\circ$)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Same-probe inference ($n=24$)", loc="left", fontweight="bold", fontsize=7.8)
    ax.legend(loc="lower right", frameon=False, fontsize=6.4, handlelength=1.35)
    finish_axis(ax, grid_axis="both")

    ax = fig.add_subplot(left[1, 0])
    metric_names = [r"Within $15^\circ$", "First-ray capture", "Any-ray capture", "Verified terminal"]
    metric_values: dict[str, list[int]] = {key: [] for key in keys}
    for key in keys:
        rows = [by_id[key][trial_id] for trial_id in trial_ids]
        metric_values[key] = [
            int(np.sum(errors[key] <= 15.0)),
            sum(row.get("direct_successful_attempt") == 1 for row in rows),
            sum(row.get("direct_successful_attempt") is not None for row in rows),
            sum(is_success(row) for row in rows),
        ]
    ypos = np.arange(len(metric_names))[::-1]
    for index, y in enumerate(ypos):
        x4 = metric_values[keys[0]][index]
        x3 = metric_values[keys[1]][index]
        ax.plot([x4, x3], [y, y], color="#aeb5bd", lw=1.0, zorder=1)
        ax.scatter(x4, y, s=23, color=colors[0], edgecolor="white", lw=0.4, zorder=2)
        ax.scatter(x3, y, s=23, color=colors[1], edgecolor="white", lw=0.4, zorder=2)
        if x4 == x3:
            ax.text(x4, y + 0.18, f"{x4} / {x3}", ha="center", va="bottom", fontsize=6.0, color=COLORS["ink"])
        else:
            ax.text(x4, y + 0.17, str(x4), ha="center", va="bottom", fontsize=6.0, color=colors[0])
            ax.text(x3, y - 0.17, str(x3), ha="center", va="top", fontsize=6.0, color=colors[1])
    ax.set_yticks(ypos, metric_names)
    ax.set_xlim(14.5, 24.4)
    ax.set_xticks([16, 20, 24])
    ax.set_xlabel("Successful trials ($n=24$ paired starts)")
    ax.set_title("(b) Paired outcomes", loc="left", fontweight="bold", fontsize=7.8)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0], markeredgecolor="white", label="Direct-4C"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[1], markeredgecolor="white", label="Direct-3C proxy"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=5.8,
        handletextpad=0.3,
    )
    finish_axis(ax, grid_axis="x")

    video = VIDEO_ROOT / "sobol_005" / "comparison_improved4_vs_jin3_retry3.mp4"
    frame_indices = [[24, 82, 168], [24, 232, 482]]
    crops = [(0, 0, 640, 420), (640, 0, 1280, 420)]
    column_titles = ["First probe", "Ray execution", "Terminal state"]
    for row_index, (label, color) in enumerate(zip(labels, colors)):
        for column_index, frame_index in enumerate(frame_indices[row_index]):
            ax = fig.add_subplot(outer[row_index, column_index + 1])
            frame = read_video_frame(video, frame_index)
            show_keyframe(ax, frame, crops[row_index], color)
            if row_index == 0:
                ax.set_title(column_titles[column_index], fontsize=7.2, pad=3)
            if column_index == 0:
                ax.text(
                    -0.045,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=6.7,
                    color=color,
                    fontweight="bold",
                )
    fig.text(
        0.655,
        0.985,
        "(c) Paired rollout: identical first fingerprint (Sobol trial 005)",
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="bold",
    )
    save(fig, "m8_four_channel_representation_ablation")


def plot_policy_keyframes() -> None:
    """Show actual chronological frames for the four task-level policies."""

    video = VIDEO_ROOT / "sobol_016" / "comparison_fuzzy_retry3_2x2.mp4"
    manifest = json.loads(
        (VIDEO_ROOT / "sobol_016" / "comparison_fuzzy_retry3_2x2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    methods = [
        ("proposed_plus", "PGS-G", COLORS["green"], (0, 0, 640, 400)),
        ("proposed", "PGS-O", COLORS["blue"], (640, 0, 1280, 400)),
        ("outward", "Outward", COLORS["orange"], (0, 400, 640, 800)),
        ("improved_direct_retry3", "Direct-4C", COLORS["purple"], (640, 400, 1280, 800)),
    ]
    fractions = [0.08, 0.43, 0.90]
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(7.15, 5.05),
        gridspec_kw={"wspace": 0.045, "hspace": 0.10},
    )
    for row_index, (key, label, color, crop) in enumerate(methods):
        active_frames = int(manifest["composition"]["active_frames"][key])
        result = manifest["results"][key]
        for column_index, fraction in enumerate(fractions):
            frame_index = max(1, min(active_frames - 1, int(round(fraction * active_frames))))
            frame = read_video_frame(video, frame_index)
            ax = axes[row_index, column_index]
            show_keyframe(ax, frame, crop, color)
            if row_index == 0:
                ax.set_title(["First probe", "Route / residual search", "Terminal state"][column_index], fontsize=7.5, pad=3)
            if column_index == 0:
                ax.text(
                    -0.025,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=6.8,
                    color=color,
                    fontweight="bold",
                )
        terminal = axes[row_index, -1]
        if is_success(result):
            badge = f"Verified — {result['time_to_bottom_s']:.2f} s, {result['actual_xy_path_mm']:.2f} mm"
        else:
            badge = f"No capture — {result['total_active_time_s']:.2f} s, {result['actual_xy_path_mm']:.2f} mm"
        terminal.text(
            0.98,
            0.04,
            badge,
            transform=terminal.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.6,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#111820", "edgecolor": color, "alpha": 0.92},
        )
    fig.subplots_adjust(left=0.055, right=0.997, top=0.94, bottom=0.015)
    save(fig, "m8_four_policy_keyframe_matrix")


def plot_budgeted_profiles(records: dict[str, list[dict]]) -> None:
    """Show how often engagement is reached under time and travel budgets.

    The 24 frozen starts are stratified only for visualization by their true
    initial radial offset.  Each row contains eight paired tasks.  A failed
    trial never enters the cumulative curve, so the plateau reports both
    reliability and conditional efficiency without censoring failures away.
    """

    proposed = records["proposed"]
    ordered_ids = sorted(
        (row["trial_id"] for row in proposed),
        key=lambda trial_id: float(
            np.linalg.norm(
                next(row for row in proposed if row["trial_id"] == trial_id)["initial_xy_mm"]
            )
        ),
    )
    groups = [ordered_ids[:8], ordered_ids[8:16], ordered_ids[16:]]
    group_titles = [
        "Small initial offset",
        "Intermediate initial offset",
        "Large initial offset",
    ]
    id_lookup = {
        key: {row["trial_id"]: row for row in values}
        for key, values in records.items()
    }

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.15, 5.35),
        sharex="col",
        sharey=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.25},
    )
    grids = {
        "time_to_bottom_s": np.linspace(0.0, 130.0, 261),
        "actual_xy_path_mm": np.linspace(0.0, 220.0, 441),
    }
    column_specs = [
        ("time_to_bottom_s", "Time budget [s]"),
        ("actual_xy_path_mm", "Planar travel budget [mm]"),
    ]

    for row_index, (trial_ids, group_title) in enumerate(zip(groups, group_titles)):
        radii = [
            float(np.linalg.norm(id_lookup["proposed"][trial_id]["initial_xy_mm"]))
            for trial_id in trial_ids
        ]
        for column_index, (metric, xlabel) in enumerate(column_specs):
            ax = axes[row_index, column_index]
            grid = grids[metric]
            for key, label, color in METHODS[:-1]:
                achieved = np.asarray(
                    [
                        float(id_lookup[key][trial_id][metric])
                        if is_success(id_lookup[key][trial_id])
                        else np.inf
                        for trial_id in trial_ids
                    ],
                    dtype=float,
                )
                cumulative = np.asarray([(achieved <= value).sum() / len(trial_ids) for value in grid])
                linewidth = 1.75 if key in {"proposed", "proposed_plus"} else 1.15
                ax.step(grid, cumulative, where="post", color=color, lw=linewidth, label=label)
                finite = np.sort(achieved[np.isfinite(achieved)])
                if finite.size:
                    ax.scatter(
                        finite,
                        np.arange(1, finite.size + 1) / len(trial_ids),
                        s=8,
                        color=color,
                        edgecolor="white",
                        linewidth=0.25,
                        zorder=3,
                    )
            ax.set_ylim(0.0, 1.04)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylabel("Verified fraction ($n=8$)")
            if column_index == 0:
                ax.text(
                    0.02,
                    0.95,
                    f"$r_0={min(radii):.2f}$--${max(radii):.2f}$ mm",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    color=COLORS["muted"],
                )
                ax.text(
                    0.98,
                    0.95,
                    f"({chr(97 + row_index)}) {group_title}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.2,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.5},
                )
            if row_index == 2:
                ax.set_xlabel(xlabel)
            finish_axis(ax, grid_axis="both")

    axes[0, 0].set_title("Completion under a time budget", fontweight="bold")
    axes[0, 1].set_title("Completion under a travel budget", fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.005),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.7,
    )
    fig.text(0.995, 0.012, "PGS-O / PGS-G: proposed  |  Outward: global spiral  |  Direct-4C: bounded-ray baseline", ha="right", va="bottom", fontsize=5.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.095, right=0.995, top=0.92, bottom=0.105)
    save(fig, "m8_budgeted_engagement_profiles")


def event_elapsed(events: pd.DataFrame, event: str, wall_start: pd.Timestamp) -> float | None:
    rows = events.loc[events["event"] == event]
    if rows.empty:
        return None
    return float((pd.to_datetime(rows.iloc[0]["wall_time"]) - wall_start).total_seconds())


def plot_hardware_trace() -> None:
    state = pd.read_csv(HARDWARE_DIR / "robot_state.csv")
    wrench = pd.read_csv(HARDWARE_DIR / "world_star_relative_wrench.csv")
    cone = pd.read_csv(HARDWARE_DIR / "conical_directservo_trace.csv")
    events = pd.read_csv(HARDWARE_DIR / "events.csv")
    summary = json.loads((HARDWARE_DIR / "summary.json").read_text(encoding="utf-8"))

    touchdown_rows = events.loc[events["event"] == "FRESH_SURFACE_CONTACT"]
    touchdown_wall = pd.to_datetime(touchdown_rows.iloc[0]["wall_time"])
    state["t_task_s"] = (pd.to_datetime(state["wall_time"]) - touchdown_wall).dt.total_seconds()
    wrench["t_task_s"] = (pd.to_datetime(wrench["wall_time"]) - touchdown_wall).dt.total_seconds()
    cone["t_task_s"] = (pd.to_datetime(cone["wall_time"]) - touchdown_wall).dt.total_seconds()
    terminal_event_time = event_elapsed(events, "M12_TERMINAL_SET_SUCCESS_CANDIDATE", touchdown_wall)
    terminal_time = float(terminal_event_time if terminal_event_time is not None else state["t_task_s"].max())
    state = state[(state["t_task_s"] >= -0.5) & (state["t_task_s"] <= terminal_time + 0.8)].copy()
    wrench = wrench[(wrench["t_task_s"] >= -0.5) & (wrench["t_task_s"] <= terminal_time + 0.8)].copy()
    cone = cone[(cone["t_task_s"] >= -0.5) & (cone["t_task_s"] <= terminal_time + 0.8)].copy()

    anchor = np.asarray(summary["provisional_surface_datum_world_point_mm"], dtype=float)
    state["x_local_mm"] = state["x_mm"] - anchor[0]
    state["y_local_mm"] = state["y_mm"] - anchor[1]
    state["depth_mm"] = anchor[2] - state["z_mm"]
    cone["x_local_mm"] = cone["actual_x_mm"] - anchor[0]
    cone["y_local_mm"] = cone["actual_y_mm"] - anchor[1]

    phase = state["phase"].astype(str)
    masks = {
        "Probe": phase.str.contains("CONE|CONICAL", regex=True),
        "Shortcut/residual": phase.str.contains("SHORTCUT|CORRIDOR|COVERAGE|BAYES", regex=True),
        "Capture/insertion": phase.str.contains("CAPTURE|INSERT|BOTTOM", regex=True),
    }
    segment_colors = {"Probe": COLORS["blue"], "Shortcut/residual": COLORS["green"], "Capture/insertion": COLORS["red"]}

    probe_end = event_elapsed(events, "CONICAL_ROUTING_QUALITY_GRADED", touchdown_wall)
    shortcut_end = event_elapsed(events, "SHORTCUT_ENDPOINT_NEGATIVE_CAPTURE_EVIDENCE", touchdown_wall)
    capture_time = event_elapsed(events, "SURFACE_TRANSPORT_CAPTURE_CONFIRMED", touchdown_wall)
    if probe_end is None or capture_time is None:
        raise RuntimeError("Named hardware run is missing audited probe/capture event times")

    def shade_phases(ax: plt.Axes) -> None:
        ax.axvspan(0.0, probe_end, color=COLORS["blue"], alpha=0.055, lw=0)
        ax.axvspan(probe_end, capture_time, color=COLORS["green"], alpha=0.055, lw=0)
        ax.axvspan(capture_time, terminal_time, color=COLORS["red"], alpha=0.045, lw=0)
        if shortcut_end is not None:
            ax.axvline(shortcut_end, color=COLORS["green"], ls=":", lw=0.8)
        ax.axvline(capture_time, color=COLORS["red"], ls=":", lw=0.8)
        ax.axvline(terminal_time, color=COLORS["ink"], ls="--", lw=0.75)

    if not HARDWARE_PHOTO.exists():
        raise FileNotFoundError(f"Missing documented hardware photograph: {HARDWARE_PHOTO}")
    fig = plt.figure(figsize=(7.15, 6.30))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.995,
        top=0.985,
        bottom=0.075,
        hspace=0.48,
        wspace=0.27,
        height_ratios=[1.04, 1.0, 1.0],
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(plt.imread(HARDWARE_PHOTO))
    ax.set_axis_off()
    ax.set_title("(a) KUKA LBR iiwa, F/T sensor, and M12 fixture", loc="left", fontweight="bold", fontsize=7.8)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(state["x_local_mm"], state["y_local_mm"], color="#a7adb4", lw=0.8, zorder=1)
    ax.plot(cone["x_local_mm"], cone["y_local_mm"], color=COLORS["blue"], lw=1.2, label="Probe", zorder=3)
    for label, mask in masks.items():
        if label == "Probe":
            continue
        ax.plot(state.loc[mask, "x_local_mm"], state.loc[mask, "y_local_mm"], color=segment_colors[label], lw=1.7, label=label, zorder=2)
    ax.scatter([0], [0], marker="o", s=24, facecolor="white", edgecolor=COLORS["ink"], lw=0.8, zorder=4, label="Touchdown")
    shortcut_cmd = np.asarray(summary["learned_active_shortcut_offset_mm"], dtype=float)
    shortcut_meas = np.asarray(summary["shortcut_measured_endpoint_offset_mm"], dtype=float)
    ax.plot([0, shortcut_cmd[0]], [0, shortcut_cmd[1]], color=COLORS["purple"], ls="--", lw=1.25, label=r"Shortcut $L^*=3.00$ mm")
    ax.scatter(shortcut_meas[0], shortcut_meas[1], marker="x", s=28, color=COLORS["purple"], lw=1.1, zorder=5, label="Measured endpoint")
    capture = np.asarray(summary["provisional_capture_axis_top_datum_world_point_mm"], dtype=float)
    ax.scatter([capture[0] - anchor[0]], [capture[1] - anchor[1]], marker="*", s=55, color=COLORS["red"], edgecolor="white", lw=0.4, zorder=5, label="Capture")
    ax.set_xlabel("$x-x_0$ [mm]")
    ax.set_ylabel("$y-y_0$ [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("(b) Measured task-plane route", loc="left", fontweight="bold", fontsize=7.8)
    ax.legend(loc="best", frameon=False, fontsize=5.7, handlelength=1.0, ncol=2, columnspacing=0.7)
    finish_axis(ax, grid_axis="both")

    fingerprint_grid = grid[1, 0].subgridspec(4, 1, hspace=0.05)
    phi_deg = np.degrees(np.asarray(summary["fingerprint_phase_rad"], dtype=float))
    force_fingerprint = np.asarray(summary["fingerprint_force_world_n"], dtype=float)
    torque_fingerprint = 1000.0 * np.asarray(summary["fingerprint_torque_world_nm"], dtype=float)
    tip_height = np.asarray(summary["fingerprint_tip_z_mm"], dtype=float)
    fingerprint_specs = [
        (force_fingerprint[:, 2], "$F_z$ [N]", COLORS["blue"]),
        (torque_fingerprint[:, 0], r"$\tau_x\,[\mathrm{mN}\!\cdot\!\mathrm{m}]$", COLORS["orange"]),
        (torque_fingerprint[:, 1], r"$\tau_y\,[\mathrm{mN}\!\cdot\!\mathrm{m}]$", COLORS["green"]),
        (tip_height - np.mean(tip_height), "$\\Delta z$ [mm]", COLORS["purple"]),
    ]
    fingerprint_axes: list[plt.Axes] = []
    for index, (values, ylabel, color) in enumerate(fingerprint_specs):
        ax_fp = fig.add_subplot(fingerprint_grid[index, 0])
        fingerprint_axes.append(ax_fp)
        ax_fp.plot(phi_deg, values, color=color, lw=0.85)
        ax_fp.set_xlim(0, 360)
        ax_fp.set_ylabel(ylabel, fontsize=5.7, labelpad=1.5)
        ax_fp.tick_params(axis="both", labelsize=5.6, pad=1)
        ax_fp.spines[["top", "right"]].set_visible(False)
        ax_fp.grid(axis="both", color=COLORS["grid"], lw=0.45, alpha=0.65)
        if index < 3:
            ax_fp.set_xticklabels([])
        else:
            ax_fp.set_xticks([0, 90, 180, 270, 360])
            ax_fp.set_xlabel(r"Probe phase $\phi$ ($^\circ$)", fontsize=6.2, labelpad=1.5)
    fingerprint_axes[0].set_title("(c) Four-channel real tactile fingerprint", loc="left", fontweight="bold", fontsize=7.8, pad=2)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(state["t_task_s"], state["depth_mm"], color=COLORS["purple"], lw=1.35, marker="o", ms=1.6)
    ax.axhline(float(summary["measured_m12_socket_top_to_bottom_depth_mm"]), color=COLORS["muted"], ls="--", lw=0.8, label=r"$-6$ mm reference datum")
    shade_phases(ax)
    ax.set_xlim(-0.5, terminal_time + 0.8)
    ax.set_xlabel("Time after touchdown [s]")
    ax.set_ylabel("Axial progress [mm]")
    ax.set_title("(d) Capture and insertion progress", loc="left", fontweight="bold", fontsize=7.8)
    phase_handles = [
        patches.Patch(facecolor=COLORS["blue"], alpha=0.18, label="probe"),
        patches.Patch(facecolor=COLORS["green"], alpha=0.18, label="route"),
        patches.Patch(facecolor=COLORS["red"], alpha=0.16, label="insert"),
    ]
    depth_handle = Line2D([0], [0], color=COLORS["muted"], ls="--", lw=0.8, label=r"$-6$ mm datum")
    ax.legend(handles=phase_handles + [depth_handle], loc="upper left", frameon=False, fontsize=5.8, ncol=2, columnspacing=0.7, handlelength=1.2)
    finish_axis(ax)

    ax = fig.add_subplot(grid[2, 0])
    for field, label, color in [
        ("fx_world_star_n", "$F_x^*$", COLORS["red"]),
        ("fy_world_star_n", "$F_y^*$", COLORS["green"]),
        ("fz_world_star_n", "$F_z^*$", COLORS["blue"]),
    ]:
        ax.plot(wrench["t_task_s"], wrench[field], label=label, color=color, lw=0.95)
    shade_phases(ax)
    ax.set_xlim(-0.5, terminal_time + 0.8)
    ax.set_xlabel("Time after touchdown [s]")
    ax.set_ylabel(r"Relative force $F_x^*,F_y^*,F_z^*$ [N]")
    ax.set_title("(e) Baseline-relative force", loc="left", fontweight="bold", fontsize=7.8)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=0.6, handlelength=1.2)
    finish_axis(ax)

    ax = fig.add_subplot(grid[2, 1])
    for field, label, color in [
        ("tx_world_star_nm", "$\\tau_x^*$", COLORS["red"]),
        ("ty_world_star_nm", "$\\tau_y^*$", COLORS["green"]),
        ("tz_world_star_nm", "$\\tau_z^*$", COLORS["blue"]),
    ]:
        ax.plot(wrench["t_task_s"], 1000.0 * wrench[field], label=label, color=color, lw=0.95)
    shade_phases(ax)
    ax.set_xlim(-0.5, terminal_time + 0.8)
    ax.set_xlabel("Time after touchdown [s]")
    ax.set_ylabel(r"Relative torque $\tau_x^*,\tau_y^*,\tau_z^*$ [mN$\cdot$m]")
    ax.set_title("(f) Baseline-relative torque", loc="left", fontweight="bold", fontsize=7.8)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=0.6, handlelength=1.2)
    finish_axis(ax)
    save(fig, "hardware_m12_named_feasibility")


def plot_hardware_same_start_pair() -> None:
    """Compare two diagnostic M12 executions from the same operator start.

    This is deliberately a mechanism figure, not a population comparison.  It
    uses only timestamped robot and Bota streams and marks the recorded miss,
    capture, and terminal events without smoothing.
    """

    runs: list[dict] = []
    specs = [
        (HARDWARE_PAIR_DIRS[0], "PGS-G", COLORS["green"]),
        (HARDWARE_PAIR_DIRS[1], "Direct-4C", COLORS["purple"]),
    ]
    for directory, label, color in specs:
        state = pd.read_csv(directory / "robot_state.csv")
        wrench = pd.read_csv(directory / "world_star_relative_wrench.csv")
        events = pd.read_csv(directory / "events.csv")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        start_rows = events.loc[events["event"] == "FRESH_SURFACE_CONTACT"]
        if start_rows.empty:
            raise RuntimeError(f"No touchdown event in {directory.name}")
        wall_start = pd.to_datetime(start_rows.iloc[0]["wall_time"])
        terminal = event_elapsed(events, "M12_TERMINAL_SET_SUCCESS_CANDIDATE", wall_start)
        if terminal is None:
            raise RuntimeError(f"No terminal event in {directory.name}")
        state["t"] = (pd.to_datetime(state["wall_time"]) - wall_start).dt.total_seconds()
        wrench["t"] = (pd.to_datetime(wrench["wall_time"]) - wall_start).dt.total_seconds()
        state = state[(state["t"] >= 0.0) & (state["t"] <= terminal + 0.5)].copy()
        wrench = wrench[(wrench["t"] >= 0.0) & (wrench["t"] <= terminal + 0.5)].copy()
        anchor = np.asarray(summary["provisional_surface_datum_world_point_mm"], dtype=float)
        state["x_local"] = state["x_mm"] - anchor[0]
        state["y_local"] = state["y_mm"] - anchor[1]
        state["depth"] = anchor[2] - state["z_mm"]
        probe_end = event_elapsed(events, "CONICAL_ROUTING_QUALITY_GRADED", wall_start)
        capture_names = ["SURFACE_ENDPOINT_LOW_FORCE_CAPTURE_CONFIRMED", "SURFACE_TRANSPORT_CAPTURE_CONFIRMED"]
        capture = next((value for value in (event_elapsed(events, name, wall_start) for name in capture_names) if value is not None), None)
        action_names = ["SHORTCUT_ENDPOINT_NEGATIVE_CAPTURE_EVIDENCE", "IMPROVED_DIRECT_ATTEMPT_RAY_MISS"]
        action = next((value for value in (event_elapsed(events, name, wall_start) for name in action_names) if value is not None), None)
        if probe_end is None or capture is None:
            raise RuntimeError(f"Missing probe/capture boundary in {directory.name}")
        force = np.linalg.norm(wrench[["fx_world_star_n", "fy_world_star_n", "fz_world_star_n"]].to_numpy(float), axis=1)
        torque = 1000.0 * np.linalg.norm(wrench[["tx_world_star_nm", "ty_world_star_nm", "tz_world_star_nm"]].to_numpy(float), axis=1)
        path = float(np.sum(np.linalg.norm(np.diff(state[["x_local", "y_local"]].to_numpy(float), axis=0), axis=1)))
        runs.append(
            {
                "label": label,
                "color": color,
                "state": state,
                "wrench": wrench,
                "force": force,
                "torque": torque,
                "probe_end": probe_end,
                "action": action,
                "capture": capture,
                "terminal": terminal,
                "path": path,
            }
        )

    all_xy = np.vstack([run["state"][["x_local", "y_local"]].to_numpy(float) for run in runs])
    xpad, ypad = 0.45, 0.35
    xlim = (float(all_xy[:, 0].min() - xpad), float(all_xy[:, 0].max() + xpad))
    ylim = (float(all_xy[:, 1].min() - ypad), float(all_xy[:, 1].max() + ypad))
    max_time = max(float(run["terminal"]) for run in runs)

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.15), gridspec_kw={"wspace": 0.38, "hspace": 0.38})
    for row, run in enumerate(runs):
        state = run["state"]
        wrench = run["wrench"]
        boundaries = [0.0, run["probe_end"], run["capture"], run["terminal"]]
        segment_colors = [COLORS["blue"], run["color"], COLORS["red"]]

        ax = axes[row, 0]
        ax.plot(state["x_local"], state["y_local"], color="#c1c7ce", lw=0.75, zorder=1)
        for lo, hi, segment_color in zip(boundaries[:-1], boundaries[1:], segment_colors):
            mask = (state["t"] >= lo) & (state["t"] <= hi)
            ax.plot(state.loc[mask, "x_local"], state.loc[mask, "y_local"], color=segment_color, lw=1.45, zorder=2)
        ax.scatter([0], [0], s=25, facecolor="white", edgecolor=COLORS["ink"], lw=0.8, zorder=4)
        for event_time, marker, event_color in [(run["action"], "x", COLORS["orange"]), (run["capture"], "*", COLORS["red"])]:
            if event_time is None:
                continue
            idx = (state["t"] - event_time).abs().idxmin()
            marker_style = {} if marker == "x" else {"edgecolor": "white", "lw": 0.45}
            ax.scatter(
                state.loc[idx, "x_local"],
                state.loc[idx, "y_local"],
                marker=marker,
                s=48,
                color=event_color,
                zorder=5,
                **marker_style,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x-x_0$ [mm]")
        ax.set_ylabel(r"$y-y_0$ [mm]")
        finish_axis(ax, grid_axis="both")

        ax = axes[row, 1]
        ax.plot(state["t"], state["depth"], color=run["color"], lw=1.25)
        ax.axhline(6.0, color=COLORS["muted"], ls="--", lw=0.75)
        for event_time, event_color, event_style in [
            (run["action"], COLORS["orange"], ":"),
            (run["capture"], COLORS["red"], ":"),
            (run["terminal"], COLORS["ink"], "--"),
        ]:
            if event_time is not None:
                ax.axvline(event_time, color=event_color, ls=event_style, lw=0.8)
        ax.set_xlim(0, max_time + 2.0)
        ax.set_ylim(-0.15, 6.35)
        ax.set_xlabel("Time after touchdown [s]")
        ax.set_ylabel("Axial progress [mm]")
        ax.text(0.98, 0.05, f"Terminal: {run['terminal']:.1f} s, $L_{{xy}}={run['path']:.1f}$ mm", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.7, color=run["color"], bbox={"facecolor": "white", "edgecolor": run["color"], "boxstyle": "round,pad=0.18", "alpha": 0.9})
        finish_axis(ax, grid_axis="both")

        ax = axes[row, 2]
        ax.plot(wrench["t"], run["force"], color=run["color"], lw=1.0, label=r"$\|F^*\|$")
        ax2 = ax.twinx()
        ax2.plot(wrench["t"], run["torque"], color=COLORS["muted"], lw=0.9, ls="--", label=r"$\|\tau^*\|$")
        for event_time, event_color, event_style in [
            (run["action"], COLORS["orange"], ":"),
            (run["capture"], COLORS["red"], ":"),
            (run["terminal"], COLORS["ink"], "--"),
        ]:
            if event_time is not None:
                ax.axvline(event_time, color=event_color, ls=event_style, lw=0.75)
        ax.set_xlim(0, max_time + 2.0)
        ax.set_xlabel("Time after touchdown [s]")
        ax.set_ylabel(r"$\|F^*\|$ [N]", color=run["color"])
        ax2.set_ylabel(r"$\|\tau^*\|$ [mN$\cdot$m]", color=COLORS["muted"])
        ax.tick_params(axis="y", colors=run["color"])
        ax2.tick_params(axis="y", colors=COLORS["muted"], labelsize=6.4)
        finish_axis(ax, grid_axis="both")
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        axes[row, 0].text(-0.43, 0.5, run["label"], transform=axes[row, 0].transAxes, rotation=90, ha="center", va="center", fontsize=8.1, fontweight="bold", color=run["color"])

    axes[0, 0].set_title("(a) Measured task-plane route", fontweight="bold", fontsize=7.7)
    axes[0, 1].set_title("(b) Capture and insertion", fontweight="bold", fontsize=7.7)
    axes[0, 2].set_title("(c) Baseline-relative load", fontweight="bold", fontsize=7.7)
    handles = [
        Line2D([0], [0], color=COLORS["blue"], lw=1.4, label="Conical probe"),
        Line2D([0], [0], color=COLORS["green"], lw=1.4, label="PGS-G route"),
        Line2D([0], [0], color=COLORS["purple"], lw=1.4, label="Direct-4C rays"),
        Line2D([0], [0], color=COLORS["red"], lw=1.4, label="Insertion"),
        Line2D([0], [0], marker="x", color=COLORS["orange"], lw=0, label="Shortcut negative / ray miss"),
        Line2D([0], [0], marker="*", color=COLORS["red"], lw=0, label="Capture"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=6, frameon=False, fontsize=5.35, columnspacing=0.65, handlelength=1.25)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.91, bottom=0.10)
    save(fig, "hardware_m12_same_start_mechanism_pair")


def plot_hardware_four_policy_matrix() -> None:
    """Show the inference-to-contact chain for one fixed-seed hardware task.

    Columns are four timestamped M12 development executions that share the
    same registered offset.  The matrix is intentionally a mechanism view:
    posterior bars, measured routes, and relative wrench streams are plotted
    without smoothing or interpolation.  Population claims remain with the
    preregistered randomized study.
    """

    specs = [
        (HARDWARE_FOUR_POLICY_DIRS[0], "PGS-G", COLORS["green"]),
        (HARDWARE_FOUR_POLICY_DIRS[1], "PGS-O", COLORS["blue"]),
        (HARDWARE_FOUR_POLICY_DIRS[2], "Direct-4C", COLORS["purple"]),
        (HARDWARE_FOUR_POLICY_DIRS[3], "Outward", COLORS["orange"]),
    ]
    capture_events = [
        "SURFACE_TRANSPORT_CAPTURE_CONFIRMED",
        "SURFACE_ENDPOINT_LOW_FORCE_CAPTURE_CONFIRMED",
        "SURFACE_ENDPOINT_CUMULATIVE_DEPTH_CAPTURE_CONFIRMED",
    ]
    miss_events = [
        "SHORTCUT_ENDPOINT_NEGATIVE_CAPTURE_EVIDENCE",
        "IMPROVED_DIRECT_ATTEMPT_RAY_MISS",
    ]

    runs: list[dict] = []
    for directory, label, color in specs:
        state = pd.read_csv(directory / "robot_state.csv")
        wrench = pd.read_csv(directory / "world_star_relative_wrench.csv")
        events = pd.read_csv(directory / "events.csv")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        touchdown_rows = events.loc[events["event"] == "FRESH_SURFACE_CONTACT"]
        if touchdown_rows.empty:
            raise RuntimeError(f"No touchdown event in {directory.name}")
        wall_start = pd.to_datetime(touchdown_rows.iloc[0]["wall_time"])
        terminal = event_elapsed(events, "M12_TERMINAL_SET_SUCCESS_CANDIDATE", wall_start)
        if terminal is None:
            raise RuntimeError(f"No terminal event in {directory.name}")
        probe_end = event_elapsed(events, "CONICAL_ROUTING_QUALITY_GRADED", wall_start)
        capture = next(
            (
                value
                for value in (event_elapsed(events, name, wall_start) for name in capture_events)
                if value is not None
            ),
            None,
        )
        if capture is None:
            raise RuntimeError(f"No audited capture event in {directory.name}")
        miss = next(
            (
                value
                for value in (event_elapsed(events, name, wall_start) for name in miss_events)
                if value is not None
            ),
            None,
        )

        state["t"] = (pd.to_datetime(state["wall_time"]) - wall_start).dt.total_seconds()
        wrench["t"] = (pd.to_datetime(wrench["wall_time"]) - wall_start).dt.total_seconds()
        state = state[(state["t"] >= 0.0) & (state["t"] <= terminal + 0.5)].copy()
        wrench = wrench[(wrench["t"] >= 0.0) & (wrench["t"] <= terminal + 0.5)].copy()
        anchor = np.asarray(summary["provisional_surface_datum_world_point_mm"], dtype=float)
        state["x_local"] = state["x_mm"] - anchor[0]
        state["y_local"] = state["y_mm"] - anchor[1]
        state["depth"] = anchor[2] - state["z_mm"]
        force = np.linalg.norm(
            wrench[["fx_world_star_n", "fy_world_star_n", "fz_world_star_n"]].to_numpy(float),
            axis=1,
        )
        torque = 1000.0 * np.linalg.norm(
            wrench[["tx_world_star_nm", "ty_world_star_nm", "tz_world_star_nm"]].to_numpy(float),
            axis=1,
        )
        cone_path = directory / "conical_directservo_trace.csv"
        cone = None
        if cone_path.exists():
            cone = pd.read_csv(cone_path)
            cone["x_local"] = cone["actual_x_mm"] - anchor[0]
            cone["y_local"] = cone["actual_y_mm"] - anchor[1]

        posterior_path = directory / "learned_conical_direction_active.json"
        posterior = (
            json.loads(posterior_path.read_text(encoding="utf-8"))
            if posterior_path.exists()
            else None
        )
        runs.append(
            {
                "label": label,
                "color": color,
                "state": state,
                "wrench": wrench,
                "force": force,
                "torque": torque,
                "summary": summary,
                "posterior": posterior,
                "cone": cone,
                "probe_end": probe_end,
                "miss": miss,
                "capture": capture,
                "terminal": terminal,
                "depth_required": float(summary["measured_m12_socket_top_to_bottom_depth_mm"]),
            }
        )

    all_xy = np.vstack([run["state"][["x_local", "y_local"]].to_numpy(float) for run in runs])
    xlim = (float(all_xy[:, 0].min() - 0.55), float(all_xy[:, 0].max() + 0.55))
    ylim = (float(all_xy[:, 1].min() - 0.45), float(all_xy[:, 1].max() + 0.45))
    force_max = 1.08 * max(float(np.nanmax(run["force"])) for run in runs)
    torque_max = 1.08 * max(float(np.nanmax(run["torque"])) for run in runs)
    posterior_max = 1.10 * max(
        float(np.max(np.asarray(run["posterior"]["direction_probabilities"], dtype=float)))
        for run in runs
        if run["posterior"] is not None
    )

    fig = plt.figure(figsize=(7.15, 7.55))
    grid = fig.add_gridspec(
        4,
        4,
        left=0.082,
        right=0.955,
        top=0.945,
        bottom=0.085,
        hspace=0.53,
        wspace=0.44,
        height_ratios=[1.03, 1.05, 0.92, 0.70],
    )

    for column, run in enumerate(runs):
        color = run["color"]
        posterior = run["posterior"]

        ax = fig.add_subplot(grid[0, column], projection="polar")
        if posterior is not None:
            angles = np.radians(np.asarray(posterior["direction_class_angles_deg"], dtype=float))
            probs = np.asarray(posterior["direction_probabilities"], dtype=float)
            aps_classes = set(int(value) for value in posterior["aps_direction_classes"])
            bar_colors = [color if index in aps_classes else "#d8dde3" for index in range(len(probs))]
            edge_colors = [color if index in aps_classes else "#aeb5bd" for index in range(len(probs))]
            ax.bar(
                angles,
                probs,
                width=0.92 * 2.0 * np.pi / len(probs),
                color=bar_colors,
                edgecolor=edge_colors,
                linewidth=0.32,
                alpha=0.9,
            )
            top_angle = math.radians(float(posterior["top_direction_deg"]))
            ax.annotate(
                "",
                xy=(top_angle, float(np.max(probs)) * 1.07),
                xytext=(top_angle, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": COLORS["red"], "lw": 1.15},
            )
            ax.set_ylim(0, posterior_max)
            ax.set_yticklabels([])
            ax.text(
                0.5,
                -0.18,
                rf"$\hat\theta={posterior['top_direction_deg']:.1f}^\circ$, "
                rf"$|\mathcal{{A}}|={posterior['aps_set_size']}$, "
                rf"$\kappa={posterior['circular_concentration']:.2f}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.55,
                color=color,
            )
        else:
            ax.set_ylim(0, posterior_max)
            ax.set_yticklabels([])
            ax.text(
                0.5,
                0.52,
                "Posterior\nnot used",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7.0,
                color=COLORS["muted"],
            )
            ax.text(
                0.5,
                -0.18,
                r"Global spiral, $L^*=0$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.55,
                color=color,
            )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids([0, 90, 180, 270], labels=[r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"])
        ax.tick_params(axis="x", labelsize=5.8, pad=0.5)
        ax.grid(color=COLORS["grid"], lw=0.45)
        ax.spines["polar"].set_linewidth(0.65)
        ax.set_title(run["label"], color=color, fontweight="bold", fontsize=8.3, pad=8)

        ax = fig.add_subplot(grid[1, column])
        state = run["state"]
        ax.plot(state["x_local"], state["y_local"], color="#c7ccd2", lw=0.7, zorder=1)
        route_mask = state["t"] <= run["capture"]
        insertion_mask = state["t"] >= run["capture"]
        ax.plot(
            state.loc[route_mask, "x_local"],
            state.loc[route_mask, "y_local"],
            color=color,
            lw=1.35,
            zorder=2,
        )
        ax.plot(
            state.loc[insertion_mask, "x_local"],
            state.loc[insertion_mask, "y_local"],
            color=COLORS["red"],
            lw=1.15,
            zorder=2,
        )
        if run["cone"] is not None:
            ax.plot(
                run["cone"]["x_local"],
                run["cone"]["y_local"],
                color=COLORS["blue"],
                lw=0.9,
                zorder=3,
            )
        ax.scatter([0], [0], s=18, facecolor="white", edgecolor=COLORS["ink"], lw=0.65, zorder=4)
        if run["miss"] is not None:
            index = (state["t"] - run["miss"]).abs().idxmin()
            ax.scatter(
                state.loc[index, "x_local"],
                state.loc[index, "y_local"],
                marker="x",
                s=29,
                color=COLORS["orange"],
                lw=0.9,
                zorder=5,
            )
        index = (state["t"] - run["capture"]).abs().idxmin()
        ax.scatter(
            state.loc[index, "x_local"],
            state.loc[index, "y_local"],
            marker="*",
            s=43,
            color=COLORS["red"],
            edgecolor="white",
            lw=0.35,
            zorder=5,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-6, -3, 0, 3])
        ax.set_yticks([-4, 0, 4])
        ax.set_xlabel(r"$x-x_0$ [mm]", fontsize=6.3, labelpad=1)
        if column == 0:
            ax.set_ylabel(r"$y-y_0$ [mm]", fontsize=6.3, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=5.8, pad=1)
        ax.text(
            0.98,
            0.04,
            rf"$T={run['terminal']:.1f}$ s"
            "\n"
            rf"$L_{{xy}}={run['summary']['path_length_mm']:.1f}$ mm",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.45,
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "boxstyle": "round,pad=0.16", "alpha": 0.88},
        )
        finish_axis(ax, grid_axis="both")

        ax = fig.add_subplot(grid[2, column])
        wrench = run["wrench"]
        time_fraction = np.clip(wrench["t"].to_numpy(float) / run["terminal"], 0.0, 1.0)
        ax.plot(time_fraction, run["force"], color=color, lw=0.9)
        ax2 = ax.twinx()
        ax2.plot(time_fraction, run["torque"], color=COLORS["muted"], lw=0.75, ls="--")
        for event_time, event_color, style in [
            (run["probe_end"], COLORS["blue"], ":"),
            (run["miss"], COLORS["orange"], ":"),
            (run["capture"], COLORS["red"], "--"),
        ]:
            if event_time is not None:
                ax.axvline(event_time / run["terminal"], color=event_color, ls=style, lw=0.65)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, force_max)
        ax2.set_ylim(0, torque_max)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(r"Normalized execution time $t/T$", fontsize=6.1, labelpad=1)
        if column == 0:
            ax.set_ylabel(r"$\|F^*\|$ [N]", fontsize=6.3, color=color, labelpad=1)
        else:
            ax.set_yticklabels([])
        if column == 3:
            ax2.set_ylabel(r"$\|\tau^*\|$ [mN$\cdot$m]", fontsize=6.3, color=COLORS["muted"], labelpad=1)
        else:
            ax2.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=5.8, pad=1)
        ax2.tick_params(axis="y", labelsize=5.8, pad=1, colors=COLORS["muted"])
        finish_axis(ax, grid_axis="both")
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        ax = fig.add_subplot(grid[3, column])
        state_fraction = np.clip(state["t"].to_numpy(float) / run["terminal"], 0.0, 1.0)
        axial_progress = state["depth"].to_numpy(float)
        ax.plot(state_fraction, axial_progress, color=color, lw=1.0)
        ax.axhline(run["depth_required"], color=COLORS["muted"], ls="--", lw=0.7)
        ax.axvline(run["capture"] / run["terminal"], color=COLORS["red"], ls="--", lw=0.65)
        ax.axvline(1.0, color=COLORS["ink"], ls="--", lw=0.7)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.2, 1.05 * run["depth_required"])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(r"Normalized execution time $t/T$", fontsize=6.1, labelpad=1)
        if column == 0:
            ax.set_ylabel(r"Axial progress $d$ [mm]", fontsize=6.3, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=5.8, pad=1)
        finish_axis(ax, grid_axis="both")

    fig.text(0.015, 0.820, "Direction\nevidence", rotation=90, ha="center", va="center", fontsize=7.0, fontweight="bold")
    fig.text(0.015, 0.575, "Measured\nroute", rotation=90, ha="center", va="center", fontsize=7.0, fontweight="bold")
    fig.text(0.015, 0.350, "Contact\nload", rotation=90, ha="center", va="center", fontsize=7.0, fontweight="bold")
    fig.text(0.015, 0.150, "Axial\nprogress", rotation=90, ha="center", va="center", fontsize=7.0, fontweight="bold")
    legend_handles = [
        patches.Patch(facecolor="#d8dde3", edgecolor="#aeb5bd", label="posterior bin"),
        Line2D([0], [0], color=COLORS["red"], lw=1.1, label=r"top direction $\hat\theta$"),
        Line2D([0], [0], color=COLORS["blue"], lw=1.1, label="conical probe"),
        Line2D([0], [0], marker="x", color=COLORS["orange"], lw=0, label="negative evidence"),
        Line2D([0], [0], marker="*", color=COLORS["red"], lw=0, label="capture"),
        Line2D([0], [0], color=COLORS["muted"], lw=0.8, ls="--", label=r"$\|\tau^*\|$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.008),
        ncol=6,
        frameon=False,
        fontsize=5.55,
        columnspacing=0.85,
        handlelength=1.25,
    )
    save(fig, "hardware_m12_four_policy_evidence_matrix")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR,
                        help="Destination for new plots; input figures always remain in figures/.")
    OUTPUT_DIR = parser.parse_args().output_dir
    configure_style()
    records = load_trials()
    # Figs. 2 and 3 are maintained as supplied/hybrid publication composites.
    # Do not overwrite them with legacy Matplotlib layouts when regenerating
    # the remaining data-derived figures.  The frozen tactile raster remains
    # the data source consumed by tools/build_fig3_hybrid.m.
    plot_shortcut_geometry()
    plot_representation_ablation(records)
    plot_policy_keyframes()
    plot_budgeted_profiles(records)
    plot_hardware_trace()
    plot_hardware_same_start_pair()
    plot_hardware_four_policy_matrix()


if __name__ == "__main__":
    main()
