#!/usr/bin/env python3
"""Re-render the successful no-time-limit M12 Pure Helix hardware run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REAL_DIR / "data" / "M12_baseline_pure_outward_spiral_20260729_225552_920"
)
DEFAULT_OUTPUT = (
    REAL_DIR / "docs" / "paper_figures" / "m12_pure_helix_success_without_time_limit"
)

COLORS = {
    "x": "#D1495B",
    "y": "#2A9D55",
    "z": "#2468B4",
    "touchdown": "#7A8793",
    "helix": "#0072B2",
    "insert": "#7B2CBF",
    "command": "#E07A1F",
    "capture": "#D62828",
    "terminal": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def read_run(run_dir: Path) -> tuple[pd.DataFrame, ...]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    states = pd.read_csv(run_dir / "robot_state.csv")
    commands = pd.read_csv(run_dir / "commands.csv")
    wrench = pd.read_csv(run_dir / "world_star_relative_wrench.csv")
    transitions = pd.read_csv(run_dir / "transitions.csv")
    return summary, states, commands, wrench, transitions


def wall_time_to_elapsed(
    wall_time: str, reference: pd.DataFrame
) -> float:
    sample_wall = pd.to_datetime(reference["wall_time"]).astype("int64").to_numpy() / 1e9
    sample_elapsed = reference["elapsed_s"].to_numpy(dtype=float)
    target = pd.Timestamp(wall_time).value / 1e9
    return float(np.interp(target, sample_wall, sample_elapsed))


def transition_elapsed(
    transitions: pd.DataFrame, wrench: pd.DataFrame, to_state: str
) -> float:
    rows = transitions.loc[transitions["to_state"] == to_state]
    if rows.empty:
        raise ValueError(f"Transition to {to_state!r} was not found")
    return wall_time_to_elapsed(str(rows.iloc[0]["wall_time"]), wrench)


def phase_masks(phases: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    text = phases.astype(str)
    touchdown = text.str.startswith("TOUCHDOWN")
    helix = text.str.startswith("PURE_OUTWARD")
    insertion = (
        text.str.startswith("M12_INSERT")
        | text.str.startswith("INSERTION_")
        | text.str.startswith("BOTTOM_")
    )
    return touchdown, helix, insertion


def plot_segmented_path(ax: plt.Axes, states: pd.DataFrame) -> None:
    touchdown, helix, insertion = phase_masks(states["phase"])
    for mask, color, label, width in (
        (touchdown, COLORS["touchdown"], "Touchdown", 1.7),
        (helix, COLORS["helix"], "Pure helix search", 2.5),
        (insertion, COLORS["insert"], "Capture + insertion", 2.6),
    ):
        data = states.loc[mask].sort_values("elapsed_s")
        if data.empty:
            continue
        ax.plot(
            data["x_mm"], data["y_mm"], data["z_mm"],
            color=color, linewidth=width, label=label,
            solid_capstyle="round", zorder=4,
        )
        ax.scatter(
            data["x_mm"], data["y_mm"], data["z_mm"],
            color=color, s=7, alpha=0.42, depthshade=False, zorder=5,
        )


def plot_commanded_path(ax: plt.Axes, commands: pd.DataFrame) -> None:
    done = commands.loc[commands["status"] == "DONE"].copy()
    touchdown, helix, insertion = phase_masks(done["phase"])
    done = done.loc[touchdown | helix | insertion]
    done = done.dropna(subset=["post_x_mm", "post_y_mm", "post_z_mm"])
    done = done.sort_values("elapsed_s")
    ax.plot(
        done["post_x_mm"], done["post_y_mm"], done["post_z_mm"],
        color=COLORS["command"], linewidth=1.0, linestyle=(0, (4, 3)),
        alpha=0.78, label="Commanded endpoints", zorder=2,
    )


def style_3d_axes(ax: plt.Axes, states: pd.DataFrame) -> None:
    touchdown, helix, insertion = phase_masks(states["phase"])
    visible = states.loc[touchdown | helix | insertion]
    ranges = []
    for key in ("x_mm", "y_mm", "z_mm"):
        values = visible[key].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        pad = max(0.35, 0.06 * (hi - lo))
        ranges.append((lo - pad, hi + pad))
    ax.set_xlim(*ranges[0])
    ax.set_ylim(*ranges[1])
    ax.set_zlim(*ranges[2])
    ax.set_box_aspect(tuple(hi - lo for lo, hi in ranges))
    # Match the original MATLAB dashboard camera exactly: view(135, 24).
    # This keeps the WORLD X/Y screen directions consistent with the
    # low-resolution reference figure instead of mirroring the trajectory.
    ax.view_init(elev=24, azim=135)
    ax.set_proj_type("ortho")
    ax.set_xlabel("WORLD X [mm]", labelpad=10)
    ax.set_ylabel("WORLD Y [mm]", labelpad=10)
    ax.set_zlabel("WORLD Z [mm]", labelpad=10)
    ax.set_title("Registered peg-tip TCP trajectory", pad=15, fontweight="semibold")
    ax.grid(True, alpha=0.30)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 1.00, 1.0))
        axis.pane.set_edgecolor((0.82, 0.85, 0.89, 1.0))


def add_stage_spans(
    ax: plt.Axes,
    touchdown_s: float,
    search_s: float,
    capture_s: float,
    terminal_s: float,
) -> None:
    ax.axvspan(touchdown_s, search_s, color="#94A3B8", alpha=0.09, lw=0)
    ax.axvspan(search_s, capture_s, color="#38BDF8", alpha=0.09, lw=0)
    ax.axvspan(capture_s, terminal_s, color="#A855F7", alpha=0.07, lw=0)
    for x, color in (
        (search_s, COLORS["touchdown"]),
        (capture_s, COLORS["capture"]),
        (terminal_s, COLORS["terminal"]),
    ):
        ax.axvline(x, color=color, linewidth=1.15, linestyle=(0, (3, 3)), alpha=0.82)


def make_figure(
    summary: dict,
    states: pd.DataFrame,
    commands: pd.DataFrame,
    wrench: pd.DataFrame,
    transitions: pd.DataFrame,
    run_name: str,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.5, 8.5), facecolor="white")
    fig.subplots_adjust(
        left=0.035, right=0.985, bottom=0.095, top=0.855,
        wspace=0.18, hspace=0.34,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0), hspace=0.16, wspace=0.08)
    ax_path = fig.add_subplot(grid[:, 0], projection="3d")
    ax_force = fig.add_subplot(grid[0, 1])
    ax_torque = fig.add_subplot(grid[1, 1], sharex=ax_force)

    plot_commanded_path(ax_path, commands)
    plot_segmented_path(ax_path, states)
    style_3d_axes(ax_path, states)

    capture = np.asarray(summary["capture_pose_mm_rad"][:3], dtype=float)
    insertion_states = states.loc[
        states["phase"].astype(str).str.startswith(("M12_INSERT", "INSERTION_"))
    ].sort_values("elapsed_s")
    terminal = insertion_states.iloc[-1][["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    ax_path.scatter(*capture, marker="*", s=150, color=COLORS["capture"],
                    edgecolor="white", linewidth=0.8, depthshade=False, zorder=10)
    ax_path.scatter(*terminal, marker="D", s=55, color=COLORS["terminal"],
                    edgecolor="white", linewidth=0.7, depthshade=False, zorder=10)

    handles, labels = ax_path.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["capture"],
                   markeredgecolor="white", markersize=13, label="Capture"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["terminal"],
                   markeredgecolor="white", markersize=7, label="Terminal"),
        ]
    )
    labels.extend(["Capture", "Terminal"])
    ax_path.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                   frameon=True, framealpha=0.94, ncol=2)

    time = wrench["elapsed_s"].to_numpy(dtype=float)
    for component, color, label in (
        ("fx_world_star_n", COLORS["x"], r"$F_{X_W^*}$"),
        ("fy_world_star_n", COLORS["y"], r"$F_{Y_W^*}$"),
        ("fz_world_star_n", COLORS["z"], r"$F_{Z_W^*}$"),
    ):
        ax_force.plot(time, wrench[component], color=color, linewidth=1.35, label=label)
    for component, color, label in (
        ("tx_world_star_nm", COLORS["x"], r"$M_{X_W^*}$"),
        ("ty_world_star_nm", COLORS["y"], r"$M_{Y_W^*}$"),
        ("tz_world_star_nm", COLORS["z"], r"$M_{Z_W^*}$"),
    ):
        ax_torque.plot(time, wrench[component], color=color, linewidth=1.35, label=label)

    touchdown_s = transition_elapsed(transitions, wrench, "SHORT_EVENT_TRIGGER_TOUCHDOWN")
    search_s = transition_elapsed(transitions, wrench, "PURE_OUTWARD_ARCHIMEDEAN_SEARCH")
    capture_s = transition_elapsed(transitions, wrench, "PROVISIONAL_SENSOR_DOWNWARD_PROGRESS_CAPTURE")
    terminal_s = transition_elapsed(transitions, wrench, "BOTTOM_HOLD_AND_END")
    for ax in (ax_force, ax_torque):
        add_stage_spans(ax, touchdown_s, search_s, capture_s, terminal_s)
        ax.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.62)
        ax.axhline(0.0, color="#475569", linewidth=0.7, alpha=0.65)
        ax.set_xlim(0.0, max(400.0, float(np.ceil(time.max() / 10.0) * 10.0)))
        ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.94)
        ax.spines[["top", "right"]].set_visible(False)

    ax_force.set_ylabel("Force [N]")
    ax_force.set_title("WORLD* relative force @ Bota measurement origin",
                       fontweight="semibold")
    ax_force.tick_params(labelbottom=False)
    ax_torque.set_xlabel("Elapsed time [s]")
    ax_torque.set_ylabel("Torque [N m]")
    ax_torque.set_title("WORLD* relative torque @ same origin (no tip shift)",
                        fontweight="semibold")

    force_values = wrench[["fx_world_star_n", "fy_world_star_n", "fz_world_star_n"]].to_numpy()
    force_min, force_max = np.nanmin(force_values), np.nanmax(force_values)
    ax_force.set_ylim(np.floor(force_min - 0.8), np.ceil(force_max + 0.8))
    torque_values = wrench[["tx_world_star_nm", "ty_world_star_nm", "tz_world_star_nm"]].to_numpy()
    torque_min, torque_max = np.nanmin(torque_values), np.nanmax(torque_values)
    ax_torque.set_ylim(torque_min - 0.025, torque_max + 0.035)

    y_top = ax_force.get_ylim()[1]
    for x, label, color, align in (
        (search_s, "search", COLORS["touchdown"], "left"),
        (capture_s, "capture", COLORS["capture"], "right"),
        (terminal_s, "terminal", COLORS["terminal"], "right"),
    ):
        ax_force.annotate(
            label, xy=(x, y_top), xytext=(3 if align == "left" else -3, -5),
            textcoords="offset points", ha=align, va="top", color=color,
            fontsize=9, fontweight="semibold",
        )

    fig.suptitle(
        "M12 Pure Helix (Success, without time limit)",
        fontsize=18, fontweight="bold", y=0.982,
    )
    fig.text(
        0.5, 0.948,
        (
            f"Real-robot closed loop | run {run_name} | "
            f"capture offset [{summary['capture_offset_mm'][0]:+.3f}, "
            f"{summary['capture_offset_mm'][1]:+.3f}] mm | "
            f"final insertion depth {summary['insertion_depth_mm']:.3f} mm | "
            f"peak relative wrench {summary['peak_relative_force_n']:.2f} N / "
            f"{summary['peak_relative_torque_nm']:.3f} N m"
        ),
        ha="center", va="top", fontsize=10.2, color="#334155",
    )
    fig.text(
        0.5, 0.022,
        (
            "Measured sensor-to-TCP rotation; current and upright baselines rotated separately at the "
            "Bota origin. Relative signal only; no online gravity/tip-moment correction or certified tare."
        ),
        ha="center", va="bottom", fontsize=8.5, color="#475569",
    )
    return fig


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_stem = args.output_stem.resolve()
    summary, states, commands, wrench, transitions = read_run(run_dir)
    if summary.get("method_outcome") != "TASK_SUCCESS_CAPTURE_AND_INSERTION":
        raise RuntimeError("Selected run is not a successful capture-and-insertion result")
    if "persistent phase-shifted outward/inward Archimedean" not in str(
        summary.get("policy_description", "")
    ):
        raise RuntimeError("Selected run is not the no-stage-time-limit Pure Outward policy")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(summary, states, commands, wrench, transitions, run_dir.name)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
