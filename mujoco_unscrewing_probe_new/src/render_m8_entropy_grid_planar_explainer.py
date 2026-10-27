#!/usr/bin/env python3
"""Render an auditable top-view explanation of the selected M8 entropy-grid run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/current/diagnostics/historical_comparisons/three_method_race/manifest.json"
)
DEFAULT_OUT = ROOT / "outputs/current/active_comparison/entropy_grid_planar_explainer"
FONT_FAMILY = "Noto Sans CJK JP"


@dataclass(frozen=True)
class PlanarRun:
    pose_id: str
    initial_xy_mm: np.ndarray
    action_nodes_mm: np.ndarray
    capture_eligible: np.ndarray
    visited_xy_mm: np.ndarray
    observations: list[dict]
    capture_event_xy_mm: np.ndarray
    capture_radius_mm: float
    action_spacing_mm: float
    visual_bound_mm: float
    socket_apothem_mm: float
    head_radius_mm: float
    active_budget_s: float
    capture_time_s: float
    bottom_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def _disk_grid(center: np.ndarray, radius: float, spacing: float) -> np.ndarray:
    count = int(math.ceil(radius / spacing))
    values = np.arange(-count, count + 1, dtype=float) * spacing
    offsets = np.asarray(
        [
            (x, y)
            for x in values
            for y in values
            if x * x + y * y <= radius * radius + 1.0e-12
        ],
        dtype=float,
    )
    return center[None, :] + offsets


def load_run(path: Path) -> PlanarRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"]["entropy"]
    geometry = result["entropy_grid_geometry_mm"]
    config = result["entropy_grid_config"]
    initial = np.asarray(result["initial_xy_mm"], dtype=float)
    spacing = 1000.0 * float(config["action_spacing_m"])
    # This is the frozen visual 95% radius used by the selected six-method run.
    visual_bound = 4.895303872079853
    nodes = _disk_grid(initial, visual_bound, spacing)
    capture_radius = float(geometry["capture_radius"])
    eligible = np.linalg.norm(nodes, axis=1) <= capture_radius + 1.0e-12
    observations = result["entropy_grid_observations"]
    visited = np.asarray([item["action_xy_mm"] for item in observations], dtype=float)
    return PlanarRun(
        pose_id=str(payload["task"]["pose_id"]),
        initial_xy_mm=initial,
        action_nodes_mm=nodes,
        capture_eligible=eligible,
        visited_xy_mm=visited,
        observations=observations,
        capture_event_xy_mm=np.asarray(result["event_xy_mm"], dtype=float),
        capture_radius_mm=capture_radius,
        action_spacing_mm=spacing,
        visual_bound_mm=visual_bound,
        socket_apothem_mm=float(geometry["socket_top_apothem"]),
        head_radius_mm=float(geometry["head_outer_radius"]),
        active_budget_s=130.0,
        capture_time_s=float(result["time_to_inner_wall_s"]),
        bottom_time_s=float(result["time_to_bottom_s"]),
    )


def _hexagon(apothem: float) -> np.ndarray:
    circumradius = 2.0 * apothem / math.sqrt(3.0)
    angle = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    return np.column_stack((circumradius * np.cos(angle), circumradius * np.sin(angle)))


def _draw_geometry(ax: plt.Axes, run: PlanarRun, *, local: bool) -> None:
    ax.add_patch(
        Circle((0.0, 0.0), run.head_radius_mm, facecolor="#e5e7eb", edgecolor="#64748b", lw=1.4, zorder=0)
    )
    ax.add_patch(
        Polygon(_hexagon(run.socket_apothem_mm), closed=True, facecolor="#ffffff", edgecolor="#334155", lw=1.6, zorder=1)
    )
    ax.add_patch(
        Circle((0.0, 0.0), run.capture_radius_mm, facecolor="#86efac", edgecolor="#15803d", alpha=0.52, lw=2.0, zorder=4)
    )
    if not local:
        ax.add_patch(
            Circle(
                tuple(run.initial_xy_mm),
                run.visual_bound_mm,
                facecolor="none",
                edgecolor="#94a3b8",
                linestyle="--",
                lw=1.4,
                zorder=2,
            )
        )


def _draw_nodes_and_path(ax: plt.Axes, run: PlanarRun, upto: int | None = None) -> None:
    ax.scatter(
        run.action_nodes_mm[:, 0],
        run.action_nodes_mm[:, 1],
        s=8,
        color="#94a3b8",
        alpha=0.55,
        linewidths=0,
        label="candidate action nodes",
        zorder=3,
    )
    eligible = run.action_nodes_mm[run.capture_eligible]
    ax.scatter(
        eligible[:, 0],
        eligible[:, 1],
        s=62,
        facecolor="#22c55e",
        edgecolor="#14532d",
        linewidth=1.0,
        label="nominal capture-eligible nodes",
        zorder=6,
    )
    count = len(run.visited_xy_mm) if upto is None else max(1, min(upto, len(run.visited_xy_mm)))
    visited = run.visited_xy_mm[:count]
    if len(visited) > 1:
        ax.plot(
            visited[:, 0],
            visited[:, 1],
            color="#7c3aed",
            lw=2.5,
            marker="o",
            markersize=4.5,
            label="commanded entropy-grid route",
            zorder=7,
        )
    else:
        ax.scatter(visited[:, 0], visited[:, 1], s=35, color="#7c3aed", zorder=7)
    for index, point in enumerate(visited, start=1):
        ax.annotate(
            str(index),
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            color="#5b21b6",
            zorder=8,
        )
    ax.scatter(
        [run.initial_xy_mm[0]],
        [run.initial_xy_mm[1]],
        marker="*",
        s=180,
        facecolor="#f59e0b",
        edgecolor="#7c2d12",
        linewidth=1.2,
        label="initial tool position",
        zorder=9,
    )
    if upto is None or upto >= len(run.visited_xy_mm):
        ax.scatter(
            [run.capture_event_xy_mm[0]],
            [run.capture_event_xy_mm[1]],
            marker="X",
            s=130,
            facecolor="#ef4444",
            edgecolor="#7f1d1d",
            linewidth=1.2,
            label="measured capture event",
            zorder=10,
        )
    ax.scatter([0.0], [0.0], marker="+", s=120, color="#111827", linewidth=2.0, label="true center (offline)", zorder=11)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.tick_params(labelsize=8)


def render_static(run: PlanarRun, path: Path) -> None:
    plt.rcParams.update({"font.family": FONT_FAMILY, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.22, 0.78], wspace=0.12)
    global_ax = fig.add_subplot(grid[0, 0])
    local_ax = fig.add_subplot(grid[0, 1])
    _draw_geometry(global_ax, run, local=False)
    _draw_nodes_and_path(global_ax, run)
    _style_axis(global_ax)
    margin = 0.45
    global_ax.set_xlim(run.initial_xy_mm[0] - run.visual_bound_mm - margin, run.initial_xy_mm[0] + run.visual_bound_mm + margin)
    global_ax.set_ylim(run.initial_xy_mm[1] - run.visual_bound_mm - margin, run.initial_xy_mm[1] + run.visual_bound_mm + margin)
    global_ax.set_title("全局：视觉支持域、熵网格与访问轨迹", fontsize=15, weight="bold")
    global_ax.legend(loc="lower left", fontsize=8, framealpha=0.95, ncol=2)

    _draw_geometry(local_ax, run, local=True)
    _draw_nodes_and_path(local_ax, run)
    _style_axis(local_ax)
    local_ax.set_xlim(-0.85, 0.85)
    local_ax.set_ylim(-0.85, 0.85)
    local_ax.set_title("局部：保守 capture tube 与实际接触捕获", fontsize=15, weight="bold")
    local_ax.annotate(
        "nominal endpoint 6",
        run.visited_xy_mm[-1],
        xytext=(18, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#5b21b6"},
        fontsize=9,
        color="#5b21b6",
    )
    local_ax.annotate(
        "measured tip enters capture region",
        run.capture_event_xy_mm,
        xytext=(16, -44),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#b91c1c"},
        fontsize=9,
        color="#b91c1c",
    )

    fig.suptitle(
        f"M8 熵网格平面解释 | {run.pose_id} | h={run.action_spacing_mm:.3f} mm, "
        f"r_eff={run.capture_radius_mm:.2f} mm",
        fontsize=19,
        weight="bold",
        color="#17375e",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "绿色节点：nominal XY 落入保守 capture disk；红叉：MuJoCo 中实际 measured tip 捕获位置。"
        "真中心只作离线解释，不进入在线 entropy-grid 控制。",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_video(run: PlanarRun, path: Path, poster_path: Path, fps: int) -> int:
    plt.rcParams.update({"font.family": FONT_FAMILY, "axes.unicode_minus": False})
    holds = [1.0, 1.25, 1.25, 1.25, 1.25, 1.5]
    total_frames = int(round(sum(holds) * fps))
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    final_frame: np.ndarray | None = None
    frame_index = 0
    try:
        for step, hold_s in enumerate(holds, start=1):
            frames = int(round(hold_s * fps))
            observation = run.observations[step - 1]
            for _ in range(frames):
                fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f8fafc")
                grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], left=0.055, right=0.975, top=0.86, bottom=0.12, wspace=0.13)
                ax = fig.add_subplot(grid[0, 0])
                zoom = fig.add_subplot(grid[0, 1])
                for target, local in ((ax, False), (zoom, True)):
                    _draw_geometry(target, run, local=local)
                    _draw_nodes_and_path(target, run, upto=step)
                    _style_axis(target)
                margin = 0.35
                ax.set_xlim(run.initial_xy_mm[0] - run.visual_bound_mm - margin, run.initial_xy_mm[0] + run.visual_bound_mm + margin)
                ax.set_ylim(run.initial_xy_mm[1] - run.visual_bound_mm - margin, run.initial_xy_mm[1] + run.visual_bound_mm + margin)
                zoom.set_xlim(-0.85, 0.85)
                zoom.set_ylim(-0.85, 0.85)
                ax.set_title("candidate grid + commanded route", fontsize=12, weight="bold")
                zoom.set_title("capture-region close-up", fontsize=12, weight="bold")
                if step < len(holds):
                    status = f"probe {step}: {observation['observation']} → update belief → choose next node"
                    entropy = observation.get("posterior_entropy_nats")
                    radius = observation.get("posterior_credible_radius_95_mm")
                    detail = (
                        ""
                        if entropy is None
                        else f"posterior H={entropy:.2f} nat | r95={radius:.2f} mm"
                    )
                else:
                    status = "segment 6: sensor capture preempts further grid motion"
                    detail = f"capture={run.capture_time_s:.2f} s | CAD bottom={run.bottom_time_s:.2f} s | budget={run.active_budget_s:.0f} s"
                fig.suptitle("M8 entropy-grid closed loop: observe → update → move → observe", fontsize=17, weight="bold", color="#17375e", y=0.965)
                fig.text(0.5, 0.905, status, ha="center", fontsize=12.5, color="#5b21b6", weight="bold")
                fig.text(0.5, 0.055, detail, ha="center", fontsize=11, color="#334155")
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                plt.close(fig)
                writer.append_data(frame)
                final_frame = frame
                frame_index += 1
    finally:
        writer.close()
    if final_frame is None or frame_index != total_frames:
        raise RuntimeError("video rendering did not produce the expected frames")
    imageio.imwrite(poster_path, final_frame)
    return frame_index


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    run = load_run(args.source)
    figure = args.out / "m8_entropy_grid_planar_map.png"
    video = args.out / "m8_entropy_grid_planar_explainer.mp4"
    poster = args.out / "m8_entropy_grid_planar_explainer_poster.png"
    render_static(run, figure)
    frames = render_video(run, video, poster, args.fps)
    manifest = {
        "source": str(args.source.relative_to(ROOT)),
        "pose_id": run.pose_id,
        "interpretation_only_true_center": True,
        "online_true_center_used": False,
        "initial_xy_mm": run.initial_xy_mm.tolist(),
        "visual_bound_mm": run.visual_bound_mm,
        "action_spacing_mm": run.action_spacing_mm,
        "action_node_count": int(len(run.action_nodes_mm)),
        "capture_radius_mm": run.capture_radius_mm,
        "nominal_capture_eligible_nodes_mm": run.action_nodes_mm[run.capture_eligible].tolist(),
        "visited_action_nodes_mm": run.visited_xy_mm.tolist(),
        "measured_capture_event_xy_mm": run.capture_event_xy_mm.tolist(),
        "capture_time_s": run.capture_time_s,
        "bottom_time_s": run.bottom_time_s,
        "active_time_budget_s": run.active_budget_s,
        "claim_boundary": (
            "The green nodes satisfy the conservative nominal XY capture-radius test. "
            "The red capture event is the measured simulated tip position under compliant contact. "
            "This selected run explains the closed-loop mechanism; it is not distribution-level evidence."
        ),
        "figure": str(figure.relative_to(ROOT)),
        "video": str(video.relative_to(ROOT)),
        "poster": str(poster.relative_to(ROOT)),
        "fps": args.fps,
        "frames": frames,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {figure}")
    print(f"wrote {video} ({frames} frames)")


if __name__ == "__main__":
    main()
