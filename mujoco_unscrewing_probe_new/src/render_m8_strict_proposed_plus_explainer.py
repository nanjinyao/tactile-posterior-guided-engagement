#!/usr/bin/env python3
"""Rerun and explain one strict-budget M8 Proposed+ mechanism example.

The MuJoCo rollout uses the same task and policy arguments as
``run_m8_strict_complete_loop_audit._run_one(..., "proposed_plus")``.  The
only additions are video and trace outputs.  The top-view figure is built
after the rollout from the measured trace and the serialized policy record;
true center is used only for offline explanation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

if not os.environ.get("DISPLAY"):
    # The explainer is normally rendered over SSH.  Select MuJoCo's headless
    # backend before importing modules that initialize its renderer, while
    # respecting an explicit caller override.
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
import torch

from src.entropy_grid_baseline import (
    DiskCoverageCertificate,
    certify_disk_grid_coverage,
    disk_grid,
)
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import RunResult, run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
)
from src.run_m8_strict_complete_loop_audit import (
    ACTIVE_TIME_BUDGET_S,
    _strict_bottom_success,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_MANIFEST = (
    ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop/manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/active_comparison/proposed_plus_strict_explainer"
)
DEFAULT_POSE_ID = "sobol_008"
EXPECTED_PADDED_ACTION_NODES = 749
RUNTIME_METHOD = "hybrid"
DISPLAY_METHOD = "proposed_plus"


@dataclass(frozen=True)
class TraceData:
    measured_xy_mm: np.ndarray
    stage_group: tuple[str, ...]
    stage: tuple[str, ...]
    clock_s: np.ndarray
    active_time_s: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one strict M8 Proposed+ MuJoCo rollout and an offline "
            "measured-path/lattice explanation."
        )
    )
    parser.add_argument(
        "--strict-manifest",
        type=Path,
        default=DEFAULT_STRICT_MANIFEST,
        help="strict 24-pose manifest containing the frozen task",
    )
    parser.add_argument("--pose-id", default=DEFAULT_POSE_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise ValueError(f"output must remain inside project root: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_task(manifest_path: Path, pose_id: str) -> PoseTask:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in payload.get("tasks", []) if item.get("pose_id") == pose_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one task {pose_id!r} in {manifest_path}, found {len(matches)}"
        )
    task = PoseTask(**matches[0])
    if task.sensor_noise_seed is None:
        raise ValueError("strict explainer requires the frozen sensor-noise seed")
    return task


def _run_strict_proposed_plus(
    task: PoseTask,
    *,
    video_path: Path,
    trace_path: Path,
) -> RunResult:
    """Mirror strict-audit `_run_one` and add only render/trace destinations."""

    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    return run_method(
        RUNTIME_METHOD,
        output_path=video_path,
        trace_output_path=trace_path,
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=runtime,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        max_transport_m=None,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed), delay_s=task.sensor_delay_s
        ),
        observation_label="shared pre-hardware wrench engineering prior",
        random_search_seed=task.posterior_seed + 700000,
        cad_conditioned_insertion=True,
        conditioned_on_ambiguous_top_contact=True,
    )


def _load_trace(path: Path) -> TraceData:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    required = {"tip_x_m", "tip_y_m", "stage_group", "stage", "clock_s", "active_time_s"}
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"trace is missing columns: {sorted(missing)}")
    measured = 1000.0 * np.asarray(
        [[float(row["tip_x_m"]), float(row["tip_y_m"])] for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(measured)):
        raise RuntimeError("trace contains non-finite measured tip XY")
    return TraceData(
        measured_xy_mm=measured,
        stage_group=tuple(row["stage_group"] for row in rows),
        stage=tuple(row["stage"] for row in rows),
        clock_s=np.asarray([float(row["clock_s"]) for row in rows], dtype=float),
        active_time_s=np.asarray(
            [float(row["active_time_s"]) for row in rows], dtype=float
        ),
    )


def _commanded_endpoints_mm(result: dict) -> np.ndarray:
    points = [
        item["action_xy_mm"]
        for item in (result.get("entropy_grid_observations") or [])
        if item.get("action_xy_mm") is not None
    ]
    if not points:
        return np.empty((0, 2), dtype=float)
    endpoints = np.asarray(points, dtype=float)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2 or not np.all(np.isfinite(endpoints)):
        raise RuntimeError("invalid commanded adaptive endpoints in RunResult")
    return endpoints


def _lattice_from_result(
    task: PoseTask,
    result: dict,
) -> tuple[np.ndarray, np.ndarray, float, float, DiskCoverageCertificate]:
    config = result.get("entropy_grid_config")
    geometry = result.get("entropy_grid_geometry_mm")
    if not isinstance(config, dict) or not isinstance(geometry, dict):
        raise RuntimeError("Proposed+ result did not serialize its entropy-grid contract")
    spacing_m = float(config["action_spacing_m"])
    capture_radius_m = 0.001 * float(geometry["capture_radius"])
    configured_padding = config.get("action_padding_m")
    requested_padding_m = (
        capture_radius_m
        if configured_padding is None
        else float(configured_padding)
    )
    effective_padding_m = max(requested_padding_m, spacing_m / math.sqrt(2.0))
    initial_xy_m = np.array([task.x_m, task.y_m], dtype=float)
    nodes_m = disk_grid(
        initial_xy_m,
        VISUAL_BOUND_M + effective_padding_m,
        spacing_m,
    )
    if len(nodes_m) != EXPECTED_PADDED_ACTION_NODES:
        raise RuntimeError(
            "strict M8 padded action lattice drifted: "
            f"expected {EXPECTED_PADDED_ACTION_NODES}, got {len(nodes_m)}"
        )
    certificate = certify_disk_grid_coverage(
        nodes_m,
        initial_xy_m,
        support_radius_m=VISUAL_BOUND_M,
        lattice_spacing_m=spacing_m,
        effective_capture_radius_m=capture_radius_m,
        dense_subdivisions_per_cell=8,
        boundary_sample_count=8192,
    )
    if not certificate.certified:
        raise RuntimeError("the reconstructed padded M8 action lattice is not certified")
    offset = nodes_m - initial_xy_m[None, :]
    inside_visual = np.linalg.norm(offset, axis=1) <= VISUAL_BOUND_M + 1.0e-12
    return (
        1000.0 * nodes_m,
        inside_visual,
        1000.0 * spacing_m,
        1000.0 * effective_padding_m,
        certificate,
    )


def _hexagon(apothem_mm: float) -> np.ndarray:
    radius = 2.0 * apothem_mm / math.sqrt(3.0)
    angle = np.deg2rad(np.arange(6, dtype=float) * 60.0 + 30.0)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def _decimate(points: np.ndarray, maximum: int = 18000) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
    return points[indices]


def _adaptive_segments(trace: TraceData) -> list[np.ndarray]:
    mask = np.asarray(
        [group == "hybrid_entropy_probe" for group in trace.stage_group],
        dtype=bool,
    )
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [trace.measured_xy_mm[part] for part in np.split(indices, split)]


def _capture_trace_record(trace: TraceData, event_xy_mm: np.ndarray | None) -> dict | None:
    if event_xy_mm is None:
        return None
    distance = np.linalg.norm(trace.measured_xy_mm - event_xy_mm[None, :], axis=1)
    index = int(np.argmin(distance))
    return {
        "row_index": index,
        "clock_s": float(trace.clock_s[index]),
        "active_time_s": float(trace.active_time_s[index]),
        "stage_group": trace.stage_group[index],
        "stage": trace.stage[index],
        "nearest_trace_distance_mm": float(distance[index]),
    }


def _render_planar_map(
    path: Path,
    *,
    task: PoseTask,
    result: dict,
    trace: TraceData,
    nodes_mm: np.ndarray,
    inside_visual: np.ndarray,
    spacing_mm: float,
    effective_padding_mm: float,
    certificate: DiskCoverageCertificate,
) -> None:
    initial_mm = np.array([1000.0 * task.x_m, 1000.0 * task.y_m])
    geometry = result["entropy_grid_geometry_mm"]
    capture_radius_mm = float(geometry["capture_radius"])
    event = result.get("event_xy_mm")
    event_mm = None if event is None else np.asarray(event, dtype=float)
    endpoints = _commanded_endpoints_mm(result)
    capture_eligible = np.linalg.norm(nodes_mm, axis=1) <= capture_radius_mm + 1.0e-12

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(16, 9), dpi=170, facecolor="#f8fafc")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.42, 0.58],
        left=0.055,
        right=0.975,
        top=0.89,
        bottom=0.09,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("white")

    head_radius_mm = float(geometry["head_outer_radius"])
    socket_apothem_mm = float(geometry["socket_top_apothem"])
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            head_radius_mm,
            facecolor="#e2e8f0",
            edgecolor="#64748b",
            lw=1.2,
            alpha=0.45,
            zorder=0,
        )
    )
    ax.add_patch(
        Polygon(
            _hexagon(socket_apothem_mm),
            closed=True,
            facecolor="white",
            edgecolor="#334155",
            lw=1.5,
            alpha=0.75,
            zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            tuple(initial_mm),
            1000.0 * VISUAL_BOUND_M,
            facecolor="none",
            edgecolor="#475569",
            linestyle="--",
            lw=1.8,
            label="visual-support disk boundary",
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            capture_radius_mm,
            facecolor="#86efac",
            edgecolor="#15803d",
            lw=2.0,
            alpha=0.62,
            label=r"offline $r_{eff}$ capture disk",
            zorder=5,
        )
    )

    padding_nodes = nodes_mm[~inside_visual]
    support_nodes = nodes_mm[inside_visual]
    ax.scatter(
        padding_nodes[:, 0],
        padding_nodes[:, 1],
        s=12,
        color="#06b6d4",
        alpha=0.65,
        linewidths=0,
        label="padding-only lattice nodes",
        zorder=3,
    )
    ax.scatter(
        support_nodes[:, 0],
        support_nodes[:, 1],
        s=8,
        color="#94a3b8",
        alpha=0.48,
        linewidths=0,
        label="lattice nodes inside visual disk",
        zorder=3,
    )
    eligible = nodes_mm[capture_eligible]
    if len(eligible):
        ax.scatter(
            eligible[:, 0],
            eligible[:, 1],
            s=62,
            facecolor="#22c55e",
            edgecolor="#14532d",
            linewidth=1.0,
            label="nominal capture-eligible nodes",
            zorder=7,
        )

    full_path = _decimate(trace.measured_xy_mm)
    ax.plot(
        full_path[:, 0],
        full_path[:, 1],
        color="#60a5fa",
        lw=1.0,
        alpha=0.55,
        label="actual measured tip XY (full trace)",
        zorder=6,
    )
    adaptive_segments = _adaptive_segments(trace)
    first_adaptive = True
    for segment in adaptive_segments:
        segment = _decimate(segment, maximum=6000)
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="#1d4ed8",
            lw=2.3,
            alpha=0.92,
            label="measured adaptive-contact path" if first_adaptive else None,
            zorder=8,
        )
        first_adaptive = False

    if len(endpoints):
        # Proposed+ reaches a posterior-shortcut anchor before its adaptive
        # endpoint sequence. Use the first measured adaptive-contact sample as
        # the route origin instead of falsely drawing a direct initial->node-1
        # command that the controller never issued.
        route_origin = (
            adaptive_segments[0][0]
            if adaptive_segments and len(adaptive_segments[0])
            else initial_mm
        )
        commanded_route = np.vstack((route_origin[None, :], endpoints))
        ax.plot(
            commanded_route[:, 0],
            commanded_route[:, 1],
            color="#f97316",
            lw=1.4,
            linestyle=":",
            alpha=0.85,
            label="commanded adaptive endpoint sequence",
            zorder=8,
        )
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            marker="D",
            s=48,
            facecolor="#fb923c",
            edgecolor="#9a3412",
            linewidth=1.0,
            zorder=9,
        )
        for index, point in enumerate(endpoints, start=1):
            ax.annotate(
                str(index),
                point,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#9a3412",
                zorder=10,
            )

    ax.scatter(
        [initial_mm[0]],
        [initial_mm[1]],
        marker="*",
        s=230,
        facecolor="#facc15",
        edgecolor="#713f12",
        linewidth=1.3,
        label="initial XY",
        zorder=11,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=190,
        color="#111827",
        linewidth=2.4,
        label="true center (offline only)",
        zorder=12,
    )
    if event_mm is not None:
        ax.scatter(
            [event_mm[0]],
            [event_mm[1]],
            marker="X",
            s=150,
            facecolor="#ef4444",
            edgecolor="#7f1d1d",
            linewidth=1.3,
            label="measured capture event",
            zorder=13,
        )

    extent = max(
        head_radius_mm,
        float(np.max(np.abs(nodes_mm))) + 0.45,
        float(np.max(np.abs(full_path))) + 0.45,
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e2e8f0", linewidth=0.65)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        "Offline top view: padded lattice, commands, and measured compliant motion",
        fontsize=14,
        weight="bold",
        color="#17375e",
    )
    ax.legend(loc="upper left", fontsize=7.8, ncol=2, framealpha=0.95)

    info.axis("off")
    strict_success = _strict_bottom_success(result)
    capture_text = (
        "none"
        if result.get("time_to_inner_wall_s") is None
        else f"{float(result['time_to_inner_wall_s']):.2f} s"
    )
    bottom_text = (
        "not verified"
        if result.get("time_to_bottom_s") is None
        else f"{float(result['time_to_bottom_s']):.2f} s"
    )
    rows = [
        ("SELECTED STRICT MECHANISM CASE", "#0f766e", 13, "bold"),
        (f"pose: {task.pose_id}", "#0f172a", 11, "bold"),
        ("", "#0f172a", 8, "normal"),
        (f"padded action nodes: {len(nodes_mm)}", "#0f172a", 10, "normal"),
        (f"inside visual disk: {int(np.sum(inside_visual))}", "#0f172a", 10, "normal"),
        (f"spacing h: {spacing_mm:.4f} mm", "#0f172a", 10, "normal"),
        (f"effective padding: {effective_padding_mm:.4f} mm", "#0f172a", 10, "normal"),
        (f"analytic gap h/sqrt(2): {1000.0 * certificate.analytic_worst_case_gap_m:.4f} mm", "#0f172a", 10, "normal"),
        (f"r_eff: {capture_radius_mm:.4f} mm", "#0f172a", 10, "normal"),
        (f"coverage certificate: {'PASS' if certificate.certified else 'FAIL'}", "#15803d" if certificate.certified else "#b91c1c", 10, "bold"),
        ("", "#0f172a", 8, "normal"),
        (f"trace samples: {len(trace.measured_xy_mm)}", "#0f172a", 10, "normal"),
        (f"adaptive endpoints: {len(endpoints)}", "#0f172a", 10, "normal"),
        (f"capture: {capture_text}", "#0f172a", 10, "normal"),
        (f"CAD bottom: {bottom_text}", "#0f172a", 10, "normal"),
        (f"terminal: {result.get('terminal_verification_mode')}", "#0f172a", 10, "normal"),
        (f"exit: {result.get('result')}", "#0f172a", 10, "normal"),
        (f"strict success: {strict_success}", "#15803d" if strict_success else "#b91c1c", 10, "bold"),
        (f"active time: {float(result['total_active_time_s']):.2f} / {ACTIVE_TIME_BUDGET_S:.0f} s", "#0f172a", 10, "normal"),
        ("", "#0f172a", 8, "normal"),
        ("TRUE CENTER IS OFFLINE ONLY", "#b91c1c", 10, "bold"),
        ("The controller never receives the", "#475569", 9.5, "normal"),
        ("black center marker or green disk.", "#475569", 9.5, "normal"),
        ("", "#0f172a", 8, "normal"),
        ("Mechanism example, not", "#7c3aed", 10.5, "bold"),
        ("distribution-level evidence.", "#7c3aed", 10.5, "bold"),
    ]
    y = 0.97
    for text, color, size, weight in rows:
        info.text(
            0.04,
            y,
            text,
            transform=info.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=size,
            weight=weight,
        )
        y -= 0.043 if text else 0.024

    fig.suptitle(
        "M8 Proposed+ strict closed loop | observe -> update -> guarded motion -> capture -> CAD bottom",
        fontsize=17,
        weight="bold",
        color="#17375e",
    )
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _extract_last_video_frame(video_path: Path, poster_path: Path) -> None:
    reader = imageio.get_reader(video_path)
    try:
        count = int(reader.count_frames())
        if count <= 0:
            raise RuntimeError("video has no frames")
        frame = reader.get_data(count - 1)
    except Exception:
        reader.close()
        reader = imageio.get_reader(video_path)
        frame = None
        for candidate in reader:
            frame = candidate
        if frame is None:
            raise RuntimeError("video has no decodable frames")
    finally:
        reader.close()
    imageio.imwrite(poster_path, np.asarray(frame))


def main() -> int:
    args = parse_args()
    strict_manifest = args.strict_manifest.resolve()
    output = args.output.resolve()
    _relative(output)
    output.mkdir(parents=True, exist_ok=True)

    task = _load_task(strict_manifest, args.pose_id)
    video_path = output / f"proposed_plus_strict_{task.pose_id}.mp4"
    trace_path = output / "trace.csv"
    result_path = output / "result.json"
    poster_path = output / "poster.png"
    planar_path = output / "planar_map.png"
    manifest_path = output / "manifest.json"

    result = _run_strict_proposed_plus(
        task,
        video_path=video_path,
        trace_path=trace_path,
    )
    result_dict = asdict(result)
    _write_json(
        result_path,
        {
            "task": asdict(task),
            "display_method": DISPLAY_METHOD,
            "runtime_method": RUNTIME_METHOD,
            "strict_bottom_success": _strict_bottom_success(result_dict),
            "result": result_dict,
        },
    )

    trace = _load_trace(trace_path)
    nodes_mm, inside_visual, spacing_mm, padding_mm, certificate = (
        _lattice_from_result(task, result_dict)
    )
    _render_planar_map(
        planar_path,
        task=task,
        result=result_dict,
        trace=trace,
        nodes_mm=nodes_mm,
        inside_visual=inside_visual,
        spacing_mm=spacing_mm,
        effective_padding_mm=padding_mm,
        certificate=certificate,
    )
    _extract_last_video_frame(video_path, poster_path)

    event = result_dict.get("event_xy_mm")
    event_xy = None if event is None else np.asarray(event, dtype=float)
    endpoints = _commanded_endpoints_mm(result_dict)
    manifest = {
        "experiment": "M8_strict_Proposed_plus_selected_mechanism_explainer",
        "created_utc": _utc_now(),
        "source_strict_manifest": _relative(strict_manifest),
        "pose_id": task.pose_id,
        "task": asdict(task),
        "method": {
            "display_name": DISPLAY_METHOD,
            "runtime_method": RUNTIME_METHOD,
            "strict_parameter_source": "run_m8_strict_complete_loop_audit._run_one",
        },
        "contract": {
            "selected_mechanism_example_not_distribution_evidence": True,
            "conditioned_on_ambiguous_top_contact": True,
            "hard_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "budget_is_accumulated_simulated_active_controller_time": True,
            "continue_to_cad_bottom": True,
            "online_true_center_used": False,
            "online_contact_tags_used": False,
            "true_center_in_planar_map_is_offline_interpretation_only": True,
        },
        "action_lattice": {
            "visual_disk_center_xy_mm": [1000.0 * task.x_m, 1000.0 * task.y_m],
            "visual_support_radius_mm": 1000.0 * VISUAL_BOUND_M,
            "visual_boundary_drawn": True,
            "padded_action_node_count": int(len(nodes_mm)),
            "nodes_inside_visual_disk": int(np.sum(inside_visual)),
            "padding_only_nodes": int(np.sum(~inside_visual)),
            "spacing_mm": spacing_mm,
            "effective_padding_mm": padding_mm,
            "coverage_certificate": asdict(certificate),
        },
        "trajectory": {
            "source": "trace.csv measured tip_x_m/tip_y_m",
            "measured_trace_samples": int(len(trace.measured_xy_mm)),
            "adaptive_stage_group": "hybrid_entropy_probe",
            "commanded_adaptive_endpoints_mm": endpoints.tolist(),
            "capture_event_xy_mm": None if event_xy is None else event_xy.tolist(),
            "capture_event_trace_record": _capture_trace_record(trace, event_xy),
            "effective_capture_radius_mm": float(
                result_dict["entropy_grid_geometry_mm"]["capture_radius"]
            ),
        },
        "capture": {
            "sensor_handoff": bool(result_dict.get("inner_wall_confirmed")),
            "offline_capture_validated": bool(
                result_dict.get("offline_capture_validated")
            ),
            "time_to_capture_s": result_dict.get("time_to_inner_wall_s"),
            "mode": result_dict.get("capture_mode"),
            "event_xy_mm": result_dict.get("event_xy_mm"),
        },
        "bottom": {
            "reached": bool(result_dict.get("bottom_reached")),
            "time_s": result_dict.get("time_to_bottom_s"),
            "depth_error_mm": result_dict.get("bottom_depth_error_mm"),
            "terminal_verification_mode": result_dict.get(
                "terminal_verification_mode"
            ),
        },
        "exit": {
            "result": result_dict.get("result"),
            "strict_bottom_success": _strict_bottom_success(result_dict),
            "total_active_time_s": result_dict.get("total_active_time_s"),
            "within_hard_active_budget": bool(
                float(result_dict["total_active_time_s"])
                <= ACTIVE_TIME_BUDGET_S + 1.0e-9
            ),
            "fallback_budget_extension_s": result_dict.get(
                "fallback_budget_extension_s"
            ),
            "hard_stop": bool(result_dict.get("hard_stop")),
            "hard_stop_reason": result_dict.get("hard_stop_reason"),
            "funnel_abort_reason": result_dict.get("funnel_abort_reason"),
            "budget_exhaustion_stage": result_dict.get("budget_exhaustion_stage"),
        },
        "claim_boundary": (
            "This selected MuJoCo rollout explains the strict Proposed+ mechanism; "
            "it is not distribution-level evidence. The true center, capture disk, "
            "capture-eligible node labels, and offline capture validation are used "
            "only after execution for explanation/scoring. The run is conditioned "
            "on the declared ambiguous-top-contact task set and does not establish "
            "an upstream router, hardware transfer, passivity, threaded unscrewing, "
            "extraction, or universal superiority."
        ),
        "artifacts": {
            "video": _artifact(video_path),
            "trace": _artifact(trace_path),
            "result": _artifact(result_path),
            "poster": _artifact(poster_path),
            "planar_map": _artifact(planar_path),
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": _relative(output), "exit": manifest["exit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
