#!/usr/bin/env python3
"""Replay and explain the frozen strict M8 entropy-grid rollout.

This module deliberately keeps three different finite sets separate:

* 1201 discrete hole-center hypotheses used by the Bayesian belief;
* 673 action-lattice nodes inside the declared visual-support disk;
* 76 boundary-padding action nodes used to close finite-disk coverage gaps.

The controller selects five actions in the frozen ``sobol_003`` rollout.  It
does not traverse all 749 action candidates.  The MuJoCo top view, true socket
center, CAD clearance/chamfer disks, and node eligibility colours are offline
explanations only.  The saved measured trace is produced by replaying the same
task and strict runtime contract as the frozen 24-pose audit.
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
import shutil
from typing import Iterable

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.entropy_grid_baseline import EntropyGridBelief, EntropyGridConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_sampling_position_map import (
    _project_world,
    render_mujoco_top_view,
)
from src.run_m8_fair_policy_benchmark import (
    MODEL_ROOT,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
)
from src.run_m8_strict_complete_loop_audit import (
    ACTIVE_TIME_BUDGET_S,
    _run_with_runtime,
    _strict_bottom_success,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIAL = (
    ROOT
    / "outputs/current/evaluations/m8_strict_24_complete_loop/trials/sobol_003.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003"
)
DEFAULT_OUTPUT_EN = (
    ROOT
    / "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003_en"
)
SPEC_PATH = ROOT / "configs/allen_m8_source.json"
POSE_ID = "sobol_003"
METHOD = "entropy_grid"
EXPECTED_BELIEF_HYPOTHESES = 1201
EXPECTED_SUPPORT_ACTIONS = 673
EXPECTED_PADDING_ACTIONS = 76
EXPECTED_SELECTED_ACTIONS = 5
FROZEN_DISPLAY_ONLY_FIELDS = {"video_path", "frame_count"}

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
MAP_SIZE = 900
PANEL_X = 930
FPS = 20
VIDEO_DURATION_S = 10.0

VISIBLE_TEXT = {
    "zh": {
        "panel_title": "M8 熵网格：真实轨迹与离线解释",
        "set_heading": "三个集合不能混为一谈",
        "capture_heading": "CAD capture sets（offline overlay）",
        "belief_heading": "在线 belief 变化（完成观测后）",
        "trajectory_heading": "轨迹图例",
        "solid_trajectory": "接触/受力段：实线",
        "dashed_trajectory": "卸载、抬升、平移、复触：虚线",
        "offline_warning": (
            "真中心、捕获区域、节点着色均为事后解释；\n"
            "在线控制不读取真中心或接触标签。"
        ),
        "map_title": "MuJoCo M8 真实几何俯视 + measured tip XY",
    },
    "en": {
        "panel_title": "M8 entropy grid: measured path\nand offline explanation",
        "set_heading": "Keep the three finite sets distinct",
        "capture_heading": "CAD capture sets (offline overlay)",
        "belief_heading": "Online belief progression (after each observation)",
        "trajectory_heading": "Trajectory legend",
        "solid_trajectory": "Contact / force-bearing motion: solid",
        "dashed_trajectory": "Unload, lift, shift, and retouch: dashed",
        "offline_warning": (
            "True center, capture regions, and node colors are post hoc explanations;\n"
            "the online controller reads neither true center nor contact labels."
        ),
        "map_title": "MuJoCo M8 actual-geometry top view + measured tip XY",
    },
}


@dataclass(frozen=True)
class TraceData:
    xy_mm: np.ndarray
    active_time_s: np.ndarray
    clock_s: np.ndarray
    stage_group: tuple[str, ...]
    stage: tuple[str, ...]
    contact_count: np.ndarray


@dataclass(frozen=True)
class LatticeData:
    belief_hypotheses_mm: np.ndarray
    action_nodes_mm: np.ndarray
    inside_visual_support: np.ndarray
    direct_clearance_eligible: np.ndarray
    chamfer_eligible: np.ndarray
    certificate: dict
    configured_padding_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay strict sobol_003 entropy-grid and render an audited M8 top view."
    )
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument(
        "--language",
        choices=tuple(VISIBLE_TEXT),
        default="zh",
        help="Language used by every visible annotation in the rendered assets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to the language-specific audited directory.",
    )
    parser.add_argument(
        "--reuse-replay-from",
        type=Path,
        default=None,
        help=(
            "Render-only mode: copy trace.csv and result.json from this audited output "
            "directory instead of running MuJoCo again."
        ),
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--duration-s", type=float, default=VIDEO_DURATION_S)
    return parser.parse_args()


def visible_text(language: str) -> dict[str, str]:
    """Return the complete language-dependent on-frame text catalogue."""

    try:
        return VISIBLE_TEXT[language]
    except KeyError as exc:
        raise ValueError(f"unsupported render language: {language!r}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise ValueError(f"artifact must remain inside project root: {path}") from exc


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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        if bold
        else Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_frozen_trial(path: Path) -> tuple[PoseTask, dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_payload = payload.get("task")
    methods = payload.get("methods")
    if not isinstance(task_payload, dict) or not isinstance(methods, dict):
        raise RuntimeError(f"invalid strict trial schema: {path}")
    if task_payload.get("pose_id") != POSE_ID:
        raise RuntimeError(
            f"this renderer is frozen to {POSE_ID}, got {task_payload.get('pose_id')!r}"
        )
    if METHOD not in methods:
        raise RuntimeError(f"strict trial has no {METHOD!r} result")
    return PoseTask(**task_payload), methods[METHOD], payload


def replay_strict_entropy(task: PoseTask, trace_path: Path) -> dict:
    """Replay through the strict audit's single policy-contract entry point."""

    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    result = _run_with_runtime(
        task,
        METHOD,
        runtime,
        output_path=None,
        trace_output_path=trace_path,
    )
    return asdict(result)


def verify_frozen_result(frozen: dict, replay: dict) -> dict:
    """Require exact equality for every substantive field in the old result."""

    compared = sorted(set(frozen).difference(FROZEN_DISPLAY_ONLY_FIELDS))
    missing = [key for key in compared if key not in replay]
    mismatches = [
        {
            "field": key,
            "frozen": frozen[key],
            "replay": replay.get(key),
        }
        for key in compared
        if key in replay and replay[key] != frozen[key]
    ]
    report = {
        "frozen_field_count": len(frozen),
        "replay_field_count": len(replay),
        "compared_field_count": len(compared),
        "excluded_display_only_fields": sorted(FROZEN_DISPLAY_ONLY_FIELDS),
        "missing_fields": missing,
        "mismatches": mismatches,
        "new_replay_only_fields": sorted(set(replay).difference(frozen)),
        "exact_substantive_match": not missing and not mismatches,
    }
    if not report["exact_substantive_match"]:
        raise RuntimeError(
            "strict entropy replay drifted from frozen sobol_003: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def reuse_replay_artifacts(
    *,
    source: Path,
    task: PoseTask,
    frozen_result: dict,
    trace_path: Path,
    result_path: Path,
) -> tuple[dict, dict, dict]:
    """Copy an audited trace/result pair without invoking the physics runtime."""

    source = source.resolve()
    _relative(source)
    if source == trace_path.parent.resolve():
        raise ValueError("reuse-replay-from must differ from the output directory")
    source_trace = source / "trace.csv"
    source_result = source / "result.json"
    for path in (source_trace, source_result):
        if not path.is_file():
            raise FileNotFoundError(f"missing audited replay artifact: {path}")

    payload = json.loads(source_result.read_text(encoding="utf-8"))
    replay = payload.get("result")
    if payload.get("method") != METHOD or not isinstance(replay, dict):
        raise RuntimeError(f"invalid reused entropy result wrapper: {source_result}")
    expected_task = asdict(task)
    if payload.get("task") != expected_task:
        raise RuntimeError(
            "reused result task does not match the frozen trial: "
            + json.dumps(
                {"expected": expected_task, "reused": payload.get("task")},
                sort_keys=True,
            )
        )
    expected_success = _strict_bottom_success(replay)
    if payload.get("strict_bottom_success") != expected_success:
        raise RuntimeError("reused result wrapper has an inconsistent success label")

    verification = verify_frozen_result(frozen_result, replay)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_trace, trace_path)
    shutil.copy2(source_result, result_path)
    return (
        replay,
        verification,
        {
            "physics_rollout_rerun_for_this_render": False,
            "render_only_reuse": True,
            "reused_replay_from": _relative(source),
            "source_trace": _artifact(source_trace),
            "source_result": _artifact(source_result),
        },
    )


def load_trace(path: Path) -> TraceData:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty measured trace: {path}")
    required = {
        "tip_x_m",
        "tip_y_m",
        "active_time_s",
        "clock_s",
        "stage_group",
        "stage",
        "contact_count",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"measured trace missing columns: {sorted(missing)}")
    xy_mm = 1000.0 * np.asarray(
        [[float(row["tip_x_m"]), float(row["tip_y_m"])] for row in rows],
        dtype=float,
    )
    if xy_mm.shape != (len(rows), 2) or not np.all(np.isfinite(xy_mm)):
        raise RuntimeError("measured XY trace is malformed")
    return TraceData(
        xy_mm=xy_mm,
        active_time_s=np.asarray([float(row["active_time_s"]) for row in rows]),
        clock_s=np.asarray([float(row["clock_s"]) for row in rows]),
        stage_group=tuple(row["stage_group"] for row in rows),
        stage=tuple(row["stage"] for row in rows),
        contact_count=np.asarray([int(row["contact_count"]) for row in rows]),
    )


def build_lattice(task: PoseTask, result: dict) -> LatticeData:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    config = EntropyGridConfig(**result["entropy_grid_config"])
    belief = EntropyGridBelief(
        np.asarray([task.x_m, task.y_m], dtype=float),
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        support_inner_radius_m=0.5 * float(spec["derived"]["socket_top_af_m"]),
        support_outer_radius_m=0.5 * float(spec["screw_head_diameter_m"]),
        probe_radius_m=float(spec["key_af_m"]) / math.sqrt(3.0),
        config=config,
        effective_capture_radius_m=float(spec["derived"]["r_cap_top_m"]),
    )
    candidate_mm = 1000.0 * belief.candidate_xy_m
    actions_mm = 1000.0 * belief.action_xy_m
    anchor_mm = 1000.0 * belief.anchor_xy_m
    inside = (
        np.linalg.norm(actions_mm - anchor_mm[None, :], axis=1)
        <= 1000.0 * VISUAL_BOUND_M + 1.0e-9
    )
    radius_mm = np.linalg.norm(actions_mm, axis=1)
    r_wall_mm = 1000.0 * float(spec["derived"]["r_cap_wall_m"])
    r_top_mm = 1000.0 * float(spec["derived"]["r_cap_top_m"])
    direct = radius_mm <= r_wall_mm + 1.0e-12
    chamfer = radius_mm <= r_top_mm + 1.0e-12
    certificate = asdict(
        belief.certify_action_coverage(
            dense_subdivisions_per_cell=8,
            boundary_sample_count=8192,
        )
    )
    counts = (
        len(candidate_mm),
        int(np.count_nonzero(inside)),
        int(np.count_nonzero(~inside)),
    )
    expected = (
        EXPECTED_BELIEF_HYPOTHESES,
        EXPECTED_SUPPORT_ACTIONS,
        EXPECTED_PADDING_ACTIONS,
    )
    if counts != expected:
        raise RuntimeError(f"strict entropy lattice drifted: expected {expected}, got {counts}")
    if not certificate["certified"]:
        raise RuntimeError("offline finite-disk action coverage certificate failed")
    return LatticeData(
        belief_hypotheses_mm=candidate_mm,
        action_nodes_mm=actions_mm,
        inside_visual_support=inside,
        direct_clearance_eligible=direct,
        chamfer_eligible=chamfer,
        certificate=certificate,
        configured_padding_mm=1000.0 * belief.action_padding_m,
    )


def _capture_trace_index(trace: TraceData, result: dict) -> tuple[int, float]:
    event = np.asarray(result["event_xy_mm"], dtype=float)
    search_indices = np.flatnonzero(
        trace.active_time_s <= float(result["time_to_inner_wall_s"]) + 1.0e-9
    )
    if len(search_indices) == 0:
        raise RuntimeError("capture time lies before measured trace")
    distance = np.linalg.norm(trace.xy_mm[search_indices] - event[None, :], axis=1)
    local = int(np.argmin(distance))
    index = int(search_indices[local])
    return index, float(distance[local])


def _is_lift_or_shift(stage_group: str, stage: str) -> bool:
    if stage_group != "fallback_reset":
        return False
    return any(
        marker in stage
        for marker in (
            "unload contact",
            "vertical retract",
            "shift above selected target",
            "guarded retouchdown",
        )
    )


def _segments(trace: TraceData, end_index: int) -> list[tuple[bool, np.ndarray]]:
    """Return contiguous measured XY segments; True means lift/shift."""

    end_index = max(1, min(int(end_index), len(trace.xy_mm) - 1))
    flags = np.asarray(
        [
            _is_lift_or_shift(trace.stage_group[i], trace.stage[i])
            for i in range(end_index + 1)
        ],
        dtype=bool,
    )
    change = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    parts = np.split(np.arange(end_index + 1), change)
    output: list[tuple[bool, np.ndarray]] = []
    for part in parts:
        if len(part) < 2:
            continue
        # Include the previous endpoint so visual segments meet.
        start = max(0, int(part[0]) - 1)
        indices = np.arange(start, int(part[-1]) + 1)
        output.append((bool(flags[part[0]]), trace.xy_mm[indices]))
    return output


def _observation_trace_indices(trace: TraceData, observations: list[dict]) -> list[int]:
    indices: list[int] = []
    cursor = 0
    for observation in observations:
        probe = int(observation["probe"])
        entropy = observation.get("posterior_entropy_nats")
        if entropy is None:
            matches = [
                i
                for i in range(cursor, len(trace.stage))
                if "landed in socket" in trace.stage[i]
            ]
        else:
            target = f"belief updated | H={float(entropy):.2f} nat"
            matches = [
                i
                for i in range(cursor, len(trace.stage))
                if target in trace.stage[i]
            ]
        if not matches:
            raise RuntimeError(f"cannot align entropy observation {probe} to measured trace")
        index = int(matches[0])
        indices.append(index)
        cursor = index + 1
    return indices


def _map_point(
    xy_mm: Iterable[float],
    camera: dict,
    *,
    z_m: float = 0.00045,
) -> tuple[int, int]:
    xy = np.asarray(tuple(xy_mm), dtype=float)
    projected = _project_world(
        np.array([0.001 * xy[0], 0.001 * xy[1], z_m]),
        camera,
    )
    if projected is None:
        raise RuntimeError(f"point projects behind top-view camera: {xy.tolist()}")
    scale = MAP_SIZE / float(camera["width"])
    return int(round(projected[0] * scale)), int(round(projected[1] * scale))


def _mapped_circle(radius_mm: float, camera: dict, *, samples: int = 241) -> list[tuple[int, int]]:
    angle = np.linspace(0.0, 2.0 * math.pi, samples)
    return [
        _map_point((radius_mm * math.cos(a), radius_mm * math.sin(a)), camera)
        for a in angle
    ]


def _draw_dashed(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 9.0,
    gap: float = 7.0,
) -> None:
    if len(points) < 2:
        return
    for a, b in zip(points[:-1], points[1:]):
        start = np.asarray(a, dtype=float)
        delta = np.asarray(b, dtype=float) - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        direction = delta / length
        position = 0.0
        while position < length:
            finish = min(position + dash, length)
            p0 = tuple(np.round(start + position * direction).astype(int))
            p1 = tuple(np.round(start + finish * direction).astype(int))
            draw.line((p0, p1), fill=fill, width=width)
            position += dash + gap


def _diamond(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int, fill, outline) -> None:
    x, y = xy
    draw.polygon(
        ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
        fill=fill,
        outline=outline,
    )


def _star(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int, fill, outline) -> None:
    points = []
    x, y = xy
    for index in range(10):
        r = radius if index % 2 == 0 else 0.42 * radius
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def _build_static_map(
    topview: Image.Image,
    camera: dict,
    *,
    task: PoseTask,
    lattice: LatticeData,
    observations: list[dict],
    r_wall_mm: float,
    r_top_mm: float,
) -> Image.Image:
    base = topview.convert("RGBA").resize((MAP_SIZE, MAP_SIZE), Image.Resampling.LANCZOS)
    veil = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(veil, "RGBA")

    # CAD-derived capture sets.  Both use the offline true center and are never
    # supplied to the online entropy policy.
    chamfer_polygon = _mapped_circle(r_top_mm, camera)
    direct_polygon = _mapped_circle(r_wall_mm, camera)
    draw.polygon(chamfer_polygon, fill=(34, 197, 94, 74))
    draw.line(chamfer_polygon, fill=(74, 222, 128, 245), width=4, joint="curve")
    draw.polygon(direct_polygon, fill=(37, 99, 235, 115))
    draw.line(direct_polygon, fill=(96, 165, 250, 255), width=4, joint="curve")

    support_circle = [
        _map_point(
            (
                1000.0 * task.x_m + 1000.0 * VISUAL_BOUND_M * math.cos(a),
                1000.0 * task.y_m + 1000.0 * VISUAL_BOUND_M * math.sin(a),
            ),
            camera,
        )
        for a in np.linspace(0.0, 2.0 * math.pi, 361)
    ]
    draw.line(support_circle, fill=(226, 232, 240, 210), width=2, joint="curve")

    for point, inside, direct, chamfer in zip(
        lattice.action_nodes_mm,
        lattice.inside_visual_support,
        lattice.direct_clearance_eligible,
        lattice.chamfer_eligible,
    ):
        xy = _map_point(point, camera)
        if direct:
            fill = (96, 165, 250, 255)
            radius = 7
        elif chamfer:
            fill = (74, 222, 128, 255)
            radius = 6
        elif inside:
            fill = (203, 213, 225, 150)
            radius = 2
        else:
            fill = (34, 211, 238, 215)
            radius = 3
        draw.ellipse(
            (xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius),
            fill=fill,
            outline=(15, 23, 42, 220) if radius >= 6 else None,
            width=1,
        )

    initial = _map_point((1000.0 * task.x_m, 1000.0 * task.y_m), camera)
    _star(draw, initial, 12, (250, 204, 21, 255), (113, 63, 18, 255))
    center = _map_point((0.0, 0.0), camera)
    draw.line((center[0] - 11, center[1], center[0] + 11, center[1]), fill=(255, 255, 255, 255), width=3)
    draw.line((center[0], center[1] - 11, center[0], center[1] + 11), fill=(255, 255, 255, 255), width=3)

    for observation in observations:
        point = _map_point(observation["action_xy_mm"], camera)
        _diamond(draw, point, 7, (251, 146, 60, 245), (124, 45, 18, 255))
        draw.text(
            (point[0] + 8, point[1] - 12),
            str(observation["probe"]),
            font=_font(18, bold=True),
            fill=(255, 237, 213, 255),
            stroke_width=2,
            stroke_fill=(67, 20, 7, 255),
        )

    base.alpha_composite(veil)
    return base


def _draw_measured_path(
    image: Image.Image,
    camera: dict,
    trace: TraceData,
    end_index: int,
    capture_index: int,
    event_xy_mm: np.ndarray,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for is_lift, points in _segments(trace, end_index):
        # Decimation affects rendering only; the saved CSV remains complete.
        stride = max(1, int(math.ceil(len(points) / 1800)))
        mapped = [_map_point(point, camera) for point in points[::stride]]
        if len(mapped) < 2:
            continue
        if is_lift:
            _draw_dashed(
                draw,
                mapped,
                fill=(15, 23, 42, 245),
                width=9,
            )
            _draw_dashed(
                draw,
                mapped,
                fill=(251, 113, 133, 255),
                width=6,
            )
        else:
            draw.line(mapped, fill=(15, 23, 42, 245), width=10, joint="curve")
            draw.line(mapped, fill=(96, 165, 250, 255), width=6, joint="curve")
    if end_index >= capture_index:
        event = _map_point(event_xy_mm, camera)
        x, y = event
        draw.line((x - 10, y - 10, x + 10, y + 10), fill=(255, 68, 68, 255), width=5)
        draw.line((x - 10, y + 10, x + 10, y - 10), fill=(255, 68, 68, 255), width=5)
    image.alpha_composite(overlay)


def _draw_capture_inset(
    image: Image.Image,
    *,
    trace: TraceData,
    end_index: int,
    capture_index: int,
    event_xy_mm: np.ndarray,
    r_wall_mm: float,
    r_top_mm: float,
) -> None:
    """Add a clearly labelled offline close-up of the two CAD capture sets."""

    left, top, size = 615, 610, 255
    half_width_mm = 0.55
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(
        (left, top, left + size, top + size),
        radius=10,
        fill=(8, 13, 22, 235),
        outline=(148, 163, 184, 245),
        width=2,
    )

    def local(point: Iterable[float]) -> tuple[int, int]:
        x_mm, y_mm = tuple(point)
        scale = size / (2.0 * half_width_mm)
        return (
            int(round(left + 0.5 * size + x_mm * scale)),
            int(round(top + 0.5 * size - y_mm * scale)),
        )

    center = local((0.0, 0.0))
    scale = size / (2.0 * half_width_mm)
    for radius_mm, fill, outline in (
        (r_top_mm, (34, 197, 94, 70), (74, 222, 128, 255)),
        (r_wall_mm, (37, 99, 235, 115), (96, 165, 250, 255)),
    ):
        radius_px = int(round(radius_mm * scale))
        draw.ellipse(
            (
                center[0] - radius_px,
                center[1] - radius_px,
                center[0] + radius_px,
                center[1] + radius_px,
            ),
            fill=fill,
            outline=outline,
            width=3,
        )

    # Only show the measured portion that lies in this local window.
    search_end = min(end_index, capture_index)
    points = trace.xy_mm[: search_end + 1]
    inside = np.max(np.abs(points), axis=1) <= half_width_mm
    indices = np.flatnonzero(inside)
    if len(indices) >= 2:
        split = np.flatnonzero(np.diff(indices) > 1) + 1
        for part in np.split(indices, split):
            if len(part) < 2:
                continue
            mapped = [local(point) for point in points[part][:: max(1, len(part) // 500)]]
            draw.line(mapped, fill=(250, 204, 21, 255), width=5, joint="curve")
    if end_index >= capture_index:
        event = local(event_xy_mm)
        draw.line((event[0] - 8, event[1] - 8, event[0] + 8, event[1] + 8), fill=(255, 68, 68, 255), width=4)
        draw.line((event[0] - 8, event[1] + 8, event[0] + 8, event[1] - 8), fill=(255, 68, 68, 255), width=4)
    draw.text(
        (left + 10, top + 8),
        "offline capture-zone zoom ±0.55 mm",
        font=_font(14, bold=True),
        fill=(248, 250, 252, 255),
    )
    image.alpha_composite(overlay)


def _panel_text(
    canvas: Image.Image,
    *,
    language: str,
    trace: TraceData,
    end_index: int,
    capture_index: int,
    result: dict,
    lattice: LatticeData,
    observation_indices: list[int],
    observations: list[dict],
    r_wall_mm: float,
    r_top_mm: float,
) -> None:
    labels = visible_text(language)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle(
        (PANEL_X, 22, CANVAS_WIDTH - 22, CANVAS_HEIGHT - 22),
        radius=14,
        fill=(15, 23, 42, 248),
        outline=(71, 85, 105, 255),
        width=2,
    )
    x = PANEL_X + 28
    y = 42
    title_lines = labels["panel_title"].splitlines()
    for line_index, line in enumerate(title_lines):
        draw.text(
            (x, y + 35 * line_index),
            line,
            font=_font(29, bold=True),
            fill=(248, 250, 252, 255),
        )
    y += 76 if len(title_lines) > 1 else 45
    draw.text((x, y), "observe → Bayesian update → choose one action", font=_font(18), fill=(125, 211, 252, 255))
    y += 42
    draw.text((x, y), labels["set_heading"], font=_font(22, bold=True), fill=(253, 186, 116, 255))
    y += 34
    rows = (
        (f"belief hypotheses: {len(lattice.belief_hypotheses_mm)}", (203, 213, 225, 255)),
        (f"action nodes: {EXPECTED_SUPPORT_ACTIONS} inside + {EXPECTED_PADDING_ACTIONS} padding", (103, 232, 249, 255)),
        (f"selected actions in this run: {len(observations)} (not 749 traversed)", (251, 146, 60, 255)),
    )
    for text, color in rows:
        draw.text((x + 8, y), "• " + text, font=_font(18), fill=color)
        y += 29

    y += 12
    draw.text((x, y), labels["capture_heading"], font=_font(22, bold=True), fill=(134, 239, 172, 255))
    y += 34
    draw.text((x + 8, y), f"blue: direct clearance  r ≤ {r_wall_mm:.2f} mm", font=_font(18), fill=(147, 197, 253, 255))
    y += 28
    draw.text((x + 8, y), f"green: chamfer eligible r ≤ {r_top_mm:.2f} mm", font=_font(18), fill=(134, 239, 172, 255))
    y += 28
    draw.text((x + 8, y), "grey/cyan: support/padding action candidates", font=_font(17), fill=(203, 213, 225, 255))

    y += 40
    draw.text((x, y), labels["belief_heading"], font=_font(22, bold=True), fill=(196, 181, 253, 255))
    y += 36
    for index, observation in enumerate(observations):
        complete = end_index >= observation_indices[index]
        highlight = complete and (
            index == len(observations) - 1
            or end_index < observation_indices[min(index + 1, len(observations) - 1)]
        )
        if observation.get("posterior_entropy_nats") is None:
            text = f"{index + 1}. CAPTURE (sensor handoff)"
        else:
            text = (
                f"{index + 1}. H={observation['posterior_entropy_nats']:.2f} nat, "
                f"r95={observation['posterior_credible_radius_95_mm']:.2f} mm"
            )
        color = (
            (250, 204, 21, 255)
            if highlight
            else (226, 232, 240, 255)
            if complete
            else (100, 116, 139, 255)
        )
        draw.text((x + 8, y), text, font=_font(18, bold=highlight), fill=color)
        y += 30

    y += 16
    current_time = float(trace.active_time_s[min(end_index, len(trace.active_time_s) - 1)])
    if end_index < capture_index:
        status = f"SEARCH  |  active time {current_time:5.2f} s"
        status_color = (125, 211, 252, 255)
    elif current_time < float(result["time_to_bottom_s"]) - 1.0e-3:
        status = (
            f"CAPTURE {float(result['time_to_inner_wall_s']):.2f} s  → shared physical funnel"
        )
        status_color = (250, 204, 21, 255)
    else:
        status = f"CAD BOTTOM VERIFIED  |  {float(result['time_to_bottom_s']):.2f} s"
        status_color = (74, 222, 128, 255)
    draw.rounded_rectangle((x, y, CANVAS_WIDTH - 50, y + 47), radius=8, fill=(30, 41, 59, 255))
    draw.text((x + 12, y + 9), status, font=_font(19, bold=True), fill=status_color)
    y += 66
    draw.text((x, y), labels["trajectory_heading"], font=_font(20, bold=True), fill=(248, 250, 252, 255))
    y += 32
    draw.line((x + 8, y + 10, x + 62, y + 10), fill=(37, 99, 235, 255), width=5)
    draw.text((x + 76, y - 3), labels["solid_trajectory"], font=_font(17), fill=(203, 213, 225, 255))
    y += 30
    _draw_dashed(draw, [(x + 8, y + 10), (x + 62, y + 10)], fill=(248, 113, 113, 255), width=4)
    draw.text((x + 76, y - 3), labels["dashed_trajectory"], font=_font(17), fill=(203, 213, 225, 255))

    draw.text(
        (x, CANVAS_HEIGHT - 88),
        labels["offline_warning"],
        font=_font(16, bold=True),
        fill=(253, 164, 175, 255),
        spacing=5,
    )


def compose_frame(
    static_map: Image.Image,
    camera: dict,
    *,
    language: str,
    trace: TraceData,
    end_index: int,
    capture_index: int,
    result: dict,
    lattice: LatticeData,
    observation_indices: list[int],
    observations: list[dict],
    r_wall_mm: float,
    r_top_mm: float,
) -> Image.Image:
    labels = visible_text(language)
    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (8, 13, 22, 255))
    map_image = static_map.copy()
    _draw_measured_path(
        map_image,
        camera,
        trace,
        end_index,
        capture_index,
        np.asarray(result["event_xy_mm"], dtype=float),
    )
    _draw_capture_inset(
        map_image,
        trace=trace,
        end_index=end_index,
        capture_index=capture_index,
        event_xy_mm=np.asarray(result["event_xy_mm"], dtype=float),
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )
    canvas.alpha_composite(map_image, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, MAP_SIZE, 48), fill=(8, 13, 22, 225))
    draw.text((24, 9), labels["map_title"], font=_font(23, bold=True), fill=(248, 250, 252, 255))
    draw.text((22, 858), "★ initial XY   ◆ selected actions   × measured capture", font=_font(17), fill=(248, 250, 252, 255), stroke_width=2, stroke_fill=(8, 13, 22, 255))
    _panel_text(
        canvas,
        language=language,
        trace=trace,
        end_index=end_index,
        capture_index=capture_index,
        result=result,
        lattice=lattice,
        observation_indices=observation_indices,
        observations=observations,
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )
    return canvas.convert("RGB")


def render_video_and_posters(
    *,
    output: Path,
    language: str,
    task: PoseTask,
    result: dict,
    trace: TraceData,
    lattice: LatticeData,
    fps: int,
    duration_s: float,
) -> tuple[Path, Path, Path, Path, int, int, float]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    r_wall_mm = 1000.0 * float(spec["derived"]["r_cap_wall_m"])
    r_top_mm = 1000.0 * float(spec["derived"]["r_cap_top_m"])
    observations = list(result["entropy_grid_observations"])
    if len(observations) != EXPECTED_SELECTED_ACTIONS:
        raise RuntimeError(
            f"expected {EXPECTED_SELECTED_ACTIONS} selected actions, got {len(observations)}"
        )
    observation_indices = _observation_trace_indices(trace, observations)
    capture_index, capture_distance_mm = _capture_trace_index(trace, result)

    topview, camera = render_mujoco_top_view()
    topview_path = output / "m8_mujoco_topview_base.png"
    topview.save(topview_path)
    static_map = _build_static_map(
        topview,
        camera,
        task=task,
        lattice=lattice,
        observations=observations,
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )

    video_name = (
        "m8_strict_entropy_grid_topview_sobol_003_en.mp4"
        if language == "en"
        else "m8_strict_entropy_grid_topview_sobol_003.mp4"
    )
    video_path = output / video_name
    poster_path = output / "poster.png"
    planar_path = output / "planar_map.png"
    total_frames = int(round(fps * duration_s))
    if total_frames < 2:
        raise ValueError("video must contain at least two frames")
    final_time = float(result["time_to_bottom_s"])
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last_frame: Image.Image | None = None
    try:
        for frame_index in range(total_frames):
            fraction = frame_index / float(total_frames - 1)
            active_time = fraction * final_time
            end_index = int(np.searchsorted(trace.active_time_s, active_time, side="right") - 1)
            end_index = max(1, min(end_index, len(trace.xy_mm) - 1))
            frame = compose_frame(
                static_map,
                camera,
                language=language,
                trace=trace,
                end_index=end_index,
                capture_index=capture_index,
                result=result,
                lattice=lattice,
                observation_indices=observation_indices,
                observations=observations,
                r_wall_mm=r_wall_mm,
                r_top_mm=r_top_mm,
            )
            writer.append_data(np.asarray(frame))
            last_frame = frame
    finally:
        writer.close()
    if last_frame is None:
        raise RuntimeError("top-view video produced no frames")
    last_frame.save(poster_path)
    final_map = compose_frame(
        static_map,
        camera,
        language=language,
        trace=trace,
        end_index=len(trace.xy_mm) - 1,
        capture_index=capture_index,
        result=result,
        lattice=lattice,
        observation_indices=observation_indices,
        observations=observations,
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )
    final_map.save(planar_path)
    return (
        video_path,
        poster_path,
        planar_path,
        topview_path,
        total_frames,
        capture_index,
        capture_distance_mm,
    )


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.duration_s <= 0.0:
        raise ValueError("fps and duration-s must be positive")
    trial_path = args.trial.resolve()
    default_output = DEFAULT_OUTPUT_EN if args.language == "en" else DEFAULT_OUTPUT
    output = (args.output or default_output).resolve()
    _relative(output)
    output.mkdir(parents=True, exist_ok=True)

    task, frozen_result, _ = _load_frozen_trial(trial_path)
    trace_path = output / "trace.csv"
    result_path = output / "result.json"
    verification_path = output / "frozen_result_verification.json"
    if args.reuse_replay_from is not None:
        replay, verification, replay_provenance = reuse_replay_artifacts(
            source=args.reuse_replay_from,
            task=task,
            frozen_result=frozen_result,
            trace_path=trace_path,
            result_path=result_path,
        )
    else:
        replay = replay_strict_entropy(task, trace_path)
        verification = verify_frozen_result(frozen_result, replay)
        replay_provenance = {
            "physics_rollout_rerun_for_this_render": True,
            "render_only_reuse": False,
            "reused_replay_from": None,
        }
        _write_json(
            result_path,
            {
                "task": asdict(task),
                "method": METHOD,
                "strict_bottom_success": _strict_bottom_success(replay),
                "result": replay,
            },
        )
    _write_json(verification_path, verification)

    trace = load_trace(trace_path)
    lattice = build_lattice(task, replay)
    (
        video_path,
        poster_path,
        planar_path,
        topview_path,
        frame_count,
        capture_index,
        capture_trace_distance_mm,
    ) = render_video_and_posters(
        output=output,
        language=args.language,
        task=task,
        result=replay,
        trace=trace,
        lattice=lattice,
        fps=args.fps,
        duration_s=args.duration_s,
    )

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    r_wall_mm = 1000.0 * float(spec["derived"]["r_cap_wall_m"])
    r_top_mm = 1000.0 * float(spec["derived"]["r_cap_top_m"])
    selected = replay["entropy_grid_observations"]
    event_radius_mm = float(np.linalg.norm(np.asarray(replay["event_xy_mm"])))
    manifest = {
        "experiment": "M8_strict_entropy_grid_real_topview_selected_mechanism_explainer",
        "created_utc": _utc_now(),
        "language": args.language,
        "visible_text_language": args.language,
        "pose_id": POSE_ID,
        "method": METHOD,
        "source_frozen_trial": _relative(trial_path),
        "source_frozen_trial_sha256": _sha256(trial_path),
        "task": asdict(task),
        "replay_contract": {
            "entry_point": "run_m8_strict_complete_loop_audit._run_with_runtime",
            "same_frozen_task_and_runtime_arguments": True,
            "trace_is_measured_tip_xy_from_replayed_MuJoCo_rollout": True,
            "hard_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "frozen_result_verification": verification,
            **replay_provenance,
        },
        "finite_sets": {
            "belief_hypotheses": int(len(lattice.belief_hypotheses_mm)),
            "belief_hypotheses_are_not_action_grid": True,
            "actions_inside_visual_support": int(
                np.count_nonzero(lattice.inside_visual_support)
            ),
            "boundary_padding_actions": int(
                np.count_nonzero(~lattice.inside_visual_support)
            ),
            "total_available_action_candidates": int(len(lattice.action_nodes_mm)),
            "selected_action_count": len(selected),
            "controller_traversed_all_action_candidates": False,
            "selected_actions_mm": [item["action_xy_mm"] for item in selected],
            "configured_radial_padding_mm": lattice.configured_padding_mm,
            "coverage_certificate_offline_audit": lattice.certificate,
        },
        "capture_sets": {
            "direct_clearance_radius_mm": r_wall_mm,
            "chamfer_eligible_radius_mm": r_top_mm,
            "nominal_direct_clearance_action_nodes_mm": lattice.action_nodes_mm[
                lattice.direct_clearance_eligible
            ].tolist(),
            "nominal_chamfer_only_action_nodes_mm": lattice.action_nodes_mm[
                lattice.chamfer_eligible & ~lattice.direct_clearance_eligible
            ].tolist(),
            "measured_capture_event_xy_mm": replay["event_xy_mm"],
            "measured_capture_event_radius_mm": event_radius_mm,
            "measured_capture_inside_direct_clearance": event_radius_mm <= r_wall_mm,
            "measured_capture_inside_chamfer_eligible": event_radius_mm <= r_top_mm,
            "nominal_selected_action_5_radius_mm": float(
                np.linalg.norm(np.asarray(selected[-1]["action_xy_mm"], dtype=float))
            ),
            "interpretation": (
                "The fifth commanded endpoint is outside the nominal CAD disks, while the "
                "measured compliant tip path reaches the direct-clearance disk before sensor "
                "handoff. Node eligibility is therefore not substituted for measured capture."
            ),
        },
        "trajectory": {
            "samples": int(len(trace.xy_mm)),
            "capture_trace_row": capture_index,
            "capture_event_nearest_trace_distance_mm": capture_trace_distance_mm,
            "solid_segments": "measured contact/force-bearing motion",
            "dashed_segments": "measured unload, lift, shift, and retouch motion",
            "capture_time_s": replay["time_to_inner_wall_s"],
            "bottom_time_s": replay["time_to_bottom_s"],
            "result": replay["result"],
            "strict_bottom_success": _strict_bottom_success(replay),
        },
        "belief_progression": [
            {
                "probe": item["probe"],
                "observation": item["observation"],
                "posterior_entropy_nats": item.get("posterior_entropy_nats"),
                "posterior_credible_radius_95_mm": item.get(
                    "posterior_credible_radius_95_mm"
                ),
                "action_xy_mm": item["action_xy_mm"],
            }
            for item in selected
        ],
        "offline_explanation_boundary": {
            "mujoco_geometry_is_actual_allen_m8_source_model": True,
            "true_center_used_online": False,
            "contact_tags_used_online": False,
            "capture_disks_and_node_eligibility_used_online": False,
            "true_center_capture_disks_and_eligibility_are_offline_overlays": True,
            "selected_single_rollout_is_not_distribution_level_evidence": True,
            "coverage_certificate_does_not_guarantee_tracking_capture_or_finite_time_success": True,
        },
        "claim_boundary": (
            "This selected strict MuJoCo replay explains the discrete Bayesian entropy-grid "
            "mechanism and its measured physical path. It demonstrates five adaptive actions, "
            "sensor capture, and shared-funnel CAD-bottom completion for sobol_003 only. It does "
            "not claim that 749 nodes were traversed, that 1201 belief hypotheses are actions, "
            "or that the offline geometric certificate guarantees hardware or finite-time success."
        ),
        "artifacts": {
            "video": _artifact(video_path),
            "poster": _artifact(poster_path),
            "planar_map": _artifact(planar_path),
            "mujoco_topview_base": _artifact(topview_path),
            "trace": _artifact(trace_path),
            "result": _artifact(result_path),
            "frozen_result_verification": _artifact(verification_path),
        },
        "video": {
            "fps": args.fps,
            "duration_s": args.duration_s,
            "frame_count": frame_count,
            "display_time_is_compressed_from_active_simulation_time": True,
            "display_compression_changes_simulation_or_metrics": False,
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": _relative(output),
                "language": args.language,
                "physics_rollout_rerun_for_this_render": replay_provenance[
                    "physics_rollout_rerun_for_this_render"
                ],
                "strict_match": verification["exact_substantive_match"],
                "action_nodes": [
                    int(np.count_nonzero(lattice.inside_visual_support)),
                    int(np.count_nonzero(~lattice.inside_visual_support)),
                ],
                "selected_actions": len(selected),
                "capture_s": replay["time_to_inner_wall_s"],
                "bottom_s": replay["time_to_bottom_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
