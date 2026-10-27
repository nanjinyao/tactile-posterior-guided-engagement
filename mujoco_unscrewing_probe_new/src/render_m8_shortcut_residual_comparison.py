#!/usr/bin/env python3
"""Replay and compare the measured residual-search paths for M8 ``sobol_008``.

The figure is intentionally a selected-mechanism explanation, not an isolated
causal ablation.  Proposed and Proposed+ replay the same frozen first probe,
task, nuisance, sensor stream, 130 s clock, shared post-capture R3 funnel, and
CAD-bottom verifier used by the five-method audit.  Pure outward is replayed
only as a faint same-start geometric reference.

The true socket center and the 0.10/0.25 mm CAD regions are offline overlays.
In particular, Proposed+ does *not* promise monotonic entropy contraction: its
residual actions are ordered by task value (capture probability + information
gain - motion/contact cost), and the measured capture cue terminates the fifth
segment before its nominal endpoint.
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
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_sampling_position_map import _project_world, render_mujoco_top_view
from src.render_m8_search_comparison import run_method
from src.run_m8_fair_policy_benchmark import PoseTask
from src.run_m8_five_method_retry3_audit import (
    ACTIVE_TIME_BUDGET_S,
    IMPROVED_MODEL_ROOT,
    SHARED_FUNNEL,
    _common_kwargs,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIAL = (
    ROOT
    / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
    / "trials/sobol_008.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/active_comparison/m8_shortcut_residual_sobol_008"
)
SPEC_PATH = ROOT / "configs/allen_m8_source.json"
POSE_ID = "sobol_008"
METHODS = ("proposed", "proposed_plus", "outward")
RUNTIME_METHOD = {
    "proposed": "learned",
    "proposed_plus": "hybrid",
    "outward": "baseline",
}
FROZEN_DISPLAY_ONLY_FIELDS = {"video_path", "frame_count"}

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
PANEL_SIZE = 850
PANEL_Y = 132
PANEL_X = {"proposed": 65, "proposed_plus": 1005}
WORLD_Z_M = 0.00045


@dataclass(frozen=True)
class TraceData:
    xy_mm: np.ndarray
    active_time_s: np.ndarray
    clock_s: np.ndarray
    stage_group: tuple[str, ...]
    stage: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the five-method sobol_008 Proposed/Proposed+/outward arms "
            "and render a same-start measured-path mechanism figure."
        )
    )
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise ValueError(f"artifact must stay inside the project root: {path}") from exc


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


def load_frozen_trial(path: Path) -> tuple[PoseTask, dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_payload = payload.get("task")
    methods = payload.get("methods")
    if not isinstance(task_payload, dict) or not isinstance(methods, dict):
        raise RuntimeError(f"invalid five-method trial schema: {path}")
    if task_payload.get("pose_id") != POSE_ID:
        raise RuntimeError(
            f"this renderer is frozen to {POSE_ID}, got {task_payload.get('pose_id')!r}"
        )
    missing = [method for method in METHODS if method not in methods]
    if missing:
        raise RuntimeError(f"frozen trial is missing methods: {missing}")
    return PoseTask(**task_payload), {key: methods[key] for key in METHODS}, payload


def replay_methods(task: PoseTask, output: Path) -> dict[str, dict]:
    """Replay through exactly the five-method audit's common argument contract."""

    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(IMPROVED_MODEL_ROOT)
    replay: dict[str, dict] = {}
    for method in METHODS:
        trace_path = output / f"{method}_trace.csv"
        kwargs = _common_kwargs(task)
        if method == "proposed":
            result = run_method(
                RUNTIME_METHOD[method],
                direction_runtime=runtime,
                trace_output_path=trace_path,
                **kwargs,
            )
        elif method == "proposed_plus":
            result = run_method(
                RUNTIME_METHOD[method],
                direction_runtime=runtime,
                trace_output_path=trace_path,
                **kwargs,
            )
        else:
            result = run_method(
                RUNTIME_METHOD[method],
                trace_output_path=trace_path,
                **kwargs,
            )
        replay[method] = asdict(result)
    return replay


def verify_frozen_results(frozen: dict[str, dict], replay: dict[str, dict]) -> dict:
    reports: dict[str, dict] = {}
    for method in METHODS:
        expected = frozen[method]
        actual = replay[method]
        compared = sorted(set(expected).difference(FROZEN_DISPLAY_ONLY_FIELDS))
        missing = [key for key in compared if key not in actual]
        mismatches = [
            {"field": key, "frozen": expected[key], "replay": actual.get(key)}
            for key in compared
            if key in actual and actual[key] != expected[key]
        ]
        reports[method] = {
            "frozen_field_count": len(expected),
            "replay_field_count": len(actual),
            "compared_field_count": len(compared),
            "excluded_display_only_fields": sorted(FROZEN_DISPLAY_ONLY_FIELDS),
            "missing_fields": missing,
            "mismatches": mismatches,
            "new_replay_only_fields": sorted(set(actual).difference(expected)),
            "exact_substantive_match": not missing and not mismatches,
        }
    report = {
        "methods": reports,
        "all_exact_substantive_match": all(
            item["exact_substantive_match"] for item in reports.values()
        ),
    }
    if not report["all_exact_substantive_match"]:
        raise RuntimeError(
            "same-start replay drifted from frozen five-method sobol_008: "
            + json.dumps(report, sort_keys=True)
        )
    return report


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
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"trace is missing columns: {sorted(missing)}")
    xy_mm = 1000.0 * np.asarray(
        [[float(row["tip_x_m"]), float(row["tip_y_m"])] for row in rows],
        dtype=float,
    )
    if xy_mm.shape != (len(rows), 2) or not np.all(np.isfinite(xy_mm)):
        raise RuntimeError(f"malformed measured XY in {path}")
    return TraceData(
        xy_mm=xy_mm,
        active_time_s=np.asarray([float(row["active_time_s"]) for row in rows]),
        clock_s=np.asarray([float(row["clock_s"]) for row in rows]),
        stage_group=tuple(row["stage_group"] for row in rows),
        stage=tuple(row["stage"] for row in rows),
    )


def _contiguous_segments(points: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    segments = []
    for part in np.split(indices, split):
        if len(part) >= 2:
            # Keep the measured path but bound rendering cost deterministically.
            if len(part) > 5000:
                part = part[np.linspace(0, len(part) - 1, 5000, dtype=int)]
            segments.append(points[part])
    return segments


def path_segments(trace: TraceData, result: dict, method: str) -> dict[str, list[np.ndarray]]:
    capture_time = float(result["time_to_inner_wall_s"])
    before_capture = trace.active_time_s <= capture_time + 1.0e-9
    groups = np.asarray(trace.stage_group, dtype=object)
    stages = np.asarray(trace.stage, dtype=object)
    if method == "proposed":
        shortcut = before_capture & (groups == "transport")
        residual = before_capture & (groups == "spiral")
    elif method == "proposed_plus":
        shortcut_marker = np.asarray(
            ["joint-posterior bounded shortcut" in stage for stage in stages],
            dtype=bool,
        )
        shortcut = before_capture & shortcut_marker
        residual = before_capture & (groups == "hybrid_entropy_probe") & ~shortcut_marker
    elif method == "outward":
        shortcut = np.zeros(len(trace.xy_mm), dtype=bool)
        residual = before_capture & (groups == "spiral")
    else:
        raise ValueError(method)
    return {
        "shortcut": _contiguous_segments(trace.xy_mm, shortcut),
        "residual": _contiguous_segments(trace.xy_mm, residual),
    }


def _polyline_length_mm(segments: list[np.ndarray]) -> float:
    return float(
        sum(
            np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
            for segment in segments
        )
    )


def _project_mm(points_mm: np.ndarray, camera: dict) -> list[tuple[int, int]]:
    projected: list[tuple[int, int]] = []
    for point in points_mm:
        pixel = _project_world(
            np.array([0.001 * point[0], 0.001 * point[1], WORLD_Z_M]),
            camera,
        )
        if pixel is not None:
            projected.append(pixel)
    return projected


def _draw_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[np.ndarray],
    camera: dict,
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    for segment in segments:
        points = _project_mm(segment, camera)
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")


def _draw_projected_circle(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    radius_mm: float,
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    points = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 241):
        pixel = _project_world(
            np.array(
                [
                    0.001 * radius_mm * math.cos(angle),
                    0.001 * radius_mm * math.sin(angle),
                    WORLD_Z_M,
                ]
            ),
            camera,
        )
        if pixel is not None:
            points.append(pixel)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def _draw_star(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    radius: int = 13,
) -> None:
    points = []
    for index in range(10):
        r = radius if index % 2 == 0 else 0.44 * radius
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        points.append((xy[0] + r * math.cos(angle), xy[1] + r * math.sin(angle)))
    draw.polygon(points, fill=(250, 204, 21, 255), outline=(66, 32, 6, 255))


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    radius: int,
    width: int,
) -> None:
    x, y = xy
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def _draw_center_inset(
    panel: Image.Image,
    *,
    trace: TraceData,
    result: dict,
    r_wall_mm: float,
    r_top_mm: float,
) -> None:
    draw = ImageDraw.Draw(panel, "RGBA")
    x0, y0, size = 635, 640, 245
    draw.rounded_rectangle(
        (x0, y0, x0 + size, y0 + size),
        radius=9,
        fill=(248, 250, 252, 238),
        outline=(51, 65, 85, 245),
        width=2,
    )
    draw.text(
        (x0 + 10, y0 + 8),
        "离线中心放大（±0.65 mm）",
        font=_font(16, bold=True),
        fill=(15, 23, 42, 255),
    )
    plot_y0 = y0 + 35
    plot_size = size - 44
    center = np.array([x0 + 0.5 * size, plot_y0 + 0.5 * plot_size])
    pixels_per_mm = plot_size / 1.30

    def inset_xy(point_mm: np.ndarray) -> tuple[int, int]:
        return (
            int(round(center[0] + pixels_per_mm * float(point_mm[0]))),
            int(round(center[1] - pixels_per_mm * float(point_mm[1]))),
        )

    for radius_mm, fill, outline in (
        (r_top_mm, (250, 204, 21, 74), (202, 138, 4, 255)),
        (r_wall_mm, (34, 197, 94, 92), (21, 128, 61, 255)),
    ):
        radius_px = pixels_per_mm * radius_mm
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
    capture_time = float(result["time_to_inner_wall_s"])
    mask = (trace.active_time_s <= capture_time + 1.0e-9) & (
        np.linalg.norm(trace.xy_mm, axis=1) <= 0.65
    )
    for segment in _contiguous_segments(trace.xy_mm, mask):
        points = [inset_xy(point) for point in segment]
        if len(points) >= 2:
            draw.line(points, fill=(37, 99, 235, 225), width=3, joint="curve")
    event = np.asarray(result["event_xy_mm"], dtype=float)
    _draw_cross(
        draw,
        inset_xy(event),
        color=(220, 38, 38, 255),
        radius=7,
        width=3,
    )
    _draw_cross(
        draw,
        (int(center[0]), int(center[1])),
        color=(15, 23, 42, 255),
        radius=4,
        width=2,
    )
    draw.text(
        (x0 + 10, y0 + size - 25),
        "绿: 直接间隙 0.10  黄: 倒角捕获 0.25 mm",
        font=_font(12),
        fill=(30, 41, 59, 255),
    )


def _draw_action_ordering(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    result: dict,
) -> None:
    observations = list(result.get("entropy_grid_observations") or [])
    for item in observations:
        point = np.asarray(item["action_xy_mm"], dtype=float)
        pixel = _project_mm(point[None, :], camera)
        if not pixel:
            continue
        x, y = pixel[0]
        radius = 8
        draw.polygon(
            [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)],
            fill=(244, 114, 182, 245),
            outline=(80, 7, 36, 255),
        )
        draw.text(
            (x + 9, y - 12),
            str(item["probe"]),
            font=_font(17, bold=True),
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(80, 7, 36, 255),
        )


def _render_panel(
    base: Image.Image,
    camera: dict,
    *,
    method: str,
    task: PoseTask,
    result: dict,
    trace: TraceData,
    segments: dict[str, list[np.ndarray]],
    outward_segments: list[np.ndarray],
    r_wall_mm: float,
    r_top_mm: float,
) -> Image.Image:
    panel = base.copy().convert("RGBA")
    draw = ImageDraw.Draw(panel, "RGBA")

    # Same-start pure outward is a reference trajectory only; it is not part of
    # either learned policy's controller state.
    _draw_segments(
        draw,
        outward_segments,
        camera,
        color=(203, 213, 225, 125),
        width=3,
    )
    _draw_segments(
        draw,
        segments["shortcut"],
        camera,
        color=(249, 115, 22, 245),
        width=7,
    )
    _draw_segments(
        draw,
        segments["residual"],
        camera,
        color=(37, 99, 235, 245)
        if method == "proposed"
        else (192, 38, 211, 245),
        width=7,
    )

    _draw_projected_circle(
        draw,
        camera,
        r_top_mm,
        color=(250, 204, 21, 255),
        width=5,
    )
    _draw_projected_circle(
        draw,
        camera,
        r_wall_mm,
        color=(34, 197, 94, 255),
        width=5,
    )
    initial = _project_mm(
        np.asarray([[1000.0 * task.x_m, 1000.0 * task.y_m]]), camera
    )[0]
    center = _project_mm(np.asarray([[0.0, 0.0]]), camera)[0]
    event = _project_mm(np.asarray([result["event_xy_mm"]], dtype=float), camera)[0]
    _draw_star(draw, initial)
    _draw_cross(draw, center, color=(255, 255, 255, 255), radius=9, width=3)
    _draw_cross(draw, event, color=(239, 68, 68, 255), radius=11, width=5)
    if method == "proposed_plus":
        _draw_action_ordering(draw, camera, result)

    draw.rounded_rectangle(
        (18, 16, 615, 145),
        radius=9,
        fill=(7, 12, 20, 220),
        outline=(148, 163, 184, 190),
        width=2,
    )
    title = (
        "Proposed：指纹捷径 → local outward"
        if method == "proposed"
        else "Proposed+：指纹捷径 → task-aware entropy grid"
    )
    draw.text((36, 29), title, font=_font(25, bold=True), fill=(248, 250, 252, 255))
    draw.text(
        (36, 70),
        (
            f"捕获 {float(result['time_to_inner_wall_s']):.4f} s  |  "
            f"CAD-bottom {float(result['time_to_bottom_s']):.4f} s"
        ),
        font=_font(19, bold=True),
        fill=(226, 232, 240, 255),
    )
    draw.text(
        (36, 104),
        f"measured XY {float(result['actual_xy_path_mm']):.2f} mm  |  shared R3 recoveries {int(result['funnel_recovery_count'])}",
        font=_font(17),
        fill=(203, 213, 225, 255),
    )

    if method == "proposed_plus":
        observations = list(result["entropy_grid_observations"])
        entropy = [
            float(item["posterior_entropy_nats"])
            for item in observations
            if item.get("posterior_entropy_nats") is not None
        ]
        draw.rounded_rectangle(
            (18, 746, 615, 882),
            radius=9,
            fill=(7, 12, 20, 224),
            outline=(244, 114, 182, 205),
            width=2,
        )
        draw.text(
            (34, 758),
            "动作 1→5：按 task value 排序，第 5 段由捕获 cue 终止",
            font=_font(17, bold=True),
            fill=(251, 207, 232, 255),
        )
        draw.text(
            (34, 792),
            "H[nat] 5.62 → " + " → ".join(f"{value:.2f}" for value in entropy) + " → capture",
            font=_font(17),
            fill=(241, 245, 249, 255),
        )
        draw.text(
            (34, 826),
            "熵非单调：目标是尽快进入可捕获集，不是逐步压低 H。",
            font=_font(17, bold=True),
            fill=(253, 186, 116, 255),
        )
    else:
        draw.rounded_rectangle(
            (18, 793, 615, 882),
            radius=9,
            fill=(7, 12, 20, 224),
            outline=(96, 165, 250, 205),
            width=2,
        )
        draw.text(
            (34, 807),
            "捷径只缩短起步距离；剩余不确定性由局部 outward 覆盖。",
            font=_font(18, bold=True),
            fill=(219, 234, 254, 255),
        )
        draw.text(
            (34, 844),
            "浅灰：同起点 pure outward 实测参考轨迹",
            font=_font(16),
            fill=(226, 232, 240, 255),
        )

    _draw_center_inset(
        panel,
        trace=trace,
        result=result,
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )
    return panel.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)


def render_figure(
    output_path: Path,
    *,
    task: PoseTask,
    replay: dict[str, dict],
    traces: dict[str, TraceData],
    segments: dict[str, dict[str, list[np.ndarray]]],
    r_wall_mm: float,
    r_top_mm: float,
) -> Path:
    base, camera = render_mujoco_top_view()
    base_path = output_path.parent / "m8_mujoco_topview_base.png"
    base.save(base_path)
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (248, 250, 252))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text(
        (60, 22),
        "同起点 measured path：捷径之后，比的是 residual search",
        font=_font(38, bold=True),
        fill=(23, 55, 94, 255),
    )
    draw.text(
        (62, 78),
        "M8 sobol_008｜同一 raw probe / nuisance / 130 s 时钟；离线中心与捕获区域不进入控制器",
        font=_font(23),
        fill=(71, 85, 105, 255),
    )
    outward_reference = segments["outward"]["residual"]
    for method in ("proposed", "proposed_plus"):
        panel = _render_panel(
            base,
            camera,
            method=method,
            task=task,
            result=replay[method],
            trace=traces[method],
            segments=segments[method],
            outward_segments=outward_reference,
            r_wall_mm=r_wall_mm,
            r_top_mm=r_top_mm,
        )
        canvas.paste(panel, (PANEL_X[method], PANEL_Y))

    footer_y = 1003
    draw.line((65, footer_y - 12, 1855, footer_y - 12), fill=(203, 213, 225, 255), width=2)
    legend = (
        ("橙：measured shortcut", (249, 115, 22, 255)),
        ("蓝/紫：measured residual", (37, 99, 235, 255)),
        ("浅灰：pure outward reference", (148, 163, 184, 255)),
        ("红×：sensor capture", (239, 68, 68, 255)),
    )
    x = 70
    for label, color in legend:
        draw.line((x, footer_y + 14, x + 44, footer_y + 14), fill=color, width=6)
        draw.text((x + 55, footer_y - 2), label, font=_font(17), fill=(30, 41, 59, 255))
        x += 395
    draw.text(
        (70, 1044),
        "捕获以后两者都进入同一 tip-RCC + 最多 3 次 post-capture recovery + CAD-bottom verifier；本页解释机制，不是单因素因果消融。",
        font=_font(18, bold=True),
        fill=(126, 34, 206, 255),
    )
    canvas.save(output_path, quality=95)
    return base_path


def main() -> int:
    args = parse_args()
    trial_path = args.trial.resolve()
    output = args.output.resolve()
    _relative(output)
    output.mkdir(parents=True, exist_ok=True)

    task, frozen, frozen_payload = load_frozen_trial(trial_path)
    replay = replay_methods(task, output)
    verification = verify_frozen_results(frozen, replay)
    verification_path = output / "frozen_result_verification.json"
    _write_json(verification_path, verification)

    result_path = output / "results.json"
    _write_json(
        result_path,
        {
            "task": asdict(task),
            "methods": replay,
            "strict_bottom_success": {
                method: _strict_bottom_success(replay[method]) for method in METHODS
            },
        },
    )
    traces = {
        method: load_trace(output / f"{method}_trace.csv") for method in METHODS
    }
    segments = {
        method: path_segments(traces[method], replay[method], method)
        for method in METHODS
    }

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    r_wall_mm = 1000.0 * float(spec["derived"]["r_cap_wall_m"])
    r_top_mm = 1000.0 * float(spec["derived"]["r_cap_top_m"])
    figure_path = output / "same_start_shortcut_residual_sobol_008.png"
    base_path = render_figure(
        figure_path,
        task=task,
        replay=replay,
        traces=traces,
        segments=segments,
        r_wall_mm=r_wall_mm,
        r_top_mm=r_top_mm,
    )

    proposed_plus_observations = list(replay["proposed_plus"]["entropy_grid_observations"])
    entropy_after = [
        float(item["posterior_entropy_nats"])
        for item in proposed_plus_observations
        if item.get("posterior_entropy_nats") is not None
    ]
    entropy_nonmonotonic = any(
        right > left + 1.0e-12
        for left, right in zip(entropy_after, entropy_after[1:])
    )
    method_summary = {}
    for method in METHODS:
        event = np.asarray(replay[method]["event_xy_mm"], dtype=float)
        method_summary[method] = {
            "runtime_method": RUNTIME_METHOD[method],
            "strict_bottom_success": _strict_bottom_success(replay[method]),
            "capture_time_s": replay[method]["time_to_inner_wall_s"],
            "bottom_time_s": replay[method]["time_to_bottom_s"],
            "actual_xy_path_mm": replay[method]["actual_xy_path_mm"],
            "capture_event_xy_mm": replay[method]["event_xy_mm"],
            "capture_event_radius_mm": float(np.linalg.norm(event)),
            "capture_mode": replay[method]["capture_mode"],
            "shortcut_measured_path_length_mm": _polyline_length_mm(
                segments[method]["shortcut"]
            ),
            "residual_measured_path_length_mm": _polyline_length_mm(
                segments[method]["residual"]
            ),
            "shortcut_segment_count": len(segments[method]["shortcut"]),
            "residual_segment_count": len(segments[method]["residual"]),
            "funnel_recovery_count": replay[method]["funnel_recovery_count"],
            "terminal_verification_mode": replay[method]["terminal_verification_mode"],
        }
    manifest = {
        "experiment": "M8_same_start_shortcut_then_residual_search_selected_mechanism",
        "created_utc": _utc_now(),
        "pose_id": POSE_ID,
        "source_frozen_trial": _relative(trial_path),
        "source_frozen_trial_sha256": _sha256(trial_path),
        "source_frozen_trial_created_utc": frozen_payload.get("created_utc"),
        "task": asdict(task),
        "same_rollout_contract": {
            "entry_contract": "run_m8_five_method_retry3_audit._common_kwargs",
            "same_initial_pose_contact_nuisance_sensor_seed": True,
            "same_first_probe_fingerprint_for_proposed_and_proposed_plus": (
                replay["proposed"]["fingerprint_sha256"]
                == replay["proposed_plus"]["fingerprint_sha256"]
            ),
            "hard_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "zero_budget_extension": all(
                float(replay[method]["fallback_budget_extension_s"]) == 0.0
                for method in METHODS
            ),
            "shared_post_capture_funnel": asdict(SHARED_FUNNEL),
            "same_terminal_verifier": True,
            "frozen_result_verification": verification,
        },
        "m8_geometry_offline_overlay": {
            "source": _relative(SPEC_PATH),
            "source_sha256": _sha256(SPEC_PATH),
            "mujoco_geometry_is_actual_allen_m8_source_model": True,
            "direct_clearance_radius_mm": r_wall_mm,
            "chamfer_capture_radius_mm": r_top_mm,
            "true_center_used_online": False,
            "capture_regions_used_online": False,
        },
        "methods": method_summary,
        "proposed_plus_task_aware_ordering": {
            "selected_action_count": len(proposed_plus_observations),
            "selected_actions_mm": [
                item["action_xy_mm"] for item in proposed_plus_observations
            ],
            "post_shortcut_entropy_before_first_action_nats": float(
                replay["proposed_plus"]["hybrid_probe_gate_records"][1][
                    "entropy_before_nats"
                ]
            ),
            "posterior_entropy_after_non_capture_actions_nats": entropy_after,
            "entropy_is_monotonic_decreasing": not entropy_nonmonotonic,
            "fifth_action_outcome": proposed_plus_observations[-1]["observation"],
            "action_objective": (
                "capture probability reward + weighted expected information gain "
                "- contact/motion cost"
            ),
            "termination": "measured sensor capture cue stops the fifth smooth segment",
            "claim": (
                "Task-aware action ordering seeks the capture set; it does not "
                "guarantee monotonic posterior-entropy contraction."
            ),
        },
        "figure_contract": {
            "pure_outward_is_faint_same_start_reference_only": True,
            "paths_are_measured_tip_xy_from_replayed_mujoco_traces": True,
            "commanded_entropy_endpoints_are_numbered_separately": True,
            "capture_and_bottom_times_are_serialized_runresult_values": True,
            "capture_then_all_methods_use_shared_funnel": True,
            "selected_case_is_not_isolated_causal_ablation": True,
            "selected_case_is_not_distribution_level_evidence": True,
        },
        "artifacts": {
            "figure": _artifact(figure_path),
            "m8_mujoco_topview_base": _artifact(base_path),
            "results": _artifact(result_path),
            "frozen_verification": _artifact(verification_path),
            **{
                f"{method}_trace": _artifact(output / f"{method}_trace.csv")
                for method in METHODS
            },
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": _relative(output),
                "verification": verification["all_exact_substantive_match"],
                "capture_bottom_s": {
                    method: [
                        method_summary[method]["capture_time_s"],
                        method_summary[method]["bottom_time_s"],
                    ]
                    for method in ("proposed", "proposed_plus")
                },
                "entropy_is_monotonic_decreasing": not entropy_nonmonotonic,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
