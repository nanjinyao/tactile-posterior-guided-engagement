from __future__ import annotations

import bisect
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.funnel_guard import FunnelGuardConfig
from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult, run_method


EVALUATION = ROOT / "outputs/current/evaluations/m8_funnel_guard_ablation"
OUT = ROOT / "outputs/current/active_comparison/funnel_guard"
LEGACY_VIDEO = OUT / "01_legacy_post_capture_insertion.mp4"
GUARD_VIDEO = OUT / "02_tip_rcc_funnel_guard.mp4"
COMPARISON_VIDEO = OUT / "comparison_post_capture_funnel_guard.mp4"
KEYFRAME = OUT / "comparison_post_capture_funnel_guard_keyframe.png"
MANIFEST = OUT / "manifest.json"
LEGACY_TRACE = OUT / "01_legacy_trace.csv"
GUARD_TRACE = OUT / "02_guard_trace.csv"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _load_result(value: dict) -> RunResult:
    allowed = {field.name for field in fields(RunResult)}
    return RunResult(**{key: item for key, item in value.items() if key in allowed})


def _select_audit_case() -> dict:
    hard_stop_defenses = []
    bounded_abort_defenses = []
    for path in sorted((EVALUATION / "trials").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        legacy = record["methods"]["legacy_insertion"]
        guarded = record["methods"]["funnel_guard"]
        if legacy["hard_stop"] and not guarded["hard_stop"]:
            hard_stop_defenses.append(record)
        elif legacy["result"] == "bottom_not_verified" and guarded["result"] == "funnel_safe_abort":
            bounded_abort_defenses.append(record)
    candidates = hard_stop_defenses or bounded_abort_defenses
    if not candidates:
        raise RuntimeError("no audit-selected CAD-depth-defense case is available")
    return candidates[0]


def _run(record: dict, *, enabled: bool, output: Path, trace: Path) -> RunResult:
    task = record["task"]
    return run_method(
        "baseline",
        output,
        initial_xy_m=np.asarray(task["initial_xy_m"], dtype=float),
        baseline_max_radius_m=0.0020,
        baseline_phase_rad=float(task["baseline_phase_rad"]),
        target_down_force_n=float(task["target_down_force_n"]),
        friction_slide=float(task["friction_slide"]),
        contact_timeconst_s=float(task["contact_timeconst_s"]),
        active_time_budget_s=30.0,
        continue_to_bottom=True,
        funnel_guard=FunnelGuardConfig(enabled=enabled),
        insertion_command_bias_xy_m=np.asarray(task["insertion_bias_xy_m"], dtype=float),
        trace_output_path=trace,
    )


def _active_frames(path: Path, result: RunResult) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        summary = max(1, int(round(SUMMARY_TIME_S * FPS)))
        count = min(reader.count_frames(), max(1, result.frame_count - summary))
        return [reader.get_data(index) for index in range(count)]
    finally:
        reader.close()


def _load_trace(path: Path) -> list[dict]:
    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            force = np.array(
                [float(row["Fx_N"]), float(row["Fy_N"]), float(row["Fz_N"])],
                dtype=float,
            )
            torque = np.array(
                [float(row["Tx_Nm"]), float(row["Ty_Nm"]), float(row["Tz_Nm"])],
                dtype=float,
            )
            result.append(
                {
                    "time_s": float(row["clock_s"]),
                    "force_n": float(np.linalg.norm(force)),
                    "torque_mnm": 1000.0 * float(np.linalg.norm(torque)),
                    "tip_z_mm": 1000.0 * float(row["tip_z_m"]),
                    "stage": row["stage"],
                    "hard_stop": row["hard_stop"] == "True",
                    "guard_active": row["funnel_guard_active"] == "True",
                }
            )
    return result


def _trace_at(trace: list[dict], time_s: float) -> dict:
    times = [row["time_s"] for row in trace]
    index = max(0, min(len(trace) - 1, bisect.bisect_right(times, time_s) - 1))
    return trace[index]


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(box, radius=5, fill=(3, 7, 12, 232), outline=(*color, 245), width=2)
    draw.text((box[0] + 12, box[1] + 8), label, fill=(*color, 255), font=_font(14, bold=True))


def _series_points(
    trace: list[dict],
    key: str,
    time_s: float,
    box: tuple[int, int, int, int],
    time_max_s: float,
    value_max: float,
) -> list[tuple[float, float]]:
    values = [row for row in trace if row["time_s"] <= time_s + 1.0e-12]
    if not values:
        return []
    # Keep the full trace: the hard-stop impulse is physically narrow and a
    # naive fixed-stride decimation can erase the very peak being audited.
    sampled = values
    x0, y0, x1, y1 = box
    return [
        (
            x0 + (x1 - x0) * min(row["time_s"], time_max_s) / time_max_s,
            y1 - (y1 - y0) * min(float(row[key]), value_max) / value_max,
        )
        for row in sampled
    ]


def _draw_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    key: str,
    threshold: float,
    value_max: float,
    time_s: float,
    time_max_s: float,
    legacy_trace: list[dict],
    guard_trace: list[dict],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill=(5, 9, 14), outline=(80, 100, 120), width=1)
    threshold_y = y1 - (y1 - y0) * threshold / value_max
    draw.line((x0, threshold_y, x1, threshold_y), fill=(255, 95, 95), width=2)
    draw.text((x0 + 8, y0 + 6), title, fill=(230, 237, 245), font=_font(12, bold=True))
    draw.text(
        (x1 - 170, threshold_y - 17),
        f"hard limit {threshold:g}",
        fill=(255, 115, 115),
        font=_font(10),
    )
    for trace, color in ((legacy_trace, (255, 205, 105)), (guard_trace, (90, 225, 255))):
        points = _series_points(trace, key, time_s, box, time_max_s, value_max)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
    draw.text((x0 + 8, y1 - 20), "legacy", fill=(255, 205, 105), font=_font(10, bold=True))
    draw.text((x0 + 80, y1 - 20), "guard", fill=(90, 225, 255), font=_font(10, bold=True))
    draw.text((x1 - 78, y1 - 20), f"{time_s:3.1f}s", fill=(185, 200, 215), font=_font(10))


def _comparison_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    source_index: int,
    source_total: int,
    legacy: RunResult,
    guard: RunResult,
    legacy_trace: list[dict],
    guard_trace: list[dict],
    task_id: str,
) -> np.ndarray:
    time_s = source_index / FPS
    time_max_s = max(legacy_trace[-1]["time_s"], guard_trace[-1]["time_s"])
    left_state = _trace_at(legacy_trace, time_s)
    right_state = _trace_at(guard_trace, time_s)
    canvas = Image.new("RGB", (1280, 720), (8, 10, 13))
    left = Image.fromarray(left_frame).resize((640, 430), Image.Resampling.LANCZOS)
    right = Image.fromarray(right_frame).resize((640, 430), Image.Resampling.LANCZOS)
    canvas.paste(left, (0, 50))
    canvas.paste(right, (640, 50))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.rectangle((0, 0, 640, 50), fill=(20, 15, 8, 255))
    draw.rectangle((640, 0, 1280, 50), fill=(4, 18, 25, 255))
    draw.text((16, 13), "WITHOUT SHIELD", fill=(255, 205, 105), font=_font(19, bold=True))
    draw.text((656, 13), "TIP-RCC + FUNNEL SHIELD", fill=(90, 225, 255), font=_font(19, bold=True))
    draw.text((470, 17), f"paired {task_id}", fill=(210, 218, 226), font=_font(11))
    draw.text((1110, 17), f"same input", fill=(210, 218, 226), font=_font(11))

    legacy_status = (
        "HARD STOP"
        if time_s >= legacy_trace[-1]["time_s"] - 1.0 / FPS and legacy.hard_stop
        else "UNPROTECTED INSERTION"
    )
    guard_status = (
        "SAFE ABORT / CONTACT UNLOADED"
        if source_index >= source_total - 1 and guard.result == "funnel_safe_abort"
        else "GUARD MONITORING"
        if right_state["guard_active"]
        else "SAME SEARCH BEFORE CAPTURE"
    )
    _draw_badge(draw, (20, 61, 245, 101), legacy_status, (255, 95, 95) if "HARD" in legacy_status else (255, 205, 105))
    _draw_badge(draw, (660, 61, 1005, 101), guard_status, (90, 225, 255))

    draw.rectangle((4, 440, 636, 478), fill=(3, 7, 12, 232))
    draw.rectangle((644, 440, 1276, 478), fill=(3, 7, 12, 232))
    draw.text(
        (14, 450),
        f"now |F|={left_state['force_n']:.1f} N  |tau|={left_state['torque_mnm']:.1f} mNm  z={left_state['tip_z_mm']:.2f} mm  | peak {legacy.max_force_n:.1f} N",
        fill=(255, 220, 165),
        font=_font(11, bold=True),
    )
    draw.text(
        (654, 450),
        f"now |F|={right_state['force_n']:.1f} N  |tau|={right_state['torque_mnm']:.1f} mNm  z={right_state['tip_z_mm']:.2f} mm  | peak {guard.max_force_n:.1f} N",
        fill=(165, 235, 255),
        font=_font(11, bold=True),
    )

    _draw_plot(
        draw,
        (35, 500, 620, 685),
        "external force norm [N]",
        "force_n",
        20.0,
        26.0,
        time_s,
        time_max_s,
        legacy_trace,
        guard_trace,
    )
    _draw_plot(
        draw,
        (660, 500, 1245, 685),
        "external torque norm [mNm]",
        "torque_mnm",
        60.0,
        150.0,
        time_s,
        time_max_s,
        legacy_trace,
        guard_trace,
    )
    footer = (
        "0.5x near divergence | same search, nuisance, down-force, phase and TCP bias | "
        "MuJoCo paired ablation; damage defense, not completion gain"
        if source_index >= int(2.0 * FPS)
        else "same search, nuisance, down-force, phase and TCP bias | waiting for post-capture divergence"
    )
    draw.text((35, 695), footer, fill=(210, 220, 230), font=_font(10))
    return np.asarray(canvas)


def _compose(legacy: RunResult, guard: RunResult, task_id: str) -> int:
    left = _active_frames(LEGACY_VIDEO, legacy)
    right = _active_frames(GUARD_VIDEO, guard)
    total = max(len(left), len(right))
    legacy_trace = _load_trace(LEGACY_TRACE)
    guard_trace = _load_trace(GUARD_TRACE)
    writer = imageio.get_writer(COMPARISON_VIDEO, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    last: np.ndarray | None = None
    written = 0
    try:
        for index in range(total):
            last = _comparison_frame(
                left[min(index, len(left) - 1)],
                right[min(index, len(right) - 1)],
                index,
                total,
                legacy,
                guard,
                legacy_trace,
                guard_trace,
                task_id,
            )
            repeat = 2 if index >= int(2.0 * FPS) else 1
            for _ in range(repeat):
                writer.append_data(last)
                written += 1
        assert last is not None
        imageio.imwrite(KEYFRAME, last)
        summary_frames = max(1, int(round(3.0 * FPS)))
        for _ in range(summary_frames):
            writer.append_data(last)
            written += 1
    finally:
        writer.close()
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = _select_audit_case()
    task_id = record["task"]["task_id"]
    print(f"[1/3] render legacy paired case {task_id}", flush=True)
    legacy = _run(record, enabled=False, output=LEGACY_VIDEO, trace=LEGACY_TRACE)
    print("[2/3] render tip-RCC funnel guard", flush=True)
    guard = _run(record, enabled=True, output=GUARD_VIDEO, trace=GUARD_TRACE)
    print("[3/3] compose synchronized comparison", flush=True)
    frames = _compose(legacy, guard, task_id)
    result_payload = {}
    for key, result in (("legacy_insertion", legacy), ("funnel_guard", guard)):
        value = asdict(result)
        value["component_video_path_archival_provenance"] = value["video_path"]
        value["video_path"] = str(COMPARISON_VIDEO.relative_to(ROOT))
        result_payload[key] = value
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "first sorted paired audit case where legacy hard-stops and guard avoids hard-stop; bounded-abort case is fallback",
        "task": record["task"],
        "results": result_payload,
        "videos": {"comparison": str(COMPARISON_VIDEO.relative_to(ROOT))},
        "archived_component_video_keys": ["legacy", "guard"],
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "comparison_frames": frames,
        "visualization": {
            "synchronized_trace_overlay": True,
            "force_hard_limit_n": 20.0,
            "torque_hard_limit_mnm": 60.0,
            "slow_motion_start_s": 2.0,
            "slow_motion_factor": 0.5,
            "keyframe_is_dynamic_divergence_frame": True
        },
        "video_retention_manifest": "outputs/current/VIDEO_RETENTION.json",
        "claim_boundary": "The paired MuJoCo case demonstrates prevention of a hard stop and bounded CAD-depth defense. Synchronized curves are derived from the same rollout traces, not a separate simulation. It is not a success-rate gain and does not demonstrate a passivity proof or hardware robustness.",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LEGACY_VIDEO.unlink(missing_ok=True)
    GUARD_VIDEO.unlink(missing_ok=True)
    print(json.dumps({"task": task_id, "legacy": legacy.result, "guard": guard.result}, indent=2))


if __name__ == "__main__":
    main()
