from __future__ import annotations

import bisect
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.funnel_guard import FunnelGuardConfig
from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult, run_method


OUT = ROOT / "outputs/current/active_comparison/funnel_extreme_recovery"
EVALUATION = ROOT / "outputs/current/evaluations/m8_extreme_funnel_recovery"
LEGACY_VIDEO = OUT / "01_no_physical_funnel.mp4"
GUARD_VIDEO = OUT / "02_tip_rcc_funnel_recovery.mp4"
LEGACY_TRACE = OUT / "01_no_physical_funnel_trace.csv"
GUARD_TRACE = OUT / "02_tip_rcc_funnel_recovery_trace.csv"
COMPARISON_VIDEO = OUT / "comparison_extreme_tilt_funnel_recovery.mp4"
KEYFRAME = OUT / "comparison_extreme_tilt_funnel_recovery_keyframe.png"
MANIFEST = OUT / "manifest.json"
REPORT = EVALUATION / "report.json"


SCENARIO = {
    "initial_xy_m": [0.0010, 0.0],
    "baseline_max_radius_m": 0.0020,
    "target_down_force_n": 5.0,
    "friction_slide": 1.2,
    "contact_timeconst_s": 0.02,
    "insertion_command_bias_xy_m": [0.00025, 0.0],
    "post_capture_tilt_disturbance_deg": 8.0,
    "post_capture_tilt_axis_deg": 0.0,
    "post_capture_lateral_disturbance_m": 0.00060,
    "post_capture_disturbance_persistent": True,
    "post_capture_tilt_ramp_s": 0.12,
    "post_capture_tilt_hold_s": 0.20,
}


GUARD_CONFIG = FunnelGuardConfig(
    enabled=True,
    max_tilt_deg=3.0,
    tilt_dwell_s=0.010,
    injected_power_limit_w=10.0,
    power_dwell_s=0.10,
    no_progress_energy_j=0.020,
    recovery_force_n=0.10,
    recovery_retract_m=0.00060,
    recovery_unload_s=0.12,
    recovery_retract_s=0.20,
    recovery_retouch_s=0.30,
)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _run(enabled: bool, output: Path, trace: Path) -> RunResult:
    return run_method(
        "baseline",
        output,
        initial_xy_m=np.asarray(SCENARIO["initial_xy_m"], dtype=float),
        baseline_max_radius_m=float(SCENARIO["baseline_max_radius_m"]),
        target_down_force_n=float(SCENARIO["target_down_force_n"]),
        friction_slide=float(SCENARIO["friction_slide"]),
        contact_timeconst_s=float(SCENARIO["contact_timeconst_s"]),
        active_time_budget_s=45.0,
        continue_to_bottom=True,
        cad_conditioned_insertion=False,
        funnel_guard=GUARD_CONFIG if enabled else FunnelGuardConfig(enabled=False),
        insertion_command_bias_xy_m=np.asarray(
            SCENARIO["insertion_command_bias_xy_m"], dtype=float
        ),
        post_capture_tilt_disturbance_deg=float(
            SCENARIO["post_capture_tilt_disturbance_deg"]
        ),
        post_capture_tilt_axis_deg=float(SCENARIO["post_capture_tilt_axis_deg"]),
        post_capture_lateral_disturbance_m=float(
            SCENARIO["post_capture_lateral_disturbance_m"]
        ),
        post_capture_disturbance_persistent=bool(
            SCENARIO["post_capture_disturbance_persistent"]
        ),
        post_capture_tilt_ramp_s=float(SCENARIO["post_capture_tilt_ramp_s"]),
        post_capture_tilt_hold_s=float(SCENARIO["post_capture_tilt_hold_s"]),
        trace_output_path=trace,
        observation_label=(
            "extreme paired residual-pose stress | shield enabled"
            if enabled
            else "extreme paired residual-pose stress | no shield"
        ),
    )


def _load_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            force = np.array(
                [
                    float(row.get("raw_physical_Fx_N", row["Fx_N"])),
                    float(row.get("raw_physical_Fy_N", row["Fy_N"])),
                    float(row.get("raw_physical_Fz_N", row["Fz_N"])),
                ],
                dtype=float,
            )
            torque = np.array(
                [
                    float(row.get("raw_physical_Tx_Nm", row["Tx_Nm"])),
                    float(row.get("raw_physical_Ty_Nm", row["Ty_Nm"])),
                    float(row.get("raw_physical_Tz_Nm", row["Tz_Nm"])),
                ],
                dtype=float,
            )
            rows.append(
                {
                    "time_s": float(row["clock_s"]),
                    "tilt_deg": math.degrees(
                        math.hypot(float(row["roll_rad"]), float(row["pitch_rad"]))
                    ),
                    "force_n": float(np.linalg.norm(force)),
                    "torque_mnm": 1000.0 * float(np.linalg.norm(torque)),
                    "tip_z_mm": 1000.0 * float(row["tip_z_m"]),
                    "stage": row["stage"],
                    "stage_group": row["stage_group"],
                    "hard_stop": row["hard_stop"] == "True",
                    "guard_active": row["funnel_guard_active"] == "True",
                }
            )
    return rows


def _trace_at(trace: list[dict], time_s: float) -> dict:
    times = [row["time_s"] for row in trace]
    index = max(0, min(len(trace) - 1, bisect.bisect_right(times, time_s) - 1))
    return trace[index]


def _active_frames(path: Path, result: RunResult) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        summary = max(1, int(round(SUMMARY_TIME_S * FPS)))
        count = min(reader.count_frames(), max(1, result.frame_count - summary))
        return [reader.get_data(index) for index in range(count)]
    finally:
        reader.close()


def _impact_start(trace: list[dict]) -> float:
    return min(
        row["time_s"]
        for row in trace
        if row["stage_group"] == "post_capture_tilt_impact"
    )


def _series_points(
    trace: list[dict],
    key: str,
    time_s: float,
    box: tuple[int, int, int, int],
    start_s: float,
    end_s: float,
    value_max: float,
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    samples = [row for row in trace if start_s <= row["time_s"] <= time_s]
    return [
        (
            x0 + (x1 - x0) * (row["time_s"] - start_s) / (end_s - start_s),
            y1 - (y1 - y0) * min(float(row[key]), value_max) / value_max,
        )
        for row in samples
    ]


def _draw_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    key: str,
    value_max: float,
    time_s: float,
    start_s: float,
    end_s: float,
    legacy_trace: list[dict],
    guard_trace: list[dict],
    thresholds: tuple[tuple[float, str, tuple[int, int, int]], ...],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill=(5, 9, 14), outline=(80, 100, 120))
    draw.text((x0 + 8, y0 + 6), title, fill=(230, 237, 245), font=_font(12, bold=True))
    for value, label, color in thresholds:
        y = y1 - (y1 - y0) * value / value_max
        draw.line((x0, y, x1, y), fill=(*color, 220), width=2)
        draw.text((x1 - 180, y - 16), label, fill=(*color, 255), font=_font(9))
    for trace, color in (
        (legacy_trace, (255, 190, 90)),
        (guard_trace, (80, 225, 255)),
    ):
        points = _series_points(trace, key, time_s, box, start_s, end_s, value_max)
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
    draw.text((x0 + 8, y1 - 19), "no funnel", fill=(255, 190, 90), font=_font(9, bold=True))
    draw.text((x0 + 90, y1 - 19), "funnel", fill=(80, 225, 255), font=_font(9, bold=True))


def _status(row: dict, result: RunResult, guarded: bool, final: bool) -> str:
    if final:
        if result.bottom_reached:
            return "CAD BOTTOM VERIFIED"
        if result.hard_stop:
            return "RAW-WRENCH HARD STOP"
        if result.post_terminal_event_display_new_hard_stop:
            return "BOTTOM REJECTED -> POST-EVENT RAW HARD"
        if result.funnel_abort_reason == "cad_depth_overshoot":
            return "CAD OVERSHOOT -> NAMED SAFE ABORT"
        return "TILTED / CAD BOTTOM NOT VERIFIED"
    if row["hard_stop"]:
        return "RAW-WRENCH HARD STOP"
    if row["stage_group"] == "funnel_recovery":
        return "CAD OVERSHOOT -> BOUNDED UNLOAD"
    if row["stage_group"] == "post_capture_tilt_impact":
        return "SAME SIDE-IMPACT + RESIDUAL POSE ERROR"
    if guarded and row["stage_group"] in {"insertion_handoff", "upright_insertion"}:
        return "SUPERVISED COMPLIANT INSERTION"
    return "POST-CAPTURE INSERTION"


def _comparison_frame(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    time_s: float,
    start_s: float,
    end_s: float,
    legacy: RunResult,
    guard: RunResult,
    legacy_trace: list[dict],
    guard_trace: list[dict],
    final: bool,
) -> np.ndarray:
    left_row = _trace_at(legacy_trace, time_s)
    right_row = _trace_at(guard_trace, time_s)
    canvas = Image.new("RGB", (1280, 720), (8, 10, 13))
    canvas.paste(Image.fromarray(left_frame).resize((640, 430), Image.Resampling.LANCZOS), (0, 50))
    canvas.paste(Image.fromarray(right_frame).resize((640, 430), Image.Resampling.LANCZOS), (640, 50))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.rectangle((0, 0, 640, 50), fill=(25, 13, 8, 255))
    draw.rectangle((640, 0, 1280, 50), fill=(4, 20, 25, 255))
    draw.text((16, 12), "NO SAFETY SUPERVISOR", fill=(255, 190, 90), font=_font(19, bold=True))
    draw.text((656, 12), "TIP-RCC + SAFETY SUPERVISOR", fill=(80, 225, 255), font=_font(19, bold=True))

    for x, label, color in (
        (18, _status(left_row, legacy, False, final), (255, 95, 95)),
        (658, _status(right_row, guard, True, final), (80, 225, 255)),
    ):
        draw.rounded_rectangle((x, 58, x + 430, 96), radius=5, fill=(2, 6, 11, 230), outline=(*color, 245), width=2)
        draw.text((x + 10, 68), label, fill=(*color, 255), font=_font(13, bold=True))

    draw.rectangle((4, 438, 636, 480), fill=(3, 7, 12, 235))
    draw.rectangle((644, 438, 1276, 480), fill=(3, 7, 12, 235))
    draw.text(
        (13, 447),
        f"tilt={left_row['tilt_deg']:.2f} deg | |F|={left_row['force_n']:.1f} N | |tau|={left_row['torque_mnm']:.1f} mNm | z={left_row['tip_z_mm']:.2f} mm",
        fill=(255, 216, 160),
        font=_font(11, bold=True),
    )
    draw.text(
        (653, 447),
        f"tilt={right_row['tilt_deg']:.2f} deg | |F|={right_row['force_n']:.1f} N | |tau|={right_row['torque_mnm']:.1f} mNm | z={right_row['tip_z_mm']:.2f} mm",
        fill=(160, 235, 255),
        font=_font(11, bold=True),
    )

    _draw_plot(
        draw,
        (35, 500, 620, 685),
        "actual tilt magnitude [deg]",
        "tilt_deg",
        11.0,
        time_s,
        start_s,
        end_s,
        legacy_trace,
        guard_trace,
        (
            (1.43, "M8 deep geometric scale 1.43 deg", (255, 95, 95)),
            (3.0, "stress-demo recovery gate 3 deg", (120, 180, 255)),
        ),
    )
    _draw_plot(
        draw,
        (660, 500, 1245, 685),
        "raw physical force norm [N]",
        "force_n",
        34.0,
        time_s,
        start_s,
        end_s,
        legacy_trace,
        guard_trace,
        ((20.0, "hard limit 20 N", (255, 95, 95)),),
    )
    draw.text(
        (35, 697),
        "paired MuJoCo severe-fault demo | 8 deg + 0.6 mm residual command | safe rejection, not recovery-success or hardware evidence",
        fill=(205, 216, 228),
        font=_font(9),
    )
    return np.asarray(canvas)


def _compose(legacy: RunResult, guard: RunResult) -> int:
    left = _active_frames(LEGACY_VIDEO, legacy)
    right = _active_frames(GUARD_VIDEO, guard)
    legacy_trace = _load_trace(LEGACY_TRACE)
    guard_trace = _load_trace(GUARD_TRACE)
    start_s = min(_impact_start(legacy_trace), _impact_start(guard_trace)) - 0.45
    start_s = max(0.0, start_s)
    end_s = max(legacy_trace[-1]["time_s"], guard_trace[-1]["time_s"])
    source_frames = max(2, int(math.ceil((end_s - start_s) * FPS)))
    writer = imageio.get_writer(
        COMPARISON_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last: np.ndarray | None = None
    written = 0
    try:
        for local_index in range(source_frames):
            time_s = min(end_s, start_s + local_index / FPS)
            left_index = min(len(left) - 1, max(0, int(round(time_s * FPS))))
            right_index = min(len(right) - 1, max(0, int(round(time_s * FPS))))
            last = _comparison_frame(
                left[left_index],
                right[right_index],
                time_s,
                start_s,
                end_s,
                legacy,
                guard,
                legacy_trace,
                guard_trace,
                False,
            )
            repeat = 2
            for _ in range(repeat):
                writer.append_data(last)
                written += 1
        assert last is not None
        last = _comparison_frame(
            left[-1],
            right[-1],
            end_s,
            start_s,
            end_s,
            legacy,
            guard,
            legacy_trace,
            guard_trace,
            True,
        )
        imageio.imwrite(KEYFRAME, last)
        for _ in range(3 * FPS):
            writer.append_data(last)
            written += 1
    finally:
        writer.close()
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVALUATION.mkdir(parents=True, exist_ok=True)
    print("[1/3] real MuJoCo rollout: no physical funnel", flush=True)
    legacy = _run(False, LEGACY_VIDEO, LEGACY_TRACE)
    print("[2/3] same rollout: tip-RCC + funnel recovery", flush=True)
    guard = _run(True, GUARD_VIDEO, GUARD_TRACE)

    if (
        legacy.result != "bottom_not_verified"
        or legacy.hard_stop
        or legacy.bottom_reached
        or not legacy.post_terminal_event_display_new_hard_stop
    ):
        raise RuntimeError(
            "frozen unsupervised severe-fault branch must reject CAD bottom "
            "and expose the subsequent post-event raw-wrench violation"
        )
    if (
        guard.result != "funnel_safe_abort"
        or guard.hard_stop
        or guard.bottom_reached
        or guard.funnel_abort_reason != "cad_depth_overshoot"
    ):
        raise RuntimeError(
            "frozen supervised severe-fault branch must reject CAD depth "
            "overshoot without a hard stop"
        )
    if guard.post_capture_disturbance_recovered or guard.funnel_recovery_count != 0:
        raise RuntimeError(
            "terminal CAD-depth overshoot must not be relabelled as recoverable"
        )

    print("[3/3] compose synchronized extreme comparison", flush=True)
    frames = _compose(legacy, guard)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": SCENARIO,
        "guard_config": asdict(GUARD_CONFIG),
        "results": {
            "no_physical_funnel": asdict(legacy),
            "tip_rcc_funnel_recovery": asdict(guard),
        },
        "videos": {"comparison": str(COMPARISON_VIDEO.relative_to(ROOT))},
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "comparison_frames": frames,
        "derived_geometry": {
            "m8_single_side_clearance_mm": 0.10,
            "m8_socket_depth_mm": 4.0,
            "deep_tilt_scale_deg": math.degrees(math.atan(0.10 / 4.0)),
        },
        "selection_rule": (
            "Frozen paired severe-fault case must leave the unsupervised branch "
            "without CAD-bottom verification and expose its subsequent inactive-display "
            "raw-wrench violation; the supervised branch must reject CAD-depth overshoot "
            "without either a primary or post-event hard stop. This case is not a "
            "successful-recovery selector."
        ),
        "claim_boundary": (
            "Real MuJoCo paired mechanism stress demo with deliberately severe synthetic residual pose error: "
            "8 deg orientation command and 0.6 mm lateral TCP disturbance exceed the nominal M8 deep-clearance scale. "
            "The current controller correctly treats the resulting CAD-depth overshoot as an unrecoverable terminal "
            "event and performs a named safe abort without a raw-wrench violation. The unsupervised branch first returns "
            "bottom-not-verified and then exceeds the raw limit on the explicitly labelled inactive display step; that "
            "late step is not relabelled as the primary terminal outcome. This trial does not exercise the full "
            "unload/retract/upright/retouch restart branch. It is not evidence of recovery success, fault frequency, "
            "hardware robustness, strict passivity, or a larger nominal task success rate."
        ),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LEGACY_VIDEO.unlink(missing_ok=True)
    GUARD_VIDEO.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "legacy": legacy.result,
                "unsupervised": legacy.result,
                "unsupervised_peak_tilt_deg": legacy.max_tilt_deg,
                "supervised": guard.result,
                "supervised_abort_reason": guard.funnel_abort_reason,
                "supervised_peak_tilt_deg": guard.max_tilt_deg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
