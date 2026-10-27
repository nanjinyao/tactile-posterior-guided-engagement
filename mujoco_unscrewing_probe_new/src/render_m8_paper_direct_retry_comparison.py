from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.mujoco_admittance_rollout import DynamicSnapshot
from src.render_m8_dynamic_videos import SIM_TIMESTEP_S, setup
from src.render_m8_sampling_videos import CONTROLLER_THRESHOLDS, FPS, ROOT, smoothstep
from src.render_m8_search_comparison import (
    ALPHA,
    COMMON_XY_SPEED_M_S,
    INITIAL_XY_M,
    KEYFRAME_OUT,
    NOMINAL_TIP_Z_M,
    POST_PROBE_SETTLE_TIME_S,
    PROBE_RAMP_TIME_S,
    PROBE_RETURN_TIME_S,
    PROBE_SCAN_TIME_S,
    RESULT_PATH as PROPOSED_RESULT_PATH,
    SETTLE_TIME_S,
    SUMMARY_TIME_S,
    TARGET_DOWN_FORCE_N,
    VIDEO_OUT,
    LiveMetrics,
    RunResult,
    comparison_case,
    fast_contact_tags,
    render_overlay_frame,
)


PAPER_PDF = ROOT / "reference" / "锥形探索.pdf"
DIAGNOSTIC_DIR = ROOT / "outputs/current/diagnostics/paper_direct_scripted_stress"
RESULT_PATH = DIAGNOSTIC_DIR / "result.json"
PAPER_VIDEO = DIAGNOSTIC_DIR / "scripted_direct_retry_to_inner_wall.mp4"
COMPARISON_VIDEO = DIAGNOSTIC_DIR / "scripted_stress_vs_assumed_proposed.mp4"
PAPER_KEYFRAME = DIAGNOSTIC_DIR / "scripted_direct_retry_keyframe.png"
COMPARISON_KEYFRAME = DIAGNOSTIC_DIR / "scripted_stress_comparison_keyframe.png"
PROPOSED_VIDEO = VIDEO_OUT / "m8_outer_partial_proposed_probe_to_inner_wall.mp4"

MAX_ATTEMPTS = 3
DIRECT_DIRECTION_ERRORS_DEG = (22.5, -22.5, 0.0)
RESET_OFFSETS_M = (
    np.array([0.0, 0.0]),
    np.array([0.0, 0.00025]),
    np.array([0.0, -0.00025]),
)
DIRECT_OVERSHOOT_M = 0.00012
INSERTION_FORCE_N = 0.8
INSERTION_VERIFY_TIME_S = 1.0
RESET_CLEARANCE_Z_M = 0.0010
YAW_OSCILLATION_RAD = math.radians(1.0)
YAW_OSCILLATION_HZ = 1.5


@dataclass
class AttemptResult:
    attempt: int
    reset_xy_mm: list[float]
    true_direction_deg: float
    direction_error_deg: float
    predicted_direction_deg: float
    commanded_travel_mm: float
    outcome: str
    end_xy_mm: list[float]
    active_time_end_s: float


@dataclass
class PaperDirectResult:
    method: str
    result: str
    capture_confirmed: bool
    successful_attempt: int | None
    time_to_capture_s: float | None
    total_active_time_s: float
    actual_xy_path_mm: float
    contact_xy_path_mm: float
    wear_proxy_mj: float
    max_force_n: float
    max_torque_mnm: float
    min_tip_z_mm: float
    hard_stop: bool
    attempts: list[AttemptResult]
    frame_count: int
    active_frame_count: int
    video_path: str


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _angle_deg(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0


def _capture_condition(
    snapshot: DynamicSnapshot,
    spec: dict,
    model,
    data,
) -> bool:
    tags = fast_contact_tags(model, data)
    capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    inside = float(np.linalg.norm(snapshot.tip_position[:2])) <= capture_radius_m
    below_rim = (
        1000.0 * float(snapshot.tip_position[2])
        <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
    )
    lateral_force = float(np.linalg.norm(snapshot.measured_wrench[:2]))
    force_norm = float(np.linalg.norm(snapshot.measured_wrench[:3]))
    torque_xy_mnm = 1000.0 * float(np.linalg.norm(snapshot.measured_wrench[3:5]))
    wall_guided = "wall" in tags and (
        lateral_force >= CONTROLLER_THRESHOLDS["inner_wall_ft_N"]
        or torque_xy_mnm >= CONTROLLER_THRESHOLDS["inner_wall_tau_xy_mNm"]
    )
    centered = (
        "head_top" not in tags
        and force_norm <= CONTROLLER_THRESHOLDS["soft_force_N"]
        and torque_xy_mnm <= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
    )
    return bool(inside and below_rim and (wall_guided or centered))


def _paper_summary_frame(frame: np.ndarray, result: PaperDirectResult) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (112, 104, 848, 608),
        radius=8,
        fill=(2, 6, 10, 232),
        outline=(255, 205, 105, 225),
        width=2,
    )
    draw.text(
        (142, 130),
        "PAPER-INSPIRED DIRECT + RETRY RESULT",
        fill=(255, 225, 155, 255),
        font=_font(21, bold=True),
    )
    y = 178
    for attempt in result.attempts:
        draw.text(
            (146, y),
            (
                f"attempt {attempt.attempt}: error={attempt.direction_error_deg:+.1f} deg  "
                f"phi={attempt.predicted_direction_deg:.1f} deg  {attempt.outcome}"
            ),
            fill=(235, 242, 248, 250),
            font=_font(15),
        )
        y += 30
    rows = [
        f"capture: {result.capture_confirmed} on attempt {result.successful_attempt}",
        f"active time: {result.time_to_capture_s:.2f} s" if result.time_to_capture_s is not None else "active time: not reached",
        f"actual XY path: {result.actual_xy_path_mm:.2f} mm",
        f"work proxy: {result.wear_proxy_mj:.2f} mJ",
        "paper contract: direction class only; no error magnitude output",
        "recovery: shifted re-initialization + full tilt-then-rotate, max 3 attempts",
        "errors are scripted class-boundary stress, not paper-CNN measurements",
    ]
    y += 18
    for row in rows:
        draw.text((146, y), row, fill=(220, 232, 242, 245), font=_font(15))
        y += 29
    return np.asarray(image)


def run_paper_direct_retry(output_path: Path) -> PaperDirectResult:
    case = comparison_case("paper_direct")
    spec, model, data, rollout, renderer, camera, scene_option = setup(case, render=True)
    assert renderer is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    nominal_tip = np.array([*INITIAL_XY_M, NOMINAL_TIP_Z_M], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip)
    rollout.set_target_down_force(TARGET_DOWN_FORCE_N)
    metrics = LiveMetrics()
    clock_s = 0.0
    next_render_s = 0.0
    frame_count = 0
    current_stage = "paper baseline | initial contact settle"
    current_phase: float | None = None
    polar_history: list[tuple[float, float, float, float, float]] = []
    trail: list[np.ndarray] = []
    baseline_z_m: float | None = None
    last_frame: np.ndarray | None = None
    min_tip_z_m = float("inf")
    hard_stop = False
    capture_count = 0
    capture_required = max(
        1,
        int(
            round(
                CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                / 1000.0
                / SIM_TIMESTEP_S
            )
        ),
    )
    attempt_results: list[AttemptResult] = []
    successful_attempt: int | None = None
    time_to_capture_s: float | None = None

    def maybe_render(snapshot: DynamicSnapshot, show_cone: bool = False) -> None:
        nonlocal next_render_s, frame_count, last_frame
        while clock_s + 1.0e-12 >= next_render_s:
            if (
                "conical scan" in current_stage
                and current_phase is not None
                and (
                    not polar_history
                    or current_phase > polar_history[-1][0] + 1.0e-6
                )
            ):
                polar_history.append(
                    (
                        current_phase,
                        float(snapshot.measured_wrench[2]),
                        float(snapshot.measured_wrench[3]),
                        float(snapshot.measured_wrench[4]),
                        float(snapshot.tip_position[2]),
                    )
                )
            frame = render_overlay_frame(
                "paper_direct",
                current_stage,
                snapshot,
                metrics,
                model,
                data,
                renderer,
                camera,
                scene_option,
                spec,
                case,
                trail,
                baseline_z_m,
                show_cone,
                1.0,
                polar_history,
            )
            writer.append_data(frame)
            last_frame = frame
            frame_count += 1
            next_render_s += 1.0 / FPS

    def step_once(
        command_tip: np.ndarray,
        command_rpy: np.ndarray,
        stage_group: str,
        *,
        active: bool = True,
        show_cone: bool = False,
    ) -> DynamicSnapshot:
        nonlocal clock_s, min_tip_z_m, hard_stop, capture_count
        snapshot = rollout.step(command_tip, command_rpy)
        clock_s += SIM_TIMESTEP_S
        metrics.observe(snapshot, SIM_TIMESTEP_S, stage_group, data.ncon > 0, active)
        min_tip_z_m = min(min_tip_z_m, float(snapshot.tip_position[2]))
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        capture_count = (
            capture_count + 1
            if active and _capture_condition(snapshot, spec, model, data)
            else 0
        )
        maybe_render(snapshot, show_cone=show_cone)
        return snapshot

    def run_segment(
        duration_s: float,
        command,
        stage: str,
        stage_group: str,
        *,
        active: bool = True,
        show_cone: bool = False,
        stop_on_capture: bool = False,
    ) -> tuple[DynamicSnapshot, bool]:
        nonlocal current_stage
        current_stage = stage
        steps = max(1, int(round(duration_s / SIM_TIMESTEP_S)))
        snapshot: DynamicSnapshot | None = None
        for index in range(steps):
            fraction = (index + 1) / steps
            tip, rpy = command(fraction, index * SIM_TIMESTEP_S)
            snapshot = step_once(
                tip,
                rpy,
                stage_group,
                active=active,
                show_cone=show_cone,
            )
            if hard_stop or (stop_on_capture and capture_count >= capture_required):
                break
        assert snapshot is not None
        return snapshot, bool(capture_count >= capture_required)

    snapshot, _ = run_segment(
        SETTLE_TIME_S,
        lambda _f, _t: (nominal_tip.copy(), np.zeros(3)),
        current_stage,
        "settle",
        active=False,
    )
    metrics.reset_reference(snapshot)
    baseline_z_m = float(snapshot.tip_position[2])

    for attempt_index in range(MAX_ATTEMPTS):
        attempt = attempt_index + 1
        reset_xy = INITIAL_XY_M + RESET_OFFSETS_M[attempt_index]
        nominal_tip = np.array([*reset_xy, NOMINAL_TIP_Z_M], dtype=float)
        polar_history.clear()
        current_phase = None
        capture_count = 0

        snapshot, _ = run_segment(
            PROBE_RAMP_TIME_S,
            lambda f, _t: (
                nominal_tip.copy(),
                np.array([0.0, ALPHA * smoothstep(f), 0.0]),
            ),
            f"attempt {attempt}/{MAX_ATTEMPTS} | conical ramp",
            "probe",
            show_cone=True,
        )
        if hard_stop:
            break

        def scan_command(fraction: float, _elapsed: float):
            nonlocal current_phase
            current_phase = 2.0 * math.pi * fraction
            return (
                nominal_tip.copy(),
                np.array(
                    [
                        ALPHA * math.sin(current_phase),
                        ALPHA * math.cos(current_phase),
                        0.0,
                    ]
                ),
            )

        snapshot, _ = run_segment(
            PROBE_SCAN_TIME_S,
            scan_command,
            f"attempt {attempt}/{MAX_ATTEMPTS} | full 360 deg conical scan",
            "probe",
            show_cone=True,
        )
        if hard_stop:
            break
        snapshot, _ = run_segment(
            PROBE_RETURN_TIME_S,
            lambda f, _t: (
                nominal_tip.copy(),
                np.array([0.0, ALPHA * (1.0 - smoothstep(f)), 0.0]),
            ),
            f"attempt {attempt}/{MAX_ATTEMPTS} | return upright",
            "probe",
            show_cone=True,
        )
        if hard_stop:
            break
        snapshot, _ = run_segment(
            POST_PROBE_SETTLE_TIME_S,
            lambda _f, _t: (nominal_tip.copy(), np.zeros(3)),
            f"attempt {attempt}/{MAX_ATTEMPTS} | direction class ready",
            "probe",
        )
        if hard_stop:
            break

        true_vector = -reset_xy
        true_direction_deg = _angle_deg(true_vector)
        direction_error_deg = DIRECT_DIRECTION_ERRORS_DEG[attempt_index]
        predicted_direction_deg = (true_direction_deg + direction_error_deg) % 360.0
        predicted_direction_rad = math.radians(predicted_direction_deg)
        direction_xy = np.array(
            [math.cos(predicted_direction_rad), math.sin(predicted_direction_rad)],
            dtype=float,
        )
        travel_m = float(np.linalg.norm(reset_xy)) + DIRECT_OVERSHOOT_M
        target_xy = reset_xy + travel_m * direction_xy
        slide_duration_s = travel_m / COMMON_XY_SPEED_M_S

        def direct_slide(fraction: float, elapsed: float):
            xy = reset_xy + smoothstep(fraction) * (target_xy - reset_xy)
            yaw = YAW_OSCILLATION_RAD * math.sin(
                2.0 * math.pi * YAW_OSCILLATION_HZ * elapsed
            )
            return np.array([xy[0], xy[1], NOMINAL_TIP_Z_M]), np.array([0.0, 0.0, yaw])

        snapshot, captured = run_segment(
            slide_duration_s,
            direct_slide,
            (
                f"attempt {attempt}/{MAX_ATTEMPTS} | direct slide "
                f"phi={predicted_direction_deg:.1f} deg | error={direction_error_deg:+.1f} deg"
            ),
            "direct_slide",
            stop_on_capture=True,
        )
        nominal_tip = np.array([*target_xy, NOMINAL_TIP_Z_M], dtype=float)

        if not captured and not hard_stop:
            rollout.set_target_down_force(INSERTION_FORCE_N)
            rollout.reset_admittance()

            def insertion_hold(_fraction: float, elapsed: float):
                yaw = YAW_OSCILLATION_RAD * math.sin(
                    2.0 * math.pi * YAW_OSCILLATION_HZ * elapsed
                )
                return nominal_tip.copy(), np.array([0.0, 0.0, yaw])

            snapshot, captured = run_segment(
                INSERTION_VERIFY_TIME_S,
                insertion_hold,
                f"attempt {attempt}/{MAX_ATTEMPTS} | down-press + yaw verify",
                "insertion_verify",
                stop_on_capture=True,
            )

        outcome = "capture_confirmed" if captured else "failed_no_capture"
        attempt_results.append(
            AttemptResult(
                attempt=attempt,
                reset_xy_mm=(1000.0 * reset_xy).tolist(),
                true_direction_deg=true_direction_deg,
                direction_error_deg=direction_error_deg,
                predicted_direction_deg=predicted_direction_deg,
                commanded_travel_mm=1000.0 * travel_m,
                outcome=outcome if not hard_stop else "hard_stop",
                end_xy_mm=(1000.0 * snapshot.tip_position[:2]).tolist(),
                active_time_end_s=metrics.active_time_s,
            )
        )

        if captured:
            successful_attempt = attempt
            time_to_capture_s = metrics.active_time_s
            current_stage = f"attempt {attempt}/{MAX_ATTEMPTS} | capture confirmed"
            run_segment(
                0.50,
                lambda _f, _t: (nominal_tip.copy(), np.zeros(3)),
                current_stage,
                "handoff",
                active=False,
            )
            break
        if hard_stop or attempt == MAX_ATTEMPTS:
            break

        current_stage = (
            f"attempt {attempt}/{MAX_ATTEMPTS} failed | unload, shifted reset, re-probe"
        )
        rollout.set_target_down_force(0.15)
        rollout.reset_admittance()
        lift_start = nominal_tip.copy()
        lift_target = np.array([lift_start[0], lift_start[1], RESET_CLEARANCE_Z_M])
        snapshot, _ = run_segment(
            0.45,
            lambda f, _t: (
                lift_start + smoothstep(f) * (lift_target - lift_start),
                np.zeros(3),
            ),
            current_stage,
            "retry_retreat",
        )
        if hard_stop:
            break
        next_reset_xy = INITIAL_XY_M + RESET_OFFSETS_M[attempt_index + 1]
        reset_start = snapshot.tip_position.copy()
        reset_target = np.array([*next_reset_xy, RESET_CLEARANCE_Z_M])
        reset_duration_s = max(
            0.60,
            float(np.linalg.norm(reset_target[:2] - reset_start[:2])) / 0.004,
        )
        snapshot, _ = run_segment(
            reset_duration_s,
            lambda f, _t: (
                reset_start + smoothstep(f) * (reset_target - reset_start),
                np.zeros(3),
            ),
            f"retry {attempt}->{attempt + 1} | shifted re-initialization",
            "retry_reset",
        )
        if hard_stop:
            break
        nominal_tip = np.array([*next_reset_xy, NOMINAL_TIP_Z_M], dtype=float)
        rollout.set_target_down_force(TARGET_DOWN_FORCE_N)
        rollout.reset_admittance()
        snapshot, _ = run_segment(
            0.50,
            lambda _f, _t: (nominal_tip.copy(), np.zeros(3)),
            f"attempt {attempt + 1}/{MAX_ATTEMPTS} | reset contact settle",
            "retry_settle",
        )
        if hard_stop:
            break

    active_frame_count = frame_count
    result = PaperDirectResult(
        method="paper_direct_retry",
        result=(
            "capture_confirmed"
            if successful_attempt is not None
            else "hard_stop"
            if hard_stop
            else "failed_after_three_attempts"
        ),
        capture_confirmed=successful_attempt is not None,
        successful_attempt=successful_attempt,
        time_to_capture_s=time_to_capture_s,
        total_active_time_s=metrics.active_time_s,
        actual_xy_path_mm=1000.0 * metrics.actual_xy_path_m,
        contact_xy_path_mm=1000.0 * metrics.contact_xy_path_m,
        wear_proxy_mj=1000.0 * metrics.wear_proxy_j,
        max_force_n=metrics.max_force_n,
        max_torque_mnm=1000.0 * metrics.max_torque_nm,
        min_tip_z_mm=1000.0 * min_tip_z_m,
        hard_stop=hard_stop,
        attempts=attempt_results,
        frame_count=frame_count,
        active_frame_count=active_frame_count,
        video_path=str(output_path.relative_to(ROOT)),
    )
    assert last_frame is not None
    summary = _paper_summary_frame(last_frame, result)
    for _ in range(max(1, int(round(SUMMARY_TIME_S * FPS)))):
        writer.append_data(summary)
        frame_count += 1
    writer.close()
    renderer.close()
    result.frame_count = frame_count
    return result


def _load_proposed_result() -> RunResult:
    payload = json.loads(PROPOSED_RESULT_PATH.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(RunResult)}
    return RunResult(
        **{key: value for key, value in payload["proposed"].items() if key in allowed}
    )


def _comparison_summary(
    frame: np.ndarray,
    proposed: RunResult,
    paper: PaperDirectResult,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (240, 66, 1040, 440),
        radius=8,
        fill=(2, 6, 10, 238),
        outline=(225, 232, 240, 220),
        width=2,
    )
    draw.text(
        (274, 92),
        "ONE-PROBE COVERING SEARCH VS DIRECTION-ONLY RETRY",
        fill=(245, 248, 252, 255),
        font=_font(20, bold=True),
    )
    proposed_time = proposed.time_to_inner_wall_s or float("nan")
    paper_time = paper.time_to_capture_s or float("nan")
    rows = [
        f"proposed: attempt 1 | {proposed_time:.2f}s | XY {proposed.actual_xy_path_mm:.2f}mm",
        f"paper-inspired: attempt {paper.successful_attempt} | {paper_time:.2f}s | XY {paper.actual_xy_path_mm:.2f}mm",
        f"paper attempts: {', '.join(a.outcome for a in paper.attempts)}",
        "paper contract: CNN direction class only; error magnitude is unavailable",
        "paper recovery: shifted reset + repeat full tilt-then-rotate; max 3 attempts",
        "proposed advantage: local spiral covers residual error inside its bounded search region",
        "selected scripted boundary errors (+22.5, -22.5, 0 deg); not measured CNN outputs",
        "single M8 MuJoCo stress illustration; no distribution or hardware superiority claim",
    ]
    y = 140
    for index, row in enumerate(rows):
        color = (90, 225, 255, 250) if index == 0 else (255, 215, 140, 250) if index == 1 else (225, 234, 242, 245)
        draw.text((278, y), row, fill=color, font=_font(15))
        y += 35
    return np.asarray(image)


def compose_comparison(
    proposed: RunResult,
    paper: PaperDirectResult,
) -> int:
    proposed_active = max(
        1,
        proposed.frame_count - max(1, int(round(SUMMARY_TIME_S * FPS))),
    )
    proposed_reader = imageio.get_reader(PROPOSED_VIDEO)
    paper_reader = imageio.get_reader(PAPER_VIDEO)
    proposed_count = min(proposed_active, proposed_reader.count_frames())
    paper_count = min(paper.active_frame_count, paper_reader.count_frames())
    count = max(proposed_count, paper_count)
    writer = imageio.get_writer(
        COMPARISON_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last_proposed: np.ndarray | None = None
    last_paper: np.ndarray | None = None
    last_composite: np.ndarray | None = None
    try:
        for index in range(count):
            if index < proposed_count:
                last_proposed = proposed_reader.get_data(index)
            if index < paper_count:
                last_paper = paper_reader.get_data(index)
            assert last_proposed is not None and last_paper is not None
            canvas = Image.new("RGB", (1280, 520), (8, 10, 13))
            canvas.paste(
                Image.fromarray(last_proposed).resize((640, 480), Image.Resampling.LANCZOS),
                (0, 40),
            )
            canvas.paste(
                Image.fromarray(last_paper).resize((640, 480), Image.Resampling.LANCZOS),
                (640, 40),
            )
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text(
                (18, 10),
                "PROPOSED | posterior transport + covering spiral",
                fill=(90, 225, 255, 255),
                font=_font(17, bold=True),
            )
            draw.text(
                (658, 10),
                "JIN 2021-INSPIRED | direction-only direct slide + retry",
                fill=(255, 205, 105, 255),
                font=_font(16, bold=True),
            )
            draw.text(
                (1162, 12),
                f"1x {index / FPS:05.1f}s",
                fill=(220, 230, 240, 255),
                font=_font(13),
            )
            last_composite = np.asarray(canvas)
            writer.append_data(last_composite)
        assert last_composite is not None
        summary = _comparison_summary(last_composite, proposed, paper)
        imageio.imwrite(COMPARISON_KEYFRAME, summary)
        for _ in range(max(1, int(round(3.0 * FPS)))):
            writer.append_data(summary)
            count += 1
    finally:
        writer.close()
        proposed_reader.close()
        paper_reader.close()
    return count


def main() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    KEYFRAME_OUT.mkdir(parents=True, exist_ok=True)
    if not PAPER_PDF.is_file():
        raise FileNotFoundError(PAPER_PDF)
    if not PROPOSED_VIDEO.is_file() or not PROPOSED_RESULT_PATH.is_file():
        raise FileNotFoundError("regenerate the proposed comparison artifacts first")

    proposed = _load_proposed_result()
    paper = run_paper_direct_retry(PAPER_VIDEO)
    comparison_frames = compose_comparison(proposed, paper)

    reader = imageio.get_reader(PAPER_VIDEO)
    try:
        imageio.imwrite(
            PAPER_KEYFRAME,
            reader.get_data(max(0, paper.active_frame_count - 1)),
        )
    finally:
        reader.close()

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "paper_source": {
            "path": str(PAPER_PDF.relative_to(ROOT)),
            "title": "Contact Pose Identification for Peg-in-Hole Assembly under Uncertainties",
            "implemented_contract": (
                "Section III-A/C/D: tilt-then-rotate, direction-class output, direct admittance guidance, "
                "and shifted re-initialization with a repeated full probe after failed insertion"
            ),
            "reported_boundary": (
                "the paper reports wrong direction/admittance compensation as failure causes and "
                "declares failure after more than three attempts"
            ),
        },
        "claim_boundary": (
            "Scripted recovery diagnostic, not an active baseline and not an exact reproduction: the original square/pentagonal CNN, "
            "training data, controller gains, and geometry are unavailable here. Direction errors are "
            "scripted class-boundary stress values selected to visualize the paper's documented recovery path."
        ),
        "fairness": {
            "same_initial_xy_mm": (1000.0 * INITIAL_XY_M).tolist(),
            "same_probe_alpha_deg": math.degrees(ALPHA),
            "same_target_down_force_n": TARGET_DOWN_FORCE_N,
            "same_direct_or_spiral_xy_speed_mm_s": 1000.0 * COMMON_XY_SPEED_M_S,
            "same_capture_evaluator": (
                "below-rim inner-wall/inserted gate + 50 ms dwell + privileged M8 CAD capture region"
            ),
            "proposed_probe_cost_included": True,
            "paper_retry_probe_cost_included": True,
        },
        "scripted_stress": {
            "direction_errors_deg": list(DIRECT_DIRECTION_ERRORS_DEG),
            "reset_offsets_mm": [(1000.0 * value).tolist() for value in RESET_OFFSETS_M],
            "selection_boundary": (
                "selected recovery illustration; not a random draw and not evidence of the paper method's failure rate"
            ),
        },
        "proposed": asdict(proposed),
        "paper_inspired": asdict(paper),
        "comparison": {
            "proposed_faster": (
                proposed.time_to_inner_wall_s is not None
                and paper.time_to_capture_s is not None
                and proposed.time_to_inner_wall_s < paper.time_to_capture_s
            ),
            "proposed_has_bounded_residual_coverage": True,
            "paper_direct_has_residual_coverage": False,
            "comparison_video": str(COMPARISON_VIDEO.relative_to(ROOT)),
            "comparison_frames": comparison_frames,
        },
        "videos": {
            "proposed": str(PROPOSED_VIDEO.relative_to(ROOT)),
            "paper_inspired": str(PAPER_VIDEO.relative_to(ROOT)),
            "side_by_side": str(COMPARISON_VIDEO.relative_to(ROOT)),
        },
        "keyframes": {
            "paper_inspired": str(PAPER_KEYFRAME.relative_to(ROOT)),
            "side_by_side": str(COMPARISON_KEYFRAME.relative_to(ROOT)),
        },
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
