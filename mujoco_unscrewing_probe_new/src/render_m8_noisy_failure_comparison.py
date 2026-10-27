from __future__ import annotations

import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import FPS, ROOT, RunResult, run_method
from src.render_m8_search_comparison import (
    LiveMetrics,
    NOMINAL_TIP_Z_M,
    SUMMARY_TIME_S,
    comparison_case,
    render_overlay_frame,
)
from src.render_m8_dynamic_videos import SIM_TIMESTEP_S, setup
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
)
from src.wrench_observation import WrenchObservationConfig


AUDIT = ROOT / "outputs/current/evaluations/m8_pre_hardware_noise_audit"
OUT = ROOT / "outputs/current/diagnostics/historical_comparisons/robustness_noise"
COMPARISON_VIDEO = OUT / "comparison_shared_noise_b_retry_success.mp4"
COMPARISON_KEYFRAME = OUT / "comparison_shared_noise_b_retry_success_keyframe.png"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _select_record() -> tuple[dict, str]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((AUDIT / "trials").glob("*.json"))
    ]
    retry_success = [
        item
        for item in records
        if item["methods"]["proposed_posterior"]["inner_wall_confirmed"]
        and item["methods"]["baseline_b"]["inner_wall_confirmed"]
        and not item["methods"]["baseline_b"]["first_attempt_success"]
    ]
    if retry_success:
        return retry_success[0], (
            "first lexicographic trial: Proposed succeeds and Baseline B succeeds only after retry"
        )
    first_attempt = [
        item
        for item in records
        if item["methods"]["proposed_posterior"]["inner_wall_confirmed"]
        and not item["methods"]["baseline_b"]["first_attempt_success"]
    ]
    if first_attempt:
        return first_attempt[0], "first lexicographic trial: Proposed succeeds and Baseline-B attempt 1 fails"
    raise RuntimeError("audit contains no honest Proposed-success/Baseline-B-retry illustration")


def _active_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [reader.get_data(index) for index in range(reader.count_frames())]
    finally:
        reader.close()


def _attempt_active_frames(path: Path) -> list[np.ndarray]:
    frames = _active_frames(path)
    summary_frames = max(1, int(round(SUMMARY_TIME_S * FPS)))
    return frames[:-summary_frames] if len(frames) > summary_frames else frames


def _banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (590, 252, 946, 292), radius=6, fill=(0, 0, 0, 205), outline=(*color, 225), width=2
    )
    draw.text((606, 263), text, fill=(*color, 255), font=_font(15, bold=True))
    return np.asarray(image)


def _last_tip(trace_path: Path) -> np.ndarray:
    with trace_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")
    row = rows[-1]
    return np.array([float(row["tip_x_m"]), float(row["tip_y_m"]), float(row["tip_z_m"])])


def _trace_terminal_diagnostic(trace_path: Path) -> dict:
    with trace_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")
    row = rows[-1]
    force = np.array([float(row[key]) for key in ("Fx_N", "Fy_N", "Fz_N")])
    torque_xy = np.array([float(row[key]) for key in ("Tx_Nm", "Ty_Nm")])
    return {
        "terminal_tip_xy_mm": [1000.0 * float(row["tip_x_m"]), 1000.0 * float(row["tip_y_m"])],
        "terminal_tip_z_mm": 1000.0 * float(row["tip_z_m"]),
        "terminal_force_norm_N": float(np.linalg.norm(force)),
        "terminal_tau_xy_mNm": 1000.0 * float(np.linalg.norm(torque_xy)),
        "terminal_contact_tags_offline_only": row["offline_contact_tags"],
        "progress_candidate": row["progress_candidate"] == "True",
        "inner_wall_gate": row["inner_wall_gate"] == "True",
        "hard_stop": row["hard_stop"] == "True",
    }


def _render_retry_reset(
    output: Path,
    *,
    start_tip: np.ndarray,
    target_xy: np.ndarray,
    task: dict,
    sensor_seed: int,
    label: str,
) -> float:
    case = comparison_case("direct")
    observation = WrenchObservationConfig(
        seed=sensor_seed,
        delay_s=task["sensor_delay_s"],
    )
    spec, model, data, rollout, renderer, camera, scene_option = setup(
        case, render=True, wrench_observation=observation
    )
    assert renderer is not None
    model.geom_friction[:, 0] = float(task["friction_slide"])
    model.geom_solref[:, 0] = float(task["contact_timeconst_s"])
    low_force = 0.20
    safe_start_tip = np.asarray(start_tip, dtype=float).copy()
    # A new MuJoCo scene cannot inherit the previous contact solver state.
    # Reconstruct at the same XY but at the nominal surface height with the
    # contact force already released, then render the physical retract/shift.
    safe_start_tip[2] = max(NOMINAL_TIP_Z_M, float(safe_start_tip[2]))
    rollout.set_target_down_force(low_force)
    rollout.set_initial_tip_pose(safe_start_tip, np.zeros(3))
    metrics = LiveMetrics()
    trail: list[np.ndarray] = []
    polar_history: list[tuple[float, float, float, float, float]] = []
    clock_s = 0.0
    next_frame_s = 0.0
    current_stage = "retry force release"
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

    def step(command_tip: np.ndarray, force_n: float, stage: str) -> None:
        nonlocal clock_s, next_frame_s, current_stage
        current_stage = stage
        rollout.set_target_down_force(max(0.15, float(force_n)))
        snapshot = rollout.step(command_tip, np.zeros(3))
        clock_s += SIM_TIMESTEP_S
        metrics.observe(snapshot, SIM_TIMESTEP_S, "retry_reset", data.ncon > 0, True)
        while clock_s + 1.0e-12 >= next_frame_s:
            frame = render_overlay_frame(
                "direct",
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
                float(safe_start_tip[2]),
                False,
                1.0,
                polar_history,
                observation_label="physical unload + lift + shifted reset + retouchdown",
            )
            writer.append_data(_banner(frame, label, (150, 235, 155)))
            next_frame_s += 1.0 / FPS
        if snapshot.command.hard_stop or snapshot.state_invalid:
            raise RuntimeError(f"retry reset failed during {stage}: {snapshot.failure_reason}")

    def segment(
        start: np.ndarray,
        end: np.ndarray,
        duration_s: float,
        force_start: float,
        force_end: float,
        stage: str,
    ) -> np.ndarray:
        steps = max(1, int(round(duration_s / SIM_TIMESTEP_S)))
        for index in range(steps):
            fraction = (index + 1) / steps
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            tip = start + smooth * (end - start)
            force = force_start + smooth * (force_end - force_start)
            step(tip, force, stage)
        return end.copy()

    try:
        target_force = float(task["target_down_force_n"])
        current = safe_start_tip.copy()
        current = segment(current, current, 0.20, low_force, low_force, "retry force released")
        hover = current.copy()
        hover[2] = max(0.0020, float(current[2]) + 0.0015)
        current = segment(current, hover, 0.55, low_force, low_force, "retry vertical retract")
        shifted = np.array([target_xy[0], target_xy[1], hover[2]], dtype=float)
        shift_time = max(0.30, float(np.linalg.norm(shifted[:2] - current[:2])) / 0.00225)
        current = segment(current, shifted, shift_time, low_force, low_force, "retry shifted re-initialization")
        touchdown = np.array([target_xy[0], target_xy[1], NOMINAL_TIP_Z_M], dtype=float)
        current = segment(current, touchdown, 0.60, low_force, target_force, "retry guarded retouchdown")
        segment(current, current, 0.30, target_force, target_force, "retry contact settle")
    finally:
        writer.close()
        renderer.close()
    return clock_s


def _compose_full_direct_retry(
    attempt_paths: list[Path],
    reset_paths: list[Path],
    aggregate: dict,
) -> tuple[Path, int, float]:
    output = OUT / "03_baseline_b_full_three_attempt_retry.mp4"
    sequences: list[np.ndarray] = []
    for index, attempt_path in enumerate(attempt_paths):
        sequences.extend(
            _banner(frame, f"BASELINE B | ATTEMPT {index + 1}/3", (150, 235, 155))
            for frame in _attempt_active_frames(attempt_path)
        )
        if index < len(reset_paths):
            sequences.extend(_active_frames(reset_paths[index]))
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in sequences:
            writer.append_data(frame)
        final = Image.fromarray(sequences[-1])
        draw = ImageDraw.Draw(final, "RGBA")
        draw.rounded_rectangle(
            (135, 120, 825, 590), radius=8, fill=(0, 0, 0, 230), outline=(150, 235, 155, 230), width=2
        )
        draw.text((165, 148), "BASELINE B | FULL RETRY RESULT", fill=(180, 255, 190, 255), font=_font(22, bold=True))
        rows = [
            *[
                f"attempt {index + 1}: "
                + ("capture" if item["inner_wall_confirmed"] else "ray miss")
                for index, item in enumerate(aggregate["attempt_results"])
            ],
            f"audit active time (reset excluded): {aggregate['total_active_time_s']:.2f} s",
            f"audit contact path / work: {aggregate['contact_xy_path_mm']:.2f} mm / {aggregate['wear_proxy_mj']:.2f} mJ",
            (
                f"result: success on attempt {aggregate['attempts_used']}"
                if aggregate["inner_wall_confirmed"]
                else f"result: failed after {aggregate['attempts_used']} attempts"
            ),
            "video reset motion is physical; audit reset cost remains excluded",
        ]
        y = 195
        for row in rows:
            draw.text((165, y), row, fill=(235, 241, 247, 255), font=_font(15))
            y += 38
        summary = np.asarray(final)
        for _ in range(3 * FPS):
            writer.append_data(summary)
            sequences.append(summary)
    finally:
        writer.close()
    reset_duration = sum(len(_active_frames(path)) / FPS for path in reset_paths)
    return output, len(sequences), reset_duration


def _summary_frame(frame: np.ndarray, results: dict, record: dict, criterion: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (75, 45, 1365, 405),
        radius=8,
        fill=(3, 7, 12, 242),
        outline=(225, 232, 240, 225),
        width=2,
    )
    draw.text(
        (105, 75),
        "SHARED PRE-HARDWARE SENSOR PRIOR | AUDIT-SELECTED RETRY ILLUSTRATION",
        fill=(255, 235, 170, 255),
        font=_font(19, bold=True),
    )
    labels = (
        ("PROPOSED", "proposed", (90, 225, 255)),
        ("BASELINE A", "outward", (255, 205, 105)),
        ("BASELINE B | 3 ATTEMPTS", "direct", (150, 235, 155)),
    )
    columns = (350, 735, 1120)
    for x, (label, _, color) in zip(columns, labels):
        draw.text((x - 120, 117), label, fill=(*color, 255), font=_font(16, bold=True))
    rows = (
        ("capture", "inner_wall_confirmed", ""),
        ("result", "result", ""),
        ("audit active", "total_active_time_s", " s"),
        ("XY path", "actual_xy_path_mm", " mm"),
        ("work", "wear_proxy_mj", " mJ"),
        ("max force", "max_force_n", " N"),
        ("max torque", "max_torque_mnm", " mNm"),
    )
    y = 155
    for title, key, suffix in rows:
        draw.text((105, y), title, fill=(180, 194, 208, 255), font=_font(14))
        for x, (_, method, _) in zip(columns, labels):
            value = getattr(results[method], key)
            if isinstance(value, bool):
                text = "YES" if value else "NO"
            elif value is None:
                text = "n/a"
            elif isinstance(value, float):
                text = f"{value:.2f}{suffix}"
            else:
                text = str(value)
                if key == "result":
                    text = {
                        "cad_bottom_verified": "CAD bottom verified",
                        "safe_downward_progress_confirmed": "capture confirmed",
                        "inner_wall_gate_confirmed": "inner-wall confirmed",
                        "centered_inserted_gate_confirmed": "inserted confirmed",
                        "search_radius_exhausted": "search exhausted",
                    }.get(text, text)
            draw.text((x, y), text[:31], fill=(235, 241, 247, 255), font=_font(14))
        y += 30
    task = record["task"]
    draw.text(
        (105, 376),
        f"{task['pose_id']} | r={1000*task['radius_m']:.2f} mm | shared noisy probe | {criterion}",
        fill=(205, 218, 230, 255),
        font=_font(12),
    )
    return np.asarray(image)


def _compose(paths: dict[str, Path], results: dict, record: dict, criterion: str) -> int:
    frames = {key: _active_frames(path) for key, path in paths.items()}
    total = max(len(items) for items in frames.values())
    output = COMPARISON_VIDEO
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    last = None
    try:
        for index in range(total):
            canvas = Image.new("RGB", (1440, 416), (8, 10, 13))
            draw = ImageDraw.Draw(canvas, "RGBA")
            for column, key in enumerate(("proposed", "outward", "direct")):
                source = frames[key][min(index, len(frames[key]) - 1)]
                panel = Image.fromarray(source).resize((480, 360), Image.Resampling.LANCZOS)
                canvas.paste(panel, (480 * column, 56))
            draw.text((14, 13), "PROPOSED | posterior + residual coverage", fill=(90, 225, 255, 255), font=_font(14, bold=True))
            draw.text((505, 13), "BASELINE A | outward coverage", fill=(255, 205, 105, 255), font=_font(14, bold=True))
            draw.text((980, 13), "BASELINE B | shifted reset + 3 probes", fill=(150, 235, 155, 255), font=_font(14, bold=True))
            draw.text((1205, 35), f"shared sensor prior | {index / FPS:05.1f}s", fill=(215, 224, 234, 255), font=_font(11))
            last = np.asarray(canvas)
            writer.append_data(last)
        assert last is not None
        summary = _summary_frame(last, results, record, criterion)
        imageio.imwrite(COMPARISON_KEYFRAME, summary)
        for _ in range(3 * FPS):
            writer.append_data(summary)
            total += 1
    finally:
        writer.close()
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--compose-only" in sys.argv:
        manifest_path = OUT / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = {
            key: RunResult(**value) for key, value in manifest["results"].items()
        }
        paths = {
            "proposed": ROOT / manifest["videos"]["proposed"],
            "outward": ROOT / manifest["videos"]["outward"],
            "direct": ROOT / manifest["videos"]["direct"],
        }
        frames = _compose(
            paths,
            results,
            {"task": manifest["task"]},
            manifest["selection_rule"],
        )
        manifest["comparison_frames"] = frames
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    record, criterion = _select_record()
    task = record["task"]
    initial_xy = np.array([task["x_m"], task["y_m"]], dtype=float)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    common = dict(
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        baseline_phase_rad=task["baseline_phase_rad"],
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task["target_down_force_n"],
        friction_slide=task["friction_slide"],
        contact_timeconst_s=task["contact_timeconst_s"],
        active_time_budget_s=130.0,
        continue_to_bottom=True,
        observation_label="shared F/T bias + colored noise + coupling + delay",
    )
    paths = {
        "proposed": OUT / "01_proposed_shared_noise.mp4",
        "outward": OUT / "02_baseline_a_shared_noise.mp4",
    }
    proposed_trace = OUT / "01_proposed_shared_noise_trace.csv"
    outward_trace = OUT / "02_baseline_a_shared_noise_trace.csv"
    results = {}
    print(f"[1/4] render Proposed from {task['pose_id']}", flush=True)
    results["proposed"] = run_method(
        "learned",
        paths["proposed"],
        initial_xy_m=initial_xy,
        posterior_seed=task["posterior_seed"],
        direction_runtime=runtime,
        max_transport_m=None,
        trace_output_path=proposed_trace,
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
        **common,
    )
    print("[2/4] render Baseline A outward", flush=True)
    results["outward"] = run_method(
        "baseline",
        paths["outward"],
        initial_xy_m=initial_xy,
        posterior_seed=task["posterior_seed"],
        trace_output_path=outward_trace,
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
        **common,
    )
    attempt_dir = OUT / "baseline_b_attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    audit_attempts = record["methods"]["baseline_b"]["attempt_results"]
    attempt_results = []
    attempt_paths = []
    trace_paths = []
    for index, audit_attempt in enumerate(audit_attempts):
        attempt_number = index + 1
        attempt_path = attempt_dir / f"attempt_{attempt_number:02d}.mp4"
        trace_path = attempt_dir / f"attempt_{attempt_number:02d}_trace.csv"
        attempt_start = np.asarray(audit_attempt["initial_xy_mm"], dtype=float) / 1000.0
        print(f"[3/6] render Baseline B attempt {attempt_number}/3", flush=True)
        attempt_result = run_method(
            "direct",
            attempt_path,
            initial_xy_m=attempt_start,
            posterior_seed=task["posterior_seed"] + index,
            direction_runtime=runtime,
            trace_output_path=trace_path,
            wrench_observation=WrenchObservationConfig(
                seed=task["sensor_noise_seed"] + index,
                delay_s=task["sensor_delay_s"],
            ),
            **common,
        )
        attempt_results.append(attempt_result)
        attempt_paths.append(attempt_path)
        trace_paths.append(trace_path)
    if results["proposed"].fingerprint_sha256 != attempt_results[0].fingerprint_sha256:
        raise RuntimeError("Proposed and Baseline B did not receive the same noisy fingerprint")
    reset_paths = []
    for index in range(2):
        reset_path = attempt_dir / f"reset_{index + 1:02d}_to_{index + 2:02d}.mp4"
        next_xy = np.asarray(audit_attempts[index + 1]["initial_xy_mm"], dtype=float) / 1000.0
        print(f"[4/6] render physical reset {index + 1}-> {index + 2}", flush=True)
        _render_retry_reset(
            reset_path,
            start_tip=_last_tip(trace_paths[index]),
            target_xy=next_xy,
            task=task,
            sensor_seed=task["sensor_noise_seed"] + index + 1,
            label=f"ATTEMPT {index + 1} FAILED | PHYSICAL RESET",
        )
        reset_paths.append(reset_path)
    aggregate = record["methods"]["baseline_b"]
    full_direct_path, full_direct_frames, reset_duration_s = _compose_full_direct_retry(
        attempt_paths, reset_paths, aggregate
    )
    stage_times: dict[str, float] = {}
    for attempt_result in attempt_results:
        for key, value in attempt_result.stage_time_s.items():
            stage_times[key] = stage_times.get(key, 0.0) + float(value)
    results["direct"] = replace(
        attempt_results[0],
        result=aggregate["result"],
        inner_wall_confirmed=aggregate["inner_wall_confirmed"],
        time_to_inner_wall_s=aggregate["time_to_inner_wall_s"],
        candidate_attempts=sum(item.candidate_attempts for item in attempt_results),
        total_active_time_s=aggregate["total_active_time_s"],
        actual_xy_path_mm=sum(item.actual_xy_path_mm for item in attempt_results),
        contact_xy_path_mm=aggregate["contact_xy_path_mm"],
        lateral_work_mj=sum(item.lateral_work_mj for item in attempt_results),
        orientation_work_mj=sum(item.orientation_work_mj for item in attempt_results),
        wear_proxy_mj=aggregate["wear_proxy_mj"],
        max_force_n=aggregate["max_force_n"],
        max_filtered_force_n=aggregate["max_filtered_force_n"],
        max_torque_mnm=aggregate["max_torque_mnm"],
        stage_time_s=stage_times,
        video_path=str(full_direct_path.relative_to(ROOT)),
        frame_count=full_direct_frames,
        direction_source="shared_three_seed_ensemble_direction_only_plus_shifted_full_reprobe",
    )
    paths["direct"] = full_direct_path
    obsolete = OUT / "03_baseline_b_first_attempt_shared_noise.mp4"
    if obsolete.exists():
        obsolete.unlink()
    print("[5/6] compose synchronized comparison and keyframe", flush=True)
    frames = _compose(paths, results, record, criterion)
    true_direction = math.degrees(math.atan2(-task["y_m"], -task["x_m"])) % 360.0
    control_direction = float(attempt_results[0].control_direction_deg or 0.0)
    direction_error = abs((control_direction - true_direction + 180.0) % 360.0 - 180.0)
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": criterion,
        "selected_audit_trial": task["pose_id"],
        "task": task,
        "shared_noise_model": WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ).to_dict(),
        "fairness": {
            "same_initial_pose": True,
            "same_contact_nuisance": True,
            "proposed_and_b_same_noisy_fingerprint": True,
            "same_frozen_p1v3": True,
            "baseline_b_video_scope": "all three attempts with physical unload/lift/shift/retouchdown reset motion",
            "baseline_b_audit_reset_motion_time_excluded": True,
            "baseline_b_rendered_reset_duration_s": reset_duration_s,
            "baseline_b_reset_scene_boundary": (
                "attempts are independent rollout episodes; each reset scene reconstructs the same final XY "
                "at released nominal surface height before dynamically rendering retract/shift/retouchdown"
            ),
        },
        "direction": {
            "true_simulation_direction_deg_offline_only": true_direction,
            "shared_control_direction_deg": control_direction,
            "error_deg_offline_only": direction_error,
        },
        "continuous_transport": {
            "policy": results["proposed"].transport_policy,
            "distance_mm": results["proposed"].transport_distance_mm,
            "direction_quality": results["proposed"].transport_direction_quality,
            "posterior_quantile": results["proposed"].transport_quantile,
            "reverse_probability": results["proposed"].transport_reverse_probability,
            "projected_quantiles_mm": results["proposed"].transport_projected_quantiles_mm,
            "residual_spiral_radius_mm": results["proposed"].spiral_max_radius_mm,
        },
        "baseline_a_terminal_diagnostic": _trace_terminal_diagnostic(outward_trace),
        "results": {key: asdict(value) for key, value in results.items()},
        "baseline_b_attempt_results": [asdict(value) for value in attempt_results],
        "baseline_b_attempt_videos": [str(path.relative_to(ROOT)) for path in attempt_paths],
        "baseline_b_reset_videos": [str(path.relative_to(ROOT)) for path in reset_paths],
        "videos": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "traces": {
            "proposed": str(proposed_trace.relative_to(ROOT)),
            "baseline_a": str(outward_trace.relative_to(ROOT)),
            "baseline_b_attempts": [str(path.relative_to(ROOT)) for path in trace_paths],
        },
        "comparison_video": str(COMPARISON_VIDEO.relative_to(ROOT)),
        "keyframe": str(COMPARISON_KEYFRAME.relative_to(ROOT)),
        "comparison_frames": frames,
        "claim_boundary": (
            "Audit-selected MuJoCo illustration under a frozen pre-hardware F/T observation prior. "
            "The noise magnitudes are defensible engineering values, not measurements from the future "
            "hardware sensor. Selection is explicitly conditional on a Baseline-B retry and must not "
            "be used as deployment-weighted success-rate evidence. Baseline-B reset motion is now rendered "
            "physically, but the frozen distribution audit still excludes reset travel/time in B's favor. "
            "Baseline A and Proposed now share the same sensor-trend capture handoff. All branches share "
            "the same post-capture tip-RCC energy/CAD guard; it does not alter search or retry selection. The video is a "
            "mechanism/recovery illustration rather than distribution-weighted evidence."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# M8 Pre-Hardware Noise Comparison\n\n"
        "This directory preserves the original clean comparison and adds a separate, audit-selected "
        "shared-noise retry illustration. Baseline B shows every required full probe plus physical "
        "unload/lift/shift/retouchdown resets. Read `manifest.json` before using the video in a claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
