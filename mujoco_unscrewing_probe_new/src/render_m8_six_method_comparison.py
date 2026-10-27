from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult, run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
)
from src.wrench_observation import WrenchObservationConfig


FOUR_SOURCE = ROOT / "outputs/current/diagnostics/historical_comparisons/four_methods/manifest.json"
RETRY_SOURCE = ROOT / "outputs/current/diagnostics/historical_comparisons/robustness_noise/manifest.json"
OUT = ROOT / "outputs/current/diagnostics/historical_comparisons/six_methods"
HYBRID_VIDEO = OUT / "01_proposed_plus_joint_funnel.mp4"
ENTROPY_VIDEO = OUT / "05_baseline_c_cad_scaled_entropy_grid.mp4"
RANDOM_VIDEO = OUT / "06_baseline_d_seeded_random_route.mp4"
COMPARISON = OUT / "comparison_six_methods_3x2.mp4"
KEYFRAME = OUT / "comparison_six_methods_3x2_keyframe.png"
MANIFEST = OUT / "manifest.json"
WIDTH = 1280
HEIGHT = 1200


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _completed(result: RunResult) -> bool:
    return bool(result.bottom_reached and result.result == "cad_bottom_verified")


def _aggregate_direct(source: dict, retry_source: dict) -> RunResult:
    result = RunResult(**source["results"]["direct"])
    attempts = retry_source["baseline_b_attempt_results"]
    terminal = attempts[-1]
    return replace(
        result,
        result=terminal["result"],
        bottom_reached=bool(terminal["bottom_reached"]),
        time_to_bottom_s=float(sum(item["total_active_time_s"] for item in attempts)),
        total_active_time_s=float(sum(item["total_active_time_s"] for item in attempts)),
        bottom_depth_error_mm=terminal.get("bottom_depth_error_mm"),
        terminal_verification_mode=terminal.get("terminal_verification_mode"),
    )


def _summary(
    frame: np.ndarray,
    results: dict[str, RunResult],
    task: dict,
    retry_attempts: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((28, 28, WIDTH - 28, HEIGHT - 28), fill=(3, 7, 12, 248), outline=(225, 235, 244, 235), width=2)
    draw.text(
        (58, 52),
        "SIX METHODS | SAME START, CONTACT NUISANCE AND PRE-HARDWARE SENSOR PRIOR",
        fill=(245, 248, 252, 255),
        font=_font(20, bold=True),
    )
    draw.text(
        (58, 88),
        f"selected {task['pose_id']} | r={1000.0 * task['radius_m']:.2f} mm | conditional mechanism video, not a success-rate estimate",
        fill=(185, 202, 218, 255),
        font=_font(12),
    )

    columns = (
        ("method", 58),
        ("result", 292),
        ("capture", 405),
        ("complete", 515),
        ("active", 630),
        ("XY", 735),
        ("contact", 830),
        ("work", 940),
        ("max F", 1040),
        ("mechanism", 1130),
    )
    for label, x in columns:
        draw.text((x, 132), label, fill=(150, 173, 195, 255), font=_font(11, bold=True))

    rows = (
        ("PROPOSED+", "hybrid", (100, 245, 235), "shortcut + task grid"),
        ("PROPOSED", "proposed", (90, 225, 255), "cone + spiral"),
        ("A OUTWARD", "outward", (255, 205, 105), "coverage"),
        ("B DIRECT", "direct", (150, 235, 155), f"{retry_attempts} attempts"),
        (
            "C ENTROPY",
            "entropy",
            (225, 155, 240),
            f"{results['entropy'].entropy_grid_probe_count} contacts",
        ),
        ("D RANDOM", "random", (255, 145, 105), "43 waypoints"),
    )
    y = 170
    for label, key, color, mechanism in rows:
        result = results[key]
        values = (
            label,
            "complete" if _completed(result) else result.result,
            "n/a" if result.time_to_inner_wall_s is None else f"{result.time_to_inner_wall_s:.2f}s",
            "n/a" if result.time_to_bottom_s is None else f"{result.time_to_bottom_s:.2f}s",
            f"{result.total_active_time_s:.2f}s",
            f"{result.actual_xy_path_mm:.1f}mm",
            f"{result.contact_xy_path_mm:.1f}mm",
            f"{result.wear_proxy_mj:.1f}mJ",
            f"{result.max_force_n:.1f}N",
            mechanism,
        )
        for (header, x), value in zip(columns, values):
            fill = (*color, 255) if header == "method" else (232, 239, 246, 255)
            draw.text((x, y), str(value)[:18], fill=fill, font=_font(11, bold=header == "method"))
        draw.line((58, y + 31, WIDTH - 58, y + 31), fill=(100, 120, 140, 75), width=1)
        y += 68

    notes = [
        "Proposed+ and Proposed share the byte-identical conical fingerprint and frozen P1v3 output.",
        "Proposed+ first takes the bounded posterior shortcut, then maximizes capture probability plus information gain at low force.",
        "A missed segment is a negative observation and surprising F/T is downweighted. Local spiral is disabled; full outward is last-resort only.",
        "Depth/F/T capture also needs one-sided consistency with the conical top-1 direction; true center/contact tags remain offline only.",
        "C has no conical learner or spiral. D has no conical learner, CAD likelihood, entropy objective or spiral; its seed is frozen before rollout.",
        "All methods use sensor-only capture, identical hard limits, tip-RCC/energy guard and the same dual-channel terminal verifier.",
        "D is fastest in this selected lucky path; historical 36-pose random coverage is 19/36, so this clip cannot establish method ranking.",
        "Fresh frozen 24-pose strict capture: Proposed+/Proposed/A/B/C = 21/23/23/22/23; all have zero hard stops.",
        "Proposed+ reduces some path/force exposure but is slower than Proposed and less contact-efficient than C; it remains experimental.",
    ]
    y = 630
    draw.text((58, y), "claim boundary", fill=(255, 225, 150, 255), font=_font(13, bold=True))
    y += 36
    for note in notes:
        draw.text((72, y), note, fill=(215, 227, 238, 255), font=_font(12))
        y += 42
    return np.asarray(image)


def _compose(
    paths: dict[str, Path],
    results: dict[str, RunResult],
    task: dict,
    retry_attempts: int,
) -> int:
    labels = (
        ("PROPOSED+ | shared belief + task-aware posterior grid", "hybrid", (100, 245, 235), (0, 0)),
        ("PROPOSED | conical posterior + residual outward", "proposed", (90, 225, 255), (640, 0)),
        ("BASELINE A | ordinary outward coverage", "outward", (255, 205, 105), (0, 400)),
        ("BASELINE B | shared direction + full shifted retries", "direct", (150, 235, 155), (640, 400)),
        ("BASELINE C | maximum-entropy grid contacts", "entropy", (225, 155, 240), (0, 800)),
        ("BASELINE D | seeded random route", "random", (255, 145, 105), (640, 800)),
    )
    readers = {key: imageio.get_reader(path) for key, path in paths.items()}
    active_counts = {}
    last_frames: dict[str, np.ndarray] = {}
    try:
        for _, key, _, _ in labels:
            count = readers[key].count_frames()
            summary_s = 3.0 if key == "direct" else SUMMARY_TIME_S
            active_counts[key] = max(1, count - int(round(summary_s * FPS)))
        total = max(active_counts.values())
        writer = imageio.get_writer(COMPARISON, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        last_canvas: np.ndarray | None = None
        try:
            for index in range(total):
                canvas = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 13))
                draw = ImageDraw.Draw(canvas)
                for label, key, color, (x, y) in labels:
                    if index < active_counts[key]:
                        last_frames[key] = readers[key].get_data(index)
                    frame = last_frames[key]
                    panel = Image.fromarray(frame).resize((640, 360), Image.Resampling.LANCZOS)
                    canvas.paste(panel, (x, y + 40))
                    draw.text((x + 14, y + 10), label, fill=color, font=_font(12, bold=True))
                last_canvas = np.asarray(canvas)
                writer.append_data(last_canvas)
            assert last_canvas is not None
            final = _summary(last_canvas, results, task, retry_attempts)
            imageio.imwrite(KEYFRAME, final)
            for _ in range(3 * FPS):
                writer.append_data(final)
                total += 1
        finally:
            writer.close()
    finally:
        for reader in readers.values():
            reader.close()
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(FOUR_SOURCE.read_text(encoding="utf-8"))
    retry_source = json.loads(RETRY_SOURCE.read_text(encoding="utf-8"))
    task = source["task"]
    initial_xy = np.array([task["x_m"], task["y_m"]], dtype=float)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    common = {
        "visual_sigma_m": VISUAL_SIGMA_M,
        "visual_bound_m": VISUAL_BOUND_M,
        "posterior_coverage": POSTERIOR_COVERAGE,
        "posterior_seed": task["posterior_seed"],
        "baseline_phase_rad": task["baseline_phase_rad"],
        "baseline_max_radius_m": BASELINE_MAX_RADIUS_M,
        "target_down_force_n": task["target_down_force_n"],
        "friction_slide": task["friction_slide"],
        "contact_timeconst_s": task["contact_timeconst_s"],
        "active_time_budget_s": 130.0,
        "continue_to_bottom": True,
        "observation_label": "shared F/T bias + noise + coupling + delay",
    }
    print("[1/4] render Proposed+ joint posterior funnel", flush=True)
    started = time.perf_counter()
    hybrid = run_method(
        "hybrid",
        HYBRID_VIDEO,
        initial_xy_m=initial_xy,
        direction_runtime=runtime,
        max_transport_m=None,
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
        **common,
    )
    print(f"[1/4] Proposed+ complete in {time.perf_counter() - started:.1f}s", flush=True)

    print("[2/4] render Baseline C with CAD-scaled entropy grid", flush=True)
    started = time.perf_counter()
    entropy = run_method(
        "entropy_grid",
        ENTROPY_VIDEO,
        initial_xy_m=initial_xy,
        direction_runtime=None,
        max_transport_m=None,
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
        **common,
    )
    print(f"[2/4] Baseline C complete in {time.perf_counter() - started:.1f}s", flush=True)

    random_seed = int(task["posterior_seed"] + 700000)
    print(f"[3/4] render Baseline D seeded random route | seed={random_seed}", flush=True)
    started = time.perf_counter()
    random = run_method(
        "random",
        RANDOM_VIDEO,
        initial_xy_m=initial_xy,
        random_search_seed=random_seed,
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
        **common,
    )
    print(f"[3/4] Baseline D complete in {time.perf_counter() - started:.1f}s", flush=True)

    results = {
        "hybrid": hybrid,
        "proposed": RunResult(**source["results"]["proposed"]),
        "outward": RunResult(**source["results"]["outward"]),
        "direct": _aggregate_direct(source, retry_source),
        "entropy": entropy,
        "random": random,
    }
    if hybrid.fingerprint_sha256 != results["proposed"].fingerprint_sha256:
        raise RuntimeError("Proposed+ and Proposed did not receive the same fingerprint")
    if not all(np.allclose(result.initial_xy_mm, results["proposed"].initial_xy_mm) for key, result in results.items() if key != "direct"):
        raise RuntimeError("six-method initial pose mismatch")
    if not all(_completed(result) for result in results.values()):
        raise RuntimeError("selected six-method video no longer has complete geometric insertion")

    paths = {
        "hybrid": HYBRID_VIDEO,
        "proposed": ROOT / source["videos"]["proposed"],
        "outward": ROOT / source["videos"]["outward"],
        "direct": ROOT / source["videos"]["direct"],
        "entropy": ENTROPY_VIDEO,
        "random": RANDOM_VIDEO,
    }
    print("[4/4] compose synchronized 3x2 and keyframe", flush=True)
    frames = _compose(paths, results, task, len(retry_source["baseline_b_attempt_results"]))
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": source["selection_rule"],
        "selected_audit_trial": task["pose_id"],
        "task": task,
        "method_order": ["hybrid", "proposed", "outward", "direct", "entropy", "random"],
        "proposed_plus_execution_order": [
            "conical_fingerprint",
            "bounded_posterior_shortcut",
            "capture_probability_plus_information_grid_nodes",
            "direction_consistent_sensor_capture_or_full_visual_support_last_resort",
            "shared_post_capture_funnel",
        ],
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_prior": True,
            "proposed_plus_and_proposed_identical_conical_fingerprint": True,
            "baseline_b_all_attempts_visible": True,
            "baseline_d_seed_fixed_before_rollout": random_seed,
            "online_true_center_or_contact_tags": False,
            "shared_capture_guard_and_terminal_verifier": True,
            "baseline_b_reset_motion_excluded_from_numeric_active_time": True,
        },
        "results": {key: asdict(value) for key, value in results.items()},
        "videos": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "frames": frames,
        "claim_boundary": (
            "Conditional same-start mechanism video. Fresh frozen 24-pose strict capture is "
            "Proposed+/Proposed/A/B/C=21/23/23/22/23 with zero hard stops; Proposed+ remains a tradeoff, "
            "not the frozen main method. "
            "Baseline D is allowed to win on this frozen lucky seed; historical random-C statistics "
            "remain appendix evidence and cannot establish the new six-method ranking."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value.result for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
