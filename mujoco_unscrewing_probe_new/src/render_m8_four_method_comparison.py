from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult, run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
)
from src.wrench_observation import WrenchObservationConfig


SOURCE = ROOT / "outputs/current/diagnostics/historical_comparisons/robustness_noise/manifest.json"
OUT = ROOT / "outputs/current/diagnostics/historical_comparisons/four_methods"
ENTROPY_VIDEO = OUT / "04_baseline_c_entropy_grid_direct.mp4"
COMPARISON = OUT / "comparison_four_methods_2x2.mp4"
KEYFRAME = OUT / "comparison_four_methods_2x2_keyframe.png"
MANIFEST = OUT / "manifest.json"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _active_frames(path: Path, key: str) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        count = reader.count_frames()
        summary_s = 3.0 if key == "direct" else SUMMARY_TIME_S
        active = max(1, count - int(round(summary_s * FPS)))
        return [reader.get_data(index) for index in range(active)]
    finally:
        reader.close()


def _short_result(result: RunResult) -> str:
    if result.bottom_reached:
        return "complete"
    capture = (
        result.inner_wall_confirmed
        if result.offline_capture_validated is None
        else result.offline_capture_validated
    )
    if capture:
        return "capture"
    if result.hard_stop:
        return "hard stop"
    return "no capture"


def _summary(frame: np.ndarray, results: dict[str, RunResult], source: dict) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (80, 52, 1200, 735),
        radius=8,
        fill=(3, 7, 12, 244),
        outline=(225, 235, 244, 230),
        width=2,
    )
    draw.text(
        (112, 76),
        "FOUR METHODS | SAME START AND PRE-HARDWARE SENSOR PRIOR",
        fill=(245, 248, 252, 255),
        font=_font(21, bold=True),
    )
    labels = (
        ("PROPOSED", "proposed", (90, 225, 255)),
        ("A OUTWARD", "outward", (255, 205, 105)),
        ("B DIRECT RETRY", "direct", (150, 235, 155)),
        ("C MAX ENTROPY", "entropy", (225, 155, 240)),
    )
    columns = (410, 620, 825, 1040)
    for x, (label, _, color) in zip(columns, labels):
        draw.text((x - 72, 128), label, fill=(*color, 255), font=_font(14, bold=True))
    terminal_complete = {
        "proposed": results["proposed"].bottom_reached,
        "outward": results["outward"].bottom_reached,
        "direct": bool(
            source["baseline_b_attempt_results"]
            and source["baseline_b_attempt_results"][-1].get("bottom_reached")
        ),
        "entropy": results["entropy"].bottom_reached,
    }
    rows = (
        ("result", lambda r: _short_result(r)),
        ("capture time", lambda r: "n/a" if r.time_to_inner_wall_s is None else f"{r.time_to_inner_wall_s:.2f}s"),
        ("active time", lambda r: f"{r.total_active_time_s:.2f}s"),
        ("XY path", lambda r: f"{r.actual_xy_path_mm:.1f}mm"),
        ("work proxy", lambda r: f"{r.wear_proxy_mj:.1f}mJ"),
        ("max force", lambda r: f"{r.max_force_n:.1f}N"),
    )
    y = 177
    for title, getter in rows:
        draw.text((118, y), title, fill=(180, 194, 208, 255), font=_font(14))
        for x, (_, key, _) in zip(columns, labels):
            value = (
                "complete"
                if title == "result" and terminal_complete[key]
                else getter(results[key])
            )
            draw.text((x - 42, y), value[:16], fill=(235, 241, 247, 255), font=_font(13))
        y += 42
    draw.text((118, 438), "method-specific", fill=(180, 194, 208, 255), font=_font(14))
    entropy_probe_capture = bool(
        results["entropy"].entropy_grid_observations
        and results["entropy"].entropy_grid_observations[-1].get("observation")
        == "direct_capture"
    )
    method_notes = (
        f"move {results['proposed'].transport_distance_mm:.2f} mm + local spiral",
        "no probe; full outward coverage",
        f"success on attempt {len(source['baseline_b_attempt_results'])}",
        (
            f"{results['entropy'].entropy_grid_probe_count} probes; probe capture; no spiral"
            if entropy_probe_capture
            else f"{results['entropy'].entropy_grid_probe_count} probes; guarded mean touchdown; no spiral"
        ),
    )
    for x, note in zip(columns, method_notes):
        draw.multiline_text((x - 72, 438), note, fill=(225, 232, 240, 255), font=_font(11), spacing=3)
    task = source["task"]
    notes = [
        f"Audit-selected {task['pose_id']}: r={1000.0 * task['radius_m']:.2f} mm; B first ray misses by physics and succeeds after full re-probes.",
        "P and B share the identical first noisy 15 deg fingerprint and frozen P1v3 output; no direction error is injected for display.",
        "C uses guarded touchdown -> max expected entropy reduction; only if probes miss does it lift-shift-touch down at the posterior mean; no conical or spiral.",
        "All branches share the sensor-only capture handoff and post-capture tip-RCC energy/CAD guard; MuJoCo tags are offline scoring only.",
        "Terminal completion is method-independent: force-seat dwell OR stable low-wrench CAD-depth dwell; no true center/contact tags.",
        "This conditional video explains mechanisms. Distribution-level claims must use the paired multi-pose audit, not this selected case.",
    ]
    y = 535
    for note in notes:
        draw.text((118, y), note, fill=(218, 228, 238, 255), font=_font(12))
        y += 34
    return np.asarray(image)


def _compose(paths: dict[str, Path], results: dict[str, RunResult], source: dict) -> int:
    frames = {key: _active_frames(path, key) for key, path in paths.items()}
    total = max(len(item) for item in frames.values())
    labels = (
        ("PROPOSED | one probe + joint posterior + residual outward", "proposed", (90, 225, 255), (0, 0)),
        ("BASELINE A | ordinary outward coverage", "outward", (255, 205, 105), (640, 0)),
        ("BASELINE B | shared direction + full shifted retries", "direct", (150, 235, 155), (0, 400)),
        ("BASELINE C | max-entropy grid probes + direct commit", "entropy", (225, 155, 240), (640, 400)),
    )
    writer = imageio.get_writer(COMPARISON, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    last: np.ndarray | None = None
    try:
        for index in range(total):
            canvas = Image.new("RGB", (1280, 800), (8, 10, 13))
            draw = ImageDraw.Draw(canvas)
            for label, key, color, (x, y) in labels:
                source_frame = frames[key][min(index, len(frames[key]) - 1)]
                panel = Image.fromarray(source_frame).resize((640, 360), Image.Resampling.LANCZOS)
                canvas.paste(panel, (x, y + 40))
                draw.text((x + 14, y + 10), label, fill=color, font=_font(13, bold=True))
            last = np.asarray(canvas)
            writer.append_data(last)
        assert last is not None
        final = _summary(last, results, source)
        imageio.imwrite(KEYFRAME, final)
        for _ in range(3 * FPS):
            writer.append_data(final)
            total += 1
    finally:
        writer.close()
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    task = source["task"]
    paths = {
        "proposed": ROOT / source["videos"]["proposed"],
        "outward": ROOT / source["videos"]["outward"],
        "direct": ROOT / source["videos"]["direct"],
        "entropy": ENTROPY_VIDEO,
    }
    results = {
        key: RunResult(**source["results"][key])
        for key in ("proposed", "outward", "direct")
    }
    print("[1/2] render Baseline C entropy-grid on the audit-selected shared task", flush=True)
    started = time.perf_counter()
    results["entropy"] = run_method(
        "entropy_grid",
        ENTROPY_VIDEO,
        initial_xy_m=np.array([task["x_m"], task["y_m"]], dtype=float),
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
        observation_label="shared F/T bias + noise + coupling + delay",
        wrench_observation=WrenchObservationConfig(
            seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
        ),
    )
    print(f"[1/2] C complete in {time.perf_counter() - started:.1f}s", flush=True)
    print("[2/2] compose synchronized 2x2 and keyframe", flush=True)
    frames = _compose(paths, results, source)
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": source["selection_rule"],
        "selected_audit_trial": task["pose_id"],
        "task": task,
        "fairness": {
            "same_initial_pose_and_contact_nuisance_all_methods": True,
            "proposed_and_b_identical_first_noisy_fingerprint": True,
            "baseline_b_all_attempts_and_physical_resets_visible": True,
            "baseline_c_online_true_center_or_contact_tags": False,
            "baseline_c_spiral_fallback": False,
            "all_interaction_motion_time_path_and_work_counted_for_c": True,
            "shared_sensor_capture_handoff_and_post_capture_guard": True,
            "method_independent_terminal_verifier": True,
            "terminal_completion_channels": [
                "bottom_force_dwell",
                "cad_depth_low_wrench_dwell",
            ],
        },
        "results": {key: asdict(value) for key, value in results.items()},
        "videos": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "frames": frames,
        "claim_boundary": (
            "Audit-selected mechanism illustration, conditional on Baseline B requiring retry. "
            "It must not be used as a success-rate estimate. Baseline C is a Chen-inspired adaptation "
            "to wrist F/T edge-load observations, not an exact reproduction of the paper's full contact-footprint funnel."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value.result for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
