from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
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


AUDIT = ROOT / "outputs/current/evaluations/m8_four_method_sensor_only_audit"
OUT = ROOT / "outputs/current/active_comparison/four_methods_clear_fingerprint"
COMPONENTS = OUT / ".components"
COMPARISON = OUT / "comparison_four_methods_clear_fingerprint_2x2.mp4"
KEYFRAME = OUT / "comparison_four_methods_clear_fingerprint_2x2_keyframe.png"
MANIFEST = OUT / "manifest.json"

# Full closed-loop, no-render preflight on the audit-filtered ranking. The
# preflight uses the exact render configuration below and stops after the first
# candidate for which Proposed/A/B all pass the shared CAD-bottom verifier.
FULL_CLOSED_LOOP_PREFLIGHT = {
    "sobol_020": {
        "proposed": "bottom_not_verified",
        "decision": "reject_after_proposed",
    },
    "sobol_026": {
        "proposed": "funnel_safe_abort",
        "decision": "reject_after_proposed",
    },
    "sobol_007": {
        "proposed": "cad_bottom_verified",
        "baseline_a": "bottom_not_verified",
        "baseline_b": "cad_bottom_verified",
        "decision": "reject_terminal_mismatch",
    },
    "sobol_004": {
        "proposed": "funnel_safe_abort",
        "decision": "reject_after_proposed",
    },
    "sobol_019": {
        "proposed": "cad_bottom_verified",
        "baseline_a": "cad_bottom_verified",
        "baseline_b": "cad_bottom_verified",
        "decision": "select",
    },
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _success(value: dict) -> bool:
    return bool(value.get("offline_capture_validated", value.get("inner_wall_confirmed")))


def _complete(result: RunResult) -> bool:
    return bool(result.bottom_reached and result.result == "cad_bottom_verified")


def _select_trial() -> tuple[dict, dict, list[dict]]:
    candidates: list[dict] = []
    records: dict[str, dict] = {}
    for path in sorted((AUDIT / "trials").glob("sobol_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        methods = record["methods"]
        proposed = methods["proposed_posterior"]
        baseline_a = methods["baseline_a"]
        baseline_b = methods["baseline_b"]
        random = methods["baseline_c_random"]
        if not (
            _success(proposed)
            and _success(baseline_a)
            and bool(baseline_b["first_attempt_success"])
            and not _success(random)
            and not proposed["hard_stop"]
            and not baseline_a["hard_stop"]
            and not baseline_b["hard_stop"]
            and not random["hard_stop"]
        ):
            continue
        item = {
            "pose_id": record["task"]["pose_id"],
            "proposed_transport_mm": float(proposed["transport_distance_mm"]),
            "direction_error_deg": float(
                record["diagnostics"]["control_direction_error_deg"]
            ),
            "direct_ray_margin_mm": float(
                record["diagnostics"]["direct_ray_geometric_margin_mm"]
            ),
        }
        candidates.append(item)
        records[item["pose_id"]] = record
    if not candidates:
        raise RuntimeError("no frozen trial satisfies the clear-fingerprint rule")
    candidates.sort(
        key=lambda item: (
            -item["proposed_transport_mm"],
            item["direction_error_deg"],
            item["pose_id"],
        )
    )
    selected = next(
        item
        for item in candidates
        if FULL_CLOSED_LOOP_PREFLIGHT.get(item["pose_id"], {}).get("decision")
        == "select"
    )
    return records[selected["pose_id"]], selected, candidates


def _active_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        count = reader.count_frames()
        active = max(1, count - int(round(SUMMARY_TIME_S * FPS)))
        return [reader.get_data(index) for index in range(active)]
    finally:
        reader.close()


def _short_result(result: RunResult) -> str:
    if _complete(result):
        return "complete"
    if result.offline_capture_validated or result.inner_wall_confirmed:
        return "capture"
    if result.hard_stop:
        return "hard stop"
    return "no capture"


def _paper_panel(frame: np.ndarray, key: str) -> Image.Image:
    panel = Image.fromarray(frame).resize((640, 360), Image.Resampling.LANCZOS)
    if key != "random":
        return panel
    draw = ImageDraw.Draw(panel)
    draw.rectangle((4, 4, 414, 28), fill=(6, 8, 11))
    draw.text(
        (10, 8),
        "BASELINE C: seeded area-uniform random waypoints",
        fill=(255, 218, 165),
        font=_font(10, bold=True),
    )
    draw.rectangle((4, 294, 438, 316), fill=(6, 8, 11))
    draw.text(
        (10, 298),
        "baseline C: frozen random waypoints",
        fill=(225, 232, 240),
        font=_font(9),
    )
    return panel


def _summary(
    frame: np.ndarray,
    results: dict[str, RunResult],
    task: dict,
    selection: dict,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (42, 35, 1238, 765),
        radius=8,
        fill=(3, 7, 12, 247),
        outline=(225, 235, 244, 235),
        width=2,
    )
    draw.text(
        (70, 58),
        "CLEAR FINGERPRINT CASE | FROZEN SAME-START COMPLEMENTARY EXAMPLE",
        fill=(245, 248, 252, 255),
        font=_font(19, bold=True),
    )
    draw.text(
        (70, 94),
        (
            f"{task['pose_id']} | r={1000.0 * task['radius_m']:.2f} mm | "
            f"move={selection['proposed_transport_mm']:.2f} mm | "
            f"direction error={selection['direction_error_deg']:.2f} deg"
        ),
        fill=(183, 202, 220, 255),
        font=_font(12),
    )

    methods = (
        ("PROPOSED", "proposed", (90, 225, 255), "posterior move + residual spiral"),
        ("A OUTWARD", "outward", (255, 205, 105), "ordinary outward coverage"),
        ("B DIRECT", "direct", (150, 235, 155), "shared direction; first attempt"),
        ("C RANDOM", "random", (255, 145, 105), "frozen random waypoints"),
    )
    columns = (
        ("method", 70),
        ("result", 250),
        ("capture", 370),
        ("complete", 480),
        ("active", 590),
        ("XY", 695),
        ("contact", 790),
        ("work", 900),
        ("max F", 1000),
        ("mechanism", 1090),
    )
    for label, x in columns:
        draw.text((x, 142), label, fill=(150, 173, 195, 255), font=_font(11, bold=True))
    y = 180
    for label, key, color, mechanism in methods:
        result = results[key]
        values = (
            label,
            _short_result(result),
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
            draw.text((x, y), str(value)[:19], fill=fill, font=_font(11, bold=header == "method"))
        draw.line((70, y + 31, 1210, y + 31), fill=(100, 120, 140, 75), width=1)
        y += 66

    notes = (
        "Selection was frozen: P/A capture, B first-attempt capture, C failure, zero hard stops; rank by largest P move.",
        "Proposed and B receive a byte-identical noisy 15 deg fingerprint and the same frozen direction ensemble.",
        f"The shared direction error is {selection['direction_error_deg']:.2f} deg; B's direct ray has {selection['direct_ray_margin_mm']:.3f} mm capture margin.",
        "All methods share the sensor-only handoff, force/torque limits, tip-RCC energy guard and CAD-depth terminal verifier.",
        "This complements the retry case. It illustrates a favorable mechanism regime; the frozen 36-pose audit supports claims.",
    )
    draw.text((70, 470), "claim boundary", fill=(255, 225, 150, 255), font=_font(13, bold=True))
    y = 507
    for note in notes:
        draw.text((82, y), note, fill=(215, 227, 238, 255), font=_font(12))
        y += 42
    return np.asarray(image)


def _compose(
    paths: dict[str, Path],
    results: dict[str, RunResult],
    task: dict,
    selection: dict,
) -> int:
    frames = {key: _active_frames(path) for key, path in paths.items()}
    total_active = max(len(items) for items in frames.values())
    labels = (
        ("PROPOSED | clear fingerprint -> longer bounded shortcut", "proposed", (90, 225, 255), (0, 0)),
        ("BASELINE A | ordinary outward coverage", "outward", (255, 205, 105), (640, 0)),
        ("BASELINE B | same direction -> first-attempt direct success", "direct", (150, 235, 155), (0, 400)),
        ("BASELINE C | frozen random route -> no capture", "random", (255, 145, 105), (640, 400)),
    )
    keyframe_index = min(int(round(7.0 * FPS)), total_active - 1)
    writer = imageio.get_writer(
        COMPARISON, fps=FPS, codec="libx264", quality=8, macro_block_size=1
    )
    last: np.ndarray | None = None
    try:
        for index in range(total_active):
            canvas = Image.new("RGB", (1280, 800), (8, 10, 13))
            draw = ImageDraw.Draw(canvas)
            for label, key, color, (x, y) in labels:
                source = frames[key][min(index, len(frames[key]) - 1)]
                panel = _paper_panel(source, key)
                canvas.paste(panel, (x, y + 40))
                draw.text((x + 14, y + 10), label, fill=color, font=_font(12, bold=True))
            last = np.asarray(canvas)
            writer.append_data(last)
            if index == keyframe_index:
                imageio.imwrite(KEYFRAME, last)
        if last is None:
            raise RuntimeError("component videos yielded no active frame")
        final = _summary(last, results, task, selection)
        for _ in range(3 * FPS):
            writer.append_data(final)
            total_active += 1
    finally:
        writer.close()
    return total_active


def main() -> None:
    source, selection, candidates = _select_trial()
    task = source["task"]
    if task["pose_id"] != "sobol_019":
        raise RuntimeError(
            f"frozen selection changed from preflighted sobol_019 to {task['pose_id']}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    if COMPONENTS.exists():
        shutil.rmtree(COMPONENTS)
    COMPONENTS.mkdir(parents=True)
    paths = {
        "proposed": COMPONENTS / "01_proposed.mp4",
        "outward": COMPONENTS / "02_baseline_a.mp4",
        "direct": COMPONENTS / "03_baseline_b.mp4",
        "random": COMPONENTS / "04_baseline_c_random.mp4",
    }
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    initial_xy = np.array([task["x_m"], task["y_m"]], dtype=float)
    common = {
        "initial_xy_m": initial_xy,
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
        "observation_label": "same frozen pre-hardware F/T bias + noise + coupling + delay",
    }

    results: dict[str, RunResult] = {}
    jobs = (
        ("proposed", "learned", {"direction_runtime": runtime, "max_transport_m": None}),
        ("outward", "baseline", {"fast_forward_after_active_s": 12.0, "fast_forward_speed": 6.0}),
        ("direct", "direct", {"direction_runtime": runtime}),
        (
            "random",
            "random",
            {
                "random_search_seed": task["posterior_seed"] + 700000,
                "fast_forward_after_active_s": 12.0,
                "fast_forward_speed": 6.0,
            },
        ),
    )
    try:
        for index, (key, method, extra) in enumerate(jobs, start=1):
            print(f"[{index}/5] render {key}", flush=True)
            started = time.perf_counter()
            results[key] = run_method(
                method,
                paths[key],
                wrench_observation=WrenchObservationConfig(
                    seed=task["sensor_noise_seed"], delay_s=task["sensor_delay_s"]
                ),
                **common,
                **extra,
            )
            print(
                f"[{index}/5] {key}: {results[key].result} in {time.perf_counter() - started:.1f}s",
                flush=True,
            )

        if results["proposed"].fingerprint_sha256 != results["direct"].fingerprint_sha256:
            raise RuntimeError("Proposed and Baseline B did not receive the same fingerprint")
        if not all(_complete(results[key]) for key in ("proposed", "outward", "direct")):
            raise RuntimeError("P/A/B no longer reach the shared CAD-bottom verifier")
        if results["random"].offline_capture_validated or results["random"].inner_wall_confirmed:
            raise RuntimeError("frozen random baseline unexpectedly captured in the rerender")
        if any(result.hard_stop for result in results.values()):
            raise RuntimeError("selected clear-fingerprint render contains a hard stop")
        if not math.isclose(
            float(results["proposed"].transport_distance_mm or 0.0),
            selection["proposed_transport_mm"],
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("rerendered Proposed transport differs from frozen audit")

        print("[5/5] compose synchronized 2x2 and dynamic keyframe", flush=True)
        frames = _compose(paths, results, task, selection)
    finally:
        shutil.rmtree(COMPONENTS, ignore_errors=True)

    persisted_results = {}
    for key, result in results.items():
        value = asdict(result)
        value["component_video_retained"] = False
        value["video_path"] = str(COMPARISON.relative_to(ROOT))
        persisted_results[key] = value
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "From the frozen 36-pose random-baseline audit, require Proposed and A capture, "
            "B first-attempt capture, C-random failure, and zero hard stops; rank by descending "
            "Proposed transport, then ascending direction error, then pose_id. In that frozen "
            "order, run the same no-render full closed-loop preflight and select the first case "
            "for which Proposed, A and B all pass the shared CAD-bottom verifier."
        ),
        "candidate_count": len(candidates),
        "candidate_ranking": candidates,
        "full_closed_loop_preflight": FULL_CLOSED_LOOP_PREFLIGHT,
        "selected_audit_trial": task["pose_id"],
        "selection_diagnostics": selection,
        "task": task,
        "paper_method_order": ["proposed", "baseline_a", "baseline_b", "baseline_c_random"],
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_prior": True,
            "proposed_and_b_identical_first_fingerprint": True,
            "baseline_b_first_attempt_only_because_frozen_audit_first_attempt_succeeds": True,
            "baseline_c_seed_fixed_before_rollout": task["posterior_seed"] + 700000,
            "online_true_center_or_contact_tags": False,
            "shared_capture_guard_and_terminal_verifier": True,
            "component_videos_removed_after_composition": True,
        },
        "results": persisted_results,
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "frames": frames,
        "claim_boundary": (
            "Complementary frozen mechanism illustration for a clear-fingerprint regime. "
            "It is not an additional independent benchmark and cannot replace the paired 36-pose audit."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value.result for key, value in results.items()}, indent=2))
    print(COMPARISON.relative_to(ROOT))


if __name__ == "__main__":
    main()
