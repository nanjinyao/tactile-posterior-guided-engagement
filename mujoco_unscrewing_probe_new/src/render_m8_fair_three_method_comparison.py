from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import (
    BASELINE_MAX_RADIUS_M,
    FPS,
    INITIAL_XY_M,
    ROOT,
    SUMMARY_TIME_S,
    RunResult,
    run_method,
)


MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SOURCE_RESULT = ROOT / "outputs/current/reports/m8_outer_partial_cad_fused_full_closed_loop_comparison.json"
ACTIVE = ROOT / "outputs/current/active_comparison"

PROPOSED_SOURCE = ROOT / "outputs/current/media/videos/comparisons/m8_outer_partial_p1v3_cad_fused_full_to_bottom.mp4"
OUTWARD_SOURCE = ROOT / "outputs/current/media/videos/comparisons/m8_outer_partial_source_outward_full_to_bottom.mp4"
DIRECT_SOURCE = ROOT / "outputs/current/media/videos/baselines/m8_shared_probe_direction_direct_to_bottom.mp4"

PROPOSED_LINK = ACTIVE / "01_proposed_learning_vision_cad.mp4"
OUTWARD_LINK = ACTIVE / "02_baseline_outward_spiral.mp4"
DIRECT_LINK = ACTIVE / "03_baseline_shared_probe_direct_retry.mp4"
COMPARISON_VIDEO = ACTIVE / "comparison_three_methods.mp4"
DIRECT_KEYFRAME = ACTIVE / "03_baseline_shared_probe_direct_keyframe.png"
COMPARISON_KEYFRAME = ACTIVE / "comparison_three_methods_keyframe.png"
ROBUSTNESS_DIR = ACTIVE / "robustness_noise"
MANIFEST = ACTIVE / "manifest.json"
INDEX = ACTIVE / "README.md"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _load_run_result(value: dict) -> RunResult:
    allowed = {item.name for item in fields(RunResult)}
    return RunResult(**{key: item for key, item in value.items() if key in allowed})


def _replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, start=link.parent))


def _active_frames(path: Path, result: RunResult) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        count = max(1, result.frame_count - max(1, int(round(SUMMARY_TIME_S * FPS))))
        return [reader.get_data(index) for index in range(min(count, reader.count_frames()))]
    finally:
        reader.close()


def _summary_frame(frame: np.ndarray, results: dict[str, RunResult]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (115, 58, 1325, 385),
        radius=8,
        fill=(3, 7, 12, 240),
        outline=(220, 232, 242, 220),
        width=2,
    )
    draw.text(
        (145, 79),
        "FAIR SAME-START COMPARISON | SHARED DIRECTION EVIDENCE",
        fill=(245, 248, 252, 255),
        font=_font(20, bold=True),
    )
    labels = (
        ("PROPOSED", "proposed", (90, 225, 255)),
        ("BASELINE A | OUTWARD", "outward", (255, 205, 105)),
        ("BASELINE B | DIRECT", "direct", (150, 235, 155)),
    )
    columns = (310, 700, 1090)
    for x, (label, _, color) in zip(columns, labels):
        draw.text((x - 90, 119), label, fill=(*color, 255), font=_font(17, bold=True))
    rows = (
        ("capture", "time_to_inner_wall_s", "s"),
        ("CAD bottom", "time_to_bottom_s", "s"),
        ("XY path", "actual_xy_path_mm", "mm"),
        ("work proxy", "wear_proxy_mj", "mJ"),
        ("max force", "max_force_n", "N"),
    )
    y = 160
    for row, field, suffix in rows:
        draw.text((145, y), row, fill=(180, 194, 208, 255), font=_font(15))
        for x, (_, key, _) in zip(columns, labels):
            value = getattr(results[key], field)
            text = "n/a" if value is None else f"{value:.2f}{suffix}"
            draw.text((x, y), text, fill=(235, 241, 247, 255), font=_font(15))
        y += 34
    draw.text(
        (145, 341),
        "Proposed and Baseline B share the same conical F/T trace and frozen P1v3 direction. Direct wins this case.",
        fill=(210, 222, 234, 255),
        font=_font(13),
    )
    return np.asarray(image)


def compose(results: dict[str, RunResult], paths: dict[str, Path]) -> int:
    frames = {key: _active_frames(paths[key], results[key]) for key in paths}
    total = max(len(values) for values in frames.values())
    writer = imageio.get_writer(
        COMPARISON_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last: np.ndarray | None = None
    try:
        for index in range(total):
            canvas = Image.new("RGB", (1440, 416), (8, 10, 13))
            draw = ImageDraw.Draw(canvas, "RGBA")
            for column, key in enumerate(("proposed", "outward", "direct")):
                source = frames[key][min(index, len(frames[key]) - 1)]
                panel = Image.fromarray(source).resize((480, 360), Image.Resampling.LANCZOS)
                canvas.paste(panel, (480 * column, 56))
            draw.text((14, 13), "PROPOSED | joint posterior + local outward", fill=(90, 225, 255, 255), font=_font(14, bold=True))
            draw.text((500, 13), "BASELINE A | ordinary outward", fill=(255, 205, 105, 255), font=_font(14, bold=True))
            draw.text((980, 13), "BASELINE B | shared direction + direct ray", fill=(150, 235, 155, 255), font=_font(14, bold=True))
            draw.text((1270, 35), f"1x {index / FPS:05.1f}s", fill=(215, 224, 234, 255), font=_font(12))
            last = np.asarray(canvas)
            writer.append_data(last)
        assert last is not None
        summary = _summary_frame(last, results)
        imageio.imwrite(COMPARISON_KEYFRAME, summary)
        for _ in range(max(1, int(round(3.0 * FPS)))):
            writer.append_data(summary)
            total += 1
    finally:
        writer.close()
    return total


def _extract_keyframe(path: Path, result: RunResult, output: Path) -> None:
    reader = imageio.get_reader(path)
    try:
        index = max(0, result.frame_count - max(1, int(round(SUMMARY_TIME_S * FPS))) - 1)
        imageio.imwrite(output, reader.get_data(min(index, reader.count_frames() - 1)))
    finally:
        reader.close()


def main() -> None:
    ACTIVE.mkdir(parents=True, exist_ok=True)
    DIRECT_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    proposed = _load_run_result(source["methods"]["learned"])
    outward = _load_run_result(source["methods"]["baseline"])

    print("[1/2] render fair shared-probe direct baseline", flush=True)
    started = time.perf_counter()
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    direct = run_method(
        "direct",
        DIRECT_SOURCE,
        initial_xy_m=INITIAL_XY_M.copy(),
        direction_runtime=runtime,
        visual_sigma_m=0.0020,
        visual_bound_m=0.004895,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        active_time_budget_s=90.0,
        continue_to_bottom=True,
    )
    print(f"[1/2] complete in {time.perf_counter() - started:.1f}s", flush=True)

    _replace_link(PROPOSED_LINK, PROPOSED_SOURCE)
    _replace_link(OUTWARD_LINK, OUTWARD_SOURCE)
    _replace_link(DIRECT_LINK, DIRECT_SOURCE)
    _extract_keyframe(DIRECT_SOURCE, direct, DIRECT_KEYFRAME)
    print("[2/2] compose synchronized three-method comparison", flush=True)
    results = {"proposed": proposed, "outward": outward, "direct": direct}
    paths = {"proposed": PROPOSED_SOURCE, "outward": OUTWARD_SOURCE, "direct": DIRECT_SOURCE}
    frames = compose(results, paths)

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "active_methods": {
            "proposed": "shared conical F/T -> frozen P1v3 direction/state -> visual + soft CAD posterior -> posterior-limited transport -> Q99 local outward -> full outward fallback if exhausted -> insertion",
            "baseline_a": "ordinary outward Archimedean spiral under the same force/speed/pitch -> insertion",
            "baseline_b": "shared conical F/T -> same frozen P1v3 direction -> direct ray up to visual bound; capture stops motion; shifted reset + full re-probe is the failure policy",
        },
        "fairness": {
            "same_initial_xy_mm": [4.0, 0.0],
            "proposed_and_direct_same_probe": True,
            "proposed_and_direct_same_direction_estimator": "outputs/current/models/p1v3_m8_direction_models",
            "proposed_and_direct_control_direction_deg": proposed.control_direction_deg,
            "same_down_force_n": 3.0,
            "same_xy_speed_mm_s": 2.25,
            "same_capture_and_cad_bottom_endpoint": True,
            "direct_uses_true_center_or_distance": False,
            "outward_has_no_probe_cost": True,
        },
        "results": {key: asdict(value) for key, value in results.items()},
        "claim_boundary": (
            "Single selected M8 simulation case with a synthetic visual prior and sensor/depth-trend capture handoff. "
            "All methods share the same post-capture tip-RCC energy/CAD guard, which does not alter search. "
            "The CAD-bottom endpoint remains simulation-stage. The fair direct baseline wins this case; proposed superiority is "
            "not supported. Retry is part of the Baseline-B policy but is not exercised because attempt one "
            "succeeds. The separate 36-pose pre-hardware audit reports Proposed/A/B=36/35/29 with "
            "0/1/0 hard stops; this video is explanatory, not distribution evidence."
        ),
        "videos": {
            "proposed": str(PROPOSED_LINK.relative_to(ROOT)),
            "baseline_a_outward": str(OUTWARD_LINK.relative_to(ROOT)),
            "baseline_b_direct": str(DIRECT_LINK.relative_to(ROOT)),
            "comparison": str(COMPARISON_VIDEO.relative_to(ROOT)),
        },
        "keyframes": {
            "baseline_b_direct": str(DIRECT_KEYFRAME.relative_to(ROOT)),
            "comparison": str(COMPARISON_KEYFRAME.relative_to(ROOT)),
        },
        "comparison_frames": frames,
        "robustness_extension": {
            "audit": "outputs/current/evaluations/m8_pre_hardware_noise_audit/report.json",
            "manifest": "outputs/current/diagnostics/historical_comparisons/robustness_noise/manifest.json",
            "comparison": "archived; see outputs/current/VIDEO_RETENTION.json",
            "scope": "separate pre-hardware observation-prior audit; does not replace the clean selected comparison",
        },
        "diagnostic_not_active_baseline": "outputs/current/diagnostics/paper_direct_scripted_stress/",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    INDEX.write_text(
        "# Active M8 Method Comparison\n\n"
        "This directory is the clean entry point for the current paper comparison.\n\n"
        "1. `01_proposed_learning_vision_cad.mp4`: proposed joint posterior, posterior-limited transport, Q99 local coverage, and full outward fallback if exhausted.\n"
        "2. `02_baseline_outward_spiral.mp4`: ordinary outward-spiral baseline.\n"
        "3. `03_baseline_shared_probe_direct_retry.mp4`: fair shared-probe direction-only direct baseline.\n"
        "4. `comparison_three_methods.mp4`: synchronized comparison.\n"
        "5. `manifest.json`: exact fairness contract, metrics, and claim boundary.\n\n"
        "The older scripted +/-22.5 degree retry video is a diagnostic stress illustration, not an active baseline.\n"
        "\n`robustness_noise/` adds a separate audit-selected comparison under a shared "
        "pre-hardware F/T observation prior. Its Baseline-B video shows all three probes/direct "
        "rays and both physical reset clips; the audit still excludes reset cost in B's favor.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
