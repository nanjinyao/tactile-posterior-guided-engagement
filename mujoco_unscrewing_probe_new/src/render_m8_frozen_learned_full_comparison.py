from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.capture_yaw_assist import CaptureYawAssistConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import (
    BASELINE_MAX_RADIUS_M,
    BASELINE_PHASE_RAD,
    FPS,
    INITIAL_XY_M,
    KEYFRAME_OUT,
    ROOT,
    SUMMARY_TIME_S,
    VIDEO_OUT,
    RunResult,
    run_method,
)


MODEL_ROOT = ROOT / "outputs" / "current" / "models" / "p1v3_m8_direction_models"
RESULT_PATH = (
    ROOT
    / "outputs"
    / "current"
    / "reports"
    / "m8_outer_partial_cad_fused_full_closed_loop_comparison.json"
)

ASSUMED_VIDEO = VIDEO_OUT / "m8_outer_partial_assumed_posterior_full_to_bottom.mp4"
LEARNED_VIDEO = VIDEO_OUT / "m8_outer_partial_p1v3_cad_fused_full_to_bottom.mp4"
BASELINE_VIDEO = VIDEO_OUT / "m8_outer_partial_source_outward_full_to_bottom.mp4"
TRIPLE_VIDEO = VIDEO_OUT / "m8_outer_partial_assumed_vs_cad_fused_vs_source_full.mp4"

ASSUMED_KEYFRAME = KEYFRAME_OUT / "m8_outer_partial_assumed_full_keyframe.png"
LEARNED_KEYFRAME = KEYFRAME_OUT / "m8_outer_partial_p1v3_cad_fused_full_keyframe.png"
BASELINE_KEYFRAME = KEYFRAME_OUT / "m8_outer_partial_source_full_keyframe.png"
TRIPLE_KEYFRAME = KEYFRAME_OUT / "m8_outer_partial_cad_fused_triptych_keyframe.png"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _active_frame_count(result: RunResult) -> int:
    return max(1, result.frame_count - max(1, int(round(SUMMARY_TIME_S * FPS))))


def _read_active_frames(path: Path, count: int) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [reader.get_data(index) for index in range(min(count, reader.count_frames()))]
    finally:
        reader.close()


def _metric(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _summary_frame(frame: np.ndarray, results: dict[str, RunResult]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (160, 68, 1280, 390),
        radius=8,
        fill=(3, 7, 12, 238),
        outline=(210, 225, 235, 220),
        width=2,
    )
    draw.text(
        (195, 88),
        "SAME START / SAME FORCE / SAME PITCH / CAD-BOTTOM ENDPOINT",
        fill=(245, 247, 250, 255),
        font=_font(21, bold=True),
    )
    headers = ("ASSUMED POSTERIOR", "P1V3 + SOFT CAD", "SOURCE OUTWARD")
    colors = ((90, 225, 255), (244, 114, 182), (255, 205, 105))
    x_columns = (205, 560, 925)
    for x, header, color in zip(x_columns, headers, colors):
        draw.text((x, 130), header, fill=(*color, 255), font=_font(18, bold=True))
    rows = (
        ("capture", "time_to_inner_wall_s", "s"),
        ("bottom", "time_to_bottom_s", "s"),
        ("XY path", "actual_xy_path_mm", "mm"),
        ("work proxy", "wear_proxy_mj", "mJ"),
        ("max force", "max_force_n", "N"),
    )
    ordered = (results["assumed"], results["learned"], results["baseline"])
    y = 172
    for label, field, suffix in rows:
        draw.text((175, y), label, fill=(180, 192, 207, 255), font=_font(16))
        for x, result in zip(x_columns, ordered):
            draw.text(
                (x + 110, y),
                _metric(getattr(result, field), suffix),
                fill=(235, 240, 246, 255),
                font=_font(16),
            )
        y += 34
    draw.text(
        (195, 348),
        "Learned: q(phi), p(state|F/T), visual prior + soft CAD annuli. Handoff uses sensor/depth trend only.",
        fill=(190, 202, 216, 255),
        font=_font(14),
    )
    return np.asarray(image)


def compose_triptych(paths: dict[str, Path], results: dict[str, RunResult]) -> int:
    counts = {key: _active_frame_count(results[key]) for key in paths}
    frames = {key: _read_active_frames(path, counts[key]) for key, path in paths.items()}
    total = max(len(values) for values in frames.values())
    writer = imageio.get_writer(
        TRIPLE_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last_frame: np.ndarray | None = None
    try:
        for index in range(total):
            canvas = Image.new("RGB", (1440, 416), (8, 10, 13))
            draw = ImageDraw.Draw(canvas, "RGBA")
            for column, key in enumerate(("assumed", "learned", "baseline")):
                source = frames[key][min(index, len(frames[key]) - 1)]
                panel = Image.fromarray(source).resize((480, 360), Image.Resampling.LANCZOS)
                canvas.paste(panel, (480 * column, 56))
            draw.text((16, 13), "ASSUMED POSTERIOR | hand radial shortcut", fill=(90, 225, 255, 255), font=_font(17, bold=True))
            draw.text((495, 13), "P1V3 + SOFT CAD | real F/T", fill=(244, 114, 182, 255), font=_font(17, bold=True))
            draw.text((982, 13), "SOURCE OUTWARD | no posterior", fill=(255, 205, 105, 255), font=_font(17, bold=True))
            draw.text((1238, 35), f"DISPLAY 1x | {index / FPS:05.1f}s", fill=(215, 224, 234, 255), font=_font(13))
            last_frame = np.asarray(canvas)
            writer.append_data(last_frame)
        assert last_frame is not None
        summary = _summary_frame(last_frame, results)
        imageio.imwrite(TRIPLE_KEYFRAME, summary)
        for _ in range(max(1, int(round(3.0 * FPS)))):
            writer.append_data(summary)
            total += 1
    finally:
        writer.close()
    return total


def _extract_keyframe(video: Path, result: RunResult, output: Path) -> None:
    reader = imageio.get_reader(video)
    try:
        index = min(_active_frame_count(result), reader.count_frames()) - 1
        imageio.imwrite(output, reader.get_data(max(index, 0)))
    finally:
        reader.close()


def main() -> None:
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    KEYFRAME_OUT.mkdir(parents=True, exist_ok=True)
    if "--compose-only" in sys.argv:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        results = {
            key: RunResult(**value) for key, value in payload["methods"].items()
        }
        paths = {
            key: ROOT / value for key, value in payload["videos"].items()
        }
        payload["triple_frames"] = compose_triptych(paths, results)
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        payload["claim_boundary"] = (
            "Same-start M8 simulation comparison with continuous probe/search/insertion-to-CAD-bottom dynamics. "
            "The frozen P1v3 branch uses simulated F/T and tip-z for direction/state beliefs, then fuses a "
            "synthetic visual prior with soft approximate-CAD radial annuli. Search-to-insertion handoff uses "
            "only persistent tip-z progress and bounded observed wrench; it does not use the true center or "
            "MuJoCo contact tags. All branches share the same post-capture tip-RCC energy/CAD guard; it does not alter search. "
            "CAD-bottom verification remains a simulation-stage geometric endpoint. This is "
            "not hardware, random-distribution, or general-superiority evidence."
        )
        RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"recomposed {TRIPLE_VIDEO}")
        return
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    yaw_assist = CaptureYawAssistConfig(enabled="--capture-yaw-assist" in sys.argv)
    common = {
        "initial_xy_m": INITIAL_XY_M.copy(),
        "baseline_max_radius_m": BASELINE_MAX_RADIUS_M,
        "active_time_budget_s": 90.0,
        "continue_to_bottom": True,
        "capture_yaw_assist": yaw_assist,
    }
    started = time.perf_counter()
    print("[1/4] rendering assumed-posterior full rollout", flush=True)
    assumed = run_method("proposed", ASSUMED_VIDEO, **common)
    print(f"[1/4] complete in {time.perf_counter() - started:.1f}s", flush=True)
    stage_started = time.perf_counter()
    print("[2/4] rendering P1v3 + F/T-state + soft-CAD full rollout", flush=True)
    learned = run_method(
        "learned",
        LEARNED_VIDEO,
        direction_runtime=runtime,
        visual_sigma_m=0.0020,
        visual_bound_m=0.004895,
        posterior_coverage=0.99,
        posterior_seed=20260710,
        max_transport_m=None,
        **common,
    )
    print(f"[2/4] complete in {time.perf_counter() - stage_started:.1f}s", flush=True)
    stage_started = time.perf_counter()
    print("[3/4] rendering source-style full rollout", flush=True)
    baseline = run_method(
        "baseline",
        BASELINE_VIDEO,
        baseline_phase_rad=BASELINE_PHASE_RAD,
        **common,
    )
    print(f"[3/4] complete in {time.perf_counter() - stage_started:.1f}s", flush=True)
    results = {"assumed": assumed, "learned": learned, "baseline": baseline}
    paths = {"assumed": ASSUMED_VIDEO, "learned": LEARNED_VIDEO, "baseline": BASELINE_VIDEO}
    print("[4/4] composing synchronized triptych", flush=True)
    triple_frames = compose_triptych(paths, results)
    _extract_keyframe(ASSUMED_VIDEO, assumed, ASSUMED_KEYFRAME)
    _extract_keyframe(LEARNED_VIDEO, learned, LEARNED_KEYFRAME)
    _extract_keyframe(BASELINE_VIDEO, baseline, BASELINE_KEYFRAME)

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "capture_yaw_assist": asdict(yaw_assist),
        "claim_boundary": (
            "Same-start M8 simulation comparison with continuous probe/search/insertion-to-CAD-bottom dynamics. "
            "The frozen P1v3 branch uses measured simulated F/T and tip-z for direction and support-state beliefs, "
            "then fuses a synthetic visual prior with soft approximate-CAD radial annuli. The CAD likelihood has a "
            "0.35 mm transition width and a nonzero floor; it does not use the true center or MuJoCo contact points. "
            "A conservative posterior planner selects transport and residual coverage. Search-to-insertion handoff "
            "uses persistent tip-z progress and bounded observed wrench without true-center or MuJoCo contact tags; "
            "bottom uses CAD depth plus Fz. All branches share the same post-capture tip-RCC energy/CAD guard, which "
            "does not alter search. This is not hardware, random-distribution, or general-superiority evidence."
        ),
        "preserved_originals": [
            "outputs/current/media/videos/comparisons/m8_outer_partial_proposed_probe_to_inner_wall.mp4",
            "outputs/current/media/videos/comparisons/m8_outer_partial_source_outward_to_inner_wall.mp4",
            "outputs/current/media/videos/comparisons/m8_outer_partial_proposed_vs_source_side_by_side.mp4",
            "outputs/current/media/videos/comparisons/m8_outer_partial_p1v3_learned_full_to_bottom.mp4",
            "outputs/current/media/videos/comparisons/m8_outer_partial_assumed_vs_learned_vs_source_full.mp4",
        ],
        "model_root": str(MODEL_ROOT.relative_to(ROOT)),
        "visual_prior": {
            "sigma_mm": 2.0,
            "truncated_radius_mm": 4.895,
            "source": "synthetic current simulation assumption; real covariance not yet measured",
        },
        "cad_fusion": {
            "source": "approximate M8 head, bit, socket-top and chamfer geometry from configs/allen_m8_source.json",
            "evidence": "P1v3 support-state probabilities inferred from the same F/T and tip-z phase trace",
            "likelihood": "soft radial state bands with 0.35 mm transition sigma and 0.02 floor",
            "not_used": "true hole center, MuJoCo contact points, or a learned distance head",
        },
        "methods": {key: asdict(value) for key, value in results.items()},
        "videos": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "triple_video": str(TRIPLE_VIDEO.relative_to(ROOT)),
        "triple_frames": triple_frames,
        "keyframes": {
            "assumed": str(ASSUMED_KEYFRAME.relative_to(ROOT)),
            "learned": str(LEARNED_KEYFRAME.relative_to(ROOT)),
            "baseline": str(BASELINE_KEYFRAME.relative_to(ROOT)),
            "triptych": str(TRIPLE_KEYFRAME.relative_to(ROOT)),
        },
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] complete; total {time.perf_counter() - started:.1f}s", flush=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
