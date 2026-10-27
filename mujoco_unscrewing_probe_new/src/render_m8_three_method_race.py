from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult


SIX_SOURCE = ROOT / "outputs/current/diagnostics/historical_comparisons/six_methods/manifest.json"
OUT = ROOT / "outputs/current/diagnostics/historical_comparisons/three_method_race"
VIDEO = OUT / "comparison_proposed_plus_proposed_entropy_race_1x3.mp4"
KEYFRAME = OUT / "comparison_proposed_plus_proposed_entropy_race_keyframe.png"
MANIFEST = OUT / "manifest.json"
WIDTH = 1440
HEIGHT = 600
PANEL_WIDTH = 480
PANEL_HEIGHT = 360
FAST_FORWARD = 4


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _completed(result: RunResult) -> bool:
    return bool(result.bottom_reached and result.result == "cad_bottom_verified")


def _summary(frame: np.ndarray, results: dict[str, RunResult], task: dict) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(
        (22, 22, WIDTH - 22, HEIGHT - 22),
        fill=(3, 7, 12, 246),
        outline=(225, 235, 244, 235),
        width=2,
    )
    draw.text(
        (46, 42),
        "THREE-METHOD RACE | SAME START, CAD CAPTURE SCALE AND TERMINAL VERIFIER",
        fill=(245, 248, 252, 255),
        font=_font(19, bold=True),
    )
    draw.text(
        (46, 76),
        f"selected {task['pose_id']} | display acceleration does not change simulated active time",
        fill=(175, 198, 218, 255),
        font=_font(12),
    )

    columns = (
        ("method", 48),
        ("capture", 245),
        ("bottom", 360),
        ("XY path", 480),
        ("contact", 600),
        ("wear", 720),
        ("max F", 830),
        ("probes", 930),
        ("mechanism", 1050),
    )
    for label, x in columns:
        draw.text((x, 132), label, fill=(145, 170, 194, 255), font=_font(11, bold=True))

    rows = (
        ("PROPOSED+", "hybrid", (100, 245, 235), "shortcut -> task-aware grid -> funnel"),
        ("PROPOSED", "proposed", (90, 225, 255), "cone -> posterior transport -> outward"),
        ("BASELINE C", "entropy", (225, 155, 240), "wide-prior CAD grid -> max entropy"),
    )
    y = 170
    for label, key, color, mechanism in rows:
        result = results[key]
        values = (
            label,
            "n/a" if result.time_to_inner_wall_s is None else f"{result.time_to_inner_wall_s:.2f}s",
            "n/a" if result.time_to_bottom_s is None else f"{result.time_to_bottom_s:.2f}s",
            f"{result.actual_xy_path_mm:.1f}mm",
            f"{result.contact_xy_path_mm:.1f}mm",
            f"{result.wear_proxy_mj:.1f}mJ",
            f"{result.max_force_n:.1f}N",
            (
                "cone only"
                if result.entropy_grid_probe_count is None
                else str(result.entropy_grid_probe_count)
            ),
            mechanism,
        )
        for (header, x), value in zip(columns, values):
            fill = (*color, 255) if header == "method" else (232, 239, 246, 255)
            draw.text((x, y), str(value)[:42], fill=fill, font=_font(11, bold=header == "method"))
        draw.line((48, y + 32, WIDTH - 48, y + 32), fill=(100, 120, 140, 75), width=1)
        y += 66

    notes = [
        "Geometry: grid spacing and outward pitch obey the same CAD-derived effective capture radius.",
        "Proposed and Proposed+ pay the full conical-probe cost; C receives no conical fingerprint or learned direction.",
        "Proposed+ scores segment capture probability plus information gain; no local spiral is available.",
        "Fresh frozen 24-pose strict capture: Proposed+/Proposed/C = 21/23/23, all with zero hard stops.",
        "Proposed+ reduces some path/force exposure but is slower than Proposed and much less contact-efficient than C.",
        "This selected race visualizes mechanism only; it cannot override the fresh paired statistics.",
    ]
    y = 390
    draw.text((48, y), "claim boundary", fill=(255, 225, 150, 255), font=_font(13, bold=True))
    y += 34
    for note in notes:
        draw.text((62, y), note, fill=(215, 227, 238, 255), font=_font(12))
        y += 34
    return np.asarray(image)


def _compose(
    paths: dict[str, Path], results: dict[str, RunResult], task: dict
) -> tuple[int, int, str]:
    labels = (
        ("PROPOSED+ | posterior field + task-aware contacts", "hybrid", (100, 245, 235)),
        ("PROPOSED | posterior-guided local outward", "proposed", (90, 225, 255)),
        ("BASELINE C | wide-prior maximum-entropy grid", "entropy", (225, 155, 240)),
    )
    readers = {key: imageio.get_reader(path) for key, path in paths.items()}
    active_counts: dict[str, int] = {}
    try:
        for _, key, _ in labels:
            count = readers[key].count_frames()
            active_counts[key] = max(1, count - int(round(SUMMARY_TIME_S * FPS)))

        position = {key: 0 for key in paths}
        last_frames = {key: readers[key].get_data(0) for key in paths}
        accelerated = False
        acceleration_start_frame = -1
        first_finished_method = ""
        frame_index = 0
        last_canvas: np.ndarray | None = None
        writer = imageio.get_writer(
            VIDEO, fps=FPS, codec="libx264", quality=8, macro_block_size=1
        )
        try:
            while True:
                canvas = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 13))
                draw = ImageDraw.Draw(canvas)
                for panel_index, (label, key, color) in enumerate(labels):
                    x = panel_index * PANEL_WIDTH
                    frame = last_frames[key]
                    panel = Image.fromarray(frame).resize(
                        (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS
                    )
                    canvas.paste(panel, (x, 52))
                    draw.text((x + 12, 14), label, fill=color, font=_font(11, bold=True))
                    status = (
                        "COMPLETE"
                        if position[key] >= active_counts[key] - 1
                        else (f"DISPLAY x{FAST_FORWARD}" if accelerated else "DISPLAY x1")
                    )
                    draw.text((x + 345, 34), status, fill=(235, 240, 245), font=_font(10, bold=True))
                draw.text(
                    (18, 430),
                    "Playback policy: physical-time synchronized until first completion; unfinished panels then fast-forward for presentation only.",
                    fill=(190, 207, 222),
                    font=_font(11),
                )
                draw.text(
                    (18, 460),
                    "All reported capture/bottom times, paths, F/T and wear are computed from the unaccelerated MuJoCo rollouts.",
                    fill=(190, 207, 222),
                    font=_font(11),
                )
                last_canvas = np.asarray(canvas)
                writer.append_data(last_canvas)
                frame_index += 1

                finished = [
                    key for key in paths if position[key] >= active_counts[key] - 1
                ]
                if len(finished) == len(paths):
                    break
                if finished and not accelerated:
                    accelerated = True
                    acceleration_start_frame = frame_index
                    first_finished_method = min(
                        finished, key=lambda key: active_counts[key]
                    )
                step = FAST_FORWARD if accelerated else 1
                for key in paths:
                    if position[key] >= active_counts[key] - 1:
                        continue
                    position[key] = min(
                        active_counts[key] - 1,
                        position[key] + step,
                    )
                    last_frames[key] = readers[key].get_data(position[key])

            assert last_canvas is not None
            final = _summary(last_canvas, results, task)
            imageio.imwrite(KEYFRAME, final)
            for _ in range(3 * FPS):
                writer.append_data(final)
                frame_index += 1
        finally:
            writer.close()
    finally:
        for reader in readers.values():
            reader.close()
    return frame_index, acceleration_start_frame, first_finished_method


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SIX_SOURCE.read_text(encoding="utf-8"))
    results = {
        key: RunResult(**source["results"][key])
        for key in ("hybrid", "proposed", "entropy")
    }
    if not all(_completed(result) for result in results.values()):
        raise RuntimeError("race source requires geometric bottom completion for all methods")
    if results["hybrid"].fingerprint_sha256 != results["proposed"].fingerprint_sha256:
        raise RuntimeError("Proposed+ and Proposed fingerprint mismatch")
    paths = {key: ROOT / source["videos"][key] for key in results}
    frames, acceleration_start_frame, first_finished_method = _compose(
        paths, results, source["task"]
    )
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_audit_trial": source["selected_audit_trial"],
        "selection_rule": (
            "frozen same-start six-method case; all three reach CAD-bottom; video is mechanism evidence"
        ),
        "task": source["task"],
        "method_order": ["hybrid", "proposed", "entropy"],
        "proposed_plus_execution_order": source["proposed_plus_execution_order"],
        "paths": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "results": {key: asdict(value) for key, value in results.items()},
        "comparison": str(VIDEO.relative_to(ROOT)),
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "frames": frames,
        "display_fast_forward": {
            "factor": FAST_FORWARD,
            "start_frame": acceleration_start_frame,
            "first_finished_method": first_finished_method,
            "changes_simulation_or_metrics": False,
        },
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_prior": True,
            "proposed_plus_and_proposed_identical_conical_fingerprint": True,
            "same_cad_effective_capture_scale": True,
            "same_capture_guard_and_terminal_verifier": True,
            "true_center_online": False,
        },
        "claim_boundary": (
            "Selected conditional mechanism race. Display acceleration begins only after the first physical "
            "completion and never changes rollout metrics. Fresh frozen 24-pose strict capture is "
            "Proposed+/Proposed/C=21/23/23 with zero hard stops; Proposed+ remains an experimental tradeoff."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["display_fast_forward"], indent=2))


if __name__ == "__main__":
    main()
