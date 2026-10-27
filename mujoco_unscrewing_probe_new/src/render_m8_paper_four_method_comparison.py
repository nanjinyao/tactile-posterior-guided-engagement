from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.render_m8_search_comparison import FPS, ROOT


ACTIVE_SIX = ROOT / "outputs/current/active_comparison/six_methods"
HISTORICAL_SIX = (
    ROOT / "outputs/current/diagnostics/historical_comparisons/six_methods"
)
OUT = ROOT / "outputs/current/active_comparison/four_methods"
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


def _source_dir() -> Path:
    for candidate in (ACTIVE_SIX, HISTORICAL_SIX):
        if (candidate / "comparison_six_methods_3x2.mp4").is_file():
            return candidate
    raise FileNotFoundError("six-method source video is not available")


def _complete(result: dict) -> bool:
    return bool(result["bottom_reached"] and result["result"] == "cad_bottom_verified")


def _panel(frame: np.ndarray, source_xy: tuple[int, int], label: str, color: tuple[int, int, int]) -> Image.Image:
    x, y = source_xy
    panel = Image.fromarray(frame).crop((x, y, x + 640, y + 400))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 640, 40), fill=(8, 10, 13))
    draw.text((14, 10), label, fill=color, font=_font(12, bold=True))
    if label.startswith("BASELINE C"):
        draw.rectangle((8, 45, 365, 68), fill=(6, 8, 11))
        draw.text(
            (14, 49),
            "BASELINE C: frozen seeded random route",
            fill=(255, 225, 170),
            font=_font(10, bold=True),
        )
        draw.rectangle((8, 334, 390, 356), fill=(6, 8, 11))
        draw.text(
            (14, 338),
            "baseline C: frozen random waypoints",
            fill=(225, 232, 240),
            font=_font(9),
        )
    return panel


def _summary(frame: np.ndarray, source: dict) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (44, 38, 1236, 762),
        radius=8,
        fill=(3, 7, 12, 247),
        outline=(225, 235, 244, 235),
        width=2,
    )
    draw.text(
        (72, 62),
        "FOUR METHODS | SAME START, NUISANCE AND TERMINAL SAFETY LAYER",
        fill=(245, 248, 252, 255),
        font=_font(20, bold=True),
    )
    task = source["task"]
    draw.text(
        (72, 99),
        f"selected {task['pose_id']} | r={1000.0 * task['radius_m']:.2f} mm | mechanism example, not a success-rate estimate",
        fill=(183, 202, 220, 255),
        font=_font(12),
    )

    methods = (
        ("PROPOSED", "proposed", (90, 225, 255), "fingerprint + move + spiral"),
        ("A OUTWARD", "outward", (255, 205, 105), "ordinary outward coverage"),
        ("B DIRECT", "direct", (150, 235, 155), "direction + retries"),
        ("C RANDOM", "random", (255, 145, 105), "frozen random waypoints"),
    )
    headers = (
        ("method", 72),
        ("result", 255),
        ("capture", 380),
        ("complete", 490),
        ("active", 600),
        ("XY", 700),
        ("contact", 790),
        ("work", 900),
        ("max F", 1000),
        ("mechanism", 1090),
    )
    for label, x in headers:
        draw.text((x, 146), label, fill=(150, 173, 195, 255), font=_font(11, bold=True))

    y = 184
    for label, key, color, mechanism in methods:
        result = source["results"][key]
        values = (
            label,
            "complete" if _complete(result) else result["result"],
            "n/a" if result["time_to_inner_wall_s"] is None else f"{result['time_to_inner_wall_s']:.2f}s",
            "n/a" if result["time_to_bottom_s"] is None else f"{result['time_to_bottom_s']:.2f}s",
            f"{result['total_active_time_s']:.2f}s",
            f"{result['actual_xy_path_mm']:.1f}mm",
            f"{result['contact_xy_path_mm']:.1f}mm",
            f"{result['wear_proxy_mj']:.1f}mJ",
            f"{result['max_force_n']:.1f}N",
            mechanism,
        )
        for (header, x), value in zip(headers, values):
            fill = (*color, 255) if header == "method" else (232, 239, 246, 255)
            draw.text((x, y), str(value)[:18], fill=fill, font=_font(11, bold=header == "method"))
        draw.line((72, y + 31, 1208, y + 31), fill=(100, 120, 140, 75), width=1)
        y += 67

    notes = (
        "Proposed and B use the same first conical fingerprint and frozen learned direction; no display-only error is injected.",
        "A removes active touch; B removes radial belief and geometric fallback; C removes the structured information model.",
        "All four use the same sensor-only capture handoff, hard limits, tip-RCC/energy guard and CAD-depth terminal verifier.",
        "Random happens to complete quickly in this selected seed. The frozen 36-pose paired audit, not this video, supports comparisons.",
        "Paper-facing names are Proposed / Baseline A / Baseline B / Baseline C; use the frozen paired report for claims.",
    )
    draw.text((72, 486), "claim boundary", fill=(255, 225, 150, 255), font=_font(13, bold=True))
    y = 524
    for note in notes:
        draw.text((84, y), note, fill=(215, 227, 238, 255), font=_font(12))
        y += 42
    return np.asarray(image)


def main() -> None:
    source_dir = _source_dir()
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    video_path = source_dir / "comparison_six_methods_3x2.mp4"
    method_keys = ("proposed", "outward", "direct", "random")
    if not all(_complete(source["results"][key]) for key in method_keys):
        raise RuntimeError("selected paper-facing four-method example is no longer complete")
    if source["results"]["proposed"]["fingerprint_sha256"] != source["results"]["direct"]["fingerprint_sha256"]:
        raise RuntimeError("Proposed and Baseline B do not share the same first fingerprint")

    OUT.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    total_source = reader.count_frames()
    active_frames = max(1, total_source - 3 * FPS)
    layout = (
        ("PROPOSED | fingerprint posterior + residual outward", (640, 0), (0, 0), (90, 225, 255)),
        ("BASELINE A | ordinary outward coverage", (0, 400), (640, 0), (255, 205, 105)),
        ("BASELINE B | shared direction + full shifted retries", (640, 400), (0, 400), (150, 235, 155)),
        ("BASELINE C | frozen seeded random route", (640, 800), (640, 400), (255, 145, 105)),
    )
    writer = imageio.get_writer(COMPARISON, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    last: np.ndarray | None = None
    keyframe_index = min(120, active_frames - 1)
    try:
        for index in range(active_frames):
            frame = reader.get_data(index)
            canvas = Image.new("RGB", (1280, 800), (8, 10, 13))
            for label, source_xy, target_xy, color in layout:
                canvas.paste(_panel(frame, source_xy, label, color), target_xy)
            last = np.asarray(canvas)
            writer.append_data(last)
            if index == keyframe_index:
                imageio.imwrite(KEYFRAME, last)
        if last is None:
            raise RuntimeError("source video yielded no active frames")
        final = _summary(last, source)
        for _ in range(3 * FPS):
            writer.append_data(final)
    finally:
        reader.close()
        writer.close()

    source_result_keys = {
        "proposed": "proposed",
        "baseline_a": "outward",
        "baseline_b": "direct",
        "baseline_c_random": "random",
    }
    paper_results = {}
    for key, source_key in source_result_keys.items():
        value = dict(source["results"][source_key])
        value["component_video_path_archival_provenance"] = value.get("video_path")
        value["video_path"] = str(COMPARISON.relative_to(ROOT))
        paper_results[key] = value

    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": source["selection_rule"],
        "selected_audit_trial": source["selected_audit_trial"],
        "task": source["task"],
        "paper_method_order": ["proposed", "baseline_a", "baseline_b", "baseline_c_random"],
        "source_result_keys": source_result_keys,
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_prior": True,
            "proposed_and_b_identical_first_fingerprint": True,
            "baseline_b_all_attempts_visible": True,
            "baseline_c_seed_fixed_before_rollout": source["fairness"]["baseline_d_seed_fixed_before_rollout"],
            "online_true_center_or_contact_tags": False,
            "shared_capture_guard_and_terminal_verifier": True,
        },
        "results": paper_results,
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "keyframe": str(KEYFRAME.relative_to(ROOT)),
        "source_video": str(video_path.relative_to(ROOT)),
        "source_active_frames": active_frames,
        "claim_boundary": (
            "Conditional same-start mechanism illustration. Random Baseline C is frozen before rollout and may be lucky in this selected case. "
            "Success-rate and efficiency claims must use the frozen 36-pose paired random-baseline audit."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(COMPARISON.relative_to(ROOT))


if __name__ == "__main__":
    main()
