#!/usr/bin/env python3
"""Render selected videos from the frozen five-method retry3/R3 audit.

Two products are supported:

* ``matrix``: Proposed+, Proposed, pure outward, improved fingerprint direct x3.
* ``direct``: improved four-channel direct x3 versus Jin-style three-channel
  direct x3.

The component rollouts are rerun with video observation enabled and every
substantive RunResult field is compared with the frozen audit trial.  The
selected videos explain mechanisms; distribution-level claims must come from
the complete paired audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.m8_direction_runtime import M8DirectionEnsemble
from src.m8_jin_style_direction import JinStyleDirectionRuntime
from src.render_m8_search_comparison import FPS, SUMMARY_TIME_S, RunResult, run_method
from src.run_m8_fair_policy_benchmark import PoseTask
from src.run_m8_five_method_retry3_audit import (
    DEFAULT_OUTPUT as AUDIT_ROOT,
    DIRECT_FAILURE_POLICY,
    IMPROVED_MODEL_ROOT,
    JIN_MODEL_ROOT,
    METHOD_LABELS,
    _common_kwargs,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/current/active_comparison/m8_retry3_selected_media"

MATRIX_METHODS = (
    "proposed_plus",
    "proposed",
    "outward",
    "improved_direct_retry3",
)
DIRECT_METHODS = ("improved_direct_retry3", "jin_direct_retry3")
DISPLAY_NAMES = {
    "proposed_plus": "PROPOSED+ | shortcut + task-aware grid",
    "proposed": "PROPOSED | shortcut + residual outward",
    "outward": "PURE OUTWARD | no fingerprint",
    "improved_direct_retry3": "IMPROVED 4-CHANNEL | DIRECT x3",
    "jin_direct_retry3": "JIN-STYLE 3-CHANNEL | DIRECT x3",
}
COLORS = {
    "proposed_plus": (91, 232, 208),
    "proposed": (91, 205, 255),
    "outward": (255, 204, 102),
    "improved_direct_retry3": (255, 139, 122),
    "jin_direct_retry3": (196, 148, 255),
}
RENDER_ONLY_FIELDS = {"video_path", "frame_count"}
FLOAT_ABS_TOL = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("matrix", "direct"), required=True)
    parser.add_argument("--pose-id", required=True)
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem")
    parser.add_argument("--case-label", default="SELECTED SAME-START CASE")
    parser.add_argument("--case-note", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_case(audit_root: Path, pose_id: str) -> tuple[PoseTask, Path, dict]:
    trial_path = audit_root / "trials" / f"{pose_id}.json"
    if not trial_path.is_file():
        raise FileNotFoundError(trial_path)
    payload = json.loads(trial_path.read_text(encoding="utf-8"))
    task = PoseTask(**payload["task"])
    return task, trial_path, payload


def _run_current(
    method: str,
    task: PoseTask,
    output_path: Path,
    improved: M8DirectionEnsemble,
    jin: JinStyleDirectionRuntime,
) -> RunResult:
    kwargs = _common_kwargs(task)
    if method == "proposed":
        return run_method(
            "learned", output_path=output_path, direction_runtime=improved, **kwargs
        )
    if method == "proposed_plus":
        return run_method(
            "hybrid", output_path=output_path, direction_runtime=improved, **kwargs
        )
    if method == "outward":
        return run_method("baseline", output_path=output_path, **kwargs)
    if method == "improved_direct_retry3":
        return run_method(
            "direct",
            output_path=output_path,
            direction_runtime=improved,
            direct_max_attempts=3,
            direct_adjustment_m=0.00025,
            direct_failure_policy_overlay=DIRECT_FAILURE_POLICY,
            fingerprint_display_mode="improved_four_channel",
            **kwargs,
        )
    if method == "jin_direct_retry3":
        return run_method(
            "direct",
            output_path=output_path,
            direction_runtime=jin,
            direct_max_attempts=3,
            direct_adjustment_m=0.00025,
            direct_failure_policy_overlay=DIRECT_FAILURE_POLICY,
            fingerprint_display_mode="jin_three_channel",
            **kwargs,
        )
    raise ValueError(method)


def _compare(expected: Any, actual: Any, path: str, mismatches: list[dict]) -> int:
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
            return 1
        checked = 0
        for key in sorted(set(expected) | set(actual)):
            if key in RENDER_ONLY_FIELDS:
                continue
            if key not in expected or key not in actual:
                mismatches.append(
                    {
                        "path": f"{path}.{key}",
                        "expected": expected.get(key, "<missing>"),
                        "actual": actual.get(key, "<missing>"),
                    }
                )
                checked += 1
            else:
                checked += _compare(
                    expected[key], actual[key], f"{path}.{key}", mismatches
                )
        return checked
    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
            return 1
        if len(expected) != len(actual):
            mismatches.append(
                {"path": f"{path}.length", "expected": len(expected), "actual": len(actual)}
            )
        checked = 1
        for index, (left, right) in enumerate(zip(expected, actual)):
            checked += _compare(left, right, f"{path}[{index}]", mismatches)
        return checked
    if isinstance(expected, bool) or isinstance(actual, bool):
        equal = type(expected) is type(actual) and expected == actual
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        left, right = float(expected), float(actual)
        equal = (
            math.isnan(left) and math.isnan(right)
            if math.isnan(left) or math.isnan(right)
            else math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_ABS_TOL)
        )
    else:
        equal = type(expected) is type(actual) and expected == actual
    if not equal:
        mismatches.append({"path": path, "expected": expected, "actual": actual})
    return 1


def _verify(expected: dict, result: RunResult, component: Path) -> dict:
    actual = asdict(result)
    mismatches: list[dict] = []
    checked = _compare(expected, actual, "RunResult", mismatches)
    renderer_valid = (
        component.is_file()
        and isinstance(actual.get("frame_count"), int)
        and int(actual["frame_count"]) > 0
        and bool(actual.get("video_path"))
    )
    if mismatches or not renderer_valid:
        raise RuntimeError(
            "selected render drifted from frozen audit: "
            + json.dumps(mismatches[:12], ensure_ascii=False)
        )
    return {
        "checked_substantive_leaves": checked,
        "mismatch_count": 0,
        "render_only_fields": sorted(RENDER_ONLY_FIELDS),
        "render_fields_valid": renderer_valid,
        "frozen_strict_bottom_success": _strict_bottom_success(expected),
        "rendered_strict_bottom_success": _strict_bottom_success(actual),
    }


def _active_readers(paths: dict[str, Path]) -> tuple[dict, dict[str, int]]:
    tail = int(round(SUMMARY_TIME_S * FPS))
    readers: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for method, path in paths.items():
        reader = imageio.get_reader(path)
        count = int(reader.count_frames()) - tail
        if count <= 0:
            reader.close()
            raise RuntimeError(f"invalid component frame count: {path}")
        readers[method] = reader
        counts[method] = count
    return readers, counts


def _status(result: RunResult) -> tuple[str, tuple[int, int, int]]:
    if _strict_bottom_success(asdict(result)):
        return "CAD BOTTOM", (131, 235, 161)
    if result.hard_stop:
        return "HARD STOP", (255, 116, 110)
    if result.offline_capture_validated:
        return "CAPTURE / NO BOTTOM", (255, 198, 105)
    return "NO CAPTURE", (255, 160, 122)


def _matrix_frame(
    frames: dict[str, np.ndarray],
    counts: dict[str, int],
    results: dict[str, RunResult],
    index: int,
    case_label: str,
) -> np.ndarray:
    image = Image.new("RGB", (1280, 800), (7, 10, 14))
    draw = ImageDraw.Draw(image)
    positions = {
        "proposed_plus": (0, 0),
        "proposed": (640, 0),
        "outward": (0, 400),
        "improved_direct_retry3": (640, 400),
    }
    for method in MATRIX_METHODS:
        x, y = positions[method]
        panel = Image.fromarray(frames[method]).resize((640, 360), Image.Resampling.LANCZOS)
        image.paste(panel, (x, y + 40))
        draw.rectangle((x, y, x + 640, y + 40), fill=(7, 10, 14))
        draw.text((x + 12, y + 11), DISPLAY_NAMES[method], fill=COLORS[method], font=_font(11, bold=True))
        status = "ACTIVE" if index < counts[method] - 1 else _status(results[method])[0]
        color = (235, 240, 246) if status == "ACTIVE" else _status(results[method])[1]
        bbox = draw.textbbox((0, 0), status, font=_font(10, bold=True))
        draw.text((x + 627 - (bbox[2] - bbox[0]), y + 12), status, fill=color, font=_font(10, bold=True))
    draw.rectangle((0, 780, 1280, 800), fill=(7, 10, 14))
    draw.text((12, 783), case_label, fill=(174, 194, 212), font=_font(9))
    return np.asarray(image)


def _direct_frame(
    frames: dict[str, np.ndarray],
    counts: dict[str, int],
    results: dict[str, RunResult],
    index: int,
    case_label: str,
) -> np.ndarray:
    image = Image.new("RGB", (1280, 720), (7, 10, 14))
    draw = ImageDraw.Draw(image)
    for column, method in enumerate(DIRECT_METHODS):
        x = 640 * column
        panel = Image.fromarray(frames[method]).resize((640, 360), Image.Resampling.LANCZOS)
        image.paste(panel, (x, 48))
        draw.text((x + 12, 14), DISPLAY_NAMES[method], fill=COLORS[method], font=_font(13, bold=True))
        status = "ACTIVE" if index < counts[method] - 1 else _status(results[method])[0]
        color = (235, 240, 246) if status == "ACTIVE" else _status(results[method])[1]
        draw.text((x + 500, 15), status, fill=color, font=_font(10, bold=True))
        result = results[method]
        attempts = result.direct_attempt_records
        draw.text(
            (x + 20, 432),
            (
                f"attempts={result.direct_attempts_completed}/3  "
                f"capture_attempt={result.direct_successful_attempt or '-'}  "
                f"bottom={bool(_strict_bottom_success(asdict(result)))}"
            ),
            fill=(225, 233, 241),
            font=_font(11, bold=True),
        )
        y = 468
        for record in attempts[:3]:
            draw.text(
                (x + 28, y),
                (
                    f"ray {record.get('attempt')}: dir={record.get('control_direction_deg', 0.0):.1f}deg  "
                    f"err={record.get('offline_control_error_deg', 0.0):.1f}deg  "
                    f"{record.get('outcome')}"
                ),
                fill=(190, 207, 222),
                font=_font(9),
            )
            y += 28
        draw.text(
            (x + 20, 570),
            f"active={result.total_active_time_s:.2f}s  XY={result.actual_xy_path_mm:.2f}mm  sim resets={result.simulator_reset_count}",
            fill=(190, 207, 222),
            font=_font(9),
        )
    draw.line((640, 0, 640, 720), fill=(72, 91, 108), width=2)
    draw.text((20, 673), case_label, fill=(174, 194, 212), font=_font(10))
    draw.text(
        (660, 673),
        "same pose / nuisance / sensor seed / controller / retry3 / post-capture R3",
        fill=(174, 194, 212),
        font=_font(9),
    )
    return np.asarray(image)


def _summary(
    methods: tuple[str, ...],
    results: dict[str, RunResult],
    pose_id: str,
    case_label: str,
    case_note: str,
    resolution: tuple[int, int],
) -> np.ndarray:
    width, height = resolution
    image = Image.new("RGB", resolution, (5, 9, 14))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 35, width - 38, height - 35), radius=10, outline=(207, 224, 238), width=2)
    draw.text((66, 59), f"{case_label} | {pose_id}", fill=(246, 248, 251), font=_font(18, bold=True))
    y = 112
    for method in methods:
        result = results[method]
        status, color = _status(result)
        direct = ""
        if method.endswith("retry3"):
            direct = f" | direct attempts {result.direct_attempts_completed}/3 | hit {result.direct_successful_attempt or '-'}"
        draw.text((72, y), DISPLAY_NAMES[method], fill=COLORS[method], font=_font(12, bold=True))
        draw.text(
            (430 if width >= 1000 else 300, y),
            f"{status} | {result.total_active_time_s:.2f}s | XY {result.actual_xy_path_mm:.2f}mm{direct}",
            fill=color,
            font=_font(10),
        )
        y += 55
    notes = [
        "Pre-capture direct opportunities: <=3 physical ray attempts; miss -> unload/lift/shift/full re-probe.",
        "Post-capture contract for every method: tip-RCC + shield + <=3 recovery cycles; no simulator reset.",
        "Four-channel panel = measured Fz, Tx, Ty, tip-z + improved posterior; Jin panel keeps its three polar channels.",
        "Selected mechanism video only; success/time comparison comes from the complete paired 24-pose audit.",
    ]
    if case_note:
        notes.insert(0, case_note)
    y = max(y + 12, int(height * 0.52))
    for note in notes:
        draw.text((72, y), note, fill=(198, 214, 228), font=_font(10))
        y += 34
    return np.asarray(image)


def _compose(
    mode: str,
    paths: dict[str, Path],
    results: dict[str, RunResult],
    video_path: Path,
    poster_path: Path,
    keyframe_path: Path,
    pose_id: str,
    case_label: str,
    case_note: str,
) -> dict:
    methods = MATRIX_METHODS if mode == "matrix" else DIRECT_METHODS
    resolution = (1280, 800) if mode == "matrix" else (1280, 720)
    readers, counts = _active_readers(paths)
    total = max(counts.values())
    last: dict[str, np.ndarray] = {}
    poster_index = max(0, min(total - 1, int(round(0.56 * total))))
    writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for index in range(total):
            for method in methods:
                if index < counts[method]:
                    last[method] = readers[method].get_data(index)
                elif method not in last:
                    last[method] = readers[method].get_data(counts[method] - 1)
            if mode == "matrix":
                frame = _matrix_frame(last, counts, results, index, case_label)
            else:
                frame = _direct_frame(last, counts, results, index, case_label)
            writer.append_data(frame)
            if index == poster_index:
                imageio.imwrite(poster_path, frame)
        keyframe = _summary(methods, results, pose_id, case_label, case_note, resolution)
        imageio.imwrite(keyframe_path, keyframe)
        for _ in range(int(round(SUMMARY_TIME_S * FPS))):
            writer.append_data(keyframe)
    finally:
        writer.close()
        for reader in readers.values():
            reader.close()
    return {
        "fps": FPS,
        "resolution": list(resolution),
        "active_frames": counts,
        "dynamic_frames": total,
        "poster_frame": poster_index,
        "summary_frames": int(round(SUMMARY_TIME_S * FPS)),
    }


def main() -> int:
    args = parse_args()
    audit_root = args.audit_root.resolve()
    task, trial_path, trial = _load_case(audit_root, args.pose_id)
    methods = MATRIX_METHODS if args.mode == "matrix" else DIRECT_METHODS
    output = (args.output / args.pose_id).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{args.pose_id}_{args.mode}_retry3_r3"
    video_path = output / f"{stem}.mp4"
    poster_path = output / f"{stem}_poster.png"
    keyframe_path = output / f"{stem}_keyframe.png"
    manifest_path = output / f"{stem}_manifest.json"
    targets = (video_path, poster_path, keyframe_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(existing)
    for path in existing:
        path.unlink()

    torch.set_num_threads(1)
    improved = M8DirectionEnsemble(IMPROVED_MODEL_ROOT)
    jin = JinStyleDirectionRuntime(JIN_MODEL_ROOT)
    results: dict[str, RunResult] = {}
    verifications: dict[str, dict] = {}
    component_paths: dict[str, Path] = {}
    for method in methods:
        component = output / f".{stem}_{method}.mp4"
        component.unlink(missing_ok=True)
        print(f"rendering {args.pose_id} {method}", flush=True)
        result = _run_current(method, task, component, improved, jin)
        verification = _verify(trial["methods"][method], result, component)
        results[method] = result
        verifications[method] = verification
        component_paths[method] = component

    if args.mode == "matrix":
        hashes = {
            results[method].fingerprint_sha256
            for method in ("proposed_plus", "proposed", "improved_direct_retry3")
        }
        if None in hashes or len(hashes) != 1:
            raise RuntimeError("matrix fingerprint hashes do not match")
    else:
        hashes = {results[method].fingerprint_sha256 for method in methods}
        if None in hashes or len(hashes) != 1:
            raise RuntimeError("direct-comparison first fingerprints do not match")

    composition = _compose(
        args.mode,
        component_paths,
        results,
        video_path,
        poster_path,
        keyframe_path,
        task.pose_id,
        args.case_label,
        args.case_note,
    )
    component_artifacts = {method: _artifact(path) for method, path in component_paths.items()}
    for path in component_paths.values():
        path.unlink()

    manifest = {
        "experiment": "M8_retry3_R3_selected_mechanism_media",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pose_id": task.pose_id,
        "case_label": args.case_label,
        "case_note": args.case_note,
        "task": asdict(task),
        "methods": list(methods),
        "source_trial": _artifact(trial_path),
        "source_audit_report": _artifact(audit_root / "report.json"),
        "results": {method: asdict(result) for method, result in results.items()},
        "verifications": verifications,
        "component_artifacts_before_deletion": component_artifacts,
        "component_videos_deleted": True,
        "contract": {
            "same_frozen_task_nuisance_sensor_seed": True,
            "same_first_raw_fingerprint_for_fingerprint_methods": True,
            "direct_ray_attempts_max": 3,
            "direct_retry_is_physical_same_rollout": True,
            "shared_post_capture_recoveries_max": 3,
            "simulator_resets": sum(result.simulator_reset_count for result in results.values()),
            "selected_mechanism_not_distribution_ranking": True,
            "online_true_center_or_contact_tags": False,
        },
        "composition": composition,
        "artifacts": {
            "video": _artifact(video_path),
            "poster": _artifact(poster_path),
            "keyframe": _artifact(keyframe_path),
        },
        "claim_boundary": (
            "Selected same-start MuJoCo mechanism visualization. The paired 24-pose "
            "five-method audit is the only success/time ranking source. Jin-style is "
            "a paper-inspired M8 proxy, not an exact reproduction of the paper task."
        ),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest["artifacts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
