#!/usr/bin/env python3
"""Render a frozen strict-M8 four-method selected-case matrix.

The four panels are Proposed, Proposed+, Outward, and Entropy-grid on the
frozen ``sobol_008`` task.  Proposed+ reuses the already verified strict
explainer video; the other three methods are rerun through the exact shared
strict-audit call contract.  Every displayed result is compared against the
frozen trial before composition, so a renderer/policy drift fails closed.

This artifact explains one same-start mechanism case.  It is not a method
ranking and never replaces the paired 24-pose report.
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
from src.render_m8_search_comparison import FPS, ROOT, SUMMARY_TIME_S, RunResult
from src.run_m8_fair_policy_benchmark import MODEL_ROOT, PoseTask
from src.run_m8_strict_complete_loop_audit import (
    ACTIVE_TIME_BUDGET_S,
    METHOD_ROLES,
    RUNTIME_METHOD,
    _run_with_runtime,
    _strict_bottom_success,
)


DEFAULT_POSE_ID = "sobol_008"
DEFAULT_STRICT_MANIFEST = (
    ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop/manifest.json"
)
DEFAULT_PROPOSED_PLUS_DIR = (
    ROOT / "outputs/current/active_comparison/proposed_plus_strict_explainer"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/active_comparison/strict_four_methods"
)

METHODS = ("proposed", "proposed_plus", "outward", "entropy_grid")
DISPLAY_NAMES = {
    "proposed": "PROPOSED | fingerprint -> posterior move -> outward",
    "proposed_plus": "PROPOSED+ | shortcut -> adaptive grid -> outward reserve",
    "outward": "OUTWARD | geometry-only Archimedean coverage",
    "entropy_grid": "ENTROPY-GRID | wide-prior active contacts",
}
COLORS = {
    "proposed": (90, 225, 255),
    "proposed_plus": (100, 245, 215),
    "outward": (255, 205, 105),
    "entropy_grid": (225, 155, 240),
}
FLOAT_FIELDS = (
    "time_to_inner_wall_s",
    "time_to_bottom_s",
    "total_active_time_s",
)
EXACT_FIELDS = (
    "result",
    "fingerprint_sha256",
    "hard_stop",
    "hard_stop_reason",
    "terminal_verification_mode",
    "inner_wall_confirmed",
    "offline_capture_validated",
    "bottom_reached",
    "fallback_budget_extension_s",
)
TIME_ABS_TOL_S = 1.0e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final strict sobol_008 four-method 2x2 matrix."
    )
    parser.add_argument(
        "--strict-manifest", type=Path, default=DEFAULT_STRICT_MANIFEST
    )
    parser.add_argument("--pose-id", default=DEFAULT_POSE_ID)
    parser.add_argument("--proposed-plus-dir", type=Path, default=DEFAULT_PROPOSED_PLUS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise ValueError(f"artifact must remain inside project root: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSansMono.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _load_frozen_case(
    strict_manifest: Path, pose_id: str
) -> tuple[PoseTask, Path, dict[str, dict[str, Any]]]:
    if pose_id != DEFAULT_POSE_ID:
        raise ValueError(
            f"this final artifact is frozen to {DEFAULT_POSE_ID}; received {pose_id}"
        )
    manifest = json.loads(strict_manifest.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("tasks", []) if item.get("pose_id") == pose_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one task {pose_id!r} in {strict_manifest}, found {len(matches)}"
        )
    task_payload = matches[0]
    task = PoseTask(**task_payload)
    if task.sensor_noise_seed is None:
        raise ValueError("strict rendering requires the frozen sensor-noise seed")

    trial_path = strict_manifest.parent / "trials" / f"{pose_id}.json"
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if trial.get("task") != task_payload:
        raise RuntimeError("strict manifest task and frozen trial task differ")
    frozen = trial.get("methods", {})
    missing = set(METHODS).difference(frozen)
    if missing:
        raise RuntimeError(f"frozen trial is missing methods: {sorted(missing)}")
    return task, trial_path, {method: frozen[method] for method in METHODS}


def _field_verification(
    method: str, actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    all_match = True
    for field in EXACT_FIELDS:
        match = actual.get(field) == expected.get(field)
        fields[field] = {
            "expected": expected.get(field),
            "actual": actual.get(field),
            "match": match,
        }
        all_match = all_match and match
    for field in FLOAT_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if expected_value is None or actual_value is None:
            match = expected_value is None and actual_value is None
            delta = None
        else:
            delta = float(actual_value) - float(expected_value)
            match = math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=TIME_ABS_TOL_S,
            )
        fields[field] = {
            "expected": expected_value,
            "actual": actual_value,
            "delta": delta,
            "absolute_tolerance_s": TIME_ABS_TOL_S,
            "match": match,
        }
        all_match = all_match and match

    expected_strict = _strict_bottom_success(expected)
    actual_strict = _strict_bottom_success(actual)
    strict_match = expected_strict == actual_strict
    all_match = all_match and strict_match
    return {
        "method": method,
        "fingerprint_applicable": method in {"proposed", "proposed_plus"},
        "fields": fields,
        "strict_bottom_success": {
            "expected": expected_strict,
            "actual": actual_strict,
            "match": strict_match,
        },
        "all_fields_match": all_match,
    }


def _verify_or_raise(
    method: str, actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    verification = _field_verification(method, actual, expected)
    if not verification["all_fields_match"]:
        failures = [
            field
            for field, item in verification["fields"].items()
            if not item["match"]
        ]
        if not verification["strict_bottom_success"]["match"]:
            failures.append("strict_bottom_success")
        raise RuntimeError(
            f"{method} render drifted from frozen {DEFAULT_POSE_ID}: {failures}"
        )
    return verification


def _load_reused_proposed_plus(
    directory: Path,
    task: PoseTask,
    expected: dict[str, Any],
) -> tuple[Path, RunResult, dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    result_path = directory / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("pose_id") != task.pose_id or payload.get("task") != asdict(task):
        raise RuntimeError("reused Proposed+ explainer is not the frozen task")
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        raise RuntimeError("reused Proposed+ result.json has no RunResult payload")

    video_record = manifest.get("artifacts", {}).get("video", {})
    result_record = manifest.get("artifacts", {}).get("result", {})
    video_path = ROOT / video_record.get("path", "")
    if video_path != directory / f"proposed_plus_strict_{task.pose_id}.mp4":
        raise RuntimeError("unexpected Proposed+ video path in explainer manifest")
    for path, record, label in (
        (video_path, video_record, "video"),
        (result_path, result_record, "result"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"reused Proposed+ {label} byte count mismatch")
        if _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"reused Proposed+ {label} hash mismatch")

    verification = _verify_or_raise("proposed_plus", result_payload, expected)
    source = {
        "reused_verified_strict_explainer": True,
        "manifest": _artifact(manifest_path),
        "result": _artifact(result_path),
        "video": _artifact(video_path),
    }
    return video_path, RunResult(**result_payload), verification, source


def _read_active_counts(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, int]]:
    readers: dict[str, Any] = {}
    active_counts: dict[str, int] = {}
    summary_frames = int(round(SUMMARY_TIME_S * FPS))
    for method, path in paths.items():
        reader = imageio.get_reader(path)
        count = int(reader.count_frames())
        if count <= summary_frames:
            reader.close()
            raise RuntimeError(f"video too short to remove summary tail: {path}")
        readers[method] = reader
        active_counts[method] = count - summary_frames
    return readers, active_counts


def _panel_canvas(
    frames: dict[str, np.ndarray],
    active_counts: dict[str, int],
    index: int,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 800), (8, 10, 13))
    draw = ImageDraw.Draw(canvas)
    positions = {
        "proposed": (0, 0),
        "proposed_plus": (640, 0),
        "outward": (0, 400),
        "entropy_grid": (640, 400),
    }
    for method in METHODS:
        x, y = positions[method]
        panel = Image.fromarray(frames[method]).resize(
            (640, 360), Image.Resampling.LANCZOS
        )
        canvas.paste(panel, (x, y + 40))
        draw.rectangle((x, y, x + 640, y + 40), fill=(8, 10, 13))
        draw.text(
            (x + 13, y + 11),
            DISPLAY_NAMES[method],
            fill=COLORS[method],
            font=_font(12, bold=True),
        )
        status = "COMPLETE" if index >= active_counts[method] - 1 else "ACTIVE"
        status_color = (140, 235, 165) if status == "COMPLETE" else (230, 236, 242)
        draw.text(
            (x + 555, y + 11), status, fill=status_color, font=_font(10, bold=True)
        )
    return np.asarray(canvas)


def _summary_frame(
    results: dict[str, RunResult], task: PoseTask, verifications: dict[str, dict[str, Any]]
) -> np.ndarray:
    image = Image.new("RGB", (1280, 800), (5, 9, 14))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        (42, 35, 1238, 765),
        radius=10,
        fill=(3, 7, 12, 250),
        outline=(215, 230, 242, 235),
        width=2,
    )
    draw.text(
        (72, 62),
        "STRICT SAME-START M8 | FROZEN sobol_008 | SELECTED MECHANISM, NOT RANKING",
        fill=(245, 248, 252),
        font=_font(19, bold=True),
    )
    draw.text(
        (72, 99),
        (
            f"initial XY = ({1000.0 * task.x_m:+.3f}, {1000.0 * task.y_m:+.3f}) mm  |  "
            f"hard accumulated active budget = {ACTIVE_TIME_BUDGET_S:.0f} s"
        ),
        fill=(173, 199, 221),
        font=_font(12),
    )

    headers = (
        ("method", 74),
        ("result", 285),
        ("capture", 500),
        ("CAD bottom", 635),
        ("active", 785),
        ("hard stop", 905),
        ("terminal", 1000),
        ("frozen", 1135),
    )
    for label, x in headers:
        draw.text((x, 160), label, fill=(148, 174, 197), font=_font(12, bold=True))
    draw.line((70, 188, 1208, 188), fill=(120, 145, 166, 150), width=1)

    short_names = {
        "proposed": "PROPOSED",
        "proposed_plus": "PROPOSED+",
        "outward": "OUTWARD",
        "entropy_grid": "ENTROPY-GRID",
    }
    y = 215
    for method in METHODS:
        result = results[method]
        terminal = {
            "cad_depth_low_wrench_dwell": "low-wrench dwell",
            "bottom_force_dwell": "axial-force dwell",
            None: "none",
        }.get(result.terminal_verification_mode, result.terminal_verification_mode)
        values = (
            short_names[method],
            result.result,
            "n/a" if result.time_to_inner_wall_s is None else f"{result.time_to_inner_wall_s:.3f}s",
            "n/a" if result.time_to_bottom_s is None else f"{result.time_to_bottom_s:.3f}s",
            f"{result.total_active_time_s:.3f}s",
            "YES" if result.hard_stop else "no",
            terminal,
            "MATCH" if verifications[method]["all_fields_match"] else "MISMATCH",
        )
        for (header, x), value in zip(headers, values):
            fill = COLORS[method] if header == "method" else (232, 239, 246)
            draw.text((x, y), str(value), fill=fill, font=_font(11, bold=header == "method"))
        draw.line((70, y + 32, 1208, y + 32), fill=(85, 104, 122, 100), width=1)
        y += 67

    notes = (
        "All four use the identical strict runner policy/environment contract; rendering is the only added argument.",
        "Proposed and Proposed+ fingerprints match the frozen trial byte-for-byte; geometry-only methods have no fingerprint.",
        "True center/contact tags are unavailable online. Offline capture scoring and CAD-bottom verification follow the strict report.",
        "This selected rollout shows mechanisms under one nuisance draw. Method claims come only from the frozen paired 24-pose report.",
        "Entropy-grid finishing first here is visible by design; the video is not edited to manufacture a Proposed+ win.",
    )
    draw.text((72, 520), "claim boundary", fill=(255, 220, 145), font=_font(13, bold=True))
    y = 555
    for note in notes:
        draw.text((86, y), note, fill=(213, 226, 237), font=_font(11))
        y += 34
    return np.asarray(image)


def _compose(
    paths: dict[str, Path],
    results: dict[str, RunResult],
    task: PoseTask,
    verifications: dict[str, dict[str, Any]],
    video_path: Path,
    poster_path: Path,
    keyframe_path: Path,
) -> dict[str, Any]:
    readers, active_counts = _read_active_counts(paths)
    total_active_frames = max(active_counts.values())
    poster_index = int(round(0.55 * (total_active_frames - 1)))
    last_frames: dict[str, np.ndarray] = {}
    poster_written = False
    writer = imageio.get_writer(
        video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for index in range(total_active_frames):
            for method in METHODS:
                if index < active_counts[method]:
                    last_frames[method] = readers[method].get_data(index)
                elif method not in last_frames:
                    last_frames[method] = readers[method].get_data(
                        active_counts[method] - 1
                    )
            canvas = _panel_canvas(last_frames, active_counts, index)
            writer.append_data(canvas)
            if index == poster_index:
                imageio.imwrite(poster_path, canvas)
                poster_written = True

        if not poster_written:
            imageio.imwrite(poster_path, canvas)
        keyframe = _summary_frame(results, task, verifications)
        imageio.imwrite(keyframe_path, keyframe)
        summary_frames = int(round(SUMMARY_TIME_S * FPS))
        for _ in range(summary_frames):
            writer.append_data(keyframe)
    finally:
        writer.close()
        for reader in readers.values():
            reader.close()
    return {
        "fps": FPS,
        "resolution": [1280, 800],
        "active_frames_by_method": active_counts,
        "synchronized_dynamic_frames": total_active_frames,
        "summary_frames": int(round(SUMMARY_TIME_S * FPS)),
        "total_frames": total_active_frames + int(round(SUMMARY_TIME_S * FPS)),
        "poster_frame": poster_index,
        "policy": (
            "physical-time synchronized at 1x; completed panels hold their final "
            "active frame; one shared summary tail is appended"
        ),
    }


def main() -> int:
    args = parse_args()
    strict_manifest = args.strict_manifest.resolve()
    proposed_plus_dir = args.proposed_plus_dir.resolve()
    output = args.output.resolve()
    _relative(output)
    output.mkdir(parents=True, exist_ok=True)

    task, trial_path, frozen = _load_frozen_case(strict_manifest, args.pose_id)
    video_path = output / "comparison_strict_four_methods_2x2.mp4"
    poster_path = output / "comparison_strict_four_methods_2x2_poster.png"
    keyframe_path = output / "comparison_strict_four_methods_2x2_keyframe.png"
    manifest_path = output / "manifest.json"

    results: dict[str, RunResult] = {}
    verifications: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    component_records: dict[str, dict[str, Any]] = {}

    (
        paths["proposed_plus"],
        results["proposed_plus"],
        verifications["proposed_plus"],
        reused_record,
    ) = _load_reused_proposed_plus(
        proposed_plus_dir, task, frozen["proposed_plus"]
    )
    component_records["proposed_plus"] = {
        "generation": "reused verified strict explainer",
        "retained_after_composition": True,
        **reused_record,
    }

    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    regenerated = ("proposed", "outward", "entropy_grid")
    for method in regenerated:
        component_path = output / f"component_{method}.mp4"
        print(f"rendering strict {method} -> {component_path.name}", flush=True)
        result = _run_with_runtime(
            task,
            method,
            runtime,
            output_path=component_path,
        )
        result_payload = asdict(result)
        verification = _verify_or_raise(method, result_payload, frozen[method])
        results[method] = result
        verifications[method] = verification
        paths[method] = component_path
        component_records[method] = {
            "generation": "rerendered through run_m8_strict_complete_loop_audit._run_with_runtime",
            "renderer_only_extra_argument": "output_path",
            "artifact_before_deletion": _artifact(component_path),
            "retained_after_composition": False,
        }

    if set(results) != set(METHODS) or set(verifications) != set(METHODS):
        raise RuntimeError("incomplete method set before composition")
    if not all(item["all_fields_match"] for item in verifications.values()):
        raise RuntimeError("at least one rendered method does not match the frozen trial")

    composition = _compose(
        paths,
        results,
        task,
        verifications,
        video_path,
        poster_path,
        keyframe_path,
    )

    for method in regenerated:
        paths[method].unlink()
        component_records[method]["deleted_after_composition"] = True

    manifest = {
        "experiment": "M8_strict_four_method_selected_mechanism_matrix",
        "created_utc": _utc_now(),
        "pose_id": task.pose_id,
        "task": asdict(task),
        "source": {
            "strict_manifest": _artifact(strict_manifest),
            "frozen_trial": _artifact(trial_path),
            "strict_parameter_source": (
                "run_m8_strict_complete_loop_audit._run_with_runtime"
            ),
        },
        "method_order": list(METHODS),
        "methods": {
            method: {
                "display_name": DISPLAY_NAMES[method],
                "runtime_method": RUNTIME_METHOD[method],
                "strict_role": METHOD_ROLES[method],
                "frozen_result": {
                    field: frozen[method].get(field)
                    for field in (*EXACT_FIELDS, *FLOAT_FIELDS)
                },
                "verification": verifications[method],
                "component_video": component_records[method],
            }
            for method in METHODS
        },
        "contract": {
            "selected_mechanism_example_not_method_ranking": True,
            "distribution_claim_source": (
                "outputs/current/evaluations/m8_strict_24_complete_loop/report.json"
            ),
            "conditioned_on_ambiguous_top_contact": True,
            "same_frozen_initial_pose_contact_nuisance_and_sensor_seed": True,
            "same_hard_accumulated_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "same_capture_handoff_and_terminal_verifier": True,
            "online_true_center_or_contact_tags": False,
            "all_rendered_results_match_frozen_trial": True,
            "float_field_absolute_tolerance_s": TIME_ABS_TOL_S,
            "proposed_plus_video_reused_after_hash_and_result_verification": True,
            "regenerated_component_videos_deleted_after_composition": True,
        },
        "composition": composition,
        "claim_boundary": (
            "Selected same-start MuJoCo mechanism visualization from the frozen "
            "strict sobol_008 trial. It preserves Entropy-grid's faster finish "
            "in this case and is not a ranking, success-rate estimate, hardware "
            "claim, passivity proof, threaded-unscrewing result, or replacement "
            "for the paired 24-pose report."
        ),
        "artifacts": {
            "video": _artifact(video_path),
            "poster": _artifact(poster_path),
            "keyframe": _artifact(keyframe_path),
        },
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": _relative(output),
                "video": manifest["artifacts"]["video"],
                "all_frozen_fields_match": True,
                "component_videos_retained": ["proposed_plus"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
