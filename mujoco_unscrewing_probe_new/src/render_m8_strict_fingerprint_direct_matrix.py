#!/usr/bin/env python3
"""Render one frozen strict-M8 fingerprint/direct 2x2 mechanism matrix.

The four panels are fixed to the paper-facing comparison requested for the
group-meeting deck:

* Proposed+
* Proposed
* pure outward Archimedean search
* fingerprint direct-first

``--pose-id`` selects any task already frozen in the strict audit manifest.
All four panels are rerun through
``run_m8_strict_complete_loop_audit._run_with_runtime``.  Consequently this
renderer does not carry a second copy of policy, environment, force, budget,
or posterior parameters.  Every ``RunResult`` leaf is checked against the
frozen trial.  Only ``video_path`` and ``frame_count`` are allowed to change,
because they are the two declared observation-only outputs enabled by passing
``output_path`` to the shared runner.

This is a selected same-start mechanism visualization, not a success-rate or
method-ranking experiment.  The paired strict report remains the only source
for distribution-level claims.
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
import re
import textwrap
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


DEFAULT_STRICT_MANIFEST = (
    ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop/manifest.json"
)
DEFAULT_STRATIFICATION_REPORT = (
    ROOT
    / "outputs/current/evaluations/"
    / "m8_strict24_fingerprint_quality_stratification/report.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/current/active_comparison/strict_fingerprint_direct_matrices"
)
LEGACY_STRICT_FOUR_METHODS_OUTPUT = (
    ROOT / "outputs/current/active_comparison/strict_four_methods"
)

METHODS = ("proposed_plus", "proposed", "outward", "direct_first")
DISPLAY_NAMES = {
    "proposed_plus": "PROPOSED+ | adaptive shortcut + search reserve",
    "proposed": "PROPOSED | fingerprint shortcut + residual outward",
    "outward": "PURE OUTWARD | no fingerprint | Archimedean search",
    "direct_first": "FINGERPRINT DIRECT-FIRST | one ray | no retry/reset",
}
SHORT_NAMES = {
    "proposed_plus": "PROPOSED+",
    "proposed": "PROPOSED",
    "outward": "PURE OUTWARD",
    "direct_first": "DIRECT-FIRST",
}
COLORS = {
    "proposed_plus": (100, 245, 215),
    "proposed": (90, 225, 255),
    "outward": (255, 205, 105),
    "direct_first": (255, 150, 130),
}

# These are the only RunResult fields changed by enabling video observation on
# an otherwise identical rollout.  They are still explicitly validated below.
RENDER_ONLY_FIELDS = ("video_path", "frame_count")
FLOAT_ABS_TOL = 1.0e-9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a frozen strict Proposed+/Proposed/outward/direct-first "
            "2x2 matrix."
        )
    )
    parser.add_argument(
        "--strict-manifest", type=Path, default=DEFAULT_STRICT_MANIFEST
    )
    parser.add_argument(
        "--stratification-report",
        type=Path,
        default=DEFAULT_STRATIFICATION_REPORT,
        help=(
            "Post-hoc strict-24 fingerprint-quality stratification used only "
            "to justify the selected-case label; never a ranking source."
        ),
    )
    parser.add_argument("--pose-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. Defaults to strict_fingerprint_direct_matrices/"
            "<pose-id>."
        ),
    )
    parser.add_argument(
        "--artifact-stem",
        help="Filename stem; defaults to comparison_<pose-id>_fingerprint_direct_2x2.",
    )
    parser.add_argument(
        "--case-label",
        default="FROZEN SELECTED CASE",
        help="Short visible label such as MUTED FINGERPRINT or CLEAR FINGERPRINT.",
    )
    parser.add_argument(
        "--case-note",
        default="",
        help="Optional visible interpretation note; stored verbatim in the manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only same-named outputs in the selected output directory.",
    )
    return parser.parse_args(argv)


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


def _validate_artifact_stem(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            "artifact stem may contain only letters, numbers, dot, underscore, and dash"
        )
    return value


def _resolve_output_and_stem(args: argparse.Namespace) -> tuple[Path, str]:
    output = (
        args.output.resolve()
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / args.pose_id).resolve()
    )
    _relative(output)
    if output == LEGACY_STRICT_FOUR_METHODS_OUTPUT.resolve():
        raise ValueError(
            "refusing to overwrite the retained historical strict_four_methods artifact"
        )
    stem = args.artifact_stem or (
        f"comparison_{args.pose_id}_fingerprint_direct_2x2"
    )
    return output, _validate_artifact_stem(stem)


def _load_frozen_case(
    strict_manifest: Path, pose_id: str
) -> tuple[PoseTask, Path, dict[str, dict[str, Any]]]:
    manifest = json.loads(strict_manifest.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("tasks", []) if item.get("pose_id") == pose_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one task {pose_id!r} in {strict_manifest}, "
            f"found {len(matches)}"
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


def _load_stratification_selection(
    report_path: Path,
    *,
    pose_id: str,
    strict_manifest: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    design = report.get("analysis_design", {})
    if design.get("type") != "post_hoc_descriptive_stratification":
        raise ValueError("fingerprint-quality selection source is not post-hoc")
    if design.get("preregistered") is not False:
        raise ValueError("fingerprint-quality selection must remain non-preregistered")
    source_manifest = report.get("source", {}).get("manifest", {})
    if source_manifest.get("sha256") != _sha256(strict_manifest):
        raise RuntimeError(
            "fingerprint-quality stratification and strict manifest SHA differ"
        )

    memberships: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name, stratum in report.get("strata", {}).items():
        for member in stratum.get("members", []):
            if member.get("pose_id") == pose_id:
                memberships.append((name, stratum, member))
    if len(memberships) > 1:
        raise RuntimeError(f"{pose_id} belongs to multiple quality strata")
    if not memberships:
        return {
            "report": _artifact(report_path),
            "pose_id": pose_id,
            "stratum": None,
            "member_metrics": None,
            "post_hoc": True,
            "preregistered": False,
            "claim_boundary": report.get("claim_boundary"),
        }

    name, stratum, member = memberships[0]
    return {
        "report": _artifact(report_path),
        "pose_id": pose_id,
        "stratum": name,
        "member_metrics": member,
        "stratum_member_count": stratum.get("member_count"),
        "direct_first_observed_count_ratio": stratum.get("observed_count_ratio"),
        "post_hoc": True,
        "preregistered": False,
        "no_probability_estimate": bool(design.get("no_probability_estimate")),
        "no_significance_test": bool(design.get("no_significance_test")),
        "claim_boundary": report.get("claim_boundary"),
    }


def _compare_tree(
    expected: Any,
    actual: Any,
    *,
    path: str,
    mismatches: list[dict[str, Any]],
) -> int:
    """Compare a JSON-like tree and return the number of checked leaves."""

    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            mismatches.append(
                {"path": path, "expected": expected, "actual": actual, "reason": "type"}
            )
            return 1
        checked = 0
        for key in sorted(set(expected).union(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                mismatches.append(
                    {"path": child, "expected": "<missing>", "actual": actual[key], "reason": "extra"}
                )
                checked += 1
            elif key not in actual:
                mismatches.append(
                    {"path": child, "expected": expected[key], "actual": "<missing>", "reason": "missing"}
                )
                checked += 1
            else:
                checked += _compare_tree(
                    expected[key], actual[key], path=child, mismatches=mismatches
                )
        return checked

    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            mismatches.append(
                {"path": path, "expected": expected, "actual": actual, "reason": "type"}
            )
            return 1
        checked = 0
        if len(expected) != len(actual):
            mismatches.append(
                {
                    "path": f"{path}.length",
                    "expected": len(expected),
                    "actual": len(actual),
                    "reason": "length",
                }
            )
            checked += 1
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            checked += _compare_tree(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                mismatches=mismatches,
            )
        return checked

    if isinstance(expected, bool) or isinstance(actual, bool):
        match = type(expected) is type(actual) and expected == actual
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isnan(expected_float) or math.isnan(actual_float):
            match = math.isnan(expected_float) and math.isnan(actual_float)
        else:
            match = math.isclose(
                expected_float,
                actual_float,
                rel_tol=0.0,
                abs_tol=FLOAT_ABS_TOL,
            )
    else:
        match = type(expected) is type(actual) and expected == actual

    if not match:
        mismatches.append(
            {"path": path, "expected": expected, "actual": actual, "reason": "value"}
        )
    return 1


def _full_result_verification(
    method: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    component_path: Path,
) -> dict[str, Any]:
    substantive_actual = {
        key: value for key, value in actual.items() if key not in RENDER_ONLY_FIELDS
    }
    substantive_expected = {
        key: value for key, value in expected.items() if key not in RENDER_ONLY_FIELDS
    }
    mismatches: list[dict[str, Any]] = []
    checked_leaf_fields = _compare_tree(
        substantive_expected,
        substantive_actual,
        path="RunResult",
        mismatches=mismatches,
    )

    renderer_fields = {
        "video_path": {
            "frozen": expected.get("video_path"),
            "rendered": actual.get("video_path"),
            "expected_rendered": _relative(component_path),
            "valid": actual.get("video_path") == _relative(component_path),
            "reason": "output_path enables video observation only",
        },
        "frame_count": {
            "frozen": expected.get("frame_count"),
            "rendered": actual.get("frame_count"),
            "valid": (
                isinstance(actual.get("frame_count"), int)
                and int(actual["frame_count"]) > 0
            ),
            "reason": "output_path records emitted video frames only",
        },
    }
    renderer_fields_valid = all(item["valid"] for item in renderer_fields.values())
    expected_strict = _strict_bottom_success(expected)
    actual_strict = _strict_bottom_success(actual)
    strict_match = expected_strict == actual_strict
    all_match = not mismatches and renderer_fields_valid and strict_match
    return {
        "method": method,
        "comparison_scope": (
            "all RunResult leaves except two declared renderer-only fields; "
            "both renderer-only fields are independently validated"
        ),
        "excluded_from_value_equality": list(RENDER_ONLY_FIELDS),
        "checked_substantive_leaf_fields": checked_leaf_fields,
        "substantive_mismatch_count": len(mismatches),
        "substantive_mismatches": mismatches,
        "renderer_only_fields": renderer_fields,
        "strict_bottom_success": {
            "expected": expected_strict,
            "actual": actual_strict,
            "match": strict_match,
        },
        "all_fields_match_or_are_declared_renderer_outputs": all_match,
    }


def _verify_or_raise(
    method: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    component_path: Path,
) -> dict[str, Any]:
    verification = _full_result_verification(
        method, actual, expected, component_path=component_path
    )
    if not verification["all_fields_match_or_are_declared_renderer_outputs"]:
        paths = [
            item["path"] for item in verification["substantive_mismatches"][:12]
        ]
        raise RuntimeError(
            f"{method} render drifted from frozen trial; mismatches={paths}, "
            f"renderer_fields={verification['renderer_only_fields']}"
        )
    return verification


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


def _finished_status(result: RunResult) -> tuple[str, tuple[int, int, int]]:
    if _strict_bottom_success(asdict(result)):
        return "CAD BOTTOM", (140, 235, 165)
    if result.offline_capture_validated:
        return "CAPTURE", (140, 235, 165)
    if result.hard_stop:
        return "HARD STOP", (255, 125, 115)
    return "NO CAPTURE", (255, 190, 115)


def _executed_control_direction_error_deg(result: RunResult) -> float | None:
    """Return circular error for the direction actually consumed by control.

    ``RunResult.direction_error_deg`` is intentionally retained as the frozen
    top-bin diagnostic.  The improved policy executes the circular-mean
    ``control_direction_deg``, so a video summary must not relabel the former
    as an operational control error.
    """

    if result.control_direction_deg is None or result.true_direction_deg is None:
        return None
    delta = (
        float(result.control_direction_deg)
        - float(result.true_direction_deg)
        + 180.0
    ) % 360.0 - 180.0
    return abs(delta)


def _panel_canvas(
    frames: dict[str, np.ndarray],
    active_counts: dict[str, int],
    results: dict[str, RunResult],
    index: int,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 800), (8, 10, 13))
    draw = ImageDraw.Draw(canvas)
    positions = {
        "proposed_plus": (0, 0),
        "proposed": (640, 0),
        "outward": (0, 400),
        "direct_first": (640, 400),
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
            font=_font(11, bold=True),
        )
        if index >= active_counts[method] - 1:
            status, status_color = _finished_status(results[method])
        else:
            status, status_color = "ACTIVE", (230, 236, 242)
        status_font = _font(10, bold=True)
        bbox = draw.textbbox((0, 0), status, font=status_font)
        draw.text(
            (x + 625 - (bbox[2] - bbox[0]), y + 11),
            status,
            fill=status_color,
            font=status_font,
        )
    return np.asarray(canvas)


def _summary_frame(
    results: dict[str, RunResult],
    task: PoseTask,
    verifications: dict[str, dict[str, Any]],
    *,
    case_label: str,
    case_note: str,
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
        (72, 61),
        f"STRICT SAME-START M8 | {task.pose_id} | {case_label}",
        fill=(245, 248, 252),
        font=_font(19, bold=True),
    )
    draw.text(
        (72, 99),
        (
            f"initial XY = ({1000.0 * task.x_m:+.3f}, "
            f"{1000.0 * task.y_m:+.3f}) mm  |  hard active budget = "
            f"{ACTIVE_TIME_BUDGET_S:.0f} s"
        ),
        fill=(173, 199, 221),
        font=_font(12),
    )

    headers = (
        ("method", 74),
        ("result", 240),
        ("control err", 475),
        ("capture", 570),
        ("CAD bottom", 690),
        ("active", 825),
        ("XY path", 930),
        ("frozen", 1060),
    )
    for label, x in headers:
        draw.text((x, 158), label, fill=(148, 174, 197), font=_font(11, bold=True))
    draw.line((70, 186, 1208, 186), fill=(120, 145, 166, 150), width=1)

    y = 211
    for method in METHODS:
        result = results[method]
        control_error_deg = _executed_control_direction_error_deg(result)
        values = (
            SHORT_NAMES[method],
            result.result,
            "n/a" if control_error_deg is None else f"{control_error_deg:.1f}deg",
            "yes" if result.offline_capture_validated else "no",
            "yes" if result.bottom_reached else "no",
            f"{result.total_active_time_s:.3f}s",
            f"{result.actual_xy_path_mm:.2f}mm",
            (
                "MATCH"
                if verifications[method][
                    "all_fields_match_or_are_declared_renderer_outputs"
                ]
                else "MISMATCH"
            ),
        )
        for (header, x), value in zip(headers, values):
            fill = COLORS[method] if header == "method" else (232, 239, 246)
            draw.text(
                (x, y),
                str(value),
                fill=fill,
                font=_font(10, bold=header == "method"),
            )
        draw.line((70, y + 31, 1208, y + 31), fill=(85, 104, 122, 100), width=1)
        y += 62

    note_lines: list[str] = []
    if case_note.strip():
        note_lines.extend(textwrap.wrap(case_note.strip(), width=112))
    note_lines.extend(
        [
            "All four panels use one frozen pose/nuisance/seed and the identical shared strict runner contract.",
            "Proposed+, Proposed and Direct-first use the byte-identical conical fingerprint; Outward reads no fingerprint.",
            "Direct-first is exactly one inferred ray with no simulator reset/retry. A miss is a valid mechanism outcome.",
            "This selected rollout explains shortcut length and first-ray capture; paired success rates require the strict batch.",
        ]
    )
    draw.text((72, 490), "claim boundary", fill=(255, 220, 145), font=_font(13, bold=True))
    y = 523
    for note in note_lines[:6]:
        draw.text((86, y), note, fill=(213, 226, 237), font=_font(10))
        y += 31
    return np.asarray(image)


def _compose(
    paths: dict[str, Path],
    results: dict[str, RunResult],
    task: PoseTask,
    verifications: dict[str, dict[str, Any]],
    *,
    case_label: str,
    case_note: str,
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
            canvas = _panel_canvas(last_frames, active_counts, results, index)
            writer.append_data(canvas)
            if index == poster_index:
                imageio.imwrite(poster_path, canvas)
                poster_written = True

        if not poster_written:
            imageio.imwrite(poster_path, canvas)
        keyframe = _summary_frame(
            results,
            task,
            verifications,
            case_label=case_label,
            case_note=case_note,
        )
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
            "physical-time synchronized at 1x; a finished panel holds its final "
            "active frame; one shared audit summary tail is appended"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strict_manifest = args.strict_manifest.resolve()
    stratification_report = args.stratification_report.resolve()
    output, stem = _resolve_output_and_stem(args)
    output.mkdir(parents=True, exist_ok=True)

    task, trial_path, frozen = _load_frozen_case(strict_manifest, args.pose_id)
    stratification_selection = _load_stratification_selection(
        stratification_report,
        pose_id=args.pose_id,
        strict_manifest=strict_manifest,
    )
    video_path = output / f"{stem}.mp4"
    poster_path = output / f"{stem}_poster.png"
    keyframe_path = output / f"{stem}_keyframe.png"
    manifest_path = output / "manifest.json"
    targets = (video_path, poster_path, keyframe_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    results: dict[str, RunResult] = {}
    rendered_payloads: dict[str, dict[str, Any]] = {}
    verifications: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    component_records: dict[str, dict[str, Any]] = {}

    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(MODEL_ROOT)
    for method in METHODS:
        component_path = output / f".{stem}_{method}_component.mp4"
        if component_path.exists():
            if not args.overwrite:
                raise FileExistsError(component_path)
            component_path.unlink()
        print(
            f"rendering frozen {task.pose_id} strict {method} -> "
            f"{component_path.name}",
            flush=True,
        )
        result = _run_with_runtime(
            task,
            method,
            runtime,
            output_path=component_path,
        )
        actual = asdict(result)
        verification = _verify_or_raise(
            method,
            actual,
            frozen[method],
            component_path=component_path,
        )
        results[method] = result
        rendered_payloads[method] = actual
        verifications[method] = verification
        paths[method] = component_path
        component_records[method] = {
            "generation": (
                "rerendered through "
                "run_m8_strict_complete_loop_audit._run_with_runtime"
            ),
            "renderer_only_extra_argument": "output_path",
            "artifact_before_deletion": _artifact(component_path),
            "retained_after_composition": False,
        }

    fingerprint_hashes = {
        rendered_payloads[method].get("fingerprint_sha256")
        for method in ("proposed_plus", "proposed", "direct_first")
    }
    if None in fingerprint_hashes or len(fingerprint_hashes) != 1:
        raise RuntimeError(
            "Proposed+, Proposed and Direct-first did not use one identical fingerprint"
        )
    if rendered_payloads["outward"].get("fingerprint_sha256") is not None:
        raise RuntimeError("pure outward unexpectedly consumed a fingerprint")

    composition = _compose(
        paths,
        results,
        task,
        verifications,
        case_label=args.case_label,
        case_note=args.case_note,
        video_path=video_path,
        poster_path=poster_path,
        keyframe_path=keyframe_path,
    )

    for method in METHODS:
        paths[method].unlink()
        component_records[method]["deleted_after_composition"] = True

    manifest = {
        "experiment": "M8_strict_fingerprint_direct_selected_mechanism_matrix",
        "created_utc": _utc_now(),
        "pose_id": task.pose_id,
        "case_label": args.case_label,
        "case_note": args.case_note,
        "task": asdict(task),
        "source": {
            "strict_manifest": _artifact(strict_manifest),
            "frozen_trial": _artifact(trial_path),
            "post_hoc_fingerprint_quality_selection": stratification_selection,
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
                "frozen_result": frozen[method],
                "rendered_result": rendered_payloads[method],
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
            "all_substantive_runresult_fields_match_frozen_trial": True,
            "declared_renderer_only_fields": list(RENDER_ONLY_FIELDS),
            "renderer_only_fields_independently_validated": True,
            "float_absolute_tolerance": FLOAT_ABS_TOL,
            "shared_fingerprint_for_proposed_plus_proposed_direct_first": True,
            "pure_outward_reads_no_fingerprint": True,
            "direct_first_has_no_retry_or_simulator_reset": True,
            "summary_direction_metric": (
                "circular error of executed control_direction_deg versus "
                "true_direction_deg; frozen direction_error_deg remains the "
                "top-bin diagnostic"
            ),
            "fingerprint_quality_label_is_post_hoc_descriptive": True,
            "fingerprint_quality_label_is_not_probability_or_significance": True,
            "all_component_videos_deleted_after_composition": True,
        },
        "composition": composition,
        "claim_boundary": (
            "Selected same-start MuJoCo mechanism visualization from one frozen "
            "strict trial. It is not a ranking, success-rate estimate, hardware "
            "claim, passivity proof, threaded-unscrewing result, or replacement "
            "for the paired strict batch. Fingerprint clarity labels are "
            "interpretations that must be justified by separately declared "
            "selection metrics."
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
                "pose_id": task.pose_id,
                "output": _relative(output),
                "video": manifest["artifacts"]["video"],
                "all_frozen_fields_accounted_for": True,
                "shared_fingerprint": next(iter(fingerprint_hashes)),
                "component_videos_retained": [],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
