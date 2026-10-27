#!/usr/bin/env python3
"""Offline QC and observability audit for the first complete M8/M10/M12 runs.

This script is deliberately disconnected from ROS, KST, Bota subscribers, and
all robot command interfaces.  The three input campaigns are an explicit
allowlist so an old resume lineage or a quarantined tooling/program mismatch
cannot silently enter normalization, fitting, or evaluation.

The audit has three intentionally small stages:

1. verify immutable evidence and acquisition quality;
2. fit direction models from ``clean_candidate`` only;
3. perform the first development reveal of ``holdout`` and evaluate CAD
   geometry.  Those points are thereafter marked consumed and must not be
   reused for model selection.

The first model is a simulation-inspired 48-D Fourier ridge baseline fitted on
real clean samples; it is not the frozen simulation CNN or its full 64-D input
contract.  The second is a real-domain, state-conditioned circular template
using the lateral force channels that are most repeatable in the sentinel
scans.  It is a compact observability probe, not a promoted hardware controller.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REAL_ROOT / "offline_analysis" / "multisize_real_fingerprint_v1"
)
COMPLETE_STATUS = "COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT"

# Authority is intentionally explicit.  Do not replace this with a glob.
RUN_ALLOWLIST = {
    "M12": REAL_ROOT
    / "data"
    / "M12_conical_fingerprints_26pt_20260728_175502_318",
    "M10": REAL_ROOT
    / "data"
    / "M10_conical_fingerprints_26pt_20260728_152808_877",
    "M8": REAL_ROOT
    / "data"
    / "M8_conical_fingerprints_26pt_20260728_161249_458",
}
KNOWN_REJECTED_RUN = (
    REAL_ROOT
    / "data"
    / "M8_conical_fingerprints_26pt_20260728_172108_490"
)

PARTITION_COUNTS = {
    "clean_candidate": 16,
    "holdout": 4,
    "quarantine": 4,
    "sentinel": 2,
}
CONTACT_STATES = (
    "supported_search",
    "double_overhang",
    "outer_partial_support",
)
WORLD_CHANNELS = (
    "fx_world_star_n",
    "fy_world_star_n",
    "fz_world_star_n",
    "tx_world_star_nm",
    "ty_world_star_nm",
    "tz_world_star_nm",
)
SIMULATION_SHAPE_CHANNELS = (2, 3, 4, 6)  # Fz, Tx, Ty, command-z
TEMPLATE_CHANNELS = (0, 1)  # real-domain lateral force
PHASE_BINS = 360


@dataclass
class Sample:
    size: str
    run_dir: Path
    attempt: dict[str, Any]
    phase_sequence: np.ndarray
    scan_source_stamp_s: np.ndarray
    scan_wrench_age_s: np.ndarray
    maximum_phase_gap_rad: float

    @property
    def partition(self) -> str:
        return str(self.attempt["effective_partition"])

    @property
    def contact_state(self) -> str:
        return str(self.attempt["contact_state"])

    @property
    def direction_deg(self) -> float:
        return float(self.attempt["true_direction_fixture_deg"])

    @property
    def measured_offset_mm(self) -> np.ndarray:
        return np.asarray(
            self.attempt["measured_contact_fixture_xy_mm"], dtype=float
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(
    path: Path, fieldnames: Sequence[str], records: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value {value!r}")
    return number


def circular_error_deg(first: float | np.ndarray, second: float | np.ndarray):
    first_value = np.asarray(first, dtype=float)
    second_value = np.asarray(second, dtype=float)
    result = np.abs(
        (first_value - second_value + 180.0) % 360.0 - 180.0
    )
    if result.ndim == 0:
        return float(result)
    return result


def periodic_resample(
    phase_rad: np.ndarray, values: np.ndarray, bins: int = PHASE_BINS
) -> np.ndarray:
    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * math.pi)
    samples = np.asarray(values, dtype=float)
    if phase.ndim != 1 or samples.ndim != 2:
        raise ValueError("phase must be 1-D and values must be 2-D")
    if len(phase) != len(samples):
        raise ValueError("phase and value sample counts differ")
    finite = np.isfinite(phase) & np.all(np.isfinite(samples), axis=1)
    phase = phase[finite]
    samples = samples[finite]
    if len(phase) < 32:
        raise ValueError("fingerprint has fewer than 32 finite samples")
    order = np.argsort(phase)
    phase = phase[order]
    samples = samples[order]
    phase, unique = np.unique(phase, return_index=True)
    samples = samples[unique]
    if len(phase) < 32:
        raise ValueError("fingerprint has fewer than 32 unique phases")
    extended_phase = np.concatenate(
        (phase[-1:] - 2.0 * math.pi, phase, phase[:1] + 2.0 * math.pi)
    )
    grid = np.linspace(0.0, 2.0 * math.pi, bins, endpoint=False)
    output = []
    for index in range(samples.shape[1]):
        extended_values = np.concatenate(
            (
                samples[-1:, index],
                samples[:, index],
                samples[:1, index],
            )
        )
        output.append(np.interp(grid, extended_phase, extended_values))
    return np.vstack(output)


def sample_file_path(attempt: dict[str, Any], key: str) -> Path:
    path = Path(str(attempt[key]))
    if not path.is_absolute():
        path = REAL_ROOT / path
    return path.resolve()


def verify_attempt_hashes(attempt: dict[str, Any]) -> int:
    mapping = {
        "attempt_mat_sha256": "mat_path",
        "baseline_csv_sha256": "baseline_csv_path",
        "contact_settle_csv_sha256": "contact_settle_csv_path",
        "cone_csv_sha256": "cone_csv_path",
    }
    count = 0
    hashes = attempt["file_hashes"]
    for hash_key, path_key in mapping.items():
        path = sample_file_path(attempt, path_key)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(hashes[hash_key])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{attempt['sample_id']} {path.name} hash mismatch: "
                f"{actual} != {expected}"
            )
        count += 1
    return count


def load_scan(
    attempt: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    path = sample_file_path(attempt, "cone_csv_path")
    all_rows = read_csv(path)
    rows = [
        row
        for row in all_rows
        if row.get("stage") == "SCAN"
        and row.get("phase_rad") not in ("", "NaN", None)
    ]
    if len(rows) != 240:
        raise ValueError(
            f"{attempt['sample_id']} has {len(rows)} SCAN rows, expected 240"
        )
    phase = np.asarray([finite_float(row["phase_rad"]) for row in rows])
    world = np.asarray(
        [
            [finite_float(row[name]) for name in WORLD_CHANNELS]
            for row in rows
        ]
    )
    command_z = np.asarray(
        [finite_float(row["cmd_z_mm"]) for row in rows]
    )[:, None]
    source_stamp = np.asarray(
        [finite_float(row["source_stamp_s"]) for row in rows]
    )
    wrench_age = np.asarray(
        [finite_float(row["wrench_age_s"]) for row in rows]
    )
    if np.any(np.diff(source_stamp) <= 0.0):
        raise ValueError(
            f"{attempt['sample_id']} source timestamps are not strictly fresh"
        )
    if np.any(wrench_age < 0.0):
        raise ValueError(f"{attempt['sample_id']} has negative wrench age")
    sorted_phase = np.sort(np.mod(phase, 2.0 * math.pi))
    cyclic_gaps = np.diff(
        np.concatenate((sorted_phase, sorted_phase[:1] + 2.0 * math.pi))
    )
    maximum_phase_gap_rad = float(np.max(cyclic_gaps))
    if maximum_phase_gap_rad > math.radians(2.0):
        raise ValueError(
            f"{attempt['sample_id']} cyclic phase gap exceeds 2 degrees"
        )
    sequence = periodic_resample(
        phase, np.column_stack((world, command_z)), bins=PHASE_BINS
    )
    return sequence, source_stamp, wrench_age, maximum_phase_gap_rad


def validate_campaign(size: str, run_dir: Path) -> tuple[list[Sample], dict]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    for flag in ("DO_NOT_TRAIN.flag", "RUN_QUARANTINE.json"):
        if (run_dir / flag).exists():
            raise ValueError(f"allowlisted run unexpectedly carries {flag}")
    campaign = read_json(run_dir / "sampling_campaign.json")
    summary = read_json(run_dir / "summary.json")
    if campaign["size"] != size or summary["size"] != size:
        raise ValueError(f"{run_dir.name} size identity mismatch")
    if campaign["status"] != COMPLETE_STATUS:
        raise ValueError(f"{run_dir.name} is not complete")
    if summary["completion_status"] != COMPLETE_STATUS:
        raise ValueError(f"{run_dir.name} summary is not complete")
    if not bool(summary["task_completed"]):
        raise ValueError(f"{run_dir.name} task_completed is false")
    if int(campaign["resume_generation"]) != 0 or bool(
        campaign["resume_enabled"]
    ):
        raise ValueError(f"{run_dir.name} is not a fresh lineage")
    for key in (
        "attempts_completed",
        "attempt_records_persisted",
        "cones_completed",
        "points_committed",
        "original_retracts_verified",
        "motion_lifecycles_closed",
    ):
        if int(campaign[key]) != 26:
            raise ValueError(f"{run_dir.name} {key} != 26")
    attempts = list(campaign["attempts"])
    if len(attempts) != 26:
        raise ValueError(f"{run_dir.name} attempt record count != 26")
    sample_ids = [str(attempt["sample_id"]) for attempt in attempts]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{run_dir.name} has duplicate sample IDs")
    for path_key in (
        "json_path",
        "mat_path",
        "baseline_csv_path",
        "contact_settle_csv_path",
        "cone_csv_path",
    ):
        paths = [sample_file_path(attempt, path_key) for attempt in attempts]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{run_dir.name} has duplicate {path_key}")
        if any(run_dir.resolve() not in path.parents for path in paths):
            raise ValueError(
                f"{run_dir.name} {path_key} escapes the locked run directory"
            )

    plan_path = Path(str(campaign["sampling_plan_path"])).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if sha256_file(plan_path) != str(campaign["sampling_plan_sha256"]):
        raise ValueError(f"{run_dir.name} sampling plan hash mismatch")
    plan = read_json(plan_path)
    if int(plan["attempt_count"]) != 26:
        raise ValueError(f"{run_dir.name} sampling plan count != 26")
    plan_by_index = {
        int(point["attempt_index"]): point for point in plan["points"]
    }
    for attempt in attempts:
        plan_point = plan_by_index[int(attempt["source_attempt_index"])]
        if attempt["effective_partition"] != attempt["requested_partition"]:
            raise ValueError(
                f"{attempt['sample_id']} fresh-run partition was rewritten"
            )
        expected = {
            "sample_id": plan_point["sample_id"],
            "requested_partition": plan_point["partition"],
            "contact_state": plan_point["contact_state"],
            "cone_sweep_subtype": plan_point["cone_sweep_subtype"],
        }
        for key, expected_value in expected.items():
            if attempt[key] != expected_value:
                raise ValueError(
                    f"{attempt['sample_id']} disagrees with plan field {key}"
                )
        attempt_json = read_json(sample_file_path(attempt, "json_path"))
        for key in expected:
            if attempt_json[key] != attempt[key]:
                raise ValueError(
                    f"{attempt['sample_id']} campaign/attempt JSON {key} mismatch"
                )
        if attempt_json["effective_partition"] != attempt["effective_partition"]:
            raise ValueError(
                f"{attempt['sample_id']} effective partition JSON mismatch"
            )
    partition_count = {
        partition: sum(
            attempt["effective_partition"] == partition
            for attempt in attempts
        )
        for partition in PARTITION_COUNTS
    }
    if partition_count != PARTITION_COUNTS:
        raise ValueError(
            f"{run_dir.name} partition contract mismatch {partition_count}"
        )

    samples: list[Sample] = []
    hash_checks = 0
    maximum_wrench_age_s = 0.0
    maximum_source_interval_s = 0.0
    baseline_rows = []
    for attempt in attempts:
        hash_checks += verify_attempt_hashes(attempt)
        baseline = read_csv(sample_file_path(attempt, "baseline_csv_path"))
        baseline_rows.append(len(baseline))
        if len(baseline) != 30:
            raise ValueError(
                f"{attempt['sample_id']} baseline count {len(baseline)} != 30"
            )
        sequence, source_stamp, wrench_age, maximum_phase_gap_rad = load_scan(
            attempt
        )
        maximum_wrench_age_s = max(
            maximum_wrench_age_s, float(np.max(wrench_age))
        )
        maximum_source_interval_s = max(
            maximum_source_interval_s,
            float(np.max(np.diff(source_stamp))),
        )
        if not bool(attempt["point_air_baseline_valid"]):
            raise ValueError(
                f"{attempt['sample_id']} point air baseline is invalid"
            )
        if not bool(attempt["contact_settle_passed"]):
            raise ValueError(
                f"{attempt['sample_id']} contact settle was not confirmed"
            )
        if float(attempt["phase_coverage_fraction"]) < 0.98:
            raise ValueError(
                f"{attempt['sample_id']} phase coverage is below 0.98"
            )
        if bool(attempt["capture_during_cone"]):
            raise ValueError(
                f"{attempt['sample_id']} captured during acquisition cone"
            )
        if bool(attempt["automatic_training_admission"]):
            raise ValueError(
                f"{attempt['sample_id']} was unexpectedly admitted online"
            )
        expected_offline_candidate = (
            attempt["effective_partition"] == "clean_candidate"
        )
        if bool(attempt["offline_training_candidate"]) != expected_offline_candidate:
            raise ValueError(
                f"{attempt['sample_id']} offline candidate role mismatch"
            )
        if not (
            bool(attempt["cone_completed"])
            and bool(attempt["cone_physical_schedule_completed"])
            and bool(attempt["cone_direction_fingerprint_valid"])
            and bool(attempt["cone_routing_fingerprint_valid"])
            and bool(attempt["safe_retract_completed"])
            and bool(attempt["immutable_evidence_committed"])
        ):
            raise ValueError(
                f"{attempt['sample_id']} has incomplete acquisition evidence"
            )
        samples.append(
            Sample(
                size=size,
                run_dir=run_dir,
                attempt=attempt,
                phase_sequence=sequence,
                scan_source_stamp_s=source_stamp,
                scan_wrench_age_s=wrench_age,
                maximum_phase_gap_rad=maximum_phase_gap_rad,
            )
        )

    quality = {
        "size": size,
        "run_dir": str(run_dir),
        "run_session_id": campaign["run_session_id"],
        "status": campaign["status"],
        "resume_generation": campaign["resume_generation"],
        "attempts": len(samples),
        "partition_counts": partition_count,
        "hash_checks": hash_checks,
        "hash_mismatches": 0,
        "baseline_rows_min": min(baseline_rows),
        "baseline_rows_max": max(baseline_rows),
        "scan_rows_per_attempt": 240,
        "phase_bins_after_resample": PHASE_BINS,
        "maximum_wrench_age_s": maximum_wrench_age_s,
        "maximum_source_interval_s": maximum_source_interval_s,
        "maximum_command_gap_s": max(
            float(sample.attempt["cone_maximum_command_gap_s"])
            for sample in samples
        ),
        "minimum_phase_coverage_fraction": min(
            float(sample.attempt["phase_coverage_fraction"])
            for sample in samples
        ),
        "maximum_cyclic_phase_gap_deg": math.degrees(
            max(sample.maximum_phase_gap_rad for sample in samples)
        ),
        "point_air_baseline_valid_count": sum(
            bool(sample.attempt["point_air_baseline_valid"])
            for sample in samples
        ),
        "contact_settle_passed_count": sum(
            bool(sample.attempt["contact_settle_passed"])
            for sample in samples
        ),
        "preferred_stable_load_count": sum(
            sample.attempt["contact_settle_quality_grade"]
            == "PREFERRED_STABLE_LOAD"
            for sample in samples
        ),
        "automatic_training_admission_count": sum(
            bool(sample.attempt["automatic_training_admission"])
            for sample in samples
        ),
        "offline_training_candidate_count": sum(
            bool(sample.attempt["offline_training_candidate"])
            for sample in samples
        ),
        "maximum_sampling_error_mm": max(
            float(sample.attempt["measured_sampling_error_mm"])
            for sample in samples
        ),
        "maximum_label_deviation_deg": max(
            float(sample.attempt["measured_label_direction_deviation_deg"])
            for sample in samples
        ),
        "direction_fingerprint_valid_count": sum(
            bool(sample.attempt["cone_direction_fingerprint_valid"])
            for sample in samples
        ),
        "training_fingerprint_valid_count": sum(
            bool(sample.attempt["cone_training_fingerprint_valid"])
            for sample in samples
        ),
        "training_flag_interpretation": (
            "false because DirectServo intentionally performed no in-loop KST "
            "pose query; it is not evidence that the fresh Bota phase trace is "
            "invalid"
        ),
    }
    return samples, quality


def channel_standardize(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    mean = np.mean(values, axis=1, keepdims=True)
    standard_deviation = np.std(values, axis=1, keepdims=True)
    return (values - mean) / np.maximum(standard_deviation, 1.0e-9)


def circular_smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    half = window // 2
    extended = np.concatenate((vector[-half:], vector, vector[:half]))
    return np.convolve(
        extended, np.ones(window, dtype=float) / window, mode="valid"
    )


def simulation_fourier_features(sample: Sample, harmonics: int = 6) -> np.ndarray:
    parts = []
    for index in SIMULATION_SHAPE_CHANNELS:
        values = circular_smooth(sample.phase_sequence[index])
        values = (values - np.mean(values)) / max(float(np.std(values)), 1.0e-9)
        spectrum = np.fft.rfft(values) / len(values)
        parts.extend(spectrum[1 : harmonics + 1].real)
        parts.extend(spectrum[1 : harmonics + 1].imag)
    return np.asarray(parts, dtype=float)


def fit_ridge_direction(
    training: list[Sample], alpha: float
) -> dict[str, np.ndarray | float]:
    if not training or any(
        sample.partition != "clean_candidate" for sample in training
    ):
        raise ValueError("ridge fitting accepts clean_candidate samples only")
    features = np.vstack(
        [simulation_fourier_features(sample) for sample in training]
    )
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std[feature_std < 1.0e-8] = 1.0
    normalized = (features - feature_mean) / feature_std
    angle = np.deg2rad([sample.direction_deg for sample in training])
    target = np.column_stack((np.cos(angle), np.sin(angle)))
    target_mean = np.mean(target, axis=0)
    centered_target = target - target_mean
    if normalized.shape[1] > normalized.shape[0]:
        weights = normalized.T @ np.linalg.solve(
            normalized @ normalized.T
            + float(alpha) * np.eye(len(normalized)),
            centered_target,
        )
    else:
        weights = np.linalg.solve(
            normalized.T @ normalized
            + float(alpha) * np.eye(normalized.shape[1]),
            normalized.T @ centered_target,
        )
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "weights": weights,
        "alpha": float(alpha),
    }


def predict_ridge(model: dict[str, Any], samples: list[Sample]) -> np.ndarray:
    features = np.vstack(
        [simulation_fourier_features(sample) for sample in samples]
    )
    normalized = (
        features - np.asarray(model["feature_mean"])
    ) / np.asarray(model["feature_std"])
    vector = (
        normalized @ np.asarray(model["weights"])
        + np.asarray(model["target_mean"])
    )
    return np.mod(np.rad2deg(np.arctan2(vector[:, 1], vector[:, 0])), 360.0)


def select_ridge_alpha(clean: list[Sample]) -> tuple[float, dict[str, Any]]:
    candidates = np.logspace(-3.0, 3.0, 13)
    records = []
    for alpha in candidates:
        errors = []
        for held_size in RUN_ALLOWLIST:
            training = [sample for sample in clean if sample.size != held_size]
            testing = [sample for sample in clean if sample.size == held_size]
            model = fit_ridge_direction(training, float(alpha))
            predicted = predict_ridge(model, testing)
            errors.extend(
                circular_error_deg(
                    predicted,
                    np.asarray([sample.direction_deg for sample in testing]),
                )
            )
        errors_array = np.asarray(errors)
        records.append(
            {
                "alpha": float(alpha),
                "median_error_deg": float(np.median(errors_array)),
                "p90_error_deg": float(np.quantile(errors_array, 0.90)),
            }
        )
    selected = min(
        records,
        key=lambda item: item["median_error_deg"]
        + 0.20 * item["p90_error_deg"],
    )
    return float(selected["alpha"]), {
        "selection_policy": (
            "clean-only leave-one-size-out; holdout never participates"
        ),
        "candidates": records,
        "selected": selected,
    }


def fractional_roll(values: np.ndarray, shift_bins: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    frequencies = np.fft.fftfreq(vector.shape[-1])
    factor = np.exp(-2j * math.pi * frequencies * float(shift_bins))
    return np.fft.ifft(np.fft.fft(vector) * factor).real


def align_lateral_force(
    sequence: np.ndarray, direction_deg: float
) -> np.ndarray:
    lateral = np.asarray(sequence[list(TEMPLATE_CHANNELS)], dtype=float)
    shifted = np.vstack(
        [fractional_roll(channel, direction_deg) for channel in lateral]
    )
    angle = math.radians(-float(direction_deg))
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    return channel_standardize(rotation @ shifted)


def fit_template_model(clean: list[Sample]) -> dict[str, np.ndarray]:
    if not clean or any(
        sample.partition != "clean_candidate" for sample in clean
    ):
        raise ValueError("template fitting accepts clean_candidate samples only")
    templates: dict[str, np.ndarray] = {}
    for size in RUN_ALLOWLIST:
        for state in CONTACT_STATES:
            selected = [
                sample
                for sample in clean
                if sample.size == size and sample.contact_state == state
            ]
            if len(selected) < 4:
                raise ValueError(f"{size}/{state} has too few clean templates")
            aligned = np.stack(
                [
                    align_lateral_force(
                        sample.phase_sequence, sample.direction_deg
                    )
                    for sample in selected
                ]
            )
            templates[f"{size}__{state}"] = channel_standardize(
                np.mean(aligned, axis=0)
            )
    return templates


def template_score_curve(
    template: np.ndarray, sample: Sample
) -> np.ndarray:
    score = []
    for direction_deg in range(360):
        aligned = align_lateral_force(
            sample.phase_sequence, float(direction_deg)
        )
        score.append(float(np.mean(aligned * template)))
    return np.asarray(score)


def separated_peaks(
    score: np.ndarray, count: int = 5, separation_deg: float = 20.0
) -> list[int]:
    output: list[int] = []
    for index in np.argsort(-np.asarray(score)):
        direction = int(index)
        if all(
            circular_error_deg(direction, selected) >= separation_deg
            for selected in output
        ):
            output.append(direction)
        if len(output) == count:
            break
    return output


def direction_metrics(errors: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    return {
        "count": int(len(values)),
        "median_error_deg": float(np.median(values)),
        "p90_error_deg": float(np.quantile(values, 0.90)),
        "within_7p5_fraction": float(np.mean(values <= 7.5)),
        "within_15_fraction": float(np.mean(values <= 15.0)),
        "within_30_fraction": float(np.mean(values <= 30.0)),
        "catastrophic_gt90_fraction": float(np.mean(values > 90.0)),
        "catastrophic_gt90_count": int(np.sum(values > 90.0)),
    }


def evaluate_template_holdout(
    templates: dict[str, np.ndarray], holdout: list[Sample]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(sample.partition != "holdout" for sample in holdout):
        raise ValueError("template holdout evaluation accepts holdout only")
    records = []
    for sample in holdout:
        key = f"{sample.size}__{sample.contact_state}"
        score = template_score_curve(templates[key], sample)
        peaks = separated_peaks(score, count=5, separation_deg=20.0)
        peak_errors = [
            circular_error_deg(direction, sample.direction_deg)
            for direction in peaks
        ]
        records.append(
            {
                "size": sample.size,
                "sample_id": sample.attempt["sample_id"],
                "contact_state": sample.contact_state,
                "true_direction_deg": sample.direction_deg,
                "top1_direction_deg": peaks[0],
                "top1_error_deg": peak_errors[0],
                "top2_direction_deg": peaks[1],
                "top2_error_deg": peak_errors[1],
                "best_of_top2_error_deg": min(peak_errors[:2]),
                "top5_directions_deg": ";".join(map(str, peaks)),
                "top5_errors_deg": ";".join(
                    f"{value:.6f}" for value in peak_errors
                ),
                "state_conditioning": (
                    "ground-truth CAD state for observability upper bound"
                ),
            }
        )
    top1 = [record["top1_error_deg"] for record in records]
    top2_best = [record["best_of_top2_error_deg"] for record in records]
    by_size = {}
    for size in RUN_ALLOWLIST:
        selected = [
            record for record in records if record["size"] == size
        ]
        by_size[size] = {
            "top1": direction_metrics(
                [record["top1_error_deg"] for record in selected]
            ),
            "best_of_top2": direction_metrics(
                [record["best_of_top2_error_deg"] for record in selected]
            ),
        }
    return records, {
        "role": (
            "clean-trained state-conditioned circular-template observability "
            "upper bound from the first development holdout reveal; not a "
            "hardware promotion or final locked audit"
        ),
        "training_channels": ["Fx_WORLD_star_N", "Fy_WORLD_star_N"],
        "training_partition": "clean_candidate_only",
        "evaluation_partition": "holdout_only",
        "state_input_boundary": (
            "true CAD contact state is used only to isolate direction "
            "observability; deployable fusion must marginalize calibrated "
            "state likelihood"
        ),
        "top1": direction_metrics(top1),
        "best_of_top2": direction_metrics(top2_best),
        "best_of_top2_interpretation": (
            "oracle posterior-support coverage only; there is no online oracle "
            "that selects the better of the two modes"
        ),
        "by_size": by_size,
    }


def pairwise_channel_correlation(sequences: list[np.ndarray]) -> float:
    values = []
    for first_index in range(len(sequences)):
        for second_index in range(first_index + 1, len(sequences)):
            first = channel_standardize(sequences[first_index])
            second = channel_standardize(sequences[second_index])
            values.append(float(np.mean(first * second)))
    return float(np.mean(values)) if values else 0.0


def align_simulation_channels(
    residual: np.ndarray, direction_deg: float
) -> np.ndarray:
    selected = residual[[2, 3, 4]].copy()  # Fz, Tx, Ty
    selected = np.vstack(
        [fractional_roll(channel, direction_deg) for channel in selected]
    )
    angle = math.radians(-direction_deg)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    selected[1:3] = rotation @ selected[1:3]
    return selected


def c6_alignment_report(clean: list[Sample]) -> dict[str, Any]:
    report = {}
    for size_index, size in enumerate(RUN_ALLOWLIST):
        size_samples = [sample for sample in clean if sample.size == size]
        residual_by_id = {}
        for sample in size_samples:
            others = [
                other.phase_sequence
                for other in size_samples
                if other.attempt["sample_id"] != sample.attempt["sample_id"]
            ]
            residual_by_id[sample.attempt["sample_id"]] = (
                sample.phase_sequence - np.median(np.stack(others), axis=0)
            )
        raw_groups = []
        aligned_groups = []
        for state in CONTACT_STATES:
            state_samples = [
                sample
                for sample in size_samples
                if sample.contact_state == state
            ]
            raw_groups.append(
                pairwise_channel_correlation(
                    [
                        residual_by_id[sample.attempt["sample_id"]][[2, 3, 4]]
                        for sample in state_samples
                    ]
                )
            )
            aligned_groups.append(
                pairwise_channel_correlation(
                    [
                        align_simulation_channels(
                            residual_by_id[sample.attempt["sample_id"]],
                            sample.direction_deg,
                        )
                        for sample in state_samples
                    ]
                )
            )
        observed = float(np.mean(aligned_groups))
        rng = np.random.default_rng(20260728 + size_index)
        null_values = []
        for _ in range(256):
            shuffled_groups = []
            for state in CONTACT_STATES:
                state_samples = [
                    sample
                    for sample in size_samples
                    if sample.contact_state == state
                ]
                shuffled_direction = rng.permutation(
                    [sample.direction_deg for sample in state_samples]
                )
                shuffled_groups.append(
                    pairwise_channel_correlation(
                        [
                            align_simulation_channels(
                                residual_by_id[sample.attempt["sample_id"]],
                                float(direction),
                            )
                            for sample, direction in zip(
                                state_samples, shuffled_direction
                            )
                        ]
                    )
                )
            null_values.append(float(np.mean(shuffled_groups)))
        null_array = np.asarray(null_values)
        report[size] = {
            "channels": ["Fz", "Tx", "Ty"],
            "common_mode_estimator": "leave-one-clean-sample-out median",
            "common_mode_removed_unaligned_mean_pairwise_correlation": float(
                np.mean(raw_groups)
            ),
            "common_mode_removed_direction_aligned_mean_pairwise_correlation": float(
                observed
            ),
            "within_state_direction_permutation_null": {
                "permutations": int(len(null_array)),
                "mean": float(np.mean(null_array)),
                "p95": float(np.quantile(null_array, 0.95)),
                "one_sided_p_value": float(
                    (1 + np.sum(null_array >= observed))
                    / (len(null_array) + 1)
                ),
            },
            "interpretation": (
                "increase after direction alignment is evidence for a "
                "direction-dependent residual only when it also exceeds the "
                "within-state shuffled-label null; magnitude is not classifier "
                "accuracy or cross-size transfer"
            ),
        }
    return report


def sentinel_repeatability(samples: list[Sample]) -> dict[str, Any]:
    report = {}
    for size in RUN_ALLOWLIST:
        size_samples = [sample for sample in samples if sample.size == size]
        sentinels = [
            sample for sample in size_samples if sample.partition == "sentinel"
        ]
        correlations = []
        for sentinel in sentinels:
            duplicate = str(sentinel.attempt.get("duplicate_of_sample_id", ""))
            reference = next(
                (
                    sample
                    for sample in size_samples
                    if sample.attempt["sample_id"] == duplicate
                ),
                None,
            )
            if reference is None:
                # The registered plans preserve the source index even when the
                # duplicate string is omitted by an older writer.
                reference = min(
                    (
                        sample
                        for sample in size_samples
                        if sample.partition == "clean_candidate"
                        and sample.contact_state == sentinel.contact_state
                    ),
                    key=lambda sample: circular_error_deg(
                        sample.direction_deg, sentinel.direction_deg
                    ),
                )
            first = channel_standardize(reference.phase_sequence[:6])
            second = channel_standardize(sentinel.phase_sequence[:6])
            correlations.append(
                {
                    "sentinel_sample_id": sentinel.attempt["sample_id"],
                    "reference_sample_id": reference.attempt["sample_id"],
                    "per_channel_correlation": np.mean(
                        first * second, axis=1
                    ).tolist(),
                    "mean_correlation": float(np.mean(first * second)),
                }
            )
        report[size] = correlations
    return report


def regular_hexagon(af_mm: float) -> np.ndarray:
    radius = float(af_mm) / math.sqrt(3.0)
    return np.asarray(
        [
            [
                radius * math.cos(math.radians(30.0 + 60.0 * index)),
                radius * math.sin(math.radians(30.0 + 60.0 * index)),
            ]
            for index in range(6)
        ]
    )


def ray_entry_distance_mm(
    start_xy_mm: np.ndarray, direction_xy: np.ndarray, feasible_af_mm: float
) -> float | None:
    """Return first ray entry into an aligned regular hexagon.

    The polygon is the exact same-yaw Minkowski erosion
    ``mouth_hex minus tip_hex``.  It certifies geometric mouth containment,
    not chamfer capture.
    """

    start = np.asarray(start_xy_mm, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    norm = float(np.linalg.norm(direction))
    if start.shape != (2,) or direction.shape != (2,) or norm <= 1.0e-12:
        raise ValueError("ray inputs must be finite nonzero 2-D vectors")
    direction = direction / norm
    polygon = regular_hexagon(feasible_af_mm)
    lower = 0.0
    upper = float("inf")
    for index in range(len(polygon)):
        edge_start = polygon[index]
        edge_end = polygon[(index + 1) % len(polygon)]
        edge = edge_end - edge_start
        constant = (
            edge[0] * (start[1] - edge_start[1])
            - edge[1] * (start[0] - edge_start[0])
        )
        slope = edge[0] * direction[1] - edge[1] * direction[0]
        if abs(slope) <= 1.0e-12:
            if constant < 0.0:
                return None
            continue
        boundary = -constant / slope
        if slope > 0.0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
        if lower > upper:
            return None
    if upper < 0.0:
        return None
    return max(0.0, float(lower))


def cad_shortcut_report(
    samples: list[Sample], template_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_by_id = {
        record["sample_id"]: record for record in template_records
    }
    records = []
    by_size = {}
    for size in RUN_ALLOWLIST:
        size_samples = [
            sample
            for sample in samples
            if sample.size == size
            and sample.partition in ("clean_candidate", "holdout")
        ]
        geometry = read_json(
            size_samples[0].run_dir / "sampling_campaign.json"
        )["printed_cad_geometry_mm"]
        peg_yaw = float(geometry["peg_tip_face_flat_normal_fixture_deg"])
        socket_yaw = float(
            geometry["socket_selected_flat_normal_fixture_deg"]
        )
        yaw_error = circular_error_deg(peg_yaw, socket_yaw)
        if yaw_error > 1.0e-9:
            raise ValueError(
                f"{size} CAD erosion requires same-yaw hexagons; "
                f"peg/socket error is {yaw_error:.6f} deg"
            )
        feasible_af = float(geometry["socket_mouth_af"]) - float(
            geometry["peg_tip_face_af"]
        )
        oracle_distances = []
        for sample in size_samples:
            true_direction = np.asarray(
                sample.attempt["true_direction_fixture_unit"], dtype=float
            )
            oracle = ray_entry_distance_mm(
                sample.measured_offset_mm, true_direction, feasible_af
            )
            if oracle is None:
                raise ValueError(
                    f"true ray misses CAD mouth erosion for {sample.attempt['sample_id']}"
                )
            oracle_distances.append(oracle)
            record = {
                "size": size,
                "sample_id": sample.attempt["sample_id"],
                "partition": sample.partition,
                "contact_state": sample.contact_state,
                "feasible_center_hex_af_mm": feasible_af,
                "oracle_true_direction_entry_distance_mm": oracle,
                "predicted_top1_entry_distance_mm": "",
                "predicted_top2_entry_distance_mm": "",
                "predicted_top1_ray_hits": "",
                "predicted_top2_any_ray_hits": "",
            }
            prediction = prediction_by_id.get(sample.attempt["sample_id"])
            if prediction is not None:
                first_angle = float(prediction["top1_direction_deg"])
                second_angle = float(prediction["top2_direction_deg"])
                first = ray_entry_distance_mm(
                    sample.measured_offset_mm,
                    np.asarray(
                        [
                            math.cos(math.radians(first_angle)),
                            math.sin(math.radians(first_angle)),
                        ]
                    ),
                    feasible_af,
                )
                second = ray_entry_distance_mm(
                    sample.measured_offset_mm,
                    np.asarray(
                        [
                            math.cos(math.radians(second_angle)),
                            math.sin(math.radians(second_angle)),
                        ]
                    ),
                    feasible_af,
                )
                record.update(
                    {
                        "predicted_top1_entry_distance_mm": (
                            "" if first is None else first
                        ),
                        "predicted_top2_entry_distance_mm": (
                            "" if second is None else second
                        ),
                        "predicted_top1_ray_hits": first is not None,
                        "predicted_top2_any_ray_hits": (
                            first is not None or second is not None
                        ),
                    }
                )
            records.append(record)
        values = np.asarray(oracle_distances)
        holdout_records = [
            record
            for record in records
            if record["size"] == size and record["partition"] == "holdout"
        ]
        by_size[size] = {
            "feasible_center_hex_af_mm": feasible_af,
            "peg_socket_flat_normal_yaw_error_deg": yaw_error,
            "oracle_entry_distance_mm": {
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            },
            "holdout_top1_ray_hit_fraction": float(
                np.mean(
                    [
                        bool(record["predicted_top1_ray_hits"])
                        for record in holdout_records
                    ]
                )
            ),
            "holdout_top2_any_ray_hit_fraction": float(
                np.mean(
                    [
                        bool(record["predicted_top2_any_ray_hits"])
                        for record in holdout_records
                    ]
                )
            ),
        }
    return records, {
        "geometry": (
            "same-yaw regular-hex Minkowski erosion: "
            "H(AF_mouth) minus H(AF_tip)"
        ),
        "meaning": (
            "first ray entry is a mouth-containment geometry distance; it is "
            "not a calibrated chamfer-capture distance"
        ),
        "oracle_offset_boundary": (
            "measured run-local offsets and true direction are used only for "
            "post-hoc scoring; deployment requires a visual/direction/CAD "
            "center posterior"
        ),
        "by_size": by_size,
    }


def save_template_model(
    path: Path, templates: dict[str, np.ndarray], clean: list[Sample]
) -> None:
    payload: dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in templates.items()
    }
    payload["phase_grid_rad"] = np.linspace(
        0.0, 2.0 * math.pi, PHASE_BINS, endpoint=False
    ).astype(np.float32)
    payload["training_sample_ids"] = np.asarray(
        [sample.attempt["sample_id"] for sample in clean]
    )
    payload["training_partitions"] = np.asarray(
        [sample.partition for sample in clean]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_template_model_contract(
    path: Path, model_path: Path, clean: list[Sample]
) -> None:
    run_contracts = {}
    for size, run_dir in RUN_ALLOWLIST.items():
        campaign = read_json(run_dir / "sampling_campaign.json")
        run_contracts[size] = {
            "run_dir": str(run_dir),
            "sampling_campaign_sha256": sha256_file(
                run_dir / "sampling_campaign.json"
            ),
            "sampling_plan_path": campaign["sampling_plan_path"],
            "sampling_plan_sha256": campaign["sampling_plan_sha256"],
            "fixture_frame_commissioning_sha256": campaign[
                "fixture_frame_commissioning_sha256"
            ],
            "active_peg_step_sha256": campaign[
                "cad_and_configuration_hashes"
            ]["active_peg_step_sha256"],
            "printed_socket_step_sha256": campaign[
                "cad_and_configuration_hashes"
            ]["printed_socket_step_sha256"],
        }
    write_json(
        path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "model_id": "real_fxy_state_templates_v1",
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "model_role": (
                "EXPLORATORY_DIRECTION_OBSERVABILITY_UPPER_BOUND_ONLY"
            ),
            "hardware_control_allowed": False,
            "training_partition": "clean_candidate_only",
            "training_sample_count": len(clean),
            "training_sample_ids": sorted(
                sample.attempt["sample_id"] for sample in clean
            ),
            "holdout_used_for_fit_or_normalization": False,
            "sentinel_used_for_fit_or_normalization": False,
            "quarantine_used_for_fit_or_normalization": False,
            "input_contract": {
                "channels": ["Fx_WORLD_star_N", "Fy_WORLD_star_N"],
                "wrench_origin": "Bota measurement origin, no tip shift",
                "phase_bins": PHASE_BINS,
                "phase_source": "recorded conical command phase",
                "frame": "WORLD_star_relative_to_fresh_per_point_air_baseline",
                "normalization": (
                    "per-channel per-scan mean removal and standard-deviation "
                    "scaling after circular direction alignment"
                ),
                "template_axes": ["lateral_x", "lateral_y"],
            },
            "output_contract": {
                "sizes": list(RUN_ALLOWLIST),
                "state_order": list(CONTACT_STATES),
                "score": (
                    "circular correlation curve over 360 candidate directions"
                ),
                "posterior_status": (
                    "uncalibrated multimodal score, not a probability"
                ),
            },
            "state_conditioning_boundary": (
                "uses registered true CAD contact state and is therefore an "
                "observability upper bound, not deployable inference"
            ),
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "run_contracts": run_contracts,
        },
    )


def holdout_analysis_signature(
    clean: list[Sample], templates: dict[str, np.ndarray]
) -> dict[str, str]:
    clean_contract = []
    for sample in sorted(clean, key=lambda value: value.attempt["sample_id"]):
        clean_contract.append(
            {
                "sample_id": sample.attempt["sample_id"],
                "partition": sample.partition,
                "contact_state": sample.contact_state,
                "direction_deg": sample.direction_deg,
                "cone_csv_sha256": sample.attempt["file_hashes"][
                    "cone_csv_sha256"
                ],
                "baseline_csv_sha256": sample.attempt["file_hashes"][
                    "baseline_csv_sha256"
                ],
            }
        )
    clean_signature = hashlib.sha256(
        json.dumps(
            clean_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    template_digest = hashlib.sha256()
    for key in sorted(templates):
        value = np.asarray(templates[key], dtype="<f8")
        template_digest.update(key.encode("utf-8"))
        template_digest.update(str(value.shape).encode("ascii"))
        template_digest.update(value.tobytes(order="C"))
    components = {
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "clean_input_signature_sha256": clean_signature,
        "template_parameter_signature_sha256": template_digest.hexdigest(),
    }
    components["analysis_signature_sha256"] = hashlib.sha256(
        json.dumps(
            components, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return components


def plot_holdout_errors(
    path: Path, records: list[dict[str, Any]]
) -> None:
    labels = [
        f"{record['size']}\n{record['contact_state'].replace('_', ' ')}"
        for record in records
    ]
    top1 = [record["top1_error_deg"] for record in records]
    top2 = [record["best_of_top2_error_deg"] for record in records]
    x = np.arange(len(records))
    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    axis.bar(x - 0.19, top1, 0.38, label="top-1 mode")
    axis.bar(x + 0.19, top2, 0.38, label="best of top-2 modes")
    axis.axhline(30.0, color="#d97706", linestyle="--", label="30 deg")
    axis.axhline(90.0, color="#dc2626", linestyle=":", label="90 deg")
    axis.set_ylabel("circular direction error [deg]")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axis.set_title(
        "First development holdout reveal: clean-trained Fx/Fy template"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_alignment(path: Path, report: dict[str, Any]) -> None:
    sizes = list(RUN_ALLOWLIST)
    raw = [
        report[size][
            "common_mode_removed_unaligned_mean_pairwise_correlation"
        ]
        for size in sizes
    ]
    aligned = [
        report[size][
            "common_mode_removed_direction_aligned_mean_pairwise_correlation"
        ]
        for size in sizes
    ]
    x = np.arange(len(sizes))
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.bar(
        x - 0.18,
        raw,
        0.36,
        label="common-mode removed, unaligned",
    )
    axis.bar(
        x + 0.18,
        aligned,
        0.36,
        label="common-mode removed + direction aligned",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(sizes)
    axis.set_ylabel("mean pairwise correlation")
    axis.set_title("C6 direction-equivariance sanity check: Fz/Tx/Ty")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cad_distances(path: Path, records: list[dict[str, Any]]) -> None:
    sizes = list(RUN_ALLOWLIST)
    values = [
        [
            float(record["oracle_true_direction_entry_distance_mm"])
            for record in records
            if record["size"] == size
        ]
        for size in sizes
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    axis.boxplot(values, labels=sizes, showmeans=True)
    axis.set_ylabel("CAD mouth-containment entry distance [mm]")
    axis.set_title(
        "Post-hoc true-direction geometry (clean + holdout; not capture calibration)"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_summary_markdown(report: dict[str, Any]) -> str:
    template = report["direction_observability"]["real_fxy_template_holdout"]
    ridge = report["direction_observability"]["simulation_fourier_ridge_holdout"]
    zero_shot = report["direction_observability"].get("simulation_zero_shot")
    lines = [
        "# M8/M10/M12 real conical fingerprint offline QC v1",
        "",
        f"Generated: `{report['created_utc']}`",
        "",
        "## Verdict",
        "",
        "- All three explicitly allowlisted fresh runs pass immutable-data and "
        "acquisition QC.",
        "- The frozen simulation feature contract does **not** transfer directly "
        "to this first real dataset.",
        "- Real lateral-force traces contain useful circular direction structure, "
        "but the current posterior remains multimodal and has one catastrophic "
        "top-1 holdout alias.",
        "- CAD can supply an interpretable mouth-containment distance after a "
        "center posterior is formed; these samples do not calibrate physical "
        "chamfer capture.",
        "- The 12 holdout points were revealed once for this v1 development "
        "assessment. They are now consumed and must not be used for further "
        "feature selection, fitting, or threshold tuning.",
        "",
        "## Data authority",
        "",
        "| size | run | clean | holdout | quarantine | sentinel | hash checks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for size, quality in report["quality_control"]["runs"].items():
        partitions = quality["partition_counts"]
        lines.append(
            f"| {size} | `{Path(quality['run_dir']).name}` | "
            f"{partitions['clean_candidate']} | {partitions['holdout']} | "
            f"{partitions['quarantine']} | {partitions['sentinel']} | "
            f"{quality['hash_checks']} |"
        )
    lines.extend(
        [
            "",
            "## Direction observability",
            "",
            "| model | median | p90 | <=30 deg | >90 deg |",
            "|---|---:|---:|---:|---:|",
            (
                "| simulation-inspired Fz/Tx/Ty/z 48-D Fourier ridge | "
                f"{ridge['median_error_deg']:.2f} | "
                f"{ridge['p90_error_deg']:.2f} | "
                f"{ridge['within_30_fraction']:.3f} | "
                f"{ridge['catastrophic_gt90_fraction']:.3f} |"
            ),
            (
                "| real Fx/Fy circular template top-1 | "
                f"{template['top1']['median_error_deg']:.2f} | "
                f"{template['top1']['p90_error_deg']:.2f} | "
                f"{template['top1']['within_30_fraction']:.3f} | "
                f"{template['top1']['catastrophic_gt90_fraction']:.3f} |"
            ),
            (
                "| real Fx/Fy best-of-top-2 oracle posterior coverage | "
                f"{template['best_of_top2']['median_error_deg']:.2f} | "
                f"{template['best_of_top2']['p90_error_deg']:.2f} | "
                f"{template['best_of_top2']['within_30_fraction']:.3f} | "
                f"{template['best_of_top2']['catastrophic_gt90_fraction']:.3f} |"
            ),
        ]
    )
    if zero_shot and zero_shot.get("status") == "COMPLETE":
        raw = zero_shot["metrics"]["nonboundary_raw_runtime_mean"]
        lines.append(
            "| frozen simulation CNN zero-shot on real | "
            f"{raw['median_error_deg']:.2f} | "
            f"{raw['p90_error_deg']:.2f} | "
            f"{raw['within_30_fraction']:.3f} | "
            f"{raw['catastrophic_gt90_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The real template is conditioned on the registered CAD contact state "
            "to isolate direction observability. A deployable model must replace "
            "that label with a calibrated state/CAD likelihood.",
            "",
            "This is a development holdout reveal, not a final untouched audit. "
            "Freeze the next adapter before collecting a new physical audit set.",
            "The best-of-top-2 row only asks whether either posterior mode covers "
            "the truth; it is not an executable online direction selector.",
            "",
            "## CAD geometric distance envelope (posterior fusion pending)",
            "",
            "| size | feasible center hex AF [mm] | oracle entry min/median/max [mm] | top-1 ray hit | top-2 any hit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for size, values in report["cad_shortcut"]["by_size"].items():
        distance = values["oracle_entry_distance_mm"]
        lines.append(
            f"| {size} | {values['feasible_center_hex_af_mm']:.3f} | "
            f"{distance['minimum']:.3f} / {distance['median']:.3f} / "
            f"{distance['maximum']:.3f} | "
            f"{values['holdout_top1_ray_hit_fraction']:.3f} | "
            f"{values['holdout_top2_any_ray_hit_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These are geometric mouth-containment distances. They are not "
            "validated chamfer-capture distances because no acquisition sample "
            "contains a capture/landing outcome.",
            "",
            "## Next training step",
            "",
            "1. Keep the simulation encoder frozen as a teacher/initialization only.",
            "2. Fit a small real-domain common-mode, frame, and phase adapter from "
            "clean samples.",
            "3. Train/calibrate state and circular-direction heads without using "
            "holdout, sentinel, or quarantine.",
            "4. Preserve a multimodal direction posterior and fuse it with exact "
            "CAD plus an independently calibrated visual center prior.",
            "5. Use short full-loop trials to calibrate physical capture margin; "
            "do not learn continuous distance from these 48 clean scans.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and fit the first complete real M8/M10/M12 scans."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-development-holdout-rereveal",
        action="store_true",
        help=(
            "explicitly consume the same development holdout after the "
            "analysis source, clean inputs, or fitted template changed"
        ),
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not (
        (KNOWN_REJECTED_RUN / "DO_NOT_TRAIN.flag").is_file()
        and (KNOWN_REJECTED_RUN / "RUN_QUARANTINE.json").is_file()
    ):
        raise ValueError(
            "known M8-program/M12-physical mismatch is not quarantined"
        )

    samples: list[Sample] = []
    quality = {}
    for size, run_dir in RUN_ALLOWLIST.items():
        loaded, run_quality = validate_campaign(size, run_dir)
        samples.extend(loaded)
        quality[size] = run_quality

    clean = [
        sample for sample in samples if sample.partition == "clean_candidate"
    ]
    holdout = [
        sample for sample in samples if sample.partition == "holdout"
    ]
    if len(clean) != 48 or len(holdout) != 12:
        raise ValueError("multisize clean/holdout count contract failed")

    alpha, alpha_selection = select_ridge_alpha(clean)
    ridge_model = fit_ridge_direction(clean, alpha)
    templates = fit_template_model(clean)
    template_model_path = output / "real_fxy_state_templates_v1.npz"
    save_template_model(template_model_path, templates, clean)
    save_template_model_contract(
        output / "real_fxy_state_templates_v1_contract.json",
        template_model_path,
        clean,
    )
    reveal_path = output / "holdout_reveal_record.json"
    reveal_signature = holdout_analysis_signature(clean, templates)
    existing_reveal = read_json(reveal_path) if reveal_path.is_file() else None
    reveal_matches = bool(
        existing_reveal
        and existing_reveal.get("analysis_signature_sha256")
        == reveal_signature["analysis_signature_sha256"]
    )
    if (
        existing_reveal is not None
        and not reveal_matches
        and not args.allow_development_holdout_rereveal
    ):
        raise RuntimeError(
            "the v1 holdout was already revealed under a different analysis "
            "signature; freeze a new model and collect a new physical audit "
            "set, or pass --allow-development-holdout-rereveal only for an "
            "explicitly documented development re-analysis"
        )

    # The one-time reveal guard must precede every read of holdout labels for
    # scoring.  All fitting and hyperparameter selection above used clean only.
    ridge_prediction = predict_ridge(ridge_model, holdout)
    ridge_errors = circular_error_deg(
        ridge_prediction,
        np.asarray([sample.direction_deg for sample in holdout]),
    )
    ridge_metrics = direction_metrics(ridge_errors)
    ridge_metrics.update(
        {
            "role": (
                "simulation-inspired 48-D Fourier ridge trained on real clean "
                "samples; selected by clean-only leave-one-size-out. This is "
                "not the frozen simulation CNN or its full 64-D input contract"
            ),
            "channels": ["Fz", "Tx", "Ty", "command_z_proxy"],
            "alpha_selection": alpha_selection,
        }
    )
    template_records, template_metrics = evaluate_template_holdout(
        templates, holdout
    )
    template_csv_fields = list(template_records[0])
    write_csv(
        output / "holdout_direction_predictions.csv",
        template_csv_fields,
        template_records,
    )

    cad_records, cad_report = cad_shortcut_report(samples, template_records)
    write_csv(
        output / "cad_shortcut_geometry.csv",
        list(cad_records[0]),
        cad_records,
    )

    c6_report = c6_alignment_report(clean)
    sentinel_report = sentinel_repeatability(samples)
    zero_shot_path = output / "simulation_zero_shot_report.json"
    if zero_shot_path.is_file():
        zero_shot_report = read_json(zero_shot_path)
    else:
        zero_shot_report = {
            "status": "PENDING_SEPARATE_TORCH_REPLAY",
            "reason": (
                "the system-Python QC lane intentionally has no Torch "
                "dependency"
            ),
        }
    if reveal_matches:
        reveal_record = existing_reveal
    else:
        reveal_record = {
            "schema_version": 1,
            "analysis_id": "multisize_real_fingerprint_v1",
            "revealed_utc": utc_now(),
            "status_after_analysis": (
                "REVEALED_ONCE_V1_DO_NOT_RETUNE_ON_THESE_RESULTS"
            ),
            "role": "development_holdout_not_final_locked_audit",
            "sample_ids": sorted(
                sample.attempt["sample_id"] for sample in holdout
            ),
            "policy": (
                "exclude these 12 samples from every subsequent feature "
                "choice, fit, normalization, calibration, and threshold "
                "selection"
            ),
            "development_rereveal_override_used": bool(
                existing_reveal is not None
            ),
            "previous_reveal_record_sha256": (
                sha256_file(reveal_path)
                if existing_reveal is not None
                else None
            ),
            **reveal_signature,
        }
        write_json(reveal_path, reveal_record)
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "analysis_id": "multisize_real_fingerprint_v1",
        "authority": (
            "REAL_DEVICE_RAW_SIGNAL_HIGHEST_AUTHORITY; simulation is a "
            "feature prior only"
        ),
        "robot_motion": False,
        "run_allowlist": {
            size: str(path) for size, path in RUN_ALLOWLIST.items()
        },
        "known_rejected_run": str(KNOWN_REJECTED_RUN),
        "quality_control": {
            "status": "PASS",
            "scope": (
                "ACQUISITION_AND_FILE_INTEGRITY_PASS_FOR_EXPLORATORY_"
                "OBSERVABILITY_ONLY"
            ),
            "training_admission": {
                "status": (
                    "NOT_POSE_QUALIFIED_FOR_SUPERVISED_MODEL_PROMOTION"
                ),
                "reason": (
                    "all 78 traces have fresh complete Bota command-phase "
                    "fingerprints, but all cone_training_fingerprint_valid and "
                    "conical_phase_pose_tracking_qualified_for_training flags "
                    "are false because there was no in-loop measured KST pose"
                ),
                "permitted_now": (
                    "exploratory command-phase observability fit on 48 clean "
                    "candidates only"
                ),
            },
            "runs": quality,
            "total_attempts": len(samples),
            "clean_training_count": len(clean),
            "holdout_evaluation_count": len(holdout),
            "holdout_status_after_analysis": reveal_record[
                "status_after_analysis"
            ],
            "quarantine_diagnostic_count": sum(
                sample.partition == "quarantine" for sample in samples
            ),
            "sentinel_repeatability_count": sum(
                sample.partition == "sentinel" for sample in samples
            ),
            "total_hash_checks": sum(
                value["hash_checks"] for value in quality.values()
            ),
            "sentinel_repeatability": sentinel_report,
        },
        "direction_observability": {
            "verdict": (
                "real direction information is present but is not yet uniquely "
                "decoded across all sizes; preserve a multimodal posterior"
            ),
            "simulation_fourier_ridge_holdout": ridge_metrics,
            "real_fxy_template_holdout": template_metrics,
            "c6_alignment": c6_report,
            "simulation_zero_shot": zero_shot_report,
        },
        "cad_shortcut": cad_report,
        "promotion_decision": "NOT_READY_FOR_DIRECT_SINGLE_RAY_HARDWARE_CONTROL",
        "next_model": (
            "frozen simulation encoder plus small real common-mode/frame/phase "
            "adapter and calibrated state/direction heads"
        ),
        "claim_boundary": (
            "This report demonstrates data integrity and partial real-domain "
            "direction observability. It does not certify a real direction "
            "posterior, chamfer capture distance, insertion success, or online "
            "hardware authority."
        ),
    }
    write_json(output / "report.json", report)
    (output / "summary.md").write_text(
        build_summary_markdown(report), encoding="utf-8"
    )
    plot_holdout_errors(
        output / "holdout_direction_errors.png", template_records
    )
    plot_alignment(output / "c6_alignment.png", c6_report)
    plot_cad_distances(output / "cad_mouth_entry_distances.png", cad_records)

    artifact_hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifact_hashes[path.name] = sha256_file(path)
    write_json(
        output / "artifact_hashes.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "artifacts": artifact_hashes,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "samples": len(samples),
                "clean": len(clean),
                "holdout": len(holdout),
                "template_top1_median_deg": template_metrics["top1"][
                    "median_error_deg"
                ],
                "template_top1_p90_deg": template_metrics["top1"][
                    "p90_error_deg"
                ],
                "template_top2_within30": template_metrics["best_of_top2"][
                    "within_30_fraction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
