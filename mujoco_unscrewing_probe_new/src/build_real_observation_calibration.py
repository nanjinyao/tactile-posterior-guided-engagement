#!/usr/bin/env python3
"""Fit a real-observation calibration and build deterministic replay variants.

This module calibrates only the observation side of the MuJoCo-to-real
interface.  It does not identify contact friction, printed-part stiffness, or
controller dynamics from one 26-point campaign per size.  Those quantities
remain explicitly declared domain-randomization variables.

The three source campaigns are an allowlist.  In particular, the later M8
program run on the physical M12 socket is named as rejected evidence and can
never enter this fit through directory discovery.
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
from typing import Any, Iterable

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = SIM_ROOT.parent
REAL_ROOT = GROUP_ROOT / "real_device"
DEFAULT_CONFIG = SIM_ROOT / "configs" / "real_calibrated_v1.json"
DEFAULT_OUTPUT = (
    REAL_ROOT
    / "offline_analysis"
    / "real_multisize_learning_v1"
    / "sim_calibration_v1"
)

# Authority is deliberately explicit; do not replace this with a glob.
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
COMPLETE_STATUS = "COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT"
PARTITION_COUNTS = {
    "clean_candidate": 16,
    "holdout": 4,
    "quarantine": 4,
    "sentinel": 2,
}
FIT_PARTITIONS = ("clean_candidate",)
DRIFT_PARTITIONS = ("sentinel",)
SEEDS_PER_GEOMETRY_TASK = 3
PHASE_BINS = 360
WRENCH_CHANNELS = (
    "fx_world_star_n",
    "fy_world_star_n",
    "fz_world_star_n",
    "tx_world_star_nm",
    "ty_world_star_nm",
    "tz_world_star_nm",
)
RAW_CHANNELS = (
    "raw_fx_n",
    "raw_fy_n",
    "raw_fz_n",
    "raw_tx_nm",
    "raw_ty_nm",
    "raw_tz_nm",
)
SEQUENCE_CHANNELS = WRENCH_CHANNELS + ("cmd_z_relative_mm",)
SIZE_ORDER = ("M8", "M10", "M12")
CONTACT_STATES = (
    "supported_search",
    "double_overhang",
    "outer_partial_support",
)


@dataclass
class RealSample:
    size: str
    run_dir: Path
    attempt: dict[str, Any]
    phase_rad: np.ndarray
    sequence: np.ndarray
    source_stamp_s: np.ndarray
    wrench_age_s: np.ndarray
    tick_lateness_s: np.ndarray
    baseline_raw: np.ndarray
    baseline_stamp_s: np.ndarray

    @property
    def sample_id(self) -> str:
        return str(self.attempt["sample_id"])

    @property
    def partition(self) -> str:
        return str(self.attempt["effective_partition"])

    @property
    def contact_state(self) -> str:
        return str(self.attempt["contact_state"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | float | int) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def periodic_resample(
    phase_rad: np.ndarray, values: np.ndarray, bins: int = PHASE_BINS
) -> np.ndarray:
    phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * math.pi)
    samples = np.asarray(values, dtype=float)
    if phase.ndim != 1 or samples.ndim != 2 or len(phase) != len(samples):
        raise ValueError("phase/value shape mismatch")
    finite = np.isfinite(phase) & np.all(np.isfinite(samples), axis=1)
    phase = phase[finite]
    samples = samples[finite]
    if len(phase) < 32:
        raise ValueError("fewer than 32 finite phase samples")
    order = np.argsort(phase)
    phase = phase[order]
    samples = samples[order]
    phase, unique = np.unique(phase, return_index=True)
    samples = samples[unique]
    extended_phase = np.concatenate(
        (phase[-1:] - 2.0 * math.pi, phase, phase[:1] + 2.0 * math.pi)
    )
    grid = np.linspace(0.0, 2.0 * math.pi, bins, endpoint=False)
    result = []
    for channel in range(samples.shape[1]):
        extended_value = np.concatenate(
            (
                samples[-1:, channel],
                samples[:, channel],
                samples[:1, channel],
            )
        )
        result.append(np.interp(grid, extended_phase, extended_value))
    return np.asarray(result)


def _attempt_path(
    run_dir: Path, attempt: dict[str, Any], key: str
) -> Path:
    path = Path(str(attempt[key]))
    if not path.is_absolute():
        path = REAL_ROOT / path
    path = path.resolve()
    if run_dir.resolve() not in path.parents:
        raise ValueError(f"{attempt['sample_id']} {key} escapes allowlisted run")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_attempt_hashes(
    run_dir: Path, attempt: dict[str, Any]
) -> int:
    mapping = {
        "attempt_mat_sha256": "mat_path",
        "baseline_csv_sha256": "baseline_csv_path",
        "contact_settle_csv_sha256": "contact_settle_csv_path",
        "cone_csv_sha256": "cone_csv_path",
    }
    hashes = attempt["file_hashes"]
    count = 0
    for hash_key, path_key in mapping.items():
        path = _attempt_path(run_dir, attempt, path_key)
        actual = sha256_file(path)
        expected = str(hashes[hash_key])
        if actual != expected:
            raise ValueError(
                f"{attempt['sample_id']} {path.name} hash mismatch"
            )
        count += 1
    return count


def _load_sample(
    size: str, run_dir: Path, attempt: dict[str, Any]
) -> RealSample:
    cone_rows = [
        row
        for row in read_csv(_attempt_path(run_dir, attempt, "cone_csv_path"))
        if row.get("stage") == "SCAN"
        and row.get("phase_rad") not in ("", "NaN", None)
    ]
    if len(cone_rows) != 240:
        raise ValueError(
            f"{attempt['sample_id']} has {len(cone_rows)} SCAN rows"
        )
    phase = np.asarray(
        [finite_float(row["phase_rad"]) for row in cone_rows], dtype=float
    )
    wrench = np.asarray(
        [
            [finite_float(row[channel]) for channel in WRENCH_CHANNELS]
            for row in cone_rows
        ],
        dtype=float,
    )
    command_z = np.asarray(
        [finite_float(row["cmd_z_mm"]) for row in cone_rows], dtype=float
    )
    command_z -= command_z[0]
    sequence = periodic_resample(
        phase, np.column_stack((wrench, command_z)), bins=PHASE_BINS
    )
    source_stamp = np.asarray(
        [finite_float(row["source_stamp_s"]) for row in cone_rows]
    )
    if np.any(np.diff(source_stamp) <= 0.0):
        raise ValueError(f"{attempt['sample_id']} source stamp is not fresh")
    wrench_age = np.asarray(
        [finite_float(row["wrench_age_s"]) for row in cone_rows]
    )
    tick_lateness = np.asarray(
        [finite_float(row["tick_lateness_s"]) for row in cone_rows]
    )
    if np.any(wrench_age < 0.0):
        raise ValueError(f"{attempt['sample_id']} has negative wrench age")

    baseline_rows = read_csv(
        _attempt_path(run_dir, attempt, "baseline_csv_path")
    )
    if len(baseline_rows) < 20:
        raise ValueError(f"{attempt['sample_id']} has a short air baseline")
    baseline_raw = np.asarray(
        [
            [finite_float(row[channel]) for channel in RAW_CHANNELS]
            for row in baseline_rows
        ],
        dtype=float,
    )
    baseline_stamp = np.asarray(
        [finite_float(row["source_stamp_s"]) for row in baseline_rows]
    )
    if np.any(np.diff(baseline_stamp) <= 0.0):
        raise ValueError(
            f"{attempt['sample_id']} baseline stamp is not fresh"
        )
    return RealSample(
        size=size,
        run_dir=run_dir,
        attempt=attempt,
        phase_rad=phase,
        sequence=sequence,
        source_stamp_s=source_stamp,
        wrench_age_s=wrench_age,
        tick_lateness_s=tick_lateness,
        baseline_raw=baseline_raw,
        baseline_stamp_s=baseline_stamp,
    )


def load_allowlisted_campaigns() -> tuple[list[RealSample], dict[str, Any]]:
    samples: list[RealSample] = []
    source_hashes: dict[str, Any] = {}
    for size, run_dir in RUN_ALLOWLIST.items():
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for flag in ("DO_NOT_TRAIN.flag", "RUN_QUARANTINE.json"):
            if (run_dir / flag).exists():
                raise ValueError(f"{run_dir.name} carries {flag}")
        campaign_path = run_dir / "sampling_campaign.json"
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        campaign = read_json(campaign_path)
        summary = read_json(summary_path)
        if campaign["size"] != size or summary["size"] != size:
            raise ValueError(f"{run_dir.name} size mismatch")
        if campaign["status"] != COMPLETE_STATUS:
            raise ValueError(f"{run_dir.name} campaign is not complete")
        if summary["completion_status"] != COMPLETE_STATUS or not bool(
            summary["task_completed"]
        ):
            raise ValueError(f"{run_dir.name} summary is not complete")
        if bool(campaign["resume_enabled"]) or int(
            campaign["resume_generation"]
        ) != 0:
            raise ValueError(f"{run_dir.name} is not a fresh generation-zero run")

        attempt_paths = sorted(
            (run_dir / "sampling_attempts").glob("*/attempt.json")
        )
        if len(attempt_paths) != 26:
            raise ValueError(f"{run_dir.name} has {len(attempt_paths)} attempts")
        partition_counts = {key: 0 for key in PARTITION_COUNTS}
        hash_checks = 0
        for attempt_path in attempt_paths:
            attempt = read_json(attempt_path)
            partition = str(attempt["effective_partition"])
            if partition not in partition_counts:
                raise ValueError(
                    f"{attempt['sample_id']} has unknown partition {partition}"
                )
            partition_counts[partition] += 1
            if not bool(attempt["cone_physical_schedule_completed"]):
                raise ValueError(f"{attempt['sample_id']} cone is incomplete")
            if not bool(attempt["cone_direction_fingerprint_valid"]):
                raise ValueError(
                    f"{attempt['sample_id']} direction fingerprint is invalid"
                )
            hash_checks += _verify_attempt_hashes(run_dir, attempt)
            samples.append(_load_sample(size, run_dir, attempt))
        if partition_counts != PARTITION_COUNTS:
            raise ValueError(
                f"{run_dir.name} partition counts {partition_counts}"
            )
        source_hashes[size] = {
            "run_id": run_dir.name,
            "campaign_sha256": sha256_file(campaign_path),
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(manifest_path),
            "attempt_hash_checks": hash_checks,
            "partition_counts": partition_counts,
        }
    if len(samples) != 78:
        raise ValueError(f"expected 78 samples, got {len(samples)}")
    if KNOWN_REJECTED_RUN in {sample.run_dir for sample in samples}:
        raise ValueError("known M8-program-on-M12-socket run leaked into fit")
    return samples, source_hashes


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    median = np.median(array, axis=axis)
    deviation = np.median(np.abs(array - np.expand_dims(median, axis)), axis=axis)
    scale = 1.4826 * deviation
    standard = np.std(array, axis=axis, ddof=1)
    return np.where(scale > 1.0e-12, scale, np.maximum(standard, 1.0e-12))


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution needs finite observations")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
    }


def _fit_frame_phase(samples: list[RealSample]) -> dict[str, Any]:
    fixture = np.asarray(
        [
            sample.attempt["measured_contact_fixture_xy_mm"]
            for sample in samples
        ],
        dtype=float,
    )
    world = np.asarray(
        [
            np.asarray(sample.attempt["touchdown_pose_mm_rad"][:2], dtype=float)
            - np.asarray(
                sample.attempt["run_local_hole_center_world_xy_mm"], dtype=float
            )
            for sample in samples
        ],
        dtype=float,
    )
    left, _, right = np.linalg.svd(fixture.T @ world)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1, :] *= -1.0
        rotation = right.T @ left.T
    prediction = fixture @ rotation.T
    residual = world - prediction
    yaw_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))

    coefficient = 0.0j
    energy = 0.0
    for sample in samples:
        theta = math.radians(
            finite_float(sample.attempt["true_direction_fixture_deg"])
            + yaw_deg
        )
        inward = np.asarray([math.cos(theta), math.sin(theta)])
        projected = sample.sequence[:2, :].T @ inward
        projected -= np.mean(projected)
        phase_grid = np.linspace(
            0.0, 2.0 * math.pi, PHASE_BINS, endpoint=False
        )
        relative_phase = phase_grid - theta
        coefficient += np.mean(projected * np.exp(-1.0j * relative_phase))
        energy += float(np.mean(projected * projected))
    phase_peak_deg = (-math.degrees(np.angle(coefficient))) % 360.0
    return {
        "fixture_to_world_xy_rotation": rotation,
        "fixture_to_world_yaw_deg": yaw_deg,
        "orthogonal_fit_rms_mm": float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        ),
        "orthogonal_fit_max_mm": float(
            np.max(np.linalg.norm(residual, axis=1))
        ),
        "conical_phase_sign": 1,
        "inward_response_peak_relative_phase_deg": phase_peak_deg,
        "inward_response_first_harmonic_amplitude": float(
            2.0 * abs(coefficient) / len(samples)
        ),
        "inward_response_first_harmonic_energy_fraction": float(
            min(
                1.0,
                (2.0 * abs(coefficient) / len(samples)) ** 2
                / max(2.0 * energy / len(samples), 1.0e-12),
            )
        ),
        "identifiability": (
            "fixture/world transform is fitted from measured touchdown offsets; "
            "response phase is fitted separately from the direction-labelled "
            "first harmonic and is not a friction or stiffness estimate"
        ),
    }


def _lag_one(values: np.ndarray) -> np.ndarray:
    result = []
    for channel in range(values.shape[1]):
        first = values[:-1, channel]
        second = values[1:, channel]
        first = first - np.mean(first)
        second = second - np.mean(second)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        result.append(
            0.0 if denominator <= 1.0e-15 else float(first @ second / denominator)
        )
    return np.asarray(result)


def _fit_channel_noise(
    samples: list[RealSample],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sequence = np.concatenate([sample.sequence.T for sample in samples], axis=0)
    location = np.median(sequence, axis=0)
    scale = robust_scale(sequence, axis=0)
    multiply = 1.0 / scale
    add = -location / scale

    residuals = []
    differences = []
    rho = []
    baseline_locations = []
    for sample in samples:
        baseline_locations.append(np.median(sample.baseline_raw, axis=0))
        centered = sample.baseline_raw - np.median(
            sample.baseline_raw, axis=0, keepdims=True
        )
        residuals.append(centered)
        differences.append(np.diff(centered, axis=0) / math.sqrt(2.0))
        rho.append(_lag_one(centered))
    residual = np.concatenate(residuals, axis=0)
    white = np.concatenate(differences, axis=0)
    baseline_location = np.asarray(baseline_locations)
    noise = {
        "channels": WRENCH_CHANNELS,
        "air_residual_cov": np.cov(residual, rowvar=False, ddof=1),
        "air_residual_std": np.std(residual, axis=0, ddof=1),
        "white_difference_cov": np.cov(white, rowvar=False, ddof=1),
        "white_difference_std": np.std(white, axis=0, ddof=1),
        "colored_rho_median": np.median(np.asarray(rho), axis=0),
        "raw_air_baseline_location_mean": np.mean(
            baseline_location, axis=0
        ),
        "raw_air_baseline_location_cov": np.cov(
            baseline_location, rowvar=False, ddof=1
        ),
        "raw_air_baseline_note": (
            "contains installed-tool gravity/payload and is provenance only; "
            "do not add it to baseline-relative simulated contact wrench"
        ),
    }
    affine = {
        "channels": SEQUENCE_CHANNELS,
        "physical_location": location,
        "physical_robust_scale": scale,
        "real_to_normalized_multiply": multiply,
        "real_to_normalized_add": add,
        "formula": "normalized = physical * multiply + add",
        "claim_boundary": (
            "real-domain feature normalization; paired simulation/real physical "
            "gain is not identifiable from these campaigns"
        ),
    }
    return affine, noise


def _fit_drift(clean_samples: list[RealSample]) -> dict[str, Any]:
    ordered = sorted(
        clean_samples, key=lambda sample: float(np.median(sample.baseline_stamp_s))
    )
    if len(ordered) < 3:
        raise ValueError("clean baseline drift fit requires at least three points")
    time_s = np.asarray(
        [float(np.median(sample.baseline_stamp_s)) for sample in ordered]
    )
    time_s -= time_s[0]
    location = np.asarray(
        [np.median(sample.baseline_raw, axis=0) for sample in ordered]
    )
    pairwise_slopes = []
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            elapsed = time_s[second] - time_s[first]
            if elapsed > 0.0:
                pairwise_slopes.append(
                    (location[second] - location[first]) / elapsed
                )
    slope = np.median(np.asarray(pairwise_slopes), axis=0)
    detrended = location - time_s[:, None] * slope[None, :]
    elapsed = np.diff(time_s)
    if np.any(elapsed <= 0.0):
        raise ValueError("clean baseline timestamps are not strictly ordered")
    increment_per_sqrt_s = np.diff(detrended, axis=0) / np.sqrt(
        elapsed[:, None]
    )
    return {
        "channels": WRENCH_CHANNELS,
        "fit_partition": "clean_candidate",
        "clean_baseline_count": len(ordered),
        "random_walk_std_per_sqrt_s": np.sqrt(
            np.mean(increment_per_sqrt_s * increment_per_sqrt_s, axis=0)
        ),
        "observed_linear_slope_median_per_s": slope,
        "fit_elapsed_s": float(time_s[-1]),
        "claim_boundary": (
            "short-run engineering drift fitted only from clean per-point air "
            "baselines; holdout, quarantine, and sentinel do not estimate parameters"
        ),
    }


def _sentinel_repeatability_validation(
    all_samples: list[RealSample], size_samples: list[RealSample]
) -> dict[str, Any]:
    by_id = {sample.sample_id: sample for sample in all_samples}
    per_sqrt_s = []
    per_s = []
    pair_records = []
    for sentinel in size_samples:
        if sentinel.partition not in DRIFT_PARTITIONS:
            continue
        source_id = str(sentinel.attempt["duplicate_of_sample_id"])
        if source_id not in by_id:
            raise ValueError(f"{sentinel.sample_id} source {source_id} missing")
        source = by_id[source_id]
        signed_delta_t = float(
            np.median(sentinel.baseline_stamp_s)
            - np.median(source.baseline_stamp_s)
        )
        delta_t = abs(signed_delta_t)
        if delta_t <= 0.0:
            raise ValueError(f"{sentinel.sample_id} has zero repeat interval")
        delta = np.median(sentinel.baseline_raw, axis=0) - np.median(
            source.baseline_raw, axis=0
        )
        per_sqrt_s.append(delta / math.sqrt(delta_t))
        per_s.append(delta / signed_delta_t)
        pair_records.append(
            {
                "sentinel_sample_id": sentinel.sample_id,
                "source_sample_id": source_id,
                "elapsed_s": delta_t,
                "sentinel_minus_source_time_s": signed_delta_t,
                "baseline_delta": delta,
            }
        )
    if not per_sqrt_s:
        raise ValueError("no sentinel drift pairs")
    sqrt_values = np.asarray(per_sqrt_s)
    slope_values = np.asarray(per_s)
    return {
        "channels": WRENCH_CHANNELS,
        "pair_count": len(pair_records),
        "sentinel_pairs": pair_records,
        "observed_delta_per_sqrt_s_rms": np.sqrt(
            np.mean(sqrt_values * sqrt_values, axis=0)
        ),
        "observed_linear_slope_median_per_s": np.median(
            slope_values, axis=0
        ),
        "claim_boundary": (
            "sentinel is validation-only repeatability evidence and estimates no "
            "calibration parameter; it is not an Allan-variance certificate"
        ),
    }


def _fit_timing(samples: list[RealSample]) -> dict[str, Any]:
    source_dt = np.concatenate(
        [np.diff(sample.source_stamp_s) for sample in samples]
    )
    wrench_age = np.concatenate(
        [sample.wrench_age_s for sample in samples]
    )
    tick_lateness = np.concatenate(
        [sample.tick_lateness_s for sample in samples]
    )
    command_gap = np.asarray(
        [
            finite_float(sample.attempt["cone_maximum_command_gap_s"])
            for sample in samples
        ]
    )
    scan_duration = np.asarray(
        [
            sample.source_stamp_s[-1] - sample.source_stamp_s[0]
            for sample in samples
        ]
    )
    return {
        "source_sample_dt_s": distribution(source_dt),
        "effective_source_rate_hz_from_median_dt": float(
            1.0 / np.median(source_dt)
        ),
        "wrench_age_s": distribution(wrench_age),
        "directservo_tick_lateness_s": distribution(tick_lateness),
        "maximum_command_gap_s": distribution(command_gap),
        "scan_source_duration_s": distribution(scan_duration),
        "replay_phase_jitter_std_bins": float(
            np.std(source_dt, ddof=1)
            / max(np.mean(source_dt), 1.0e-12)
        ),
    }


def _fit_force(samples: list[RealSample]) -> dict[str, Any]:
    settle_mean = np.asarray(
        [
            finite_float(sample.attempt["contact_settle_normal_mean_n"])
            for sample in samples
        ]
    )
    settle_std = np.asarray(
        [
            finite_float(sample.attempt["contact_settle_normal_std_n"])
            for sample in samples
        ]
    )
    scan_normal = np.concatenate(
        [sample.sequence[2, :] for sample in samples]
    )
    by_state = {}
    for state in CONTACT_STATES:
        selected = [sample for sample in samples if sample.contact_state == state]
        by_state[state] = {
            "sample_count": len(selected),
            "scan_normal_force_n": distribution(
                np.concatenate([sample.sequence[2, :] for sample in selected])
            ),
            "settle_normal_mean_n": distribution(
                np.asarray(
                    [
                        sample.attempt["contact_settle_normal_mean_n"]
                        for sample in selected
                    ],
                    dtype=float,
                )
            ),
        }
    return {
        "contact_settle_normal_mean_n": distribution(settle_mean),
        "contact_settle_normal_std_n": distribution(settle_std),
        "scan_normal_force_n": distribution(scan_normal),
        "by_contact_state": by_state,
        "sampling_target_down_force_n": 3.5,
        "claim_boundary": (
            "empirical load distribution for observation replay; it is not an "
            "identified contact stiffness or a new controller force target"
        ),
    }


def fit_calibration(
    samples: list[RealSample], source_hashes: dict[str, Any]
) -> dict[str, Any]:
    clean = [
        sample for sample in samples if sample.partition in FIT_PARTITIONS
    ]
    if len(clean) != 48:
        raise ValueError(f"expected 48 clean fit samples, got {len(clean)}")
    frame_phase_by_size = {}
    affine_by_size = {}
    noise_by_size = {}
    drift_by_size = {}
    timing_by_size = {}
    force_by_size = {}
    sentinel_validation_by_size = {}
    for size in SIZE_ORDER:
        selected = [sample for sample in clean if sample.size == size]
        size_all = [sample for sample in samples if sample.size == size]
        frame_phase_by_size[size] = _fit_frame_phase(selected)
        affine_by_size[size], noise_by_size[size] = _fit_channel_noise(
            selected
        )
        drift_by_size[size] = _fit_drift(selected)
        sentinel_validation_by_size[size] = (
            _sentinel_repeatability_validation(samples, size_all)
        )
        timing_by_size[size] = _fit_timing(selected)
        force_by_size[size] = _fit_force(selected)

    partition_counts = {
        partition: sum(sample.partition == partition for sample in samples)
        for partition in PARTITION_COUNTS
    }
    config = {
        "schema_version": 1,
        "calibration_id": "real_calibrated_v1",
        "created_utc": utc_now(),
        "authority": (
            "REAL_DEVICE_FRESH_RUN_OBSERVATION_CALIBRATION; exact CAD retained; "
            "simulation remains secondary synthetic evidence"
        ),
        "source_policy": {
            "allowlisted_runs": {
                size: run_dir.name for size, run_dir in RUN_ALLOWLIST.items()
            },
            "known_rejected_run": KNOWN_REJECTED_RUN.name,
            "samples_total": len(samples),
            "fit_samples": len(clean),
            "fit_partitions": FIT_PARTITIONS,
            "sentinel_role": "validation_only_never_fit",
            "excluded_from_fit": ("holdout", "quarantine", "sentinel"),
            "partition_counts": partition_counts,
        },
        "frame_phase_transform": {
            "by_size": frame_phase_by_size,
            "online_coordinate_contract": (
                "fixture XY to KUKA WORLD XY fitted from each fresh run; conical "
                "phase response is expressed relative to the true inward direction"
            ),
        },
        "channel_affine": {
            "by_size": affine_by_size,
            "sequence_channels": SEQUENCE_CHANNELS,
        },
        "noise_cov": {
            "by_size": noise_by_size,
            "units": ("N", "N", "N", "N_m", "N_m", "N_m"),
        },
        "drift": {"by_size": drift_by_size},
        "timing": {"by_size": timing_by_size},
        "force": {"by_size": force_by_size},
        "validation": {
            "sentinel_repeatability": {
                "by_size": sentinel_validation_by_size,
                "parameter_fit_effect": "NONE",
            }
        },
        "domain_randomization": {
            "identified_from_real_runs": False,
            "variables": {
                "friction_slide": {
                    "distribution": "uniform",
                    "minimum": 0.40,
                    "maximum": 1.00,
                },
                "contact_time_constant_s": {
                    "distribution": "log_uniform",
                    "minimum": 0.02,
                    "maximum": 0.08,
                },
                "printed_polymer_stiffness_scale": {
                    "distribution": "log_uniform",
                    "minimum": 0.70,
                    "maximum": 1.30,
                },
                "printed_polymer_damping_scale": {
                    "distribution": "log_uniform",
                    "minimum": 0.70,
                    "maximum": 1.30,
                },
            },
            "claim_boundary": (
                "ranges are engineering domain randomization only; no friction, "
                "stiffness, damping, or contact-time parameter is presented as a "
                "fit from conical wrench traces"
            ),
        },
        "generation": {
            "generator": "deterministic_real_observation_replay",
            "phase_bins": PHASE_BINS,
            "sequence_channels": SEQUENCE_CHANNELS,
            "geometry_tasks": len(samples),
            "seeds_per_geometry_task": SEEDS_PER_GEOMETRY_TASK,
            "generated_records": len(samples) * SEEDS_PER_GEOMETRY_TASK,
            "seed_formula": (
                "202607280000 + 10000*size_index + 10*execution_index + repeat"
            ),
            "partition_preservation": (
                "source partition is immutable; only clean_candidate replay "
                "records are training-eligible"
            ),
        },
        "hashes": {
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "source_campaigns": source_hashes,
        },
        "claim_boundary": (
            "observation/frame/timing/load calibration plus deterministic replay; "
            "not paired MuJoCo system identification, capture calibration, or "
            "hardware control authorization"
        ),
    }
    config["hashes"]["fitted_payload_sha256"] = canonical_sha256(
        {
            key: config[key]
            for key in (
                "source_policy",
                "frame_phase_transform",
                "channel_affine",
                "noise_cov",
                "drift",
                "timing",
                "force",
                "validation",
                "domain_randomization",
                "generation",
            )
        }
    )
    return jsonable(config)


def _circular_shift(sequence: np.ndarray, shift_bins: float) -> np.ndarray:
    bins = sequence.shape[1]
    source = (np.arange(bins, dtype=float) - shift_bins) % bins
    lower = np.floor(source).astype(int)
    upper = (lower + 1) % bins
    weight = source - lower
    return (
        sequence[:, lower] * (1.0 - weight)[None, :]
        + sequence[:, upper] * weight[None, :]
    )


def _draw_replay(
    sample: RealSample,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    size = sample.size
    timing = config["timing"]["by_size"][size]
    noise = config["noise_cov"]["by_size"][size]
    drift = config["drift"]["by_size"][size]
    force = config["force"]["by_size"][size]
    affine = config["channel_affine"]["by_size"][size]

    phase_shift = float(
        rng.normal(0.0, timing["replay_phase_jitter_std_bins"])
    )
    replay = _circular_shift(sample.sequence.copy(), phase_shift)
    target_force = max(
        0.5,
        float(
            rng.normal(
                force["contact_settle_normal_mean_n"]["mean"],
                max(force["contact_settle_normal_mean_n"]["std"], 0.05),
            )
        ),
    )
    source_force = max(
        finite_float(sample.attempt["contact_settle_normal_mean_n"]), 0.5
    )
    force_scale = target_force / source_force
    replay[:6, :] *= force_scale

    white_cov = np.asarray(noise["white_difference_cov"], dtype=float)
    white_cov = 0.5 * (white_cov + white_cov.T)
    eigenvalue, eigenvector = np.linalg.eigh(white_cov)
    white_cov = (eigenvector * np.maximum(eigenvalue, 0.0)) @ eigenvector.T
    innovation = rng.multivariate_normal(
        np.zeros(6), white_cov, size=PHASE_BINS
    ).T
    rho = np.clip(
        np.asarray(noise["colored_rho_median"], dtype=float), 0.0, 0.995
    )
    colored = np.zeros_like(innovation)
    for index in range(1, PHASE_BINS):
        colored[:, index] = (
            rho * colored[:, index - 1]
            + np.sqrt(1.0 - rho * rho) * innovation[:, index]
        )
    replay[:6, :] += colored

    drift_std = np.asarray(
        drift["random_walk_std_per_sqrt_s"], dtype=float
    )
    sample_dt = timing["source_sample_dt_s"]["median"]
    random_walk = np.cumsum(
        rng.normal(
            0.0,
            drift_std[:, None] * math.sqrt(sample_dt),
            size=(6, PHASE_BINS),
        ),
        axis=1,
    )
    replay[:6, :] += random_walk

    multiply = np.asarray(
        affine["real_to_normalized_multiply"], dtype=float
    )[:, None]
    add = np.asarray(affine["real_to_normalized_add"], dtype=float)[:, None]
    normalized = replay * multiply + add
    metadata = {
        "seed": seed,
        "phase_shift_bins": phase_shift,
        "sampled_settle_force_n": target_force,
        "force_scale": force_scale,
        "source_sample_id": sample.sample_id,
    }
    return replay.astype(np.float32), normalized.astype(np.float32), metadata


def build_replay(
    samples: list[RealSample], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    physical = []
    normalized = []
    records = []
    for sample in sorted(
        samples,
        key=lambda item: (
            SIZE_ORDER.index(item.size),
            int(item.attempt["execution_index"]),
        ),
    ):
        size_index = SIZE_ORDER.index(sample.size)
        execution_index = int(sample.attempt["execution_index"])
        for repeat in range(SEEDS_PER_GEOMETRY_TASK):
            seed = 202607280000 + 10000 * size_index + 10 * execution_index + repeat
            replay, standard, draw = _draw_replay(sample, config, seed)
            record = {
                "record_index": len(records),
                "replay_id": f"{sample.sample_id}__realcal_seed{repeat}",
                "source_sample_id": sample.sample_id,
                "source_run_id": sample.run_dir.name,
                "size": sample.size,
                "execution_index": execution_index,
                "partition": sample.partition,
                "contact_state": sample.contact_state,
                "direction_fixture_deg": finite_float(
                    sample.attempt["true_direction_fixture_deg"]
                ),
                "measured_contact_fixture_xy_mm": sample.attempt[
                    "measured_contact_fixture_xy_mm"
                ],
                "seed": seed,
                "training_use": (
                    "eligible_real_calibrated_replay"
                    if sample.partition == "clean_candidate"
                    else "forbidden_source_partition_preserved"
                ),
                "source_cone_sha256": sample.attempt["file_hashes"][
                    "cone_csv_sha256"
                ],
                "draw": draw,
            }
            physical.append(replay)
            normalized.append(standard)
            records.append(record)
    arrays = {
        "sequence_physical": np.asarray(physical, dtype=np.float32),
        "sequence_normalized": np.asarray(normalized, dtype=np.float32),
        "seed": np.asarray([record["seed"] for record in records], dtype=np.int64),
        "direction_deg": np.asarray(
            [record["direction_fixture_deg"] for record in records],
            dtype=np.float32,
        ),
        "measured_offset_mm": np.asarray(
            [
                record["measured_contact_fixture_xy_mm"]
                for record in records
            ],
            dtype=np.float32,
        ),
        "size_index": np.asarray(
            [SIZE_ORDER.index(record["size"]) for record in records],
            dtype=np.int8,
        ),
        "sample_id": np.asarray(
            [record["source_sample_id"] for record in records], dtype="U96"
        ),
        "partition": np.asarray(
            [record["partition"] for record in records], dtype="U24"
        ),
        "contact_state": np.asarray(
            [record["contact_state"] for record in records], dtype="U32"
        ),
    }
    return arrays, records


def build_bundle(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if config_path.exists():
        raise FileExistsError(f"refusing to overwrite config: {config_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    samples, source_hashes = load_allowlisted_campaigns()
    config = fit_calibration(samples, source_hashes)
    arrays, records = build_replay(samples, config)

    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir()
    try:
        replay_path = staging / "calibrated_observation_replay_v1.npz"
        np.savez_compressed(replay_path, **arrays)
        task_path = staging / "simulation_tasks.json"
        atomic_json(
            task_path,
            {
                "schema_version": 1,
                "calibration_id": config["calibration_id"],
                "records": records,
            },
        )
        manifest = {
            "schema_version": 1,
            "calibration_id": config["calibration_id"],
            "created_utc": utc_now(),
            "geometry_tasks": len(samples),
            "seeds_per_geometry_task": SEEDS_PER_GEOMETRY_TASK,
            "records": len(records),
            "sequence_shape": list(arrays["sequence_physical"].shape),
            "sequence_channels": SEQUENCE_CHANNELS,
            "training_eligible_records": sum(
                record["partition"] == "clean_candidate"
                for record in records
            ),
            "nontraining_records": sum(
                record["partition"] != "clean_candidate"
                for record in records
            ),
            "partition_record_counts": {
                partition: sum(
                    record["partition"] == partition for record in records
                )
                for partition in PARTITION_COUNTS
            },
            "source_run_ids": {
                size: run.name for size, run in RUN_ALLOWLIST.items()
            },
            "known_rejected_run": KNOWN_REJECTED_RUN.name,
            "artifacts": {
                replay_path.name: sha256_file(replay_path),
                task_path.name: sha256_file(task_path),
            },
            "claim_boundary": config["claim_boundary"],
        }
        atomic_json(staging / "manifest.json", manifest)
        output_dir.mkdir()
        for path in staging.iterdir():
            path.replace(output_dir / path.name)
        staging.rmdir()
        atomic_json(config_path, config)
    except Exception:
        # Preserve generated staging evidence for diagnosis; never replace a
        # previously completed bundle.
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle(args.config, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
