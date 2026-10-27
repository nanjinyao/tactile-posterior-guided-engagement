from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.m8_jin_style_direction import (
    DEFAULT_MODEL_ROOT as DEFAULT_JIN_MODEL_ROOT,
    JinStyleDirectionRuntime,
    circular_error_deg,
    direction_class,
)
from src.run_m8_fair_policy_benchmark import (
    MODEL_ROOT as DEFAULT_IMPROVED_MODEL_ROOT,
    PoseTask,
    build_tasks,
)
from src.run_m8_strict_complete_loop_audit import (
    ACTIVE_TIME_BUDGET_S,
    _capture_success,
    _failure_category,
    _run_with_runtime,
    _strict_bottom_success,
)
from src.sample_m8_conical_pilot import SPEC_PATH, _load_trial
import src.train_m8_direction_models as improved_training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTION_SOURCE = (
    ROOT
    / "outputs/current/evaluations/p1v6_m8_jin_comparison_direction_audit"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/evaluations/m8_jin_style_vs_improved"
)
DEFAULT_FIGURE = (
    ROOT
    / "outputs/current/media/figures/paper/m8_jin_style_vs_improved_fingerprint_direct.png"
)
AMENDMENT_FILENAME = "contract_amendment_derived_rebuild.json"
TASK_SEED = 20260733
TASK_COUNT = 48
EXPECTED_DIRECTION_TRIALS = 180
BOOTSTRAP_SEED = 20260734
BOOTSTRAP_RESAMPLES = 20_000
METHODS = ("improved_fingerprint", "jin_2021_style_proxy")
METHOD_LABELS = {
    "improved_fingerprint": "Improved fingerprint",
    "jin_2021_style_proxy": "Jin-2021-style proxy",
}
SCHEMA_VERSION = 1


_IMPROVED_RUNTIME: M8DirectionEnsemble | None = None
_JIN_RUNTIME: JinStyleDirectionRuntime | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _observed_mtime(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "observed_filesystem_mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(timespec="microseconds"),
        "observed_filesystem_mtime_ns": stat.st_mtime_ns,
    }


def _experiment_contract(
    *,
    direction_source: Path,
    improved_model_root: Path,
    jin_model_root: Path,
    tasks: list[PoseTask],
    expected_direction_trials: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "M8_Jin_2021_style_proxy_vs_improved_fingerprint",
        "direction_layer": {
            "source": _display_path(direction_source),
            "expected_trials": expected_direction_trials,
            "training_use_required_to_contain": "forbidden",
            "shared_input_contract": (
                "Each NPZ is loaded once; the identical Fz/Tx/Ty/tip_z phase "
                "sequence is passed to both frozen runtimes."
            ),
            "continuous_metrics_deg": [
                "median",
                "p90",
                "within_7.5",
                "within_15",
                "greater_than_90",
            ],
            "common_classification": (
                "Both continuous control directions and the true direction are "
                "quantized by the same 12-sector M8 proxy rule."
            ),
        },
        "task_layer": {
            "builder": "src.run_m8_fair_policy_benchmark.build_tasks",
            "seed": TASK_SEED,
            "poses": len(tasks),
            "wrench_noise": True,
            "policy": (
                "src.run_m8_strict_complete_loop_audit._run_with_runtime"
                "(task, 'direct_first', runtime)"
            ),
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "paired_execution": (
                "Same PoseTask, nuisance parameters, sensor seed, strict direct-ray "
                "controller, capture handoff, CAD-bottom verifier, and 130 s budget. "
                "Only the frozen direction estimator changes."
            ),
            "tasks": [asdict(task) for task in tasks],
        },
        "estimators": {
            "improved_fingerprint": {
                "model_root": _display_path(improved_model_root),
                "runtime": "src.m8_direction_runtime.M8DirectionEnsemble",
            },
            "jin_2021_style_proxy": {
                "model_root": _display_path(jin_model_root),
                "runtime": "src.m8_jin_style_direction.JinStyleDirectionRuntime",
                "status": (
                    "M8 proxy, not a faithful reproduction: Jin et al. do not "
                    "specify an M8 label set, exact rasterization/normalization, "
                    "all CNN hyperparameters, or the direct-ray controller."
                ),
            },
        },
        "completion_rule": (
            "A completed manifest is immutable. Resume reuses complete paired pose "
            "records and never overwrites a completed experiment."
        ),
        "claim_boundary": (
            "Fresh paired MuJoCo M8 evidence only. The Jin-style arm is a frozen "
            "engineering proxy, not a faithful paper reproduction. Because the M8 "
            "geometry, labels, data distribution, and controller differ, these "
            "numbers cannot be compared directly with the paper's reported 97.4% "
            "square classification accuracy. No hardware or universal superiority "
            "claim is licensed."
        ),
    }


def _ensure_contract(output: Path, expected: dict[str, Any]) -> None:
    path = output / "task_contract.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "existing task_contract.json differs from the requested frozen contract"
            )
        return
    _atomic_json(path, expected)


def _completed_manifest_is_valid(
    output: Path, expected_contract: dict[str, Any]
) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    contract_path = output / "task_contract.json"
    if not contract_path.exists():
        raise ValueError("completed manifest has no task_contract.json")
    actual_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if actual_contract != expected_contract:
        raise ValueError("completed experiment contract does not match this request")
    if manifest.get("task_contract_sha256") != _sha256(contract_path):
        raise ValueError("completed task contract hash mismatch")
    _verify_hashed_path_entries(manifest.get("artifacts", {}), "artifacts")
    _verify_hashed_path_entries(manifest.get("sources", {}), "sources")
    _verify_hashed_path_entries(
        manifest.get("post_completion_contract_amendment", {}),
        "post_completion_contract_amendment",
    )
    return True


def _verify_hashed_path_entries(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = Path(str(value["path"]))
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists() or _sha256(path) != value["sha256"]:
                raise ValueError(f"completed provenance verification failed: {label}")
            return
        for key, item in value.items():
            _verify_hashed_path_entries(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_hashed_path_entries(item, f"{label}[{index}]")


def _continuous_direction_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    errors = np.asarray([row[f"{prefix}_error_deg"] for row in rows], dtype=float)
    sector_correct = np.asarray(
        [row[f"{prefix}_sector_correct"] for row in rows], dtype=bool
    )
    return {
        "samples": len(rows),
        "angle_error_median_deg": float(np.median(errors)),
        "angle_error_p90_deg": float(np.quantile(errors, 0.90)),
        "angle_within_7p5_count": int(np.sum(errors <= 7.5 + 1.0e-9)),
        "angle_within_7p5_rate": float(np.mean(errors <= 7.5 + 1.0e-9)),
        "angle_within_15_count": int(np.sum(errors <= 15.0 + 1.0e-9)),
        "angle_within_15_rate": float(np.mean(errors <= 15.0 + 1.0e-9)),
        "catastrophic_gt90_count": int(np.sum(errors > 90.0)),
        "catastrophic_gt90_rate": float(np.mean(errors > 90.0)),
        "common_12_sector_correct": int(np.sum(sector_correct)),
        "common_12_sector_accuracy": float(np.mean(sector_correct)),
    }


def _paired_base_pose_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Percentile CI for median paired absolute-error difference.

    ``base_pose_id`` is the preregistered cluster unit. In P1v6 it identifies
    one state/radius/local-angle/physical-sector pose and keeps its three
    nuisance repeats together. The estimand is
    median_i(|e_improved,i| - |e_proxy,i|), so negative values favor the
    improved estimator.
    """

    if resamples <= 0:
        raise ValueError("cluster bootstrap resamples must be positive")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        cluster = str(row["base_pose_id"])
        paired_difference = float(row["improved_error_deg"]) - float(
            row["jin_error_deg"]
        )
        grouped.setdefault(cluster, []).append(paired_difference)
    if len(grouped) < 2:
        raise ValueError("paired cluster bootstrap requires at least two base poses")
    clusters = [np.asarray(grouped[key], dtype=float) for key in sorted(grouped)]
    all_values = np.concatenate(clusters)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=float)
    cluster_count = len(clusters)
    cluster_sizes = {len(values) for values in clusters}
    if len(cluster_sizes) == 1:
        matrix = np.stack(clusters)
        batch_size = 1000
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            sampled = rng.integers(
                0, cluster_count, size=(stop - start, cluster_count)
            )
            bootstrap[start:stop] = np.median(
                matrix[sampled].reshape(stop - start, -1), axis=1
            )
    else:
        for index in range(resamples):
            sampled = rng.integers(0, cluster_count, size=cluster_count)
            bootstrap[index] = float(
                np.median(np.concatenate([clusters[item] for item in sampled]))
            )
    low, high = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    size_counts = Counter(len(values) for values in clusters)
    return {
        "estimand": (
            "median over paired scans of "
            "abs_error_improved_deg - abs_error_jin_proxy_deg"
        ),
        "point_estimate_deg": float(np.median(all_values)),
        "confidence_level": 0.95,
        "ci_percentile_low_deg": float(low),
        "ci_percentile_high_deg": float(high),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "interval_interpretation": (
            "The 95% interval includes zero; this paired-median P1v6 contrast "
            "does not establish a direction-error advantage."
            if low <= 0.0 <= high
            else "The 95% interval excludes zero."
        ),
        "method": "paired base-pose cluster bootstrap, percentile interval",
        "cluster_unit": (
            "base_pose_id = state/radius/local-angle/physical-sector; all nuisance "
            "repeats remain together"
        ),
        "cluster_count": cluster_count,
        "cluster_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "samples": len(rows),
        "seed": seed,
        "resamples": resamples,
        "negative_favors": "improved_fingerprint",
        "implementation_note": (
            "The cluster interval was preregistered; deterministic seed and "
            "resample count were fixed during the post-run consistency repair."
        ),
    }


def _evaluate_direction_layer(
    source: Path,
    improved_model_root: Path,
    jin_model_root: Path,
    expected_trials: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_path = source / "sampling_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"missing frozen direction contract: {contract_path}")
    source_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if "forbidden" not in str(source_contract.get("training_use", "")).lower():
        raise ValueError("direction comparison source is not frozen against training use")
    trial_paths = sorted((source / "trials").glob("*.npz"))
    if len(trial_paths) != expected_trials:
        raise ValueError(
            f"expected {expected_trials} frozen direction trials, found {len(trial_paths)}"
        )

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    improved_runtime = M8DirectionEnsemble(improved_model_root)
    jin_runtime = JinStyleDirectionRuntime(jin_model_root)
    rows: list[dict[str, Any]] = []
    for path in tqdm(
        trial_paths,
        desc="shared-scan Jin proxy vs improved",
        unit="scan",
        dynamic_ncols=True,
    ):
        metadata, phase_sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            raise ValueError(f"failed frozen direction trial: {path.name}")
        indices = [columns.index(name) for name in improved_training.INPUT_CHANNELS]
        shared_sequence = np.asarray(phase_sequence[indices], dtype=float)
        kwargs = {
            "target_down_force_n": float(metadata["target_down_force_n"]),
            "bit_radius_m": bit_radius_m,
        }
        improved = improved_runtime.infer_phase_sequence(shared_sequence, **kwargs)
        jin = jin_runtime.infer_phase_sequence(shared_sequence, **kwargs)
        true_angle = float(improved_training._direction_angle_deg(metadata))
        improved_angle = float(improved.circular_mean_direction_deg)
        jin_angle = float(jin.circular_mean_direction_deg)
        improved_error = float(
            circular_error_deg(np.asarray([improved_angle]), np.asarray([true_angle]))[0]
        )
        jin_error = float(
            circular_error_deg(np.asarray([jin_angle]), np.asarray([true_angle]))[0]
        )
        true_sector = direction_class(true_angle)
        rows.append(
            {
                "trial_id": str(metadata["trial_id"]),
                "base_pose_id": str(metadata["base_pose_id"]),
                "search_state": str(metadata["search_state"]),
                "radial_level": int(metadata["radial_level"]),
                "radius_mm": 1000.0 * float(metadata["radius_m"]),
                "local_angle_deg": float(metadata["base_angle_deg"]),
                "sector": int(metadata["rotation_index"]),
                "nuisance_id": str(metadata["nuisance_id"]),
                "shared_phase_sequence_sha256": _array_sha256(shared_sequence),
                "true_direction_deg": true_angle,
                "true_common_12_sector": true_sector,
                "improved_direction_deg": improved_angle,
                "improved_error_deg": improved_error,
                "improved_common_12_sector": direction_class(improved_angle),
                "improved_sector_correct": direction_class(improved_angle)
                == true_sector,
                "improved_confidence": float(improved.confidence),
                "jin_direction_deg": jin_angle,
                "jin_error_deg": jin_error,
                "jin_common_12_sector": direction_class(jin_angle),
                "jin_sector_correct": direction_class(jin_angle) == true_sector,
                "jin_confidence": float(jin.confidence),
                "improved_minus_jin_error_deg": improved_error - jin_error,
            }
        )

    paired_bootstrap = _paired_base_pose_cluster_bootstrap(rows)
    metrics = {
        "shared_raw_scan_for_both_estimators": True,
        "shared_scan_unique_hashes": len(
            {row["shared_phase_sequence_sha256"] for row in rows}
        ),
        "improved_fingerprint": _continuous_direction_metrics(rows, "improved"),
        "jin_2021_style_proxy": _continuous_direction_metrics(rows, "jin"),
        "paired_improved_minus_jin_absolute_error_median_deg": float(
            np.median([row["improved_minus_jin_error_deg"] for row in rows])
        ),
        "paired_base_pose_cluster_bootstrap": paired_bootstrap,
    }
    return rows, metrics


def _worker_init(improved_model_root: str, jin_model_root: str) -> None:
    global _IMPROVED_RUNTIME, _JIN_RUNTIME
    torch.set_num_threads(1)
    _IMPROVED_RUNTIME = M8DirectionEnsemble(Path(improved_model_root))
    _JIN_RUNTIME = JinStyleDirectionRuntime(Path(jin_model_root))


def _validate_task_pair(
    task: PoseTask,
    improved: dict[str, Any],
    jin: dict[str, Any],
) -> dict[str, Any]:
    left_hash = improved.get("fingerprint_sha256")
    right_hash = jin.get("fingerprint_sha256")
    if (left_hash is None) != (right_hash is None):
        raise ValueError(f"one-sided missing fingerprint for {task.pose_id}")
    if left_hash is not None and left_hash != right_hash:
        raise ValueError(f"fingerprint SHA mismatch for {task.pose_id}")
    return {
        "same_pose_nuisance_and_sensor_seed": True,
        "same_strict_direct_controller_and_130s_budget": True,
        "only_direction_estimator_changes": True,
        "fingerprint_available_for_both": left_hash is not None,
        "fingerprint_sha256_match": None if left_hash is None else True,
        "shared_fingerprint_sha256": left_hash,
    }


def _run_task_pair(task: PoseTask) -> dict[str, Any]:
    if _IMPROVED_RUNTIME is None or _JIN_RUNTIME is None:
        raise RuntimeError("paired worker runtimes are not initialized")
    started = time.perf_counter()
    improved = asdict(
        _run_with_runtime(task, "direct_first", _IMPROVED_RUNTIME)
    )
    jin = asdict(_run_with_runtime(task, "direct_first", _JIN_RUNTIME))
    return {
        "task": asdict(task),
        "methods": {
            "improved_fingerprint": improved,
            "jin_2021_style_proxy": jin,
        },
        "fairness": _validate_task_pair(task, improved, jin),
        "worker_elapsed_s": time.perf_counter() - started,
        "created_utc": _utc_now(),
    }


def _safe_termination(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("hard_stop")
        and float(row.get("total_active_time_s") or 0.0)
        <= ACTIVE_TIME_BUDGET_S + 1.0e-9
        and abs(float(row.get("fallback_budget_extension_s") or 0.0))
        <= 1.0e-12
    )


def _circular_absolute_error_deg(predicted_deg: float, true_deg: float) -> float:
    return abs((float(predicted_deg) - float(true_deg) + 180.0) % 360.0 - 180.0)


def _direction_error_distribution(errors: np.ndarray) -> dict[str, Any]:
    if errors.ndim != 1 or len(errors) == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("direction-error distribution must be finite and non-empty")
    return {
        "samples": len(errors),
        "error_median_deg": float(np.median(errors)),
        "error_p90_deg": float(np.quantile(errors, 0.90)),
        "within_7p5_count": int(np.sum(errors <= 7.5 + 1.0e-9)),
        "within_7p5_rate": float(np.mean(errors <= 7.5 + 1.0e-9)),
        "within_15_count": int(np.sum(errors <= 15.0 + 1.0e-9)),
        "within_15_rate": float(np.mean(errors <= 15.0 + 1.0e-9)),
        "gt90_count": int(np.sum(errors > 90.0)),
        "gt90_rate": float(np.mean(errors > 90.0)),
    }


def _method_task_summary(
    records: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    rows = [record["methods"][method] for record in records]
    control_direction_errors = np.asarray(
        [
            _circular_absolute_error_deg(
                float(row["control_direction_deg"]),
                float(row["true_direction_deg"]),
            )
            for row in rows
            if row.get("control_direction_deg") is not None
            and row.get("true_direction_deg") is not None
        ],
        dtype=float,
    )
    if len(control_direction_errors) != len(rows):
        raise ValueError(
            f"executed control direction is missing for {method}: "
            f"{len(control_direction_errors)}/{len(rows)}"
        )
    top_class_direction_errors = np.asarray(
        [
            float(row["direction_error_deg"])
            for row in rows
            if row.get("direction_error_deg") is not None
        ],
        dtype=float,
    )
    if len(top_class_direction_errors) != len(rows):
        raise ValueError(
            f"top-class direction diagnostic is missing for {method}: "
            f"{len(top_class_direction_errors)}/{len(rows)}"
        )
    control_metrics = _direction_error_distribution(control_direction_errors)
    top_class_metrics = _direction_error_distribution(top_class_direction_errors)
    capture = np.asarray([_capture_success(row) for row in rows], dtype=bool)
    strict = np.asarray([_strict_bottom_success(row) for row in rows], dtype=bool)
    hard_stop = np.asarray([bool(row.get("hard_stop")) for row in rows], dtype=bool)
    safe = np.asarray([_safe_termination(row) for row in rows], dtype=bool)
    budget_violation = np.asarray(
        [
            float(row.get("total_active_time_s") or 0.0)
            > ACTIVE_TIME_BUDGET_S + 1.0e-9
            for row in rows
        ],
        dtype=bool,
    )
    return {
        "trials": len(rows),
        "executed_control_direction_definition": (
            "circular absolute error between frozen RunResult.control_direction_deg "
            "(the direction actually executed by direct-first) and "
            "RunResult.true_direction_deg"
        ),
        "control_direction_error_samples": control_metrics["samples"],
        "control_direction_error_median_deg": control_metrics["error_median_deg"],
        "control_direction_error_p90_deg": control_metrics["error_p90_deg"],
        "control_direction_within_7p5_count": control_metrics["within_7p5_count"],
        "control_direction_within_7p5_rate": control_metrics["within_7p5_rate"],
        "control_direction_within_15_count": control_metrics["within_15_count"],
        "control_direction_within_15_rate": control_metrics["within_15_rate"],
        "control_direction_gt90_count": control_metrics["gt90_count"],
        "control_direction_gt90_rate": control_metrics["gt90_rate"],
        "top_class_direction_diagnostic": {
            "role": "diagnostic only; this is not the direct-first ray that was executed",
            "source": (
                "frozen RunResult.direction_error_deg = circular absolute error of "
                "RunResult.predicted_direction_deg (ensemble top class / top_direction_deg)"
            ),
            **top_class_metrics,
        },
        "first_ray_capture_count": int(np.sum(capture)),
        "first_ray_capture_rate": float(np.mean(capture)),
        "strict_cad_bottom_count": int(np.sum(strict)),
        "strict_cad_bottom_rate": float(np.mean(strict)),
        "hard_stop_count": int(np.sum(hard_stop)),
        "hard_stop_rate": float(np.mean(hard_stop)),
        "safe_termination_count": int(np.sum(safe)),
        "safe_termination_rate": float(np.mean(safe)),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "budget_violation_count": int(np.sum(budget_violation)),
        "fallback_budget_extension_nonzero_count": sum(
            abs(float(row.get("fallback_budget_extension_s") or 0.0)) > 1.0e-12
            for row in rows
        ),
        "total_active_time_median_s": float(
            np.median([float(row["total_active_time_s"]) for row in rows])
        ),
        "failure_categories": dict(
            sorted(Counter(_failure_category(row) for row in rows).items())
        ),
    }


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    lower = min(int(left_only), int(right_only))
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_binary(
    records: list[dict[str, Any]], predicate
) -> dict[str, Any]:
    outcomes = [
        (
            bool(predicate(record["methods"]["improved_fingerprint"])),
            bool(predicate(record["methods"]["jin_2021_style_proxy"])),
        )
        for record in records
    ]
    improved_only = sum(pair == (True, False) for pair in outcomes)
    jin_only = sum(pair == (False, True) for pair in outcomes)
    return {
        "both_event": sum(pair == (True, True) for pair in outcomes),
        "improved_only_event": improved_only,
        "jin_proxy_only_event": jin_only,
        "neither_event": sum(pair == (False, False) for pair in outcomes),
        "discordant_pairs": improved_only + jin_only,
        "mcnemar_exact_two_sided_p": _exact_mcnemar(improved_only, jin_only),
    }


def _paired_task_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first_ray_capture": _paired_binary(records, _capture_success),
        "strict_cad_bottom": _paired_binary(records, _strict_bottom_success),
        "hard_stop": _paired_binary(records, lambda row: bool(row.get("hard_stop"))),
        "safe_termination": _paired_binary(records, _safe_termination),
    }


def _task_pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        task = record["task"]
        improved = record["methods"]["improved_fingerprint"]
        jin = record["methods"]["jin_2021_style_proxy"]
        if improved.get("true_direction_deg") != jin.get("true_direction_deg"):
            raise ValueError(
                f"paired true-direction mismatch for frozen pose {task['pose_id']}"
            )
        true_direction_deg = float(improved["true_direction_deg"])
        improved_control_direction_deg = float(improved["control_direction_deg"])
        jin_control_direction_deg = float(jin["control_direction_deg"])
        rows.append(
            {
                "pose_id": task["pose_id"],
                "radius_mm": 1000.0 * float(task["radius_m"]),
                "angle_deg": math.degrees(float(task["angle_rad"])) % 360.0,
                "sensor_noise_seed": task["sensor_noise_seed"],
                "sensor_delay_s": task["sensor_delay_s"],
                "shared_fingerprint_sha256": record["fairness"][
                    "shared_fingerprint_sha256"
                ],
                "fingerprint_sha256_match": record["fairness"][
                    "fingerprint_sha256_match"
                ],
                "true_direction_deg": true_direction_deg,
                "improved_control_direction_deg": improved_control_direction_deg,
                "jin_control_direction_deg": jin_control_direction_deg,
                "improved_control_direction_error_deg": _circular_absolute_error_deg(
                    improved_control_direction_deg, true_direction_deg
                ),
                "jin_control_direction_error_deg": _circular_absolute_error_deg(
                    jin_control_direction_deg, true_direction_deg
                ),
                "improved_top_class_direction_error_deg": improved.get(
                    "direction_error_deg"
                ),
                "jin_top_class_direction_error_deg": jin.get("direction_error_deg"),
                "improved_first_ray_capture": _capture_success(improved),
                "jin_first_ray_capture": _capture_success(jin),
                "improved_strict_cad_bottom": _strict_bottom_success(improved),
                "jin_strict_cad_bottom": _strict_bottom_success(jin),
                "improved_hard_stop": bool(improved.get("hard_stop")),
                "jin_hard_stop": bool(jin.get("hard_stop")),
                "improved_safe_termination": _safe_termination(improved),
                "jin_safe_termination": _safe_termination(jin),
                "improved_active_time_s": improved.get("total_active_time_s"),
                "jin_active_time_s": jin.get("total_active_time_s"),
                "improved_failure_category": _failure_category(improved),
                "jin_failure_category": _failure_category(jin),
            }
        )
    return rows


def _write_summary(output: Path, report: dict[str, Any]) -> None:
    direction = report["direction_layer"]["methods"]
    bootstrap = report["direction_layer"]["paired_base_pose_cluster_bootstrap"]
    task = report["task_layer"]["methods"]
    paired = report["task_layer"]["paired"]
    lines = [
        "# M8 Jin-2021-Style Proxy vs Improved Fingerprint",
        "",
        "This is a fresh paired MuJoCo comparison. `Jin-2021-style` means an M8 "
        "proxy with locked paper-inspired preprocessing/CNN structure, not a faithful reproduction.",
        "",
        "## Structured fresh P1v6 direction stress (mixed result)",
        "",
        "| estimator | n | error median / p90 | within 7.5 deg | within 15 deg | >90 deg | common 12-sector accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = direction[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['samples']} | "
            f"{item['angle_error_median_deg']:.2f} / {item['angle_error_p90_deg']:.2f} deg | "
            f"{item['angle_within_7p5_rate']:.3f} | {item['angle_within_15_rate']:.3f} | "
            f"{item['catastrophic_gt90_rate']:.3f} | {item['common_12_sector_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Paired base-pose cluster bootstrap for "
                "`median(|e_improved| - |e_proxy|)`: "
                f"{bootstrap['point_estimate_deg']:.3f} deg, 95% percentile CI "
                f"[{bootstrap['ci_percentile_low_deg']:.3f}, "
                f"{bootstrap['ci_percentile_high_deg']:.3f}] deg; "
                f"{bootstrap['cluster_count']} `base_pose_id` clusters, "
                f"{bootstrap['resamples']} resamples, seed {bootstrap['seed']}. "
                "Negative favors the improved estimator. Each cluster is one "
                "state/radius/local-angle/physical-sector pose and retains all "
                f"three nuisance repeats. {bootstrap['interval_interpretation']}"
            ),
            "",
            "Each P1v6 NPZ phase sequence was loaded once and passed unchanged to both frozen runtimes. "
            "This structured radial/sector stress result is mixed: the improved estimator has the lower "
            "median and higher within-15-deg rate, while the proxy has the better p90, within-7.5-deg, "
            "catastrophic-tail, and common-12-sector metrics. It is not evidence of uniformly better "
            "direction accuracy.",
            "",
            "## Fresh strict-48 operational direction distribution",
            "",
            "Operational errors are recomputed from each frozen `RunResult.control_direction_deg` "
            "against `RunResult.true_direction_deg` on the same 48 noisy poses used by the direct-first "
            "task comparison. `control_direction_deg` is the direction actually executed by the ray. "
            "This is distinct from the P1v6 structured stress set; no simulator rollout was rerun.",
            "",
            "| estimator | n | executed-control error median / p90 | within 7.5 deg | within 15 deg | >90 deg |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = task[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['control_direction_error_samples']} | "
            f"{item['control_direction_error_median_deg']:.2f} / "
            f"{item['control_direction_error_p90_deg']:.2f} deg | "
            f"{item['control_direction_within_7p5_rate']:.3f} | "
            f"{item['control_direction_within_15_rate']:.3f} | "
            f"{item['control_direction_gt90_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Top-class diagnostic (not the executed ray)",
            "",
            "The frozen `RunResult.direction_error_deg` is retained only as a diagnostic of "
            "`predicted_direction_deg` (the ensemble top class / `top_direction_deg`). It must not be "
            "read as strict-48 controller performance.",
            "",
            "| estimator | n | top-class error median / p90 | within 7.5 deg | within 15 deg | >90 deg |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = task[method]["top_class_direction_diagnostic"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['samples']} | "
            f"{item['error_median_deg']:.2f} / {item['error_p90_deg']:.2f} deg | "
            f"{item['within_7p5_rate']:.3f} | {item['within_15_rate']:.3f} | "
            f"{item['gt90_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Fresh paired strict direct-first task layer",
            "",
            f"All rows use the same strict controller and accumulated {ACTIVE_TIME_BUDGET_S:.0f} s active-time budget.",
            "",
            "| estimator | first-ray capture | strict CAD-bottom | hard stop | safe termination | budget violations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        item = task[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['first_ray_capture_count']}/{item['trials']} | "
            f"{item['strict_cad_bottom_count']}/{item['trials']} | "
            f"{item['hard_stop_count']}/{item['trials']} | "
            f"{item['safe_termination_count']}/{item['trials']} | "
            f"{item['budget_violation_count']} |"
        )
    lines.extend(["", "## Exact paired tests", ""])
    for endpoint, item in paired.items():
        lines.append(
            f"- `{endpoint}`: both={item['both_event']}, improved-only={item['improved_only_event']}, "
            f"Jin-proxy-only={item['jin_proxy_only_event']}, neither={item['neither_event']}, "
            f"exact McNemar p={item['mcnemar_exact_two_sided_p']:.6g}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    _atomic_text(output / "summary.md", "\n".join(lines))


def _write_figure(
    path: Path,
    direction_methods: dict[str, Any],
    task_methods: dict[str, Any],
    paired_bootstrap: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#167D9A", "#D8792F")
    labels = [METHOD_LABELS[method] for method in METHODS]
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)

    direction_names = ("<=7.5 deg", "<=15 deg", "12-sector acc.")
    direction_keys = (
        "angle_within_7p5_rate",
        "angle_within_15_rate",
        "common_12_sector_accuracy",
    )
    x = np.arange(len(direction_names), dtype=float)
    width = 0.34
    for index, method in enumerate(METHODS):
        values = [direction_methods[method][key] for key in direction_keys]
        bars = axes[0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=labels[index],
            color=colors[index],
        )
        axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in values], fontsize=8)
    axes[0].set_xticks(x, direction_names)
    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("P1v6 structured stress\nMIXED direction result")
    axes[0].grid(axis="y", alpha=0.25)
    med_p90 = "   ".join(
        f"{labels[index]}: {direction_methods[method]['angle_error_median_deg']:.1f}/"
        f"{direction_methods[method]['angle_error_p90_deg']:.1f} deg med/p90"
        for index, method in enumerate(METHODS)
    )
    axes[0].text(0.0, -0.22, med_p90, transform=axes[0].transAxes, fontsize=7.5)
    axes[0].text(
        0.0,
        -0.31,
        (
            f"paired median delta={paired_bootstrap['point_estimate_deg']:.2f} deg; "
            f"95% cluster CI [{paired_bootstrap['ci_percentile_low_deg']:.2f}, "
            f"{paired_bootstrap['ci_percentile_high_deg']:.2f}] (includes 0)\n"
            f"base_pose_id clusters={paired_bootstrap['cluster_count']}; "
            f"B={paired_bootstrap['resamples']}; seed={paired_bootstrap['seed']}"
        ),
        transform=axes[0].transAxes,
        fontsize=7.2,
    )

    operational_names = ("<=7.5 deg", "<=15 deg", ">90 deg")
    operational_keys = (
        "control_direction_within_7p5_rate",
        "control_direction_within_15_rate",
        "control_direction_gt90_rate",
    )
    x = np.arange(len(operational_names), dtype=float)
    for index, method in enumerate(METHODS):
        values = [task_methods[method][key] for key in operational_keys]
        bars = axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=labels[index],
            color=colors[index],
        )
        axes[1].bar_label(
            bars, labels=[f"{value:.2f}" for value in values], fontsize=8
        )
    axes[1].set_xticks(x, operational_names)
    axes[1].set_ylim(0.0, 1.12)
    axes[1].set_ylabel("Rate")
    axes[1].set_title(
        "Strict-48 executed-control distribution\nMIXED direction result"
    )
    axes[1].grid(axis="y", alpha=0.25)
    operational_med_p90 = "   ".join(
        f"{labels[index]}: "
        f"{task_methods[method]['control_direction_error_median_deg']:.1f}/"
        f"{task_methods[method]['control_direction_error_p90_deg']:.1f} deg med/p90"
        for index, method in enumerate(METHODS)
    )
    axes[1].text(
        0.0,
        -0.22,
        operational_med_p90,
        transform=axes[1].transAxes,
        fontsize=7.5,
    )
    axes[1].text(
        0.0,
        -0.31,
        "error of control_direction_deg actually executed by direct-first",
        transform=axes[1].transAxes,
        fontsize=7.2,
    )

    task_names = ("First-ray capture", "Strict bottom", "Safe termination", "Hard stop")
    task_keys = (
        "first_ray_capture_rate",
        "strict_cad_bottom_rate",
        "safe_termination_rate",
        "hard_stop_rate",
    )
    x = np.arange(len(task_names), dtype=float)
    for index, method in enumerate(METHODS):
        values = [task_methods[method][key] for key in task_keys]
        bars = axes[2].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=labels[index],
            color=colors[index],
        )
        axes[2].bar_label(
            bars, labels=[f"{value:.2f}" for value in values], fontsize=8
        )
    axes[2].set_xticks(x, task_names, rotation=12, ha="right")
    axes[2].set_ylim(0.0, 1.12)
    axes[2].set_ylabel("Rate")
    axes[2].set_title("Strict-48 direct-first task\nIMPROVED success")
    axes[2].grid(axis="y", alpha=0.25)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "M8 estimator-only comparison | Jin arm is a paper-inspired proxy",
        fontsize=13,
        weight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    figure.savefig(temporary, dpi=220, format="png", bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def _source_manifest(
    output: Path,
    direction_source: Path,
    improved_model_root: Path,
    jin_model_root: Path,
    direction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = {
        "preregistration": output / "preregistration.json",
        "direction_sampling_contract": direction_source / "sampling_contract.json",
        "direction_manifest": direction_source / "manifest.json",
        "improved_runtime_contract": improved_model_root
        / "ensemble_interface_contract.json",
        "jin_proxy_runtime_contract": jin_model_root / "runtime_contract.json",
        "jin_proxy_training_report": jin_model_root / "training_report.json",
        "comparison_audit_source": Path(__file__).resolve(),
        "strict_complete_loop_audit_source": (
            ROOT / "src/run_m8_strict_complete_loop_audit.py"
        ),
        "jin_proxy_runtime_source": ROOT / "src/m8_jin_style_direction.py",
        "improved_runtime_source": ROOT / "src/m8_direction_runtime.py",
        "improved_model_definition_source": (
            ROOT / "src/train_m8_direction_models.py"
        ),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing frozen source artifact: {path}")
        result[name] = {"path": _display_path(path), "sha256": _sha256(path)}

    def checkpoint_entries(contract_path: Path, expected: int) -> list[dict[str, str]]:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        values = list(contract.get("checkpoints", []))
        if len(values) != expected:
            raise ValueError(
                f"expected {expected} checkpoints in {contract_path}, found {len(values)}"
            )
        entries = []
        for value in values:
            checkpoint = Path(value)
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing frozen checkpoint: {checkpoint}")
            entries.append(
                {
                    "path": _display_path(checkpoint),
                    "sha256": _sha256(checkpoint),
                    **_observed_mtime(checkpoint),
                }
            )
        return entries

    result["jin_proxy_checkpoints"] = checkpoint_entries(
        jin_model_root / "runtime_contract.json", 3
    )
    result["improved_ensemble_checkpoints"] = checkpoint_entries(
        improved_model_root / "ensemble_interface_contract.json", 3
    )
    preregistration_path = output / "preregistration.json"
    preregistration_payload = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    result["preregistration"].update(
        {
            **_observed_mtime(preregistration_path),
            "declared_created_utc": preregistration_payload.get("created_utc"),
            "declared_created_utc_interpretation": (
                "date-level placeholder, retained verbatim; not an exact event timestamp"
            ),
        }
    )
    first_jin_checkpoint = min(
        result["jin_proxy_checkpoints"],
        key=lambda item: int(item["observed_filesystem_mtime_ns"]),
    )
    prereg_mtime_ns = int(
        result["preregistration"]["observed_filesystem_mtime_ns"]
    )
    first_checkpoint_mtime_ns = int(
        first_jin_checkpoint["observed_filesystem_mtime_ns"]
    )
    result["preregistration_timing_observation"] = {
        "observed_order": "preregistration_file_mtime_before_first_jin_checkpoint_mtime",
        "observed_gap_s": (
            first_checkpoint_mtime_ns - prereg_mtime_ns
        )
        / 1.0e9,
        "first_jin_checkpoint_path": first_jin_checkpoint["path"],
        "created_utc_is_date_level_placeholder": True,
        "boundary": (
            "Filesystem mtimes support observed ordering but are not cryptographic "
            "proof of authorship time."
        ),
    }
    result["improved_ensemble_runtime_auxiliaries"] = []
    for checkpoint_entry in result["improved_ensemble_checkpoints"]:
        checkpoint = Path(checkpoint_entry["path"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        seed_dir = checkpoint.parent
        auxiliary = {}
        for name, filename in (
            ("normalization", "normalization.npz"),
            ("calibration_metrics", "metrics.json"),
        ):
            path = seed_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"missing improved runtime auxiliary artifact: {path}"
                )
            auxiliary[name] = {
                "path": _display_path(path),
                "sha256": _sha256(path),
            }
        result["improved_ensemble_runtime_auxiliaries"].append(auxiliary)
    amendment_path = output / AMENDMENT_FILENAME
    if amendment_path.exists():
        result["post_completion_contract_amendment"] = {
            "path": _display_path(amendment_path),
            "sha256": _sha256(amendment_path),
        }
    result["shared_phase_sequence_set_sha256"] = _json_sha256(
        [
            (row["trial_id"], row["shared_phase_sequence_sha256"])
            for row in direction_rows
        ]
    )
    return result


def _write_manifest(
    output: Path,
    figure: Path,
    contract: dict[str, Any],
    direction_source: Path,
    improved_model_root: Path,
    jin_model_root: Path,
    direction_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    derived_rebuild: dict[str, Any] | None = None,
    amendment_reference: dict[str, Any] | None = None,
) -> None:
    artifact_paths = {
        "report": output / "report.json",
        "summary": output / "summary.md",
        "predictions": output / "predictions.csv",
        "paired_tasks": output / "paired_tasks.csv",
        "paper_figure": figure,
    }
    artifacts = {
        name: {"path": _display_path(path), "sha256": _sha256(path)}
        for name, path in artifact_paths.items()
    }
    trial_paths = sorted((output / "trials").glob("*.json"))
    payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "experiment": contract["experiment"],
            "completed_direction_trials": len(direction_rows),
            "completed_paired_task_poses": len(records),
            "task_contract_sha256": _sha256(output / "task_contract.json"),
            "paired_trial_record_set_sha256": _json_sha256(
                [(path.name, _sha256(path)) for path in trial_paths]
            ),
            "sources": _source_manifest(
                output,
                direction_source,
                improved_model_root,
                jin_model_root,
                direction_rows,
            ),
            "artifacts": artifacts,
            "claim_boundary": contract["claim_boundary"],
            "updated_utc": _utc_now(),
        }
    if derived_rebuild is not None:
        payload["derived_rebuild"] = derived_rebuild
    if amendment_reference is not None:
        payload["post_completion_contract_amendment"] = amendment_reference
    _atomic_json(output / "manifest.json", payload)


def _load_resume_record(path: Path, task: PoseTask) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("task") != asdict(task):
        raise ValueError(f"resume task mismatch: {task.pose_id}")
    methods = record.get("methods", {})
    if set(methods) != set(METHODS):
        raise ValueError(f"resume method mismatch: {task.pose_id}")
    expected_fairness = _validate_task_pair(
        task,
        methods["improved_fingerprint"],
        methods["jin_2021_style_proxy"],
    )
    if record.get("fairness") != expected_fairness:
        raise ValueError(f"resume fairness mismatch: {task.pose_id}")
    return record


def _paired_trial_record_set_sha256(output: Path) -> str:
    trial_paths = sorted((output / "trials").glob("*.json"))
    return _json_sha256([(path.name, _sha256(path)) for path in trial_paths])


def _verify_contract_amendment(
    output: Path,
    *,
    task_contract_sha256: str,
    trial_record_set_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output / AMENDMENT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"derived rebuild requires the post-completion amendment: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("provenance_status", {}).get("preregistered") is not False:
        raise ValueError("contract amendment must explicitly be non-preregistered")
    if not payload.get("provenance_status", {}).get(
        "post_completion_protocol_correction"
    ):
        raise ValueError("contract amendment lacks post-completion status")
    if payload.get("original_contract", {}).get("sha256") != task_contract_sha256:
        raise ValueError("contract amendment original task-contract SHA mismatch")
    frozen = payload.get("frozen_evidence", {})
    if (
        frozen.get("strict_paired_trial_record_set_sha256")
        != trial_record_set_sha256
    ):
        raise ValueError("contract amendment frozen trial-set SHA mismatch")
    if frozen.get("p1v6_scan_count") != EXPECTED_DIRECTION_TRIALS:
        raise ValueError("contract amendment P1v6 scan count mismatch")
    if frozen.get("strict_paired_trial_count") != TASK_COUNT:
        raise ValueError("contract amendment paired trial count mismatch")
    timing = payload.get("preregistration_timestamp_provenance", {})
    if timing.get("declared_created_utc") != "2026-07-15T00:00:00Z":
        raise ValueError("contract amendment does not retain placeholder created_utc")
    prereg = timing.get("preregistration_file", {})
    first_checkpoint = timing.get("first_jin_proxy_checkpoint", {})
    for label, item in (
        ("preregistration", prereg),
        ("first_jin_proxy_checkpoint", first_checkpoint),
    ):
        artifact = Path(str(item.get("path", "")))
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.exists() or _sha256(artifact) != item.get("sha256"):
            raise ValueError(f"contract amendment {label} SHA mismatch")
        if artifact.stat().st_mtime_ns != item.get("observed_filesystem_mtime_ns"):
            raise ValueError(f"contract amendment {label} observed mtime mismatch")
    if not (
        int(prereg["observed_filesystem_mtime_ns"])
        < int(first_checkpoint["observed_filesystem_mtime_ns"])
    ):
        raise ValueError("contract amendment observed preregistration order mismatch")
    reference = {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "amendment_id": payload.get("amendment_id"),
        "post_completion_protocol_correction": True,
        "preregistered": False,
    }
    return payload, reference


def _load_verified_derived_rebuild_records(
    output: Path,
    contract: dict[str, Any],
    tasks: list[PoseTask],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Read a completed frozen run without modifying or rerunning any trial."""

    if len(tasks) != TASK_COUNT:
        raise ValueError(
            f"--rebuild-derived requires the complete frozen {TASK_COUNT}-pose design"
        )
    contract_path = output / "task_contract.json"
    manifest_path = output / "manifest.json"
    if not contract_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "--rebuild-derived requires task_contract.json and a completed manifest"
        )
    actual_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if actual_contract != contract:
        raise ValueError("derived rebuild task contract mismatch")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete":
        raise ValueError("derived rebuild requires a completed manifest")
    if manifest.get("completed_paired_task_poses") != TASK_COUNT:
        raise ValueError("derived rebuild manifest is not a complete 48-pose run")
    if manifest.get("task_contract_sha256") != _sha256(contract_path):
        raise ValueError("derived rebuild task contract SHA mismatch")

    expected_names = {f"{task.pose_id}.json" for task in tasks}
    trial_paths = sorted((output / "trials").glob("*.json"))
    actual_names = {path.name for path in trial_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"derived rebuild requires exactly 48 frozen trials; missing={missing}, extra={extra}"
        )
    trial_set_sha = _paired_trial_record_set_sha256(output)
    if trial_set_sha != manifest.get("paired_trial_record_set_sha256"):
        raise ValueError("derived rebuild frozen trial record-set SHA mismatch")
    _verify_contract_amendment(
        output,
        task_contract_sha256=_sha256(contract_path),
        trial_record_set_sha256=trial_set_sha,
    )
    records = [
        _load_resume_record(output / "trials" / f"{task.pose_id}.json", task)
        for task in tasks
    ]
    records.sort(key=lambda record: record["task"]["pose_id"])
    return records, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def run(args: argparse.Namespace) -> int:
    direction_source = Path(args.direction_source).resolve()
    output = Path(args.output).resolve()
    figure = Path(args.figure).resolve()
    improved_model_root = Path(args.improved_model_root).resolve()
    jin_model_root = Path(args.jin_model_root).resolve()
    all_tasks = build_tasks(TASK_COUNT, TASK_SEED, wrench_noise=True)
    if args.poses > TASK_COUNT:
        raise ValueError(f"--poses cannot exceed the frozen {TASK_COUNT}-pose design")
    tasks = all_tasks[: args.poses]
    contract = _experiment_contract(
        direction_source=direction_source,
        improved_model_root=improved_model_root,
        jin_model_root=jin_model_root,
        tasks=tasks,
        expected_direction_trials=args.expected_direction_trials,
    )

    derived_rebuild_metadata: dict[str, Any] | None = None
    amendment_reference: dict[str, Any] | None = None
    if args.rebuild_derived:
        if args.poses != TASK_COUNT or args.expected_direction_trials != EXPECTED_DIRECTION_TRIALS:
            raise ValueError(
                "--rebuild-derived requires the frozen 48-pose/180-direction full design"
            )
        records, previous_manifest, previous_manifest_sha = (
            _load_verified_derived_rebuild_records(output, contract, tasks)
        )
        _, amendment_reference = _verify_contract_amendment(
            output,
            task_contract_sha256=_sha256(output / "task_contract.json"),
            trial_record_set_sha256=previous_manifest[
                "paired_trial_record_set_sha256"
            ],
        )
        previous_rebuild = previous_manifest.get("derived_rebuild", {})
        derived_rebuild_metadata = {
            "performed": True,
            "rebuild_count": int(previous_rebuild.get("rebuild_count", 0)) + 1,
            "previous_manifest_sha256": previous_manifest_sha,
            "frozen_trial_record_set_sha256": previous_manifest[
                "paired_trial_record_set_sha256"
            ],
            "frozen_trials_read": len(records),
            "simulation_rerun": False,
            "trial_files_overwritten": 0,
            "contract_amendment_sha256": amendment_reference["sha256"],
            "reason": (
                "Corrected the derived strict-48 operational direction summary to "
                "use the actually executed RunResult.control_direction_deg rather "
                "than the ensemble-top-class RunResult.direction_error_deg; rebuilt "
                "CSVs/report/summary/figure/manifest from unchanged frozen evidence."
            ),
            "rebuilt_utc": _utc_now(),
        }
    else:
        if _completed_manifest_is_valid(output, contract):
            print(f"completed immutable experiment verified: {output}")
            return 0
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(
                f"refusing to overwrite non-empty output without --resume: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "trials").mkdir(parents=True, exist_ok=True)
        _ensure_contract(output, contract)

        records = []
        pending: list[PoseTask] = []
        for task in tasks:
            path = output / "trials" / f"{task.pose_id}.json"
            if args.resume and path.exists():
                records.append(_load_resume_record(path, task))
            else:
                pending.append(task)

        started = time.perf_counter()

        def checkpoint() -> None:
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            _atomic_json(
                output / "progress.json",
                {
                    "total_paired_poses": len(tasks),
                    "completed_paired_poses": len(records),
                    "pending_paired_poses": len(tasks) - len(records),
                    "elapsed_s_this_invocation": elapsed,
                    "updated_utc": _utc_now(),
                },
            )

        checkpoint()
        errors: list[dict[str, Any]] = []
        if pending:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(pending)),
                mp_context=context,
                initializer=_worker_init,
                initargs=(str(improved_model_root), str(jin_model_root)),
            ) as pool:
                futures = {pool.submit(_run_task_pair, task): task for task in pending}
                with tqdm(
                    total=len(tasks),
                    initial=len(records),
                    desc="paired strict direct-first",
                    unit="pose",
                    dynamic_ncols=True,
                ) as bar:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            record = future.result()
                        except Exception as exc:  # retain successful pairs for resume
                            errors.append(
                                {
                                    "pose_id": task.pose_id,
                                    "error": repr(exc),
                                    "updated_utc": _utc_now(),
                                }
                            )
                        else:
                            _atomic_json(
                                output / "trials" / f"{task.pose_id}.json", record
                            )
                            records.append(record)
                        checkpoint()
                        bar.update(1)

        if errors:
            _atomic_json(output / "errors.json", {"errors": errors})
            print(json.dumps(errors, indent=2))
            return 1
        errors_path = output / "errors.json"
        if errors_path.exists():
            errors_path.unlink()

    direction_rows, direction_metrics = _evaluate_direction_layer(
        direction_source,
        improved_model_root,
        jin_model_root,
        args.expected_direction_trials,
    )

    records.sort(key=lambda record: record["task"]["pose_id"])
    if len(records) != len(tasks):
        raise RuntimeError(
            f"paired task completion mismatch: {len(records)} != {len(tasks)}"
        )
    task_methods = {
        method: _method_task_summary(records, method) for method in METHODS
    }
    available_hashes = [
        record["fairness"]["fingerprint_sha256_match"]
        for record in records
        if record["fairness"]["fingerprint_sha256_match"] is not None
    ]
    task_report = {
        "poses": len(records),
        "seed": TASK_SEED,
        "wrench_noise": True,
        "direction_metric_schema_version": 2,
        "operational_direction_scope": (
            "Fresh strict-48 executed-control distribution: circular absolute error "
            "of frozen RunResult.control_direction_deg against "
            "RunResult.true_direction_deg on the same noisy paired direct-first poses. "
            "This is the direction actually used by the controller and is distinct "
            "from the structured P1v6 direction stress set."
        ),
        "top_class_direction_diagnostic_scope": (
            "Frozen RunResult.direction_error_deg measures predicted_direction_deg "
            "(ensemble top class / top_direction_deg) against true_direction_deg. It "
            "is retained inside each method only as a non-operational diagnostic."
        ),
        "post_completion_derived_metric_correction": {
            "status": "non-preregistered derived-only semantic correction",
            "previous_incorrect_operational_source": "RunResult.direction_error_deg",
            "correct_operational_source": "RunResult.control_direction_deg",
            "truth_source": "RunResult.true_direction_deg",
            "simulator_rerun": False,
            "frozen_trial_files_modified": 0,
        },
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "same_strict_controller_for_both": True,
        "only_direction_estimator_changes": True,
        "fingerprint_available_pair_count": len(available_hashes),
        "fingerprint_unavailable_pair_count": len(records) - len(available_hashes),
        "fingerprint_sha_mismatch_count": sum(not value for value in available_hashes),
        "methods": task_methods,
        "paired": _paired_task_summary(records),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": contract["experiment"],
        "direction_layer": {
            "source": _display_path(direction_source),
            "evaluation_role": (
                "Fresh structured P1v6 radial/sector direction stress; separate "
                "from the strict-48 operational task distribution."
            ),
            "result_characterization": (
                "mixed: improved has lower median and better within-15 rate, while "
                "the proxy has better p90, within-7.5, >90 tail, and common "
                "12-sector accuracy; the paired base-pose cluster-bootstrap "
                "interval for the median absolute-error difference includes zero"
            ),
            "methods": {
                "improved_fingerprint": direction_metrics["improved_fingerprint"],
                "jin_2021_style_proxy": direction_metrics[
                    "jin_2021_style_proxy"
                ],
            },
            "shared_raw_scan_for_both_estimators": direction_metrics[
                "shared_raw_scan_for_both_estimators"
            ],
            "paired_improved_minus_jin_absolute_error_median_deg": direction_metrics[
                "paired_improved_minus_jin_absolute_error_median_deg"
            ],
            "paired_base_pose_cluster_bootstrap": direction_metrics[
                "paired_base_pose_cluster_bootstrap"
            ],
        },
        "task_layer": task_report,
        "interpretation_rule": (
            "Report the measured paired differences and exact McNemar p-values. "
            "For strict-48 direction, use the executed control_direction_deg rather "
            "than the ensemble top class. Do not infer superiority merely from a "
            "higher point estimate."
        ),
        "claim_boundary": contract["claim_boundary"],
        "updated_utc": _utc_now(),
    }
    if derived_rebuild_metadata is not None:
        report["derived_rebuild"] = derived_rebuild_metadata
    if amendment_reference is not None:
        report["post_completion_contract_amendment"] = amendment_reference

    staging: Path | None = None
    artifact_output = output
    artifact_figure = figure
    if derived_rebuild_metadata is not None:
        staging = Path(tempfile.mkdtemp(prefix=".derived_rebuild_", dir=output))
        artifact_output = staging
        artifact_figure = staging / "paper_figure.png"
    try:
        _atomic_csv(artifact_output / "predictions.csv", direction_rows)
        _atomic_csv(artifact_output / "paired_tasks.csv", _task_pair_rows(records))
        _atomic_json(artifact_output / "report.json", report)
        _write_summary(artifact_output, report)
        _write_figure(
            artifact_figure,
            report["direction_layer"]["methods"],
            task_methods,
            report["direction_layer"]["paired_base_pose_cluster_bootstrap"],
        )
        if staging is not None:
            for name in ("predictions.csv", "paired_tasks.csv", "report.json", "summary.md"):
                (staging / name).replace(output / name)
            figure.parent.mkdir(parents=True, exist_ok=True)
            artifact_figure.replace(figure)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    _write_manifest(
        output,
        figure,
        contract,
        direction_source,
        improved_model_root,
        jin_model_root,
        direction_rows,
        records,
        derived_rebuild=derived_rebuild_metadata,
        amendment_reference=amendment_reference,
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh paired M8 audit of the frozen Jin-2021-style direction proxy "
            "against the improved fingerprint estimator."
        )
    )
    parser.add_argument("--direction-source", default=str(DEFAULT_DIRECTION_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument(
        "--improved-model-root", default=str(DEFAULT_IMPROVED_MODEL_ROOT)
    )
    parser.add_argument("--jin-model-root", default=str(DEFAULT_JIN_MODEL_ROOT))
    parser.add_argument("--poses", type=int, default=TASK_COUNT)
    parser.add_argument(
        "--expected-direction-trials",
        type=int,
        default=EXPECTED_DIRECTION_TRIALS,
    )
    parser.add_argument(
        "--workers", type=int, default=min(24, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help=(
            "Rebuild report/CSVs/figure/manifest only after verifying the frozen "
            "task contract and all 48 paired trial hashes; never reruns or writes trials."
        ),
    )
    args = parser.parse_args(argv)
    if args.poses <= 0 or args.expected_direction_trials <= 0 or args.workers <= 0:
        parser.error("--poses, --expected-direction-trials, and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
