#!/usr/bin/env python3
"""Replay the frozen simulation CNN on the three complete real campaigns.

This is an offline diagnostic only.  It imports no ROS, KST, or Bota client,
opens no robot socket, and cannot produce a motion command.  The explicit run
allowlist must match the system-Python QC report before inference starts.

The raw zero-shot result tests the simulation-to-real feature contract.  Two
additional fixed-frame calibrations (global and per-size sign/offset) are fit
on ``clean_candidate`` only and scored on the already-designated holdout.  They
diagnose whether a simple mirror/rotation convention explains the gap; they
are not deployable model training.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REAL_ROOT = HERE.parent
OUTPUT_ROOT = (
    REAL_ROOT / "offline_analysis" / "multisize_real_fingerprint_v1"
)
QC_REPORT = OUTPUT_ROOT / "report.json"
INFERENCE_MODULE_PATH = HERE / "infer_conical_direction.py"
DEPLOYMENT_MANIFEST = HERE / "deployment_manifest.json"

# Deliberately explicit: the wrong M8-program/M12-physical run is absent.
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
COMPLETE_STATUS = "COMPLETE_RETURNED_TO_INITIAL_AIR_POSE_KST_END_SENT"
STATE_TO_SIM = {
    "supported_search": "supported_search",
    "double_overhang": "double_overhang",
    "outer_partial_support": "outer_partial",
}


def load_inference_module():
    spec = importlib.util.spec_from_file_location(
        "offline_frozen_conical_inference", INFERENCE_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(INFERENCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INFER = load_inference_module()
MODEL_ROOT = INFER.DEFAULT_MODEL_ROOT.resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(
    path: Path, fieldnames: list[str], records: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def circular_error_deg(
    predicted_deg: np.ndarray | float, true_deg: np.ndarray | float
) -> np.ndarray:
    predicted = np.asarray(predicted_deg, dtype=float)
    truth = np.asarray(true_deg, dtype=float)
    return np.abs((predicted - truth + 180.0) % 360.0 - 180.0)


def angle_from_xy(vector: np.ndarray) -> float:
    xy = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(xy))
    if xy.shape != (2,) or not np.all(np.isfinite(xy)) or norm <= 1.0e-12:
        raise ValueError("direction vector is not a finite nonzero XY vector")
    return float(np.mod(np.rad2deg(np.arctan2(xy[1], xy[0])), 360.0))


def direction_metrics(errors: np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("direction metrics require a nonempty 1-D array")
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


def score_records(
    records: list[dict[str, Any]], prediction_key: str
) -> dict[str, Any]:
    errors = circular_error_deg(
        np.asarray([record[prediction_key] for record in records]),
        np.asarray([record["true_direction_world_deg"] for record in records]),
    )
    return direction_metrics(errors)


def verify_qc_authority() -> dict[str, Any]:
    if not QC_REPORT.is_file():
        raise FileNotFoundError(
            f"run the system-Python QC first; absent: {QC_REPORT}"
        )
    report = read_json(QC_REPORT)
    if report["quality_control"]["status"] != "PASS":
        raise ValueError("multisize QC report is not PASS")
    expected = {
        size: str(path.resolve()) for size, path in RUN_ALLOWLIST.items()
    }
    if report["run_allowlist"] != expected:
        raise ValueError("zero-shot run allowlist differs from the QC authority")
    return report


def load_attempts() -> tuple[
    list[tuple[str, dict[str, Any], float]], dict[str, float]
]:
    output: list[tuple[str, dict[str, Any], float]] = []
    target_by_size: dict[str, float] = {}
    for size, run_dir in RUN_ALLOWLIST.items():
        campaign = read_json(run_dir / "sampling_campaign.json")
        if (
            campaign["size"] != size
            or campaign["status"] != COMPLETE_STATUS
            or int(campaign["resume_generation"]) != 0
            or bool(campaign["resume_enabled"])
            or len(campaign["attempts"]) != 26
        ):
            raise ValueError(f"{run_dir.name} is not the locked fresh campaign")
        target_down_force_n = float(
            campaign["contact_observation_policy"][
                "preferred_target_normal_force_n"
            ]
        )
        target_by_size[size] = target_down_force_n
        for attempt in campaign["attempts"]:
            trace = Path(str(attempt["cone_csv_path"])).resolve()
            if not trace.is_file():
                raise FileNotFoundError(trace)
            expected_hash = attempt["file_hashes"]["cone_csv_sha256"]
            if INFER.sha256_file(trace) != expected_hash:
                raise ValueError(f"{attempt['sample_id']} trace hash mismatch")
            output.append((size, attempt, target_down_force_n))
    if len(output) != 78:
        raise ValueError(f"expected 78 attempts, got {len(output)}")
    return output, target_by_size


def infer_attempt(
    size: str,
    attempt: dict[str, Any],
    bit_radius_m: float,
    target_down_force_n: float,
) -> dict[str, Any]:
    trace_path = Path(str(attempt["cone_csv_path"])).resolve()
    input_value = INFER.trace_csv_to_input(trace_path)
    phase, columns = INFER.validate_input(input_value)
    sequence = INFER.phase_resample(phase, columns, bins=360)
    estimate = INFER.run_ensemble(
        MODEL_ROOT,
        sequence,
        target_down_force_n=target_down_force_n,
        bit_radius_m=bit_radius_m,
    )
    world_from_task_xy = np.asarray(
        input_value["world_from_task_xy"], dtype=float
    )
    mean_world = world_from_task_xy @ np.asarray(
        estimate["direction_task_xy"], dtype=float
    )
    mean_world_deg = angle_from_xy(mean_world)
    top_task_rad = math.radians(float(estimate["top_direction_deg"]))
    top_world = world_from_task_xy @ np.asarray(
        [math.cos(top_task_rad), math.sin(top_task_rad)]
    )
    top_world_deg = angle_from_xy(top_world)
    true_world_deg = angle_from_xy(
        np.asarray(attempt["true_direction_world_xy"], dtype=float)
    )
    real_state = str(attempt["contact_state"])
    # Planned boundary-quarantine states have no class in the frozen
    # three-state simulation head.  Preserve their names in the CSV; they are
    # excluded from every state/direction headline metric below.
    true_state = STATE_TO_SIM.get(real_state, f"UNMAPPED_{real_state}")
    return {
        "size": size,
        "sample_id": attempt["sample_id"],
        "partition": attempt["effective_partition"],
        "contact_state": attempt["contact_state"],
        "true_state_sim_name": true_state,
        "predicted_state_sim_name": estimate["top_state"],
        "state_correct": int(estimate["top_state"] == true_state),
        "true_direction_world_deg": true_world_deg,
        "raw_runtime_mean_world_deg": mean_world_deg,
        "raw_runtime_mean_error_deg": float(
            circular_error_deg(mean_world_deg, true_world_deg)
        ),
        "raw_top_class_world_deg": top_world_deg,
        "raw_top_class_error_deg": float(
            circular_error_deg(top_world_deg, true_world_deg)
        ),
        "max_probability": estimate["max_probability"],
        "circular_concentration": estimate["circular_concentration"],
        "aps_set_size": estimate["aps_set_size"],
        "ensemble_top1_span_deg": estimate["ensemble_top1_span_deg"],
        "simulation_transport_gate_passed": int(
            estimate["simulation_transport_gate_passed"]
        ),
        "target_down_force_n": target_down_force_n,
    }


def fit_fixed_frame_transform(
    clean_records: list[dict[str, Any]], prediction_key: str
) -> dict[str, Any]:
    predicted = np.asarray(
        [record[prediction_key] for record in clean_records], dtype=float
    )
    truth = np.asarray(
        [record["true_direction_world_deg"] for record in clean_records],
        dtype=float,
    )
    best: dict[str, Any] | None = None
    for sign in (1, -1):
        for offset in np.arange(0.0, 360.0, 0.25):
            transformed = np.mod(sign * predicted + offset, 360.0)
            errors = circular_error_deg(transformed, truth)
            objective = float(
                np.median(errors) + 0.20 * np.quantile(errors, 0.90)
            )
            candidate = {
                "sign": int(sign),
                "offset_deg": float(offset),
                "clean_selection_objective": objective,
                "clean_metrics": direction_metrics(errors),
            }
            if best is None or objective < best["clean_selection_objective"]:
                best = candidate
    if best is None:
        raise RuntimeError("fixed-frame transform search produced no candidate")
    return best


def apply_transform(
    value: float, transform: dict[str, Any]
) -> float:
    return float(
        np.mod(
            int(transform["sign"]) * float(value)
            + float(transform["offset_deg"]),
            360.0,
        )
    )


def evaluation_signature(
    attempts: list[tuple[str, dict[str, Any], float]],
    target_by_size: dict[str, float],
    contract_path: Path,
) -> dict[str, str]:
    input_contract = []
    for size, attempt, target in sorted(
        attempts, key=lambda value: value[1]["sample_id"]
    ):
        input_contract.append(
            {
                "size": size,
                "sample_id": attempt["sample_id"],
                "partition": attempt["effective_partition"],
                "contact_state": attempt["contact_state"],
                "true_direction_world_xy": attempt[
                    "true_direction_world_xy"
                ],
                "cone_csv_sha256": attempt["file_hashes"][
                    "cone_csv_sha256"
                ],
                "target_down_force_n": target,
            }
        )
    # Bind the complete ordered input contract, not merely the first trace.
    input_signature = hashlib.sha256(
        json.dumps(
            input_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    components = {
        "evaluation_source_sha256": INFER.sha256_file(Path(__file__).resolve()),
        "inference_runtime_sha256": INFER.sha256_file(INFERENCE_MODULE_PATH),
        "deployment_manifest_sha256": INFER.sha256_file(DEPLOYMENT_MANIFEST),
        "model_contract_sha256": INFER.sha256_file(contract_path),
        "real_input_contract_sha256": input_signature,
        "campaign_target_force_contract_sha256": hashlib.sha256(
            json.dumps(
                target_by_size, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    components["evaluation_signature_sha256"] = hashlib.sha256(
        json.dumps(
            components, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return components


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline frozen-CNN replay on the three locked real runs."
    )
    parser.add_argument(
        "--allow-development-holdout-rereveal",
        action="store_true",
        help=(
            "explicitly rerun the consumed v1 holdout after evaluation code, "
            "inputs, model, or force contract changed"
        ),
    )
    args = parser.parse_args()
    qc_report = verify_qc_authority()
    manifest = INFER.load_and_verify_manifest(
        DEPLOYMENT_MANIFEST.resolve(), MODEL_ROOT
    )
    contract_path = MODEL_ROOT / "ensemble_interface_contract.json"
    contract = read_json(contract_path)
    bit_radii = {
        key.upper(): float(value)
        for key, value in contract["runtime_input_contract"][
            "size_bit_radius_m"
        ].items()
    }
    if set(bit_radii) != set(RUN_ALLOWLIST):
        raise ValueError("frozen model size-radius contract is incomplete")

    attempts, target_by_size = load_attempts()
    if set(target_by_size.values()) != {3.5}:
        raise ValueError(
            f"expected the three real campaigns to share 3.5 N, got "
            f"{target_by_size}"
        )
    signature = evaluation_signature(
        attempts, target_by_size, contract_path
    )
    output_json = OUTPUT_ROOT / "simulation_zero_shot_report.json"
    existing_report = read_json(output_json) if output_json.is_file() else None
    signature_matches = bool(
        existing_report
        and existing_report.get("evaluation_signature_sha256")
        == signature["evaluation_signature_sha256"]
    )
    if (
        existing_report is not None
        and not signature_matches
        and not args.allow_development_holdout_rereveal
    ):
        raise RuntimeError(
            "the v1 holdout was already replayed with a different zero-shot "
            "signature; pass --allow-development-holdout-rereveal only for "
            "an explicitly documented development sensitivity update"
        )

    records = [
        infer_attempt(size, attempt, bit_radii[size], target)
        for size, attempt, target in attempts
    ]
    runtime_default_records = [
        infer_attempt(size, attempt, bit_radii[size], 3.0)
        for size, attempt, _target in attempts
    ]
    clean = [
        record for record in records
        if record["partition"] == "clean_candidate"
    ]
    holdout = [
        record for record in records if record["partition"] == "holdout"
    ]
    nonboundary = [
        record for record in records if record["partition"] != "quarantine"
    ]
    runtime_default_nonboundary = [
        record
        for record in runtime_default_records
        if record["partition"] != "quarantine"
    ]
    runtime_default_holdout = [
        record
        for record in runtime_default_records
        if record["partition"] == "holdout"
    ]
    if len(clean) != 48 or len(holdout) != 12 or len(nonboundary) != 66:
        raise ValueError("partition count contract failed")

    global_transform = fit_fixed_frame_transform(
        clean, "raw_runtime_mean_world_deg"
    )
    size_transforms = {
        size: fit_fixed_frame_transform(
            [record for record in clean if record["size"] == size],
            "raw_runtime_mean_world_deg",
        )
        for size in RUN_ALLOWLIST
    }
    for record in records:
        record["global_clean_frame_calibrated_deg"] = apply_transform(
            record["raw_runtime_mean_world_deg"], global_transform
        )
        record["per_size_clean_frame_calibrated_deg"] = apply_transform(
            record["raw_runtime_mean_world_deg"],
            size_transforms[record["size"]],
        )
        record["global_clean_frame_calibrated_error_deg"] = float(
            circular_error_deg(
                record["global_clean_frame_calibrated_deg"],
                record["true_direction_world_deg"],
            )
        )
        record["per_size_clean_frame_calibrated_error_deg"] = float(
            circular_error_deg(
                record["per_size_clean_frame_calibrated_deg"],
                record["true_direction_world_deg"],
            )
        )

    by_size = {}
    for size in RUN_ALLOWLIST:
        selected = [
            record for record in nonboundary if record["size"] == size
        ]
        by_size[size] = {
            "raw_runtime_mean": score_records(
                selected, "raw_runtime_mean_world_deg"
            ),
            "state_accuracy": float(
                np.mean([record["state_correct"] for record in selected])
            ),
            "simulation_transport_gate_pass_fraction": float(
                np.mean(
                    [
                        record["simulation_transport_gate_passed"]
                        for record in selected
                    ]
                )
            ),
        }

    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "role": (
            "offline frozen simulation-to-real zero-shot diagnostic; no robot "
            "connection and no hardware authority"
        ),
        "robot_motion": False,
        "hardware_control_allowed": False,
        "model_id": MODEL_ROOT.name,
        "model_root": str(MODEL_ROOT),
        "model_contract_sha256": INFER.sha256_file(contract_path),
        "deployment_manifest_sha256": INFER.sha256_file(
            DEPLOYMENT_MANIFEST
        ),
        "deployment_audit_status": manifest["audit_decision"]["global_status"],
        "qc_authority_status_at_replay": qc_report["quality_control"]["status"],
        "development_rereveal_override_used": bool(
            existing_report is not None and not signature_matches
        ),
        "previous_zero_shot_report_sha256": (
            INFER.sha256_file(output_json)
            if existing_report is not None and not signature_matches
            else None
        ),
        **signature,
        "run_allowlist": {
            size: str(path.resolve()) for size, path in RUN_ALLOWLIST.items()
        },
        "preprocessing": {
            "implementation": str(INFERENCE_MODULE_PATH),
            "phase_bins": 360,
            "channels": ["Fz", "Tx", "Ty", "tip_z"],
            "primary_target_down_force_n_by_size": target_by_size,
            "sensitivity_target_down_force_n": 3.0,
            "bit_radius_source": "frozen ensemble interface contract",
            "frame_adapter": "existing provisional real-trace runtime adapter",
            "tip_z_boundary": (
                "actual_z is command-propagated when feedback_observed=0; this "
                "is not in-loop measured KST tip position"
            ),
        },
        "counts": {
            "all": len(records),
            "clean": len(clean),
            "holdout": len(holdout),
            "nonboundary_clean_holdout_sentinel": len(nonboundary),
        },
        "metrics": {
            "nonboundary_raw_runtime_mean": score_records(
                nonboundary, "raw_runtime_mean_world_deg"
            ),
            "holdout_raw_runtime_mean": score_records(
                holdout, "raw_runtime_mean_world_deg"
            ),
            "holdout_raw_top_class": score_records(
                holdout, "raw_top_class_world_deg"
            ),
            "holdout_after_global_clean_sign_offset": score_records(
                holdout, "global_clean_frame_calibrated_deg"
            ),
            "holdout_after_per_size_clean_sign_offset": score_records(
                holdout, "per_size_clean_frame_calibrated_deg"
            ),
            "target_force_sensitivity_3p0_n": {
                "nonboundary_raw_runtime_mean": score_records(
                    runtime_default_nonboundary,
                    "raw_runtime_mean_world_deg",
                ),
                "holdout_raw_runtime_mean": score_records(
                    runtime_default_holdout,
                    "raw_runtime_mean_world_deg",
                ),
                "interpretation": (
                    "3.0 N is the old runtime default; 3.5 N is the real "
                    "campaign contract and the primary result"
                ),
            },
            "by_size_nonboundary": by_size,
            "nonboundary_state_accuracy": float(
                np.mean([record["state_correct"] for record in nonboundary])
            ),
        },
        "clean_only_frame_diagnostic": {
            "global_transform": global_transform,
            "per_size_transforms": size_transforms,
            "selection_partition": "clean_candidate_only",
            "interpretation": (
                "if holdout remains poor, a single coordinate sign/offset is "
                "insufficient to explain simulation-to-real mismatch"
            ),
        },
        "verdict": (
            "the frozen simulation CNN does not transfer zero-shot to these "
            "real Bota fingerprints; use it only as a frozen teacher or "
            "initialization for a small real-domain adapter"
        ),
        "claim_boundary": (
            "These holdout points have now been revealed for v1 development. "
            "No result here is a final locked audit or motion authorization."
        ),
        "qc_authority_analysis_id": qc_report["analysis_id"],
    }
    output_csv = OUTPUT_ROOT / "simulation_zero_shot_predictions.csv"
    INFER.write_json(output_json, report)
    write_csv(output_csv, list(records[0]), records)
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(records),
                "nonboundary_median_deg": report["metrics"][
                    "nonboundary_raw_runtime_mean"
                ]["median_error_deg"],
                "holdout_median_deg": report["metrics"][
                    "holdout_raw_runtime_mean"
                ]["median_error_deg"],
                "holdout_per_size_frame_median_deg": report["metrics"][
                    "holdout_after_per_size_clean_sign_offset"
                ]["median_error_deg"],
                "state_accuracy": report["metrics"][
                    "nonboundary_state_accuracy"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
