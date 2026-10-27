#!/usr/bin/env python3
"""Build the immutable real-first M8/M10/M12 offline-learning dataset.

Only the three explicitly allowlisted, fresh, completed campaigns are read.
This program has no ROS/KST/Bota imports and cannot command the robot.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from common import (
    ANALYSIS_ROOT,
    CONTEXT_ORDER,
    C6_TRANSFORM_ID,
    DATASET_ROOT,
    INPUT_CHANNELS,
    PHASE_BINS,
    REAL_ROOT,
    REAL_STATE_TO_CANONICAL,
    SIZE_BIT_RADIUS_M,
    SIZE_TO_INDEX,
    STATE_TO_INDEX,
    TARGET_FORCE_REFERENCE_N,
    direction_class,
    sha256_file,
    write_json,
)
import common as common_module


ANALYZER_PATH = (
    REAL_ROOT / "tools" / "analyze_multisize_real_fingerprints.py"
)
RUN_ALLOWLIST = {
    "M8": REAL_ROOT
    / "data"
    / "M8_conical_fingerprints_26pt_20260728_161249_458",
    "M10": REAL_ROOT
    / "data"
    / "M10_conical_fingerprints_26pt_20260728_152808_877",
    "M12": REAL_ROOT
    / "data"
    / "M12_conical_fingerprints_26pt_20260728_175502_318",
}
REJECTED_WRONG_PROGRAM_RUN = (
    REAL_ROOT
    / "data"
    / "M8_conical_fingerprints_26pt_20260728_172108_490"
)
CAD_PARAMETERS_PATH = (
    REAL_ROOT / "model" / "parametric_tooling" / "design_parameters.json"
)
CAD_MANIFEST_PATH = (
    REAL_ROOT
    / "model"
    / "parametric_tooling"
    / "generated"
    / "manifest.json"
)
PARTITION_ORDER = (
    "clean_candidate",
    "holdout",
    "quarantine",
    "sentinel",
)


def _require_number(
    mapping: dict[str, Any], key: str, expected: float
) -> None:
    value = float(mapping[key])
    if not np.isclose(value, expected, atol=1.0e-9):
        raise ValueError(f"clocking contract {key}={value} != {expected}")


def _validate_cad_clocking(
    analyzer: Any,
    campaign: dict[str, Any],
) -> dict[str, Any]:
    """Reject stale pre-v1.5 CAD or a second application of its -15 deg yaw."""

    design = analyzer.read_json(CAD_PARAMETERS_PATH)
    manifest = analyzer.read_json(CAD_MANIFEST_PATH)
    if design["design_revision"] != "v1.5-integrated-flange-tools-yaw-datum":
        raise ValueError("offline learning requires the printed v1.5 CAD")
    if manifest["design_revision"] != design["design_revision"]:
        raise ValueError("generated CAD manifest revision does not match source")
    coordinate = design["coordinate_system"]
    _require_number(
        coordinate, "integrated_peg_selected_hex_flat_normal_deg", 45.0
    )
    _require_number(coordinate, "flange_reference_tangent_normal_deg", 45.0)
    _require_number(coordinate, "socket_selected_hex_flat_normal_deg", 0.0)
    _require_number(design["peg"], "flange_relative_yaw_deg", -15.0)

    plan_path = Path(campaign["sampling_plan_path"])
    plan = analyzer.read_json(plan_path)
    yaw = plan["coordinate_contract"]["yaw_contract"]
    _require_number(
        yaw, "printed_peg_selected_flat_normal_in_flange_cad_deg", 45.0
    )
    _require_number(
        yaw, "commissioned_peg_flat_normal_in_fixture_deg", 0.0
    )
    _require_number(yaw, "socket_flat_normal_in_fixture_deg", 0.0)
    if (
        plan["cone_command_geometry"]["rotation_convention"]
        != "R(phi)=Rz(-phi)*Ry(15deg)*Rz(phi)"
    ):
        raise ValueError("sampling cone phase convention is not canonical")
    printed = campaign["printed_cad_geometry_mm"]
    _require_number(printed, "peg_tip_face_flat_normal_fixture_deg", 0.0)
    _require_number(
        printed, "peg_selected_flat_normal_in_flange_cad_deg", 45.0
    )
    _require_number(printed, "socket_selected_flat_normal_fixture_deg", 0.0)
    return {
        "design_revision": design["design_revision"],
        "shape_rotation_from_legacy_deg": -15.0,
        "printed_peg_flat_normal_in_flange_cad_deg": 45.0,
        "operator_taught_peg_flat_normal_in_fixture_deg": 0.0,
        "socket_flat_normal_in_fixture_deg": 0.0,
        "cone_rotation_convention": (
            "R(phi)=Rz(-phi)*Ry(15deg)*Rz(phi)"
        ),
        "runtime_rule": (
            "the CAD -15 degree shape correction is absorbed by the taught "
            "flange datum and must not be applied a second time"
        ),
        "measurement_status": (
            "OPERATOR_TAUGHT_FIXTURE_ALIGNMENT_NOT_AS_BUILT_YAW_METROLOGY"
        ),
        "design_parameters_path": str(CAD_PARAMETERS_PATH),
        "design_parameters_sha256": sha256_file(CAD_PARAMETERS_PATH),
        "generated_manifest_path": str(CAD_MANIFEST_PATH),
        "generated_manifest_sha256": sha256_file(CAD_MANIFEST_PATH),
    }


def _load_analyzer():
    specification = importlib.util.spec_from_file_location(
        "multisize_real_fingerprint_analyzer", ANALYZER_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import {ANALYZER_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _save_npz(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        sequence = np.stack([record["sequence"] for record in records])
        context = np.stack([record["context"] for record in records])
    else:
        sequence = np.zeros(
            (0, len(INPUT_CHANNELS), PHASE_BINS), dtype=np.float32
        )
        context = np.zeros((0, len(CONTEXT_ORDER)), dtype=np.float32)
    np.savez_compressed(
        path,
        sequence=sequence.astype(np.float32),
        context=context.astype(np.float32),
        state_index=np.asarray(
            [record["state_index"] for record in records], dtype=np.int64
        ),
        direction_deg=np.asarray(
            [record["direction_deg"] for record in records], dtype=np.float32
        ),
        direction_class=np.asarray(
            [record["direction_class"] for record in records], dtype=np.int64
        ),
        size_index=np.asarray(
            [record["size_index"] for record in records], dtype=np.int64
        ),
        sample_id=np.asarray(
            [record["sample_id"] for record in records], dtype="U96"
        ),
        partition=np.asarray(
            [record["partition"] for record in records], dtype="U24"
        ),
        origin_group=np.asarray(
            [record["origin_group"] for record in records], dtype="U128"
        ),
    )


def _write_index(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "sample_id",
        "size",
        "size_index",
        "partition",
        "contact_state",
        "state_index",
        "direction_deg",
        "direction_class",
        "origin_group",
        "run_session_id",
        "campaign_sha256",
        "attempt_json_sha256",
        "cone_csv_sha256",
        "training_role",
    )
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    temporary.replace(path)


def build_dataset(output: Path = DATASET_ROOT) -> dict[str, Any]:
    analyzer = _load_analyzer()
    records: list[dict[str, Any]] = []
    campaign_quality: dict[str, Any] = {}
    campaign_sources: dict[str, Any] = {}
    cad_clocking_contract: dict[str, Any] | None = None

    rejected_flags = [
        flag
        for flag in ("DO_NOT_TRAIN.flag", "RUN_QUARANTINE.json")
        if (REJECTED_WRONG_PROGRAM_RUN / flag).is_file()
    ]
    if len(rejected_flags) != 2:
        raise ValueError(
            "the known M8-on-M12 wrong-program lineage is not fully quarantined"
        )

    for size, run_dir in RUN_ALLOWLIST.items():
        samples, quality = analyzer.validate_campaign(size, run_dir)
        campaign = analyzer.read_json(run_dir / "sampling_campaign.json")
        current_clocking = _validate_cad_clocking(analyzer, campaign)
        if cad_clocking_contract is None:
            cad_clocking_contract = current_clocking
        elif current_clocking != cad_clocking_contract:
            raise ValueError(
                f"{size} CAD/frame clocking differs from the other sizes"
            )
        target_force = float(
            campaign["contact_observation_policy"][
                "preferred_target_normal_force_n"
            ]
        )
        campaign_hash = sha256_file(run_dir / "sampling_campaign.json")
        campaign_quality[size] = quality
        campaign_sources[size] = {
            "run_directory": str(run_dir),
            "run_session_id": campaign["run_session_id"],
            "sampling_campaign_sha256": campaign_hash,
            "sampling_plan_path": campaign["sampling_plan_path"],
            "sampling_plan_sha256": campaign["sampling_plan_sha256"],
            "dataset_group_key": campaign["dataset_group_key"],
            "target_normal_force_n": target_force,
        }
        size_key = size.lower()
        size_index = SIZE_TO_INDEX[size_key]
        for sample in samples:
            sequence = np.asarray(sample.phase_sequence, dtype=np.float32).copy()
            if sequence.shape != (len(INPUT_CHANNELS), PHASE_BINS):
                raise ValueError(
                    f"{sample.attempt['sample_id']} shape {sequence.shape}"
                )
            sequence[6] -= sequence[6, 0]
            partition = sample.partition
            if partition == "quarantine":
                state_name = "boundary_diagnostic"
                state_index = -1
            else:
                state_name = REAL_STATE_TO_CANONICAL[sample.contact_state]
                state_index = STATE_TO_INDEX[state_name]
            context = np.zeros(len(CONTEXT_ORDER), dtype=np.float32)
            context[size_index] = 1.0
            context[3] = target_force / TARGET_FORCE_REFERENCE_N
            context[4] = (
                SIZE_BIT_RADIUS_M[size_key] / SIZE_BIT_RADIUS_M["m12"]
            )
            attempt_json = analyzer.sample_file_path(
                sample.attempt, "json_path"
            )
            cone_csv = analyzer.sample_file_path(
                sample.attempt, "cone_csv_path"
            )
            direction_deg = float(sample.direction_deg) % 360.0
            records.append(
                {
                    "sample_id": str(sample.attempt["sample_id"]),
                    "size": size_key,
                    "size_index": size_index,
                    "partition": partition,
                    "contact_state": state_name,
                    "state_index": state_index,
                    "direction_deg": direction_deg,
                    "direction_class": int(direction_class(direction_deg)),
                    "origin_group": (
                        f"{campaign['run_session_id']}::"
                        f"{sample.attempt['sample_id']}"
                    ),
                    "run_session_id": campaign["run_session_id"],
                    "campaign_sha256": campaign_hash,
                    "attempt_json_sha256": sha256_file(attempt_json),
                    "cone_csv_sha256": sha256_file(cone_csv),
                    "training_role": (
                        "TRAIN_CLEAN_ONLY"
                        if partition == "clean_candidate"
                        else (
                            "CONSUMED_HOLDOUT_DIAGNOSTIC_ONLY"
                            if partition == "holdout"
                            else (
                                "DRIFT_DIAGNOSTIC_ONLY"
                                if partition == "sentinel"
                                else "QUARANTINE_NEVER_TRAIN"
                            )
                        )
                    ),
                    "sequence": sequence,
                    "context": context,
                }
            )

    records.sort(
        key=lambda record: (
            record["size_index"],
            PARTITION_ORDER.index(record["partition"]),
            record["sample_id"],
        )
    )
    if len(records) != 78:
        raise ValueError(f"expected 78 records, got {len(records)}")
    counts = {
        partition: sum(
            record["partition"] == partition for record in records
        )
        for partition in PARTITION_ORDER
    }
    expected_counts = {
        "clean_candidate": 48,
        "holdout": 12,
        "quarantine": 12,
        "sentinel": 6,
    }
    if counts != expected_counts:
        raise ValueError(f"partition counts {counts} != {expected_counts}")

    output.mkdir(parents=True, exist_ok=True)
    clean = [
        record
        for record in records
        if record["partition"] == "clean_candidate"
    ]
    diagnostic = [
        record
        for record in records
        if record["partition"] != "clean_candidate"
    ]
    _save_npz(output / "train_clean.npz", clean)
    _save_npz(output / "nontraining_diagnostics.npz", diagnostic)
    _write_index(output / "sample_index.csv", records)

    contract = {
        "schema_version": 1,
        "dataset_id": "real_multisize_conical_fingerprints_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "REAL_DEVICE_RAW_SIGNAL_HIGHEST_AUTHORITY",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "validation_analyzer_path": str(ANALYZER_PATH),
            "validation_analyzer_sha256": sha256_file(ANALYZER_PATH),
            "common_contract_path": str(
                Path(common_module.__file__).resolve()
            ),
            "common_contract_sha256": sha256_file(
                Path(common_module.__file__).resolve()
            ),
        },
        "campaign_allowlist": campaign_sources,
        "cad_clocking_contract": cad_clocking_contract,
        "known_rejected_run": {
            "path": str(REJECTED_WRONG_PROGRAM_RUN),
            "required_flags": rejected_flags,
            "admitted": False,
        },
        "sample_count": len(records),
        "partition_counts": counts,
        "training_partition": "clean_candidate",
        "training_sample_count": len(clean),
        "holdout_status": (
            "CONSUMED_ONCE_BY_PRIOR_V1_ANALYSIS_DIAGNOSTIC_ONLY_"
            "NOT_MODEL_SELECTION"
        ),
        "input": {
            "shape": [len(INPUT_CHANNELS), PHASE_BINS],
            "channels": list(INPUT_CHANNELS),
            "phase_grid": "360 uniform bins on [0,2pi)",
            "command_z_semantics": "relative to phase-bin-zero command z",
        },
        "context": {
            "shape": [len(CONTEXT_ORDER)],
            "order": list(CONTEXT_ORDER),
        },
        "outputs": {
            "state_names": list(STATE_TO_INDEX),
            "direction_classes": 48,
            "direction_bin_centres_deg": (
                3.75 + 7.5 * np.arange(48)
            ).tolist(),
        },
        "grouping": {
            "cross_validation": "leave_one_size_out",
            "origin_group_is_indivisible": True,
            "augmentation_counted_as_independent_samples": False,
        },
        "c6_augmentation": {
            "transform_id": C6_TRANSFORM_ID,
            "training_fold_only": True,
            "validation_augmentation": False,
            "rotation_semantics": (
                "rotate Fx/Fy and Tx/Ty by +k*60 degrees; roll phase by "
                "-k*60 bins; add k*60 degrees to direction label"
            ),
            "derivation": (
                "fixture direction is CCW +X_F to +Y_F while the recorded "
                "cone tilt direction is -phase under "
                "R(phi)=Rz(-phi)*Ry(15deg)*Rz(phi)"
            ),
        },
        "campaign_quality": campaign_quality,
        "prohibitions": [
            "no online model update",
            "no holdout use for candidate selection",
            "no quarantine or sentinel training",
            "no simulation data in normalization",
            "no glob-discovered campaigns",
        ],
    }
    write_json(output / "dataset_contract.json", contract)
    hashes = {
        path.name: sha256_file(path)
        for path in (
            output / "dataset_contract.json",
            output / "sample_index.csv",
            output / "train_clean.npz",
            output / "nontraining_diagnostics.npz",
        )
    }
    write_json(output / "artifact_hashes.json", hashes)
    write_json(
        ANALYSIS_ROOT / "dataset_build_report.json",
        {
            "status": "PASS",
            "dataset_directory": str(output),
            "partition_counts": counts,
            "training_sample_count": len(clean),
            "nontraining_diagnostic_sample_count": len(diagnostic),
            "artifact_hashes": hashes,
        },
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATASET_ROOT)
    arguments = parser.parse_args()
    contract = build_dataset(arguments.output.resolve())
    print(
        "dataset PASS:",
        contract["sample_count"],
        "records;",
        contract["training_sample_count"],
        "clean training records",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
