#!/usr/bin/env python3
"""Export a hash-bound, CPU-only real multisize direction ensemble bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from common import (
    BUNDLE_ROOT,
    CONTEXT_ORDER,
    DIRECTION_BIN_DEG,
    DIRECTION_CLASSES,
    DIRECTION_OFFSET_DEG,
    INPUT_CHANNELS,
    MODEL_ID,
    PHASE_BINS,
    SIZES,
    STATE_NAMES,
    sha256_file,
    write_json,
)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def export_bundle(
    *,
    selected_architecture: str,
    seed_records: list[dict[str, Any]],
    split_manifest_path: Path,
    training_report_path: Path,
    candidate_metrics_path: Path,
    dataset_contract_path: Path,
    aps_quantile: float,
    output: Path = BUNDLE_ROOT,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    candidate_metrics = json.loads(
        candidate_metrics_path.read_text(encoding="utf-8")
    )
    selected_cv = next(
        record
        for record in candidate_metrics["candidates"]
        if record.get("selected", False)
    )
    copied_files: list[Path] = []
    for source, name in (
        (split_manifest_path, "split_manifest.json"),
        (training_report_path, "training_report.json"),
        (candidate_metrics_path, "candidate_metrics.json"),
        (dataset_contract_path, "dataset_contract.json"),
    ):
        target = output / name
        _copy(source, target)
        copied_files.append(target)

    exported_seeds: list[dict[str, Any]] = []
    for record in seed_records:
        seed = int(record["seed"])
        seed_root = output / "seeds" / f"seed_{seed}"
        sources = {
            "model_path": Path(record["model_path"]),
            "normalization_path": Path(record["normalization_path"]),
            "metrics_path": Path(record["metrics_path"]),
            "model_config_path": Path(record["model_config_path"]),
        }
        targets = {
            "model_path": seed_root / "model.pt",
            "normalization_path": seed_root / "normalization.npz",
            "metrics_path": seed_root / "metrics.json",
            "model_config_path": seed_root / "model_config.json",
        }
        for key in sources:
            _copy(sources[key], targets[key])
            copied_files.append(targets[key])
        exported_seeds.append(
            {
                "seed": seed,
                **{
                    key: str(path.relative_to(output))
                    for key, path in targets.items()
                },
                "model_sha256": sha256_file(targets["model_path"]),
                "normalization_sha256": sha256_file(
                    targets["normalization_path"]
                ),
                "metrics_sha256": sha256_file(targets["metrics_path"]),
                "model_config_sha256": sha256_file(
                    targets["model_config_path"]
                ),
            }
        )

    interface = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "architecture_type": selected_architecture,
        "combiner": "mean seed probabilities",
        "supported_sizes": list(SIZES),
        "input": {
            "shape": [len(INPUT_CHANNELS), PHASE_BINS],
            "channels": list(INPUT_CHANNELS),
            "phase_bins": PHASE_BINS,
            "context_order": list(CONTEXT_ORDER),
            "context_shape": [len(CONTEXT_ORDER)],
        },
        "output": {
            "state_names": list(STATE_NAMES),
            "direction_classes": DIRECTION_CLASSES,
            "direction_bin_offset_deg": DIRECTION_OFFSET_DEG,
            "direction_bin_width_deg": DIRECTION_BIN_DEG,
            "returns": [
                "state_probabilities",
                "direction_probabilities",
                "top_direction_deg",
                "circular_mean_direction_deg",
                "direction_world_xy",
                "max_probability",
                "circular_concentration",
                "aps_direction_classes",
                "aps_direction_angles_deg",
                "aps_set_size",
                "sample_quality",
            ],
        },
        "calibration": {
            "method": "clean-only leave-one-size-out APS",
            "miscoverage_target": 0.10,
            "aps_quantile": float(aps_quantile),
            "holdout_used": False,
        },
        "quality": {
            "semantics": "continuous diagnostic and shortcut-scale signal, never a validity veto",
            "formula": (
                "clip(0.45*maximum_probability + "
                "0.35*circular_concentration + "
                "0.20*(1-(aps_set_size-1)/47),0,1)"
            ),
            "weak_complete_fingerprint_remains_actionable": True,
        },
        "policy_boundary": {
            "role": (
                "high-level contact-state and direction posterior plus "
                "CAD-bounded shortcut, residual target, and corridor proposal"
            ),
            "capture_insertion_or_terminal_authority": False,
            "pure_outward_spiral_consumer": False,
            "hardware_control_sizes": ["m12"],
        },
        "seeds": exported_seeds,
    }
    interface_path = output / "ensemble_interface_contract.json"
    write_json(interface_path, interface)
    copied_files.append(interface_path)

    promotion = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "QUALIFIED",
        "qualification": "DEVELOPMENT_CV_CONTROL",
        "evidence": (
            "real clean-only leave-one-size-out cross-validation; original "
            "consumed holdout excluded from selection and calibration"
        ),
        "authorization": {
            "capped_shortcut_hardware_control": True,
            "hardware_control_sizes": ["m12"],
            "capture_control": False,
            "insertion_control": False,
            "terminal_verification": False,
        },
        "policy_control_scope": {
            "allowed": [
                "state posterior",
                "direction posterior",
                "CAD-bounded capped shortcut",
                "residual target selection",
                "CAD-bounded corridor selection",
            ],
            "forbidden": [
                "pure-outward-spiral consumption",
                "capture declaration",
                "insertion completion",
                "terminal success",
                "online weight update",
            ],
        },
        "selected_architecture": selected_architecture,
        "selected_clean_loso_metrics": selected_cv["aggregate"],
        "known_limitations": [
            (
                "leave-one-size-out direction transfer remains multimodal; "
                "this is development capped control, not production accuracy"
            ),
            (
                "M10 and M8 are inference-capable training domains but do not "
                "yet have full-loop hardware control authority"
            ),
            (
                "the v1.5 CAD clocking is nominal/operator-taught rather than "
                "independently measured as-built yaw metrology"
            ),
        ],
        "ensemble_interface_contract": "ensemble_interface_contract.json",
        "ensemble_interface_contract_sha256": sha256_file(interface_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    promotion_path = output / "promotion_contract.json"
    write_json(promotion_path, promotion)
    copied_files.append(promotion_path)

    deployment = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "architecture_type": selected_architecture,
        "deployment_authority": "CAPPED_SHORTCUT",
        "default_execution_mode": "CAPPED_SHORTCUT",
        "allowed_execution_modes": ["CAPPED_SHORTCUT", "DIAGNOSTIC_ONLY"],
        "hardware_control_sizes": ["m12"],
        "promotion_contract": "promotion_contract.json",
        "promotion_contract_sha256": sha256_file(promotion_path),
        "ensemble_interface_contract": "ensemble_interface_contract.json",
        "ensemble_interface_contract_sha256": sha256_file(interface_path),
        "seed_count": len(exported_seeds),
        "seed_models": exported_seeds,
    }
    deployment_path = output / "deployment_manifest.json"
    write_json(deployment_path, deployment)
    copied_files.append(deployment_path)

    summary = (
        "# Real multisize direction bundle v1\n\n"
        f"- model: `{MODEL_ID}`\n"
        f"- selected architecture: `{selected_architecture}`\n"
        "- evidence: 48 real clean samples, clean-only LOSO by M8/M10/M12\n"
        "- control authority: capped CAD shortcut only\n"
        "- hardware control scope: M12 only; M10/M8 remain inference-only\n"
        "- no authority: capture, insertion, terminal verification, or "
        "Pure-Outward Spiral\n"
        "- consumed holdout, quarantine, and sentinel samples were not used "
        "for fitting or candidate selection\n"
    )
    summary_path = output / "README.md"
    temporary = summary_path.with_suffix(".md.tmp")
    temporary.write_text(summary, encoding="utf-8")
    temporary.replace(summary_path)
    copied_files.append(summary_path)

    hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in copied_files
    }
    write_json(output / "artifact_hashes.json", hashes)
    return {
        "bundle_root": str(output),
        "selected_architecture": selected_architecture,
        "seed_count": len(exported_seeds),
        "promotion_status": promotion["status"],
        "deployment_authority": deployment["deployment_authority"],
        "artifact_count": len(hashes),
    }


def main() -> int:
    raise SystemExit(
        "This exporter is called by train_real_multisize_direction.py so "
        "checkpoint and split hashes cannot be supplied inconsistently."
    )


if __name__ == "__main__":
    main()
