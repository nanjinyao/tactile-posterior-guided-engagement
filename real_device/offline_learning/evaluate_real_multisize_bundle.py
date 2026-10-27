#!/usr/bin/env python3
"""Report-only replay of the frozen real bundle on non-training evidence.

The 12 holdout samples were already consumed by the earlier observability
study.  They are never used here to fit normalization, tune a hyperparameter,
choose an architecture, or update a weight.  This script only records how the
already-selected frozen ensemble behaves, with M12 called out because it is
the sole initial full-loop hardware-control scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    ANALYSIS_ROOT,
    BUNDLE_ROOT,
    DATASET_ROOT,
    FrozenSimulationAdapter,
    FourierLinearModel,
    SIZES,
    SimulationAdapterTailL2SP,
    TinyCircularCNN,
    direction_metrics,
    posterior_direction_deg,
    sha256_file,
    state_metrics,
    write_json,
)
from train_real_multisize_direction import _predict


def _construct(
    architecture: str, state: dict[str, torch.Tensor]
) -> torch.nn.Module:
    if architecture == FourierLinearModel.architecture_type:
        return FourierLinearModel()
    if architecture == TinyCircularCNN.architecture_type:
        return TinyCircularCNN()
    if architecture == FrozenSimulationAdapter.architecture_type:
        return FrozenSimulationAdapter(
            state["legacy_context_mean"].cpu().numpy(),
            state["legacy_context_std"].cpu().numpy(),
        )
    if architecture == SimulationAdapterTailL2SP.architecture_type:
        return SimulationAdapterTailL2SP(
            state["legacy_context_mean"].cpu().numpy(),
            state["legacy_context_std"].cpu().numpy(),
        )
    raise ValueError(f"unsupported architecture {architecture}")


def _arrays(path: Path, partition: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        mask = values["partition"].astype(str) == partition
        return {
            key: np.asarray(values[key][mask])
            for key in (
                "sequence",
                "context",
                "state_index",
                "direction_deg",
                "size_index",
                "sample_id",
            )
        }


def _evaluate(
    root: Path, arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    deployment = json.loads(
        (root / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    architecture = str(deployment["architecture_type"])
    state_predictions = []
    direction_predictions = []
    for seed in deployment["seed_models"]:
        model_path = root / str(seed["model_path"])
        normalization_path = root / str(seed["normalization_path"])
        if sha256_file(model_path) != seed["model_sha256"]:
            raise ValueError(f"model hash mismatch: {model_path}")
        if (
            sha256_file(normalization_path)
            != seed["normalization_sha256"]
        ):
            raise ValueError(
                f"normalization hash mismatch: {normalization_path}"
            )
        state = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        model = _construct(architecture, state)
        model.load_state_dict(state)
        normalizer = dict(np.load(normalization_path))
        state_probability, direction_probability = _predict(
            model,
            arrays["sequence"].astype(np.float32),
            arrays["context"].astype(np.float32),
            normalizer,
        )
        state_predictions.append(state_probability)
        direction_predictions.append(direction_probability)
    state_probability = np.mean(state_predictions, axis=0)
    direction_probability = np.mean(direction_predictions, axis=0)
    report: dict[str, Any] = {
        "count": int(len(arrays["direction_deg"])),
        "state": state_metrics(
            state_probability, arrays["state_index"].astype(np.int64)
        ),
        "direction_top1": direction_metrics(
            direction_probability, arrays["direction_deg"]
        ),
        "direction_circular_mean": {
            "errors_deg": np.abs(
                (
                    posterior_direction_deg(direction_probability)
                    - arrays["direction_deg"]
                    + 180.0
                )
                % 360.0
                - 180.0
            ).tolist()
        },
        "by_size": {},
    }
    for size_index, size in enumerate(SIZES):
        mask = arrays["size_index"] == size_index
        report["by_size"][size] = {
            "count": int(np.sum(mask)),
            "state": state_metrics(
                state_probability[mask],
                arrays["state_index"][mask].astype(np.int64),
            ),
            "direction_top1": direction_metrics(
                direction_probability[mask],
                arrays["direction_deg"][mask],
            ),
        }
    return report, state_probability, direction_probability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=BUNDLE_ROOT)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DATASET_ROOT / "nontraining_diagnostics.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ANALYSIS_ROOT / "bundle_nontraining_replay_report.json",
    )
    args = parser.parse_args()

    holdout = _arrays(args.diagnostics, "holdout")
    sentinel = _arrays(args.diagnostics, "sentinel")
    holdout_report, _, _ = _evaluate(args.bundle, holdout)
    sentinel_report, _, _ = _evaluate(args.bundle, sentinel)
    m12 = holdout_report["by_size"]["m12"]["direction_top1"]
    m12_development_support = bool(
        m12["count"] == 4
        and m12["catastrophic_gt90_count"] == 0
        and m12["p90_error_deg"] <= 45.0
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(args.bundle.resolve()),
        "deployment_manifest_sha256": sha256_file(
            args.bundle / "deployment_manifest.json"
        ),
        "data_firewall": {
            "fit_partitions": ["clean_candidate"],
            "report_only_partitions_opened": ["holdout", "sentinel"],
            "holdout_status": (
                "PREVIOUSLY_CONSUMED_REPORT_ONLY_NOT_SELECTION_OR_CALIBRATION"
            ),
            "quarantine_opened": False,
            "weights_or_normalization_updated": False,
        },
        "holdout": holdout_report,
        "sentinel": sentinel_report,
        "initial_hardware_scope": {
            "sizes": ["m12"],
            "authority": "DEVELOPMENT_CV_CONTROL_CAPPED_SHORTCUT",
            "report_only_m12_support": m12_development_support,
            "boundary": (
                "small consumed holdout supports rapid bounded iteration; "
                "it is not production qualification or an untouched audit"
            ),
        },
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "m12_holdout": m12,
                "m12_development_support": m12_development_support,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
