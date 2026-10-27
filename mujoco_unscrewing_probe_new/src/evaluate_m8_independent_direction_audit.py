from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.sample_m8_conical_pilot import DEFAULT_P1V2_OUT, SPEC_PATH, _load_trial
import src.train_m8_direction_models as training


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if np.any(selected):
            value += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return value


def run(args: argparse.Namespace) -> int:
    source = Path(args.input or DEFAULT_P1V2_OUT).resolve()
    contract = json.loads((source / "sampling_contract.json").read_text(encoding="utf-8"))
    stage_name = str(contract["stage"])
    trial_paths = sorted((source / "trials").glob("*.npz"))
    if len(trial_paths) != 180:
        raise ValueError(f"expected frozen 180 trials, found {len(trial_paths)}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    runtime = M8DirectionEnsemble(Path(args.model_root) if args.model_root else None) if args.model_root else M8DirectionEnsemble()
    rows = []
    for path in tqdm(trial_paths, desc=f"{stage_name} ensemble", unit="trial", dynamic_ncols=True):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            raise ValueError(f"failed trial in frozen audit: {path.name}")
        indices = [columns.index(name) for name in training.INPUT_CHANNELS]
        estimate = runtime.infer_phase_sequence(
            sequence[indices],
            target_down_force_n=float(metadata["target_down_force_n"]),
            bit_radius_m=bit_radius,
        )
        true_angle = training._direction_angle_deg(metadata)
        top1_error = float(
            training._circular_error_deg(
                np.asarray([estimate.top_direction_deg]), np.asarray([true_angle])
            )[0]
        )
        error = float(
            training._circular_error_deg(
                np.asarray([estimate.circular_mean_direction_deg]), np.asarray([true_angle])
            )[0]
        )
        true_class = training._direction_class(true_angle)
        rows.append(
            {
                "trial_id": metadata["trial_id"],
                "search_state": metadata["search_state"],
                "radial_level": int(metadata["radial_level"]),
                "radius_mm": 1000.0 * float(metadata["radius_m"]),
                "local_angle_deg": float(metadata["base_angle_deg"]),
                "sector": int(metadata["rotation_index"]),
                "nuisance_id": metadata["nuisance_id"],
                "true_direction_deg": true_angle,
                "pred_direction_deg": estimate.top_direction_deg,
                "control_direction_deg": estimate.circular_mean_direction_deg,
                "top1_angle_error_deg": top1_error,
                "angle_error_deg": error,
                "within_7p5": error <= 7.5 + 1.0e-9,
                "confidence": estimate.confidence,
                "transport_allowed": estimate.transport_allowed,
                "aps_size": len(estimate.aps_direction_classes),
                "aps_contains_true": true_class in estimate.aps_direction_classes,
                "true_state": metadata["search_state"],
                "pred_state": estimate.top_state,
                **{
                    f"state_probability_{state}": float(
                        estimate.state_probabilities[index]
                    )
                    for index, state in enumerate(training.STATE_NAMES)
                },
            }
        )

    errors = np.asarray([row["angle_error_deg"] for row in rows], dtype=float)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    acceptable = np.asarray([row["within_7p5"] for row in rows], dtype=bool)
    aps_coverage = np.asarray([row["aps_contains_true"] for row in rows], dtype=bool)
    state_correct = np.asarray([row["true_state"] == row["pred_state"] for row in rows])
    state_probabilities = np.asarray(
        [
            [row[f"state_probability_{state}"] for state in training.STATE_NAMES]
            for row in rows
        ],
        dtype=float,
    )
    state_true = np.asarray(
        [training.STATE_TO_INDEX[row["true_state"]] for row in rows], dtype=int
    )
    state_confidence = np.max(state_probabilities, axis=1)
    state_true_probability = state_probabilities[np.arange(len(rows)), state_true]
    state_one_hot = np.eye(len(training.STATE_NAMES))[state_true]
    catastrophic = errors > 90.0
    transport_allowed = np.asarray([row["transport_allowed"] for row in rows], dtype=bool)
    aps_sizes = np.asarray([row["aps_size"] for row in rows], dtype=int)
    sector_metrics = {}
    for sector in range(6):
        selected = np.asarray([row["sector"] == sector for row in rows])
        sector_metrics[str(sector)] = {
            "samples": int(np.sum(selected)),
            "within_7p5_rate": float(np.mean(acceptable[selected])),
            "angle_p90_deg": float(np.quantile(errors[selected], 0.90)),
        }
    radial_metrics = {}
    for radial_level in range(5):
        selected = np.asarray([row["radial_level"] == radial_level for row in rows])
        radial_metrics[str(radial_level)] = {
            "radius_mm": float(rows[int(np.flatnonzero(selected)[0])]["radius_mm"]),
            "state": rows[int(np.flatnonzero(selected)[0])]["search_state"],
            "samples": int(np.sum(selected)),
            "within_7p5_rate": float(np.mean(acceptable[selected])),
            "angle_p90_deg": float(np.quantile(errors[selected], 0.90)),
            "aps_coverage": float(np.mean(aps_coverage[selected])),
        }
    local_angle_metrics = {}
    for local_angle in sorted({row["local_angle_deg"] for row in rows}):
        selected = np.asarray([row["local_angle_deg"] == local_angle for row in rows])
        local_angle_metrics[f"{local_angle:.1f}"] = {
            "samples": int(np.sum(selected)),
            "within_7p5_rate": float(np.mean(acceptable[selected])),
            "angle_p90_deg": float(np.quantile(errors[selected], 0.90)),
            "catastrophic_gt90_count": int(np.sum(catastrophic[selected])),
        }

    report = {
        "stage": f"{stage_name}_evaluation",
        "samples": len(rows),
        "metrics": {
            "angle_error_median_deg": float(np.median(errors)),
            "angle_error_p90_deg": float(np.quantile(errors, 0.90)),
            "angle_within_7p5_rate": float(np.mean(acceptable)),
            "catastrophic_gt90_rate": float(np.mean(catastrophic)),
            "catastrophic_gt90_count": int(np.sum(catastrophic)),
            "catastrophic_transport_allowed_count": int(
                np.sum(catastrophic & transport_allowed)
            ),
            "catastrophic_aps_size_min": (
                int(np.min(aps_sizes[catastrophic])) if np.any(catastrophic) else None
            ),
            "aps_coverage": float(np.mean(aps_coverage)),
            "aps_mean_set_size": float(np.mean(aps_sizes)),
            "confidence_accept_rate": float(np.mean(transport_allowed)),
            "transport_accepted_count": int(np.sum(transport_allowed)),
            "transport_rejected_count": int(np.sum(~transport_allowed)),
            "transport_accepted_within_7p5_rate": (
                float(np.mean(acceptable[transport_allowed]))
                if np.any(transport_allowed)
                else None
            ),
            "acceptable_event_ece_10bin": _ece(confidence, acceptable),
            "diagnostic_state_accuracy": float(np.mean(state_correct)),
            "state_nll": float(
                -np.mean(np.log(np.clip(state_true_probability, 1.0e-12, 1.0)))
            ),
            "state_brier": float(
                np.mean(np.sum((state_probabilities - state_one_hot) ** 2, axis=1))
            ),
            "state_ece_10bin": _ece(state_confidence, state_correct),
            "state_mean_confidence": float(np.mean(state_confidence)),
            "sector_within_rate_range": float(
                max(item["within_7p5_rate"] for item in sector_metrics.values())
                - min(item["within_7p5_rate"] for item in sector_metrics.values())
            ),
        },
        "radial_metrics": radial_metrics,
        "local_angle_metrics": local_angle_metrics,
        "sector_metrics": sector_metrics,
        "claim_boundary": contract["claim_boundary"] + " Not real-robot posterior or closed-loop benefit evidence.",
        "updated_utc": _utc_now(),
    }
    _atomic_json(source / "independent_direction_report.json", report)
    with (source / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = report["metrics"]
    lines = [
        f"# {stage_name}",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "| samples | angle median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | direction ECE | state accuracy / ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {len(rows)} | {metrics['angle_error_median_deg']:.2f} / {metrics['angle_error_p90_deg']:.2f} deg "
            f"| {metrics['angle_within_7p5_rate']:.3f} | {metrics['aps_coverage']:.3f} / "
            f"{metrics['aps_mean_set_size']:.2f} | {metrics['confidence_accept_rate']:.3f} / "
            f"{metrics['transport_accepted_within_7p5_rate']:.3f} | "
            f"{metrics['acceptable_event_ece_10bin']:.3f} | {metrics['diagnostic_state_accuracy']:.3f} / "
            f"{metrics['state_ece_10bin']:.3f} |"
        ),
        "",
        "## Radial strata",
        "",
        "| level | radius | state | within 7.5 deg | p90 | APS coverage |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for level, item in radial_metrics.items():
        lines.append(
            f"| {level} | {item['radius_mm']:.3f} mm | {item['state']} | "
            f"{item['within_7p5_rate']:.3f} | {item['angle_p90_deg']:.2f} deg | {item['aps_coverage']:.3f} |"
        )
    lines.extend(["", "## Boundary", "", report["claim_boundary"], ""])
    (source / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen M8 P1v2 direction audit.")
    parser.add_argument("--input")
    parser.add_argument("--model-root")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
