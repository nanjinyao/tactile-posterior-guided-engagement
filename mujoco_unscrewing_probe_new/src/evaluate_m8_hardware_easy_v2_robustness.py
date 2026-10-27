"""Untouched strict/easy evaluation for the hardware-easy-v2 M8 adapter."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.m8_direction_runtime import M8DirectionEnsemble
from src.sample_m8_conical_pilot import _load_trial
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "outputs" / "current" / "models" / "p1v3_m8_direction_models"
ADAPTED_MODEL = (
    ROOT
    / "outputs"
    / "current"
    / "models"
    / "m8_hardware_easy_v2_robust_direction"
)
STRICT_AUDIT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh_strict_untouched"
)
EASY_AUDIT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh_easy_untouched"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_robustness_fresh"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (
            confidence < high if high < 1.0 else confidence <= high
        )
        if np.any(selected):
            value += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected]))
                - float(np.mean(correct[selected]))
            )
    return value


def _load_dataset(source: Path) -> tuple[dict, list[tuple[Path, dict, np.ndarray, tuple[str, ...]]]]:
    contract = json.loads((source / "sampling_contract.json").read_text(encoding="utf-8"))
    if contract.get("training_use") != "forbidden":
        raise ValueError(f"{source}: untouched evaluation requires training_use=forbidden")
    records = []
    for path in sorted((source / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") != "ok":
            raise ValueError(f"{path}: non-ok untouched trial")
        records.append((path, metadata, sequence, columns))
    if len(records) != int(contract["total"]):
        raise ValueError(f"{source}: contract/trial count mismatch")
    return contract, records


def _evaluate(
    runtime: M8DirectionEnsemble,
    records: list[tuple[Path, dict, np.ndarray, tuple[str, ...]]],
    bit_radius_m: float = 0.006 / math.sqrt(3.0),
) -> tuple[dict, list[dict]]:
    rows = []
    excluded = []
    bit_radius = float(bit_radius_m)
    for path, metadata, sequence, columns in records:
        valid = bool(metadata.get("direction_supervision_valid", True))
        if not valid:
            excluded.append(
                {
                    "trial_id": metadata["trial_id"],
                    "early_capture": bool(metadata.get("early_capture", False)),
                    "capture_during_scan": bool(
                        metadata.get("capture_during_scan", False)
                    ),
                    "radius_m": float(metadata["radius_m"]),
                }
            )
            continue
        indices = [columns.index(name) for name in training.INPUT_CHANNELS]
        estimate = runtime.infer_phase_sequence(
            sequence[indices],
            target_down_force_n=float(metadata["target_down_force_n"]),
            bit_radius_m=bit_radius,
        )
        true_angle = training._direction_angle_deg(metadata)
        error = float(
            training._circular_error_deg(
                np.asarray([estimate.circular_mean_direction_deg]),
                np.asarray([true_angle]),
            )[0]
        )
        true_class = training._direction_class(true_angle)
        rows.append(
            {
                "trial_id": metadata["trial_id"],
                "radius_m": float(metadata["radius_m"]),
                "radial_level": int(metadata["radial_level"]),
                "sector": int(metadata["rotation_index"]),
                "search_state": metadata["search_state"],
                "observation_stratum": (
                    "ideal"
                    if metadata.get("wrench_observation", "ideal") == "ideal"
                    else "sensor_prior"
                ),
                "true_direction_deg": true_angle,
                "predicted_direction_deg": estimate.circular_mean_direction_deg,
                "angle_error_deg": error,
                "within_7p5": error <= 7.5 + 1.0e-9,
                "confidence": float(estimate.confidence),
                "aps_size": len(estimate.aps_direction_classes),
                "aps_contains_true": true_class in estimate.aps_direction_classes,
                "transport_allowed": bool(estimate.transport_allowed),
                "state_correct": estimate.top_state == metadata["search_state"],
            }
        )
    if not rows:
        raise ValueError("no valid full fingerprints in untouched dataset")
    errors = np.asarray([row["angle_error_deg"] for row in rows], dtype=float)
    within = np.asarray([row["within_7p5"] for row in rows], dtype=bool)
    aps = np.asarray([row["aps_contains_true"] for row in rows], dtype=bool)
    aps_size = np.asarray([row["aps_size"] for row in rows], dtype=int)
    accepted = np.asarray([row["transport_allowed"] for row in rows], dtype=bool)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    metrics = {
        "total_task_outcomes": len(records),
        "valid_full_fingerprint_samples": len(rows),
        "capture_or_incomplete_routed_samples": len(excluded),
        "valid_full_fingerprint_rate": len(rows) / len(records),
        "capture_routing_rate": len(excluded) / len(records),
        "angle_error_median_deg": float(np.median(errors)),
        "angle_error_p90_deg": float(np.quantile(errors, 0.90)),
        "angle_within_7p5_rate": float(np.mean(within)),
        "aps_coverage": float(np.mean(aps)),
        "aps_mean_set_size": float(np.mean(aps_size)),
        "transport_accept_rate": float(np.mean(accepted)),
        "transport_accepted_count": int(np.sum(accepted)),
        "transport_accepted_within_7p5_rate": (
            float(np.mean(within[accepted])) if np.any(accepted) else None
        ),
        "transport_accepted_catastrophic_gt90_count": int(
            np.sum((errors > 90.0) & accepted)
        ),
        "all_catastrophic_gt90_count": int(np.sum(errors > 90.0)),
        "acceptable_event_ece_10bin": _ece(confidence, within),
        "diagnostic_state_accuracy": float(
            np.mean([row["state_correct"] for row in rows])
        ),
        "zero_observed_accepted_catastrophe_rule_of_three_95pct_upper": (
            3.0 / int(np.sum(accepted)) if np.any(accepted) else None
        ),
    }
    strata = {}
    for key in ("radial_level", "sector", "observation_stratum", "search_state"):
        strata[key] = {}
        for value in sorted({row[key] for row in rows}, key=str):
            selected = np.asarray([row[key] == value for row in rows], dtype=bool)
            accepted_selected = accepted & selected
            strata[key][str(value)] = {
                "samples": int(np.sum(selected)),
                "within_7p5_rate": float(np.mean(within[selected])),
                "angle_error_p90_deg": float(np.quantile(errors[selected], 0.90)),
                "transport_accept_rate": float(np.mean(accepted[selected])),
                "transport_accepted_count": int(np.sum(accepted_selected)),
                "transport_accepted_within_7p5_rate": (
                    float(np.mean(within[accepted_selected]))
                    if np.any(accepted_selected)
                    else None
                ),
                "accepted_catastrophic_gt90_count": int(
                    np.sum((errors > 90.0) & accepted & selected)
                ),
            }
    metrics["worst_radial_within_7p5_rate"] = min(
        item["within_7p5_rate"] for item in strata["radial_level"].values()
    )
    metrics["worst_sector_within_7p5_rate"] = min(
        item["within_7p5_rate"] for item in strata["sector"].values()
    )
    radial_accepted_rates = [
        item["transport_accepted_within_7p5_rate"]
        for item in strata["radial_level"].values()
        if item["transport_accepted_within_7p5_rate"] is not None
    ]
    metrics["worst_radial_transport_accepted_within_7p5_rate"] = (
        min(radial_accepted_rates) if radial_accepted_rates else None
    )
    return {"metrics": metrics, "strata": strata, "excluded": excluded}, rows


def _strict_gate(adapted: dict, base: dict) -> dict:
    checks = {
        "within_drop_le_0p03": adapted["angle_within_7p5_rate"]
        >= base["angle_within_7p5_rate"] - 0.03,
        "p90_increase_le_2deg": adapted["angle_error_p90_deg"]
        <= base["angle_error_p90_deg"] + 2.0,
        "aps_drop_le_0p02": adapted["aps_coverage"] >= base["aps_coverage"] - 0.02,
        "aps_size_increase_le_1": adapted["aps_mean_set_size"]
        <= base["aps_mean_set_size"] + 1.0,
        "accept_drop_le_0p10": adapted["transport_accept_rate"]
        >= base["transport_accept_rate"] - 0.10,
        "accepted_within_ge_0p95": (
            adapted["transport_accepted_within_7p5_rate"] is not None
            and adapted["transport_accepted_within_7p5_rate"] >= 0.95
        ),
        "accepted_catastrophic_zero": adapted[
            "transport_accepted_catastrophic_gt90_count"
        ]
        == 0,
        "worst_radial_accepted_within_ge_0p95": (
            adapted["worst_radial_transport_accepted_within_7p5_rate"] is not None
            and adapted["worst_radial_transport_accepted_within_7p5_rate"] >= 0.95
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _easy_gate(adapted: dict) -> dict:
    checks = {
        "within_ge_0p95": adapted["angle_within_7p5_rate"] >= 0.95,
        "p90_le_7p5deg": adapted["angle_error_p90_deg"] <= 7.5 + 1.0e-9,
        "aps_coverage_ge_0p95": adapted["aps_coverage"] >= 0.95,
        "aps_size_le_4": adapted["aps_mean_set_size"] <= 4.0,
        "accept_ge_0p75": adapted["transport_accept_rate"] >= 0.75,
        "accepted_within_ge_0p97": (
            adapted["transport_accepted_within_7p5_rate"] is not None
            and adapted["transport_accepted_within_7p5_rate"] >= 0.97
        ),
        "accepted_catastrophic_zero": adapted[
            "transport_accepted_catastrophic_gt90_count"
        ]
        == 0,
        "worst_radial_accepted_within_ge_0p95": (
            adapted["worst_radial_transport_accepted_within_7p5_rate"] is not None
            and adapted["worst_radial_transport_accepted_within_7p5_rate"] >= 0.95
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    strict_source = Path(args.strict_audit).resolve()
    easy_source = Path(args.easy_audit).resolve()
    strict_contract, strict_records = _load_dataset(strict_source)
    easy_contract, easy_records = _load_dataset(easy_source)
    models = {
        "base_p1v3": M8DirectionEnsemble(Path(args.base_model).resolve()),
        "adapted_easy_v2": M8DirectionEnsemble(Path(args.adapted_model).resolve()),
    }
    results = {}
    prediction_rows = []
    for model_name, runtime in models.items():
        results[model_name] = {}
        for domain, records in (("strict", strict_records), ("easy_v2", easy_records)):
            result, rows = _evaluate(runtime, records)
            results[model_name][domain] = result
            prediction_rows.extend(
                {"model": model_name, "domain": domain, **row} for row in rows
            )
    base_strict = results["base_p1v3"]["strict"]["metrics"]
    adapted_strict = results["adapted_easy_v2"]["strict"]["metrics"]
    adapted_easy = results["adapted_easy_v2"]["easy_v2"]["metrics"]
    gates = {
        "strict_non_forgetting": _strict_gate(adapted_strict, base_strict),
        "easy_v2_untouched": _easy_gate(adapted_easy),
    }
    report = {
        "stage": "M8_hardware_easy_v2_robustness_untouched_evaluation",
        "results": results,
        "promotion_gates": gates,
        "promotion_ready_for_geometry_conditioned_simulation": all(
            gate["passed"] for gate in gates.values()
        ),
        "artifacts": {
            "base_model": str(Path(args.base_model).resolve()),
            "base_contract_sha256": _sha256(
                Path(args.base_model).resolve() / "ensemble_interface_contract.json"
            ),
            "adapted_model": str(Path(args.adapted_model).resolve()),
            "adapted_contract_sha256": _sha256(
                Path(args.adapted_model).resolve()
                / "ensemble_interface_contract.json"
            ),
            "strict_sampling_contract_sha256": _sha256(
                strict_source / "sampling_contract.json"
            ),
            "easy_sampling_contract_sha256": _sha256(
                easy_source / "sampling_contract.json"
            ),
        },
        "data_contracts": {
            "strict_training_use": strict_contract["training_use"],
            "easy_training_use": easy_contract["training_use"],
            "strict_stage": strict_contract["stage"],
            "easy_stage": easy_contract["stage"],
        },
        "claim_boundary": (
            "Untouched simulation comparison across the frozen strict scene and the hardware-easy-v2 "
            "contact-geometry surrogate. Captured traces are routed as mechanics outcomes and excluded from "
            "direction accuracy. This is not real-hardware fine-tuning or full-loop validation."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    lines = [
        "# M8 hardware-easy-v2 robustness audit",
        "",
        "Captured traces are counted as mechanics-routing outcomes, not as direction predictions.",
        "",
        "| model / domain | valid / total | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy | accepted >90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, domains in results.items():
        for domain, result in domains.items():
            item = result["metrics"]
            lines.append(
                f"| {model_name} / {domain} | {item['valid_full_fingerprint_samples']} / "
                f"{item['total_task_outcomes']} | {item['angle_error_median_deg']:.2f} / "
                f"{item['angle_error_p90_deg']:.2f} deg | {item['angle_within_7p5_rate']:.3f} | "
                f"{item['aps_coverage']:.3f} / {item['aps_mean_set_size']:.2f} | "
                f"{item['transport_accept_rate']:.3f} / "
                f"{item['transport_accepted_within_7p5_rate'] if item['transport_accepted_within_7p5_rate'] is not None else 'n/a'} | "
                f"{item['transport_accepted_catastrophic_gt90_count']} |"
            )
    lines.extend(
        [
            "",
            f"Promotion gate: **{'PASS' if report['promotion_ready_for_geometry_conditioned_simulation'] else 'FAIL'}**.",
            "",
            "Boundary: " + report["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "promotion_ready": report[
                    "promotion_ready_for_geometry_conditioned_simulation"
                ],
                "gates": gates,
                "adapted_strict": adapted_strict,
                "adapted_easy": adapted_easy,
            },
            indent=2,
        )
    )
    return 0 if report["promotion_ready_for_geometry_conditioned_simulation"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(BASE_MODEL))
    parser.add_argument("--adapted-model", default=str(ADAPTED_MODEL))
    parser.add_argument("--strict-audit", default=str(STRICT_AUDIT))
    parser.add_argument("--easy-audit", default=str(EASY_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
