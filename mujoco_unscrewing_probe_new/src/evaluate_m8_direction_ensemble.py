from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "current" / "models" / "p1_m8_direction_models"
DEFAULT_SEEDS = (20260710, 20260711, 20260712)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{Path('/proc/self').resolve().name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / np.sum(probability, axis=1, keepdims=True)


def _aps_quantile(probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.10) -> float:
    scores = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == target)[0])
        scores.append(float(cumulative[rank]))
    level = min(math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def _select_gate(
    probabilities: np.ndarray,
    angles: np.ndarray,
    target_within_rate: float = 0.98,
    min_accepted: int = 24,
) -> float:
    confidence = np.max(probabilities, axis=1)
    predicted = training._class_angle_deg(np.argmax(probabilities, axis=1))
    error = training._circular_error_deg(predicted, angles)
    for threshold in sorted(set(map(float, confidence))):
        accepted = confidence >= threshold
        if np.sum(accepted) < min_accepted:
            continue
        if np.mean(error[accepted] <= 7.5 + 1.0e-9) >= target_within_rate:
            return threshold
    return 1.0


def _aps_sizes(probabilities: np.ndarray, aps_quantile: float) -> np.ndarray:
    sizes = []
    for probability in probabilities:
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        sizes.append(
            min(int(np.searchsorted(cumulative, aps_quantile, side="left")) + 1, len(order))
        )
    return np.asarray(sizes, dtype=int)


def _circular_concentration(probabilities: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(training._class_angle_deg(np.arange(training.DIRECTION_CLASSES)))
    vector = probabilities @ np.exp(1j * angles)
    return np.abs(vector)


def _circular_mean_angle_deg(probabilities: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(training._class_angle_deg(np.arange(training.DIRECTION_CLASSES)))
    vector = probabilities @ np.exp(1j * angles)
    return np.mod(np.rad2deg(np.angle(vector)), 360.0)


def _ensemble_span_deg(per_seed_probabilities: np.ndarray) -> np.ndarray:
    top = training._class_angle_deg(np.argmax(per_seed_probabilities, axis=2))
    spans = np.zeros(top.shape[1], dtype=float)
    for index in range(top.shape[1]):
        values = top[:, index]
        difference = np.abs((values[:, None] - values[None, :] + 180.0) % 360.0 - 180.0)
        spans[index] = float(np.max(difference))
    return spans


def _select_structured_gate(
    probabilities: np.ndarray,
    per_seed_probabilities: np.ndarray,
    angles: np.ndarray,
    aps_quantile: float,
    *,
    target_within_rate: float = 0.98,
    min_accepted: int = 24,
) -> dict:
    confidence = np.max(probabilities, axis=1)
    aps_size = _aps_sizes(probabilities, aps_quantile)
    concentration = _circular_concentration(probabilities)
    span = _ensemble_span_deg(per_seed_probabilities)
    predicted = _circular_mean_angle_deg(probabilities)
    error = training._circular_error_deg(predicted, angles)
    quantiles = np.linspace(0.0, 0.9, 10)
    confidence_candidates = sorted(set([0.0, *np.quantile(confidence, quantiles).tolist()]))
    concentration_candidates = sorted(set([0.0, *np.quantile(concentration, quantiles).tolist()]))
    aps_candidates = sorted(set(map(int, aps_size)))
    span_candidates = sorted(set([180.0, *np.quantile(span, quantiles).tolist()]))
    best = None
    for confidence_min in confidence_candidates:
        for aps_max in aps_candidates:
            for concentration_min in concentration_candidates:
                for span_max in span_candidates:
                    accepted = (
                        (confidence >= confidence_min)
                        & (aps_size <= aps_max)
                        & (concentration >= concentration_min)
                        & (span <= span_max + 1.0e-9)
                    )
                    count = int(np.sum(accepted))
                    if count < min_accepted:
                        continue
                    within = float(np.mean(error[accepted] <= 7.5 + 1.0e-9))
                    catastrophic = float(np.mean(error[accepted] > 90.0))
                    if within < target_within_rate or catastrophic > 0.0:
                        continue
                    candidate = (
                        count,
                        within,
                        -float(np.mean(aps_size[accepted])),
                        {
                            "min_max_probability": float(confidence_min),
                            "max_aps_set_size": int(aps_max),
                            "min_circular_concentration": float(concentration_min),
                            "max_ensemble_top1_span_deg": float(span_max),
                        },
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
    if best is None:
        return {
            "min_max_probability": 1.0,
            "max_aps_set_size": 0,
            "min_circular_concentration": 1.0,
            "max_ensemble_top1_span_deg": 0.0,
        }
    return best[3]


def _gate_mask(
    probabilities: np.ndarray,
    per_seed_probabilities: np.ndarray,
    aps_quantile: float,
    gate: dict,
) -> np.ndarray:
    return (
        (np.max(probabilities, axis=1) >= gate["min_max_probability"])
        & (_aps_sizes(probabilities, aps_quantile) <= gate["max_aps_set_size"])
        & (_circular_concentration(probabilities) >= gate["min_circular_concentration"])
        & (
            _ensemble_span_deg(per_seed_probabilities)
            <= gate["max_ensemble_top1_span_deg"] + 1.0e-9
        )
    )


def _evaluate(
    state_probability: np.ndarray,
    direction_probability: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    per_seed_direction_probability: np.ndarray,
    aps_quantile: float,
    gate: dict,
) -> dict:
    state_true, direction_true, angles = labels
    state_pred = np.argmax(state_probability, axis=1)
    direction_pred = np.argmax(direction_probability, axis=1)
    top1_error = training._circular_error_deg(training._class_angle_deg(direction_pred), angles)
    error = training._circular_error_deg(_circular_mean_angle_deg(direction_probability), angles)
    coverage, sizes = [], []
    for probability, target in zip(direction_probability, direction_true):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        size = min(int(np.searchsorted(cumulative, aps_quantile, side="left")) + 1, len(order))
        coverage.append(int(target) in set(order[:size].tolist()))
        sizes.append(size)
    accepted = _gate_mask(
        direction_probability, per_seed_direction_probability, aps_quantile, gate
    )
    return {
        "samples": len(angles),
        "state_accuracy": float(np.mean(state_pred == state_true)),
        "direction_top1_accuracy": float(np.mean(direction_pred == direction_true)),
        "top1_angle_error_p90_deg": float(np.quantile(top1_error, 0.90)),
        "angle_error_median_deg": float(np.median(error)),
        "angle_error_p90_deg": float(np.quantile(error, 0.90)),
        "angle_within_7p5_rate": float(np.mean(error <= 7.5 + 1.0e-9)),
        "aps_coverage": float(np.mean(coverage)),
        "aps_mean_set_size": float(np.mean(sizes)),
        "gate": gate,
        "gate_accept_rate": float(np.mean(accepted)),
        "gate_accepted_samples": int(np.sum(accepted)),
        "gate_within_7p5_rate": (
            float(np.mean(error[accepted] <= 7.5 + 1.0e-9)) if np.any(accepted) else None
        ),
        "gate_catastrophic_gt90_rate": (
            float(np.mean(error[accepted] > 90.0)) if np.any(accepted) else None
        ),
    }


def _examples(split_seed: int, targeted_data: Path | None = None) -> dict[str, list[dict]]:
    p0b = training.load_trials(training.P0B)
    p0c = training.load_trials(training.P0C)
    p0d = training.load_trials(training.P0D)
    p0a = training.load_trials(training.P0A)
    split = training.grouped_split(p0b, split_seed)
    sector_split = training.grouped_sector_split(p0d, split_seed)
    p0b_val = training.make_examples(p0b, set(split["val"]), (0,))
    p0d_val = training.make_sector_examples(p0d, set(sector_split["val"]))
    targeted_val = []
    if targeted_data is not None:
        targeted = training.load_trials(targeted_data)
        targeted_split = training.grouped_targeted_split(targeted, split_seed)
        targeted_val = training.make_examples(targeted, set(targeted_split["val"]), (0,))
    result = {
        "calibration": p0b_val + p0d_val + targeted_val,
        "p0b_test": training.make_examples(p0b, set(split["test"]), (0,)),
        "p0d_test": training.make_sector_examples(p0d, set(sector_split["test"])),
        "p0c_hard": training.make_examples(p0c, None, (0,)),
        "p0a_full_angle": training.make_examples(p0a, None, (0,)),
    }
    if targeted_val:
        result["targeted_val"] = targeted_val
    return result


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    seed_dirs = [output / "seeds" / f"seed_{seed}" for seed in args.seeds]
    targeted_data = Path(args.targeted_data).resolve() if args.targeted_data else None
    examples = _examples(args.split_seed, targeted_data)
    accumulated = {
        name: {"state": [], "direction": [], "labels": None} for name in examples
    }
    bar = tqdm(seed_dirs, desc="P1 ensemble", unit="model", dynamic_ncols=True)
    for seed_dir in bar:
        normalization = np.load(seed_dir / "normalization.npz")
        metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        model = training.CircularDirectionCNN()
        model.load_state_dict(
            torch.load(seed_dir / "circular_cnn_best.pt", map_location="cpu", weights_only=True)
        )
        for name, split_examples in examples.items():
            arrays = training._arrays(
                split_examples, normalization["context_mean"], normalization["context_std"]
            )
            state_logits, direction_logits, states, directions, angles = training._collect_logits(
                model, arrays, torch.device("cpu")
            )
            accumulated[name]["state"].append(_softmax(state_logits))
            accumulated[name]["direction"].append(
                _softmax(direction_logits, float(metrics["temperature"]))
            )
            accumulated[name]["labels"] = (states, directions, angles)

    averaged = {
        name: (
            np.mean(values["state"], axis=0),
            np.mean(values["direction"], axis=0),
            np.asarray(values["direction"]),
            values["labels"],
        )
        for name, values in accumulated.items()
    }
    calibration = averaged["calibration"]
    aps = _aps_quantile(calibration[1], calibration[3][1])
    gate = _select_structured_gate(calibration[1], calibration[2], calibration[3][2], aps)
    results = {
        name: _evaluate(state, direction, labels, per_seed, aps, gate)
        for name, (state, direction, per_seed, labels) in averaged.items()
    }
    report = {
        "stage": "P1_M8_three_seed_probability_ensemble",
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "aps_alpha": 0.10,
        "aps_quantile": aps,
        "structured_transport_gate": gate,
        "targeted_data": str(targeted_data) if targeted_data else None,
        "metrics": results,
        "claim_boundary": (
            "Three-seed probability ensemble on simulation-only grouped splits. Calibration uses only the combined "
            "P0b/P0d validation plus optional grouped targeted calibration. P1v2 is never loaded. "
            "It is not a real-robot posterior or closed-loop performance proof."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "ensemble_metrics.json", report)
    _atomic_json(
        output / "ensemble_interface_contract.json",
        {
            "checkpoints": [str(path / "circular_cnn_best.pt") for path in seed_dirs],
            "combiner": "mean calibrated class probabilities",
            "direction_bins": training.DIRECTION_CLASSES,
            "direction_bin_width_deg": training.DIRECTION_BIN_DEG,
            "confidence_gate": gate["min_max_probability"],
            "structured_transport_gate": gate,
            "aps_quantile": aps,
            "state_role": (
                "soft action-relevant radial likelihood only: combine calibrated support-state probabilities "
                "with approximate CAD using a nonzero likelihood floor; never use a hard state gate, direct "
                "distance command, capture trigger, or terminal verifier"
            ),
            "direction_role": "allow bounded transport only when all calibrated set/concentration/disagreement gates pass",
            "fallback": "guarded local outward spiral or re-probe",
            "claim_boundary": report["claim_boundary"],
            "updated_utc": _utc_now(),
        },
    )
    lines = [
        "# M8 P1 Three-Seed Direction Ensemble",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        f"Seeds: `{args.seeds}`; structured gate: `{gate}`; APS quantile: `{aps:.3f}`",
        "",
        "| split | state acc. | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy / >90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        lines.append(
            f"| {name} | {item['state_accuracy']:.3f} | {item['angle_error_median_deg']:.2f} / "
            f"{item['angle_error_p90_deg']:.2f} deg | {item['angle_within_7p5_rate']:.3f} | "
            f"{item['aps_coverage']:.3f} / {item['aps_mean_set_size']:.2f} | "
            f"{item['gate_accept_rate']:.3f} / {item['gate_within_7p5_rate']:.3f} / "
            f"{item['gate_catastrophic_gt90_rate']:.3f} |"
        )
    lines.extend(["", "## Boundary", "", report["claim_boundary"], ""])
    (output / "ensemble_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fixed-split M8 direction ensemble.")
    parser.add_argument("--output", default=str(DEFAULT_ROOT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--targeted-data")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
