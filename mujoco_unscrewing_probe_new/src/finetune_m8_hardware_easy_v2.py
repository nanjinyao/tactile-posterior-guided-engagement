"""Balanced strict/easy replay adaptation for the M8 direction ensemble."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn

import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "outputs" / "current" / "models" / "p1v3_m8_direction_models"
EASY_DATA = (
    ROOT
    / "outputs"
    / "current"
    / "datasets"
    / "m8_hardware_easy_v2_direction_training"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "current"
    / "models"
    / "m8_hardware_easy_v2_robust_direction"
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


def _valid_direction_trial(trial: training.LoadedTrial) -> bool:
    return bool(trial.metadata.get("direction_supervision_valid", True))


def _prepare_examples(
    split_seed: int,
    easy_data: Path,
    strict_development_data: tuple[Path, ...] = (),
    easy_development_data: tuple[Path, ...] = (),
    development_replay_factor: int = 1,
) -> tuple[dict, dict]:
    p0b = training.load_trials(training.P0B)
    p0d = training.load_trials(training.P0D)
    targeted = training.load_trials(training.P1V3_TARGETED)
    p0b_split = training.grouped_split(p0b, split_seed)
    p0d_split = training.grouped_sector_split(p0d, split_seed)
    targeted_split = training.grouped_targeted_split(targeted, split_seed)

    strict_train = (
        training.make_examples(p0b, set(p0b_split["train"]), tuple(range(6)))
        + training.make_sector_examples(p0d, set(p0d_split["train"]))
        + training.make_examples(targeted, set(targeted_split["train"]), (0,))
    )
    strict_calibration = (
        training.make_examples(p0b, set(p0b_split["val"]), (0,))
        + training.make_sector_examples(p0d, set(p0d_split["val"]))
        + training.make_examples(targeted, set(targeted_split["val"]), (0,))
    )

    easy_all = training.load_trials(easy_data)
    easy_split = training.grouped_targeted_split(easy_all, split_seed + 1000)
    easy_valid = [trial for trial in easy_all if _valid_direction_trial(trial)]
    easy_train = training.make_examples(
        easy_valid, set(easy_split["train"]), (0,)
    )
    easy_calibration = training.make_examples(
        easy_valid, set(easy_split["val"]), (0,)
    )
    if not easy_train or not easy_calibration:
        raise ValueError("easy-v2 split has no valid full fingerprints")
    development_records = []
    for domain, paths in (
        ("strict", strict_development_data),
        ("easy", easy_development_data),
    ):
        for path_index, path in enumerate(paths):
            contract_path = path / "sampling_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if contract.get("training_use") != "forbidden":
                raise ValueError(f"{path}: expected a previously untouched contract")
            all_trials = training.load_trials(path)
            split_dev = training.grouped_targeted_split(
                all_trials, split_seed + 2000 + 100 * path_index + (0 if domain == "strict" else 50)
            )
            valid_trials = [trial for trial in all_trials if _valid_direction_trial(trial)]
            train_dev = training.make_examples(
                valid_trials, set(split_dev["train"]), (0,)
            )
            calibration_dev = training.make_examples(
                valid_trials, set(split_dev["val"]), (0,)
            )
            replayed_train = train_dev * development_replay_factor
            if domain == "strict":
                strict_train.extend(replayed_train)
                strict_calibration.extend(calibration_dev)
            else:
                easy_train.extend(replayed_train)
                easy_calibration.extend(calibration_dev)
            development_records.append(
                {
                    "domain": domain,
                    "path": str(path),
                    "sampling_contract_sha256": _sha256(contract_path),
                    "original_training_use": "forbidden",
                    "new_role": "development_after_failed_v2_evaluation",
                    "all_ok_trials": len(all_trials),
                    "valid_full_fingerprints": len(valid_trials),
                    "train_examples": len(train_dev),
                    "development_replay_factor": development_replay_factor,
                    "effective_train_examples": len(replayed_train),
                    "calibration_examples": len(calibration_dev),
                    "train_groups": sorted(split_dev["train"]),
                    "calibration_groups": sorted(split_dev["val"]),
                    "group_intersection": sorted(
                        set(split_dev["train"]) & set(split_dev["val"])
                    ),
                }
            )
    provenance = {
        "split_seed": split_seed,
        "split_before_capture_filter": True,
        "strict": {
            "train_examples": len(strict_train),
            "calibration_examples": len(strict_calibration),
            "p0b_train_groups": sorted(p0b_split["train"]),
            "p0d_train_groups": sorted(p0d_split["train"]),
            "targeted_train_groups": sorted(targeted_split["train"]),
        },
        "easy": {
            "all_ok_trials": len(easy_all),
            "valid_full_fingerprints": len(easy_valid),
            "capture_or_incomplete_excluded": len(easy_all) - len(easy_valid),
            "train_examples": len(easy_train),
            "calibration_examples": len(easy_calibration),
            "train_groups": sorted(easy_split["train"]),
            "calibration_groups": sorted(easy_split["val"]),
            "group_intersection": sorted(
                set(easy_split["train"]) & set(easy_split["val"])
            ),
        },
        "previously_untouched_now_consumed_as_development": development_records,
        "development_replay_factor": development_replay_factor,
    }
    return {
        "strict_train": strict_train,
        "strict_calibration": strict_calibration,
        "easy_train": easy_train,
        "easy_calibration": easy_calibration,
    }, provenance


def _domain_arrays(
    examples: dict[str, list[dict]], context_mean: np.ndarray, context_std: np.ndarray
) -> dict[str, tuple[np.ndarray, ...]]:
    return {
        name: training._arrays(items, context_mean, context_std)
        for name, items in examples.items()
    }


def _batch_loss(
    model: training.CircularDirectionCNN,
    arrays: tuple[np.ndarray, ...],
    indices: np.ndarray,
    device: torch.device,
    soft_sigma_deg: float,
) -> torch.Tensor:
    sequence, context, states, _, angles = arrays
    state_logits, direction_logits = model(
        torch.from_numpy(sequence[indices]).to(device),
        torch.from_numpy(context[indices]).to(device),
    )
    state = torch.from_numpy(states[indices]).to(device)
    soft = torch.from_numpy(
        training._circular_soft_targets(angles[indices], soft_sigma_deg)
    ).to(device)
    return nn.functional.cross_entropy(state_logits, state) - torch.mean(
        torch.sum(soft * nn.functional.log_softmax(direction_logits, dim=1), dim=1)
    )


def _validation_loss(
    model: training.CircularDirectionCNN,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    soft_sigma_deg: float,
) -> float:
    logits = training._collect_logits(model, arrays, device)
    state_loss = nn.functional.cross_entropy(
        torch.from_numpy(logits[0]), torch.from_numpy(logits[2])
    )
    soft = torch.from_numpy(
        training._circular_soft_targets(logits[4], soft_sigma_deg)
    )
    direction_loss = -torch.mean(
        torch.sum(soft * nn.functional.log_softmax(torch.from_numpy(logits[1]), dim=1), dim=1)
    )
    return float(state_loss + direction_loss)


def _fit_seed(
    base_checkpoint: Path,
    arrays: dict[str, tuple[np.ndarray, ...]],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2sp: float,
    strict_weight: float,
    soft_sigma_deg: float,
    patience: int,
    adaptation_mode: str,
) -> tuple[training.CircularDirectionCNN, list[dict], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = training.CircularDirectionCNN().to(device)
    model.load_state_dict(
        torch.load(base_checkpoint, map_location=device, weights_only=True)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.state_head, model.direction_head]
    if adaptation_mode == "tail":
        modules.extend([model.encoder[8], model.shared])
    elif adaptation_mode != "heads":
        raise ValueError(f"unknown adaptation mode: {adaptation_mode}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    model.state_head.train()
    model.direction_head.train()
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    strict_n = len(arrays["strict_train"][0])
    easy_n = len(arrays["easy_train"][0])
    steps = max(math.ceil(strict_n / batch_size), math.ceil(easy_n / batch_size))
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(steps):
            strict_indices = rng.choice(strict_n, size=min(batch_size, strict_n), replace=True)
            easy_indices = rng.choice(easy_n, size=min(batch_size, easy_n), replace=True)
            optimizer.zero_grad()
            strict_loss = _batch_loss(
                model, arrays["strict_train"], strict_indices, device, soft_sigma_deg
            )
            easy_loss = _batch_loss(
                model, arrays["easy_train"], easy_indices, device, soft_sigma_deg
            )
            penalty = sum(
                torch.mean((parameter - initial[name]) ** 2)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            loss = (
                strict_weight * strict_loss
                + (1.0 - strict_weight) * easy_loss
                + l2sp * penalty
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        strict_val = _validation_loss(
            model, arrays["strict_calibration"], device, soft_sigma_deg
        )
        easy_val = _validation_loss(
            model, arrays["easy_calibration"], device, soft_sigma_deg
        )
        score = max(strict_val, easy_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "strict_calibration_loss": strict_val,
                "easy_calibration_loss": easy_val,
                "worst_domain_loss": score,
            }
        )
        if score < best_score - 1.0e-4:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("adaptation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _balanced_indices(first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
    target = max(first, second)
    return np.resize(np.arange(first), target), np.resize(np.arange(second), target)


def _aps_quantile(probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.10) -> float:
    scores = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == int(target))[0])
        scores.append(float(cumulative[rank]))
    level = min(math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def _aps_sizes(probabilities: np.ndarray, quantile: float) -> np.ndarray:
    sizes = []
    for probability in probabilities:
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        sizes.append(
            min(int(np.searchsorted(cumulative, quantile, side="left")) + 1, len(order))
        )
    return np.asarray(sizes, dtype=int)


def _circular_mean(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles = np.deg2rad(
        training._class_angle_deg(np.arange(training.DIRECTION_CLASSES))
    )
    vector = probabilities @ np.exp(1j * angles)
    return np.mod(np.rad2deg(np.angle(vector)), 360.0), np.abs(vector)


def _ensemble_span(per_seed: np.ndarray) -> np.ndarray:
    tops = training._class_angle_deg(np.argmax(per_seed, axis=2))
    spans = []
    for values in tops.T:
        delta = np.abs(
            (values[:, None] - values[None, :] + 180.0) % 360.0 - 180.0
        )
        spans.append(float(np.max(delta)))
    return np.asarray(spans)


def _gate_features(
    probabilities: np.ndarray,
    per_seed: np.ndarray,
    angles: np.ndarray,
    aps_quantile: float,
) -> dict[str, np.ndarray]:
    predicted, concentration = _circular_mean(probabilities)
    return {
        "confidence": np.max(probabilities, axis=1),
        "aps_size": _aps_sizes(probabilities, aps_quantile),
        "concentration": concentration,
        "span": _ensemble_span(per_seed),
        "error": training._circular_error_deg(predicted, angles),
    }


def _mask(features: dict[str, np.ndarray], gate: dict) -> np.ndarray:
    return (
        (features["confidence"] >= gate["min_max_probability"])
        & (features["aps_size"] <= gate["max_aps_set_size"])
        & (features["concentration"] >= gate["min_circular_concentration"])
        & (features["span"] <= gate["max_ensemble_top1_span_deg"] + 1.0e-9)
    )


def _select_worst_domain_gate(
    domains: dict[str, dict[str, np.ndarray]], *, max_aps_size_cap: int
) -> dict:
    all_confidence = np.concatenate([value["confidence"] for value in domains.values()])
    all_concentration = np.concatenate(
        [value["concentration"] for value in domains.values()]
    )
    all_spans = np.concatenate([value["span"] for value in domains.values()])
    all_sizes = np.concatenate([value["aps_size"] for value in domains.values()])
    quantiles = (0.0, 0.25, 0.50, 0.70, 0.80, 0.90)
    confidence_candidates = sorted(
        set([0.0, *np.quantile(all_confidence, quantiles).tolist()])
    )
    concentration_candidates = sorted(
        set([0.0, *np.quantile(all_concentration, quantiles).tolist()])
    )
    span_candidates = sorted(
        set([180.0, *np.quantile(all_spans, quantiles).tolist()])
    )
    size_candidates = sorted(
        size for size in set(map(int, all_sizes)) if size <= max_aps_size_cap
    )
    if not size_candidates:
        size_candidates = [max_aps_size_cap]
    best = None
    for confidence in confidence_candidates:
        for max_size in size_candidates:
            for concentration in concentration_candidates:
                for span in span_candidates:
                    gate = {
                        "min_max_probability": float(confidence),
                        "max_aps_set_size": int(max_size),
                        "min_circular_concentration": float(concentration),
                        "max_ensemble_top1_span_deg": float(span),
                    }
                    domain_rates = []
                    valid = True
                    total = 0
                    for features in domains.values():
                        accepted = _mask(features, gate)
                        count = int(np.sum(accepted))
                        if count < min(12, max(4, len(accepted) // 5)):
                            valid = False
                            break
                        errors = features["error"][accepted]
                        if np.mean(errors <= 7.5 + 1.0e-9) < 0.97 or np.any(errors > 90.0):
                            valid = False
                            break
                        domain_rates.append(float(np.mean(accepted)))
                        total += count
                    if not valid:
                        continue
                    score = (min(domain_rates), total, -max_size)
                    if best is None or score > best[0]:
                        best = (score, gate)
    if best is None:
        return {
            "min_max_probability": 1.0,
            "max_aps_set_size": 0,
            "min_circular_concentration": 1.0,
            "max_ensemble_top1_span_deg": 0.0,
        }
    return best[1]


def _metrics(features: dict[str, np.ndarray], gate: dict, probabilities: np.ndarray, labels: np.ndarray) -> dict:
    accepted = _mask(features, gate)
    errors = features["error"]
    aps_size = features["aps_size"]
    covered = []
    for probability, target, size in zip(probabilities, labels, aps_size):
        covered.append(int(target) in set(np.argsort(-probability)[: int(size)].tolist()))
    return {
        "samples": len(errors),
        "angle_error_median_deg": float(np.median(errors)),
        "angle_error_p90_deg": float(np.quantile(errors, 0.90)),
        "angle_within_7p5_rate": float(np.mean(errors <= 7.5 + 1.0e-9)),
        "aps_coverage": float(np.mean(covered)),
        "aps_mean_set_size": float(np.mean(aps_size)),
        "transport_accept_rate": float(np.mean(accepted)),
        "transport_accepted_count": int(np.sum(accepted)),
        "transport_accepted_within_7p5_rate": (
            float(np.mean(errors[accepted] <= 7.5 + 1.0e-9)) if np.any(accepted) else None
        ),
        "transport_accepted_catastrophic_gt90_count": int(
            np.sum(errors[accepted] > 90.0)
        ),
    }


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    easy_data = Path(args.easy_data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    strict_development = tuple(
        Path(value).resolve() for value in args.strict_development_data
    )
    easy_development = tuple(
        Path(value).resolve() for value in args.easy_development_data
    )
    examples, provenance = _prepare_examples(
        args.split_seed,
        easy_data,
        strict_development,
        easy_development,
        args.development_replay_factor,
    )
    _atomic_json(output / "split_manifest.json", provenance)
    base_contract_path = BASE_ROOT / "ensemble_interface_contract.json"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    per_domain_probabilities: dict[str, list[np.ndarray]] = {
        "strict_calibration": [],
        "easy_calibration": [],
    }
    labels_by_domain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    checkpoint_paths = []
    seed_records = []
    for base_checkpoint_value in base_contract["checkpoints"]:
        base_checkpoint = Path(base_checkpoint_value)
        if not base_checkpoint.is_absolute():
            base_checkpoint = ROOT / base_checkpoint
        seed = int(base_checkpoint.parent.name.rsplit("_", 1)[-1])
        normalization = np.load(base_checkpoint.parent / "normalization.npz")
        context_mean = np.asarray(normalization["context_mean"], dtype=np.float32)
        context_std = np.asarray(normalization["context_std"], dtype=np.float32)
        arrays = _domain_arrays(examples, context_mean, context_std)
        model, history, best_epoch = _fit_seed(
            base_checkpoint,
            arrays,
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2sp=args.l2sp,
            strict_weight=args.strict_weight,
            soft_sigma_deg=args.soft_sigma_deg,
            patience=args.patience,
            adaptation_mode=args.adaptation_mode,
        )
        logits = {
            domain: training._collect_logits(model, arrays[domain], device)
            for domain in ("strict_calibration", "easy_calibration")
        }
        strict_count = len(logits["strict_calibration"][3])
        easy_count = len(logits["easy_calibration"][3])
        strict_index, easy_index = _balanced_indices(strict_count, easy_count)
        balanced_logits = np.concatenate(
            (
                logits["strict_calibration"][1][strict_index],
                logits["easy_calibration"][1][easy_index],
            )
        )
        balanced_labels = np.concatenate(
            (
                logits["strict_calibration"][3][strict_index],
                logits["easy_calibration"][3][easy_index],
            )
        )
        temperature = training._fit_temperature(
            balanced_logits, balanced_labels, device
        )
        for domain in per_domain_probabilities:
            per_domain_probabilities[domain].append(
                _softmax(logits[domain][1], temperature)
            )
            labels_by_domain[domain] = (logits[domain][3], logits[domain][4])
        seed_dir = output / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "circular_cnn_best.pt"
        torch.save(model.state_dict(), checkpoint)
        shutil.copy2(base_checkpoint.parent / "normalization.npz", seed_dir / "normalization.npz")
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        metrics = {
            "stage": "M8_hardware_easy_v2_strict_replay_adaptation",
            "seed": seed,
            "temperature": temperature,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": _sha256(base_checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "best_epoch": best_epoch,
            "history": history,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "encoder_and_shared_frozen": args.adaptation_mode == "heads",
            "adaptation_mode": args.adaptation_mode,
            "strict_replay_weight": args.strict_weight,
            "easy_weight": 1.0 - args.strict_weight,
            "l2sp": args.l2sp,
            "updated_utc": _utc_now(),
        }
        _atomic_json(seed_dir / "metrics.json", metrics)
        checkpoint_paths.append(checkpoint)
        seed_records.append(metrics)

    mean_probabilities = {
        domain: np.mean(np.stack(per_seed, axis=0), axis=0)
        for domain, per_seed in per_domain_probabilities.items()
    }
    strict_count = len(labels_by_domain["strict_calibration"][0])
    easy_count = len(labels_by_domain["easy_calibration"][0])
    strict_index, easy_index = _balanced_indices(strict_count, easy_count)
    balanced_probability = np.concatenate(
        (
            mean_probabilities["strict_calibration"][strict_index],
            mean_probabilities["easy_calibration"][easy_index],
        )
    )
    balanced_labels = np.concatenate(
        (
            labels_by_domain["strict_calibration"][0][strict_index],
            labels_by_domain["easy_calibration"][0][easy_index],
        )
    )
    aps_quantile = _aps_quantile(balanced_probability, balanced_labels)
    features = {
        domain: _gate_features(
            mean_probabilities[domain],
            np.stack(per_domain_probabilities[domain], axis=0),
            labels_by_domain[domain][1],
            aps_quantile,
        )
        for domain in mean_probabilities
    }
    inherited_aps_cap = int(
        base_contract.get("structured_transport_gate", {}).get(
            "max_aps_set_size", training.DIRECTION_CLASSES
        )
    )
    gate = _select_worst_domain_gate(
        features, max_aps_size_cap=inherited_aps_cap
    )
    domain_metrics = {
        domain: _metrics(
            features[domain],
            gate,
            mean_probabilities[domain],
            labels_by_domain[domain][0],
        )
        for domain in mean_probabilities
    }
    contract = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "combiner": "mean calibrated class probabilities",
        "confidence_gate": gate["min_max_probability"],
        "aps_quantile": aps_quantile,
        "direction_bins": training.DIRECTION_CLASSES,
        "direction_bin_width_deg": training.DIRECTION_BIN_DEG,
        "structured_transport_gate": gate,
        "state_role": base_contract["state_role"],
        "direction_role": base_contract["direction_role"],
        "fallback": base_contract["fallback"],
        "adaptation_contract": {
            "geometry_profile": "hardware_easy_v2",
            "strict_replay": True,
            "domain_weights": {
                "strict": args.strict_weight,
                "hardware_easy_v2": 1.0 - args.strict_weight,
            },
            "trainable_modules": (
                ["state_head", "direction_head"]
                if args.adaptation_mode == "heads"
                else ["encoder.last_conv", "shared", "state_head", "direction_head"]
            ),
            "encoder_and_shared_frozen": args.adaptation_mode == "heads",
            "batchnorm_running_statistics_frozen": True,
            "captured_or_incomplete_traces_used_for_direction_supervision": 0,
            "final_promotion_audit_samples_seen_during_fitting": 0,
            "previously_untouched_development_datasets_consumed": [
                str(path) for path in (*strict_development, *easy_development)
            ],
            "max_aps_set_size_cap_inherited_from_base": inherited_aps_cap,
        },
        "claim_boundary": (
            "Simulation-only hardware-easy-v2 contact-geometry adaptation with frozen strict replay. "
            "It is not real-hardware fine-tuning, calibrated Bota sensing, FDM compliance, or full-loop evidence."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "ensemble_interface_contract.json", contract)
    report = {
        "stage": "M8_hardware_easy_v2_strict_replay_adaptation",
        "device": str(device),
        "base_model_root": str(BASE_ROOT),
        "base_contract_sha256": _sha256(base_contract_path),
        "easy_dataset": str(easy_data),
        "easy_sampling_contract_sha256": _sha256(easy_data / "sampling_contract.json"),
        "split_manifest_sha256": _sha256(output / "split_manifest.json"),
        "seeds": seed_records,
        "aps_quantile": aps_quantile,
        "structured_transport_gate": gate,
        "calibration_metrics": domain_metrics,
        "test_samples_seen": 0,
        "previously_untouched_development_datasets_consumed": [
            str(path) for path in (*strict_development, *easy_development)
        ],
        "claim_boundary": contract["claim_boundary"],
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "adaptation_report.json", report)
    lines = [
        "# M8 hardware-easy-v2 strict-replay adaptation",
        "",
        (
            "Only the state and direction heads were adapted."
            if args.adaptation_mode == "heads"
            else "The last convolution, shared block, and two heads were adapted at low learning rate."
        ),
        "Batch-normalization statistics and per-seed normalization remain frozen from P1v3.",
        "Captured or incomplete traces were excluded from direction supervision.",
        "",
        "| calibration domain | samples | median / p90 | within 7.5 deg | APS coverage / size | gate accept / accepted accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, item in domain_metrics.items():
        lines.append(
            f"| {domain} | {item['samples']} | {item['angle_error_median_deg']:.2f} / "
            f"{item['angle_error_p90_deg']:.2f} deg | {item['angle_within_7p5_rate']:.3f} | "
            f"{item['aps_coverage']:.3f} / {item['aps_mean_set_size']:.2f} | "
            f"{item['transport_accept_rate']:.3f} / "
            f"{item['transport_accepted_within_7p5_rate'] if item['transport_accepted_within_7p5_rate'] is not None else 'n/a'} |"
        )
    lines.extend(["", "Boundary: " + contract["claim_boundary"], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "calibration": domain_metrics}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-data", default=str(EASY_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--l2sp", type=float, default=1.0e-2)
    parser.add_argument("--strict-weight", type=float, default=0.30)
    parser.add_argument("--soft-sigma-deg", type=float, default=6.0)
    parser.add_argument("--split-seed", type=int, default=20260721)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--adaptation-mode", choices=("heads", "tail"), default="heads")
    parser.add_argument("--strict-development-data", action="append", default=[])
    parser.add_argument("--easy-development-data", action="append", default=[])
    parser.add_argument("--development-replay-factor", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 < args.strict_weight < 1.0:
        parser.error("--strict-weight must be in (0, 1)")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("epoch, patience, and batch size must be positive")
    if args.development_replay_factor <= 0:
        parser.error("--development-replay-factor must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
