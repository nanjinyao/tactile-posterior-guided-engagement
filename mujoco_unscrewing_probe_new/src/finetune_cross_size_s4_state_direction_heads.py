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

from src.m8_direction_runtime import M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs/current/datasets/cross_size_s4_adaptation"
DEFAULT_OUTPUT = ROOT / "outputs/current/models/cross_size_s4_state_direction_heads"
BASE_MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SIZES = ("m10", "m12")
FROZEN_PREFIXES = ("encoder.", "shared.")
TRAINABLE_PREFIXES = ("state_head.", "direction_head.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(
    state: dict[str, torch.Tensor], prefixes: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if not key.startswith(prefixes):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_variant_checkpoint(
    variant: str,
    base_checkpoint: Path,
    checkpoint: Path,
    model: training.CircularDirectionCNN,
) -> None:
    """Write a variant checkpoint while making cal-only byte identity explicit."""

    if variant == "cal_only":
        shutil.copy2(base_checkpoint, checkpoint)
    elif variant == "fine_tuned":
        torch.save(model.state_dict(), checkpoint)
    else:
        raise ValueError(variant)


def _load_contract(data_root: Path, size: str) -> dict:
    contract_path = data_root / size / "dataset_contract.json"
    manifest_path = data_root / size / "dataset_manifest.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(
            f"{size}: S4 data are {manifest.get('status')!r}; run the opt-in sampler first"
        )
    if manifest["dataset_contract_sha256"] != _sha256(contract_path):
        raise ValueError(f"{size}: S4 dataset contract hash drift")
    intersections = contract.get("pairwise_orbit_intersections", {})
    if any(intersections.values()):
        raise ValueError(f"{size}: S4 C6 orbit leakage")
    if set(contract["tasks"]) != {"train", "validation", "calibration"}:
        raise ValueError(f"{size}: incomplete S4 data roles")
    return contract


def _load_role_examples(
    data_root: Path, size: str, contract: dict, role: str
) -> dict:
    spec = json.loads(
        (ROOT / "configs" / f"allen_{size}_source.json").read_text(
            encoding="utf-8"
        )
    )
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    tasks = {task["trial_id"]: task for task in contract["tasks"][role]}
    sequences: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    state_labels: list[int] = []
    direction_labels: list[int] = []
    angles: list[float] = []
    trial_ids: list[str] = []
    orbit_ids: list[str] = []
    for trial_id, task in sorted(tasks.items()):
        path = data_root / size / "trials" / f"{trial_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            phase_sequence = np.asarray(data["phase_sequence"], dtype=float)
            columns = tuple(map(str, data["phase_columns"].tolist()))
        if metadata.get("status") != "ok":
            raise ValueError(f"{trial_id}: non-ok S4 scan")
        if metadata.get("wrench_observation") == "ideal":
            raise ValueError(f"{trial_id}: S4 scan omitted its observation-chain prior")
        raw = np.vstack(
            [
                phase_sequence[columns.index(name)]
                for name in training.INPUT_CHANNELS
            ]
        )
        shape, context = M8DirectionEnsemble._network_inputs(
            raw, float(metadata["target_down_force_n"]), bit_radius_m
        )
        state = str(metadata["search_state"])
        if state not in training.STATE_TO_INDEX:
            raise ValueError(f"{trial_id}: unknown contact state {state}")
        angle = float(training._direction_angle_deg(metadata, 0))
        sequences.append(shape)
        contexts.append(context)
        state_labels.append(training.STATE_TO_INDEX[state])
        direction_labels.append(training._direction_class(angle))
        angles.append(angle)
        trial_ids.append(trial_id)
        orbit_ids.append(str(task["symmetry_orbit_id"]))
    return {
        "sequence": np.stack(sequences).astype(np.float32),
        "context": np.stack(contexts).astype(np.float32),
        "state": np.asarray(state_labels, dtype=np.int64),
        "direction": np.asarray(direction_labels, dtype=np.int64),
        "angle_deg": np.asarray(angles, dtype=np.float32),
        "trial_ids": trial_ids,
        "orbit_ids": orbit_ids,
        "bit_radius_m": bit_radius_m,
    }


def configure_trainable_heads(model: training.CircularDirectionCNN) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    return tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def _hidden(
    model: training.CircularDirectionCNN,
    examples: dict,
    context_mean: np.ndarray,
    context_std: np.ndarray,
) -> torch.Tensor:
    sequence = torch.from_numpy(examples["sequence"])
    normalized_context = (
        examples["context"] - context_mean[None, :]
    ) / context_std[None, :]
    context = torch.from_numpy(normalized_context.astype(np.float32))
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(sequence).flatten(1)
        hidden = model.shared(torch.cat((encoded, context), dim=1))
    return hidden.detach()


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probability = np.exp(scaled)
    return probability / np.sum(probability, axis=1, keepdims=True)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            logits_t / torch.exp(log_temperature), labels_t
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(0.25, 10.0))


def _aps_quantile(
    probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.10
) -> float:
    scores = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == int(target))[0])
        scores.append(float(cumulative[rank]))
    level = min(
        math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores), 1.0
    )
    return float(np.quantile(scores, level, method="higher"))


def probability_metrics(
    state_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    examples: dict,
    *,
    member_direction_probabilities: np.ndarray | None = None,
    aps_quantile: float | None = None,
) -> dict:
    state_probability = np.asarray(state_probabilities, dtype=float)
    direction_probability = np.asarray(direction_probabilities, dtype=float)
    states = np.asarray(examples["state"], dtype=int)
    angles = np.asarray(examples["angle_deg"], dtype=float)
    if state_probability.shape != (len(states), len(training.STATE_NAMES)):
        raise ValueError("state probability shape mismatch")
    if direction_probability.shape != (
        len(states),
        training.DIRECTION_CLASSES,
    ):
        raise ValueError("direction probability shape mismatch")

    state_pred = np.argmax(state_probability, axis=1)
    top_class = np.argmax(direction_probability, axis=1)
    top_deg = training._class_angle_deg(top_class)
    top_error = training._circular_error_deg(top_deg, angles)
    class_angles_rad = np.deg2rad(
        training._class_angle_deg(np.arange(training.DIRECTION_CLASSES))
    )
    posterior_vector = np.sum(
        direction_probability * np.exp(1j * class_angles_rad)[None, :], axis=1
    )
    concentration = np.abs(posterior_vector)
    control_deg = np.mod(np.rad2deg(np.angle(posterior_vector)), 360.0)
    control_error = training._circular_error_deg(control_deg, angles)

    ensemble_span = None
    if member_direction_probabilities is not None:
        members = np.asarray(member_direction_probabilities, dtype=float)
        if members.ndim != 3 or members.shape[1:] != direction_probability.shape:
            raise ValueError("member direction probability shape mismatch")
        member_top = training._class_angle_deg(np.argmax(members, axis=2))
        differences = np.abs(
            (member_top[:, None, :] - member_top[None, :, :] + 180.0)
            % 360.0
            - 180.0
        )
        ensemble_span = np.max(differences, axis=(0, 1))

    def summarize(mask: np.ndarray) -> dict:
        result = {
            "samples": int(np.sum(mask)),
            "state_accuracy": float(np.mean(state_pred[mask] == states[mask])),
            "top1_error_median_deg": float(np.median(top_error[mask])),
            "top1_error_p90_deg": float(np.quantile(top_error[mask], 0.90)),
            "top1_error_gt90_count": int(np.sum(top_error[mask] > 90.0)),
            "control_direction_error_median_deg": float(
                np.median(control_error[mask])
            ),
            "control_direction_error_p90_deg": float(
                np.quantile(control_error[mask], 0.90)
            ),
            "control_direction_error_gt90_count": int(
                np.sum(control_error[mask] > 90.0)
            ),
            "circular_concentration_median": float(
                np.median(concentration[mask])
            ),
        }
        if ensemble_span is not None:
            result["ensemble_top1_span_median_deg"] = float(
                np.median(ensemble_span[mask])
            )
            result["ensemble_top1_span_p90_deg"] = float(
                np.quantile(ensemble_span[mask], 0.90)
            )
        return result

    all_mask = np.ones(len(states), dtype=bool)
    result = summarize(all_mask)
    result["per_state"] = {}
    for state_index, state_name in enumerate(training.STATE_NAMES):
        mask = states == state_index
        if np.any(mask):
            result["per_state"][state_name] = summarize(mask)
    result["worst_state_top1_error_p90_deg"] = max(
        item["top1_error_p90_deg"] for item in result["per_state"].values()
    )
    result["worst_state_control_direction_error_p90_deg"] = max(
        item["control_direction_error_p90_deg"]
        for item in result["per_state"].values()
    )
    result["worst_state_control_direction_error_gt90_count"] = max(
        item["control_direction_error_gt90_count"]
        for item in result["per_state"].values()
    )
    if aps_quantile is not None:
        covered = []
        sizes = []
        labels = np.asarray(examples["direction"], dtype=int)
        for probability, target in zip(direction_probability, labels):
            order = np.argsort(-probability)
            cumulative = np.cumsum(probability[order])
            size = min(
                int(np.searchsorted(cumulative, aps_quantile, side="left")) + 1,
                len(order),
            )
            covered.append(int(target) in set(map(int, order[:size])))
            sizes.append(size)
        result["aps_coverage"] = float(np.mean(covered))
        result["aps_mean_set_size"] = float(np.mean(sizes))
    return result


def _head_logits(
    model: training.CircularDirectionCNN, hidden: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        return (
            model.state_head(hidden).cpu().numpy(),
            model.direction_head(hidden).cpu().numpy(),
        )


def _selection_metrics(
    model: training.CircularDirectionCNN,
    hidden: torch.Tensor,
    examples: dict,
) -> dict:
    state_logits, direction_logits = _head_logits(model, hidden)
    return probability_metrics(
        _softmax(state_logits), _softmax(direction_logits), examples
    )


def _selection_key(metrics: dict, validation_loss: float) -> tuple[float, ...]:
    # Catastrophic control-direction errors dominate; then protect the worst
    # contact state instead of selecting on a pooled median that can hide the
    # double-overhang failures observed in S3.
    return (
        float(metrics["control_direction_error_gt90_count"]),
        float(metrics["worst_state_control_direction_error_p90_deg"]),
        float(1.0 - metrics["state_accuracy"]),
        float(validation_loss),
    )


def _fit_heads(
    model: training.CircularDirectionCNN,
    train: dict,
    validation: dict,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2sp: float,
    state_weight: float,
    soft_sigma_deg: float,
) -> tuple[list[dict], dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainable_names = configure_trainable_heads(model)
    train_hidden = _hidden(model, train, context_mean, context_std)
    validation_hidden = _hidden(model, validation, context_mean, context_std)
    state_target = torch.from_numpy(train["state"])
    direction_target = torch.from_numpy(
        training._circular_soft_targets(train["angle_deg"], soft_sigma_deg)
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(TRAINABLE_PREFIXES)
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    history: list[dict] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_heads: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        model.state_head.train()
        model.direction_head.train()
        optimizer.zero_grad()
        state_logits = model.state_head(train_hidden)
        direction_logits = model.direction_head(train_hidden)
        state_loss = nn.functional.cross_entropy(state_logits, state_target)
        direction_loss = -torch.mean(
            torch.sum(
                direction_target
                * nn.functional.log_softmax(direction_logits, dim=1),
                dim=1,
            )
        )
        penalty = sum(
            torch.sum((parameter - initial[name]) ** 2)
            for name, parameter in model.named_parameters()
            if name in initial
        )
        loss = state_weight * state_loss + direction_loss + l2sp * penalty
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_state = model.state_head(validation_hidden)
            val_direction = model.direction_head(validation_hidden)
            validation_loss = float(
                state_weight
                * nn.functional.cross_entropy(
                    val_state, torch.from_numpy(validation["state"])
                )
                - torch.mean(
                    torch.sum(
                        torch.from_numpy(
                            training._circular_soft_targets(
                                validation["angle_deg"], soft_sigma_deg
                            )
                        )
                        * nn.functional.log_softmax(val_direction, dim=1),
                        dim=1,
                    )
                )
            )
        metrics = _selection_metrics(model, validation_hidden, validation)
        key = _selection_key(metrics, validation_loss)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_heads = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if name.startswith(TRAINABLE_PREFIXES)
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "state_loss": float(state_loss.detach()),
                    "direction_loss": float(direction_loss.detach()),
                    "l2sp_penalty": float(penalty.detach()),
                    "validation_loss": validation_loss,
                    "validation_state_accuracy": metrics["state_accuracy"],
                    "validation_control_gt90_count": metrics[
                        "control_direction_error_gt90_count"
                    ],
                    "validation_worst_state_control_p90_deg": metrics[
                        "worst_state_control_direction_error_p90_deg"
                    ],
                    "is_selected_so_far": epoch == best_epoch,
                }
            )
    if not best_heads or best_key is None:
        raise RuntimeError("S4 head fitting failed to select an epoch")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_heads:
                parameter.copy_(best_heads[name])
    model.eval()
    return history, {
        "selection_rule": (
            "lexicographic(control_gt90_count, worst_state_control_p90, "
            "1-state_accuracy, validation_loss)"
        ),
        "selected_epoch": best_epoch,
        "selection_key": list(best_key),
        "trainable_parameter_names": list(trainable_names),
    }


def _build_variant(
    size: str,
    variant: str,
    examples: dict[str, dict],
    output: Path,
    base_contract: dict,
    args: argparse.Namespace,
) -> dict:
    if variant not in {"cal_only", "fine_tuned"}:
        raise ValueError(variant)
    size_root = output / size / variant
    seed_records = []
    probabilities: dict[str, dict[str, list[np.ndarray]]] = {
        role: {"state": [], "direction": []}
        for role in ("train", "validation", "calibration")
    }
    member_probabilities: dict[str, list[np.ndarray]] = {
        role: [] for role in probabilities
    }
    for checkpoint_value in base_contract["checkpoints"]:
        base_checkpoint = _checkpoint_path(checkpoint_value)
        seed_name = base_checkpoint.parent.name
        seed = int(seed_name.rsplit("_", 1)[1])
        normalization_path = base_checkpoint.parent / "normalization.npz"
        normalization = np.load(normalization_path)
        context_mean = np.asarray(normalization["context_mean"], dtype=np.float32)
        context_std = np.asarray(normalization["context_std"], dtype=np.float32)
        base_state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        model = training.CircularDirectionCNN()
        model.load_state_dict(base_state)
        if variant == "fine_tuned":
            history, selection = _fit_heads(
                model,
                examples["train"],
                examples["validation"],
                context_mean,
                context_std,
                seed=seed,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2sp=args.l2sp,
                state_weight=args.state_weight,
                soft_sigma_deg=args.soft_sigma_deg,
            )
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            history = []
            selection = {
                "selection_rule": "none; base checkpoint weights remain byte-identical",
                "selected_epoch": None,
                "selection_key": None,
                "trainable_parameter_names": [],
            }
        trained_state = model.state_dict()
        frozen_exact = all(
            torch.equal(base_state[key], trained_state[key])
            for key in base_state
            if key.startswith(FROZEN_PREFIXES)
        )
        heads_changed = {
            prefix.removesuffix("."): any(
                not torch.equal(base_state[key], trained_state[key])
                for key in base_state
                if key.startswith(prefix)
            )
            for prefix in TRAINABLE_PREFIXES
        }
        hidden = {
            role: _hidden(model, role_examples, context_mean, context_std)
            for role, role_examples in examples.items()
        }
        logits = {
            role: _head_logits(model, role_hidden)
            for role, role_hidden in hidden.items()
        }
        direction_temperature = _fit_temperature(
            logits["calibration"][1], examples["calibration"]["direction"]
        )
        state_temperature = _fit_temperature(
            logits["calibration"][0], examples["calibration"]["state"]
        )
        for role in probabilities:
            probabilities[role]["state"].append(
                _softmax(logits[role][0], state_temperature)
            )
            direction_probability = _softmax(
                logits[role][1], direction_temperature
            )
            probabilities[role]["direction"].append(direction_probability)
            member_probabilities[role].append(direction_probability)

        seed_dir = size_root / "seeds" / seed_name
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "circular_cnn_best.pt"
        _write_variant_checkpoint(variant, base_checkpoint, checkpoint, model)
        shutil.copy2(normalization_path, seed_dir / "normalization.npz")
        record = {
            "stage": "cross_size_S4_state_direction_head_adaptation",
            "size": size,
            "variant": variant,
            "seed": seed,
            "base_checkpoint": _relative(base_checkpoint),
            "base_checkpoint_sha256": _sha256(base_checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "encoder_shared_frozen_exact": frozen_exact,
            "heads_changed": heads_changed,
            "frozen_state_hash_before": _state_hash(
                base_state, FROZEN_PREFIXES
            ),
            "frozen_state_hash_after": _state_hash(
                trained_state, FROZEN_PREFIXES
            ),
            "head_state_hash_before": _state_hash(
                base_state, TRAINABLE_PREFIXES
            ),
            "head_state_hash_after": _state_hash(
                trained_state, TRAINABLE_PREFIXES
            ),
            "trainable_parameter_names": selection["trainable_parameter_names"],
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "direction_temperature": direction_temperature,
            # Runtime currently reads `temperature` for the direction head.
            "temperature": direction_temperature,
            "state_temperature": state_temperature,
            "weight_update_role": "train" if variant == "fine_tuned" else "none",
            "model_selection_role": (
                "validation" if variant == "fine_tuned" else "none"
            ),
            "temperature_APS_role": "calibration",
            "checkpoint_byte_identical_to_base": (
                _sha256(checkpoint) == _sha256(base_checkpoint)
            ),
            "selection": selection,
            "training_hyperparameters": {
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "l2sp": float(args.l2sp),
                "state_weight": float(args.state_weight),
                "soft_sigma_deg": float(args.soft_sigma_deg),
            },
            "history": history,
            "updated_utc": _utc_now(),
        }
        _atomic_json(seed_dir / "metrics.json", record)
        _atomic_json(
            seed_dir / "interface_contract.json",
            {
                "input": {
                    "phase_bins": 360,
                    "sequence_channels": list(training.INPUT_CHANNELS),
                    "dimensionless_context_features": 16,
                    "normalization_source": _relative(normalization_path),
                },
                "adaptation": {
                    "variant": variant,
                    "trainable_modules": (
                        ["state_head", "direction_head"]
                        if variant == "fine_tuned"
                        else []
                    ),
                    "encoder_shared_frozen": True,
                    "all_checkpoint_weights_frozen": variant == "cal_only",
                    "weight_update_role": (
                        "train" if variant == "fine_tuned" else "none"
                    ),
                    "model_selection_role": (
                        "validation" if variant == "fine_tuned" else "none"
                    ),
                    "temperature_APS_role": "calibration",
                },
                "updated_utc": _utc_now(),
            },
        )
        seed_records.append(record)

    calibration_direction_mean = np.mean(
        probabilities["calibration"]["direction"], axis=0
    )
    aps_quantile = _aps_quantile(
        calibration_direction_mean, examples["calibration"]["direction"]
    )
    role_metrics = {}
    for role in probabilities:
        state_mean = np.mean(probabilities[role]["state"], axis=0)
        direction_mean = np.mean(probabilities[role]["direction"], axis=0)
        role_metrics[role] = probability_metrics(
            state_mean,
            direction_mean,
            examples[role],
            member_direction_probabilities=np.stack(member_probabilities[role]),
            aps_quantile=aps_quantile if role == "calibration" else None,
        )

    checkpoints = [
        size_root
        / "seeds"
        / Path(value).parent.name
        / "circular_cnn_best.pt"
        for value in base_contract["checkpoints"]
    ]
    ensemble_contract = {
        "checkpoints": [_relative(path) for path in checkpoints],
        "combiner": "mean size-calibrated class probabilities",
        "confidence_gate": float(base_contract["confidence_gate"]),
        "aps_quantile": aps_quantile,
        "direction_bins": training.DIRECTION_CLASSES,
        "direction_bin_width_deg": training.DIRECTION_BIN_DEG,
        "structured_transport_gate": base_contract["structured_transport_gate"],
        "state_role": (
            "size-adapted diagnostic soft CAD likelihood; never a terminal verifier"
        ),
        "direction_role": base_contract["direction_role"],
        "fallback": base_contract["fallback"],
        "adaptation_contract": {
            "size": size,
            "variant": variant,
            "encoder_shared_frozen": True,
            "trainable_modules": (
                ["state_head", "direction_head"]
                if variant == "fine_tuned"
                else []
            ),
            "all_checkpoint_weights_frozen": variant == "cal_only",
            "weight_update_role": (
                "train" if variant == "fine_tuned" else "none"
            ),
            "model_selection_role": (
                "validation" if variant == "fine_tuned" else "none"
            ),
            "temperature_and_APS_role": "calibration",
            "S3_full_loop_samples_seen": 0,
            "future_S4_full_loop_samples_seen": 0,
        },
        "claim_boundary": (
            "Simulation-only S4 observed-wrench head adaptation. Head metrics do "
            "not establish full-loop success."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(size_root / "ensemble_interface_contract.json", ensemble_contract)
    report = {
        "size": size,
        "variant": variant,
        "seeds": seed_records,
        "all_encoder_shared_frozen_exact": all(
            record["encoder_shared_frozen_exact"] for record in seed_records
        ),
        "all_state_heads_changed": all(
            record["heads_changed"]["state_head"] for record in seed_records
        ),
        "all_direction_heads_changed": all(
            record["heads_changed"]["direction_head"] for record in seed_records
        ),
        "all_cal_only_checkpoints_equal_base": all(
            record["checkpoint_byte_identical_to_base"] for record in seed_records
        ),
        "aps_quantile": aps_quantile,
        "role_metrics": role_metrics,
        "ensemble_contract_sha256": _sha256(
            size_root / "ensemble_interface_contract.json"
        ),
        "claim_boundary": (
            "Per-state and circular-control metrics are train/validation/calibration "
            "diagnostics only; a new untouched full-loop set is still required."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(size_root / "adaptation_report.json", report)
    return report


def _write_summary(output: Path, report: dict) -> None:
    lines = [
        "# Cross-size S4 state + direction head adaptation",
        "",
        "The calibration-only variant preserves every base checkpoint byte and fits",
        "only state/direction temperature plus APS on calibration. The fine-tuned",
        "variant freezes the encoder/shared trunk; complete C6 orbits are split into",
        "train, validation and calibration before sampling. Model selection uses",
        "catastrophic circular-control error and worst-state p90, not a pooled top-1",
        "median. These are model diagnostics, not full-loop evidence.",
        "",
        "| size | variant | role | state acc | MAP med/p90 | control med/p90 | control >90 | worst-state control p90 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for size in report["sizes"]:
        for variant in ("cal_only", "fine_tuned"):
            for role in ("train", "validation", "calibration"):
                metrics = report["sizes"][size][variant]["role_metrics"][role]
                lines.append(
                    f"| {size.upper()} | {variant} | {role} | "
                    f"{metrics['state_accuracy']:.3f} | "
                    f"{metrics['top1_error_median_deg']:.2f}/{metrics['top1_error_p90_deg']:.2f} deg | "
                    f"{metrics['control_direction_error_median_deg']:.2f}/"
                    f"{metrics['control_direction_error_p90_deg']:.2f} deg | "
                    f"{metrics['control_direction_error_gt90_count']} | "
                    f"{metrics['worst_state_control_direction_error_p90_deg']:.2f} deg |"
                )
    lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_contract_path = BASE_MODEL_ROOT / "ensemble_interface_contract.json"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    size_reports = {}
    dataset_contracts = {}
    for size in args.sizes:
        contract = _load_contract(data_root, size)
        examples = {
            role: _load_role_examples(data_root, size, contract, role)
            for role in ("train", "validation", "calibration")
        }
        size_reports[size] = {
            variant: _build_variant(
                size, variant, examples, output, base_contract, args
            )
            for variant in ("cal_only", "fine_tuned")
        }
        dataset_contracts[size] = {
            "path": _relative(data_root / size / "dataset_contract.json"),
            "sha256": _sha256(data_root / size / "dataset_contract.json"),
        }
        _atomic_json(
            output / size / "model_manifest.json",
            {
                "status": "complete",
                "size": size,
                "base_model_contract": _relative(base_contract_path),
                "base_model_contract_sha256": _sha256(base_contract_path),
                "dataset_contract": dataset_contracts[size],
                "variants": {
                    variant: {
                        "root": _relative(output / size / variant),
                        "report_sha256": _sha256(
                            output / size / variant / "adaptation_report.json"
                        ),
                        "ensemble_contract_sha256": _sha256(
                            output
                            / size
                            / variant
                            / "ensemble_interface_contract.json"
                        ),
                    }
                    for variant in ("cal_only", "fine_tuned")
                },
                "S3_full_loop_samples_used": 0,
                "future_S4_full_loop_samples_used": 0,
                "updated_utc": _utc_now(),
            },
        )
    report = {
        "stage": "cross_size_S4_state_direction_head_adaptation",
        "sizes": size_reports,
        "base_model_root": _relative(BASE_MODEL_ROOT),
        "base_model_contract_sha256": _sha256(base_contract_path),
        "dataset_contracts": dataset_contracts,
        "frozen_modules": ["encoder", "shared"],
        "trainable_modules": ["state_head", "direction_head"],
        "cal_only_weight_updates": False,
        "cal_only_temperature_APS_role": "calibration",
        "selection_role": "validation",
        "temperature_APS_role": "calibration",
        "S3_full_loop_samples_used": 0,
        "future_S4_full_loop_samples_used": 0,
        "training_hyperparameters": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "l2sp": float(args.l2sp),
            "state_weight": float(args.state_weight),
            "soft_sigma_deg": float(args.soft_sigma_deg),
        },
        "hyperparameter_selection_contract": {
            "selection_role": "validation only",
            "calibration_role_used_for_selection": False,
            "full_loop_rollouts_used_for_selection": 0,
            "diagnostic_root": _relative(
                Path(
                    getattr(
                        args,
                        "selection_diagnostic_root",
                        "outputs/current/diagnostics/s4_head_hyperparam_sweep",
                    )
                ).resolve()
            ),
            "selection_rule": (
                "lexicographic(sum validation control >90 counts across M10/M12, "
                "maximum validation worst-state control p90, one minus minimum "
                "validation state accuracy, mean validation control p90)"
            ),
        },
        "claim_boundary": (
            "This stage can establish only held-out head calibration under the S4 "
            "simulation observation prior. It cannot claim an S4 full-loop pass or "
            "real-robot fine-tuning necessity."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "sizes": list(args.sizes),
            "report_sha256": _sha256(output / "report.json"),
            "summary_sha256": _sha256(output / "summary.md"),
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a weight-identical calibration-only S4 control and fine-tune "
            "only the state/direction heads with validation-selected epochs; "
            "temperature/APS use calibration only."
        )
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--l2sp", type=float, default=1.0e-2)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--soft-sigma-deg", type=float, default=6.0)
    parser.add_argument(
        "--selection-diagnostic-root",
        default="outputs/current/diagnostics/s4_head_hyperparam_sweep",
        help=(
            "validation-only sweep directory recorded in the model provenance; "
            "this option does not read or select from the directory"
        ),
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.learning_rate <= 0.0
        or args.l2sp < 0.0
        or args.state_weight <= 0.0
        or args.soft_sigma_deg < 0.0
    ):
        parser.error("invalid S4 adaptation hyperparameter")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
