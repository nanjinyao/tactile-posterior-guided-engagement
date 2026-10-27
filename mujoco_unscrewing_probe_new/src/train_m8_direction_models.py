from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.sample_m8_conical_pilot import (
    PHASE_COLUMNS,
    _fingerprint_features,
    _load_trial,
    _rotate_xy,
    _smoothed_standardized,
)


ROOT = Path(__file__).resolve().parents[1]
P0A = ROOT / "outputs" / "current" / "datasets" / "p0a_m8_symmetry"
P0B = ROOT / "outputs" / "current" / "datasets" / "p0b_m8_observability"
P0C = ROOT / "outputs" / "current" / "datasets" / "p0c_m8_collision_followup"
P0D = ROOT / "outputs" / "current" / "datasets" / "p0d_m8_sector_anchors"
P1V3_TARGETED = ROOT / "outputs" / "current" / "datasets" / "p1v3_m8_targeted_direction_training"
DEFAULT_OUT = ROOT / "outputs" / "current" / "models" / "p1_m8_direction_models"
SPEC = json.loads((ROOT / "configs" / "allen_m8_source.json").read_text(encoding="utf-8"))
BIT_RADIUS_M = float(SPEC["key_af_m"]) / math.sqrt(3.0)

DIRECTION_BIN_DEG = 7.5
DIRECTION_OFFSET_DEG = 3.75
DIRECTION_CLASSES = 48
STATE_NAMES = ("supported_search", "double_overhang", "outer_partial")
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}
INPUT_CHANNELS = ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")


@dataclass(frozen=True)
class LoadedTrial:
    metadata: dict
    sequence: np.ndarray
    columns: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_trials(directory: Path) -> list[LoadedTrial]:
    trials = []
    for path in sorted((directory / "trials").glob("*.npz")):
        metadata, sequence, columns = _load_trial(path)
        if metadata.get("status") == "ok":
            trials.append(LoadedTrial(metadata, sequence, columns))
    return trials


def grouped_split(trials: list[LoadedTrial], seed: int = 20260710) -> dict[str, list[str]]:
    groups_by_state: dict[str, list[str]] = {state: [] for state in STATE_NAMES}
    for trial in trials:
        group = trial.metadata["base_pose_id"]
        state = trial.metadata["search_state"]
        if group not in groups_by_state[state]:
            groups_by_state[state].append(group)
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for state, groups in groups_by_state.items():
        ordered = sorted(groups)
        rng.shuffle(ordered)
        if len(ordered) != 40:
            raise ValueError(f"expected 40 P0b groups for {state}, got {len(ordered)}")
        split["val"].extend(ordered[:8])
        split["test"].extend(ordered[8:16])
        split["train"].extend(ordered[16:])
    return split


def _sector_pose_id(metadata: dict) -> str:
    return str(metadata["symmetry_orbit_id"]).rsplit("_n", 1)[0]


def grouped_sector_split(
    trials: list[LoadedTrial], seed: int = 20260710
) -> dict[str, list[str]]:
    groups_by_state: dict[str, list[str]] = {state: [] for state in STATE_NAMES}
    for trial in trials:
        group = _sector_pose_id(trial.metadata)
        state = trial.metadata["search_state"]
        if group not in groups_by_state[state]:
            groups_by_state[state].append(group)
    rng = np.random.default_rng(seed + 1)
    split = {"train": [], "val": [], "test": []}
    for state, groups in groups_by_state.items():
        ordered = sorted(groups)
        rng.shuffle(ordered)
        if len(ordered) != 9:
            raise ValueError(f"expected 9 P0d spatial groups for {state}, got {len(ordered)}")
        split["val"].extend(ordered[:2])
        split["test"].extend(ordered[2:4])
        split["train"].extend(ordered[4:])
    return split


def grouped_targeted_split(
    trials: list[LoadedTrial], seed: int = 20260710
) -> dict[str, list[str]]:
    groups_by_state: dict[str, list[str]] = {}
    for trial in trials:
        state = trial.metadata["search_state"]
        groups_by_state.setdefault(state, [])
        group = trial.metadata["base_pose_id"]
        if group not in groups_by_state[state]:
            groups_by_state[state].append(group)
    rng = np.random.default_rng(seed + 2)
    split = {"train": [], "val": []}
    for state, groups in sorted(groups_by_state.items()):
        ordered = sorted(groups)
        rng.shuffle(ordered)
        val_count = max(1, int(round(len(ordered) / 6.0)))
        split["val"].extend(ordered[:val_count])
        split["train"].extend(ordered[val_count:])
    return split


def _augment_sequence(
    sequence: np.ndarray, columns: tuple[str, ...], rotation_index: int
) -> np.ndarray:
    transformed = np.asarray(sequence, dtype=float).copy()
    angle = rotation_index * math.pi / 3.0
    for x_name, y_name in (("Fx_N", "Fy_N"), ("Tx_Nm", "Ty_Nm"), ("tip_x_m", "tip_y_m")):
        ix, iy = columns.index(x_name), columns.index(y_name)
        transformed[[ix, iy]] = _rotate_xy(transformed[[ix, iy]], angle)
    shift = int(round(rotation_index * transformed.shape[1] / 6.0))
    return np.roll(transformed, -shift, axis=1)


def _direction_angle_deg(metadata: dict, rotation_index: int = 0) -> float:
    physical_rotation_deg = float(metadata.get("rotation_deg", 0.0))
    return (
        float(metadata["base_angle_deg"])
        + physical_rotation_deg
        + 180.0
        + 60.0 * rotation_index
    ) % 360.0


def _direction_class(angle_deg: float) -> int:
    return int(round(((angle_deg - DIRECTION_OFFSET_DEG) % 360.0) / DIRECTION_BIN_DEG)) % DIRECTION_CLASSES


def _class_angle_deg(class_index: np.ndarray | int) -> np.ndarray:
    return (DIRECTION_OFFSET_DEG + DIRECTION_BIN_DEG * np.asarray(class_index)) % 360.0


def _circular_soft_targets(angles_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=float)[:, None]
    centers = _class_angle_deg(np.arange(DIRECTION_CLASSES))[None, :]
    difference = (centers - angles + 180.0) % 360.0 - 180.0
    if sigma_deg <= 0.0:
        targets = np.zeros((len(angles), DIRECTION_CLASSES), dtype=np.float32)
        targets[np.arange(len(angles)), np.argmin(np.abs(difference), axis=1)] = 1.0
        return targets
    weights = np.exp(-0.5 * (difference / float(sigma_deg)) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights.astype(np.float32)


def _circular_error_deg(predicted_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return np.abs((predicted_deg - true_deg + 180.0) % 360.0 - 180.0)


def make_example(
    trial: LoadedTrial,
    rotation_index: int = 0,
    bit_radius_m: float = BIT_RADIUS_M,
) -> dict:
    sequence = _augment_sequence(trial.sequence, trial.columns, rotation_index)
    shape_input = np.vstack(
        [_smoothed_standardized(sequence[trial.columns.index(name)]) for name in INPUT_CHANNELS]
    ).astype(np.float32)
    metadata = dict(trial.metadata)
    shape_fourier, absolute = _fingerprint_features(
        metadata, sequence, trial.columns, float(bit_radius_m)
    )
    angle_deg = _direction_angle_deg(metadata, rotation_index)
    return {
        "sequence": shape_input,
        "context": absolute.astype(np.float32),
        "fourier": np.concatenate((shape_fourier, absolute)).astype(np.float32),
        "state": STATE_TO_INDEX[metadata["search_state"]],
        "direction_class": _direction_class(angle_deg),
        "direction_angle_deg": angle_deg,
        "base_pose_id": metadata["base_pose_id"],
        "trial_id": metadata["trial_id"],
        "rotation_index": rotation_index,
        "radial_level": int(metadata.get("radial_level", -1)),
        "radius_m": float(metadata.get("radius_m", float("nan"))),
        "observation_stratum": (
            "ideal"
            if metadata.get("wrench_observation", "ideal") == "ideal"
            else "sensor_prior"
        ),
    }


def make_examples(
    trials: list[LoadedTrial],
    groups: set[str] | None,
    rotations: tuple[int, ...],
    bit_radius_m: float = BIT_RADIUS_M,
) -> list[dict]:
    selected = [trial for trial in trials if groups is None or trial.metadata["base_pose_id"] in groups]
    return [
        make_example(trial, rotation, bit_radius_m)
        for trial in selected
        for rotation in rotations
    ]


def make_sector_examples(trials: list[LoadedTrial], groups: set[str]) -> list[dict]:
    selected = [trial for trial in trials if _sector_pose_id(trial.metadata) in groups]
    return [make_example(trial, 0) for trial in selected]


class CircularDirectionCNN(nn.Module):
    def __init__(self, context_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 24, 9, padding=4, padding_mode="circular"),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, 7, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(45),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 45 + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.state_head = nn.Linear(128, len(STATE_NAMES))
        self.direction_head = nn.Linear(128, DIRECTION_CLASSES)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sequence).flatten(1)
        hidden = self.shared(torch.cat((encoded, context), dim=1))
        return self.state_head(hidden), self.direction_head(hidden)


def _arrays(examples: list[dict], context_mean: np.ndarray, context_std: np.ndarray) -> tuple[np.ndarray, ...]:
    sequence = np.stack([example["sequence"] for example in examples])
    context = np.stack([example["context"] for example in examples])
    context = (context - context_mean) / context_std
    state = np.asarray([example["state"] for example in examples], dtype=np.int64)
    direction = np.asarray([example["direction_class"] for example in examples], dtype=np.int64)
    angles = np.asarray([example["direction_angle_deg"] for example in examples], dtype=np.float32)
    return sequence, context.astype(np.float32), state, direction, angles


def _evaluate_logits(
    state_logits: np.ndarray,
    direction_logits: np.ndarray,
    state_true: np.ndarray,
    direction_true: np.ndarray,
    angle_true: np.ndarray,
    *,
    temperature: float = 1.0,
    aps_quantile: float | None = None,
    confidence_gate: float | None = None,
) -> dict:
    state_pred = np.argmax(state_logits, axis=1)
    scaled = direction_logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= np.sum(probs, axis=1, keepdims=True)
    direction_pred = np.argmax(probs, axis=1)
    angle_pred = _class_angle_deg(direction_pred)
    error = _circular_error_deg(angle_pred, angle_true)
    result = {
        "samples": len(state_true),
        "state_accuracy": float(np.mean(state_pred == state_true)),
        "direction_top1_accuracy": float(np.mean(direction_pred == direction_true)),
        "angle_error_median_deg": float(np.median(error)),
        "angle_error_p90_deg": float(np.quantile(error, 0.90)),
        "angle_within_7p5_rate": float(np.mean(error <= 7.5 + 1.0e-9)),
    }
    if aps_quantile is not None:
        coverage = []
        set_sizes = []
        for probability, target in zip(probs, direction_true):
            order = np.argsort(-probability)
            cumulative = np.cumsum(probability[order])
            size = min(int(np.searchsorted(cumulative, aps_quantile, side="left")) + 1, len(order))
            prediction_set = set(order[:size].tolist())
            coverage.append(int(target) in prediction_set)
            set_sizes.append(size)
        result["aps_coverage"] = float(np.mean(coverage))
        result["aps_mean_set_size"] = float(np.mean(set_sizes))
    if confidence_gate is not None:
        accepted = np.max(probs, axis=1) >= confidence_gate
        result["gate_threshold"] = confidence_gate
        result["gate_accept_rate"] = float(np.mean(accepted))
        result["gate_accepted_samples"] = int(np.sum(accepted))
        result["gate_within_7p5_rate"] = (
            float(np.mean(error[accepted] <= 7.5 + 1.0e-9)) if np.any(accepted) else None
        )
        result["gate_catastrophic_gt90_rate"] = (
            float(np.mean(error[accepted] > 90.0)) if np.any(accepted) else None
        )
    return result


def _select_confidence_gate(
    direction_logits: np.ndarray,
    angle_true: np.ndarray,
    temperature: float,
    *,
    target_within_rate: float = 0.98,
    min_accepted: int = 24,
) -> float:
    scaled = direction_logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    confidence = np.max(probabilities, axis=1)
    predicted = _class_angle_deg(np.argmax(probabilities, axis=1))
    errors = _circular_error_deg(predicted, angle_true)
    for threshold in sorted(set(map(float, confidence))):
        accepted = confidence >= threshold
        if np.sum(accepted) < min_accepted:
            continue
        if np.mean(errors[accepted] <= 7.5 + 1.0e-9) >= target_within_rate:
            return threshold
    return 1.0


def _collect_logits(
    model: nn.Module, arrays: tuple[np.ndarray, ...], device: torch.device, batch_size: int = 128
) -> tuple[np.ndarray, ...]:
    sequence, context, states, directions, angles = arrays
    dataset = TensorDataset(torch.from_numpy(sequence), torch.from_numpy(context))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    state_logits, direction_logits = [], []
    model.eval()
    with torch.no_grad():
        for seq, ctx in loader:
            state, direction = model(seq.to(device), ctx.to(device))
            state_logits.append(state.cpu().numpy())
            direction_logits.append(direction.cpu().numpy())
    return np.vstack(state_logits), np.vstack(direction_logits), states, directions, angles


def _fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits_t / torch.exp(log_temperature), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().clamp(0.25, 10.0))


def _aps_quantile(logits: np.ndarray, labels: np.ndarray, temperature: float, alpha: float = 0.10) -> float:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    scores = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == target)[0])
        scores.append(float(cumulative[rank]))
    level = min(math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def _fit_fourier_baseline(train_examples: list[dict]) -> tuple[object, object]:
    features = np.stack([example["fourier"] for example in train_examples])
    state = np.asarray([example["state"] for example in train_examples])
    direction = np.asarray([example["direction_class"] for example in train_examples])
    state_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, C=1.0))
    direction_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, C=1.0))
    state_model.fit(features, state)
    direction_model.fit(features, direction)
    return state_model, direction_model


def _evaluate_fourier(models: tuple[object, object], examples: list[dict]) -> dict:
    features = np.stack([example["fourier"] for example in examples])
    states = np.asarray([example["state"] for example in examples])
    direction = np.asarray([example["direction_class"] for example in examples])
    angles = np.asarray([example["direction_angle_deg"] for example in examples])
    state_pred = models[0].predict(features)
    direction_pred = models[1].predict(features)
    errors = _circular_error_deg(_class_angle_deg(direction_pred), angles)
    return {
        "samples": len(examples),
        "state_accuracy": float(np.mean(state_pred == states)),
        "direction_top1_accuracy": float(np.mean(direction_pred == direction)),
        "angle_error_median_deg": float(np.median(errors)),
        "angle_error_p90_deg": float(np.quantile(errors, 0.90)),
        "angle_within_7p5_rate": float(np.mean(errors <= 7.5 + 1.0e-9)),
    }


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    p0b = load_trials(P0B)
    p0c = load_trials(P0C)
    p0d = load_trials(P0D)
    p0a = load_trials(P0A)
    targeted_path = Path(args.targeted_data).resolve() if args.targeted_data else None
    targeted = load_trials(targeted_path) if targeted_path else []
    split = grouped_split(p0b, args.split_seed)
    sector_split = grouped_sector_split(p0d, args.split_seed)
    targeted_split = grouped_targeted_split(targeted, args.split_seed) if targeted else {
        "train": [], "val": []
    }
    _atomic_json(
        output / "split_manifest.json",
        {
            "seed": args.seed,
            "split_seed": args.split_seed,
            "split_before_augmentation": True,
            "train_groups": sorted(split["train"]),
            "val_groups": sorted(split["val"]),
            "test_groups": sorted(split["test"]),
            "counts": {key: len(value) for key, value in split.items()},
            "p0d_split_unit": "spatial orbit before nuisance and sector expansion",
            "p0d_train_groups": sorted(sector_split["train"]),
            "p0d_val_groups": sorted(sector_split["val"]),
            "p0d_test_groups": sorted(sector_split["test"]),
            "p0d_counts": {key: len(value) for key, value in sector_split.items()},
            "targeted_data": str(targeted_path) if targeted_path else None,
            "targeted_split_unit": "base_pose_id before nuisance repeats",
            "targeted_train_groups": sorted(targeted_split["train"]),
            "targeted_val_groups": sorted(targeted_split["val"]),
            "targeted_counts": {key: len(value) for key, value in targeted_split.items()},
            "updated_utc": _utc_now(),
        },
    )

    p0b_train_examples = make_examples(p0b, set(split["train"]), tuple(range(6)))
    p0d_train_examples = make_sector_examples(p0d, set(sector_split["train"]))
    targeted_train_examples = make_examples(targeted, set(targeted_split["train"]), (0,))
    targeted_val_examples = make_examples(targeted, set(targeted_split["val"]), (0,))
    train_examples = p0b_train_examples + p0d_train_examples + targeted_train_examples
    val_examples = make_examples(p0b, set(split["val"]), (0,))
    p0d_val_examples = make_sector_examples(p0d, set(sector_split["val"]))
    calibration_examples = val_examples + p0d_val_examples + targeted_val_examples
    test_examples = make_examples(p0b, set(split["test"]), (0,))
    p0d_test_examples = make_sector_examples(p0d, set(sector_split["test"]))
    hard_examples = make_examples(p0c, None, (0,))
    full_angle_examples = make_examples(p0a, None, (0,))

    context_train = np.stack([example["context"] for example in train_examples])
    context_mean = np.mean(context_train, axis=0)
    context_std = np.maximum(np.std(context_train, axis=0), 1.0e-6)
    arrays = {
        "train": _arrays(train_examples, context_mean, context_std),
        "val": _arrays(calibration_examples, context_mean, context_std),
        "p0b_val": _arrays(val_examples, context_mean, context_std),
        "p0b_test": _arrays(test_examples, context_mean, context_std),
        "p0d_val": _arrays(p0d_val_examples, context_mean, context_std),
        "p0d_test": _arrays(p0d_test_examples, context_mean, context_std),
        "p0c_hard": _arrays(hard_examples, context_mean, context_std),
        "p0a_full_angle": _arrays(full_angle_examples, context_mean, context_std),
        **(
            {"targeted_val": _arrays(targeted_val_examples, context_mean, context_std)}
            if targeted_val_examples
            else {}
        ),
    }

    progress = {
        "stage": "P1_direction_models",
        "device": str(device),
        "epochs_total": args.epochs,
        "epochs_completed": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "progress.json", progress)

    fourier_models = _fit_fourier_baseline(train_examples)
    joblib.dump(fourier_models[0], output / "fourier_state_logistic.joblib")
    joblib.dump(fourier_models[1], output / "fourier_direction_logistic.joblib")
    fourier_metrics = {
        name: _evaluate_fourier(fourier_models, examples)
        for name, examples in (
            ("p0b_val", val_examples),
            ("p0b_test", test_examples),
            ("p0d_val", p0d_val_examples),
            ("p0d_test", p0d_test_examples),
            ("p0c_hard", hard_examples),
            ("p0a_full_angle", full_angle_examples),
        )
    }

    train_sequence, train_context, train_state, train_direction, train_angles = arrays["train"]
    train_soft_direction = _circular_soft_targets(
        train_angles, args.soft_direction_sigma_deg
    )
    dataset = TensorDataset(
        torch.from_numpy(train_sequence),
        torch.from_numpy(train_context),
        torch.from_numpy(train_state),
        torch.from_numpy(train_direction),
        torch.from_numpy(train_soft_direction),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CircularDirectionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    bar = tqdm(range(1, args.epochs + 1), desc="P1 circular CNN", unit="epoch", dynamic_ncols=True)
    for epoch in bar:
        model.train()
        losses = []
        for sequence, context, state, direction, soft_direction in loader:
            sequence = sequence.to(device)
            context = context.to(device)
            state = state.to(device)
            direction = direction.to(device)
            soft_direction = soft_direction.to(device)
            optimizer.zero_grad()
            state_logits, direction_logits = model(sequence, context)
            loss_state = nn.functional.cross_entropy(state_logits, state)
            loss_direction = -torch.mean(
                torch.sum(soft_direction * nn.functional.log_softmax(direction_logits, dim=1), dim=1)
            )
            loss = loss_state + loss_direction
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_logits = _collect_logits(model, arrays["val"], device)
        val_state_loss = nn.functional.cross_entropy(
            torch.tensor(val_logits[0]), torch.tensor(val_logits[2])
        ).item()
        val_soft_direction = torch.tensor(
            _circular_soft_targets(val_logits[4], args.soft_direction_sigma_deg)
        )
        val_direction_loss = -torch.mean(
            torch.sum(
                val_soft_direction * nn.functional.log_softmax(torch.tensor(val_logits[1]), dim=1),
                dim=1,
            )
        ).item()
        val_loss = val_state_loss + val_direction_loss
        val_metrics = _evaluate_logits(*val_logits)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_state_accuracy": val_metrics["state_accuracy"],
            "val_angle_p90_deg": val_metrics["angle_error_p90_deg"],
        }
        history.append(record)
        if val_loss < best_loss - 1.0e-4:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), output / "circular_cnn_best.pt")
        else:
            stale += 1
        progress.update(
            {
                "epochs_completed": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "updated_utc": _utc_now(),
            }
        )
        _atomic_json(output / "progress.json", progress)
        bar.set_postfix(
            loss=f"{record['train_loss']:.3f}",
            val=f"{val_loss:.3f}",
            state=f"{val_metrics['state_accuracy']:.2f}",
            p90=f"{val_metrics['angle_error_p90_deg']:.1f}",
            refresh=False,
        )
        if stale >= args.patience:
            break
    bar.close()

    model.load_state_dict(torch.load(output / "circular_cnn_best.pt", map_location=device, weights_only=True))
    val_logits = _collect_logits(model, arrays["val"], device)
    temperature = _fit_temperature(val_logits[1], val_logits[3], device)
    aps_quantile = _aps_quantile(val_logits[1], val_logits[3], temperature)
    confidence_gate = _select_confidence_gate(val_logits[1], val_logits[4], temperature)
    cnn_metrics = {}
    evaluation_names = [
        "p0b_val",
        "p0b_test",
        "p0d_val",
        "p0d_test",
        "p0c_hard",
        "p0a_full_angle",
    ]
    if "targeted_val" in arrays:
        evaluation_names.append("targeted_val")
    for name in evaluation_names:
        logits = _collect_logits(model, arrays[name], device)
        cnn_metrics[name] = _evaluate_logits(
            *logits,
            temperature=temperature,
            aps_quantile=aps_quantile,
            confidence_gate=confidence_gate,
        )

    metrics = {
        "device": str(device),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_examples_after_C6": len(train_examples),
        "p0b_train_examples_after_C6": len(p0b_train_examples),
        "p0d_train_examples": len(p0d_train_examples),
        "targeted_train_examples": len(targeted_train_examples),
        "targeted_val_examples": len(targeted_val_examples),
        "soft_direction_sigma_deg": args.soft_direction_sigma_deg,
        "calibration_examples": len(calibration_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "p0d_val_examples": len(p0d_val_examples),
        "p0d_test_examples": len(p0d_test_examples),
        "p0c_hard_examples": len(hard_examples),
        "p0a_full_angle_examples": len(full_angle_examples),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "temperature": temperature,
        "aps_alpha": 0.10,
        "aps_quantile": aps_quantile,
        "confidence_gate": confidence_gate,
        "confidence_gate_selection": {
            "split": "combined P0b validation plus P0d spatial-orbit validation",
            "target_within_7p5_rate": 0.98,
            "minimum_accepted_samples": 24,
        },
        "fourier_logistic": fourier_metrics,
        "circular_cnn": cnn_metrics,
        "history": history,
        "claim_boundary": (
            "Simulation-only grouped direction model. P0b C6 augmentation is supplemented by physical P0d sector "
            "anchors and optional targeted off-grid training/calibration data split by base pose. P0c and P0a remain "
            "independent diagnostics. P1v2 is never loaded. Not hardware validation or closed-loop gain."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "metrics.json", metrics)
    np.savez(
        output / "normalization.npz",
        context_mean=context_mean,
        context_std=context_std,
        direction_class_angles_deg=_class_angle_deg(np.arange(DIRECTION_CLASSES)),
    )
    _atomic_json(
        output / "interface_contract.json",
        {
            "input": {
                "phase_bins": 360,
                "sequence_channels": list(INPUT_CHANNELS),
                "sequence_preprocessing": "circular smooth window 7 then per-trial standardization",
                "dimensionless_context_features": 16,
            },
            "output": {
                "state_classes": list(STATE_NAMES),
                "direction_bins": DIRECTION_CLASSES,
                "direction_bin_width_deg": DIRECTION_BIN_DEG,
                "direction_bin_centers_deg": _class_angle_deg(
                    np.arange(DIRECTION_CLASSES)
                ).tolist(),
                "temperature": temperature,
                "aps_alpha": 0.10,
                "aps_quantile": aps_quantile,
            },
            "control_gate": {
                "short_move_allowed_if_max_direction_probability_gte": confidence_gate,
                "otherwise": "do not execute learned transport; use guarded local outward spiral or re-probe",
                "distance_policy": "no learned distance output; choose a bounded mechanics-derived transport length",
            },
            "claim_boundary": metrics["claim_boundary"],
            "updated_utc": _utc_now(),
        },
    )
    lines = [
        "# M8 P1 Direction Model Baselines",
        "",
        f"Updated: {metrics['updated_utc']}",
        "",
        f"Device: `{device}`; best CNN epoch: {best_epoch}; temperature: {temperature:.3f}; confidence gate: {confidence_gate:.3f}",
        "",
        "| model / split | state accuracy | angle median / p90 | within 7.5 deg | APS coverage / size | gated accept / accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_metrics in (("fourier", fourier_metrics), ("cnn", cnn_metrics)):
        for split_name, item in model_metrics.items():
            aps = (
                f"{item['aps_coverage']:.3f} / {item['aps_mean_set_size']:.2f}"
                if "aps_coverage" in item
                else "n/a"
            )
            gated = (
                f"{item['gate_accept_rate']:.3f} / {item['gate_within_7p5_rate']:.3f}"
                if "gate_accept_rate" in item and item["gate_within_7p5_rate"] is not None
                else "n/a"
            )
            lines.append(
                f"| {model_name} / {split_name} | {item['state_accuracy']:.3f} | "
                f"{item['angle_error_median_deg']:.2f} / {item['angle_error_p90_deg']:.2f} deg | "
                f"{item['angle_within_7p5_rate']:.3f} | {aps} | {gated} |"
            )
    lines.extend(["", "## Boundary", "", metrics["claim_boundary"], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    progress.update({"status": "complete", "updated_utc": _utc_now()})
    _atomic_json(output / "progress.json", progress)
    print(json.dumps({"output": str(output), "best_epoch": best_epoch, "cnn": cnn_metrics}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped M8 direction-first baselines.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--targeted-data")
    parser.add_argument("--soft-direction-sigma-deg", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.soft_direction_sigma_deg < 0.0:
        parser.error("--soft-direction-sigma-deg must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
