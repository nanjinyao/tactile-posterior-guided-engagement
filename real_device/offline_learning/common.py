from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


REAL_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REAL_ROOT.parent / "mujoco_unscrewing_probe_new"
DATASET_ROOT = (
    REAL_ROOT / "offline_analysis" / "real_multisize_learning_v1" / "dataset"
)
ANALYSIS_ROOT = REAL_ROOT / "offline_analysis" / "real_multisize_learning_v1"
BUNDLE_ROOT = (
    REAL_ROOT
    / "model_runtime"
    / "bundles"
    / "real_multisize_direction_v1"
)
MODEL_ID = "real_multisize_direction_v1"

SIZES = ("m8", "m10", "m12")
SIZE_TO_INDEX = {name: index for index, name in enumerate(SIZES)}
SIZE_BIT_RADIUS_M = {
    "m8": 0.006 / math.sqrt(3.0),
    "m10": 0.008 / math.sqrt(3.0),
    "m12": 0.010 / math.sqrt(3.0),
}
BIT_RADIUS_REFERENCE_M = SIZE_BIT_RADIUS_M["m12"]
TARGET_FORCE_REFERENCE_N = 3.5

STATE_NAMES = (
    "supported_search",
    "double_overhang",
    "outer_partial_support",
)
REAL_STATE_TO_CANONICAL = {
    "supported_search": "supported_search",
    "double_overhang": "double_overhang",
    "outer_partial_support": "outer_partial_support",
}
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}

PHASE_BINS = 360
INPUT_CHANNELS = (
    "fx_world_star_n",
    "fy_world_star_n",
    "fz_world_star_n",
    "tx_world_star_nm",
    "ty_world_star_nm",
    "tz_world_star_nm",
    "cmd_z_relative_mm",
)
CONTEXT_ORDER = (
    "size_onehot_m8",
    "size_onehot_m10",
    "size_onehot_m12",
    "target_down_force_over_3p5N",
    "bit_radius_over_m12_radius",
)
DIRECTION_CLASSES = 48
DIRECTION_BIN_DEG = 7.5
DIRECTION_OFFSET_DEG = 3.75
C6_ROTATIONS = tuple(range(6))
C6_TRANSFORM_ID = "REAL_FRAME_C6_V2_RPOS_PHASE_NEG_CAD_V1P5"


def utc_json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=utc_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direction_class(angle_deg: float | np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return (
        np.rint(((angle - DIRECTION_OFFSET_DEG) % 360.0) / DIRECTION_BIN_DEG)
        .astype(np.int64)
        % DIRECTION_CLASSES
    )


def class_angle_deg(class_index: int | np.ndarray) -> np.ndarray:
    return (
        DIRECTION_OFFSET_DEG
        + DIRECTION_BIN_DEG * np.asarray(class_index, dtype=float)
    ) % 360.0


def circular_error_deg(
    predicted_deg: float | np.ndarray, true_deg: float | np.ndarray
) -> np.ndarray:
    predicted = np.asarray(predicted_deg, dtype=float)
    truth = np.asarray(true_deg, dtype=float)
    return np.abs((predicted - truth + 180.0) % 360.0 - 180.0)


def circular_soft_targets(
    angles_deg: np.ndarray, sigma_deg: float = 7.5
) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=float)[:, None]
    centers = class_angle_deg(np.arange(DIRECTION_CLASSES))[None, :]
    difference = (centers - angles + 180.0) % 360.0 - 180.0
    weights = np.exp(-0.5 * (difference / float(sigma_deg)) ** 2)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights.astype(np.float32)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    values = values - np.max(values, axis=1, keepdims=True)
    probability = np.exp(values)
    return probability / np.sum(probability, axis=1, keepdims=True)


def posterior_direction_deg(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    angles = np.deg2rad(class_angle_deg(np.arange(DIRECTION_CLASSES)))
    vector = values @ np.exp(1j * angles)
    return np.mod(np.rad2deg(np.angle(vector)), 360.0)


def top1_direction_deg(probability: np.ndarray) -> np.ndarray:
    return class_angle_deg(np.argmax(np.asarray(probability), axis=1))


def direction_metrics(
    probability: np.ndarray, truth_deg: np.ndarray
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=float)
    predicted = top1_direction_deg(probability)
    errors = circular_error_deg(predicted, truth_deg)
    soft_targets = circular_soft_targets(np.asarray(truth_deg, dtype=float))
    circular_nll = -np.mean(
        np.sum(soft_targets * np.log(np.clip(probability, 1.0e-12, 1.0)), axis=1)
    )
    return {
        "count": int(len(errors)),
        "median_error_deg": float(np.median(errors)),
        "p90_error_deg": float(np.quantile(errors, 0.90)),
        "circular_soft_target_nll": float(circular_nll),
        "within_7p5_fraction": float(np.mean(errors <= 7.5)),
        "within_15_fraction": float(np.mean(errors <= 15.0)),
        "within_30_fraction": float(np.mean(errors <= 30.0)),
        "catastrophic_gt90_count": int(np.sum(errors > 90.0)),
        "catastrophic_gt90_fraction": float(np.mean(errors > 90.0)),
    }


def state_metrics(
    probability: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    predicted = np.argmax(np.asarray(probability), axis=1)
    target = np.asarray(truth, dtype=np.int64)
    matrix = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=np.int64)
    for actual, estimate in zip(target, predicted):
        matrix[int(actual), int(estimate)] += 1
    class_f1 = []
    for index in range(len(STATE_NAMES)):
        true_positive = float(matrix[index, index])
        false_positive = float(np.sum(matrix[:, index]) - true_positive)
        false_negative = float(np.sum(matrix[index, :]) - true_positive)
        denominator = 2.0 * true_positive + false_positive + false_negative
        class_f1.append(
            0.0 if denominator <= 0.0 else 2.0 * true_positive / denominator
        )
    return {
        "count": int(len(target)),
        "accuracy": float(np.mean(predicted == target)),
        "class_f1": class_f1,
        "macro_f1": float(np.mean(class_f1)),
        "confusion_matrix": matrix.tolist(),
    }


def aps_quantile(
    probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.10
) -> float:
    scores: list[float] = []
    for probability, target in zip(probabilities, labels):
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        rank = int(np.flatnonzero(order == int(target))[0])
        scores.append(float(cumulative[rank]))
    if not scores:
        raise ValueError("APS calibration requires samples")
    level = min(
        math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores),
        1.0,
    )
    return float(np.quantile(scores, level, method="higher"))


def c6_augment(
    sequence: np.ndarray,
    direction_deg: np.ndarray,
    state: np.ndarray,
    context: np.ndarray,
    origin_index: np.ndarray,
    rotations: Iterable[int] = C6_ROTATIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the real-frame C6 transform without changing split membership.

    The fixture-frame direction label is counter-clockwise from +X_F to +Y_F.
    With the recorded cone convention
    ``R(phi)=Rz(-phi)*Ry(15deg)*Rz(phi)``, rotating a physical sample by
    ``delta`` therefore rotates lateral force/torque vectors by ``+delta`` and
    advances the source phase by ``+delta``.  In array form this is
    ``R(+delta)`` followed by ``roll(-delta)``.  The v1 transform incorrectly
    used ``R(-delta)`` with the forward ``+delta`` label and is rejected.

    This is a training regularizer, never an independent sample.
    """

    source = np.asarray(sequence, dtype=np.float32)
    outputs: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    states: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    rotation_ids: list[np.ndarray] = []
    for rotation_index in rotations:
        angle = rotation_index * math.pi / 3.0
        c, s = math.cos(angle), math.sin(angle)
        transformed = source.copy()
        for x_index, y_index in ((0, 1), (3, 4)):
            x_value = source[:, x_index].copy()
            y_value = source[:, y_index].copy()
            transformed[:, x_index] = c * x_value - s * y_value
            transformed[:, y_index] = s * x_value + c * y_value
        shift = -int(rotation_index * PHASE_BINS / 6)
        transformed = np.roll(transformed, shift, axis=2)
        outputs.append(transformed)
        angles.append((np.asarray(direction_deg) + 60.0 * rotation_index) % 360.0)
        states.append(np.asarray(state))
        contexts.append(np.asarray(context))
        origins.append(np.asarray(origin_index))
        rotation_ids.append(
            np.full(len(source), rotation_index, dtype=np.int64)
        )
    return (
        np.concatenate(outputs, axis=0),
        np.concatenate(angles, axis=0).astype(np.float32),
        np.concatenate(states, axis=0).astype(np.int64),
        np.concatenate(contexts, axis=0).astype(np.float32),
        np.concatenate(origins, axis=0).astype(np.int64),
        np.concatenate(rotation_ids, axis=0),
    )


def normalization(
    sequence: np.ndarray, context: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(sequence, dtype=np.float32)
    sequence_mean = np.mean(values, axis=(0, 2), keepdims=False)
    sequence_std = np.std(values, axis=(0, 2), keepdims=False)
    sequence_std[sequence_std < 1.0e-6] = 1.0
    context_mean = np.mean(context, axis=0)
    context_std = np.std(context, axis=0)
    context_std[context_std < 1.0e-6] = 1.0
    return {
        "sequence_mean": sequence_mean.astype(np.float32),
        "sequence_std": sequence_std.astype(np.float32),
        "context_mean": context_mean.astype(np.float32),
        "context_std": context_std.astype(np.float32),
    }


def apply_normalization(
    sequence: np.ndarray,
    context: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    normalized_sequence = (
        sequence - values["sequence_mean"][None, :, None]
    ) / values["sequence_std"][None, :, None]
    normalized_context = (
        context - values["context_mean"][None, :]
    ) / values["context_std"][None, :]
    return (
        normalized_sequence.astype(np.float32),
        normalized_context.astype(np.float32),
    )


class FourierLinearModel(nn.Module):
    architecture_type = "REAL_FOURIER_LINEAR_V1"

    def __init__(self, harmonics: int = 6) -> None:
        super().__init__()
        self.harmonics = int(harmonics)
        feature_dim = len(INPUT_CHANNELS) * 2 * self.harmonics + len(CONTEXT_ORDER)
        self.state_head = nn.Linear(feature_dim, len(STATE_NAMES))
        self.direction_head = nn.Linear(feature_dim, DIRECTION_CLASSES)

    def features(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        spectrum = torch.fft.rfft(sequence, dim=2) / sequence.shape[2]
        selected = spectrum[:, :, 1 : self.harmonics + 1]
        fourier = torch.cat((selected.real, selected.imag), dim=2).flatten(1)
        return torch.cat((fourier, context), dim=1)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(sequence, context)
        return self.state_head(features), self.direction_head(features)


class TinyCircularCNN(nn.Module):
    architecture_type = "REAL_TINY_CIRCULAR_CNN_V1"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                len(INPUT_CHANNELS),
                16,
                9,
                padding=4,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(16, 24, 7, padding=3, padding_mode="circular"),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Conv1d(24, 24, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(30),
        )
        self.shared = nn.Sequential(
            nn.Linear(24 * 30 + len(CONTEXT_ORDER), 48),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.state_head = nn.Linear(48, len(STATE_NAMES))
        self.direction_head = nn.Linear(48, DIRECTION_CLASSES)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(sequence).flatten(1)
        hidden = self.shared(torch.cat((encoded, context), dim=1))
        return self.state_head(hidden), self.direction_head(hidden)


class FrozenSimulationAdapter(nn.Module):
    """Frozen legacy simulation backbone plus a small real residual adapter."""

    architecture_type = "REAL_FROZEN_SIM_ADAPTER_V1"

    def __init__(
        self,
        legacy_context_mean: np.ndarray,
        legacy_context_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.input_adapter = nn.Conv1d(len(INPUT_CHANNELS), 4, 1)
        nn.init.zeros_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 24, 9, padding=4, padding_mode="circular"),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, 7, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(45),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 45 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.teacher_state_head = nn.Linear(128, len(STATE_NAMES))
        self.teacher_direction_head = nn.Linear(128, DIRECTION_CLASSES)
        self.state_head = nn.Linear(128, len(STATE_NAMES))
        self.direction_head = nn.Linear(128, DIRECTION_CLASSES)
        self.register_buffer(
            "legacy_context_mean",
            torch.as_tensor(legacy_context_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "legacy_context_std",
            torch.as_tensor(legacy_context_std, dtype=torch.float32),
        )

    @staticmethod
    def _legacy_shape(sequence_raw: torch.Tensor) -> torch.Tensor:
        selected = sequence_raw[:, [2, 3, 4, 6]]
        mean = torch.mean(selected, dim=2, keepdim=True)
        standard_deviation = torch.std(
            selected, dim=2, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        return (selected - mean) / standard_deviation

    @staticmethod
    def _legacy_context(
        sequence_raw: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        target_force = context[:, 3].clamp_min(1.0e-6) * TARGET_FORCE_REFERENCE_N
        bit_radius = context[:, 4].clamp_min(1.0e-6) * BIT_RADIUS_REFERENCE_M
        selected = sequence_raw[:, [2, 3, 4, 6]]
        scales = torch.stack(
            (
                target_force,
                target_force * bit_radius,
                target_force * bit_radius,
                bit_radius * 1000.0,
            ),
            dim=1,
        )
        dimensionless = selected / scales[:, :, None]
        features = torch.stack(
            (
                torch.mean(dimensionless, dim=2),
                torch.std(dimensionless, dim=2, unbiased=False),
                torch.amin(dimensionless, dim=2),
                torch.amax(dimensionless, dim=2),
            ),
            dim=2,
        )
        return features.flatten(1)

    def hidden(
        self,
        sequence_raw: torch.Tensor,
        sequence_normalized: torch.Tensor,
        context_raw: torch.Tensor,
    ) -> torch.Tensor:
        legacy_shape = self._legacy_shape(sequence_raw)
        shape = legacy_shape + self.input_adapter(sequence_normalized)
        legacy_context = self._legacy_context(sequence_raw, context_raw)
        normalized_context = (
            legacy_context - self.legacy_context_mean[None, :]
        ) / self.legacy_context_std[None, :]
        encoded = self.encoder(shape).flatten(1)
        return self.shared(torch.cat((encoded, normalized_context), dim=1))

    def forward(
        self,
        sequence_raw: torch.Tensor,
        sequence_normalized: torch.Tensor,
        context_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(sequence_raw, sequence_normalized, context_raw)
        return self.state_head(hidden), self.direction_head(hidden)

    def teacher(
        self,
        sequence_raw: torch.Tensor,
        context_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            zero_residual = torch.zeros_like(sequence_raw)
            hidden = self.hidden(sequence_raw, zero_residual, context_raw)
            return (
                self.teacher_state_head(hidden),
                self.teacher_direction_head(hidden),
            )


class SimulationAdapterTailL2SP(FrozenSimulationAdapter):
    """Simulation adapter with a narrowly unfrozen shared tail.

    The convolutional encoder and teacher heads remain frozen exactly as in
    :class:`FrozenSimulationAdapter`.  Training may update the final legacy
    shared linear layer in addition to the real-domain input/output adapters,
    but the trainer anchors that layer to its simulation initialization with
    an explicit L2-SP penalty.
    """

    architecture_type = "REAL_SIM_ADAPTER_TAIL_L2SP_V1"
    l2sp_parameter_names = ("shared.0.weight", "shared.0.bias")


def _initialize_simulation_adapter_model(
    model: FrozenSimulationAdapter,
    checkpoint: Path,
) -> FrozenSimulationAdapter:
    """Load one legacy simulation checkpoint into an adapter topology."""

    legacy = torch.load(checkpoint, map_location="cpu", weights_only=True)
    translated: dict[str, torch.Tensor] = {}
    for key, value in legacy.items():
        if key.startswith("encoder.") or key.startswith("shared."):
            translated[key] = value
        elif key.startswith("state_head."):
            suffix = key[len("state_head.") :]
            translated[f"teacher_state_head.{suffix}"] = value
            translated[f"state_head.{suffix}"] = value.clone()
        elif key.startswith("direction_head."):
            suffix = key[len("direction_head.") :]
            translated[f"teacher_direction_head.{suffix}"] = value
            translated[f"direction_head.{suffix}"] = value.clone()
    missing, unexpected = model.load_state_dict(translated, strict=False)
    allowed_missing = {
        "input_adapter.weight",
        "input_adapter.bias",
        "legacy_context_mean",
        "legacy_context_std",
    }
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"legacy adapter state mismatch missing={missing} unexpected={unexpected}"
        )
    return model


def initialize_simulation_adapter(
    checkpoint: Path, normalization_path: Path
) -> FrozenSimulationAdapter:
    normalization_values = np.load(normalization_path)
    model = FrozenSimulationAdapter(
        normalization_values["context_mean"],
        normalization_values["context_std"],
    )
    _initialize_simulation_adapter_model(model, checkpoint)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("input_adapter.")
            or name.startswith("state_head.")
            or name.startswith("direction_head.")
        )
    model.encoder.eval()
    model.shared.eval()
    model.teacher_state_head.eval()
    model.teacher_direction_head.eval()
    return model


def initialize_simulation_tail_l2sp(
    checkpoint: Path, normalization_path: Path
) -> SimulationAdapterTailL2SP:
    """Initialize candidate 3 from the same frozen simulation checkpoint."""

    normalization_values = np.load(normalization_path)
    model = SimulationAdapterTailL2SP(
        normalization_values["context_mean"],
        normalization_values["context_std"],
    )
    _initialize_simulation_adapter_model(model, checkpoint)
    trainable_prefixes = (
        "input_adapter.",
        "shared.0.",
        "state_head.",
        "direction_head.",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    model.encoder.eval()
    model.teacher_state_head.eval()
    model.teacher_direction_head.eval()
    return model


@dataclass(frozen=True)
class TrainingArrays:
    sequence: np.ndarray
    context: np.ndarray
    state: np.ndarray
    direction_deg: np.ndarray
    size_index: np.ndarray
    sample_id: np.ndarray


def load_training_arrays(path: Path) -> TrainingArrays:
    with np.load(path, allow_pickle=False) as data:
        return TrainingArrays(
            sequence=np.asarray(data["sequence"], dtype=np.float32),
            context=np.asarray(data["context"], dtype=np.float32),
            state=np.asarray(data["state_index"], dtype=np.int64),
            direction_deg=np.asarray(data["direction_deg"], dtype=np.float32),
            size_index=np.asarray(data["size_index"], dtype=np.int64),
            sample_id=np.asarray(data["sample_id"]).astype(str),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
