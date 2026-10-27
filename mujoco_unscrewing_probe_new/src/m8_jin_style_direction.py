from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from src.m8_direction_runtime import DirectionEstimate, M8DirectionEnsemble
import src.train_m8_direction_models as improved


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = (
    ROOT / "outputs/current/models/m8_jin_2021_style_direction_proxy"
)

# Jin et al. use 9 classes for a square and 11 for a pentagon: one aligned
# class plus edge/vertex direction classes.  The natural M8 hex adaptation is
# therefore 12 directional sectors; the aligned class is absent because this
# benchmark is conditioned on ambiguous top contact.
DIRECTION_SECTORS = 12
DIRECTION_SECTOR_DEG = 360.0 / DIRECTION_SECTORS
PAPER_INPUT_CHANNELS = ("tip_z_m", "Tx_Nm", "Ty_Nm")
POLAR_CANVAS_SIZE = 200
NETWORK_IMAGE_SIZE = 20
MOVING_AVERAGE_SAMPLES = 20


def direction_centers_deg() -> np.ndarray:
    return np.arange(DIRECTION_SECTORS, dtype=float) * DIRECTION_SECTOR_DEG


def direction_class(angle_deg: float) -> int:
    return int(np.floor((float(angle_deg) + 0.5 * DIRECTION_SECTOR_DEG) / DIRECTION_SECTOR_DEG)) % DIRECTION_SECTORS


def circular_error_deg(predicted_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(predicted_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0)


def _circular_resample(values: np.ndarray, samples: int) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    source_phase = np.linspace(0.0, 1.0, len(source), endpoint=False)
    target_phase = np.linspace(0.0, 1.0, samples, endpoint=False)
    extended_phase = np.concatenate((source_phase[-1:] - 1.0, source_phase, source_phase[:1] + 1.0))
    extended_values = np.concatenate((source[-1:], source, source[:1]))
    return np.interp(target_phase, extended_phase, extended_values)


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float)
    left = window // 2
    right = window - left - 1
    padded = np.pad(np.asarray(values, dtype=float), (left, right), mode="wrap")
    return np.convolve(padded, np.ones(window, dtype=float) / window, mode="valid")


def polar_binary_image(sequence: np.ndarray) -> np.ndarray:
    """Convert Xz/Mx/My phase traces to the paper's 3x20x20 polar pattern.

    The paper fixes the 200x200 binary polar image, 20-sample moving average,
    and 20x20 network input, but omits the exact normalization and rasterizer.
    This proxy locks per-scan min-max normalization, a two-pixel polyline, and
    deterministic 10x10 max downsampling before any evaluation is read.
    """

    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] < 32:
        raise ValueError("paper-style input must have shape (3, N), N >= 32")
    if not np.all(np.isfinite(values)):
        raise ValueError("paper-style input contains non-finite values")

    images = []
    samples = 2000
    phase = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    center = 0.5 * (POLAR_CANVAS_SIZE - 1)
    scale = 0.48 * (POLAR_CANVAS_SIZE - 1)
    for channel in values:
        signal = _circular_resample(channel, samples)
        signal = _circular_moving_average(signal, MOVING_AVERAGE_SAMPLES)
        low, high = float(np.min(signal)), float(np.max(signal))
        normalized = np.full_like(signal, 0.5) if high - low < 1.0e-12 else (signal - low) / (high - low)
        radius = 0.12 + 0.82 * normalized
        x = center + scale * radius * np.cos(phase)
        y = center - scale * radius * np.sin(phase)
        canvas = Image.new("L", (POLAR_CANVAS_SIZE, POLAR_CANVAS_SIZE), 0)
        points = list(zip(np.rint(x).astype(int), np.rint(y).astype(int)))
        ImageDraw.Draw(canvas).line(points + points[:1], fill=255, width=2, joint="curve")
        binary = np.asarray(canvas, dtype=np.uint8) > 0
        factor = POLAR_CANVAS_SIZE // NETWORK_IMAGE_SIZE
        pooled = binary.reshape(
            NETWORK_IMAGE_SIZE,
            factor,
            NETWORK_IMAGE_SIZE,
            factor,
        ).max(axis=(1, 3))
        images.append(pooled.astype(np.float32))
    return np.stack(images)


class JinStylePolarCNN(nn.Module):
    """Paper-specified two-convolution/two-pooling/one-FC M8 proxy."""

    def __init__(self) -> None:
        super().__init__()
        # Filter counts and 3x3 kernels are not reported by Jin et al.; these
        # conventional fixed values are part of the preregistered proxy.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 5 * 5, DIRECTION_SECTORS)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


class JinStyleDirectionRuntime:
    """Duck-typed direct-ray runtime for the frozen strict controller."""

    def __init__(self, model_root: Path = DEFAULT_MODEL_ROOT) -> None:
        self.model_root = Path(model_root).resolve()
        contract = json.loads(
            (self.model_root / "runtime_contract.json").read_text(encoding="utf-8")
        )
        self.models: list[JinStylePolarCNN] = []
        for value in contract["checkpoints"]:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            model = JinStylePolarCNN()
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            model.eval()
            self.models.append(model)

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        del target_down_force_n, bit_radius_m
        value = np.asarray(sequence, dtype=float)
        if value.shape[0] != 4:
            raise ValueError("shared phase sequence must contain Fz, Tx, Ty, tip_z")
        image = polar_binary_image(value[[3, 1, 2]])
        probabilities = []
        seed_top = []
        with torch.no_grad():
            tensor = torch.from_numpy(image[None, ...])
            for model in self.models:
                logits = model(tensor).numpy()[0]
                logits -= np.max(logits)
                probability = np.exp(logits)
                probability /= np.sum(probability)
                probabilities.append(probability)
                seed_top.append(int(np.argmax(probability)))
        probability = np.mean(probabilities, axis=0)
        top_class = int(np.argmax(probability))
        top_angle = float(direction_centers_deg()[top_class])
        vector = np.sum(
            probability * np.exp(1j * np.deg2rad(direction_centers_deg()))
        )
        concentration = float(abs(vector))
        seed_angles = direction_centers_deg()[np.asarray(seed_top, dtype=int)]
        spans = np.abs(
            (seed_angles[:, None] - seed_angles[None, :] + 180.0) % 360.0
            - 180.0
        )

        # The strict direct controller expects the project's 48-bin container,
        # but only direction_xy/top_direction are consumed. Keep the proxy's
        # twelve probabilities losslessly at their nearest 48-bin locations.
        probability_48 = np.zeros(improved.DIRECTION_CLASSES, dtype=float)
        for sector, item in enumerate(probability):
            probability_48[improved._direction_class(direction_centers_deg()[sector])] += item
        order = np.argsort(-probability)
        cumulative = np.cumsum(probability[order])
        aps_size = int(np.searchsorted(cumulative, 0.90, side="left")) + 1
        aps_sectors = order[:aps_size]
        aps_48 = tuple(
            improved._direction_class(direction_centers_deg()[index])
            for index in aps_sectors
        )
        angle_rad = math.radians(top_angle)
        return DirectionEstimate(
            state_probabilities=np.asarray([1.0, 0.0, 0.0]),
            direction_probabilities=probability_48,
            top_state="supported_search",
            top_direction_deg=top_angle,
            circular_mean_direction_deg=top_angle,
            direction_xy=np.asarray([math.cos(angle_rad), math.sin(angle_rad)]),
            confidence=float(probability[top_class]),
            aps_direction_classes=aps_48,
            aps_direction_angles_deg=tuple(
                float(direction_centers_deg()[index]) for index in aps_sectors
            ),
            circular_concentration=concentration,
            ensemble_top1_span_deg=float(np.max(spans)),
            transport_allowed=True,
        )

    def infer_scan(
        self,
        phase_rad: np.ndarray,
        wrench: np.ndarray,
        tip_z_m: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        wrench_value = np.asarray(wrench, dtype=float)
        values = np.column_stack(
            (wrench_value[:, 2], wrench_value[:, 3], wrench_value[:, 4], tip_z_m)
        )
        sequence = M8DirectionEnsemble._phase_resample(phase_rad, values)
        return self.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
