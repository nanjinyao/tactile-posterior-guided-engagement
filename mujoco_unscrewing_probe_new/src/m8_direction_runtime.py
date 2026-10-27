from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

import src.train_m8_direction_models as training
from src.checkpoint_paths import resolve_checkpoint_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "outputs" / "current" / "models" / "p1_m8_direction_models"


@dataclass(frozen=True)
class DirectionEstimate:
    state_probabilities: np.ndarray
    direction_probabilities: np.ndarray
    top_state: str
    top_direction_deg: float
    circular_mean_direction_deg: float
    direction_xy: np.ndarray
    confidence: float
    aps_direction_classes: tuple[int, ...]
    aps_direction_angles_deg: tuple[float, ...]
    circular_concentration: float
    ensemble_top1_span_deg: float
    transport_allowed: bool


class M8DirectionEnsemble:
    def __init__(self, model_root: Path = DEFAULT_MODEL_ROOT) -> None:
        self.model_root = Path(model_root).resolve()
        contract = json.loads(
            (self.model_root / "ensemble_interface_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.confidence_gate = float(contract["confidence_gate"])
        self.aps_quantile = float(contract["aps_quantile"])
        structured = contract.get("structured_transport_gate", {})
        self.max_aps_set_size = int(structured.get("max_aps_set_size", training.DIRECTION_CLASSES))
        self.min_circular_concentration = float(structured.get("min_circular_concentration", 0.0))
        self.max_ensemble_top1_span_deg = float(
            structured.get("max_ensemble_top1_span_deg", 180.0)
        )
        self.models: list[tuple[training.CircularDirectionCNN, np.ndarray, np.ndarray, float]] = []
        # Historical checkpoints do not carry a state temperature.  Keep the
        # existing public model tuple stable and store the optional S4 state
        # calibration in a parallel list so old audits remain bit-for-bit
        # compatible at the default value 1.0.
        self.state_temperatures: list[float] = []
        for checkpoint_value in contract["checkpoints"]:
            checkpoint = resolve_checkpoint_path(checkpoint_value, self.model_root)
            seed_dir = checkpoint.parent
            normalization = np.load(seed_dir / "normalization.npz")
            metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
            model = training.CircularDirectionCNN()
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            model.eval()
            self.models.append(
                (
                    model,
                    np.asarray(normalization["context_mean"], dtype=np.float32),
                    np.asarray(normalization["context_std"], dtype=np.float32),
                    float(metrics["temperature"]),
                )
            )
            self.state_temperatures.append(
                float(metrics.get("state_temperature", 1.0))
            )

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / temperature
        scaled -= np.max(scaled)
        probability = np.exp(scaled)
        return probability / np.sum(probability)

    @staticmethod
    def _phase_resample(
        phase_rad: np.ndarray,
        values: np.ndarray,
        bins: int = 360,
    ) -> np.ndarray:
        phase = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * math.pi)
        samples = np.asarray(values, dtype=float)
        if phase.ndim != 1 or samples.ndim != 2 or samples.shape[0] != len(phase):
            raise ValueError("phase must be (N,) and values must be (N, C)")
        if len(phase) < 32 or not np.all(np.isfinite(samples)):
            raise ValueError("fingerprint scan is incomplete or non-finite")
        order = np.argsort(phase)
        phase = phase[order]
        samples = samples[order]
        phase, unique = np.unique(phase, return_index=True)
        samples = samples[unique]
        phase_extended = np.concatenate((phase[-1:] - 2.0 * math.pi, phase, phase[:1] + 2.0 * math.pi))
        grid = np.linspace(0.0, 2.0 * math.pi, bins, endpoint=False)
        channels = []
        for index in range(samples.shape[1]):
            values_extended = np.concatenate((samples[-1:, index], samples[:, index], samples[:1, index]))
            channels.append(np.interp(grid, phase_extended, values_extended))
        return np.vstack(channels)

    @staticmethod
    def _network_inputs(
        sequence: np.ndarray,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sequence.shape != (4, 360):
            raise ValueError("sequence must contain Fz, Tx, Ty, tip_z with shape (4, 360)")
        shape = np.vstack(
            [training._smoothed_standardized(channel) for channel in sequence]
        ).astype(np.float32)
        force_scale = max(float(target_down_force_n), 1.0e-6)
        torque_scale = force_scale * bit_radius_m
        dimensionless = (
            sequence[0] / force_scale,
            sequence[1] / torque_scale,
            sequence[2] / torque_scale,
            sequence[3] / bit_radius_m,
        )
        context = []
        for values in dimensionless:
            context.extend((np.mean(values), np.std(values), np.min(values), np.max(values)))
        return shape, np.asarray(context, dtype=np.float32)

    def infer_phase_sequence(
        self,
        sequence: np.ndarray,
        *,
        target_down_force_n: float,
        bit_radius_m: float,
    ) -> DirectionEstimate:
        shape, context = self._network_inputs(
            np.asarray(sequence, dtype=float), target_down_force_n, bit_radius_m
        )
        state_probabilities = []
        direction_probabilities = []
        with torch.no_grad():
            for (
                model,
                context_mean,
                context_std,
                temperature,
            ), state_temperature in zip(self.models, self.state_temperatures):
                normalized_context = (context - context_mean) / context_std
                state_logits, direction_logits = model(
                    torch.from_numpy(shape[None, ...]),
                    torch.from_numpy(normalized_context[None, ...].astype(np.float32)),
                )
                state_probabilities.append(
                    self._softmax(state_logits.numpy()[0], state_temperature)
                )
                direction_probabilities.append(
                    self._softmax(direction_logits.numpy()[0], temperature)
                )
        state_probability = np.mean(state_probabilities, axis=0)
        direction_probability = np.mean(direction_probabilities, axis=0)
        top_class = int(np.argmax(direction_probability))
        top_angle_deg = float(training._class_angle_deg(top_class))
        order = np.argsort(-direction_probability)
        cumulative = np.cumsum(direction_probability[order])
        size = min(
            int(np.searchsorted(cumulative, self.aps_quantile, side="left")) + 1,
            len(order),
        )
        aps_classes = tuple(map(int, order[:size]))
        aps_angles = tuple(
            map(float, training._class_angle_deg(np.asarray(aps_classes, dtype=int)))
        )
        confidence = float(direction_probability[top_class])
        class_angles = np.deg2rad(
            training._class_angle_deg(np.arange(training.DIRECTION_CLASSES))
        )
        posterior_vector = np.sum(direction_probability * np.exp(1j * class_angles))
        concentration = float(abs(posterior_vector))
        circular_mean_direction_deg = float(np.mod(np.rad2deg(np.angle(posterior_vector)), 360.0))
        circular_mean_direction_rad = math.radians(circular_mean_direction_deg)
        seed_top_angles = training._class_angle_deg(
            np.argmax(np.asarray(direction_probabilities), axis=1)
        )
        seed_difference = np.abs(
            (seed_top_angles[:, None] - seed_top_angles[None, :] + 180.0) % 360.0 - 180.0
        )
        ensemble_span = float(np.max(seed_difference))
        transport_allowed = (
            confidence >= self.confidence_gate
            and len(aps_classes) <= self.max_aps_set_size
            and concentration >= self.min_circular_concentration
            and ensemble_span <= self.max_ensemble_top1_span_deg + 1.0e-9
        )
        return DirectionEstimate(
            state_probabilities=state_probability,
            direction_probabilities=direction_probability,
            top_state=training.STATE_NAMES[int(np.argmax(state_probability))],
            top_direction_deg=top_angle_deg,
            circular_mean_direction_deg=circular_mean_direction_deg,
            direction_xy=np.array(
                [math.cos(circular_mean_direction_rad), math.sin(circular_mean_direction_rad)]
            ),
            confidence=confidence,
            aps_direction_classes=aps_classes,
            aps_direction_angles_deg=aps_angles,
            circular_concentration=concentration,
            ensemble_top1_span_deg=ensemble_span,
            transport_allowed=transport_allowed,
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
        wrench_values = np.asarray(wrench, dtype=float)
        if wrench_values.ndim != 2 or wrench_values.shape[1] != 6:
            raise ValueError("wrench must have shape (N, 6)")
        values = np.column_stack(
            (wrench_values[:, 2], wrench_values[:, 3], wrench_values[:, 4], tip_z_m)
        )
        sequence = self._phase_resample(phase_rad, values)
        return self.infer_phase_sequence(
            sequence,
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
