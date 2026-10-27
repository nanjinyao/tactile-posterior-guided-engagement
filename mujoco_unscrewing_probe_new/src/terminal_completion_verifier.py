from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalVerifierConfig:
    """Sensor/CAD terminal-set verifier shared by every search policy."""

    depth_tolerance_mm: float
    bottom_force_n: float
    low_lateral_force_n: float
    low_torque_xy_mnm: float
    stable_depth_span_mm: float
    dwell_steps: int
    expected_axial_force_sign: int = 1
    force_max_lateral_force_n: float | None = None
    force_max_torque_xy_mnm: float | None = None
    low_axial_force_n: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "depth_tolerance_mm",
            "bottom_force_n",
            "low_lateral_force_n",
            "low_torque_xy_mnm",
            "stable_depth_span_mm",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.dwell_steps <= 0:
            raise ValueError("dwell_steps must be positive")
        if self.expected_axial_force_sign not in (-1, 1):
            raise ValueError("expected_axial_force_sign must be -1 or +1")
        for name in (
            "force_max_lateral_force_n",
            "force_max_torque_xy_mnm",
            "low_axial_force_n",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive when provided")


class TerminalCompletionVerifier:
    """Verify either force seating or stable low-wrench CAD-depth insertion.

    A tight, centered fit can reach the known insertion depth with almost no
    preload. Requiring a bottom-force threshold for every contact mode creates
    a false negative. The second channel therefore accepts a stable CAD-depth
    dwell only when lateral force and tilting moment are both small.
    """

    FORCE_DWELL = "bottom_force_dwell"
    LOW_WRENCH_DEPTH_DWELL = "cad_depth_low_wrench_dwell"

    def __init__(self, config: TerminalVerifierConfig) -> None:
        self.config = config
        self._force_count = 0
        self._low_wrench_depth_mm: deque[float] = deque(
            maxlen=config.dwell_steps
        )
        self._reset_telemetry()

    def _reset_telemetry(self) -> None:
        self._updates = 0
        self._depth_band_samples = 0
        self._depth_band_entries = 0
        self._depth_band_exits = 0
        self._consecutive_depth_band = 0
        self._max_consecutive_depth_band = 0
        self._max_force_dwell = 0
        self._max_low_wrench_dwell = 0
        self._was_in_depth_band = False
        self._gate_failure_counts = {
            "out_of_depth_band": 0,
            "force_axial": 0,
            "force_lateral": 0,
            "force_torque": 0,
            "low_axial": 0,
            "low_lateral": 0,
            "low_torque": 0,
        }
        self._last_diagnostics: dict[str, float | bool | int | None] = {}

    def _reset_gate_state(self) -> None:
        self._force_count = 0
        self._low_wrench_depth_mm.clear()

    def reset(self) -> None:
        self._reset_gate_state()
        self._reset_telemetry()

    def diagnostics(self) -> dict:
        """Return JSON-safe gate telemetry without changing verification logic."""

        return {
            "updates": self._updates,
            "depth_band_samples": self._depth_band_samples,
            "depth_band_entries": self._depth_band_entries,
            "depth_band_exits": self._depth_band_exits,
            "max_consecutive_depth_band_samples": self._max_consecutive_depth_band,
            "max_force_dwell_samples": self._max_force_dwell,
            "max_low_wrench_dwell_samples": self._max_low_wrench_dwell,
            "required_dwell_samples": self.config.dwell_steps,
            "gate_failure_counts": dict(self._gate_failure_counts),
            "last": dict(self._last_diagnostics),
        }

    def update(
        self,
        *,
        depth_error_mm: float,
        tip_z_mm: float,
        measured_wrench: np.ndarray,
    ) -> str | None:
        wrench = np.asarray(measured_wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("measured_wrench must contain six finite values")
        if not np.isfinite(depth_error_mm) or not np.isfinite(tip_z_mm):
            raise ValueError("depth values must be finite")

        self._updates += 1
        in_depth_band = abs(float(depth_error_mm)) <= self.config.depth_tolerance_mm
        if not in_depth_band:
            if self._was_in_depth_band:
                self._depth_band_exits += 1
            self._was_in_depth_band = False
            self._consecutive_depth_band = 0
            self._gate_failure_counts["out_of_depth_band"] += 1
            self._reset_gate_state()
            self._last_diagnostics = {
                "in_depth_band": False,
                "depth_error_mm": float(depth_error_mm),
                "tip_z_mm": float(tip_z_mm),
                "lateral_force_n": None,
                "signed_axial_force_n": None,
                "torque_xy_mnm": None,
                "force_seated": False,
                "low_wrench": False,
                "low_wrench_depth_span_mm": None,
            }
            return None

        if not self._was_in_depth_band:
            self._depth_band_entries += 1
        self._was_in_depth_band = True
        self._depth_band_samples += 1
        self._consecutive_depth_band += 1
        self._max_consecutive_depth_band = max(
            self._max_consecutive_depth_band, self._consecutive_depth_band
        )

        lateral_force = float(np.linalg.norm(wrench[:2]))
        torque_xy_mnm = 1000.0 * float(np.linalg.norm(wrench[3:5]))
        force_lateral_limit = (
            self.config.low_lateral_force_n
            if self.config.force_max_lateral_force_n is None
            else self.config.force_max_lateral_force_n
        )
        force_torque_limit = (
            self.config.low_torque_xy_mnm
            if self.config.force_max_torque_xy_mnm is None
            else self.config.force_max_torque_xy_mnm
        )
        signed_axial_force = (
            self.config.expected_axial_force_sign * float(wrench[2])
        )
        force_seated = bool(
            signed_axial_force >= self.config.bottom_force_n
            and lateral_force <= force_lateral_limit
            and torque_xy_mnm <= force_torque_limit
        )
        if signed_axial_force < self.config.bottom_force_n:
            self._gate_failure_counts["force_axial"] += 1
        if lateral_force > force_lateral_limit:
            self._gate_failure_counts["force_lateral"] += 1
        if torque_xy_mnm > force_torque_limit:
            self._gate_failure_counts["force_torque"] += 1
        self._force_count = self._force_count + 1 if force_seated else 0
        self._max_force_dwell = max(self._max_force_dwell, self._force_count)

        low_axial_limit = (
            self.config.bottom_force_n
            if self.config.low_axial_force_n is None
            else self.config.low_axial_force_n
        )
        low_wrench = bool(
            abs(float(wrench[2])) < low_axial_limit
            and lateral_force <= self.config.low_lateral_force_n
            and torque_xy_mnm <= self.config.low_torque_xy_mnm
        )
        if abs(float(wrench[2])) >= low_axial_limit:
            self._gate_failure_counts["low_axial"] += 1
        if lateral_force > self.config.low_lateral_force_n:
            self._gate_failure_counts["low_lateral"] += 1
        if torque_xy_mnm > self.config.low_torque_xy_mnm:
            self._gate_failure_counts["low_torque"] += 1
        if low_wrench:
            self._low_wrench_depth_mm.append(float(tip_z_mm))
        else:
            self._low_wrench_depth_mm.clear()
        self._max_low_wrench_dwell = max(
            self._max_low_wrench_dwell, len(self._low_wrench_depth_mm)
        )
        depth_span = (
            max(self._low_wrench_depth_mm) - min(self._low_wrench_depth_mm)
            if self._low_wrench_depth_mm
            else None
        )
        self._last_diagnostics = {
            "in_depth_band": True,
            "depth_error_mm": float(depth_error_mm),
            "tip_z_mm": float(tip_z_mm),
            "lateral_force_n": lateral_force,
            "signed_axial_force_n": signed_axial_force,
            "torque_xy_mnm": torque_xy_mnm,
            "force_seated": force_seated,
            "low_wrench": low_wrench,
            "low_wrench_depth_span_mm": depth_span,
        }

        if self._force_count >= self.config.dwell_steps:
            return self.FORCE_DWELL
        if len(self._low_wrench_depth_mm) >= self.config.dwell_steps:
            assert depth_span is not None
            if depth_span <= self.config.stable_depth_span_mm:
                return self.LOW_WRENCH_DEPTH_DWELL
        return None
