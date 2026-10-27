from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec6(values: tuple[float, ...] | list[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{name} must contain 6 values, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector.copy()


@dataclass(frozen=True)
class AdmittanceConfig:
    """Parameters for a task-frame Cartesian admittance controller.

    Translation uses SI units (m, m/s, N); rotation uses axis-angle units
    (rad, rad/s, N m). The measured wrench convention is environment-on-tool.
    """

    dt_s: float = 0.002
    virtual_mass: tuple[float, ...] = (4.0, 4.0, 2.0, 0.08, 0.08, 0.05)
    damping: tuple[float, ...] = (180.0, 180.0, 95.0, 3.6, 3.6, 2.5)
    stiffness: tuple[float, ...] = (2200.0, 2200.0, 0.0, 18.0, 18.0, 10.0)
    wrench_filter_hz: float = 25.0
    max_velocity: tuple[float, ...] = (0.010, 0.010, 0.006, 0.35, 0.35, 0.25)
    max_acceleration: tuple[float, ...] = (0.20, 0.20, 0.12, 4.0, 4.0, 3.0)
    max_offset: tuple[float, ...] = (0.002, 0.002, 0.003, 0.10, 0.10, 0.08)
    hard_wrench: tuple[float, ...] = (20.0, 20.0, 20.0, 0.060, 0.060, 0.060)

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if not np.isfinite(self.wrench_filter_hz) or self.wrench_filter_hz <= 0.0:
            raise ValueError("wrench_filter_hz must be positive")
        for name in (
            "virtual_mass",
            "damping",
            "stiffness",
            "max_velocity",
            "max_acceleration",
            "max_offset",
            "hard_wrench",
        ):
            vector = _vec6(getattr(self, name), name)
            if name in {"virtual_mass", "damping", "max_velocity", "max_acceleration", "max_offset", "hard_wrench"}:
                if np.any(vector <= 0.0):
                    raise ValueError(f"{name} must be strictly positive")
            elif np.any(vector < 0.0):
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AdmittanceCommand:
    pose_offset: np.ndarray
    twist: np.ndarray
    filtered_wrench: np.ndarray
    wrench_error: np.ndarray
    acceleration: np.ndarray
    hard_stop: bool
    saturated_axes: np.ndarray = field(repr=False)


class CartesianAdmittanceController:
    """Discrete 6D admittance with filtering, saturation, and a latched stop.

    The implemented dynamics are

        M x_ddot + D x_dot + K x = wrench_measured - wrench_target.

    The returned ``pose_offset`` is added to the nominal task-frame pose.
    For a vertical contact whose reaction is +Fz, a +Fz target causes a
    negative z correction while measured Fz is below target.
    """

    def __init__(self, config: AdmittanceConfig | None = None) -> None:
        self.config = config or AdmittanceConfig()
        self._mass = _vec6(self.config.virtual_mass, "virtual_mass")
        self._damping = _vec6(self.config.damping, "damping")
        self._stiffness = _vec6(self.config.stiffness, "stiffness")
        self._max_velocity = _vec6(self.config.max_velocity, "max_velocity")
        self._max_acceleration = _vec6(self.config.max_acceleration, "max_acceleration")
        self._max_offset = _vec6(self.config.max_offset, "max_offset")
        self._hard_wrench = _vec6(self.config.hard_wrench, "hard_wrench")
        rc_s = 1.0 / (2.0 * np.pi * self.config.wrench_filter_hz)
        self._filter_alpha = self.config.dt_s / (rc_s + self.config.dt_s)
        self.reset()

    @property
    def hard_stop_latched(self) -> bool:
        return self._hard_stop_latched

    @property
    def pose_offset(self) -> np.ndarray:
        return self._offset.copy()

    def observe_wrench_safety(self, measured_wrench: np.ndarray) -> bool:
        """Latch the raw-wrench stop without advancing admittance dynamics.

        MuJoCo may produce a contact peak between two outer-loop controller
        updates. Hardware safety monitoring also runs independently of the
        slower force-control law, so inspect every physics sample here.
        """
        measured = _vec6(measured_wrench, "measured_wrench")
        if np.any(np.abs(measured) > self._hard_wrench):
            self._hard_stop_latched = True
        return self._hard_stop_latched

    def reset(self, pose_offset: np.ndarray | None = None) -> None:
        self._offset = np.zeros(6, dtype=float) if pose_offset is None else _vec6(pose_offset, "pose_offset")
        self._offset = np.clip(self._offset, -self._max_offset, self._max_offset)
        self._twist = np.zeros(6, dtype=float)
        self._filtered_wrench = np.zeros(6, dtype=float)
        self._filter_initialized = False
        self._hard_stop_latched = False

    def step(
        self,
        measured_wrench: np.ndarray,
        target_wrench: np.ndarray | None = None,
        enabled_axes: np.ndarray | None = None,
    ) -> AdmittanceCommand:
        measured = _vec6(measured_wrench, "measured_wrench")
        target = np.zeros(6, dtype=float) if target_wrench is None else _vec6(target_wrench, "target_wrench")
        enabled = np.ones(6, dtype=bool) if enabled_axes is None else np.asarray(enabled_axes, dtype=bool)
        if enabled.shape != (6,):
            raise ValueError(f"enabled_axes must contain 6 values, got shape {enabled.shape}")

        if np.any(np.abs(measured) > self._hard_wrench):
            self._hard_stop_latched = True

        if not self._filter_initialized:
            self._filtered_wrench = measured.copy()
            self._filter_initialized = True
        else:
            self._filtered_wrench += self._filter_alpha * (measured - self._filtered_wrench)

        error = self._filtered_wrench - target
        acceleration = (error - self._damping * self._twist - self._stiffness * self._offset) / self._mass
        acceleration[~enabled] = 0.0
        acceleration = np.clip(acceleration, -self._max_acceleration, self._max_acceleration)

        saturated = np.zeros(6, dtype=bool)
        if self._hard_stop_latched:
            self._twist.fill(0.0)
            acceleration.fill(0.0)
        else:
            self._twist += acceleration * self.config.dt_s
            self._twist[~enabled] = 0.0
            self._twist = np.clip(self._twist, -self._max_velocity, self._max_velocity)
            candidate = self._offset + self._twist * self.config.dt_s
            clipped = np.clip(candidate, -self._max_offset, self._max_offset)
            saturated = ~np.isclose(candidate, clipped, rtol=0.0, atol=1.0e-15)
            self._offset = clipped
            # Anti-windup: do not keep integrating velocity into a pose limit.
            self._twist[saturated] = 0.0

        return AdmittanceCommand(
            pose_offset=self._offset.copy(),
            twist=self._twist.copy(),
            filtered_wrench=self._filtered_wrench.copy(),
            wrench_error=error.copy(),
            acceleration=acceleration.copy(),
            hard_stop=self._hard_stop_latched,
            saturated_axes=saturated,
        )


def critical_damping(virtual_mass: np.ndarray, stiffness: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    """Return diagonal damping for the requested damping ratio."""

    mass = _vec6(virtual_mass, "virtual_mass")
    spring = _vec6(stiffness, "stiffness")
    if np.any(mass <= 0.0) or np.any(spring < 0.0):
        raise ValueError("virtual_mass must be positive and stiffness non-negative")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be positive")
    return 2.0 * ratio * np.sqrt(mass * spring)
