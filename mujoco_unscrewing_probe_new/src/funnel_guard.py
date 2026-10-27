from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class FunnelGuardConfig:
    """Simulation-initial safety supervisor for post-capture insertion.

    The guard is deliberately independent of the search policy and contact
    label. It observes tip motion and the external-sensor wrench, then requests
    a bounded unload/re-touch recovery when contact consumes energy without
    producing insertion progress.
    """

    enabled: bool = True
    window_s: float = 0.15
    min_window_progress_m: float = 0.00002
    no_progress_energy_j: float = 0.0015
    injected_power_limit_w: float = 0.50
    power_dwell_s: float = 0.010
    max_tilt_deg: float = 0.75
    tilt_dwell_s: float = 0.020
    force_floor_n: float = 0.20
    moment_lever_m: float = 0.003
    max_recoveries: int = 1
    retry_on_terminal_rejection: bool = False
    hold_terminal_set_on_retry: bool = False
    bumpless_terminal_entry_rebase: bool = False
    axial_position_hold_in_terminal_set: bool = False
    recovery_force_n: float = 0.20
    recovery_retract_m: float = 0.00035
    recovery_unload_s: float = 0.10
    recovery_retract_s: float = 0.15
    recovery_retouch_s: float = 0.20
    capture_force_release_s: float = 0.30
    insertion_handoff_s: float = 1.00
    descent_speed_m_s: float = 0.00035
    planar_stiffness_n_m: float = 1500.0
    planar_damping_n_s_m: float = 100.0
    shallow_tilt_stiffness_nm_rad: float = 1.8
    shallow_tilt_damping_nm_s_rad: float = 0.35
    deep_tilt_stiffness_nm_rad: float = 3.0
    deep_tilt_damping_nm_s_rad: float = 0.50
    yaw_stiffness_nm_rad: float = 1.5
    yaw_damping_nm_s_rad: float = 0.25
    cad_bottom_overshoot_margin_m: float = 0.00025

    def __post_init__(self) -> None:
        for name in (
            "window_s",
            "min_window_progress_m",
            "no_progress_energy_j",
            "injected_power_limit_w",
            "power_dwell_s",
            "max_tilt_deg",
            "tilt_dwell_s",
            "force_floor_n",
            "moment_lever_m",
            "recovery_force_n",
            "recovery_retract_m",
            "recovery_unload_s",
            "recovery_retract_s",
            "recovery_retouch_s",
            "capture_force_release_s",
            "insertion_handoff_s",
            "descent_speed_m_s",
            "planar_stiffness_n_m",
            "planar_damping_n_s_m",
            "shallow_tilt_stiffness_nm_rad",
            "shallow_tilt_damping_nm_s_rad",
            "deep_tilt_stiffness_nm_rad",
            "deep_tilt_damping_nm_s_rad",
            "yaw_stiffness_nm_rad",
            "yaw_damping_nm_s_rad",
            "cad_bottom_overshoot_margin_m",
        ):
            _finite_positive(getattr(self, name), name)
        if self.max_recoveries < 0:
            raise ValueError("max_recoveries must be non-negative")


@dataclass(frozen=True)
class FunnelGuardObservation:
    injected_power_w: float
    injected_energy_j: float
    window_energy_j: float
    window_progress_m: float
    lateral_force_ratio: float
    tilt_moment_ratio: float
    health: float
    recovery_requested: bool
    reason: str | None


class FunnelGuard:
    """Contact-energy and progress observer for a tip-frame insertion funnel."""

    def __init__(self, config: FunnelGuardConfig | None = None) -> None:
        self.config = config or FunnelGuardConfig()
        self.reset()

    def reset(self) -> None:
        self._time_s = 0.0
        self._energy_j = 0.0
        self._previous_tip: np.ndarray | None = None
        self._previous_rpy: np.ndarray | None = None
        self._history: deque[tuple[float, float, float]] = deque()
        self._power_over_s = 0.0
        self._tilt_over_s = 0.0
        self._recovery_latched = False
        self.last = FunnelGuardObservation(
            injected_power_w=0.0,
            injected_energy_j=0.0,
            window_energy_j=0.0,
            window_progress_m=0.0,
            lateral_force_ratio=0.0,
            tilt_moment_ratio=0.0,
            health=0.0,
            recovery_requested=False,
            reason=None,
        )

    @property
    def injected_energy_j(self) -> float:
        return self._energy_j

    @staticmethod
    def _wrapped_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        return (current - previous + math.pi) % (2.0 * math.pi) - math.pi

    def observe(
        self,
        tip_position: np.ndarray,
        rpy: np.ndarray,
        wrench_environment_on_tool: np.ndarray,
        dt_s: float,
        *,
        allow_stall: bool = False,
        minimum_tip_z_m: float | None = None,
    ) -> FunnelGuardObservation:
        tip = np.asarray(tip_position, dtype=float)
        angles = np.asarray(rpy, dtype=float)
        wrench = np.asarray(wrench_environment_on_tool, dtype=float)
        dt = _finite_positive(dt_s, "dt_s")
        if tip.shape != (3,) or angles.shape != (3,) or wrench.shape != (6,):
            raise ValueError("tip_position/rpy/wrench must have shape 3/3/6")
        if not np.all(np.isfinite(tip)) or not np.all(np.isfinite(angles)) or not np.all(np.isfinite(wrench)):
            raise ValueError("funnel observations must be finite")
        if minimum_tip_z_m is not None and not math.isfinite(minimum_tip_z_m):
            raise ValueError("minimum_tip_z_m must be finite when provided")

        self._time_s += dt
        injected_work_j = 0.0
        if self._previous_tip is not None and self._previous_rpy is not None:
            delta_tip = tip - self._previous_tip
            delta_rpy = self._wrapped_delta(angles, self._previous_rpy)
            environment_work = float(
                np.dot(wrench[:3], delta_tip) + np.dot(wrench[3:], delta_rpy)
            )
            # Wrench is environment-on-tool. Negative environment work means
            # the robot injected positive work into the contact.
            injected_work_j = max(0.0, -environment_work)
        self._previous_tip = tip.copy()
        self._previous_rpy = angles.copy()
        injected_power_w = injected_work_j / dt
        self._energy_j += injected_work_j

        self._history.append((self._time_s, float(tip[2]), self._energy_j))
        while self._history and self._history[0][0] < self._time_s - self.config.window_s:
            self._history.popleft()
        mature = bool(
            len(self._history) >= 2
            and self._history[-1][0] - self._history[0][0] >= 0.8 * self.config.window_s
        )
        window_progress_m = (
            self._history[0][1] - self._history[-1][1] if mature else 0.0
        )
        window_energy_j = (
            self._history[-1][2] - self._history[0][2] if mature else 0.0
        )

        axial = abs(float(wrench[2])) + self.config.force_floor_n
        lateral_ratio = float(np.linalg.norm(wrench[:2])) / axial
        moment_ratio = float(np.linalg.norm(wrench[3:5])) / (
            self.config.moment_lever_m * axial
        )
        progress_scale = max(self.config.min_window_progress_m, 1.0e-12)
        health = (
            window_progress_m / progress_scale
            - 0.10 * lateral_ratio
            - 0.05 * moment_ratio
        )

        self._power_over_s = (
            self._power_over_s + dt
            if injected_power_w >= self.config.injected_power_limit_w
            else 0.0
        )
        tilt_deg = math.degrees(float(np.linalg.norm(angles[:2])))
        self._tilt_over_s = (
            self._tilt_over_s + dt if tilt_deg >= self.config.max_tilt_deg else 0.0
        )

        reason: str | None = None
        if self.config.enabled and not self._recovery_latched:
            if minimum_tip_z_m is not None and float(tip[2]) < minimum_tip_z_m:
                reason = "cad_depth_overshoot"
            elif self._power_over_s >= self.config.power_dwell_s:
                reason = "contact_power_limit"
            elif self._tilt_over_s >= self.config.tilt_dwell_s:
                reason = "tilt_limit"
            elif (
                mature
                and not allow_stall
                and window_progress_m < self.config.min_window_progress_m
                and window_energy_j >= self.config.no_progress_energy_j
            ):
                reason = "energy_without_progress"
            if reason is not None:
                self._recovery_latched = True

        self.last = FunnelGuardObservation(
            injected_power_w=injected_power_w,
            injected_energy_j=self._energy_j,
            window_energy_j=window_energy_j,
            window_progress_m=window_progress_m,
            lateral_force_ratio=lateral_ratio,
            tilt_moment_ratio=moment_ratio,
            health=health,
            recovery_requested=reason is not None,
            reason=reason,
        )
        return self.last
