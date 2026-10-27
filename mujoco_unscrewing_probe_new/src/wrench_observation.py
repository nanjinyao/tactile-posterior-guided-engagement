from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


def _vec6(values: tuple[float, ...], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain six finite values")
    return vector


@dataclass(frozen=True)
class WrenchObservationConfig:
    """Pre-hardware engineering prior for a six-axis F/T observation chain.

    Force entries use N and torque entries use N m. The values are intentionally
    modest: they exercise bias, colored noise, frame/cross-axis error, and delay
    without pretending that a specific real sensor has already been measured.
    """

    seed: int = 20260721
    sample_dt_s: float = 0.0005
    delay_s: float = 0.004
    bias_std: tuple[float, ...] = (0.06, 0.06, 0.08, 0.00030, 0.00030, 0.00020)
    white_std: tuple[float, ...] = (0.020, 0.020, 0.030, 0.00010, 0.00010, 0.00008)
    drift_std_per_sqrt_s: tuple[float, ...] = (
        0.004,
        0.004,
        0.006,
        0.000025,
        0.000025,
        0.000020,
    )
    colored_rho: float = 0.92
    cross_axis_std: float = 0.004
    gain_std: float = 0.006

    def __post_init__(self) -> None:
        if self.sample_dt_s <= 0.0 or self.delay_s < 0.0:
            raise ValueError("sample_dt_s must be positive and delay_s non-negative")
        if not 0.0 <= self.colored_rho < 1.0:
            raise ValueError("colored_rho must be in [0, 1)")
        for name in ("bias_std", "white_std", "drift_std_per_sqrt_s"):
            if np.any(_vec6(getattr(self, name), name) < 0.0):
                raise ValueError(f"{name} must be non-negative")
        if self.cross_axis_std < 0.0 or self.gain_std < 0.0:
            raise ValueError("cross_axis_std and gain_std must be non-negative")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["delay_steps"] = int(round(self.delay_s / self.sample_dt_s))
        value["claim_boundary"] = (
            "pre-hardware engineering prior; replace ranges with measured bias, "
            "Allan/covariance, frame-calibration, and latency distributions"
        )
        return value


class WrenchObservationModel:
    """Stateful deterministic sensor model applied after MuJoCo contact physics."""

    def __init__(self, config: WrenchObservationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.bias = self.rng.normal(0.0, _vec6(config.bias_std, "bias_std"))
        gains = 1.0 + self.rng.normal(0.0, config.gain_std, size=6)
        coupling = self.rng.normal(0.0, config.cross_axis_std, size=(6, 6))
        np.fill_diagonal(coupling, 0.0)
        # Keep force/torque unit families separate. Lever-arm effects already
        # belong in the physical sensor transform and should be calibrated.
        coupling[:3, 3:] = 0.0
        coupling[3:, :3] = 0.0
        self.calibration = np.diag(gains) + coupling
        self.drift = np.zeros(6, dtype=float)
        self.colored = np.zeros(6, dtype=float)
        self.white_std = _vec6(config.white_std, "white_std")
        self.drift_step_std = _vec6(
            config.drift_std_per_sqrt_s, "drift_std_per_sqrt_s"
        ) * np.sqrt(config.sample_dt_s)
        self.delay_steps = int(round(config.delay_s / config.sample_dt_s))
        self.history: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)
        # A delay line represents elapsed sensor time, not the number of times
        # a caller happens to inspect a wrench.  Seed it with the zero-load
        # initial condition so that a non-zero first sample is delayed too.
        for _ in range(self.delay_steps + 1):
            self.history.append(np.zeros(6, dtype=float))
        self._sample_count = 0
        self._latest_observation = np.zeros(6, dtype=float)

    @property
    def sample_count(self) -> int:
        """Number of physical-time samples that advanced noise/delay state."""

        return self._sample_count

    @property
    def latest_observation(self) -> np.ndarray:
        """Return the most recent sample without advancing sensor time."""

        return self._latest_observation.copy()

    def observe(self, physical_wrench: np.ndarray) -> np.ndarray:
        """Advance the sensor model by exactly one configured sample period."""

        physical = np.asarray(physical_wrench, dtype=float)
        if physical.shape != (6,) or not np.all(np.isfinite(physical)):
            raise ValueError("physical_wrench must contain six finite values")
        self.history.append(physical.copy())
        delayed = self.history[0]
        self.drift += self.rng.normal(0.0, self.drift_step_std)
        innovation = self.rng.normal(0.0, self.white_std)
        rho = self.config.colored_rho
        self.colored = rho * self.colored + np.sqrt(1.0 - rho * rho) * innovation
        observation = self.calibration @ delayed + self.bias + self.drift + self.colored
        self._sample_count += 1
        self._latest_observation = observation.copy()
        return observation
