from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CaptureYawAssistConfig:
    """Optional local-search yaw dither for hex clocking and chamfer capture.

    The assist is intentionally disabled by default. It starts with the local
    spiral and may continue through capture confirmation and insertion.
    """

    enabled: bool = False
    amplitude_deg: float = 0.5
    frequency_hz: float = 1.5
    ramp_time_s: float = 0.25
    soft_tau_z_limit_mnm: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "amplitude_deg",
            "frequency_hz",
            "ramp_time_s",
            "soft_tau_z_limit_mnm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.amplitude_deg > 3.0:
            raise ValueError("amplitude_deg exceeds the bounded micro-yaw range")

    def command_rad(self, elapsed_s: float, *, aborted: bool = False) -> float:
        if not self.enabled or aborted:
            return 0.0
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("elapsed_s must be finite and non-negative")
        ramp_fraction = min(1.0, elapsed_s / self.ramp_time_s)
        envelope = ramp_fraction * ramp_fraction * (3.0 - 2.0 * ramp_fraction)
        return (
            math.radians(self.amplitude_deg)
            * envelope
            * math.sin(2.0 * math.pi * self.frequency_hz * elapsed_s)
        )

    def exceeds_soft_tau_z(self, measured_tau_z_nm: float) -> bool:
        if not self.enabled:
            return False
        if not math.isfinite(measured_tau_z_nm):
            return True
        return 1000.0 * abs(measured_tau_z_nm) >= self.soft_tau_z_limit_mnm
