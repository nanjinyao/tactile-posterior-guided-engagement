from __future__ import annotations

from dataclasses import dataclass, replace
import math

from src.cartesian_admittance import AdmittanceConfig
from src.funnel_guard import FunnelGuardConfig


KEY_AF_M_BY_SIZE = {
    "M8": 0.006,
    "M10": 0.008,
    "M12": 0.010,
}
REFERENCE_SIZE = "M8"
REFERENCE_BIT_RADIUS_M = KEY_AF_M_BY_SIZE[REFERENCE_SIZE] / math.sqrt(3.0)


@dataclass(frozen=True)
class CrossSizeCommissioning:
    """Deterministic starting controller for one independently tested size.

    This is a geometry-scaled commissioning *interface*, not evidence that the
    parameters work for a size.  Each returned configuration must still pass a
    size-specific safety and full-loop campaign before it is used in a claim.
    """

    size: str
    bit_radius_m: float
    bit_radius_scale: float
    response_time_scale: float
    commissioning_version: str
    funnel_guard: FunnelGuardConfig
    admittance: AdmittanceConfig
    oracle_transport_soft_escape_recovery: bool
    oracle_center_tolerance_capture_fraction: float
    claim_boundary: str = (
        "v2 geometry/dynamics-scaled commissioning start only; independent size "
        "validation required"
    )


def _scaled(values: tuple[float, ...], divisor: float) -> tuple[float, ...]:
    return tuple(float(value) / divisor for value in values)


def _scaled_translation_rotation(
    values: tuple[float, ...],
    *,
    translation_divisor: float,
    rotation_divisor: float,
) -> tuple[float, ...]:
    """Scale Cartesian and angular entries with separate physical exponents."""

    if len(values) != 6:
        raise ValueError("a Cartesian admittance vector must contain six values")
    return tuple(
        float(value)
        / (translation_divisor if index < 3 else rotation_divisor)
        for index, value in enumerate(values)
    )


def commissioning_for_bit_radius(
    bit_radius_m: float,
    *,
    size: str,
    base_funnel: FunnelGuardConfig | None = None,
    base_admittance: AdmittanceConfig | None = None,
) -> CrossSizeCommissioning:
    """Return the v2 geometry/dynamics-scaled commissioning starting point.

    Let ``s = r_bit / r_bit,M8``.  The source meshes keep the same 0.5 kg bit
    mass across M8/M10/M12, so a stiffness reduction ``K_t ~ 1/s`` gives a
    translational response time ``T_t ~ sqrt(M/K_t) ~ sqrt(s)`` rather than
    ``T ~ s``.  v2 therefore uses ``sqrt(s)`` consistently for capture,
    handoff, recovery, descent velocity, and translational rate limits.

    Rotation is separated from translation.  A similar angular error acts at
    a lever arm proportional to ``s`` and produces contact moment proportional
    to ``s**2``.  The commanded rotational stiffness is therefore scaled by
    ``1/s**2`` and its critical-damping trend by ``1/s``.  This is a
    commissioning hypothesis, not a success guarantee.

    Raw hard-wrench thresholds and independent safety-supervisor thresholds
    are copied unchanged: scaling never turns a hard stop into an accepted
    contact.
    """

    radius = float(bit_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("bit_radius_m must be finite and positive")
    normalized_size = str(size).strip().upper()
    if not normalized_size:
        raise ValueError("size must be non-empty")
    scale = radius / REFERENCE_BIT_RADIUS_M
    if scale < 1.0 - 1.0e-12:
        raise ValueError("this commissioning rule is defined for M8-or-larger tools")

    base_guard = base_funnel or FunnelGuardConfig()
    base_outer = base_admittance or AdmittanceConfig()
    response_time_scale = math.sqrt(scale)

    guard = replace(
        base_guard,
        # Keep every commanded transition around the recovery loop on the same
        # response-time scale.  v1 slowed capture/handoff by s but left the
        # recovery unload/retract/retouch at M8 timing, creating a discontinuous
        # terminal-retry schedule for larger tools.
        capture_force_release_s=(
            base_guard.capture_force_release_s * response_time_scale
        ),
        insertion_handoff_s=(
            base_guard.insertion_handoff_s * response_time_scale
        ),
        recovery_unload_s=base_guard.recovery_unload_s * response_time_scale,
        recovery_retract_s=base_guard.recovery_retract_s * response_time_scale,
        recovery_retouch_s=base_guard.recovery_retouch_s * response_time_scale,
        descent_speed_m_s=base_guard.descent_speed_m_s / response_time_scale,
        # A rejected dwell after first entry into the CAD terminal depth band
        # is a verifier retry, not evidence that the tool left the socket.
        # Retract/re-touch there created the measured second-impact failure in
        # the M10 development audit.  Larger-size v2 holds the already valid
        # axial command and restarts only the verifier.  The ordinary guarded
        # recovery remains active when the terminal set was never reached.
        hold_terminal_set_on_retry=(
            True
            if scale > 1.0 + 1.0e-12
            else base_guard.hold_terminal_set_on_retry
        ),
        bumpless_terminal_entry_rebase=(
            True
            if scale > 1.0 + 1.0e-12
            else base_guard.bumpless_terminal_entry_rebase
        ),
        axial_position_hold_in_terminal_set=(
            True
            if scale > 1.0 + 1.0e-12
            else base_guard.axial_position_hold_in_terminal_set
        ),
        planar_stiffness_n_m=base_guard.planar_stiffness_n_m / scale,
        planar_damping_n_s_m=(
            base_guard.planar_damping_n_s_m / response_time_scale
        ),
        shallow_tilt_stiffness_nm_rad=(
            base_guard.shallow_tilt_stiffness_nm_rad / (scale * scale)
        ),
        shallow_tilt_damping_nm_s_rad=(
            base_guard.shallow_tilt_damping_nm_s_rad / scale
        ),
        deep_tilt_stiffness_nm_rad=(
            base_guard.deep_tilt_stiffness_nm_rad / (scale * scale)
        ),
        deep_tilt_damping_nm_s_rad=(
            base_guard.deep_tilt_damping_nm_s_rad / scale
        ),
        yaw_stiffness_nm_rad=(
            base_guard.yaw_stiffness_nm_rad / (scale * scale)
        ),
        yaw_damping_nm_s_rad=base_guard.yaw_damping_nm_s_rad / scale,
    )
    admittance = replace(
        base_outer,
        damping=_scaled_translation_rotation(
            base_outer.damping,
            translation_divisor=response_time_scale,
            rotation_divisor=scale,
        ),
        stiffness=_scaled_translation_rotation(
            base_outer.stiffness,
            translation_divisor=scale,
            rotation_divisor=scale * scale,
        ),
        max_velocity=_scaled_translation_rotation(
            base_outer.max_velocity,
            translation_divisor=response_time_scale,
            rotation_divisor=scale,
        ),
        max_acceleration=_scaled_translation_rotation(
            base_outer.max_acceleration,
            translation_divisor=scale,
            rotation_divisor=scale * scale,
        ),
        # Hard thresholds are independent of nominal compliance and speed.
        hard_wrench=tuple(base_outer.hard_wrench),
    )
    if tuple(admittance.hard_wrench) != tuple(base_outer.hard_wrench):
        raise RuntimeError("cross-size commissioning must not relax hard-wrench limits")

    return CrossSizeCommissioning(
        size=normalized_size,
        bit_radius_m=radius,
        bit_radius_scale=scale,
        response_time_scale=response_time_scale,
        commissioning_version=(
            "v2_sqrt_time_separate_rotation_lever_bumpless_terminal_hold_"
            "oracle_soft_escape_resume"
        ),
        funnel_guard=guard,
        admittance=admittance,
        # The privileged straight-to-center arm is an execution lower bound,
        # not a search policy.  If its unchanged soft wrench guard interrupts
        # the ray, larger-tool commissioning explicitly unloads, rebases at the
        # measured tip and resumes the remaining oracle ray.  M8 and every
        # non-oracle policy retain the historical behavior.
        oracle_transport_soft_escape_recovery=bool(
            scale > 1.0 + 1.0e-12
        ),
        # Require measured tracking to enter a strict inner quarter of the CAD
        # capture radius before the oracle is allowed to start guarded descent.
        # This is an execution tolerance, not a relaxed capture/verifier gate.
        oracle_center_tolerance_capture_fraction=0.25,
    )


def commissioning_for_size(
    size: str,
    *,
    base_funnel: FunnelGuardConfig | None = None,
    base_admittance: AdmittanceConfig | None = None,
) -> CrossSizeCommissioning:
    """Return the deterministic starting configuration for M8, M10, or M12."""

    normalized_size = str(size).strip().upper()
    try:
        key_af_m = KEY_AF_M_BY_SIZE[normalized_size]
    except KeyError as exc:
        raise ValueError(
            f"unsupported size {size!r}; expected one of {sorted(KEY_AF_M_BY_SIZE)}"
        ) from exc
    return commissioning_for_bit_radius(
        key_af_m / math.sqrt(3.0),
        size=normalized_size,
        base_funnel=base_funnel,
        base_admittance=base_admittance,
    )


def funnel_guard_for_size(
    size: str,
    *,
    base: FunnelGuardConfig | None = None,
) -> FunnelGuardConfig:
    """Convenience interface returning only the size-conditioned funnel guard."""

    return commissioning_for_size(size, base_funnel=base).funnel_guard
