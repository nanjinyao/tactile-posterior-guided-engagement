from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PosteriorSearchPlan:
    direction_xy: np.ndarray
    transport_distance_m: float
    transport_target_xy: np.ndarray
    spiral_max_radius_m: float
    spiral_pitch_m_per_rev: float
    posterior_coverage: float
    transport_quantile: float
    direction_quality: float
    reverse_probability: float
    projected_q10_m: float
    projected_q50_m: float
    projected_q90_m: float
    transport_distance_raw_m: float
    transport_distance_cap_m: float
    transport_mode: str


def plan_posterior_guided_search(
    current_xy: np.ndarray,
    center_samples_xy: np.ndarray,
    direction_xy: np.ndarray,
    effective_capture_radius_m: float,
    *,
    transport_quantile: float | None = None,
    posterior_coverage: float = 0.95,
    direction_quality: float = 1.0,
    min_transport_quantile: float = 0.10,
    max_transport_quantile: float = 0.25,
    min_transport_fraction: float = 0.35,
    max_transport_fraction: float = 0.65,
    capture_margin_fraction: float = 0.25,
    reverse_probability_limit: float = 0.20,
    max_transport_m: float | None = None,
    pitch_safety_factor: float = 0.80,
) -> PosteriorSearchPlan:
    """Plan a conservative transport followed by a finite outward search.

    ``center_samples_xy`` should represent the posterior formed from the
    measured visual prior, conical direction likelihood, and CAD feasibility.
    The transport consumes a quality-adaptive lower posterior quantile along
    the learned direction. The local spiral retains a dense finite-width search
    opportunity over residual hypotheses. Pitch limits adjacent-turn spacing,
    but the finite path has no outer-boundary padding and therefore is not a
    complete-domain coverage certificate.
    """

    current = np.asarray(current_xy, dtype=float)
    samples = np.asarray(center_samples_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    if current.shape != (2,):
        raise ValueError("current_xy must contain 2 values")
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < 2:
        raise ValueError("center_samples_xy must have shape (N, 2), N >= 2")
    if direction.shape != (2,):
        raise ValueError("direction_xy must contain 2 values")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(samples)) or not np.all(np.isfinite(direction)):
        raise ValueError("all position and direction inputs must be finite")
    if effective_capture_radius_m <= 0.0:
        raise ValueError("effective_capture_radius_m must be positive")
    if transport_quantile is not None and not 0.0 < transport_quantile < 0.5:
        raise ValueError("transport_quantile must be between 0 and 0.5")
    if not 0.5 < posterior_coverage < 1.0:
        raise ValueError("posterior_coverage must be between 0.5 and 1")
    if not 0.0 <= direction_quality <= 1.0:
        raise ValueError("direction_quality must be in [0, 1]")
    if not 0.0 < min_transport_quantile <= max_transport_quantile < 0.5:
        raise ValueError("transport quantile bounds must satisfy 0 < min <= max < 0.5")
    if not 0.0 < min_transport_fraction <= max_transport_fraction <= 1.0:
        raise ValueError("transport fraction bounds must satisfy 0 < min <= max <= 1")
    if not 0.0 <= capture_margin_fraction <= 1.0:
        raise ValueError("capture_margin_fraction must be in [0, 1]")
    if not 0.0 <= reverse_probability_limit < 0.5:
        raise ValueError("reverse_probability_limit must be in [0, 0.5)")
    if max_transport_m is not None and max_transport_m <= 0.0:
        raise ValueError("max_transport_m must be positive when provided")
    if not 0.0 < pitch_safety_factor <= 1.0:
        raise ValueError("pitch_safety_factor must be in (0, 1]")

    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-12:
        raise ValueError("direction_xy must be non-zero")
    unit = direction / norm
    deltas = samples - current
    projected = deltas @ unit
    radial = np.linalg.norm(deltas, axis=1)

    reverse_probability = float(np.mean(projected <= 0.0))
    # Ensemble concentration and seed agreement enter as a continuous risk
    # scale. Posterior reverse mass further discounts apparent confidence.
    effective_quality = float(
        np.clip(direction_quality * (1.0 - 2.0 * reverse_probability), 0.0, 1.0)
    )
    selected_quantile = (
        float(transport_quantile)
        if transport_quantile is not None
        else min_transport_quantile
        + effective_quality * (max_transport_quantile - min_transport_quantile)
    )
    projected_q10 = float(np.quantile(projected, 0.10))
    projected_q50 = float(np.quantile(projected, 0.50))
    projected_q90 = float(np.quantile(projected, 0.90))
    projected_bound = float(np.quantile(projected, selected_quantile))
    transport_raw = max(
        0.0,
        projected_bound - capture_margin_fraction * effective_capture_radius_m,
    )
    posterior_radius = float(np.quantile(radial, posterior_coverage))
    transport_fraction = min_transport_fraction + effective_quality * (
        max_transport_fraction - min_transport_fraction
    )
    distance_cap = transport_fraction * posterior_radius
    if max_transport_m is not None:
        distance_cap = min(distance_cap, max_transport_m)
    if reverse_probability >= reverse_probability_limit:
        transport_distance = 0.0
        transport_mode = "reverse_mass_fallback"
    elif transport_raw <= 0.0:
        transport_distance = 0.0
        transport_mode = "nonpositive_projection_fallback"
    else:
        transport_distance = min(transport_raw, distance_cap)
        transport_mode = "continuous_risk_bounded"
    target = current + transport_distance * unit

    remaining_radius = np.linalg.norm(samples - target, axis=1)
    covered_radius = float(np.quantile(remaining_radius, posterior_coverage))
    spiral_max_radius = max(0.0, covered_radius - effective_capture_radius_m)
    spiral_pitch = 2.0 * effective_capture_radius_m * pitch_safety_factor

    return PosteriorSearchPlan(
        direction_xy=unit.copy(),
        transport_distance_m=transport_distance,
        transport_target_xy=target.copy(),
        spiral_max_radius_m=spiral_max_radius,
        spiral_pitch_m_per_rev=spiral_pitch,
        posterior_coverage=posterior_coverage,
        transport_quantile=selected_quantile,
        direction_quality=effective_quality,
        reverse_probability=reverse_probability,
        projected_q10_m=projected_q10,
        projected_q50_m=projected_q50,
        projected_q90_m=projected_q90,
        transport_distance_raw_m=transport_raw,
        transport_distance_cap_m=distance_cap,
        transport_mode=transport_mode,
    )
