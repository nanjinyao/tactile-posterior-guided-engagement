from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EntropyGridConfig:
    """Configuration for the Chen-inspired planar belief baseline."""

    belief_spacing_m: float = 0.00025
    action_spacing_m: float = 0.00075
    likelihood_floor: float = 0.01
    # This is a real-system watchdog, not the nominal stopping rule. Normal
    # termination is capture, posterior contraction, or exhausted information
    # gain; the active-time and wrench guards remain independent safety limits.
    max_probes: int = 64
    credible_radius_stop_m: float = 0.00045
    min_information_gain_nats: float = 0.005
    torque_direction_bins: int = 8
    low_lever_threshold_m: float = 0.00025
    support_centroid_scale: float = 0.75
    quadrature_points: int = 81
    radial_lookup_points: int = 256
    # ``None`` uses the CAD-derived conservative capture radius.  The action
    # lattice is padded beyond the visual-center support so clipping the
    # lattice at the circular prior boundary cannot create uncovered slivers.
    # The effective padding is always at least half a lattice-cell diagonal.
    action_padding_m: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "belief_spacing_m",
            "action_spacing_m",
            "credible_radius_stop_m",
            "min_information_gain_nats",
            "low_lever_threshold_m",
            "support_centroid_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.likelihood_floor < 0.1:
            raise ValueError("likelihood_floor must be in [0, 0.1)")
        if self.max_probes <= 0:
            raise ValueError("max_probes must be positive")
        if self.torque_direction_bins < 4:
            raise ValueError("torque_direction_bins must be at least four")
        if self.quadrature_points < 21 or self.radial_lookup_points < 32:
            raise ValueError("contact-centroid lookup is too coarse")
        if self.action_padding_m is not None and (
            not math.isfinite(float(self.action_padding_m))
            or float(self.action_padding_m) < 0.0
        ):
            raise ValueError("action_padding_m must be finite and nonnegative")


@dataclass(frozen=True)
class HybridProbeConfig:
    """Value-of-information gate for Proposed+ residual probing."""

    safety_probe_ceiling: int = 64
    fixed_contact_cost_nats: float = 0.08
    travel_cost_nats_per_m: float = 20.0
    minimum_net_utility_nats: float = 0.05
    minimum_coverage_utility_nats: float = 0.0
    posterior_kernel_bandwidth_m: float = 0.00035
    residual_spiral_trigger_m: float = 0.00120
    posterior_sample_count: int = 4096
    minimum_credible_radius_reduction_fraction: float = 0.05
    maximum_residual_radius_ratio: float = 1.10
    maximum_consecutive_noncontracting_updates: int = 2
    continuous_contact_path: bool = True
    minimum_learned_prior_weight: float = 0.15
    maximum_learned_prior_weight: float = 0.75
    information_gain_weight: float = 0.35
    capture_probability_reward_nats: float = 1.50
    coverage_information_gain_weight: float = 0.05
    coverage_capture_probability_reward_nats: float = 2.25
    capture_miss_inside_likelihood: float = 0.20
    capture_miss_outside_likelihood: float = 0.995
    predictive_probability_reference: float = 0.15
    minimum_likelihood_temperature: float = 0.20
    local_spiral_fallback_enabled: bool = False
    fallback_transition_overhead_s: float = 1.15
    fallback_return_position_error_bound_m: float = 0.00025
    fallback_candidate_contingency_s: float = 6.0
    terminal_execution_reserve_s: float = 12.0

    def __post_init__(self) -> None:
        if self.safety_probe_ceiling <= 0:
            raise ValueError("safety_probe_ceiling must be positive")
        for name in (
            "fixed_contact_cost_nats",
            "travel_cost_nats_per_m",
            "minimum_net_utility_nats",
            "minimum_coverage_utility_nats",
            "posterior_kernel_bandwidth_m",
            "residual_spiral_trigger_m",
            "fallback_transition_overhead_s",
            "fallback_return_position_error_bound_m",
            "fallback_candidate_contingency_s",
            "terminal_execution_reserve_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.posterior_kernel_bandwidth_m <= 0.0:
            raise ValueError("posterior_kernel_bandwidth_m must be positive")
        if self.terminal_execution_reserve_s <= 0.0:
            raise ValueError("terminal_execution_reserve_s must be positive")
        if self.posterior_sample_count < 128:
            raise ValueError("posterior_sample_count is too small")
        if not 0.0 <= self.minimum_credible_radius_reduction_fraction < 1.0:
            raise ValueError(
                "minimum_credible_radius_reduction_fraction must be in [0, 1)"
            )
        if self.maximum_residual_radius_ratio < 1.0:
            raise ValueError("maximum_residual_radius_ratio must be at least one")
        if self.maximum_consecutive_noncontracting_updates <= 0:
            raise ValueError(
                "maximum_consecutive_noncontracting_updates must be positive"
            )
        if not (
            0.0 <= self.minimum_learned_prior_weight
            <= self.maximum_learned_prior_weight
            < 1.0
        ):
            raise ValueError("learned-prior mixture bounds must satisfy 0 <= min <= max < 1")
        for name in (
            "information_gain_weight",
            "capture_probability_reward_nats",
            "coverage_information_gain_weight",
            "coverage_capture_probability_reward_nats",
            "predictive_probability_reference",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(
                getattr(self, name)
            ) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.predictive_probability_reference <= 0.0:
            raise ValueError("predictive_probability_reference must be positive")
        if not (
            0.0 < self.capture_miss_inside_likelihood
            < self.capture_miss_outside_likelihood
            <= 1.0
        ):
            raise ValueError(
                "capture-miss likelihoods must satisfy 0 < inside < outside <= 1"
            )
        if not 0.0 < self.minimum_likelihood_temperature <= 1.0:
            raise ValueError("minimum_likelihood_temperature must be in (0, 1]")


@dataclass(frozen=True)
class EntropyAction:
    index: int
    xy_m: np.ndarray
    expected_information_gain_nats: float
    predicted_outcome_probabilities: list[float]
    travel_distance_m: float = 0.0
    net_utility_nats: float | None = None
    segment_capture_probability: float = 0.0
    information_gain_weight: float = 1.0
    capture_probability_reward_nats: float = 0.0


@dataclass(frozen=True)
class EntropyUpdate:
    action_index: int
    action_xy_m: np.ndarray
    observation_category: int
    information_gain_nats: float
    entropy_nats: float
    credible_radius_95_m: float
    posterior_mean_xy_m: np.ndarray
    predictive_probability: float = 1.0
    likelihood_temperature: float = 1.0


@dataclass(frozen=True)
class CaptureMissUpdate:
    segment_start_xy_m: np.ndarray
    segment_end_xy_m: np.ndarray
    covered_probability_before: float
    entropy_nats: float
    credible_radius_95_m: float


@dataclass(frozen=True)
class DiskCoverageCertificate:
    """Auditable coverage certificate for a padded square action lattice.

    ``analytic_worst_case_gap_m`` is a continuum guarantee, not a sampled
    estimate.  It applies when the actual action set contains the complete
    square lattice through ``support_radius + h/sqrt(2)``.  The dense-disk and
    boundary values independently audit the concrete generated node set.
    """

    support_radius_m: float
    action_support_radius_m: float
    action_padding_m: float
    lattice_spacing_m: float
    effective_capture_radius_m: float
    action_node_count: int
    analytic_worst_case_gap_m: float
    dense_worst_gap_m: float
    boundary_worst_gap_m: float
    observed_worst_gap_m: float
    dense_sample_count: int
    boundary_sample_count: int
    required_lattice_node_count: int
    complete_required_lattice: bool
    certified: bool


def weighted_mass_radius(
    points_xy_m: np.ndarray,
    weights: np.ndarray,
    center_xy_m: np.ndarray,
    *,
    mass: float = 0.95,
) -> float:
    """Return the smallest center-fixed radius containing ``mass`` weight.

    This is the exact weighted quantile of the discrete center belief.  In
    particular, it makes no Gaussian or elliptical-posterior assumption.
    """

    points = np.asarray(points_xy_m, dtype=float)
    probability = np.asarray(weights, dtype=float)
    center = np.asarray(center_xy_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points_xy_m must have shape (N, 2)")
    if probability.shape != (len(points),):
        raise ValueError("weights must have shape (N,)")
    if center.shape != (2,):
        raise ValueError("center_xy_m must contain two values")
    if not (
        np.all(np.isfinite(points))
        and np.all(np.isfinite(probability))
        and np.all(np.isfinite(center))
    ):
        raise ValueError("mass-radius inputs must be finite")
    if np.any(probability < 0.0):
        raise ValueError("weights must be nonnegative")
    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1]")
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("weights must contain positive mass")

    distance = np.linalg.norm(points - center[None, :], axis=1)
    order = np.argsort(distance, kind="stable")
    cumulative = np.cumsum(probability[order]) / total
    index = min(int(np.searchsorted(cumulative, mass, side="left")), len(order) - 1)
    return float(distance[order[index]])


def disk_grid(center_xy_m: np.ndarray, radius_m: float, spacing_m: float) -> np.ndarray:
    center = np.asarray(center_xy_m, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_m must contain two finite values")
    if radius_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("radius_m and spacing_m must be positive")
    count = int(math.ceil(radius_m / spacing_m))
    coordinate = np.arange(-count, count + 1, dtype=float) * spacing_m
    offset = np.array(
        [
            (x, y)
            for x in coordinate
            for y in coordinate
            if x * x + y * y <= radius_m * radius_m + 1.0e-15
        ],
        dtype=float,
    )
    return center[None, :] + offset


def _nearest_node_gap(
    query_xy_m: np.ndarray,
    action_xy_m: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> float:
    """Maximum query-to-node distance without allocating one giant matrix."""

    query = np.asarray(query_xy_m, dtype=float)
    action = np.asarray(action_xy_m, dtype=float)
    if query.ndim != 2 or query.shape[1] != 2 or len(query) == 0:
        raise ValueError("query_xy_m must have shape (N, 2)")
    if action.ndim != 2 or action.shape[1] != 2 or len(action) == 0:
        raise ValueError("action_xy_m must have shape (M, 2)")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(action)):
        raise ValueError("coverage points must be finite")
    worst = 0.0
    for start in range(0, len(query), chunk_size):
        delta = query[start : start + chunk_size, None, :] - action[None, :, :]
        nearest = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
        worst = max(worst, float(np.max(nearest)))
    return worst


def certify_disk_grid_coverage(
    action_xy_m: np.ndarray,
    center_xy_m: np.ndarray,
    *,
    support_radius_m: float,
    lattice_spacing_m: float,
    effective_capture_radius_m: float,
    dense_subdivisions_per_cell: int = 8,
    boundary_sample_count: int = 4096,
) -> DiskCoverageCertificate:
    """Certify capture coverage of a finite circular visual support.

    The analytic part checks that the *actual* action set contains every node
    of the center-aligned square lattice through radius
    ``support_radius + h/sqrt(2)``.  Any point in the declared visual disk then
    has its nearest infinite-lattice node in the finite action set, hence its
    distance to an action node is at most ``h/sqrt(2)``.  Certification also
    requires this continuum bound not to exceed the physical capture radius.

    Dense Cartesian samples and deterministic boundary samples are reported as
    an independent numerical audit; they do not replace the analytic bound.
    """

    action = np.asarray(action_xy_m, dtype=float)
    center = np.asarray(center_xy_m, dtype=float)
    if action.ndim != 2 or action.shape[1] != 2 or len(action) == 0:
        raise ValueError("action_xy_m must have shape (N, 2)")
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy_m must contain two finite values")
    for name, value in (
        ("support_radius_m", support_radius_m),
        ("lattice_spacing_m", lattice_spacing_m),
        ("effective_capture_radius_m", effective_capture_radius_m),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if dense_subdivisions_per_cell <= 0:
        raise ValueError("dense_subdivisions_per_cell must be positive")
    if boundary_sample_count < 32:
        raise ValueError("boundary_sample_count must be at least 32")

    half_diagonal = float(lattice_spacing_m) / math.sqrt(2.0)
    required_radius = float(support_radius_m) + half_diagonal
    required = disk_grid(center, required_radius, float(lattice_spacing_m))

    # Compare integer lattice coordinates instead of raw floats.  The allclose
    # checks reject off-lattice nodes before the set-containment proof is used.
    action_lattice = (action - center[None, :]) / float(lattice_spacing_m)
    required_lattice = (required - center[None, :]) / float(lattice_spacing_m)
    rounded_action = np.rint(action_lattice).astype(np.int64)
    rounded_required = np.rint(required_lattice).astype(np.int64)
    action_is_lattice = bool(
        np.allclose(action_lattice, rounded_action, atol=1.0e-8, rtol=0.0)
    )
    actual_keys = {tuple(value) for value in rounded_action.tolist()}
    required_keys = {tuple(value) for value in rounded_required.tolist()}
    complete_required_lattice = action_is_lattice and required_keys.issubset(actual_keys)

    dense_spacing = float(lattice_spacing_m) / dense_subdivisions_per_cell
    dense_count = int(math.ceil(float(support_radius_m) / dense_spacing))
    coordinate = np.arange(-dense_count, dense_count + 1, dtype=float) * dense_spacing
    dense_offset = np.asarray(
        [
            (x, y)
            for x in coordinate
            for y in coordinate
            if x * x + y * y <= float(support_radius_m) ** 2 + 1.0e-15
        ],
        dtype=float,
    )
    dense_query = center[None, :] + dense_offset
    angle = 2.0 * math.pi * np.arange(boundary_sample_count) / boundary_sample_count
    boundary_query = center[None, :] + float(support_radius_m) * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )
    dense_gap = _nearest_node_gap(dense_query, action)
    boundary_gap = _nearest_node_gap(boundary_query, action)
    radial_extent = float(np.max(np.linalg.norm(action - center[None, :], axis=1)))
    action_padding = radial_extent - float(support_radius_m)
    certified = bool(
        complete_required_lattice
        and half_diagonal <= float(effective_capture_radius_m) + 1.0e-12
        and max(dense_gap, boundary_gap)
        <= float(effective_capture_radius_m) + 1.0e-12
    )
    return DiskCoverageCertificate(
        support_radius_m=float(support_radius_m),
        action_support_radius_m=radial_extent,
        action_padding_m=action_padding,
        lattice_spacing_m=float(lattice_spacing_m),
        effective_capture_radius_m=float(effective_capture_radius_m),
        action_node_count=len(action),
        analytic_worst_case_gap_m=half_diagonal,
        dense_worst_gap_m=dense_gap,
        boundary_worst_gap_m=boundary_gap,
        observed_worst_gap_m=max(dense_gap, boundary_gap),
        dense_sample_count=len(dense_query),
        boundary_sample_count=boundary_sample_count,
        required_lattice_node_count=len(required),
        complete_required_lattice=complete_required_lattice,
        certified=certified,
    )


def normalized_torque_lever(wrench: np.ndarray, force_floor_n: float = 0.50) -> np.ndarray:
    value = np.asarray(wrench, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("wrench must contain six finite values")
    if force_floor_n <= 0.0:
        raise ValueError("force_floor_n must be positive")
    force_norm = max(force_floor_n, float(np.linalg.norm(value[:3])))
    return value[3:5] / force_norm


def torque_category_from_wrench(
    wrench: np.ndarray,
    *,
    low_lever_threshold_m: float,
    direction_bins: int,
) -> tuple[int, np.ndarray]:
    """Quantize the signed F/T moment arm into low + angular categories.

    Category zero is a low-moment/near-symmetric contact. Categories 1..B are
    angular bins of ``[tau_x, tau_y] / ||F||``. Keeping the sign is essential:
    the inner socket edge and the outer screw-head edge generate opposite
    moment directions and cannot be distinguished by moment magnitude alone.
    """

    if low_lever_threshold_m <= 0.0 or direction_bins < 4:
        raise ValueError("invalid torque quantizer")
    lever = normalized_torque_lever(wrench)
    if float(np.linalg.norm(lever)) < low_lever_threshold_m:
        return 0, lever
    angle = math.atan2(float(lever[1]), float(lever[0])) % (2.0 * math.pi)
    index = int(math.floor((angle + math.pi / direction_bins) / (2.0 * math.pi / direction_bins)))
    return 1 + index % direction_bins, lever


def edge_loaded_from_wrench(
    wrench: np.ndarray,
    *,
    lever_threshold_m: float,
    force_floor_n: float = 0.50,
) -> tuple[bool, float]:
    """Legacy magnitude diagnostic retained for the observability audit."""

    lever = normalized_torque_lever(wrench, force_floor_n)
    magnitude = float(np.linalg.norm(lever))
    return magnitude >= lever_threshold_m, magnitude


class EntropyGridBelief:
    """Discrete hole-center belief with information-maximizing F/T probes.

    The likelihood is a CAD-derived support-centroid model.  A circular peg
    approximation is intersected with the known annular screw-head top.  The
    centroid of the supported patch predicts the *signed* normalized torque
    direction.  This is still a simulation-initial model that needs a small
    iiwa/F-T calibration; it never receives the true center online.
    """

    def __init__(
        self,
        visual_anchor_xy_m: np.ndarray,
        visual_sigma_m: float,
        visual_bound_m: float,
        support_inner_radius_m: float,
        support_outer_radius_m: float,
        probe_radius_m: float,
        config: EntropyGridConfig | None = None,
        effective_capture_radius_m: float | None = None,
    ) -> None:
        self.config = config or EntropyGridConfig()
        self.anchor_xy_m = np.asarray(visual_anchor_xy_m, dtype=float).copy()
        if self.anchor_xy_m.shape != (2,) or not np.all(np.isfinite(self.anchor_xy_m)):
            raise ValueError("visual_anchor_xy_m must contain two finite values")
        if visual_sigma_m <= 0.0 or visual_bound_m <= 0.0:
            raise ValueError("visual prior scales must be positive")
        if not 0.0 < support_inner_radius_m < support_outer_radius_m:
            raise ValueError("support radii must define a positive annulus")
        if probe_radius_m <= 0.0:
            raise ValueError("probe_radius_m must be positive")
        self.visual_sigma_m = float(visual_sigma_m)
        self.visual_bound_m = float(visual_bound_m)
        self.support_inner_radius_m = float(support_inner_radius_m)
        self.support_outer_radius_m = float(support_outer_radius_m)
        self.probe_radius_m = float(probe_radius_m)
        inferred_capture_radius_m = (
            self.support_inner_radius_m
            - 0.5 * math.sqrt(3.0) * self.probe_radius_m
        )
        if effective_capture_radius_m is not None and (
            not math.isfinite(float(effective_capture_radius_m))
            or float(effective_capture_radius_m) <= 0.0
        ):
            raise ValueError("effective_capture_radius_m must be finite and positive")
        self.effective_capture_radius_m = (
            float(effective_capture_radius_m)
            if effective_capture_radius_m is not None
            else (
                float(inferred_capture_radius_m)
                if inferred_capture_radius_m > 0.0
                else None
            )
        )
        self.candidate_xy_m = disk_grid(
            self.anchor_xy_m, self.visual_bound_m, self.config.belief_spacing_m
        )
        requested_padding_m = (
            float(self.config.action_padding_m)
            if self.config.action_padding_m is not None
            else (
                float(self.effective_capture_radius_m)
                if self.effective_capture_radius_m is not None
                else 0.0
            )
        )
        # Padding by h/sqrt(2) is the finite-disk condition needed for the
        # nearest node of the infinite square lattice to remain available at
        # every point on the circular prior boundary.
        self.action_padding_m = max(
            requested_padding_m,
            self.config.action_spacing_m / math.sqrt(2.0),
        )
        self.action_support_radius_m = self.visual_bound_m + self.action_padding_m
        self.action_xy_m = disk_grid(
            self.anchor_xy_m,
            self.action_support_radius_m,
            self.config.action_spacing_m,
        )
        offset = self.candidate_xy_m - self.anchor_xy_m[None, :]
        log_weight = -0.5 * np.sum(offset * offset, axis=1) / self.visual_sigma_m**2
        log_weight -= float(np.max(log_weight))
        self.weights = np.exp(log_weight)
        self.weights /= float(np.sum(self.weights))
        self.visual_prior_weights = self.weights.copy()
        self._used_actions: set[int] = set()
        self.history: list[EntropyUpdate] = []
        self._category_likelihood = self._build_category_likelihood_template()
        self._radial_distance, self._radial_centroid = self._build_support_centroid_lookup()

    def _build_category_likelihood_template(self) -> np.ndarray:
        """Precompute P(observation | predicted category).

        The previous implementation rebuilt this fixed confusion model once
        per center hypothesis and action. Indexing a small template preserves
        the exact likelihood while removing the dominant Python-loop cost for
        fine CAD grids.
        """

        bins = self.config.torque_direction_bins
        outcomes = bins + 1
        floor = self.config.likelihood_floor
        template = np.full((outcomes, outcomes), floor, dtype=float)
        template[0, 0] += 0.70
        template[0, 1:] += 0.30 / bins
        for category in range(1, outcomes):
            k = category - 1
            template[category, 0] += 0.10
            template[category, 1 + k] += 0.56
            template[category, 1 + (k - 1) % bins] += 0.14
            template[category, 1 + (k + 1) % bins] += 0.14
            remaining = [
                j
                for j in range(bins)
                if j not in {k, (k - 1) % bins, (k + 1) % bins}
            ]
            for j in remaining:
                template[category, 1 + j] += 0.06 / len(remaining)
        template /= np.sum(template, axis=1, keepdims=True)
        return template

    def _build_support_centroid_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.linspace(
            -self.probe_radius_m,
            self.probe_radius_m,
            self.config.quadrature_points,
        )
        local_x, local_y = np.meshgrid(q, q, indexing="xy")
        in_probe = local_x * local_x + local_y * local_y <= self.probe_radius_m**2
        maximum = self.visual_bound_m * 2.0 + max(
            self.config.action_spacing_m,
            self.action_padding_m,
        )
        distance = np.linspace(0.0, maximum, self.config.radial_lookup_points)
        centroid = np.zeros_like(distance)
        for index, radius in enumerate(distance):
            world_x = local_x + radius
            world_radius_sq = world_x * world_x + local_y * local_y
            supported = in_probe & (
                world_radius_sq >= self.support_inner_radius_m**2
            ) & (world_radius_sq <= self.support_outer_radius_m**2)
            centroid[index] = (
                0.0 if not np.any(supported) else float(np.mean(local_x[supported]))
            )
        return distance, centroid

    @property
    def entropy_nats(self) -> float:
        positive = self.weights > 0.0
        return float(-np.sum(self.weights[positive] * np.log(self.weights[positive])))

    @property
    def posterior_mean_xy_m(self) -> np.ndarray:
        return np.sum(self.weights[:, None] * self.candidate_xy_m, axis=0)

    @property
    def posterior_map_xy_m(self) -> np.ndarray:
        return self.candidate_xy_m[int(np.argmax(self.weights))].copy()

    @property
    def credible_radius_95_m(self) -> float:
        return weighted_mass_radius(
            self.candidate_xy_m,
            self.weights,
            self.posterior_mean_xy_m,
            mass=0.95,
        )

    @property
    def action_lattice_gap_bound_m(self) -> float:
        """Continuum nearest-node bound for the padded square lattice."""

        return self.config.action_spacing_m / math.sqrt(2.0)

    @property
    def action_grid_coverage_certified(self) -> bool:
        """Cheap analytic status; use ``certify_action_coverage`` to audit nodes."""

        return bool(
            self.effective_capture_radius_m is not None
            and self.action_padding_m + 1.0e-12 >= self.action_lattice_gap_bound_m
            and self.action_lattice_gap_bound_m
            <= self.effective_capture_radius_m + 1.0e-12
        )

    def certify_action_coverage(
        self,
        *,
        dense_subdivisions_per_cell: int = 8,
        boundary_sample_count: int = 4096,
    ) -> DiskCoverageCertificate:
        """Run the deterministic continuum-plus-numerical coverage audit."""

        if self.effective_capture_radius_m is None:
            raise RuntimeError(
                "effective_capture_radius_m is required for a capture certificate"
            )
        return certify_disk_grid_coverage(
            self.action_xy_m,
            self.anchor_xy_m,
            support_radius_m=self.visual_bound_m,
            lattice_spacing_m=self.config.action_spacing_m,
            effective_capture_radius_m=self.effective_capture_radius_m,
            dense_subdivisions_per_cell=dense_subdivisions_per_cell,
            boundary_sample_count=boundary_sample_count,
        )

    @property
    def should_stop(self) -> bool:
        return bool(
            len(self.history) >= self.config.max_probes
            or self.credible_radius_95_m <= self.config.credible_radius_stop_m
        )

    def _predicted_category(self, action_xy_m: np.ndarray) -> np.ndarray:
        delta = np.asarray(action_xy_m, dtype=float)[None, :] - self.candidate_xy_m
        radius = np.linalg.norm(delta, axis=1)
        radial_centroid = np.interp(
            radius,
            self._radial_distance,
            self._radial_centroid,
            left=0.0,
            right=0.0,
        ) * self.config.support_centroid_scale
        unit = np.zeros_like(delta)
        nonzero = radius > 1.0e-12
        unit[nonzero] = delta[nonzero] / radius[nonzero, None]
        contact_centroid = radial_centroid[:, None] * unit
        # r_contact x Fz: [cy, -cx] for a positive measured support force.
        lever = np.column_stack((contact_centroid[:, 1], -contact_centroid[:, 0]))
        magnitude = np.linalg.norm(lever, axis=1)
        category = np.zeros(len(radius), dtype=int)
        informative = magnitude >= self.config.low_lever_threshold_m
        angle = np.arctan2(lever[informative, 1], lever[informative, 0]) % (2.0 * math.pi)
        bin_width = 2.0 * math.pi / self.config.torque_direction_bins
        direction_index = np.floor((angle + 0.5 * bin_width) / bin_width).astype(int)
        category[informative] = 1 + direction_index % self.config.torque_direction_bins
        return category

    def category_probabilities(self, action_xy_m: np.ndarray) -> np.ndarray:
        predicted = self._predicted_category(action_xy_m)
        return self._category_likelihood[predicted]

    def _information_gain(self, action_xy_m: np.ndarray) -> tuple[float, np.ndarray]:
        likelihood = self.category_probabilities(action_xy_m)
        marginal = self.weights @ likelihood
        ratio = likelihood / np.clip(marginal[None, :], 1.0e-300, None)
        information = np.sum(
            self.weights[:, None]
            * likelihood
            * np.log(np.clip(ratio, 1.0e-300, None))
        )
        return float(information), marginal

    def set_prior_from_samples(
        self,
        samples_xy_m: np.ndarray,
        *,
        bandwidth_m: float,
        density_floor_fraction: float = 1.0e-6,
        learned_mixture_weight: float = 1.0,
    ) -> None:
        """Project an upstream continuous posterior onto the CAD belief grid."""

        samples = np.asarray(samples_xy_m, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) == 0:
            raise ValueError("samples_xy_m must have shape (N, 2)")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples_xy_m contains non-finite values")
        if bandwidth_m <= 0.0:
            raise ValueError("bandwidth_m must be positive")
        if not 0.0 <= density_floor_fraction < 0.1:
            raise ValueError("density_floor_fraction must be in [0, 0.1)")
        if not 0.0 <= learned_mixture_weight <= 1.0:
            raise ValueError("learned_mixture_weight must be in [0, 1]")

        density = np.zeros(len(self.candidate_xy_m), dtype=float)
        denominator = 2.0 * bandwidth_m * bandwidth_m
        for start in range(0, len(samples), 512):
            chunk = samples[start : start + 512]
            delta = self.candidate_xy_m[:, None, :] - chunk[None, :, :]
            density += np.sum(
                np.exp(-np.sum(delta * delta, axis=2) / denominator),
                axis=1,
            )
        maximum = float(np.max(density))
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise RuntimeError("sample posterior does not overlap the belief grid")
        density += density_floor_fraction * maximum
        learned = density / float(np.sum(density))
        self.weights = (
            learned_mixture_weight * learned
            + (1.0 - learned_mixture_weight) * self.visual_prior_weights
        )
        self.weights /= float(np.sum(self.weights))
        self._used_actions.clear()
        self.history.clear()

    def sample_posterior(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        indices = rng.choice(
            len(self.candidate_xy_m),
            size=count,
            replace=True,
            p=self.weights,
        )
        return self.candidate_xy_m[indices].copy()

    def select_action(
        self,
        current_xy_m: np.ndarray,
        *,
        fixed_action_cost_nats: float = 0.0,
        travel_penalty_nats_per_m: float = 0.0,
    ) -> EntropyAction:
        current = np.asarray(current_xy_m, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("current_xy_m must contain two finite values")
        if fixed_action_cost_nats < 0.0 or travel_penalty_nats_per_m < 0.0:
            raise ValueError("action costs must be nonnegative")
        gain = np.full(len(self.action_xy_m), -np.inf, dtype=float)
        marginal: list[np.ndarray | None] = [None] * len(self.action_xy_m)
        for index, action in enumerate(self.action_xy_m):
            if index in self._used_actions:
                continue
            gain[index], marginal[index] = self._information_gain(action)
        distance = np.linalg.norm(self.action_xy_m - current[None, :], axis=1)
        utility = (
            gain
            - fixed_action_cost_nats
            - travel_penalty_nats_per_m * distance
        )
        best_utility = float(np.max(utility))
        if not math.isfinite(best_utility):
            raise RuntimeError("no unused entropy-grid action remains")
        tied = np.flatnonzero(utility >= best_utility - 1.0e-12)
        best = int(tied[int(np.argmin(distance[tied]))])
        assert marginal[best] is not None
        return EntropyAction(
            index=best,
            xy_m=self.action_xy_m[best].copy(),
            expected_information_gain_nats=float(gain[best]),
            predicted_outcome_probabilities=np.asarray(marginal[best]).tolist(),
            travel_distance_m=float(distance[best]),
            net_utility_nats=float(utility[best]),
        )

    @staticmethod
    def _point_segment_distance(
        points_xy_m: np.ndarray,
        start_xy_m: np.ndarray,
        end_xy_m: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points_xy_m, dtype=float)
        start = np.asarray(start_xy_m, dtype=float)
        end = np.asarray(end_xy_m, dtype=float)
        delta = end - start
        length_sq = float(delta @ delta)
        if length_sq <= 1.0e-18:
            return np.linalg.norm(points - start[None, :], axis=1)
        fraction = np.clip(((points - start[None, :]) @ delta) / length_sq, 0.0, 1.0)
        projection = start[None, :] + fraction[:, None] * delta[None, :]
        return np.linalg.norm(points - projection, axis=1)

    def segment_capture_probability(
        self,
        start_xy_m: np.ndarray,
        end_xy_m: np.ndarray,
        effective_capture_radius_m: float,
    ) -> float:
        if effective_capture_radius_m <= 0.0:
            raise ValueError("effective_capture_radius_m must be positive")
        distance = self._point_segment_distance(
            self.candidate_xy_m,
            start_xy_m,
            end_xy_m,
        )
        return float(np.sum(self.weights[distance <= effective_capture_radius_m]))

    def select_task_aware_action(
        self,
        current_xy_m: np.ndarray,
        *,
        effective_capture_radius_m: float,
        information_gain_weight: float,
        capture_probability_reward_nats: float,
        fixed_action_cost_nats: float = 0.0,
        travel_penalty_nats_per_m: float = 0.0,
    ) -> EntropyAction:
        """Select an endpoint whose whole contact segment has task value.

        Pure information gain can prefer a diagnostic boundary far from the
        capture basin. This score adds the posterior probability that the
        guarded segment from the current point to the candidate endpoint
        intersects the conservative CAD capture disk.
        """

        current = np.asarray(current_xy_m, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("current_xy_m must contain two finite values")
        if effective_capture_radius_m <= 0.0:
            raise ValueError("effective_capture_radius_m must be positive")
        for value in (
            information_gain_weight,
            capture_probability_reward_nats,
            fixed_action_cost_nats,
            travel_penalty_nats_per_m,
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError("task-aware action weights must be nonnegative")

        gain = np.full(len(self.action_xy_m), -np.inf, dtype=float)
        capture_probability = np.zeros(len(self.action_xy_m), dtype=float)
        marginal: list[np.ndarray | None] = [None] * len(self.action_xy_m)
        for index, action in enumerate(self.action_xy_m):
            if index in self._used_actions:
                continue
            gain[index], marginal[index] = self._information_gain(action)
            capture_probability[index] = self.segment_capture_probability(
                current,
                action,
                effective_capture_radius_m,
            )
        distance = np.linalg.norm(self.action_xy_m - current[None, :], axis=1)
        utility = (
            information_gain_weight * gain
            + capture_probability_reward_nats * capture_probability
            - fixed_action_cost_nats
            - travel_penalty_nats_per_m * distance
        )
        best_utility = float(np.max(utility))
        if not math.isfinite(best_utility):
            raise RuntimeError("no unused task-aware entropy-grid action remains")
        tied = np.flatnonzero(utility >= best_utility - 1.0e-12)
        best = int(tied[int(np.argmin(distance[tied]))])
        assert marginal[best] is not None
        return EntropyAction(
            index=best,
            xy_m=self.action_xy_m[best].copy(),
            expected_information_gain_nats=float(gain[best]),
            predicted_outcome_probabilities=np.asarray(marginal[best]).tolist(),
            travel_distance_m=float(distance[best]),
            net_utility_nats=float(utility[best]),
            segment_capture_probability=float(capture_probability[best]),
            information_gain_weight=float(information_gain_weight),
            capture_probability_reward_nats=float(capture_probability_reward_nats),
        )

    def apply_no_capture_segment(
        self,
        start_xy_m: np.ndarray,
        end_xy_m: np.ndarray,
        *,
        effective_capture_radius_m: float,
        miss_inside_likelihood: float,
        miss_outside_likelihood: float,
    ) -> CaptureMissUpdate:
        """Use absence of a capture cue as a soft geometric observation."""

        if not (
            0.0 < miss_inside_likelihood < miss_outside_likelihood <= 1.0
        ):
            raise ValueError("invalid capture-miss likelihoods")
        distance = self._point_segment_distance(
            self.candidate_xy_m,
            start_xy_m,
            end_xy_m,
        )
        inside = distance <= effective_capture_radius_m
        covered_probability = float(np.sum(self.weights[inside]))
        likelihood = np.full(len(self.weights), miss_outside_likelihood, dtype=float)
        likelihood[inside] = miss_inside_likelihood
        self.weights *= likelihood
        total = float(np.sum(self.weights))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("capture-miss update collapsed the center belief")
        self.weights /= total
        return CaptureMissUpdate(
            segment_start_xy_m=np.asarray(start_xy_m, dtype=float).copy(),
            segment_end_xy_m=np.asarray(end_xy_m, dtype=float).copy(),
            covered_probability_before=covered_probability,
            entropy_nats=self.entropy_nats,
            credible_radius_95_m=self.credible_radius_95_m,
        )

    def predictive_observation_probability(
        self,
        action: EntropyAction,
        observation_category: int,
    ) -> float:
        outcomes = self.config.torque_direction_bins + 1
        if not 0 <= observation_category < outcomes:
            raise ValueError("observation_category is outside the quantizer")
        likelihood = self.category_probabilities(action.xy_m)[:, observation_category]
        return float(self.weights @ likelihood)

    def update(
        self,
        action: EntropyAction,
        *,
        observation_category: int,
        likelihood_temperature: float = 1.0,
    ) -> EntropyUpdate:
        if action.index in self._used_actions:
            raise ValueError("an entropy-grid action can only be evaluated once")
        outcomes = self.config.torque_direction_bins + 1
        if not 0 <= observation_category < outcomes:
            raise ValueError("observation_category is outside the quantizer")
        if not 0.0 <= likelihood_temperature <= 1.0:
            raise ValueError("likelihood_temperature must be in [0, 1]")
        before = self.entropy_nats
        likelihood = self.category_probabilities(action.xy_m)[:, observation_category]
        predictive_probability = float(self.weights @ likelihood)
        if likelihood_temperature > 0.0:
            self.weights *= np.power(likelihood, likelihood_temperature)
        total = float(np.sum(self.weights))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("entropy-grid posterior collapsed")
        self.weights /= total
        self._used_actions.add(action.index)
        update = EntropyUpdate(
            action_index=action.index,
            action_xy_m=action.xy_m.copy(),
            observation_category=int(observation_category),
            information_gain_nats=before - self.entropy_nats,
            entropy_nats=self.entropy_nats,
            credible_radius_95_m=self.credible_radius_95_m,
            posterior_mean_xy_m=self.posterior_mean_xy_m.copy(),
            predictive_probability=predictive_probability,
            likelihood_temperature=float(likelihood_temperature),
        )
        self.history.append(update)
        return update

    def overlay(self, selected_action_xy_m: np.ndarray | None = None) -> dict:
        return {
            "anchor_xy_m": self.anchor_xy_m.copy(),
            "visual_bound_m": self.visual_bound_m,
            "action_support_radius_m": self.action_support_radius_m,
            "action_padding_m": self.action_padding_m,
            "effective_capture_radius_m": self.effective_capture_radius_m,
            "action_lattice_gap_bound_m": self.action_lattice_gap_bound_m,
            "action_grid_coverage_certified": self.action_grid_coverage_certified,
            "candidate_xy_m": self.candidate_xy_m.copy(),
            "weights": self.weights.copy(),
            "probed_xy_m": np.array(
                [item.action_xy_m for item in self.history], dtype=float
            ).reshape((-1, 2)),
            "selected_action_xy_m": (
                None
                if selected_action_xy_m is None
                else np.asarray(selected_action_xy_m, dtype=float).copy()
            ),
            "posterior_mean_xy_m": self.posterior_mean_xy_m.copy(),
            "entropy_nats": self.entropy_nats,
            "credible_radius_95_m": self.credible_radius_95_m,
            "probe_count": len(self.history),
            "max_probes": self.config.max_probes,
        }
