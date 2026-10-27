from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.cartesian_admittance import AdmittanceConfig
from src.capture_yaw_assist import CaptureYawAssistConfig
from src.entropy_grid_baseline import (
    CaptureMissUpdate,
    EntropyAction,
    EntropyGridBelief,
    EntropyGridConfig,
    HybridProbeConfig,
    torque_category_from_wrench,
)
from src.funnel_guard import FunnelGuard, FunnelGuardConfig, FunnelGuardObservation
from src.m8_direction_runtime import DirectionEstimate, M8DirectionEnsemble
from src.mujoco_admittance_rollout import DynamicSnapshot
from src.wrench_observation import WrenchObservationConfig
from src.posterior_search_plan import PosteriorSearchPlan, plan_posterior_guided_search
from src.terminal_completion_verifier import (
    TerminalCompletionVerifier,
    TerminalVerifierConfig,
)
import src.train_m8_direction_models as direction_training
from src.render_m8_dynamic_videos import CONTACT_TIMECONST_S, SIM_TIMESTEP_S, setup
from src.render_m8_sampling_videos import (
    ALPHA,
    CONTROLLER_THRESHOLDS,
    FPS,
    HEIGHT,
    ROOT,
    SPIRAL_PROFILES,
    WIDTH,
    VideoCase,
    camera_snapshot,
    contact_tags,
    draw_axis_and_probe_cone,
    draw_contact_points,
    draw_plan_panel,
    draw_polar_fingerprint_panel,
    draw_six_axis_wrench,
    font_pair,
    planned_outward_spiral,
    smoothstep,
    valid_contact_infos,
    wrench_at_sensor,
)


OUT = ROOT / "outputs" / "current"
VIDEO_OUT = OUT / "media" / "videos" / "comparisons"
KEYFRAME_OUT = OUT / "media" / "keyframes" / "comparisons"
RESULT_PATH = OUT / "reports" / "m8_outer_partial_search_comparison.json"

INITIAL_XY_M = np.array([0.00400, 0.0], dtype=float)
NOMINAL_TIP_Z_M = 0.00030
TARGET_DOWN_FORCE_N = 3.0
SPIRAL_PITCH_M = 0.00040
COMMON_XY_SPEED_M_S = 0.00225
PROPOSED_PROFILE = SPIRAL_PROFILES["V6a"]
PROPOSED_TARGET_XY_M = np.asarray(PROPOSED_PROFILE.predicted_center_xy_m, dtype=float)
PROPOSED_LOCAL_RADIUS_M = float(PROPOSED_PROFILE.spiral_max_radius_m)
BASELINE_MAX_RADIUS_M = float(
    np.linalg.norm(PROPOSED_TARGET_XY_M - INITIAL_XY_M) + PROPOSED_LOCAL_RADIUS_M
)
BASELINE_PHASE_RAD = 0.0
SETTLE_TIME_S = 0.80
PROBE_RAMP_TIME_S = 10.0 / FPS
PROBE_SCAN_TIME_S = 52.0 / FPS
PROBE_RETURN_TIME_S = 22.0 / FPS
POST_PROBE_SETTLE_TIME_S = 0.20
SUMMARY_TIME_S = 2.0
COMMON_ACTIVE_TIME_BUDGET_S = 70.0
BASELINE_FAST_FORWARD_SPEED = 8.0
DIRECT_FIRST_FAILURE_POLICY_OVERLAY = (
    "failure: stop after 1 ray; no retry/reset"
)
DIRECT_RETRY_FAILURE_POLICY_OVERLAY = (
    "failure policy: shifted reset + full re-probe"
)


def _bounded_active_dt(
    active_time_s: float,
    requested_dt_s: float,
    active_time_budget_s: float,
) -> float:
    """Return the executable part of one active step under a hard deadline."""

    if requested_dt_s < 0.0:
        raise ValueError("requested_dt_s must be non-negative")
    if active_time_budget_s <= 0.0:
        raise ValueError("active_time_budget_s must be positive")
    remaining_s = max(0.0, float(active_time_budget_s) - float(active_time_s))
    return min(float(requested_dt_s), remaining_s)


def _hybrid_fallback_reserve_seconds(
    current_xy_m: np.ndarray,
    initial_xy_m: np.ndarray,
    *,
    fallback_path_length_m: float,
    xy_speed_m_s: float,
    return_position_error_bound_m: float,
    transition_overhead_s: float,
    candidate_contingency_s: float,
    terminal_execution_reserve_s: float,
) -> float:
    """Reserve enough active time to return, cover, and verify the terminal.

    This is a scheduling contract rather than a success guarantee: unsafe
    interruption, tracking error, or a failed terminal verifier can still end
    the episode. The geometric path length and commanded speed determine the
    finite-width search term; fixed controller phases provide the remaining overhead.
    """

    current = np.asarray(current_xy_m, dtype=float)
    initial = np.asarray(initial_xy_m, dtype=float)
    if current.shape != (2,) or initial.shape != (2,):
        raise ValueError("current_xy_m and initial_xy_m must contain two values")
    values = (
        fallback_path_length_m,
        xy_speed_m_s,
        return_position_error_bound_m,
        transition_overhead_s,
        candidate_contingency_s,
        terminal_execution_reserve_s,
    )
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(initial)):
        raise ValueError("fallback reserve positions must be finite")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("fallback reserve inputs must be finite")
    if (
        fallback_path_length_m < 0.0
        or return_position_error_bound_m < 0.0
        or transition_overhead_s < 0.0
        or candidate_contingency_s < 0.0
    ):
        raise ValueError("fallback path and transition overhead must be non-negative")
    if xy_speed_m_s <= 0.0 or terminal_execution_reserve_s < 0.0:
        raise ValueError("fallback speed must be positive and terminal reserve non-negative")
    return float(
        fallback_path_length_m / xy_speed_m_s
        + max(
            0.25,
            (
                np.linalg.norm(current - initial)
                + return_position_error_bound_m
            )
            / xy_speed_m_s,
        )
        + transition_overhead_s
        + candidate_contingency_s
        + terminal_execution_reserve_s
    )


def _nominal_terminal_execution_reserve_seconds(
    *,
    continue_to_bottom: bool,
    socket_depth_m: float,
    conservative_capture_z_m: float,
    axial_command_speed_m_s: float,
    minimum_descent_s: float,
    force_release_s: float,
    insertion_handoff_s: float,
    recovery_count: int,
    recovery_duration_s: float,
    disturbance_duration_s: float,
    completion_event_margin_s: float,
    minimum_reserve_s: float,
) -> float:
    """Derive a nominal post-capture schedule from CAD depth and controller rates.

    ``completion_event_margin_s`` covers the in-loop terminal-verifier dwell
    and time-step rounding.  The 0.50 s visualization hold emitted *after* a
    terminal event is deliberately excluded because it is marked inactive and
    is neither required for success nor charged to the hard active-time budget.
    """

    if not continue_to_bottom:
        return 0.0
    numeric = (
        socket_depth_m,
        conservative_capture_z_m,
        axial_command_speed_m_s,
        minimum_descent_s,
        force_release_s,
        insertion_handoff_s,
        recovery_duration_s,
        disturbance_duration_s,
        completion_event_margin_s,
        minimum_reserve_s,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("terminal reserve inputs must be finite")
    if socket_depth_m <= 0.0 or axial_command_speed_m_s <= 0.0:
        raise ValueError("socket depth and axial speed must be positive")
    if recovery_count < 0 or any(float(value) < 0.0 for value in numeric[3:]):
        raise ValueError("terminal reserve durations must be non-negative")
    final_command_z_m = -socket_depth_m + 0.0010
    descent_s = max(
        minimum_descent_s,
        abs(final_command_z_m - conservative_capture_z_m)
        / axial_command_speed_m_s,
    )
    derived_s = (
        force_release_s
        + insertion_handoff_s
        + descent_s
        + recovery_count * recovery_duration_s
        + disturbance_duration_s
        + completion_event_margin_s
    )
    return float(max(minimum_reserve_s, derived_s))


def _optional_action_fits_nominal_reserve(
    remaining_active_s: float,
    optional_action_s: float,
    nominal_fallback_reserve_s: float,
) -> bool:
    values = (remaining_active_s, optional_action_s, nominal_fallback_reserve_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("budget gate inputs must be finite")
    if optional_action_s < 0.0 or nominal_fallback_reserve_s < 0.0:
        raise ValueError("action duration and fallback reserve must be non-negative")
    return bool(
        remaining_active_s
        >= optional_action_s + nominal_fallback_reserve_s - 1.0e-12
    )


def _advance_dwell_count(count: int, cue: bool, required: int) -> tuple[int, bool]:
    """Advance a consecutive-sample latch; a transient cue never confirms."""

    if required <= 0:
        raise ValueError("required must be positive")
    updated = count + 1 if cue else 0
    return updated, updated >= required


def _bounded_measured_pose_rebase_orientation(
    measured_rpy: np.ndarray,
    exact_held_rpy_command: np.ndarray,
    rotational_offset_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Choose an unclipped nominal/offset pair for a measured-pose rebase.

    The preferred representation preserves the exact held nominal command and
    expresses the measured orientation as a bounded admittance offset.  If that
    shortest angular displacement exceeds any configured offset limit, using
    it would be silently clipped by ``CartesianAdmittanceController.reset``.
    The safe representation then uses measured RPY as nominal with zero offset.
    """

    measured = np.asarray(measured_rpy, dtype=float)
    exact = np.asarray(exact_held_rpy_command, dtype=float)
    limit = np.asarray(rotational_offset_limit, dtype=float)
    if measured.shape != (3,) or exact.shape != (3,) or limit.shape != (3,):
        raise ValueError("measured, exact, and rotational limit must have shape (3,)")
    if (
        not np.all(np.isfinite(measured))
        or not np.all(np.isfinite(exact))
        or not np.all(np.isfinite(limit))
        or np.any(limit <= 0.0)
    ):
        raise ValueError("rebase orientation inputs must be finite with positive limits")
    requested = (measured - exact + math.pi) % (2.0 * math.pi) - math.pi
    fallback = bool(np.any(np.abs(requested) > limit + 1.0e-12))
    return (
        measured.copy() if fallback else exact.copy(),
        np.zeros(3, dtype=float) if fallback else requested,
        fallback,
    )


def _declared_preprobe_route(
    method: str,
    conditioned_on_ambiguous_top_contact: bool,
) -> str:
    """Expose whether conical probing is conditioned or conservatively assumed.

    There is intentionally no invented settle-only ambiguity classifier here:
    the current benchmark supplies the ambiguous-top-contact condition upstream.
    """

    if method not in {"proposed", "learned", "hybrid", "direct"}:
        return "not_applicable_no_conical_probe"
    if conditioned_on_ambiguous_top_contact:
        return "conditioned_ambiguous_top_contact_probe"
    return "no_upstream_ambiguity_router_conservative_probe"


def _deduplicate_xy_polyline(
    samples_xy_m: list[np.ndarray] | np.ndarray,
    *,
    tolerance_m: float = 1.0e-9,
) -> np.ndarray:
    """Validate a measured XY trace and remove stationary duplicate samples."""

    samples = np.asarray(samples_xy_m, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) == 0:
        raise ValueError("samples_xy_m must have shape (N, 2), N >= 1")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples_xy_m contains non-finite values")
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    keep = [0]
    for index in range(1, len(samples)):
        if float(np.linalg.norm(samples[index] - samples[keep[-1]])) > tolerance_m:
            keep.append(index)
    return samples[np.asarray(keep, dtype=int)].copy()


def _minimum_distance_to_xy_polyline_m(
    point_xy_m: np.ndarray,
    samples_xy_m: list[np.ndarray] | np.ndarray,
) -> float:
    """Offline geometry helper; callers must not use its value for control."""

    point = np.asarray(point_xy_m, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("point_xy_m must be a finite two-vector")
    samples = _deduplicate_xy_polyline(samples_xy_m)
    if len(samples) == 1:
        return float(np.linalg.norm(samples[0] - point))
    minimum = float("inf")
    for start, end in zip(samples[:-1], samples[1:]):
        delta = end - start
        denominator = float(np.dot(delta, delta))
        fraction = (
            0.0
            if denominator <= 0.0
            else float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
        )
        minimum = min(
            minimum,
            float(np.linalg.norm(start + fraction * delta - point)),
        )
    return minimum


def _apply_no_capture_measured_polyline(
    belief: EntropyGridBelief,
    samples_xy_m: list[np.ndarray] | np.ndarray,
    *,
    effective_capture_radius_m: float,
    miss_inside_likelihood: float,
    miss_outside_likelihood: float,
) -> CaptureMissUpdate:
    """Apply one soft miss update to the union of the measured swept tube.

    The likelihood is applied once per center hypothesis, even when several
    adjacent measured segments cover it. This preserves the semantics of one
    no-capture observation and avoids multiplying the same negative evidence
    once per simulation sample.
    """

    if effective_capture_radius_m <= 0.0:
        raise ValueError("effective_capture_radius_m must be positive")
    if not (
        0.0 < miss_inside_likelihood < miss_outside_likelihood <= 1.0
    ):
        raise ValueError("invalid capture-miss likelihoods")
    samples = _deduplicate_xy_polyline(samples_xy_m)
    if len(samples) == 1:
        distance = np.linalg.norm(
            belief.candidate_xy_m - samples[0][None, :], axis=1
        )
    else:
        distance = np.full(len(belief.candidate_xy_m), np.inf, dtype=float)
        for start, end in zip(samples[:-1], samples[1:]):
            distance = np.minimum(
                distance,
                belief._point_segment_distance(  # noqa: SLF001 - audited union update
                    belief.candidate_xy_m,
                    start,
                    end,
                ),
            )
    inside = distance <= effective_capture_radius_m
    covered_probability = float(np.sum(belief.weights[inside]))
    likelihood = np.full(len(belief.weights), miss_outside_likelihood, dtype=float)
    likelihood[inside] = miss_inside_likelihood
    belief.weights *= likelihood
    total = float(np.sum(belief.weights))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("capture-miss update collapsed the center belief")
    belief.weights /= total
    return CaptureMissUpdate(
        segment_start_xy_m=samples[0].copy(),
        segment_end_xy_m=samples[-1].copy(),
        covered_probability_before=covered_probability,
        entropy_nats=belief.entropy_nats,
        credible_radius_95_m=belief.credible_radius_95_m,
    )


@dataclass(frozen=True)
class ConicalContactTransitionConfig:
    """Observable-only guard for contact-mode changes during a conical probe.

    The guard deliberately has no MuJoCo model, contact-count, or geometry
    input.  It sees the same quantities available to a robot controller:
    elapsed time, measured tip height, target normal load, and measured
    six-axis wrench.
    """

    low_support_force_fraction: float = 0.40
    low_support_recovery_fraction: float = 0.65
    low_support_gap_dwell_s: float = 0.004
    sustained_low_support_s: float = 0.012
    progress_window_s: float = 0.050
    progress_entry_drop_m: float = 0.000070
    progress_window_drop_m: float = 0.000040
    safe_total_force_n: float = 8.0
    safe_lateral_force_n: float = 2.0
    safe_tau_xy_mnm: float = 20.0

    def __post_init__(self) -> None:
        finite_positive = (
            self.low_support_force_fraction,
            self.low_support_recovery_fraction,
            self.low_support_gap_dwell_s,
            self.sustained_low_support_s,
            self.progress_window_s,
            self.progress_entry_drop_m,
            self.progress_window_drop_m,
            self.safe_total_force_n,
            self.safe_lateral_force_n,
            self.safe_tau_xy_mnm,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in finite_positive):
            raise ValueError("conical transition thresholds must be finite and positive")
        if self.low_support_force_fraction >= self.low_support_recovery_fraction:
            raise ValueError("low-support recovery threshold must exceed the entry threshold")
        if self.low_support_recovery_fraction > 1.0:
            raise ValueError("low-support recovery fraction must not exceed one")
        if self.low_support_gap_dwell_s > self.sustained_low_support_s:
            raise ValueError("gap dwell must not exceed sustained-low-support dwell")


@dataclass(frozen=True)
class ConicalContactTransitionDecision:
    stop_cone: bool
    event: str | None
    low_support: bool
    safe_wrench: bool
    support_force_n: float
    lateral_force_n: float
    tau_xy_mnm: float
    entry_drop_m: float
    window_drop_m: float
    fingerprint_incomplete: bool


class ConicalContactTransitionMonitor:
    """Latch a probe-to-capture transition without privileged contact truth."""

    def __init__(
        self,
        *,
        start_time_s: float,
        start_tip_z_m: float,
        target_down_force_n: float,
        config: ConicalContactTransitionConfig | None = None,
    ) -> None:
        values = (start_time_s, start_tip_z_m, target_down_force_n)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("conical transition monitor inputs must be finite")
        if start_time_s < 0.0 or target_down_force_n <= 0.0:
            raise ValueError("start time must be non-negative and target force positive")
        self.config = config or ConicalContactTransitionConfig()
        self.start_time_s = float(start_time_s)
        self.start_tip_z_m = float(start_tip_z_m)
        self.target_down_force_n = float(target_down_force_n)
        self._last_time_s = float(start_time_s)
        self._history: deque[tuple[float, float]] = deque(
            [(float(start_time_s), float(start_tip_z_m))]
        )
        self._low_support_elapsed_s = 0.0
        self._low_support_active = False
        self._event: str | None = None
        self._event_time_s: float | None = None
        self._fingerprint_incomplete = False
        self._low_support_episode_count = 0
        self._transient_low_support_count = 0
        self._minimum_support_force_n = float("inf")
        self._maximum_lateral_force_n = 0.0
        self._maximum_tau_xy_mnm = 0.0
        self._maximum_entry_drop_m = 0.0

    @property
    def event(self) -> str | None:
        return self._event

    @property
    def fingerprint_incomplete(self) -> bool:
        return self._fingerprint_incomplete

    def observe(
        self,
        *,
        time_s: float,
        tip_z_m: float,
        measured_wrench: np.ndarray,
        evidence_relevant: bool,
    ) -> ConicalContactTransitionDecision:
        time_value = float(time_s)
        z_value = float(tip_z_m)
        wrench = np.asarray(measured_wrench, dtype=float)
        if (
            not math.isfinite(time_value)
            or not math.isfinite(z_value)
            or wrench.shape != (6,)
            or not np.all(np.isfinite(wrench))
        ):
            raise ValueError("conical transition observation must be finite")
        if time_value < self._last_time_s - 1.0e-12:
            raise ValueError("conical transition observation time must be monotonic")
        dt_s = max(0.0, time_value - self._last_time_s)
        self._last_time_s = time_value

        support_force_n = max(0.0, float(wrench[2]))
        lateral_force_n = float(np.linalg.norm(wrench[:2]))
        total_force_n = float(np.linalg.norm(wrench[:3]))
        tau_xy_mnm = 1000.0 * float(np.linalg.norm(wrench[3:5]))
        self._minimum_support_force_n = min(
            self._minimum_support_force_n, support_force_n
        )
        self._maximum_lateral_force_n = max(
            self._maximum_lateral_force_n, lateral_force_n
        )
        self._maximum_tau_xy_mnm = max(
            self._maximum_tau_xy_mnm, tau_xy_mnm
        )

        low_threshold_n = (
            self.config.low_support_force_fraction * self.target_down_force_n
        )
        recovery_threshold_n = (
            self.config.low_support_recovery_fraction * self.target_down_force_n
        )
        low_support = bool(
            support_force_n <= (
                recovery_threshold_n if self._low_support_active else low_threshold_n
            )
        )
        if low_support:
            if not self._low_support_active:
                self._low_support_episode_count += 1
            self._low_support_active = True
            self._low_support_elapsed_s += dt_s
            if (
                evidence_relevant
                and self._low_support_elapsed_s
                >= self.config.low_support_gap_dwell_s
            ):
                self._fingerprint_incomplete = True
        else:
            if (
                self._low_support_active
                and 0.0 < self._low_support_elapsed_s
                < self.config.sustained_low_support_s
            ):
                self._transient_low_support_count += 1
            self._low_support_active = False
            self._low_support_elapsed_s = 0.0

        self._history.append((time_value, z_value))
        cutoff_s = time_value - self.config.progress_window_s
        while len(self._history) > 1 and self._history[1][0] <= cutoff_s:
            self._history.popleft()
        entry_drop_m = self.start_tip_z_m - z_value
        self._maximum_entry_drop_m = max(self._maximum_entry_drop_m, entry_drop_m)
        history_span_s = time_value - self._history[0][0]
        window_drop_m = self._history[0][1] - z_value
        safe_wrench = bool(
            total_force_n <= self.config.safe_total_force_n
            and lateral_force_n <= self.config.safe_lateral_force_n
            and tau_xy_mnm <= self.config.safe_tau_xy_mnm
        )
        safe_downward_progress = bool(
            entry_drop_m >= self.config.progress_entry_drop_m
            and history_span_s >= 0.8 * self.config.progress_window_s
            and window_drop_m >= self.config.progress_window_drop_m
            and safe_wrench
        )
        sustained_low_support = bool(
            self._low_support_elapsed_s >= self.config.sustained_low_support_s
        )
        if self._event is None and safe_downward_progress:
            self._event = "safe_downward_progress_during_conical_probe"
            self._event_time_s = time_value
            self._fingerprint_incomplete = True
        elif self._event is None and sustained_low_support:
            self._event = "sustained_low_support_during_conical_probe"
            self._event_time_s = time_value
            self._fingerprint_incomplete = True

        return ConicalContactTransitionDecision(
            stop_cone=self._event is not None,
            event=self._event,
            low_support=low_support,
            safe_wrench=safe_wrench,
            support_force_n=support_force_n,
            lateral_force_n=lateral_force_n,
            tau_xy_mnm=tau_xy_mnm,
            entry_drop_m=entry_drop_m,
            window_drop_m=window_drop_m,
            fingerprint_incomplete=self._fingerprint_incomplete,
        )

    def summary(self) -> dict:
        return {
            "transition_event": self._event,
            "transition_active_time_s": self._event_time_s,
            "fingerprint_incomplete": self._fingerprint_incomplete,
            "low_support_episode_count": self._low_support_episode_count,
            "transient_low_support_count": self._transient_low_support_count,
            "minimum_support_force_n": (
                None
                if not math.isfinite(self._minimum_support_force_n)
                else self._minimum_support_force_n
            ),
            "maximum_lateral_force_n": self._maximum_lateral_force_n,
            "maximum_tau_xy_mnm": self._maximum_tau_xy_mnm,
            "maximum_downward_progress_mm": 1000.0 * self._maximum_entry_drop_m,
            "uses_contact_count": False,
            "uses_contact_geometry_truth": False,
        }


@dataclass
class LiveMetrics:
    active_time_s: float = 0.0
    actual_xy_path_m: float = 0.0
    contact_xy_path_m: float = 0.0
    lateral_work_j: float = 0.0
    orientation_work_j: float = 0.0
    max_force_n: float = 0.0
    max_torque_nm: float = 0.0
    max_filtered_force_n: float = 0.0
    max_filtered_torque_nm: float = 0.0
    max_raw_physical_force_n: float = 0.0
    max_raw_physical_torque_nm: float = 0.0
    max_tilt_deg: float = 0.0
    stage_time_s: dict[str, float] = field(default_factory=dict)
    stage_max_force_n: dict[str, float] = field(default_factory=dict)
    stage_max_torque_nm: dict[str, float] = field(default_factory=dict)
    stage_max_filtered_force_n: dict[str, float] = field(default_factory=dict)
    stage_max_raw_physical_force_n: dict[str, float] = field(default_factory=dict)
    stage_max_raw_physical_torque_nm: dict[str, float] = field(default_factory=dict)
    _previous_tip: np.ndarray | None = field(default=None, repr=False)
    _previous_rpy: np.ndarray | None = field(default=None, repr=False)

    def reset_reference(self, snapshot: DynamicSnapshot) -> None:
        self._previous_tip = snapshot.tip_position.copy()
        self._previous_rpy = np.array(
            [snapshot.roll, snapshot.pitch, snapshot.yaw], dtype=float
        )

    def observe(
        self,
        snapshot: DynamicSnapshot,
        dt_s: float,
        stage_group: str,
        contact_valid: bool,
        active: bool,
    ) -> None:
        raw_force_norm = float(np.linalg.norm(snapshot.physical_wrench[:3]))
        raw_torque_norm = float(np.linalg.norm(snapshot.physical_wrench[3:]))
        self.max_raw_physical_force_n = max(
            self.max_raw_physical_force_n, raw_force_norm
        )
        self.max_raw_physical_torque_nm = max(
            self.max_raw_physical_torque_nm, raw_torque_norm
        )
        self.stage_max_raw_physical_force_n[stage_group] = max(
            self.stage_max_raw_physical_force_n.get(stage_group, 0.0),
            raw_force_norm,
        )
        self.stage_max_raw_physical_torque_nm[stage_group] = max(
            self.stage_max_raw_physical_torque_nm.get(stage_group, 0.0),
            raw_torque_norm,
        )
        rpy = np.array([snapshot.roll, snapshot.pitch, snapshot.yaw], dtype=float)
        if self._previous_tip is None or self._previous_rpy is None:
            self._previous_tip = snapshot.tip_position.copy()
            self._previous_rpy = rpy
            return
        delta_tip = snapshot.tip_position - self._previous_tip
        delta_rpy = rpy - self._previous_rpy
        self._previous_tip = snapshot.tip_position.copy()
        self._previous_rpy = rpy
        if not active:
            return
        self.active_time_s += dt_s
        self.stage_time_s[stage_group] = self.stage_time_s.get(stage_group, 0.0) + dt_s
        delta_xy = delta_tip[:2]
        xy_distance = float(np.linalg.norm(delta_xy))
        self.actual_xy_path_m += xy_distance
        if contact_valid:
            self.contact_xy_path_m += xy_distance
        wrench = snapshot.measured_wrench
        self.lateral_work_j += abs(float(np.dot(wrench[:2], delta_xy)))
        self.orientation_work_j += abs(float(np.dot(wrench[3:], delta_rpy)))
        force_norm = float(np.linalg.norm(wrench[:3]))
        torque_norm = float(np.linalg.norm(wrench[3:]))
        filtered_force_norm = float(np.linalg.norm(snapshot.command.filtered_wrench[:3]))
        filtered_torque_norm = float(np.linalg.norm(snapshot.command.filtered_wrench[3:]))
        tilt_deg = math.degrees(float(np.linalg.norm(rpy[:2])))
        self.max_force_n = max(self.max_force_n, force_norm)
        self.max_torque_nm = max(self.max_torque_nm, torque_norm)
        self.max_filtered_force_n = max(self.max_filtered_force_n, filtered_force_norm)
        self.max_filtered_torque_nm = max(self.max_filtered_torque_nm, filtered_torque_norm)
        self.max_tilt_deg = max(self.max_tilt_deg, tilt_deg)
        self.stage_max_force_n[stage_group] = max(
            self.stage_max_force_n.get(stage_group, 0.0), force_norm
        )
        self.stage_max_torque_nm[stage_group] = max(
            self.stage_max_torque_nm.get(stage_group, 0.0), torque_norm
        )
        self.stage_max_filtered_force_n[stage_group] = max(
            self.stage_max_filtered_force_n.get(stage_group, 0.0), filtered_force_norm
        )

    @property
    def wear_proxy_j(self) -> float:
        return self.lateral_work_j + self.orientation_work_j


@dataclass
class ShortcutPlanTelemetry:
    """Keep the first shortcut distinct from later Proposed+ replans.

    Distances are stored in millimetres because this object is copied directly
    into :class:`RunResult`.  Recording a new plan deliberately resets
    ``latest_plan_executed``: planning a motion and executing it are separate
    events in the hybrid controller.  Here ``executed`` means dispatched to the
    physical controller; it does not claim that the measured tip reached the
    commanded endpoint.
    """

    first_shortcut_distance_mm: float | None = None
    first_shortcut_target_xy_mm: list[float] | None = None
    first_shortcut_direction_quality: float | None = None
    first_shortcut_quantile: float | None = None
    first_shortcut_reverse_probability: float | None = None
    first_shortcut_distance_cap_mm: float | None = None
    first_shortcut_executed: bool | None = None
    latest_planned_distance_mm: float | None = None
    latest_planned_target_xy_mm: list[float] | None = None
    latest_plan_update_index: int | None = None
    latest_plan_executed: bool | None = None

    def record(self, plan: PosteriorSearchPlan, *, first: bool = False) -> None:
        distance_mm = 1000.0 * float(plan.transport_distance_m)
        target_xy_mm = (1000.0 * np.asarray(plan.transport_target_xy)).tolist()
        if first:
            if self.first_shortcut_distance_mm is not None:
                raise RuntimeError("first shortcut telemetry is already frozen")
            self.first_shortcut_distance_mm = distance_mm
            self.first_shortcut_target_xy_mm = target_xy_mm.copy()
            self.first_shortcut_direction_quality = float(plan.direction_quality)
            self.first_shortcut_quantile = float(plan.transport_quantile)
            self.first_shortcut_reverse_probability = float(plan.reverse_probability)
            self.first_shortcut_distance_cap_mm = (
                1000.0 * float(plan.transport_distance_cap_m)
            )
            self.first_shortcut_executed = False
            update_index = 0
        else:
            if self.latest_plan_update_index is None:
                raise RuntimeError("cannot record a replan before the first shortcut")
            update_index = self.latest_plan_update_index + 1
        self.latest_planned_distance_mm = distance_mm
        self.latest_planned_target_xy_mm = target_xy_mm.copy()
        self.latest_plan_update_index = update_index
        self.latest_plan_executed = False

    def mark_latest_executed(self) -> None:
        if self.latest_plan_update_index is None:
            raise RuntimeError("cannot execute an unrecorded shortcut plan")
        self.latest_plan_executed = True
        if self.latest_plan_update_index == 0:
            self.first_shortcut_executed = True


@dataclass
class CaptureDirectionGateLifecycle:
    """One-way lifecycle for the initial learned-direction capture gate."""

    initially_enabled: bool
    active: bool = field(init=False)
    retired_for_global_fallback: bool = False
    retired_active_time_s: float | None = None
    retirement_reason: str | None = None

    def __post_init__(self) -> None:
        self.initially_enabled = bool(self.initially_enabled)
        self.active = self.initially_enabled

    def retire_for_global_fallback(
        self, *, active_time_s: float, reason: str
    ) -> None:
        if not self.initially_enabled:
            return
        time_s = float(active_time_s)
        normalized_reason = str(reason).strip()
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("capture-gate retirement time must be finite and non-negative")
        if not normalized_reason:
            raise ValueError("capture-gate retirement reason must be non-empty")
        if self.retired_for_global_fallback:
            if self.retirement_reason != normalized_reason:
                raise RuntimeError("capture direction gate retirement reason drift")
            return
        self.active = False
        self.retired_for_global_fallback = True
        self.retired_active_time_s = time_s
        self.retirement_reason = normalized_reason


@dataclass
class RunResult:
    method: str
    result: str
    inner_wall_confirmed: bool
    time_to_inner_wall_s: float | None
    candidate_attempts: int
    total_active_time_s: float
    actual_xy_path_mm: float
    contact_xy_path_mm: float
    lateral_work_mj: float
    orientation_work_mj: float
    wear_proxy_mj: float
    max_force_n: float
    max_torque_mnm: float
    stage_time_s: dict[str, float]
    event_xy_mm: list[float] | None
    baseline_depth_mm: float
    min_tip_z_mm: float
    hard_stop: bool
    video_path: str | None = None
    frame_count: int = 0
    initial_xy_mm: list[float] | None = None
    direction_source: str | None = None
    predicted_direction_deg: float | None = None
    true_direction_deg: float | None = None
    direction_error_deg: float | None = None
    direction_confidence: float | None = None
    aps_set_size: int | None = None
    transport_allowed: bool | None = None
    legacy_transport_gate: bool | None = None
    transport_distance_mm: float | None = None
    transport_target_xy_mm: list[float] | None = None
    spiral_max_radius_mm: float | None = None
    capture_mode: str | None = None
    display_fast_forward_after_s: float | None = None
    display_fast_forward_speed: float = 1.0
    display_fast_forward_start_frame: int | None = None
    control_direction_deg: float | None = None
    aps_direction_angles_deg: list[float] | None = None
    circular_concentration: float | None = None
    ensemble_top1_span_deg: float | None = None
    bottom_reached: bool = False
    time_to_bottom_s: float | None = None
    bottom_depth_error_mm: float | None = None
    terminal_verification_mode: str | None = None
    terminal_outcome_hard_stop: bool | None = None
    post_terminal_event_display_new_hard_stop: bool = False
    post_terminal_event_display_hard_stop_reason: str | None = None
    offline_capture_scoring_snapshot: str | None = None
    top_state: str | None = None
    state_probabilities: dict[str, float] | None = None
    posterior_model: str | None = None
    cad_radial_boundaries_mm: dict[str, float] | None = None
    posterior_radial_quantiles_mm: dict[str, float] | None = None
    transport_policy: str | None = None
    transport_direction_quality: float | None = None
    transport_quantile: float | None = None
    transport_reverse_probability: float | None = None
    transport_projected_quantiles_mm: dict[str, float] | None = None
    transport_raw_distance_mm: float | None = None
    transport_distance_cap_mm: float | None = None
    # Frozen semantics for the first posterior shortcut.  The legacy
    # ``transport_*`` fields above remain backward compatible and may describe
    # a later Proposed+ replan.
    first_shortcut_distance_mm: float | None = None
    first_shortcut_target_xy_mm: list[float] | None = None
    first_shortcut_direction_quality: float | None = None
    first_shortcut_quantile: float | None = None
    first_shortcut_reverse_probability: float | None = None
    first_shortcut_distance_cap_mm: float | None = None
    first_shortcut_executed: bool | None = None
    # Latest posterior plan, whether or not that plan was subsequently
    # executed.  ``latest_plan_update_index == 0`` denotes the first shortcut.
    latest_planned_distance_mm: float | None = None
    latest_planned_target_xy_mm: list[float] | None = None
    latest_plan_update_index: int | None = None
    latest_plan_executed: bool | None = None
    max_filtered_force_n: float | None = None
    max_filtered_torque_mnm: float | None = None
    max_raw_physical_force_n: float | None = None
    max_raw_physical_torque_mnm: float | None = None
    stage_max_force_n: dict[str, float] | None = None
    stage_max_torque_mnm: dict[str, float] | None = None
    stage_max_filtered_force_n: dict[str, float] | None = None
    stage_max_raw_physical_force_n: dict[str, float] | None = None
    stage_max_raw_physical_torque_mnm: dict[str, float] | None = None
    fingerprint_sha256: str | None = None
    global_outward_fallback_used: bool = False
    global_outward_fallback_started_active_time_s: float | None = None
    capture_yaw_assist_enabled: bool = False
    capture_yaw_assist_amplitude_deg: float = 0.5
    capture_yaw_assist_frequency_hz: float = 1.5
    capture_yaw_assist_ramp_time_s: float = 0.25
    capture_yaw_assist_soft_tau_z_limit_mnm: float = 10.0
    capture_yaw_assist_used: bool = False
    capture_yaw_assist_aborted: bool = False
    max_abs_tau_z_mnm: float = 0.0
    capture_handoff_used: bool = False
    sensor_progress_candidate_count: int = 0
    search_recovery_count: int = 0
    oracle_transport_soft_escape_recovery_enabled: bool = False
    oracle_transport_soft_escape_recovery_count: int = 0
    oracle_transport_center_tolerance_mm: float | None = None
    oracle_transport_final_center_residual_mm: float | None = None
    oracle_transport_recovery_outcome: str | None = None
    oracle_transport_recovery_records: list[dict] | None = None
    fallback_physical_reset_used: bool = False
    fallback_budget_extension_s: float = 0.0
    active_time_budget_s: float | None = None
    budget_exhaustion_stage: str | None = None
    fallback_skipped_reason: str | None = None
    hard_stop_reason: str | None = None
    conditioned_on_ambiguous_top_contact: bool = False
    preprobe_route: str | None = None
    conical_probe_attempt_records: list[dict] | None = None
    conical_direction_inference_count: int = 0
    conical_capture_handoff_count: int = 0
    conical_incomplete_fallback_used: bool = False
    funnel_guard_enabled: bool = False
    funnel_tip_rcc_used: bool = False
    funnel_recovery_count: int = 0
    funnel_injected_energy_mj: float = 0.0
    funnel_peak_injected_power_w: float = 0.0
    funnel_min_health: float | None = None
    funnel_abort_reason: str | None = None
    insertion_command_bias_mm: list[float] | None = None
    post_capture_tilt_disturbance_deg: float = 0.0
    post_capture_tilt_axis_deg: float = 0.0
    post_capture_lateral_disturbance_mm: float = 0.0
    post_capture_disturbance_persistent: bool = False
    post_capture_tilt_duration_s: float = 0.0
    post_capture_disturbance_recovered: bool = False
    post_capture_disturbance_reason: str | None = None
    max_tilt_deg: float = 0.0
    final_tilt_deg: float = 0.0
    online_capture_sensor_only: bool = True
    offline_capture_validated: bool | None = None
    offline_capture_tags: list[str] | None = None
    random_search_seed: int | None = None
    random_search_path_budget_mm: float | None = None
    random_search_waypoints: int | None = None
    entropy_grid_probe_count: int | None = None
    entropy_grid_initial_nats: float | None = None
    entropy_grid_final_nats: float | None = None
    entropy_grid_credible_radius_mm: float | None = None
    entropy_grid_map_xy_mm: list[float] | None = None
    entropy_grid_mean_xy_mm: list[float] | None = None
    entropy_grid_observations: list[dict] | None = None
    entropy_grid_used_spiral: bool | None = None
    hybrid_spiral_fallback_reason: str | None = None
    entropy_grid_config: dict | None = None
    entropy_grid_geometry_mm: dict | None = None
    hybrid_probe_gate_records: list[dict] | None = None
    hybrid_probe_config: dict | None = None
    hybrid_residual_coverage_config: dict | None = None
    hybrid_fallback_nominal_reserve_s: float | None = None
    hybrid_entry_fallback_feasible: bool | None = None
    hybrid_terminal_execution_reserve_s: float | None = None
    hybrid_budget_switch_active_time_s: float | None = None
    hybrid_budget_switch_reason: str | None = None
    hybrid_guarded_motion_failure_reason: str | None = None
    capture_direction_consistency_gate_enabled: bool = False
    capture_direction_gate_active_at_outcome: bool = False
    capture_direction_gate_retired_for_global_fallback: bool = False
    capture_direction_gate_retired_active_time_s: float | None = None
    capture_direction_gate_retirement_reason: str | None = None
    capture_reverse_projection_tolerance_mm: float | None = None
    terminal_attempts: int = 0
    terminal_rejection_count: int = 0
    terminal_attempt_diagnostics: list[dict] | None = None
    terminal_entry_rebase_records: list[dict] | None = None
    terminal_axial_position_hold_used: bool = False
    transition_telemetry_records: list[dict] | None = None
    oracle_endpoint_settle_records: list[dict] | None = None
    post_release_requalification_records: list[dict] | None = None
    post_release_requalified: bool | None = None
    funnel_recovery_reasons: list[str] | None = None
    funnel_recovery_xy_rebases_mm: list[list[float]] | None = None
    funnel_recovery_transition_records: list[dict] | None = None
    simulator_reset_count: int = 0
    direct_max_attempts: int = 1
    direct_attempts_completed: int = 0
    direct_successful_attempt: int | None = None
    direct_attempt_records: list[dict] | None = None
    fingerprint_sha256_attempts: list[str] | None = None


@dataclass(frozen=True)
class CadRadialGeometry:
    """Approximate concentric M8 head/socket geometry used as a soft prior."""

    full_outer_support_radius_m: float
    inner_opening_overlap_limit_m: float
    transition_sigma_m: float


def cad_radial_geometry_from_spec(spec: dict) -> CadRadialGeometry:
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius_m = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem_m = 0.5 * float(spec["derived"]["socket_top_af_m"])
    full_outer_support_radius_m = head_radius_m - bit_radius_m
    inner_opening_overlap_limit_m = socket_top_apothem_m + bit_radius_m
    if not 0.0 < full_outer_support_radius_m < inner_opening_overlap_limit_m:
        raise ValueError("invalid CAD radial support boundaries")
    transition_sigma_m = max(
        float(spec.get("bit_tip_chamfer_m", 0.0)),
        float(spec.get("chamfer_width_m", 0.0)),
        0.00025,
    )
    return CadRadialGeometry(
        full_outer_support_radius_m=full_outer_support_radius_m,
        inner_opening_overlap_limit_m=inner_opening_overlap_limit_m,
        transition_sigma_m=transition_sigma_m,
    )


def _stable_logistic(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def cad_state_likelihood(
    radius_m: np.ndarray,
    state_probabilities: np.ndarray,
    geometry: CadRadialGeometry,
    *,
    likelihood_floor: float = 0.02,
) -> np.ndarray:
    """Score candidate radii using F/T-derived state belief and approximate CAD.

    The three support states are deliberately soft near CAD boundaries. This
    preserves model tolerance and state ambiguity instead of turning a
    diagnostic class into a privileged radial label.
    """
    radius = np.asarray(radius_m, dtype=float)
    state_probability = np.asarray(state_probabilities, dtype=float)
    if radius.ndim != 1:
        raise ValueError("radius_m must be one-dimensional")
    if state_probability.shape != (len(direction_training.STATE_NAMES),):
        raise ValueError("state_probabilities has the wrong shape")
    if np.any(state_probability < 0.0) or not np.isfinite(state_probability).all():
        raise ValueError("state_probabilities must be finite and non-negative")
    total = float(np.sum(state_probability))
    if total <= 0.0:
        raise ValueError("state_probabilities must contain positive mass")
    if not 0.0 <= likelihood_floor < 1.0:
        raise ValueError("likelihood_floor must be in [0, 1)")
    state_probability = state_probability / total

    sigma = geometry.transition_sigma_m
    supported = _stable_logistic(
        (geometry.full_outer_support_radius_m - radius) / sigma
    )
    outside_full_support = 1.0 - supported
    inside_inner_overlap = _stable_logistic(
        (geometry.inner_opening_overlap_limit_m - radius) / sigma
    )
    double_overhang = outside_full_support * inside_inner_overlap
    outer_partial = 1.0 - inside_inner_overlap
    likelihood_by_state = {
        "supported_search": supported,
        "double_overhang": double_overhang,
        "outer_partial": outer_partial,
    }
    geometry_likelihood = sum(
        float(state_probability[index]) * likelihood_by_state[state]
        for index, state in enumerate(direction_training.STATE_NAMES)
    )
    return likelihood_floor + (1.0 - likelihood_floor) * geometry_likelihood


def center_posterior_from_direction(
    current_xy_m: np.ndarray,
    direction_probabilities: np.ndarray,
    visual_sigma_m: float,
    rng: np.random.Generator,
    *,
    visual_bound_m: float | None = None,
    state_probabilities: np.ndarray | None = None,
    cad_geometry: CadRadialGeometry | None = None,
    cad_likelihood_floor: float = 0.02,
    candidates: int = 8192,
    posterior_samples: int = 4096,
) -> np.ndarray:
    """Fuse vision, F/T-derived direction/state beliefs, and approximate CAD.

    ``visual_bound_m`` truncates the Gaussian to the declared deployment
    acceptance set. It is a task assumption, not information inferred from the
    conical fingerprint. CAD state bands are soft and require both
    ``state_probabilities`` and ``cad_geometry``; omitting both preserves the
    original direction-only posterior for controlled ablations.
    """
    current = np.asarray(current_xy_m, dtype=float)
    probability = np.asarray(direction_probabilities, dtype=float)
    if current.shape != (2,) or probability.shape != (direction_training.DIRECTION_CLASSES,):
        raise ValueError("invalid current position or direction probability shape")
    if visual_sigma_m <= 0.0:
        raise ValueError("visual_sigma_m must be positive")
    if visual_bound_m is not None and visual_bound_m <= 0.0:
        raise ValueError("visual_bound_m must be positive when provided")
    samples = current + rng.normal(0.0, visual_sigma_m, size=(candidates, 2))
    delta = samples - current
    if visual_bound_m is not None:
        accepted = np.linalg.norm(delta, axis=1) <= visual_bound_m
        samples = samples[accepted]
        delta = delta[accepted]
        if len(samples) < posterior_samples:
            raise ValueError(
                "too few visual-prior candidates remain after truncation; "
                "increase candidates or relax visual_bound_m"
            )
    angles_deg = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360.0
    classes = np.rint(
        ((angles_deg - direction_training.DIRECTION_OFFSET_DEG) % 360.0)
        / direction_training.DIRECTION_BIN_DEG
    ).astype(int) % direction_training.DIRECTION_CLASSES
    weights = probability[classes] + 1.0e-9
    if (state_probabilities is None) != (cad_geometry is None):
        raise ValueError(
            "state_probabilities and cad_geometry must be provided together"
        )
    if state_probabilities is not None and cad_geometry is not None:
        radius = np.linalg.norm(delta, axis=1)
        weights *= cad_state_likelihood(
            radius,
            state_probabilities,
            cad_geometry,
            likelihood_floor=cad_likelihood_floor,
        )
    weights /= np.sum(weights)
    selected = rng.choice(len(samples), size=posterior_samples, replace=True, p=weights)
    return samples[selected]


def comparison_case(method: str) -> VideoCase:
    if method in {"proposed", "learned", "direct", "oracle"}:
        return VideoCase(
            "V6a",
            "",
            "PROPOSED | conical + posterior transport + local outward",
            "same-start dynamic comparison",
            "safe downward progress",
            float(INITIAL_XY_M[0]),
            float(INITIAL_XY_M[1]),
            0.00010,
            "spiral",
        )
    if method == "entropy_grid":
        return VideoCase(
            "ENTROPY",
            "",
            "ENTROPY-GRID | grid belief + maximum entropy probing",
            "same-start dynamic comparison",
            "guarded posterior-mean touchdown",
            float(INITIAL_XY_M[0]),
            float(INITIAL_XY_M[1]),
            0.00010,
            "entropy_grid",
        )
    if method == "hybrid":
        return VideoCase(
            "HYBRID",
            "",
            "PROPOSED+ | conical belief + task-aware posterior grid",
            "same-start dynamic comparison",
            "joint posterior funnel",
            float(INITIAL_XY_M[0]),
            float(INITIAL_XY_M[1]),
            0.00010,
            "hybrid",
        )
    if method == "random":
        return VideoCase(
            "RAND",
            "",
            "BASELINE D | seeded random-waypoint route",
            "same-start dynamic comparison",
            "safe downward progress",
            float(INITIAL_XY_M[0]),
            float(INITIAL_XY_M[1]),
            0.00010,
            "random_search",
        )
    return VideoCase(
        "BASE",
        "",
        "BASELINE | source-style ordinary outward spiral",
        "same-start dynamic comparison",
        "safe downward progress",
        float(INITIAL_XY_M[0]),
        float(INITIAL_XY_M[1]),
        0.00010,
        "comparison",
    )


def fast_contact_tags(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    tags: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        if "bit_collision" not in (geom1, geom2):
            continue
        other = geom2 if geom1 == "bit_collision" else geom1
        if other == "workpiece_base":
            continue
        if other == "screw_head_top":
            tags.add("head_top")
        elif other.startswith("socket_chamfer"):
            tags.add("chamfer")
        elif other.startswith("socket_wall"):
            tags.add("wall")
        elif other == "socket_bottom":
            tags.add("bottom")
        else:
            tags.add("other")
    return tags


def archimedean_path(
    center_xy: np.ndarray,
    pitch_m: float,
    max_radius_m: float,
    phase_rad: float,
    samples: int = 12000,
) -> tuple[np.ndarray, np.ndarray]:
    b = pitch_m / (2.0 * math.pi)
    theta_max = max_radius_m / b
    theta = np.linspace(0.0, theta_max, samples)
    radius = b * theta
    xy = np.asarray(center_xy, dtype=float) + np.column_stack(
        (radius * np.cos(phase_rad + theta), radius * np.sin(phase_rad + theta))
    )
    increments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(increments)))
    return arc, xy


@dataclass(frozen=True)
class HybridResidualCoverageConfig:
    """Finite-budget posterior corridor used when a full disk spiral is costly.

    The corridor is deliberately a *conditional* coverage policy.  It covers a
    declared angular sector around the measured direction; it never reads the
    true socket center and it does not turn the online capture cue or terminal
    verifier into an oracle.  Defaults leave all historical rollouts unchanged.
    """

    enabled: bool = False
    angular_half_width_deg: float = 12.0
    lane_spacing_capture_fraction: float = 0.80
    tracking_error_allowance_m: float = 0.0
    minimum_direction_quality: float = 0.10
    maximum_reverse_probability: float = 0.35
    transition_overhead_s: float = 1.0
    corridor_path_speed_m_s: float = 0.0
    measured_tip_outer_servo_enabled: bool = False
    outer_servo_gain: float = 1.5
    outer_servo_max_correction_m: float = 0.0010
    outer_servo_max_correction_speed_m_s: float = 0.0040
    outer_servo_abort_tracking_error_m: float = 0.0012
    outer_servo_abort_dwell_s: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.angular_half_width_deg < 90.0:
            raise ValueError("angular_half_width_deg must be in (0, 90)")
        if not 0.0 < self.lane_spacing_capture_fraction <= 1.0:
            raise ValueError(
                "lane_spacing_capture_fraction must be in (0, 1]"
            )
        if (
            not math.isfinite(self.tracking_error_allowance_m)
            or self.tracking_error_allowance_m < 0.0
        ):
            raise ValueError(
                "tracking_error_allowance_m must be finite and non-negative"
            )
        if not 0.0 <= self.minimum_direction_quality <= 1.0:
            raise ValueError("minimum_direction_quality must be in [0, 1]")
        if not 0.0 <= self.maximum_reverse_probability <= 0.5:
            raise ValueError("maximum_reverse_probability must be in [0, 0.5]")
        if not math.isfinite(self.transition_overhead_s) or self.transition_overhead_s < 0.0:
            raise ValueError("transition_overhead_s must be finite and non-negative")
        if (
            not math.isfinite(self.corridor_path_speed_m_s)
            or self.corridor_path_speed_m_s < 0.0
        ):
            raise ValueError(
                "corridor_path_speed_m_s must be finite and non-negative"
            )
        if not math.isfinite(self.outer_servo_gain) or self.outer_servo_gain < 0.0:
            raise ValueError("outer_servo_gain must be finite and non-negative")
        for name in (
            "outer_servo_max_correction_m",
            "outer_servo_max_correction_speed_m_s",
            "outer_servo_abort_tracking_error_m",
            "outer_servo_abort_dwell_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.measured_tip_outer_servo_enabled and self.outer_servo_gain <= 0.0:
            raise ValueError("enabled outer servo requires positive gain")


def posterior_directional_corridor_path(
    anchor_xy_m: np.ndarray,
    direction_xy: np.ndarray,
    *,
    axial_start_m: float,
    axial_end_m: float,
    half_width_m: float,
    lane_spacing_m: float,
    close_to_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a center-first serpentine covering a posterior direction sector.

    In coordinates aligned with ``direction_xy``, every lane spans
    ``axial_start_m <= s <= axial_end_m``.  Lateral lane separation is at most
    ``lane_spacing_m``.  Therefore an ideal point target satisfying

    ``s* in [axial_start_m, axial_end_m]`` and ``|u*| <= half_width_m``

    is within half a lane spacing of the commanded polyline.  The caller must
    enforce ``lane_spacing_m <= 2*r_capture`` for a geometric capture-tube
    condition.  Real tracking, contact and the sensor-only handoff remain
    separate failure modes, so this helper is not a task-success certificate.
    """

    anchor = np.asarray(anchor_xy_m, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    if anchor.shape != (2,) or direction.shape != (2,):
        raise ValueError("anchor_xy_m and direction_xy must have shape (2,)")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(direction)):
        raise ValueError("corridor anchor and direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("direction_xy must be nonzero")
    if not all(
        math.isfinite(float(value))
        for value in (
            axial_start_m,
            axial_end_m,
            half_width_m,
            lane_spacing_m,
        )
    ):
        raise ValueError("corridor geometry must be finite")
    if axial_start_m < 0.0 or axial_end_m <= axial_start_m:
        raise ValueError("corridor axial interval must satisfy 0 <= start < end")
    if half_width_m < 0.0 or lane_spacing_m <= 0.0:
        raise ValueError("corridor width must be non-negative and spacing positive")

    direction = direction / norm
    lateral = np.array([-direction[1], direction[0]], dtype=float)
    if half_width_m <= 1.0e-12:
        offsets = np.array([0.0], dtype=float)
    else:
        side_count = max(1, int(math.ceil(half_width_m / lane_spacing_m)))
        side_step = half_width_m / side_count
        values = [0.0]
        for index in range(1, side_count + 1):
            values.extend((index * side_step, -index * side_step))
        offsets = np.asarray(values, dtype=float)

    start = anchor + axial_start_m * direction
    end = anchor + axial_end_m * direction
    points: list[np.ndarray] = [start + offsets[0] * lateral]
    forward = True
    for index, offset in enumerate(offsets):
        lane_start = start + offset * lateral
        lane_end = end + offset * lateral
        destination = lane_end if forward else lane_start
        if float(np.linalg.norm(points[-1] - (lane_start if forward else lane_end))) > 1.0e-12:
            points.append(lane_start if forward else lane_end)
        points.append(destination)
        if index + 1 < len(offsets):
            next_offset = offsets[index + 1]
            connector = (
                end + next_offset * lateral
                if forward
                else start + next_offset * lateral
            )
            points.append(connector)
        forward = not forward

    if close_to_start and float(np.linalg.norm(points[-1] - points[0])) > 1.0e-12:
        # Return through the already searched boundary/centre line.  This makes
        # a subsequent guarded global reset explicit and keeps schedule length
        # deterministic without teleporting the contact state.
        current_axial = (
            axial_end_m
            if float(np.dot(points[-1] - anchor, direction))
            > 0.5 * (axial_start_m + axial_end_m)
            else axial_start_m
        )
        points.append(anchor + current_axial * direction)
        points.append(points[0].copy())

    xy = _deduplicate_xy_polyline(points)
    increments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(increments)))
    return arc, xy, offsets


def direction_consistent_capture(
    initial_xy_m: np.ndarray,
    candidate_xy_m: np.ndarray,
    predicted_direction_deg: float,
    reverse_tolerance_m: float,
) -> bool:
    """Check that a capture candidate is not behind the sensed hole direction."""

    initial = np.asarray(initial_xy_m, dtype=float)
    candidate = np.asarray(candidate_xy_m, dtype=float)
    if initial.shape != (2,) or candidate.shape != (2,):
        raise ValueError("initial_xy_m and candidate_xy_m must have shape (2,)")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(candidate)):
        raise ValueError("capture positions must be finite")
    if not np.isfinite(predicted_direction_deg):
        raise ValueError("predicted_direction_deg must be finite")
    if not np.isfinite(reverse_tolerance_m) or reverse_tolerance_m < 0.0:
        raise ValueError("reverse_tolerance_m must be finite and non-negative")
    displacement = candidate - initial
    if float(np.linalg.norm(displacement)) <= reverse_tolerance_m:
        return True
    angle = math.radians(float(predicted_direction_deg))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    return bool(float(np.dot(displacement, direction)) >= -reverse_tolerance_m)


def random_waypoint_path(
    center_xy: np.ndarray,
    max_radius_m: float,
    path_budget_m: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform random waypoints in the same convex visual-search disk."""
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xy must contain two finite values")
    if max_radius_m <= 0.0 or path_budget_m <= 0.0:
        raise ValueError("random search radius and path budget must be positive")
    rng = np.random.default_rng(int(seed))
    points = [center.copy()]
    arc = [0.0]
    while arc[-1] < path_budget_m:
        radius = max_radius_m * math.sqrt(float(rng.random()))
        angle = 2.0 * math.pi * float(rng.random())
        target = center + radius * np.array([math.cos(angle), math.sin(angle)])
        delta = target - points[-1]
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            continue
        remaining = path_budget_m - arc[-1]
        if length > remaining:
            target = points[-1] + (remaining / length) * delta
            length = remaining
        points.append(target)
        arc.append(arc[-1] + length)
    return np.asarray(arc, dtype=float), np.asarray(points, dtype=float)


def proposed_local_path() -> tuple[np.ndarray, np.ndarray]:
    _, xy = planned_outward_spiral(PROPOSED_PROFILE, INITIAL_XY_M, samples=8000)
    increments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(increments))), xy


def _draw_panel_text(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int, int]]],
    font: ImageFont.ImageFont,
    width: int,
) -> None:
    x, y = origin
    for value, color in lines:
        while value and draw.textlength(value, font=font) > width:
            value = value[:-1]
        draw.text((x, y), value, fill=color, font=font)
        y += 18


def _draw_entropy_grid_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    overlay: dict,
    font: ImageFont.ImageFont,
    *,
    task_aware: bool = False,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(
        box,
        radius=7,
        fill=(0, 0, 0, 188),
        outline=(225, 155, 240, 175),
        width=1,
    )
    draw.text(
        (left + 10, top + 7),
        (
            "posterior grid | capture probability + information gain"
            if task_aware
            else "grid belief | max expected entropy reduction"
        ),
        fill=(238, 210, 250, 245),
        font=font,
    )
    plot_left, plot_top = left + 12, top + 29
    plot_right, plot_bottom = right - 12, bottom - 42
    anchor = np.asarray(overlay["anchor_xy_m"], dtype=float)
    bound = float(overlay["visual_bound_m"])

    def project(point: np.ndarray) -> tuple[float, float]:
        normalized = (np.asarray(point, dtype=float) - anchor) / bound
        x = plot_left + 0.5 * (normalized[0] + 1.0) * (plot_right - plot_left)
        y = plot_bottom - 0.5 * (normalized[1] + 1.0) * (plot_bottom - plot_top)
        return float(x), float(y)

    center_x, center_y = project(anchor)
    radius_px = 0.5 * min(plot_right - plot_left, plot_bottom - plot_top)
    draw.ellipse(
        (center_x - radius_px, center_y - radius_px, center_x + radius_px, center_y + radius_px),
        outline=(165, 178, 192, 135),
        width=1,
    )
    candidate = np.asarray(overlay["candidate_xy_m"], dtype=float)
    weight = np.asarray(overlay["weights"], dtype=float)
    normalized_weight = weight / max(float(np.max(weight)), 1.0e-12)
    visible = np.flatnonzero(normalized_weight >= 0.025)
    for index in visible:
        x, y = project(candidate[index])
        intensity = int(55 + 190 * math.sqrt(float(normalized_weight[index])))
        draw.ellipse(
            (x - 1.4, y - 1.4, x + 1.4, y + 1.4),
            fill=(intensity, 95, 230, 205),
        )
    for index, point in enumerate(np.asarray(overlay["probed_xy_m"], dtype=float)):
        x, y = project(point)
        draw.ellipse((x - 3.2, y - 3.2, x + 3.2, y + 3.2), outline=(255, 205, 105, 235), width=2)
        draw.text((x + 4, y - 6), str(index + 1), fill=(255, 225, 155, 240), font=font)
    selected = overlay.get("selected_action_xy_m")
    if selected is not None:
        x, y = project(np.asarray(selected, dtype=float))
        draw.line((x - 5, y, x + 5, y), fill=(255, 255, 255, 245), width=2)
        draw.line((x, y - 5, x, y + 5), fill=(255, 255, 255, 245), width=2)
    mean_x, mean_y = project(np.asarray(overlay["posterior_mean_xy_m"], dtype=float))
    draw.ellipse((mean_x - 4, mean_y - 4, mean_x + 4, mean_y + 4), fill=(90, 225, 255, 245))
    draw.text(
        (left + 10, bottom - 35),
        (
            f"probe {overlay['probe_count']}/{overlay['max_probes']} | "
            f"H={overlay['entropy_nats']:.2f} nat | "
            f"r95={1000.0 * overlay['credible_radius_95_m']:.2f} mm"
        ),
        fill=(225, 235, 245, 240),
        font=font,
    )


def _draw_four_channel_fingerprint_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    history: list[tuple[float, float, float, float, float]],
    font: ImageFont.ImageFont,
) -> None:
    """Draw the exact four measured channels consumed by the improved model.

    Each history item is ``(phase, Fz, Tx, Ty, tip_z)`` in SI units.  The
    small plots use the extrema of the samples seen so far and print those
    extrema, so the animation is readable without pretending that a changing
    display scale is a fixed physical calibration.
    """

    left, top, right, bottom = rect
    draw.rounded_rectangle(
        rect,
        radius=7,
        fill=(0, 0, 0, 175),
        outline=(90, 225, 255, 145),
        width=1,
    )
    draw.text(
        (left + 9, top + 5),
        "improved measured fingerprint | Fz, Tx, Ty, tip-z",
        fill=(225, 242, 252, 245),
        font=font,
    )
    specs = (
        ("Fz [N]", 1, 1.0, (95, 215, 255, 245)),
        ("Tx [mNm]", 2, 1000.0, (255, 190, 95, 245)),
        ("Ty [mNm]", 3, 1000.0, (125, 235, 165, 245)),
        ("z [mm]", 4, 1000.0, (210, 155, 255, 245)),
    )
    header_h = 25
    gap = 5
    cell_w = (right - left - 3 * gap) // 2
    cell_h = (bottom - top - header_h - 3 * gap) // 2
    for index, (label, column, scale, color) in enumerate(specs):
        row, col = divmod(index, 2)
        x0 = left + gap + col * (cell_w + gap)
        y0 = top + header_h + gap + row * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        draw.rectangle((x0, y0, x1, y1), fill=(6, 11, 17, 220), outline=(100, 120, 138, 105))
        draw.text((x0 + 4, y0 + 2), label, fill=color, font=font)
        if not history:
            continue
        values = np.asarray([item[column] * scale for item in history], dtype=float)
        phases = np.asarray([item[0] for item in history], dtype=float)
        finite = np.isfinite(values) & np.isfinite(phases)
        if not np.any(finite):
            continue
        values = values[finite]
        phases = phases[finite]
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = max(hi - lo, 1.0e-9)
        plot_left, plot_right = x0 + 4, x1 - 4
        plot_top, plot_bottom = y0 + 15, y1 - 4
        points = []
        for phase, value in zip(phases, values):
            px = plot_left + (plot_right - plot_left) * float(phase % (2.0 * math.pi)) / (2.0 * math.pi)
            py = plot_bottom - (plot_bottom - plot_top) * (float(value) - lo) / span
            points.append((int(round(px)), int(round(py))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif points:
            px, py = points[0]
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
        draw.text(
            (x1 - 86, y0 + 2),
            f"{lo:+.2f}/{hi:+.2f}",
            fill=(185, 201, 215, 220),
            font=font,
        )


def _draw_direction_posterior_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    estimate: DirectionEstimate | None,
    font: ImageFont.ImageFont,
    *,
    planned_transport_distance_m: float | None,
    planned_spiral_radius_m: float | None,
) -> None:
    """Show the improved model's circular posterior and executed mean."""

    left, top, right, bottom = rect
    draw.rounded_rectangle(
        rect,
        radius=7,
        fill=(0, 0, 0, 180),
        outline=(90, 225, 255, 150),
        width=1,
    )
    if estimate is None:
        _draw_panel_text(
            draw,
            (left + 10, top + 9),
            [
                ("direction posterior pending", (225, 240, 250, 240)),
                ("4x360 measured input -> q(phi), APS, mean", (205, 220, 232, 235)),
            ],
            font,
            right - left - 18,
        )
        return
    center = np.asarray([left + 53.0, 0.5 * (top + bottom)], dtype=float)
    radius = max(22.0, min(37.0, 0.42 * (bottom - top)))
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        outline=(190, 205, 218, 120),
        width=1,
    )
    probability = np.asarray(estimate.direction_probabilities, dtype=float)
    if probability.size and float(np.max(probability)) > 0.0:
        angles = np.linspace(0.0, 2.0 * math.pi, probability.size, endpoint=False)
        scaled = probability / float(np.max(probability))
        aps = set(map(int, estimate.aps_direction_classes))
        for index, (angle, item) in enumerate(zip(angles, scaled)):
            length = radius * (0.18 + 0.72 * float(item))
            end = center + length * np.asarray([math.cos(angle), -math.sin(angle)])
            color = (255, 195, 85, 220) if index in aps else (90, 160, 205, 95)
            draw.line((center[0], center[1], end[0], end[1]), fill=color, width=2 if index in aps else 1)
    mean = math.radians(float(estimate.circular_mean_direction_deg))
    mean_end = center + radius * np.asarray([math.cos(mean), -math.sin(mean)])
    draw.line((center[0], center[1], mean_end[0], mean_end[1]), fill=(85, 245, 155, 255), width=3)
    draw.ellipse((center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2), fill=(245, 248, 250, 255))
    move = "pending" if planned_transport_distance_m is None else f"{1000.0 * planned_transport_distance_m:.3f} mm"
    reserve = "pending" if planned_spiral_radius_m is None else f"{1000.0 * planned_spiral_radius_m:.3f} mm"
    _draw_panel_text(
        draw,
        (left + 104, top + 8),
        [
            (f"mean={estimate.circular_mean_direction_deg:6.2f} deg | top={estimate.top_direction_deg:6.2f}", (225, 240, 250, 245)),
            (f"APS={len(estimate.aps_direction_classes)} | kappa={estimate.circular_concentration:.3f}", (225, 240, 250, 240)),
            (f"shortcut={move} | residual R={reserve}", (225, 240, 250, 240)),
        ],
        font,
        right - left - 112,
    )


def render_overlay_frame(
    method: str,
    stage: str,
    snapshot: DynamicSnapshot,
    metrics: LiveMetrics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    scene_option: mujoco.MjvOption,
    spec: dict,
    case: VideoCase,
    trail: list[np.ndarray],
    baseline_z_m: float | None,
    show_cone: bool,
    display_speed: float,
    polar_history: list[tuple[float, float, float, float, float]],
    estimate: DirectionEstimate | None = None,
    planned_transport_distance_m: float | None = None,
    planned_spiral_radius_m: float | None = None,
    observation_label: str | None = None,
    entropy_overlay: dict | None = None,
    direct_failure_policy_overlay: str = DIRECT_RETRY_FAILURE_POLICY_OVERLAY,
    fingerprint_display_mode: str = "improved_four_channel",
) -> np.ndarray:
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    camera_data = camera_snapshot(renderer.scene, WIDTH, HEIGHT)
    raw = renderer.render()
    raw = np.clip(raw.astype(np.float32) * 1.13 + 7.0, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(raw)
    draw = ImageDraw.Draw(image, "RGBA")
    font_title, font, font_mono = font_pair()
    infos = valid_contact_infos(model, data)
    tags = contact_tags(infos)
    xy = snapshot.tip_position[:2]
    trail.append(xy.copy())
    drawn = draw_contact_points(draw, camera_data, infos)
    sensor, force_sensor, torque_sensor = wrench_at_sensor(
        spec,
        xy,
        float(snapshot.tip_position[2]),
        float(snapshot.roll),
        float(snapshot.pitch),
        infos,
    )
    bit_len = float(spec["bit_length_m"])
    sensor_rot = np.eye(3)
    quat = np.asarray(data.qpos[3:7], dtype=float)
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quat)
    sensor_rot = matrix.reshape(3, 3)
    draw_axis_and_probe_cone(
        draw,
        camera_data,
        spec,
        xy,
        float(snapshot.tip_position[2]),
        float(snapshot.roll),
        float(snapshot.pitch),
        show_cone=show_cone,
    )
    draw_six_axis_wrench(
        draw,
        camera_data,
        sensor,
        sensor_rot,
        force_sensor,
        torque_sensor,
        font,
    )

    title = (
        "ASSUMED: probe + hand posterior + local outward"
        if method == "proposed"
        else "PROPOSED+: shortcut -> task-aware grid -> physical funnel"
        if method == "hybrid"
        else "LEARNED: frozen P1v3 + vision/F-T/CAD posterior"
        if method == "learned"
        else "JIN-STYLE DIRECT: Xz/Mx/My + adjusted re-probe"
        if method == "direct" and fingerprint_display_mode == "jin_three_channel"
        else "IMPROVED DIRECT: four-channel fingerprint + bounded ray"
        if method == "direct"
        else "PAPER-INSPIRED STRESS: scripted direct slide + retry"
        if method == "paper_direct"
        else "ENTROPY-GRID: CAD grid + maximum-entropy probing"
        if method == "entropy_grid"
        else "RANDOM: seeded area-uniform random route"
        if method == "random"
        else "BASELINE: source-style ordinary outward spiral"
    )
    draw.rounded_rectangle(
        (10, 10, 560, 214),
        radius=7,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 75),
        width=1,
    )
    draw.text((22, 20), title, fill=(255, 235, 170, 250), font=font_title)
    show_hybrid_grid = bool(
        method == "hybrid" and entropy_overlay is not None and estimate is not None
    )
    if (
        method in {"proposed", "learned", "hybrid", "direct"}
        and not show_hybrid_grid
        and fingerprint_display_mode != "jin_three_channel"
    ):
        _draw_four_channel_fingerprint_panel(
            draw,
            (570, 10, 948, 151),
            polar_history,
            font_mono,
        )
        _draw_direction_posterior_panel(
            draw,
            (570, 155, 948, 244),
            estimate,
            font_mono,
            planned_transport_distance_m=planned_transport_distance_m,
            planned_spiral_radius_m=planned_spiral_radius_m,
        )
    elif method == "paper_direct" or (
        method == "direct" and fingerprint_display_mode == "jin_three_channel"
    ):
        # The Jin-style paper proxy intentionally keeps the paper's three
        # Xz/Mx/My polar channels.  It must not be visually upgraded to the
        # improved model's four-channel interface.
        paper_history = [
            (phase, tip_z, tx, ty)
            for phase, _fz, tx, ty, tip_z in polar_history
        ]
        draw_polar_fingerprint_panel(
            draw,
            (570, 10, 948, 143),
            paper_history,
            font,
        )
        draw.rounded_rectangle(
            (570, 149, 948, 244),
            radius=7,
            fill=(0, 0, 0, 180),
            outline=(255, 195, 90, 145),
            width=1,
        )
        _draw_panel_text(
            draw,
            (582, 158),
            [
                ("Jin-2021-style M8 proxy", (255, 224, 160, 245)),
                ("input: Xz/Mx/My polar binary images", (225, 236, 245, 240)),
                ("output: 12-sector direction class", (225, 236, 245, 240)),
                ("failure -> shifted reset -> full re-probe", (225, 236, 245, 240)),
            ],
            font_mono,
            350,
        )
    elif method in {"entropy_grid", "hybrid"} and entropy_overlay is not None:
        _draw_entropy_grid_panel(
            draw,
            (570, 10, 948, 244),
            entropy_overlay,
            font_mono,
            task_aware=method == "hybrid",
        )
    else:
        speed_color = (
            (255, 205, 105, 245)
            if display_speed > 1.0
            else (210, 225, 235, 225)
        )
        draw.rounded_rectangle(
            (704, 12, 948, 48),
            radius=7,
            fill=(0, 0, 0, 180),
            outline=speed_color,
            width=1,
        )
        draw.text(
            (718, 20),
            f"DISPLAY {display_speed:.0f}x | SIM {metrics.active_time_s:05.1f}s",
            fill=speed_color,
            font=font_mono,
        )
    depth_drop_mm = 0.0 if baseline_z_m is None else 1000.0 * (
        baseline_z_m - float(snapshot.tip_position[2])
    )
    lines = [
        (f"stage: {stage}", (235, 245, 255, 240)),
        (f"comparison time: {metrics.active_time_s:.2f} s", (235, 245, 255, 240)),
        (f"tip xy=({1000*xy[0]:+.2f},{1000*xy[1]:+.2f}) mm  dz={depth_drop_mm:+.3f} mm", (235, 245, 255, 230)),
        (f"F=({force_sensor[0]:+.2f},{force_sensor[1]:+.2f},{force_sensor[2]:+.2f}) N", (235, 245, 255, 230)),
        (f"tau_xy={1000*np.linalg.norm(torque_sensor[:2]):.2f} mNm  tags={','.join(sorted(tags)) or 'none'}", (235, 245, 255, 230)),
        (f"XY path={1000*metrics.actual_xy_path_m:.1f} mm", (90, 225, 255, 245)),
        (f"wear proxy={1000*metrics.wear_proxy_j:.3f} mJ", (255, 190, 105, 245)),
        (f"contacts={len(infos)} shown={drawn}", (200, 215, 230, 220)),
    ]
    if observation_label:
        lines.append((observation_label, (255, 205, 105, 245)))
    _draw_panel_text(draw, (22, 50), lines, font_mono, 520)

    draw.rounded_rectangle(
        (14, 590, 584, 704),
        radius=7,
        fill=(0, 0, 0, 155),
        outline=(255, 255, 255, 70),
        width=1,
    )
    if method == "proposed":
        formula = [
            "same start / force / pitch / XY speed as baseline",
            "15 deg full probe -> assumed posterior transport -> local outward",
            f"Fd={TARGET_DOWN_FORCE_N:.1f} N, pitch={1000*SPIRAL_PITCH_M:.2f} mm/rev, v={1000*COMMON_XY_SPEED_M_S:.2f} mm/s",
            "probe time and probe contact work are included",
        ]
    elif method == "hybrid":
        formula = [
            "visual + conical F/T + CAD initialize a tempered center belief",
            "U = capture probability + information gain - travel/contact cost",
            "no-capture path softly excludes swept hypotheses; surprising F/T is downweighted",
            "no local spiral; global outward remains only after local grid exhaustion",
        ]
    elif method == "learned":
        formula = [
            "frozen P1v3: q(phi), p(state|F/T) + visual prior",
            "approx CAD support annuli are soft; no true-center/contact tags",
            "structured gate -> bounded transport -> dense outward search",
            f"Fd={TARGET_DOWN_FORCE_N:.1f} N, pitch={1000*SPIRAL_PITCH_M:.2f} mm/rev, v={1000*COMMON_XY_SPEED_M_S:.2f} mm/s",
        ]
    elif method == "direct":
        formula = [
            (
                "Jin-style direct: Xz/Mx/My polar CNN -> 12-sector direction"
                if fingerprint_display_mode == "jin_three_channel"
                else "improved direct: Fz/Tx/Ty/z -> 48-bin circular posterior"
            ),
            "straight admittance ray; no CAD radius and no true-distance input",
            direct_failure_policy_overlay,
            f"Fd={TARGET_DOWN_FORCE_N:.1f} N, v={1000*COMMON_XY_SPEED_M_S:.2f} mm/s",
        ]
    elif method == "paper_direct":
        formula = [
            "Jin et al. 2021-inspired: tilt-rotate -> direction class",
            "direct admittance slide has direction only, no distance estimate",
            "failed insertion -> slightly shifted reset -> full re-probe",
            "scripted class-boundary stress; not exact paper reproduction",
        ]
    elif method == "entropy_grid":
        formula = [
            "strong transparent diagnostic: visual/CAD grid + signed F/T cue",
            "each guarded touchdown selects max expected entropy reduction",
            "after K probes: unload/lift/shift + guarded posterior-mean touchdown",
            "all lift/shift/touchdown time, path and contact work are counted",
        ]
    elif method == "random":
        formula = [
            "random diagnostic: area-uniform seeded random waypoints",
            "same visual disk / XY speed / down-force / path budget as outward",
            "no conical probe, learned direction, CAD likelihood, or true center",
            "one preregistered seed per paired task; report seed distribution",
        ]
    else:
        formula = [
            "source-style baseline: uninterrupted outward Archimedean sweep",
            "stable down-force; no posterior; fixed phase in tool frame",
            f"Fd={TARGET_DOWN_FORCE_N:.1f} N, pitch={1000*SPIRAL_PITCH_M:.2f} mm/rev, v={1000*COMMON_XY_SPEED_M_S:.2f} mm/s",
            "not exact paper reproduction: no 20 rpm or motor-torque verifier",
        ]
    _draw_panel_text(
        draw,
        (26, 602),
        [(line, (235, 245, 255, 232)) for line in formula],
        font,
        540,
    )
    profile = PROPOSED_PROFILE if method == "proposed" and not show_cone else None
    draw_plan_panel(
        draw,
        spec,
        xy,
        trail,
        infos,
        font,
        "same-start XY plan | yellow = MuJoCo contacts",
        profile=profile,
    )
    return np.asarray(image)


def summary_frame(frame: np.ndarray, result: RunResult) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_title, font, font_mono = font_pair()
    draw.rounded_rectangle(
        (150, 145, 810, 570),
        radius=8,
        fill=(0, 0, 0, 220),
        outline=(255, 230, 155, 220),
        width=2,
    )
    title = (
        "ASSUMED-POSTERIOR RESULT"
        if result.method == "proposed"
        else "PROPOSED+ JOINT-FUNNEL RESULT"
        if result.method == "hybrid"
        else "FROZEN P1V3 LEARNED RESULT"
        if result.method == "learned"
        else "SHARED-PROBE DIRECT BASELINE RESULT"
        if result.method == "direct"
        else "MAX-ENTROPY GRID BASELINE RESULT"
        if result.method == "entropy_grid"
        else "SOURCE-STYLE BASELINE RESULT"
    )
    draw.text((178, 170), title, fill=(255, 235, 170, 255), font=font_title)
    rows = [
        f"result: {result.result}",
        f"time to inner-wall gate: {result.time_to_inner_wall_s:.2f} s" if result.time_to_inner_wall_s is not None else "time to inner-wall gate: not reached",
        f"actual XY path: {result.actual_xy_path_mm:.2f} mm",
        f"contact XY path: {result.contact_xy_path_mm:.2f} mm",
        f"lateral work proxy: {result.lateral_work_mj:.4f} mJ",
        f"orientation work proxy: {result.orientation_work_mj:.4f} mJ",
        f"combined wear proxy: {result.wear_proxy_mj:.4f} mJ",
        f"max force / torque: {result.max_force_n:.2f} N / {result.max_torque_mnm:.2f} mNm",
    ]
    if result.method in {"learned", "hybrid"}:
        rows.extend(
            [
                f"control direction / APS: {result.control_direction_deg:.2f} deg / {result.aps_set_size}" if result.control_direction_deg is not None else "control direction: unavailable",
                f"gate / move / Rspiral: {result.transport_allowed} / {result.transport_distance_mm:.3f} / {result.spiral_max_radius_mm:.3f} mm" if result.transport_distance_mm is not None and result.spiral_max_radius_mm is not None else "learned plan: unavailable",
                f"CAD bottom verified: {result.bottom_reached} at {result.time_to_bottom_s:.2f} s" if result.time_to_bottom_s is not None else f"CAD bottom verified: {result.bottom_reached}",
                "single simulation case; sensor-only handoff, offline geometry score",
            ]
        )
        if result.method == "hybrid":
            rows.extend(
                [
                    f"task-aware grid contacts: {result.entropy_grid_probe_count}",
                    f"local spiral / global fallback: {result.entropy_grid_used_spiral} / {result.global_outward_fallback_used}",
                    f"terminal mode: {result.terminal_verification_mode}",
                ]
            )
    elif result.method == "direct":
        rows.extend(
            [
                f"control direction / APS: {result.control_direction_deg:.2f} deg / {result.aps_set_size}" if result.control_direction_deg is not None else "direction: unavailable",
                f"direct attempts: {result.direct_attempts_completed}/{result.direct_max_attempts} | success={result.direct_successful_attempt}",
                f"direct ray limit: {result.transport_distance_mm:.3f} mm" if result.transport_distance_mm is not None else "direct ray: unavailable",
                f"CAD bottom verified: {result.bottom_reached} at {result.time_to_bottom_s:.2f} s" if result.time_to_bottom_s is not None else f"CAD bottom verified: {result.bottom_reached}",
                "same-rollout unload/lift/shift/re-probe; no simulator reset",
            ]
        )
    elif result.method == "proposed":
        rows.append("hand-assumed posterior mechanism illustration; not learned")
    elif result.method == "entropy_grid":
        rows.extend(
            [
                (
                    f"grid probes / r95: {result.entropy_grid_probe_count} / "
                    f"{result.entropy_grid_credible_radius_mm:.3f} mm"
                    if result.entropy_grid_credible_radius_mm is not None
                    else "grid posterior: unavailable"
                ),
                "guarded posterior-mean touchdown; no spiral fallback",
                "online belief uses CAD/F-T likelihood, never true center/tags",
            ]
        )
    else:
        rows.append("source-style geometry baseline; no direction posterior")
    _draw_panel_text(
        draw,
        (180, 215),
        [(row, (235, 245, 255, 245)) for row in rows],
        font_mono,
        600,
    )
    return np.asarray(image)


def run_method(
    method: str,
    output_path: Path | None = None,
    summary_result: RunResult | None = None,
    *,
    initial_xy_m: np.ndarray | None = None,
    direction_runtime: M8DirectionEnsemble | None = None,
    visual_sigma_m: float = 0.0020,
    visual_bound_m: float | None = None,
    posterior_coverage: float = 0.95,
    posterior_seed: int = 20260710,
    max_transport_m: float | None = 0.00150,
    baseline_phase_rad: float = BASELINE_PHASE_RAD,
    baseline_max_radius_m: float | None = None,
    target_down_force_n: float = TARGET_DOWN_FORCE_N,
    friction_slide: float | None = None,
    contact_timeconst_s: float | None = None,
    global_outward_fallback: bool = True,
    fast_forward_after_active_s: float | None = None,
    fast_forward_speed: float = 1.0,
    active_time_budget_s: float | None = None,
    trace_output_path: Path | None = None,
    continue_to_bottom: bool = False,
    capture_yaw_assist: CaptureYawAssistConfig | None = None,
    funnel_guard: FunnelGuardConfig | None = None,
    insertion_command_bias_xy_m: np.ndarray | None = None,
    post_capture_tilt_disturbance_deg: float = 0.0,
    post_capture_tilt_axis_deg: float = 0.0,
    post_capture_lateral_disturbance_m: float = 0.0,
    post_capture_disturbance_persistent: bool = False,
    post_capture_tilt_ramp_s: float = 0.12,
    post_capture_tilt_hold_s: float = 0.08,
    wrench_observation: WrenchObservationConfig | None = None,
    admittance_config: AdmittanceConfig | None = None,
    observation_label: str | None = None,
    use_state_cad_fusion: bool = True,
    random_search_seed: int = 20260712,
    random_search_path_budget_m: float | None = None,
    entropy_grid_config: EntropyGridConfig | None = None,
    hybrid_probe_config: HybridProbeConfig | None = None,
    hybrid_residual_coverage_config: HybridResidualCoverageConfig | None = None,
    spec_path: str | Path | None = None,
    model_path: str | Path | None = None,
    spiral_pitch_m: float | None = None,
    xy_speed_m_s: float = COMMON_XY_SPEED_M_S,
    cad_conditioned_insertion: bool = False,
    conditioned_on_ambiguous_top_contact: bool = False,
    direct_failure_policy_overlay: str = DIRECT_RETRY_FAILURE_POLICY_OVERLAY,
    direct_max_attempts: int = 1,
    direct_adjustment_m: float = 0.00025,
    fingerprint_display_mode: str = "improved_four_channel",
    oracle_transport_soft_escape_recovery: bool = False,
    oracle_center_tolerance_capture_fraction: float = 0.25,
) -> RunResult:
    if method not in {
        "proposed",
        "baseline",
        "learned",
        "direct",
        "oracle",
        "entropy_grid",
        "hybrid",
        "random",
    }:
        raise ValueError(method)
    case = comparison_case(method)
    spec, model, data, rollout, renderer, camera, scene_option = setup(
        case,
        render=output_path is not None,
        wrench_observation=wrench_observation,
        admittance_config=admittance_config,
        spec_path=spec_path,
        model_path=model_path,
    )
    if target_down_force_n <= 0.0:
        raise ValueError("target_down_force_n must be positive")
    if xy_speed_m_s <= 0.0:
        raise ValueError("xy_speed_m_s must be positive")
    if friction_slide is not None:
        if friction_slide <= 0.0:
            raise ValueError("friction_slide must be positive")
        model.geom_friction[:, 0] = float(friction_slide)
    if contact_timeconst_s is not None:
        if contact_timeconst_s <= 0.0:
            raise ValueError("contact_timeconst_s must be positive")
        model.geom_solref[:, 0] = float(contact_timeconst_s)
    initial_xy = (
        INITIAL_XY_M.copy()
        if initial_xy_m is None
        else np.asarray(initial_xy_m, dtype=float).copy()
    )
    if initial_xy.shape != (2,) or not np.all(np.isfinite(initial_xy)):
        raise ValueError("initial_xy_m must contain two finite values")
    if method in {"learned", "hybrid", "direct"} and direction_runtime is None:
        raise ValueError(f"{method} method requires direction_runtime")
    if visual_bound_m is not None and visual_bound_m <= 0.0:
        raise ValueError("visual_bound_m must be positive when provided")
    if not 0.5 < posterior_coverage < 1.0:
        raise ValueError("posterior_coverage must be between 0.5 and 1")
    if random_search_path_budget_m is not None and random_search_path_budget_m <= 0.0:
        raise ValueError("random_search_path_budget_m must be positive when provided")
    time_budget_s = (
        COMMON_ACTIVE_TIME_BUDGET_S
        if active_time_budget_s is None
        else float(active_time_budget_s)
    )
    if time_budget_s <= 0.0:
        raise ValueError("active_time_budget_s must be positive")
    if not 1 <= int(direct_max_attempts) <= 3:
        raise ValueError("direct_max_attempts must be in [1, 3]")
    direct_max_attempts = int(direct_max_attempts)
    if not math.isfinite(direct_adjustment_m) or direct_adjustment_m <= 0.0:
        raise ValueError("direct_adjustment_m must be finite and positive")
    if fingerprint_display_mode not in {
        "improved_four_channel",
        "jin_three_channel",
    }:
        raise ValueError("unsupported fingerprint_display_mode")
    if not 0.0 < float(oracle_center_tolerance_capture_fraction) < 1.0:
        raise ValueError(
            "oracle_center_tolerance_capture_fraction must lie in (0, 1)"
        )
    yaw_assist = capture_yaw_assist or CaptureYawAssistConfig()
    funnel_config = funnel_guard or FunnelGuardConfig()
    insertion_bias_xy = (
        np.zeros(2, dtype=float)
        if insertion_command_bias_xy_m is None
        else np.asarray(insertion_command_bias_xy_m, dtype=float).copy()
    )
    if insertion_bias_xy.shape != (2,) or not np.all(np.isfinite(insertion_bias_xy)):
        raise ValueError("insertion_command_bias_xy_m must contain two finite values")
    if float(np.linalg.norm(insertion_bias_xy)) > 0.00025:
        raise ValueError("insertion command bias exceeds the bounded 0.25 mm stress range")
    disturbance_tilt_deg = float(post_capture_tilt_disturbance_deg)
    disturbance_axis_deg = float(post_capture_tilt_axis_deg)
    disturbance_lateral_m = float(post_capture_lateral_disturbance_m)
    disturbance_ramp_s = float(post_capture_tilt_ramp_s)
    disturbance_hold_s = float(post_capture_tilt_hold_s)
    disturbance_persistent = bool(post_capture_disturbance_persistent)
    if not math.isfinite(disturbance_tilt_deg) or not 0.0 <= disturbance_tilt_deg <= 8.0:
        raise ValueError("post_capture_tilt_disturbance_deg must be in [0, 8]")
    if not math.isfinite(disturbance_axis_deg):
        raise ValueError("post_capture_tilt_axis_deg must be finite")
    if not math.isfinite(disturbance_lateral_m) or not 0.0 <= disturbance_lateral_m <= 0.001:
        raise ValueError("post_capture_lateral_disturbance_m must be in [0, 0.001]")
    if not math.isfinite(disturbance_ramp_s) or disturbance_ramp_s <= 0.0:
        raise ValueError("post_capture_tilt_ramp_s must be positive")
    if not math.isfinite(disturbance_hold_s) or disturbance_hold_s < 0.0:
        raise ValueError("post_capture_tilt_hold_s must be non-negative")
    nominal_tip = np.array([*initial_xy, NOMINAL_TIP_Z_M], dtype=float)
    rollout.set_target_down_force(target_down_force_n)
    rollout.set_initial_tip_pose(nominal_tip)
    metrics = LiveMetrics()
    clock_s = 0.0
    next_render_s = 0.0
    frame_count = 0
    trail: list[np.ndarray] = []
    # phase, measured Fz, measured Tx, measured Ty, measured tip-z.  The
    # improved model consumes all four physical channels; the paper proxy
    # renderer derives its historical Xz/Mx/My view from the same samples.
    polar_history: list[tuple[float, float, float, float, float]] = []
    writer = None
    last_frame: np.ndarray | None = None
    min_tip_z_m = float("inf")
    hard_stop = False
    hard_stop_reason: str | None = None
    result_name = "not_reached"
    inner_wall_time_s: float | None = None
    candidate_attempts = 0
    event_xy: np.ndarray | None = None
    baseline_z_m: float | None = None
    current_stage = "contact settle"
    current_phase: float | None = None
    scan_phase: list[float] = []
    scan_wrench: list[np.ndarray] = []
    scan_tip_z: list[float] = []
    estimate: DirectionEstimate | None = None
    fingerprint_sha256: str | None = None
    fingerprint_sha256_attempts: list[str] = []
    conical_probe_attempt_records: list[dict] = []
    conical_direction_inference_count = 0
    conical_capture_handoff_count = 0
    conical_incomplete_fallback_used = False
    initial_conical_record: dict | None = None
    direct_attempt_records: list[dict] = []
    direct_successful_attempt: int | None = None
    direct_attempts_completed = 0
    global_outward_fallback_used = False
    global_outward_fallback_started_active_time_s: float | None = None
    cad_geometry = cad_radial_geometry_from_spec(spec)
    effective_capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    configured_spiral_pitch_m = (
        2.0 * effective_capture_radius_m * 0.80
        if spiral_pitch_m is None
        else float(spiral_pitch_m)
    )
    if configured_spiral_pitch_m <= 0.0:
        raise ValueError("spiral_pitch_m must be positive")
    if configured_spiral_pitch_m > 2.0 * effective_capture_radius_m + 1.0e-12:
        raise ValueError(
            "spiral pitch exceeds the adjacent-turn spacing bound 2*r_eff"
        )
    planned_spiral_pitch_m = configured_spiral_pitch_m
    hybrid_config = hybrid_probe_config or HybridProbeConfig()
    hybrid_residual_config = (
        hybrid_residual_coverage_config or HybridResidualCoverageConfig()
    )
    if hybrid_residual_config.enabled and method != "hybrid":
        raise ValueError(
            "hybrid_residual_coverage_config is only valid for method='hybrid'"
        )
    if method == "hybrid" and global_outward_fallback:
        if not hybrid_config.continuous_contact_path:
            raise ValueError(
                "nominal fallback scheduling requires continuous_contact_path=True"
            )
        if hybrid_config.local_spiral_fallback_enabled:
            raise ValueError(
                "nominal global-fallback scheduling requires local spiral disabled"
            )
    entropy_config = entropy_grid_config or EntropyGridConfig(
        belief_spacing_m=min(0.00025, effective_capture_radius_m),
        action_spacing_m=min(
            0.00075,
            0.95 * math.sqrt(2.0) * effective_capture_radius_m,
        ),
        credible_radius_stop_m=effective_capture_radius_m,
        max_probes=(
            hybrid_config.safety_probe_ceiling if method == "hybrid" else 64
        )
    )
    entropy_belief = (
        EntropyGridBelief(
            initial_xy,
            visual_sigma_m,
            (
                float(visual_bound_m)
                if visual_bound_m is not None
                else float(baseline_max_radius_m or BASELINE_MAX_RADIUS_M)
            ),
            0.5 * float(spec["derived"]["socket_top_af_m"]),
            0.5 * float(spec["screw_head_diameter_m"]),
            float(spec["key_af_m"]) / math.sqrt(3.0),
            entropy_config,
        )
        if method in {"entropy_grid", "hybrid"}
        else None
    )
    entropy_initial_nats = (
        None if entropy_belief is None else entropy_belief.entropy_nats
    )
    entropy_selected_action: EntropyAction | None = None
    entropy_observation_records: list[dict] = []
    hybrid_gate_records: list[dict] = []
    hybrid_spiral_executed = False
    hybrid_spiral_fallback_reason: str | None = None
    hybrid_fallback_nominal_reserve_s: float | None = None
    hybrid_entry_fallback_feasible: bool | None = None
    hybrid_terminal_execution_reserve_s: float | None = None
    hybrid_budget_switch_active_time_s: float | None = None
    hybrid_budget_switch_reason: str | None = None
    hybrid_guarded_motion_failure_reason: str | None = None
    posterior_radial_quantiles_mm: dict[str, float] | None = None
    transport_policy: str | None = None
    transport_direction_quality: float | None = None
    transport_quantile: float | None = None
    transport_reverse_probability: float | None = None
    transport_projected_quantiles_mm: dict[str, float] | None = None
    transport_raw_distance_mm: float | None = None
    transport_distance_cap_mm: float | None = None
    shortcut_telemetry = ShortcutPlanTelemetry()
    planned_target_xy: np.ndarray | None = None
    planned_spiral_radius_m: float | None = None
    planned_transport_distance_m: float | None = None
    planned_target_is_latest_plan = False
    planned_start_xy = initial_xy.copy()
    actual_random_path_budget_m: float | None = None
    random_waypoint_count: int | None = None
    capture_mode: str | None = None
    bottom_reached = False
    bottom_time_s: float | None = None
    bottom_depth_error_mm: float | None = None
    terminal_verification_mode: str | None = None
    terminal_outcome_snapshot: DynamicSnapshot | None = None
    terminal_outcome_contact_tags: list[str] | None = None
    terminal_outcome_hard_stop: bool | None = None
    terminal_outcome_hard_stop_reason: str | None = None
    post_terminal_event_display_new_hard_stop = False
    post_terminal_event_display_hard_stop_reason: str | None = None
    fast_forward_start_frame: int | None = None
    depth_commit_latched = False
    yaw_assist_aborted = False
    yaw_assist_used = False
    yaw_assist_elapsed_s = 0.0
    max_abs_tau_z_mnm = 0.0
    capture_handoff_used = False
    sensor_progress_candidate_count = 0
    search_recovery_count = 0
    oracle_transport_recovery_enabled = bool(
        method == "oracle" and oracle_transport_soft_escape_recovery
    )
    oracle_transport_recovery_count = 0
    oracle_transport_center_tolerance_m = (
        float(oracle_center_tolerance_capture_fraction)
        * float(spec["derived"]["r_cap_top_m"])
        if oracle_transport_recovery_enabled
        else None
    )
    oracle_transport_final_center_residual_m: float | None = None
    oracle_transport_recovery_outcome: str | None = None
    oracle_transport_recovery_records: list[dict] = []
    fallback_physical_reset_used = False
    fallback_budget_extension_s = 0.0
    budget_exhaustion_stage: str | None = None
    fallback_skipped_reason: str | None = None
    preprobe_route = _declared_preprobe_route(
        method,
        bool(conditioned_on_ambiguous_top_contact),
    )
    soft_escape_requested = False
    funnel_monitor = FunnelGuard(funnel_config)
    funnel_observation: FunnelGuardObservation = funnel_monitor.last
    funnel_guard_active = False
    funnel_allow_stall = False
    funnel_recovery_requested = False
    funnel_recovery_reason: str | None = None
    funnel_recovery_count = 0
    funnel_recovery_reasons: list[str] = []
    funnel_recovery_xy_rebases_mm: list[list[float]] = []
    terminal_attempts = 0
    terminal_rejection_count = 0
    terminal_attempt_diagnostics: list[dict] = []
    terminal_entry_rebase_records: list[dict] = []
    transition_telemetry_records: list[dict] = []
    oracle_endpoint_settle_records: list[dict] = []
    post_release_requalification_records: list[dict] = []
    post_release_requalified: bool | None = None
    funnel_recovery_transition_records: list[dict] = []
    terminal_axial_position_hold_used = False
    funnel_abort_reason: str | None = None
    funnel_peak_power_w = 0.0
    funnel_min_health = float("inf")
    funnel_tip_rcc_used = False
    funnel_minimum_tip_z_m: float | None = None
    funnel_accumulated_energy_j = 0.0
    post_capture_disturbance_recovered = False
    post_capture_disturbance_reason: str | None = None
    residual_insertion_rpy = np.zeros(3, dtype=float)
    progress_depth_samples: deque[tuple[float, float]] = deque(maxlen=500)
    trace_rows: list[dict] | None = [] if trace_output_path is not None else None
    latest_snapshot: DynamicSnapshot | None = None
    last_tip_command = nominal_tip.copy()
    last_rpy_command = np.zeros(3, dtype=float)
    pending_transition_trace: dict[str, object] | None = None

    if fast_forward_speed < 1.0:
        raise ValueError("fast_forward_speed must be at least 1")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            output_path,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

    def maybe_render(snapshot: DynamicSnapshot, show_cone: bool = False) -> None:
        nonlocal next_render_s, frame_count, last_frame, fast_forward_start_frame
        if writer is None:
            return
        assert renderer is not None
        while clock_s + 1.0e-12 >= next_render_s:
            display_speed = (
                fast_forward_speed
                if fast_forward_after_active_s is not None
                and metrics.active_time_s >= fast_forward_after_active_s
                else 1.0
            )
            if display_speed > 1.0 and fast_forward_start_frame is None:
                fast_forward_start_frame = frame_count
            if (
                method in {"proposed", "learned", "hybrid", "direct"}
                and "full 360 deg conical fingerprint" in current_stage
                and current_phase is not None
                and (
                    not polar_history
                    or current_phase > polar_history[-1][0] + 1.0e-6
                )
            ):
                polar_history.append(
                    (
                        current_phase,
                        float(snapshot.measured_wrench[2]),
                        float(snapshot.measured_wrench[3]),
                        float(snapshot.measured_wrench[4]),
                        float(snapshot.tip_position[2]),
                    )
                )
            frame = render_overlay_frame(
                method,
                current_stage,
                snapshot,
                metrics,
                model,
                data,
                renderer,
                camera,
                scene_option,
                spec,
                case,
                trail,
                baseline_z_m,
                show_cone,
                display_speed,
                polar_history,
                estimate,
                planned_transport_distance_m,
                planned_spiral_radius_m,
                observation_label,
                (
                    None
                    if entropy_belief is None
                    else entropy_belief.overlay(
                        None
                        if entropy_selected_action is None
                        else entropy_selected_action.xy_m
                    )
                ),
                direct_failure_policy_overlay,
                fingerprint_display_mode,
            )
            writer.append_data(frame)
            last_frame = frame
            frame_count += 1
            next_render_s += display_speed / FPS

    def step_once(
        tip_command: np.ndarray,
        rpy_command: np.ndarray,
        stage_group: str,
        active: bool,
        show_cone: bool = False,
    ) -> DynamicSnapshot:
        nonlocal clock_s, min_tip_z_m, hard_stop, max_abs_tau_z_mnm, current_stage
        nonlocal yaw_assist_aborted, funnel_observation
        nonlocal funnel_recovery_requested, funnel_recovery_reason
        nonlocal funnel_peak_power_w, funnel_min_health
        nonlocal latest_snapshot, budget_exhaustion_stage, hard_stop_reason
        nonlocal last_tip_command, last_rpy_command, pending_transition_trace
        active_dt_s = (
            _bounded_active_dt(
                metrics.active_time_s,
                SIM_TIMESTEP_S,
                time_budget_s,
            )
            if active
            else SIM_TIMESTEP_S
        )
        if active and active_dt_s <= 0.0:
            if budget_exhaustion_stage is None:
                budget_exhaustion_stage = current_stage
            if latest_snapshot is None:
                raise RuntimeError("active budget exhausted before initial snapshot")
            return latest_snapshot
        snapshot = rollout.step(tip_command, rpy_command)
        last_tip_command = np.asarray(tip_command, dtype=float).copy()
        last_rpy_command = np.asarray(rpy_command, dtype=float).copy()
        latest_snapshot = snapshot
        clock_s += SIM_TIMESTEP_S
        metrics.observe(snapshot, active_dt_s, stage_group, data.ncon > 0, active)
        if active and metrics.active_time_s >= time_budget_s - 1.0e-12:
            metrics.active_time_s = min(metrics.active_time_s, time_budget_s)
            if budget_exhaustion_stage is None:
                budget_exhaustion_stage = current_stage
        if funnel_guard_active:
            funnel_observation = funnel_monitor.observe(
                snapshot.tip_position,
                np.array([snapshot.roll, snapshot.pitch, snapshot.yaw]),
                snapshot.measured_wrench,
                SIM_TIMESTEP_S,
                allow_stall=funnel_allow_stall,
                minimum_tip_z_m=funnel_minimum_tip_z_m,
            )
            funnel_peak_power_w = max(
                funnel_peak_power_w,
                funnel_observation.injected_power_w,
            )
            funnel_min_health = min(funnel_min_health, funnel_observation.health)
            if funnel_observation.recovery_requested:
                funnel_recovery_requested = True
                funnel_recovery_reason = funnel_observation.reason
        if baseline_z_m is not None and active:
            progress_depth_samples.append(
                (metrics.active_time_s, float(snapshot.tip_position[2]))
            )
        min_tip_z_m = min(min_tip_z_m, float(snapshot.tip_position[2]))
        max_abs_tau_z_mnm = max(
            max_abs_tau_z_mnm,
            1000.0 * abs(float(snapshot.measured_wrench[5])),
        )
        yaw_assist_stage = stage_group in {
            "spiral",
            "progress_dwell",
            "descent_confirmation",
            "insertion_handoff",
            "upright_insertion",
        }
        if (
            method not in {"baseline", "entropy_grid", "random"}
            and yaw_assist_stage
            and yaw_assist.exceeds_soft_tau_z(float(snapshot.measured_wrench[5]))
        ):
            yaw_assist_aborted = True
            if stage_group == "spiral":
                current_stage = (
                    "local outward spiral | micro-yaw disabled by tau_z soft limit"
                )
            elif stage_group == "progress_dwell":
                current_stage = "capture dwell | micro-yaw disabled by tau_z soft limit"
            elif stage_group == "descent_confirmation":
                current_stage = (
                    "guarded descent | micro-yaw disabled by tau_z soft limit"
                )
            else:
                current_stage = (
                    "compliant insertion | micro-yaw disabled by tau_z soft limit"
                )
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        if hard_stop and hard_stop_reason is None:
            reasons = []
            if snapshot.state_invalid:
                reasons.append(f"state_invalid:{snapshot.failure_reason}")
            if snapshot.hard_stop_reason:
                reasons.append(snapshot.hard_stop_reason)
            hard_stop_reason = ";".join(reasons) or "hard_stop_latched_unknown_source"
        if trace_rows is not None:
            trace_rows.append(
                {
                    "clock_s": clock_s,
                    "active_time_s": metrics.active_time_s,
                    "stage": current_stage,
                    "stage_group": stage_group,
                    "tip_x_m": float(snapshot.tip_position[0]),
                    "tip_y_m": float(snapshot.tip_position[1]),
                    "tip_z_m": float(snapshot.tip_position[2]),
                    "nominal_x_m": float(snapshot.nominal_tip_position[0]),
                    "nominal_y_m": float(snapshot.nominal_tip_position[1]),
                    "nominal_z_m": float(snapshot.nominal_tip_position[2]),
                    "roll_rad": float(snapshot.roll),
                    "pitch_rad": float(snapshot.pitch),
                    "yaw_rad": float(snapshot.yaw),
                    "command_yaw_rad": float(rpy_command[2]),
                    "capture_yaw_assist_enabled": bool(
                        yaw_assist.enabled
                        and method not in {"baseline", "entropy_grid", "random"}
                    ),
                    "capture_yaw_assist_aborted": bool(yaw_assist_aborted),
                    "Fx_N": float(snapshot.measured_wrench[0]),
                    "Fy_N": float(snapshot.measured_wrench[1]),
                    "Fz_N": float(snapshot.measured_wrench[2]),
                    "Tx_Nm": float(snapshot.measured_wrench[3]),
                    "Ty_Nm": float(snapshot.measured_wrench[4]),
                    "Tz_Nm": float(snapshot.measured_wrench[5]),
                    "raw_physical_Fx_N": float(snapshot.physical_wrench[0]),
                    "raw_physical_Fy_N": float(snapshot.physical_wrench[1]),
                    "raw_physical_Fz_N": float(snapshot.physical_wrench[2]),
                    "raw_physical_Tx_Nm": float(snapshot.physical_wrench[3]),
                    "raw_physical_Ty_Nm": float(snapshot.physical_wrench[4]),
                    "raw_physical_Tz_Nm": float(snapshot.physical_wrench[5]),
                    "filtered_Fx_N": float(snapshot.command.filtered_wrench[0]),
                    "filtered_Fy_N": float(snapshot.command.filtered_wrench[1]),
                    "filtered_Fz_N": float(snapshot.command.filtered_wrench[2]),
                    "filtered_Tx_Nm": float(snapshot.command.filtered_wrench[3]),
                    "filtered_Ty_Nm": float(snapshot.command.filtered_wrench[4]),
                    "filtered_Tz_Nm": float(snapshot.command.filtered_wrench[5]),
                    "contact_count": int(data.ncon),
                    "offline_contact_tags": ";".join(sorted(fast_contact_tags(model, data))),
                    "progress_candidate": False,
                    "inner_wall_gate": False,
                    "centered_inserted_gate": False,
                    "candidate_attempt": candidate_attempts,
                    "hard_stop": bool(hard_stop),
                    "hard_stop_reason": hard_stop_reason or "",
                    "state_invalid": bool(snapshot.state_invalid),
                    "funnel_guard_active": bool(funnel_guard_active),
                    "funnel_health": float(funnel_observation.health),
                    "funnel_injected_power_W": float(
                        funnel_observation.injected_power_w
                    ),
                    "funnel_injected_energy_J": float(
                        funnel_observation.injected_energy_j
                    ),
                    "funnel_window_progress_m": float(
                        funnel_observation.window_progress_m
                    ),
                    "funnel_window_energy_J": float(
                        funnel_observation.window_energy_j
                    ),
                    "funnel_recovery_requested": bool(
                        funnel_recovery_requested
                    ),
                    "funnel_recovery_reason": funnel_recovery_reason or "",
                    "transition_event": "",
                    "transition_position_reference_jump_mm": "",
                    "transition_orientation_reference_jump_deg": "",
                    "post_release_requalification_cue": False,
                    "conical_probe_transition_event": "",
                    "conical_probe_incomplete": False,
                    "conical_support_force_n": "",
                    "conical_entry_drop_mm": "",
                }
            )
            if pending_transition_trace is not None:
                trace_rows[-1].update(pending_transition_trace)
                pending_transition_trace = None
        maybe_render(snapshot, show_cone=show_cone)
        return snapshot

    def mark_trace(**values: object) -> None:
        if trace_rows:
            trace_rows[-1].update(values)

    def wrapped_rpy_delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Shortest component-wise Euler displacement for the small-angle regime."""

        return (
            np.asarray(target, dtype=float)
            - np.asarray(source, dtype=float)
            + math.pi
        ) % (2.0 * math.pi) - math.pi

    def smooth_rpy_transition(
        start_rpy: np.ndarray,
        target_rpy: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        return np.asarray(start_rpy, dtype=float) + smoothstep(fraction) * wrapped_rpy_delta(
            target_rpy,
            start_rpy,
        )

    def atomic_measured_pose_rebase(
        candidate: DynamicSnapshot,
        held_rpy_command: np.ndarray,
        transition_event: str,
    ) -> dict:
        """Rebase nominal pose and held differentiator state without physics time.

        The preferred nominal orientation remains the exact controller command
        used by the caller.  A bounded rotational admittance offset holds the
        measured orientation.  If that offset would be clipped, measured RPY
        becomes the explicit nominal instead.  In both branches neither the
        actual pose nor the inner-loop command is silently teleported when
        ownership moves between controllers.
        """

        nonlocal nominal_tip, pending_transition_trace
        nonlocal last_tip_command, last_rpy_command
        exact_held_rpy = np.asarray(held_rpy_command, dtype=float).copy()
        if exact_held_rpy.shape != (3,) or not np.all(np.isfinite(exact_held_rpy)):
            raise ValueError("held_rpy_command must contain three finite values")
        measured_tip = np.asarray(candidate.tip_position, dtype=float).copy()
        measured_rpy = np.array(
            [candidate.roll, candidate.pitch, candidate.yaw], dtype=float
        )
        nominal_tip_before = last_tip_command.copy()
        before_tip, before_rpy = rollout.held_command_pose()
        rebase_offset = np.zeros(6, dtype=float)
        requested_rotational_offset = wrapped_rpy_delta(
            measured_rpy,
            exact_held_rpy,
        )
        rotational_offset_limit = np.asarray(
            rollout.controller.config.max_offset,
            dtype=float,
        )[3:]
        (
            nominal_rpy,
            bounded_rotational_offset,
            measured_rpy_nominal_fallback,
        ) = _bounded_measured_pose_rebase_orientation(
            measured_rpy,
            exact_held_rpy,
            rotational_offset_limit,
        )
        # CartesianAdmittanceController.reset clips pose offsets.  Never let a
        # clipped rotational offset silently invalidate the measured-pose
        # continuity claim: when the shortest held-to-measured displacement is
        # outside the configured bound, use measured RPY as the nominal and a
        # zero rotational offset.  This changes no threshold and advances no
        # physics or sensor state.
        rebase_offset[3:] = bounded_rotational_offset
        nominal_tip = measured_tip.copy()
        rollout.rebase_admittance_command(
            nominal_tip,
            nominal_rpy,
            rebase_offset,
        )
        # The rebase itself changes the nominal command held by the inner loop
        # without a physics step. Synchronize local transition state so every
        # subsequent smooth interpolation starts from the actual post-rebase
        # nominal, including the measured-RPY fallback branch.
        last_tip_command = nominal_tip.copy()
        last_rpy_command = nominal_rpy.copy()
        after_tip, after_rpy = rollout.held_command_pose()
        position_jump_mm = 1000.0 * float(np.linalg.norm(after_tip - before_tip))
        orientation_jump_deg = math.degrees(
            float(np.linalg.norm(wrapped_rpy_delta(after_rpy, before_rpy)))
        )
        post_rebase_rpy_error_deg = math.degrees(
            float(np.linalg.norm(wrapped_rpy_delta(after_rpy, measured_rpy)))
        )
        if post_rebase_rpy_error_deg > 1.0e-6:
            raise RuntimeError(
                "atomic measured-pose rebase left an unreported orientation error"
            )
        record = {
            "event": str(transition_event),
            "active_time_s": metrics.active_time_s,
            "nominal_tip_before_mm": (1000.0 * nominal_tip_before).tolist(),
            "exact_held_rpy_command_deg": np.degrees(exact_held_rpy).tolist(),
            "nominal_rpy_passed_to_rebase_deg": np.degrees(nominal_rpy).tolist(),
            "nominal_rpy_after_rebase_rad": nominal_rpy.tolist(),
            "measured_tip_mm": (1000.0 * measured_tip).tolist(),
            "measured_rpy_deg": np.degrees(measured_rpy).tolist(),
            "requested_rotational_offset_deg": np.degrees(
                requested_rotational_offset
            ).tolist(),
            "rotational_offset_limit_deg": np.degrees(
                rotational_offset_limit
            ).tolist(),
            "measured_rpy_nominal_fallback": measured_rpy_nominal_fallback,
            "exact_held_rpy_used_as_nominal": not measured_rpy_nominal_fallback,
            "held_reference_before_tip_mm": (1000.0 * before_tip).tolist(),
            "held_reference_after_tip_mm": (1000.0 * after_tip).tolist(),
            "position_reference_jump_mm": position_jump_mm,
            "orientation_reference_jump_deg": orientation_jump_deg,
            "post_rebase_measured_tip_error_mm": 1000.0
            * float(np.linalg.norm(after_tip - measured_tip)),
            "post_rebase_measured_rpy_error_deg": post_rebase_rpy_error_deg,
            "physics_steps_advanced": 0,
            "raw_hard_thresholds_changed": False,
            "soft_thresholds_changed": False,
        }
        transition_telemetry_records.append(record)
        trace_event = {
            "transition_event": str(transition_event),
            "transition_position_reference_jump_mm": position_jump_mm,
            "transition_orientation_reference_jump_deg": orientation_jump_deg,
        }
        if pending_transition_trace is None:
            pending_transition_trace = trace_event
        else:
            pending_transition_trace = {
                "transition_event": (
                    f"{pending_transition_trace['transition_event']};"
                    f"{transition_event}"
                ),
                "transition_position_reference_jump_mm": max(
                    float(
                        pending_transition_trace[
                            "transition_position_reference_jump_mm"
                        ]
                    ),
                    position_jump_mm,
                ),
                "transition_orientation_reference_jump_deg": max(
                    float(
                        pending_transition_trace[
                            "transition_orientation_reference_jump_deg"
                        ]
                    ),
                    orientation_jump_deg,
                ),
            }
        return record

    def run_segment(
        duration_s: float,
        command: Callable[[float], tuple[np.ndarray, np.ndarray]],
        stage: str,
        stage_group: str,
        active: bool,
        show_cone: bool = False,
        observer: Callable[[float, DynamicSnapshot], bool | None] | None = None,
    ) -> DynamicSnapshot:
        nonlocal current_stage
        current_stage = stage
        steps = max(1, int(round(duration_s / SIM_TIMESTEP_S)))
        snapshot: DynamicSnapshot | None = None
        for index in range(steps):
            fraction = (index + 1) / steps
            tip, rpy = command(fraction)
            snapshot = step_once(tip, rpy, stage_group, active, show_cone)
            observer_requested_stop = False
            if observer is not None:
                observer_requested_stop = bool(observer(fraction, snapshot))
            if (
                observer_requested_stop
                or hard_stop
                or (active and metrics.active_time_s >= time_budget_s)
            ):
                break
        assert snapshot is not None
        return snapshot

    def execute_conical_probe(
        *,
        label_prefix: str,
        attempt_number: int,
    ) -> tuple[DynamicSnapshot, dict]:
        """Execute one interruptible cone and return an observable audit record."""

        nonlocal snapshot, current_phase, current_stage
        monitor = ConicalContactTransitionMonitor(
            start_time_s=metrics.active_time_s,
            start_tip_z_m=float(snapshot.tip_position[2]),
            target_down_force_n=target_down_force_n,
        )
        ramp_steps_expected = max(
            1, int(round(PROBE_RAMP_TIME_S / SIM_TIMESTEP_S))
        )
        scan_steps_expected = max(
            1, int(round(PROBE_SCAN_TIME_S / SIM_TIMESTEP_S))
        )
        ramp_steps_observed = 0
        scan_steps_observed = 0

        def observe_transition(
            value: DynamicSnapshot,
            *,
            evidence_relevant: bool,
        ) -> ConicalContactTransitionDecision:
            decision = monitor.observe(
                time_s=metrics.active_time_s,
                tip_z_m=float(value.tip_position[2]),
                measured_wrench=value.measured_wrench,
                evidence_relevant=evidence_relevant,
            )
            mark_trace(
                conical_probe_transition_event=decision.event or "",
                conical_probe_incomplete=decision.fingerprint_incomplete,
                conical_support_force_n=decision.support_force_n,
                conical_entry_drop_mm=1000.0 * decision.entry_drop_m,
            )
            return decision

        def observe_ramp(
            _fraction: float,
            value: DynamicSnapshot,
        ) -> bool:
            nonlocal ramp_steps_observed
            ramp_steps_observed += 1
            return observe_transition(
                value,
                evidence_relevant=True,
            ).stop_cone

        snapshot = run_segment(
            PROBE_RAMP_TIME_S,
            lambda f: (
                nominal_tip.copy(),
                np.array([0.0, ALPHA * smoothstep(f), 0.0]),
            ),
            f"{label_prefix}15 deg conical ramp",
            "probe",
            True,
            True,
            observe_ramp,
        )
        ramp_complete = bool(
            ramp_steps_observed == ramp_steps_expected
            and monitor.event is None
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        )

        if ramp_complete:

            def observe_scan(
                fraction: float,
                value: DynamicSnapshot,
            ) -> bool:
                nonlocal current_phase, scan_steps_observed
                scan_steps_observed += 1
                current_phase = 2.0 * math.pi * fraction
                scan_phase.append(current_phase)
                scan_wrench.append(value.measured_wrench.copy())
                scan_tip_z.append(float(value.tip_position[2]))
                return observe_transition(
                    value,
                    evidence_relevant=True,
                ).stop_cone

            snapshot = run_segment(
                PROBE_SCAN_TIME_S,
                lambda f: (
                    nominal_tip.copy(),
                    np.array(
                        [
                            ALPHA * math.sin(2.0 * math.pi * f),
                            ALPHA * math.cos(2.0 * math.pi * f),
                            0.0,
                        ]
                    ),
                ),
                f"{label_prefix}full 360 deg conical fingerprint",
                "probe",
                True,
                True,
                observe_scan,
            )

        scan_complete = bool(
            ramp_complete
            and scan_steps_observed == scan_steps_expected
            and monitor.event is None
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        )
        return_start_rpy = last_rpy_command.copy()
        returned_upright = False
        if not hard_stop and metrics.active_time_s < time_budget_s:
            return_steps_expected = max(
                1, int(round(PROBE_RETURN_TIME_S / SIM_TIMESTEP_S))
            )
            return_steps_observed = 0

            def observe_return(
                _fraction: float,
                _value: DynamicSnapshot,
            ) -> None:
                nonlocal return_steps_observed
                return_steps_observed += 1

            snapshot = run_segment(
                PROBE_RETURN_TIME_S,
                lambda f: (
                    nominal_tip.copy(),
                    smooth_rpy_transition(
                        return_start_rpy,
                        np.zeros(3),
                        f,
                    ),
                ),
                (
                    f"{label_prefix}observable transition | freeze cone, return upright"
                    if monitor.event is not None
                    else f"{label_prefix}return upright"
                ),
                "probe",
                True,
                True,
                observe_return,
            )
            returned_upright = bool(
                return_steps_observed == return_steps_expected
                and not hard_stop
                and metrics.active_time_s < time_budget_s
            )
        if returned_upright:
            snapshot = run_segment(
                POST_PROBE_SETTLE_TIME_S,
                lambda _: (nominal_tip.copy(), np.zeros(3)),
                (
                    f"{label_prefix}upright physical capture requalification"
                    if monitor.event is not None
                    else f"{label_prefix}upright verify"
                ),
                "probe",
                True,
            )

        record = monitor.summary()
        record.update(
            {
                "attempt": attempt_number,
                "label_prefix": label_prefix,
                "ramp_steps_observed": ramp_steps_observed,
                "ramp_steps_expected": ramp_steps_expected,
                "scan_steps_observed": scan_steps_observed,
                "scan_steps_expected": scan_steps_expected,
                "full_scan_completed": scan_complete,
                "returned_upright": returned_upright,
                "direction_inference_allowed": bool(
                    scan_complete
                    and returned_upright
                    and not monitor.fingerprint_incomplete
                ),
                "direction_inference_performed": False,
                "capture_requalification_attempted": False,
                "capture_requalification_confirmed": False,
                "fallback_policy": None,
            }
        )
        conical_probe_attempt_records.append(record)
        return snapshot, record

    snapshot = run_segment(
        SETTLE_TIME_S,
        lambda _: (nominal_tip.copy(), np.zeros(3)),
        "common guarded contact settle",
        "settle",
        False,
    )
    metrics.reset_reference(snapshot)

    if method in {"proposed", "learned", "hybrid", "direct"} and not hard_stop:
        snapshot, initial_conical_record = execute_conical_probe(
            label_prefix="",
            attempt_number=1,
        )

    if (
        method in {"learned", "hybrid", "direct"}
        and not hard_stop
        and initial_conical_record is not None
        and initial_conical_record["direction_inference_allowed"]
    ):
        assert direction_runtime is not None
        fingerprint_bytes = b"".join(
            (
                np.asarray(scan_phase, dtype=np.float64).tobytes(),
                np.asarray(scan_wrench, dtype=np.float64).tobytes(),
                np.asarray(scan_tip_z, dtype=np.float64).tobytes(),
            )
        )
        fingerprint_sha256 = hashlib.sha256(fingerprint_bytes).hexdigest()
        fingerprint_sha256_attempts.append(fingerprint_sha256)
        initial_conical_record["fingerprint_sha256"] = fingerprint_sha256
        bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
        estimate = direction_runtime.infer_scan(
            np.asarray(scan_phase),
            np.asarray(scan_wrench),
            np.asarray(scan_tip_z),
            target_down_force_n=target_down_force_n,
            bit_radius_m=bit_radius_m,
        )
        conical_direction_inference_count += 1
        initial_conical_record["direction_inference_performed"] = True
        if method in {"learned", "hybrid"}:
            posterior = center_posterior_from_direction(
                initial_xy,
                estimate.direction_probabilities,
                visual_sigma_m,
                np.random.default_rng(posterior_seed),
                visual_bound_m=visual_bound_m,
                state_probabilities=(
                    estimate.state_probabilities if use_state_cad_fusion else None
                ),
                cad_geometry=(cad_geometry if use_state_cad_fusion else None),
            )
            posterior_radius_m = np.linalg.norm(posterior - initial_xy, axis=1)
            posterior_radial_quantiles_mm = {
                "q10": 1000.0 * float(np.quantile(posterior_radius_m, 0.10)),
                "q50": 1000.0 * float(np.quantile(posterior_radius_m, 0.50)),
                "q95": 1000.0 * float(np.quantile(posterior_radius_m, 0.95)),
            }
            plan = plan_posterior_guided_search(
                initial_xy,
                posterior,
                estimate.direction_xy,
                effective_capture_radius_m=effective_capture_radius_m,
                posterior_coverage=posterior_coverage,
                direction_quality=float(
                    estimate.circular_concentration
                    * max(
                        0.0,
                        math.cos(
                            0.5
                            * math.radians(
                                min(180.0, estimate.ensemble_top1_span_deg)
                            )
                        ),
                    )
                ),
                max_transport_m=max_transport_m,
                pitch_safety_factor=0.80,
            )
            shortcut_telemetry.record(plan, first=True)
            planned_transport_distance_m = plan.transport_distance_m
            planned_target_xy = plan.transport_target_xy.copy()
            planned_target_is_latest_plan = True
            planned_spiral_radius_m = plan.spiral_max_radius_m
            planned_spiral_pitch_m = plan.spiral_pitch_m_per_rev
            transport_policy = plan.transport_mode
            transport_direction_quality = plan.direction_quality
            transport_quantile = plan.transport_quantile
            transport_reverse_probability = plan.reverse_probability
            transport_projected_quantiles_mm = {
                "q10": 1000.0 * plan.projected_q10_m,
                "q50": 1000.0 * plan.projected_q50_m,
                "q90": 1000.0 * plan.projected_q90_m,
            }
            transport_raw_distance_mm = 1000.0 * plan.transport_distance_raw_m
            transport_distance_cap_mm = 1000.0 * plan.transport_distance_cap_m
            if method == "hybrid":
                assert entropy_belief is not None
                learned_prior_weight = float(
                    np.clip(
                        plan.direction_quality,
                        hybrid_config.minimum_learned_prior_weight,
                        hybrid_config.maximum_learned_prior_weight,
                    )
                )
                entropy_belief.set_prior_from_samples(
                    posterior,
                    bandwidth_m=hybrid_config.posterior_kernel_bandwidth_m,
                    learned_mixture_weight=learned_prior_weight,
                )
                entropy_initial_nats = entropy_belief.entropy_nats
        else:
            direct_limit_m = (
                float(visual_bound_m)
                if visual_bound_m is not None
                else float(baseline_max_radius_m or BASELINE_MAX_RADIUS_M)
            )
            planned_transport_distance_m = direct_limit_m
            planned_target_xy = initial_xy + direct_limit_m * estimate.direction_xy
            planned_spiral_radius_m = 0.0
    elif (
        method == "proposed"
        and initial_conical_record is not None
        and initial_conical_record["direction_inference_allowed"]
    ):
        planned_target_xy = PROPOSED_TARGET_XY_M.copy()
        planned_transport_distance_m = float(
            np.linalg.norm(planned_target_xy - initial_xy)
        )
        planned_spiral_radius_m = PROPOSED_LOCAL_RADIUS_M
    elif method == "oracle":
        planned_target_xy = np.zeros(2, dtype=float)
        planned_transport_distance_m = float(np.linalg.norm(initial_xy))
        planned_spiral_radius_m = 0.0

    baseline_z_m = float(snapshot.tip_position[2])
    progress_depth_samples.clear()
    progress_depth_samples.append((metrics.active_time_s, baseline_z_m))

    capture_direction_gate = CaptureDirectionGateLifecycle(
        initially_enabled=bool(
            method in {"learned", "hybrid", "direct"} and estimate is not None
        )
    )
    capture_direction_anchor_xy = initial_xy.copy()
    capture_reverse_tolerance_m = float(spec["derived"]["r_cap_top_m"])

    def retire_capture_direction_gate_for_global_fallback(reason: str) -> None:
        """Permanently end the learned-ray gate before global search begins.

        The direction-consistency test is meaningful only while executing the
        initial learned posterior.  A declared global outward fallback has
        abandoned that hypothesis, so retaining the test can reject the very
        opposite-side capture that the global sweep is meant to recover.  This
        transition reads no geometry truth and changes no sensor or safety
        threshold.
        """

        capture_direction_gate.retire_for_global_fallback(
            active_time_s=metrics.active_time_s,
            reason=reason,
        )

    def capture_direction_consistent(candidate: DynamicSnapshot) -> bool:
        """Reject outer-edge drops that move opposite the sensed hole direction.

        This uses only the initial robot pose, the candidate robot pose, the
        frozen conical top-1 direction, and the CAD capture tolerance. It does
        not use the true socket center or MuJoCo contact labels.
        """

        if not capture_direction_gate.active:
            return True
        assert estimate is not None
        return direction_consistent_capture(
            capture_direction_anchor_xy,
            candidate.tip_position[:2],
            estimate.top_direction_deg,
            capture_reverse_tolerance_m,
        )

    def capture_yaw_command() -> np.ndarray:
        nonlocal yaw_assist_aborted, yaw_assist_elapsed_s, yaw_assist_used
        if method in {"baseline", "entropy_grid", "random"} or not yaw_assist.enabled:
            return np.zeros(3)
        if yaw_assist.exceeds_soft_tau_z(float(snapshot.measured_wrench[5])):
            yaw_assist_aborted = True
        yaw_assist_elapsed_s += SIM_TIMESTEP_S
        yaw_rad = yaw_assist.command_rad(
            yaw_assist_elapsed_s,
            aborted=yaw_assist_aborted,
        )
        yaw_assist_used = yaw_assist_used or abs(yaw_rad) > 1.0e-12
        return np.array([0.0, 0.0, yaw_rad])

    def progress_condition(candidate: DynamicSnapshot) -> bool:
        """Sensor-only early handoff cue; not a terminal capture verifier."""
        assert baseline_z_m is not None
        depth_drop = baseline_z_m - float(candidate.tip_position[2])
        window_s = CONTROLLER_THRESHOLDS["capture_trend_window_ms"] / 1000.0
        recent = [
            item
            for item in progress_depth_samples
            if item[0] >= metrics.active_time_s - window_s
        ]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 0.8 * window_s:
            return False
        recent_drop = recent[0][1] - recent[-1][1]
        return bool(
            depth_drop >= CONTROLLER_THRESHOLDS["chamfer_depth_drop_mm"] / 1000.0
            and recent_drop
            >= CONTROLLER_THRESHOLDS["capture_trend_drop_mm"] / 1000.0
            and float(np.linalg.norm(candidate.measured_wrench[:3]))
            <= CONTROLLER_THRESHOLDS["soft_force_N"]
            and 1000.0 * float(np.linalg.norm(candidate.measured_wrench[3:5]))
            <= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
            and capture_direction_consistent(candidate)
        )

    def search_escape_condition(candidate: DynamicSnapshot) -> bool:
        force_norm = float(np.linalg.norm(candidate.measured_wrench[:3]))
        torque_xy_mnm = 1000.0 * float(
            np.linalg.norm(candidate.measured_wrench[3:5])
        )
        return bool(
            force_norm >= CONTROLLER_THRESHOLDS["search_escape_force_N"]
            or torque_xy_mnm
            >= CONTROLLER_THRESHOLDS["search_escape_tau_xy_mNm"]
        )

    def inner_wall_condition(candidate: DynamicSnapshot) -> bool:
        """Online sensor/depth gate; geometry truth and contact tags are forbidden."""
        lateral_force = float(np.linalg.norm(candidate.measured_wrench[:2]))
        force_norm = float(np.linalg.norm(candidate.measured_wrench[:3]))
        torque_xy_mnm = 1000.0 * float(np.linalg.norm(candidate.measured_wrench[3:5]))
        return bool(
            1000.0 * float(candidate.tip_position[2])
            <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            and (
                lateral_force >= CONTROLLER_THRESHOLDS["inner_wall_ft_N"]
                or torque_xy_mnm >= CONTROLLER_THRESHOLDS["inner_wall_tau_xy_mNm"]
            )
            and force_norm < CONTROLLER_THRESHOLDS["hard_force_N"]
            and torque_xy_mnm < CONTROLLER_THRESHOLDS["hard_tau_xy_mNm"]
            and capture_direction_consistent(candidate)
        )

    def centered_inserted_condition(candidate: DynamicSnapshot) -> bool:
        """Online low-wrench below-rim gate without privileged geometry."""
        force_norm = float(np.linalg.norm(candidate.measured_wrench[:3]))
        torque_xy_mnm = 1000.0 * float(np.linalg.norm(candidate.measured_wrench[3:5]))
        return bool(
            1000.0 * float(candidate.tip_position[2])
            <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            and force_norm <= CONTROLLER_THRESHOLDS["soft_force_N"]
            and torque_xy_mnm <= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
            and capture_direction_consistent(candidate)
        )

    def post_release_requalification_condition(
        candidate: DynamicSnapshot,
    ) -> bool:
        """Sensor-only sustained-capture cue under the released force command."""

        force_norm = float(np.linalg.norm(candidate.measured_wrench[:3]))
        torque_xy_mnm = 1000.0 * float(
            np.linalg.norm(candidate.measured_wrench[3:5])
        )
        return bool(
            1000.0 * float(candidate.tip_position[2])
            <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            and force_norm <= CONTROLLER_THRESHOLDS["soft_force_N"]
            and torque_xy_mnm <= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
            and capture_direction_consistent(candidate)
        )

    def sensor_depth_commit_condition(candidate: DynamicSnapshot) -> bool:
        """Latch below-rim progress without using MuJoCo contact labels."""
        force_norm = float(np.linalg.norm(candidate.measured_wrench[:3]))
        torque_xy_mnm = 1000.0 * float(np.linalg.norm(candidate.measured_wrench[3:5]))
        return bool(
            1000.0 * float(candidate.tip_position[2])
            <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            and force_norm < CONTROLLER_THRESHOLDS["hard_force_N"]
            and torque_xy_mnm < CONTROLLER_THRESHOLDS["hard_tau_xy_mNm"]
            and capture_direction_consistent(candidate)
        )

    def confirm_progress(
        hold_tip: np.ndarray,
        *,
        require_progress_cue: bool = True,
    ) -> bool:
        nonlocal current_stage, snapshot, candidate_attempts, capture_mode, depth_commit_latched
        nonlocal capture_handoff_used, sensor_progress_candidate_count
        candidate_attempts += 1
        if require_progress_cue:
            capture_handoff_used = True
            sensor_progress_candidate_count += 1
            current_stage = "sensor progress cue | freeze XY + low-force compliant handoff"
            handoff_start_z = float(snapshot.tip_position[2])
            handoff_start_force_n = float(rollout.target_wrench[2])
            handoff_force_n = min(
                handoff_start_force_n,
                target_down_force_n,
                CONTROLLER_THRESHOLDS["capture_handoff_force_N"],
            )
            rollout.set_planar_impedance(
                CONTROLLER_THRESHOLDS["capture_planar_stiffness_N_m"],
                CONTROLLER_THRESHOLDS["capture_planar_damping_N_s_m"],
            )
            ramp_steps = max(
                1,
                int(
                    round(
                        CONTROLLER_THRESHOLDS["capture_handoff_ramp_ms"]
                        / 1000.0
                        / SIM_TIMESTEP_S
                    )
                ),
            )
            for index in range(ramp_steps):
                fraction = smoothstep((index + 1) / ramp_steps)
                rollout.set_target_down_force(
                    handoff_start_force_n
                    + fraction * (handoff_force_n - handoff_start_force_n)
                )
                snapshot = step_once(
                    hold_tip,
                    capture_yaw_command(),
                    "capture_handoff",
                    True,
                )
                mark_trace(
                    progress_candidate=progress_condition(snapshot),
                    inner_wall_gate=inner_wall_condition(snapshot),
                    centered_inserted_gate=centered_inserted_condition(snapshot),
                    candidate_attempt=candidate_attempts,
                )
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    break
            if hard_stop:
                return False

            required = max(
                1,
                int(round(CONTROLLER_THRESHOLDS["capture_dwell_ms"] / 1000.0 / SIM_TIMESTEP_S)),
            )
            count = 0
            depth_commit_count = 0
            verify_steps = max(required * 4, int(round(0.25 / SIM_TIMESTEP_S)))
            for _ in range(verify_steps):
                snapshot = step_once(
                    hold_tip,
                    capture_yaw_command(),
                    "progress_dwell",
                    True,
                )
                force_norm = float(np.linalg.norm(snapshot.measured_wrench[:3]))
                torque_xy_mnm = 1000.0 * float(
                    np.linalg.norm(snapshot.measured_wrench[3:5])
                )
                handoff_drop = handoff_start_z - float(snapshot.tip_position[2])
                cue = bool(
                    handoff_drop
                    >= CONTROLLER_THRESHOLDS["capture_handoff_drop_mm"] / 1000.0
                    and force_norm <= CONTROLLER_THRESHOLDS["soft_force_N"]
                    and torque_xy_mnm
                    <= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
                )
                mark_trace(
                    progress_candidate=cue,
                    inner_wall_gate=inner_wall_condition(snapshot),
                    centered_inserted_gate=centered_inserted_condition(snapshot),
                    candidate_attempt=candidate_attempts,
                )
                count, progress_confirmed = _advance_dwell_count(
                    count, cue, required
                )
                depth_commit_count, depth_confirmed = _advance_dwell_count(
                    depth_commit_count,
                    sensor_depth_commit_condition(snapshot),
                    required,
                )
                if progress_confirmed or depth_confirmed:
                    break
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    break
            if count >= required or depth_commit_count >= required:
                depth_commit_latched = True
                capture_mode = "safe_downward_progress"
                current_stage = "persistent safe progress | compliant capture latched"
                return True

            current_stage = "candidate rejected | restore search impedance and force"
            for index in range(ramp_steps):
                fraction = smoothstep((index + 1) / ramp_steps)
                rollout.set_target_down_force(
                    handoff_force_n
                    + fraction * (target_down_force_n - handoff_force_n)
                )
                snapshot = step_once(
                    hold_tip,
                    np.zeros(3),
                    "candidate_recovery",
                    True,
                )
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    break
            rollout.restore_planar_impedance()
            if not hard_stop:
                rebase_offset = np.zeros(6, dtype=float)
                rebase_offset[:3] = snapshot.tip_position - hold_tip
                rollout.set_target_down_force(target_down_force_n)
                rollout.reset_admittance(rebase_offset)
            return False

        direct_inner = inner_wall_condition(snapshot)
        direct_centered = centered_inserted_condition(snapshot)
        if direct_inner or direct_centered:
            required = max(
                1,
                int(
                    round(
                        CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                        / 1000.0
                        / SIM_TIMESTEP_S
                    )
                ),
            )
            direct_count = 0
            direct_mode = "inner_wall_guided" if direct_inner else "centered_free_insertion"
            verify_steps = max(required * 4, int(round(0.25 / SIM_TIMESTEP_S)))
            for _ in range(verify_steps):
                direct_inner = inner_wall_condition(snapshot)
                direct_centered = centered_inserted_condition(snapshot)
                cue = direct_inner or direct_centered
                if cue:
                    direct_mode = (
                        "inner_wall_guided"
                        if direct_inner
                        else "centered_free_insertion"
                    )
                direct_count, confirmed = _advance_dwell_count(
                    direct_count, cue, required
                )
                mark_trace(
                    inner_wall_gate=direct_inner,
                    centered_inserted_gate=direct_centered,
                    candidate_attempt=candidate_attempts,
                )
                if confirmed:
                    capture_mode = direct_mode
                    current_stage = "persistent direct capture cue | capture latched"
                    return True
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    return False
                snapshot = step_once(
                    hold_tip,
                    capture_yaw_command(),
                    "direct_capture_dwell",
                    True,
                )

        current_stage = (
            "upright guarded descent + micro-yaw clocking | confirm inner-wall gate"
            if yaw_assist.enabled
            and method not in {"baseline", "entropy_grid", "random"}
            else "upright guarded descent | confirm inner-wall gate"
        )
        confirmation_force_n = min(0.8, target_down_force_n)
        # Preserve force-command continuity when a preceding guarded recovery
        # has intentionally unloaded contact.  In the historical path the
        # current command equals ``target_down_force_n``, so this is behavior
        # preserving unless a caller explicitly performed a bumpless unload.
        confirmation_force_start_n = (
            float(rollout.target_wrench[2])
            if oracle_transport_recovery_enabled
            else target_down_force_n
        )
        release_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
        for index in range(release_steps):
            release_fraction = smoothstep((index + 1) / release_steps)
            rollout.set_target_down_force(
                confirmation_force_start_n
                + release_fraction
                * (confirmation_force_n - confirmation_force_start_n)
            )
            snapshot = step_once(
                hold_tip,
                capture_yaw_command(),
                "candidate_force_release",
                True,
            )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                break
        if hard_stop:
            return False
        confirmation_tip = np.array(
            [hold_tip[0], hold_tip[1], float(snapshot.tip_position[2])], dtype=float
        )
        rollout.set_target_down_force(confirmation_force_n)
        rollout.reset_admittance()
        confirmation_steps = max(1, int(round(3.0 / SIM_TIMESTEP_S)))
        required = max(
            1,
            int(
                round(
                    CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                    / 1000.0
                    / SIM_TIMESTEP_S
                )
            ),
        )
        confirmation_count = 0
        confirmation_mode = "inner_wall_guided"
        for index in range(confirmation_steps):
            if metrics.active_time_s >= time_budget_s:
                break
            snapshot = step_once(
                confirmation_tip,
                capture_yaw_command(),
                "descent_confirmation",
                True,
            )
            mark_trace(
                progress_candidate=progress_condition(snapshot),
                inner_wall_gate=inner_wall_condition(snapshot),
                centered_inserted_gate=centered_inserted_condition(snapshot),
                candidate_attempt=candidate_attempts,
            )
            direct_inner = inner_wall_condition(snapshot)
            direct_centered = centered_inserted_condition(snapshot)
            if direct_inner or direct_centered:
                confirmation_mode = (
                    "inner_wall_guided"
                    if direct_inner
                    else "centered_free_insertion"
                )
            confirmation_count, confirmed = _advance_dwell_count(
                confirmation_count,
                direct_inner or direct_centered,
                required,
            )
            if confirmed:
                capture_mode = confirmation_mode
                return True
            if hard_stop or metrics.active_time_s >= time_budget_s:
                break

        if not hard_stop:
            current_stage = "candidate rejected | restore search down-force"
            restore_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
            for index in range(restore_steps):
                if metrics.active_time_s >= time_budget_s:
                    break
                restore_fraction = smoothstep((index + 1) / restore_steps)
                rollout.set_target_down_force(
                    confirmation_force_n
                    + restore_fraction * (target_down_force_n - confirmation_force_n)
                )
                snapshot = step_once(
                    confirmation_tip,
                    np.zeros(3),
                    "candidate_recovery",
                    True,
                )
                if hard_stop:
                    break
            if not hard_stop:
                # Preserve the current corrected tip while returning ownership
                # to the original search-plane nominal command.
                rebase_offset = np.zeros(6, dtype=float)
                rebase_offset[:3] = snapshot.tip_position - hold_tip
                rollout.set_target_down_force(target_down_force_n)
                rollout.reset_admittance(rebase_offset)
        return False

    def physical_reset_to_anchor(
        target_xy: np.ndarray,
        label: str,
        *,
        count_as_recovery: bool = True,
        touchdown_force_n: float | None = None,
        guarded_probe: bool = False,
    ) -> bool:
        nonlocal snapshot, nominal_tip, baseline_z_m, current_stage
        nonlocal search_recovery_count, fallback_physical_reset_used
        if count_as_recovery:
            search_recovery_count += 1
            fallback_physical_reset_used = True
        if guarded_probe:
            rollout.set_planar_impedance(300.0, 55.0)
            rollout.set_rotational_impedance(
                (0.50, 0.50, 1.00),
                (0.15, 0.15, 0.20),
            )
        else:
            rollout.set_planar_impedance(
                CONTROLLER_THRESHOLDS["capture_planar_stiffness_N_m"],
                CONTROLLER_THRESHOLDS["capture_planar_damping_N_s_m"],
            )
        low_force_n = 0.10 if guarded_probe else 0.20
        contact_force_n = (
            target_down_force_n
            if touchdown_force_n is None
            else float(touchdown_force_n)
        )
        if contact_force_n <= 0.0:
            raise ValueError("touchdown_force_n must be positive")
        current_stage = f"{label} | unload contact"
        # Preserve the current nominal command while the existing admittance
        # offset unloads. Replacing nominal with the measured tip here would
        # apply the offset twice and create an artificial contact impulse.
        current = snapshot.nominal_tip_position.copy()
        unload_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
        unload_start_force_n = float(rollout.target_wrench[2])
        for index in range(unload_steps):
            fraction = smoothstep((index + 1) / unload_steps)
            rollout.set_target_down_force(
                unload_start_force_n
                + fraction * (low_force_n - unload_start_force_n)
            )
            snapshot = step_once(current, np.zeros(3), "fallback_reset", True)
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False

        hover = current.copy()
        hover[2] = max(0.0020, float(snapshot.tip_position[2]) + 0.0015)
        lift_steps = max(1, int(round(0.55 / SIM_TIMESTEP_S)))
        lift_start = current.copy()
        current_stage = f"{label} | vertical retract"
        for index in range(lift_steps):
            fraction = smoothstep((index + 1) / lift_steps)
            nominal_tip = lift_start + fraction * (hover - lift_start)
            snapshot = step_once(nominal_tip, np.zeros(3), "fallback_reset", True)
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False

        shift_start = nominal_tip.copy()
        shift_target = np.array([target_xy[0], target_xy[1], hover[2]], dtype=float)
        shift_duration_s = max(
            0.30,
            float(np.linalg.norm(shift_target[:2] - shift_start[:2]))
            / xy_speed_m_s,
        )
        shift_steps = max(1, int(round(shift_duration_s / SIM_TIMESTEP_S)))
        current_stage = f"{label} | shift above selected target"
        for index in range(shift_steps):
            fraction = smoothstep((index + 1) / shift_steps)
            nominal_tip = shift_start + fraction * (shift_target - shift_start)
            snapshot = step_once(nominal_tip, np.zeros(3), "fallback_reset", True)
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False

        touchdown = np.array(
            [target_xy[0], target_xy[1], NOMINAL_TIP_Z_M], dtype=float
        )
        touchdown_start = nominal_tip.copy()
        touchdown_duration_s = 3.00 if guarded_probe else 0.65
        touchdown_steps = max(
            1, int(round(touchdown_duration_s / SIM_TIMESTEP_S))
        )
        current_stage = f"{label} | guarded retouchdown"
        probe_contact_latched = False
        for index in range(touchdown_steps):
            fraction = smoothstep((index + 1) / touchdown_steps)
            nominal_tip = touchdown_start + fraction * (touchdown - touchdown_start)
            rollout.set_target_down_force(
                low_force_n
                if guarded_probe
                else low_force_n + fraction * (contact_force_n - low_force_n)
            )
            snapshot = step_once(nominal_tip, np.zeros(3), "fallback_reset", True)
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False
            if guarded_probe:
                force_norm = float(np.linalg.norm(snapshot.measured_wrench[:3]))
                torque_xy_mnm = 1000.0 * float(
                    np.linalg.norm(snapshot.measured_wrench[3:5])
                )
                if (
                    sensor_depth_commit_condition(snapshot)
                    or force_norm >= 0.30
                    or torque_xy_mnm >= 1.0
                ):
                    probe_contact_latched = True
                    break

        if guarded_probe:
            rollout.set_planar_impedance(300.0, 55.0)
        else:
            rollout.restore_planar_impedance()
        if probe_contact_latched and sensor_depth_commit_condition(snapshot):
            rollout.set_target_down_force(min(contact_force_n, 0.8))
            current_stage = f"{label} | below-rim cue latched at low force"
            return True
        if guarded_probe:
            current_stage = f"{label} | contact frozen, ramp low probe force"
            achieved_force_n = low_force_n
            ramp_steps = max(1, int(round(0.35 / SIM_TIMESTEP_S)))
            for index in range(ramp_steps):
                fraction = smoothstep((index + 1) / ramp_steps)
                commanded_force_n = low_force_n + fraction * (
                    contact_force_n - low_force_n
                )
                rollout.set_target_down_force(commanded_force_n)
                snapshot = step_once(
                    nominal_tip,
                    np.zeros(3),
                    "fallback_reset",
                    True,
                )
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    return False
                if sensor_depth_commit_condition(snapshot):
                    current_stage = f"{label} | below-rim cue latched during force ramp"
                    return True
                force_norm = float(np.linalg.norm(snapshot.measured_wrench[:3]))
                torque_xy_mnm = 1000.0 * float(
                    np.linalg.norm(snapshot.measured_wrench[3:5])
                )
                achieved_force_n = commanded_force_n
                if (
                    force_norm >= CONTROLLER_THRESHOLDS["soft_force_N"]
                    or torque_xy_mnm >= CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"]
                ):
                    break
            rollout.set_target_down_force(achieved_force_n)
        else:
            rollout.set_target_down_force(contact_force_n)
            rollout.reset_admittance()
        settle_steps = max(1, int(round(0.30 / SIM_TIMESTEP_S)))
        current_stage = f"{label} | contact settle"
        for _ in range(settle_steps):
            snapshot = step_once(nominal_tip, np.zeros(3), "fallback_reset", True)
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False
        baseline_z_m = float(snapshot.tip_position[2])
        progress_depth_samples.clear()
        progress_depth_samples.append((metrics.active_time_s, baseline_z_m))
        return True

    def smooth_guarded_contact_move(
        target_xy: np.ndarray,
        label: str,
        *,
        probe_force_n: float,
        stage_group: str = "hybrid_entropy_probe",
        measured_xy_path: list[np.ndarray] | None = None,
    ) -> tuple[bool, bool]:
        """Move between adaptive probe nodes without repeated impact.

        The target normal force is constant after a short ramp, while the XY
        command follows a minimum-jerk segment. A short endpoint dwell keeps
        the F/T sample approximately quasi-static. Capture cues preempt the
        tangential command and enter the same sensor-only handoff used by all
        search policies.
        """

        nonlocal snapshot, nominal_tip, current_stage, baseline_z_m
        target = np.asarray(target_xy, dtype=float)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xy must contain two finite values")
        if probe_force_n <= 0.0:
            raise ValueError("probe_force_n must be positive")
        if measured_xy_path is not None:
            measured_xy_path.append(snapshot.tip_position[:2].copy())

        rollout.set_planar_impedance(300.0, 55.0)
        rollout.set_rotational_impedance(
            (0.50, 0.50, 1.00),
            (0.15, 0.15, 0.20),
        )
        current_stage = f"{label} | low-force minimum-jerk contact segment"
        ramp_start_force_n = float(rollout.target_wrench[2])
        ramp_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
        for index in range(ramp_steps):
            fraction = smoothstep((index + 1) / ramp_steps)
            rollout.set_target_down_force(
                ramp_start_force_n
                + fraction * (probe_force_n - ramp_start_force_n)
            )
            snapshot = step_once(
                nominal_tip.copy(),
                np.zeros(3),
                stage_group,
                True,
            )
            if measured_xy_path is not None:
                measured_xy_path.append(snapshot.tip_position[:2].copy())
            captured = inner_wall_condition(snapshot) or centered_inserted_condition(
                snapshot
            )
            if captured:
                current_stage = f"{label} | capture cue during force ramp"
                return True, confirm_progress(
                    nominal_tip.copy(),
                    require_progress_cue=False,
                )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False, False
        rollout.set_target_down_force(probe_force_n)

        start = nominal_tip.copy()
        delta_xy = target - start[:2]
        duration_s = max(
            0.25,
            float(np.linalg.norm(delta_xy)) / xy_speed_m_s,
        )
        steps = max(1, int(round(duration_s / SIM_TIMESTEP_S)))
        for index in range(steps):
            fraction = smoothstep((index + 1) / steps)
            nominal_tip = np.array(
                [
                    start[0] + fraction * delta_xy[0],
                    start[1] + fraction * delta_xy[1],
                    NOMINAL_TIP_Z_M,
                ],
                dtype=float,
            )
            snapshot = step_once(
                nominal_tip,
                np.zeros(3),
                stage_group,
                True,
            )
            if measured_xy_path is not None:
                measured_xy_path.append(snapshot.tip_position[:2].copy())
            captured = inner_wall_condition(snapshot) or centered_inserted_condition(
                snapshot
            )
            if captured:
                current_stage = f"{label} | capture cue preempts tangential motion"
                return True, confirm_progress(
                    nominal_tip.copy(),
                    require_progress_cue=False,
                )
            if search_escape_condition(snapshot):
                current_stage = f"{label} | soft wrench escape before hard stop"
                return False, False
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False, False

        current_stage = f"{label} | quasi-static endpoint dwell"
        dwell_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
        for _ in range(dwell_steps):
            snapshot = step_once(
                nominal_tip.copy(),
                np.zeros(3),
                stage_group,
                True,
            )
            if measured_xy_path is not None:
                measured_xy_path.append(snapshot.tip_position[:2].copy())
            captured = inner_wall_condition(snapshot) or centered_inserted_condition(
                snapshot
            )
            if captured:
                current_stage = f"{label} | endpoint capture cue"
                return True, confirm_progress(
                    nominal_tip.copy(),
                    require_progress_cue=False,
                )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                return False, False
        baseline_z_m = float(snapshot.tip_position[2])
        progress_depth_samples.clear()
        progress_depth_samples.append((metrics.active_time_s, baseline_z_m))
        return True, False

    def set_funnel_impedance(depth_fraction: float) -> None:
        fraction = float(np.clip(depth_fraction, 0.0, 1.0))
        tilt_k = (
            funnel_config.shallow_tilt_stiffness_nm_rad
            + fraction
            * (
                funnel_config.deep_tilt_stiffness_nm_rad
                - funnel_config.shallow_tilt_stiffness_nm_rad
            )
        )
        tilt_d = (
            funnel_config.shallow_tilt_damping_nm_s_rad
            + fraction
            * (
                funnel_config.deep_tilt_damping_nm_s_rad
                - funnel_config.shallow_tilt_damping_nm_s_rad
            )
        )
        rollout.set_planar_impedance(
            funnel_config.planar_stiffness_n_m,
            funnel_config.planar_damping_n_s_m,
        )
        rollout.set_rotational_impedance(
            (tilt_k, tilt_k, funnel_config.yaw_stiffness_nm_rad),
            (tilt_d, tilt_d, funnel_config.yaw_damping_nm_s_rad),
        )

    def perform_funnel_recovery(resume_force_n: float) -> bool:
        nonlocal snapshot, nominal_tip, current_stage
        nonlocal funnel_guard_active, funnel_recovery_requested
        nonlocal funnel_recovery_reason, funnel_recovery_count, funnel_abort_reason
        nonlocal funnel_accumulated_energy_j, residual_insertion_rpy
        nonlocal terminal_axial_position_hold_used
        if funnel_recovery_count >= funnel_config.max_recoveries:
            funnel_abort_reason = funnel_recovery_reason or "funnel_recovery_limit"
            return False
        reason = funnel_recovery_reason or "funnel_stall"
        funnel_recovery_reasons.append(reason)
        funnel_guard_active = False
        funnel_recovery_requested = False
        funnel_recovery_reason = None
        # A true guard recovery leaves the terminal-set position-hold mode and
        # restores compliant axial motion before unloading/retracting.
        rollout.set_enabled_axes(
            np.array([True, True, True, False, False, False])
        )
        terminal_axial_position_hold_used = False

        # Transfer recovery ownership at the measured pose before the first
        # unload physics step.  The historical path changed RPY to zero on that
        # first step while retaining a stale nominal/held reference.  Preserve
        # the exact last command here, then return upright continuously during
        # the unload interval.
        recovery_entry_rpy_command = last_rpy_command.copy()
        recovery_entry_transition = atomic_measured_pose_rebase(
            snapshot,
            recovery_entry_rpy_command,
            "funnel_recovery_ownership_entry",
        )
        recovery_entry_rpy_command = last_rpy_command.copy()
        current_nominal = nominal_tip.copy()
        _, held_entry_rpy = rollout.held_command_pose()
        recovery_rpy_offset = wrapped_rpy_delta(
            held_entry_rpy,
            recovery_entry_rpy_command,
        )
        upright_nominal_rpy = -recovery_rpy_offset
        residual_insertion_rpy.fill(0.0)

        current_stage = f"funnel guard: {reason} | unload"
        start_force_n = float(rollout.target_wrench[2])
        unload_steps = max(
            1,
            int(round(funnel_config.recovery_unload_s / SIM_TIMESTEP_S)),
        )
        for index in range(unload_steps):
            fraction = smoothstep((index + 1) / unload_steps)
            rollout.set_target_down_force(
                start_force_n
                + fraction * (funnel_config.recovery_force_n - start_force_n)
            )
            recovery_rpy_command = smooth_rpy_transition(
                recovery_entry_rpy_command,
                upright_nominal_rpy,
                (index + 1) / unload_steps,
            )
            snapshot = step_once(
                current_nominal,
                recovery_rpy_command,
                "funnel_recovery",
                True,
            )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                funnel_abort_reason = "hard_stop_during_funnel_recovery"
                return False

        # Crossing the CAD bottom envelope is a terminal safety event, not a
        # recoverable contact-search stall. Retouching below the known socket
        # depth can only inject more energy into an already invalid state.
        if reason == "cad_depth_overshoot":
            funnel_abort_reason = reason
            return False

        funnel_recovery_count += 1
        post_unload_transition = atomic_measured_pose_rebase(
            snapshot,
            last_rpy_command,
            "funnel_recovery_post_unload",
        )
        retract_rpy_start = last_rpy_command.copy()
        _, held_post_unload_rpy = rollout.held_command_pose()
        retract_rpy_offset = wrapped_rpy_delta(
            held_post_unload_rpy,
            retract_rpy_start,
        )
        retract_upright_nominal_rpy = -retract_rpy_offset

        current_stage = (
            f"funnel guard | retract {1000.0 * funnel_config.recovery_retract_m:.2f} mm "
            "+ return upright"
        )
        retract_start = nominal_tip.copy()
        retract_target = retract_start.copy()
        retract_target[2] += funnel_config.recovery_retract_m
        retract_steps = max(
            1,
            int(round(funnel_config.recovery_retract_s / SIM_TIMESTEP_S)),
        )
        for index in range(retract_steps):
            fraction = smoothstep((index + 1) / retract_steps)
            nominal_tip = retract_start + fraction * (retract_target - retract_start)
            retract_rpy_command = smooth_rpy_transition(
                retract_rpy_start,
                retract_upright_nominal_rpy,
                (index + 1) / retract_steps,
            )
            snapshot = step_once(
                nominal_tip,
                retract_rpy_command,
                "funnel_recovery",
                True,
            )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                funnel_abort_reason = "hard_stop_during_funnel_recovery"
                return False

        # Rebase at the measured tip so the previous admittance offset is not
        # applied twice when contact is re-established.
        nominal_tip = snapshot.tip_position.copy()
        funnel_recovery_xy_rebases_mm.append(
            (1000.0 * nominal_tip[:2]).tolist()
        )
        pre_retouch_transition = atomic_measured_pose_rebase(
            snapshot,
            last_rpy_command,
            "funnel_recovery_pre_retouch",
        )
        retouch_rpy_start = last_rpy_command.copy()
        _, held_pre_retouch_rpy = rollout.held_command_pose()
        retouch_rpy_offset = wrapped_rpy_delta(
            held_pre_retouch_rpy,
            retouch_rpy_start,
        )
        retouch_upright_nominal_rpy = -retouch_rpy_offset
        current_stage = "funnel guard | low-force retouchdown"
        retouch_steps = max(
            1,
            int(round(funnel_config.recovery_retouch_s / SIM_TIMESTEP_S)),
        )
        for index in range(retouch_steps):
            fraction = smoothstep((index + 1) / retouch_steps)
            rollout.set_target_down_force(
                funnel_config.recovery_force_n
                + fraction * (resume_force_n - funnel_config.recovery_force_n)
            )
            retouch_rpy_command = smooth_rpy_transition(
                retouch_rpy_start,
                retouch_upright_nominal_rpy,
                (index + 1) / retouch_steps,
            )
            snapshot = step_once(
                nominal_tip,
                retouch_rpy_command,
                "funnel_recovery",
                True,
            )
            if hard_stop or metrics.active_time_s >= time_budget_s:
                funnel_abort_reason = "hard_stop_during_funnel_recovery"
                return False

        rollout.set_target_down_force(resume_force_n)
        # The retouchdown response changes the measured equilibrium.  Rebase
        # once more at that measured pose before returning ownership to the
        # descent controller; the caller must then restart its interpolation
        # from fraction zero rather than continuing a stale pass origin.
        post_retouch_transition = atomic_measured_pose_rebase(
            snapshot,
            last_rpy_command,
            "funnel_recovery_resume",
        )
        funnel_recovery_transition_records.append(
            {
                "recovery": funnel_recovery_count,
                "reason": reason,
                "resume_force_n": float(resume_force_n),
                "ownership_entry_transition_event": recovery_entry_transition[
                    "event"
                ],
                "ownership_entry_exact_rpy_command_deg": np.degrees(
                    recovery_entry_rpy_command
                ).tolist(),
                "upright_return_command_deg": np.degrees(
                    upright_nominal_rpy
                ).tolist(),
                "upright_return_smoothed": True,
                "retract_upright_command_deg": np.degrees(
                    retract_upright_nominal_rpy
                ).tolist(),
                "retouch_upright_command_deg": np.degrees(
                    retouch_upright_nominal_rpy
                ).tolist(),
                "post_unload_transition_event": post_unload_transition["event"],
                "pre_retouch_transition_event": pre_retouch_transition["event"],
                "post_retouch_transition_event": post_retouch_transition["event"],
                "resume_measured_tip_mm": (
                    1000.0 * snapshot.tip_position
                ).tolist(),
                "descent_restart_required": True,
                "raw_hard_thresholds_changed": False,
                "soft_thresholds_changed": False,
            }
        )
        funnel_accumulated_energy_j += funnel_monitor.injected_energy_j
        funnel_monitor.reset()
        funnel_guard_active = funnel_config.enabled
        current_stage = "funnel guard recovered | resume compliant insertion"
        return True

    def follow_path(
        arc: np.ndarray,
        xy_path: np.ndarray,
        stage: str,
        stage_group: str,
        *,
        continuous_until_capture: bool = False,
        nominal_z_m: float | None = None,
        measured_xy_path: list[np.ndarray] | None = None,
        planar_tracking_error_path_m: list[float] | None = None,
        reference_xy_path: list[np.ndarray] | None = None,
        reference_tracking_error_path_m: list[float] | None = None,
        outer_servo_correction_path_m: list[float] | None = None,
        path_outer_servo_config: HybridResidualCoverageConfig | None = None,
        path_xy_speed_m_s: float | None = None,
    ) -> bool:
        nonlocal current_stage, snapshot, nominal_tip, candidate_attempts, capture_mode
        nonlocal soft_escape_requested
        current_stage = stage
        total_length = float(arc[-1])
        elapsed = 0.0
        candidate_armed = True
        capture_count = 0
        progress_count = 0
        outer_correction_xy = np.zeros(2, dtype=float)
        outer_tracking_error_dwell_s = 0.0
        outer_servo_enabled = bool(
            path_outer_servo_config is not None
            and path_outer_servo_config.measured_tip_outer_servo_enabled
        )
        execution_xy_speed_m_s = (
            xy_speed_m_s
            if path_xy_speed_m_s is None
            else float(path_xy_speed_m_s)
        )
        if execution_xy_speed_m_s <= 0.0:
            raise ValueError("path_xy_speed_m_s must be positive")
        capture_required = max(
            1,
            int(
                round(
                    CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                    / 1000.0
                    / SIM_TIMESTEP_S
                )
            ),
        )
        while (
            elapsed * execution_xy_speed_m_s < total_length
            and not hard_stop
            and not depth_commit_latched
            and metrics.active_time_s < time_budget_s
        ):
            elapsed += SIM_TIMESTEP_S
            distance = min(total_length, elapsed * execution_xy_speed_m_s)
            x = float(np.interp(distance, arc, xy_path[:, 0]))
            y = float(np.interp(distance, arc, xy_path[:, 1]))
            reference_xy = np.array([x, y], dtype=float)
            if outer_servo_enabled:
                assert path_outer_servo_config is not None
                measured_error_xy = reference_xy - snapshot.tip_position[:2]
                desired_correction_xy = (
                    path_outer_servo_config.outer_servo_gain
                    * measured_error_xy
                )
                desired_norm = float(np.linalg.norm(desired_correction_xy))
                correction_limit_m = (
                    path_outer_servo_config.outer_servo_max_correction_m
                )
                if desired_norm > correction_limit_m:
                    desired_correction_xy *= correction_limit_m / desired_norm
                correction_delta = desired_correction_xy - outer_correction_xy
                delta_norm = float(np.linalg.norm(correction_delta))
                delta_limit_m = (
                    path_outer_servo_config.outer_servo_max_correction_speed_m_s
                    * SIM_TIMESTEP_S
                )
                if delta_norm > delta_limit_m:
                    correction_delta *= delta_limit_m / delta_norm
                outer_correction_xy += correction_delta
            command_xy = reference_xy + outer_correction_xy
            nominal_tip = np.array(
                [
                    command_xy[0],
                    command_xy[1],
                    NOMINAL_TIP_Z_M
                    if nominal_z_m is None
                    else float(nominal_z_m),
                ],
                dtype=float,
            )
            rpy_command = (
                capture_yaw_command()
                if stage_group == "spiral"
                and not continuous_until_capture
                and method in {"proposed", "learned", "hybrid", "oracle"}
                else np.zeros(3)
            )
            snapshot = step_once(nominal_tip, rpy_command, stage_group, True)
            if measured_xy_path is not None:
                measured_xy_path.append(snapshot.tip_position[:2].copy())
            if reference_xy_path is not None:
                reference_xy_path.append(reference_xy.copy())
            if planar_tracking_error_path_m is not None:
                planar_tracking_error_path_m.append(
                    float(
                        np.linalg.norm(
                            snapshot.tip_position[:2] - nominal_tip[:2]
                        )
                    )
                )
            reference_tracking_error_m = float(
                np.linalg.norm(snapshot.tip_position[:2] - reference_xy)
            )
            if reference_tracking_error_path_m is not None:
                reference_tracking_error_path_m.append(
                    reference_tracking_error_m
                )
            if outer_servo_correction_path_m is not None:
                outer_servo_correction_path_m.append(
                    float(np.linalg.norm(outer_correction_xy))
                )
            is_candidate = progress_condition(snapshot)
            mark_trace(
                progress_candidate=is_candidate,
                inner_wall_gate=inner_wall_condition(snapshot),
                centered_inserted_gate=centered_inserted_condition(snapshot),
                candidate_attempt=candidate_attempts,
            )
            if search_escape_condition(snapshot) and not is_candidate:
                soft_escape_requested = True
                current_stage = "pre-hard-stop wrench escape requested"
                break
            if outer_servo_enabled and not is_candidate:
                assert path_outer_servo_config is not None
                outer_tracking_error_dwell_s = (
                    outer_tracking_error_dwell_s + SIM_TIMESTEP_S
                    if reference_tracking_error_m
                    > path_outer_servo_config.outer_servo_abort_tracking_error_m
                    else 0.0
                )
                if (
                    outer_tracking_error_dwell_s
                    >= path_outer_servo_config.outer_servo_abort_dwell_s
                ):
                    soft_escape_requested = True
                    current_stage = "measured-tip outer-servo tracking timeout"
                    break
            if continuous_until_capture:
                captured = inner_wall_condition(snapshot) or centered_inserted_condition(
                    snapshot
                )
                capture_count = capture_count + 1 if captured else 0
                if capture_count >= capture_required:
                    capture_mode = (
                        "inner_wall_guided"
                        if inner_wall_condition(snapshot)
                        else "centered_free_insertion"
                    )
                    return True
                if not is_candidate:
                    candidate_armed = True
                elif candidate_armed:
                    if confirm_progress(nominal_tip.copy()):
                        return True
                    if depth_commit_latched:
                        return False
                    candidate_armed = False
                    current_stage = stage
                continue
            if not is_candidate:
                candidate_armed = True
            elif candidate_armed:
                if confirm_progress(nominal_tip.copy()):
                    return True
                if depth_commit_latched:
                    return False
                candidate_armed = False
                current_stage = stage
        return False

    def handle_incomplete_conical_probe(record: dict) -> bool:
        """Requalify a stopped cone, then use direction-free residual search.

        No direction model is called on this route.  A capture-transition cue
        first enters the same upright physical confirmation used by the search
        policies.  If that cue is rejected, the only optional continuation is
        the declared global outward fallback.
        """

        nonlocal conical_capture_handoff_count
        nonlocal conical_incomplete_fallback_used
        nonlocal global_outward_fallback_used
        nonlocal global_outward_fallback_started_active_time_s
        nonlocal fallback_skipped_reason, current_stage

        if record["transition_event"] is not None:
            conical_capture_handoff_count += 1
            record["capture_requalification_attempted"] = True
            current_stage = (
                "conical contact transition | upright physical capture requalification"
            )
            captured = confirm_progress(
                nominal_tip.copy(),
                require_progress_cue=False,
            )
            record["capture_requalification_confirmed"] = bool(captured)
            if captured:
                record["fallback_policy"] = "not_needed_capture_requalified"
                return True
            if hard_stop or metrics.active_time_s >= time_budget_s:
                record["fallback_policy"] = "not_run_after_requalification_termination"
                return False

        conical_incomplete_fallback_used = True
        retire_capture_direction_gate_for_global_fallback(
            "incomplete_conical_evidence_forbids_direction_control"
        )
        if not global_outward_fallback:
            record["fallback_policy"] = "safe_stop_no_direction_inference"
            fallback_skipped_reason = (
                "incomplete_conical_probe_and_global_outward_fallback_disabled"
            )
            return False

        record["fallback_policy"] = "direction_free_global_outward_spiral"
        global_outward_fallback_used = True
        global_outward_fallback_started_active_time_s = metrics.active_time_s
        current_stage = (
            "incomplete conical evidence | direction-free global outward fallback"
        )
        fallback_radius_m = (
            BASELINE_MAX_RADIUS_M
            if baseline_max_radius_m is None
            else float(baseline_max_radius_m)
        )
        fallback_anchor_xy = nominal_tip[:2].copy()
        fallback_arc, fallback_xy = archimedean_path(
            fallback_anchor_xy,
            configured_spiral_pitch_m,
            fallback_radius_m,
            baseline_phase_rad,
        )
        return follow_path(
            fallback_arc,
            fallback_xy,
            current_stage,
            "recovery_spiral",
            continuous_until_capture=True,
        )

    progress = False
    conical_probe_owns_episode = bool(
        method in {"proposed", "learned", "hybrid", "direct"}
        and initial_conical_record is not None
        and not initial_conical_record["direction_inference_allowed"]
    )
    if (
        conical_probe_owns_episode
        and not hard_stop
        and metrics.active_time_s < time_budget_s
    ):
        progress = handle_incomplete_conical_probe(initial_conical_record)

    if not hard_stop and not conical_probe_owns_episode and method == "hybrid":
        assert estimate is not None and entropy_belief is not None
        assert planned_spiral_radius_m is not None
        assert planned_target_xy is not None and posterior is not None
        frozen_posterior = posterior.copy()
        frozen_target_xy = planned_target_xy.copy()
        frozen_spiral_radius_m = float(planned_spiral_radius_m)
        frozen_transport_distance_m = float(planned_transport_distance_m or 0.0)
        frozen_transport_metadata = {
            "policy": transport_policy,
            "direction_quality": transport_direction_quality,
            "quantile": transport_quantile,
            "reverse_probability": transport_reverse_probability,
            "projected_quantiles_mm": transport_projected_quantiles_mm,
            "raw_distance_mm": transport_raw_distance_mm,
            "distance_cap_mm": transport_distance_cap_mm,
        }
        initial_entropy_nats = float(entropy_belief.entropy_nats)
        initial_credible_radius_m = float(entropy_belief.credible_radius_95_m)
        hybrid_rng = np.random.default_rng(posterior_seed + 900001)
        fallback_radius_m = (
            BASELINE_MAX_RADIUS_M
            if baseline_max_radius_m is None
            else baseline_max_radius_m
        )
        fallback_arc_for_reserve, _ = archimedean_path(
            initial_xy,
            configured_spiral_pitch_m,
            fallback_radius_m,
            baseline_phase_rad,
        )
        hybrid_terminal_execution_reserve_s = (
            _nominal_terminal_execution_reserve_seconds(
                continue_to_bottom=continue_to_bottom,
                socket_depth_m=float(spec["socket_depth_m"]),
                conservative_capture_z_m=NOMINAL_TIP_Z_M,
                axial_command_speed_m_s=funnel_config.descent_speed_m_s,
                minimum_descent_s=6.0,
                force_release_s=funnel_config.capture_force_release_s,
                insertion_handoff_s=funnel_config.insertion_handoff_s,
                recovery_count=(
                    funnel_config.max_recoveries if funnel_config.enabled else 0
                ),
                recovery_duration_s=(
                    funnel_config.recovery_unload_s
                    + funnel_config.recovery_retract_s
                    + funnel_config.recovery_retouch_s
                ),
                disturbance_duration_s=(
                    disturbance_ramp_s + disturbance_hold_s + disturbance_ramp_s
                    if disturbance_tilt_deg > 0.0
                    else 0.0
                ),
                completion_event_margin_s=0.25,
                minimum_reserve_s=hybrid_config.terminal_execution_reserve_s,
            )
        )

        def hybrid_fallback_reserve_s(current_xy_m: np.ndarray) -> float:
            return _hybrid_fallback_reserve_seconds(
                current_xy_m,
                initial_xy,
                fallback_path_length_m=float(fallback_arc_for_reserve[-1]),
                xy_speed_m_s=xy_speed_m_s,
                return_position_error_bound_m=(
                    hybrid_config.fallback_return_position_error_bound_m
                ),
                transition_overhead_s=hybrid_config.fallback_transition_overhead_s,
                candidate_contingency_s=(
                    hybrid_config.fallback_candidate_contingency_s
                ),
                terminal_execution_reserve_s=hybrid_terminal_execution_reserve_s,
            )

        current_stage = "joint posterior shortcut | before entropy-grid search"
        shortcut_measured_xy: list[np.ndarray] = []
        shortcut_distance_m = float(
            np.linalg.norm(frozen_target_xy - nominal_tip[:2])
        )
        shortcut_duration_s = 0.40 + max(
            0.25, shortcut_distance_m / xy_speed_m_s
        )
        entry_remaining_s = time_budget_s - metrics.active_time_s
        entry_reserve_s = hybrid_fallback_reserve_s(
            snapshot.tip_position[:2]
        )
        shortcut_endpoint_reserve_s = hybrid_fallback_reserve_s(frozen_target_xy)
        hybrid_entry_fallback_feasible = bool(
            not global_outward_fallback or entry_remaining_s >= entry_reserve_s
        )
        residual_direction = estimate.direction_xy.copy()
        if frozen_transport_distance_m > 1.0e-12:
            residual_direction = (
                frozen_target_xy - initial_xy
            ) / frozen_transport_distance_m
        residual_axial_end_m = float(
            visual_bound_m
            if visual_bound_m is not None
            else fallback_radius_m + effective_capture_radius_m
        )
        residual_half_width_m = residual_axial_end_m * math.sin(
            math.radians(hybrid_residual_config.angular_half_width_deg)
        )
        residual_robust_capture_radius_m = (
            effective_capture_radius_m
            - hybrid_residual_config.tracking_error_allowance_m
        )
        if residual_robust_capture_radius_m <= 0.0:
            raise ValueError(
                "tracking_error_allowance_m must be smaller than the effective "
                "capture radius"
            )
        residual_lane_spacing_m = (
            2.0
            * residual_robust_capture_radius_m
            * hybrid_residual_config.lane_spacing_capture_fraction
        )
        residual_arc, residual_xy, residual_lane_offsets = (
            posterior_directional_corridor_path(
                initial_xy,
                residual_direction,
                axial_start_m=min(
                    frozen_transport_distance_m,
                    residual_axial_end_m - 1.0e-9,
                ),
                axial_end_m=residual_axial_end_m,
                half_width_m=residual_half_width_m,
                lane_spacing_m=residual_lane_spacing_m,
                close_to_start=True,
            )
        )
        residual_direction_eligible = bool(
            hybrid_residual_config.enabled
            and float(
                0.0
                if transport_direction_quality is None
                else transport_direction_quality
            )
            >= hybrid_residual_config.minimum_direction_quality
            and float(
                1.0
                if transport_reverse_probability is None
                else transport_reverse_probability
            )
            <= hybrid_residual_config.maximum_reverse_probability
        )
        residual_path_speed_m_s = (
            xy_speed_m_s
            if hybrid_residual_config.corridor_path_speed_m_s <= 0.0
            else hybrid_residual_config.corridor_path_speed_m_s
        )
        residual_corridor_duration_s = (
            float(residual_arc[-1]) / residual_path_speed_m_s
            + hybrid_residual_config.transition_overhead_s
        )
        residual_schedule_required_s = (
            shortcut_duration_s
            + residual_corridor_duration_s
            + hybrid_terminal_execution_reserve_s
        )
        residual_schedule_feasible = bool(
            residual_direction_eligible
            and entry_remaining_s >= residual_schedule_required_s - 1.0e-12
        )
        hybrid_primary_search_schedule = (
            "posterior_residual_corridor"
            if residual_schedule_feasible
            and (
                not hybrid_entry_fallback_feasible
                or residual_schedule_required_s < entry_reserve_s
            )
            else "full_global_outward"
            if hybrid_entry_fallback_feasible
            else "best_effort_global_outward"
        )
        shortcut_budget_feasible = bool(
            not global_outward_fallback
            or hybrid_primary_search_schedule == "posterior_residual_corridor"
            or _optional_action_fits_nominal_reserve(
                entry_remaining_s,
                shortcut_duration_s,
                shortcut_endpoint_reserve_s,
            )
        )
        if shortcut_budget_feasible:
            shortcut_moved, shortcut_capture = smooth_guarded_contact_move(
                frozen_target_xy,
                "joint-posterior bounded shortcut",
                probe_force_n=min(0.8, target_down_force_n),
                measured_xy_path=shortcut_measured_xy,
            )
        else:
            shortcut_moved, shortcut_capture = False, False
            hybrid_fallback_nominal_reserve_s = entry_reserve_s
            hybrid_budget_switch_active_time_s = metrics.active_time_s
            hybrid_budget_switch_reason = (
                "entry_budget_cannot_fit_nominal_fallback_schedule"
                if not hybrid_entry_fallback_feasible
                else "nominal_fallback_schedule_reserved_before_shortcut"
            )
        if shortcut_moved:
            shortcut_telemetry.mark_latest_executed()
        if (
            shortcut_budget_feasible
            and not shortcut_moved
            and not shortcut_capture
            and not hard_stop
        ):
            hybrid_guarded_motion_failure_reason = (
                "guarded_shortcut_motion_failed_or_soft_escape"
            )
        shortcut_gate = {
            "stage": "posterior_shortcut_before_entropy_grid",
            "executed": bool(shortcut_moved),
            "outcome": (
                "capture_during_shortcut"
                if shortcut_capture
                else "arrived_at_task_aware_grid_anchor"
                if shortcut_moved
                else "skipped_to_preserve_nominal_fallback_schedule"
                if not shortcut_budget_feasible
                else "motion_failed"
            ),
            "transport_distance_mm": 1000.0
            * float(np.linalg.norm(frozen_target_xy - initial_xy)),
            "remaining_active_budget_s": entry_remaining_s,
            "entry_fallback_reserve_s": entry_reserve_s,
            "shortcut_duration_estimate_s": shortcut_duration_s,
            "shortcut_endpoint_fallback_reserve_s": shortcut_endpoint_reserve_s,
            "entry_fallback_feasible": hybrid_entry_fallback_feasible,
            "selected_primary_search_schedule": hybrid_primary_search_schedule,
            "residual_direction_eligible": residual_direction_eligible,
            "residual_schedule_feasible": residual_schedule_feasible,
            "residual_schedule_required_s": residual_schedule_required_s,
            "residual_corridor_duration_s": residual_corridor_duration_s,
            "residual_corridor_path_length_mm": 1000.0
            * float(residual_arc[-1]),
            "residual_corridor_path_speed_mm_s": 1000.0
            * residual_path_speed_m_s,
            "residual_corridor_lane_count": int(len(residual_lane_offsets)),
            "residual_corridor_half_width_mm": 1000.0
            * residual_half_width_m,
            "residual_corridor_axial_interval_mm": [
                1000.0 * frozen_transport_distance_m,
                1000.0 * residual_axial_end_m,
            ],
            "residual_corridor_direction_quality": transport_direction_quality,
            "residual_corridor_reverse_probability": (
                transport_reverse_probability
            ),
            "residual_corridor_uses_true_center": False,
            "residual_tracking_error_allowance_mm": 1000.0
            * hybrid_residual_config.tracking_error_allowance_m,
            "residual_robust_capture_radius_mm": 1000.0
            * residual_robust_capture_radius_m,
        }
        if shortcut_moved and not shortcut_capture:
            shortcut_miss = _apply_no_capture_measured_polyline(
                entropy_belief,
                shortcut_measured_xy,
                effective_capture_radius_m=effective_capture_radius_m,
                miss_inside_likelihood=hybrid_config.capture_miss_inside_likelihood,
                miss_outside_likelihood=hybrid_config.capture_miss_outside_likelihood,
            )
            shortcut_gate["no_capture_probability_mass_soft_excluded"] = (
                shortcut_miss.covered_probability_before
            )
            shortcut_gate["miss_update_path_source"] = "measured_tip_xy_polyline"
            shortcut_gate["measured_path_samples"] = len(
                _deduplicate_xy_polyline(shortcut_measured_xy)
            )
        hybrid_gate_records.append(shortcut_gate)
        progress = bool(shortcut_capture)
        current_xy = snapshot.tip_position[:2].copy()
        planned_start_xy = current_xy.copy()
        hybrid_probe_moved = False
        consecutive_noncontracting_updates = 0
        coverage_mode = False

        for probe_index in range(
            0
            if progress
            or not shortcut_moved
            or hybrid_guarded_motion_failure_reason is not None
            or hybrid_primary_search_schedule
            == "posterior_residual_corridor"
            else hybrid_config.safety_probe_ceiling
        ):
            information_weight = (
                hybrid_config.coverage_information_gain_weight
                if coverage_mode
                else hybrid_config.information_gain_weight
            )
            capture_reward = (
                hybrid_config.coverage_capture_probability_reward_nats
                if coverage_mode
                else hybrid_config.capture_probability_reward_nats
            )
            entropy_selected_action = entropy_belief.select_task_aware_action(
                current_xy,
                effective_capture_radius_m=effective_capture_radius_m,
                information_gain_weight=information_weight,
                capture_probability_reward_nats=capture_reward,
                fixed_action_cost_nats=hybrid_config.fixed_contact_cost_nats,
                travel_penalty_nats_per_m=hybrid_config.travel_cost_nats_per_m,
            )
            gate_record = {
                "probe": probe_index + 1,
                "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                "travel_distance_mm": 1000.0
                * entropy_selected_action.travel_distance_m,
                "net_utility_nats": entropy_selected_action.net_utility_nats,
                "segment_capture_probability": (
                    entropy_selected_action.segment_capture_probability
                ),
                "action_mode": (
                    "capture_weighted_coverage"
                    if coverage_mode
                    else "capture_plus_information"
                ),
                "information_gain_weight": information_weight,
                "capture_probability_reward_nats": capture_reward,
                "entropy_before_nats": entropy_belief.entropy_nats,
                "credible_radius_before_mm": 1000.0
                * entropy_belief.credible_radius_95_m,
                "spiral_radius_before_mm": 1000.0 * planned_spiral_radius_m,
            }
            estimated_probe_duration_s = float(
                0.60
                + max(
                    0.25,
                    float(
                        np.linalg.norm(
                            entropy_selected_action.xy_m - nominal_tip[:2]
                        )
                    )
                    / xy_speed_m_s,
                )
            )
            reserve_s = hybrid_fallback_reserve_s(
                entropy_selected_action.xy_m
            )
            remaining_s = time_budget_s - metrics.active_time_s
            gate_record.update(
                {
                    "estimated_probe_duration_s": estimated_probe_duration_s,
                    "remaining_active_budget_s": remaining_s,
                    "nominal_fallback_reserve_from_action_endpoint_s": reserve_s,
                }
            )
            budget_allows_probe = bool(
                not global_outward_fallback
                or _optional_action_fits_nominal_reserve(
                    remaining_s,
                    estimated_probe_duration_s,
                    reserve_s,
                )
            )
            execute_probe = bool(
                budget_allows_probe
                and
                entropy_selected_action.net_utility_nats is not None
                and entropy_selected_action.net_utility_nats
                >= (
                    hybrid_config.minimum_coverage_utility_nats
                    if coverage_mode
                    else hybrid_config.minimum_net_utility_nats
                )
            )
            gate_record["executed"] = execute_probe
            gate_record["reason"] = (
                "nominal_fallback_schedule_reserved_before_probe"
                if not budget_allows_probe
                else "positive_task_value"
                if execute_probe
                else "motion_contact_cost_exceeds_capture_and_information_value"
            )
            hybrid_gate_records.append(gate_record)
            if not execute_probe:
                if not budget_allows_probe:
                    current_reserve_s = hybrid_fallback_reserve_s(current_xy)
                    hybrid_fallback_nominal_reserve_s = current_reserve_s
                    hybrid_budget_switch_active_time_s = metrics.active_time_s
                    hybrid_budget_switch_reason = (
                        gate_record["reason"]
                        if remaining_s >= current_reserve_s
                        else "nominal_schedule_condition_violated_best_effort"
                    )
                    gate_record["current_pose_nominal_fallback_reserve_s"] = (
                        current_reserve_s
                    )
                break

            measured_probe_xy: list[np.ndarray] = []
            if hybrid_config.continuous_contact_path:
                moved, captured_during_move = smooth_guarded_contact_move(
                    entropy_selected_action.xy_m,
                    f"hybrid task-aware posterior-grid node {probe_index + 1}",
                    probe_force_n=min(0.8, target_down_force_n),
                    measured_xy_path=measured_probe_xy,
                )
            else:
                moved = physical_reset_to_anchor(
                    entropy_selected_action.xy_m,
                    f"hybrid discrete entropy probe {probe_index + 1} | VoI stop",
                    count_as_recovery=False,
                    touchdown_force_n=min(0.8, target_down_force_n),
                    guarded_probe=True,
                )
                captured_during_move = False
                measured_probe_xy.append(snapshot.tip_position[:2].copy())
            if not moved or hard_stop:
                gate_record["outcome"] = "motion_failed"
                if not hard_stop:
                    hybrid_guarded_motion_failure_reason = (
                        f"guarded_probe_{probe_index + 1}_motion_failed_or_soft_escape"
                    )
                break
            hybrid_probe_moved = True
            current_xy = snapshot.tip_position[:2].copy()
            planned_start_xy = current_xy.copy()

            if captured_during_move:
                progress = True
                gate_record["outcome"] = "capture_during_smooth_segment"
                entropy_observation_records.append(
                    {
                        "probe": probe_index + 1,
                        "action_xy_mm": (
                            1000.0 * entropy_selected_action.xy_m
                        ).tolist(),
                        "observation": "capture_during_smooth_segment",
                        "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                        "net_utility_nats": entropy_selected_action.net_utility_nats,
                    }
                )
                break

            if inner_wall_condition(snapshot) or centered_inserted_condition(snapshot):
                current_stage = "hybrid entropy probe captured | sensor-only handoff"
                progress = confirm_progress(
                    nominal_tip.copy(),
                    require_progress_cue=False,
                )
                gate_record["outcome"] = "direct_capture"
                entropy_observation_records.append(
                    {
                        "probe": probe_index + 1,
                        "action_xy_mm": (
                            1000.0 * entropy_selected_action.xy_m
                        ).tolist(),
                        "observation": "direct_capture",
                        "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                        "net_utility_nats": entropy_selected_action.net_utility_nats,
                    }
                )
                break

            miss_update = _apply_no_capture_measured_polyline(
                entropy_belief,
                measured_probe_xy,
                effective_capture_radius_m=effective_capture_radius_m,
                miss_inside_likelihood=hybrid_config.capture_miss_inside_likelihood,
                miss_outside_likelihood=hybrid_config.capture_miss_outside_likelihood,
            )
            observation_category, torque_lever = torque_category_from_wrench(
                snapshot.measured_wrench,
                low_lever_threshold_m=entropy_config.low_lever_threshold_m,
                direction_bins=entropy_config.torque_direction_bins,
            )
            predictive_probability = entropy_belief.predictive_observation_probability(
                entropy_selected_action,
                observation_category,
            )
            likelihood_temperature = float(
                np.clip(
                    predictive_probability
                    / hybrid_config.predictive_probability_reference,
                    hybrid_config.minimum_likelihood_temperature,
                    1.0,
                )
            )
            if coverage_mode:
                likelihood_temperature = min(
                    likelihood_temperature,
                    hybrid_config.minimum_likelihood_temperature,
                )
            update = entropy_belief.update(
                entropy_selected_action,
                observation_category=observation_category,
                likelihood_temperature=likelihood_temperature,
            )
            posterior_contracting = bool(
                update.information_gain_nats > 0.0
                and 1000.0 * update.credible_radius_95_m
                <= 0.99 * float(gate_record["credible_radius_before_mm"])
            )
            consecutive_noncontracting_updates = (
                0
                if posterior_contracting
                else consecutive_noncontracting_updates + 1
            )
            gate_record.update(
                {
                    "outcome": "belief_update",
                    "observation_category": observation_category,
                    "no_capture_probability_mass_soft_excluded": (
                        miss_update.covered_probability_before
                    ),
                    "miss_update_path_source": "measured_tip_xy_polyline",
                    "measured_path_samples": len(
                        _deduplicate_xy_polyline(measured_probe_xy)
                    ),
                    "predictive_observation_probability": predictive_probability,
                    "likelihood_temperature": likelihood_temperature,
                    "entropy_after_nats": update.entropy_nats,
                    "credible_radius_after_mm": 1000.0
                    * update.credible_radius_95_m,
                    "posterior_contracting": posterior_contracting,
                    "consecutive_noncontracting_updates": (
                        consecutive_noncontracting_updates
                    ),
                }
            )
            entropy_observation_records.append(
                {
                    "probe": probe_index + 1,
                    "action_xy_mm": (
                        1000.0 * entropy_selected_action.xy_m
                    ).tolist(),
                    "observation": (
                        "low_moment"
                        if observation_category == 0
                        else f"torque_direction_bin_{observation_category - 1}"
                    ),
                    "torque_category": observation_category,
                    "normalized_torque_lever_xy_mm": (
                        1000.0 * torque_lever
                    ).tolist(),
                    "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                    "net_utility_nats": entropy_selected_action.net_utility_nats,
                    "segment_capture_probability": (
                        entropy_selected_action.segment_capture_probability
                    ),
                    "no_capture_probability_mass_soft_excluded": (
                        miss_update.covered_probability_before
                    ),
                    "miss_update_path_source": "measured_tip_xy_polyline",
                    "measured_path_samples": len(
                        _deduplicate_xy_polyline(measured_probe_xy)
                    ),
                    "predictive_observation_probability": predictive_probability,
                    "likelihood_temperature": likelihood_temperature,
                    "realized_entropy_change_nats": update.information_gain_nats,
                    "posterior_entropy_nats": update.entropy_nats,
                    "posterior_credible_radius_95_mm": 1000.0
                    * update.credible_radius_95_m,
                    "posterior_mean_xy_mm": (
                        1000.0 * update.posterior_mean_xy_m
                    ).tolist(),
                }
            )
            entropy_selected_action = None

            posterior = entropy_belief.sample_posterior(
                hybrid_config.posterior_sample_count,
                hybrid_rng,
            )
            posterior_direction = (
                entropy_belief.posterior_mean_xy_m - current_xy
            )
            direction_norm = float(np.linalg.norm(posterior_direction))
            if direction_norm <= 1.0e-12:
                posterior_direction = estimate.direction_xy.copy()
            else:
                posterior_direction /= direction_norm
            plan = plan_posterior_guided_search(
                current_xy,
                posterior,
                posterior_direction,
                effective_capture_radius_m=effective_capture_radius_m,
                posterior_coverage=posterior_coverage,
                direction_quality=float(
                    estimate.circular_concentration
                    * max(
                        0.0,
                        math.cos(
                            0.5
                            * math.radians(
                                min(180.0, estimate.ensemble_top1_span_deg)
                            )
                        ),
                    )
                ),
                max_transport_m=max_transport_m,
                pitch_safety_factor=0.80,
            )
            shortcut_telemetry.record(plan)
            planned_transport_distance_m = plan.transport_distance_m
            planned_target_xy = plan.transport_target_xy.copy()
            planned_target_is_latest_plan = True
            planned_spiral_radius_m = plan.spiral_max_radius_m
            transport_policy = f"hybrid_{plan.transport_mode}"
            transport_direction_quality = plan.direction_quality
            transport_quantile = plan.transport_quantile
            transport_reverse_probability = plan.reverse_probability
            transport_projected_quantiles_mm = {
                "q10": 1000.0 * plan.projected_q10_m,
                "q50": 1000.0 * plan.projected_q50_m,
                "q90": 1000.0 * plan.projected_q90_m,
            }
            transport_raw_distance_mm = 1000.0 * plan.transport_distance_raw_m
            transport_distance_cap_mm = 1000.0 * plan.transport_distance_cap_m
            gate_record["replanned_transport_mm"] = 1000.0 * plan.transport_distance_m
            gate_record["replanned_spiral_radius_mm"] = 1000.0 * plan.spiral_max_radius_m
            current_stage = (
                f"hybrid posterior updated | H={update.entropy_nats:.2f} nat | "
                f"R={1000.0 * plan.spiral_max_radius_m:.2f} mm"
            )
            snapshot = run_segment(
                0.20,
                lambda _: (nominal_tip.copy(), np.zeros(3)),
                current_stage,
                "hybrid_entropy_probe",
                True,
            )
            if (
                consecutive_noncontracting_updates
                >= hybrid_config.maximum_consecutive_noncontracting_updates
            ):
                coverage_mode = True
                consecutive_noncontracting_updates = 0
                gate_record["mode_switch_after_update"] = (
                    "downweight_ft_and_continue_capture_weighted_grid_coverage"
                )

        if not progress and not hard_stop:
            updated_posterior = entropy_belief.sample_posterior(
                hybrid_config.posterior_sample_count,
                hybrid_rng,
            )
            execution_start_xy = (
                nominal_tip[:2].copy()
                if hybrid_config.continuous_contact_path
                else initial_xy.copy()
            )
            updated_direction = (
                entropy_belief.posterior_mean_xy_m - execution_start_xy
            )
            updated_direction_norm = float(np.linalg.norm(updated_direction))
            if updated_direction_norm <= 1.0e-12:
                updated_direction = estimate.direction_xy.copy()
            else:
                updated_direction /= updated_direction_norm
            updated_plan = plan_posterior_guided_search(
                execution_start_xy,
                updated_posterior,
                updated_direction,
                effective_capture_radius_m=effective_capture_radius_m,
                posterior_coverage=posterior_coverage,
                direction_quality=float(
                    estimate.circular_concentration
                    * max(
                        0.0,
                        math.cos(
                            0.5
                            * math.radians(
                                min(180.0, estimate.ensemble_top1_span_deg)
                            )
                        ),
                    )
                ),
                max_transport_m=max_transport_m,
                pitch_safety_factor=0.80,
            )
            shortcut_telemetry.record(updated_plan)
            final_entropy_nats = float(entropy_belief.entropy_nats)
            final_credible_radius_m = float(entropy_belief.credible_radius_95_m)
            radius_reduction_fraction = (
                initial_credible_radius_m - final_credible_radius_m
            ) / max(initial_credible_radius_m, 1.0e-12)
            information_plan_accepted = bool(
                hybrid_probe_moved
                and final_entropy_nats < initial_entropy_nats
                and radius_reduction_fraction
                >= hybrid_config.minimum_credible_radius_reduction_fraction
                and updated_plan.spiral_max_radius_m
                <= hybrid_config.maximum_residual_radius_ratio
                * frozen_spiral_radius_m
            )
            optional_commit_target_xy = (
                updated_plan.transport_target_xy.copy()
                if information_plan_accepted
                else entropy_belief.posterior_map_xy_m.copy()
            )
            optional_commit_distance_m = float(
                np.linalg.norm(optional_commit_target_xy - execution_start_xy)
            )
            optional_commit_duration_s = 0.40 + max(
                0.25, optional_commit_distance_m / xy_speed_m_s
            )
            post_probe_remaining_s = time_budget_s - metrics.active_time_s
            post_commit_reserve_s = hybrid_fallback_reserve_s(
                optional_commit_target_xy
            )
            if (
                hybrid_budget_switch_reason is None
                and hybrid_guarded_motion_failure_reason is None
                and global_outward_fallback
                and hybrid_primary_search_schedule
                != "posterior_residual_corridor"
                and not _optional_action_fits_nominal_reserve(
                    post_probe_remaining_s,
                    optional_commit_duration_s,
                    post_commit_reserve_s,
                )
            ):
                current_reserve_s = hybrid_fallback_reserve_s(
                    execution_start_xy
                )
                hybrid_fallback_nominal_reserve_s = current_reserve_s
                hybrid_budget_switch_active_time_s = metrics.active_time_s
                hybrid_budget_switch_reason = (
                    "nominal_fallback_schedule_reserved_before_optional_commit"
                    if post_probe_remaining_s >= current_reserve_s
                    else "nominal_schedule_condition_violated_best_effort"
                )
            plan_gate = {
                "stage": "post_probe_plan_selection",
                "executed_probe_count": sum(
                    bool(item.get("executed")) and "probe" in item
                    for item in hybrid_gate_records
                ),
                "entropy_reduction_nats": initial_entropy_nats
                - final_entropy_nats,
                "credible_radius_reduction_fraction": radius_reduction_fraction,
                "updated_spiral_radius_mm": 1000.0
                * updated_plan.spiral_max_radius_m,
                "frozen_spiral_radius_mm": 1000.0 * frozen_spiral_radius_m,
                "updated_plan_accepted": information_plan_accepted,
                "remaining_active_budget_s": post_probe_remaining_s,
                "optional_commit_duration_estimate_s": optional_commit_duration_s,
                "post_commit_nominal_fallback_reserve_s": post_commit_reserve_s,
                "reason": (
                    "guarded_motion_failure_skips_all_optional_transport"
                    if hybrid_guarded_motion_failure_reason is not None
                    else "posterior_residual_corridor_selected_as_primary_schedule"
                    if hybrid_primary_search_schedule
                    == "posterior_residual_corridor"
                    else "nominal_fallback_schedule_switch_skips_optional_map_commit"
                    if hybrid_budget_switch_reason is not None
                    else "posterior_contracted_and_residual_plan_did_not_inflate"
                    if information_plan_accepted
                    else "information_health_gate_failed_use_guarded_grid_map_commit"
                ),
            }
            hybrid_gate_records.append(plan_gate)
            if hybrid_primary_search_schedule == "posterior_residual_corridor":
                posterior = frozen_posterior
                planned_target_xy = nominal_tip[:2].copy()
                planned_transport_distance_m = 0.0
                planned_target_is_latest_plan = False
                planned_spiral_radius_m = 0.0
                transport_policy = (
                    "hybrid_posterior_residual_corridor_primary"
                )
                plan_gate["post_entropy_commit"] = (
                    "not_applicable_primary_corridor_starts_at_first_shortcut"
                )
            elif (
                hybrid_budget_switch_reason is not None
                or hybrid_guarded_motion_failure_reason is not None
            ):
                posterior = updated_posterior
                planned_target_xy = execution_start_xy.copy()
                planned_transport_distance_m = 0.0
                planned_target_is_latest_plan = False
                planned_spiral_radius_m = 0.0
                transport_policy = (
                    "hybrid_guarded_motion_failure_global_outward"
                    if hybrid_guarded_motion_failure_reason is not None
                    else "hybrid_nominal_budget_reserve_global_outward"
                )
                plan_gate["post_entropy_commit"] = (
                    "skipped_after_guarded_motion_failure"
                    if hybrid_guarded_motion_failure_reason is not None
                    else "skipped_to_preserve_fallback_budget"
                )
            elif information_plan_accepted:
                posterior = updated_posterior
                planned_target_xy = updated_plan.transport_target_xy.copy()
                planned_transport_distance_m = updated_plan.transport_distance_m
                planned_target_is_latest_plan = True
                planned_spiral_radius_m = 0.0
                transport_policy = (
                    f"hybrid_task_grid_accepted_{updated_plan.transport_mode}"
                )
                transport_direction_quality = updated_plan.direction_quality
                transport_quantile = updated_plan.transport_quantile
                transport_reverse_probability = updated_plan.reverse_probability
                transport_projected_quantiles_mm = {
                    "q10": 1000.0 * updated_plan.projected_q10_m,
                    "q50": 1000.0 * updated_plan.projected_q50_m,
                    "q90": 1000.0 * updated_plan.projected_q90_m,
                }
                transport_raw_distance_mm = (
                    1000.0 * updated_plan.transport_distance_raw_m
                )
                transport_distance_cap_mm = (
                    1000.0 * updated_plan.transport_distance_cap_m
                )
            else:
                posterior = updated_posterior
                planned_target_xy = entropy_belief.posterior_map_xy_m.copy()
                planned_transport_distance_m = float(
                    np.linalg.norm(planned_target_xy - execution_start_xy)
                )
                planned_target_is_latest_plan = False
                planned_spiral_radius_m = 0.0
                transport_policy = (
                    "hybrid_task_grid_health_stop_guarded_map_commit"
                )
                transport_direction_quality = frozen_transport_metadata[
                    "direction_quality"
                ]
                transport_quantile = frozen_transport_metadata["quantile"]
                transport_reverse_probability = frozen_transport_metadata[
                    "reverse_probability"
                ]
                transport_projected_quantiles_mm = frozen_transport_metadata[
                    "projected_quantiles_mm"
                ]
                transport_raw_distance_mm = frozen_transport_metadata[
                    "raw_distance_mm"
                ]
                transport_distance_cap_mm = frozen_transport_metadata[
                    "distance_cap_mm"
                ]

            if (
                hybrid_probe_moved
                and hybrid_budget_switch_reason is None
                and hybrid_guarded_motion_failure_reason is None
            ):
                if not hybrid_config.continuous_contact_path:
                    reset_ok = physical_reset_to_anchor(
                        initial_xy,
                        "hybrid sensing complete | return to frozen contact anchor",
                        count_as_recovery=False,
                        touchdown_force_n=target_down_force_n,
                        guarded_probe=False,
                    )
                    if not reset_ok:
                        plan_gate["anchor_return"] = "failed"
                    elif not progress:
                        plan_gate["anchor_return"] = "completed"
                else:
                    commit_label = (
                        "task-aware grid converged | guarded commit to updated anchor"
                        if information_plan_accepted
                        else "task-aware grid health stop | guarded MAP commit"
                    )
                    if information_plan_accepted:
                        shortcut_telemetry.mark_latest_executed()
                    reset_ok, reset_capture = smooth_guarded_contact_move(
                        planned_target_xy,
                        commit_label,
                        probe_force_n=min(0.8, target_down_force_n),
                    )
                    if reset_capture:
                        progress = True
                        plan_gate["post_entropy_commit"] = "capture_during_guarded_move"
                    elif reset_ok:
                        plan_gate["post_entropy_commit"] = "completed"
                    else:
                        plan_gate["post_entropy_commit"] = "failed"
                        if not hard_stop:
                            hybrid_guarded_motion_failure_reason = (
                                "guarded_optional_commit_failed_or_soft_escape"
                            )
                            planned_target_xy = nominal_tip[:2].copy()
                            planned_transport_distance_m = 0.0
                            planned_target_is_latest_plan = False
                            planned_spiral_radius_m = 0.0
                            transport_policy = (
                                "hybrid_guarded_commit_failure_global_outward"
                            )

                if not progress and not hybrid_config.continuous_contact_path:
                    plan_gate["anchor_return"] = "completed"
                    current_stage = (
                        "hybrid sensing complete | restore nominal search force and impedance"
                    )
                    restore_start_force_n = float(rollout.target_wrench[2])
                    restore_steps = max(
                        1, int(round(0.25 / SIM_TIMESTEP_S))
                    )
                    for index in range(restore_steps):
                        fraction = smoothstep((index + 1) / restore_steps)
                        rollout.set_target_down_force(
                            restore_start_force_n
                            + fraction
                            * (target_down_force_n - restore_start_force_n)
                        )
                        snapshot = step_once(
                            nominal_tip.copy(),
                            np.zeros(3),
                            "hybrid_entropy_probe",
                            True,
                        )
                        if hard_stop or metrics.active_time_s >= time_budget_s:
                            plan_gate["nominal_controller_restore"] = "failed"
                            break
                    if not hard_stop:
                        rollout.set_target_down_force(target_down_force_n)
                        rollout.restore_planar_impedance()
                        rollout.restore_rotational_impedance()
                        plan_gate["nominal_controller_restore"] = "completed"
                elif (
                    not progress
                    and hybrid_config.continuous_contact_path
                    and reset_ok
                    and hybrid_config.local_spiral_fallback_enabled
                ):
                    current_stage = (
                        "entropy stage complete | restore nominal search force"
                    )
                    restore_start_force_n = float(rollout.target_wrench[2])
                    restore_steps = max(1, int(round(0.25 / SIM_TIMESTEP_S)))
                    for index in range(restore_steps):
                        fraction = smoothstep((index + 1) / restore_steps)
                        rollout.set_target_down_force(
                            restore_start_force_n
                            + fraction
                            * (target_down_force_n - restore_start_force_n)
                        )
                        snapshot = step_once(
                            nominal_tip.copy(),
                            np.zeros(3),
                            "hybrid_entropy_probe",
                            True,
                        )
                        if hard_stop or metrics.active_time_s >= time_budget_s:
                            plan_gate["nominal_controller_restore"] = "failed"
                            break
                    if not hard_stop:
                        rollout.set_target_down_force(target_down_force_n)
                        rollout.restore_planar_impedance()
                        rollout.restore_rotational_impedance()
                        plan_gate["nominal_controller_restore"] = "completed"
                elif (
                    not progress
                    and hybrid_config.continuous_contact_path
                    and reset_ok
                    and not hybrid_config.local_spiral_fallback_enabled
                ):
                    # Keep the task-grid's low-force controller active through
                    # the retract/shift/retouchdown reset. Restoring full search
                    # force at an arbitrary last probe pose can wedge the tip
                    # before the global fallback even starts.
                    plan_gate["nominal_controller_restore"] = (
                        "deferred_until_guarded_global_anchor"
                    )
            planned_start_xy = (
                nominal_tip[:2].copy()
                if hybrid_config.continuous_contact_path
                else initial_xy.copy()
            )

    if not hard_stop and not conical_probe_owns_episode and method == "direct":
        assert estimate is not None
        direct_limit_m = float(planned_transport_distance_m or 0.0)
        first_direction_xy = estimate.direction_xy.copy()
        adjustment_axis = np.asarray(
            [-first_direction_xy[1], first_direction_xy[0]], dtype=float
        )
        adjustment_norm = float(np.linalg.norm(adjustment_axis))
        if adjustment_norm <= 1.0e-12:
            adjustment_axis = np.asarray([0.0, 1.0], dtype=float)
        else:
            adjustment_axis /= adjustment_norm
        adjustment_signs = (0.0, 1.0, -1.0)

        for attempt_index in range(direct_max_attempts):
            attempt_number = attempt_index + 1
            attempt_anchor = initial_xy + (
                adjustment_signs[attempt_index]
                * float(direct_adjustment_m)
                * adjustment_axis
            )
            if attempt_index > 0:
                reset_ok = physical_reset_to_anchor(
                    attempt_anchor,
                    (
                        f"direct retry {attempt_number}/{direct_max_attempts} | "
                        "low-force shifted re-initialization"
                    ),
                    count_as_recovery=False,
                    touchdown_force_n=target_down_force_n,
                    guarded_probe=False,
                )
                if not reset_ok:
                    direct_attempt_records.append(
                        {
                            "attempt": attempt_number,
                            "anchor_xy_mm": (1000.0 * attempt_anchor).tolist(),
                            "outcome": (
                                "budget_exhausted_during_adjustment"
                                if metrics.active_time_s >= time_budget_s
                                else "adjustment_failed"
                            ),
                            "active_time_end_s": metrics.active_time_s,
                        }
                    )
                    break

                scan_phase.clear()
                scan_wrench.clear()
                scan_tip_z.clear()
                polar_history.clear()
                current_phase = None
                snapshot, retry_conical_record = execute_conical_probe(
                    label_prefix=(
                        f"direct attempt {attempt_number}/{direct_max_attempts} | "
                    ),
                    attempt_number=attempt_number,
                )
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    break
                if not retry_conical_record["direction_inference_allowed"]:
                    conical_probe_owns_episode = True
                    progress = handle_incomplete_conical_probe(
                        retry_conical_record
                    )
                    direct_attempt_records.append(
                        {
                            "attempt": attempt_number,
                            "anchor_xy_mm": (1000.0 * attempt_anchor).tolist(),
                            "outcome": (
                                "sensor_capture_handoff"
                                if progress
                                else "incomplete_conical_probe_no_direction_inference"
                            ),
                            "active_time_end_s": metrics.active_time_s,
                        }
                    )
                    if progress:
                        direct_successful_attempt = attempt_number
                    break
                fingerprint_bytes = b"".join(
                    (
                        np.asarray(scan_phase, dtype=np.float64).tobytes(),
                        np.asarray(scan_wrench, dtype=np.float64).tobytes(),
                        np.asarray(scan_tip_z, dtype=np.float64).tobytes(),
                    )
                )
                retry_fingerprint_sha256 = hashlib.sha256(
                    fingerprint_bytes
                ).hexdigest()
                fingerprint_sha256_attempts.append(retry_fingerprint_sha256)
                retry_conical_record[
                    "fingerprint_sha256"
                ] = retry_fingerprint_sha256
                bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
                estimate = direction_runtime.infer_scan(
                    np.asarray(scan_phase),
                    np.asarray(scan_wrench),
                    np.asarray(scan_tip_z),
                    target_down_force_n=target_down_force_n,
                    bit_radius_m=bit_radius_m,
                )
                conical_direction_inference_count += 1
                retry_conical_record["direction_inference_performed"] = True

            capture_direction_anchor_xy = attempt_anchor.copy()
            baseline_z_m = float(snapshot.tip_position[2])
            progress_depth_samples.clear()
            progress_depth_samples.append((metrics.active_time_s, baseline_z_m))
            direct_samples = max(2, int(math.ceil(direct_limit_m / 0.00001)))
            direct_arc = np.linspace(0.0, direct_limit_m, direct_samples)
            direct_xy = (
                attempt_anchor
                + direct_arc[:, None] * estimate.direction_xy[None, :]
            )
            direct_attempts_completed += 1
            progress = follow_path(
                direct_arc,
                direct_xy,
                (
                    f"fingerprint direct ray {attempt_number}/{direct_max_attempts} | "
                    "capture stops motion"
                ),
                "direct_slide",
                continuous_until_capture=True,
            )
            offline_true_direction_deg = float(
                math.degrees(
                    math.atan2(-float(attempt_anchor[1]), -float(attempt_anchor[0]))
                )
                % 360.0
            )
            direct_attempt_records.append(
                {
                    "attempt": attempt_number,
                    "anchor_xy_mm": (1000.0 * attempt_anchor).tolist(),
                    "control_direction_deg": float(
                        estimate.circular_mean_direction_deg
                    ),
                    "top_direction_deg": float(estimate.top_direction_deg),
                    "offline_true_direction_deg": offline_true_direction_deg,
                    "offline_control_error_deg": float(
                        abs(
                            (
                                float(estimate.circular_mean_direction_deg)
                                - offline_true_direction_deg
                                + 180.0
                            )
                            % 360.0
                            - 180.0
                        )
                    ),
                    "ray_limit_mm": 1000.0 * direct_limit_m,
                    "outcome": (
                        "sensor_capture_handoff"
                        if progress
                        else "hard_stop"
                        if hard_stop
                        else "budget_exhausted"
                        if metrics.active_time_s >= time_budget_s
                        else "ray_miss"
                    ),
                    "active_time_end_s": metrics.active_time_s,
                }
            )
            if progress:
                direct_successful_attempt = attempt_number
                break
            if hard_stop or metrics.active_time_s >= time_budget_s:
                break
    elif (
        not progress
        and not hard_stop
        and not conical_probe_owns_episode
        and method in {"proposed", "learned", "hybrid", "oracle"}
    ):
        assert planned_target_xy is not None
        assert planned_spiral_radius_m is not None
        transport_delta = planned_target_xy - planned_start_xy
        transport_distance = float(np.linalg.norm(transport_delta))
        transport_duration = transport_distance / xy_speed_m_s

        def transport_command(fraction: float) -> tuple[np.ndarray, np.ndarray]:
            xy = planned_start_xy + smoothstep(fraction) * transport_delta
            return np.array([xy[0], xy[1], NOMINAL_TIP_Z_M]), np.zeros(3)

        current_stage = (
                "learned-posterior bounded transport"
            if method == "learned"
            else "joint-posterior bounded transport"
            if method == "hybrid"
            else "oracle direct transport"
            if method == "oracle"
            else "assumed-posterior bounded transport"
        )
        transport_candidate_armed = True
        steps = (
            0
            if transport_distance <= 1.0e-12
            else max(1, int(round(transport_duration / SIM_TIMESTEP_S)))
        )
        latest_plan_execution_marked = False
        for index in range(steps):
            if metrics.active_time_s >= time_budget_s:
                break
            if planned_target_is_latest_plan and not latest_plan_execution_marked:
                shortcut_telemetry.mark_latest_executed()
                latest_plan_execution_marked = True
            tip, rpy = transport_command((index + 1) / steps)
            nominal_tip = tip.copy()
            snapshot = step_once(tip, rpy, "transport", True)
            is_candidate = progress_condition(snapshot)
            mark_trace(
                progress_candidate=is_candidate,
                inner_wall_gate=inner_wall_condition(snapshot),
                centered_inserted_gate=centered_inserted_condition(snapshot),
                candidate_attempt=candidate_attempts,
            )
            if (
                oracle_transport_recovery_enabled
                and search_escape_condition(snapshot)
            ):
                # The oracle arm is a privileged execution lower bound.  A
                # soft guard may interrupt its ray, but an intermediate contact
                # cue must not turn it into a different search policy.  The
                # explicit commissioning path below unloads and resumes the
                # remaining ray; raw hard stops remain latched.
                soft_escape_requested = True
                current_stage = "oracle ray soft-wrench escape requested"
                break
            if search_escape_condition(snapshot) and not is_candidate:
                soft_escape_requested = True
                current_stage = "pre-hard-stop wrench escape requested"
                break
            if oracle_transport_recovery_enabled:
                # Defer every online capture cue until measured endpoint
                # tracking reaches the commissioned center tolerance.
                transport_candidate_armed = True
            elif not is_candidate:
                transport_candidate_armed = True
            elif transport_candidate_armed:
                if confirm_progress(nominal_tip.copy()):
                    progress = True
                    break
                transport_candidate_armed = False
                current_stage = (
                    "learned-posterior bounded transport"
                    if method == "learned"
                    else "joint-posterior bounded transport"
                    if method == "hybrid"
                    else "oracle direct transport"
                    if method == "oracle"
                    else "assumed-posterior bounded transport"
                )
            if hard_stop:
                break
        oracle_center_ready = not oracle_transport_recovery_enabled
        if (
            oracle_transport_recovery_enabled
            and not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            assert oracle_transport_center_tolerance_m is not None

            def center_residual_m() -> float:
                return float(
                    np.linalg.norm(snapshot.tip_position[:2] - planned_target_xy)
                )

            def hold_oracle_center_until_ready() -> bool:
                """Smoothly close measured endpoint error, then dwell in tolerance."""

                nonlocal snapshot, nominal_tip, current_stage
                nonlocal soft_escape_requested
                approach_start_tip = snapshot.tip_position.copy()
                approach_rpy_command = last_rpy_command.copy()
                transition_record = atomic_measured_pose_rebase(
                    snapshot,
                    approach_rpy_command,
                    "oracle_endpoint_settle_entry",
                )
                approach_rpy_command = last_rpy_command.copy()
                _, held_approach_rpy = rollout.held_command_pose()
                approach_rpy_offset = wrapped_rpy_delta(
                    held_approach_rpy,
                    approach_rpy_command,
                )
                approach_upright_nominal_rpy = -approach_rpy_offset
                approach_delta_xy = planned_target_xy - approach_start_tip[:2]
                approach_distance_m = float(np.linalg.norm(approach_delta_xy))
                approach_duration_s = max(
                    0.10,
                    approach_distance_m / xy_speed_m_s,
                )
                approach_steps = max(
                    1,
                    int(round(approach_duration_s / SIM_TIMESTEP_S)),
                )
                settle_record = {
                    "active_time_start_s": metrics.active_time_s,
                    "measured_start_xy_mm": (
                        1000.0 * approach_start_tip[:2]
                    ).tolist(),
                    "initial_center_residual_mm": 1000.0 * center_residual_m(),
                    "smooth_remaining_center_command_mm": 1000.0
                    * approach_distance_m,
                    "approach_steps": approach_steps,
                    "entry_transition_event": transition_record["event"],
                    "approach_initial_rpy_command_deg": np.degrees(
                        approach_rpy_command
                    ).tolist(),
                    "approach_upright_nominal_rpy_deg": np.degrees(
                        approach_upright_nominal_rpy
                    ).tolist(),
                    "upright_return_smoothed_during_approach": True,
                    "raw_hard_thresholds_changed": False,
                    "soft_thresholds_changed": False,
                    "oracle_only_true_center_destination": True,
                }
                current_stage = (
                    "oracle endpoint settle | smooth remaining measured-center residual"
                )
                for index in range(approach_steps):
                    fraction = smoothstep((index + 1) / approach_steps)
                    xy = approach_start_tip[:2] + fraction * approach_delta_xy
                    nominal_tip = np.array(
                        [xy[0], xy[1], approach_start_tip[2]], dtype=float
                    )
                    approach_step_rpy = smooth_rpy_transition(
                        approach_rpy_command,
                        approach_upright_nominal_rpy,
                        (index + 1) / approach_steps,
                    )
                    snapshot = step_once(
                        nominal_tip.copy(),
                        approach_step_rpy,
                        "oracle_center_settle_approach",
                        True,
                    )
                    mark_trace(
                        progress_candidate=progress_condition(snapshot),
                        inner_wall_gate=inner_wall_condition(snapshot),
                        centered_inserted_gate=centered_inserted_condition(snapshot),
                        candidate_attempt=candidate_attempts,
                    )
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        settle_record["outcome"] = (
                            "raw_hard_stop" if hard_stop else "active_budget_exhausted"
                        )
                        settle_record["final_center_residual_mm"] = (
                            1000.0 * center_residual_m()
                        )
                        oracle_endpoint_settle_records.append(settle_record)
                        return False
                    if search_escape_condition(snapshot):
                        soft_escape_requested = True
                        current_stage = (
                            "oracle endpoint smooth settle soft-wrench escape requested"
                        )
                        settle_record["outcome"] = "soft_escape_during_smooth_approach"
                        settle_record["final_center_residual_mm"] = (
                            1000.0 * center_residual_m()
                        )
                        oracle_endpoint_settle_records.append(settle_record)
                        return False

                center_count = 0
                required = max(
                    1,
                    int(
                        round(
                            CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                            / 1000.0
                            / SIM_TIMESTEP_S
                        )
                    ),
                )
                settle_steps = max(
                    required,
                    int(round(funnel_config.insertion_handoff_s / SIM_TIMESTEP_S)),
                )
                # Keep the exact last approach RPY and z command.  The first
                # dwell step therefore changes neither orientation ownership
                # nor nominal z; only the already-smooth XY endpoint remains.
                dwell_rpy_command = last_rpy_command.copy()
                hold_z_m = float(nominal_tip[2])
                nominal_tip = np.array(
                    [planned_target_xy[0], planned_target_xy[1], hold_z_m],
                    dtype=float,
                )
                settle_record["dwell_rpy_command_deg"] = np.degrees(
                    dwell_rpy_command
                ).tolist()
                settle_record["rpy_command_changed_at_dwell_entry"] = False
                current_stage = (
                    "oracle endpoint tracking | measured-center tolerance dwell"
                )
                for _ in range(settle_steps):
                    snapshot = step_once(
                        nominal_tip.copy(),
                        dwell_rpy_command,
                        "oracle_center_settle",
                        True,
                    )
                    residual = center_residual_m()
                    center_count, confirmed = _advance_dwell_count(
                        center_count,
                        residual <= oracle_transport_center_tolerance_m,
                        required,
                    )
                    mark_trace(
                        progress_candidate=progress_condition(snapshot),
                        inner_wall_gate=inner_wall_condition(snapshot),
                        centered_inserted_gate=centered_inserted_condition(snapshot),
                        candidate_attempt=candidate_attempts,
                    )
                    if confirmed:
                        settle_record["outcome"] = "center_tolerance_dwell_reached"
                        settle_record["final_center_residual_mm"] = (
                            1000.0 * residual
                        )
                        settle_record["dwell_samples"] = center_count
                        oracle_endpoint_settle_records.append(settle_record)
                        return True
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        settle_record["outcome"] = (
                            "raw_hard_stop" if hard_stop else "active_budget_exhausted"
                        )
                        settle_record["final_center_residual_mm"] = (
                            1000.0 * residual
                        )
                        settle_record["dwell_samples"] = center_count
                        oracle_endpoint_settle_records.append(settle_record)
                        return False
                    if search_escape_condition(snapshot):
                        soft_escape_requested = True
                        current_stage = (
                            "oracle endpoint soft-wrench escape requested"
                        )
                        settle_record["outcome"] = "soft_escape_during_tolerance_dwell"
                        settle_record["final_center_residual_mm"] = (
                            1000.0 * residual
                        )
                        settle_record["dwell_samples"] = center_count
                        oracle_endpoint_settle_records.append(settle_record)
                        return False
                settle_record["outcome"] = "center_tolerance_dwell_not_reached"
                settle_record["final_center_residual_mm"] = 1000.0 * center_residual_m()
                settle_record["dwell_samples"] = center_count
                oracle_endpoint_settle_records.append(settle_record)
                return False

            # If the original ray reached its commanded endpoint without a
            # soft escape, first give the measured tip one bounded response-time
            # dwell.  A soft escape during that dwell enters the same recovery
            # path below.
            if not soft_escape_requested:
                oracle_center_ready = hold_oracle_center_until_ready()

            while (
                not oracle_center_ready
                and soft_escape_requested
                and not hard_stop
                and metrics.active_time_s < time_budget_s
            ):
                if oracle_transport_recovery_count >= funnel_config.max_recoveries:
                    oracle_transport_recovery_outcome = (
                        "soft_escape_recovery_limit_reached"
                    )
                    break
                oracle_transport_recovery_count += 1
                search_recovery_count += 1
                detection_tip = snapshot.tip_position.copy()
                detection_force_n = float(np.linalg.norm(snapshot.measured_wrench[:3]))
                detection_torque_mnm = 1000.0 * float(
                    np.linalg.norm(snapshot.measured_wrench[3:5])
                )
                record = {
                    "recovery": oracle_transport_recovery_count,
                    "detected_active_time_s": metrics.active_time_s,
                    "detected_tip_xy_mm": (1000.0 * detection_tip[:2]).tolist(),
                    "detected_center_residual_mm": 1000.0 * center_residual_m(),
                    "detected_measured_force_norm_n": detection_force_n,
                    "detected_measured_tau_xy_mnm": detection_torque_mnm,
                    "search_escape_force_threshold_n": CONTROLLER_THRESHOLDS[
                        "search_escape_force_N"
                    ],
                    "search_escape_tau_xy_threshold_mnm": CONTROLLER_THRESHOLDS[
                        "search_escape_tau_xy_mNm"
                    ],
                    "raw_hard_thresholds_changed": False,
                    "soft_thresholds_changed": False,
                    "center_source": "true_center_oracle_only",
                }
                current_stage = (
                    "oracle soft escape | measured-tip freeze + force unload"
                )
                # The nominal command must be rebased before the first unload
                # physics step.  Otherwise commanding the measured tip while
                # retaining the old admittance offset applies that offset a
                # second time and creates exactly the discontinuity this
                # recovery is meant to remove.
                freeze_tip = detection_tip.copy()
                pre_unload_transition = atomic_measured_pose_rebase(
                    snapshot,
                    last_rpy_command,
                    "oracle_soft_escape_pre_unload",
                )
                oracle_unload_rpy_command = last_rpy_command.copy()
                record["pre_unload_rebase_tip_xy_mm"] = (
                    1000.0 * freeze_tip[:2]
                ).tolist()
                record["pre_unload_transition_event"] = pre_unload_transition[
                    "event"
                ]
                record["pre_unload_rebase_rpy_deg"] = pre_unload_transition[
                    "measured_rpy_deg"
                ]
                force_start_n = float(rollout.target_wrench[2])
                recovery_force_n = min(
                    force_start_n,
                    funnel_config.recovery_force_n,
                )
                unload_steps = max(
                    1,
                    int(
                        round(
                            funnel_config.recovery_unload_s / SIM_TIMESTEP_S
                        )
                    ),
                )
                for index in range(unload_steps):
                    fraction = smoothstep((index + 1) / unload_steps)
                    rollout.set_target_down_force(
                        force_start_n
                        + fraction * (recovery_force_n - force_start_n)
                    )
                    snapshot = step_once(
                        freeze_tip,
                        oracle_unload_rpy_command,
                        "oracle_transport_soft_escape_unload",
                        True,
                    )
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        break
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    record["outcome"] = (
                        "raw_hard_stop" if hard_stop else "active_budget_exhausted"
                    )
                    oracle_transport_recovery_records.append(record)
                    break

                # Rebase only at the measured robot tip.  No geometry truth is
                # involved in this continuity operation; the true center enters
                # only as the oracle arm's already-declared destination.
                resume_start_tip = snapshot.tip_position.copy()
                post_unload_transition = atomic_measured_pose_rebase(
                    snapshot,
                    last_rpy_command,
                    "oracle_soft_escape_post_unload",
                )
                oracle_resume_rpy_command = last_rpy_command.copy()
                soft_escape_requested = False
                record["rebase_tip_xy_mm"] = (
                    1000.0 * resume_start_tip[:2]
                ).tolist()
                record["rebase_center_residual_mm"] = 1000.0 * center_residual_m()
                record["post_unload_transition_event"] = post_unload_transition[
                    "event"
                ]
                record["post_unload_rebase_rpy_deg"] = post_unload_transition[
                    "measured_rpy_deg"
                ]

                resume_delta_xy = planned_target_xy - resume_start_tip[:2]
                resume_distance_m = float(np.linalg.norm(resume_delta_xy))
                resume_duration_s = max(
                    0.10,
                    resume_distance_m / xy_speed_m_s,
                )
                resume_steps = max(
                    1, int(round(resume_duration_s / SIM_TIMESTEP_S))
                )
                current_stage = (
                    "oracle soft escape recovered | resume remaining center ray"
                )
                for index in range(resume_steps):
                    fraction = smoothstep((index + 1) / resume_steps)
                    xy = (
                        resume_start_tip[:2]
                        + fraction * resume_delta_xy
                    )
                    nominal_tip = np.array(
                        [xy[0], xy[1], resume_start_tip[2]], dtype=float
                    )
                    snapshot = step_once(
                        nominal_tip.copy(),
                        oracle_resume_rpy_command,
                        "oracle_transport_soft_escape_resume",
                        True,
                    )
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        break
                    if search_escape_condition(snapshot):
                        soft_escape_requested = True
                        current_stage = (
                            "oracle resumed ray soft-wrench escape requested"
                        )
                        break
                record["resume_command_distance_mm"] = 1000.0 * resume_distance_m
                record["post_resume_center_residual_mm"] = 1000.0 * center_residual_m()
                if hard_stop or metrics.active_time_s >= time_budget_s:
                    record["outcome"] = (
                        "raw_hard_stop" if hard_stop else "active_budget_exhausted"
                    )
                    oracle_transport_recovery_records.append(record)
                    break
                if soft_escape_requested:
                    record["outcome"] = "repeat_soft_escape"
                    oracle_transport_recovery_records.append(record)
                    continue

                oracle_center_ready = hold_oracle_center_until_ready()
                record["post_settle_center_residual_mm"] = 1000.0 * center_residual_m()
                record["outcome"] = (
                    "center_tolerance_reached"
                    if oracle_center_ready
                    else "repeat_soft_escape"
                    if soft_escape_requested
                    else "center_tolerance_not_reached"
                )
                oracle_transport_recovery_records.append(record)
                if not oracle_center_ready and not soft_escape_requested:
                    break

            oracle_transport_final_center_residual_m = center_residual_m()
            if oracle_center_ready:
                oracle_transport_recovery_outcome = (
                    "center_tolerance_reached_after_soft_escape_recovery"
                    if oracle_transport_recovery_count > 0
                    else "center_tolerance_reached_without_recovery"
                )
                soft_escape_requested = False
            elif oracle_transport_recovery_outcome is None:
                oracle_transport_recovery_outcome = (
                    "center_tolerance_not_reached"
                )
                fallback_skipped_reason = (
                    "oracle_center_tracking_tolerance_not_reached"
                )
        if (
            method == "oracle"
            and not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
            and (
                not oracle_transport_recovery_enabled
                or oracle_center_ready
            )
        ):
            # At the exact CAD center there need not be a chamfer/wall cue before
            # descent. The oracle is a privileged lower bound, so route directly
            # into the same guarded descent and terminal verifier.
            progress = confirm_progress(
                nominal_tip.copy(),
                require_progress_cue=False,
            )
        if (
            not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            if (
                planned_spiral_radius_m > 1.0e-9
                and (
                    method != "hybrid"
                    or hybrid_config.local_spiral_fallback_enabled
                )
            ):
                if method == "hybrid":
                    hybrid_spiral_executed = True
                    hybrid_spiral_fallback_reason = (
                        "value_of_information_exhausted_before_capture"
                    )
                phase = math.atan2(float(transport_delta[1]), float(transport_delta[0]))
                arc, xy_path = archimedean_path(
                    planned_target_xy,
                    planned_spiral_pitch_m,
                    planned_spiral_radius_m,
                    phase,
                )
                progress = follow_path(
                    arc,
                    xy_path,
                    (
                        "VoI exhausted | local outward spiral fallback + micro-yaw"
                        if method == "hybrid" and yaw_assist.enabled
                        else "VoI exhausted | local outward spiral fallback"
                        if method == "hybrid"
                        else "local outward spiral + micro-yaw capture assist"
                        if yaw_assist.enabled
                        else "local outward spiral after direction posterior"
                    ),
                    "spiral",
                )
        if (
            method == "hybrid"
            and hybrid_primary_search_schedule
            == "posterior_residual_corridor"
            and shortcut_moved
            and not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            remaining_before_corridor_s = time_budget_s - metrics.active_time_s
            corridor_required_now_s = (
                residual_corridor_duration_s
                + hybrid_terminal_execution_reserve_s
            )
            corridor_gate = {
                "stage": "posterior_residual_corridor_coverage",
                "selected_primary_search_schedule": (
                    hybrid_primary_search_schedule
                ),
                "remaining_active_budget_s": remaining_before_corridor_s,
                "corridor_plus_terminal_required_s": (
                    corridor_required_now_s
                ),
                "corridor_path_length_mm": 1000.0
                * float(residual_arc[-1]),
                "corridor_path_speed_mm_s": 1000.0
                * residual_path_speed_m_s,
                "corridor_lane_count": int(len(residual_lane_offsets)),
                "corridor_half_width_mm": 1000.0
                * residual_half_width_m,
                "corridor_lane_spacing_mm": 1000.0
                * residual_lane_spacing_m,
                "corridor_axial_interval_mm": [
                    1000.0 * frozen_transport_distance_m,
                    1000.0 * residual_axial_end_m,
                ],
                "conditional_geometric_contract": (
                    "ideal capture tube covers centers whose predicted-direction "
                    "coordinates lie inside the declared corridor; tracking, "
                    "contact sensing and terminal verification remain fallible"
                ),
                "uses_true_center_online": False,
                "hard_wrench_limits_relaxed": False,
            }
            corridor_budget_feasible_now = bool(
                remaining_before_corridor_s
                >= corridor_required_now_s - 1.0e-12
            )
            corridor_gate["executed"] = corridor_budget_feasible_now
            if corridor_budget_feasible_now:
                soft_escape_requested = False
                residual_restore_ok = True
                restore_start_force_n = float(rollout.target_wrench[2])
                restore_start_kp = float(rollout.position_kp[0])
                restore_start_kd = float(rollout.position_kd[0])
                restore_target_kp, restore_target_kd = (
                    rollout.nominal_planar_impedance()
                )
                restore_start_rot_kp = rollout.rotation_kp.copy()
                restore_start_rot_kd = rollout.rotation_kd.copy()
                restore_target_rot_kp, restore_target_rot_kd = (
                    rollout.nominal_rotational_impedance()
                )
                restore_steps = max(1, int(round(0.75 / SIM_TIMESTEP_S)))
                for index in range(restore_steps):
                    fraction = smoothstep((index + 1) / restore_steps)
                    rollout.set_target_down_force(
                        restore_start_force_n
                        + fraction
                        * (target_down_force_n - restore_start_force_n)
                    )
                    rollout.set_planar_impedance(
                        restore_start_kp
                        + fraction * (restore_target_kp - restore_start_kp),
                        restore_start_kd
                        + fraction * (restore_target_kd - restore_start_kd),
                    )
                    rollout.set_rotational_impedance(
                        restore_start_rot_kp
                        + fraction
                        * (restore_target_rot_kp - restore_start_rot_kp),
                        restore_start_rot_kd
                        + fraction
                        * (restore_target_rot_kd - restore_start_rot_kd),
                    )
                    snapshot = step_once(
                        nominal_tip.copy(),
                        np.zeros(3),
                        "residual_corridor_restore",
                        True,
                    )
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        residual_restore_ok = False
                        break
                if residual_restore_ok:
                    rollout.set_target_down_force(target_down_force_n)
                    rollout.restore_planar_impedance()
                    rollout.restore_rotational_impedance()
                corridor_gate["nominal_controller_restore"] = (
                    "completed" if residual_restore_ok else "failed"
                )
                measured_corridor_xy: list[np.ndarray] = []
                corridor_tracking_error_m: list[float] = []
                corridor_reference_xy: list[np.ndarray] = []
                corridor_reference_tracking_error_m: list[float] = []
                corridor_outer_correction_m: list[float] = []
                candidate_count_before_corridor = sensor_progress_candidate_count
                if residual_restore_ok:
                    residual_nominal_z_m = float(nominal_tip[2])
                    progress = follow_path(
                        residual_arc,
                        residual_xy,
                        (
                            "posterior direction reliable | finite-budget "
                            "residual corridor"
                        ),
                        "residual_corridor",
                        continuous_until_capture=True,
                        nominal_z_m=residual_nominal_z_m,
                        measured_xy_path=measured_corridor_xy,
                        planar_tracking_error_path_m=(
                            corridor_tracking_error_m
                        ),
                        reference_xy_path=corridor_reference_xy,
                        reference_tracking_error_path_m=(
                            corridor_reference_tracking_error_m
                        ),
                        outer_servo_correction_path_m=(
                            corridor_outer_correction_m
                        ),
                        path_outer_servo_config=hybrid_residual_config,
                        path_xy_speed_m_s=residual_path_speed_m_s,
                    )
                if measured_corridor_xy:
                    command_tracking = np.asarray(
                        corridor_tracking_error_m, dtype=float
                    )
                    tracking = np.asarray(
                        corridor_reference_tracking_error_m, dtype=float
                    )
                    correction = np.asarray(
                        corridor_outer_correction_m, dtype=float
                    )
                    corridor_gate.update(
                        {
                            "offline_true_center_used_for_scoring_only": True,
                            "offline_min_planned_nominal_center_distance_mm": 1000.0
                            * _minimum_distance_to_xy_polyline_m(
                                np.zeros(2), residual_xy
                            ),
                            "offline_min_nominal_center_distance_mm": 1000.0
                            * _minimum_distance_to_xy_polyline_m(
                                np.zeros(2), corridor_reference_xy
                            ),
                            "offline_min_measured_center_distance_mm": 1000.0
                            * _minimum_distance_to_xy_polyline_m(
                                np.zeros(2), measured_corridor_xy
                            ),
                            "planar_tracking_error_p50_mm": 1000.0
                            * float(np.quantile(tracking, 0.50)),
                            "planar_tracking_error_p90_mm": 1000.0
                            * float(np.quantile(tracking, 0.90)),
                            "planar_tracking_error_p99_mm": 1000.0
                            * float(np.quantile(tracking, 0.99)),
                            "planar_tracking_error_max_mm": 1000.0
                            * float(np.max(tracking)),
                            "command_tracking_error_p99_mm": 1000.0
                            * float(np.quantile(command_tracking, 0.99)),
                            "command_tracking_error_max_mm": 1000.0
                            * float(np.max(command_tracking)),
                            "outer_servo_enabled": (
                                hybrid_residual_config.measured_tip_outer_servo_enabled
                            ),
                            "outer_servo_correction_p99_mm": 1000.0
                            * float(np.quantile(correction, 0.99)),
                            "outer_servo_correction_max_mm": 1000.0
                            * float(np.max(correction)),
                            "sensor_progress_candidates_during_corridor": (
                                sensor_progress_candidate_count
                                - candidate_count_before_corridor
                            ),
                        }
                    )
                corridor_gate["outcome"] = (
                    "sensor_capture_handoff"
                    if progress
                    else "hard_stop"
                    if hard_stop
                    else "nominal_controller_restore_failed"
                    if not residual_restore_ok
                    else "soft_escape"
                    if soft_escape_requested
                    else "corridor_exhausted_without_capture"
                    if metrics.active_time_s < time_budget_s
                    else "common_time_budget_exhausted"
                )
                corridor_gate["active_time_end_s"] = metrics.active_time_s
            else:
                corridor_gate["outcome"] = (
                    "skipped_corridor_plus_terminal_no_longer_fits"
                )
            hybrid_gate_records.append(corridor_gate)
        if (
            method in {"learned", "hybrid"}
            and global_outward_fallback
            and not progress
            and not hard_stop
            and metrics.active_time_s >= time_budget_s
        ):
            fallback_skipped_reason = (
                "common_active_time_budget_exhausted_before_global_fallback"
            )
        if (
            method in {"learned", "hybrid"}
            and global_outward_fallback
            and not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            retire_capture_direction_gate_for_global_fallback(
                "initial_learned_posterior_abandoned_before_global_outward_fallback"
            )
            global_outward_fallback_used = True
            global_outward_fallback_started_active_time_s = metrics.active_time_s
            if method == "hybrid":
                hybrid_spiral_fallback_reason = (
                    "posterior_residual_miss_best_effort_finite_global_outward"
                    if hybrid_primary_search_schedule
                    == "posterior_residual_corridor"
                    else "task_aware_grid_exhausted_finite_global_outward"
                )
            soft_escape_requested = False
            guarded_global_reset = method == "hybrid"
            reset_capture = False
            if guarded_global_reset:
                reset_ok, reset_capture = smooth_guarded_contact_move(
                    initial_xy,
                    "task-grid exhausted | low-force return to visual anchor",
                    probe_force_n=min(0.8, target_down_force_n),
                    stage_group="fallback_reset",
                )
                if reset_capture:
                    progress = True
                    hybrid_spiral_fallback_reason = (
                        "capture_during_low_force_global_return"
                    )
                elif not reset_ok and not hard_stop:
                    fallback_skipped_reason = (
                        "guarded_global_return_failed_safe_search_abort"
                    )
            else:
                reset_ok = physical_reset_to_anchor(
                    initial_xy,
                    "posterior exhausted/unsafe | full baseline reset",
                )
            if reset_ok and not hard_stop and not progress:
                if guarded_global_reset:
                    current_stage = (
                        "global anchor stable | bumpless force/impedance restore"
                    )
                    restore_start_force_n = float(rollout.target_wrench[2])
                    restore_start_kp = float(rollout.position_kp[0])
                    restore_start_kd = float(rollout.position_kd[0])
                    restore_target_kp, restore_target_kd = (
                        rollout.nominal_planar_impedance()
                    )
                    restore_start_rot_kp = rollout.rotation_kp.copy()
                    restore_start_rot_kd = rollout.rotation_kd.copy()
                    restore_target_rot_kp, restore_target_rot_kd = (
                        rollout.nominal_rotational_impedance()
                    )
                    restore_steps = max(
                        1, int(round(0.75 / SIM_TIMESTEP_S))
                    )
                    for index in range(restore_steps):
                        fraction = smoothstep((index + 1) / restore_steps)
                        rollout.set_target_down_force(
                            restore_start_force_n
                            + fraction
                            * (target_down_force_n - restore_start_force_n)
                        )
                        rollout.set_planar_impedance(
                            restore_start_kp
                            + fraction * (restore_target_kp - restore_start_kp),
                            restore_start_kd
                            + fraction * (restore_target_kd - restore_start_kd),
                        )
                        rollout.set_rotational_impedance(
                            restore_start_rot_kp
                            + fraction
                            * (restore_target_rot_kp - restore_start_rot_kp),
                            restore_start_rot_kd
                            + fraction
                            * (restore_target_rot_kd - restore_start_rot_kd),
                        )
                        snapshot = step_once(
                            nominal_tip.copy(),
                            np.zeros(3),
                            "fallback_reset",
                            True,
                        )
                        if hard_stop or metrics.active_time_s >= time_budget_s:
                            reset_ok = False
                            break
                    if reset_ok and not hard_stop:
                        rollout.set_target_down_force(target_down_force_n)
                        rollout.restore_planar_impedance()
                        rollout.restore_rotational_impedance()
            if reset_ok and not hard_stop and not progress:
                fallback_nominal_z_m = float(nominal_tip[2])
                fallback_arc, fallback_xy = archimedean_path(
                    initial_xy,
                    configured_spiral_pitch_m,
                    BASELINE_MAX_RADIUS_M
                    if baseline_max_radius_m is None
                    else baseline_max_radius_m,
                    baseline_phase_rad,
                )
                progress = follow_path(
                    fallback_arc,
                    fallback_xy,
                    (
                        "posterior residual exhausted | best-effort finite "
                        "visual-support outward fallback"
                        if method == "hybrid"
                        and hybrid_primary_search_schedule
                        == "posterior_residual_corridor"
                        else "posterior exhausted | finite visual-support "
                        "outward fallback"
                    ),
                    "fallback_spiral",
                    continuous_until_capture=True,
                    nominal_z_m=fallback_nominal_z_m,
                )
            if (
                not progress
                and metrics.active_time_s >= time_budget_s
                and fallback_skipped_reason is None
            ):
                fallback_skipped_reason = (
                    "common_active_time_budget_exhausted_during_global_fallback"
                )
    elif not hard_stop and method == "entropy_grid":
        assert entropy_belief is not None
        current_xy = nominal_tip[:2].copy()
        while (
            not entropy_belief.should_stop
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            entropy_selected_action = entropy_belief.select_action(current_xy)
            if (
                entropy_selected_action.expected_information_gain_nats
                < entropy_config.min_information_gain_nats
            ):
                current_stage = "entropy gain exhausted | commit posterior mean"
                break
            probe_label = (
                f"entropy probe {len(entropy_belief.history) + 1}/"
                f"{entropy_config.max_probes}"
            )
            if float(np.linalg.norm(entropy_selected_action.xy_m - current_xy)) <= 1.0e-9:
                rollout.set_planar_impedance(300.0, 55.0)
                rollout.set_rotational_impedance(
                    (0.50, 0.50, 1.00),
                    (0.15, 0.15, 0.20),
                )
                probe_force_n = min(0.8, target_down_force_n)
                start_probe_force_n = float(rollout.target_wrench[2])
                force_ramp_steps = max(1, int(round(0.20 / SIM_TIMESTEP_S)))
                for index in range(force_ramp_steps):
                    fraction = smoothstep((index + 1) / force_ramp_steps)
                    rollout.set_target_down_force(
                        start_probe_force_n
                        + fraction * (probe_force_n - start_probe_force_n)
                    )
                    snapshot = step_once(
                        nominal_tip,
                        np.zeros(3),
                        "entropy_probe_dwell",
                        True,
                    )
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        break
                rollout.set_target_down_force(probe_force_n)
                snapshot = run_segment(
                    0.30,
                    lambda _: (nominal_tip.copy(), np.zeros(3)),
                    f"{probe_label} | guarded force dwell at current contact",
                    "entropy_probe_dwell",
                    True,
                )
                moved = not hard_stop
            else:
                moved = physical_reset_to_anchor(
                    entropy_selected_action.xy_m,
                    probe_label,
                    count_as_recovery=False,
                    touchdown_force_n=min(0.8, target_down_force_n),
                    guarded_probe=True,
                )
            if not moved or hard_stop:
                break
            current_xy = entropy_selected_action.xy_m.copy()

            if inner_wall_condition(snapshot) or centered_inserted_condition(snapshot):
                current_stage = "entropy probe landed in socket | sensor-only handoff"
                progress = confirm_progress(
                    nominal_tip.copy(),
                    require_progress_cue=False,
                )
                entropy_observation_records.append(
                    {
                        "probe": len(entropy_observation_records) + 1,
                        "action_xy_mm": (1000.0 * entropy_selected_action.xy_m).tolist(),
                        "observation": "direct_capture",
                        "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                    }
                )
                break

            observation_category, torque_lever = torque_category_from_wrench(
                snapshot.measured_wrench,
                low_lever_threshold_m=entropy_config.low_lever_threshold_m,
                direction_bins=entropy_config.torque_direction_bins,
            )
            update = entropy_belief.update(
                entropy_selected_action,
                observation_category=observation_category,
            )
            entropy_observation_records.append(
                {
                    "probe": len(entropy_observation_records) + 1,
                    "action_xy_mm": (1000.0 * entropy_selected_action.xy_m).tolist(),
                    "observation": (
                        "low_moment"
                        if observation_category == 0
                        else f"torque_direction_bin_{observation_category - 1}"
                    ),
                    "torque_category": observation_category,
                    "normalized_torque_lever_xy_mm": (
                        1000.0 * torque_lever
                    ).tolist(),
                    "expected_information_gain_nats": entropy_selected_action.expected_information_gain_nats,
                    "realized_entropy_change_nats": update.information_gain_nats,
                    "posterior_entropy_nats": update.entropy_nats,
                    "posterior_credible_radius_95_mm": 1000.0
                    * update.credible_radius_95_m,
                    "posterior_mean_xy_mm": (
                        1000.0 * update.posterior_mean_xy_m
                    ).tolist(),
                }
            )
            entropy_selected_action = None
            current_stage = (
                f"belief updated | H={update.entropy_nats:.2f} nat | "
                f"r95={1000.0 * update.credible_radius_95_m:.2f} mm"
            )
            snapshot = run_segment(
                0.20,
                lambda _: (nominal_tip.copy(), np.zeros(3)),
                current_stage,
                "entropy_probe_dwell",
                True,
            )

        if (
            not progress
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            planned_target_xy = entropy_belief.posterior_mean_xy_m.copy()
            start_xy = nominal_tip[:2].copy()
            planned_transport_distance_m = float(
                np.linalg.norm(planned_target_xy - start_xy)
            )
            planned_spiral_radius_m = 0.0
            current_stage = (
                "entropy converged/budget reached | guarded posterior-mean commit"
            )
            committed = physical_reset_to_anchor(
                planned_target_xy,
                "guarded posterior-mean touchdown | no spiral fallback",
                count_as_recovery=False,
                touchdown_force_n=min(0.8, target_down_force_n),
                guarded_probe=True,
            )
            if committed and not hard_stop:
                if inner_wall_condition(snapshot) or centered_inserted_condition(snapshot):
                    progress = confirm_progress(
                        nominal_tip.copy(),
                        require_progress_cue=False,
                    )
                else:
                    current_stage = (
                        "posterior mean contacted top support | safe miss, no spiral"
                    )
    elif not hard_stop and method == "random":
        random_radius_m = (
            BASELINE_MAX_RADIUS_M
            if baseline_max_radius_m is None
            else float(baseline_max_radius_m)
        )
        if random_search_path_budget_m is None:
            reference_arc, _ = archimedean_path(
                initial_xy,
                configured_spiral_pitch_m,
                random_radius_m,
                baseline_phase_rad,
            )
            actual_random_path_budget_m = float(reference_arc[-1])
        else:
            actual_random_path_budget_m = float(random_search_path_budget_m)
        random_arc, random_xy = random_waypoint_path(
            initial_xy,
            random_radius_m,
            actual_random_path_budget_m,
            random_search_seed,
        )
        random_waypoint_count = int(len(random_xy) - 1)
        progress = follow_path(
            random_arc,
            random_xy,
            "seeded area-uniform random waypoints | no probe / no direction",
            "random_search",
            continuous_until_capture=True,
        )
        if (
            not progress
            and soft_escape_requested
            and not hard_stop
            and metrics.active_time_s >= time_budget_s
        ):
            fallback_skipped_reason = (
                "common_active_time_budget_exhausted_before_random_recovery"
            )
        if (
            not progress
            and soft_escape_requested
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            soft_escape_requested = False
            if physical_reset_to_anchor(initial_xy, "random pre-hard-stop recovery"):
                random_arc, random_xy = random_waypoint_path(
                    initial_xy,
                    random_radius_m,
                    actual_random_path_budget_m,
                    random_search_seed + 1,
                )
                random_waypoint_count += int(len(random_xy) - 1)
                progress = follow_path(
                    random_arc,
                    random_xy,
                    "random recovered | independent waypoint sequence",
                    "random_search",
                    continuous_until_capture=True,
                )
            if not progress and metrics.active_time_s >= time_budget_s:
                fallback_skipped_reason = (
                    "common_active_time_budget_exhausted_during_random_recovery"
                )
    elif not hard_stop and method == "baseline":
        arc, xy_path = archimedean_path(
            initial_xy,
            configured_spiral_pitch_m,
            BASELINE_MAX_RADIUS_M
            if baseline_max_radius_m is None
            else baseline_max_radius_m,
            baseline_phase_rad,
        )
        progress = follow_path(
            arc,
            xy_path,
            "continuous outward spiral | stable 3 N down-force",
            "spiral",
            continuous_until_capture=True,
        )
        if (
            not progress
            and soft_escape_requested
            and not hard_stop
            and metrics.active_time_s >= time_budget_s
        ):
            fallback_skipped_reason = (
                "common_active_time_budget_exhausted_before_baseline_recovery"
            )
        if (
            not progress
            and soft_escape_requested
            and not hard_stop
            and metrics.active_time_s < time_budget_s
        ):
            soft_escape_requested = False
            if physical_reset_to_anchor(
                initial_xy,
                "baseline pre-hard-stop recovery",
            ):
                arc, xy_path = archimedean_path(
                    initial_xy,
                    configured_spiral_pitch_m,
                    BASELINE_MAX_RADIUS_M
                    if baseline_max_radius_m is None
                    else baseline_max_radius_m,
                    baseline_phase_rad,
                )
                progress = follow_path(
                    arc,
                    xy_path,
                    "baseline recovered | restart ordinary outward spiral",
                    "recovery_spiral",
                    continuous_until_capture=True,
                )
            if not progress and metrics.active_time_s >= time_budget_s:
                fallback_skipped_reason = (
                    "common_active_time_budget_exhausted_during_baseline_recovery"
                )

    if hard_stop:
        result_name = "hard_stop"
    elif progress:
        result_name = (
            "centered_inserted_gate_confirmed"
            if capture_mode == "centered_free_insertion"
            else "safe_downward_progress_confirmed"
            if capture_mode == "safe_downward_progress"
            else "inner_wall_gate_confirmed"
        )
        inner_wall_time_s = metrics.active_time_s
        event_xy = snapshot.tip_position[:2].copy()
        if continue_to_bottom:
            funnel_minimum_tip_z_m = (
                -float(spec["socket_depth_m"])
                - funnel_config.cad_bottom_overshoot_margin_m
            )
            funnel_tip_rcc_used = funnel_config.enabled
            rollout.set_tip_impedance_mode(funnel_config.enabled)
            if funnel_config.enabled:
                set_funnel_impedance(0.0)
            funnel_monitor.reset()
            funnel_guard_active = funnel_config.enabled
            funnel_allow_stall = False
            current_stage = (
                "peg-tip RCC funnel insertion + micro-yaw clocking | CAD-bottom verifier"
                if funnel_config.enabled
                and yaw_assist.enabled
                and method not in {"baseline", "entropy_grid", "random"}
                else "peg-tip RCC funnel insertion | CAD-bottom verifier"
                if funnel_config.enabled
                else "upright compliant insertion | CAD-bottom verifier"
            )
            # Arrest the capture transient before rebasing the insertion
            # controller. An instantaneous 3 N -> 0.15 N reset produced a
            # one-physics-step wall impulse even though the filtered wrench
            # remained small. A short continuous release is closer to a real
            # force-controller handoff and avoids commanding against chamfer
            # guidance while the contact mode is changing.
            release_force_n = min(
                (
                    funnel_config.recovery_force_n
                    if cad_conditioned_insertion
                    else 0.8
                ),
                target_down_force_n,
            )
            capture_force_start_n = float(rollout.target_wrench[2])
            insertion_xy_command = snapshot.tip_position[:2].copy() + insertion_bias_xy
            insertion_handoff_tip = snapshot.tip_position.copy()
            insertion_handoff_tip[:2] = insertion_xy_command

            if disturbance_tilt_deg > 0.0 and not hard_stop:
                axis_rad = math.radians(disturbance_axis_deg)
                disturbance_rpy = math.radians(disturbance_tilt_deg) * np.array(
                    [math.sin(axis_rad), math.cos(axis_rad), 0.0], dtype=float
                )
                disturbance_xy = disturbance_lateral_m * np.array(
                    [math.cos(axis_rad), math.sin(axis_rad)], dtype=float
                )
                ramp_steps = max(
                    1, int(round(disturbance_ramp_s / SIM_TIMESTEP_S))
                )
                hold_steps = max(
                    0, int(round(disturbance_hold_s / SIM_TIMESTEP_S))
                )
                current_stage = (
                    f"paired post-capture side-impact transient | "
                    f"tilt command {disturbance_tilt_deg:.1f} deg"
                )
                disturbance_interrupted = False
                for index in range(ramp_steps + hold_steps):
                    fraction = (
                        smoothstep((index + 1) / ramp_steps)
                        if index < ramp_steps
                        else 1.0
                    )
                    snapshot = step_once(
                        insertion_handoff_tip
                        + np.array(
                            [
                                fraction * disturbance_xy[0],
                                fraction * disturbance_xy[1],
                                0.0,
                            ]
                        ),
                        fraction * disturbance_rpy,
                        "post_capture_tilt_impact",
                        True,
                    )
                    if funnel_recovery_requested:
                        post_capture_disturbance_reason = funnel_recovery_reason
                        post_capture_disturbance_recovered = perform_funnel_recovery(
                            release_force_n
                        )
                        disturbance_interrupted = True
                        break
                    if hard_stop or funnel_abort_reason is not None:
                        disturbance_interrupted = True
                        break

                if (
                    not disturbance_interrupted
                    and not hard_stop
                    and funnel_abort_reason is None
                ):
                    if disturbance_persistent:
                        insertion_xy_command = insertion_xy_command + disturbance_xy
                        insertion_handoff_tip[:2] = insertion_xy_command
                        residual_insertion_rpy = disturbance_rpy.copy()
                        current_stage = (
                            "unrejected residual tilt/TCP bias | continue insertion"
                        )
                    else:
                        current_stage = "side-impact transient released | command upright"
                        return_steps = max(
                            1, int(round(disturbance_ramp_s / SIM_TIMESTEP_S))
                        )
                        for index in range(return_steps):
                            fraction = 1.0 - smoothstep((index + 1) / return_steps)
                            snapshot = step_once(
                                insertion_handoff_tip
                                + np.array(
                                    [
                                        fraction * disturbance_xy[0],
                                        fraction * disturbance_xy[1],
                                        0.0,
                                    ]
                                ),
                                fraction * disturbance_rpy,
                                "post_capture_tilt_release",
                                True,
                            )
                            if funnel_recovery_requested:
                                post_capture_disturbance_reason = funnel_recovery_reason
                                post_capture_disturbance_recovered = perform_funnel_recovery(
                                    release_force_n
                                )
                                break
                            if hard_stop or funnel_abort_reason is not None:
                                break

            release_steps = max(
                1,
                int(
                    round(
                        funnel_config.capture_force_release_s / SIM_TIMESTEP_S
                    )
                ),
            )
            # Capture-to-insertion ownership transfer begins at the measured
            # pose and the exact command held by the search controller.  Force
            # is released while this pose is held; pose/orientation motion is
            # introduced only after the sensor cue is requalified at low force.
            # Axial force integration is suspended during this bounded handoff
            # so a centered free-fit cannot descend through the CAD terminal
            # band merely because no contact is available to balance the small
            # released force command. Raw six-axis safety remains active.
            rollout.set_enabled_axes(
                np.array([True, True, False, False, False, False])
            )
            release_entry_rpy_command = last_rpy_command.copy()
            release_entry_transition = atomic_measured_pose_rebase(
                snapshot,
                release_entry_rpy_command,
                "capture_release_entry",
            )
            release_entry_rpy_command = last_rpy_command.copy()
            release_entry_tip = nominal_tip.copy()
            release_recovery_completed = False
            for index in range(release_steps):
                release_fraction = smoothstep((index + 1) / release_steps)
                rollout.set_target_down_force(
                    capture_force_start_n
                    + release_fraction
                    * (release_force_n - capture_force_start_n)
                )
                snapshot = step_once(
                    release_entry_tip,
                    release_entry_rpy_command,
                    "capture_force_release",
                    True,
                )
                if funnel_recovery_requested:
                    if not perform_funnel_recovery(release_force_n):
                        break
                    release_recovery_completed = True
                    break
                if hard_stop or funnel_abort_reason is not None:
                    break
            rollout.set_target_down_force(release_force_n)

            # Requalify capture using only measured depth and the six-axis
            # wrench under the released force command.  A latched pre-release
            # cue is not sufficient evidence that the bit remained captured
            # across a hybrid-controller transition.
            termination_before_requalification = bool(
                hard_stop or funnel_abort_reason is not None
            )
            termination_reason_before_requalification = (
                hard_stop_reason if hard_stop else funnel_abort_reason
            )
            requalification_transition: dict | None = None
            requalification_entry_rpy_command = last_rpy_command.copy()
            if not termination_before_requalification:
                requalification_transition = atomic_measured_pose_rebase(
                    snapshot,
                    requalification_entry_rpy_command,
                    "post_release_requalification_entry",
                )
                requalification_entry_rpy_command = last_rpy_command.copy()
            requalification_tip = nominal_tip.copy()
            requalification_required = max(
                1,
                int(
                    round(
                        CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                        / 1000.0
                        / SIM_TIMESTEP_S
                    )
                ),
            )
            requalification_steps = max(
                4 * requalification_required,
                int(round(0.25 / SIM_TIMESTEP_S)),
            )
            requalification_count = 0
            requalification_peak_force_n = 0.0
            requalification_peak_torque_mnm = 0.0
            if not termination_before_requalification:
                current_stage = (
                    "post-release sensor-only low-force capture requalification"
                )
                for _ in range(requalification_steps):
                    snapshot = step_once(
                        requalification_tip,
                        requalification_entry_rpy_command,
                        "post_release_requalification",
                        True,
                    )
                    force_norm = float(
                        np.linalg.norm(snapshot.measured_wrench[:3])
                    )
                    torque_xy_mnm = 1000.0 * float(
                        np.linalg.norm(snapshot.measured_wrench[3:5])
                    )
                    requalification_peak_force_n = max(
                        requalification_peak_force_n,
                        force_norm,
                    )
                    requalification_peak_torque_mnm = max(
                        requalification_peak_torque_mnm,
                        torque_xy_mnm,
                    )
                    cue = post_release_requalification_condition(snapshot)
                    mark_trace(
                        post_release_requalification_cue=cue,
                        inner_wall_gate=inner_wall_condition(snapshot),
                        centered_inserted_gate=centered_inserted_condition(snapshot),
                        candidate_attempt=candidate_attempts,
                    )
                    requalification_count, confirmed = _advance_dwell_count(
                        requalification_count,
                        cue,
                        requalification_required,
                    )
                    if confirmed:
                        post_release_requalified = True
                        break
                    if hard_stop or metrics.active_time_s >= time_budget_s:
                        break
            if post_release_requalified is not True:
                post_release_requalified = False
                if not hard_stop and funnel_abort_reason is None:
                    funnel_abort_reason = (
                        "post_release_capture_requalification_failed"
                    )
                    current_stage = (
                        "post-release capture cue lost | named low-force safe abort"
                    )
                    safe_abort_force_n = min(
                        release_force_n,
                        funnel_config.force_floor_n,
                    )
                    safe_abort_start_force_n = float(rollout.target_wrench[2])
                    safe_abort_rpy_command = last_rpy_command.copy()
                    atomic_measured_pose_rebase(
                        snapshot,
                        safe_abort_rpy_command,
                        "post_release_requalification_safe_abort",
                    )
                    safe_abort_rpy_command = last_rpy_command.copy()
                    safe_abort_tip = nominal_tip.copy()
                    safe_abort_steps = max(
                        1,
                        int(round(0.10 / SIM_TIMESTEP_S)),
                    )
                    for index in range(safe_abort_steps):
                        fraction = smoothstep((index + 1) / safe_abort_steps)
                        rollout.set_target_down_force(
                            safe_abort_start_force_n
                            + fraction
                            * (safe_abort_force_n - safe_abort_start_force_n)
                        )
                        snapshot = step_once(
                            safe_abort_tip,
                            safe_abort_rpy_command,
                            "post_release_safe_abort_unload",
                            True,
                        )
                        if hard_stop or metrics.active_time_s >= time_budget_s:
                            break
            post_release_requalification_records.append(
                {
                    "active_time_end_s": metrics.active_time_s,
                    "entry_transition_event": (
                        None
                        if requalification_transition is None
                        else requalification_transition["event"]
                    ),
                    "capture_release_entry_transition_event": (
                        release_entry_transition["event"]
                    ),
                    "required_consecutive_samples": requalification_required,
                    "achieved_consecutive_samples": requalification_count,
                    "requalified": bool(post_release_requalified),
                    "release_recovery_completed": release_recovery_completed,
                    "peak_measured_force_n": requalification_peak_force_n,
                    "peak_measured_tau_xy_mnm": (
                        requalification_peak_torque_mnm
                    ),
                    "skipped_due_to_existing_terminal_outcome": (
                        termination_before_requalification
                    ),
                    "preexisting_terminal_reason": (
                        termination_reason_before_requalification
                    ),
                    "failure_action": (
                        None
                        if (
                            post_release_requalified
                            or termination_before_requalification
                        )
                        else "post_release_capture_requalification_failed_safe_abort"
                    ),
                    "true_center_used": False,
                    "offline_contact_tags_used": False,
                    "raw_hard_thresholds_changed": False,
                    "soft_thresholds_changed": False,
                }
            )

            bottom_z_m = -float(spec["socket_depth_m"])
            # Paired controller ablations may inject a bounded post-capture
            # flange/TCP command bias. It is hidden from the guard and models
            # calibration/execution error, not privileged center knowledge.
            hold_xy = insertion_xy_command.copy()
            start_command_z_m = float(snapshot.tip_position[2])
            handoff_start_tip = snapshot.tip_position.copy()
            handoff_start_rpy_command = last_rpy_command.copy()
            if post_release_requalified and not hard_stop and funnel_abort_reason is None:
                # Re-enable compliant axial motion only after the low-force cue
                # has been sustained.  The following full handoff interval then
                # restores force ownership gradually instead of applying it as
                # a one-step change at descent entry.
                rollout.set_enabled_axes(
                    np.array([True, True, True, False, False, False])
                )
                atomic_measured_pose_rebase(
                    snapshot,
                    handoff_start_rpy_command,
                    "insertion_handoff_entry",
                )
                handoff_start_rpy_command = last_rpy_command.copy()
            handoff_nominal_start = nominal_tip.copy()
            held_tip_start, held_rpy_start = rollout.held_command_pose()
            handoff_tip_offset = held_tip_start - handoff_nominal_start
            handoff_rpy_offset = wrapped_rpy_delta(
                held_rpy_start,
                handoff_start_rpy_command,
            )
            handoff_target_tip = np.array(
                [hold_xy[0], hold_xy[1], handoff_start_tip[2]], dtype=float
            )
            handoff_recovery_completed = False
            handoff_steps = max(
                1,
                int(round(funnel_config.insertion_handoff_s / SIM_TIMESTEP_S)),
            )
            for index in range(handoff_steps if post_release_requalified else 0):
                insertion_fraction = smoothstep((index + 1) / handoff_steps)
                if funnel_config.enabled:
                    set_funnel_impedance(0.5 * insertion_fraction)
                physical_tip_reference = (
                    held_tip_start
                    + insertion_fraction * (handoff_target_tip - held_tip_start)
                )
                nominal_tip = physical_tip_reference - handoff_tip_offset
                target_physical_rpy = (
                    residual_insertion_rpy + capture_yaw_command()
                )
                nominal_rpy = smooth_rpy_transition(
                    handoff_start_rpy_command,
                    target_physical_rpy - handoff_rpy_offset,
                    insertion_fraction,
                )
                snapshot = step_once(
                    nominal_tip,
                    nominal_rpy,
                    "insertion_handoff",
                    True,
                )
                if funnel_recovery_requested:
                    if not perform_funnel_recovery(release_force_n):
                        break
                    handoff_recovery_completed = True
                    post_release_requalified = False
                    funnel_abort_reason = (
                        "capture_lost_during_insertion_handoff_after_recovery"
                    )
                    break
                if hard_stop or funnel_abort_reason is not None:
                    break
            if post_release_requalified and not hard_stop and funnel_abort_reason is None:
                # Descent begins at the measured end of the smooth handoff,
                # never at the pre-capture search nominal.
                atomic_measured_pose_rebase(
                    snapshot,
                    last_rpy_command,
                    "insertion_descent_entry",
                )
                hold_xy = snapshot.tip_position[:2].copy()
                start_command_z_m = float(snapshot.tip_position[2])
            if cad_conditioned_insertion:
                # Command to a CAD standoff and let the low-force admittance
                # close the final millimetre. The previous fixed -1.5 mm
                # command was specific to the 4 mm M8 socket and crossed the
                # M10 depth band too quickly for the terminal dwell.
                final_command_z_m = bottom_z_m + 0.0010
                descent_duration_s = max(
                    6.0,
                    abs(final_command_z_m - start_command_z_m)
                    / funnel_config.descent_speed_m_s,
                )
            else:
                final_command_z_m = min(-0.0015, start_command_z_m - 0.0010)
                descent_duration_s = 6.0
            descent_steps = max(
                1,
                int(round(descent_duration_s / SIM_TIMESTEP_S)),
            )
            bottom_required = max(
                1,
                int(
                    round(
                        CONTROLLER_THRESHOLDS["capture_dwell_ms"]
                        / 1000.0
                        / SIM_TIMESTEP_S
                    )
                ),
            )
            # When a dwell is rejected after the measured tip has already
            # entered the CAD terminal set, a v2 cross-size retry keeps the
            # frozen axial command.  A terminal verifier rejection is not, by
            # itself, evidence of lost capture; retracting and re-touching the
            # socket bottom injected a second contact impulse on M10.  The
            # default remains the legacy recovery, and any actual funnel-guard
            # event still takes the ordinary unload/retract/re-touch path.
            terminal_retry_hold_command_z_m: float | None = None
            terminal_retry_hold_rpy_command: np.ndarray | None = None
            while not hard_stop and funnel_abort_reason is None and not bottom_reached:
                terminal_attempts += 1
                terminal_verifier = TerminalCompletionVerifier(
                    TerminalVerifierConfig(
                        depth_tolerance_mm=CONTROLLER_THRESHOLDS[
                            "bottom_depth_tolerance_mm"
                        ],
                        bottom_force_n=CONTROLLER_THRESHOLDS["bottom_fz_N"],
                        low_lateral_force_n=CONTROLLER_THRESHOLDS[
                            "bottom_low_wrench_lateral_force_N"
                        ],
                        low_torque_xy_mnm=CONTROLLER_THRESHOLDS[
                            "bottom_low_wrench_tau_xy_mNm"
                        ],
                        stable_depth_span_mm=CONTROLLER_THRESHOLDS[
                            "bottom_stable_depth_span_mm"
                        ],
                        dwell_steps=bottom_required,
                    )
                )
                bottom_hold_command_z_m = terminal_retry_hold_command_z_m
                terminal_hold_rpy_command = (
                    None
                    if terminal_retry_hold_rpy_command is None
                    else terminal_retry_hold_rpy_command.copy()
                )
                pass_start_z_m = (
                    start_command_z_m
                    if terminal_attempts == 1
                    else float(snapshot.tip_position[2])
                )
                if terminal_attempts == 1:
                    pass_duration_s = descent_duration_s
                elif cad_conditioned_insertion:
                    pass_duration_s = max(
                        6.0,
                        abs(final_command_z_m - pass_start_z_m)
                        / funnel_config.descent_speed_m_s,
                    )
                else:
                    pass_duration_s = descent_duration_s
                pass_steps = max(
                    1,
                    int(round(pass_duration_s / SIM_TIMESTEP_S)),
                )
                pass_rpy_start = last_rpy_command.copy()
                pass_rpy_transition_steps = max(
                    1,
                    min(
                        pass_steps,
                        int(
                            round(
                                funnel_config.insertion_handoff_s
                                / SIM_TIMESTEP_S
                            )
                        ),
                    ),
                )
                descent_restart_after_recovery = False
                for index in range(pass_steps):
                    if (
                        metrics.active_time_s >= time_budget_s
                        or funnel_abort_reason is not None
                    ):
                        break
                    fraction = smoothstep((index + 1) / pass_steps)
                    if funnel_config.enabled:
                        set_funnel_impedance(0.5 + 0.5 * fraction)
                    commanded_z = (
                        bottom_hold_command_z_m
                        if bottom_hold_command_z_m is not None
                        else pass_start_z_m
                        + fraction * (final_command_z_m - pass_start_z_m)
                    )
                    nominal_tip = np.array([hold_xy[0], hold_xy[1], commanded_z])
                    funnel_allow_stall = bool(
                        bottom_depth_error_mm is not None
                        and bottom_depth_error_mm
                        <= 2.0 * CONTROLLER_THRESHOLDS[
                            "bottom_depth_tolerance_mm"
                        ]
                    )
                    if terminal_hold_rpy_command is not None:
                        terminal_rpy_command = terminal_hold_rpy_command.copy()
                    else:
                        desired_descent_rpy = (
                            residual_insertion_rpy + capture_yaw_command()
                        )
                        terminal_rpy_command = smooth_rpy_transition(
                            pass_rpy_start,
                            desired_descent_rpy,
                            min(
                                1.0,
                                (index + 1) / pass_rpy_transition_steps,
                            ),
                        )
                    snapshot = step_once(
                        nominal_tip,
                        terminal_rpy_command,
                        "upright_insertion",
                        True,
                    )
                    bottom_depth_error_mm = 1000.0 * abs(
                        float(snapshot.tip_position[2]) - bottom_z_m
                    )
                    if (
                        funnel_config.enabled
                        and bottom_hold_command_z_m is None
                        and bottom_depth_error_mm
                        <= CONTROLLER_THRESHOLDS["bottom_depth_tolerance_mm"]
                    ):
                        # The CAD depth band is a terminal set. Freeze the axial
                        # command on first entry and let the F/T dwell verifier
                        # settle instead of driving through the valid band.
                        # Terminal-set ownership transfer is a controller
                        # continuity invariant, not a tunable performance
                        # option.  Keep the legacy config bit in telemetry, but
                        # every enabled physical funnel uses the atomic rebase.
                        if funnel_config.enabled:
                            previous_hold_xy = hold_xy.copy()
                            previous_command_z_m = float(commanded_z)
                            measured_tip = snapshot.tip_position.copy()
                            # The socket/chamfer has already corrected some of
                            # the capture-event eccentricity.  Preserve that
                            # physical correction instead of letting the
                            # lateral servo pull the bit back toward its old
                            # contact point.  Rebasing nominal XYZ at the
                            # measured tip and zeroing stored admittance
                            # velocity/offset is pose-continuous, unlike a
                            # retract/re-touch or command teleport.
                            hold_xy = measured_tip[:2].copy()
                            bottom_hold_command_z_m = float(measured_tip[2])
                            nominal_tip = measured_tip.copy()
                            terminal_transition = atomic_measured_pose_rebase(
                                snapshot,
                                terminal_rpy_command,
                                "cad_terminal_set_entry",
                            )
                            terminal_hold_rpy_command = last_rpy_command.copy()
                            if (
                                funnel_config.axial_position_hold_in_terminal_set
                            ):
                                # Depth-set entry terminates the force-driven
                                # descent.  Hold the measured z pose while the
                                # unchanged low-wrench/depth verifier observes
                                # a dwell.  Raw wrench safety remains active on
                                # every physics step; only the admittance z
                                # integrator is disabled.
                                rollout.set_enabled_axes(
                                    np.array(
                                        [
                                            True,
                                            True,
                                            False,
                                            False,
                                            False,
                                            False,
                                        ]
                                    )
                                )
                                terminal_axial_position_hold_used = True
                            funnel_accumulated_energy_j += (
                                funnel_monitor.injected_energy_j
                            )
                            funnel_monitor.reset()
                            terminal_entry_rebase_records.append(
                                {
                                    "attempt": terminal_attempts,
                                    "active_time_s": metrics.active_time_s,
                                    "previous_hold_xy_mm": (
                                        1000.0 * previous_hold_xy
                                    ).tolist(),
                                    "measured_tip_xy_mm": (
                                        1000.0 * measured_tip[:2]
                                    ).tolist(),
                                    "released_lateral_command_error_mm": 1000.0
                                    * float(
                                        np.linalg.norm(
                                            measured_tip[:2] - previous_hold_xy
                                        )
                                    ),
                                    "previous_command_z_mm": 1000.0
                                    * previous_command_z_m,
                                    "measured_tip_z_mm": 1000.0
                                    * float(measured_tip[2]),
                                    "exact_just_used_rpy_command_deg": np.degrees(
                                        terminal_rpy_command
                                    ).tolist(),
                                    "transition_event": terminal_transition[
                                        "event"
                                    ],
                                    "position_reference_jump_mm": (
                                        terminal_transition[
                                            "position_reference_jump_mm"
                                        ]
                                    ),
                                    "orientation_reference_jump_deg": (
                                        terminal_transition[
                                            "orientation_reference_jump_deg"
                                        ]
                                    ),
                                    "post_rebase_measured_tip_error_mm": (
                                        terminal_transition[
                                            "post_rebase_measured_tip_error_mm"
                                        ]
                                    ),
                                    "raw_hard_thresholds_changed": False,
                                    "terminal_verifier_thresholds_changed": False,
                                    "configured_bumpless_terminal_entry_rebase": bool(
                                        funnel_config.bumpless_terminal_entry_rebase
                                    ),
                                    "atomic_terminal_entry_enforced": True,
                                    "axial_admittance_disabled_for_dwell": bool(
                                        terminal_axial_position_hold_used
                                    ),
                                }
                            )
                        else:
                            bottom_hold_command_z_m = float(commanded_z)
                    terminal_verification_mode = terminal_verifier.update(
                        depth_error_mm=bottom_depth_error_mm,
                        tip_z_mm=1000.0 * float(snapshot.tip_position[2]),
                        measured_wrench=snapshot.measured_wrench,
                    )
                    if terminal_verification_mode is not None:
                        bottom_reached = True
                        bottom_time_s = metrics.active_time_s
                        break
                    if funnel_recovery_requested:
                        if not perform_funnel_recovery(release_force_n):
                            break
                        hold_xy = snapshot.tip_position[:2].copy()
                        # Never continue the old interpolation fraction after
                        # recovery.  The next outer pass takes its origin from
                        # this measured z and starts again at fraction zero.
                        descent_restart_after_recovery = True
                        if funnel_recovery_transition_records:
                            funnel_recovery_transition_records[-1].update(
                                {
                                    "descent_pass_restarted": True,
                                    "restart_origin_tip_mm": (
                                        1000.0 * snapshot.tip_position
                                    ).tolist(),
                                    "stale_pass_fraction_reused": False,
                                }
                            )
                        break
                    if hard_stop:
                        break

                attempt_diagnostic = terminal_verifier.diagnostics()
                attempt_diagnostic.update(
                    {
                        "attempt": terminal_attempts,
                        "verified_mode": terminal_verification_mode,
                        "bottom_reached": bottom_reached,
                        "hard_stop": hard_stop,
                        "funnel_abort_reason": funnel_abort_reason,
                        "hold_command_xy_mm": (1000.0 * hold_xy).tolist(),
                        "actual_tip_xy_mm": (
                            1000.0 * snapshot.tip_position[:2]
                        ).tolist(),
                        "actual_center_eccentricity_mm": 1000.0
                        * float(np.linalg.norm(snapshot.tip_position[:2])),
                        "nominal_actual_xy_error_mm": 1000.0
                        * float(
                            np.linalg.norm(
                                snapshot.tip_position[:2] - hold_xy
                            )
                        ),
                        "actual_tip_z_mm": 1000.0
                        * float(snapshot.tip_position[2]),
                        "nominal_command_z_mm": 1000.0
                        * float(nominal_tip[2]),
                        "terminal_hold_command_z_mm": (
                            None
                            if bottom_hold_command_z_m is None
                            else 1000.0 * float(bottom_hold_command_z_m)
                        ),
                        "target_down_force_n": float(rollout.target_wrench[2]),
                    }
                )
                terminal_attempt_diagnostics.append(attempt_diagnostic)

                if (
                    descent_restart_after_recovery
                    and not hard_stop
                    and funnel_abort_reason is None
                    and metrics.active_time_s < time_budget_s
                ):
                    current_stage = (
                        "funnel recovery complete | restart descent from measured pose"
                    )
                    continue

                if bottom_reached or hard_stop or funnel_abort_reason is not None:
                    break
                if metrics.active_time_s >= time_budget_s:
                    budget_exhaustion_stage = (
                        budget_exhaustion_stage or "post_capture_terminal"
                    )
                    break
                if not funnel_config.retry_on_terminal_rejection:
                    break
                terminal_rejection_count += 1
                if (
                    funnel_config.hold_terminal_set_on_retry
                    and bottom_hold_command_z_m is not None
                ):
                    if terminal_rejection_count > funnel_config.max_recoveries:
                        funnel_abort_reason = "terminal_dwell_rejected"
                        break
                    terminal_retry_hold_command_z_m = float(
                        bottom_hold_command_z_m
                    )
                    terminal_retry_hold_rpy_command = (
                        None
                        if terminal_hold_rpy_command is None
                        else terminal_hold_rpy_command.copy()
                    )
                    nominal_tip = np.array(
                        [hold_xy[0], hold_xy[1], terminal_retry_hold_command_z_m]
                    )
                    rollout.set_target_down_force(release_force_n)
                    funnel_accumulated_energy_j += funnel_monitor.injected_energy_j
                    funnel_monitor.reset()
                    current_stage = (
                        f"terminal dwell retry {terminal_attempts + 1}/"
                        f"{funnel_config.max_recoveries + 1} | "
                        "hold first-entry CAD terminal command"
                    )
                    continue
                funnel_recovery_reason = "terminal_dwell_rejected"
                if not perform_funnel_recovery(release_force_n):
                    if funnel_abort_reason is None:
                        funnel_abort_reason = "post_capture_recovery_exhausted"
                    break
                hold_xy = snapshot.tip_position[:2].copy()
                current_stage = (
                    f"terminal retry {terminal_attempts + 1}/"
                    f"{funnel_config.max_recoveries + 1} | measured-tip rebase"
                )
            if hard_stop:
                result_name = "hard_stop"
            elif funnel_abort_reason is not None:
                result_name = "funnel_safe_abort"
                current_stage = (
                    f"funnel safe abort: {funnel_abort_reason} | contact unloaded"
                )
            elif bottom_reached:
                result_name = "cad_bottom_verified"
                current_stage = (
                    "CAD bottom verified | full engagement complete | "
                    f"{terminal_verification_mode}"
                )
            else:
                result_name = "bottom_not_verified"
                current_stage = "insertion descent ended without bottom verification"
            # Freeze the completed active-controller outcome before the
            # non-scoring visualization tail.  The tail still advances MuJoCo
            # so raw all-step diagnostics remain observable, but it must not
            # retroactively change success, hard-stop, or offline capture.
            terminal_outcome_snapshot = snapshot
            terminal_outcome_contact_tags = sorted(fast_contact_tags(model, data))
            terminal_outcome_hard_stop = hard_stop
            terminal_outcome_hard_stop_reason = hard_stop_reason
            run_segment(
                0.50,
                lambda _: (nominal_tip.copy(), np.zeros(3)),
                current_stage,
                "terminal_hold",
                False,
            )
            if (
                terminal_outcome_hard_stop_reason is None
                and hard_stop_reason is not None
            ):
                post_terminal_event_display_new_hard_stop = True
                post_terminal_event_display_hard_stop_reason = hard_stop_reason
            funnel_guard_active = False
        else:
            current_stage = "inner-wall confirmed | handoff to insertion"
            terminal_outcome_snapshot = snapshot
            terminal_outcome_contact_tags = sorted(fast_contact_tags(model, data))
            terminal_outcome_hard_stop = hard_stop
            terminal_outcome_hard_stop_reason = hard_stop_reason
            run_segment(
                0.50,
                lambda _: (nominal_tip.copy(), np.zeros(3)),
                current_stage,
                "handoff",
                False,
            )
            if (
                terminal_outcome_hard_stop_reason is None
                and hard_stop_reason is not None
            ):
                post_terminal_event_display_new_hard_stop = True
                post_terminal_event_display_hard_stop_reason = hard_stop_reason
    elif depth_commit_latched:
        result_name = "depth_commit_unverified"
    elif metrics.active_time_s >= time_budget_s:
        result_name = "common_time_budget_exhausted"
    elif fallback_skipped_reason == "guarded_global_return_failed_safe_search_abort":
        result_name = "guarded_search_abort"
    elif fallback_skipped_reason == "oracle_center_tracking_tolerance_not_reached":
        result_name = "oracle_center_tracking_not_reached"
    else:
        result_name = "search_radius_exhausted"

    # MuJoCo geometry is retained only as an offline scorer. It must never
    # affect the online capture handoff above.  When an inactive post-terminal
    # display tail exists, score the frozen active-controller outcome rather
    # than the later display-only physics state.
    scoring_snapshot = (
        terminal_outcome_snapshot
        if terminal_outcome_snapshot is not None
        else snapshot
    )
    scoring_hard_stop = (
        terminal_outcome_hard_stop
        if terminal_outcome_hard_stop is not None
        else hard_stop
    )
    scoring_hard_stop_reason = (
        terminal_outcome_hard_stop_reason
        if terminal_outcome_snapshot is not None
        else hard_stop_reason
    )
    event_tags = (
        list(terminal_outcome_contact_tags or [])
        if progress and terminal_outcome_snapshot is not None
        else sorted(fast_contact_tags(model, data))
        if progress
        else []
    )
    event_inside_capture = bool(
        progress
        and np.linalg.norm(scoring_snapshot.tip_position[:2])
        <= float(spec["derived"]["r_cap_top_m"])
    )
    offline_capture_validated = bool(
        progress
        and event_inside_capture
        and (
            "chamfer" in event_tags
            or "wall" in event_tags
            or "bottom" in event_tags
            or (
                "head_top" not in event_tags
                and 1000.0 * float(scoring_snapshot.tip_position[2])
                <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            )
        )
    )

    true_direction_deg = (
        math.degrees(math.atan2(float(-initial_xy[1]), float(-initial_xy[0]))) % 360.0
        if np.linalg.norm(initial_xy) > 1.0e-12
        else 0.0
    )
    predicted_direction_deg = (
        estimate.top_direction_deg
        if estimate is not None
        else true_direction_deg
        if method == "oracle"
        else None
    )
    direction_error_deg = (
        float(
            direction_training._circular_error_deg(
                np.asarray([predicted_direction_deg]),
                np.asarray([true_direction_deg]),
            )[0]
        )
        if predicted_direction_deg is not None
        else None
    )
    result = RunResult(
        method=method,
        result=result_name,
        inner_wall_confirmed=progress,
        time_to_inner_wall_s=inner_wall_time_s,
        candidate_attempts=candidate_attempts,
        total_active_time_s=metrics.active_time_s,
        actual_xy_path_mm=1000.0 * metrics.actual_xy_path_m,
        contact_xy_path_mm=1000.0 * metrics.contact_xy_path_m,
        lateral_work_mj=1000.0 * metrics.lateral_work_j,
        orientation_work_mj=1000.0 * metrics.orientation_work_j,
        wear_proxy_mj=1000.0 * metrics.wear_proxy_j,
        max_force_n=metrics.max_force_n,
        max_torque_mnm=1000.0 * metrics.max_torque_nm,
        stage_time_s={key: round(value, 6) for key, value in metrics.stage_time_s.items()},
        event_xy_mm=None if event_xy is None else (1000.0 * event_xy).tolist(),
        baseline_depth_mm=1000.0 * float(baseline_z_m),
        min_tip_z_mm=1000.0 * min_tip_z_m,
        hard_stop=bool(scoring_hard_stop),
        video_path=None if output_path is None else str(output_path.relative_to(ROOT)),
        frame_count=frame_count,
        initial_xy_mm=(1000.0 * initial_xy).tolist(),
        direction_source=(
            "prior_valid_inference_then_incomplete_retry_direction_free_fallback"
            if conical_probe_owns_episode and conical_direction_inference_count > 0
            else "none_incomplete_or_capture_transition_conical_probe"
            if conical_probe_owns_episode
            else "three_seed_ensemble"
            if method == "learned"
            else "three_seed_ensemble_plus_task_aware_grid"
            if method == "hybrid"
            else "jin_2021_style_three_channel_direction_proxy"
            if method == "direct" and fingerprint_display_mode == "jin_three_channel"
            else "shared_three_seed_ensemble_direction_only"
            if method == "direct"
            else "true_center_oracle"
            if method == "oracle"
            else "assumed_profile"
            if method == "proposed"
            else "none"
        ),
        predicted_direction_deg=predicted_direction_deg,
        true_direction_deg=true_direction_deg,
        direction_error_deg=direction_error_deg,
        direction_confidence=None if estimate is None else estimate.confidence,
        aps_set_size=None if estimate is None else len(estimate.aps_direction_classes),
        transport_allowed=(
            None
            if planned_transport_distance_m is None
            else planned_transport_distance_m > 1.0e-9
        ),
        legacy_transport_gate=None if estimate is None else estimate.transport_allowed,
        transport_distance_mm=(
            None
            if planned_transport_distance_m is None
            else 1000.0 * planned_transport_distance_m
        ),
        transport_target_xy_mm=(
            None if planned_target_xy is None else (1000.0 * planned_target_xy).tolist()
        ),
        spiral_max_radius_mm=(
            None
            if planned_spiral_radius_m is None
            else 1000.0 * planned_spiral_radius_m
        ),
        capture_mode=capture_mode,
        display_fast_forward_after_s=fast_forward_after_active_s,
        display_fast_forward_speed=fast_forward_speed,
        display_fast_forward_start_frame=fast_forward_start_frame,
        control_direction_deg=(
            None if estimate is None else estimate.circular_mean_direction_deg
        ),
        aps_direction_angles_deg=(
            None if estimate is None else list(estimate.aps_direction_angles_deg)
        ),
        circular_concentration=(
            None if estimate is None else estimate.circular_concentration
        ),
        ensemble_top1_span_deg=(
            None if estimate is None else estimate.ensemble_top1_span_deg
        ),
        bottom_reached=bottom_reached,
        time_to_bottom_s=bottom_time_s,
        bottom_depth_error_mm=bottom_depth_error_mm,
        terminal_verification_mode=terminal_verification_mode,
        terminal_outcome_hard_stop=terminal_outcome_hard_stop,
        post_terminal_event_display_new_hard_stop=(
            post_terminal_event_display_new_hard_stop
        ),
        post_terminal_event_display_hard_stop_reason=(
            post_terminal_event_display_hard_stop_reason
        ),
        offline_capture_scoring_snapshot=(
            "pre_inactive_post_terminal_display"
            if terminal_outcome_snapshot is not None
            else "final_active_outcome"
        ),
        top_state=None if estimate is None else estimate.top_state,
        state_probabilities=(
            None
            if estimate is None
            else {
                state: float(estimate.state_probabilities[index])
                for index, state in enumerate(direction_training.STATE_NAMES)
            }
        ),
        posterior_model=(
            None
            if conical_probe_owns_episode
            else "visual_x_soft_cad_x_conical_fingerprint_x_signed_ft_entropy_updates"
            if method == "hybrid"
            else
            (
                "synthetic_visual_gaussian_x_ft_direction_and_state_x_soft_cad_radial_annuli"
                if use_state_cad_fusion
                else "synthetic_visual_gaussian_x_ft_direction_only"
            )
            if method == "learned"
            else "visual_grid_x_soft_cad_edge_load_likelihood"
            if method == "entropy_grid"
            else None
        ),
        cad_radial_boundaries_mm=(
            {
                "full_outer_support": 1000.0 * cad_geometry.full_outer_support_radius_m,
                "inner_opening_overlap_limit": 1000.0
                * cad_geometry.inner_opening_overlap_limit_m,
                "transition_sigma": 1000.0 * cad_geometry.transition_sigma_m,
            }
            if method in {"learned", "hybrid"}
            else None
        ),
        posterior_radial_quantiles_mm=posterior_radial_quantiles_mm,
        transport_policy=transport_policy,
        transport_direction_quality=transport_direction_quality,
        transport_quantile=transport_quantile,
        transport_reverse_probability=transport_reverse_probability,
        transport_projected_quantiles_mm=transport_projected_quantiles_mm,
        transport_raw_distance_mm=transport_raw_distance_mm,
        transport_distance_cap_mm=transport_distance_cap_mm,
        first_shortcut_distance_mm=(
            shortcut_telemetry.first_shortcut_distance_mm
        ),
        first_shortcut_target_xy_mm=(
            shortcut_telemetry.first_shortcut_target_xy_mm
        ),
        first_shortcut_direction_quality=(
            shortcut_telemetry.first_shortcut_direction_quality
        ),
        first_shortcut_quantile=shortcut_telemetry.first_shortcut_quantile,
        first_shortcut_reverse_probability=(
            shortcut_telemetry.first_shortcut_reverse_probability
        ),
        first_shortcut_distance_cap_mm=(
            shortcut_telemetry.first_shortcut_distance_cap_mm
        ),
        first_shortcut_executed=shortcut_telemetry.first_shortcut_executed,
        latest_planned_distance_mm=shortcut_telemetry.latest_planned_distance_mm,
        latest_planned_target_xy_mm=(
            shortcut_telemetry.latest_planned_target_xy_mm
        ),
        latest_plan_update_index=shortcut_telemetry.latest_plan_update_index,
        latest_plan_executed=shortcut_telemetry.latest_plan_executed,
        max_filtered_force_n=metrics.max_filtered_force_n,
        max_filtered_torque_mnm=1000.0 * metrics.max_filtered_torque_nm,
        max_raw_physical_force_n=metrics.max_raw_physical_force_n,
        max_raw_physical_torque_mnm=(
            1000.0 * metrics.max_raw_physical_torque_nm
        ),
        stage_max_force_n={
            key: float(value) for key, value in metrics.stage_max_force_n.items()
        },
        stage_max_torque_mnm={
            key: 1000.0 * float(value)
            for key, value in metrics.stage_max_torque_nm.items()
        },
        stage_max_filtered_force_n={
            key: float(value)
            for key, value in metrics.stage_max_filtered_force_n.items()
        },
        stage_max_raw_physical_force_n={
            key: float(value)
            for key, value in metrics.stage_max_raw_physical_force_n.items()
        },
        stage_max_raw_physical_torque_mnm={
            key: 1000.0 * float(value)
            for key, value in metrics.stage_max_raw_physical_torque_nm.items()
        },
        fingerprint_sha256=fingerprint_sha256,
        global_outward_fallback_used=global_outward_fallback_used,
        global_outward_fallback_started_active_time_s=(
            global_outward_fallback_started_active_time_s
        ),
        capture_yaw_assist_enabled=(
            yaw_assist.enabled
            and method not in {"baseline", "entropy_grid", "random"}
        ),
        capture_yaw_assist_amplitude_deg=yaw_assist.amplitude_deg,
        capture_yaw_assist_frequency_hz=yaw_assist.frequency_hz,
        capture_yaw_assist_ramp_time_s=yaw_assist.ramp_time_s,
        capture_yaw_assist_soft_tau_z_limit_mnm=yaw_assist.soft_tau_z_limit_mnm,
        capture_yaw_assist_used=yaw_assist_used,
        capture_yaw_assist_aborted=yaw_assist_aborted,
        max_abs_tau_z_mnm=max_abs_tau_z_mnm,
        capture_handoff_used=capture_handoff_used,
        sensor_progress_candidate_count=sensor_progress_candidate_count,
        search_recovery_count=search_recovery_count,
        oracle_transport_soft_escape_recovery_enabled=(
            oracle_transport_recovery_enabled
        ),
        oracle_transport_soft_escape_recovery_count=(
            oracle_transport_recovery_count
        ),
        oracle_transport_center_tolerance_mm=(
            None
            if oracle_transport_center_tolerance_m is None
            else 1000.0 * oracle_transport_center_tolerance_m
        ),
        oracle_transport_final_center_residual_mm=(
            None
            if oracle_transport_final_center_residual_m is None
            else 1000.0 * oracle_transport_final_center_residual_m
        ),
        oracle_transport_recovery_outcome=(
            oracle_transport_recovery_outcome
        ),
        oracle_transport_recovery_records=(
            oracle_transport_recovery_records
            if oracle_transport_recovery_enabled
            else None
        ),
        fallback_physical_reset_used=fallback_physical_reset_used,
        fallback_budget_extension_s=fallback_budget_extension_s,
        active_time_budget_s=time_budget_s,
        budget_exhaustion_stage=budget_exhaustion_stage,
        fallback_skipped_reason=fallback_skipped_reason,
        hard_stop_reason=scoring_hard_stop_reason,
        conditioned_on_ambiguous_top_contact=bool(
            conditioned_on_ambiguous_top_contact
        ),
        preprobe_route=preprobe_route,
        conical_probe_attempt_records=(
            conical_probe_attempt_records
            if method in {"proposed", "learned", "hybrid", "direct"}
            else None
        ),
        conical_direction_inference_count=conical_direction_inference_count,
        conical_capture_handoff_count=conical_capture_handoff_count,
        conical_incomplete_fallback_used=conical_incomplete_fallback_used,
        funnel_guard_enabled=funnel_config.enabled and continue_to_bottom,
        funnel_tip_rcc_used=funnel_tip_rcc_used,
        funnel_recovery_count=funnel_recovery_count,
        funnel_injected_energy_mj=1000.0
        * (funnel_accumulated_energy_j + funnel_monitor.injected_energy_j),
        funnel_peak_injected_power_w=funnel_peak_power_w,
        funnel_min_health=(
            None if not np.isfinite(funnel_min_health) else funnel_min_health
        ),
        funnel_abort_reason=funnel_abort_reason,
        insertion_command_bias_mm=(1000.0 * insertion_bias_xy).tolist(),
        post_capture_tilt_disturbance_deg=disturbance_tilt_deg,
        post_capture_tilt_axis_deg=disturbance_axis_deg,
        post_capture_lateral_disturbance_mm=1000.0 * disturbance_lateral_m,
        post_capture_disturbance_persistent=disturbance_persistent,
        post_capture_tilt_duration_s=(
            disturbance_ramp_s + disturbance_hold_s + disturbance_ramp_s
            if disturbance_tilt_deg > 0.0
            else 0.0
        ),
        post_capture_disturbance_recovered=post_capture_disturbance_recovered,
        post_capture_disturbance_reason=post_capture_disturbance_reason,
        max_tilt_deg=metrics.max_tilt_deg,
        final_tilt_deg=math.degrees(
            float(
                np.linalg.norm(
                    [scoring_snapshot.roll, scoring_snapshot.pitch]
                )
            )
        ),
        online_capture_sensor_only=True,
        offline_capture_validated=offline_capture_validated,
        offline_capture_tags=event_tags,
        random_search_seed=(random_search_seed if method == "random" else None),
        random_search_path_budget_mm=(
            None
            if actual_random_path_budget_m is None
            else 1000.0 * actual_random_path_budget_m
        ),
        random_search_waypoints=random_waypoint_count,
        entropy_grid_probe_count=(
            None
            if entropy_belief is None
            else len(entropy_observation_records)
        ),
        entropy_grid_initial_nats=entropy_initial_nats,
        entropy_grid_final_nats=(
            None if entropy_belief is None else entropy_belief.entropy_nats
        ),
        entropy_grid_credible_radius_mm=(
            None
            if entropy_belief is None
            else 1000.0 * entropy_belief.credible_radius_95_m
        ),
        entropy_grid_map_xy_mm=(
            None
            if entropy_belief is None
            else (1000.0 * entropy_belief.posterior_map_xy_m).tolist()
        ),
        entropy_grid_mean_xy_mm=(
            None
            if entropy_belief is None
            else (1000.0 * entropy_belief.posterior_mean_xy_m).tolist()
        ),
        entropy_grid_observations=(
            entropy_observation_records
            if method in {"entropy_grid", "hybrid"}
            else None
        ),
        entropy_grid_used_spiral=(
            False
            if method == "entropy_grid"
            else hybrid_spiral_executed
            if method == "hybrid"
            else None
        ),
        hybrid_spiral_fallback_reason=(
            hybrid_spiral_fallback_reason if method == "hybrid" else None
        ),
        entropy_grid_config=(
            asdict(entropy_config)
            if method in {"entropy_grid", "hybrid"}
            else None
        ),
        entropy_grid_geometry_mm=(
            {
                "socket_top_apothem": 500.0
                * float(spec["derived"]["socket_top_af_m"]),
                "head_outer_radius": 500.0 * float(spec["screw_head_diameter_m"]),
                "probe_circumradius": 1000.0
                * float(spec["key_af_m"])
                / math.sqrt(3.0),
                "capture_radius": 1000.0 * float(spec["derived"]["r_cap_top_m"]),
            }
            if method in {"entropy_grid", "hybrid"}
            else None
        ),
        hybrid_probe_gate_records=(
            hybrid_gate_records if method == "hybrid" else None
        ),
        hybrid_probe_config=(
            asdict(hybrid_config) if method == "hybrid" else None
        ),
        hybrid_residual_coverage_config=(
            asdict(hybrid_residual_config) if method == "hybrid" else None
        ),
        hybrid_fallback_nominal_reserve_s=(
            hybrid_fallback_nominal_reserve_s if method == "hybrid" else None
        ),
        hybrid_entry_fallback_feasible=(
            hybrid_entry_fallback_feasible if method == "hybrid" else None
        ),
        hybrid_terminal_execution_reserve_s=(
            hybrid_terminal_execution_reserve_s if method == "hybrid" else None
        ),
        hybrid_budget_switch_active_time_s=(
            hybrid_budget_switch_active_time_s if method == "hybrid" else None
        ),
        hybrid_budget_switch_reason=(
            hybrid_budget_switch_reason if method == "hybrid" else None
        ),
        hybrid_guarded_motion_failure_reason=(
            hybrid_guarded_motion_failure_reason
            if method == "hybrid"
            else None
        ),
        capture_direction_consistency_gate_enabled=(
            capture_direction_gate.initially_enabled
        ),
        capture_direction_gate_active_at_outcome=(
            capture_direction_gate.active
        ),
        capture_direction_gate_retired_for_global_fallback=(
            capture_direction_gate.retired_for_global_fallback
        ),
        capture_direction_gate_retired_active_time_s=(
            capture_direction_gate.retired_active_time_s
        ),
        capture_direction_gate_retirement_reason=(
            capture_direction_gate.retirement_reason
        ),
        capture_reverse_projection_tolerance_mm=(
            1000.0 * capture_reverse_tolerance_m
            if capture_direction_gate.initially_enabled
            else None
        ),
        terminal_attempts=terminal_attempts,
        terminal_rejection_count=terminal_rejection_count,
        terminal_attempt_diagnostics=terminal_attempt_diagnostics,
        terminal_entry_rebase_records=terminal_entry_rebase_records,
        terminal_axial_position_hold_used=terminal_axial_position_hold_used,
        transition_telemetry_records=transition_telemetry_records,
        oracle_endpoint_settle_records=(
            oracle_endpoint_settle_records if method == "oracle" else None
        ),
        post_release_requalification_records=(
            post_release_requalification_records if continue_to_bottom else None
        ),
        post_release_requalified=(
            post_release_requalified if continue_to_bottom and progress else None
        ),
        funnel_recovery_reasons=funnel_recovery_reasons,
        funnel_recovery_xy_rebases_mm=funnel_recovery_xy_rebases_mm,
        funnel_recovery_transition_records=funnel_recovery_transition_records,
        simulator_reset_count=0,
        direct_max_attempts=(direct_max_attempts if method == "direct" else 1),
        direct_attempts_completed=(
            direct_attempts_completed if method == "direct" else 0
        ),
        direct_successful_attempt=(
            direct_successful_attempt if method == "direct" else None
        ),
        direct_attempt_records=(
            direct_attempt_records if method == "direct" else None
        ),
        fingerprint_sha256_attempts=(
            fingerprint_sha256_attempts
            if method in {"learned", "hybrid", "direct"}
            else None
        ),
    )

    if writer is not None:
        assert last_frame is not None
        display_result = summary_result or result
        final = summary_frame(last_frame, display_result)
        for _ in range(max(1, int(round(SUMMARY_TIME_S * FPS)))):
            writer.append_data(final)
            frame_count += 1
        writer.close()
        result.frame_count = frame_count
    if renderer is not None:
        renderer.close()
    if trace_output_path is not None:
        trace_output_path = Path(trace_output_path)
        trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        assert trace_rows is not None and trace_rows
        fieldnames = list(trace_rows[0])
        with trace_output_path.open("w", encoding="utf-8", newline="") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(trace_rows)
    return result


def add_comparison_summary(
    frame: np.ndarray,
    proposed: RunResult,
    baseline: RunResult,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
    except OSError:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.rounded_rectangle(
        (290, 86, 990, 430),
        radius=8,
        fill=(0, 0, 0, 230),
        outline=(255, 230, 155, 230),
        width=2,
    )
    draw.text((320, 112), "SAME-START SINGLE-CASE RESULT", fill=(255, 235, 170, 255), font=title_font)
    time_delta = None
    time_reduction = None
    path_reduction = None
    if proposed.time_to_inner_wall_s is not None and baseline.time_to_inner_wall_s is not None:
        time_delta = baseline.time_to_inner_wall_s - proposed.time_to_inner_wall_s
        time_reduction = 100.0 * time_delta / baseline.time_to_inner_wall_s
    if baseline.actual_xy_path_mm > 1.0e-12:
        path_reduction = 100.0 * (
            baseline.actual_xy_path_mm - proposed.actual_xy_path_mm
        ) / baseline.actual_xy_path_mm
    proposed_time = (
        f"{proposed.time_to_inner_wall_s:.2f}s"
        if proposed.time_to_inner_wall_s is not None
        else f">{COMMON_ACTIVE_TIME_BUDGET_S:.0f}s (not reached)"
    )
    baseline_time = (
        f"{baseline.time_to_inner_wall_s:.2f}s"
        if baseline.time_to_inner_wall_s is not None
        else f">{COMMON_ACTIVE_TIME_BUDGET_S:.0f}s (not reached)"
    )
    rows = [
        f"both complete same capture gate: proposed={proposed.inner_wall_confirmed}, baseline={baseline.inner_wall_confirmed}",
        f"proposed time / XY path: {proposed_time} / {proposed.actual_xy_path_mm:.2f}mm",
        f"baseline time / XY path: {baseline_time} / {baseline.actual_xy_path_mm:.2f}mm",
        f"time saved: {time_delta:.2f}s ({time_reduction:.1f}% shorter)" if time_delta is not None and time_reduction is not None else "time saved: unavailable",
        f"XY path reduction: {path_reduction:.1f}%" if path_reduction is not None else "XY path reduction: unavailable",
        "includes full 15 deg probe cost for proposed",
        "single phase / assumed posterior / sensor-only handoff; offline CAD score",
        "diagnostic example only; general superiority requires a sweep",
    ]
    _draw_panel_text(
        draw,
        (320, 160),
        [(row, (235, 245, 255, 245)) for row in rows],
        font,
        640,
    )
    return np.asarray(image)


def compose_side_by_side(
    proposed_path: Path,
    baseline_path: Path,
    output_path: Path,
    proposed: RunResult,
    baseline: RunResult,
) -> tuple[int, np.ndarray]:
    left_reader = imageio.get_reader(proposed_path)
    right_reader = imageio.get_reader(baseline_path)
    result_summary_frames = max(1, int(round(SUMMARY_TIME_S * FPS)))
    left_frames = min(
        left_reader.count_frames(),
        max(1, proposed.frame_count - result_summary_frames),
    )
    right_frames = min(
        right_reader.count_frames(),
        max(1, baseline.frame_count - result_summary_frames),
    )
    count = max(left_frames, right_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    last_left: np.ndarray | None = None
    last_right: np.ndarray | None = None
    representative: np.ndarray | None = None
    try:
        for index in range(count):
            if index < left_frames:
                last_left = left_reader.get_data(index)
            if index < right_frames:
                last_right = right_reader.get_data(index)
            assert last_left is not None and last_right is not None
            left = Image.fromarray(last_left).resize((640, 480), Image.Resampling.LANCZOS)
            right = Image.fromarray(last_right).resize((640, 480), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1280, 520), (8, 10, 13))
            canvas.paste(left, (0, 40))
            canvas.paste(right, (640, 40))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            draw.text((18, 10), "PROPOSED | probe + posterior + local outward", fill=(90, 225, 255), font=font)
            draw.text((660, 10), "BASELINE | continuous spiral", fill=(255, 205, 105), font=font)
            fast_forwarding = (
                baseline.display_fast_forward_start_frame is not None
                and index >= baseline.display_fast_forward_start_frame
            )
            badge = (
                f"BASELINE DISPLAY {baseline.display_fast_forward_speed:.0f}x"
                if fast_forwarding
                else "DISPLAY 1x"
            )
            badge_color = (255, 205, 105) if fast_forwarding else (210, 225, 235)
            draw.rounded_rectangle(
                (1038, 5, 1272, 35),
                radius=6,
                fill=(20, 22, 26),
                outline=badge_color,
                width=1,
            )
            draw.text((1050, 10), badge, fill=badge_color, font=font)
            frame = np.asarray(canvas)
            writer.append_data(frame)
            representative = frame.copy()
        assert representative is not None
        summary = add_comparison_summary(representative, proposed, baseline)
        for _ in range(max(1, int(round(SUMMARY_TIME_S * FPS)))):
            writer.append_data(summary)
        count += max(1, int(round(SUMMARY_TIME_S * FPS)))
    finally:
        writer.close()
        left_reader.close()
        right_reader.close()
    return count, representative


def main(capture_yaw_assist: CaptureYawAssistConfig | None = None) -> None:
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    KEYFRAME_OUT.mkdir(parents=True, exist_ok=True)
    proposed_path = VIDEO_OUT / "m8_outer_partial_proposed_probe_to_inner_wall.mp4"
    baseline_path = VIDEO_OUT / "m8_outer_partial_source_outward_to_inner_wall.mp4"
    comparison_path = VIDEO_OUT / "m8_outer_partial_proposed_vs_source_side_by_side.mp4"

    yaw_assist = capture_yaw_assist or CaptureYawAssistConfig()
    proposed = run_method(
        "proposed",
        proposed_path,
        capture_yaw_assist=yaw_assist,
    )
    baseline = run_method(
        "baseline",
        baseline_path,
        fast_forward_after_active_s=proposed.time_to_inner_wall_s,
        fast_forward_speed=BASELINE_FAST_FORWARD_SPEED,
    )
    comparison_frames, representative = compose_side_by_side(
        proposed_path,
        baseline_path,
        comparison_path,
        proposed,
        baseline,
    )

    proposed_reader = imageio.get_reader(proposed_path)
    baseline_reader = imageio.get_reader(baseline_path)
    proposed_key = proposed_reader.get_data(max(0, proposed.frame_count - int(SUMMARY_TIME_S * FPS) - 1))
    baseline_key = baseline_reader.get_data(max(0, baseline.frame_count - int(SUMMARY_TIME_S * FPS) - 1))
    proposed_reader.close()
    baseline_reader.close()
    imageio.imwrite(KEYFRAME_OUT / "m8_outer_partial_proposed_keyframe.png", proposed_key)
    imageio.imwrite(KEYFRAME_OUT / "m8_outer_partial_baseline_keyframe.png", baseline_key)
    comparison_reader = imageio.get_reader(comparison_path)
    comparison_key = comparison_reader.get_data(comparison_reader.count_frames() - 1)
    comparison_reader.close()
    imageio.imwrite(
        KEYFRAME_OUT / "m8_outer_partial_comparison_keyframe.png",
        comparison_key,
    )

    time_saved_s = None
    time_reduction_pct = None
    if proposed.time_to_inner_wall_s is not None and baseline.time_to_inner_wall_s is not None:
        time_saved_s = baseline.time_to_inner_wall_s - proposed.time_to_inner_wall_s
        time_reduction_pct = 100.0 * time_saved_s / baseline.time_to_inner_wall_s
    wear_reduction_pct = None
    if baseline.wear_proxy_mj > 1.0e-12:
        wear_reduction_pct = 100.0 * (
            baseline.wear_proxy_mj - proposed.wear_proxy_mj
        ) / baseline.wear_proxy_mj
    path_reduction_pct = None
    if baseline.actual_xy_path_mm > 1.0e-12:
        path_reduction_pct = 100.0 * (
            baseline.actual_xy_path_mm - proposed.actual_xy_path_mm
        ) / baseline.actual_xy_path_mm

    payload = {
        "claim_boundary": (
            "Single M8 MuJoCo stress case with an assumed posterior, sensor-only "
            "safe-progress handoff, and offline CAD scoring. It compares search policies under matched simulation "
            "conditions; it is not learned, hardware-validated, phase-averaged, or an exact "
            "reproduction of the source paper."
        ),
        "fairness_contract": {
            "same_initial_xy_mm": (1000.0 * INITIAL_XY_M).tolist(),
            "same_target_down_force_n": TARGET_DOWN_FORCE_N,
            "same_spiral_pitch_mm_per_rev": 1000.0 * SPIRAL_PITCH_M,
            "same_xy_speed_mm_s": 1000.0 * COMMON_XY_SPEED_M_S,
            "same_active_time_budget_s": COMMON_ACTIVE_TIME_BUDGET_S,
            "same_endpoint": (
                "below-rim sensor/depth/wrench gate + dwell; M8 geometry only for offline scoring"
            ),
            "proposed_probe_cost_included": True,
            "baseline_max_radius_mm": 1000.0 * BASELINE_MAX_RADIUS_M,
            "baseline_phase_rad": BASELINE_PHASE_RAD,
            "capture_yaw_assist": asdict(yaw_assist),
        },
        "source_paper_boundary": {
            "matched": "outward increasing-radius spiral geometry",
            "not_matched": "paper used about 5 N, 20 rpm tool rotation, and motor torque coupling verification",
            "reason": "3 N and no motor rotation isolate the search policy under the current project controller",
        },
        "initial_condition_boundary": (
            "x=4.00 mm is an outer-partial stress case and is outside the source paper's "
            "reported 1-2 mm visual positioning error; do not generalize from this one video."
        ),
        "proposed": asdict(proposed),
        "baseline": asdict(baseline),
        "comparison": {
            "time_saved_s": time_saved_s,
            "time_reduction_pct": time_reduction_pct,
            "xy_path_reduction_pct": path_reduction_pct,
            "wear_proxy_reduction_pct": wear_reduction_pct,
            "wear_comparison_boundary": (
                "Both rollouts reach the same simulation capture evaluator; this remains a proxy, "
                "not a hardware wear measurement."
            ),
            "display_schedule": {
                "before_proposed_completion": "1x synchronized playback",
                "after_proposed_completion": f"baseline {BASELINE_FAST_FORWARD_SPEED:.0f}x; proposed result held",
                "clocks": "overlay values remain physical simulation task time",
                "proposed_upper_right": (
                    "live measured Xz/Mx/My polar fingerprint during the conical scan; "
                    "completed fingerprint held during transport and local search"
                ),
                "render_backend": os.environ.get("MUJOCO_GL", "default"),
            },
            "comparison_video": str(comparison_path.relative_to(ROOT)),
            "comparison_frames": comparison_frames,
        },
        "keyframes": {
            "proposed": str((KEYFRAME_OUT / "m8_outer_partial_proposed_keyframe.png").relative_to(ROOT)),
            "baseline": str((KEYFRAME_OUT / "m8_outer_partial_baseline_keyframe.png").relative_to(ROOT)),
            "comparison": str((KEYFRAME_OUT / "m8_outer_partial_comparison_keyframe.png").relative_to(ROOT)),
        },
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the M8 proposed-versus-source search comparison."
    )
    parser.add_argument(
        "--capture-yaw-assist",
        action="store_true",
        help="enable bounded post-progress micro-yaw clocking for the proposed method",
    )
    parser.add_argument("--capture-yaw-amplitude-deg", type=float, default=0.5)
    parser.add_argument("--capture-yaw-frequency-hz", type=float, default=1.5)
    parser.add_argument("--capture-yaw-ramp-time-s", type=float, default=0.25)
    parser.add_argument(
        "--capture-yaw-soft-tau-z-limit-mnm",
        type=float,
        default=10.0,
    )
    args = parser.parse_args()
    main(
        CaptureYawAssistConfig(
            enabled=args.capture_yaw_assist,
            amplitude_deg=args.capture_yaw_amplitude_deg,
            frequency_hz=args.capture_yaw_frequency_hz,
            ramp_time_s=args.capture_yaw_ramp_time_s,
            soft_tau_z_limit_mnm=args.capture_yaw_soft_tau_z_limit_mnm,
        )
    )
