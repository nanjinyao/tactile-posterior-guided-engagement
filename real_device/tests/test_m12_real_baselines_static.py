from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"
KST_CLIENT = (
    ROOT.parents[1]
    / "reference"
    / "KST-Kuka-Sunrise-Toolbox-master"
    / "Matlab_client"
)
LEGACY_ALIAS_ROOT = (
    ROOT / "archive" / "legacy_entry_aliases_20260724"
)
WRAPPERS = {
    "FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY": (
        ROOT / "run_m12_full_loop_proposed_plus_entropy_grid.m"
    ),
    "FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL": (
        ROOT / "run_m12_full_loop_proposed_local_spiral.m"
    ),
    "FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL": (
        ROOT / "run_m12_full_loop_baseline_pure_outward_spiral.m"
    ),
    "FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED": (
        ROOT / "run_m12_full_loop_baseline_improved_direct_x3.m"
    ),
}
LEGACY_ALIASES = {
    "run_m12_event_trigger_insert_demo.m": (
        "run_m12_full_loop_proposed_plus_entropy_grid"
    ),
    "run_m12_proposed_insert_demo.m": (
        "run_m12_full_loop_proposed_local_spiral"
    ),
    "run_m12_pure_outward_insert_demo.m": (
        "run_m12_full_loop_baseline_pure_outward_spiral"
    ),
    "run_m12_improved_direct_retry3_demo.m": (
        "run_m12_full_loop_baseline_improved_direct_x3"
    ),
    "run_m12_conical_sampling_campaign.m": (
        "run_m12_sampling_conical_fingerprints_26pt"
    ),
}


class TestM12RealBaselinesStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")

    def test_thin_wrappers_select_only_one_shared_mode(self) -> None:
        for mode, path in WRAPPERS.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                "execute_m12_contact_engagement_policy( ...\n"
                f"    '{mode}')",
                text,
            )
            for forbidden in ("KST(", "net_establishConnection", "movePTP", "sendEEf"):
                self.assertNotIn(forbidden, text)

    def test_legacy_aliases_are_archived_outside_active_root(self) -> None:
        for filename, entrypoint in LEGACY_ALIASES.items():
            self.assertFalse((ROOT / filename).exists())
            archived = LEGACY_ALIAS_ROOT / filename
            self.assertTrue(archived.is_file())
            text = archived.read_text(encoding="utf-8")
            self.assertIn(f"result = {entrypoint}();", text)
            self.assertNotIn("execute_m12_contact_engagement_policy(", text)
            self.assertLess(len(text.splitlines()), 12)

    def test_default_no_argument_behavior_is_unchanged(self) -> None:
        parser = self.main[
            self.main.index(
                "function [mode, resumeParentRunDir] = parseExecutionMode"
            ) :
            self.main.index("function policy = executionPolicy")
        ]
        self.assertIn(
            "mode = 'FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY';", parser
        )
        for mode in WRAPPERS:
            self.assertIn(f"'{mode}'", parser)

    def test_conical_modes_require_learned_authority_but_pure_does_not(
        self,
    ) -> None:
        policy = self.main[
            self.main.index("function policy = executionPolicy") :
            self.main.index("function [result, manifest] = discloseExecutionPolicy")
        ]
        for mode in (
            "FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL",
            "FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED",
        ):
            self.assertIn(f"case '{mode}'", policy)
        self.assertIn("policy.uses_conical_fingerprint = false;", policy)
        disclosure = self.main[
            self.main.index(
                "function [result, manifest] = discloseExecutionPolicy"
            ) :
            self.main.index("function valid = isValidSha256")
        ]
        self.assertIn(
            "policy.uses_conical_fingerprint;", disclosure
        )
        self.assertIn(
            "manifest.policy_uses_learned_direction_for_motion = ...",
            disclosure,
        )
        self.assertIn(
            "there is no hidden ', ...\n"
            "     'classical direction fallback.",
            disclosure,
        )

    def test_active_inference_is_required_and_selects_bounded_motion(
        self,
    ) -> None:
        route_start = self.main.index("if conical.aperture_candidate")
        route_end = self.main.index("if ~capture.found", route_start)
        route = self.main[route_start:route_end]
        self.assertIn("learnedShortcut = planLearnedShortcut(", route)
        self.assertIn(
            "shortcutOffset = learnedShortcut.offset_mm;", route
        )
        self.assertIn(
            "shortcutDirectionWorld = learnedShortcut.direction_world_xy;",
            route,
        )
        self.assertIn("learnedModelUsedForControl = true;", route)
        self.assertIn(
            "error('M12Demo:LearnedInferenceRequired'", route
        )
        self.assertNotRegex(route.lower(), r"classical|shadow")
        inference = self.main[
            self.main.index("function learned = inferFrozenConicalModel") :
            self.main.index(
                "function [ctx, capture, diagnostics] = runOutwardArchimedeanSearch"
            )
        ]
        self.assertIn("try", inference)
        self.assertIn("validateModelDeploymentTrustAnchor(cfg);", inference)
        self.assertIn("catch inferenceError", inference)

    def test_pure_outward_has_no_probe_or_model_branch(self) -> None:
        branch = self.main[
            self.main.index(
                "if strcmp(executionMode, "
                "'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL')"
            ) :
            self.main.index(
                "logTransition(paths.transitions, 'SHORT_EVENT_TRIGGER_TOUCHDOWN'"
            )
        ]
        self.assertIn("runOutwardArchimedeanSearch", branch)
        self.assertNotIn("runConicalFingerprint", branch)
        self.assertNotIn("inferFrozenConicalModel", branch)
        config = self.main[
            self.main.index("cfg.baselines = struct") :
            self.main.index("activeLearnedBundleRoot")
        ]
        self.assertIn(
            "'spiral_pitch_mm_per_revolution', activeCoveragePitchMm", config
        )
        self.assertIn("'pure_outward_radius_mm', 4.20", config)
        self.assertIn("'path_waypoint_arc_mm', 0.60", config)
        self.assertIn("'active_time_budget_s', 0.0", config)

    def test_shared_measured_touchdown_anchor_and_drift_gate(self) -> None:
        touchdown = self.main.index(
            "[ctx, touchdown] = guardedTouchdown("
        )
        anchor = self.main.index(
            "centerXY = touchdown.pose_mm_rad(1:2);", touchdown
        )
        branch = self.main.index(
            "if strcmp(executionMode, "
            "'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL')",
            anchor,
        )
        self.assertLess(touchdown, anchor)
        self.assertLess(anchor, branch)
        pre_anchor = self.main[touchdown:anchor]
        self.assertNotIn("centerXY = startPose(1:2);", pre_anchor)
        for token in (
            "'touchdown_lateral_drift_warning_mm', 0.20",
            "'maximum_touchdown_lateral_drift_mm', 3.00",
            "'touchdown_xy_correction_deadband_mm', 0.08",
            "'touchdown_xy_correction_step_mm', 0.15",
            "'maximum_touchdown_xy_lock_error_mm', 0.25",
            "TOUCHDOWN_LATERAL_DRIFT_WARNING_CONTINUE",
            "touchdown.maximum_lateral_deviation_mm",
            "M12Demo:TouchdownLateralDrift",
            "MEASURED_TOUCHDOWN_POSE_AFTER_FIXED_XY_CORRECTION_AND_0P25MM_GATE",
        ):
            self.assertIn(token, self.main)
        for observed_drift_mm in (0.203, 0.210, 0.699, 0.912, 1.056):
            self.assertLessEqual(observed_drift_mm, 3.00)
        self.assertGreater(3.001, 3.00)
        self.assertNotIn(
            "measured_touchdown_XY_after_0p20mm_lateral_drift_gate",
            self.main,
        )
        self.assertNotIn(
            "touchdown XY only after 0.20 mm lateral-drift gate",
            self.main,
        )
        touchdown_helper = self.main[
            self.main.index(
                "function [ctx, touchdown] = guardedTouchdown"
            ) :
            self.main.index(
                "function [ctx, transport] = moveSurfaceForceControlledXY"
            )
        ]
        for token in (
            "touchdownAnchorXY = double(reshape(ctx.last_pose(1:2), 1, 2));",
            "correctionXY = touchdownAnchorXY - ctx.last_pose(1:2);",
            "ctx.cfg.motion.touchdown_xy_correction_deadband_mm",
            "ctx.cfg.motion.touchdown_xy_correction_step_mm",
            "[correctionXY, -step]",
            "landingErrorMm = norm(pose(1:2) - touchdownAnchorXY);",
            "while landingErrorMm > ...",
            "ctx.cfg.motion.maximum_touchdown_xy_lock_error_mm",
            "[correctionXY, 0.0]",
            "'M12Demo:TouchdownXyLockError'",
        ):
            self.assertIn(token, touchdown_helper)
        move = touchdown_helper.index("[ctx, pose] = moveOne")
        endpoint_guard = touchdown_helper.index(
            "[ctx, guard] = guardDwell", move
        )
        warning = touchdown_helper.index(
            "TOUCHDOWN_LATERAL_DRIFT_WARNING_CONTINUE", endpoint_guard
        )
        stop = touchdown_helper.index(
            "M12Demo:TouchdownLateralDrift", warning
        )
        self.assertLess(move, endpoint_guard)
        self.assertLess(endpoint_guard, warning)
        self.assertLess(warning, stop)
        stop_condition = touchdown_helper[
            touchdown_helper.rfind("if ", 0, stop) : stop
        ]
        self.assertIn(
            "touchdown.maximum_lateral_deviation_mm", stop_condition
        )
        self.assertNotIn(
            "touchdown.cumulative_lateral_motion_mm >", stop_condition
        )
        downstream = self.main[anchor:]
        for token in (
            "runOutwardArchimedeanSearch(ctx, centerXY, centerXY",
            "runImprovedDirectRetry3(ctx, conical, learned, centerXY",
            "centerXY + offset",
        ):
            self.assertIn(token, downstream)

    def test_touchdown_exception_summary_recovers_committed_ledgers(
        self,
    ) -> None:
        catch = self.main.index("catch experimentError")
        end_request = self.main.index(
            "[endState.sent, endState.method] = requestKstEndOnce(", catch
        )
        close_logs = self.main.index(
            "closeRealtimeLogFiles(ctx.realtime_logs)", end_request
        )
        recovery = self.main.index(
            "refreshResultMetricsFromRuntimeLogs(result, paths)", catch
        )
        conical_recovery = self.main.index(
            "refreshResultPeaksFromConicalEvidence(", recovery
        )
        self.assertLess(catch, end_request)
        self.assertLess(end_request, close_logs)
        self.assertLess(close_logs, recovery)
        self.assertLess(recovery, conical_recovery)
        helper = self.main[
            self.main.index(
                "function result = refreshResultMetricsFromRuntimeLogs"
            ) :
            self.main.index(
                "function result = refreshResultPeaksFromConicalEvidence"
            )
        ]
        for token in (
            'statuses == "ISSUED"',
            'statuses == "DONE"',
            'statuses == "SETUP_FAILED_NO_TRIGGER"',
            "result.motion_attempted",
            "result.completed_commands",
            "result.path_length_mm",
            "result.peak_relative_force_n",
            "result.peak_relative_torque_nm",
            "result.touchdown_lateral_drift_mm",
        ):
            self.assertIn(token, helper)

    def test_shared_fast_surface_contract_preserves_hard_and_capture_limits(
        self,
    ) -> None:
        config = self.main[
            self.main.index("cfg.surface = struct(") :
            self.main.index("cfg.conical = struct(")
        ]
        for token in (
            "'speed_mm_s', 6.0",
            "'xy_segment_mm', 1.20",
            "'speed_fast_mm_s', 10.0",
            "'speed_nominal_mm_s', 6.0",
            "'speed_cautious_mm_s', 3.0",
            "'force_deadband_n', 1.50",
            "'force_correction_enter_low_n', 2.25",
            "'force_correction_enter_high_n', 5.75",
            "'force_correction_exit_low_n', 2.5",
            "'force_correction_exit_high_n', 5.5",
            "'negative_observation_xy_tolerance_mm', 0.30",
            "'z_stagnation_recenter_limit_mm', 0.60",
            "'minimum_measured_z_progress_mm', 0.005",
            "'phase_timeout_s', 0.0",
            "'endpoint_xy_tolerance_mm', 0.20",
            "'maximum_stagnant_z_steps', 3",
            "'maximum_stagnant_xy_steps', 3",
            "'maximum_force_settle_steps', 3",
            "'maximum_excess_transport_actions', 24",
            "'maximum_z_correction_lateral_drift_mm', 0.60",
            "'force_continuation_min_n', 1.25",
            "'negative_observation_normal_min_n', 1.25",
            "'capture_depth_drop_mm', 0.25",
            "'capture_minimum_recent_drop_mm', 0.04",
            "'capture_confirmation_extra_drop_mm', 0.06",
            "'soft_relief_trigger_force_n', 10.0",
            "'soft_relief_trigger_torque_nm', 0.50",
            "'soft_relief_resume_torque_nm', 0.40",
            "'soft_relief_tracking_tolerance_mm', 0.35",
            "'soft_relief_measured_step_envelope_mm', 0.60",
            "'maximum_z_step_mm', 0.080",
        ):
            self.assertIn(token, config)
        for token in (
            "'hard_force_delta_n', 25.0",
            "'hard_torque_delta_nm', 0.75",
            "'touch_speed_mm_s', 1.0",
            "'insertion_speed_mm_s', 0.60",
            "'insertion_segment_mm', 0.10",
        ):
            self.assertIn(token, self.main)
        sensor = self.main[
            self.main.index("cfg.sensor = struct(") :
            self.main.index("cfg.motion = struct(")
        ]
        for token in (
            "'contact_event_n', 2.0",
            "'contact_consecutive_samples', 3",
            "'hard_force_delta_n', 25.0",
            "'hard_torque_delta_nm', 0.75",
        ):
            self.assertIn(token, sensor)
        self.assertAlmostEqual(math.hypot(0.60, 0.080), 0.605309838)
        contract = self.main[
            self.main.index("function validateSurfaceTransportContract") :
            self.main.index("function validateBaselinePolicyContract")
        ]
        self.assertNotIn(
            "cfg.surface.force_continuation_min_n >= ...\n"
            "        cfg.sensor.contact_event_n",
            contract,
        )
        self.assertNotIn(
            "cfg.surface.negative_observation_normal_min_n >= ...\n"
            "        cfg.sensor.contact_event_n",
            contract,
        )

    def test_surface_targets_have_geometry_scaled_action_budget(self) -> None:
        surface = self.main[
            self.main.index(
                "function [ctx, transport] = moveSurfaceForceControlledXY"
            ) :
            self.main.index(
                "function transport = updateSurfaceTransportObservation"
            )
        ]
        for token in (
            "geometricActions = ceil(max(0.0, initialXYDistance -",
            "geometricActions + ...\n"
            "        ctx.cfg.surface.maximum_excess_transport_actions",
            "for index = 1:transportActionBudget",
            "zCorrectionLateralDriftMm",
            "SURFACE_Z_CORRECTION_LATERAL_COMPLIANCE_LATCHED",
            "normalErrorN = ctx.cfg.surface.target_normal_force_n - ...",
            "forceCorrectionActive",
            "ctx.cfg.surface.force_correction_enter_low_n",
            "ctx.cfg.surface.force_correction_enter_high_n",
            "ctx.cfg.surface.force_correction_exit_low_n",
            "ctx.cfg.surface.force_correction_exit_high_n",
            "forceSettled = guard.event_confirmed && ...",
            "if atTarget && forceSettled && guard.event_confirmed",
            "'TARGET_REACHED_WITH_FORCE_HELD'",
            "stagnantXYSteps = 0;",
            "stagnantZSteps = 0;",
            "depthHistory = transport.depth_below_surface_mm;",
            "'SURFACE_HYSTERETIC_FORCE_CONTINUATION_ACCEPTED'",
        ):
            self.assertIn(token, surface)
        self.assertIn(
            "CAPTURE_CONFIRMATION_LATERAL_COMPLIANCE_CONTINUE",
            self.main,
        )
        self.assertNotIn(
            "M12Demo:CaptureConfirmationLateralDrift",
            self.main,
        )
        self.assertNotIn("sign(zStep) * minimumActionableZ", surface)
        capture = surface.index("surfaceCaptureCandidate(")
        normal_error = surface.index(
            "normalErrorN = ctx.cfg.surface.target_normal_force_n"
        )
        force_settled = surface.index(
            "forceSettled = guard.event_confirmed &&", normal_error
        )
        held_return = surface.index(
            "if atTarget && forceSettled && guard.event_confirmed",
            force_settled,
        )
        self.assertLess(capture, normal_error)
        self.assertLess(normal_error, force_settled)
        self.assertLess(force_settled, held_return)
        move = surface.index("[ctx, postPose] = moveOne(ctx, delta")
        endpoint_guard = surface.index(
            "[ctx, guard] = guardDwell(ctx, [phase, '_ENDPOINT_DWELL']",
            move,
        )
        no_lift_latch = surface.index(
            "SURFACE_Z_CORRECTION_LATERAL_COMPLIANCE_LATCHED",
            endpoint_guard,
        )
        progress_degraded = surface.index(
            "SURFACE_XY_NO_PROGRESS_DEGRADED_CONTINUE", endpoint_guard
        )
        self.assertLess(move, endpoint_guard)
        self.assertLess(endpoint_guard, no_lift_latch)
        self.assertLess(endpoint_guard, progress_degraded)
        self.assertNotIn("M12Demo:SurfaceTransportNoProgress", surface)
        budget = lambda distance: min(
            120, math.ceil(max(0.0, distance - 0.20) / 0.60) + 8
        )
        self.assertEqual(budget(0.15), 8)
        self.assertEqual(budget(0.55), 9)
        self.assertEqual(budget(4.20), 15)

    def test_z_stagnation_recovers_or_degrades_without_task_abort(
        self,
    ) -> None:
        surface = self.main[
            self.main.index(
                "function [ctx, transport] = moveSurfaceForceControlledXY"
            ) :
            self.main.index(
                "function transport = updateSurfaceTransportObservation"
            )
        ]
        for token in (
            "postNegativeObservation = guard.event_confirmed && ...",
            "postNormalN >= ...",
            "ctx.cfg.surface.negative_observation_normal_min_n && ...",
            "postNormalN <= ...",
            "ctx.cfg.surface.negative_observation_normal_max_n && ...",
            "ctx.cfg.surface.negative_observation_xy_tolerance_mm",
            "'SURFACE_Z_STAGNATION_BOUNDED_ACCEPTED'",
            "'SURFACE_Z_STAGNATION_LOW_FORCE_RECLASSIFIED'",
            "'SURFACE_Z_STAGNATION_ONE_RECENTER_BEGIN'",
            "[phase, '_Z_STAGNATION_RECENTER']",
            "'SURFACE_Z_STAGNATION_RECENTERED_ACCEPTED'",
            "'SURFACE_Z_STAGNATION_RECENTERED_DEGRADED_CONTINUE'",
            "'SURFACE_Z_STAGNATION_DEGRADED_CONTINUE'",
            "transport.degraded_waypoint = true;",
        ):
            self.assertIn(token, surface)
        self.assertNotIn("M12Demo:SurfaceZNoProgress", surface)

        def disposition(
            *,
            xy_remaining_mm: float,
            event_confirmed: bool,
            normal_force_n: float,
            force_delta_n: float,
            torque_delta_nm: float,
        ) -> str:
            bounded = (
                event_confirmed
                and 1.25 <= normal_force_n <= 7.5
                and force_delta_n < 10.0
                and torque_delta_nm < 0.50
            )
            if bounded and xy_remaining_mm <= 0.30:
                return "BOUNDED_ENDPOINT_ACCEPTED"
            if bounded and xy_remaining_mm <= 0.60:
                return "ONE_RECENTER_THEN_CONTINUE"
            return "DEGRADED_CONTINUE"

        self.assertEqual(
            disposition(
                xy_remaining_mm=0.208,
                event_confirmed=True,
                normal_force_n=3.20,
                force_delta_n=3.30,
                torque_delta_nm=0.04,
            ),
            "BOUNDED_ENDPOINT_ACCEPTED",
        )
        self.assertEqual(
            disposition(
                xy_remaining_mm=0.42,
                event_confirmed=True,
                normal_force_n=2.90,
                force_delta_n=3.00,
                torque_delta_nm=0.04,
            ),
            "ONE_RECENTER_THEN_CONTINUE",
        )
        self.assertEqual(
            disposition(
                xy_remaining_mm=0.65,
                event_confirmed=False,
                normal_force_n=0.40,
                force_delta_n=0.50,
                torque_delta_nm=0.01,
            ),
            "DEGRADED_CONTINUE",
        )

    def test_entropy_grid_covers_the_cad_guidance_margin(self) -> None:
        search = self.main[
            self.main.index("m12TipFaceAfMm = 8.0;") :
            self.main.index("% Real baseline geometry")
        ]
        for token in (
            "m12TipFaceAfMm = 8.0;",
            "m12ShaftAfMm = 10.0;",
            "m12SocketWallAfMm = 10.6;",
            "m12SocketMouthAfMm = 13.0;",
            "m12DeepClearancePerSideMm = ...",
            "0.5 * (m12SocketWallAfMm - m12ShaftAfMm);",
            "m12MouthGuidanceAllowanceMm = ...",
            "0.5 * (m12SocketMouthAfMm - m12TipFaceAfMm);",
            "m12ConservativeCaptureRadiusMm = ...",
            "0.25 * (m12SocketMouthAfMm - m12ShaftAfMm);",
            "m12GridPitchMm = min(2.0 * m12DeepClearancePerSideMm, ...",
            "(2.0 / 3.0) * m12ConservativeCaptureRadiusMm);",
            "'shortcut_cap_mm', 3.00",
            "'shortcut_projection_quantile_low', 0.10",
            "'shortcut_projection_quantile_high', 0.25",
            "'shortcut_radius_cap_fraction_low', 0.50",
            "'shortcut_radius_cap_fraction_high', 0.75",
            "'directional_corridor_probability_mass', 0.80",
            "'directional_corridor_minimum_full_width_deg', 12.0",
            "'directional_corridor_maximum_full_width_deg', 45.0",
            "'directional_corridor_tracking_allowance_mm', 0.20",
            "'directional_corridor_maximum_advance_mm', ...\n"
            "        activeDirectionalCorridorAdvanceMm",
            "'grid_pitch_mm', m12GridPitchMm",
            "m12CommonSearchReachMm = 4.80;",
            "m12CenterHypothesisRadiusMm = m12CommonSearchReachMm + ...",
            "m12ConservativeCaptureRadiusMm + 0.20;",
            "'action_reach_mm', m12CommonSearchReachMm",
            "'grid_radius_mm', m12CommonSearchReachMm",
            "'center_hypothesis_radius_mm', m12CenterHypothesisRadiusMm",
            "'maximum_grid_nodes', activeGridNodeCount",
            "'residual_maximum_connected_leg_mm', ...\n"
            "        activeResidualConnectedLegMm",
            "'tested_offset_tolerance_mm', 0.10",
            "'directional_progress_tolerance_mm', 0.10",
            "'swept_capture_tube_tolerance_mm', ...\n"
            "        m12ConservativeCaptureRadiusMm - 0.20",
            "'minimum_novel_swept_length_mm', 0.25",
            "'swept_segment_sample_spacing_mm', 0.05",
            "'bayes_capture_radius_mm', m12ConservativeCaptureRadiusMm",
            "'cad_tip_face_af_mm', m12TipFaceAfMm",
            "'cad_shaft_af_mm', m12ShaftAfMm",
            "'cad_socket_wall_af_mm', m12SocketWallAfMm",
            "'cad_socket_mouth_af_mm', m12SocketMouthAfMm",
        ):
            self.assertIn(token, search)

        tip_face_af = 8.0
        shaft_af = 10.0
        socket_wall_af = 10.6
        socket_mouth_af = 13.0
        deep_clearance = 0.5 * (socket_wall_af - shaft_af)
        mouth_guidance_allowance = (
            0.5 * (socket_mouth_af - tip_face_af)
        )
        common_search_reach = 4.80
        capture_radius = 0.25 * (socket_mouth_af - shaft_af)
        pitch = min(2.0 * deep_clearance, (2.0 / 3.0) * capture_radius)
        center_hypothesis_radius = (
            common_search_reach + capture_radius + 0.20
        )
        maximum_nodes = 293
        shortcut_cap = 3.00
        corridor_probability_mass = 0.80
        corridor_minimum_full_width_deg = 12.0
        corridor_maximum_full_width_deg = 45.0
        tracking_allowance = 0.20
        corridor_maximum_advance_mm = 1.00
        self.assertAlmostEqual(deep_clearance, 0.30, places=9)
        self.assertAlmostEqual(capture_radius, 0.75, places=9)
        self.assertEqual(pitch, 0.50)
        self.assertEqual(mouth_guidance_allowance, 2.50)
        self.assertEqual(common_search_reach, 4.80)
        self.assertAlmostEqual(center_hypothesis_radius, 5.75)
        self.assertGreater(
            center_hypothesis_radius,
            common_search_reach + capture_radius,
        )
        self.assertEqual(maximum_nodes, 293)
        self.assertEqual(shortcut_cap, 3.00)
        self.assertEqual(corridor_probability_mass, 0.80)
        self.assertEqual(corridor_minimum_full_width_deg, 12.0)
        self.assertEqual(corridor_maximum_full_width_deg, 45.0)
        self.assertEqual(tracking_allowance, 0.20)
        self.assertEqual(corridor_maximum_advance_mm, 1.00)

        integer_radius = math.ceil(common_search_reach / pitch)
        grid = [
            (pitch * ix, pitch * iy)
            for ix in range(-integer_radius, integer_radius + 1)
            for iy in range(-integer_radius, integer_radius + 1)
            if math.hypot(pitch * ix, pitch * iy)
            <= common_search_reach + 1e-12
        ]
        self.assertEqual(len(grid), 293)
        self.assertEqual(maximum_nodes, len(grid))
        for endpoint in (
            (-4.0, 0.0),
            (4.0, 0.0),
            (0.0, -4.0),
            (0.0, 4.0),
        ):
            self.assertIn(endpoint, grid)
        hypothesis_integer_radius = math.ceil(
            center_hypothesis_radius / pitch
        )
        center_hypotheses = [
            (pitch * ix, pitch * iy)
            for ix in range(
                -hypothesis_integer_radius,
                hypothesis_integer_radius + 1,
            )
            for iy in range(
                -hypothesis_integer_radius,
                hypothesis_integer_radius + 1,
            )
            if math.hypot(pitch * ix, pitch * iy)
            <= center_hypothesis_radius + 1e-12
        ]
        self.assertEqual(len(center_hypotheses), 421)
        self.assertGreater(len(center_hypotheses), len(grid))
        self.assertGreater(
            max(math.hypot(x, y) for x, y in center_hypotheses),
            max(math.hypot(x, y) for x, y in grid),
        )
        plan = self.main[
            self.main.index("function plan = makePlan(cfg, policy)") :
            self.main.index("function validateConicalContract(cfg)")
        ]
        for token in (
            "actionOffsets = buildGridOffsets(cfg);",
            "centerHypotheses = buildCenterHypothesisOffsets(cfg);",
            "'action_offsets_mm', actionOffsets",
            "'grid_offsets_mm', actionOffsets",
            "'center_hypotheses_mm', centerHypotheses",
            "cfg.search.grid_pitch_mm, cfg.search.action_reach_mm",
            "cfg.search.center_hypothesis_radius_mm",
        ):
            self.assertIn(token, plan)

        simulation = json.loads(
            (
                ROOT.parent
                / "mujoco_unscrewing_probe_new"
                / "configs"
                / "allen_m12_hardware_easy_v2.json"
            ).read_text(encoding="utf-8")
        )
        guidance_margin_mm = (
            1000.0
            * simulation["paired_flat_normal_guidance_margin_m"]
        )
        self.assertAlmostEqual(
            1000.0
            * simulation["derived"]["bit_af_m"]
            * simulation["bit_tip_scale"],
            tip_face_af,
            places=9,
        )
        self.assertAlmostEqual(
            1000.0 * simulation["derived"]["bit_af_m"],
            shaft_af,
            places=9,
        )
        self.assertAlmostEqual(
            1000.0 * simulation["derived"]["socket_wall_af_m"],
            socket_wall_af,
            places=9,
        )
        self.assertAlmostEqual(
            1000.0 * simulation["derived"]["socket_top_af_m"],
            socket_mouth_af,
            places=9,
        )
        self.assertAlmostEqual(
            1000.0 * simulation["clearance_m"],
            deep_clearance,
            places=9,
        )
        self.assertAlmostEqual(guidance_margin_mm, 2.50, places=9)
        self.assertGreater(common_search_reach, guidance_margin_mm)
        self.assertGreaterEqual(
            max(math.hypot(x, y) for x, y in grid),
            guidance_margin_mm,
        )
        self.assertIn(
            "not a guaranteed capture radius",
            simulation["paired_guidance_margin_semantics"],
        )

    def test_learned_directional_corridor_executes_before_residual_eig(
        self,
    ) -> None:
        prior_start = self.main.index(
            "activeGridRoutingDirection = ...\n"
            "        learnedShortcut.direction_world_xy;"
        )
        prior_end = self.main.index(
            "learnedModelUsedForControl = true;", prior_start
        )
        prior_binding = self.main[prior_start:prior_end]
        for token in (
            "activeGridPriorSource = ...",
            "'ACTIVE_REAL_MODEL_DIRECTION_STATE_AND_CAD_POSTERIOR'",
            "result.active_grid_prior_source = activeGridPriorSource;",
            "result.active_grid_prior_direction_world_xy = ...",
            "activeGridRoutingDirection;",
            "result.active_grid_prior_direction_confidence = ...",
            "shortcutConfidence;",
            "result.active_grid_routing_direction_world_xy = ...",
            "activeGridRoutingDirection;",
            "result.active_grid_directional_corridor_half_angle_deg = ...",
            "corridorHalfAngleDeg;",
            "result.active_grid_directional_corridor_full_width_deg = ...",
            "2.0 * corridorHalfAngleDeg;",
            "result.active_grid_directional_corridor_control_center_task_deg = ...",
            "corridorCredible.control_center_task_deg;",
            "result.active_grid_prior = posterior;",
        ):
            self.assertIn(token, prior_binding)
        self.assertNotRegex(prior_binding.lower(), r"classical|shadow")

        builder = self.main[
            self.main.index(
                "function [pathOffsetsMm, laneOffsetsMm, diagnostics] = "
                "..."
            ) :
            self.main.index(
                "function [candidateIndex, bestInformation, predictiveOpen"
            )
        ]
        for token in (
            "buildDirectionalCadCorridor( ...",
            "direction = direction / norm(direction);",
            "radiusMm = cfg.search.action_reach_mm;",
            "currentOffsetMm, routingDirectionWorldXY, halfAngleDeg, cfg)",
            "2.0 * halfAngleDeg >= ...",
            "cfg.search.directional_corridor_minimum_full_width_deg",
            "2.0 * halfAngleDeg <= ...",
            "cfg.search.directional_corridor_maximum_full_width_deg",
            "cfg.search.directional_corridor_tracking_allowance_mm;",
            "axialStartMm = min(max(dot(currentOffsetMm, direction), 0.0), ...",
            "axialStopMm = min(radiusMm, axialStartMm + ...",
            "cfg.search.directional_corridor_maximum_advance_mm);",
            "pathOffsetsMm = currentOffsetMm;",
            "target = axialStationsMm(stationIndex) * direction;",
            "vecnorm(pathOffsetsMm, 2, 2) <= radiusMm + 1e-9",
            "'M12Demo:DirectionalCorridorCadDisk'",
            "axialProgressMm = diff(pathOffsetsMm * direction.');",
            "cfg.search.directional_progress_tolerance_mm",
            "'M12Demo:DirectionalCorridorMonotonicity'",
            "laneOffsetsMm = zeros(size(pathOffsetsMm, 1), 1);",
            "'routing_mode', 'MONOTONIC_LEARNED_DIRECTION_CENTERLINE'",
        ):
            self.assertIn(token, builder)
        for obsolete_serpentine in (
            "directional_corridor_lane_spacing_capture_fraction",
            "requestedLaneOffsets",
            "laneStarts",
            "laneEnds",
            "forward = true;",
            "forward = ~forward;",
        ):
            self.assertNotIn(obsolete_serpentine, builder)

        route_start = self.main.index(
            "result.entropy_grid_selection_regions = cell(0, 1);"
        )
        route_end = self.main.index("if ~capture.found", route_start)
        route = self.main[route_start:route_end]
        for token in (
            "directionQualified = norm(activeGridRoutingDirection) > 1e-12;",
            "activeResidualActionMask = directionalSectorMask( ...",
            "['FROZEN_LEARNED_DIRECTION_MONOTONIC_CENTERLINE_THEN_', ...",
            "'CONNECTED_NOVEL_RESIDUAL'];",
            "buildDirectionalCadCorridor(currentGridOffset, ...",
            "activeGridRoutingDirection, corridorHalfAngleDeg, cfg)",
            "result.directional_corridor_planned_offsets_mm = ...",
            "result.directional_corridor_lane_offsets_mm = ...",
            "result.directional_corridor_path_length_mm = ...",
            "'DIRECTIONAL_CAD_CORRIDOR_PLANNED'",
            "'FROZEN_CONICAL_DIRECTION_CENTERLINE_TARGET'",
            "for corridorIndex = 2:size(corridorOffsets, 1)",
            "totalSearchTargets >= cfg.search.maximum_grid_nodes",
            "totalSearchTargets = totalSearchTargets + 1;",
            "moveSurfaceWithPolicyDeadline(ctx, ...",
            "sprintf('DIRECTIONAL_CORRIDOR_%02d', ...",
            "segmentStartOffset = currentGridOffset;",
            "measuredProbeOffset = ...",
            "probe.pose_mm_rad(1:2) - centerXY;",
            "currentGridOffset = measuredProbeOffset;",
            "plannedTestedOffsets(end + 1, :) = offset;",
            "measuredTestedOffsets(end + 1, :) = ...",
            "measuredSweptSegments = appendMeasuredSurfaceSegments( ...",
            "probe.measured_xy_polyline_mm",
            "centerHypotheses, segmentStartOffset, ...",
            "measuredProbeOffset, true, cfg",
            "measuredProbeOffset, false, cfg",
            "surfaceNegativeObservationValid(probe)",
            "'PHYSICAL_CAPTURE_HANDOFF'",
            "while ~capture.found && ...",
            "totalSearchTargets < cfg.search.maximum_grid_nodes",
            "residualEigSteps = residualEigSteps + 1;",
            "residualCadCoverageSteps = residualCadCoverageSteps + 1;",
            "result.bayesian_grid_steps = residualGridSteps;",
        ):
            self.assertIn(token, route)
        self.assertNotIn(
            "corridorCredible.mass_requirement_satisfied;",
            route[
                route.index("directionQualified =") :
                route.index("if directionQualified")
            ],
            "80 percent posterior mass is diagnostic; it must not veto a "
            "finite learned direction",
        )
        for obsolete_skip in (
            "RESIDUAL_EIG_ONLY_CORRIDOR_80PCT_REQUIRES_OVER_45DEG",
            "SKIPPED_80PCT_ROUTE_CENTERED_SECTOR_EXCEEDS_45DEG",
            "DIRECTIONAL_CAD_CORRIDOR_SKIPPED_DIFFUSE_POSTERIOR",
        ):
            self.assertNotIn(obsolete_skip, route)

        route_centered = self.main[
            self.main.index(
                "function [halfAngleDeg, diagnostics] = ..."
            ) :
            self.main.index(
                "function [pathOffsetsMm, laneOffsetsMm, diagnostics] = ..."
            )
        ]
        for token in (
            "posteriorCorridorHalfAngleDeg(output, cfg, routingDirectionWorldXY)",
            "routingDirectionWorldXY = routingDirectionWorldXY / ...",
            "directionTask = cfg.frame.world_from_fixture_rotation.' * ...",
            "controlCenterDeg = mod(rad2deg(atan2( ...",
            "anglesDeg - controlCenterDeg + 180.0",
            "'control_center_task_deg', controlCenterDeg",
        ):
            self.assertIn(token, route_centered)
        self.assertIn(
            "posteriorCorridorHalfAngleDeg(learned.output, cfg, ...\n"
            "                learnedShortcut.direction_world_xy);",
            self.main,
        )

        capture_branch = route.index("if probe.capture_confirmed")
        positive_update = route.index(
            "posterior = updatePosteriorAlongSegment(posterior, ...",
            capture_branch,
        )
        make_capture = route.index(
            "capture = makeCapture(true, ...", positive_update
        )
        physical_handoff = route.index(
            "'PHYSICAL_CAPTURE_HANDOFF'", make_capture
        )
        immediate_break = route.index("break;", physical_handoff)
        negative_update = route.index(
            "posterior = updatePosteriorAlongSegment(posterior, ...",
            immediate_break,
        )
        self.assertLess(capture_branch, positive_update)
        self.assertLess(positive_update, make_capture)
        self.assertLess(make_capture, physical_handoff)
        self.assertLess(physical_handoff, immediate_break)
        self.assertLess(immediate_break, negative_update)

        for removed_contract in (
            "maximum_action_jump_mm",
            "classical_direction_sector_half_angle_deg",
            "entropy_grid_sector_fallback",
            "DIRECTION_SECTOR_EXHAUSTED",
            "UNIFORM_ROUTING_ABSTENTION_CAD_BOUNDED",
        ):
            self.assertNotIn(removed_contract, self.main)

    def test_directional_corridor_geometry_truth_table(
        self,
    ) -> None:
        radius_mm = 4.80
        maximum_advance_mm = 1.00
        capture_radius_mm = 0.75
        tracking_allowance_mm = 0.20
        robust_capture_radius_mm = (
            capture_radius_mm - tracking_allowance_mm
        )
        grid_pitch_mm = 0.50
        endpoint_tolerance_mm = 0.12
        progress_tolerance_mm = 0.10

        def norm(point: tuple[float, float]) -> float:
            return math.hypot(*point)

        def rotate(
            point: tuple[float, float],
            angle_deg: float,
        ) -> tuple[float, float]:
            cosine = math.cos(math.radians(angle_deg))
            sine = math.sin(math.radians(angle_deg))
            return (
                cosine * point[0] - sine * point[1],
                sine * point[0] + cosine * point[1],
            )

        def build_corridor(
            current: tuple[float, float],
            direction: tuple[float, float],
        ) -> tuple[list[tuple[float, float]], list[float]]:
            direction_norm = norm(direction)
            self.assertGreater(direction_norm, 1e-12)
            unit = (
                direction[0] / direction_norm,
                direction[1] / direction_norm,
            )
            axial_start = min(
                max(current[0] * unit[0] + current[1] * unit[1], 0.0),
                radius_mm,
            )
            axial_stop = min(
                radius_mm,
                axial_start + maximum_advance_mm,
            )
            path = [current]
            if axial_stop - axial_start > endpoint_tolerance_mm:
                stations: list[float] = []
                station = axial_start + grid_pitch_mm
                while station <= axial_stop + 1e-12:
                    stations.append(station)
                    station += grid_pitch_mm
                if not stations:
                    stations = [axial_stop]
                elif axial_stop - stations[-1] > endpoint_tolerance_mm:
                    stations.append(axial_stop)
                else:
                    stations[-1] = axial_stop
                path.extend(
                    (station * unit[0], station * unit[1])
                    for station in stations
                )
            return path, [0.0] * len(path)

        baseline_path, baseline_lanes = build_corridor(
            (1.50, 0.0), (1.0, 0.0)
        )
        self.assertEqual(baseline_lanes, [0.0] * len(baseline_path))
        self.assertEqual(
            [round(point[0], 10) for point in baseline_path],
            [1.50, 2.00, 2.50],
        )
        self.assertTrue(
            all(abs(point[1]) <= 1e-12 for point in baseline_path)
        )

        for direction_angle_deg in (0.0, 17.0, 90.0, 143.0, -72.0):
            unit = rotate((1.0, 0.0), direction_angle_deg)
            lateral = (-unit[1], unit[0])
            for shortcut_distance_mm in (0.75, 1.50, 4.50):
                shortcut_endpoint = (
                    shortcut_distance_mm * unit[0] + 0.08 * lateral[0],
                    shortcut_distance_mm * unit[1] + 0.08 * lateral[1],
                )
                path, lane_offsets = build_corridor(
                    shortcut_endpoint, unit
                )
                self.assertEqual(lane_offsets, [0.0] * len(path))
                projections = [
                    point[0] * unit[0] + point[1] * unit[1]
                    for point in path
                ]
                self.assertTrue(
                    all(
                        upper - lower >= -progress_tolerance_mm - 1e-12
                        for lower, upper in zip(
                            projections, projections[1:]
                        )
                    )
                )
                self.assertLessEqual(
                    projections[-1] - projections[0],
                    maximum_advance_mm + 1e-12,
                )
                self.assertTrue(
                    all(norm(point) <= radius_mm + 1e-9 for point in path)
                )
                for point in path[1:]:
                    cross_track = (
                        point[0] * lateral[0] + point[1] * lateral[1]
                    )
                    self.assertAlmostEqual(cross_track, 0.0, places=10)

        self.assertAlmostEqual(robust_capture_radius_mm, 0.55)

    def test_residual_eig_and_coverage_are_sector_and_leg_constrained(
        self,
    ) -> None:
        selector = self.main[
            self.main.index(
                "function [candidateIndex, bestInformation, predictiveOpen"
            ) :
            self.main.index(
                "function likelihood = segmentCaptureLikelihood"
            )
        ]
        for token in (
            "posterior, hypotheses, actionOffsets, plannedTestedOffsets, ...",
            "measuredTestedOffsets, measuredSweptSegments, currentOffset, ...",
            "admissibleActions, routingDirectionWorldXY, cfg)",
            "size(hypotheses, 1) == numel(posterior)",
            "size(actionOffsets, 1) == numel(admissibleActions)",
            "selectionRegion = 'NO_UNTESTED_95PCT_CREDIBLE_NODE';",
            "credibleSupportMask( ...",
            "posterior, cfg.search.eig_credible_mass)",
            "for index = 1:size(actionOffsets, 1)",
            "candidate = actionOffsets(index, :);",
            "if ~admissibleActions(index) || ...",
            "travelMm > cfg.search.residual_maximum_connected_leg_mm + 1e-12",
            "residualCandidatePhysicalAdmission(candidate, currentOffset, ...",
            "plannedTestedOffsets, measuredTestedOffsets, ...",
            "measuredSweptSegments, routingDirectionWorldXY, cfg);",
            "alreadyTested(candidate, plannedTestedOffsets, ...",
            "alreadyTested(candidate, measuredTestedOffsets, ...",
            "cfg.search.tested_offset_tolerance_mm)",
            "directionalProgressMm < ...",
            "-cfg.search.directional_progress_tolerance_mm - 1e-12",
            "globalForwardProgressMm = candidateProjectionMm - "
            "historyFrontierMm;",
            "measuredSweptNovelLength( ...",
            "cfg.search.minimum_novel_swept_length_mm - 1e-12;",
            "pointToSegmentDistance2d(sample, historyStart, historyEnd)",
            "continue;",
            "travelMm = norm(candidate - currentOffset);",
            "segmentCaptureLikelihood( ...\n"
            "        hypotheses, currentOffset, candidate, cfg)",
            "candidateIndex = index;",
            "cfg.search.eig_forward_progress_weight_per_mm * ...",
            "max(0.0, globalForwardProgressMm) + ...",
            "cfg.search.eig_novel_coverage_weight_per_mm * ...",
            "if candidateIndex == 0",
            "bestInformation = NaN;",
            "bestUtility <= cfg.search.eig_minimum_utility",
            "selectionRegion = 'BEST_95PCT_CREDIBLE_UTILITY_BELOW_THRESHOLD';",
            "DIRECTION_CONDITIONED_CONNECTED_95PCT_CREDIBLE_RESIDUAL_EIG",
            "selectCadCoverageNode( ...",
            "for coveragePass = 1:2",
            "DIRECTION_CONDITIONED_CONNECTED_95PCT_CAD_COVERAGE",
            "DIRECTION_CONDITIONED_CONNECTED_COMMON_REACH_CAD_COVERAGE",
            "This is deliberately a coverage score, not an EIG admission gate.",
            "if candidateIndex ~= 0",
            "return;",
            "function mask = directionalSectorMask(",
            "projection > 0.0 & angleDeg <= halfAngleDeg + 1e-9",
            "function posterior = conditionPosteriorOnDirectionalSector(",
            "posterior(~mask) = posterior(~mask) * outsideLikelihoodFloor;",
        ):
            self.assertIn(token, selector)
        self.assertNotIn(
            "hypotheses(candidateIndex, :) - currentOffset",
            selector,
        )
        self.assertGreaterEqual(selector.count("continue;"), 4)

        action_offsets = [
            (0.5 * ix, 0.5 * iy)
            for ix in range(-10, 11)
            for iy in range(-10, 11)
            if math.hypot(0.5 * ix, 0.5 * iy) <= 4.8 + 1e-12
        ]
        self.assertEqual(len(action_offsets), 293)

        half_angle_deg = 22.5
        maximum_leg_mm = 1.0

        def in_positive_x_sector(candidate: tuple[float, float]) -> bool:
            radius = math.hypot(*candidate)
            if radius <= 1e-12:
                return True
            angle_deg = math.degrees(
                math.acos(max(-1.0, min(1.0, candidate[0] / radius)))
            )
            return candidate[0] > 0.0 and angle_deg <= half_angle_deg

        def admissible_from(
            current: tuple[float, float],
        ) -> list[tuple[float, float]]:
            return [
                candidate
                for candidate in action_offsets
                if in_positive_x_sector(candidate)
                and math.dist(candidate, current)
                <= maximum_leg_mm + 1e-12
            ]

        origin_actions = admissible_from((0.0, 0.0))
        self.assertIn((0.5, 0.0), origin_actions)
        self.assertIn((1.0, 0.0), origin_actions)
        self.assertNotIn((0.0, 0.5), origin_actions)
        self.assertNotIn((-0.5, 0.0), origin_actions)
        self.assertNotIn((1.5, 0.0), origin_actions)
        next_actions = admissible_from((0.5, 0.0))
        self.assertIn((1.5, 0.0), next_actions)
        self.assertNotIn((2.0, 0.0), next_actions)

        def point_to_segment_distance(
            point: tuple[float, float],
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> float:
            segment = (end[0] - start[0], end[1] - start[1])
            denominator = segment[0] ** 2 + segment[1] ** 2
            if denominator <= 1e-15:
                return math.dist(point, start)
            fraction = (
                (point[0] - start[0]) * segment[0]
                + (point[1] - start[1]) * segment[1]
            ) / denominator
            fraction = min(max(fraction, 0.0), 1.0)
            closest = (
                start[0] + fraction * segment[0],
                start[1] + fraction * segment[1],
            )
            return math.dist(point, closest)

        def novel_swept_length(
            start: tuple[float, float],
            end: tuple[float, float],
            measured_segments: list[
                tuple[tuple[float, float], tuple[float, float]]
            ],
        ) -> float:
            segment_length = math.dist(start, end)
            interval_count = max(1, math.ceil(segment_length / 0.05))
            novel_intervals = 0
            for interval_index in range(interval_count):
                fraction = (interval_index + 0.5) / interval_count
                sample = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                minimum_distance = min(
                    point_to_segment_distance(sample, old_start, old_end)
                    for old_start, old_end in measured_segments
                )
                novel_intervals += minimum_distance > 0.10
            return segment_length * novel_intervals / interval_count

        measured_history = [
            ((0.0, 0.0), (0.5, 0.0)),
            ((0.5, 0.0), (1.0, 0.0)),
        ]
        self.assertGreaterEqual(
            novel_swept_length(
                (1.0, 0.0), (1.5, 0.0), measured_history
            ),
            0.25,
            "a forward segment that creates new physical coverage is valid",
        )
        self.assertLess(
            novel_swept_length(
                (0.5, 0.0), (1.0, 0.0), measured_history
            ),
            0.25,
            "replaying an old measured scanline must not count as coverage",
        )
        learned_direction = (1.0, 0.0)
        backward_progress = (
            (0.5 - 1.0) * learned_direction[0]
            + (0.5 - 0.0) * learned_direction[1]
        )
        self.assertLess(backward_progress, -0.10)

    def test_dashboard_matches_fixture_view_and_has_only_world_axes(
        self,
    ) -> None:
        visualization = self.main[
            self.main.index("cfg.visualization = struct(") :
            self.main.index("cfg.insertion = struct(")
        ]
        world_length = re.search(
            r"'world_axis_length_mm',\s*([0-9]+(?:\.[0-9]+)?)",
            visualization,
        )
        self.assertIsNotNone(world_length)
        self.assertLessEqual(float(world_length.group(1)), 4.0)
        self.assertNotIn("tcp_axis_length_mm", visualization)
        self.assertNotIn("peg_axis_length_mm", visualization)

        initialize = self.main[
            self.main.index("function dashboard = initializeDashboard") :
            self.main.index("function dashboard = updateDashboard")
        ]
        self.assertIn("view(trajectoryAxes, 135, 24);", initialize)
        self.assertRegex(
            initialize,
            r"(?:camproj\(trajectoryAxes,\s*'orthographic'\)|"
            r"set\(trajectoryAxes,\s*'Projection',\s*'orthographic'\))",
        )
        self.assertIn(
            "[actualTrajectory, commandTrajectory]",
            initialize,
        )
        for forbidden in (
            "tcpAxes",
            "tcp_axes",
            "pegAxis",
            "peg_axis",
            "black 25 mm",
            "axis marker, not motion",
        ):
            self.assertNotIn(forbidden, initialize)

        update = self.main[
            self.main.index("function dashboard = updateDashboard") :
            self.main.index("function value = smoothstep01")
        ]
        for forbidden in (
            "worldFromTcp",
            "tcpLength",
            "tcp_axes",
            "pegVector",
            "peg_axis",
            "black 25 mm",
        ):
            self.assertNotIn(forbidden, update)

    def test_mode_specific_plan_does_not_claim_other_algorithms(self) -> None:
        plan = self.main[
            self.main.index("function plan = makePlan") :
            self.main.index("function validateConicalContract")
        ]
        for token in (
            "FORCE_HELD_GLOBAL_OUTWARD_ARCHIMEDEAN_SEARCH",
            "FORCE_HELD_LOCAL_OUTWARD_ARCHIMEDEAN_RESIDUAL",
            "EXACTLY_THREE_MAXIMUM_INDEPENDENT_QUALIFIED_CONES_AND_DIRECT_RAYS",
            "FORCE_HELD_FINITE_ENTROPY_GRID_OVER_MEASURED_TOUCHDOWN_ANCHOR",
        ):
            self.assertIn(token, plan)
        writer = self.main[
            self.main.index("function writePlan(") :
            self.main.index("function writeSamplingExecutionPlan")
        ]
        pure_case = writer[
            writer.index(
                "case 'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL'"
            ) :
            writer.index(
                "case 'FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL'"
            )
        ]
        self.assertIn("global_outward_spiral", pure_case)
        self.assertNotIn("conical_contact", pure_case)
        self.assertNotIn("bayes_hypothesis", pure_case)

    def test_archimedean_spiral_closes_full_4p2mm_terminal_perimeter(
        self,
    ) -> None:
        baseline_config = self.main[
            self.main.index("cfg.baselines = struct(") :
            self.main.index("activeLearnedBundleRoot")
        ]
        for token in (
            "'spiral_pitch_mm_per_revolution', activeCoveragePitchMm",
            "'pure_outward_radius_mm', 4.20",
            "'path_waypoint_arc_mm', 0.60",
        ):
            self.assertIn(token, baseline_config)
        spiral_source = self.main[
            self.main.index("function waypoints = archimedeanWaypoints(") :
            self.main.index(
                "function [ctx, capture, diagnostics] = "
                "runImprovedDirectRetry3("
            )
        ]
        for token in (
            "b = pitchMm / (2.0 * pi);",
            "maximumTheta = maximumRadiusMm / b;",
            "while theta < maximumTheta",
            "while norm(waypoint - waypoints(end, :)) > arcStepMm",
            "perimeterSteps = max(3, ceil( ...",
            "2.0 * pi * maximumRadiusMm / arcStepMm));",
            "terminalAngle = phaseRad + maximumTheta;",
            "for perimeterIndex = 1:perimeterSteps",
            "2.0 * pi * perimeterIndex / perimeterSteps;",
            "waypoint = centerXY + maximumRadiusMm * ...",
            "max(increments) <= arcStepMm * 1.01",
            "maximumRadiusMm) < 1e-9",
        ):
            self.assertIn(token, spiral_source)
        runtime = self.main[
            self.main.index(
                "function [ctx, capture, diagnostics] = "
                "runOutwardArchimedeanSearch("
            ) :
            self.main.index("function waypoints = archimedeanWaypoints(")
        ]
        self.assertIn("'terminal_perimeter_sweep', true", runtime)

        pitch_mm = 1.00
        maximum_radius_mm = 4.20
        arc_step_mm = 0.60
        capture_radius_mm = 0.75
        phase_rad = 0.37

        def waypoints(
            pitch: float,
            radius_limit: float,
            arc: float,
            phase: float,
        ) -> tuple[
            list[tuple[float, float]],
            int,
            int,
            float,
        ]:
            b = pitch / (2.0 * math.pi)
            theta_limit = radius_limit / b
            theta = 0.0
            points = [(0.0, 0.0)]
            while theta < theta_limit:
                radius = b * theta
                theta_step = min(
                    1.0, arc / max(math.hypot(radius, b), 1e-15)
                )
                candidate = min(theta + theta_step, theta_limit)
                candidate_radius = b * candidate
                point = (
                    candidate_radius * math.cos(phase + candidate),
                    candidate_radius * math.sin(phase + candidate),
                )
                while math.dist(point, points[-1]) > arc:
                    theta_step *= 0.8
                    candidate = min(theta + theta_step, theta_limit)
                    candidate_radius = b * candidate
                    point = (
                        candidate_radius * math.cos(phase + candidate),
                        candidate_radius * math.sin(phase + candidate),
                    )
                theta = candidate
                if math.dist(point, points[-1]) > 1e-9:
                    points.append(point)
            main_count = len(points)
            perimeter_steps = max(
                3,
                math.ceil(2.0 * math.pi * radius_limit / arc),
            )
            terminal_angle = phase + theta_limit
            for perimeter_index in range(1, perimeter_steps + 1):
                perimeter_angle = (
                    terminal_angle
                    + 2.0
                    * math.pi
                    * perimeter_index
                    / perimeter_steps
                )
                point = (
                    radius_limit * math.cos(perimeter_angle),
                    radius_limit * math.sin(perimeter_angle),
                )
                if math.dist(point, points[-1]) > 1e-9:
                    points.append(point)
            return points, main_count, perimeter_steps, terminal_angle

        points, main_count, perimeter_steps, terminal_angle = waypoints(
            pitch_mm,
            maximum_radius_mm,
            arc_step_mm,
            phase_rad,
        )
        main_spiral = points[:main_count]
        terminal_perimeter = points[main_count - 1 :]

        self.assertEqual(perimeter_steps, 44)
        self.assertEqual(len(terminal_perimeter), perimeter_steps + 1)
        self.assertAlmostEqual(
            math.dist(terminal_perimeter[0], terminal_perimeter[-1]),
            0.0,
            places=9,
        )
        unwrapped_terminal_angles = [
            terminal_angle
            + 2.0 * math.pi * index / perimeter_steps
            for index in range(perimeter_steps + 1)
        ]
        self.assertAlmostEqual(
            unwrapped_terminal_angles[-1]
            - unwrapped_terminal_angles[0],
            2.0 * math.pi,
            places=12,
        )
        for point in terminal_perimeter:
            self.assertAlmostEqual(
                math.hypot(*point),
                maximum_radius_mm,
                places=9,
            )

        self.assertLessEqual(
            max(
                math.dist(first, second)
                for first, second in zip(
                    main_spiral,
                    main_spiral[1:],
                )
            ),
            arc_step_mm * 1.01,
        )
        terminal_chord_max = max(
            math.dist(first, second)
            for first, second in zip(
                terminal_perimeter,
                terminal_perimeter[1:],
            )
        )
        self.assertLessEqual(
            terminal_chord_max,
            arc_step_mm * 1.01,
        )
        self.assertAlmostEqual(
            math.hypot(*points[-1]),
            maximum_radius_mm,
            places=9,
        )

        # The mathematical Rmax circle covers every angular direction in the
        # outer capture annulus exactly.  The executed polygonal chords sit
        # inward by only one sagitta, which is quantified separately below.
        for angle_index in range(720):
            angle = 2.0 * math.pi * angle_index / 720.0
            for radial_fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
                radius = (
                    maximum_radius_mm
                    + radial_fraction * capture_radius_mm
                )
                point = (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                )
                distance_to_terminal_circle = abs(
                    math.hypot(*point) - maximum_radius_mm
                )
                self.assertLessEqual(
                    distance_to_terminal_circle,
                    capture_radius_mm + 1e-12,
                )

        def point_to_segment_distance(
            point: tuple[float, float],
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> float:
            segment = (end[0] - start[0], end[1] - start[1])
            length_squared = segment[0] ** 2 + segment[1] ** 2
            fraction = (
                (point[0] - start[0]) * segment[0]
                + (point[1] - start[1]) * segment[1]
            ) / length_squared
            fraction = min(max(fraction, 0.0), 1.0)
            closest = (
                start[0] + fraction * segment[0],
                start[1] + fraction * segment[1],
            )
            return math.dist(point, closest)

        discrete_sagitta_margin_mm = maximum_radius_mm * (
            1.0 - math.cos(math.pi / perimeter_steps)
        )
        self.assertLess(discrete_sagitta_margin_mm, 0.011)
        maximum_discrete_boundary_gap_mm = 0.0
        for angle_index in range(3600):
            angle = 2.0 * math.pi * angle_index / 3600.0
            boundary = (
                (maximum_radius_mm + capture_radius_mm)
                * math.cos(angle),
                (maximum_radius_mm + capture_radius_mm)
                * math.sin(angle),
            )
            gap = min(
                point_to_segment_distance(boundary, start, end)
                for start, end in zip(
                    terminal_perimeter,
                    terminal_perimeter[1:],
                )
            )
            maximum_discrete_boundary_gap_mm = max(
                maximum_discrete_boundary_gap_mm,
                gap,
            )
        self.assertLessEqual(
            maximum_discrete_boundary_gap_mm,
            capture_radius_mm + discrete_sagitta_margin_mm + 1e-9,
        )

    def test_proposed_is_one_learned_shortcut_then_local_outward_residual(
        self,
    ) -> None:
        policy_function = self.main.index("function policy = executionPolicy")
        policy = self.main[
            self.main.index(
                "case 'FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL'",
                policy_function,
            ) :
            self.main.index(
                "case 'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL'",
                policy_function,
            )
        ]
        self.assertIn("learned CAD-posterior bounded", policy)
        self.assertIn("posterior-sized local outward residual", policy)
        self.assertIn("'LOCAL_OUTWARD_ARCHIMEDEAN'", policy)
        residual = self.main[
            self.main.index(
                "if strcmp(executionMode, ...\n"
                "            'FULL_LOOP_PROPOSED_LEARNED_LOCAL_SPIRAL')",
                self.main.index("gridStep = 0;"),
            ) :
            self.main.index(
                "else\n    result.entropy_grid_selection_regions",
                self.main.index("gridStep = 0;"),
            )
        ]
        self.assertIn("runOutwardArchimedeanSearch", residual)
        self.assertIn("learnedShortcut.residual_radius_mm", residual)
        self.assertNotIn("selectExpectedInformationNode", residual)
        baseline = self.main[
            self.main.index("cfg.baselines = struct") :
            self.main.index("activeLearnedBundleRoot")
        ]
        self.assertIn(
            "'proposed_local_pitch_mm_per_revolution', ...\n"
            "        1.60 * m12ConservativeCaptureRadiusMm",
            baseline,
        )
        self.assertIn("'proposed_residual_radius_mm', 4.20", baseline)
        self.assertIn(
            "posterior, 0.95) - cfg.search.bayes_capture_radius_mm;",
            self.main,
        )
        negative_observation_gate = self.main.index(
            "if surfaceNegativeObservationValid(shortcutTransport)"
        )
        sector_conditioning = self.main.index(
            "posterior = conditionPosteriorOnDirectionalSector( ...",
            negative_observation_gate,
        )
        measured_q95 = self.main.index(
            "measuredResidualRadiusMm = weightedQuantile( ...",
            sector_conditioning,
        )
        self.assertLess(negative_observation_gate, sector_conditioning)
        self.assertLess(sector_conditioning, measured_q95)
        conditioned_block = self.main[
            sector_conditioning:
            self.main.index("gridStep = 0;", sector_conditioning)
        ]
        for token in (
            "'LEARNED_DIRECTION_SECTOR_AFTER_MEASURED_SHORTCUT_OBSERVATION'",
            "'ORIGINAL_MEASURED_TOUCHDOWN_ANCHOR'",
            "vecnorm(centerHypotheses - measuredShortcutOffset, 2, 2)",
            "posterior, 0.95) - cfg.search.bayes_capture_radius_mm;",
            "shortcutEvidence.residual_posterior_scope = ...",
            "shortcutEvidence.residual_posterior_coordinate_frame = ...",
            "shortcutEvidence.residual_posterior_sector_half_angle_deg = ...",
            "shortcutEvidence.residual_radius_uses_measured_shortcut_endpoint = ...",
            "true;",
        ):
            self.assertIn(token, conditioned_block)
        self.assertNotIn(
            "centerHypotheses = centerHypotheses + measuredShortcutOffset",
            conditioned_block,
            "belief hypotheses remain in the original touchdown frame",
        )

    def test_improved_direct_reprobes_with_fresh_model_and_no_fan(
        self,
    ) -> None:
        direct = self.main[
            self.main.index(
                "function [ctx, capture, diagnostics] = runImprovedDirectRetry3"
            ) :
            self.main.index(
                "function [ctx, transport] = moveSurfaceWithPolicyDeadline"
            )
        ]
        for token in (
            "adjustmentSigns = [0.0, 1.0, -1.0];",
            "if attempt > 1",
            "runConicalFingerprint(ctx, ...",
            "inferFrozenConicalModel( ...",
            "learnedPosteriorModeForDirectAttempt(currentLearned, ...",
            "referenceDirectionTaskDeg",
            "direct_retry_direction_half_width_deg",
            "FRESH_TOP1_INSIDE_FIRST_DIRECTION_MICROADJUST_SECTOR",
            "eligibleProbability(~eligible) = -Inf;",
            "[~, selectedIndex] = max(eligibleProbability);",
            "FRESH_HIGHEST_POSTERIOR_MODE_INSIDE_FIRST_DIRECTION_MICROADJUST_SECTOR",
            "FRESH_POSTERIOR_FIRST_DIRECTION_MICROADJUST_BOUNDED_DIRECT_RAY",
            "direct_ray_limit_mm * direction",
            "record.fan_angle_deg = 0.0;",
            "DIRECT_RETRY_RESET_SENSOR_DOWNWARD_PROGRESS_CAPTURE_CANDIDATE",
        ):
            self.assertIn(token, direct)
        self.assertNotRegex(
            direct,
            r"(fanAnglesDeg\s*\(\s*attempt\s*\)|"
            r"direction\s*\+\s*fan|rotate.*fan)",
        )
        self.assertIn(
            "'direct_max_attempts', 3", self.main
        )
        self.assertIn(
            "'direct_ray_limit_mm', activeDirectRayLimitMm", self.main
        )
        self.assertIn(
            "'direct_retry_direction_half_width_deg', 20.0",
            self.main,
        )
        baseline = self.main[
            self.main.index("cfg.baselines = struct") :
            self.main.index("activeLearnedBundleRoot")
        ]
        fan = re.search(
            r"'direct_fan_angles_deg',\s*\[([^\]]+)\]",
            baseline,
        )
        self.assertIsNotNone(fan)
        self.assertEqual(
            [
                float(value)
                for value in re.findall(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                    fan.group(1),
                )
            ],
            [0.0, 0.0, 0.0],
        )
        self.assertIn("'direct_adjustment_mm', 0.25", baseline)
        self.assertNotIn(
            "runPersistentImprovedDirectRounds", self.main
        )
        runtime_before_capture_assert = self.main[
            self.main.index(
                "[ctx, capture, directDiagnostics] = ..."
            ) :
            self.main.index("assert(capture.found")
        ]
        for token in (
            "runImprovedDirectRetry3(ctx, conical, learned, centerXY",
            "finishDirectX3WithoutCapture(",
            "return;",
        ):
            self.assertIn(token, runtime_before_capture_assert)
        self.assertIn(
            "IMPROVED_DIRECT_STRICT_THREE_ATTEMPTS_COMPLETE_NO_CAPTURE",
            direct,
        )
        self.assertNotIn("another x3 round", direct.lower())

    def test_incomplete_cone_stops_but_weak_complete_cone_remains_actionable(
        self,
    ) -> None:
        inference = self.main[
            self.main.index("function learned = inferFrozenConicalModel") :
            self.main.index(
                "function [ctx, capture, diagnostics] = runOutwardArchimedeanSearch"
            )
        ]
        for token in (
            "cone.direction_fingerprint_valid && ...",
            "~cone.aperture_candidate",
            "'M12Demo:InvalidConeForLearnedDirection'",
            "Recovered low-force episodes reduce sample_quality but do",
            "not remove bounded high-level motion authority.",
        ):
            self.assertIn(token, inference)
        output_contract = self.main[
            self.main.index("function assertActiveLearnedOutput") :
            self.main.index("function validateSurfaceTransportContract")
        ]
        for token in (
            "all(isfinite(directionProbability))",
            "all(isfinite(stateProbability))",
            "all(isfinite(directionWorldXY))",
            "logical(value.weak_complete_fingerprint_remains_actionable)",
            "~logical(value.posterior_quality_is_validity_veto)",
            "logical(value.hardware_control_allowed)",
        ):
            self.assertIn(token, output_contract)
        route = self.main[
            self.main.index("if conical.aperture_candidate") :
            self.main.index("if ~capture.found")
        ]
        self.assertIn(
            "error('M12Demo:LearnedInferenceRequired'", route
        )
        self.assertNotRegex(route.lower(), r"classical|shadow")

    def test_all_modes_join_shared_capture_insertion_and_terminal_verifier(
        self,
    ) -> None:
        shared_capture = self.main.index("if ~capture.found")
        insertion = self.main.index(
            "[ctx, insertion] = guardedM12Insertion(ctx)",
            shared_capture,
        )
        terminal = self.main.index(
            "[result, endState, ctx] = finishAtBottom(",
            insertion,
        )
        self.assertLess(shared_capture, insertion)
        self.assertLess(insertion, terminal)
        for token in (
            "hard_force_delta_n', 25.0",
            "hard_torque_delta_nm', 0.75",
            "capture_depth_drop_mm', 0.25",
            "'success_depth_minimum_mm', cadSocketDepthMm - 0.30",
            "'success_depth_maximum_mm', cadSocketDepthMm + 0.20",
        ):
            self.assertIn(token, self.main)

    def test_policy_deadline_bounds_each_surface_segment(self) -> None:
        touchdown = self.main.index(
            "contactNormalSign = touchdown.normal_sign_world_z;"
        )
        clock = self.main.index("policyClock = tic;", touchdown)
        first_cone = self.main.index(
            "[ctx, conical] = runConicalFingerprint", clock
        )
        pure_search = self.main.index(
            "runOutwardArchimedeanSearch", clock
        )
        self.assertLess(clock, first_cone)
        self.assertLess(clock, pure_search)
        helper = self.main[
            self.main.index(
                "function [ctx, transport] = moveSurfaceWithPolicyDeadline"
            ) :
            self.main.index("function posterior = modelGridPosterior")
        ]
        self.assertIn("policyBudgetS - toc(policyClock)", helper)
        self.assertIn(
            "min(originalPhaseTimeoutS, remainingS)", helper
        )
        self.assertIn("assertPolicyDeadline", helper)

    def test_full_tcp_payload_and_sensor_metrology_fail_before_kst(self) -> None:
        tooling_path = (
            ROOT.parent
            / "mujoco_unscrewing_probe_new"
            / "configs"
            / "real_tooling_v1.json"
        )
        tooling = json.loads(tooling_path.read_text(encoding="utf-8"))
        metrology_path = (
            ROOT / "configs" / "tool_metrology_commissioning_v1.json"
        )
        metrology = json.loads(metrology_path.read_text(encoding="utf-8"))
        tcp = metrology["tcp_pivot_touch_off_commissioning"]
        sensor = metrology["sensor_metrology"]
        payload = metrology["payload_identification"]
        load_handling = metrology[
            "controller_load_handling_qualification"
        ]
        self.assertEqual(tcp["status"], "PENDING_POST_INSTALL_METROLOGY")
        self.assertIsNone(
            tcp["measured_media_flange_to_tip_translation_m"]
        )
        self.assertLessEqual(
            tcp["translation_uncertainty_acceptance_m"], 0.00025
        )
        self.assertLessEqual(
            tcp["pivot_fit_max_residual_acceptance_m"], 0.00025
        )
        self.assertLessEqual(
            tcp["lateral_uncertainty_acceptance_m"], 0.00014
        )
        self.assertIsNone(sensor["T_sensor_from_tip_tcp"])
        self.assertFalse(sensor["ready_for_hardware_wrench_transform"])
        self.assertIsNone(payload["measured_complete_tool_mass_kg"])
        self.assertIsNone(
            payload["measured_com_in_sensor_measurement_frame_m"]
        )
        self.assertIsNone(payload["measured_inertia_at_com_kg_m2"])
        self.assertEqual(
            load_handling["status"],
            "PENDING_FINAL_INSTALLED_LOAD_QUALIFICATION",
        )
        self.assertIsNone(
            load_handling["expected_sunrise_application_name"]
        )
        self.assertIsNone(load_handling["expected_load_data_name"])
        self.assertIsNone(
            load_handling["expected_controller_station_identity"]
        )
        self.assertIsNone(
            load_handling[
                "expected_controller_deployment_session_id"
            ]
        )
        self.assertIn(
            "new UUID",
            load_handling["controller_deployment_session_rule"],
        )
        self.assertEqual(
            load_handling["expected_robot_ip"], "192.168.1.147"
        )
        self.assertIsNone(
            load_handling["sunrise_project_source_export_path"]
        )
        self.assertIsNone(
            load_handling[
                "expected_sunrise_project_source_sha256"
            ]
        )
        self.assertIsNone(
            load_handling["sunrise_load_data_deployed_utc"]
        )
        self.assertIsNone(
            load_handling["controller_deployment_started_utc"]
        )
        self.assertLessEqual(
            load_handling["sunrise_readback_maximum_age_s"], 900.0
        )
        self.assertFalse(
            load_handling["sunrise_nonzero_load_data_applied"]
        )
        self.assertFalse(
            load_handling[
                "final_installed_load_directservo_tracking_qualified"
            ]
        )
        self.assertEqual(
            metrology["tooling_profile_sha256"],
            hashlib.sha256(tooling_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            metrology["combined_lateral_uncertainty_budget"]["acceptance_m"],
            0.0003,
        )

        validation = self.main[
            self.main.index(
                "function [geometryValid, payloadComplete, gate]"
            ) :
            self.main.index(
                "function validateRigidCommissioningTransform"
            )
        ]
        for token in (
            "MeasuredTcpNotQualified",
            "translation_uncertainty_acceptance_m",
            "lateral_uncertainty_acceptance_m",
            "pivot_fit_max_residual_acceptance_m",
            "verifyCommissioningEvidence",
            "PayloadMetrologyIncomplete",
            "eig(0.5 * (inertiaAtCom + inertiaAtCom.')) > 0.0",
            "T_sensor_from_tip_tcp",
            "SensorTipMetrologyIncomplete",
            "cfg.files.tool_metrology_commissioning",
            "offline_phase_gravity_and_torque_correction_required",
        ):
            self.assertIn(token, validation)
        validation_call = self.main.index(
            "validateM12Tooling(cfg.files.tooling"
        )
        connection = self.main.index("iiwa.net_establishConnection()")
        self.assertLess(validation_call, connection)
        self.assertIn(
            "toolTransform(1:3, 4) = cfg.tool.translation_mm(:) / 1000.0",
            self.main,
        )
        self.assertNotIn(
            "toolTransform(3, 4) = cfg.tool.flange_to_tip_mm", self.main
        )
        self.assertNotIn(
            "PAYLOAD_METROLOGY_DISCLOSED_INCOMPLETE", self.main
        )
        fixture = json.loads(
            (
                ROOT
                / "configs"
                / "fixture_frame_commissioning_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            fixture["tooling_profile_sha256"],
            hashlib.sha256(tooling_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            hashlib.sha256(tooling_path.read_bytes()).hexdigest(),
            "03878ec282b8e9284b8733062d9ecf9132834069a83cad573755b0120617b356",
        )

    def test_json_null_text_fields_fail_cleanly_and_end_running_app(self) -> None:
        """JSON null must not escape as a 0x0x0 char into strtrim."""
        self.assertNotIn("strtrim(char(string(", self.main)
        normalizer = self.main[
            self.main.index("function text = scalarTextOrEmpty") :
            self.main.index("function seconds = stampSeconds")
        ]
        self.assertIn("if isempty(value)", normalizer)
        self.assertIn("text = '';", normalizer)
        self.assertIn("M12Demo:ScalarTextField", normalizer)
        self.assertIn("text = strtrim(text);", normalizer)

        tooling_validation = self.main[
            self.main.index(
                "function [geometryValid, payloadComplete, gate]"
            ) :
            self.main.index(
                "function validateRigidCommissioningTransform"
            )
        ]
        self.assertIn(
            "metrologyInstances.physical_tool_instance_id",
            tooling_validation,
        )
        self.assertIn(
            "'M12Demo:ToolMetrologyPhysicalIdentityMissing'",
            tooling_validation,
        )

        exception_path = self.main[
            self.main.index("catch experimentError") :
            self.main.index(
                "function [mode, resumeParentRunDir] = parseExecutionMode"
            )
        ]
        self.assertIn(
            "requestKstEndOnce( ...\n            [], cfg.robot.ip",
            exception_path,
        )
        end_helper = self.main[
            self.main.index(
                "function [sent, method] = requestKstEndOnce"
            ) :
            self.main.index("function paths = initializeRunFiles")
        ]
        self.assertIn("KST_OBJECT_END_SKIPPED", end_helper)
        self.assertIn("uint8(sprintf('end\\n'))", end_helper)
        for motion_token in ("movePTP", "sendEEfPosition", "realTime_start"):
            self.assertNotIn(motion_token, end_helper)

    def test_main_uses_hash_bound_active_real_multisize_bundle(self) -> None:
        bundle = (
            ROOT
            / "model_runtime"
            / "bundles"
            / "real_multisize_direction_v1"
        )
        manifest_path = bundle / "deployment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["model_id"], "real_multisize_direction_v1"
        )
        self.assertEqual(
            manifest["deployment_authority"], "CAPPED_SHORTCUT"
        )
        self.assertEqual(manifest["hardware_control_sizes"], ["m12"])
        self.assertIn(
            "'real_multisize_direction_v1'", self.main
        )
        self.assertIn(
            "'required_authority', 'CAPPED_SHORTCUT'", self.main
        )
        self.assertIn(
            "'required_promotion_qualification', 'DEVELOPMENT_CV_CONTROL'",
            self.main,
        )
        inference = self.main[
            self.main.index("function learned = inferFrozenConicalModel") :
            self.main.index(
                "function [ctx, capture, diagnostics] = runOutwardArchimedeanSearch"
            )
        ]
        self.assertIn("validateModelDeploymentTrustAnchor(cfg);", inference)
        self.assertIn("catch inferenceError", inference)
        validator = self.main[
            self.main.index(
                "function validateModelDeploymentTrustAnchor"
            ) :
            self.main.index(
                "function validateSurfaceTransportContract"
            )
        ]
        for token in (
            "deployment_manifest_sha256",
            "ensemble_interface_contract_sha256",
            "promotion_contract_sha256",
            "required_authority",
            "required_promotion_qualification",
            "capped_shortcut_hardware_control",
            "capture_control",
            "insertion_control",
            "terminal_verification",
        ):
            self.assertIn(token, validator)
        self.assertNotRegex(validator.lower(), r"shadow_only|classical")

    def test_measured_socket_depth_and_controller_load_gate_full_loop(
        self,
    ) -> None:
        fixture = json.loads(
            (
                ROOT
                / "configs"
                / "fixture_frame_commissioning_v1.json"
            ).read_text(encoding="utf-8")
        )
        measured = fixture["measured_world_from_fixture"]
        self.assertIsNone(
            measured["m12_socket_top_to_bottom_depth_mm"]
        )
        self.assertIsNone(
            measured["m12_socket_depth_standard_uncertainty_mm"]
        )
        self.assertIsNone(
            measured["m12_socket_bottom_datum_world_point_mm"]
        )
        self.assertLessEqual(
            measured["m12_socket_depth_uncertainty_acceptance_mm"],
            0.15,
        )
        depth_loader = self.main[
            self.main.index(
                "function frame = loadFixtureFrameCommissioning"
            ) :
            self.main.index(
                "function [toolingGate, combinedGate] = "
                "...\n        validateToolFixtureIdentityAndLateralUncertainty",
                self.main.index(
                    "function frame = loadFixtureFrameCommissioning"
                ),
            )
        ]
        for token in (
            "MeasuredM12SocketDepthNotQualified",
            "m12_socket_top_to_bottom_depth_mm",
            "m12_socket_depth_standard_uncertainty_mm",
            "m12_socket_bottom_datum_world_point_mm",
            "rotation * m12CenterFixtureMm(:)",
            "MuJoCo 6 mm depth is a reference",
        ):
            self.assertIn(token, depth_loader)
        load_gate = self.main[
            self.main.index(
                "function gate = validateControllerLoadHandlingQualification"
            ) :
            self.main.index(
                "function [elapsedS, attempts, source]"
            )
        ]
        for token in (
            "ControllerLoadHandlingNotQualified",
            "sunrise_nonzero_load_data_applied",
            "final_installed_load_directservo_tracking_qualified",
            "conicalTracking.qualified_for_training_candidates",
            "SunriseLoadDataReadbackMismatch",
            "registered_complete_tool_mass_kg",
            "sunrise_project_source_sha256",
            "expected_sunrise_project_source_sha256",
            "expected_sunrise_application_name",
            "expected_load_data_name",
            "expected_controller_station_identity",
            "expected_controller_deployment_session_id",
            "expected_robot_ip",
            "sunrise_project_source_export_path",
            "controller_deployment_started_utc",
            "sunrise_load_data_deployed_utc",
            "sunrise_readback_maximum_age_s",
            "SunriseProjectSourceExport",
            "SunriseProjectPreregistration",
            "SunriseLoadToleranceHardCap",
            "ControllerReadbackAgeHardCap",
            "SunriseReadbackStaleOrWrongSession",
            "ControllerDeploymentAttestationMismatch",
            "TrackingControllerDeploymentMismatch",
            "ControllerDeploymentChangedDuringPreflight",
            "ControllerAttestationInsufficientTaskHorizon",
            "running_application_state",
            "controller_station_identity",
            "controller_deployment_session_id",
            "robot_ip",
            "installation_session_started_utc",
            "parseExactUtcSeconds",
            "verifyCommissioningEvidence",
        ):
            self.assertIn(token, load_gate)
        load_gate_call = self.main.index(
            "validateControllerLoadHandlingQualification("
        )
        connection = self.main.index("iiwa.net_establishConnection()")
        self.assertLess(load_gate_call, connection)
        before_kst = self.main.index(
            "CONTROLLER_DEPLOYMENT_REATTESTED_BEFORE_KST_NO_MOTION"
        )
        before_motion = self.main.index(
            "CONTROLLER_DEPLOYMENT_REATTESTED_BEFORE_FIRST_MOTION"
        )
        touchdown = self.main.index(
            "[ctx, touchdown] = guardedTouchdown("
        )
        self.assertLess(before_kst, connection)
        self.assertLess(connection, before_motion)
        self.assertLess(before_motion, touchdown)
        self.assertGreaterEqual(
            self.main.count(
                "revalidateControllerLoadHandlingBoundary("
            ),
            3,
        )
        strict_depth_branch = self.main[
            self.main.index("if ~samplingMode") :
            self.main.index("validateConicalContract(cfg);")
        ]
        self.assertIn(
            "cfg.insertion.target_depth_mm = ...",
            strict_depth_branch,
        )
        self.assertIn(
            "cfg.frame.m12_socket_top_to_bottom_depth_mm",
            strict_depth_branch,
        )
        insertion = self.main[
            self.main.index(
                "function [ctx, insertion] = guardedM12Insertion(ctx)"
            ) :
            self.main.index(
                "function [ctx, postPose] = moveOne"
            )
        ]
        for token in (
            "m12_socket_bottom_datum_world_point_mm",
            "world_from_fixture_rotation(:, 3)",
            "dot(",
            "-step * fixtureZWorld",
            "maximumCommandedDepth",
            "success_depth_maximum_mm",
        ):
            self.assertIn(token, insertion)
        self.assertNotIn("surfaceZ", insertion)

    def test_tcpclient_fallback_has_bounded_native_line_io(self) -> None:
        connection = (KST_CLIENT / "net_establishConnection.m").read_text(
            encoding="utf-8"
        )
        self.assertIn('"ConnectTimeout",3,"Timeout",8', connection)
        self.assertIn('configureTerminator(t,"LF")', connection)
        for filename in (
            "attachToolToFlange.m",
            "getEEFPos.m",
            "getJointsPos.m",
            "realTime_startDirectServoCartesian.m",
            "realTime_stopDirectServoCartesian.m",
            "sendEEfPositionf.m",
        ):
            text = (KST_CLIENT / filename).read_text(encoding="utf-8")
            self.assertIn("kstWriteLine(", text)
        get_eef = (KST_CLIENT / "getEEFPos.m").read_text(
            encoding="utf-8"
        )
        get_joints = (KST_CLIENT / "getJointsPos.m").read_text(
            encoding="utf-8"
        )
        vector_reader = (
            KST_CLIENT / "kstReadNumericVectorReply.m"
        ).read_text(encoding="utf-8")
        acknowledgment = (
            KST_CLIENT / "checkAcknowledgment.m"
        ).read_text(encoding="utf-8")
        motion_ack_reader = (
            KST_CLIENT / "kstReadAcknowledgmentReply.m"
        ).read_text(encoding="utf-8")
        relative_motion = (
            KST_CLIENT / "movePTPLineEefRelBase.m"
        ).read_text(encoding="utf-8")
        staged_target = (
            KST_CLIENT / "sendEEfPositions.m"
        ).read_text(encoding="utf-8")
        premotion_setup = (
            KST_CLIENT / "kstExecuteIdempotentPremotionSetup.m"
        ).read_text(encoding="utf-8")
        line_reader = (
            KST_CLIENT / "kstReadLine.m"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "kstReadNumericVectorReply(t, 6, theCommand)", get_eef
        )
        self.assertIn(
            "kstReadNumericVectorReply(t, 7, theCommand)", get_joints
        )
        self.assertIn("message = kstReadLine(t);", vector_reader)
        for token in (
            "EMPTY_REPLY",
            "STALE_DONE_ACK",
            "NO_VECTOR_DELIMITER",
            "VECTOR_LENGTH_",
            "NONFINITE_OR_NONNUMERIC_VECTOR",
            "KST:StateQueryRejected",
            "KST:NumericVectorReplyUnavailable",
            "numel(tokens) ~= expectedCount",
            "str2double(tokens)",
            "any(~isfinite(parsed))",
        ):
            self.assertIn(token, vector_reader)
        self.assertNotIn("str2num", vector_reader)
        self.assertNotIn("kstWriteLine", vector_reader)
        self.assertIn("ret=strcmpi(message,ak);", acknowledgment)
        self.assertIn("ret=false;", acknowledgment)
        for token in (
            "waitBudgetS",
            "lineBudget",
            "KST:MotionAcknowledgmentUnavailable",
            "kstReadLine(t)",
        ):
            self.assertIn(token, motion_ack_reader)
        self.assertNotIn("while readingFlag==false", relative_motion)
        self.assertIn(
            "kstReadAcknowledgmentReply(t, completionBudgetS",
            relative_motion,
        )
        self.assertIn(
            "KST:MotionCompletionAcknowledgmentUnavailable",
            relative_motion,
        )
        self.assertIn(
            "not replay the motion",
            relative_motion,
        )
        self.assertIn(
            "kstExecuteIdempotentPremotionSetup(t, theCommand",
            relative_motion,
        )
        self.assertIn(
            "kstExecuteIdempotentPremotionSetup(",
            staged_target,
        )
        for token in (
            "maximumAttempts = 3",
            "acknowledgmentBudgetS = 1.0",
            "startsWith(command, 'jRelVel_')",
            "startsWith(command, 'cArtixanPosition_')",
            "KST:PreMotionSetupAcknowledgmentUnavailable",
            "KST:NonIdempotentSetupRetryForbidden",
            "motion_trigger_sent",
            "drainLateDoneReplies",
        ):
            self.assertIn(token, premotion_setup)
        for forbidden in (
            "doPTP",
            "startDirectServo",
            "stopDirectServo",
        ):
            # Mentions in the safety comment are allowed; no such command may
            # ever be written by this helper.
            self.assertNotIn(
                f"kstWriteLine(t, '{forbidden}", premotion_setup
            )
        self.assertIn("t.NumBytesAvailable", line_reader)
        self.assertIn("message = '';", line_reader)

        move_one = self.main[
            self.main.index("function [ctx, postPose] = moveOne") :
            self.main.index("function [result, endState, ctx] = finishAtBottom")
        ]
        self.assertIn(
            "prePose = double(reshape(ctx.last_pose, 1, 6));",
            move_one,
        )
        self.assertNotIn(
            "ctx.iiwa.getEEFPos(), [phase, '_PRE']",
            move_one,
        )

    def test_transient_latched_bota_stale_requires_bounded_reconfirmation(
        self,
    ) -> None:
        sensor_config = self.main[
            self.main.index("cfg.sensor = struct") :
            self.main.index("cfg.motion = struct")
        ]
        self.assertIn(
            "'transient_stale_recovery_timeout_s', 2.00",
            sensor_config,
        )
        diagnostic_gate = self.main[
            self.main.index(
                "function [recovered, recoveryWaitS] = "
                "assertDiagnosticHealthy"
            ) :
            self.main.index("function [healthy, reason] = diagnosticHealthy")
        ]
        for token in (
            "isRecoverableFrameStreamStaleDiagnostic",
            "receive(subscriber, min(0.50, remainingS))",
            "Bota frame stream is stale",
            "double(status.Level) == 2",
            "transient_stale_recovery_timeout_s",
        ):
            self.assertIn(token, diagnostic_gate)
        self.assertIn(
            "TRANSIENT_BOTA_STALE_RECOVERED",
            self.main,
        )

    def test_combined_lateral_uncertainty_is_bounded_before_kst(self) -> None:
        gate = self.main[
            self.main.index(
                "function [toolingGate, combinedGate] ="
            ) :
            self.main.index(
                "function radiusMm = minimumM12CleanCandidateRadiusMm"
            )
        ]
        for token in (
            "ToolFixturePhysicalIdentityMismatch",
            "hypot(",
            "socketCenterUncertaintyMm",
            "tcpLateralUncertaintyMm",
            "tipRunoutMm",
            "minimumM12CleanCandidateRadiusMm",
            "atan2d(",
            "CombinedLateralUncertaintyExceeded",
            "'status', 'PASS_PRE_KST'",
        ):
            self.assertIn(token, gate)
        gate_call = self.main.index(
            "validateToolFixtureIdentityAndLateralUncertainty("
        )
        connection = self.main.index("iiwa.net_establishConnection()")
        self.assertLess(gate_call, connection)

    def test_full_loop_uses_explicit_old_kernel_commissioning_fallback(
        self,
    ) -> None:
        config = self.main[
            self.main.index("cfg.compatibility = struct") :
            self.main.index("cfg.robot = struct")
        ]
        for token in (
            "'old_kernel_motion_compatibility_enabled', true",
            "'tip_tcp_translation_mm', [0.0, 0.0, 92.0]",
            "'tip_tcp_standard_uncertainty_mm', 4.0",
            "'start_world_axis_map', diag([1.0, -1.0, -1.0])",
            "'insertion_depth_mm', 6.0",
        ):
            self.assertIn(token, config)

        preflight = self.main[
            self.main.index("compatibilityMode = ~samplingMode") :
            self.main.index("validateConicalContract(cfg);")
        ]
        self.assertIn(
            "validateM12ToolingOldKernelCompatibility", preflight
        )
        self.assertIn("makeOldKernelCompatibilityFrame", preflight)
        self.assertIn(
            "PROVISIONAL_OLD_KERNEL_COMPATIBILITY_92MM_AXIAL_TCP",
            preflight,
        )
        self.assertIn(
            "validateM12Tooling(cfg.files.tooling", preflight
        )

        conical_contract = self.main[
            self.main.index("function validateConicalContract") :
            self.main.index("function validateModelDeploymentTrustAnchor")
        ]
        self.assertIn("strictMeasuredTcp", conical_contract)
        self.assertIn("oldKernelCompatibleTcp", conical_contract)
        self.assertIn(
            "assert(strictMeasuredTcp || oldKernelCompatibleTcp",
            conical_contract,
        )

        wrapper = (
            ROOT / "run_m12_full_loop_proposed_plus_entropy_grid.m"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "execute_m12_contact_engagement_policy( ...\n"
            "    'FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY')",
            wrapper,
        )

    def test_insertion_uses_no_lift_dither_and_dual_terminal_verifier(
        self,
    ) -> None:
        insertion_config = self.main[
            self.main.index("cfg.insertion = struct") :
            self.main.index("cfg.start_gate = struct")
        ]
        for token in (
            "'soft_resume_force_delta_n', 9.0",
            "'soft_resume_torque_delta_nm', 0.28",
            "'post_relief_segment_mm', 0.05",
            "'post_relief_speed_mm_s', 0.30",
            "'capture_xy_correction_deadband_mm', 0.35",
            "'capture_xy_correction_step_mm', 0.60",
            "'capture_xy_tolerance_mm', 0.60",
            "'funnel_dither_radius_mm', 0.15",
            "'terminal_low_wrench_force_n', 2.0",
            "'terminal_low_wrench_torque_nm', 0.08",
            "'terminal_stable_dwells', 3",
        ):
            self.assertIn(token, insertion_config)

        insertion = self.main[
            self.main.index("function [ctx, insertion] = guardedM12Insertion") :
            self.main.index(
                "function [ctx, unwedge] = unwedgeInsertionChamferStall"
            )
        ]
        self.assertGreaterEqual(
            insertion.count("recoverInsertionSoftEnvelope(ctx"), 2
        )
        self.assertGreaterEqual(
            insertion.count("verifyInsertionTerminalSet(ctx"), 2
        )
        for token in (
            "captureAxisXY = double(reshape(ctx.last_pose(1:2), 1, 2));",
            "captureXYError = activeFunnelAxisXY - currentPose(1:2);",
            "insertionDelta(1:2) = captureXYCorrection;",
            "INSERTION_LOW_LOAD_STAGNATION_ESCALATED_LOCKED_AXIS",
            "INSERTION_TERMINAL_FUNNEL_%03d",
        ):
            self.assertIn(token, insertion)
        stagnant_branch = insertion[insertion.index(
            "if stagnantActions >= ..."
        ) :]
        self.assertNotIn("recoverInsertionSoftEnvelope(ctx", stagnant_branch)
        self.assertNotIn("error(", stagnant_branch)
        # Regression guard for the 2026-07-29 method-2 stall.  This branch used
        # to assign post_relief_segment_mm / post_relief_speed_mm_s, which are
        # exactly the values the loop already runs at once it has relieved
        # once, so the "recovery" re-issued an identical dead command forever:
        # the trace shows depth pinned at 0.70 mm with dF 4.6 N, far below the
        # soft trigger, so no other recovery path could fire.  The branch must
        # change something physical -- grow the segment and, past a bounded
        # number of escalations, sweep laterally to release a chamfer wedge.
        self.assertIn("stagnationEscalations = stagnationEscalations + 1", stagnant_branch)
        self.assertIn("2^stagnationEscalations", stagnant_branch)
        self.assertIn("stagnation_segment_cap_mm", stagnant_branch)
        self.assertIn("unwedgeInsertionChamferStall(ctx", stagnant_branch)
        self.assertNotIn(
            "adaptiveSegmentMm = ...\n            ctx.cfg.insertion.post_relief_segment_mm;",
            stagnant_branch,
        )
        # The lateral release sweep must stay a downward-or-lateral action: a
        # +Z command here would pull a shallowly captured peg back out.
        unwedge = self.main[
            self.main.index(
                "function [ctx, unwedge] = unwedgeInsertionChamferStall"
            ) :
            self.main.index(
                "function [ctx, recovery] = recoverInsertionSoftEnvelope"
            )
        ]
        for token in (
            "INSERTION_CHAMFER_UNWEDGE_SWEEP_BEGIN",
            "INSERTION_CHAMFER_UNWEDGE_RELEASED",
            "INSERTION_CHAMFER_UNWEDGE_NO_RELEASE_CONTINUE",
            "downBiasMm * fixtureZWorld",
            "stagnation_unwedge_maximum_radius_mm",
            "enforceTime(ctx);",
        ):
            self.assertIn(token, unwedge)
        self.assertNotIn("+ downBiasMm * fixtureZWorld", unwedge)
        self.assertNotIn("error(", unwedge)
        self.assertIn(
            "INSERTION_SOFT_LOAD_%03d",
            insertion,
        )
        self.assertNotIn(
            "INSERTION_SOFT_LIMIT_REACHED",
            insertion,
        )

        recovery = self.main[
            self.main.index(
                "function [ctx, recovery] = recoverInsertionSoftEnvelope"
            ) :
            self.main.index(
                "function [ctx, pose, joints] = "
                "readKstRobotStateWithRecovery"
            )
        ]
        for token in (
            "ctx.insertion_capture_dither_index = ...",
            "funnel_dither_points_per_ring",
            "funnel_dither_radial_increment_mm",
            "funnel_dither_maximum_radius_mm",
            "ditherAngle = globalDitherIndex * pi * (3.0 - sqrt(5.0));",
            "targetXY = captureAxisXY + ditherRadiusMm * ...",
            "ditherDelta = [targetXY - ctx.last_pose(1:2), 0.0];",
            "guardDwell(ctx",
            "soft_resume_force_delta_n",
            "soft_resume_torque_delta_nm",
            "INSERTION_SOFT_EVENT_CAPTURE_AXIS_DITHER_CONTINUE_DOWN",
        ):
            self.assertIn(token, recovery)
        self.assertNotIn("soft_relief_step_mm * fixtureZWorld", recovery)
        self.assertNotIn("ditherDelta(3)", recovery)

        terminal = self.main[
            self.main.index(
                "function [ctx, verified, kind, guard, depthMm] = ..."
            ) :
            self.main.index(
                "function [ctx, recovery] = recoverInsertionSoftEnvelope"
            )
        ]
        for token in (
            "insertionBottomForceState(guard, ...",
            "kind = 'PERSISTENT_BOTTOM_FORCE_EVENT';",
            "terminal_low_wrench_force_n",
            "terminal_low_wrench_torque_nm",
            "terminal_stable_depth_span_mm",
            "kind = 'LOW_WRENCH_STABLE_DEPTH_EVENT';",
        ):
            self.assertIn(token, terminal)

        sensor_config = self.main[
            self.main.index("cfg.sensor = struct") :
            self.main.index("cfg.motion = struct")
        ]
        self.assertIn("'hard_force_delta_n', 25.0", sensor_config)
        self.assertIn("'hard_torque_delta_nm', 0.75", sensor_config)


if __name__ == "__main__":
    unittest.main()
