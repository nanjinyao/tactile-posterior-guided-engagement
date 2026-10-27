from __future__ import annotations

import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"


class TestM12ConicalLowForceGuardStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")

    def section(self, start: str, end: str) -> str:
        begin = self.main.index(start)
        return self.main[begin : self.main.index(end, begin)]

    def test_thresholds_are_run_noise_and_target_scaled(self) -> None:
        config = self.section(
            "cfg.conical = struct(",
            "cfg.visualization = struct(",
        )
        for token in (
            "'target_normal_force_n', 3.5",
            "'normal_force_deadband_n', 0.50",
            "'contact_admittance_mm_per_n_s', 0.20",
            "'maximum_z_press_speed_mm_s', 0.70",
            "'maximum_z_press_offset_mm', 0.70",
            "'force_relief_hold_n', 12.0",
            "'force_relief_resume_n', 9.0",
            "'force_relief_fast_trigger_n', 18.0",
            "'force_relief_fast_step_mm', 0.10",
            "'low_force_guard_enabled', true",
            "'low_force_hold_target_fraction', 0.25",
            "'low_force_hold_total_force_target_fraction', 0.50",
            "'low_force_recover_target_fraction', 0.55",
            "'low_force_noise_sigma_multiplier', 6.0",
            "'low_force_persistent_samples', 5",
            "'low_force_recover_samples', 3",
            "'low_force_recover_total_force_target_fraction', 0.70",
            "'low_force_maximum_reacquire_mm', 0.25",
            "'low_force_maximum_resume_drop_mm', 0.25",
            "'low_force_maximum_episode_s', 9.0",
            "'low_force_maximum_total_s', 60.0",
            "'low_force_obstruction_force_target_fraction', 3.0",
            "'low_force_obstruction_moment_target_fraction', 1.50",
            "'low_force_obstruction_persistent_samples', 3",
        ):
            self.assertIn(token, config)
        threshold = self.section(
            "function thresholds = conicalLowForceThresholds",
            "function [ctx, outcome, previousCommandWallS",
        )
        for token in (
            "ctx.baseline_force_std_n",
            "ctx.baseline_torque_std_nm",
            "target_normal_force_n",
            "low_force_hold_target_fraction * targetN",
            "low_force_recover_target_fraction * targetN",
            "low_force_recover_total_force_target_fraction *",
            "targetN * ctx.cfg.tool.flange_to_tip_mm / 1000.0",
        ):
            self.assertIn(token, threshold)

    def test_readme_matches_weak_direction_and_conical_force_contract(
        self,
    ) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for token in (
            "`real_multisize_direction_v1` 的三 seed real-first posterior",
            "模型拥有方向、状态 posterior、",
            "弱但完整、新鲜、相位覆盖充分的 360° 指纹仍发出连续缩短的",
            "不会成为 validity veto",
            "deadband 从 `0.75` 收紧到 `0.50 N`",
            "admittance 从 `0.15` 提到 `0.20 mm/(N·s)`",
            "`0.50 mm/s, 0.50 mm` 提到 `0.70 mm/s, 0.70 mm`",
            "`≥18 N` 时冻结当前锥扫相位",
            "一次 `+0.10 mm` 卸载",
            "任务硬停止统一为 `25 N / 0.75 Nm`",
            "完整停止且模型推理返回后，额外批量更新一个 learned-decision 窗口",
            "soft torque trigger/resume 使用 `0.50/0.40 Nm`",
        ):
            self.assertIn(token, readme)

    def test_recoverable_contact_search_uses_broad_soft_envelopes_only(
        self,
    ) -> None:
        sensor = self.section(
            "cfg.sensor = struct(",
            "cfg.motion = struct(",
        )
        for token in (
            "'hard_force_delta_n', 25.0",
            "'hard_torque_delta_nm', 0.75",
            "'insertion_soft_force_delta_n', 16.0",
            "'insertion_soft_torque_delta_nm', 0.50",
        ):
            self.assertIn(token, sensor)

        motion = self.section(
            "cfg.motion = struct(",
            "cfg.search = struct(",
        )
        self.assertIn("'maximum_up_excursion_mm', 2.25", motion)

        surface = self.section(
            "cfg.surface = struct(",
            "cfg.conical = struct(",
        )
        for token in (
            "'target_normal_force_n', 4.0",
            "'soft_relief_trigger_force_n', 10.0",
            "'soft_relief_resume_force_n', 7.0",
            "'capture_soft_force_delta_n', 12.0",
            "'capture_soft_torque_delta_nm', 0.40",
            "'force_continuation_min_n', 1.25",
            "'force_continuation_max_n', 9.0",
            "'xy_segment_mm', 1.20",
            "'speed_fast_mm_s', 10.0",
            "'speed_nominal_mm_s', 6.0",
            "'speed_cautious_mm_s', 3.0",
            "'force_correction_enter_low_n', 2.25",
            "'force_correction_enter_high_n', 5.75",
            "'maximum_transport_steps', 200",
            "'maximum_excess_transport_actions', 24",
            "'maximum_force_settle_steps', 3",
            "'maximum_stagnant_z_steps', 3",
            "'maximum_stagnant_xy_steps', 3",
            "'endpoint_xy_tolerance_mm', 0.20",
            "'transport_normal_force_min_n', 2.5",
            "'transport_normal_force_max_n', 5.5",
            "'negative_observation_normal_min_n', 1.25",
            "'negative_observation_normal_max_n', 7.5",
        ):
            self.assertIn(token, surface)

    def test_point19_spike_enters_bounded_fast_relief_below_hard_stop(
        self,
    ) -> None:
        cone = self.section(
            "function [ctx, cone] = runConicalFingerprint",
            "function thresholds = conicalLowForceThresholds",
        )
        fast_relief = (
            "if sample.force_delta_n >= ...\n"
            "            ctx.cfg.conical.force_relief_fast_trigger_n"
        )
        phase_hold = (
            "if sample.force_delta_n >= ...\n"
            "            ctx.cfg.conical.force_relief_hold_n"
        )
        self.assertIn(fast_relief, cone)
        self.assertIn(
            "zOffsetMm + ...\n"
            "                ctx.cfg.conical.force_relief_fast_step_mm",
            cone,
        )
        self.assertLess(cone.index(fast_relief), cone.index(phase_hold))

        point19_peak_n = 22.1535677
        self.assertGreaterEqual(point19_peak_n, 18.0)
        self.assertLess(point19_peak_n, 25.0)
        one_tick_rate_relief_mm = 1.0 / 40.0
        self.assertAlmostEqual(
            one_tick_rate_relief_mm + 0.10,
            0.125,
        )

    def test_phase_hold_freezes_xy_orientation_and_bounds_downward_motion(
        self,
    ) -> None:
        hold = self.section(
            "function [ctx, outcome, previousCommandWallS",
            "function [ctx, finalCommand] = accountConicalStreamCommand",
        )
        self.assertIn(
            "ctx.cfg.conical.low_force_persistent_samples", hold
        )
        self.assertIn(
            "lastTotalForceN >= thresholds.recover_total_force_n", hold
        )
        self.assertIn("commandPose(1:2) = initialCommandPose(1:2);", hold)
        self.assertIn("commandPose(4:6) = initialCommandPose(4:6);", hold)
        self.assertIn(
            "ctx.cfg.conical.low_force_maximum_reacquire_mm - reacquireMm",
            hold,
        )
        self.assertIn("'QUICK_RECONTACT_RESUME_TAGGED'", hold)
        self.assertIn("'APERTURE_CANDIDATE'", hold)
        self.assertIn(
            "ConicalLowForceRecoveredOutsideResumeLane", hold
        )
        self.assertNotIn(
            "if reacquireMm <= ...\n"
            "                ctx.cfg.conical.low_force_maximum_resume_drop_mm",
            hold,
        )

    def test_low_force_observation_clock_reclassifies_without_abort(
        self,
    ) -> None:
        hold = self.section(
            "function [ctx, outcome, previousCommandWallS",
            "function [ctx, finalCommand] = accountConicalStreamCommand",
        )
        self.assertIn(
            "if ticks > maximumTicks || toc(episodeClock) > ...",
            hold,
        )
        self.assertIn(
            "'OBSERVATION_WINDOW_AMBIGUOUS_NO_LIFT_REQUALIFICATION'",
            hold,
        )
        self.assertIn("outcomeName = 'APERTURE_CANDIDATE';", hold)
        self.assertNotIn("M12Demo:ConicalLowForceEpisodeClock", hold)

    def test_low_force_hold_requires_low_normal_and_low_total_load(self) -> None:
        cone = self.section(
            "function [ctx, cone] = runConicalFingerprint",
            "function thresholds = conicalLowForceThresholds",
        )
        self.assertIn(
            "measuredNormalN <= lowForceThresholds.hold_normal_n && ...",
            cone,
        )
        self.assertIn(
            "sample.force_delta_n <= ...\n"
            "                lowForceThresholds.hold_total_force_n",
            cone,
        )
        thresholds = self.section(
            "function thresholds = conicalLowForceThresholds",
            "function [ctx, outcome, previousCommandWallS",
        )
        self.assertIn("holdTotalForceN < recoverTotalForceN", thresholds)
        self.assertIn("'hold_total_force_n', holdTotalForceN", thresholds)
        hold = self.section(
            "function [ctx, outcome, previousCommandWallS",
            "function [ctx, finalCommand] = accountConicalStreamCommand",
        )
        self.assertIn("normalRecovered || totalRecovered", hold)
        self.assertIn("'trigger_normal_force_n', triggerNormalN", hold)
        self.assertIn("'trigger_total_force_n', triggerTotalForceN", hold)
        self.assertIn("'recovery_channel', recoveryChannel", hold)
        enters_hold = lambda normal, total: normal <= 0.875 and total <= 1.75
        self.assertTrue(enters_hold(0.0, 0.0))
        self.assertFalse(enters_hold(0.0, 3.0))
        self.assertFalse(enters_hold(1.0, 1.0))

    def test_lateral_or_torque_obstruction_holds_xyz_and_requalifies(
        self,
    ) -> None:
        handoff = self.section(
            "function handleConicalLowForceObstructionNoLift",
            "function [ctx, outcome, previousCommandWallS, "
            "maximumCommandGapS] = ...\n"
            "        executeConicalEarlyUprightReturn",
        )
        self.assertIn("classification handoff, not a motion primitive", handoff)
        self.assertIn("Keep XYZ fixed", handoff)
        self.assertIn("ConicalLowForceObstructionCueLost", handoff)
        self.assertNotIn("sendEEfPositionf", handoff)
        self.assertNotIn("receiveWrench", handoff)
        self.assertNotIn("commandPose(3) =", handoff)
        self.assertNotIn("error(", handoff)
        hold = self.section(
            "function [ctx, outcome, previousCommandWallS",
            "function [ctx, finalCommand] = accountConicalStreamCommand",
        )
        self.assertIn("obstructionSamples = obstructionSamples + 1;", hold)
        self.assertIn(
            "low_force_obstruction_persistent_samples", hold
        )
        self.assertIn("handleConicalLowForceObstructionNoLift", hold)
        self.assertIn(
            "outcomeName = 'APERTURE_CANDIDATE';", hold
        )
        self.assertIn("break;", hold)

    def test_early_return_is_smooth_fixed_xyz_and_histories_are_truncated(
        self,
    ) -> None:
        upright = self.section(
            "function [ctx, outcome, previousCommandWallS, "
            "maximumCommandGapS] = ...\n"
            "        executeConicalEarlyUprightReturn",
            "function rotation = conicalRotationAtPhaseTilt",
        )
        self.assertIn("commandPose = startCommandPose;", upright)
        self.assertIn("startTiltDeg * (1.0 - u)", upright)
        self.assertNotIn("commandPose(3) =", upright)
        cone = self.section(
            "function [ctx, cone] = runConicalFingerprint",
            "function thresholds = conicalLowForceThresholds",
        )
        self.assertIn("executedScheduleMask", cone)
        self.assertIn("executedIndices = find(executedScheduleMask);", cone)
        self.assertIn("'post_stop_pose_mm_rad', postPose", cone)
        self.assertIn(
            "physicalScheduleCompleted = logical(stopAck) && ...", cone
        )
        self.assertIn(
            "~apertureCandidate && all(executedScheduleMask)", cone
        )
        self.assertIn(
            "'schedule_completed', physicalScheduleCompleted", cone
        )
        self.assertIn(
            "'physical_schedule_completed', physicalScheduleCompleted",
            cone,
        )
        self.assertIn(
            "'direction_fingerprint_valid', directionFingerprintValid",
            cone,
        )
        self.assertIn(
            "'routing_fingerprint_valid', routingFingerprintValid",
            cone,
        )
        self.assertIn(
            "'training_fingerprint_valid', trainingFingerprintValid",
            cone,
        )

    def test_routing_validity_requires_recovered_low_force_and_quality(
        self,
    ) -> None:
        config = self.section(
            "cfg.conical = struct(",
            "cfg.visualization = struct(",
        )
        for token in (
            "'routing_quality_minimum', 0.10",
            "'routing_ramp_episode_log_penalty', 0.02",
            "'routing_scan_episode_log_penalty', 0.12",
            "'routing_low_force_wall_reference_s', 3.0",
            "'routing_low_force_wall_log_penalty', 0.10",
            "'routing_reacquire_log_penalty', 0.10",
        ):
            self.assertIn(token, config)

        cone = self.section(
            "function [ctx, cone] = runConicalFingerprint",
            "function thresholds = conicalLowForceThresholds",
        )
        for token in (
            "executedScanMask = strcmp(executedStage, 'SCAN');",
            "lowForceRampEpisodes",
            "lowForceScanEpisodes",
            "routingQualityFactors = struct(",
            "'ramp_episode', exp(",
            "'scan_episode', exp(",
            "'low_force_wall', exp(",
            "'reacquire', exp(",
            "routingFingerprintQuality = prod(cell2mat(",
            "struct2cell(routingQualityFactors)))",
            "onlyRecoveredLowForce = ...",
            "lowForceQuickRecontacts == lowForceEpisodes",
            "routingFingerprintValid = directionFingerprintValid && ...",
            "onlyRecoveredLowForce && ...",
            "routingFingerprintQuality >= ...",
            "ctx.cfg.conical.routing_quality_minimum",
            "trainingFingerprintValid = routingFingerprintValid && ...",
            "lowForceEpisodes == 0",
            "forceReliefEpisodes == 0",
            "fastForceReliefKicks",
            "'fast_force_relief_kicks', fastForceReliefKicks",
            "'routing_fingerprint_quality', routingFingerprintQuality",
            "'routing_fingerprint_quality_factors', routingQualityFactors",
            "'low_force_ramp_episodes', lowForceRampEpisodes",
            "'low_force_scan_episodes', lowForceScanEpisodes",
            "'low_force_quick_recontacts', lowForceQuickRecontacts",
        ):
            self.assertIn(token, cone)
        self.assertNotIn("lowForceScanEpisodes == 0", cone)

        def classify(
            *,
            complete: bool,
            aperture: bool,
            low_force_episodes: int,
            quick_recontacts: int,
            routing_quality: float,
            minimum_quality: float = 0.10,
        ) -> tuple[bool, bool]:
            direction_valid = complete and not aperture
            routing_valid = (
                direction_valid
                and quick_recontacts == low_force_episodes
                and routing_quality >= minimum_quality
            )
            training_valid = (
                routing_valid and low_force_episodes == 0
            )
            return routing_valid, training_valid

        self.assertEqual(
            classify(
                complete=True,
                aperture=False,
                low_force_episodes=0,
                quick_recontacts=0,
                routing_quality=1.0,
            ),
            (True, True),
        )
        representative_recovered_quality = math.exp(
            -0.02 - 0.12 - 0.10 - 0.10
        )
        self.assertEqual(
            classify(
                complete=True,
                aperture=False,
                low_force_episodes=2,
                quick_recontacts=2,
                routing_quality=representative_recovered_quality,
            ),
            (True, False),
            "Recovered RAMP/SCAN episodes are quality penalties, not vetoes.",
        )
        self.assertEqual(
            classify(
                complete=True,
                aperture=False,
                low_force_episodes=1,
                quick_recontacts=0,
                routing_quality=0.95,
            ),
            (False, False),
            "An unrecovered low-force episode must still veto routing.",
        )
        self.assertEqual(
            classify(
                complete=True,
                aperture=False,
                low_force_episodes=3,
                quick_recontacts=3,
                routing_quality=0.099,
            ),
            (False, False),
            "Recovered evidence below the quality floor must abstain.",
        )
        self.assertEqual(
            classify(
                complete=False,
                aperture=True,
                low_force_episodes=1,
                quick_recontacts=1,
                routing_quality=0.95,
            ),
            (False, False),
        )

    def test_weak_complete_full_loop_fingerprint_reaches_model_action(
        self,
    ) -> None:
        main_route = self.section(
            "if conical.aperture_candidate",
            "if ~capture.found",
        )
        learned = self.section(
            "function learned = inferFrozenConicalModel",
            "function [ctx, capture, diagnostics] = runOutwardArchimedeanSearch",
        )
        for token in (
            "assert(conical.direction_fingerprint_valid",
            "learned = inferFrozenConicalModel(conical, cfg, paths);",
            "modelPosterior = modelGridPosterior(",
            "learnedShortcut = planLearnedShortcut(",
            "activeGridRoutingDirection = ...",
            "learnedShortcut.direction_world_xy;",
            "learnedModelUsedForControl = true;",
        ):
            self.assertIn(token, main_route)
        self.assertNotIn(
            "activeLearnedDirection(learned, 'POST_CONE_ACTIVE_ROUTE')",
            main_route,
        )
        self.assertNotIn("training_fingerprint_valid", main_route)
        self.assertIn("cone.direction_fingerprint_valid", learned)
        self.assertIn("~cone.aperture_candidate", learned)
        self.assertIn(
            "Recovered low-force episodes reduce sample_quality", learned
        )
        self.assertIn(
            "not remove bounded high-level motion authority", learned
        )
        self.assertNotIn("training_fingerprint_valid", learned)

    def test_incomplete_capture_cone_never_reaches_direction_consumers(
        self,
    ) -> None:
        main_route = self.section(
            "captureRequalified = false;",
            "result.fingerprint_phase_rad = conical.phase_rad;",
        )
        requal = main_route.index("requalifyConicalApertureCandidate")
        rescan = main_route.index(
            "runConicalFingerprint(ctx, capturePose", requal
        )
        self.assertLess(requal, rescan)
        self.assertIn(
            "while conical.aperture_candidate && ~captureRequalified",
            main_route,
        )
        self.assertIn(
            "CONTACT_CONE_AMBIGUOUS_LOW_FORCE_RESCAN_IN_PLACE",
            main_route,
        )
        self.assertNotIn("inferFrozenConicalModel", main_route)
        self.assertNotIn(
            "ConicalApertureCandidateNotRequalified", main_route
        )

        aperture_branch = self.section(
            "if conical.aperture_candidate",
            "\n    else\n    assert(conical.direction_fingerprint_valid",
        )
        self.assertIn(
            "BYPASSED_INCOMPLETE_CONE_CAPTURE_HANDOFF", aperture_branch
        )
        self.assertIn(
            "ConicalApertureRescanInvariant", aperture_branch
        )
        self.assertIn(
            "result.learned_inference_attempted = false;",
            aperture_branch,
        )
        self.assertNotIn("inferFrozenConicalModel", aperture_branch)
        learned = self.section(
            "function learned = inferFrozenConicalModel",
            "function [ctx, capture, diagnostics] = runOutwardArchimedeanSearch",
        )
        self.assertIn("~cone.aperture_candidate", learned)

    def test_sampling_keeps_recovered_low_force_but_quarantines_contact_loss(
        self,
    ) -> None:
        campaign = self.section(
            "function [result, endState, ctx] = runM12SamplingCampaign",
            "function attempt = makeSamplingAttempt",
        )
        self.assertIn("if cone.aperture_candidate", campaign)
        self.assertIn(
            "CONTACT_LOSS_OR_APERTURE_DURING_CONE", campaign
        )
        self.assertIn("cone.low_force_episodes > 0", campaign)
        self.assertIn("cone.force_relief_episodes > 0", campaign)
        self.assertIn(
            "attempt.weak_but_complete_fingerprint", campaign
        )
        self.assertIn("~cone.training_fingerprint_valid", campaign)
        quality_gate_start = campaign.index(
            "if ~attempt.cone_post_stop_state_known"
        )
        quality_gate_end = campaign.index(
            "attempt.operational_status = ...",
            quality_gate_start,
        )
        self.assertNotIn(
            "training_fingerprint_valid",
            campaign[quality_gate_start:quality_gate_end],
        )
        self.assertNotIn("inferFrozenConicalModel(cone", campaign)

    def test_direct_retry_routes_low_force_capture_before_direction(self) -> None:
        direct = self.section(
            "function [ctx, capture, diagnostics] = runImprovedDirectRetry3",
            "function record = directAttemptTemplate",
        )
        cone_call = direct.index("runConicalFingerprint(ctx")
        capture_branch = direct.index(
            "while currentCone.aperture_candidate", cone_call
        )
        inference = direct.index(
            "currentLearned = inferFrozenConicalModel(", cone_call
        )
        direction = direct.index(
            "learnedPosteriorModeForDirectAttempt(currentLearned",
            cone_call,
        )
        self.assertLess(capture_branch, inference)
        self.assertLess(capture_branch, direction)
        self.assertLess(inference, direction)
        self.assertIn("requalifyConicalApertureCandidate", direct)
        self.assertIn(
            "DIRECT_LOW_FORCE_CONE_AMBIGUOUS_RESCAN_SAME_CONTACT", direct
        )
        self.assertIn(
            "runConicalFingerprint(ctx, capturePose", direct
        )
        self.assertIn(
            "LOW_FORCE_CONE_PHYSICAL_CAPTURE_HANDOFF_NO_DIRECTION",
            direct,
        )


if __name__ == "__main__":
    unittest.main()
