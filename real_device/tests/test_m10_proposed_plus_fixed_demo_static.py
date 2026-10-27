"""Static contracts for the fixed, in-sample-adjacent M10 Proposed+ pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


REAL_DIR = Path(__file__).resolve().parent.parent
ENGINE_PATH = REAL_DIR / "execute_m12_contact_engagement_policy.m"
LAUNCHER_PATH = REAL_DIR / "run_m10_proposed_plus_entropy_trial.m"
PILOT_PATH = REAL_DIR / "configs" / "m10_fixed_demo_pilot_authorization_v1.json"
INSTALLATION_PATH = REAL_DIR / "configs" / "m10_fixed_demo_installation_v1.json"
EXPECTED_PILOT_SHA256 = (
    "8f09bdc86949ee448d29795ca7100347be4b865083f93dfa7761eae9834dc825"
)


def circular_grid_count(pitch_mm: float, radius_mm: float) -> int:
    integer_radius = math.ceil(radius_mm / pitch_mm)
    return sum(
        math.hypot(pitch_mm * ix, pitch_mm * iy) <= radius_mm + 1.0e-12
        for ix in range(-integer_radius, integer_radius + 1)
        for iy in range(-integer_radius, integer_radius + 1)
    )


def matlab_function_body(source: str, function_name: str) -> str:
    """Return one MATLAB local-function block without parsing nested ends."""
    starts = [match.start() for match in re.finditer(r"(?m)^function\b", source)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(source)
        body = source[start:end]
        lines = body.splitlines()
        header_lines = [lines[0]]
        while header_lines[-1].rstrip().endswith("..."):
            header_lines.append(lines[len(header_lines)])
        header = "\n".join(header_lines)
        if re.search(rf"\b{re.escape(function_name)}\s*\(", header):
            return body
    raise AssertionError(f"MATLAB function {function_name!r} was not found")


class TestM10ProposedPlusFixedDemoStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ENGINE_PATH.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        cls.pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
        cls.installation = json.loads(
            INSTALLATION_PATH.read_text(encoding="utf-8")
        )

    def test_launcher_is_one_m10_proposed_method_at_one_fixed_offset(self) -> None:
        for token in (
            "function result = run_m10_proposed_plus_entropy_trial()",
            "FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY_M10",
            "M10_PROPOSED_PLUS_FIXED_DEMO_PRIVATE_TRIAL_V1",
            "demoOffsetFixtureMm = [ ...\n"
            "    4.590151046897575, 0.3008543944586581];",
            "'campaign_center_world_xy_mm', []",
            "'public_entrypoint', 'run_m10_proposed_plus_entropy_trial'",
            "maximumLauncherAttempts = 3;",
            "clear execute_m12_contact_engagement_policy",
        ):
            self.assertIn(token, self.launcher)
        self.assertNotIn("selectedMethodIndex", self.launcher)
        self.assertNotIn("FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY'", self.launcher)
        self.assertNotRegex(self.launcher, r"\b(input|questdlg|menu)\s*\(")

    def test_private_m10_mode_and_fresh_center_are_enforced_before_kst(self) -> None:
        entry = self.engine[
            : self.engine.index(
                "function [mode, resumeParentRunDir] = parseExecutionMode"
            )
        ]
        parse_call = entry.index(
            "pairedTrialSpec = parsePairedTrialSpec(executionMode, varargin{:});"
        )
        private_gate = entry.index(
            "'M12Demo:M10FixedDemoPrivateSpecRequired'"
        )
        kst_connection = entry.index("iiwa.net_establishConnection()")
        self.assertLess(parse_call, private_gate)
        self.assertLess(private_gate, kst_connection)

        # Local MATLAB functions are defined after the entry body, so their
        # source offsets do not represent execution order.  The parser call
        # above runs before KST and delegates to the validator; pin the
        # fresh-centre assertion in that validator instead of comparing its
        # later definition offset to KST.
        parser = matlab_function_body(self.engine, "parsePairedTrialSpec")
        self.assertIn("validatePairedTrialSpec(varargin{2}, mode)", parser)
        validator = matlab_function_body(
            self.engine, "validatePairedTrialSpec"
        )
        fresh_center_gate = validator.index(
            "'M12Demo:M10FreshTaughtCenterRequired'"
        )
        campaign_center_read = validator.index(
            "campaignCenter = double(reshape("
        )
        self.assertLess(campaign_center_read, fresh_center_gate)
        self.assertIn("assert(isempty(campaignCenter)", validator)
        self.assertIn("double(spec.trial_index) >= 1", validator)
        self.assertIn("double(spec.trial_index) <= 3", validator)
        for token in (
            "M10_PROPOSED_PLUS_FIXED_DEMO_PRIVATE_TRIAL_V1",
            "M10_FIXED_PROPOSED_PLUS_NEAR_A003P75_R4P600_V1",
            "M10-CLEAN_CANDIDATE-SUPPORTED-A003p75",
            "M10_FIXED_PROPOSED_PLUS_DEMO_20260730_V1",
            "norm(offset - expectedOffset) <= 1e-9",
            "assert(isempty(campaignCenter)",
        ):
            self.assertIn(token, self.engine)

    def test_m10_geometry_derives_depth_grid_and_motion_bounds(self) -> None:
        pitch_mm = min(
            2.0 * (0.5 * (8.54 - 8.0)),
            (2.0 / 3.0) * (0.25 * (10.54 - 8.0)),
        )
        capture_radius_mm = 0.25 * (10.54 - 8.0)
        connected_leg_mm = min(1.0, 2.0 * pitch_mm)
        coverage_pitch_mm = min(1.0, 2.0 * (capture_radius_mm - 0.20))
        direct_ray_mm = min(5.70, 4.20 + 2.0 * capture_radius_mm)
        supported_unwedge_mm = min(
            1.60, 0.5 * 10.54 - 6.4 / math.sqrt(3.0) - 0.20
        )
        self.assertAlmostEqual(pitch_mm, 0.42333333333333334)
        self.assertEqual(circular_grid_count(pitch_mm, 4.8), 405)
        self.assertAlmostEqual(connected_leg_mm, 0.8466666666666667)
        self.assertAlmostEqual(coverage_pitch_mm, 0.87)
        self.assertAlmostEqual(direct_ray_mm, 5.47)
        self.assertAlmostEqual(supported_unwedge_mm, 1.3749582772, places=9)
        for token in (
            "cadSocketDepthMm = 5.0;",
            "'success_depth_minimum_mm', cadSocketDepthMm - 0.30",
            "'success_depth_maximum_mm', cadSocketDepthMm + 0.20",
            "'maximum_grid_nodes', activeGridNodeCount",
            "activeResidualConnectedLegMm = min(1.00, 2.00 * m12GridPitchMm);",
            "activeCoveragePitchMm = min(1.00, ...",
            "activeInsertionUnwedgeMaximumRadiusMm = min( ...",
        ):
            self.assertIn(token, self.engine)

    def test_m10_real_run_tuning_is_mode_scoped_and_keeps_hard_guards(
        self,
    ) -> None:
        make_config = matlab_function_body(self.engine, "makeConfig")
        sensor = make_config[
            make_config.index("cfg.sensor = struct") :
            make_config.index("cfg.motion = struct")
        ]
        insertion = make_config[
            make_config.index("cfg.insertion = struct") :
            make_config.index("cfg.start_gate = struct")
        ]
        for token in (
            "'hard_force_delta_n', 25.0",
            "'hard_torque_delta_nm', 0.75",
            "'insertion_soft_force_delta_n', 16.0",
        ):
            self.assertIn(token, sensor)
        for token in (
            "'controller_tuning_revision', "
            "'SHARED_INSERTION_BASELINE_V1'",
            "'terminal_low_wrench_force_n', 2.0",
            "'terminal_low_wrench_torque_nm', 0.08",
            "'terminal_stable_dwells', 3",
        ):
            self.assertIn(token, insertion)

        profile = matlab_function_body(
            self.engine, "applySamplingSizeProfile"
        )
        tuned_start = profile.index(
            "if isM10FixedDemo\n"
            "    runLocalInstallationRecord = canonicalPath("
        )
        tuned_end = profile.index(
            "else\n"
            "    runLocalInstallationRecord = canonicalPath(",
            tuned_start,
        )
        tuned = profile[tuned_start:tuned_end]
        for token in (
            "M10_CHAMFER_FORCE14_TERMINAL_TORQUE010_V2",
            "cfg.sensor.insertion_soft_force_delta_n = 14.0;",
            "cfg.insertion.terminal_low_wrench_torque_nm = 0.10;",
        ):
            self.assertIn(token, tuned)
        self.assertNotIn(
            "cfg.sensor.hard_force_delta_n =", tuned
        )
        self.assertNotIn(
            "cfg.sensor.hard_torque_delta_nm =", tuned
        )

        initial_result = matlab_function_body(
            self.engine, "makeInitialResult"
        )
        manifest = matlab_function_body(self.engine, "makeManifest")
        for token in (
            "insertion_controller_tuning_revision",
            "insertion_soft_force_delta_n",
            "insertion_terminal_low_wrench_force_n",
            "insertion_terminal_low_wrench_torque_nm",
        ):
            self.assertIn(token, initial_result)
            self.assertIn(token, manifest)

    def test_tuning_validation_appends_after_success_without_erasing_it(
        self,
    ) -> None:
        main = self.launcher[
            : self.launcher.index("function validateM10DemoCampaign")
        ]
        for token in (
            "M10_CHAMFER_FORCE14_TERMINAL_TORQUE010_V2",
            "SHARED_INSERTION_BASELINE_V1",
            "tuningRevisionAlreadyAttempted",
            "double(campaign.attempt_count) == 1",
            "M10Demo:TuningValidationAlreadyComplete",
            "campaign.status = 'TUNING_VALIDATION_READY';",
            "campaign.tuning_validation_source_run_directory",
            "'controller_tuning_revision', controllerTuningRevision",
        ):
            self.assertIn(token, main)
        reopen = main[
            main.index("if strcmp(char(campaign.status), 'COMPLETE')") :
            main.index(
                "assert(~strcmp(char(campaign.status), 'RUNNING')"
            )
        ]
        self.assertIn(
            "campaign.attempt_records = attemptRecords;", reopen
        )
        self.assertIn(
            "writeM10DemoCheckpoint(campaignPath, campaign);", reopen
        )

        normalize = matlab_function_body(
            self.launcher, "m10DemoCampaignAttemptRecords"
        )
        self.assertIn(
            "if ~isfield(records, 'controller_tuning_revision')",
            normalize,
        )
        self.assertIn(
            "char(legacyControllerTuningRevision)", normalize
        )

        validator = matlab_function_body(
            self.engine, "validatePairedTrialSpec"
        )
        self.assertIn("'controller_tuning_revision'", validator)
        self.assertIn(
            "M10_CHAMFER_FORCE14_TERMINAL_TORQUE010_V2", validator
        )
        entry = self.engine[
            : self.engine.index(
                "function [mode, resumeParentRunDir] = parseExecutionMode"
            )
        ]
        self.assertIn(
            "M12Demo:M10ControllerTuningRevisionMismatch", entry
        )

    def test_m10_fail_stop_defaults_and_arms_before_air_preposition(self) -> None:
        make_context = matlab_function_body(self.engine, "makeContext")
        default_envelope = make_context[
            make_context.index(
                "'m10_fixed_demo_support_envelope', struct("
            ) :
            make_context.index("'dashboard', struct(")
        ]
        for token in (
            "'enabled', false",
            "'center_world_xy_mm', [NaN, NaN]",
            "'maximum_commanded_radius_mm', NaN",
            "'maximum_physical_radius_mm', NaN",
            "'endpoint_tracking_allowance_mm', NaN",
            "'safety_fail_stop_only', false",
            "'posterior_input', false",
            "'target_ranking_input', false",
            "'steering_input', false",
        ):
            self.assertIn(token, default_envelope)

        preposition = matlab_function_body(
            self.engine, "executePairedAirPreposition"
        )
        schema_gate = preposition.index(
            "'M10_PROPOSED_PLUS_FIXED_DEMO_PRIVATE_TRIAL_V1'"
        )
        arm = preposition.index(
            "ctx = enableM10FixedDemoSupportEnvelope("
        )
        preposition_motion = preposition.index(
            "[ctx, ~] = moveAirSegmented("
        )
        measured_endpoint_guard = preposition.index(
            "'PAIRED_AIR_PREPOSITION_MEASURED_ENDPOINT'"
        )
        self.assertLess(schema_gate, arm)
        self.assertLess(arm, preposition_motion)
        self.assertLess(preposition_motion, measured_endpoint_guard)

    def test_m10_fail_stop_guards_commands_and_measured_robot_state(self) -> None:
        move_one = matlab_function_body(self.engine, "moveOne")
        move_command_guard = move_one.index(
            "ctx, ctx.last_command_pose(1:2), phase, 'COMMAND_TARGET'"
        )
        blocking_command = move_one.index(
            "ctx.iiwa.movePTPLineEefRelBase("
        )
        self.assertLess(move_command_guard, blocking_command)

        conical = matlab_function_body(self.engine, "runConicalFingerprint")
        directservo_command_guard = conical.index(
            "assertConicalCommandEnvelope(ctx, commandPose, runKind);"
        )
        directservo_command = conical.index(
            "ctx.iiwa.sendEEfPositionf(num2cell(commandPose));"
        )
        self.assertLess(directservo_command_guard, directservo_command)

        conical_envelope = matlab_function_body(
            self.engine, "assertConicalCommandEnvelope"
        )
        self.assertIn(
            "ctx, commandPose(1:2), phase, 'COMMAND_TARGET'",
            conical_envelope,
        )

        measured_state = matlab_function_body(self.engine, "assertRobotState")
        measured_guard = measured_state.index(
            "ctx, pose(1:2), phase, 'MEASURED_STATE'"
        )
        ordinary_workspace_guard = measured_state.index(
            "orientationDrift = max("
        )
        self.assertLess(measured_guard, ordinary_workspace_guard)

    def test_m10_fail_stop_geometry_is_safety_only_and_not_policy_input(
        self,
    ) -> None:
        arm = matlab_function_body(
            self.engine, "enableM10FixedDemoSupportEnvelope"
        )
        for token in (
            "expectedMaximumPhysicalRadiusMm = 5.792458277186394;",
            "expectedTrackingAllowanceMm = 0.600;",
            "expectedMaximumCommandedRadiusMm = 5.192458277186394;",
            "'safety_fail_stop_only', true",
            "'posterior_input', false",
            "'target_ranking_input', false",
            "'steering_input', false",
        ):
            self.assertIn(token, arm)

        fail_stop = matlab_function_body(
            self.engine, "assertM10FixedDemoSupportEnvelope"
        )
        for token in (
            "case 'COMMAND_TARGET'",
            "limitMm = envelope.maximum_commanded_radius_mm;",
            "case 'MEASURED_STATE'",
            "limitMm = envelope.maximum_physical_radius_mm;",
            "~envelope.posterior_input",
            "~envelope.target_ranking_input",
            "~envelope.steering_input",
        ):
            self.assertIn(token, fail_stop)

        forbidden = (
            "m10_fixed_demo_support_envelope",
            "assertM10FixedDemoSupportEnvelope",
            "5.792458277186394",
            "0.600",
            "5.192458277186394",
        )
        posterior_ranking_steering_functions = (
            "learnedPosteriorModeForDirectAttempt",
            "activeLearnedDirection",
            "modelGridPosterior",
            "planLearnedShortcut",
            "selectPosteriorResidualTarget",
            "posteriorCorridorHalfAngleDeg",
            "buildDirectionalCadCorridor",
            "selectExpectedInformationNode",
            "selectCadCoverageNode",
            "conditionPosteriorOnDirectionalSector",
        )
        for function_name in posterior_ranking_steering_functions:
            body = matlab_function_body(self.engine, function_name)
            for token in forbidden:
                self.assertNotIn(
                    token,
                    body,
                    f"{token!r} leaked into {function_name}",
                )

    def test_launcher_bounds_trial_indices_and_persists_attempt_audit(
        self,
    ) -> None:
        main = self.launcher[
            : self.launcher.index("function validateM10DemoCampaign")
        ]
        for token in (
            "maximumLauncherAttempts = 3;",
            "assert(double(campaign.attempt_count) < maximumLauncherAttempts",
            "upcomingAttempt = double(campaign.attempt_count) + 1;",
            "'trial_index', upcomingAttempt",
            "'attempt_records', m10DemoEmptyAttemptRecords()",
            "result.m10_fixed_demo_attempt_records = campaign.attempt_records;",
        ):
            self.assertIn(token, main)

        append_record = main.index(
            "attemptRecords(end + 1) = m10DemoNewAttemptRecord("
        )
        persist_records = main.index(
            "campaign.attempt_records = attemptRecords;", append_record
        )
        checkpoint = main.index(
            "writeM10DemoCheckpoint(campaignPath, campaign);",
            persist_records,
        )
        invoke_engine = main.index(
            "result = execute_m12_contact_engagement_policy("
        )
        self.assertLess(append_record, persist_records)
        self.assertLess(persist_records, checkpoint)
        self.assertLess(checkpoint, invoke_engine)

        catch_start = main.index("catch trialError")
        catch_end = main.index("    rethrow(trialError);\nend", catch_start)
        catch_body = main[catch_start:catch_end]
        self.assertIn(
            "campaign.attempt_records = attemptRecords;", catch_body
        )
        self.assertIn(
            "writeM10DemoCheckpoint(campaignPath, campaign);", catch_body
        )
        success_body = main[catch_end:]
        self.assertIn(
            "attemptRecords(recordIndex).status = 'COMPLETE';", success_body
        )
        self.assertIn(
            "campaign.attempt_records = attemptRecords;", success_body
        )
        self.assertIn(
            "writeM10DemoCheckpoint(campaignPath, campaign);", success_body
        )

        validator = matlab_function_body(
            self.launcher, "validateM10DemoCampaign"
        )
        self.assertIn(
            "isequal([attemptRecords.attempt_index], 1:attemptCount)",
            validator,
        )
        self.assertIn(
            "numel(unique(string( ...", validator
        )

    def test_pilot_is_hash_pinned_and_does_not_claim_general_promotion(self) -> None:
        self.assertEqual(
            hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest(),
            EXPECTED_PILOT_SHA256,
        )
        self.assertEqual(
            self.pilot["authorization_id"],
            "M10_FIXED_PROPOSED_PLUS_DEMO_20260730_V1",
        )
        self.assertTrue(self.pilot["authorization"]["fixed_demo_only"])
        self.assertFalse(self.pilot["authorization"]["capture_control"])
        self.assertFalse(self.pilot["authorization"]["insertion_control"])
        self.assertFalse(self.pilot["authorization"]["terminal_verification"])
        self.assertIn(
            "not general M10 full-loop promotion",
            self.pilot["control_boundary"]["claim_limit"],
        )
        self.assertIn(EXPECTED_PILOT_SHA256, self.launcher)
        self.assertIn(EXPECTED_PILOT_SHA256, self.engine)

    def test_installation_record_discloses_unverified_metrology(self) -> None:
        self.assertEqual(self.installation["size"], "m10")
        self.assertFalse(
            self.installation["operator_report"]["independently_verified_by_software"]
        )
        self.assertEqual(
            self.installation["printed_cad"]["socket_depth_mm"], 5.0
        )
        self.assertIn(
            "general M10 full-loop promotion",
            self.installation["not_authority_for"],
        )


if __name__ == "__main__":
    unittest.main()
