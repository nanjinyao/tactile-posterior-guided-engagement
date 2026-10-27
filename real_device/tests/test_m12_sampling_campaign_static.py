from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"
WRAPPER = ROOT / "run_m12_sampling_conical_fingerprints_26pt.m"
RESUME_WRAPPER = (
    ROOT / "run_m12_sampling_conical_fingerprints_26pt_resume.m"
)
PLAN = ROOT / "configs" / "m12_printed_cad_conical_sampling_v2.json"
FRAME = ROOT / "configs" / "fixture_frame_commissioning_v1.json"
DASHBOARD = ROOT / "m12_sampling_topview_dashboard.m"
KST_CLIENT = (
    ROOT.parents[1]
    / "reference"
    / "KST-Kuka-Sunrise-Toolbox-master"
    / "Matlab_client"
)
GENERATOR = ROOT / "tools" / "generate_m12_printed_cad_sampling_plan.py"
CANONICAL_PLAN_PNG = (
    ROOT / "docs" / "m12_printed_cad_sampling_top_view.png"
)
OPERATOR_PLAN_PNG = ROOT / "m12采样规划图.png"

EXPECTED_PARTITION_COUNTS = {
    "clean_candidate": 16,
    "holdout": 4,
    "quarantine": 4,
    "sentinel": 2,
}
EXPECTED_CLEAN_CONTACT_STATE_COUNTS = {
    "supported_search": 6,
    "double_overhang": 6,
    "outer_partial_support": 4,
}
EXPECTED_CLEAN_CONE_SUBTYPE_COUNTS = {
    "landing_supported_cone_transition": 6,
    "landing_double_overhang": 6,
    "landing_mouth_clear_cone_transition": 2,
    "full_frustum_mouth_clear": 2,
}
EXPECTED_SOURCE_KEYS = {
    "design_parameters",
    "generated_cad_manifest",
    "printed_m12_integrated_peg_step",
    "printed_hardware_easy_v2_socket_step",
}


class TestM12SamplingCampaignStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.resume_wrapper = RESUME_WRAPPER.read_text(encoding="utf-8")
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.frame = json.loads(FRAME.read_text(encoding="utf-8"))
        cls.dashboard = (
            DASHBOARD.read_text(encoding="utf-8")
            if DASHBOARD.exists()
            else cls.main
        )

    def _numeric_field(self, source: str, field: str) -> float:
        match = re.search(
            rf"'{re.escape(field)}'\s*,\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            source,
        )
        self.assertIsNotNone(match, f"missing MATLAB field {field}")
        assert match is not None
        return float(match.group(1))

    def _matlab_function(self, source: str, name: str) -> str:
        declaration = re.search(
            rf"^function(?:(?:[^\n]*\.\.\.\n\s*)*)"
            rf"[^\n]*\b{re.escape(name)}\b[^\n]*$",
            source,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(
            declaration,
            f"missing MATLAB function {name}",
        )
        assert declaration is not None
        following = re.search(
            r"^function\b",
            source[declaration.end() :],
            flags=re.MULTILINE,
        )
        end = (
            len(source)
            if following is None
            else declaration.end() + following.start()
        )
        return source[declaration.start() : end]

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_wrapper_selects_only_the_shared_sampling_mode(self) -> None:
        self.assertIn(
            "execute_m12_contact_engagement_policy( ...\n"
            "    'DATASET_M12_CONICAL_FINGERPRINTS_26PT')",
            self.wrapper,
        )
        self.assertNotIn("movePTP", self.wrapper)
        self.assertNotIn("sendEEf", self.wrapper)

    def test_resume_entrypoint_and_parent_validation_are_read_only(
        self,
    ) -> None:
        self.assertIn(
            "function result = "
            "run_m12_sampling_conical_fingerprints_26pt_resume("
            "previousRunDir)",
            self.resume_wrapper,
        )
        self.assertIn(
            "execute_m12_contact_engagement_policy( ...\n"
            "    'DATASET_M12_CONICAL_FINGERPRINTS_26PT', "
            "previousRunDir)",
            self.resume_wrapper,
        )
        for token in (
            "if nargin == 0 || isempty(previousRunDir)",
            "selectLatestResumableM12Campaign()",
            "M12 AUTO RESUME",
            "sampling_campaign.json",
            "m12_run_local_conical_fingerprints_26pt_v2",
            "attempt_count_planned",
            "hasCompletedCone(campaign.attempts)",
        ):
            self.assertIn(token, self.resume_wrapper)
        for forbidden in (
            "movePTP",
            "sendEEf",
            "DirectServo",
            "KST(",
        ):
            self.assertNotIn(forbidden, self.resume_wrapper)

        parser = self._matlab_function(
            self.main, "parseExecutionMode"
        )
        self.assertIn("resumeParentRunDir = ''", parser)
        self.assertIn("if nargin == 2", parser)
        self.assertIn(
            "'DATASET_M12_CONICAL_FINGERPRINTS_26PT'",
            parser,
        )
        self.assertIn("M12Demo:SamplingResumeInput", parser)

        loader = self._matlab_function(
            self.main, "loadM12SamplingResumeParent"
        )
        for token in (
            "resumeFrame = makeOldKernelCompatibilityFrame(",
            "resumeCfg = cfg",
            "resumeCfg.sampling.conical_phase_pose_tracking = struct(",
            "COLLECTION_ONLY_PRE_POST_MEASURED_POSE",
            "'qualified_for_training_candidates', false",
            "sampling_campaign.json",
            "SamplingResumeParentScope",
            "SamplingResumeParentStatus",
            "SamplingResumeExecutionOrder",
            "SamplingResumeCadHash",
            "SamplingResumeAttemptIndices",
            "SamplingResumeAttemptIdentity",
            "SamplingResumeConeQuality",
            "validateSamplingResumeEvidencePath",
            "completedConeIndices",
            "pendingExecutionIndices",
            "campaign_lineage_id",
            "dataset_group_key",
            "inherited_attempts",
            "rawAttempts(coneMask)",
        ):
            self.assertIn(token, loader)
        frame_build = loader.index(
            "resumeFrame = makeOldKernelCompatibilityFrame("
        )
        cfg_build = loader.index("resumeCfg = cfg", frame_build)
        parent_read = loader.index(
            "parent = jsondecode(fileread(summaryPath))"
        )
        self.assertLess(frame_build, cfg_build)
        self.assertLess(cfg_build, parent_read)

        complete_rows = (
            "completeEvidenceRawIndices = "
            "reshape(find(coneMask), 1, [])"
        )
        self.assertIn(complete_rows, loader)
        self.assertIn(
            "for rawIndex = completeEvidenceRawIndices",
            loader,
        )
        self.assertNotIn(
            "for rawIndex = find(coneMask)",
            loader,
            "MATLAB for iterates columns; find(coneMask) is normally a "
            "column and must be reshaped to a row of scalar indices",
        )

        parent_center = loader.index("resumeCenterPose = double(")
        center_xy = loader.index(
            "parent.run_local_hole_center_world_xy_mm",
            parent_center,
        )
        plan_rebase = loader.index(
            "resumePlan = rebaseM12SamplingPlanToRunLocalStart(",
            center_xy,
        )
        normalize = loader.index(
            "normalizedAttempts = normalizeSamplingResumeAttempts(",
            plan_rebase,
        )
        self.assertLess(parent_center, center_xy)
        self.assertLess(center_xy, plan_rebase)
        self.assertLess(plan_rebase, normalize)
        normalize_call = loader[
            normalize : loader.index(";", normalize) + 1
        ]
        self.assertIn(
            "rawAttempts(coneMask), resumePlan, resumeCfg",
            normalize_call,
        )
        for forbidden in (
            "movePTP",
            "sendEEf",
            "DirectServo",
            "writeJson",
            "writeText",
            "save(",
        ):
            self.assertNotIn(forbidden, loader)
        evidence_validator = self._matlab_function(
            self.main, "validateSamplingResumeEvidencePath"
        )
        for token in (
            "dataRoot, originRunSessionId, evidencePath",
            "dataRoot = canonicalPath(dataRoot)",
            "parentRunDir = canonicalPath(fullfile(dataRoot, "
            "originRunSessionId))",
            "startsWith(parentRunDir, [dataRoot, filesep])",
            "strcmp(originDirectoryName, originRunSessionId)",
            "startsWith(evidencePath, [parentRunDir, filesep])",
            "isfile(evidencePath)",
            "isValidSha256(expectedHash)",
            "strcmp(sha256File(evidencePath), expectedHash)",
        ):
            self.assertIn(token, evidence_validator)
        for forbidden in (
            "movePTP",
            "sendEEf",
            "DirectServo",
            "writeJson",
            "writeText",
            "save(",
        ):
            self.assertNotIn(forbidden, evidence_validator)

        parent_load = self.main.index(
            "samplingResume = loadM12SamplingResumeParent("
        )
        main_try = self.main.index("\ntry\n")
        kst_preflight = self.main.index("kstClientDir = fullfile(")
        self.assertLess(
            parent_load,
            main_try,
            "resume parent validation must construct its own read-only "
            "frame/config before entering the motion execution try-block",
        )
        self.assertLess(
            parent_load,
            kst_preflight,
            "resume evidence must fail before KST connection setup",
        )

        normalizer = self._matlab_function(
            self.main, "normalizeSamplingResumeAttempts"
        )
        self.assertIn(
            "attempt.origin_run_session_id = "
            "...\n        samplingResumeAttemptOriginRunSessionId(",
            normalizer,
        )
        self.assertIn(
            "CONE_COMPLETE_RETRACT_UNVERIFIED_PARENT",
            normalizer,
        )
        self.assertNotIn(
            "attempt.safe_retract_completed = true",
            normalizer,
            "a child may close the lifecycle after a fresh clear check, "
            "but it must not forge the parent's original retract evidence",
        )

    def test_multigeneration_resume_preserves_origin_scope_and_recovery(
        self,
    ) -> None:
        loader = self._matlab_function(
            self.main, "loadM12SamplingResumeParent"
        )
        origin_resolver = self._matlab_function(
            self.main, "samplingResumeAttemptOriginRunSessionId"
        )
        evidence_validator = self._matlab_function(
            self.main, "validateSamplingResumeEvidencePath"
        )
        normalizer = self._matlab_function(
            self.main, "normalizeSamplingResumeAttempts"
        )

        for token in (
            "immediateParentRunSessionId",
            "rawAttempt.origin_run_session_id",
            "if ~isempty(recordedOrigin)",
            "originRunSessionId = recordedOrigin",
            "assert(~isempty(originRunSessionId)",
            "SamplingResumeEvidenceOrigin",
        ):
            self.assertIn(token, origin_resolver)
        self.assertLess(
            origin_resolver.index(
                "originRunSessionId = scalarTextOrEmpty("
            ),
            origin_resolver.index(
                "originRunSessionId = recordedOrigin"
            ),
        )

        self.assertIn(
            "attemptOriginRunSessionIds{rawIndex} = "
            "...\n        samplingResumeAttemptOriginRunSessionId(",
            loader,
        )
        origin_scoped_calls = re.findall(
            r"validateSamplingResumeEvidencePath\(dataRoot,\s*\.\.\.\s*"
            r"attemptOriginRunSessionIds\{rawIndex\}",
            loader,
        )
        self.assertEqual(
            len(origin_scoped_calls),
            4,
            "MAT, baseline, settle and cone evidence must each resolve "
            "through the attempt's immutable origin run",
        )
        self.assertNotRegex(
            loader,
            r"validateSamplingResumeEvidencePath\(\s*parentRunDir",
            "multi-generation evidence must not be constrained to only "
            "the immediate parent directory",
        )
        for token in (
            "dataRoot, originRunSessionId, evidencePath",
            "fullfile(dataRoot, originRunSessionId)",
            "startsWith(parentRunDir, [dataRoot, filesep])",
            "strcmp(originDirectoryName, originRunSessionId)",
            "startsWith(evidencePath, [parentRunDir, filesep])",
            "isfile(evidencePath)",
            "isValidSha256(expectedHash)",
            "strcmp(sha256File(evidencePath), expectedHash)",
        ):
            self.assertIn(token, evidence_validator)

        field_copy = normalizer.index(
            "attempt.(fieldName) = rawAttempt.(fieldName)"
        )
        recovery_snapshot = normalizer.index(
            "recoveryClearVerified = "
            "logical(attempt.recovery_clear_verified)"
        )
        lifecycle_snapshot = normalizer.index(
            "motionLifecycleClosed = "
            "logical(attempt.motion_lifecycle_closed)"
        )
        self.assertLess(field_copy, recovery_snapshot)
        self.assertLess(recovery_snapshot, lifecycle_snapshot)
        for token in (
            "attempt.origin_run_session_id = "
            "...\n        samplingResumeAttemptOriginRunSessionId(",
            "attempt.recovery_run_session_id",
            "attempt.recovery_clear_verified_local",
            "attempt.recovery_clear_max_force_n",
            "attempt.recovery_clear_max_torque_nm",
            "SamplingResumeRecoveryProvenance",
            "attempt.recovery_clear_verified = recoveryClearVerified",
            "attempt.motion_lifecycle_closed = safeRetractCompleted ||",
            "(recoveryClearVerified && motionLifecycleClosed)",
        ):
            self.assertIn(token, normalizer)
        for field, reset in (
            ("origin_run_session_id", "''"),
            ("recovery_run_session_id", "''"),
            ("recovery_clear_verified_local", "''"),
            ("recovery_clear_verified", "false"),
            ("motion_lifecycle_closed", "false"),
            ("recovery_clear_max_force_n", "NaN"),
            ("recovery_clear_max_torque_nm", "NaN"),
        ):
            self.assertNotRegex(
                normalizer,
                rf"attempt\.{field}\s*=\s*{re.escape(reset)}",
                f"normalizer must not erase inherited {field}",
            )

    def test_resume_skips_every_completed_cone_before_target_motion(
        self,
    ) -> None:
        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        declaration = campaign.splitlines()[0] + campaign.splitlines()[1]
        self.assertIn("samplingResume", declaration)
        for token in (
            "samplingResume.enabled",
            "samplingResume.inherited_attempts",
            "samplingResume.completed_cone_indices",
            "SAMPLING_RESUME_CHILD_CHECKPOINT_CREATED",
            "SAMPLING_RESUME_START_CLEAR_VERIFIED",
            "SAMPLING_RESUME_POINT_SKIPPED_EXISTING_CONE",
        ):
            self.assertIn(token, campaign)

        loop_start = campaign.index(
            "for executionIndex = 1:numel(plan.execution_points)"
        )
        target = campaign.index(
            "bestEffortSamplingTopviewAction(ctx, ...",
            loop_start,
        )
        self.assertIn(
            "'target', struct(",
            campaign[target : target + 300],
        )
        skip_event = campaign.index(
            "SAMPLING_RESUME_POINT_SKIPPED_EXISTING_CONE",
            loop_start,
        )
        self.assertLess(skip_event, target)
        skip_block = campaign[
            campaign.rfind("if ", loop_start, skip_event) :
            target
        ]
        self.assertIn("completed_cone_indices", skip_block)
        self.assertIn("continue;", skip_block)
        for forbidden in (
            "movePTP",
            "moveOne(",
            "moveAirSegmented(",
            "runConicalFingerprint(",
            "sendEEf",
            "DirectServo",
        ):
            self.assertNotIn(forbidden, skip_block)

        clear_verified = campaign.index(
            "SAMPLING_RESUME_START_CLEAR_VERIFIED"
        )
        self.assertLess(clear_verified, loop_start)
        self.assertIn(
            "retractSamplingUntilClear",
            campaign[:clear_verified],
        )
        self.assertIn(
            "writeSamplingCheckpoint(paths, campaign)",
            campaign[:loop_start],
        )

        baseline_gate = campaign.index(
            "SAMPLING_RESUME_BASELINE_COMPATIBLE_NO_MOTION"
        )
        initial_clear = campaign.index(
            "retractSamplingUntilClear( ...",
            baseline_gate,
        )
        self.assertLess(baseline_gate, initial_clear)
        self.assertLess(initial_clear, clear_verified)
        self.assertLess(clear_verified, loop_start)

        resume_clear_block = campaign[
            campaign.index(
                "if samplingResume.enabled\n"
                "    campaign.resume_start_clear_verified = true"
            ) :
            campaign.index("safeToMoveLaterally = true")
        ]
        for token in (
            "attempts(inheritedIndex).recovery_clear_verified = true",
            "attempts(inheritedIndex).motion_lifecycle_closed = true",
            "CONE_COMPLETE_RECOVERY_CLEAR_VERIFIED_CHILD",
        ):
            self.assertIn(token, resume_clear_block)
        self.assertNotIn(
            "safe_retract_completed = true",
            resume_clear_block,
            "child clear recovery must remain distinct from parent retract",
        )

        top_level_sampling = self.main[
            self.main.index("if samplingMode\n        samplingCenterPose") :
            self.main.index(
                "[result, endState, ctx] = runM12SamplingCampaign("
            )
        ]
        parent_center = top_level_sampling.index(
            "samplingResume.run_local_hole_center_world_xy_mm"
        )
        frame_binding = top_level_sampling.index(
            "[samplingCenterPose(1:2), startPose(3)]",
            parent_center,
        )
        plan_rebase = top_level_sampling.index(
            "rebaseM12SamplingPlanToRunLocalStart( ...",
            frame_binding,
        )
        self.assertLess(parent_center, frame_binding)
        self.assertLess(frame_binding, plan_rebase)

        upsert = self._matlab_function(
            self.main, "appendSamplingAttempt"
        )
        for token in (
            "[campaign.attempts.execution_index]",
            "SamplingCampaignDuplicateExecutionIndex",
            "campaign.attempts(existingIndex, 1) = attempt",
            "refreshSamplingCampaignCounters(campaign)",
        ):
            self.assertIn(token, upsert)

    def test_sampling_entrypoint_authorizes_without_a_second_phrase(
        self,
    ) -> None:
        confirmer = self._matlab_function(
            self.main, "confirmSamplingStart"
        )
        self.assertNotRegex(confirmer, r"\binput\s*\(")
        for forbidden in (
            "captureStationaryStart",
            "getEEFPos",
            "getJointsPos",
            "movePTP",
            "sendEEf",
            "realTime_startDirectServoCartesian",
        ):
            self.assertNotIn(forbidden, confirmer)
        self.assertNotIn("required_operator_phrase", self.main)
        self.assertIn(
            "'interactive_confirmation_required', false", self.main
        )
        self.assertIn(
            "EXPLICIT_DEDICATED_SAMPLING_ENTRYPOINT_INVOCATION",
            self.main,
        )
        self.assertIn(
            "SAMPLING_ENTRYPOINT_AUTHORIZED_AUTOMATIC_CONTINUE_NO_MOTION",
            confirmer,
        )

    def test_default_entry_remains_full_loop(self) -> None:
        parser = self._matlab_function(self.main, "parseExecutionMode")
        self.assertIn(
            "mode = 'FULL_LOOP_PROPOSED_PLUS_LEARNED_ENTROPY';", parser
        )
        branch = self.main.index("if samplingMode\n        [result, endState, ctx]")
        full_touchdown = self.main.index(
            "manualStartXY = startPose(1:2);", branch
        )
        self.assertLess(branch, full_touchdown)
        measured_anchor = self.main.index(
            "centerXY = touchdown.pose_mm_rad(1:2);", full_touchdown
        )
        self.assertLess(full_touchdown, measured_anchor)
        self.assertIn(
            "guardedM12Insertion(ctx)", self.main[measured_anchor:]
        )

    def test_schema2_plan_has_top_level_points_and_required_coverage(
        self,
    ) -> None:
        self.assertEqual(self.plan["schema_version"], 2)
        self.assertEqual(
            self.plan["name"], "m12_printed_cad_conical_sampling_v2"
        )
        self.assertNotIn("sizes", self.plan)
        self.assertIn("points", self.plan)
        points = self.plan["points"]
        self.assertEqual(len(points), 26)
        self.assertEqual(self.plan["attempt_count"], 26)
        self.assertEqual(self.plan["sampling_design"]["attempt_count"], 26)
        counts: dict[str, int] = {}
        clean_contact_state_counts: dict[str, int] = {}
        clean_cone_subtype_counts: dict[str, int] = {}
        ids = set()
        fixture_center = self.plan["fixture_centre_xy_mm"]
        for attempt_index, point in enumerate(points, start=1):
            self.assertEqual(point["attempt_index"], attempt_index)
            counts[point["partition"]] = counts.get(point["partition"], 0) + 1
            ids.add(point["sample_id"])
            dx, dy = point["local_offset_fixture_mm"]
            self.assertGreater(math.hypot(dx, dy), 0.0)
            fixture_x, fixture_y = point["fixture_xy_mm"]
            self.assertAlmostEqual(
                fixture_x, fixture_center[0] + dx, places=8
            )
            self.assertAlmostEqual(
                fixture_y, fixture_center[1] + dy, places=8
            )
            tx, ty = point["true_direction_to_centre_fixture_unit"]
            radius = math.hypot(dx, dy)
            self.assertAlmostEqual(tx, -dx / radius, places=8)
            self.assertAlmostEqual(ty, -dy / radius, places=8)
            metrics = point["nominal_touchdown_geometry"]
            cone_metrics = point[
                "printed_frustum_cone_phase_diagnostics"
            ]
            self.assertEqual(cone_metrics["phase_bins"], 360)
            for field in (
                "maximum_mouth_overlap_fraction",
                "minimum_mouth_overlap_fraction",
                "mouth_overlap_phase_fraction",
                "outer_overhang_phase_fraction",
            ):
                self.assertGreaterEqual(cone_metrics[field], -1e-8)
                self.assertLessEqual(cone_metrics[field], 1.0 + 1e-8)
            self.assertGreater(
                cone_metrics["minimum_projected_area_mm2"], 0.0
            )
            self.assertGreaterEqual(
                cone_metrics["maximum_projected_area_mm2"],
                cone_metrics["minimum_projected_area_mm2"],
            )
            if point["cone_sweep_subtype"] == "full_frustum_mouth_clear":
                self.assertLessEqual(
                    cone_metrics["maximum_mouth_overlap_fraction"], 1e-6
                )
            fractions = [
                metrics["outer_overhang_fraction"],
                metrics["mouth_overlap_fraction"],
                metrics["head_top_support_fraction"],
            ]
            for fraction in fractions:
                self.assertGreaterEqual(fraction, -1e-8)
                self.assertLessEqual(fraction, 1.0 + 1e-8)
            self.assertAlmostEqual(sum(fractions), 1.0, places=6)
            if point["partition"] == "clean_candidate":
                state = point["contact_state"]
                clean_contact_state_counts[state] = (
                    clean_contact_state_counts.get(state, 0) + 1
                )
                subtype = point["cone_sweep_subtype"]
                clean_cone_subtype_counts[subtype] = (
                    clean_cone_subtype_counts.get(subtype, 0) + 1
                )
        self.assertEqual(len(ids), 26)
        self.assertEqual(counts, EXPECTED_PARTITION_COUNTS)
        self.assertEqual(
            self.plan["sampling_design"]["partition_counts"],
            EXPECTED_PARTITION_COUNTS,
        )
        self.assertEqual(
            clean_contact_state_counts,
            EXPECTED_CLEAN_CONTACT_STATE_COUNTS,
        )
        self.assertEqual(
            self.plan["sampling_design"]["clean_contact_state_counts"],
            EXPECTED_CLEAN_CONTACT_STATE_COUNTS,
        )
        self.assertEqual(
            clean_cone_subtype_counts,
            EXPECTED_CLEAN_CONE_SUBTYPE_COUNTS,
        )
        self.assertEqual(
            self.plan["sampling_design"][
                "clean_cone_sweep_subtype_counts"
            ],
            EXPECTED_CLEAN_CONE_SUBTYPE_COUNTS,
        )
        self.assertTrue(
            self.plan["validation"]["all_three_clean_contact_states_present"]
        )
        self.assertEqual(
            self.plan["validation"]["outer_partial_clean_count"], 4
        )
        self.assertTrue(self.plan["claim_boundary"]["no_centre_attempt"])
        loader = self._matlab_function(self.main, "loadM12SamplingPlan")
        for state, expected_count in (
            EXPECTED_CLEAN_CONTACT_STATE_COUNTS.items()
        ):
            self.assertRegex(
                loader,
                rf'contactStates\s*==\s*"{re.escape(state)}"\s*\)\s*'
                rf"==\s*{expected_count}",
                msg=(
                    "the MATLAB runtime gate must accept the generated "
                    f"{state} clean count {expected_count}"
                ),
            )
        self.assertIn(
            "6/6/4 printed-CAD touchdown states",
            loader,
        )

    def test_execution_order_comes_from_json_and_separates_sentinels(
        self,
    ) -> None:
        order_match = re.search(
            r"'execution_attempt_indices'\s*,\s*"
            r"(?:\.\.\.\s*)?\[([^\]]+)\]",
            self.main,
        )
        self.assertIsNone(
            order_match,
            "the MATLAB source must not duplicate the generated JSON route",
        )
        execution = self.plan["execution"]
        configured_order = execution["attempt_indices"]
        self.assertEqual(len(configured_order), 26)
        self.assertEqual(sorted(configured_order), list(range(1, 27)))
        points = self.plan["points"]
        route = [[0.0, 0.0]]
        route.extend(
            points[attempt_index - 1]["local_offset_fixture_mm"]
            for attempt_index in configured_order
        )
        route.append([0.0, 0.0])
        closed_route_mm = sum(
            math.dist(start, end)
            for start, end in zip(route, route[1:])
        )
        source_order_route = [[0.0, 0.0]]
        source_order_route.extend(
            point["local_offset_fixture_mm"] for point in points
        )
        source_order_route.append([0.0, 0.0])
        source_order_closed_route_mm = sum(
            math.dist(start, end)
            for start, end in zip(
                source_order_route, source_order_route[1:]
            )
        )
        self.assertAlmostEqual(
            closed_route_mm, execution["closed_route_distance_mm"], places=6
        )
        self.assertAlmostEqual(
            source_order_closed_route_mm,
            execution["source_order_closed_route_distance_mm"],
            places=6,
        )
        self.assertLess(closed_route_mm, source_order_closed_route_mm)

        execution_position = {
            attempt_index: position
            for position, attempt_index in enumerate(
                configured_order, start=1
            )
        }
        sample_id_to_index = {
            point["sample_id"]: point["attempt_index"] for point in points
        }
        sentinel_count = 0
        for point in points:
            if point["partition"] != "sentinel":
                continue
            sentinel_count += 1
            original_index = sample_id_to_index[point["duplicate_of"]]
            self.assertEqual(
                point["local_offset_fixture_mm"],
                points[original_index - 1]["local_offset_fixture_mm"],
            )
            separation = abs(
                execution_position[point["attempt_index"]]
                - execution_position[original_index]
            )
            self.assertGreaterEqual(separation, 10)
            self.assertEqual(
                separation,
                execution["sentinel_duplicate_execution_separations"][
                    sentinel_count - 1
                ],
            )
        self.assertEqual(sentinel_count, 2)
        self.assertGreaterEqual(
            self.plan["validation"]["sentinel_minimum_execution_separation"],
            10,
        )

        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        loader = self.main[
            self.main.index("function plan = loadM12SamplingPlan") :
            self.main.index(
                "function plan = rebaseM12SamplingPlanToRunLocalStart"
            )
        ]
        ordered_points_are_built = re.search(
            r"(?:executionPoints|plan\.execution_points)\s*=\s*"
            r"(?:points|plan\.points)\s*\(\s*"
            r"(?:executionAttemptIndices|plan\.execution_attempt_indices)"
            r"\s*\)",
            loader,
        )
        self.assertIsNotNone(
            ordered_points_are_built,
            "the generated JSON order must select the plan points",
        )
        self.assertIn(
            "record.execution.attempt_indices",
            loader,
            "the route must be loaded from the schema2 JSON execution block",
        )
        self.assertTrue(
            "plan.execution_points" in campaign
            or re.search(
                r"plan\.points\s*\(\s*"
                r"(?:plan|cfg\.sampling)\.execution_attempt_indices",
                campaign,
            ),
            "the campaign must iterate the fixed-order point sequence",
        )

    def test_photo_projection_is_right_handed_for_upward_fixture_z(self) -> None:
        contract = self.plan["coordinate_contract"]["photo_top_view"]
        self.assertEqual(
            contract["mapping"],
            "u_photo_mm = -x_fixture_mm; "
            "v_photo_up_mm = -y_fixture_mm",
        )
        self.assertEqual(contract["plus_x_fixture_points"], "left")
        self.assertEqual(contract["plus_y_fixture_points"], "down")
        for point in self.plan["points"]:
            x_fixture, y_fixture = point["fixture_xy_mm"]
            u_photo, v_photo = point["photo_uv_mm"]
            self.assertAlmostEqual(u_photo, -x_fixture, places=8)
            self.assertAlmostEqual(v_photo, -y_fixture, places=8)

    def test_plan_is_hash_bound_to_the_active_printed_cad_artifacts(
        self,
    ) -> None:
        validation = self.plan["validation"]
        self.assertTrue(validation["actual_printed_cad_hashes_verified"])
        self.assertTrue(validation["no_simulation_geometry_input"])
        sources = self.plan["source_files"]
        self.assertEqual(set(sources), EXPECTED_SOURCE_KEYS)
        resolved_root = ROOT.resolve()
        for name, record in sources.items():
            relative_path = Path(record["path_from_real_device"])
            self.assertFalse(
                relative_path.is_absolute(),
                f"{name} must be recorded relative to real_device",
            )
            source_path = (ROOT / relative_path).resolve()
            try:
                source_path.relative_to(resolved_root)
            except ValueError:
                self.fail(f"{name} escapes the real_device tree")
            self.assertTrue(source_path.is_file(), f"missing {name}")
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                record["sha256"],
                self._sha256(source_path),
                f"{name} hash is not the active printed-CAD artifact",
            )

        geometry = self.plan["geometry_mm"]
        expected_geometry = {
            "head_outer_diameter": 21.275,
            "socket_mouth_af": 13.0,
            "socket_deep_wall_af": 10.6,
            "socket_chamfer_width": 1.2,
            "socket_chamfer_depth": 1.2,
            "center_through_hole_diameter": 0.5,
            "peg_tip_face_af": 8.0,
            "peg_shaft_af": 10.0,
            "peg_chamfer_radial_setback": 1.0,
            "peg_chamfer_axial_length": 1.0,
        }
        for field, expected in expected_geometry.items():
            self.assertAlmostEqual(geometry[field], expected, places=9)

        loader = self._matlab_function(self.main, "loadM12SamplingPlan")
        for source_field in EXPECTED_SOURCE_KEYS - {
            "printed_m12_integrated_peg_step"
        }:
            self.assertIn(
                f"record.source_files.{source_field}.sha256", loader
            )
        self.assertIn(
            "record.source_files.(cfg.sampling.plan_source_peg_key)",
            loader,
        )
        self.assertIn("sha256File(cfg.files.printed_cad_manifest)", loader)
        self.assertIn("sha256File(cfg.files.active_peg_step)", loader)
        self.assertIn("sha256File(cfg.files.printed_socket_step)", loader)

    def test_sampling_only_lateral_envelope_is_derived_from_the_plan(
        self,
    ) -> None:
        motion = self.plan["motion_contract"]
        maximum_radius = max(
            math.hypot(*point["local_offset_fixture_mm"])
            for point in self.plan["points"]
        )
        self.assertAlmostEqual(
            maximum_radius,
            motion["maximum_planned_radius_mm"],
            places=6,
        )
        required = motion["required_sampling_only_lateral_excursion_mm"]
        minimum_required = (
            maximum_radius
            + motion["measured_landing_tolerance_mm"]
            + motion["air_tracking_tolerance_mm"]
            + motion["additional_workspace_margin_mm"]
        )
        self.assertGreaterEqual(required + 1e-9, minimum_required)
        self.assertLess(required, 15.0)

        sampling_setup = self.main[
            self.main.index("samplingPlan = struct();") :
            self.main.index("runStamp = char(datetime")
        ]
        self.assertRegex(
            sampling_setup,
            r"if samplingMode[\s\S]*?"
            r"samplingPlan\s*=\s*loadM12SamplingPlan\([\s\S]*?"
            r"cfg\.motion\.maximum_lateral_excursion_mm\s*=\s*max\("
            r"[\s\S]*?samplingPlan\."
            r"required_sampling_only_lateral_excursion_mm",
        )
        loader = self._matlab_function(self.main, "loadM12SamplingPlan")
        self.assertIn(
            "record.motion_contract."
            "required_sampling_only_lateral_excursion_mm",
            loader,
        )
        self.assertIn(
            "'required_sampling_only_lateral_excursion_mm'",
            loader,
        )
        motion_configuration = self.main[
            self.main.index("cfg.motion = struct(") :
            self.main.index("cfg.search = struct(")
        ]
        self.assertAlmostEqual(
            self._numeric_field(
                motion_configuration, "maximum_lateral_excursion_mm"
            ),
            10.0,
            msg="the wider envelope must be sampling-only, not global",
        )

    def test_generated_plan_and_operator_figure_are_current(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "printed-CAD generator --check failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )
        self.assertIn(
            "M12 printed-CAD sampling outputs are current",
            completed.stdout,
        )
        self.assertTrue(CANONICAL_PLAN_PNG.is_file())
        self.assertTrue(OPERATOR_PLAN_PNG.is_file())
        self.assertEqual(
            OPERATOR_PLAN_PNG.read_bytes(),
            CANONICAL_PLAN_PNG.read_bytes(),
            "m12采样规划图.png must be the canonical generated M12 CAD PNG",
        )

    def test_tooling_gate_avoids_sampling_requalification_cycle(self) -> None:
        self.assertIn(
            "validateM12Tooling(cfg.files.tooling, cfg, ~samplingMode)",
            self.main,
        )
        tooling = self.main[
            self.main.index("function [geometryValid, payloadComplete, gate]") :
            self.main.index("function [elapsedS, attempts, source]")
        ]
        self.assertRegex(
            tooling,
            r"if requireFullLoopQualification\s+"
            r"assert\(strcmp\(clockingStatus, 'QUALIFIED'\)[\s\S]*?"
            r"lowLoadRequalified",
        )
        self.assertIn(
            "COMMISSIONING_DATA_ONLY_NOT_FULL_LOOP_AUTHORIZATION", tooling
        )
        self.assertIn("geometryCommissioned", tooling)

    def test_fixture_uses_a_measured_full_right_handed_rotation(self) -> None:
        measured = self.frame["measured_world_from_fixture"]
        self.assertIn("rotation_matrix", measured)
        self.assertIsNone(measured["rotation_matrix"])
        self.assertIn(
            "expected_world_from_upright_tcp_rotation_matrix", measured
        )
        self.assertIsNone(
            measured["expected_world_from_upright_tcp_rotation_matrix"]
        )
        loader = self.main[
            self.main.index("function frame = loadFixtureFrameCommissioning") :
            self.main.index("function qualification = loadQualification")
        ]
        for token in (
            "rotation.' * rotation - eye(3)",
            "det(rotation)",
            "matrixYawDeg",
            "matrixTiltDeg",
            "expectedWorldFromUprightTcp",
            "tcpOrthogonalityError",
            "expectedPegAxisErrorDeg",
            "'world_from_fixture_rotation', rotation",
            "PhysicalInstanceIdentityMissing",
            "SAMPLING_QUALIFIED",
        ):
            self.assertIn(token, loader)
        self.assertNotIn(
            "'world_from_fixture_rotation', rotationZ", loader
        )
        instances = self.frame["physical_instances"]
        self.assertIn("physical_tool_instance_id", instances)
        self.assertIn("fixture_instance_id", instances)

    def test_sampling_authorization_follows_all_readiness_gates_and_one_capture(
        self,
    ) -> None:
        ordered = [
            "assertRootStatus(workspaceRoot);",
            "assertNoRosControlConflict(workspaceRoot);",
            "waitForHealthyDiagnostic(diagnosticSub, cfg",
            "waitForFreshWrench(",
            "connectionFlag = iiwa.net_establishConnection();",
            "captureStationaryStart(iiwa, cfg, paths);",
            "bindOldKernelCompatibilityFrameToLiveStart( ...",
            "confirmSamplingStart(iiwa, cfg, paths, startPose",
            "measureBaseline(ctx, 'START_BASELINE');",
            "[result, endState, ctx] = runM12SamplingCampaign( ...",
        ]
        positions = [self.main.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        startup_flow = self.main[
            positions[4] :
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            )
        ]
        self.assertEqual(
            startup_flow.count(
                "captureStationaryStart(iiwa, cfg, paths);"
            ),
            1,
        )
        self.assertEqual(
            startup_flow.count(
                "bindOldKernelCompatibilityFrameToLiveStart( ..."
            ),
            1,
        )

    def test_motion_order_is_lift_clear_xy_touch_cone_retract_clear(self) -> None:
        campaign = self.main[
            self.main.index("function [result, endState, ctx] = runM12SamplingCampaign") :
            self.main.index("function attempt = makeSamplingAttempt")
        ]
        ordered = [
            "SAMPLING_INITIAL_LIFT",
            "SAMPLE_%02d_AIR_XY",
            "SAMPLE_%02d_PRE_TOUCH_CLEAR",
            "acquireSamplingTouchdownWithRetries(",
            "observeSamplingContactBeforeCone(",
            "runConicalFingerprint(ctx",
            "SAMPLE_%02d_RETRACT",
            "attempt.safe_retract_completed = true",
            "SAMPLING_RETURN_HOME_XY",
            "SAMPLING_RETURN_INITIAL_AIR_Z",
        ]
        positions = [campaign.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "cfg.sampling.preferred_retract_mm", self.main
        )
        self.assertIn("retractSamplingUntilClear", campaign)
        self.assertIn("wrench_clear_dwell_count', 2", self.main)
        acquisition = self._matlab_function(
            self.main, "acquireSamplingTouchdownWithRetries"
        )
        self.assertLess(
            acquisition.index("captureSamplingPointBaseline("),
            acquisition.index("guardedSamplingTouchdown("),
        )
        self.assertIn(
            "SAMPLING_POINT_ACQUISITION_RETRY_AFTER_CLEAR_RETRACT",
            acquisition,
        )
        self.assertIn("retractSamplingUntilClear(", acquisition)
        self.assertIn("maximum_point_acquisition_attempts", acquisition)

    def test_sampling_motion_and_xy_lock_parameters_are_fast_and_bounded(
        self,
    ) -> None:
        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        self.assertAlmostEqual(
            self._numeric_field(sampling, "preferred_retract_mm"), 0.75
        )
        maximum_retract = re.search(
            r"'maximum_retract_mm'\s*,\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            sampling,
        )
        if maximum_retract is None:
            self.assertAlmostEqual(
                self._numeric_field(sampling, "minimum_retract_mm"), 3.0
            )
        else:
            self.assertAlmostEqual(float(maximum_retract.group(1)), 3.0)
        self.assertAlmostEqual(
            self._numeric_field(sampling, "air_translation_segment_mm"), 1.50
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "air_translation_speed_mm_s"), 8.0
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "baseline_samples"), 30.0
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "maximum_active_time_s"), 0.0
        )
        self.assertAlmostEqual(
            self._numeric_field(
                sampling, "touchdown_xy_lock_warning_mm"
            ),
            0.15,
        )
        self.assertAlmostEqual(
            self._numeric_field(
                sampling, "maximum_touchdown_xy_lock_error_mm"
            ),
            0.25,
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "touchdown_step_mm"), 0.10
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "touchdown_speed_mm_s"), 1.0
        )
        tracking_tolerance = self._numeric_field(
            sampling, "step_tracking_tolerance_mm"
        )
        tracking_recheck = self._numeric_field(
            sampling, "step_tracking_recheck_acceptance_mm"
        )
        self.assertAlmostEqual(tracking_tolerance, 0.12)
        self.assertAlmostEqual(tracking_recheck, 0.15)
        self.assertGreater(tracking_recheck, tracking_tolerance)
        self.assertLess(tracking_recheck, 0.25)
        self.assertAlmostEqual(
            self._numeric_field(
                sampling, "touchdown_xy_correction_deadband_mm"
            ),
            0.08,
        )
        self.assertAlmostEqual(
            self._numeric_field(
                sampling, "minimum_touchdown_down_progress_mm"
            ),
            0.0,
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "step_tracking_recheck_count"),
            2.0,
        )
        self.assertAlmostEqual(
            self._numeric_field(sampling, "step_tracking_recheck_pause_s"),
            0.08,
        )
        self.assertRegex(
            self.main,
            r"if samplingMode[\s\S]{0,500}"
            r"cfg\.sensor\.baseline_samples\s*=\s*"
            r"cfg\.sampling\.baseline_samples",
        )
        touchdown = self._matlab_function(
            self.main, "guardedSamplingTouchdown"
        )
        self.assertIn(
            "SAMPLING_TOUCHDOWN_CONFIG_DEFAULT_RECOVERED_CONTINUE",
            touchdown,
        )
        self.assertIn("touchdown_xy_lock_warning_mm", touchdown)
        self.assertIn("maximum_touchdown_xy_lock_error_mm", touchdown)
        self.assertIn("step_tracking_tolerance_mm", touchdown)
        move_one = self._matlab_function(self.main, "moveOne")
        for token in (
            "step_tracking_recheck_acceptance_mm",
            "step_tracking_recheck_count",
            "step_tracking_recheck_pause_s",
            "SAMPLING_TOUCHDOWN_TRACKING_RECHECK_BEGIN_NO_MOTION",
            "SAMPLING_TOUCHDOWN_TRACKING_RECHECK_OBSERVED_NO_MOTION",
            "SAMPLING_TOUCHDOWN_TRACKING_RECHECK_ACCEPTED_CONTINUE",
            "TRACKING_REJECTED",
        ):
            self.assertIn(token, move_one)
        recheck = move_one[
            move_one.index(
                "SAMPLING_TOUCHDOWN_TRACKING_RECHECK_BEGIN_NO_MOTION"
            ) :
            move_one.index("downwardProgressMm =")
        ]
        self.assertIn("pause(", recheck)
        self.assertIn("readKstPoseWithRecovery", recheck)
        self.assertNotIn("getEEFPos()", recheck)
        self.assertNotIn("movePTP", recheck)
        self.assertNotIn("sendEEf", recheck)
        enforce_time = self._matlab_function(self.main, "enforceTime")
        self.assertIn("timeBudgetS > 0.0", enforce_time)

    def test_kst_post_motion_recovery_is_bounded_and_never_replays_motion(
        self,
    ) -> None:
        kst_config = self.main[
            self.main.index("cfg.kst = struct(") :
            self.main.index("cfg.ros = struct(")
        ]
        expected = {
            "post_motion_protocol_settle_s": 0.04,
            "inter_state_query_settle_s": 0.03,
            "read_only_recovery_attempts": 3.0,
            "read_only_recovery_pause_s": 0.05,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(
                self._numeric_field(kst_config, field), value
            )
        self.assertIn(
            "'motion_reissue_permitted_during_state_recovery', false",
            kst_config,
        )

        move_one = self._matlab_function(self.main, "moveOne")
        self.assertEqual(
            move_one.count("movePTPLineEefRelBase("),
            1,
            "the acknowledged motion must remain outside every retry loop",
        )
        for token in (
            "KST:MotionCompletionAcknowledgmentUnavailable",
            "KST_COMPLETION_ACK_MISSING_READ_ONLY_CONFIRMATION_BEGIN",
            "KST_COMPLETION_ACK_RECOVERED_FROM_MEASURED_ENDPOINT",
            "completionAcknowledgmentMissing",
            "cachedLegacyCompletionAckMissing",
            "RELATIVE_LINEAR_MOTION_COMPLETE",
            "command_reissued=0",
        ):
            self.assertIn(token, move_one)
        issued = move_one.index("movePTPLineEefRelBase(")
        recovered_state = move_one.index(
            "readKstRobotStateWithRecovery(", issued
        )
        self.assertLess(issued, recovered_state)
        post_read = move_one[issued:recovered_state + 250]
        self.assertIn("[phase, '_POST'], true", post_read)

        state_reader = self._matlab_function(
            self.main, "readKstRobotStateWithRecovery"
        )
        pose_reader = self._matlab_function(
            self.main, "readKstPoseWithRecovery"
        )
        recovery = self._matlab_function(
            self.main, "recoverKstRobotStateReadOnly"
        )
        self.assertIn(
            "post_motion_protocol_settle_s", state_reader
        )
        self.assertIn(
            "post_motion_protocol_settle_s", pose_reader
        )
        for token in (
            "read_only_recovery_attempts",
            "KST_STATE_READ_RETRY_NO_MOTION",
            "KST_STATE_READ_RECOVERED_NO_MOTION",
            "KST_STATE_READ_EXHAUSTED_NO_MOTION",
            "motion_reissued=0",
            "KstStateReadRecoveryExhausted",
            "recovery_position_consistency_mm",
            "recovery_orientation_consistency_deg",
            "recovery_joint_consistency_rad",
        ):
            self.assertIn(token, recovery)
        for forbidden in (
            "movePTP",
            "sendEEf",
            "realTime_start",
            "realTime_stop",
            "DirectServo",
        ):
            self.assertNotIn(forbidden, recovery)

        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        catch_start = campaign.index("catch attemptError")
        catch_block = campaign[catch_start:]
        for token in (
            "attempt.campaign_must_abort = true",
            "writeSamplingCheckpoint(paths, campaign)",
            "rethrow(attemptError)",
        ):
            self.assertIn(token, catch_block)
        self.assertIn(
            "CONE_COMPLETE_RETRACT_UNVERIFIED", catch_block
        )

        provenance = self.main[
            self.main.index("kstClassPath = canonicalPath") :
            self.main.index(
                "KST_CLIENT_PROVENANCE_PINNED_NO_MOTION"
            )
        ]
        for token in (
            "expectedKstClassPath",
            "expectedEefPath",
            "expectedJointsPath",
            "expectedVectorReaderPath",
            "expectedMotionHelperPath",
            "expectedTargetHelperPath",
            "expectedSetupRecoveryPath",
            "expectedAcknowledgmentReaderPath",
            "expectedLineReaderPath",
            "expectedLineWriterPath",
            "strcmp(kstClassPath, expectedKstClassPath)",
            "sha256File(kstEefPath)",
            "sha256File(kstJointsPath)",
            "sha256File(kstVectorReaderPath)",
            "sha256File(kstMotionHelperPath)",
            "sha256File(kstTargetHelperPath)",
            "sha256File(kstSetupRecoveryPath)",
            "sha256File(kstAcknowledgmentReaderPath)",
            "sha256File(kstLineReaderPath)",
            "sha256File(kstLineWriterPath)",
        ):
            self.assertIn(token, provenance)

    def test_confirmed_touchdown_force_quality_cannot_skip_cone(
        self,
    ) -> None:
        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        contact_match = re.search(
            r"'contact_stability'\s*,\s*struct\(([\s\S]*?)\)"
            r"\s*,\s*\.\.\.\s*'wrench_clear_dwell_count'",
            sampling,
        )
        self.assertIsNotNone(
            contact_match,
            "sampling.contact_stability must be an explicit sub-structure",
        )
        assert contact_match is not None
        contact = contact_match.group(1)
        expected = {
            "target_normal_force_n": 3.5,
            "force_deadband_n": 0.5,
            "minimum_dwell_s": 0.5,
            "minimum_samples": 20.0,
            "maximum_force_std_n": 0.2,
            "passive_observation_s": 0.60,
            "maximum_observation_s": 1.50,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(
                self._numeric_field(contact, field),
                value,
                msg=f"unexpected sampling.contact_stability.{field}",
            )

        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        touchdown = campaign.index(
            "acquireSamplingTouchdownWithRetries("
        )
        self.assertIn(
            "observeSamplingContactBeforeCone",
            campaign[touchdown:],
            "each confirmed touchdown must be observed before cone motion",
        )
        observation = campaign.index(
            "observeSamplingContactBeforeCone", touchdown
        )
        cone = campaign.index("runConicalFingerprint(ctx", observation)
        self.assertLess(touchdown, observation)
        self.assertLess(observation, cone)
        self.assertIn(
            "physicalTouchdownForCone", campaign[touchdown:cone]
        )
        for token in (
            "assert(contactSettle.valid",
            "M12Demo:SamplingCompletionFirstInvariant",
        ):
            self.assertIn(token, campaign[observation:cone])
        for forbidden in (
            "CONTACT_FORCE_NOT_STABLE_",
            "~contactSettle.valid",
            "elseif contactSettle.valid",
        ):
            self.assertNotIn(forbidden, campaign)

    def test_precone_contact_observation_is_passive_fresh_and_nonblocking(
        self,
    ) -> None:
        observer = self._matlab_function(
            self.main, "observeSamplingContactBeforeCone"
        )
        for token in (
            "receiveWrench(ctx, phase, [], true, false)",
            "enforceSamplingHardWrenchSample",
            "minimum_dwell_s",
            "minimum_samples",
            "maximum_force_std_n",
            "passive_observation_s",
            "maximum_observation_s",
            "PASS_PREFERRED_STABLE_LOAD",
            "PASS_COMPLETION_FIRST_CONTACT_OBSERVED",
            "settle.valid = true",
            "[0.0, 0.0, 0.0]",
            "std(",
        ):
            self.assertIn(token, observer)
        for forbidden in (
            "moveOne(",
            "moveAirSegmented(",
            "movePTP",
            "getEEFPos",
            "getJointsPos",
            "sendEEf",
            "sendEEfPositionf",
            "realTime_",
            "drawnow",
            "dashboard",
            "fopen(",
            "save(",
        ):
            self.assertNotIn(forbidden, observer)

        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        contact_match = re.search(
            r"'contact_stability'\s*,\s*struct\(([\s\S]*?)\)"
            r"\s*,\s*\.\.\.\s*'wrench_clear_dwell_count'",
            sampling,
        )
        self.assertIsNotNone(contact_match)
        assert contact_match is not None
        contact = contact_match.group(1)
        expected = {
            "passive_observation_s": 0.60,
            "maximum_observation_s": 1.50,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(
                self._numeric_field(contact, field),
                value,
                msg=f"unexpected sampling.contact_stability.{field}",
            )
        self.assertIn("'issues_motion_commands', false", contact)
        self.assertIn(
            "'soft_force_quality_can_skip_cone', false", contact
        )
        self.assertIn(
            "contact.issues_motion_commands", self.main
        )
        self.assertIn(
            "contact.soft_force_quality_can_skip_cone", self.main
        )
        self.assertEqual(
            observer.count("appendSamplingSettleObservation("), 1
        )
        self.assertIn(
            "settle.motion_commands_issued = false", observer
        )
        self.assertIn(
            "settle.completion_first_continuation = true", observer
        )

    def test_point_acquisition_soft_failures_get_three_bounded_retries(
        self,
    ) -> None:
        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        self.assertAlmostEqual(
            self._numeric_field(
                sampling, "maximum_point_acquisition_attempts"
            ),
            3.0,
        )
        acquisition = self._matlab_function(
            self.main, "acquireSamplingTouchdownWithRetries"
        )
        for token in (
            "captureSamplingPointBaseline(",
            "guardedSamplingTouchdown(",
            "POINT_AIR_BASELINE_CONTAMINATED_",
            "NO_PERSISTENT_CONTACT_WITHIN_RUN_LOCAL_TOUCHDOWN_BUDGET",
            "MEASURED_CONTACT_XY_OUTSIDE_PLAN_TOLERANCE",
            "SAMPLING_POINT_ACQUISITION_RETRY_AFTER_CLEAR_RETRACT",
            "retractSamplingUntilClear(",
            "moveAirSegmented(",
            "requireWrenchClearDwells(",
        ):
            self.assertIn(token, acquisition)
        self.assertRegex(
            acquisition,
            r"for acquisitionIndex\s*=\s*\.\.\.\s*"
            r"[\s\S]{0,100}maximum_point_acquisition_attempts",
        )
        self.assertLess(
            acquisition.index("retractSamplingUntilClear("),
            acquisition.index("moveAirSegmented("),
        )
        self.assertNotIn("catch ", acquisition)
        self.assertNotIn("try\n", acquisition)
        self.assertIn(
            "captureAtTouchdown || acquisitionIndex >=", acquisition
        )

    def test_completion_first_policy_is_manifested_and_label_only_tags(
        self,
    ) -> None:
        self.assertGreaterEqual(
            self.main.count(
                "'soft_force_quality_can_skip_cone', false"
            ),
            3,
        )
        self.assertGreaterEqual(
            self.main.count(
                "'confirmed_contact_continues_to_cone', true"
            ),
            2,
        )
        self.assertGreaterEqual(
            self.main.count(
                "'hard_or_unknown_state_is_not_caught', true"
            ),
            2,
        )
        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        label_gate = campaign.index(
            "if ~attempt.measured_label_gate_passed"
        )
        observation = campaign.index(
            "observeSamplingContactBeforeCone(", label_gate
        )
        cone = campaign.index("runConicalFingerprint(ctx", observation)
        self.assertLess(label_gate, observation)
        self.assertLess(observation, cone)
        self.assertIn(
            "elseif isempty(attempt.quarantine_reason)",
            campaign[cone:],
            "successful cone must not overwrite an earlier label quarantine",
        )
        cone_runner = self._matlab_function(
            self.main, "runConicalFingerprint"
        )
        pre_stream = cone_runner[
            : cone_runner.index(
                "realTime_startDirectServoCartesian()"
            )
        ]
        for token in (
            "readKstRobotStateWithRecovery(",
            "assertRobotState(",
        ):
            self.assertIn(token, pre_stream)
        self.assertNotIn("getEEFPos()", pre_stream)
        self.assertNotIn("getJointsPos()", pre_stream)

    def test_sampling_touchdown_uses_endpoint_and_contact_without_progress_abort(
        self,
    ) -> None:
        move_one = self._matlab_function(self.main, "moveOne")
        touchdown = self._matlab_function(
            self.main, "guardedSamplingTouchdown"
        )
        self.assertIn(
            "samplingMeasuredMicroMove =",
            move_one,
        )
        self.assertRegex(
            move_one,
            r"ratioGateRequired\s*=\s*"
            r"~samplingMeasuredMicroMove\s*&&\s*(?:\.\.\.\s*)?"
            r"~contactReliefMeasuredRebase\s*&&\s*commandDistance",
        )
        self.assertIn(
            "SAMPLING_TOUCHDOWN_ENDPOINT_AND_CONTACT_SEQUENCE_AUTHORITATIVE",
            move_one,
        )
        self.assertNotIn("SAMPLING_CONTACT_SETTLE_", move_one)
        self.assertIn(
            "if ratioGateRequired && alongCommand < minimumProgress",
            move_one,
            "the generic ratio gate must remain active for other motions",
        )
        self.assertNotRegex(
            move_one,
            r"trackingAccepted\s*=\s*trackingAccepted\s*&&[\s\S]{0,160}"
            r"minimum_touchdown_down_progress_mm",
        )
        self.assertIn(
            "maximum_touchdown_xy_lock_error_mm",
            move_one,
        )
        for token in (
            "segmentCount = ceil(maximumDownMm / segment)",
            "for index = 1:segmentCount",
            "guardDwell(",
            "if guard.event_confirmed",
        ):
            self.assertIn(
                token,
                touchdown,
                "touchdown must remain finite and force-feedback-driven",
            )

    def test_invalid_attempts_are_quarantined_and_abort_when_state_unknown(
        self,
    ) -> None:
        for token in (
            "NO_PERSISTENT_CONTACT_WITHIN_RUN_LOCAL_TOUCHDOWN_BUDGET",
            "CAPTURE_OR_BORE_ENTRY_DETECTED_DURING_TOUCHDOWN",
            "INCOMPLETE_CONICAL_PHASE_OR_TIMING_COVERAGE",
            "CAPTURE_OR_BORE_ENTRY_DETECTED_DURING_CONE",
            "ABORTED_NO_BLIND_RETURN",
            "SAMPLING_ATTEMPT_QUARANTINED_ABORT_NO_BLIND_RETURN",
        ):
            self.assertIn(token, self.main)
        self.assertIn(
            "attempt.automatic_training_admission = false", self.main
        )
        self.assertIn(
            "CLEAN_CANDIDATE_PENDING_OFFLINE_QC", self.main
        )
        self.assertIn("HOLDOUT_NEVER_TRAIN", self.main)
        self.assertIn(
            "SENTINEL_REPEATABILITY_ONLY_NOT_INDEPENDENT_TRAINING",
            self.main,
        )
        checkpoint_recovery = self._matlab_function(
            self.main, "refreshSamplingMetricsFromCheckpoint"
        )
        for token in (
            "sampling_attempts_completed",
            "sampling_attempts_quarantined",
            "sampling_returned_to_initial_air_pose",
            "sampling_plan_sha256",
        ):
            self.assertIn(token, checkpoint_recovery)
        self.assertRegex(
            self.main,
            r"refreshResultMetricsFromRuntimeLogs\(result, paths\);"
            r"\s*if samplingMode\s*"
            r"result = refreshSamplingMetricsFromCheckpoint\(result, paths\);",
        )

    def test_training_label_comes_from_measured_touchdown_not_commanded_xy(
        self,
    ) -> None:
        labeler = self.main[
            self.main.index("function attempt = addMeasuredSamplingLabel") :
            self.main.index(
                "function attempt = classifyCompletedSamplingAttempt"
            )
        ]
        self.assertIn("touchdownPose(1:2)", labeler)
        self.assertRegex(
            labeler,
            r"run_local_(?:hole_)?center_world_xy_mm",
        )
        self.assertRegex(
            labeler,
            r"labelVector\s*=\s*"
            r"plan\.run_local_(?:hole_)?center_world_xy_mm\s*-\s*"
            r"(?:measuredWorldXY|touchdownPose\(1:2\))",
        )
        self.assertNotRegex(
            labeler,
            r"labelVector\s*=\s*attempt\.target_world_xy_mm",
        )
        self.assertIn(
            "MEASURED_CONTACT_XY_OUTSIDE_PLAN_TOLERANCE", labeler
        )
        self.assertIn("measured_label_direction_warning", labeler)
        self.assertIn(
            "PASS_DIRECTION_DEVIATION_RECORDED_FOR_OFFLINE_QC",
            labeler,
        )
        self.assertNotIn(
            "MEASURED_LABEL_DIRECTION_DEVIATES_FROM_PREREGISTERED_PLAN",
            labeler,
        )
        self.assertIn(
            "maximum_measured_sampling_error_mm', 0.25", self.main
        )
        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        flow = campaign[
            campaign.index("acquireSamplingTouchdownWithRetries(") :
            campaign.index("runConicalFingerprint(ctx")
        ]
        self.assertIn("addMeasuredSamplingLabel(", flow)
        self.assertIn("~attempt.measured_label_gate_passed", flow)

    def test_sampling_uses_run_local_hole_center_and_world_xy_axes(self) -> None:
        campaign = self.main[
            self.main.index("function [result, endState, ctx] = runM12SamplingCampaign") :
            self.main.index("function attempt = makeSamplingAttempt")
        ]
        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        self.assertIn(
            "RUN_LOCAL_START_HOLE_CENTER_WORLD_XY_ALIGNED", sampling
        )
        self.assertIn(
            "'sampling_coordinate_authority', plan.coordinate_authority",
            campaign,
        )
        rebase = self._matlab_function(
            self.main, "rebaseM12SamplingPlanToRunLocalStart"
        )
        direct_center = re.search(
            r"runLocal\w*(?:Hole)?Center\w*WorldXY\s*=\s*"
            r"homePose\(1:2\)",
            campaign,
        )
        rebased_center = re.search(
            r"plan\.run_local_(?:hole_)?center_world_xy_mm\s*=\s*"
            r"startPose\(1:2\)",
            rebase,
        )
        self.assertTrue(
            direct_center or rebased_center,
            "the confirmed run-local start XY must define the hole center",
        )
        direct_target = re.search(
            r"targetWorldXY\s*=\s*runLocal\w*(?:Hole)?Center\w*WorldXY"
            r"\s*\+\s*double\(\s*point\.local_offset_fixture_mm",
            campaign,
        )
        rebased_target = re.search(
            r"(?:plan\.points\(index\)\.)?world_target_xy_mm\s*=\s*"
            r"startPose\(1:2\)\s*\+\s*offset",
            rebase,
        )
        campaign_uses_rebased_target = (
            "point.world_target_xy_mm" in campaign
        )
        self.assertTrue(
            direct_target
            or (rebased_target and campaign_uses_rebased_target),
            "target WORLD XY must be run-local center plus CAD local offset",
        )
        self.assertNotIn(
            "fixturePointToWorld( ...\n"
            "        double(point.fixture_point_mm(:).'), cfg.frame)",
            campaign,
        )
        labeler = self.main[
            self.main.index("function attempt = addMeasuredSamplingLabel") :
            self.main.index(
                "function attempt = classifyCompletedSamplingAttempt"
            )
        ]
        self.assertRegex(
            labeler,
            r"run_local_(?:hole_)?center_world_xy_mm",
        )
        self.assertIn("touchdownPose(1:2)", labeler)
        self.assertNotRegex(
            labeler,
            r"labelVector\s*=\s*attempt\.target_world_xy_mm",
        )

    def test_sampling_has_no_model_preflight_or_inference(self) -> None:
        preflight_call = self.main[
            self.main.index("if policy.uses_conical_fingerprint") :
            self.main.index("writeJson(paths.manifest", self.main.index(
                "if policy.uses_conical_fingerprint"
            ))
        ]
        self.assertIn("~samplingMode", preflight_call)
        campaign = self.main[
            self.main.index("function [result, endState, ctx] = runM12SamplingCampaign") :
            self.main.index("function attempt = makeSamplingAttempt")
        ]
        for forbidden in (
            "inferFrozenConicalModel",
            "modelGridPosterior",
            "planLearnedShortcut",
            "learned.output",
        ):
            self.assertNotIn(forbidden, campaign)
        self.assertNotRegex(
            campaign,
            re.compile(r"\b(train|fitnet|trainingOptions|backprop)\s*\("),
        )
        self.assertIn(
            "automatic_weight_update_permitted', false", campaign
        )
        attempt_schema = self.main[
            self.main.index("function attempt = makeSamplingAttempt") :
            self.main.index("function baseline = emptySamplingPointBaseline")
        ]
        for stale_model_field in (
            "model_shadow_",
            "predicted_direction_",
            "shadow_shortcut_",
            "classical_shortcut_",
        ):
            self.assertNotIn(stale_model_field, attempt_schema)

    def test_complete_weak_fingerprint_is_not_quarantined_by_low_force_or_model(
        self,
    ) -> None:
        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        for forbidden in (
            "LOW_FORCE_RECONTACT_TRACE_TAGGED_QUARANTINE",
            "model_shadow_completed",
            "model_shadow_error",
        ):
            self.assertNotIn(forbidden, campaign)
        self.assertIn("cone.schedule_completed", campaign)
        self.assertIn("cone.phase_coverage_fraction", campaign)
        self.assertIn("cone.low_force_episodes", campaign)
        classifier = self.main[
            self.main.index(
                "function attempt = classifyCompletedSamplingAttempt"
            ) : self.main.index(
                "function attempt = quarantineSamplingAttempt"
            )
        ]
        self.assertNotIn("model", classifier.lower())
        self.assertNotIn("low_force", classifier.lower())

    def test_complete_cone_resume_ignores_direction_and_routing_quality(
        self,
    ) -> None:
        loader = self._matlab_function(
            self.main, "loadM12SamplingResumeParent"
        )
        complete_evidence = loader[
            loader.index(
                "completeEvidenceRawIndices = "
                "reshape(find(coneMask), 1, [])"
            ) :
            loader.index("pendingExecutionIndices = setdiff(")
        ]
        for forbidden in (
            "cone_direction_fingerprint_valid",
            "cone_routing_fingerprint_valid",
            "direction_fingerprint_valid",
            "routing_fingerprint_valid",
        ):
            self.assertNotIn(
                forbidden,
                complete_evidence,
                "direction/routing are offline quality metadata and must "
                "never force a physically complete cone to replay",
            )
        for token in (
            "logical(attempt.cone_completed)",
            "phase_coverage_fraction",
            "validateSamplingResumeCompleteConeMat(",
        ):
            self.assertIn(token, complete_evidence)

        mat_validator = self._matlab_function(
            self.main, "validateSamplingResumeCompleteConeMat"
        )
        for token in (
            "physical_schedule_completed",
            "schedule_fraction",
            "stage",
            "raw_force_n",
            "raw_torque_nm",
            "source_stamp_s",
            "all_command_pose_mm_rad",
            "all(diff(double(cone.source_stamp_s(:))) > 0.0)",
            "minimum_phase_coverage_fraction",
        ):
            self.assertIn(token, mat_validator)
        self.assertNotIn("direction_fingerprint_valid", mat_validator)
        self.assertNotIn("routing_fingerprint_valid", mat_validator)

    def test_resume_accepts_all_cones_for_recovery_or_finalization_only(
        self,
    ) -> None:
        loader = self._matlab_function(
            self.main, "loadM12SamplingResumeParent"
        )
        self.assertIn(
            "RETURNED_TO_INITIAL_POSE_BUT_CAMPAIGN_FINALIZATION_FAILED",
            loader,
        )
        self.assertIn(
            "'recovery_only_resume', isempty(pendingExecutionIndices)",
            loader,
        )
        self.assertNotIn(
            "SamplingResumeAlreadyComplete",
            loader,
            "26 complete cones may still need clear-state recovery or "
            "child-run finalization and must never be replayed",
        )
        self.assertNotRegex(
            loader,
            r"assert\(\s*~isempty\(pendingExecutionIndices\)",
        )

        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        clear_verified = campaign.index(
            "SAMPLING_RESUME_START_CLEAR_VERIFIED"
        )
        point_loop = campaign.index(
            "for executionIndex = 1:numel(plan.execution_points)"
        )
        home_return = campaign.index(
            "SAMPLING_RETURN_HOME_XY", point_loop
        )
        self.assertLess(clear_verified, point_loop)
        self.assertLess(point_loop, home_return)
        self.assertIn(
            "SAMPLING_RESUME_POINT_SKIPPED_EXISTING_CONE",
            campaign[point_loop:home_return],
        )

    def test_post_stop_failure_keeps_complete_cone_before_abort_and_no_replay(
        self,
    ) -> None:
        cone_runner = self._matlab_function(
            self.main, "runConicalFingerprint"
        )
        for token in (
            "physical_schedule_completed",
            "directservo_stop_acknowledged",
            "post_stop_state_known",
            "post_stop_state_error_identifier",
            "post_stop_state_error_message",
        ):
            self.assertIn(token, cone_runner)
        post_read = cone_runner.index(
            "readKstRobotStateWithRecovery(",
            cone_runner.index("postStopStateKnown = false"),
        )
        self.assertIn(
            "[runKind, '_POST_DIRECTSERVO'], true",
            cone_runner[post_read : post_read + 220],
        )
        cone_build = cone_runner.index("cone = struct(")
        self.assertLess(
            post_read,
            cone_build,
            "the post-stop read may enrich a cone but its failure must be "
            "caught locally so a complete cone value is still returned",
        )
        post_region = cone_runner[
            cone_runner.rfind("try", 0, post_read) :
            cone_build
        ]
        self.assertIn("catch postStateError", post_region)
        self.assertIn(
            "postStopStateKnown = false",
            cone_runner[:post_read],
        )
        self.assertNotIn("rethrow(", post_region)

        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        cone_call = campaign.index("runConicalFingerprint(ctx")
        evidence = campaign.index(
            "addSamplingConeEvidence(", cone_call
        )
        atomic_checkpoint = campaign.index(
            "SAMPLING_CONE_COMPLETE_ATOMIC_CHECKPOINT",
            evidence,
        )
        post_state_gate = campaign.index(
            "throwIfConicalPostStateUnknown", atomic_checkpoint
        )
        retract = campaign.index(
            "retractSamplingUntilClear(ctx", post_state_gate
        )
        self.assertLess(cone_call, evidence)
        self.assertLess(evidence, atomic_checkpoint)
        self.assertLess(atomic_checkpoint, post_state_gate)
        self.assertLess(post_state_gate, retract)
        self.assertEqual(
            campaign.count("runConicalFingerprint(ctx"),
            1,
            "neither persistence nor post-stop state recovery may replay "
            "the physical cone",
        )

    def test_cone_evidence_precedes_best_effort_ui_and_servo_event_io_is_deferred(
        self,
    ) -> None:
        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        cone_call = campaign.index("runConicalFingerprint(ctx")
        evidence = campaign.index(
            "addSamplingConeEvidence(", cone_call
        )
        batch_ui = campaign.index(
            "bestEffortAppendSamplingConeToTopview", cone_call
        )
        cone_ui = campaign.index("'cone_complete'", batch_ui)
        self.assertLess(evidence, batch_ui)
        self.assertLess(batch_ui, cone_ui)
        best_effort_ui = self._matlab_function(
            self.main, "bestEffortAppendSamplingConeToTopview"
        )
        self.assertIn("try", best_effort_ui)
        self.assertIn("catch dashboardError", best_effort_ui)

        cone_runner = self._matlab_function(
            self.main, "runConicalFingerprint"
        )
        active_start = cone_runner.index(
            "ctx.direct_servo_active = true"
        )
        active_stop = cone_runner.index(
            "ctx.direct_servo_active = false", active_start
        )
        active_region = cone_runner[active_start:active_stop]
        self.assertNotIn("logEvent(", active_region)
        self.assertNotIn("logTransition(", active_region)
        self.assertIn("deferred_transition_records", cone_runner)
        flush = cone_runner.index(
            "flushDeferredConicalTransitions", active_stop
        )
        self.assertGreater(flush, active_stop)

        for helper_name in (
            "runConicalLowForcePhaseHold",
            "handleConicalLowForceObstructionNoLift",
            "executeConicalEarlyUprightReturn",
            "accountConicalStreamCommand",
        ):
            helper = self._matlab_function(self.main, helper_name)
            self.assertNotIn("logEvent(", helper)
            self.assertNotIn("logTransition(", helper)

    def test_campaign_dashboard_updates_only_after_safe_retract(self) -> None:
        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        self.assertLess(
            campaign.index("attempt.safe_retract_completed = true"),
            campaign.index(
                "bestEffortSamplingTopviewAction(ctx, ...\n"
                "        'complete'"
            ),
        )
        self.assertIn(
            "'锥扫中，界面冻结'", campaign
        )
        self.assertIn("'allow_drawnow', true", campaign)

    def test_cone_complete_and_secured_commit_counts_are_independent(
        self,
    ) -> None:
        initializer = self._matlab_function(
            self.dashboard, "initializeDashboard"
        )
        cone_complete = self._matlab_function(
            self.dashboard, "applyConeComplete"
        )
        secured_complete = self._matlab_function(
            self.dashboard, "applyComplete"
        )
        status = self._matlab_function(
            self.dashboard, "updateStatusText"
        )
        for token in (
            "'cone_completed_mask', false(26, 1)",
            "'committed_mask', false(26, 1)",
            "cones: 0/26",
            "secured: 0/26",
        ):
            self.assertIn(token, initializer)
        self.assertIn(
            "dashboard.cone_completed_mask(index) = true",
            cone_complete,
        )
        self.assertNotIn("committed_mask(index)", cone_complete)
        self.assertIn(
            "dashboard.committed_mask(index) = true",
            secured_complete,
        )
        self.assertNotIn(
            "cone_completed_mask(index) = true", secured_complete
        )
        for token in (
            "cones: %d/26",
            "secured: %d/26",
            "sum(dashboard.cone_completed_mask)",
            "sum(dashboard.committed_mask)",
        ):
            self.assertIn(token, status)
        self.assertNotIn(
            "committed:",
            self.dashboard,
            "operator-facing lifecycle count must be labelled secured",
        )

        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        cone_evidence = campaign.index("addSamplingConeEvidence(")
        cone_mark = campaign.index("'cone_complete'", cone_evidence)
        retract = campaign.index(
            "retractSamplingUntilClear(ctx", cone_mark
        )
        secured = campaign.index(
            "attempt.safe_retract_completed = true", retract
        )
        committed_mark = campaign.index(
            "bestEffortSamplingTopviewAction(ctx, ...\n"
            "        'complete'",
            secured,
        )
        self.assertLess(
            cone_evidence, cone_mark
        )
        self.assertLess(cone_mark, retract)
        self.assertLess(retract, secured)
        self.assertLess(secured, committed_mark)
        cone_payload = campaign[cone_mark:retract]
        self.assertIn("'force_render', true", cone_payload)

        aggregator = self._matlab_function(
            self.main, "refreshSamplingCampaignCounters"
        )
        self.assertIn(
            "campaign.cones_completed = sum(arrayfun(", aggregator
        )
        self.assertIn(
            "logical(value.cone_completed)", aggregator
        )
        self.assertIn(
            "campaign.points_committed = sum(arrayfun(", aggregator
        )
        self.assertIn(
            "logical(value.motion_lifecycle_closed)", aggregator
        )
        self.assertIn(
            "campaign.original_retracts_verified = sum(arrayfun(",
            aggregator,
        )
        self.assertIn(
            "logical(value.safe_retract_completed)", aggregator
        )
        self.assertIn(
            "campaign.motion_lifecycles_closed = sum(arrayfun(",
            aggregator,
        )
        self.assertIn(
            "numel(unique(executionIndices)) == numel(executionIndices)",
            aggregator,
        )

        catch_start = campaign.index("catch attemptError")
        cone_failure = campaign.index(
            "elseif attempt.cone_completed", catch_start
        )
        acquisition_failure = campaign.index(
            "\n        else", cone_failure
        )
        cone_failure_branch = campaign[
            cone_failure:acquisition_failure
        ]
        self.assertIn(
            "CONE_COMPLETE_RETRACT_UNVERIFIED",
            cone_failure_branch,
        )
        self.assertNotIn(
            "quarantineSamplingAttempt", cone_failure_branch
        )
        self.assertIn(
            "SAMPLING_CONE_COMPLETE_CHECKPOINTED_",
            campaign[catch_start:],
        )

    def test_sampling_has_one_top_view_window_and_servo_has_no_ui_calls(
        self,
    ) -> None:
        dashboard = self._matlab_function(
            self.dashboard, "initializeDashboard"
        )
        self.assertEqual(
            len(re.findall(r"\bfigure\s*\(", dashboard)), 1
        )
        self.assertIn("top view", dashboard.lower())
        self.assertIn(
            "printedCad = unpackPrintedCadGeometry(payload, sizeLabel)",
            dashboard,
        )
        self.assertRegex(
            dashboard,
            r"isfield\(previousDashboard, 'figure'\)\s*&&\s*\.\.\.\s*"
            r"isscalar\(previousDashboard\.figure\)\s*&&\s*\.\.\.\s*"
            r"isgraphics\(previousDashboard\.figure\)",
        )
        geometry_bindings = {
            "outerDiameterMm": "head_outer_diameter",
            "mouthAfMm": "socket_mouth_af",
            "wallAfMm": "socket_deep_wall_af",
            "throughHoleDiameterMm": "center_through_hole_diameter",
            "pegTipAfMm": "peg_tip_face_af",
            "pegTipFlatNormalYawDeg":
                "peg_tip_face_flat_normal_fixture_deg",
        }
        for local_name, plan_field in geometry_bindings.items():
            self.assertRegex(
                dashboard,
                rf"{re.escape(local_name)}\s*=\s*(?:\.\.\.\s*)?"
                rf"printedCad\.{re.escape(plan_field)}",
            )
        for token in (
            "outerDiameterMm",
            "mouthAfMm",
            "wallAfMm",
            "throughHoleDiameterMm",
            "pegTipAfMm",
            "+X_F",
            "+Y_F",
            "+Z_F",
        ):
            self.assertIn(token, dashboard)
        geometry_loader = self._matlab_function(
            self.dashboard, "unpackPrintedCadGeometry"
        )
        self.assertIn("plan.geometry_mm", geometry_loader)
        for field in geometry_bindings.values():
            self.assertIn(field, geometry_loader)
        self.assertRegex(
            self.main,
            r"callSamplingTopview\(\s*\.\.\.\s*"
            r"cfg,\s*'initialize'",
        )
        dispatcher = self._matlab_function(
            self.main, "callSamplingTopview"
        )
        self.assertIn("m12_sampling_topview_dashboard", dispatcher)
        self.assertIn("m10_sampling_topview_dashboard", dispatcher)
        self.assertIn("feval(dashboardFunction", dispatcher)

        servo = self.main[
            self.main.index(
                "function [ctx, cone] = runConicalFingerprint"
            ) : self.main.index(
                "function thresholds = conicalLowForceThresholds"
            )
        ]
        forbidden_calls = (
            r"\bdrawnow\b",
            r"\btimer\s*\(",
            r"\bfigure\s*\(",
            r"\bplot\s*\(",
            r"\buicontrol\s*\(",
            r"\bupdateDashboard\s*\(",
        )
        for pattern in forbidden_calls:
            self.assertNotRegex(servo, pattern)

        servo_stream = servo[
            servo.index("for index = 1:count\n    enforceTime(ctx);") :
            servo.index("streamElapsedS = toc(servoClock);")
        ]
        for token in (
            "getEEFPos",
            "getJointsPos",
            "getEEFForce",
            "m12_sampling_topview_dashboard",
            "rossubscriber",
            "rosmessage",
            "receive(",
            "fopen(",
            "fclose(",
            "fprintf(",
            "writetable",
            "writematrix",
            "writecell",
            "writeJson",
            "writeText",
            "readtable",
            "readmatrix",
            "matfile(",
            "save(",
        ):
            self.assertNotIn(token, servo_stream)
        self.assertIn(
            "logConicalTrace(",
            servo_stream,
            "only the pre-existing real-time trace sink is allowed in-stream",
        )

    def test_each_attempt_exports_machine_readable_trace_files(self) -> None:
        campaign = self.main[
            self.main.index(
                "function [result, endState, ctx] = runM12SamplingCampaign"
            ) : self.main.index("function attempt = makeSamplingAttempt")
        ]
        self.assertRegex(
            campaign,
            r"fullfile\(\s*attemptDir\s*,\s*"
            r"'[^']*(?:trace|wrench)[^']*\.csv'\s*\)",
        )
        self.assertNotIn(
            "exportSamplingAttemptEvidence(",
            campaign,
            "campaign control must use the bounded persistence helper; "
            "immutable evidence export is an internal precommit phase",
        )
        persistence = self._matlab_function(
            self.main, "persistSamplingAttemptWithRetry"
        )
        self.assertIn("exportSamplingAttemptEvidence(", persistence)
        self.assertIn("saveSamplingAttempt(", persistence)
        cone_checkpoint = campaign.index(
            "SAMPLING_CONE_COMPLETE_ATOMIC_CHECKPOINT"
        )
        precommit = campaign.rfind(
            "persistSamplingAttemptWithRetry(", 0, cone_checkpoint
        )
        retract = campaign.index(
            "retractSamplingUntilClear(ctx", cone_checkpoint
        )
        secured = campaign.index(
            "attempt.safe_retract_completed = true", retract
        )
        lifecycle_overlay = campaign.index(
            "persistSamplingAttemptWithRetry(", secured
        )
        self.assertLess(
            precommit,
            cone_checkpoint,
        )
        self.assertLess(cone_checkpoint, retract)
        self.assertLess(retract, secured)
        self.assertLess(secured, lifecycle_overlay)

    def test_sampling_persistence_retries_are_bounded_and_never_skip_clear(
        self,
    ) -> None:
        sampling = self.main[
            self.main.index("cfg.sampling = struct(") :
            self.main.index("cfg.files = struct(")
        ]
        for token in (
            "'persistence'",
            "'maximum_attempts'",
            "'retry_pause_s'",
        ):
            self.assertIn(token, sampling)

        persistence = self._matlab_function(
            self.main, "persistSamplingAttemptWithRetry"
        )
        for token in (
            "maximum_attempts",
            "retry_pause_s",
            "for ",
            "try",
            "catch",
            "PRE_RETRACT_PERSISTENCE_RETRY_NO_MOTION",
            "PRE_RETRACT_PERSISTENCE_DEFERRED_UNTIL_CLEAR",
            "POST_RETRACT_PERSISTENCE_RECOVERED",
            "POST_RETRACT_PERSISTENCE_EXHAUSTED_ROBOT_CLEAR",
        ):
            self.assertIn(token, persistence)
        self.assertNotIn("movePTP", persistence)
        self.assertNotIn("sendEEf", persistence)
        self.assertNotIn("DirectServo", persistence)

        campaign = self._matlab_function(
            self.main, "runM12SamplingCampaign"
        )
        pre_persist = campaign.index(
            "persistSamplingAttemptWithRetry("
        )
        retract = campaign.index(
            "retractSamplingUntilClear(ctx", pre_persist
        )
        secured = campaign.index(
            "attempt.safe_retract_completed = true", retract
        )
        post_persist = campaign.index(
            "persistSamplingAttemptWithRetry(", secured
        )
        self.assertLess(pre_persist, retract)
        self.assertLess(retract, secured)
        self.assertLess(secured, post_persist)
        pre_to_clear = campaign[pre_persist:retract]
        for forbidden in (
            "rethrow(",
            "assert(persistence",
            "error('M12Demo:SamplingPersistence",
        ):
            self.assertNotIn(
                forbidden,
                pre_to_clear,
                "pre-retract persistence failure must be represented as "
                "state and may not bypass physical clearance",
            )

        post_clear = campaign[
            secured : campaign.index("catch attemptError", secured)
        ]
        self.assertNotIn(
            "exportSamplingAttemptEvidence(",
            post_clear,
            "immutable MAT and three evidence CSVs are committed once "
            "before retract and never overwritten by a lifecycle overlay",
        )
        self.assertNotIn(
            "saveSamplingAttempt(",
            post_clear,
            "post-retract lifecycle closure must update JSON/campaign only",
        )

    def test_sampling_csv_outputs_are_atomic_snapshots_not_appends(
        self,
    ) -> None:
        exporter = self._matlab_function(
            self.main, "exportSamplingAttemptEvidence"
        )
        for writer in (
            "writeSamplingPointBaselineCsvAtomic",
            "writeSamplingContactSettleCsvAtomic",
            "writeSamplingConeTraceCsvAtomic",
        ):
            self.assertIn(f"{writer}(", exporter)
            writer_source = self._matlab_function(self.main, writer)
            self.assertNotRegex(
                writer_source,
                r"fopen\(\s*path\s*,\s*'w'\s*\)",
                "final CSV path must never be opened for in-place overwrite",
            )
            self.assertTrue(
                "writeText(" in writer_source
                or (
                    "tempname(fileparts(path))" in writer_source
                    and "movefile(" in writer_source
                ),
                f"{writer} must close a same-directory temporary file "
                "and atomically replace the final CSV",
            )

        snapshot = self._matlab_function(
            self.main, "writeSamplingAttemptsCsvSnapshot"
        )
        for token in (
            "campaign.attempts",
            "execution_index",
            "writeText(",
        ):
            self.assertIn(token, snapshot)
        self.assertNotIn("appendLine(", snapshot)
        self.assertNotIn("appendSamplingAttemptCsv(", self.main)

    def test_sampling_attempt_csv_includes_settle_robustness_evidence(
        self,
    ) -> None:
        header_source = self._matlab_function(
            self.main, "samplingAttemptsCsvHeader"
        )
        writer = self._matlab_function(
            self.main, "samplingAttemptCsvRow"
        )
        new_fields = (
            "contact_settle_reason",
            "contact_settle_normal_mean_n",
            "contact_settle_normal_std_n",
            "contact_settle_off_target_continuation",
            "contact_settle_quality_grade",
            "contact_settle_completion_first_continuation",
            "contact_settle_observed_dwell_s",
            "contact_settle_normal_min_n",
            "contact_settle_normal_max_n",
            "contact_settle_adjustment_commands",
            "contact_settle_unobserved_adjustments",
        )
        for field in new_fields:
            self.assertIn(field, header_source)
            self.assertIn(f"attempt.{field}", writer)

        header_literals = re.findall(r"'([^']*)'", header_source)
        header = "".join(header_literals)
        format_literals = re.findall(
            r"'([^']*%(?:[^']|''*)*)'",
            writer[writer.index("sprintf([") : writer.index("], ...")],
        )
        format_string = "".join(format_literals)
        conversion_count = len(
            re.findall(r"%(?:\.\d+)?[dfs]", format_string)
        )
        self.assertEqual(header.count(",") + 1, 41)
        self.assertEqual(conversion_count, 41)

    def test_command_propagated_cone_pose_cannot_make_training_candidate(
        self,
    ) -> None:
        metrology = json.loads(
            (
                ROOT / "configs" / "tool_metrology_commissioning_v1.json"
            ).read_text(encoding="utf-8")
        )
        tracking = metrology[
            "conical_phase_pose_tracking_qualification"
        ]
        self.assertEqual(
            tracking["status"],
            "PENDING_INDEPENDENT_MEASURED_POSE_TRACE",
        )
        self.assertFalse(
            tracking["qualified_for_training_candidates"]
        )
        loader = self.main[
            self.main.index(
                "function gate = loadConicalPhasePoseTrackingQualification"
            ) :
            self.main.index(
                "function [elapsedS, attempts, source]"
            )
        ]
        for token in (
            "COMMAND_PROPAGATED",
            "phase_error_acceptance_deg",
            "tip_xy_tracking_error_acceptance_m",
            "timestamp_alignment_error_acceptance_s",
            "verifyCommissioningEvidence",
            "independently measured",
            "ConicalTrackingEvidenceBinding",
            "physical_tool_instance_id",
            "installation_session_id",
            "active_matlab_source_sha256",
            "conical_command_rate_hz",
            "conical_feedback_policy",
        ):
            self.assertIn(token, loader)
        classifier = self.main[
            self.main.index(
                "function attempt = classifyCompletedSamplingAttempt"
            ) :
            self.main.index(
                "function attempt = quarantineSamplingAttempt"
            )
        ]
        self.assertIn("CLEAN_CANDIDATE_PENDING_OFFLINE_QC", classifier)
        self.assertIn("attempt.automatic_training_admission = false", classifier)
        self.assertNotIn(
            "CONICAL_PHASE_POSE_TRACKING_NOT_QUALIFIED_FOR_TRAINING",
            classifier,
        )
        self.assertNotIn("quarantineSamplingAttempt", classifier)


if __name__ == "__main__":
    unittest.main()
