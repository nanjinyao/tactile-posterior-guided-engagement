from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "execute_m12_contact_engagement_policy.m"
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


def function_slice(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


class TestM12LearnedPolicyContractStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN_PATH.read_text(encoding="utf-8")

    def test_public_wrappers_are_callable_functions_and_select_one_mode(
        self,
    ) -> None:
        for mode, path in WRAPPERS.items():
            text = path.read_text(encoding="utf-8")
            first_code = next(
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("%")
            )
            self.assertTrue(
                first_code.startswith("function result = "),
                f"{path.name} must remain a callable MATLAB function file",
            )
            self.assertIn(
                "execute_m12_contact_engagement_policy( ...\n"
                f"    '{mode}')",
                text,
            )
            self.assertEqual(
                text.count("execute_m12_contact_engagement_policy("), 1
            )

    def test_pure_outward_has_zero_cone_and_model_calls(self) -> None:
        policy = function_slice(
            self.main,
            "policy = executionPolicy(mode)",
            "[result, manifest] = discloseExecutionPolicy(",
        )
        pure_case = policy[
            policy.index("case 'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL'") :
            policy.index(
                "case 'FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED'"
            )
        ]
        self.assertIn("policy.uses_conical_fingerprint = false;", pure_case)
        self.assertIn("no conical probe and no learned inference", pure_case)
        self.assertIn(
            "if policy.uses_conical_fingerprint && ~samplingMode",
            self.main,
        )
        branch_start = self.main.index(
            "if strcmp(executionMode, "
            "'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL')"
        )
        branch_end = self.main.index(
            "logTransition(paths.transitions, 'SHORT_EVENT_TRIGGER_TOUCHDOWN'",
            branch_start,
        )
        pure_runtime = self.main[branch_start:branch_end]
        self.assertIn("runOutwardArchimedeanSearch", pure_runtime)
        self.assertNotIn("runConicalFingerprint", pure_runtime)
        self.assertNotIn("inferFrozenConicalModel", pure_runtime)

    def test_direct_x3_reprobes_inside_first_learned_direction_sector(
        self,
    ) -> None:
        direct = function_slice(
            self.main,
            "[ctx, capture, diagnostics] = runImprovedDirectRetry3(",
            "[direction, quality, evidence] =",
        )
        for token in (
            "adjustmentSigns = [0.0, 1.0, -1.0];",
            "if attempt > 1",
            "runConicalFingerprint(ctx, ...",
            "inferFrozenConicalModel( ...",
            "learnedPosteriorModeForDirectAttempt(currentLearned, ...",
            "referenceDirectionTaskDeg",
            "direct_retry_direction_half_width_deg",
            "FRESH_TOP1_INSIDE_FIRST_DIRECTION_MICROADJUST_SECTOR",
            "FRESH_HIGHEST_POSTERIOR_MODE_INSIDE_FIRST_DIRECTION_MICROADJUST_SECTOR",
            "record.fan_angle_deg = 0.0;",
            "direct_ray_limit_mm",
        ):
            self.assertIn(token, direct)
        mode_selector = function_slice(
            self.main,
            "[direction, quality, evidence, selectedTaskDeg] = ...\n"
            "        learnedPosteriorModeForDirectAttempt(",
            "[direction, quality, evidence] =",
        )
        for token in (
            "probability = double(output.direction_probabilities(:));",
            "anglesDeg = mod(double(output.direction_class_angles_deg(:))",
            "referenceDeltaDeg = abs(mod( ...",
            "eligible = referenceDeltaDeg <= ...",
            "cfg.baselines.direct_retry_direction_half_width_deg + 1e-12;",
            "eligibleProbability(~eligible) = -Inf;",
            "[~, selectedIndex] = max(eligibleProbability);",
        ):
            self.assertIn(token, mode_selector)
        self.assertIn(
            "top1TaskDeg = mod(double(output.top_direction_deg)",
            mode_selector,
            "each fresh cone may keep TOP1 inside the first-route sector",
        )
        self.assertIn(
            "isequal(double(ctx.cfg.baselines.direct_fan_angles_deg), ...\n"
            "        [0.0, 0.0, 0.0])",
            direct,
        )
        self.assertNotRegex(
            direct,
            r"(fan_angles_deg\s*\(\s*attempt\s*\)|"
            r"direction\s*\+\s*fan|rotate.*fan)",
        )
        self.assertLess(
            direct.index("runConicalFingerprint(ctx, ..."),
            direct.index("inferFrozenConicalModel( ..."),
        )

    def test_shortcut_keeps_learned_top1_with_posterior_cad_distance_cap(
        self,
    ) -> None:
        planner = function_slice(
            self.main,
            "plan = planLearnedShortcut(",
            "offset = selectPosteriorResidualTarget(",
        )
        config = self.main[
            self.main.index("cfg.search = struct(") :
            self.main.index("cfg.baselines = struct(")
        ]
        for token in (
            "'shortcut_cap_mm', 3.00",
            "'shortcut_projection_quantile_low', 0.10",
            "'shortcut_projection_quantile_high', 0.25",
            "'shortcut_radius_cap_fraction_low', 0.50",
            "'shortcut_radius_cap_fraction_high', 0.75",
            "'shortcut_residual_fallback_minimum_cap_fraction', 0.75",
        ):
            self.assertIn(token, config)
        self.assertIn(
            "captureTransportMarginMm = ...\n"
            "    0.25 * cfg.search.bayes_capture_radius_mm;",
            planner,
        )
        self.assertIn(
            "radiusTransportCap = qualityRadiusFraction * radius95 - ...\n"
            "    captureTransportMarginMm;",
            planner,
        )
        self.assertIn(
            "cfg.learning.maximum_authorized_shortcut_mm", planner
        )
        for token in (
            "distance = distanceCap;",
            "direction = top1Direction;",
            "offset = distance * direction;",
            "LEARNED_TOP1_DIRECTION_WITH_POSTERIOR_CAD_DISTANCE",
            "segmentCaptureLikelihood(hypotheses, [0.0, 0.0], offset, cfg)",
            "'raw_top1_direction_world_xy', top1Direction",
            "'swept_capture_probability', ...",
            "'low_quality_still_commands_bounded_action', true",
        ):
            self.assertIn(token, planner)
        self.assertNotIn(
            "selectPosteriorSweptCaptureTarget(", planner,
            "the diffuse posterior may size the route but may not rotate a "
            "finite learned direction onto another grid mode",
        )

    def test_center_hypotheses_extend_beyond_bounded_action_reach(
        self,
    ) -> None:
        config = self.main[
            self.main.index("m12CommonSearchReachMm = 4.80;") :
            self.main.index("cfg.baselines = struct(")
        ]
        for token in (
            "m12CenterHypothesisRadiusMm = m12CommonSearchReachMm + ...",
            "m12ConservativeCaptureRadiusMm + 0.20;",
            "'action_reach_mm', m12CommonSearchReachMm",
            "'grid_radius_mm', m12CommonSearchReachMm",
            "'center_hypothesis_radius_mm', m12CenterHypothesisRadiusMm",
            "'maximum_grid_nodes', activeGridNodeCount",
        ):
            self.assertIn(token, config)

        plan = function_slice(
            self.main,
            "plan = makePlan(cfg, policy)",
            "grid = buildGridOffsets(cfg)",
        )
        for token in (
            "actionOffsets = buildGridOffsets(cfg);",
            "centerHypotheses = buildCenterHypothesisOffsets(cfg);",
            "'action_offsets_mm', actionOffsets",
            "'grid_offsets_mm', actionOffsets",
            "'center_hypotheses_mm', centerHypotheses",
        ):
            self.assertIn(token, plan)
        builders = function_slice(
            self.main,
            "grid = buildGridOffsets(cfg)",
            "validateConicalContract(cfg)",
        )
        self.assertIn(
            "cfg.search.grid_pitch_mm, cfg.search.action_reach_mm",
            builders,
        )
        self.assertIn(
            "cfg.search.center_hypothesis_radius_mm",
            builders,
        )

        action_reach_mm = 4.80
        capture_radius_mm = 0.75
        hypothesis_radius_mm = action_reach_mm + capture_radius_mm + 0.20
        pitch_mm = 0.50

        def circular_count(radius_mm: float) -> int:
            integer_radius = math.ceil(radius_mm / pitch_mm)
            return sum(
                math.hypot(pitch_mm * ix, pitch_mm * iy)
                <= radius_mm + 1e-12
                for ix in range(-integer_radius, integer_radius + 1)
                for iy in range(-integer_radius, integer_radius + 1)
            )

        self.assertGreater(
            hypothesis_radius_mm,
            action_reach_mm + capture_radius_mm,
        )
        self.assertEqual(circular_count(action_reach_mm), 293)
        self.assertEqual(circular_count(hypothesis_radius_mm), 421)

        route = self.main[
            self.main.index(
                "centerHypotheses = plan.center_hypotheses_mm;"
            ) :
            self.main.index("result.residual_eig_steps = residualEigSteps;")
        ]
        for token in (
            "actionOffsets = plan.action_offsets_mm;",
            "modelGridPosterior( ...\n"
            "            centerHypotheses, learned.output, conical, cfg)",
            "planLearnedShortcut( ...\n"
            "            centerHypotheses, modelPosterior",
            "updatePosteriorAlongSegment(posterior, ...\n"
            "                    centerHypotheses",
            "offset = actionOffsets(candidateIndex, :);",
        ):
            self.assertIn(token, route)
        self.assertNotIn(
            "offset = centerHypotheses(candidateIndex, :);",
            route,
            "outer center hypotheses describe belief support, not commands",
        )

    def test_corridor_is_centered_on_selected_posterior_cad_route(
        self,
    ) -> None:
        corridor_angle = function_slice(
            self.main,
            "[halfAngleDeg, diagnostics] = ...\n"
            "        posteriorCorridorHalfAngleDeg("
            "output, cfg, routingDirectionWorldXY)",
            "[pathOffsetsMm, laneOffsetsMm, diagnostics] =",
        )
        config = self.main[
            self.main.index("cfg.search = struct(") :
            self.main.index("cfg.baselines = struct(")
        ]
        for token in (
            "'directional_corridor_probability_mass', 0.80",
            "'directional_corridor_minimum_full_width_deg', 12.0",
            "'directional_corridor_maximum_full_width_deg', 45.0",
        ):
            self.assertIn(token, config)
        for token in (
            "routingDirectionWorldXY = double(reshape( ...",
            "directionTask = cfg.frame.world_from_fixture_rotation.' * ...",
            "controlCenterDeg = mod(rad2deg(atan2( ...",
        ):
            self.assertIn(token, corridor_angle)
        self.assertIn(
            "anglesDeg - controlCenterDeg + 180.0", corridor_angle
        )
        self.assertIn(
            "[sortedDistanceDeg, order] = sort(circularDistanceDeg",
            corridor_angle,
        )
        self.assertIn("targetMass =", corridor_angle)
        self.assertIn(
            "requiredFullWidthDeg = 2.0 * requiredHalfAngleDeg;",
            corridor_angle,
        )
        self.assertIn(
            "fullWidthDeg = min(max(requiredFullWidthDeg", corridor_angle
        )
        self.assertIn("halfAngleDeg = 0.5 * fullWidthDeg;", corridor_angle)
        self.assertIn(
            "containedMass = sum(probability(containedMask));",
            corridor_angle,
        )
        self.assertIn(
            "containedMass >= targetMass - 1e-12", corridor_angle
        )
        self.assertNotIn("extendedAngles", corridor_angle)
        self.assertNotIn("minimumWidthDeg", corridor_angle)
        self.assertNotIn("minimum_half_angle_deg", self.main)
        self.assertNotIn("maximum_half_angle_deg", self.main)
        self.assertIn(
            "posteriorCorridorHalfAngleDeg(learned.output, cfg, ...\n"
            "                learnedShortcut.direction_world_xy)",
            self.main,
        )
        self.assertIn(
            "activeGridRoutingDirection = ...\n"
            "        learnedShortcut.direction_world_xy;",
            self.main,
        )
        self.assertIn(
            "directionQualified = norm(activeGridRoutingDirection) > 1e-12;",
            self.main,
        )
        self.assertNotIn(
            "directionQualified = norm(activeGridRoutingDirection) > "
            "1e-12 && ...\n"
            "        corridorCredible.mass_requirement_satisfied;",
            self.main,
            "posterior diffuseness may widen the capped corridor, not erase "
            "a finite learned direction",
        )

    def test_directional_corridor_is_monotonic_centerline_without_serpentine(
        self,
    ) -> None:
        corridor = function_slice(
            self.main,
            "[pathOffsetsMm, laneOffsetsMm, diagnostics] = ...\n"
            "        buildDirectionalCadCorridor(",
            "[candidateIndex, bestInformation, predictiveOpen,",
        )
        config = self.main[
            self.main.index("cfg.search = struct(") :
            self.main.index("cfg.baselines = struct(")
        ]
        for token in (
            "'directional_corridor_maximum_advance_mm', ...\n"
            "        activeDirectionalCorridorAdvanceMm",
            "'directional_progress_tolerance_mm', 0.10",
        ):
            self.assertIn(token, config)
        for token in (
            "axialStartMm = min(max(dot(currentOffsetMm, direction), 0.0),",
            "axialStopMm = min(radiusMm, axialStartMm + ...",
            "cfg.search.directional_corridor_maximum_advance_mm);",
            "pathOffsetsMm = currentOffsetMm;",
            "target = axialStationsMm(stationIndex) * direction;",
            "axialProgressMm = diff(pathOffsetsMm * direction.');",
            "-cfg.search.directional_progress_tolerance_mm - 1e-12",
            "laneOffsetsMm = zeros(size(pathOffsetsMm, 1), 1);",
            "'routing_mode', 'MONOTONIC_LEARNED_DIRECTION_CENTERLINE'",
        ):
            self.assertIn(token, corridor)
        for stale_serpentine_token in (
            "laneStarts",
            "laneEnds",
            "forward = true;",
            "directional_corridor_lane_spacing_capture_fraction",
        ):
            self.assertNotIn(stale_serpentine_token, corridor)

        direction_angle_rad = math.radians(30.0)
        direction = (
            math.cos(direction_angle_rad),
            math.sin(direction_angle_rad),
        )
        normal = (-direction[1], direction[0])
        current = (
            direction[0] + 0.25 * normal[0],
            direction[1] + 0.25 * normal[1],
        )
        axial_start = max(
            0.0,
            current[0] * direction[0] + current[1] * direction[1],
        )
        axial_stop = min(4.80, axial_start + 1.00)
        stations = [axial_start + 0.50, axial_stop]
        path = [current] + [
            (station * direction[0], station * direction[1])
            for station in stations
        ]
        projections = [
            point[0] * direction[0] + point[1] * direction[1]
            for point in path
        ]
        self.assertTrue(
            all(
                later - earlier >= -0.10
                for earlier, later in zip(projections, projections[1:])
            )
        )
        self.assertLessEqual(projections[-1] - projections[0], 1.00)
        for point in path[1:]:
            lateral = point[0] * normal[0] + point[1] * normal[1]
            self.assertAlmostEqual(lateral, 0.0, places=12)

    def test_residual_eig_and_coverage_stay_directional_and_connected(
        self,
    ) -> None:
        eig_selector = function_slice(
            self.main,
            "[candidateIndex, bestInformation, predictiveOpen,",
            "[candidateIndex, bestInformation, predictiveOpen, ...\n"
            "        bestUtility, selectionRegion] = ...\n"
            "        selectCadCoverageNode(",
        )
        coverage_selector = function_slice(
            self.main,
            "[candidateIndex, bestInformation, predictiveOpen, ...\n"
            "        bestUtility, selectionRegion] = ...\n"
            "        selectCadCoverageNode(",
            "[admitted, novelSweptLengthMm, directionalProgressMm, ...\n"
            "        globalForwardProgressMm] = ...\n"
            "        residualCandidatePhysicalAdmission(",
        )
        physical_admission = function_slice(
            self.main,
            "[admitted, novelSweptLengthMm, directionalProgressMm, ...\n"
            "        globalForwardProgressMm] = ...\n"
            "        residualCandidatePhysicalAdmission(",
            "segments = appendMeasuredSurfaceSegments(",
        )
        swept_novelty = function_slice(
            self.main,
            "novelLengthMm = measuredSweptNovelLength(",
            "distanceMm = pointToSegmentDistance2d(",
        )
        config = self.main[
            self.main.index("cfg.search = struct(") :
            self.main.index("cfg.baselines = struct(")
        ]
        for token in (
            "'eig_information_weight', 0.35",
            "'eig_capture_reward_weight', 1.50",
            "'eig_action_cost', 0.08",
            "'eig_travel_penalty_per_mm', 0.020",
            "'eig_credible_mass', 0.95",
            "'eig_minimum_utility', 0.05",
            "'eig_maximum_noncontracting_updates', 2",
            "'residual_maximum_connected_leg_mm', ...\n"
            "        activeResidualConnectedLegMm",
            "'tested_offset_tolerance_mm', 0.10",
            "'directional_progress_tolerance_mm', 0.10",
            "'swept_capture_tube_tolerance_mm', ...\n"
            "        m12ConservativeCaptureRadiusMm - 0.20",
            "'minimum_novel_swept_length_mm', 0.25",
            "'swept_segment_sample_spacing_mm', 0.05",
        ):
            self.assertIn(token, config)
        common_tokens = (
            "posterior, hypotheses, actionOffsets, plannedTestedOffsets, ...",
            "measuredTestedOffsets, measuredSweptSegments, currentOffset, ...",
            "admissibleActions, routingDirectionWorldXY, cfg)",
            "size(hypotheses, 1) == numel(posterior)",
            "size(actionOffsets, 1) == numel(admissibleActions)",
            "for index = 1:size(actionOffsets, 1)",
            "candidate = actionOffsets(index, :);",
            "travelMm = norm(candidate - currentOffset);",
            "residualCandidatePhysicalAdmission(candidate, currentOffset, ...",
            "plannedTestedOffsets, measuredTestedOffsets, ...",
            "measuredSweptSegments, routingDirectionWorldXY, cfg);",
            "if ~admissibleActions(index) || ...",
            "cfg.search.residual_maximum_connected_leg_mm + 1e-12",
            "~physicallyNovelAndForward",
            "pOpenGivenHypothesis = segmentCaptureLikelihood( ...",
            "hypotheses, currentOffset, candidate, cfg);",
            "actionOffsets(candidateIndex, :) - currentOffset",
        )
        for selector in (eig_selector, coverage_selector):
            for token in common_tokens:
                self.assertIn(token, selector)
            self.assertNotIn(
                "hypotheses(candidateIndex, :) - currentOffset",
                selector,
                "candidate indices belong to the action set, not the "
                "larger center-hypothesis set",
            )
        for token in (
            "size(plannedTestedOffsets, 2) == 2",
            "size(measuredTestedOffsets, 2) == 2",
            "size(measuredSweptSegments, 2) == 4",
            "directionalProgressMm = dot(candidate - currentOffset, direction);",
            "globalForwardProgressMm = candidateProjectionMm - "
            "historyFrontierMm;",
            "alreadyTested(candidate, plannedTestedOffsets, ...",
            "alreadyTested(candidate, measuredTestedOffsets, ...",
            "-cfg.search.directional_progress_tolerance_mm - 1e-12",
            "novelSweptLengthMm = measuredSweptNovelLength( ...",
            "cfg.search.minimum_novel_swept_length_mm - 1e-12",
        ):
            self.assertIn(token, physical_admission)
        for token in (
            "cfg.search.swept_segment_sample_spacing_mm",
            "fraction = (intervalIndex - 0.5) / intervalCount;",
            "pointToSegmentDistance2d(sample, historyStart, historyEnd)",
            "minimumDistanceMm > ...\n"
            "            cfg.search.swept_capture_tube_tolerance_mm",
        ):
            self.assertIn(token, swept_novelty)
        for token in (
            "credibleSupportMask( ...\n"
            "    posterior, cfg.search.eig_credible_mass)",
            "cfg.search.eig_information_weight * information + ...",
            "cfg.search.eig_capture_reward_weight * pOpen + ...",
            "cfg.search.eig_forward_progress_weight_per_mm * ...",
            "max(0.0, globalForwardProgressMm) + ...",
            "cfg.search.eig_novel_coverage_weight_per_mm * ...",
            "novelSweptLengthMm - ...",
            "cfg.search.eig_action_cost - ...",
            "cfg.search.eig_travel_penalty_per_mm * travelMm",
            "DIRECTION_CONDITIONED_CONNECTED_95PCT_CREDIBLE_RESIDUAL_EIG",
        ):
            self.assertIn(token, eig_selector)
        for token in (
            "for coveragePass = 1:2",
            "DIRECTION_CONDITIONED_CONNECTED_95PCT_CAD_COVERAGE",
            "DIRECTION_CONDITIONED_CONNECTED_COMMON_REACH_CAD_COVERAGE",
            "segmentCaptureLikelihood( ...",
            "cfg.search.coverage_capture_reward_weight * pOpen + ...",
            "cfg.search.eig_forward_progress_weight_per_mm * ...",
            "cfg.search.eig_novel_coverage_weight_per_mm * ...",
            "cfg.search.eig_travel_penalty_per_mm * travelMm",
        ):
            self.assertIn(token, coverage_selector)
        self.assertNotIn(
            "cfg.search.eig_minimum_utility",
            coverage_selector,
            "CAD coverage must not inherit the EIG admission threshold",
        )

        search_loop = self.main[
            self.main.index("while ~capture.found && ...\n"
                            "            totalSearchTargets <") :
            self.main.index(
                "result.residual_eig_steps = residualEigSteps;",
                self.main.index("while ~capture.found && ...\n"
                                "            totalSearchTargets <"),
            )
        ]
        for event in (
            "RESIDUAL_EIG_CREDIBLE_RADIUS_SWITCH_TO_CAD_COVERAGE",
            "RESIDUAL_EIG_UTILITY_SWITCH_TO_CAD_COVERAGE",
            "RESIDUAL_EIG_NONCONTRACTING_SWITCH_TO_CAD_COVERAGE",
        ):
            self.assertIn(event, search_loop)
        residual_route = self.main[
            self.main.index(
                "result.entropy_grid_selection_regions = cell(0, 1);"
            ) :
            self.main.index("result.residual_eig_steps = residualEigSteps;")
        ]
        for token in (
            "activeResidualActionMask = true(size(actionOffsets, 1), 1);",
            "directionQualified = norm(activeGridRoutingDirection) > 1e-12;",
            "activeResidualActionMask = directionalSectorMask( ...\n"
            "            actionOffsets, activeGridRoutingDirection, ...",
            "centerHypotheses, actionOffsets, ...\n"
            "                plannedTestedOffsets, measuredTestedOffsets, ...",
            "measuredSweptSegments, currentGridOffset, ...",
            "activeResidualActionMask, activeGridRoutingDirection, cfg",
            "plannedTestedOffsets(end + 1, :) = offset;",
            "measuredTestedOffsets(end + 1, :) = ...",
            "measuredSweptSegments = appendMeasuredSurfaceSegments( ...",
            "probe.measured_xy_polyline_mm",
            "surfaceNegativeObservationValid(probe)",
        ):
            self.assertIn(token, residual_route)
        self.assertGreaterEqual(
            search_loop.count("selectCadCoverageNode(posterior, ..."),
            2,
        )
        self.assertIn(
            "noncontractingUpdates >= ...\n"
            "                cfg.search.eig_maximum_noncontracting_updates",
            search_loop,
        )
        self.assertNotIn("RESIDUAL_EIG_NONCONTRACTING_STOP", search_loop)
        self.assertNotIn(
            "RESIDUAL_EIG_CREDIBLE_RADIUS_CAPTURE_SCALE_STOP",
            search_loop,
        )
        noncontract_switch = search_loop[
            search_loop.index(
                "RESIDUAL_EIG_NONCONTRACTING_SWITCH_TO_CAD_COVERAGE"
            ) :
        ]
        self.assertNotIn(
            "break;",
            noncontract_switch,
            "noncontracting evidence must switch mode, not stop the search",
        )

    def test_residual_physical_history_rejects_old_sweep_and_backtracking(
        self,
    ) -> None:
        endpoint_tolerance_mm = 0.10
        swept_tolerance_mm = 0.10
        minimum_novel_length_mm = 0.25
        sample_spacing_mm = 0.05
        planned_endpoints = [(0.0, 0.0), (1.0, 0.0)]
        measured_endpoints = [(0.0, 0.0), (1.0, 0.0)]
        measured_segments = [(0.0, 0.0, 1.0, 0.0)]
        learned_direction = (1.0, 0.0)

        def distance_to_segment(
            point: tuple[float, float],
            segment: tuple[float, float, float, float],
        ) -> float:
            start = segment[:2]
            end = segment[2:]
            delta = (end[0] - start[0], end[1] - start[1])
            denominator = delta[0] ** 2 + delta[1] ** 2
            if denominator <= 1e-15:
                return math.dist(point, start)
            fraction = (
                (point[0] - start[0]) * delta[0]
                + (point[1] - start[1]) * delta[1]
            ) / denominator
            fraction = min(max(fraction, 0.0), 1.0)
            closest = (
                start[0] + fraction * delta[0],
                start[1] + fraction * delta[1],
            )
            return math.dist(point, closest)

        def novel_length(
            current: tuple[float, float],
            candidate: tuple[float, float],
        ) -> float:
            length = math.dist(current, candidate)
            interval_count = max(1, math.ceil(length / sample_spacing_mm))
            novel_intervals = 0
            for index in range(interval_count):
                fraction = (index + 0.5) / interval_count
                sample = (
                    current[0] + fraction * (candidate[0] - current[0]),
                    current[1] + fraction * (candidate[1] - current[1]),
                )
                if min(
                    distance_to_segment(sample, segment)
                    for segment in measured_segments
                ) > swept_tolerance_mm:
                    novel_intervals += 1
            return length * novel_intervals / interval_count

        def admitted(
            current: tuple[float, float],
            candidate: tuple[float, float],
        ) -> bool:
            if any(
                math.dist(candidate, endpoint) <= endpoint_tolerance_mm
                for endpoint in planned_endpoints + measured_endpoints
            ):
                return False
            progress = (
                (candidate[0] - current[0]) * learned_direction[0]
                + (candidate[1] - current[1]) * learned_direction[1]
            )
            if progress < -endpoint_tolerance_mm - 1e-12:
                return False
            return novel_length(current, candidate) >= (
                minimum_novel_length_mm - 1e-12
            )

        self.assertFalse(
            admitted((0.50, 0.0), (0.75, 0.0)),
            "a new endpoint on an old measured sweep is not new coverage",
        )
        self.assertFalse(
            admitted((1.0, 0.0), (0.75, 0.0)),
            "the learned-direction route may not reverse by 0.25 mm",
        )
        self.assertFalse(
            admitted((1.0, 0.0), (1.0, 0.05)),
            "planned and measured endpoint history are both exclusion sets",
        )
        self.assertTrue(admitted((1.0, 0.0), (1.50, 0.0)))
        self.assertTrue(
            admitted((1.0, 0.0), (0.90, 0.90)),
            "exactly 0.10 mm measured back-projection remains admissible",
        )
        self.assertFalse(admitted((1.0, 0.0), (0.899, 0.90)))

    def test_local_spiral_uses_measured_endpoint_q95_and_cad_pitch(
        self,
    ) -> None:
        config = self.main[
            self.main.index("cfg.baselines = struct(") :
            self.main.index("cfg.learning = struct(")
        ]
        self.assertIn(
            "'proposed_local_pitch_mm_per_revolution', ...\n"
            "        1.60 * m12ConservativeCaptureRadiusMm",
            config,
        )
        self.assertIn(
            "vecnorm(centerHypotheses - measuredShortcutOffset, 2, 2)",
            self.main,
        )
        self.assertIn(
            "posterior, 0.95) - cfg.search.bayes_capture_radius_mm;",
            self.main,
        )
        shortcut_posterior = self.main[
            self.main.index(
                "% A miss updates the posterior only when the measured endpoint"
            ) :
            self.main.index("gridStep = 0;")
        ]
        for token in (
            "posterior = updatePosteriorAlongSegment(posterior, ...",
            "posterior = conditionPosteriorOnDirectionalSector( ...",
            "activeGridRoutingDirection, corridorHalfAngleDeg, ...",
            "cfg.search.directional_sector_outside_likelihood_floor);",
            "'LEARNED_DIRECTION_SECTOR_AFTER_MEASURED_SHORTCUT_OBSERVATION'",
            "'ORIGINAL_MEASURED_TOUCHDOWN_ANCHOR'",
            "vecnorm(centerHypotheses - measuredShortcutOffset, 2, 2)",
            "posterior_residual_radius_measured_endpoint_mm",
            "residual_posterior_sector_half_angle_deg",
            "residual_radius_uses_measured_shortcut_endpoint = ...\n"
            "            true;",
        ):
            self.assertIn(token, shortcut_posterior)
        self.assertLess(
            shortcut_posterior.index(
                "posterior = updatePosteriorAlongSegment(posterior, ..."
            ),
            shortcut_posterior.index(
                "posterior = conditionPosteriorOnDirectionalSector( ..."
            ),
        )
        self.assertLess(
            shortcut_posterior.index(
                "posterior = conditionPosteriorOnDirectionalSector( ..."
            ),
            shortcut_posterior.index(
                "measuredResidualRadiusMm = weightedQuantile( ..."
            ),
        )
        self.assertNotIn(
            "centerHypotheses = centerHypotheses - measuredShortcutOffset",
            shortcut_posterior,
        )
        self.assertNotIn(
            "centerHypotheses = centerHypotheses + measuredShortcutOffset",
            shortcut_posterior,
        )
        local_call = re.search(
            r"runOutwardArchimedeanSearch\(ctx, shortcutAnchorXY,.*?"
            r"learnedShortcut\.residual_radius_mm",
            self.main,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(local_call)

    def test_all_surface_paths_share_10_6_3_speed_120_segment_and_guards(
        self,
    ) -> None:
        config = self.main[
            self.main.index("cfg.surface = struct(") :
            self.main.index("cfg.conical = struct(")
        ]
        for token in (
            "'xy_segment_mm', 1.20",
            "'speed_fast_mm_s', 10.0",
            "'speed_nominal_mm_s', 6.0",
            "'speed_cautious_mm_s', 3.0",
            "'transport_normal_force_min_n', 2.5",
            "'transport_normal_force_max_n', 5.5",
            "'force_correction_enter_low_n', 2.25",
            "'force_correction_enter_high_n', 5.75",
        ):
            self.assertIn(token, config)
        surface = function_slice(
            self.main,
            "[ctx, transport] = moveSurfaceForceControlledXY(",
            "[speedMmS, speedClass] = selectAdaptiveSurfaceSpeed(",
        )
        self.assertIn("selectAdaptiveSurfaceSpeed( ...", surface)
        selector = function_slice(
            self.main,
            "[speedMmS, speedClass] = selectAdaptiveSurfaceSpeed(",
            "transport = updateSurfaceTransportObservation(",
        )
        self.assertIn("speedMmS = cfg.speed_fast_mm_s;", selector)
        self.assertIn("speedMmS = cfg.speed_nominal_mm_s;", selector)
        self.assertIn("speedMmS = cfg.speed_cautious_mm_s;", selector)
        sensor = self.main[
            self.main.index("cfg.sensor = struct(") :
            self.main.index("cfg.motion = struct(")
        ]
        self.assertIn("'hard_force_delta_n', 25.0", sensor)
        self.assertIn("'hard_torque_delta_nm', 0.75", sensor)

    def test_active_route_has_no_classical_or_shadow_fallback(self) -> None:
        start = self.main.index("if conical.aperture_candidate")
        end = self.main.index("if ~capture.found", start)
        route = self.main[start:end]
        self.assertNotRegex(route.lower(), r"classical|shadow")
        self.assertIn("error('M12Demo:LearnedInferenceRequired'", route)
        self.assertNotIn("shortcutOffset = [0.0, 0.0]", route)
        self.assertIn(
            "result.learned_direction_task_deg = ...\n"
            "            learned.output.top_direction_deg;",
            route,
        )
        self.assertIn(
            "result.learned_circular_mean_direction_task_deg = ...",
            route,
        )

    def test_preflight_is_hash_bound_finite_and_under_three_seconds(
        self,
    ) -> None:
        preflight = function_slice(
            self.main,
            "preflight = validateLearnedRuntimePreflight(cfg, paths)",
            "assertActiveLearnedOutput(value, cfg, identifier)",
        )
        self.assertIn("'maximum_preflight_runtime_s', 3.0", self.main)
        self.assertIn("preflightClock = tic;", preflight)
        self.assertIn(
            "preflightElapsedS = toc(preflightClock);", preflight
        )
        self.assertIn(
            "preflightElapsedS < "
            "cfg.learning.maximum_preflight_runtime_s",
            preflight,
        )
        self.assertIn(
            "'M12Demo:LearnedRuntimePreflightTooSlow'", preflight
        )
        self.assertIn(
            "assertActiveLearnedOutput(value, cfg, ...", preflight
        )

    def test_controller_input_and_bundle_trust_anchor_share_new_schema(
        self,
    ) -> None:
        inference = function_slice(
            self.main,
            "learned = inferFrozenConicalModel(cone, cfg, paths)",
            "[ctx, capture, diagnostics] = runOutwardArchimedeanSearch(",
        )
        for token in (
            "'schema_version', 2",
            "'model_fx_n'",
            "'model_fy_n'",
            "'model_fz_n'",
            "'model_tx_nm'",
            "'model_ty_nm'",
            "'model_tz_nm'",
            "'model_cmd_z_relative_mm'",
        ):
            self.assertIn(token, inference)
        output_contract = function_slice(
            self.main,
            "assertActiveLearnedOutput(value, cfg, identifier)",
            "validateSurfaceTransportContract(cfg)",
        )
        for token in (
            "abs(sum(directionProbability) - 1.0) <= 1e-6",
            "abs(sum(stateProbability) - 1.0) <= 1e-6",
            '"supported_search"; ...',
            '"double_overhang"; "outer_partial_support"',
            "'TOP1_POSTERIOR_MODE'",
            "weak_complete_fingerprint_remains_actionable",
            "posterior_quality_is_validity_veto",
            "hardware_control_sizes",
        ):
            self.assertIn(token, output_contract)

        bundle = ROOT / "model_runtime" / "bundles" / (
            "real_multisize_direction_v1"
        )
        manifest = json.loads(
            (bundle / "deployment_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["model_id"], "real_multisize_direction_v1")
        self.assertEqual(manifest["deployment_authority"], "CAPPED_SHORTCUT")
        self.assertIn("CAPPED_SHORTCUT", manifest["allowed_execution_modes"])
        for path_key, hash_key in (
            (
                "ensemble_interface_contract",
                "ensemble_interface_contract_sha256",
            ),
            ("promotion_contract", "promotion_contract_sha256"),
        ):
            artifact = bundle / manifest[path_key]
            self.assertTrue(artifact.is_file())
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                manifest[hash_key],
            )
        interface = json.loads(
            (bundle / manifest["ensemble_interface_contract"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("m12", interface["supported_sizes"])
        self.assertFalse(
            interface["policy_boundary"][
                "capture_insertion_or_terminal_authority"
            ]
        )
        self.assertFalse(
            interface["policy_boundary"]["pure_outward_spiral_consumer"]
        )
        promotion = json.loads(
            (bundle / manifest["promotion_contract"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(promotion["qualification"], "DEVELOPMENT_CV_CONTROL")
        self.assertTrue(
            promotion["authorization"]["capped_shortcut_hardware_control"]
        )
        self.assertFalse(promotion["authorization"]["capture_control"])
        self.assertFalse(promotion["authorization"]["insertion_control"])
        self.assertFalse(promotion["authorization"]["terminal_verification"])


if __name__ == "__main__":
    unittest.main()
