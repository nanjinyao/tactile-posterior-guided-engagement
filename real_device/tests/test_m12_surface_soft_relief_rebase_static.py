from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"


def matlab_function(source: str, name: str) -> str:
    declaration = re.search(
        rf"^function[^\n]*\b{re.escape(name)}\b[^\n]*$",
        source,
        flags=re.MULTILINE,
    )
    if declaration is None:
        raise AssertionError(f"missing MATLAB function {name}")
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


def matlab_struct(source: str, name: str) -> str:
    start_token = f"cfg.{name} = struct("
    start = source.find(start_token)
    if start < 0:
        raise AssertionError(f"missing MATLAB cfg.{name} struct")
    end = source.find(");", start)
    if end < 0:
        raise AssertionError(f"unterminated MATLAB cfg.{name} struct")
    return source[start : end + 2]


def numeric_field(struct_source: str, field: str) -> float:
    match = re.search(
        rf"'{re.escape(field)}'\s*,\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        struct_source,
    )
    if match is None:
        raise AssertionError(f"missing numeric MATLAB field {field}")
    return float(match.group(1))


def matlab_call(source: str, assignment: str, function: str) -> re.Match[str]:
    match = re.search(
        rf"{re.escape(assignment)}\s*=\s*{re.escape(function)}\(.*?\);",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(
            f"missing MATLAB call {assignment} = {function}(...)"
        )
    return match


class TestM12SurfaceSoftReliefRebaseStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.sensor = matlab_struct(cls.main, "sensor")
        cls.surface_config = matlab_struct(cls.main, "surface")
        cls.surface_controller = matlab_function(
            cls.main, "moveSurfaceForceControlledXY"
        )
        cls.move_one = matlab_function(cls.main, "moveOne")

        branch_start = cls.surface_controller.index(
            "if softReliefRequested"
        )
        branch_end = cls.surface_controller.index(
            "elseif guard.maximum_force_delta_n <=", branch_start
        )
        cls.soft_relief_branch = cls.surface_controller[
            branch_start:branch_end
        ]

    def test_contact_release_has_a_dedicated_bounded_tracking_contract(
        self,
    ) -> None:
        self.assertAlmostEqual(
            numeric_field(
                self.surface_config,
                "soft_relief_tracking_tolerance_mm",
            ),
            0.35,
        )
        self.assertAlmostEqual(
            numeric_field(
                self.surface_config,
                "maximum_z_correction_lateral_drift_mm",
            ),
            0.60,
        )
        self.assertAlmostEqual(
            numeric_field(
                self.surface_config,
                "soft_relief_measured_step_envelope_mm",
            ),
            0.60,
        )

        relief_move = matlab_call(
            self.soft_relief_branch,
            "[ctx, reliefPose]",
            "moveOne",
        ).group(0)
        self.assertIn(
            "ctx.cfg.surface.soft_relief_tracking_tolerance_mm",
            relief_move,
        )
        self.assertNotIn(
            "ctx.cfg.surface.tracking_tolerance_mm",
            relief_move,
        )
        self.assertIn(
            "'CONTACT_RELIEF_MEASURED_REBASE'",
            relief_move,
        )

    def test_fresh_wrench_guard_immediately_follows_relief_motion(self) -> None:
        relief_move = matlab_call(
            self.soft_relief_branch,
            "[ctx, reliefPose]",
            "moveOne",
        )
        fresh_guard = matlab_call(
            self.soft_relief_branch,
            "[ctx, guard]",
            "guardDwell",
        )
        self.assertLess(relief_move.end(), fresh_guard.start())

        between = self.soft_relief_branch[
            relief_move.end() : fresh_guard.start()
        ]
        self.assertIn("transport.steps = index;", between)
        for forbidden in (
            "moveOne(",
            "readKstPoseWithRecovery(",
            "pause(",
            "continue;",
            "return;",
            "error(",
        ):
            self.assertNotIn(forbidden, between)
        self.assertIn(
            "[phase, '_SOFT_RELIEF_DWELL']",
            fresh_guard.group(0),
        )

    def test_measured_release_pose_is_logged_before_search_continues(
        self,
    ) -> None:
        fresh_guard = matlab_call(
            self.soft_relief_branch,
            "[ctx, guard]",
            "guardDwell",
        )
        event = re.search(
            r"logEvent\(\s*ctx\.paths\.events,\s*\.\.\.\s*"
            r"'SURFACE_SOFT_RELIEF_MEASURED_REBASE_CONTINUE'.*?\);",
            self.soft_relief_branch,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(event)
        assert event is not None

        continuation = self.soft_relief_branch.rfind("continue;")
        self.assertGreater(event.start(), fresh_guard.end())
        self.assertGreater(continuation, event.end())
        event_text = event.group(0)
        self.assertIn("reliefPose(1:2) - currentPose(1:2)", event_text)
        self.assertIn("reliefPose(3) - currentPose(3)", event_text)
        self.assertIn(
            "ctx.cfg.surface.maximum_z_correction_lateral_drift_mm",
            event_text,
        )

    def test_soft_torque_relief_covers_the_observed_contact_release(
        self,
    ) -> None:
        observed_torque_nm = 0.418163747
        trigger = numeric_field(
            self.surface_config, "soft_relief_trigger_torque_nm"
        )
        resume = numeric_field(
            self.surface_config, "soft_relief_resume_torque_nm"
        )
        hard_guard = numeric_field(
            self.sensor, "hard_torque_delta_nm"
        )

        self.assertAlmostEqual(trigger, 0.50)
        self.assertAlmostEqual(resume, 0.40)
        self.assertLess(observed_torque_nm, trigger)
        self.assertLess(resume, trigger)
        self.assertLess(trigger, hard_guard)
        self.assertLessEqual(trigger, 0.70 * hard_guard)

    def test_contact_relief_rebases_on_a_finite_measured_step(self) -> None:
        for token in (
            "CONTACT_RELIEF_MEASURED_REBASE",
            "measuredStepDistanceMm = norm(measuredDelta);",
            "trackingAccepted = measuredStepDistanceMm <=",
            "ctx.cfg.surface.soft_relief_measured_step_envelope_mm",
            "CONTACT_RELIEF_TRACKING_MISMATCH_REBASED_TO_MEASURED_STATE",
            "M12Demo:ContactReliefMeasuredStepEnvelope",
        ):
            self.assertIn(token, self.move_one)

        ratio_gate = re.search(
            r"ratioGateRequired\s*=.*?;",
            self.move_one,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(ratio_gate)
        assert ratio_gate is not None
        self.assertIn(
            "~contactReliefMeasuredRebase",
            ratio_gate.group(0),
        )
        self.assertRegex(
            self.move_one,
            r"if\s+~contactReliefMeasuredRebase\s+&&\s+\.\.\.\s*"
            r"crossTrack\s*>\s*ctx\.cfg\.motion\.maximum_cross_track_mm",
        )

    def test_contact_relief_still_checks_fresh_wrench_and_surface_band(
        self,
    ) -> None:
        fresh_guard = matlab_call(
            self.soft_relief_branch,
            "[ctx, guard]",
            "guardDwell",
        )
        z_envelope = self.soft_relief_branch.index(
            "SURFACE_RELIEF_COMPLIANCE_NO_LIFT_HOLD"
        )
        continuation = self.soft_relief_branch.rfind("continue;")
        self.assertLess(fresh_guard.end(), z_envelope)
        self.assertLess(z_envelope, continuation)
        self.assertNotIn(
            "M12Demo:SurfaceReliefMeasuredZEnvelope",
            self.soft_relief_branch,
        )
        self.assertIn(
            "transport.no_lift_contact_hold = true;",
            self.soft_relief_branch,
        )
        self.assertIn(
            "ctx.cfg.surface.maximum_down_before_capture_mm",
            self.soft_relief_branch,
        )
        self.assertIn(
            "ctx.cfg.surface.maximum_up_from_surface_mm",
            self.soft_relief_branch,
        )

    def test_generic_tracking_and_hard_guards_remain_narrow(self) -> None:
        self.assertAlmostEqual(
            numeric_field(self.surface_config, "tracking_tolerance_mm"),
            0.20,
        )
        self.assertAlmostEqual(
            numeric_field(self.sensor, "hard_force_delta_n"),
            25.0,
        )
        self.assertAlmostEqual(
            numeric_field(self.sensor, "hard_torque_delta_nm"),
            0.75,
        )

        generic_move = matlab_call(
            self.surface_controller,
            "[ctx, postPose]",
            "moveOne",
        ).group(0)
        self.assertIn(
            "ctx.cfg.surface.tracking_tolerance_mm",
            generic_move,
        )
        self.assertNotIn(
            "ctx.cfg.surface.soft_relief_tracking_tolerance_mm",
            generic_move,
        )


if __name__ == "__main__":
    unittest.main()
