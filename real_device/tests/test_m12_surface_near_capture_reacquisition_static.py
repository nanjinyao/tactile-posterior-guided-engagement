from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"


def matlab_function(source: str, name: str) -> str:
    declaration = re.search(
        rf"^function(?:(?:[^\n]*\.\.\.\n\s*)*)"
        rf"[^\n]*\b{re.escape(name)}\b[^\n]*$",
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


def matlab_call(
    source: str,
    assignment: str,
    function: str,
) -> re.Match[str]:
    match = re.search(
        rf"{re.escape(assignment)}\s*=\s*"
        rf"{re.escape(function)}\(.*?\);",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(
            f"missing MATLAB call {assignment} = {function}(...)"
        )
    return match


class TestM12SurfaceNearCaptureReacquisitionStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.sensor = matlab_struct(cls.main, "sensor")
        cls.surface_config = matlab_struct(cls.main, "surface")
        cls.surface = matlab_function(
            cls.main, "moveSurfaceForceControlledXY"
        )
        cls.reacquire = matlab_function(
            cls.main, "reacquireOrConfirmNearCaptureEndpoint"
        )
        cls.low_force_cue = matlab_function(
            cls.main, "surfaceLowForceApertureCue"
        )
        cls.soft_state = matlab_function(
            cls.main, "surfaceCaptureSoftState"
        )
        cls.guard_dwell = matlab_function(cls.main, "guardDwell")

    def test_zero_load_endpoint_reacquires_then_degrades_without_abort(
        self,
    ) -> None:
        cue = self.surface.index("surfaceLowForceApertureCue(")
        reacquire = self.surface.index(
            "reacquireOrConfirmNearCaptureEndpoint(", cue
        )
        self.assertLess(cue, reacquire)
        self.assertNotIn("error('M12Demo:SurfaceForceSettle'", self.surface)
        self.assertIn(
            "SURFACE_FORCE_SETTLE_DEGRADED_CONTINUE", self.surface
        )
        self.assertIn("transport.no_lift_contact_hold = true;", self.surface)
        self.assertIn(
            "ctx.surface_capture_contact_hold_active = true;", self.surface
        )

        for token in (
            "~guard.event_confirmed",
            "normalForceN <= "
            "cfg.surface.aperture_low_force_normal_max_n",
            "guard.maximum_force_delta_n <= ...",
            "cfg.surface.aperture_low_force_total_max_n",
            "guard.maximum_torque_delta_nm <",
            "cfg.surface.capture_soft_torque_delta_nm",
            "cfg.surface.soft_relief_trigger_torque_nm",
        ):
            self.assertIn(token, self.low_force_cue)
        self.assertNotIn("error(", self.low_force_cue)

        reacquire_call = re.search(
            r"(?:\[.*?\]\s*=\s*)?"
            r"reacquireOrConfirmNearCaptureEndpoint\(.*?\);",
            self.surface,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(reacquire_call)
        assert reacquire_call is not None
        between = self.surface[
            cue : reacquire_call.start()
        ]
        self.assertNotIn(
            "M12Demo:SurfaceForceSettle",
            between,
            "a persistent near-zero aperture cue must not throw before "
            "bounded physical reacquisition",
        )

    def test_reacquisition_is_frozen_xy_downward_and_finite(self) -> None:
        self.assertRegex(
            self.reacquire,
            r"for\s+\w+\s*=\s*0\s*:\s*"
            r"ctx\.cfg\.surface\.capture_confirmation_max_steps",
        )
        for token in (
            "surfaceZ - ...",
            "ctx.cfg.surface.maximum_down_before_capture_mm",
            "ctx.cfg.surface.maximum_z_step_mm",
            "ctx.cfg.surface.maximum_z_correction_lateral_drift_mm",
        ):
            self.assertIn(token, self.reacquire)

        move = re.search(
            r"\[[^\]]*ctx[^\]]*\]\s*=\s*moveOne\("
            r"ctx\s*,\s*(?:\.\.\.\s*)?"
            r"(?P<delta>\[[^\]]+\]).*?\);",
            self.reacquire,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            move,
            "near-capture reacquisition must issue one explicit bounded move",
        )
        assert move is not None
        delta = re.sub(r"\s+", "", move.group("delta"))
        self.assertRegex(
            delta,
            r"^\[0(?:\.0+)?,0(?:\.0+)?,(?:-"
            r"[A-Za-z]\w*|zStep(?:Mm)?)\]$",
            "near-capture motion must freeze XY and command only WORLD -Z",
        )
        if not delta.split(",")[-1].startswith("-"):
            self.assertRegex(
                self.reacquire[: move.start()],
                r"zStep(?:Mm)?\s*=\s*-\s*",
                "an indirect Z step passed to moveOne must be explicitly "
                "negative",
            )
        self.assertNotRegex(
            self.reacquire,
            r"moveOne\(.*?\[\s*0(?:\.0+)?\s*,\s*"
            r"0(?:\.0+)?\s*,\s*\+",
        )

    def test_every_reacquisition_move_gets_fresh_guard_dwell(self) -> None:
        move = re.search(
            r"\[[^\]]*ctx[^\]]*\]\s*=\s*moveOne\(.*?\);",
            self.reacquire,
            flags=re.DOTALL,
        )
        dwell = re.search(
            r"\[[^\]]*ctx[^\]]*\]\s*=\s*guardDwell\(.*?\);",
            self.reacquire,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(move)
        self.assertIsNotNone(dwell)
        assert move is not None and dwell is not None
        self.assertLess(move.end(), dwell.start())
        between = self.reacquire[move.end() : dwell.start()]
        for forbidden in ("moveOne(", "return;"):
            self.assertNotIn(
                forbidden,
                between,
                "fresh F/T hard-guard dwell must immediately follow each "
                "blocking downward endpoint",
            )
        self.assertNotRegex(
            between,
            r"(?m)^\s*continue;\s*$",
            "diagnostic text may say continue, but control flow must not "
            "skip the fresh guard dwell",
        )
        self.assertNotRegex(
            between,
            r"\bguard\s*=",
            "the post-move wrench decision must come from the fresh "
            "guardDwell rather than a fabricated guard state",
        )
        dwell_call = dwell.group(0)
        self.assertIn(
            "ctx.cfg.sensor.contact_consecutive_samples",
            dwell_call,
        )
        self.assertIn("ctx.cfg.sensor.contact_event_n", dwell_call)
        self.assertNotIn("try", self.reacquire)
        self.assertNotIn("catch", self.reacquire)

    def test_low_force_measured_free_drop_can_confirm_capture(self) -> None:
        capture_outcome = "recovery.outcome = 'CAPTURE_CONFIRMED';"
        capture = self.reacquire.index(capture_outcome)
        capture_if = self.reacquire.rfind(
            "if currentDepthMm", 0, capture
        )
        self.assertGreaterEqual(capture_if, 0)
        evidence = self.reacquire[capture_if:capture]
        for token in (
            "currentDepthMm",
            "ctx.cfg.surface.capture_depth_drop_mm",
            "extraDropMm",
            "ctx.cfg.surface.capture_confirmation_extra_drop_mm",
            "surfaceCaptureSoftState(normalForceN, guard, ctx.cfg)",
        ):
            self.assertIn(token, evidence)
        self.assertNotIn(
            "guard.event_confirmed",
            evidence,
            "measured low-load aperture descent is itself capture evidence; "
            "requiring a contact event recreates the zero-newton dead end",
        )
        capture_branch = self.reacquire[
            capture : self.reacquire.index("return;", capture)
        ]
        self.assertIn(
            "surfaceLowForceApertureCue(normalForceN, guard, ctx.cfg)",
            capture_branch,
        )
        self.assertIn(
            "'LOW_FORCE_APERTURE_MEASURED_FREE_DROP'",
            capture_branch,
        )
        self.assertIn("'CONTACT_REACQUIRED'", self.reacquire)
        self.assertIn(
            "'FINITE_REACQUISITION_EXHAUSTED'",
            self.reacquire,
        )

        handled_capture = self.surface.index(
            "strcmp(nearCapture.outcome, 'CAPTURE_CONFIRMED')"
        )
        capture_return = self.surface.index("return;", handled_capture)
        handled_text = self.surface[handled_capture:capture_return]
        self.assertIn("transport.capture_confirmed = true;", handled_text)
        self.assertIn(
            "nearCapture.capture_evidence_kind",
            handled_text,
        )
        self.assertIn("transport.terminal_reason", handled_text)

    def test_capture_soft_state_is_below_relief_and_needs_no_contact(
        self,
    ) -> None:
        for token in (
            "guard.maximum_force_delta_n <",
            "min(cfg.surface.capture_soft_force_delta_n, ...",
            "cfg.surface.soft_relief_trigger_force_n)",
            "guard.maximum_torque_delta_nm <",
            "min(cfg.surface.capture_soft_torque_delta_nm, ...",
            "cfg.surface.soft_relief_trigger_torque_nm)",
        ):
            self.assertIn(token, self.soft_state)
        self.assertIn(
            "supportedChamferState = guard.event_confirmed",
            self.soft_state,
        )
        self.assertRegex(
            self.soft_state,
            r"inside\s*=\s*insideWrenchEnvelope\s*&&\s*\.\.\.\s*"
            r"\(supportedChamferState\s*\|\|\s*\.\.\.\s*"
            r"surfaceLowForceApertureCue"
            r"\(normalForceN,\s*guard,\s*cfg\)\)",
            "capture must accept either supported contact or an independent "
            "low-force aperture cue inside the soft wrench envelope",
        )
        low_force_expression = self.low_force_cue[
            self.low_force_cue.index("cue =") :
        ]
        self.assertIn("~guard.event_confirmed", low_force_expression)

    def test_no_lift_capture_hold_persists_and_expands_without_replay(
        self,
    ) -> None:
        for token in (
            "surface_capture_contact_active_target_xy_mm",
            "ctx.surface_capture_contact_dither_index",
            "funnel_dither_points_per_ring",
            "funnel_dither_radial_increment_mm",
            "funnel_dither_maximum_radius_mm",
            "pi * (3.0 - sqrt(5.0))",
        ):
            self.assertIn(token, self.surface)
        self.assertIn(
            "ctx.surface_capture_contact_active_target_xy_mm = ...",
            self.surface,
        )
        self.assertIn(
            "captureHoldTargetXY = ditherTargetXY;",
            self.surface,
        )
        self.assertIn(
            "if transport.no_lift_contact_hold && zStep > 0.0",
            self.surface,
        )
        self.assertNotIn(
            "ditherAngle = 0.5 * pi * mod",
            self.surface,
            "a fixed four-point dither would repeat forever after one "
            "ambiguous capture cue",
        )

    def test_hard_force_and_torque_guards_remain_immediate(self) -> None:
        self.assertAlmostEqual(
            numeric_field(self.sensor, "hard_force_delta_n"),
            25.0,
        )
        self.assertAlmostEqual(
            numeric_field(self.sensor, "hard_torque_delta_nm"),
            0.75,
        )
        hard_force = self.guard_dwell.index(
            "if sample.force_delta_n >= "
            "ctx.cfg.sensor.hard_force_delta_n"
        )
        hard_torque = self.guard_dwell.index(
            "if sample.torque_delta_nm >= "
            "ctx.cfg.sensor.hard_torque_delta_nm"
        )
        event_latch = self.guard_dwell.index(
            "if sample.force_delta_n >= eventThresholdN"
        )
        self.assertLess(hard_force, event_latch)
        self.assertLess(hard_torque, event_latch)
        self.assertIn("M12Demo:HardForceGuard", self.guard_dwell)
        self.assertIn("M12Demo:HardTorqueGuard", self.guard_dwell)


if __name__ == "__main__":
    unittest.main()
