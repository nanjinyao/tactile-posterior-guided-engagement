from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"
DASHBOARD = ROOT / "m12_learned_postcone_dashboard.m"


class TestM12LearnedPostconeDashboardStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = MAIN.read_text(encoding="utf-8")
        cls.dashboard = DASHBOARD.read_text(encoding="utf-8")

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

    def test_dashboard_is_standalone_observation_only_code(self) -> None:
        self.assertIn(
            "function dashboard = "
            "m12_learned_postcone_dashboard(dashboard, payload)",
            self.dashboard,
        )
        self.assertIn("DISPLAY ONLY", self.dashboard)
        self.assertIn("NO CONTROL FEEDBACK", self.dashboard)

        forbidden_calls = (
            "timer",
            "system",
            "unix",
            "dos",
            "receive",
            "rossubscriber",
            "rosmessage",
            "fopen",
            "fclose",
            "fread",
            "fwrite",
            "fileread",
            "textscan",
            "readtable",
            "readmatrix",
            "readcell",
            "writetable",
            "writematrix",
            "writecell",
            "dlmread",
            "dlmwrite",
            "csvread",
            "csvwrite",
            "xlsread",
            "xlswrite",
            "imread",
            "imwrite",
            "load",
            "save",
            "saveas",
            "savefig",
            "exportgraphics",
            "print",
            "diary",
            "copyfile",
            "movefile",
            "mkdir",
            "rmdir",
        )
        for name in forbidden_calls:
            self.assertIsNone(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(",
                    self.dashboard,
                    flags=re.IGNORECASE,
                ),
                f"standalone dashboard must not call {name}",
            )
        for interface in (
            "iiwa.",
            "wrench_sub",
            "diagnostic_sub",
            "receiveWrench",
            "sendEEfPositionf",
            "realTime_startDirectServoCartesian",
            "realTime_stopDirectServoCartesian",
        ):
            self.assertNotIn(
                interface,
                self.dashboard,
                f"standalone dashboard must not access {interface}",
            )

    def test_seven_raw_channels_and_display_only_fourier_mad(self) -> None:
        for channel in (
            "model_fx_n",
            "model_fy_n",
            "model_fz_n",
            "model_tx_nm",
            "model_ty_nm",
            "model_tz_nm",
            "model_cmd_z_relative_mm",
        ):
            self.assertIn(channel, self.dashboard)

        physical_input = self._matlab_function(
            self.dashboard, "exactPhysicalPhaseGrid"
        )
        for channel in (
            "modelInput.model_fx_n(:)",
            "modelInput.model_fy_n(:)",
            "modelInput.model_fz_n(:)",
            "modelInput.model_tx_nm(:)",
            "modelInput.model_ty_nm(:)",
            "modelInput.model_tz_nm(:)",
            "modelInput.model_cmd_z_relative_mm(:)",
        ):
            self.assertIn(channel, physical_input)
        self.assertNotIn("normalize", physical_input.lower())
        self.assertNotIn("standardize", physical_input.lower())
        self.assertNotIn("payload.model_input =", self.dashboard)

        fourier = self._matlab_function(
            self.dashboard, "displayOnlyFourier"
        )
        for token in (
            "for order = 1:6",
            "for harmonic = 1:order",
            "cos(harmonic * phase)",
            "sin(harmonic * phase)",
            "residual = sequence - reconstruction(:, :, 6)",
            "residualMedian = median(residual, 1)",
            "residualMad = 1.4826 * median(",
            "abs(residual - residualMedian)",
            "2.5 * residualMad",
        ):
            self.assertIn(token, fourier)

        sequence_panel = self._matlab_function(
            self.dashboard, "renderSequencePanel"
        )
        for token in (
            "for channel = 1:7",
            "data.sequence(:, channel)",
            "for order = 1:6",
            "Exact physical model input before per-seed normalization",
            "thin raw",
            "DISPLAY ONLY / NOT MODEL CI",
        ):
            self.assertIn(token, sequence_panel)

    def test_aps_is_discrete_classes_not_a_fake_single_sector(self) -> None:
        aps = self._matlab_function(self.dashboard, "apsAngles")
        for token in (
            "aps_direction_angles_deg",
            "aps_direction_classes",
            "indices = round(double(output.aps_direction_classes(:))) + 1",
            "angles = directionAngles(indices)",
        ):
            self.assertIn(token, aps)
        for forbidden in ("linspace(", "patch(", "fill(", "area("):
            self.assertNotIn(forbidden, aps)

        top_view = self._matlab_function(
            self.dashboard, "renderTopView"
        )
        aps_start = top_view.index("apsHandle = plot(")
        aps_end = top_view.index("shortcutUv =", aps_start)
        aps_render = top_view[aps_start:aps_end]
        for token in (
            "'p'",
            "'LineStyle', 'none'",
            "'DisplayName', 'discrete APS'",
            "data.apsAnglesDeg",
        ):
            self.assertIn(token, aps_render)
        for forbidden in ("patch(", "fill(", "linspace("):
            self.assertNotIn(forbidden, aps_render)

        decision_panel = self._matlab_function(
            self.dashboard, "renderDecisionPanel"
        )
        self.assertIn(
            "plot(posteriorAxes, mod(data.apsAnglesDeg, 360.0)",
            decision_panel,
        )
        self.assertIn("'LineStyle', 'none'", decision_panel)

    def test_three_state_labels_do_not_expand_scalar_display_data(self) -> None:
        prepare = self._matlab_function(
            self.dashboard, "prepareDisplayData"
        )
        self.assertIn(
            "'stateNames', {stateNames(output)}",
            prepare,
        )
        self.assertNotIn(
            "'stateNames', stateNames(output)",
            prepare,
        )

    def test_controller_renders_after_inference_and_route_before_motion(
        self,
    ) -> None:
        inference = self.main.index(
            "learned = inferFrozenConicalModel(conical, cfg, paths);"
        )
        route = self.main.index(
            "learnedShortcut = planLearnedShortcut(",
            inference,
        )
        render = self.main.index(
            "ctx = bestEffortRenderLearnedPostConeDecision(",
            route,
        )
        transport = self.main.index(
            "'SHORTCUT_CONTACT_TRANSPORT'",
            render,
        )
        self.assertLess(inference, route)
        self.assertLess(route, render)
        self.assertLess(render, transport)
        self.assertIn(
            "'learned_postcone_enabled', true",
            self.main,
        )
        for token in (
            "'learned_postcone_visible', true",
            # Raising the window goes through the X11 window manager and can
            # block forever; it is cosmetic, so it defaults off.  See
            # test_phase_boundary_presentation_flags_are_independent.
            "'learned_postcone_bring_to_front', false",
            "'learned_postcone_single_phase_boundary_drawnow', true",
        ):
            self.assertIn(token, self.main)

        renderer = self._matlab_function(
            self.main, "bestEffortRenderLearnedPostConeDecision"
        )
        for token in (
            "'model_input', learned.model_input",
            "'learned_output', learned.output",
            "'grid_posterior', double(gridPosterior(:))",
            "'shortcut', routePlan",
            "'execution_mode', char(ctx.cfg.execution_mode)",
            "'direct_servo_active', logical(ctx.direct_servo_active)",
            "m12_learned_postcone_dashboard(",
        ):
            self.assertIn(token, renderer)
        self.assertIn(
            "~logical(ctx.direct_servo_active)",
            renderer,
        )
        self.assertIn(
            "ctx.learned_dashboard.phase_boundary_refresh.elapsed_s",
            renderer,
        )

    def test_phase_boundary_window_is_visible_front_and_flushed_once(
        self,
    ) -> None:
        main_dashboard = self._matlab_function(
            self.dashboard, "m12_learned_postcone_dashboard"
        )
        presenter = self._matlab_function(
            self.dashboard, "presentCompletedPhaseBoundaryFigure"
        )
        self.assertIn(
            "~payload.direct_servo_active",
            main_dashboard,
        )
        self.assertIn(
            "presentCompletedPhaseBoundaryFigure(figureHandle, payload.cfg)",
            main_dashboard,
        )
        for token in (
            "'learned_postcone_visible'",
            "'learned_postcone_bring_to_front'",
            "'learned_postcone_single_phase_boundary_drawnow'",
            "set(figureHandle, 'Visible', 'on');",
            "figure(figureHandle);",
            "drawnow limitrate nocallbacks;",
            "'drawnow_count', drawnowCount",
            "'stage', 'AFTER_DIRECTSERVO_STOP_BEFORE_BLOCKING_SEARCH'",
        ):
            self.assertIn(token, presenter)
        self.assertEqual(
            len(
                re.findall(
                    r"(?<![A-Za-z0-9_])drawnow\s+limitrate\s+nocallbacks\s*;",
                    self.dashboard,
                    flags=re.IGNORECASE,
                )
            ),
            1,
            "the deferred dashboard must flush exactly once per render",
        )

    def test_phase_boundary_presentation_flags_are_independent(self) -> None:
        # These three flags were once welded together by a single assert that
        # demanded all of them, so none could be used to mitigate anything.
        # On 2026-07-30 this presenter intermittently blocked forever -- four
        # trials hung for 30 s until the operator interrupted, with no window
        # ever shown, while successful renders report 0.000-1.000 s.  Each flag
        # must now gate only its own step so the blocking one can be turned off
        # without giving up the display entirely.
        presenter = self._matlab_function(
            self.dashboard, "presentCompletedPhaseBoundaryFigure"
        )
        self.assertIn("if bringToFront\n    figure(figureHandle);", presenter)
        self.assertIn("if flushOnce\n", presenter)
        self.assertIn("'brought_to_front', bringToFront", presenter)
        # The assert may require only the visible flag; requiring the other two
        # would re-weld them.
        assert_block = presenter[
            presenter.index("assert(") : presenter.index("bringToFront =")
        ]
        self.assertIn("learned_postcone_visible", assert_block)
        self.assertNotIn("learned_postcone_bring_to_front", assert_block)
        self.assertNotIn(
            "learned_postcone_single_phase_boundary_drawnow", assert_block
        )

    def test_execution_mode_is_prominent_for_each_learned_call(self) -> None:
        main_dashboard = self._matlab_function(
            self.dashboard, "m12_learned_postcone_dashboard"
        )
        prepare = self._matlab_function(
            self.dashboard, "prepareDisplayData"
        )
        decision_text = self._matlab_function(
            self.dashboard, "decisionText"
        )
        self.assertIn(
            "printable(payload.execution_mode)",
            main_dashboard,
        )
        self.assertIn(
            "'executionMode', printable(payload.execution_mode)",
            prepare,
        )
        self.assertIn(
            "sprintf('method: %s', data.executionMode)",
            decision_text,
        )

    def test_conical_directservo_loop_has_no_dashboard_fft_or_graphics(
        self,
    ) -> None:
        conical = self._matlab_function(
            self.main, "runConicalFingerprint"
        )
        loop_start = conical.index(
            "for index = 1:count\n    enforceTime(ctx);"
        )
        loop_end = conical.index(
            "streamElapsedS = toc(servoClock);",
            loop_start,
        )
        servo_loop = conical[loop_start:loop_end]

        forbidden = (
            "m12_learned_postcone_dashboard",
            "bestEffortRenderLearnedPostConeDecision",
            "displayOnlyFourier",
            "exactPhysicalPhaseGrid",
            "fft(",
            "ifft(",
            "drawnow",
            "timer(",
            "figure(",
            "axes(",
            "tiledlayout(",
            "nexttile(",
            "plot(",
            "plot3(",
            "polarplot(",
            "scatter(",
            "polarscatter(",
            "patch(",
            "fill(",
            "bar(",
            "barh(",
            "quiver(",
            "quiver3(",
            "annotation(",
        )
        for token in forbidden:
            self.assertNotIn(
                token,
                servo_loop,
                f"DirectServo loop contains deferred display work: {token}",
            )


if __name__ == "__main__":
    unittest.main()
