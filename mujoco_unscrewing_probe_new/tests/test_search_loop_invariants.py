from __future__ import annotations

import unittest

import numpy as np

from src.entropy_grid_baseline import EntropyGridBelief, HybridProbeConfig
from src.render_m8_search_comparison import (
    _advance_dwell_count,
    _apply_no_capture_measured_polyline,
    _bounded_active_dt,
    _declared_preprobe_route,
    _hybrid_fallback_reserve_seconds,
    _nominal_terminal_execution_reserve_seconds,
    _optional_action_fits_nominal_reserve,
    run_method,
)


class TestSearchLoopInvariants(unittest.TestCase):
    def test_active_clock_cannot_cross_hard_budget(self) -> None:
        self.assertAlmostEqual(_bounded_active_dt(0.090, 0.020, 0.100), 0.010)
        self.assertEqual(_bounded_active_dt(0.100, 0.020, 0.100), 0.0)
        self.assertEqual(_bounded_active_dt(0.120, 0.020, 0.100), 0.0)

    def test_hybrid_fallback_reserve_is_geometry_and_speed_derived(self) -> None:
        reserve = _hybrid_fallback_reserve_seconds(
            np.array([0.003, 0.004]),
            np.zeros(2),
            fallback_path_length_m=0.090,
            xy_speed_m_s=0.002,
            return_position_error_bound_m=0.0,
            transition_overhead_s=1.0,
            candidate_contingency_s=0.0,
            terminal_execution_reserve_s=12.0,
        )
        self.assertAlmostEqual(reserve, 60.5)
        anchor_reserve = _hybrid_fallback_reserve_seconds(
            np.zeros(2),
            np.zeros(2),
            fallback_path_length_m=0.090,
            xy_speed_m_s=0.002,
            return_position_error_bound_m=0.0,
            transition_overhead_s=1.0,
            candidate_contingency_s=0.0,
            terminal_execution_reserve_s=12.0,
        )
        self.assertAlmostEqual(anchor_reserve, 58.25)
        config = HybridProbeConfig()
        self.assertGreater(config.terminal_execution_reserve_s, 0.0)
        self.assertGreaterEqual(config.fallback_transition_overhead_s, 0.0)

    def test_budget_gate_uses_action_endpoint_and_detects_infeasible_entry(self) -> None:
        near = _hybrid_fallback_reserve_seconds(
            np.zeros(2),
            np.zeros(2),
            fallback_path_length_m=0.120,
            xy_speed_m_s=0.002,
            return_position_error_bound_m=0.00025,
            transition_overhead_s=1.15,
            candidate_contingency_s=6.0,
            terminal_execution_reserve_s=12.0,
        )
        endpoint = _hybrid_fallback_reserve_seconds(
            np.array([0.004, 0.0]),
            np.zeros(2),
            fallback_path_length_m=0.120,
            xy_speed_m_s=0.002,
            return_position_error_bound_m=0.00025,
            transition_overhead_s=1.15,
            candidate_contingency_s=6.0,
            terminal_execution_reserve_s=12.0,
        )
        self.assertGreater(endpoint, near)
        self.assertFalse(_optional_action_fits_nominal_reserve(70.0, 0.0, endpoint))
        self.assertTrue(_optional_action_fits_nominal_reserve(100.0, 2.0, endpoint))

    def test_terminal_reserve_is_cad_depth_conditioned_and_capture_only_is_zero(self) -> None:
        kwargs = dict(
            socket_depth_m=0.004,
            conservative_capture_z_m=0.0003,
            axial_command_speed_m_s=0.00035,
            minimum_descent_s=6.0,
            force_release_s=0.30,
            insertion_handoff_s=1.0,
            recovery_count=1,
            recovery_duration_s=0.45,
            disturbance_duration_s=0.0,
            completion_event_margin_s=0.25,
            minimum_reserve_s=12.0,
        )
        self.assertGreaterEqual(
            _nominal_terminal_execution_reserve_seconds(
                continue_to_bottom=True, **kwargs
            ),
            12.0,
        )
        self.assertEqual(
            _nominal_terminal_execution_reserve_seconds(
                continue_to_bottom=False, **kwargs
            ),
            0.0,
        )

    def test_short_rollout_reports_hard_budget_without_extension(self) -> None:
        result = run_method(
            "baseline",
            active_time_budget_s=0.010,
            continue_to_bottom=False,
        )
        self.assertLessEqual(result.total_active_time_s, 0.010 + 1.0e-12)
        self.assertEqual(result.fallback_budget_extension_s, 0.0)
        self.assertEqual(result.result, "common_time_budget_exhausted")
        self.assertIsNotNone(result.budget_exhaustion_stage)

    def test_transient_capture_cue_does_not_satisfy_dwell(self) -> None:
        count = 0
        accepted = False
        for cue in (True, False, True, True):
            count, accepted = _advance_dwell_count(count, cue, required=3)
            self.assertFalse(accepted)
        count, accepted = _advance_dwell_count(count, True, required=3)
        self.assertTrue(accepted)

    def test_inactive_terminal_display_tail_cannot_redefine_success(self) -> None:
        result = run_method(
            "oracle",
            initial_xy_m=np.array([0.004, 0.0]),
            active_time_budget_s=10.0,
            continue_to_bottom=True,
            conditioned_on_ambiguous_top_contact=True,
        )
        self.assertEqual(result.result, "cad_bottom_verified")
        self.assertTrue(result.bottom_reached)
        self.assertTrue(result.offline_capture_validated)
        self.assertFalse(result.hard_stop)
        self.assertFalse(result.terminal_outcome_hard_stop)
        self.assertEqual(
            result.offline_capture_scoring_snapshot,
            "pre_inactive_post_terminal_display",
        )
        self.assertNotIn("terminal_hold", result.stage_time_s)
        self.assertIn("terminal_hold", result.stage_max_raw_physical_force_n)

    def test_miss_update_uses_measured_polyline_union_once(self) -> None:
        kwargs = dict(
            visual_anchor_xy_m=np.zeros(2),
            visual_sigma_m=0.002,
            visual_bound_m=0.004895,
            support_inner_radius_m=0.00325,
            support_outer_radius_m=0.00650,
            probe_radius_m=0.003464,
        )
        simple = EntropyGridBelief(**kwargs)
        sampled = EntropyGridBelief(**kwargs)
        measured_simple = np.array([[0.0, -0.002], [0.0, 0.002]])
        measured_sampled = np.array(
            [
                [0.0, -0.002],
                [0.0, -0.001],
                [0.0, 0.0],
                [0.0, 0.001],
                [0.0, 0.002],
                [0.0, 0.002],
            ]
        )
        update_kwargs = dict(
            effective_capture_radius_m=0.00025,
            miss_inside_likelihood=0.20,
            miss_outside_likelihood=0.995,
        )
        _apply_no_capture_measured_polyline(simple, measured_simple, **update_kwargs)
        _apply_no_capture_measured_polyline(sampled, measured_sampled, **update_kwargs)
        np.testing.assert_allclose(simple.weights, sampled.weights, atol=1.0e-15)

        vertical = int(
            np.argmin(
                np.linalg.norm(
                    simple.candidate_xy_m - np.array([[0.0, 0.0015]]), axis=1
                )
            )
        )
        horizontal = int(
            np.argmin(
                np.linalg.norm(
                    simple.candidate_xy_m - np.array([[0.0015, 0.0]]), axis=1
                )
            )
        )
        self.assertLess(simple.weights[vertical], simple.weights[horizontal])

    def test_ambiguity_condition_is_declared_not_invented(self) -> None:
        self.assertEqual(
            _declared_preprobe_route("hybrid", True),
            "conditioned_ambiguous_top_contact_probe",
        )
        self.assertEqual(
            _declared_preprobe_route("hybrid", False),
            "no_upstream_ambiguity_router_conservative_probe",
        )
        self.assertEqual(
            _declared_preprobe_route("baseline", True),
            "not_applicable_no_conical_probe",
        )


if __name__ == "__main__":
    unittest.main()
