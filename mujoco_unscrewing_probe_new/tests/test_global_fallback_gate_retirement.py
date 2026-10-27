from __future__ import annotations

import unittest

from src.render_m8_search_comparison import (
    CaptureDirectionGateLifecycle,
    RunResult,
)


class TestGlobalFallbackGateRetirement(unittest.TestCase):
    def test_enabled_gate_retires_once_and_never_reactivates(self) -> None:
        gate = CaptureDirectionGateLifecycle(initially_enabled=True)
        self.assertTrue(gate.active)
        gate.retire_for_global_fallback(
            active_time_s=12.5,
            reason="posterior_abandoned",
        )
        self.assertFalse(gate.active)
        self.assertTrue(gate.retired_for_global_fallback)
        self.assertEqual(gate.retired_active_time_s, 12.5)
        self.assertEqual(gate.retirement_reason, "posterior_abandoned")

        # An idempotent call cannot reactivate or move the original transition.
        gate.retire_for_global_fallback(
            active_time_s=99.0,
            reason="posterior_abandoned",
        )
        self.assertFalse(gate.active)
        self.assertEqual(gate.retired_active_time_s, 12.5)
        with self.assertRaises(RuntimeError):
            gate.retire_for_global_fallback(
                active_time_s=12.5,
                reason="different_reason",
            )

    def test_disabled_gate_is_unchanged_by_fallback_retirement(self) -> None:
        gate = CaptureDirectionGateLifecycle(initially_enabled=False)
        gate.retire_for_global_fallback(
            active_time_s=2.0,
            reason="posterior_abandoned",
        )
        self.assertFalse(gate.active)
        self.assertFalse(gate.retired_for_global_fallback)
        self.assertIsNone(gate.retired_active_time_s)
        self.assertIsNone(gate.retirement_reason)

    def test_run_result_defaults_preserve_nonfallback_semantics(self) -> None:
        result = RunResult(
            method="baseline",
            result="search_radius_exhausted",
            inner_wall_confirmed=False,
            time_to_inner_wall_s=None,
            candidate_attempts=0,
            total_active_time_s=1.0,
            actual_xy_path_mm=0.0,
            contact_xy_path_mm=0.0,
            lateral_work_mj=0.0,
            orientation_work_mj=0.0,
            wear_proxy_mj=0.0,
            max_force_n=0.0,
            max_torque_mnm=0.0,
            stage_time_s={},
            event_xy_mm=None,
            baseline_depth_mm=0.0,
            min_tip_z_mm=0.0,
            hard_stop=False,
        )
        self.assertFalse(result.capture_direction_gate_active_at_outcome)
        self.assertFalse(
            result.capture_direction_gate_retired_for_global_fallback
        )
        self.assertIsNone(result.capture_direction_gate_retired_active_time_s)
        self.assertIsNone(result.capture_direction_gate_retirement_reason)

    def test_invalid_retirement_metadata_is_rejected(self) -> None:
        gate = CaptureDirectionGateLifecycle(initially_enabled=True)
        with self.assertRaises(ValueError):
            gate.retire_for_global_fallback(active_time_s=-1.0, reason="x")
        with self.assertRaises(ValueError):
            gate.retire_for_global_fallback(active_time_s=0.0, reason=" ")


if __name__ == "__main__":
    unittest.main()
