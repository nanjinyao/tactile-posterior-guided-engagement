from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from src.posterior_search_plan import plan_posterior_guided_search
from src.render_m8_search_comparison import RunResult, ShortcutPlanTelemetry


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOBOL_016 = (
    ROOT
    / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
    / "trials/sobol_016.json"
)


class TestShortcutTelemetrySemantics(unittest.TestCase):
    @staticmethod
    def _plan(center_x_m: float):
        samples = np.asarray(
            [
                [center_x_m - 0.0002, -0.0001],
                [center_x_m - 0.0001, 0.0001],
                [center_x_m, 0.0],
                [center_x_m + 0.0001, -0.0001],
                [center_x_m + 0.0002, 0.0001],
            ],
            dtype=float,
        )
        return plan_posterior_guided_search(
            np.zeros(2, dtype=float),
            samples,
            np.asarray([1.0, 0.0]),
            effective_capture_radius_m=0.00025,
            posterior_coverage=0.99,
        )

    def test_first_shortcut_is_frozen_when_latest_plan_changes(self) -> None:
        first = self._plan(0.0030)
        replan = self._plan(0.0015)
        telemetry = ShortcutPlanTelemetry()

        telemetry.record(first, first=True)
        telemetry.mark_latest_executed()
        frozen_distance = telemetry.first_shortcut_distance_mm
        frozen_target = telemetry.first_shortcut_target_xy_mm

        telemetry.record(replan)

        self.assertEqual(telemetry.latest_plan_update_index, 1)
        self.assertTrue(telemetry.first_shortcut_executed)
        self.assertFalse(telemetry.latest_plan_executed)
        self.assertEqual(telemetry.first_shortcut_distance_mm, frozen_distance)
        self.assertEqual(telemetry.first_shortcut_target_xy_mm, frozen_target)
        self.assertAlmostEqual(
            telemetry.latest_planned_distance_mm,
            1000.0 * replan.transport_distance_m,
        )
        self.assertNotEqual(
            telemetry.latest_planned_distance_mm,
            telemetry.first_shortcut_distance_mm,
        )

        telemetry.mark_latest_executed()
        self.assertTrue(telemetry.latest_plan_executed)

    def test_old_run_result_payload_remains_backward_compatible(self) -> None:
        payload = json.loads(FROZEN_SOBOL_016.read_text(encoding="utf-8"))
        old_proposed = RunResult(**payload["methods"]["proposed"])

        # Historical artifacts are evidence and are intentionally not rewritten.
        # Missing semantic fields therefore deserialize to None, while every
        # pre-existing transport field retains its original value.
        self.assertIsNone(old_proposed.first_shortcut_distance_mm)
        self.assertIsNone(old_proposed.latest_planned_distance_mm)
        self.assertAlmostEqual(old_proposed.transport_distance_mm, 0.9032664222572051)

    def test_frozen_sobol_016_documents_first_vs_latest_meaning(self) -> None:
        payload = json.loads(FROZEN_SOBOL_016.read_text(encoding="utf-8"))
        proposed = payload["methods"]["proposed"]
        proposed_plus = payload["methods"]["proposed_plus"]
        records = proposed_plus["hybrid_probe_gate_records"]

        first_shortcut = records[0]
        fourth_update = next(item for item in records if item.get("probe") == 4)
        fifth_probe = next(item for item in records if item.get("probe") == 5)

        self.assertEqual(first_shortcut["stage"], "posterior_shortcut_before_entropy_grid")
        self.assertAlmostEqual(
            first_shortcut["transport_distance_mm"],
            proposed["transport_distance_mm"],
        )
        self.assertAlmostEqual(first_shortcut["transport_distance_mm"], 0.9032664222572051)
        self.assertAlmostEqual(
            fourth_update["replanned_transport_mm"],
            proposed_plus["transport_distance_mm"],
        )
        self.assertAlmostEqual(proposed_plus["transport_distance_mm"], 0.5424859387900729)
        self.assertEqual(fifth_probe["outcome"], "capture_during_smooth_segment")


if __name__ == "__main__":
    unittest.main()
