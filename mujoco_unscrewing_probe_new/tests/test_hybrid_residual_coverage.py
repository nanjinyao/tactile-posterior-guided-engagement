from __future__ import annotations

import math
import unittest

import numpy as np

from src.render_m8_search_comparison import (
    HybridResidualCoverageConfig,
    archimedean_path,
    posterior_directional_corridor_path,
)
from src.run_cross_size_s4_full_loop_evaluation import (
    S4_HYBRID_RESIDUAL_COVERAGE,
    _residual_schedule_bound,
)


def _distance_to_polyline(point: np.ndarray, xy: np.ndarray) -> float:
    starts = xy[:-1]
    delta = xy[1:] - starts
    denominator = np.sum(delta * delta, axis=1)
    fraction = np.divide(
        np.sum((point[None, :] - starts) * delta, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projection = starts + fraction[:, None] * delta
    return float(np.min(np.linalg.norm(projection - point[None, :], axis=1)))


class TestHybridResidualCoverage(unittest.TestCase):
    def test_declared_sector_is_inside_capture_tube(self) -> None:
        capture_radius_m = 0.00037
        visual_bound_m = 0.008
        config = HybridResidualCoverageConfig(enabled=True)
        spacing_m = (
            2.0
            * capture_radius_m
            * config.lane_spacing_capture_fraction
        )
        arc, xy, offsets = posterior_directional_corridor_path(
            np.zeros(2),
            np.array([1.0, 0.0]),
            axial_start_m=0.0,
            axial_end_m=visual_bound_m,
            half_width_m=visual_bound_m
            * math.sin(math.radians(config.angular_half_width_deg)),
            lane_spacing_m=spacing_m,
        )
        self.assertGreater(float(arc[-1]), 0.0)
        self.assertTrue(np.allclose(xy[0], xy[-1]))
        self.assertLessEqual(
            float(np.max(np.diff(np.sort(offsets)))), spacing_m + 1.0e-12
        )
        for radius_m in np.linspace(0.0, visual_bound_m, 17):
            for angle_deg in np.linspace(
                -config.angular_half_width_deg,
                config.angular_half_width_deg,
                13,
            ):
                point = radius_m * np.array(
                    [
                        math.cos(math.radians(angle_deg)),
                        math.sin(math.radians(angle_deg)),
                    ]
                )
                self.assertLessEqual(
                    _distance_to_polyline(point, xy),
                    capture_radius_m + 1.0e-12,
                )

    def test_m12_global_spiral_cannot_fit_130_seconds_but_corridor_can(self) -> None:
        capture_radius_m = 0.00037
        visual_bound_m = 0.008
        pitch_m = 2.0 * capture_radius_m * 0.80
        current_radius_m = visual_bound_m - capture_radius_m
        global_arc, _ = archimedean_path(
            np.zeros(2), pitch_m, current_radius_m, 0.0
        )
        self.assertGreater(float(global_arc[-1]) / 0.00225, 130.0)

        config = HybridResidualCoverageConfig(
            enabled=True,
            lane_spacing_capture_fraction=1.0,
            tracking_error_allowance_m=0.00015,
        )
        robust_capture_radius_m = (
            capture_radius_m - config.tracking_error_allowance_m
        )
        corridor_arc, _, _ = posterior_directional_corridor_path(
            np.zeros(2),
            np.array([1.0, 0.0]),
            axial_start_m=0.0025,
            axial_end_m=visual_bound_m,
            half_width_m=visual_bound_m
            * math.sin(math.radians(config.angular_half_width_deg)),
            lane_spacing_m=(
                2.0
                * robust_capture_radius_m
                * config.lane_spacing_capture_fraction
            ),
        )
        shortcut_s = 0.40 + 0.0025 / 0.00225
        terminal_s = 29.00476190476191
        schedule_s = (
            shortcut_s
            + float(corridor_arc[-1]) / 0.00225
            + config.transition_overhead_s
            + terminal_s
        )
        self.assertLess(schedule_s, 130.0 - 5.7995)
        self.assertAlmostEqual(
            2.0 * robust_capture_radius_m, 0.00044, places=12
        )

    def test_config_is_disabled_by_default_and_rejects_unsafe_values(self) -> None:
        self.assertFalse(HybridResidualCoverageConfig().enabled)
        self.assertFalse(
            HybridResidualCoverageConfig().measured_tip_outer_servo_enabled
        )
        with self.assertRaises(ValueError):
            HybridResidualCoverageConfig(lane_spacing_capture_fraction=1.01)
        with self.assertRaises(ValueError):
            HybridResidualCoverageConfig(maximum_reverse_probability=0.51)

    def test_s4_freezes_declared_tracking_allowance_not_a_looser_pitch(self) -> None:
        config = S4_HYBRID_RESIDUAL_COVERAGE
        self.assertTrue(config.enabled)
        self.assertAlmostEqual(config.tracking_error_allowance_m, 0.00015)
        self.assertEqual(config.lane_spacing_capture_fraction, 1.0)
        self.assertTrue(config.measured_tip_outer_servo_enabled)
        self.assertLessEqual(config.outer_servo_max_correction_m, 0.0010)
        self.assertLessEqual(
            config.outer_servo_max_correction_speed_m_s, 0.0040
        )
        m12_capture_radius_m = 0.00037
        spacing_m = 2.0 * (
            m12_capture_radius_m - config.tracking_error_allowance_m
        )
        self.assertAlmostEqual(spacing_m, 0.00044)
        for size in ("m10", "m12"):
            bound = _residual_schedule_bound(size)
            self.assertTrue(bound["fits_130_s"])
            self.assertLessEqual(bound["total_required_s"], 130.0)
            self.assertFalse(bound["bound_uses_true_center"])


if __name__ == "__main__":
    unittest.main()
