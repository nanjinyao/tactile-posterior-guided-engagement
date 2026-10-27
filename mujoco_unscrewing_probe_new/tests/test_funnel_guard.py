from __future__ import annotations

import unittest

import numpy as np

from src.funnel_guard import FunnelGuard, FunnelGuardConfig


class TestFunnelGuard(unittest.TestCase):
    def test_progress_with_low_energy_does_not_recover(self) -> None:
        guard = FunnelGuard()
        observation = None
        for index in range(400):
            observation = guard.observe(
                np.array([0.0, 0.0, -2.0e-7 * index]),
                np.zeros(3),
                np.array([0.1, 0.0, 1.0, 0.0, 0.0, 0.0]),
                0.0005,
            )
        assert observation is not None
        self.assertGreater(observation.window_progress_m, 0.0)
        self.assertFalse(observation.recovery_requested)

    def test_energy_without_progress_requests_recovery(self) -> None:
        guard = FunnelGuard(
            FunnelGuardConfig(
                no_progress_energy_j=2.0e-5,
                injected_power_limit_w=10.0,
            )
        )
        requested = False
        # Move laterally against a resisting environment wrench while Z stalls.
        for index in range(600):
            observation = guard.observe(
                np.array([2.0e-6 * index, 0.0, 0.0]),
                np.zeros(3),
                np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                0.0005,
            )
            requested = requested or observation.recovery_requested
        self.assertTrue(requested)
        self.assertEqual(guard.last.reason, None)
        self.assertGreater(guard.injected_energy_j, 0.0)

    def test_tilt_limit_is_dwell_guarded(self) -> None:
        guard = FunnelGuard()
        requested = False
        for index in range(60):
            observation = guard.observe(
                np.zeros(3),
                np.array([np.deg2rad(1.0), 0.0, 0.0]),
                np.zeros(6),
                0.0005,
                allow_stall=True,
            )
            requested = requested or observation.recovery_requested
        self.assertTrue(requested)

    def test_configuration_rejects_nonphysical_values(self) -> None:
        with self.assertRaises(ValueError):
            FunnelGuardConfig(window_s=0.0)
        with self.assertRaises(ValueError):
            FunnelGuardConfig(max_recoveries=-1)

    def test_cad_depth_overshoot_requests_recovery(self) -> None:
        guard = FunnelGuard()
        observation = guard.observe(
            np.array([0.0, 0.0, -0.0043]),
            np.zeros(3),
            np.zeros(6),
            0.0005,
            minimum_tip_z_m=-0.00425,
        )
        self.assertTrue(observation.recovery_requested)
        self.assertEqual(observation.reason, "cad_depth_overshoot")


if __name__ == "__main__":
    unittest.main()
