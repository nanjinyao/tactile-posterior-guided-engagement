from __future__ import annotations

import math
import unittest

from src.capture_yaw_assist import CaptureYawAssistConfig


class TestCaptureYawAssistConfig(unittest.TestCase):
    def test_default_is_disabled_and_commands_zero(self) -> None:
        config = CaptureYawAssistConfig()

        self.assertFalse(config.enabled)
        self.assertEqual(config.command_rad(0.5), 0.0)

    def test_enabled_command_is_ramped_and_bounded(self) -> None:
        config = CaptureYawAssistConfig(enabled=True)
        samples = [config.command_rad(index / 1000.0) for index in range(2001)]

        self.assertEqual(samples[0], 0.0)
        self.assertGreater(max(abs(value) for value in samples), 0.0)
        self.assertLessEqual(
            max(abs(value) for value in samples),
            math.radians(config.amplitude_deg) + 1.0e-12,
        )

    def test_abort_suppresses_command(self) -> None:
        config = CaptureYawAssistConfig(enabled=True)

        self.assertNotEqual(config.command_rad(0.5), 0.0)
        self.assertEqual(config.command_rad(0.5, aborted=True), 0.0)

    def test_tau_z_soft_limit_is_symmetric_and_fail_closed(self) -> None:
        config = CaptureYawAssistConfig(enabled=True, soft_tau_z_limit_mnm=10.0)

        self.assertFalse(config.exceeds_soft_tau_z(0.0099))
        self.assertTrue(config.exceeds_soft_tau_z(0.0100))
        self.assertTrue(config.exceeds_soft_tau_z(-0.0101))
        self.assertTrue(config.exceeds_soft_tau_z(float("nan")))

    def test_disabled_config_does_not_trip_finite_tau_z(self) -> None:
        config = CaptureYawAssistConfig(enabled=False)

        self.assertFalse(config.exceeds_soft_tau_z(1.0))
        self.assertFalse(config.exceeds_soft_tau_z(float("nan")))

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CaptureYawAssistConfig(amplitude_deg=3.1)
        with self.assertRaises(ValueError):
            CaptureYawAssistConfig(frequency_hz=0.0)
        with self.assertRaises(ValueError):
            CaptureYawAssistConfig(ramp_time_s=float("inf"))


if __name__ == "__main__":
    unittest.main()
