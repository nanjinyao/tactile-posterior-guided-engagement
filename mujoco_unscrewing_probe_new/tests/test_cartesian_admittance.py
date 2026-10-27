from __future__ import annotations

import unittest

import numpy as np

from src.cartesian_admittance import AdmittanceConfig, CartesianAdmittanceController, critical_damping


class TestCartesianAdmittanceController(unittest.TestCase):
    def test_missing_down_force_commands_negative_z_offset(self) -> None:
        controller = CartesianAdmittanceController()
        target = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        enabled = np.array([False, False, True, False, False, False])
        for _ in range(100):
            command = controller.step(np.zeros(6), target, enabled)
        self.assertLess(command.pose_offset[2], 0.0)
        self.assertTrue(np.allclose(command.pose_offset[[0, 1, 3, 4, 5]], 0.0))

    def test_target_force_reaches_bounded_steady_state(self) -> None:
        controller = CartesianAdmittanceController()
        target = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        enabled = np.array([False, False, True, False, False, False])
        measured = target.copy()
        for _ in range(500):
            command = controller.step(measured, target, enabled)
        self.assertLess(abs(command.pose_offset[2]), 1.0e-8)
        self.assertLess(abs(command.twist[2]), 1.0e-8)

    def test_pose_and_velocity_limits_are_enforced(self) -> None:
        config = AdmittanceConfig(
            max_velocity=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
            max_offset=(0.0002, 0.0002, 0.0002, 0.02, 0.02, 0.02),
        )
        controller = CartesianAdmittanceController(config)
        enabled = np.array([True, False, False, False, False, False])
        for _ in range(1000):
            command = controller.step(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), enabled_axes=enabled)
        self.assertLessEqual(abs(command.pose_offset[0]), 0.0002)
        self.assertLessEqual(abs(command.twist[0]), 0.001)

    def test_hard_wrench_latches_until_reset(self) -> None:
        controller = CartesianAdmittanceController()
        command = controller.step(np.array([21.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(command.hard_stop)
        self.assertTrue(np.allclose(command.twist, 0.0))
        command = controller.step(np.zeros(6))
        self.assertTrue(command.hard_stop)
        controller.reset()
        self.assertFalse(controller.hard_stop_latched)

    def test_raw_safety_observer_latches_between_control_updates(self) -> None:
        controller = CartesianAdmittanceController()
        self.assertTrue(
            controller.observe_wrench_safety(
                np.array([0.0, 0.0, 20.1, 0.0, 0.0, 0.0])
            )
        )
        self.assertTrue(controller.hard_stop_latched)

    def test_critical_damping(self) -> None:
        damping = critical_damping(np.ones(6), np.full(6, 4.0))
        self.assertTrue(np.allclose(damping, 4.0))


if __name__ == "__main__":
    unittest.main()
