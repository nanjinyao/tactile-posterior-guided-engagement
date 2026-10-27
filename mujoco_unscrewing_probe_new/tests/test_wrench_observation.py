from __future__ import annotations

import unittest
from pathlib import Path
import json

import mujoco
import numpy as np

from src.mujoco_admittance_rollout import MuJoCoAdmittanceRollout
from src.wrench_observation import WrenchObservationConfig, WrenchObservationModel


class TestWrenchObservationModel(unittest.TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        config = WrenchObservationConfig(seed=17)
        left = WrenchObservationModel(config)
        right = WrenchObservationModel(config)
        physical = np.array([1.0, -0.5, 3.0, 0.01, -0.02, 0.005])
        for _ in range(20):
            self.assertTrue(np.array_equal(left.observe(physical), right.observe(physical)))

    def test_delay_holds_earlier_physical_sample(self) -> None:
        config = WrenchObservationConfig(
            seed=1,
            sample_dt_s=0.001,
            delay_s=0.002,
            bias_std=(0.0,) * 6,
            white_std=(0.0,) * 6,
            drift_std_per_sqrt_s=(0.0,) * 6,
            cross_axis_std=0.0,
            gain_std=0.0,
        )
        model = WrenchObservationModel(config)
        zero = np.zeros(6)
        step = np.ones(6)
        self.assertTrue(np.allclose(model.observe(zero), zero))
        self.assertTrue(np.allclose(model.observe(step), zero))
        self.assertTrue(np.allclose(model.observe(step), zero))
        self.assertTrue(np.allclose(model.observe(step), step))

    def test_sample_counter_advances_once_per_observe(self) -> None:
        model = WrenchObservationModel(WrenchObservationConfig(seed=7))
        self.assertEqual(model.sample_count, 0)
        model.observe(np.zeros(6))
        self.assertEqual(model.sample_count, 1)
        _ = model.latest_observation
        self.assertEqual(model.sample_count, 1)
        model.observe(np.ones(6))
        self.assertEqual(model.sample_count, 2)

    def test_rollout_advances_sensor_once_per_physics_step_and_not_on_reset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = mujoco.MjModel.from_xml_path(
            str(root / "models" / "allen_m8_source.xml")
        )
        data = mujoco.MjData(model)
        spec = json.loads(
            (root / "configs" / "allen_m8_source.json").read_text(
                encoding="utf-8"
            )
        )
        config = WrenchObservationConfig(
            seed=11,
            delay_s=0.0,
            bias_std=(0.0,) * 6,
            white_std=(0.0,) * 6,
            drift_std_per_sqrt_s=(0.0,) * 6,
            cross_axis_std=0.0,
            gain_std=0.0,
        )
        rollout = MuJoCoAdmittanceRollout(
            model,
            data,
            float(spec["bit_length_m"]),
            float(spec["bit_mass_kg"]),
            controller_dt_s=float(model.opt.timestep),
            wrench_observation=config,
        )
        rollout.set_initial_tip_pose(np.array([0.0, 0.0, 0.002]))
        assert rollout.wrench_observation is not None
        self.assertEqual(rollout.wrench_observation.sample_count, 0)

        rollout.reset_admittance()
        self.assertEqual(rollout.wrench_observation.sample_count, 0)
        snapshot = rollout.step(np.array([0.0, 0.0, 0.002]), np.zeros(3))
        self.assertEqual(rollout.wrench_observation.sample_count, 1)
        self.assertTrue(
            np.allclose(snapshot.measured_wrench, snapshot.physical_wrench)
        )

        rollout.reset_admittance()
        self.assertEqual(rollout.wrench_observation.sample_count, 1)
        rollout.step(np.array([0.0, 0.0, 0.002]), np.zeros(3))
        self.assertEqual(rollout.wrench_observation.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
