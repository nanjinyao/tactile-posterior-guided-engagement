from __future__ import annotations

import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from src.mujoco_admittance_rollout import MuJoCoAdmittanceRollout
from src.render_m8_search_comparison import (
    _bounded_measured_pose_rebase_orientation,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SOURCE = ROOT / "src/render_m8_search_comparison.py"


class TestHeldCommandTelemetry(unittest.TestCase):
    def test_rebase_and_held_pose_read_advance_neither_physics_nor_sensor(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "models/allen_m8_source.xml")
        )
        data = mujoco.MjData(model)
        spec = json.loads(
            (ROOT / "configs/allen_m8_source.json").read_text(encoding="utf-8")
        )
        observation = WrenchObservationConfig(
            seed=501,
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
            wrench_observation=observation,
        )
        initial_tip = np.array([0.0, 0.0, 0.002], dtype=float)
        used_rpy = np.radians(np.array([0.20, -0.15, 0.35], dtype=float))
        rollout.set_initial_tip_pose(initial_tip)
        snapshot = rollout.step(initial_tip, used_rpy)
        assert rollout.wrench_observation is not None
        samples_before = rollout.wrench_observation.sample_count
        time_before = float(data.time)
        qpos_before = np.asarray(data.qpos, dtype=float).copy()

        held_tip_before, held_rpy_before = rollout.held_command_pose()
        self.assertEqual(rollout.wrench_observation.sample_count, samples_before)
        self.assertEqual(float(data.time), time_before)

        measured_rpy = np.array(
            [snapshot.roll, snapshot.pitch, snapshot.yaw], dtype=float
        )
        offset = np.zeros(6, dtype=float)
        offset[3:] = (measured_rpy - used_rpy + np.pi) % (2.0 * np.pi) - np.pi
        rollout.rebase_admittance_command(snapshot.tip_position, used_rpy, offset)
        held_tip_after, held_rpy_after = rollout.held_command_pose()

        self.assertEqual(rollout.wrench_observation.sample_count, samples_before)
        self.assertEqual(float(data.time), time_before)
        self.assertTrue(np.array_equal(np.asarray(data.qpos), qpos_before))
        self.assertTrue(np.all(np.isfinite(held_tip_before)))
        self.assertTrue(np.all(np.isfinite(held_rpy_before)))
        self.assertTrue(np.all(np.isfinite(held_tip_after)))
        self.assertTrue(np.all(np.isfinite(held_rpy_after)))
        # The rebase is anchored at the measured pose; a single non-advancing
        # outer update may add only its bounded sub-step admittance correction.
        self.assertLess(np.linalg.norm(held_tip_after - snapshot.tip_position), 5e-5)

    def test_out_of_bound_rpy_fallback_has_a_continuous_first_physics_step(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "models/allen_m8_source.xml")
        )
        data = mujoco.MjData(model)
        spec = json.loads(
            (ROOT / "configs/allen_m8_source.json").read_text(encoding="utf-8")
        )
        rollout = MuJoCoAdmittanceRollout(
            model,
            data,
            float(spec["bit_length_m"]),
            float(spec["bit_mass_kg"]),
            controller_dt_s=float(model.opt.timestep),
        )
        tip = np.array([0.0, 0.0, 0.002], dtype=float)
        measured_rpy = np.array([0.20, -0.16, 0.14], dtype=float)
        exact_pre_rebase_command = np.zeros(3, dtype=float)
        rollout.set_initial_tip_pose(tip, measured_rpy)
        limits = np.asarray(rollout.controller.config.max_offset, dtype=float)[3:]
        nominal_rpy, rotational_offset, fallback = (
            _bounded_measured_pose_rebase_orientation(
                measured_rpy,
                exact_pre_rebase_command,
                limits,
            )
        )
        self.assertTrue(fallback)
        self.assertTrue(np.array_equal(nominal_rpy, measured_rpy))
        self.assertTrue(np.array_equal(rotational_offset, np.zeros(3)))

        pose_offset = np.zeros(6, dtype=float)
        pose_offset[3:] = rotational_offset
        rollout.rebase_admittance_command(tip, nominal_rpy, pose_offset)
        _, held_rpy_before = rollout.held_command_pose()
        snapshot = rollout.step(tip, nominal_rpy)
        _, held_rpy_after = rollout.held_command_pose()
        wrapped_delta = (
            held_rpy_after - held_rpy_before + np.pi
        ) % (2.0 * np.pi) - np.pi
        self.assertLess(np.linalg.norm(wrapped_delta), 1.0e-10)
        self.assertFalse(snapshot.command.hard_stop)


class TestTransitionMechanismContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ROLLOUT_SOURCE.read_text(encoding="utf-8")

    def test_terminal_entry_uses_exact_just_used_rpy_atomic_rebase(self) -> None:
        self.assertIn("desired_descent_rpy = (", self.source)
        self.assertIn("terminal_rpy_command = smooth_rpy_transition(", self.source)
        self.assertIn(
            'terminal_rpy_command,\n                                "cad_terminal_set_entry"',
            self.source,
        )
        self.assertIn('"exact_just_used_rpy_command_deg"', self.source)

    def test_recovery_forces_a_new_descent_pass_from_measured_pose(self) -> None:
        self.assertIn("descent_restart_after_recovery = True", self.source)
        self.assertIn('"stale_pass_fraction_reused": False', self.source)
        self.assertIn(
            '"funnel recovery complete | restart descent from measured pose"',
            self.source,
        )
        self.assertIn('"funnel_recovery_resume"', self.source)

    def test_oracle_endpoint_closes_residual_smoothly_before_dwell(self) -> None:
        self.assertIn('"oracle_endpoint_settle_entry"', self.source)
        self.assertIn("approach_delta_xy = planned_target_xy - approach_start_tip[:2]", self.source)
        self.assertIn('"oracle_center_settle_approach"', self.source)
        self.assertLess(
            self.source.index('"oracle_center_settle_approach"'),
            self.source.index('"oracle_center_settle"'),
        )

    def test_post_release_gate_is_sensor_only_and_has_named_safe_abort(self) -> None:
        start = self.source.index("    def post_release_requalification_condition")
        end = self.source.index("    def sensor_depth_commit_condition", start)
        gate_source = self.source[start:end]
        self.assertIn("candidate.measured_wrench", gate_source)
        self.assertIn("candidate.tip_position", gate_source)
        self.assertNotIn("planned_target_xy", gate_source)
        self.assertNotIn("fast_contact_tags", gate_source)
        self.assertNotIn("event_inside_capture", gate_source)
        self.assertIn(
            '"post_release_capture_requalification_failed"',
            self.source,
        )
        self.assertIn('"true_center_used": False', self.source)
        self.assertIn('"offline_contact_tags_used": False', self.source)

    def test_preexisting_terminal_abort_is_not_relabelled_as_requalification_failure(self) -> None:
        self.assertIn("termination_before_requalification = bool(", self.source)
        self.assertIn(
            '"skipped_due_to_existing_terminal_outcome": (',
            self.source,
        )
        self.assertIn(
            '"preexisting_terminal_reason": (',
            self.source,
        )
        self.assertIn(
            "post_release_requalified\n                            or termination_before_requalification",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
