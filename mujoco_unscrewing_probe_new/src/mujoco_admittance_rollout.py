from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import mujoco
import numpy as np

from src.cartesian_admittance import AdmittanceConfig, AdmittanceCommand, CartesianAdmittanceController
from src.wrench_observation import WrenchObservationConfig, WrenchObservationModel


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=float,
    )


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    matrix = quat_to_matrix(quat)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


@dataclass(frozen=True)
class DynamicSnapshot:
    """One post-physics state and its single co-timed wrench acquisition.

    ``measured_wrench`` is the delayed/noisy observation used by the outer
    controller at its next scheduled update. ``physical_wrench`` is the
    undelayed MuJoCo contact wrench used only by the fast hard-stop layer.
    """

    tip_position: np.ndarray
    roll: float
    pitch: float
    yaw: float
    measured_wrench: np.ndarray
    physical_wrench: np.ndarray
    command: AdmittanceCommand
    nominal_tip_position: np.ndarray
    state_invalid: bool
    failure_reason: str
    hard_stop_reason: str = ""


class MuJoCoAdmittanceRollout:
    """Admittance outer loop with a wrench-driven MuJoCo free-body inner loop."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bit_length_m: float,
        bit_mass_kg: float,
        *,
        wrench_reference_to_tip_m: float | None = None,
        controller_dt_s: float = 0.002,
        target_down_force_n: float = 3.0,
        contact_timeconst_s: float = 0.05,
        effective_mass_kg: float = 5.0,
        effective_inertia_kg_m2: tuple[float, float, float] = (0.020, 0.020, 0.010),
        wrench_observation: WrenchObservationConfig | None = None,
        admittance_config: AdmittanceConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.bit_length_m = float(bit_length_m)
        self.bit_mass_kg = float(bit_mass_kg)
        if wrench_reference_to_tip_m is None:
            # Retain the frozen simulation evidence until a caller explicitly
            # opts into a calibrated hardware transform.  Hardware-aligned
            # runs must never rely on this legacy fraction.
            self.wrench_reference_to_tip_m = 0.18 * self.bit_length_m
            self.wrench_reference_mode = "legacy_0.18_bit_length"
        else:
            value = float(wrench_reference_to_tip_m)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("wrench_reference_to_tip_m must be finite and positive")
            self.wrench_reference_to_tip_m = value
            self.wrench_reference_mode = "explicit_calibrated_distance"
        tip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "bit_tip"
        )
        if tip_site_id < 0:
            raise ValueError("MuJoCo model must define a bit_tip site")
        self.body_id = int(model.site_bodyid[tip_site_id])
        if self.body_id <= 0:
            raise ValueError("bit_tip must belong to a dynamic bit body")
        self.effective_mass_kg = float(effective_mass_kg)
        self.model.body_mass[self.body_id] = self.effective_mass_kg
        self.model.body_inertia[self.body_id] = np.asarray(effective_inertia_kg_m2, dtype=float)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.geom_solref[:, 0] = float(contact_timeconst_s)
        self.controller_dt_s = float(controller_dt_s)
        self.control_stride = max(1, int(round(self.controller_dt_s / self.model.opt.timestep)))
        self.target_wrench = np.array([0.0, 0.0, target_down_force_n, 0.0, 0.0, 0.0], dtype=float)
        controller_config = (
            AdmittanceConfig(dt_s=self.controller_dt_s)
            if admittance_config is None
            else replace(admittance_config, dt_s=self.controller_dt_s)
        )
        self.controller = CartesianAdmittanceController(controller_config)
        self.enabled_axes = np.array([False, False, True, False, False, False])
        self._nominal_position_kp = np.array([12000.0, 12000.0, 6000.0])
        self._nominal_position_kd = np.array([420.0, 420.0, 340.0])
        self._nominal_rotation_kp = np.array([3.0, 3.0, 1.5])
        self._nominal_rotation_kd = np.array([0.50, 0.50, 0.25])
        self.position_kp = self._nominal_position_kp.copy()
        self.position_kd = self._nominal_position_kd.copy()
        # The inner loop represents a stiff robot flange, while contact
        # compliance is introduced explicitly by the outer admittance loop.
        self.rotation_kp = self._nominal_rotation_kp.copy()
        self.rotation_kd = self._nominal_rotation_kd.copy()
        self.tip_impedance_mode = False
        self._step_count = 0
        self._command = self.controller.step(np.zeros(6), self.target_wrench, self.enabled_axes)
        self._nominal_tip = np.zeros(3, dtype=float)
        self._nominal_rpy = np.zeros(3, dtype=float)
        self._desired_body_pos = np.zeros(3, dtype=float)
        self._desired_tip_pos = np.zeros(3, dtype=float)
        self._desired_quat = euler_to_quat(0.0, 0.0, 0.0)
        self.failed = False
        self.wrench_observation_config = wrench_observation
        self.wrench_observation = (
            None
            if wrench_observation is None
            else WrenchObservationModel(wrench_observation)
        )
        self._latest_physical_wrench = np.zeros(6, dtype=float)
        self._latest_measured_wrench = np.zeros(6, dtype=float)
        self._hard_stop_reason = ""

    def _wrench_limit_reason(self, prefix: str, wrench: np.ndarray) -> str:
        limits = np.asarray(self.controller.config.hard_wrench, dtype=float)
        exceeded = np.flatnonzero(np.abs(np.asarray(wrench, dtype=float)) > limits)
        axes = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
        suffix = ",".join(axes[int(index)] for index in exceeded)
        return f"{prefix}:{suffix or 'latched'}"

    def observe_wrench(self, physical_wrench: np.ndarray) -> np.ndarray:
        if self.wrench_observation is None:
            return np.asarray(physical_wrench, dtype=float).copy()
        return self.wrench_observation.observe(physical_wrench)

    def set_enabled_axes(self, enabled_axes: np.ndarray) -> None:
        enabled = np.asarray(enabled_axes, dtype=bool)
        if enabled.shape != (6,):
            raise ValueError("enabled_axes must have shape (6,)")
        self.enabled_axes = enabled.copy()

    def set_target_down_force(self, force_n: float) -> None:
        if force_n <= 0.0:
            raise ValueError("force_n must be positive")
        self.target_wrench[2] = float(force_n)

    def set_planar_impedance(self, stiffness_n_m: float, damping_n_s_m: float) -> None:
        if not np.isfinite(stiffness_n_m) or stiffness_n_m <= 0.0:
            raise ValueError("stiffness_n_m must be positive")
        if not np.isfinite(damping_n_s_m) or damping_n_s_m <= 0.0:
            raise ValueError("damping_n_s_m must be positive")
        self.position_kp[:2] = float(stiffness_n_m)
        self.position_kd[:2] = float(damping_n_s_m)

    def restore_planar_impedance(self) -> None:
        self.position_kp[:2] = self._nominal_position_kp[:2]
        self.position_kd[:2] = self._nominal_position_kd[:2]

    def nominal_planar_impedance(self) -> tuple[float, float]:
        """Return the configured planar inner-loop gains for bumpless ramps."""

        return (
            float(self._nominal_position_kp[0]),
            float(self._nominal_position_kd[0]),
        )

    def set_rotational_impedance(
        self,
        stiffness_nm_rad: tuple[float, float, float] | np.ndarray,
        damping_nm_s_rad: tuple[float, float, float] | np.ndarray,
    ) -> None:
        stiffness = np.asarray(stiffness_nm_rad, dtype=float)
        damping = np.asarray(damping_nm_s_rad, dtype=float)
        if stiffness.shape != (3,) or damping.shape != (3,):
            raise ValueError("rotational stiffness/damping must have shape (3,)")
        if not np.all(np.isfinite(stiffness)) or np.any(stiffness <= 0.0):
            raise ValueError("rotational stiffness must be finite and positive")
        if not np.all(np.isfinite(damping)) or np.any(damping <= 0.0):
            raise ValueError("rotational damping must be finite and positive")
        self.rotation_kp = stiffness.copy()
        self.rotation_kd = damping.copy()

    def restore_rotational_impedance(self) -> None:
        self.rotation_kp = self._nominal_rotation_kp.copy()
        self.rotation_kd = self._nominal_rotation_kd.copy()

    def nominal_rotational_impedance(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of nominal rotational gains for bumpless ramps."""

        return (
            self._nominal_rotation_kp.copy(),
            self._nominal_rotation_kd.copy(),
        )

    def set_tip_impedance_mode(self, enabled: bool) -> None:
        """Place the Cartesian compliance reference at the peg tip.

        This is a MuJoCo proxy for commanding KUKA iiwa Cartesian impedance in
        a calibrated tool frame. It is enabled only after capture; the conical
        fingerprint and planar search retain their existing controller.
        """

        self.tip_impedance_mode = bool(enabled)

    def reset_admittance(self, pose_offset: np.ndarray | None = None) -> None:
        """Reset controller state without advancing sensor/physics time."""

        self.controller.reset(pose_offset)
        self._hard_stop_reason = ""
        # The raw channel is an explicit, undelayed safety path.  Reapplying
        # the cached sample prevents reset from clearing a still-present stop,
        # while neither observation delay nor stochastic state advances.
        if self.controller.observe_wrench_safety(self._latest_physical_wrench):
            self._hard_stop_reason = self._wrench_limit_reason(
                "raw_physical_wrench_limit", self._latest_physical_wrench
            )
        raw_latched = self.controller.hard_stop_latched
        self._command = self.controller.step(
            self._latest_measured_wrench,
            self.target_wrench,
            self.enabled_axes,
        )
        if self._command.hard_stop and not raw_latched:
            self._hard_stop_reason = self._wrench_limit_reason(
                "measured_wrench_limit", self._latest_measured_wrench
            )

    def rebase_admittance_command(
        self,
        nominal_tip_position: np.ndarray,
        nominal_rpy: np.ndarray,
        pose_offset: np.ndarray | None = None,
    ) -> None:
        """Reset admittance and its held-command history without a rate impulse.

        ``step`` differentiates the corrected Cartesian command to construct
        inner-loop feed-forward velocity.  Resetting only the outer admittance
        while changing the nominal pose leaves the previous corrected command
        in that differentiator and can therefore create a one-step damping
        impulse.  This operation atomically rebases both states.  It advances
        neither MuJoCo nor the observation model and retains the same raw-wrench
        latching performed by :meth:`reset_admittance`.
        """

        nominal_tip = np.asarray(nominal_tip_position, dtype=float)
        nominal_angles = np.asarray(nominal_rpy, dtype=float)
        if nominal_tip.shape != (3,) or not np.all(np.isfinite(nominal_tip)):
            raise ValueError("nominal_tip_position must contain three finite values")
        if nominal_angles.shape != (3,) or not np.all(np.isfinite(nominal_angles)):
            raise ValueError("nominal_rpy must contain three finite values")
        self.reset_admittance(pose_offset)
        corrected_tip = nominal_tip + self._command.pose_offset[:3]
        corrected_angles = nominal_angles + self._command.pose_offset[3:]
        desired_quat = euler_to_quat(*corrected_angles)
        desired_rotation = quat_to_matrix(desired_quat)
        local_tip = np.array([0.0, 0.0, -self.bit_length_m / 2.0])
        self._desired_tip_pos = corrected_tip.copy()
        self._desired_quat = desired_quat.copy()
        self._desired_body_pos = corrected_tip - desired_rotation @ local_tip
        self._nominal_tip = nominal_tip.copy()
        self._nominal_rpy = nominal_angles.copy()

    def held_command_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the Cartesian pose held by the inner-loop differentiator.

        This is controller telemetry, not a new observation.  Reading it does
        not advance MuJoCo, the wrench model, or the admittance state.  It is
        useful at hybrid-controller transitions because continuity must be
        assessed against the command that was actually held during the last
        physics interval, rather than against a possibly stale nominal pose.
        """

        return (
            self._desired_tip_pos.copy(),
            np.asarray(quat_to_euler(self._desired_quat), dtype=float),
        )

    def set_initial_tip_pose(self, tip_position: np.ndarray, rpy: np.ndarray | None = None) -> None:
        tip = np.asarray(tip_position, dtype=float)
        rpy_value = np.zeros(3, dtype=float) if rpy is None else np.asarray(rpy, dtype=float)
        if tip.shape != (3,) or rpy_value.shape != (3,):
            raise ValueError("tip_position and rpy must have shape (3,)")
        quat = euler_to_quat(*rpy_value)
        rotation = quat_to_matrix(quat)
        local_tip = np.array([0.0, 0.0, -self.bit_length_m / 2.0])
        self.data.qpos[:3] = tip - rotation @ local_tip
        self.data.qpos[3:7] = quat
        self.data.qvel.fill(0.0)
        self.data.qfrc_applied.fill(0.0)
        self.controller.reset()
        self._hard_stop_reason = ""
        self._step_count = 0
        self._nominal_tip = tip.copy()
        self._nominal_rpy = rpy_value.copy()
        self._desired_body_pos = np.asarray(self.data.qpos[:3], dtype=float).copy()
        self._desired_tip_pos = tip.copy()
        self._desired_quat = quat.copy()
        self.failed = False
        mujoco.mj_forward(self.model, self.data)
        self._latest_physical_wrench = self.measure_wrench()
        # Initialisation has no elapsed sample period.  With no observation
        # model the physical wrench is already the sensor value; with a
        # delayed model retain its non-advancing initial observation.
        self._latest_measured_wrench = (
            self._latest_physical_wrench.copy()
            if self.wrench_observation is None
            else self.wrench_observation.latest_observation
        )
        self._command = self.controller.step(
            self._latest_measured_wrench,
            self.target_wrench,
            self.enabled_axes,
        )
        if self._command.hard_stop:
            self._hard_stop_reason = self._wrench_limit_reason(
                "measured_wrench_limit", self._latest_measured_wrench
            )

    def measure_wrench(self) -> np.ndarray:
        force_world = np.zeros(3, dtype=float)
        torque_world = np.zeros(3, dtype=float)
        quat = np.asarray(self.data.qpos[3:7], dtype=float)
        rotation = quat_to_matrix(quat)
        body_position = np.asarray(self.data.qpos[:3], dtype=float)
        tip = body_position + rotation @ np.array([0.0, 0.0, -self.bit_length_m / 2.0])
        sensor = tip + rotation @ np.array([0.0, 0.0, self.wrench_reference_to_tip_m])
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
            if "bit_collision" not in (geom1, geom2):
                continue
            other = geom2 if geom1 == "bit_collision" else geom1
            if other == "workpiece_base":
                continue
            local = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, index, local)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force = frame.T @ local[:3]
            torque = frame.T @ local[3:]
            if geom1 == "bit_collision":
                force = -force
                torque = -torque
            point = np.asarray(contact.pos, dtype=float)
            force_world += force
            torque_world += torque + np.cross(point - sensor, force)
        return np.concatenate((force_world, torque_world))

    def wrench_reference_world(self) -> np.ndarray:
        """Return the current wrench-reference point without sensing contacts.

        This is telemetry for frame calibration and tests; it does not advance
        physics or the observation model.  The vector is measured from the peg
        tip toward the sensor along the local +Z tool axis.
        """

        tip, _ = self.actual_tip_pose()
        rotation = quat_to_matrix(np.asarray(self.data.qpos[3:7], dtype=float))
        return tip + rotation @ np.array(
            [0.0, 0.0, self.wrench_reference_to_tip_m], dtype=float
        )

    def actual_tip_pose(self) -> tuple[np.ndarray, tuple[float, float, float]]:
        quat = np.asarray(self.data.qpos[3:7], dtype=float)
        rotation = quat_to_matrix(quat)
        local_tip = np.array([0.0, 0.0, -self.bit_length_m / 2.0])
        tip = np.asarray(self.data.qpos[:3], dtype=float) + rotation @ local_tip
        return tip, quat_to_euler(quat)

    def step(self, nominal_tip_position: np.ndarray, nominal_rpy: np.ndarray) -> DynamicSnapshot:
        nominal_tip = np.asarray(nominal_tip_position, dtype=float)
        nominal_angles = np.asarray(nominal_rpy, dtype=float)

        corrected_tip = nominal_tip + self._command.pose_offset[:3]
        corrected_angles = nominal_angles + self._command.pose_offset[3:]
        desired_quat = euler_to_quat(*corrected_angles)
        desired_rotation = quat_to_matrix(desired_quat)
        local_tip = np.array([0.0, 0.0, -self.bit_length_m / 2.0])
        desired_body_pos = corrected_tip - desired_rotation @ local_tip
        dt = float(self.model.opt.timestep)
        desired_linear_velocity = np.clip(
            (desired_body_pos - self._desired_body_pos) / dt,
            -0.15,
            0.15,
        )
        desired_angular_velocity = np.zeros(3, dtype=float)
        mujoco.mju_subQuat(desired_angular_velocity, desired_quat, self._desired_quat)
        desired_angular_velocity = np.clip(desired_angular_velocity / dt, -3.0, 3.0)

        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            velocity,
            0,
        )
        angular_velocity = velocity[:3]
        linear_velocity = velocity[3:]
        actual_body_pos = np.asarray(self.data.qpos[:3], dtype=float)
        actual_tip, _ = self.actual_tip_pose()
        tip_offset_world = actual_tip - actual_body_pos
        actual_tip_velocity = linear_velocity + np.cross(angular_velocity, tip_offset_world)
        desired_tip_velocity = np.clip(
            (corrected_tip - self._desired_tip_pos) / dt,
            -0.15,
            0.15,
        )
        if self.tip_impedance_mode:
            force_cmd = (
                self.position_kp * (corrected_tip - actual_tip)
                + self.position_kd * (desired_tip_velocity - actual_tip_velocity)
            )
        else:
            force_cmd = (
                self.position_kp * (desired_body_pos - actual_body_pos)
                + self.position_kd * (desired_linear_velocity - linear_velocity)
            )
        quat_error = np.zeros(3, dtype=float)
        mujoco.mju_subQuat(quat_error, desired_quat, np.asarray(self.data.qpos[3:7]))
        torque_cmd = (
            self.rotation_kp * quat_error
            + self.rotation_kd * (desired_angular_velocity - angular_velocity)
        )

        self.data.qfrc_applied.fill(0.0)
        control_point = actual_tip if self.tip_impedance_mode else actual_body_pos
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force_cmd,
            torque_cmd,
            control_point,
            self.body_id,
            self.data.qfrc_applied,
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            np.array([0.0, 0.0, self.effective_mass_kg * 9.81]),
            np.zeros(3),
            actual_body_pos,
            self.body_id,
            self.data.qfrc_applied,
        )
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._desired_body_pos = desired_body_pos.copy()
        self._desired_tip_pos = corrected_tip.copy()
        self._desired_quat = desired_quat.copy()
        self._nominal_tip = nominal_tip.copy()
        self._nominal_rpy = nominal_angles.copy()
        tip, rpy = self.actual_tip_pose()
        reasons: list[str] = []
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            reasons.append("nonfinite_state")
        if np.linalg.norm(tip - nominal_tip) > 0.010:
            reasons.append("tip_tracking_error")
        # Contact impulses can briefly create large generalized velocity at the
        # body center while the controlled tip remains bounded. Treat pose
        # divergence/non-finite state as rollout failure; report velocity peaks
        # as metrics instead of aborting on a single integration sample.
        state_invalid = bool(reasons)
        self.failed = self.failed or state_invalid
        # Sense once after each physics transition.  This is the only call
        # that advances observation delay/noise/drift state.  The outer-loop
        # command computed here is held for the following physics interval,
        # which gives a conventional act -> physics -> sense/update ordering.
        physical_wrench = self.measure_wrench()
        post_step_wrench = self.observe_wrench(physical_wrench)
        self._latest_physical_wrench = physical_wrench.copy()
        self._latest_measured_wrench = post_step_wrench.copy()

        # Fast safety deliberately uses the co-timed undelayed physical wrench
        # so an observation delay cannot hide an impulse.  Downstream policy
        # and the slower admittance update use only ``post_step_wrench``.
        hard_stop_before_raw = self.controller.hard_stop_latched
        raw_hard_stop = self.controller.observe_wrench_safety(physical_wrench)
        if raw_hard_stop and not hard_stop_before_raw:
            self._hard_stop_reason = self._wrench_limit_reason(
                "raw_physical_wrench_limit", physical_wrench
            )
        control_update_due = (self._step_count - 1) % self.control_stride == 0
        if control_update_due:
            hard_stop_before_control = self.controller.hard_stop_latched
            self._command = self.controller.step(
                post_step_wrench,
                self.target_wrench,
                self.enabled_axes,
            )
            if self._command.hard_stop and not hard_stop_before_control:
                self._hard_stop_reason = self._wrench_limit_reason(
                    "measured_wrench_limit", post_step_wrench
                )
        elif raw_hard_stop:
            self._command = replace(self._command, hard_stop=True)
        return DynamicSnapshot(
            tip_position=tip,
            roll=rpy[0],
            pitch=rpy[1],
            yaw=rpy[2],
            measured_wrench=post_step_wrench,
            physical_wrench=physical_wrench,
            command=self._command,
            nominal_tip_position=nominal_tip.copy(),
            state_invalid=state_invalid,
            failure_reason=",".join(reasons),
            hard_stop_reason=self._hard_stop_reason,
        )

    def advance(
        self,
        nominal_tip_position: np.ndarray,
        nominal_rpy: np.ndarray,
        duration_s: float,
    ) -> DynamicSnapshot:
        steps = max(1, int(round(float(duration_s) / self.model.opt.timestep)))
        snapshot: DynamicSnapshot | None = None
        for _ in range(steps):
            snapshot = self.step(nominal_tip_position, nominal_rpy)
            if snapshot.command.hard_stop or snapshot.state_invalid:
                break
        assert snapshot is not None
        return snapshot

    def advance_interpolated(
        self,
        nominal_tip_position: np.ndarray,
        nominal_rpy: np.ndarray,
        duration_s: float,
    ) -> DynamicSnapshot:
        target_tip = np.asarray(nominal_tip_position, dtype=float)
        target_rpy = np.asarray(nominal_rpy, dtype=float)
        start_tip = self._nominal_tip.copy()
        start_rpy = self._nominal_rpy.copy()
        steps = max(1, int(round(float(duration_s) / self.model.opt.timestep)))
        snapshot: DynamicSnapshot | None = None
        for index in range(steps):
            fraction = (index + 1) / steps
            tip = start_tip + fraction * (target_tip - start_tip)
            rpy = start_rpy + fraction * (target_rpy - start_rpy)
            snapshot = self.step(tip, rpy)
            if snapshot.command.hard_stop or snapshot.state_invalid:
                break
        assert snapshot is not None
        return snapshot

    def advance_interpolated_until(
        self,
        nominal_tip_position: np.ndarray,
        nominal_rpy: np.ndarray,
        duration_s: float,
        stop_predicate: Callable[[DynamicSnapshot], bool],
    ) -> tuple[DynamicSnapshot, bool]:
        """Track an interpolated setpoint and evaluate a guard at 500 Hz.

        Rendering runs much slower than the controller. Contact and depth
        events therefore must be checked inside the physics/control rollout,
        not only once per rendered frame.
        """

        target_tip = np.asarray(nominal_tip_position, dtype=float)
        target_rpy = np.asarray(nominal_rpy, dtype=float)
        start_tip = self._nominal_tip.copy()
        start_rpy = self._nominal_rpy.copy()
        steps = max(1, int(round(float(duration_s) / self.model.opt.timestep)))
        snapshot: DynamicSnapshot | None = None
        stopped = False
        for index in range(steps):
            fraction = (index + 1) / steps
            tip = start_tip + fraction * (target_tip - start_tip)
            rpy = start_rpy + fraction * (target_rpy - start_rpy)
            snapshot = self.step(tip, rpy)
            if snapshot.command.hard_stop or snapshot.state_invalid:
                break
            if (index + 1) % self.control_stride == 0 and stop_predicate(snapshot):
                stopped = True
                break
        assert snapshot is not None
        return snapshot, stopped
