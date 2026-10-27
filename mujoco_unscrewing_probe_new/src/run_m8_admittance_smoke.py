from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from src.cartesian_admittance import AdmittanceConfig, CartesianAdmittanceController


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "allen_m8_source.xml"
CONFIG_PATH = ROOT / "configs" / "allen_m8_source.json"
OUT_PATH = ROOT / "outputs" / "current" / "reports" / "m8_admittance_flat_contact_smoke.json"


def environment_wrench_on_bit(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    force_world = np.zeros(3, dtype=float)
    torque_world = np.zeros(3, dtype=float)
    sensor = np.asarray(data.qpos[:3], dtype=float)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        if "bit_collision" not in (geom1, geom2):
            continue
        other = geom2 if geom1 == "bit_collision" else geom1
        if other == "workpiece_base":
            continue
        local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, local)
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


def run(duration_s: float = 0.8, timestep_s: float = 0.0001) -> dict:
    spec = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = timestep_s
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The generated geometry uses a stiff presentation/contact default. This
    # smoke isolates the controller with a slower, explicitly reported contact
    # time constant; contact-parameter sweeps remain a separate experiment.
    model.geom_solref[:, 0] = 0.05
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "allen_bit_m8")
    bit_len = float(spec["bit_length_m"])

    # Start above a supported top-face point. The admittance controller alone
    # creates the downward correction needed to establish the target reaction.
    tip_xy = np.array([0.0048, 0.0], dtype=float)
    nominal_tip_z = 0.00030
    nominal_body_pos = np.array([tip_xy[0], tip_xy[1], nominal_tip_z + bit_len / 2.0], dtype=float)
    data.qpos[:3] = nominal_body_pos
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    controller_dt = 0.002
    controller = CartesianAdmittanceController(AdmittanceConfig(dt_s=controller_dt))
    target_wrench = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    enabled = np.array([False, False, True, False, False, False])

    position_kp = np.array([1800.0, 1800.0, 2200.0])
    position_kd = np.array([70.0, 70.0, 75.0])
    rotation_kp = np.array([14.0, 14.0, 10.0])
    rotation_kd = np.array([1.2, 1.2, 0.9])
    gravity_compensation = np.array([0.0, 0.0, float(spec["bit_mass_kg"]) * 9.81])

    control_stride = max(1, int(round(controller_dt / model.opt.timestep)))
    total_steps = int(round(duration_s / model.opt.timestep))
    command = controller.step(np.zeros(6), target_wrench, enabled)
    force_z: list[float] = []
    offset_z: list[float] = []
    body_z: list[float] = []
    body_vz: list[float] = []
    contact_steps = 0

    for step in range(total_steps):
        measured = environment_wrench_on_bit(model, data)
        if step % control_stride == 0:
            command = controller.step(measured, target_wrench, enabled)

        desired_pos = nominal_body_pos + command.pose_offset[:3]
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        angular_velocity = velocity[:3]
        linear_velocity = velocity[3:]
        desired_linear_velocity = command.twist[:3]
        force_cmd = (
            position_kp * (desired_pos - np.asarray(data.qpos[:3]))
            + position_kd * (desired_linear_velocity - linear_velocity)
            + gravity_compensation
        )

        quat_error = np.zeros(3, dtype=float)
        mujoco.mju_subQuat(quat_error, np.array([1.0, 0.0, 0.0, 0.0]), np.asarray(data.qpos[3:7]))
        torque_cmd = rotation_kp * quat_error - rotation_kd * angular_velocity
        data.qfrc_applied.fill(0.0)
        mujoco.mj_applyFT(
            model,
            data,
            force_cmd,
            torque_cmd,
            np.asarray(data.qpos[:3]),
            body_id,
            data.qfrc_applied,
        )
        mujoco.mj_step(model, data)

        force_z.append(float(measured[2]))
        offset_z.append(float(command.pose_offset[2]))
        body_z.append(float(data.qpos[2]))
        body_vz.append(float(data.qvel[2]))
        if abs(measured[2]) > 0.05:
            contact_steps += 1
        if command.hard_stop:
            break

    tail = max(1, int(round(0.25 / model.opt.timestep)))
    tail_fz = np.asarray(force_z[-tail:], dtype=float)
    result = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "duration_commanded_s": duration_s,
        "duration_executed_s": len(force_z) * float(model.opt.timestep),
        "target_environment_on_tool_fz_N": float(target_wrench[2]),
        "tail_mean_fz_N": float(np.mean(tail_fz)),
        "tail_std_fz_N": float(np.std(tail_fz)),
        "tail_mean_abs_error_N": float(np.mean(np.abs(tail_fz - target_wrench[2]))),
        "final_admittance_offset_z_mm": 1000.0 * float(offset_z[-1]),
        "final_tip_z_mm": 1000.0 * (float(body_z[-1]) - bit_len / 2.0),
        "min_tip_z_mm": 1000.0 * (min(body_z) - bit_len / 2.0),
        "max_abs_body_vz_m_s": max(abs(value) for value in body_vz),
        "contact_fraction": contact_steps / max(len(force_z), 1),
        "hard_stop": bool(command.hard_stop),
        "controller_rate_hz": 1.0 / controller_dt,
        "simulation_timestep_s": float(model.opt.timestep),
        "smoke_contact_solref_timeconst_s": 0.05,
        "claim_boundary": "single supported-contact MuJoCo smoke; not chamfer/spiral/robot validation",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
