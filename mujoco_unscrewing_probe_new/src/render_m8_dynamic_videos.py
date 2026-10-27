from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from src.cartesian_admittance import AdmittanceConfig
from src.mujoco_admittance_rollout import DynamicSnapshot, MuJoCoAdmittanceRollout
from src.render_m8_sampling_videos import (
    ALPHA,
    ALPHA_DEG,
    CONTROLLER_THRESHOLDS,
    FPS,
    HEIGHT,
    OUT,
    ROOT,
    SPIRAL_PROFILES,
    VIDEO_OUT,
    WIDTH,
    VideoCase,
    build_cases,
    color_scene,
    contact_tags,
    make_camera,
    pseudo_ft_channels,
    planned_outward_spiral,
    render_frame,
    render_model_xml,
    smoothstep,
    valid_contact_infos,
)


FRAME_DT = 1.0 / FPS
SIM_TIMESTEP_S = 0.0005
CONTACT_TIMECONST_S = 0.05
TARGET_DOWN_FORCE_N = 3.0
OUTWARD_PITCH_M = 0.00040
TRANSPORT_SPEED_M_S = 0.0030
SPIRAL_SPEED_M_S = 0.0030


def load_spec(spec_path: str | Path | None = None) -> dict:
    path = (
        ROOT / "configs" / "allen_m8_source.json"
        if spec_path is None
        else Path(spec_path)
    )
    return json.loads(path.read_text(encoding="utf-8"))


def setup(
    case: VideoCase,
    *,
    render: bool = True,
    wrench_observation=None,
    admittance_config: AdmittanceConfig | None = None,
    spec_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[
    dict,
    mujoco.MjModel,
    mujoco.MjData,
    MuJoCoAdmittanceRollout,
    mujoco.Renderer | None,
    mujoco.MjvCamera,
    mujoco.MjvOption,
]:
    spec = load_spec(spec_path)
    model_xml = (
        render_model_xml(spec)
        if model_path is None
        else Path(model_path).read_text(encoding="utf-8")
    )
    model = mujoco.MjModel.from_xml_string(model_xml)
    model.opt.timestep = SIM_TIMESTEP_S
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    color_scene(model)
    rollout = MuJoCoAdmittanceRollout(
        model,
        data,
        float(spec["bit_length_m"]),
        float(spec["bit_mass_kg"]),
        wrench_reference_to_tip_m=(
            None
            if spec.get("wrench_reference_to_tip_m") is None
            else float(spec["wrench_reference_to_tip_m"])
        ),
        target_down_force_n=TARGET_DOWN_FORCE_N,
        contact_timeconst_s=CONTACT_TIMECONST_S,
        wrench_observation=wrench_observation,
        admittance_config=admittance_config,
    )
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH) if render else None
    camera = make_camera(model)
    scene_option = mujoco.MjvOption()
    return spec, model, data, rollout, renderer, camera, scene_option


def actual_values(snapshot: DynamicSnapshot) -> tuple[np.ndarray, float, float, float]:
    return (
        snapshot.tip_position[:2].copy(),
        float(snapshot.tip_position[2]),
        float(snapshot.roll),
        float(snapshot.pitch),
    )


def append_frame(
    frames: list[np.ndarray],
    contact_counts: list[int],
    z_values: list[float],
    force_norms: list[float],
    torque_norms: list[float],
    trail: list[np.ndarray],
    polar_history: list[tuple[float, float, float, float]],
    spec: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    scene_option: mujoco.MjvOption,
    case: VideoCase,
    stage: str,
    index: int,
    frame_count: int,
    snapshot: DynamicSnapshot,
    phase: float | None,
    phase_fraction: float,
    note: str,
) -> set[str]:
    xy, tip_z, roll, pitch = actual_values(snapshot)
    infos = valid_contact_infos(model, data)
    trail.append(xy.copy())
    frames.append(
        render_frame(
            model,
            data,
            renderer,
            camera,
            scene_option,
            spec,
            case,
            stage,
            index,
            frame_count,
            xy,
            tip_z,
            roll,
            pitch,
            phase,
            phase_fraction,
            infos,
            trail,
            polar_history,
            note,
        )
    )
    contact_counts.append(len(infos))
    z_values.append(tip_z)
    force_norms.append(float(np.linalg.norm(snapshot.measured_wrench[:3])))
    torque_norms.append(float(np.linalg.norm(snapshot.measured_wrench[3:])))
    return contact_tags(infos)


def render_conical(case: VideoCase) -> tuple[list[np.ndarray], dict]:
    spec, model, data, rollout, renderer, camera, scene_option = setup(case)
    nominal_tip = np.array([case.x, case.y, 0.00030], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip)
    snapshot = rollout.advance(nominal_tip, np.zeros(3), 0.80)
    baseline_tip = snapshot.tip_position.copy()

    ramp_frames = 10
    scan_frames = 52
    return_frames = 22
    frame_count = ramp_frames + scan_frames + return_frames
    frames: list[np.ndarray] = []
    contact_counts: list[int] = []
    z_values: list[float] = []
    force_norms: list[float] = []
    torque_norms: list[float] = []
    trail: list[np.ndarray] = []
    polar_history: list[tuple[float, float, float, float]] = []
    xy_deviations: list[float] = []
    orientation_errors: list[float] = []
    hard_stop = False
    index = 0

    for ramp_index in range(ramp_frames):
        scale = smoothstep((ramp_index + 1) / ramp_frames)
        rpy = np.array([0.0, ALPHA * scale, 0.0])
        snapshot = rollout.advance_interpolated(nominal_tip, rpy, FRAME_DT)
        xy_error = float(np.linalg.norm(snapshot.tip_position[:2] - nominal_tip[:2]))
        xy_deviations.append(xy_error)
        orientation_errors.append(float(np.linalg.norm(np.array([snapshot.roll, snapshot.pitch]) - rpy[:2])))
        note = (
            f"dynamic ramp | Fz={snapshot.measured_wrench[2]:+.2f}N "
            f"e_xy={1000.0 * xy_error:.2f}mm"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "dynamic ramp to 15 deg cone", index, frame_count, snapshot, 0.0, 0.0, note,
        )
        index += 1
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        if hard_stop:
            break

    if not hard_stop:
        for scan_index in range(scan_frames):
            theta = 2.0 * math.pi * scan_index / max(scan_frames - 1, 1)
            rpy = np.array([ALPHA * math.sin(theta), ALPHA * math.cos(theta), 0.0])
            snapshot = rollout.advance_interpolated(nominal_tip, rpy, FRAME_DT)
            xy_error = float(np.linalg.norm(snapshot.tip_position[:2] - nominal_tip[:2]))
            xy_deviations.append(xy_error)
            orientation_errors.append(float(np.linalg.norm(np.array([snapshot.roll, snapshot.pitch]) - rpy[:2])))
            infos = valid_contact_infos(model, data)
            xy, tip_z, roll, pitch = actual_values(snapshot)
            polar_history.append((theta, *pseudo_ft_channels(spec, xy, tip_z, roll, pitch, infos)))
            note = (
                f"dynamic full scan | Fz={snapshot.measured_wrench[2]:+.2f}N "
                f"e_xy={1000.0 * xy_error:.2f}mm"
            )
            append_frame(
                frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
                spec, model, data, renderer, camera, scene_option, case,
                "dynamic conical fingerprint", index, frame_count, snapshot, theta,
                scan_index / max(scan_frames - 1, 1), note,
            )
            index += 1
            hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
            if hard_stop:
                break

    if not hard_stop:
        final_theta = 2.0 * math.pi
        for return_index in range(return_frames):
            scale = 1.0 - smoothstep((return_index + 1) / return_frames)
            rpy = np.array(
                [ALPHA * math.sin(final_theta) * scale, ALPHA * math.cos(final_theta) * scale, 0.0]
            )
            snapshot = rollout.advance_interpolated(nominal_tip, rpy, FRAME_DT)
            xy_error = float(np.linalg.norm(snapshot.tip_position[:2] - nominal_tip[:2]))
            xy_deviations.append(xy_error)
            orientation_errors.append(float(np.linalg.norm(np.array([snapshot.roll, snapshot.pitch]) - rpy[:2])))
            note = (
                f"dynamic return upright | Fz={snapshot.measured_wrench[2]:+.2f}N "
                f"e_xy={1000.0 * xy_error:.2f}mm"
            )
            append_frame(
                frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
                spec, model, data, renderer, camera, scene_option, case,
                "dynamic return upright / verify", index, frame_count, snapshot, final_theta, 1.0, note,
            )
            index += 1
            hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
            if hard_stop:
                break

    renderer.close()
    metrics = {
        "result": "hard_stop" if hard_stop else "dynamic_full_scan_return_upright",
        "dynamic_physics": True,
        "admittance_controlled": True,
        "max_force_N": max(force_norms, default=0.0),
        "max_torque_Nm": max(torque_norms, default=0.0),
        "max_contact_count": max(contact_counts, default=0),
        "min_tip_z_m": min(z_values, default=float(snapshot.tip_position[2])),
        "max_tip_z_m": max(z_values, default=float(snapshot.tip_position[2])),
        "hard_stop": hard_stop,
        "failure_reason": snapshot.failure_reason,
        "final_roll_deg": math.degrees(snapshot.roll),
        "final_pitch_deg": math.degrees(snapshot.pitch),
        "nominal_tip_xy_fixed": True,
        "max_tip_xy_deviation_m": max(xy_deviations, default=0.0),
        "final_tip_xy_deviation_m": float(np.linalg.norm(snapshot.tip_position[:2] - nominal_tip[:2])),
        "max_orientation_tracking_error_deg": math.degrees(max(orientation_errors, default=0.0)),
    }
    return frames, metrics


def spiral_path(profile_key: str, start_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = SPIRAL_PROFILES[profile_key]
    _, xy = planned_outward_spiral(profile, start_xy, samples=4000)
    increments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(increments)))
    return arc, xy


def render_outward(case: VideoCase) -> tuple[list[np.ndarray], dict]:
    spec, model, data, rollout, renderer, camera, scene_option = setup(case)
    profile = SPIRAL_PROFILES[case.key]
    start_xy = np.array([case.x, case.y], dtype=float)
    nominal_tip = np.array([case.x, case.y, 0.00030], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip)
    snapshot = rollout.advance(nominal_tip, np.zeros(3), 0.80)
    baseline_z = float(snapshot.tip_position[2])

    transport_target = np.asarray(profile.predicted_center_xy_m, dtype=float)
    transport_distance = float(np.linalg.norm(transport_target - start_xy))
    speed_scale = max(0.25, float(profile.nominal_speed_scale))
    transport_frames = max(
        8,
        int(math.ceil(transport_distance / (TRANSPORT_SPEED_M_S * speed_scale) * FPS)),
    )
    arc, path_xy = spiral_path(case.key, start_xy)
    spiral_frames = max(
        12,
        int(math.ceil(float(arc[-1]) / (SPIRAL_SPEED_M_S * speed_scale) * FPS)),
    )
    settle_frames = 2
    frame_count = transport_frames + spiral_frames + settle_frames

    frames: list[np.ndarray] = []
    contact_counts: list[int] = []
    z_values: list[float] = []
    force_norms: list[float] = []
    torque_norms: list[float] = []
    trail: list[np.ndarray] = []
    polar_history: list[tuple[float, float, float, float]] = []
    hard_stop = False
    capture_streak = 0
    capture_event = "not_reached"
    index = 0
    final_nominal_xy = start_xy.copy()

    for transport_index in range(transport_frames):
        fraction = smoothstep((transport_index + 1) / transport_frames)
        xy = start_xy + fraction * (transport_target - start_xy)
        final_nominal_xy = xy.copy()
        nominal_tip = np.array([xy[0], xy[1], 0.00030])
        def transport_guard(candidate: DynamicSnapshot) -> bool:
            depth_drop = baseline_z - float(candidate.tip_position[2])
            tags = contact_tags(valid_contact_infos(model, data))
            return bool(
                depth_drop >= CONTROLLER_THRESHOLDS["chamfer_depth_drop_mm"] / 1000.0
                and tags & {"chamfer", "wall"}
            )

        snapshot, transport_capture = rollout.advance_interpolated_until(
            nominal_tip,
            np.zeros(3),
            FRAME_DT,
            transport_guard,
        )
        if transport_capture:
            final_nominal_xy = snapshot.nominal_tip_position[:2].copy()
        note = (
            f"posterior transport {100.0 * fraction:.0f}% | Fz={snapshot.measured_wrench[2]:+.2f}N "
            f"d={1000.0 * transport_distance:.2f}mm"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "bounded posterior-guided transport", index, frame_count, snapshot, None,
            0.25 * fraction, note,
        )
        index += 1
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        if hard_stop:
            break
        if transport_capture:
            capture_streak = 1
            capture_event = "chamfer_depth_candidate_hold"
            break

    if not hard_stop and capture_event == "not_reached":
        for spiral_index in range(spiral_frames):
            distance = float(arc[-1]) * (spiral_index + 1) / spiral_frames
            x = float(np.interp(distance, arc, path_xy[:, 0]))
            y = float(np.interp(distance, arc, path_xy[:, 1]))
            final_nominal_xy = np.array([x, y])
            nominal_tip = np.array([x, y, 0.00030])
            def spiral_guard(candidate: DynamicSnapshot) -> bool:
                depth_drop = baseline_z - float(candidate.tip_position[2])
                tags = contact_tags(valid_contact_infos(model, data))
                return bool(
                    depth_drop >= CONTROLLER_THRESHOLDS["chamfer_depth_drop_mm"] / 1000.0
                    and tags & {"chamfer", "wall"}
                )

            snapshot, capture_signal = rollout.advance_interpolated_until(
                nominal_tip,
                np.zeros(3),
                FRAME_DT,
                spiral_guard,
            )
            if capture_signal:
                final_nominal_xy = snapshot.nominal_tip_position[:2].copy()
            progress = (spiral_index + 1) / spiral_frames
            depth_drop = baseline_z - float(snapshot.tip_position[2])
            capture_streak = 1 if capture_signal else 0
            note = (
                f"local outward coverage | pitch={1000.0 * OUTWARD_PITCH_M:.2f}mm/rev "
                f"Fz={snapshot.measured_wrench[2]:+.2f}N dz={1000.0 * depth_drop:.2f}mm"
            )
            append_frame(
                frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
                spec, model, data, renderer, camera, scene_option, case,
                "dynamic local outward Archimedean spiral", index, frame_count, snapshot, None,
                0.25 + 0.65 * progress, note,
            )
            index += 1
            hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
            if hard_stop:
                break
            if capture_signal:
                capture_event = "chamfer_depth_candidate_hold"
                break

    if not hard_stop and capture_event == "chamfer_depth_candidate_hold":
        required_cycles = max(
            1,
            int(round(CONTROLLER_THRESHOLDS["capture_dwell_ms"] / 1000.0 / rollout.controller_dt_s)),
        )
        dwell_cycles = 1

        def dwell_guard(candidate: DynamicSnapshot) -> bool:
            nonlocal dwell_cycles
            depth_drop = baseline_z - float(candidate.tip_position[2])
            tags = contact_tags(valid_contact_infos(model, data))
            persistent = bool(
                depth_drop >= CONTROLLER_THRESHOLDS["chamfer_depth_drop_mm"] / 1000.0
                and tags & {"chamfer", "wall"}
            )
            dwell_cycles = dwell_cycles + 1 if persistent else 0
            return dwell_cycles >= required_cycles

        nominal_tip = np.array([final_nominal_xy[0], final_nominal_xy[1], 0.00030])
        snapshot, latched = rollout.advance_interpolated_until(
            nominal_tip,
            np.zeros(3),
            max(FRAME_DT, CONTROLLER_THRESHOLDS["capture_dwell_ms"] / 1000.0),
            dwell_guard,
        )
        capture_event = "chamfer_depth_cue_latched" if latched else "capture_candidate_not_persistent"
        note = (
            f"XY frozen | {CONTROLLER_THRESHOLDS['capture_dwell_ms']:.0f}ms capture dwell "
            f"Fz={snapshot.measured_wrench[2]:+.2f}N"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "rim/chamfer capture dwell", index, frame_count, snapshot, None, 1.0, note,
        )
        index += 1
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid

    if capture_event == "not_reached" and not hard_stop:
        final_tags = contact_tags(valid_contact_infos(model, data))
        if "chamfer" in final_tags or "wall" in final_tags:
            capture_event = "rim_chamfer_contact_at_path_end"
        else:
            capture_event = "outward_radius_exhausted"

    renderer.close()
    metrics = {
        "result": "hard_stop" if hard_stop else capture_event,
        "dynamic_physics": True,
        "admittance_controlled": True,
        "transport_distance_m": transport_distance,
        "transport_target_xy_m": transport_target.tolist(),
        "outward_pitch_m_per_rev": profile.spiral_pitch_m_per_rev,
        "outward_max_radius_m": profile.spiral_max_radius_m,
        "spiral_phase_source": "posterior_transport_direction_only",
        "max_force_N": max(force_norms, default=0.0),
        "max_torque_Nm": max(torque_norms, default=0.0),
        "max_contact_count": max(contact_counts, default=0),
        "min_tip_z_m": min(z_values, default=float(snapshot.tip_position[2])),
        "max_tip_z_m": max(z_values, default=float(snapshot.tip_position[2])),
        "hard_stop": hard_stop,
        "failure_reason": snapshot.failure_reason,
    }
    return frames, metrics


def render_terminal(case: VideoCase) -> tuple[list[np.ndarray], dict]:
    spec, model, data, rollout, renderer, camera, scene_option = setup(case)
    rollout.target_wrench.fill(0.0)
    rollout.set_enabled_axes(np.zeros(6, dtype=bool))
    nominal_tip = np.array([0.0, 0.0, case.nominal_tip_z], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip)
    snapshot = rollout.advance(nominal_tip, np.zeros(3), 0.10)
    nominal_tip = np.array([case.x, case.y, case.nominal_tip_z], dtype=float)
    snapshot = rollout.advance_interpolated(nominal_tip, np.zeros(3), 0.40)

    gate_frames = 8
    descent_frames = 58
    hold_frames = 30
    frame_count = gate_frames + descent_frames + hold_frames
    frames: list[np.ndarray] = []
    contact_counts: list[int] = []
    z_values: list[float] = []
    force_norms: list[float] = []
    torque_norms: list[float] = []
    trail: list[np.ndarray] = []
    polar_history: list[tuple[float, float, float, float]] = []
    hard_stop = False
    bottom_reached = False
    wall_gate_triggered = False
    index = 0
    bottom_z = -float(spec["socket_depth_m"])

    for gate_index in range(gate_frames):
        snapshot = rollout.advance_interpolated(nominal_tip, np.zeros(3), FRAME_DT)
        lateral_force = float(np.linalg.norm(snapshot.measured_wrench[:2]))
        torque_xy_mnm = 1.0e3 * float(np.linalg.norm(snapshot.measured_wrench[3:5]))
        wall_gate_triggered = bool(
            1000.0 * float(snapshot.tip_position[2]) <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
            and (
                lateral_force >= CONTROLLER_THRESHOLDS["inner_wall_ft_N"]
                or torque_xy_mnm >= CONTROLLER_THRESHOLDS["inner_wall_tau_xy_mNm"]
            )
        )
        note = (
            f"inner-wall gate | Ft={lateral_force:.2f}N tau_xy={torque_xy_mnm:.2f}mNm "
            f"trigger={wall_gate_triggered}"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "direct F/T + depth gate", index, frame_count, snapshot, None,
            (gate_index + 1) / gate_frames * 0.12, note,
        )
        index += 1

    if wall_gate_triggered:
        rollout.set_enabled_axes(np.array([True, True, True, False, False, False]))
        rollout.set_target_down_force(0.8)
        rollout.reset_admittance()

    for descent_index in range(descent_frames if wall_gate_triggered else 0):
        fraction = smoothstep((descent_index + 1) / descent_frames)
        commanded_z = case.nominal_tip_z + fraction * (-0.0015 - case.nominal_tip_z)
        nominal_tip = np.array([case.x, case.y, commanded_z])
        snapshot = rollout.advance_interpolated(nominal_tip, np.zeros(3), FRAME_DT)
        actual_z = float(snapshot.tip_position[2])
        lateral_force = float(np.linalg.norm(snapshot.measured_wrench[:2]))
        bottom_reached = bool(
            abs(actual_z - bottom_z) <= CONTROLLER_THRESHOLDS["bottom_depth_tolerance_mm"] / 1000.0
            and abs(float(snapshot.measured_wrench[2])) >= CONTROLLER_THRESHOLDS["bottom_fz_N"]
        )
        note = (
            f"dynamic wall-guided insertion | Ft={lateral_force:.2f}N "
            f"Fz={snapshot.measured_wrench[2]:+.2f}N"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "dynamic inner-wall to upright insertion", index, frame_count, snapshot, None,
            fraction, note,
        )
        index += 1
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        if hard_stop or bottom_reached:
            break

    if not hard_stop:
        final_nominal = nominal_tip.copy()
        for _ in range(hold_frames):
            snapshot = rollout.advance_interpolated(final_nominal, np.zeros(3), FRAME_DT)
            note = (
                f"CAD-depth verify | z={1000.0 * snapshot.tip_position[2]:+.2f}mm "
                f"Fz={snapshot.measured_wrench[2]:+.2f}N"
            )
            append_frame(
                frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
                spec, model, data, renderer, camera, scene_option, case,
                "bottom contact / geometric insertion complete", index, frame_count, snapshot, None, 1.0, note,
            )
            index += 1
            hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
            if hard_stop:
                break

    renderer.close()
    metrics = {
        "result": "hard_stop" if hard_stop else ("cad_bottom_verified" if bottom_reached else "descent_complete"),
        "dynamic_physics": True,
        "admittance_controlled": True,
        "bottom_reached": bottom_reached,
        "wall_gate_triggered": wall_gate_triggered,
        "max_force_N": max(force_norms, default=0.0),
        "max_torque_Nm": max(torque_norms, default=0.0),
        "max_contact_count": max(contact_counts, default=0),
        "min_tip_z_m": min(z_values, default=float(snapshot.tip_position[2])),
        "max_tip_z_m": max(z_values, default=float(snapshot.tip_position[2])),
        "hard_stop": hard_stop,
        "failure_reason": snapshot.failure_reason,
    }
    return frames, metrics


def render_capture(case: VideoCase) -> tuple[list[np.ndarray], dict]:
    spec, model, data, rollout, renderer, camera, scene_option = setup(case)
    rollout.target_wrench.fill(0.0)
    rollout.set_enabled_axes(np.zeros(6, dtype=bool))
    nominal_tip = np.array([case.x, case.y, 0.00030], dtype=float)
    rollout.set_initial_tip_pose(nominal_tip)
    snapshot = rollout.advance(nominal_tip, np.zeros(3), 0.08)
    rollout.set_enabled_axes(np.array([False, False, True, False, False, False]))
    rollout.set_target_down_force(0.15)
    rollout.reset_admittance()

    frame_count = 45
    force_ramp_frames = 15
    frames: list[np.ndarray] = []
    contact_counts: list[int] = []
    z_values: list[float] = []
    force_norms: list[float] = []
    torque_norms: list[float] = []
    trail: list[np.ndarray] = []
    polar_history: list[tuple[float, float, float, float]] = []
    hard_stop = False
    inner_wall_captured = False

    for index in range(frame_count):
        force_fraction = smoothstep(min(1.0, (index + 1) / force_ramp_frames))
        target_force = 0.15 + force_fraction * (0.8 - 0.15)
        rollout.set_target_down_force(target_force)
        def inner_wall_guard(candidate: DynamicSnapshot) -> bool:
            tags = contact_tags(valid_contact_infos(model, data))
            lateral_force = float(np.linalg.norm(candidate.measured_wrench[:2]))
            torque_xy_mnm = 1.0e3 * float(np.linalg.norm(candidate.measured_wrench[3:5]))
            return bool(
                1000.0 * float(candidate.tip_position[2])
                <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
                and "wall" in tags
                and (
                    lateral_force >= CONTROLLER_THRESHOLDS["inner_wall_ft_N"]
                    or torque_xy_mnm >= CONTROLLER_THRESHOLDS["inner_wall_tau_xy_mNm"]
                )
            )

        snapshot, inner_wall_captured = rollout.advance_interpolated_until(
            nominal_tip,
            np.zeros(3),
            FRAME_DT,
            inner_wall_guard,
        )
        note = (
            f"upright chamfer guidance | Fd={target_force:.2f}N Fz={snapshot.measured_wrench[2]:+.2f}N "
            f"wall_gate={inner_wall_captured}"
        )
        append_frame(
            frames, contact_counts, z_values, force_norms, torque_norms, trail, polar_history,
            spec, model, data, renderer, camera, scene_option, case,
            "upright rim/chamfer capture", index, frame_count, snapshot, None,
            (index + 1) / frame_count, note,
        )
        hard_stop = snapshot.command.hard_stop or snapshot.state_invalid
        if hard_stop or inner_wall_captured:
            break

    renderer.close()
    metrics = {
        "result": "hard_stop" if hard_stop else (
            "inner_wall_capture_cue_latched" if inner_wall_captured else "capture_not_reached"
        ),
        "dynamic_physics": True,
        "admittance_controlled": True,
        "upright_motion": True,
        "inner_wall_captured": inner_wall_captured,
        "max_force_N": max(force_norms, default=0.0),
        "max_torque_Nm": max(torque_norms, default=0.0),
        "max_contact_count": max(contact_counts, default=0),
        "min_tip_z_m": min(z_values, default=float(snapshot.tip_position[2])),
        "max_tip_z_m": max(z_values, default=float(snapshot.tip_position[2])),
        "hard_stop": hard_stop,
        "failure_reason": snapshot.failure_reason,
    }
    return frames, metrics


def write_video(case: VideoCase, frames: list[np.ndarray], metrics: dict) -> dict:
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    output = VIDEO_OUT / case.filename
    imageio.mimsave(output, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    return {
        "key": case.key,
        "path": str(output.relative_to(ROOT)),
        "offline_label": case.offline_label,
        "protocol": case.protocol,
        "result": metrics["result"],
        "frames": len(frames),
        "fps": FPS,
        **metrics,
    }


def render_case(case: VideoCase) -> dict:
    if case.mode == "full":
        frames, metrics = render_conical(case)
    elif case.mode == "capture":
        frames, metrics = render_capture(case)
    elif case.mode == "terminal":
        frames, metrics = render_terminal(case)
    elif case.mode == "spiral":
        frames, metrics = render_outward(case)
    else:
        raise ValueError(f"Dynamic renderer does not support mode: {case.mode}")
    if not frames:
        raise RuntimeError(f"No frames generated for {case.key}")
    return write_video(case, frames, metrics)


def main(selected_keys: set[str] | None = None) -> None:
    cases = [case for case in build_cases() if case.mode != "abort"]
    if selected_keys:
        cases = [case for case in cases if case.key in selected_keys]
    records = [render_case(case) for case in cases]

    manifest_path = OUT / "reports" / "m8_sampling_video_manifest.json"
    if selected_keys and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_key = {record["key"]: record for record in old.get("videos", [])}
        by_key.update({record["key"]: record for record in records})
        records = [by_key[key] for key in sorted(by_key)]

    manifest = {
        "claim_boundary": (
            "All listed videos are segmented dynamic MuJoCo rollouts driven by a 500 Hz Cartesian "
            "admittance outer loop and a wrench-applied Cartesian PD inner loop. They are not one "
            "automatically routed episode, real-robot validation, or learned-direction results. "
            "V6 profiles use assumed posteriors, and V4/V6 safe-progress guards use privileged "
            "contact tags as simulation proxies without defining an online chamfer class."
        ),
        "physics": {
            "simulation_timestep_s": SIM_TIMESTEP_S,
            "contact_solref_timeconst_s": CONTACT_TIMECONST_S,
            "search_target_down_force_N": TARGET_DOWN_FORCE_N,
            "capture_and_insertion_force_N": "0.15->0.8 for V4; 0.8 for V5",
            "controller": "src/cartesian_admittance.py + src/mujoco_admittance_rollout.py",
            "contact_wrench": "mj_contactForce aggregated as environment-on-tool wrench at the displayed F/T sensor frame",
        },
        "nominal_conical_template": {
            "applies_to": ["V1", "V2", "V3"],
            "alpha_deg": ALPHA_DEG,
            "roll": "alpha * sin(theta)",
            "pitch": "alpha * cos(theta)",
            "full_scan": True,
            "nominal_tip_center_xy": "fixed; actual contact deflection is measured",
        },
        "search_policy": {
            "sequence": "posterior-guided transport -> local outward Archimedean spiral",
            "pitch_m_per_rev": OUTWARD_PITCH_M,
            "coverage_bound": "pitch <= 2 * r_eff; current M8 simulation r_eff=0.25 mm",
            "learning_boundary": "direction posterior and confidence are assumed visualization inputs",
            "progress_gate_boundary": (
                "safe downward-progress proxy = depth drop + privileged MuJoCo chamfer/wall "
                "contact tag + 50 ms dwell; not a sensor-only real-robot gate and not a "
                "chamfer classifier"
            ),
        },
        "controller_thresholds_m8_sim_initial": CONTROLLER_THRESHOLDS,
        "videos": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", default="", help="Comma-separated subset, e.g. V3,V6c")
    args = parser.parse_args()
    keys = {value.strip() for value in args.keys.split(",") if value.strip()}
    main(keys or None)
