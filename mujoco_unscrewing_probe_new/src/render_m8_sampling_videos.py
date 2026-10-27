from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current"
VIDEO_OUT = OUT / "media" / "videos" / "stages"
REPORT_OUT = OUT / "reports"
MODEL_PATH = ROOT / "models" / "allen_m8_source.xml"
CONFIG_PATH = ROOT / "configs" / "allen_m8_source.json"

WIDTH = 960
HEIGHT = 720
FPS = 15
ALPHA_DEG = 15.0
ALPHA = math.radians(ALPHA_DEG)
YAW = 0.0

FORCE_DISPLAY_EPS_N = 0.03
TORQUE_DISPLAY_EPS_NM = 0.00005
WRENCH_COLORS = {
    "Fx": (255, 75, 75, 245),
    "Fy": (85, 235, 110, 245),
    "Fz": (75, 155, 255, 245),
    "tau_x": (255, 80, 210, 245),
    "tau_y": (55, 225, 245, 245),
    "tau_z": (255, 210, 50, 245),
}

CONTROLLER_THRESHOLDS = {
    "target_down_force_N": 3.0,
    "soft_force_N": 8.0,
    "hard_force_N": 20.0,
    "soft_tau_xy_mNm": 20.0,
    "hard_tau_xy_mNm": 60.0,
    "chamfer_depth_drop_mm": 0.07,
    "inner_wall_depth_mm": -0.30,
    "inner_wall_ft_N": 0.25,
    "inner_wall_tau_xy_mNm": 1.5,
    "bottom_depth_tolerance_mm": 0.15,
    "bottom_fz_N": 0.70,
    "bottom_low_wrench_lateral_force_N": 0.25,
    "bottom_low_wrench_tau_xy_mNm": 1.5,
    "bottom_stable_depth_span_mm": 0.02,
    "capture_dwell_ms": 50.0,
    "capture_trend_window_ms": 50.0,
    "capture_trend_drop_mm": 0.04,
    "capture_handoff_drop_mm": 0.04,
    "capture_handoff_force_N": 2.0,
    "capture_handoff_ramp_ms": 150.0,
    "capture_planar_stiffness_N_m": 4000.0,
    "capture_planar_damping_N_s_m": 280.0,
    "search_escape_force_N": 12.0,
    "search_escape_tau_xy_mNm": 40.0,
}


@dataclass(frozen=True)
class VideoCase:
    key: str
    filename: str
    offline_label: str
    protocol: str
    result: str
    x: float
    y: float
    nominal_tip_z: float
    mode: str


@dataclass(frozen=True)
class SpiralProfile:
    search_state: str
    confidence: float
    direction_half_width_deg: float
    turns: float
    nominal_steps: int
    predicted_center_xy_m: tuple[float, float]
    terminal_radius_m: float
    spiral_pitch_m_per_rev: float
    spiral_max_radius_m: float
    nominal_speed_scale: float
    belief_note: str


SPIRAL_PROFILES = {
    "V6a": SpiralProfile(
        search_state="outer_partial",
        confidence=0.63,
        direction_half_width_deg=24.0,
        turns=5.00,
        nominal_steps=58,
        predicted_center_xy_m=(0.00180, 0.0),
        terminal_radius_m=0.0,
        spiral_pitch_m_per_rev=0.00040,
        spiral_max_radius_m=0.00200,
        nominal_speed_scale=0.75,
        belief_note="directional posterior; transport intentionally undershoots",
    ),
    "V6b": SpiralProfile(
        search_state="double_overhang",
        confidence=0.42,
        direction_half_width_deg=46.0,
        turns=5.50,
        nominal_steps=58,
        predicted_center_xy_m=(0.00200, 0.0),
        terminal_radius_m=0.0,
        spiral_pitch_m_per_rev=0.00040,
        spiral_max_radius_m=0.00220,
        nominal_speed_scale=0.50,
        belief_note="broad/top-2 belief -> dense conservative spiral",
    ),
    "V6c": SpiralProfile(
        search_state="supported_search",
        confidence=0.86,
        direction_half_width_deg=13.0,
        turns=3.50,
        nominal_steps=58,
        predicted_center_xy_m=(0.00120, 0.0),
        terminal_radius_m=0.0,
        spiral_pitch_m_per_rev=0.00040,
        spiral_max_radius_m=0.00140,
        nominal_speed_scale=1.00,
        belief_note="narrow unimodal direction belief",
    ),
}


def planned_outward_spiral(
    profile: SpiralProfile,
    start_xy: np.ndarray,
    samples: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an outward spiral using only the available posterior direction.

    The spiral is centered at the posterior center estimate. Its first radial
    direction continues the bounded transport direction; no true hole-center
    coordinate is used to select a favorable phase.
    """

    start = np.asarray(start_xy, dtype=float)
    center = np.asarray(profile.predicted_center_xy_m, dtype=float)
    transport = center - start
    phase0 = math.atan2(float(transport[1]), float(transport[0])) if np.linalg.norm(transport) > 1.0e-12 else 0.0
    b = profile.spiral_pitch_m_per_rev / (2.0 * math.pi)
    theta_max = profile.spiral_max_radius_m / b
    theta = np.linspace(0.0, theta_max, max(2, int(samples)))
    radius = b * theta
    xy = center + np.column_stack(
        (radius * np.cos(phase0 + theta), radius * np.sin(phase0 + theta))
    )
    return theta, xy


def mm(value_m: float) -> float:
    return 1000.0 * value_m


def load_spec() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=float,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def set_bit_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tip_xy: np.ndarray,
    tip_z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = YAW,
) -> None:
    bit_len = float(load_spec()["bit_length_m"])
    q = euler_to_quat(roll, pitch, yaw)
    rot = quat_to_matrix(q)
    local_tip = np.array([0.0, 0.0, -bit_len / 2.0], dtype=float)
    tip_world = np.array([float(tip_xy[0]), float(tip_xy[1]), float(tip_z)], dtype=float)
    body_pos = tip_world - rot @ local_tip
    data.qpos[:3] = body_pos
    data.qpos[3:7] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def valid_contact_infos(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    infos: list[dict] = []
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        if "bit_collision" not in (geom1, geom2):
            continue
        other = geom2 if geom1 == "bit_collision" else geom1
        if other == "workpiece_base":
            continue
        force_local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, idx, force_local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world = frame.T @ force_local[:3]
        torque_world = frame.T @ force_local[3:]
        # mj_contactForce reports the wrench on geom2. Flip it when the bit is
        # geom1 so every stored vector is environment-on-bit.
        if geom1 == "bit_collision":
            force_world = -force_world
            torque_world = -torque_world
        infos.append(
            {
                "pos": np.asarray(contact.pos, dtype=float).copy(),
                "dist": float(contact.dist),
                "geom1": geom1,
                "geom2": geom2,
                "other": other,
                "force_world": force_world,
                "torque_world": torque_world,
            }
        )
    return infos


def contact_tags(infos: list[dict]) -> set[str]:
    tags: set[str] = set()
    for item in infos:
        other = str(item["other"])
        if other.startswith("head_top_sector"):
            tags.add("head_top")
        elif other.startswith("socket_chamfer"):
            tags.add("chamfer")
        elif other.startswith("socket_wall"):
            tags.add("wall")
        elif other == "socket_bottom":
            tags.add("bottom")
        else:
            tags.add(other)
    return tags


def find_touchdown_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xy: np.ndarray,
    z_hint: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = YAW,
    max_drop: float = 0.0010,
) -> tuple[float, list[dict]]:
    start_z = z_hint + 0.00025
    steps = max(8, int(max_drop / 0.000025))
    last_infos: list[dict] = []
    last_z = z_hint
    for i in range(steps + 1):
        z = start_z - i * 0.000025
        set_bit_pose(model, data, xy, z, roll=roll, pitch=pitch, yaw=yaw)
        infos = valid_contact_infos(model, data)
        if infos:
            return z - 0.000025, infos
        last_z = z
        last_infos = infos
    return last_z, last_infos


def color_scene(model: mujoco.MjModel) -> None:
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("head_top_sector"):
            # Collision-only top sectors. They keep the real contact geometry,
            # while clean_head_top_visual carries the presentation layer.
            model.geom_rgba[gid] = np.array([0.61, 0.63, 0.56, 0.00])
        elif name == "clean_head_top_visual":
            model.geom_rgba[gid] = np.array([0.58, 0.57, 0.50, 1.00])
        elif name.startswith("socket_chamfer"):
            # Keep collision active, but hide the segmented chamfer mesh in the
            # render. Its faceted collision wedges are physically useful but
            # visually read as ugly protruding triangles.
            model.geom_rgba[gid] = np.array([0.78, 0.70, 0.46, 0.00])
        elif name.startswith("socket_wall"):
            model.geom_rgba[gid] = np.array([0.18, 0.19, 0.23, 0.00])
        elif name == "socket_bottom":
            model.geom_rgba[gid] = np.array([0.12, 0.13, 0.15, 1.00])
        elif name == "bit_collision":
            model.geom_rgba[gid] = np.array([0.02, 0.22, 0.58, 0.58])
        elif name == "screw_head_side_visual":
            model.geom_rgba[gid] = np.array([0.42, 0.42, 0.39, 0.28])
        elif name == "workpiece_base":
            model.geom_rgba[gid] = np.array([0.16, 0.16, 0.16, 1.00])
    if model.nsite:
        model.site_rgba[:, 3] = 0.0


def camera_snapshot(scene: mujoco.MjvScene, width: int, height: int) -> dict:
    c0 = scene.camera[0]
    c1 = scene.camera[1]
    forward = np.asarray(c0.forward, dtype=float)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    up = np.asarray(c0.up, dtype=float)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    pos = 0.5 * (np.asarray(c0.pos, dtype=float) + np.asarray(c1.pos, dtype=float))
    near = max(float(c0.frustum_near), 1.0e-12)
    vertical_scale = abs(float(c0.frustum_top)) / near
    return {
        "pos": pos,
        "forward": forward,
        "up": up,
        "right": right,
        "vertical_scale": max(vertical_scale, 1.0e-12),
        "aspect": float(width) / max(float(height), 1.0),
        "width": int(width),
        "height": int(height),
    }


def project_world(point: np.ndarray, camera: dict) -> tuple[int, int] | None:
    p = np.asarray(point, dtype=float)
    v = p - camera["pos"]
    z = float(np.dot(v, camera["forward"]))
    if z <= 1.0e-6:
        return None
    x = float(np.dot(v, camera["right"]))
    y = float(np.dot(v, camera["up"]))
    x_ndc = x / (z * camera["vertical_scale"] * camera["aspect"])
    y_ndc = y / (z * camera["vertical_scale"])
    sx = int(round(camera["width"] * (0.5 + 0.5 * x_ndc)))
    sy = int(round(camera["height"] * (0.5 - 0.5 * y_ndc)))
    if sx < -200 or sx > camera["width"] + 200 or sy < -200 or sy > camera["height"] + 200:
        return None
    return sx, sy


def hex_vertices(apothem: float, yaw: float = 0.0) -> list[np.ndarray]:
    radius = apothem / math.cos(math.pi / 6.0)
    return [
        np.array([radius * math.cos(yaw + k * math.pi / 3.0), radius * math.sin(yaw + k * math.pi / 3.0)])
        for k in range(6)
    ]


def format_float_list(values: list[float] | list[int]) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def clean_head_top_mesh_xml(spec: dict, segments: int = 96) -> str:
    head_r = float(spec["screw_head_diameter_m"]) / 2.0
    socket_top_apothem = float(spec["derived"]["socket_top_af_m"]) / 2.0
    z = 0.000012
    vertices: list[float] = []
    faces: list[int] = []

    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        local = (theta + math.pi / 6.0) % (math.pi / 3.0) - math.pi / 6.0
        inner_r = socket_top_apothem / max(math.cos(local), 1.0e-9)
        vertices.extend([head_r * math.cos(theta), head_r * math.sin(theta), z])
        vertices.extend([inner_r * math.cos(theta), inner_r * math.sin(theta), z])

    for i in range(segments):
        j = (i + 1) % segments
        oi = 2 * i
        ii = 2 * i + 1
        oj = 2 * j
        ij = 2 * j + 1
        faces.extend([oi, oj, ij, oi, ij, ii])
        faces.extend([ij, oj, oi, ii, ij, oi])

    return (
        '    <mesh name="m8_clean_head_top_visual" inertia="shell" '
        f'vertex="{format_float_list(vertices)}" '
        f'face="{format_float_list(faces)}"/>\n'
    )


def render_model_xml(spec: dict) -> str:
    xml = MODEL_PATH.read_text(encoding="utf-8")
    xml = xml.replace("  </asset>", clean_head_top_mesh_xml(spec) + "  </asset>", 1)
    xml = xml.replace(
        '<body name="fixed_screw_m8" pos="0 0 0">',
        (
            '<body name="fixed_screw_m8" pos="0 0 0">\n'
            '      <geom name="clean_head_top_visual" type="mesh" mesh="m8_clean_head_top_visual" '
            'contype="0" conaffinity="0" rgba="0.58 0.57 0.50 1"/>'
        ),
        1,
    )
    return xml


def draw_contact_points(draw: ImageDraw.ImageDraw, camera: dict, infos: list[dict]) -> int:
    drawn = 0
    for item in infos:
        point = project_world(np.asarray(item["pos"], dtype=float), camera)
        if point is None:
            continue
        r = 4
        draw.ellipse(
            (point[0] - r, point[1] - r, point[0] + r, point[1] + r),
            fill=(255, 205, 0, 232),
            outline=(255, 245, 120, 255),
            width=1,
        )
        drawn += 1
    return drawn


def map_plan(point_xy: np.ndarray, rect: tuple[int, int, int, int], scale_m: float) -> tuple[int, int]:
    left, top, right, bottom = rect
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    half = 0.46 * min(right - left, bottom - top)
    return (
        int(round(cx + half * float(point_xy[0]) / scale_m)),
        int(round(cy - half * float(point_xy[1]) / scale_m)),
    )


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill: tuple[int, int, int, int], width: int = 2) -> None:
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=fill, width=width)


def _draw_arrow_2d(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int, int],
    width: int = 3,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm < 4.0:
        return
    ux, uy = dx / norm, dy / norm
    size = 8
    left = (end[0] - size * ux - 0.45 * size * uy, end[1] - size * uy + 0.45 * size * ux)
    right = (end[0] - size * ux + 0.45 * size * uy, end[1] - size * uy - 0.45 * size * ux)
    draw.polygon([end, left, right], fill=color)


def _draw_label_2d(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    text_value: str,
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = point
    bbox = draw.textbbox((x, y), text_value, font=font)
    draw.rounded_rectangle(
        (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2),
        radius=3,
        fill=(0, 0, 0, 165),
        outline=(255, 255, 255, 35),
    )
    draw.text((x, y), text_value, fill=color, font=font)


def fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textlength(value, font=font) <= max_width:
        return value
    suffix = "..."
    trimmed = value
    while trimmed and draw.textlength(trimmed + suffix, font=font) > max_width:
        trimmed = trimmed[:-1]
    return trimmed.rstrip() + suffix


def _axis_perpendicular_basis(axis: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(axis, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1.0e-12)
    ref = np.asarray(camera["up"], dtype=float)
    if abs(float(np.dot(ref, a))) > 0.82:
        ref = np.asarray(camera["right"], dtype=float)
    u = ref - float(np.dot(ref, a)) * a
    u /= max(float(np.linalg.norm(u)), 1.0e-12)
    v = np.cross(a, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-12)
    return u, v


def _draw_torque_arc_3d(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    center: np.ndarray,
    axis: np.ndarray,
    sign: float,
    color: tuple[int, int, int, int],
    radius: float,
) -> tuple[int, int] | None:
    u, v = _axis_perpendicular_basis(axis, camera)
    theta0 = -0.70 * math.pi
    theta1 = theta0 + (1.45 * math.pi if sign >= 0.0 else -1.45 * math.pi)
    points: list[tuple[int, int]] = []
    for theta in np.linspace(theta0, theta1, 34):
        world = center + radius * (math.cos(float(theta)) * u + math.sin(float(theta)) * v)
        point = project_world(world, camera)
        if point is not None:
            points.append(point)
    draw_polyline(draw, points, color, width=3)
    if len(points) >= 2:
        _draw_arrow_2d(draw, points[-2], points[-1], color, width=3)
        return points[-1]
    return points[-1] if points else None


def draw_plan_panel(
    draw: ImageDraw.ImageDraw,
    spec: dict,
    current_xy: np.ndarray,
    trail: list[np.ndarray],
    infos: list[dict],
    font: ImageFont.ImageFont,
    title: str,
    profile: SpiralProfile | None = None,
) -> None:
    rect = (624, 452, 944, 704)
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0, 150), outline=(255, 255, 255, 70), width=1)
    draw.text((left + 10, top + 8), title, fill=(235, 245, 255, 235), font=font)
    plot_rect = (left + 18, top + 34, right - 14, bottom - 12)
    draw.rectangle(plot_rect, outline=(255, 255, 255, 55), width=1)
    scale_m = 0.008
    head_r = float(spec["screw_head_diameter_m"]) / 2.0
    socket_top_apothem = float(spec["derived"]["socket_top_af_m"]) / 2.0
    bit_apothem = float(spec["key_af_m"]) / 2.0

    center = map_plan(np.array([0.0, 0.0]), plot_rect, scale_m)
    head_px = int(round(0.46 * min(plot_rect[2] - plot_rect[0], plot_rect[3] - plot_rect[1]) * head_r / scale_m))
    draw.ellipse(
        (center[0] - head_px, center[1] - head_px, center[0] + head_px, center[1] + head_px),
        outline=(170, 180, 190, 150),
        width=1,
    )
    socket_pts = [map_plan(v, plot_rect, scale_m) for v in hex_vertices(socket_top_apothem, yaw=math.pi / 6.0)]
    socket_pts.append(socket_pts[0])
    draw_polyline(draw, socket_pts, (210, 220, 230, 150), width=2)

    if profile is not None and trail:
        start = np.asarray(trail[0], dtype=float)
        predicted = np.asarray(profile.predicted_center_xy_m, dtype=float)
        delta = predicted - start
        phi = math.atan2(float(delta[1]), float(delta[0]))
        ray_length = min(max(float(np.linalg.norm(delta)), 0.002), 0.0075)
        lo = phi - math.radians(profile.direction_half_width_deg)
        hi = phi + math.radians(profile.direction_half_width_deg)
        p_start = map_plan(start, plot_rect, scale_m)
        p_lo = map_plan(start + ray_length * np.array([math.cos(lo), math.sin(lo)]), plot_rect, scale_m)
        p_hi = map_plan(start + ray_length * np.array([math.cos(hi), math.sin(hi)]), plot_rect, scale_m)
        draw.polygon([p_start, p_lo, p_hi], fill=(255, 85, 210, 22))
        draw.line((p_start[0], p_start[1], p_lo[0], p_lo[1]), fill=(255, 105, 220, 120), width=1)
        draw.line((p_start[0], p_start[1], p_hi[0], p_hi[1]), fill=(255, 105, 220, 120), width=1)
        p_hat = map_plan(predicted, plot_rect, scale_m)
        draw.line((p_hat[0] - 5, p_hat[1], p_hat[0] + 5, p_hat[1]), fill=(255, 105, 220, 240), width=2)
        draw.line((p_hat[0], p_hat[1] - 5, p_hat[0], p_hat[1] + 5), fill=(255, 105, 220, 240), width=2)
        draw.text((p_hat[0] + 6, p_hat[1] - 12), "p1", fill=(255, 175, 230, 235), font=font)
        _, planned_xy = planned_outward_spiral(profile, start, samples=220)
        planned_pts = [map_plan(p, plot_rect, scale_m) for p in planned_xy]
        draw_polyline(draw, planned_pts, (255, 105, 220, 115), width=1)

    bit_pts = [map_plan(current_xy + v, plot_rect, scale_m) for v in hex_vertices(bit_apothem, yaw=math.pi / 6.0)]
    bit_pts.append(bit_pts[0])
    draw_polyline(draw, bit_pts, (55, 210, 255, 230), width=2)

    if len(trail) >= 2:
        trail_pts = [map_plan(p, plot_rect, scale_m) for p in trail[-180:]]
        draw_polyline(draw, trail_pts, (30, 230, 255, 180), width=2)
    p = map_plan(current_xy, plot_rect, scale_m)
    draw.line((p[0] - 5, p[1], p[0] + 5, p[1]), fill=(60, 240, 255, 245), width=2)
    draw.line((p[0], p[1] - 5, p[0], p[1] + 5), fill=(60, 240, 255, 245), width=2)

    for item in infos:
        q = map_plan(np.asarray(item["pos"], dtype=float)[:2], plot_rect, scale_m)
        draw.ellipse((q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3), fill=(255, 205, 0, 230), outline=(255, 245, 120, 240))


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = np.array([0.0022, 0.0, -0.0010])
    cam.distance = 0.038
    cam.azimuth = 128
    cam.elevation = -39
    return cam


def font_pair() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 18),
            ImageFont.truetype("DejaVuSans.ttf", 13),
            ImageFont.truetype("DejaVuSansMono.ttf", 12),
        )
    except OSError:
        f = ImageFont.load_default()
        return f, f, f


def draw_phase_gauge(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    phase: float | None,
    complete_fraction: float,
    font: ImageFont.ImageFont,
    title: str = "conical phase / data mask",
    detail: str = f"alpha={ALPHA_DEG:.0f} deg",
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0, 145), outline=(255, 255, 255, 65), width=1)
    draw.text((left + 10, top + 8), title, fill=(235, 245, 255, 235), font=font)
    cx = left + 58
    cy = top + 62
    r = 34
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(210, 220, 230, 120), width=2)
    if complete_fraction > 0.0:
        draw.arc(
            (cx - r, cy - r, cx + r, cy + r),
            start=-90,
            end=-90 + 360 * float(np.clip(complete_fraction, 0.0, 1.0)),
            fill=(35, 220, 255, 230),
            width=4,
        )
    if phase is not None:
        px = int(round(cx + r * math.cos(phase - math.pi / 2.0)))
        py = int(round(cy + r * math.sin(phase - math.pi / 2.0)))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 205, 0, 240))
    draw.text((left + 110, top + 48), detail, fill=(255, 235, 170, 240), font=font)
    draw.text((left + 110, top + 68), f"mask={complete_fraction:.2f}", fill=(235, 245, 255, 220), font=font)


def draw_spiral_belief_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    profile: SpiralProfile,
    start_xy: np.ndarray,
    progress: float,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0, 155), outline=(255, 255, 255, 65), width=1)
    predicted = np.asarray(profile.predicted_center_xy_m, dtype=float)
    delta = predicted - np.asarray(start_xy, dtype=float)
    direction_deg = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    start_radius = float(np.linalg.norm(delta))
    pitch_mm = mm(profile.spiral_pitch_m_per_rev)
    draw.text((left + 10, top + 8), "assumed posterior -> search schedule", fill=(235, 245, 255, 240), font=font)
    draw.text(
        (left + 10, top + 31),
        f"state={profile.search_state}  kappa={profile.confidence:.2f}",
        fill=(255, 235, 170, 240),
        font=font,
    )
    draw.text(
        (left + 10, top + 51),
        f"direction={direction_deg:+.0f} deg +/- {profile.direction_half_width_deg:.0f} deg",
        fill=(235, 245, 255, 230),
        font=font,
    )
    draw.text(
        (left + 10, top + 71),
        f"transport={mm(start_radius):.2f} mm  outward turns<={profile.turns:.2f}",
        fill=(235, 245, 255, 230),
        font=font,
    )
    draw.text(
        (left + 10, top + 91),
        f"nominal speed scale={profile.nominal_speed_scale:.2f}",
        fill=(235, 245, 255, 230),
        font=font,
    )
    bar_left = left + 10
    bar_right = right - 10
    bar_top = top + 116
    draw.rectangle((bar_left, bar_top, bar_right, bar_top + 8), outline=(255, 255, 255, 70), width=1)
    fill_right = bar_left + int((bar_right - bar_left) * float(np.clip(progress, 0.0, 1.0)))
    draw.rectangle((bar_left, bar_top, fill_right, bar_top + 8), fill=(45, 220, 255, 220))


def draw_direct_ft_gate_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    tip_z: float,
    force_sensor: np.ndarray,
    torque_sensor: np.ndarray,
    tags: set[str],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0, 155), outline=(255, 255, 255, 65), width=1)
    ft = float(np.linalg.norm(force_sensor[:2]))
    tau_xy_mnm = 1.0e3 * float(np.linalg.norm(torque_sensor[:2]))
    fz = abs(float(force_sensor[2]))
    bottom_signal = bool(
        abs(mm(tip_z) + mm(float(load_spec()["socket_depth_m"])))
        <= CONTROLLER_THRESHOLDS["bottom_depth_tolerance_mm"]
        and fz >= CONTROLLER_THRESHOLDS["bottom_fz_N"]
    )
    wall_signal = bool(
        mm(tip_z) <= CONTROLLER_THRESHOLDS["inner_wall_depth_mm"]
        and (
            ft >= CONTROLLER_THRESHOLDS["inner_wall_ft_N"]
            or tau_xy_mnm >= CONTROLLER_THRESHOLDS["inner_wall_tau_xy_mNm"]
        )
    )
    if bottom_signal:
        decision = "bottom reached -> complete"
        color = (105, 245, 150, 245)
    elif wall_signal:
        decision = "inner wall -> descend"
        color = (255, 220, 95, 245)
    else:
        decision = "guarded descent"
        color = (125, 220, 255, 245)
    draw.text((left + 10, top + 8), "direct F/T + depth gate", fill=(235, 245, 255, 240), font=font)
    draw.text((left + 10, top + 32), f"z={mm(tip_z):+.2f} mm", fill=(220, 230, 240, 230), font=font)
    draw.text((left + 10, top + 51), f"Ft={ft:.2f} N", fill=(220, 230, 240, 230), font=font)
    draw.text((left + 10, top + 70), f"tau_xy={tau_xy_mnm:.2f} mNm", fill=(220, 230, 240, 230), font=font)
    draw.text((left + 10, top + 94), decision, fill=color, font=font)
    draw.text(
        (left + 10, top + 114),
        f"wall: z<{CONTROLLER_THRESHOLDS['inner_wall_depth_mm']:.2f}mm, Ft>{CONTROLLER_THRESHOLDS['inner_wall_ft_N']:.2f}N",
        fill=(190, 205, 220, 210),
        font=font,
    )


def current_axis(spec: dict, roll: float, pitch: float, yaw: float = YAW) -> np.ndarray:
    return quat_to_matrix(euler_to_quat(roll, pitch, yaw)) @ np.array([0.0, 0.0, 1.0], dtype=float)


def wrench_reference_to_tip_m(spec: dict) -> float:
    value = spec.get("wrench_reference_to_tip_m")
    if value is None:
        return 0.18 * float(spec["bit_length_m"])
    distance = float(value)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("wrench_reference_to_tip_m must be finite and positive")
    return distance


def sensor_origin(spec: dict, xy: np.ndarray, tip_z: float, roll: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    rot = quat_to_matrix(euler_to_quat(roll, pitch, YAW))
    tip = np.array([float(xy[0]), float(xy[1]), float(tip_z)], dtype=float)
    return tip + rot @ np.array([0.0, 0.0, wrench_reference_to_tip_m(spec)], dtype=float), rot


def wrench_at_sensor(
    spec: dict,
    xy: np.ndarray,
    tip_z: float,
    roll: float,
    pitch: float,
    infos: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor, rot = sensor_origin(spec, xy, tip_z, roll, pitch)
    force_world = np.zeros(3, dtype=float)
    torque_world = np.zeros(3, dtype=float)
    for item in infos:
        force = np.asarray(item["force_world"], dtype=float)
        force_world += force
        torque_world += np.asarray(item["torque_world"], dtype=float)
        torque_world += np.cross(np.asarray(item["pos"], dtype=float) - sensor, force)
    return sensor, rot.T @ force_world, rot.T @ torque_world


def controller_wrench_gate(force_sensor: np.ndarray, torque_sensor: np.ndarray) -> tuple[float, bool, float, float]:
    force_norm = float(np.linalg.norm(force_sensor))
    tau_xy_mnm = 1.0e3 * float(np.linalg.norm(torque_sensor[:2]))
    force_scale = CONTROLLER_THRESHOLDS["soft_force_N"] / max(force_norm, 1.0e-9)
    torque_scale = CONTROLLER_THRESHOLDS["soft_tau_xy_mNm"] / max(tau_xy_mnm, 1.0e-9)
    speed_gate = float(np.clip(min(1.0, force_scale, torque_scale), 0.25, 1.0))
    hard_abort = bool(
        force_norm > CONTROLLER_THRESHOLDS["hard_force_N"]
        or tau_xy_mnm > CONTROLLER_THRESHOLDS["hard_tau_xy_mNm"]
    )
    return speed_gate, hard_abort, force_norm, tau_xy_mnm


def draw_axis_and_probe_cone(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    spec: dict,
    xy: np.ndarray,
    tip_z: float,
    roll: float,
    pitch: float,
    show_cone: bool,
) -> None:
    bit_len = float(spec["bit_length_m"])
    tip = np.array([float(xy[0]), float(xy[1]), float(tip_z)], dtype=float)
    axis = current_axis(spec, roll, pitch)
    sensor = tip + axis * wrench_reference_to_tip_m(spec)
    axis_end = tip + axis * (0.33 * bit_len)

    if show_cone:
        cone_pts: list[tuple[int, int]] = []
        cone_world: list[np.ndarray] = []
        for k in range(97):
            theta = 2.0 * math.pi * k / 96.0
            r, p = conical_pose(theta, 1.0)
            a = current_axis(spec, r, p)
            world = tip + a * (0.27 * bit_len)
            cone_world.append(world)
            point = project_world(world, camera)
            if point is not None:
                cone_pts.append(point)
        tip_px = project_world(tip, camera)
        if tip_px is not None:
            for k in range(0, 96, 4):
                p0 = project_world(cone_world[k], camera)
                p1 = project_world(cone_world[k + 4], camera)
                if p0 is not None and p1 is not None:
                    draw.polygon([tip_px, p0, p1], fill=(25, 190, 235, 9))
        draw_polyline(draw, cone_pts, (55, 225, 255, 205), width=2)
        for theta in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            r, p = conical_pose(theta, 1.0)
            a = current_axis(spec, r, p)
            p0 = project_world(tip, camera)
            p1 = project_world(tip + a * (0.27 * bit_len), camera)
            if p0 is not None and p1 is not None:
                draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(55, 225, 255, 70), width=1)

    p_tip = project_world(tip, camera)
    p_axis_end = project_world(axis_end, camera)
    p_sensor = project_world(sensor, camera)
    if p_tip is not None and p_axis_end is not None:
        _draw_arrow_2d(draw, p_tip, p_axis_end, (125, 245, 255, 245), width=3)
        draw.text((p_axis_end[0] + 6, p_axis_end[1] - 10), "axis", fill=(165, 250, 255, 245), font=font_pair()[1])
        draw.ellipse((p_tip[0] - 3, p_tip[1] - 3, p_tip[0] + 3, p_tip[1] + 3), fill=(110, 245, 255, 235))
    if p_sensor is not None:
        r = 7
        draw.ellipse(
            (p_sensor[0] - r, p_sensor[1] - r, p_sensor[0] + r, p_sensor[1] + r),
            outline=(255, 80, 190, 235),
            width=2,
        )
        draw.line((p_sensor[0] - 9, p_sensor[1], p_sensor[0] + 9, p_sensor[1]), fill=(255, 80, 190, 210), width=1)
        draw.line((p_sensor[0], p_sensor[1] - 9, p_sensor[0], p_sensor[1] + 9), fill=(255, 80, 190, 210), width=1)
        draw.text((p_sensor[0] + 9, p_sensor[1] - 9), "F/T", fill=(255, 185, 225, 230), font=font_pair()[1])


def draw_six_axis_wrench(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    sensor: np.ndarray,
    rot: np.ndarray,
    force_sensor: np.ndarray,
    torque_sensor: np.ndarray,
    font: ImageFont.ImageFont,
) -> None:
    axes = [rot[:, i].copy() for i in range(3)]
    force_names = ["Fx", "Fy", "Fz"]
    torque_names = ["tau_x", "tau_y", "tau_z"]
    for i, name in enumerate(force_names):
        value = float(force_sensor[i])
        if abs(value) < FORCE_DISPLAY_EPS_N:
            continue
        length = float(np.clip(0.0040 * value, -0.0060, 0.0060))
        p0 = project_world(sensor, camera)
        p1 = project_world(sensor + axes[i] * length, camera)
        if p0 is None or p1 is None:
            continue
        _draw_arrow_2d(draw, p0, p1, WRENCH_COLORS[name], width=4)
        _draw_label_2d(draw, (p1[0] + 5, p1[1] - 9), f"{name} {value:+.2f}", WRENCH_COLORS[name], font)

    for i, name in enumerate(torque_names):
        value = float(torque_sensor[i])
        if abs(value) < TORQUE_DISPLAY_EPS_NM:
            continue
        length = float(np.clip(3.0 * value, -0.0040, 0.0040))
        p0 = project_world(sensor, camera)
        p1 = project_world(sensor + axes[i] * length, camera)
        if p0 is not None and p1 is not None:
            _draw_arrow_2d(draw, p0, p1, WRENCH_COLORS[name], width=2)
        label = _draw_torque_arc_3d(
            draw,
            camera,
            sensor,
            axes[i],
            value,
            WRENCH_COLORS[name],
            0.00095 + 0.00030 * i,
        )
        if label is not None:
            _draw_label_2d(
                draw,
                (label[0] + 5, label[1] - 9),
                f"t{['x', 'y', 'z'][i]} {value * 1e3:+.2f}",
                WRENCH_COLORS[name],
                font,
            )


def pseudo_ft_channels(spec: dict, xy: np.ndarray, tip_z: float, roll: float, pitch: float, infos: list[dict]) -> tuple[float, float, float]:
    _, _, torque_sensor = wrench_at_sensor(spec, xy, tip_z, roll, pitch, infos)
    return float(tip_z), float(torque_sensor[0]), float(torque_sensor[1])


def normalize_channel(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1.0e-12:
        return [0.65 for _ in values]
    return [0.28 + 0.58 * float((v - lo) / (hi - lo)) for v in values]


def draw_polar_fingerprint_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    history: list[tuple[float, float, float, float]],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0, 145), outline=(255, 255, 255, 65), width=1)
    draw.text((left + 10, top + 7), "polar fingerprint channels", fill=(235, 245, 255, 235), font=font)
    labels = [("Ch1 Xz", 1), ("Ch2 Mx", 2), ("Ch3 My", 3)]
    phases = [h[0] for h in history]
    for i, (label, col) in enumerate(labels):
        if right - left >= 470:
            cx = left + 72 + i * 145
        else:
            channel_width = (right - left - 20) / 3.0
            cx = int(round(left + 10 + (i + 0.5) * channel_width))
        cy = top + 74
        rr = min(36, max(24, int(round((right - left - 32) / 10.0))))
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(210, 220, 230, 85), width=1)
        draw.line((cx - rr, cy, cx + rr, cy), fill=(210, 220, 230, 45), width=1)
        draw.line((cx, cy - rr, cx, cy + rr), fill=(210, 220, 230, 45), width=1)
        draw.text((cx - 34, top + 28), label, fill=(255, 235, 170, 235), font=font)
        if len(history) < 2:
            continue
        vals = normalize_channel([h[col] for h in history])
        pts: list[tuple[int, int]] = []
        for phase, val in zip(phases, vals):
            radius = rr * val
            x = int(round(cx + radius * math.cos(phase - math.pi / 2.0)))
            y = int(round(cy + radius * math.sin(phase - math.pi / 2.0)))
            pts.append((x, y))
        draw_polyline(draw, pts, (55, 210, 255, 220), width=2)


def draw_overlay(
    frame: np.ndarray,
    camera: dict,
    spec: dict,
    case: VideoCase,
    stage: str,
    frame_index: int,
    frame_count: int,
    xy: np.ndarray,
    tip_z: float,
    roll: float,
    pitch: float,
    phase: float | None,
    phase_fraction: float,
    infos: list[dict],
    trail: list[np.ndarray],
    polar_history: list[tuple[float, float, float, float]],
    result_note: str,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font_title, font, font_mono = font_pair()
    profile = SPIRAL_PROFILES.get(case.key)
    drawn_contacts = draw_contact_points(draw, camera, infos)
    sensor, force_sensor, torque_sensor = wrench_at_sensor(spec, xy, tip_z, roll, pitch, infos)
    _, sensor_rot = sensor_origin(spec, xy, tip_z, roll, pitch)
    draw_axis_and_probe_cone(
        draw,
        camera,
        spec,
        xy,
        tip_z,
        roll,
        pitch,
        show_cone=case.mode == "full",
    )
    draw_six_axis_wrench(draw, camera, sensor, sensor_rot, force_sensor, torque_sensor, font)
    tag_set = contact_tags(infos)
    tags = ",".join(sorted(tag_set)) if infos else "none"

    draw.rounded_rectangle((10, 10, 454, 212), radius=7, fill=(0, 0, 0, 155), outline=(255, 255, 255, 70), width=1)
    if case.mode == "spiral":
        assert profile is not None
        lines = [
            f"{case.key} | {case.offline_label}",
            f"mode: transport + outward Archimedean | frame {frame_index + 1}/{frame_count}",
            f"state: {profile.search_state} -> rim/chamfer",
            f"tip_z={mm(tip_z):+.3f} mm  xy=({mm(xy[0]):+.2f},{mm(xy[1]):+.2f}) mm",
            f"F=({force_sensor[0]:+.2f},{force_sensor[1]:+.2f},{force_sensor[2]:+.2f}) N  tau_xy={1e3 * np.linalg.norm(torque_sensor[:2]):.2f} mNm",
            f"contacts={len(infos)} drawn={drawn_contacts} tags={tags}",
            result_note,
        ]
    else:
        lines = [
            f"{case.key} | {case.offline_label}",
            f"protocol: {case.protocol}",
            f"stage: {stage} | frame {frame_index + 1}/{frame_count}",
            f"result: {case.result}",
            f"tip_z={mm(tip_z):+.3f} mm  xy=({mm(xy[0]):+.2f},{mm(xy[1]):+.2f}) mm",
            f"roll={math.degrees(roll):+.1f} deg pitch={math.degrees(pitch):+.1f} deg",
            f"F_s=({force_sensor[0]:+.2f},{force_sensor[1]:+.2f},{force_sensor[2]:+.2f}) N",
            f"tau_xy={1e3 * np.linalg.norm(torque_sensor[:2]):.2f} mNm",
            f"MuJoCo contacts={len(infos)} drawn={drawn_contacts} tags={tags}",
            result_note,
        ]
    max_panel_text_width = 420
    title_font = font_title
    if draw.textlength(lines[0], font=title_font) > max_panel_text_width:
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        except OSError:
            title_font = font
    title = fit_text_to_width(draw, lines[0], title_font, max_panel_text_width)
    draw.text((22, 20), title, fill=(255, 235, 170, 245), font=title_font)
    y = 48
    for line in lines[1:]:
        line = fit_text_to_width(draw, line, font_mono, max_panel_text_width)
        color = (245, 245, 245, 232)
        if "result:" in line:
            color = (255, 235, 170, 245)
        if "contacts=0" in line or "abort" in line:
            color = (255, 175, 145, 245)
        draw.text((22, y), line, fill=color, font=font_mono)
        y += 17

    formula_rect = (14, 596, 560, 704)
    draw.rounded_rectangle(formula_rect, radius=7, fill=(0, 0, 0, 145), outline=(255, 255, 255, 65), width=1)
    if case.mode == "spiral":
        assert profile is not None
        pitch_mm = mm(profile.spiral_pitch_m_per_rev)
        formula = [
            f"posterior transport -> outward r=b*theta, pitch={pitch_mm:.2f} mm/rev",
            f"assumed posterior: {profile.search_state}, kappa={profile.confidence:.2f}, angular set +/-{profile.direction_half_width_deg:.0f} deg",
            f"dynamic admittance: Fd={CONTROLLER_THRESHOLDS['target_down_force_N']:.1f}N; pitch <= 2*r_eff",
            f"capture dz>={CONTROLLER_THRESHOLDS['chamfer_depth_drop_mm']:.2f}mm; abort F>{CONTROLLER_THRESHOLDS['hard_force_N']:.0f}N/tau>{CONTROLLER_THRESHOLDS['hard_tau_xy_mNm']:.0f}mNm",
        ]
    elif case.mode == "capture":
        formula = [
            "rim/chamfer is outside the flat-surface CNN sampling set",
            "hold upright; low down-force lets the chamfer provide radial guidance",
            "below-rim depth + lateral wrench -> inner-wall capture gate",
            "no 15 deg conical scan after capture is observable",
        ]
    elif case.mode == "terminal":
        formula = [
            "inner-wall / inserted are white-box terminal states",
            "depth + sustained lateral wrench -> descend trigger",
            "hold upright; wall-guided XY settles while Z descends",
            f"CAD socket bottom depth = {mm(float(spec['socket_depth_m'])):.1f} mm",
        ]
    else:
        formula = [
            "nominal conical probe template",
            "roll = alpha sin(theta), pitch = alpha cos(theta), alpha=15 deg",
            "nominal tip-center XY fixed; actual contact deflection is measured",
            "yellow dots are MuJoCo data.contact.pos, not hand-drawn contact",
        ]
    yy = formula_rect[1] + 10
    for line in formula:
        draw.text((formula_rect[0] + 10, yy), line, fill=(235, 245, 255, 230), font=font)
        yy += 21

    if case.mode == "spiral":
        assert profile is not None
        draw_spiral_belief_panel(
            draw,
            (462, 10, 946, 145),
            profile,
            np.asarray(trail[0] if trail else [case.x, case.y], dtype=float),
            phase_fraction,
            font,
        )
    elif case.mode in {"terminal", "capture"}:
        draw_direct_ft_gate_panel(draw, (462, 10, 820, 153), tip_z, force_sensor, torque_sensor, tag_set, font)
    else:
        draw_polar_fingerprint_panel(draw, (462, 10, 946, 132), polar_history, font)
    draw_plan_panel(draw, spec, xy, trail, infos, font, "XY plan | debug contacts", profile=profile)
    return np.asarray(image)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    scene_option: mujoco.MjvOption,
    spec: dict,
    case: VideoCase,
    stage: str,
    idx: int,
    n_frames: int,
    xy: np.ndarray,
    tip_z: float,
    roll: float,
    pitch: float,
    phase: float | None,
    phase_fraction: float,
    infos: list[dict],
    trail: list[np.ndarray],
    polar_history: list[tuple[float, float, float, float]],
    result_note: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    camera = camera_snapshot(renderer.scene, WIDTH, HEIGHT)
    frame = renderer.render()
    frame = np.clip(frame.astype(np.float32) * 1.13 + 7.0, 0.0, 255.0).astype(np.uint8)
    return draw_overlay(
        frame,
        camera,
        spec,
        case,
        stage,
        idx,
        n_frames,
        xy,
        tip_z,
        roll,
        pitch,
        phase,
        phase_fraction,
        infos,
        trail,
        polar_history,
        result_note,
    )


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def conical_pose(theta: float, alpha_scale: float = 1.0) -> tuple[float, float]:
    a = ALPHA * alpha_scale
    return a * math.sin(theta), a * math.cos(theta)


def build_cases() -> list[VideoCase]:
    spec = load_spec()
    bit_r = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_r = float(spec["screw_head_diameter_m"]) / 2.0
    c_top = float(spec["clearance_m"]) + float(spec["chamfer_width_m"])
    return [
        VideoCase(
            "V1",
            "m8_state_01_outer_partial_conical15.mp4",
            "outer partial support",
            "nominal alpha=15 deg conical scan",
            "full scan + return upright verify",
            head_r + 0.20 * bit_r,
            0.0,
            0.00010,
            "full",
        ),
        VideoCase(
            "V2",
            "m8_state_02_double_overhang_conical15.mp4",
            "double-overhang / annular bridge",
            "nominal alpha=15 deg conical scan",
            "full scan + return upright verify",
            0.5 * (head_r - bit_r + head_r),
            0.0,
            0.00010,
            "full",
        ),
        VideoCase(
            "V3",
            "m8_state_03_supported_search_conical15.mp4",
            "supported search",
            "nominal alpha=15 deg conical scan",
            "full scan + return upright verify",
            min(head_r - bit_r, 0.0026),
            0.0,
            0.00010,
            "full",
        ),
        VideoCase(
            "V4",
            "m8_state_04_rim_chamfer_to_inner_wall.mp4",
            "rim / chamfer -> inner-wall capture",
            "upright compliant capture with direct F/T + depth gate",
            "no conical fingerprint; chamfer guidance transitions to inner-wall state",
            c_top * 1.2,
            0.0,
            -0.00004,
            "capture",
        ),
        VideoCase(
            "V5",
            "m8_state_05_inner_wall_to_inserted_bottom.mp4",
            "inner-wall -> inserted / bottom",
            "direct F/T + depth trigger, then compliant descent",
            "no conical sample / no CNN; bottom contact completes insertion",
            0.000200,
            0.0,
            -0.00075,
            "terminal",
        ),
        VideoCase(
            "V6a",
            "m8_spiral_outer_partial_to_rim_chamfer.mp4",
            "outer partial -> rim/chamfer",
            "posterior-guided transport + local outward Archimedean spiral",
            "outer partial -> near-center rim/chamfer capture",
            0.00510,
            0.0,
            0.00010,
            "spiral",
        ),
        VideoCase(
            "V6b",
            "m8_spiral_double_overhang_to_rim_chamfer.mp4",
            "double-overhang -> rim/chamfer",
            "posterior-guided transport + dense local outward spiral",
            "double-overhang -> near-center rim/chamfer capture",
            0.5 * (head_r - bit_r + head_r),
            0.0,
            0.00010,
            "spiral",
        ),
        VideoCase(
            "V6c",
            "m8_spiral_supported_search_to_rim_chamfer.mp4",
            "supported search -> rim/chamfer",
            "posterior-guided transport + short local outward spiral",
            "supported search -> near-center rim/chamfer capture",
            min(head_r - bit_r, 0.0026),
            0.0,
            0.00010,
            "spiral",
        ),
    ]


def render_video(case: VideoCase) -> dict:
    spec = load_spec()
    model = mujoco.MjModel.from_xml_string(render_model_xml(spec))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    color_scene(model)
    scene_option = mujoco.MjvOption()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = make_camera(model)
    xy0 = np.array([case.x, case.y], dtype=float)
    trail: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    contact_counts: list[int] = []
    z_values: list[float] = []
    polar_history: list[tuple[float, float, float, float]] = []
    event = case.result
    spiral_runtime: dict | None = None

    if case.mode == "abort":
        n_frames = 58
        for idx in range(n_frames):
            t = idx / max(n_frames - 1, 1)
            z = 0.0012 * (1.0 - smoothstep(min(t / 0.58, 1.0))) + case.nominal_tip_z
            set_bit_pose(model, data, xy0, z)
            infos = valid_contact_infos(model, data)
            trail.append(xy0.copy())
            note = "no valid screw/socket contact -> conical protocol not started"
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    "touchdown pre-check / abort",
                    idx,
                    n_frames,
                    xy0,
                    z,
                    0.0,
                    0.0,
                    None,
                    0.0,
                    infos,
                    trail,
                    polar_history,
                    note,
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)

    elif case.mode in {"full", "early"}:
        base_z, _ = find_touchdown_z(model, data, xy0, case.nominal_tip_z)
        scan_steps = 60 if case.mode == "full" else 42
        n_frames = 14 + scan_steps + 14
        early_stop_at: int | None = None
        idx = 0
        for ramp_i in range(14):
            frac = smoothstep((ramp_i + 1) / 14.0)
            roll, pitch = conical_pose(0.0, frac)
            z, infos = find_touchdown_z(model, data, xy0, base_z, roll=roll, pitch=pitch)
            set_bit_pose(model, data, xy0, z, roll=roll, pitch=pitch)
            trail.append(xy0.copy())
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    "ramp into fixed 15 deg cone",
                    idx,
                    n_frames,
                    xy0,
                    z,
                    roll,
                    pitch,
                    0.0,
                    0.0,
                    infos,
                    trail,
                    polar_history,
                    "same nominal template; contact response is state-dependent",
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)
            idx += 1
        for scan_i in range(scan_steps):
            theta = 2.0 * math.pi * scan_i / max(scan_steps - 1, 1)
            roll, pitch = conical_pose(theta, 1.0)
            z, infos = find_touchdown_z(model, data, xy0, base_z, roll=roll, pitch=pitch)
            set_bit_pose(model, data, xy0, z, roll=roll, pitch=pitch)
            polar_history.append((theta, *pseudo_ft_channels(spec, xy0, z, roll, pitch, infos)))
            tags = contact_tags(infos)
            if case.mode == "early" and early_stop_at is None and scan_i >= 10 and (
                "chamfer" in tags or "wall" in tags or z < -0.00010
            ):
                early_stop_at = scan_i
            phase_fraction = scan_i / max(scan_steps - 1, 1)
            note = "full 0..2pi scan" if case.mode == "full" else "early-stop gate watches chamfer/depth cue"
            if early_stop_at is not None:
                note = "chamfer/depth cue detected -> return upright, do not keep sweeping"
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    "conical scan" if early_stop_at is None else "early-stop cue",
                    idx,
                    n_frames,
                    xy0,
                    z,
                    roll,
                    pitch,
                    theta,
                    phase_fraction,
                    infos,
                    trail,
                    polar_history,
                    note,
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)
            idx += 1
            if early_stop_at is not None:
                break
        return_steps = 14
        current_theta = 2.0 * math.pi if early_stop_at is None else 2.0 * math.pi * float(early_stop_at) / max(scan_steps - 1, 1)
        for ret_i in range(return_steps):
            frac = 1.0 - smoothstep((ret_i + 1) / return_steps)
            roll, pitch = conical_pose(current_theta, frac)
            z, infos = find_touchdown_z(model, data, xy0, base_z, roll=roll, pitch=pitch)
            set_bit_pose(model, data, xy0, z, roll=roll, pitch=pitch)
            note = "return commanded roll/pitch to zero, then verify"
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    "return upright / verify",
                    idx,
                    n_frames,
                    xy0,
                    z,
                    roll,
                    pitch,
                    current_theta,
                    1.0 if early_stop_at is None else max(float(early_stop_at) / max(scan_steps - 1, 1), 0.0),
                    infos,
                    trail,
                    polar_history,
                    note,
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)
            idx += 1
        if early_stop_at is not None:
            event = f"early_stop_at_phase_{early_stop_at / max(scan_steps - 1, 1):.2f}"

    elif case.mode == "terminal":
        gate_steps = 18
        descent_steps = 58
        hold_steps = 14
        n_frames = gate_steps + descent_steps + hold_steps
        socket_bottom_z = -float(spec["socket_depth_m"]) - 0.000025
        for idx in range(n_frames):
            if idx < gate_steps:
                xy = xy0.copy()
                z = case.nominal_tip_z
                stage = "direct inner-wall F/T + depth gate"
                note = "wall contact + sub-rim depth + sustained wrench -> descend; no CNN/probe"
                fraction = 0.0
            elif idx < gate_steps + descent_steps:
                t = (idx - gate_steps) / max(descent_steps - 1, 1)
                center_fraction = smoothstep(min(t / 0.42, 1.0))
                xy = xy0 * (1.0 - center_fraction)
                z = case.nominal_tip_z + (socket_bottom_z - case.nominal_tip_z) * smoothstep(t)
                stage = "wall-guided compliant descent" if center_fraction < 1.0 else "upright insertion"
                note = "keep roll/pitch zero; wall reaction settles XY while guarded Z descends"
                fraction = t
            else:
                xy = np.zeros(2, dtype=float)
                z = socket_bottom_z
                stage = "bottom contact / complete"
                note = "socket-bottom contact reached at the CAD depth target"
                fraction = 1.0
            set_bit_pose(model, data, xy, z)
            infos = valid_contact_infos(model, data)
            trail.append(xy.copy())
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    stage,
                    idx,
                    n_frames,
                    xy,
                    z,
                    0.0,
                    0.0,
                    None,
                    fraction,
                    infos,
                    trail,
                    polar_history,
                    note,
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)
        event = "direct_inner_wall_gate_then_socket_bottom_complete"

    elif case.mode == "spiral":
        profile = SPIRAL_PROFILES[case.key]
        settle_steps = 14
        start = xy0.copy()
        predicted_center = np.asarray(profile.predicted_center_xy_m, dtype=float)
        start_vector = start - predicted_center
        start_radius = float(np.linalg.norm(start_vector))
        start_angle = math.atan2(float(start_vector[1]), float(start_vector[0]))
        radial_pitch_m = (start_radius - profile.terminal_radius_m) / max(profile.turns, 1.0e-9)
        capture_xy = start.copy()
        capture_z = case.nominal_tip_z
        depth_reference: float | None = None
        capture_streak = 0
        capture_progress = 1.0
        max_speed_gate = 1.0
        min_speed_gate = 1.0
        trajectory: list[tuple[np.ndarray, float, float, float, str, str]] = []
        progress = 0.0
        base_increment = profile.nominal_speed_scale / max(profile.nominal_steps - 1, 1)
        max_iterations = int(6 * profile.nominal_steps / max(profile.nominal_speed_scale, 0.25))

        for _ in range(max_iterations):
            phase = start_angle + 2.0 * math.pi * profile.turns * progress
            radius = start_radius - radial_pitch_m * profile.turns * progress
            xy = predicted_center + radius * np.array([math.cos(phase), math.sin(phase)], dtype=float)
            z_hint = 0.00010 * (1.0 - progress) - 0.00008 * progress
            z, infos = find_touchdown_z(model, data, xy, z_hint, max_drop=0.0012)
            set_bit_pose(model, data, xy, z)
            _, force_sensor, torque_sensor = wrench_at_sensor(spec, xy, z, 0.0, 0.0, infos)
            speed_gate, hard_abort, force_norm, tau_xy_mnm = controller_wrench_gate(force_sensor, torque_sensor)
            min_speed_gate = min(min_speed_gate, speed_gate)
            max_speed_gate = max(max_speed_gate, speed_gate)
            if depth_reference is None:
                depth_reference = z
            depth_drop_mm = mm(float(depth_reference - z))
            capture_signal = bool(
                progress >= 0.45
                and depth_drop_mm >= CONTROLLER_THRESHOLDS["chamfer_depth_drop_mm"]
                and not hard_abort
            )
            capture_streak = capture_streak + 1 if capture_signal else 0
            note = (
                f"speed_gate={speed_gate:.2f} |F|={force_norm:.1f}N "
                f"tau_xy={tau_xy_mnm:.1f}mNm dz={depth_drop_mm:.2f}mm"
            )
            stage = "confidence-modulated biased Archimedean spiral"
            trajectory.append((xy.copy(), z, phase, progress, stage, note))
            capture_xy = xy.copy()
            capture_z = z
            if hard_abort:
                event = "hard_wrench_abort"
                break
            if capture_streak >= 3:
                capture_progress = progress
                event = "chamfer_depth_cue_latched"
                break
            if progress >= 1.0 - 1.0e-9:
                if capture_signal:
                    # Hold the terminal belief point long enough for the same
                    # dwell rule used during the path to latch the cue.
                    continue
                event = "rim_chamfer_capture_at_terminal_belief"
                break
            progress = min(1.0, progress + base_increment * speed_gate)

        if event != "hard_wrench_abort":
            for settle_i in range(settle_steps):
                t = (settle_i + 1) / settle_steps
                z = capture_z - 0.00010 * smoothstep(t)
                trajectory.append(
                    (
                        capture_xy.copy(),
                        z,
                        start_angle + 2.0 * math.pi * profile.turns * max(capture_progress, progress),
                        max(capture_progress, progress),
                        "rim/chamfer capture settle",
                        "capture cue latched -> stop XY search; guarded down-force settles on chamfer",
                    )
                )

        n_frames = len(trajectory)
        for idx, (xy, z, phase, fraction, stage, note) in enumerate(trajectory):
            set_bit_pose(model, data, xy, z)
            infos = valid_contact_infos(model, data)
            trail.append(xy.copy())
            frames.append(
                render_frame(
                    model,
                    data,
                    renderer,
                    cam,
                    scene_option,
                    spec,
                    case,
                    stage,
                    idx,
                    n_frames,
                    xy,
                    z,
                    0.0,
                    0.0,
                    phase,
                    fraction,
                    infos,
                    trail,
                    polar_history,
                    note,
                )
            )
            contact_counts.append(len(infos))
            z_values.append(z)

        spiral_runtime = {
            "search_state": profile.search_state,
            "confidence": profile.confidence,
            "direction_half_width_deg": profile.direction_half_width_deg,
            "turns": profile.turns,
            "radial_pitch_mm_per_rev": mm(radial_pitch_m),
            "nominal_speed_scale": profile.nominal_speed_scale,
            "min_wrench_speed_gate": min_speed_gate,
            "max_wrench_speed_gate": max_speed_gate,
            "capture_progress": capture_progress,
            "predicted_center_mm": [mm(v) for v in profile.predicted_center_xy_m],
            "terminal_xy_mm": [mm(float(capture_xy[0])), mm(float(capture_xy[1]))],
        }
    else:
        raise ValueError(f"Unknown mode {case.mode}")

    renderer.close()
    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_OUT / case.filename
    imageio.mimsave(out_path, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    record = {
        "key": case.key,
        "path": str(out_path.relative_to(ROOT)),
        "offline_label": case.offline_label,
        "protocol": case.protocol,
        "result": event,
        "frames": len(frames),
        "fps": FPS,
        "max_contact_count": int(max(contact_counts) if contact_counts else 0),
        "min_contact_count": int(min(contact_counts) if contact_counts else 0),
        "min_tip_z_m": float(min(z_values) if z_values else case.nominal_tip_z),
        "max_tip_z_m": float(max(z_values) if z_values else case.nominal_tip_z),
    }
    if spiral_runtime is not None:
        record["spiral_runtime"] = spiral_runtime
    return record


def legacy_quasistatic_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    results = [render_video(case) for case in cases]
    expected_video_names = {case.filename for case in cases}
    for path in VIDEO_OUT.glob("*.mp4"):
        if path.name not in expected_video_names:
            path.unlink()
    manifest = {
        "claim_boundary": (
            "Videos are quasi-static commanded MuJoCo trajectories. Contact dots are real "
            "MuJoCo data.contact positions and the displayed six-axis wrench is aggregated "
            "from mj_contactForce for the current pose. The script does not claim a real "
            "robot admittance controller has already been implemented."
        ),
        "visual_layer_boundary": (
            "The visible top face is a clean render-only mesh injected into a temporary XML "
            "with contype=0 and conaffinity=0. The hidden segmented head/chamfer/wall geoms "
            "remain the active collision sources for MuJoCo contacts."
        ),
        "diagnostic_overlay": {
            "force_arrows": "Fx/Fy/Fz at the virtual sensor origin",
            "torque_arcs": "tau_x/tau_y/tau_z using the right-hand rule",
            "warning": "the wrench is a contact reaction, not a direct arrow to the hole center",
            "conical_funnel": "V1-V4 only; current peg axis plus the nominal 15 deg swept cone",
        },
        "nominal_conical_template": {
            "alpha_deg": ALPHA_DEG,
            "roll": "alpha * sin(theta)",
            "pitch": "alpha * cos(theta)",
            "theta": "0..2pi for full scans",
        },
        "learning_output_contract": {
            "search_state": "outer_partial / double_overhang / supported_search",
            "direction_distribution": "calibrated circular sector probabilities q(phi)",
            "direction_set": "finite-coverage angular set Phi_(1-alpha)",
            "confidence": "kappa modulates radial drift, turn count, and nominal speed",
            "radial_band": "derived primarily from CAD + contact state, not unconstrained point regression",
            "unsafe_score": "hard controller thresholds always override learning",
        },
        "controller_thresholds_m8_sim_initial": CONTROLLER_THRESHOLDS,
        "spiral_profiles": {
            key: {
                "search_state": value.search_state,
                "confidence": value.confidence,
                "direction_half_width_deg": value.direction_half_width_deg,
                "turns": value.turns,
                "predicted_center_mm": [mm(v) for v in value.predicted_center_xy_m],
                "terminal_radius_mm": mm(value.terminal_radius_m),
                "nominal_speed_scale": value.nominal_speed_scale,
                "belief_note": value.belief_note,
            }
            for key, value in SPIRAL_PROFILES.items()
        },
        "corrected_transition_parameters": {
            "V5_inner_wall_to_inserted": {
                "recognition": "direct depth + sustained lateral wrench + contact-geometry consistency",
                "uses_conical_probe": False,
                "uses_learning": False,
                "start_tip_z_mm": -0.75,
                "cad_socket_depth_mm": mm(float(load_spec()["socket_depth_m"])),
                "terminal_tip_z_mm": -4.025,
                "action": "wall-guided compliant descent; stop on socket-bottom contact",
            },
        },
        "videos": results,
    }
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "m8_sampling_video_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Regenerate the authoritative dynamic video set.

    The quasi-static implementation remains above for traceability and shared
    rendering helpers, but it must not overwrite current evidence.
    """

    from src.render_m8_dynamic_videos import main as dynamic_main

    dynamic_main()


if __name__ == "__main__":
    main()
