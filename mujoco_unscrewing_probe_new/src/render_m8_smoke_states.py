from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "current"
FIGURE_OUT = OUT / "media" / "figures"
REPORT_OUT = OUT / "reports"
MODEL_PATH = ROOT / "models" / "allen_m8_source.xml"
CONFIG_PATH = ROOT / "configs" / "allen_m8_source.json"


def mm(value_m: float) -> float:
    return value_m * 1000.0


def load_spec() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def set_bit_pose(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, tip_z: float, yaw: float = 0.0) -> None:
    bit_len = float(load_spec()["bit_length_m"])
    body_z = tip_z + bit_len / 2.0
    data.qpos[:3] = np.array([x, y, body_z], dtype=float)
    data.qpos[3:7] = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def color_scene(model: mujoco.MjModel, mode: str) -> None:
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("head_top_sector"):
            model.geom_rgba[gid] = np.array([0.70, 0.73, 0.67, 1.0])
        elif name.startswith("socket_chamfer"):
            model.geom_rgba[gid] = np.array([0.62, 0.66, 0.60, 0.0])
        elif name.startswith("socket_wall"):
            model.geom_rgba[gid] = np.array([0.10, 0.12, 0.16, 0.0])
        elif name == "socket_bottom":
            model.geom_rgba[gid] = np.array([0.08, 0.09, 0.11, 1.0])
        elif name == "bit_collision":
            model.geom_rgba[gid] = np.array([0.02, 0.20, 0.52, 0.74])
        elif name == "screw_head_side_visual":
            model.geom_rgba[gid] = np.array([0.44, 0.45, 0.42, 0.30])
    if mode == "partial_support":
        highlight = np.array([0.86, 0.78, 0.62, 1.0])
    elif mode == "double_overhang":
        highlight = np.array([0.78, 0.70, 0.92, 1.0])
    elif mode == "supported_search":
        highlight = np.array([0.74, 0.86, 0.76, 1.0])
    else:
        highlight = np.array([0.95, 0.82, 0.28, 1.0])
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if mode == "rim_chamfer" and name.startswith("socket_chamfer"):
            model.geom_rgba[gid] = np.array([highlight[0], highlight[1], highlight[2], 0.0])
        if mode == "double_overhang" and name.startswith("head_top_sector"):
            model.geom_rgba[gid] = highlight
        if mode == "supported_search" and name.startswith("head_top_sector"):
            model.geom_rgba[gid] = highlight
        if mode == "partial_support" and name.startswith("head_top_sector"):
            model.geom_rgba[gid] = highlight


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


def collect_mujoco_contact_positions(model: mujoco.MjModel, data: mujoco.MjData) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        if "bit_collision" not in (geom1, geom2):
            continue
        if geom1 == "workpiece_base" or geom2 == "workpiece_base":
            continue
        points.append(np.asarray(contact.pos, dtype=float).copy())
    return points


def draw_mujoco_contact_points(draw: ImageDraw.ImageDraw, camera: dict, contact_points: list[np.ndarray]) -> int:
    drawn = 0
    for point in contact_points:
        p = project_world(point, camera)
        if p is None:
            continue
        r = 3
        draw.ellipse(
            (p[0] - r, p[1] - r, p[0] + r, p[1] + r),
            fill=(255, 205, 0, 230),
            outline=(255, 244, 120, 245),
            width=1,
        )
        drawn += 1
    return drawn


def find_first_contact_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    tip_z: float,
    yaw: float,
) -> tuple[float, list[np.ndarray]]:
    z_values = [tip_z - i * 0.000025 for i in range(25)]
    last_points: list[np.ndarray] = []
    last_z = tip_z
    for z in z_values:
        set_bit_pose(model, data, x, y, z, yaw=yaw)
        points = collect_mujoco_contact_positions(model, data)
        if points:
            return z, points
        last_z = z
        last_points = points
    return last_z, last_points


def draw_screen_overlay(
    frame: np.ndarray,
    camera: dict,
    title: str,
    note: str,
    contact_points: list[np.ndarray],
    actual_tip_z: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_note = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font_title = ImageFont.load_default()
        font_note = ImageFont.load_default()

    drawn_contacts = draw_mujoco_contact_points(draw, camera, contact_points)

    draw.rounded_rectangle((14, 374, 430, 462), radius=6, fill=(0, 0, 0, 150), outline=(255, 255, 255, 55))
    draw.text((26, 384), title, fill=(255, 235, 170, 245), font=font_title)
    draw.text((26, 414), note, fill=(235, 245, 255, 232), font=font_note)
    draw.text(
        (26, 438),
        f"yellow dots: MuJoCo contact.pos x {drawn_contacts} | tip_z={mm(actual_tip_z):.3f} mm",
        fill=(235, 245, 255, 212),
        font=font_note,
    )
    return np.asarray(image)


def render_state(name: str, x: float, y: float, tip_z: float, title: str, note: str) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    model.site_rgba[:, 3] = 0.0
    color_scene(model, name)
    yaw = math.radians(12.0)
    actual_tip_z, contact_points = find_first_contact_pose(model, data, x, y, tip_z, yaw=yaw)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = np.array([0.0018, 0.0, -0.0008])
    cam.distance = 0.038
    cam.azimuth = 128
    cam.elevation = -42
    scene_option = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    camera = camera_snapshot(renderer.scene, width=640, height=480)
    frame = renderer.render()
    renderer.close()
    frame = np.clip(frame.astype(np.float32) * 1.18 + 10.0, 0.0, 255.0).astype(np.uint8)
    frame = draw_screen_overlay(frame, camera, title, note, contact_points, actual_tip_z)

    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    out = FIGURE_OUT / f"m8_smoke_{name}.png"
    imageio.imwrite(out, frame)
    return {
        "state": name,
        "path": str(out.relative_to(ROOT)),
        "requested_tip_z_m": tip_z,
        "actual_tip_z_m": actual_tip_z,
        "mujoco_contact_count": len(contact_points),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    bit_r = spec["key_af_m"] / math.sqrt(3.0)
    head_r = spec["screw_head_diameter_m"] / 2.0
    c_top = spec["clearance_m"] + spec["chamfer_width_m"]

    cases = [
        (
            "partial_support",
            head_r * 0.78,
            0.0,
            0.00015,
            "M8 outer partial support",
            f"outer side partly unsupported | offset={mm(head_r * 0.78):.1f} mm",
        ),
        (
            "double_overhang",
            0.5 * (head_r - bit_r + head_r),
            0.0,
            0.00015,
            "M8 double overhang / bridge",
            "outer side and socket side both show open-side cues",
        ),
        (
            "supported_search",
            min(head_r - bit_r, 0.0026),
            0.0,
            0.00015,
            "M8 supported search",
            "bit fully on screw-head top; press/spiral/probe is meaningful",
        ),
        (
            "rim_chamfer",
            c_top * 1.2,
            0.0,
            -0.00005,
            "M8 rim / chamfer cue",
            f"tip near socket chamfer; capture radius about {mm(c_top):.2f} mm",
        ),
        (
            "inner_wall_capture",
            spec["clearance_m"] * 0.55,
            0.0,
            -0.00075,
            "M8 inner-wall capture",
            "tip is constrained by inner wall; verify / compliant insert, not full cone",
        ),
    ]
    renders = [render_state(*case) for case in cases]
    summary = {
        "generated": [item["path"] for item in renders],
        "renders": renders,
        "note": "Smoke renders use MuJoCo data.contact positions. Yellow dots are projected contact.pos samples for contacts involving bit_collision. No marker geoms were added to the source XML.",
    }
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "m8_smoke_state_renders.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
