from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs" / "current"
MODEL_PATH = ROOT / "models" / "allen_m8_source.xml"
OUTPUT_PATH = CURRENT / "media" / "figures" / "m8_all_conical_sampling_positions_mujoco.png"
INVENTORY_PATH = CURRENT / "reports" / "m8_all_conical_sampling_positions.json"

CANVAS_WIDTH = 1900
CANVAS_HEIGHT = 1200
RENDER_SIZE = 1100
RENDER_X = 50
RENDER_Y = 72


@dataclass(frozen=True)
class DatasetStyle:
    key: str
    label: str
    role: str
    color: tuple[int, int, int]
    marker: str
    marker_radius: int


DATASETS = (
    DatasetStyle("p0a_m8_symmetry", "P0a symmetry", "geometry diagnostic", (45, 212, 191), "circle", 3),
    DatasetStyle("p0b_m8_observability", "P0b observability", "grouped foundation pool", (56, 189, 248), "circle", 4),
    DatasetStyle("p0c_m8_collision_followup", "P0c hard cases", "development follow-up", (34, 211, 238), "diamond", 6),
    DatasetStyle("p0d_m8_sector_anchors", "P0d sector anchors", "physical direction anchors", (74, 222, 128), "square", 4),
    DatasetStyle("p1v2_m8_independent_direction_audit", "P1v2 design audit", "viewed failure audit", (251, 146, 60), "square", 6),
    DatasetStyle("p1v3_m8_targeted_direction_training", "P1v3 targeted", "train / calibration", (244, 114, 182), "triangle", 6),
    DatasetStyle("p1v4_m8_untouched_direction_audit", "P1v4 interface audit", "development only", (167, 139, 250), "cross", 7),
    DatasetStyle("p1v5_m8_final_untouched_direction_audit", "P1v5 final untouched", "final test; no reuse", (250, 204, 21), "star", 8),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        bold_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        if bold_path.exists():
            return ImageFont.truetype(str(bold_path), size=size)
    regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return ImageFont.truetype(str(regular), size=size)


def _position_key(record: dict) -> tuple[int, int]:
    return round(float(record["x_m"]) * 1.0e9), round(float(record["y_m"]) * 1.0e9)


def load_sampling_inventory() -> tuple[list[dict], dict]:
    layers: list[dict] = []
    all_positions: set[tuple[int, int]] = set()
    total_rollouts = 0

    for style in DATASETS:
        manifest_path = CURRENT / style.key / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            record
            for record in manifest["records"]
            if str(record.get("status", "ok")).lower() not in {"failed", "error"}
        ]
        by_position: dict[tuple[int, int], dict] = {}
        for record in records:
            by_position.setdefault(_position_key(record), record)
        positions = [
            {
                "x_m": float(record["x_m"]),
                "y_m": float(record["y_m"]),
                "radius_m": float(record["radius_m"]),
                "search_state": str(record["search_state"]),
            }
            for record in by_position.values()
        ]
        all_positions.update(by_position)
        total_rollouts += len(records)
        layers.append(
            {
                "style": style,
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                "rollouts": len(records),
                "unique_xy_positions": len(positions),
                "positions": positions,
                "states": sorted({position["search_state"] for position in positions}),
                "radius_min_mm": 1000.0 * min(position["radius_m"] for position in positions),
                "radius_max_mm": 1000.0 * max(position["radius_m"] for position in positions),
            }
        )

    inventory = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "M8_all_conical_sampling_positions",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "total_dynamic_rollouts": total_rollouts,
        "total_unique_xy_positions": len(all_positions),
        "position_definition": "commanded initial bit-center XY relative to true socket center",
        "duplicate_rule": "nuisance repeats at the same XY are drawn once and retained in rollout counts",
        "scope": "complete 15 deg conical direction-fingerprint scans only",
        "excluded_from_direction_map": ["rim/chamfer force-ramp", "inner-wall descent", "inserted terminal traces"],
        "cad_boundaries_mm": {
            "full_support_center_radius": 3.036,
            "declared_visual_95_radius": 4.895,
            "screw_head_radius": 6.5,
        },
        "claim_boundary": (
            "The map is an exact spatial inventory of existing simulation manifests. "
            "It does not imply uniform deployment probability, real-robot coverage, or radial identifiability. "
            "P1v5 remains forbidden for fitting, calibration, threshold selection, or interface changes."
        ),
        "datasets": [
            {
                **asdict(layer["style"]),
                "manifest_path": layer["manifest_path"],
                "rollouts": layer["rollouts"],
                "unique_xy_positions": layer["unique_xy_positions"],
                "states": layer["states"],
                "radius_min_mm": layer["radius_min_mm"],
                "radius_max_mm": layer["radius_max_mm"],
            }
            for layer in layers
        ],
    }
    return layers, inventory


def _camera_snapshot(scene: mujoco.MjvScene, width: int, height: int) -> dict:
    c0 = scene.camera[0]
    c1 = scene.camera[1]
    forward = np.asarray(c0.forward, dtype=float)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    up = np.asarray(c0.up, dtype=float)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    near = max(float(c0.frustum_near), 1.0e-12)
    return {
        "pos": 0.5 * (np.asarray(c0.pos, dtype=float) + np.asarray(c1.pos, dtype=float)),
        "forward": forward,
        "up": up,
        "right": right,
        "vertical_scale": max(abs(float(c0.frustum_top)) / near, 1.0e-12),
        "aspect": float(width) / float(height),
        "width": width,
        "height": height,
    }


def _project_world(point: np.ndarray, camera: dict) -> tuple[int, int] | None:
    vector = np.asarray(point, dtype=float) - camera["pos"]
    depth = float(np.dot(vector, camera["forward"]))
    if depth <= 1.0e-6:
        return None
    x_ndc = float(np.dot(vector, camera["right"])) / (
        depth * camera["vertical_scale"] * camera["aspect"]
    )
    y_ndc = float(np.dot(vector, camera["up"])) / (depth * camera["vertical_scale"])
    return (
        int(round(camera["width"] * (0.5 + 0.5 * x_ndc))),
        int(round(camera["height"] * (0.5 - 0.5 * y_ndc))),
    )


def render_mujoco_top_view() -> tuple[Image.Image, dict]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "bit_collision":
            model.geom_rgba[geom_id, 3] = 0.0
        elif name.startswith("head_top_sector"):
            model.geom_rgba[geom_id] = np.array([0.55, 0.54, 0.49, 1.0])
        elif name.startswith("socket_chamfer"):
            # Keep the collision model unchanged but hide its segmented wedge
            # mesh; in a pure top view those facets look like protrusions.
            model.geom_rgba[geom_id] = np.array([0.80, 0.70, 0.42, 0.0])
        elif name.startswith("socket_wall"):
            model.geom_rgba[geom_id] = np.array([0.20, 0.22, 0.27, 0.0])
        elif name == "socket_bottom":
            model.geom_rgba[geom_id] = np.array([0.09, 0.10, 0.12, 1.0])
        elif name == "workpiece_base":
            model.geom_rgba[geom_id] = np.array([0.12, 0.13, 0.15, 1.0])
    model.site_rgba[:, 3] = 0.0

    data.qpos[:3] = np.array([0.0, 0.0, 0.08])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    model.vis.global_.offwidth = RENDER_SIZE
    model.vis.global_.offheight = RENDER_SIZE
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=RENDER_SIZE, width=RENDER_SIZE)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat = np.array([0.0, 0.0, -0.0035])
    camera.distance = 0.035
    camera.azimuth = 90.0
    camera.elevation = -90.0
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    snapshot = _camera_snapshot(renderer.scene, RENDER_SIZE, RENDER_SIZE)
    frame = Image.fromarray(renderer.render()).convert("RGBA")
    renderer.close()
    return frame, snapshot


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    style: DatasetStyle,
    *,
    origin: tuple[int, int] = (0, 0),
) -> None:
    x = xy[0] + origin[0]
    y = xy[1] + origin[1]
    r = style.marker_radius
    color = (*style.color, 220)
    outline = (8, 12, 18, 235)

    if style.marker == "circle":
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=1)
    elif style.marker == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=1)
    elif style.marker == "diamond":
        draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=color, outline=outline)
    elif style.marker == "triangle":
        draw.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], fill=color, outline=outline)
    elif style.marker == "cross":
        draw.line((x - r, y - r, x + r, y + r), fill=color, width=3)
        draw.line((x - r, y + r, x + r, y - r), fill=color, width=3)
    elif style.marker == "star":
        points = []
        for index in range(10):
            radius = r if index % 2 == 0 else 0.45 * r
            angle = -math.pi / 2.0 + index * math.pi / 5.0
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(255, 255, 255, 255))
    else:
        raise ValueError(f"Unsupported marker: {style.marker}")


def _draw_projected_circle(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    radius_m: float,
    color: tuple[int, int, int, int],
    *,
    width: int,
) -> None:
    points: list[tuple[int, int]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 241):
        projected = _project_world(
            np.array([radius_m * math.cos(angle), radius_m * math.sin(angle), 0.00035]),
            camera,
        )
        if projected is not None:
            points.append((projected[0] + RENDER_X, projected[1] + RENDER_Y))
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def compose_map(layers: list[dict], inventory: dict) -> Image.Image:
    background, camera = render_mujoco_top_view()
    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (10, 14, 20, 255))
    canvas.alpha_composite(background, (RENDER_X, RENDER_Y))
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.text((50, 20), "M8 锥形指纹：全部仿真采样初始位置", font=_font(34, bold=True), fill=(245, 247, 250, 255))
    draw.text(
        (50, 1168),
        "MuJoCo 真实 M8 几何俯视图；每个唯一 XY 只画一次，摩擦/下压力/yaw nuisance 重复保留在统计中。",
        font=_font(19),
        fill=(184, 194, 207, 255),
    )

    _draw_projected_circle(draw, camera, 0.003036, (74, 222, 128, 245), width=3)
    _draw_projected_circle(draw, camera, 0.004895, (34, 211, 238, 245), width=3)

    for layer in layers:
        style = layer["style"]
        for position in layer["positions"]:
            projected = _project_world(
                np.array([position["x_m"], position["y_m"], 0.00045]),
                camera,
            )
            if projected is not None:
                _draw_marker(draw, projected, style, origin=(RENDER_X, RENDER_Y))

    center = _project_world(np.array([0.0, 0.0, 0.0006]), camera)
    if center is not None:
        cx, cy = center[0] + RENDER_X, center[1] + RENDER_Y
        draw.line((cx - 12, cy, cx + 12, cy), fill=(255, 255, 255, 255), width=3)
        draw.line((cx, cy - 12, cx, cy + 12), fill=(255, 255, 255, 255), width=3)

    panel_x = 1190
    draw.rounded_rectangle((panel_x, 72, 1870, 1138), radius=8, fill=(17, 24, 34, 245), outline=(68, 81, 98, 255), width=2)
    draw.text((panel_x + 28, 94), "数据角色与空间覆盖", font=_font(28, bold=True), fill=(245, 247, 250, 255))
    draw.text(
        (panel_x + 28, 142),
        f"{inventory['total_dynamic_rollouts']} 条动态 scan  ->  {inventory['total_unique_xy_positions']} 个唯一 XY",
        font=_font(22, bold=True),
        fill=(255, 255, 255, 255),
    )
    draw.text((panel_x + 28, 180), "同一 XY 的 nuisance 重复不伪装成新位置", font=_font(18), fill=(166, 177, 191, 255))

    y = 232
    for layer in layers:
        style = layer["style"]
        _draw_marker(draw, (panel_x + 42, y + 10), style)
        draw.text((panel_x + 66, y - 4), style.label, font=_font(20, bold=True), fill=(235, 239, 244, 255))
        draw.text(
            (panel_x + 360, y - 4),
            f"{layer['rollouts']:>3} / {layer['unique_xy_positions']:>3}",
            font=_font(20),
            fill=(*style.color, 255),
        )
        draw.text((panel_x + 66, y + 24), style.role, font=_font(16), fill=(157, 170, 186, 255))
        y += 80

    draw.line((panel_x + 28, y + 2, 1840, y + 2), fill=(68, 81, 98, 255), width=2)
    y += 26
    draw.text((panel_x + 28, y), "CAD / 部署边界", font=_font(23, bold=True), fill=(245, 247, 250, 255))
    y += 48
    draw.line((panel_x + 30, y + 11, panel_x + 72, y + 11), fill=(74, 222, 128, 255), width=4)
    draw.text((panel_x + 88, y), "full support: r <= 3.036 mm", font=_font(18), fill=(211, 218, 226, 255))
    y += 38
    draw.line((panel_x + 30, y + 11, panel_x + 72, y + 11), fill=(34, 211, 238, 255), width=4)
    draw.text((panel_x + 88, y), "visual 95%: r <= 4.895 mm", font=_font(18), fill=(211, 218, 226, 255))
    y += 54
    boundary_lines = (
        "• 外圈 P0 点是历史 observability / stress，",
        "  不代表当前部署概率或 nominal claim。",
        "• 方向 CNN 只消费可完成 15° 全圈 probe 的",
        "  flat-support 数据；rim/wall/inserted 另走白盒 gate。",
        "• P1v5 是 final untouched，禁止回流训练或调 gate。",
    )
    for line in boundary_lines:
        draw.text((panel_x + 28, y), line, font=_font(14), fill=(184, 194, 207, 255))
        y += 23

    return canvas.convert("RGB")


def main() -> None:
    layers, inventory = load_sampling_inventory()
    image = compose_map(layers, inventory)
    image.save(OUTPUT_PATH, quality=95)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {INVENTORY_PATH}")
    print(
        f"rollouts={inventory['total_dynamic_rollouts']} "
        f"unique_xy={inventory['total_unique_xy_positions']}"
    )


if __name__ == "__main__":
    main()
