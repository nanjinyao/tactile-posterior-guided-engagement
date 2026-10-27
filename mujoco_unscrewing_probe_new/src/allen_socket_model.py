from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AllenSocketSpec:
    """Metric Allen socket-head engagement geometry in SI units.

    `key_af_m` is the nominal Allen key / bit across-flats size. Clearance and
    micro-chamfer are controlled simulation variables, not ISO tolerance claims.
    """

    name: str
    thread_d_m: float
    key_af_m: float
    clearance_m: float
    chamfer_width_m: float
    chamfer_depth_m: float
    socket_depth_m: float
    screw_head_diameter_m: float
    screw_head_height_m: float
    visual_offset_default_m: float
    visual_offset_sweep_m: tuple[float, float, float]
    visual_offset_min_m: float
    bit_length_m: float
    bit_tip_chamfer_m: float
    bit_tip_scale: float = 0.98
    wall_thickness_m: float = 0.0030
    head_top_segments: int = 72
    top_contact_thickness_m: float = 0.00008
    workpiece_margin_m: float = 0.0060
    bit_mass_kg: float = 0.50
    friction: tuple[float, float, float] = (0.75, 0.02, 0.001)
    solref: tuple[float, float] = (0.010, 1.0)
    solimp: tuple[float, float, float] = (0.80, 0.95, 0.010)
    # The historical source scenes use six overlapping wall boxes and six
    # chamfer wedges.  Keep that frozen behavior for the published source
    # artifacts.  Derived hardware-easy scenes opt into a circularly bounded
    # shell so the enlarged entry funnel cannot create invisible collision
    # wings outside the physical screw-head diameter.
    bounded_socket_shell: bool = False

    @property
    def bit_af_m(self) -> float:
        return self.key_af_m

    @property
    def socket_wall_af_m(self) -> float:
        return self.key_af_m + 2.0 * self.clearance_m

    @property
    def socket_top_af_m(self) -> float:
        return self.socket_wall_af_m + 2.0 * self.chamfer_width_m

    @property
    def r_cap_wall_m(self) -> float:
        return self.clearance_m

    @property
    def r_cap_top_m(self) -> float:
        return self.clearance_m + self.chamfer_width_m

    @property
    def head_radius_m(self) -> float:
        return self.screw_head_diameter_m / 2.0

    @property
    def workpiece_half_size_m(self) -> float:
        return self.head_radius_m + self.workpiece_margin_m


SPECS: dict[str, AllenSocketSpec] = {
    "m8": AllenSocketSpec(
        name="m8",
        thread_d_m=0.0080,
        key_af_m=0.0060,
        clearance_m=0.00010,
        chamfer_width_m=0.00015,
        chamfer_depth_m=0.00015,
        socket_depth_m=0.0040,
        screw_head_diameter_m=0.0130,
        screw_head_height_m=0.0080,
        visual_offset_default_m=0.0030,
        visual_offset_sweep_m=(0.0010, 0.0030, 0.0050),
        visual_offset_min_m=0.0008,
        bit_length_m=0.0350,
        bit_tip_chamfer_m=0.00035,
    ),
    "m10": AllenSocketSpec(
        name="m10",
        thread_d_m=0.0100,
        key_af_m=0.0080,
        clearance_m=0.00012,
        chamfer_width_m=0.00018,
        chamfer_depth_m=0.00018,
        socket_depth_m=0.0050,
        screw_head_diameter_m=0.0165,
        screw_head_height_m=0.0100,
        visual_offset_default_m=0.0040,
        visual_offset_sweep_m=(0.0015, 0.0040, 0.0060),
        visual_offset_min_m=0.0010,
        bit_length_m=0.0420,
        bit_tip_chamfer_m=0.00045,
        wall_thickness_m=0.0035,
    ),
    "m12": AllenSocketSpec(
        name="m12",
        thread_d_m=0.0120,
        key_af_m=0.0100,
        clearance_m=0.00015,
        chamfer_width_m=0.00022,
        chamfer_depth_m=0.00022,
        socket_depth_m=0.0060,
        screw_head_diameter_m=0.0185,
        screw_head_height_m=0.0120,
        visual_offset_default_m=0.0050,
        visual_offset_sweep_m=(0.0020, 0.0050, 0.0080),
        visual_offset_min_m=0.0012,
        bit_length_m=0.0500,
        bit_tip_chamfer_m=0.00055,
        wall_thickness_m=0.0040,
    ),
}


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _hex_ray_radius(apothem: float, theta: float) -> float:
    normals = [i * math.pi / 3.0 for i in range(6)]
    denom = max(abs(math.cos(theta - n)) for n in normals)
    return apothem / max(denom, 1.0e-12)


def _hex_bit_mesh(across_flats: float, length: float, chamfer: float, tip_scale: float) -> tuple[str, str]:
    radius = across_flats / math.sqrt(3.0)
    half = length / 2.0
    rings = [
        (radius * tip_scale, -half),
        (radius, -half + chamfer),
        (radius, half),
    ]
    vertices: list[float] = []
    for ring_radius, z in rings:
        for i in range(6):
            theta = math.pi / 6.0 + i * math.pi / 3.0
            vertices.extend([ring_radius * math.cos(theta), ring_radius * math.sin(theta), z])
    faces: list[int] = []
    faces.extend([0, 5, 4, 0, 4, 3, 0, 3, 2, 0, 2, 1])
    for ring in [0, 1]:
        base = ring * 6
        nxt = (ring + 1) * 6
        for i in range(6):
            j = (i + 1) % 6
            faces.extend([base + i, base + j, nxt + j])
            faces.extend([base + i, nxt + j, nxt + i])
    top = 12
    faces.extend([top, top + 1, top + 2, top, top + 2, top + 3, top, top + 3, top + 4, top, top + 4, top + 5])
    return _fmt(vertices), " ".join(str(i) for i in faces)


def _cylinder_side_mesh(radius: float, z_top: float, z_bottom: float, segments: int) -> tuple[str, str]:
    vertices: list[float] = []
    for z in [z_top, z_bottom]:
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            vertices.extend([radius * math.cos(theta), radius * math.sin(theta), z])
    faces: list[int] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([i, segments + i, segments + j])
        faces.extend([i, segments + j, j])
    return _fmt(vertices), " ".join(str(i) for i in faces)


def _ring_sector_prism_mesh(
    outer_radius: float,
    inner_apothem: float,
    theta0: float,
    theta1: float,
    z_top: float,
    z_bottom: float,
) -> tuple[str, str]:
    points: list[tuple[float, float, float]] = []
    for theta, radius in [
        (theta0, outer_radius),
        (theta1, outer_radius),
        (theta1, _hex_ray_radius(inner_apothem, theta1)),
        (theta0, _hex_ray_radius(inner_apothem, theta0)),
    ]:
        points.append((radius * math.cos(theta), radius * math.sin(theta), z_top))
    for theta, radius in [
        (theta0, outer_radius),
        (theta1, outer_radius),
        (theta1, _hex_ray_radius(inner_apothem, theta1)),
        (theta0, _hex_ray_radius(inner_apothem, theta0)),
    ]:
        points.append((radius * math.cos(theta), radius * math.sin(theta), z_bottom))
    vertices = [coord for point in points for coord in point]
    faces = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    return _fmt(vertices), " ".join(str(i) for i in faces)


def _chamfer_wedge_mesh(side_index: int, wall_apothem: float, chamfer_width: float, chamfer_depth: float, wall_thickness: float) -> tuple[str, str]:
    theta = side_index * math.pi / 3.0
    n = (math.cos(theta), math.sin(theta))
    t = (-math.sin(theta), math.cos(theta))
    a_wall = wall_apothem
    a_top = wall_apothem + chamfer_width
    half_tangent = (2.0 * a_top / math.sqrt(3.0)) / 2.0 + wall_thickness
    z_top = 0.0
    z_bottom = -chamfer_depth

    def point(dist: float, tang: float, z: float) -> list[float]:
        return [n[0] * dist + t[0] * tang, n[1] * dist + t[1] * tang, z]

    pts = [
        point(a_top, -half_tangent, z_top),
        point(a_top, half_tangent, z_top),
        point(a_top + wall_thickness, -half_tangent, z_top),
        point(a_top + wall_thickness, half_tangent, z_top),
        point(a_wall, -half_tangent, z_bottom),
        point(a_wall, half_tangent, z_bottom),
        point(a_wall + wall_thickness, -half_tangent, z_bottom),
        point(a_wall + wall_thickness, half_tangent, z_bottom),
    ]
    vertices = [coord for p in pts for coord in p]
    faces = [
        0, 4, 5, 0, 5, 1,
        2, 3, 7, 2, 7, 6,
        0, 1, 3, 0, 3, 2,
        4, 6, 7, 4, 7, 5,
        0, 2, 6, 0, 6, 4,
        1, 5, 7, 1, 7, 3,
    ]
    return _fmt(vertices), " ".join(str(i) for i in faces)


def _chamfer_ring_sector_mesh(
    outer_radius: float,
    wall_apothem: float,
    top_apothem: float,
    theta0: float,
    theta1: float,
    chamfer_depth: float,
) -> tuple[str, str]:
    """Convex sector of a 45-degree hex-entry chamfer, clipped to a circle.

    The inner boundary moves from ``top_apothem`` at z=0 to
    ``wall_apothem`` at z=-chamfer_depth.  Splitting the annulus into narrow
    sectors avoids both concave collision meshes and the out-of-envelope
    corners created by the legacy overlapping wedge proxy.
    """

    def polar(radius: float, theta: float, z: float) -> tuple[float, float, float]:
        return radius * math.cos(theta), radius * math.sin(theta), z

    points = [
        polar(outer_radius, theta0, 0.0),
        polar(outer_radius, theta1, 0.0),
        polar(_hex_ray_radius(top_apothem, theta1), theta1, 0.0),
        polar(_hex_ray_radius(top_apothem, theta0), theta0, 0.0),
        polar(outer_radius, theta0, -chamfer_depth),
        polar(outer_radius, theta1, -chamfer_depth),
        polar(_hex_ray_radius(wall_apothem, theta1), theta1, -chamfer_depth),
        polar(_hex_ray_radius(wall_apothem, theta0), theta0, -chamfer_depth),
    ]
    vertices = [coordinate for point in points for coordinate in point]
    faces = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    return _fmt(vertices), " ".join(str(index) for index in faces)


def _hex_disk_mesh(apothem: float, z_top: float, z_bottom: float) -> tuple[str, str]:
    radius = apothem / math.cos(math.pi / 6.0)
    vertices: list[float] = []
    for z in [z_top, z_bottom]:
        for i in range(6):
            theta = math.pi / 6.0 + i * math.pi / 3.0
            vertices.extend([radius * math.cos(theta), radius * math.sin(theta), z])
    faces: list[int] = []
    faces.extend([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5])
    faces.extend([6, 8, 7, 6, 9, 8, 6, 10, 9, 6, 11, 10])
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([i, j, 6 + j])
        faces.extend([i, 6 + j, 6 + i])
    return _fmt(vertices), " ".join(str(i) for i in faces)


def spec_to_config(spec: AllenSocketSpec) -> dict:
    data = asdict(spec)
    data.update(
        {
            "units": "SI",
            "claim_boundary": "parameterized fixed Allen socket-head engagement model; not threaded unscrewing",
            "nominal_standard_dimensions": {
                "thread_d_m": spec.thread_d_m,
                "allen_key_across_flats_m": spec.key_af_m,
                "screw_head_diameter_m": spec.screw_head_diameter_m,
                "screw_head_height_m": spec.screw_head_height_m,
            },
            "controlled_simulation_variables": {
                "single_side_clearance_m": spec.clearance_m,
                "micro_chamfer_width_m": spec.chamfer_width_m,
                "micro_chamfer_depth_m": spec.chamfer_depth_m,
            },
            "derived": {
                "bit_af_m": spec.bit_af_m,
                "socket_wall_af_m": spec.socket_wall_af_m,
                "socket_top_af_m": spec.socket_top_af_m,
                "r_cap_wall_m": spec.r_cap_wall_m,
                "r_cap_top_m": spec.r_cap_top_m,
            },
        }
    )
    return data


def build_xml(spec: AllenSocketSpec) -> str:
    head_segments = max(24, int(spec.head_top_segments))
    wall_apothem = spec.socket_wall_af_m / 2.0
    top_apothem = spec.socket_top_af_m / 2.0
    head_radius = spec.head_radius_m
    wall_height = max(spec.socket_depth_m - spec.chamfer_depth_m, 0.0005)
    wall_z = -(spec.chamfer_depth_m + wall_height / 2.0)
    side_length = 2.0 * wall_apothem / math.sqrt(3.0)
    start_z = spec.bit_length_m / 2.0 + 0.0060

    assets: list[str] = []
    geoms: list[str] = []

    bit_vertices, bit_faces = _hex_bit_mesh(spec.bit_af_m, spec.bit_length_m, spec.bit_tip_chamfer_m, spec.bit_tip_scale)
    side_vertices, side_faces = _cylinder_side_mesh(head_radius, 0.0, -spec.screw_head_height_m, head_segments)
    bottom_vertices, bottom_faces = _hex_disk_mesh(wall_apothem, -spec.socket_depth_m, -spec.socket_depth_m - 0.00012)
    assets.append(f'<mesh name="{spec.name}_bit_mesh" vertex="{bit_vertices}" face="{bit_faces}"/>')
    assets.append(f'<mesh name="{spec.name}_head_side_mesh" vertex="{side_vertices}" face="{side_faces}"/>')
    assets.append(f'<mesh name="{spec.name}_socket_bottom_mesh" vertex="{bottom_vertices}" face="{bottom_faces}"/>')

    for i in range(head_segments):
        theta0 = 2.0 * math.pi * i / head_segments
        theta1 = 2.0 * math.pi * (i + 1) / head_segments
        vertices, faces = _ring_sector_prism_mesh(head_radius, top_apothem, theta0, theta1, 0.0, -spec.top_contact_thickness_m)
        assets.append(f'<mesh name="{spec.name}_head_top_sector_{i}" vertex="{vertices}" face="{faces}"/>')
        geoms.append(
            f'<geom name="head_top_sector_{i}" type="mesh" mesh="{spec.name}_head_top_sector_{i}" '
            f'rgba="0.70 0.69 0.64 1" friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
        )

    if spec.bounded_socket_shell:
        for i in range(head_segments):
            theta0 = 2.0 * math.pi * i / head_segments
            theta1 = 2.0 * math.pi * (i + 1) / head_segments
            vertices, faces = _chamfer_ring_sector_mesh(
                head_radius,
                wall_apothem,
                top_apothem,
                theta0,
                theta1,
                spec.chamfer_depth_m,
            )
            assets.append(
                f'<mesh name="{spec.name}_socket_chamfer_sector_{i}" vertex="{vertices}" face="{faces}"/>'
            )
            geoms.append(
                f'<geom name="socket_chamfer_sector_{i}" type="mesh" '
                f'mesh="{spec.name}_socket_chamfer_sector_{i}" rgba="0.82 0.80 0.72 1" '
                f'friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
            )
            vertices, faces = _ring_sector_prism_mesh(
                head_radius,
                wall_apothem,
                theta0,
                theta1,
                -spec.chamfer_depth_m,
                -spec.socket_depth_m,
            )
            assets.append(
                f'<mesh name="{spec.name}_socket_wall_sector_{i}" vertex="{vertices}" face="{faces}"/>'
            )
            geoms.append(
                f'<geom name="socket_wall_sector_{i}" type="mesh" '
                f'mesh="{spec.name}_socket_wall_sector_{i}" rgba="0.40 0.41 0.43 1" '
                f'friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
            )
    else:
        for i in range(6):
            vertices, faces = _chamfer_wedge_mesh(i, wall_apothem, spec.chamfer_width_m, spec.chamfer_depth_m, spec.wall_thickness_m)
            assets.append(f'<mesh name="{spec.name}_socket_chamfer_{i}" vertex="{vertices}" face="{faces}"/>')
            geoms.append(
                f'<geom name="socket_chamfer_{i}" type="mesh" mesh="{spec.name}_socket_chamfer_{i}" '
                f'rgba="0.82 0.80 0.72 1" friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
            )
            theta = i * math.pi / 3.0
            nx, ny = math.cos(theta), math.sin(theta)
            yaw = theta - math.pi / 2.0
            cx = nx * (wall_apothem + spec.wall_thickness_m / 2.0)
            cy = ny * (wall_apothem + spec.wall_thickness_m / 2.0)
            geoms.append(
                f'<geom name="socket_wall_{i}" type="box" pos="{cx:.9g} {cy:.9g} {wall_z:.9g}" '
                f'euler="0 0 {yaw:.9g}" size="{side_length / 2.0 + spec.wall_thickness_m:.9g} {spec.wall_thickness_m / 2.0:.9g} {wall_height / 2.0:.9g}" '
                f'rgba="0.40 0.41 0.43 1" friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
            )

    geoms.append(
        f'<geom name="socket_bottom" type="mesh" mesh="{spec.name}_socket_bottom_mesh" '
        f'rgba="0.30 0.31 0.33 1" friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>'
    )
    geoms.append(
        f'<geom name="screw_head_side_visual" type="mesh" mesh="{spec.name}_head_side_mesh" contype="0" conaffinity="0" '
        f'rgba="0.54 0.53 0.50 1"/>'
    )
    geoms.append(
        f'<geom name="workpiece_base" type="box" pos="0 0 {-spec.screw_head_height_m - 0.001:.9g}" '
        f'size="{spec.workpiece_half_size_m:.9g} {spec.workpiece_half_size_m:.9g} 0.001" rgba="0.22 0.23 0.24 1"/>'
    )

    asset_text = "\n    ".join(assets)
    geom_text = "\n      ".join(geoms)
    xml = f'''<mujoco model="allen_socket_engagement_{spec.name}">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option timestep="0.0005" gravity="0 0 -9.81"/>
  <asset>
    {asset_text}
  </asset>
  <worldbody>
    <body name="fixed_screw_{spec.name}" pos="0 0 0">
      {geom_text}
      <site name="recess_center" pos="0 0 0" size="0.0004" rgba="0 0.7 1 1"/>
    </body>
    <body name="allen_bit_{spec.name}" pos="0 0 {start_z:.9g}">
      <freejoint name="bit_freejoint"/>
      <geom name="bit_collision" type="mesh" mesh="{spec.name}_bit_mesh" mass="{spec.bit_mass_kg:.9g}"
            rgba="0.12 0.14 0.18 1" friction="{_fmt(spec.friction)}" solref="{_fmt(spec.solref)}" solimp="{_fmt(spec.solimp)}"/>
      <site name="bit_tip" pos="0 0 {-spec.bit_length_m / 2.0:.9g}" size="0.00035" rgba="1 0.2 0.1 1"/>
      <site name="ft_sensor_site" pos="0 0 0" size="0.00045" rgba="0.1 0.8 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
'''
    return xml


def inventory_row(spec: AllenSocketSpec) -> dict:
    return {
        "size": spec.name.upper(),
        "thread_d_mm": spec.thread_d_m * 1000.0,
        "key_or_bit_af_mm": spec.key_af_m * 1000.0,
        "single_side_clearance_mm": spec.clearance_m * 1000.0,
        "socket_wall_af_mm": spec.socket_wall_af_m * 1000.0,
        "micro_chamfer_width_mm": spec.chamfer_width_m * 1000.0,
        "micro_chamfer_depth_mm": spec.chamfer_depth_m * 1000.0,
        "socket_top_af_mm": spec.socket_top_af_m * 1000.0,
        "r_cap_wall_mm": spec.r_cap_wall_m * 1000.0,
        "r_cap_top_mm": spec.r_cap_top_m * 1000.0,
        "socket_depth_mm": spec.socket_depth_m * 1000.0,
        "head_diameter_mm": spec.screw_head_diameter_m * 1000.0,
        "head_height_mm": spec.screw_head_height_m * 1000.0,
        "visual_offset_default_mm": spec.visual_offset_default_m * 1000.0,
        "visual_offset_sweep_mm": [v * 1000.0 for v in spec.visual_offset_sweep_m],
    }


def write_inventory(rows: list[dict], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"units": "mm in inventory rows", "rows": rows}, indent=2), encoding="utf-8")
    headers = [
        "size",
        "key_or_bit_af_mm",
        "socket_wall_af_mm",
        "clearance_mm",
        "chamfer_w/d_mm",
        "socket_top_af_mm",
        "r_cap_wall/top_mm",
        "socket_depth_mm",
        "head_diameter/height_mm",
        "visual_R_default_mm",
    ]
    lines = [
        "# M8 / M10 / M12 Allen Socket Model Inventory",
        "",
        "All dimensions are generated from `src/allen_socket_model.py`.",
        "The Allen key across-flats and screw-head scale are nominal size parameters.",
        "Clearance and micro-chamfer are controlled simulation variables, not manufacturing tolerance claims.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["size"]),
                    f'{row["key_or_bit_af_mm"]:.2f}',
                    f'{row["socket_wall_af_mm"]:.2f}',
                    f'{row["single_side_clearance_mm"]:.2f}',
                    f'{row["micro_chamfer_width_mm"]:.2f}/{row["micro_chamfer_depth_mm"]:.2f}',
                    f'{row["socket_top_af_mm"]:.2f}',
                    f'{row["r_cap_wall_mm"]:.2f}/{row["r_cap_top_mm"]:.2f}',
                    f'{row["socket_depth_mm"]:.2f}',
                    f'{row["head_diameter_mm"]:.1f}/{row["head_height_mm"]:.1f}',
                    f'{row["visual_offset_default_mm"]:.1f}',
                ]
            )
            + "|"
        )
    lines.extend(
        [
            "",
            "Claim boundary:",
            "",
            "```text",
            "Generated XMLs are fixed Allen socket-head engagement scenes.",
            "They do not include threads, screw extraction, tool changer, visual detector, or full unscrewing dynamics.",
            "M10/M12 are model-ready scale variants; current first evidence should remain M8-first.",
            "```",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def generate(selected: list[str]) -> None:
    rows: list[dict] = []
    for name in selected:
        spec = SPECS[name]
        config_path = ROOT / "configs" / f"allen_{name}_source.json"
        model_path = ROOT / "models" / f"allen_{name}_source.xml"
        config_path.write_text(json.dumps(spec_to_config(spec), indent=2), encoding="utf-8")
        model_path.write_text(build_xml(spec), encoding="utf-8")
        rows.append(inventory_row(spec))
    write_inventory(rows, ROOT / "outputs/current/reports/m8_m10_m12_model_inventory.json", ROOT / "docs/M8_M10_M12_MODEL_INVENTORY.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean M8/M10/M12 Allen socket MuJoCo models.")
    parser.add_argument("--sizes", default="m8,m10,m12", help="Comma-separated sizes: m8,m10,m12")
    args = parser.parse_args()
    selected = [s.strip().lower() for s in args.sizes.split(",") if s.strip()]
    unknown = sorted(set(selected) - set(SPECS))
    if unknown:
        raise SystemExit(f"unknown sizes: {unknown}")
    generate(selected)


if __name__ == "__main__":
    main()
