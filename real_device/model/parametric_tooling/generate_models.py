#!/usr/bin/env python3
"""Generate three one-piece flange-peg tools and a three-socket fixture.

The original SolidWorks STEP is used as an exact B-rep parent.  All generated
parts are in millimetres.  In the canonical tool assembly the blue flange's
sensor-side mating plane is Z=0 and the tool points along -Z.  The socket
fixture uses the same XY convention, with its socket mouths at Z=0.

This file reads the three MuJoCo JSON sources for the strict simulation-exact
socket profile.  The separately named hardware-easy-v2 profile deliberately
enlarges the straight-wall clearance, entry funnel, and exterior head; that
engineering bias remains explicit in ``design_parameters.json`` and is never
folded into the simulation-exact export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cadquery as cq


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
REAL_DEVICE_DIR = MODEL_DIR.parent
GROUP_MEETING_DIR = REAL_DEVICE_DIR.parent
SIM_ROOT = GROUP_MEETING_DIR / "mujoco_unscrewing_probe_new"
SOURCE_STEP = MODEL_DIR / "fast_mount_ATI_ft_1.STEP"
PARAMETERS_FILE = HERE / "design_parameters.json"
OUTPUT_DIR = HERE / "generated"

SIZE_ORDER = ("m8", "m10", "m12")
SOURCE_CONFIGS = {
    size: SIM_ROOT / "configs" / f"allen_{size}_source.json"
    for size in SIZE_ORDER
}
RETIRED_ACTIVE_OUTPUT_STEMS = (
    "modified_blue_flange_receiver",
    "tool_free_retaining_clip",
    "peg_cartridge_m8",
    "peg_cartridge_m10",
    "peg_cartridge_m12",
    "quick_change_fit_coupon",
    "assembly_m8",
    "assembly_m10",
    "assembly_m12",
    "socket_base_hardware_easy",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_retired_active_outputs() -> None:
    """Prevent archived v1.3 parts from being mistaken for active print files."""

    for stem in RETIRED_ACTIVE_OUTPUT_STEMS:
        for suffix in (".step", ".stl", ".3mf", ".svg"):
            path = OUTPUT_DIR / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def hex_points(
    across_flats_mm: float,
    yaw_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """Regular hexagon whose legacy selected flat normal is rotated by yaw."""

    radius = across_flats_mm / math.sqrt(3.0)
    return [
        (
            radius * math.cos(math.radians(30.0 + yaw_deg + 60.0 * index)),
            radius * math.sin(math.radians(30.0 + yaw_deg + 60.0 * index)),
        )
        for index in range(6)
    ]


def hex_prism(
    across_flats_mm: float,
    z0_mm: float,
    z1_mm: float,
    yaw_deg: float = 0.0,
) -> cq.Workplane:
    if z1_mm <= z0_mm:
        raise ValueError("hex_prism requires z1 > z0")
    return (
        cq.Workplane("XY")
        .polyline(hex_points(across_flats_mm, yaw_deg))
        .close()
        .extrude(z1_mm - z0_mm)
        .translate((0.0, 0.0, z0_mm))
    )


def hex_loft(
    lower_af_mm: float,
    upper_af_mm: float,
    z0_mm: float,
    z1_mm: float,
    yaw_deg: float = 0.0,
) -> cq.Workplane:
    if z1_mm <= z0_mm:
        raise ValueError("hex_loft requires z1 > z0")
    lower_points = [
        cq.Vector(x, y, z0_mm)
        for x, y in hex_points(lower_af_mm, yaw_deg)
    ]
    upper_points = [
        cq.Vector(x, y, z1_mm)
        for x, y in hex_points(upper_af_mm, yaw_deg)
    ]
    faces = [
        cq.Face.makeFromWires(
            cq.Wire.makePolygon(list(reversed(lower_points)), close=True)
        ),
        cq.Face.makeFromWires(
            cq.Wire.makePolygon(upper_points, close=True)
        ),
    ]
    for index in range(6):
        following = (index + 1) % 6
        side = [
            lower_points[index],
            lower_points[following],
            upper_points[following],
            upper_points[index],
        ]
        faces.append(
            cq.Face.makeFromWires(cq.Wire.makePolygon(side, close=True))
        )
    solid = cq.Solid.makeSolid(cq.Shell.makeShell(faces))
    if not solid.isValid():
        raise RuntimeError("explicit planar hex frustum is invalid")
    return cq.Workplane(obj=solid)


def circular_frustum(
    lower_diameter_mm: float,
    upper_diameter_mm: float,
    z0_mm: float,
    z1_mm: float,
) -> cq.Workplane:
    if z1_mm <= z0_mm:
        raise ValueError("circular_frustum requires z1 > z0")
    return (
        cq.Workplane("XY")
        .circle(lower_diameter_mm / 2.0)
        .workplane(offset=z1_mm - z0_mm)
        .circle(upper_diameter_mm / 2.0)
        .loft(combine=True)
        .translate((0.0, 0.0, z0_mm))
    )


def cylinder(
    diameter_mm: float,
    z0_mm: float,
    z1_mm: float,
    x_mm: float = 0.0,
    y_mm: float = 0.0,
) -> cq.Workplane:
    if z1_mm <= z0_mm:
        raise ValueError("cylinder requires z1 > z0")
    return (
        cq.Workplane("XY")
        .center(x_mm, y_mm)
        .circle(diameter_mm / 2.0)
        .extrude(z1_mm - z0_mm)
        .translate((0.0, 0.0, z0_mm))
    )


def rectangular_prism(
    size_x_mm: float,
    size_y_mm: float,
    z0_mm: float,
    z1_mm: float,
    x_mm: float = 0.0,
    y_mm: float = 0.0,
) -> cq.Workplane:
    if z1_mm <= z0_mm:
        raise ValueError("rectangular_prism requires z1 > z0")
    return (
        cq.Workplane("XY")
        .box(size_x_mm, size_y_mm, z1_mm - z0_mm, centered=(True, True, False))
        .translate((x_mm, y_mm, z0_mm))
    )


def rounded_plate(
    size_x_mm: float,
    size_y_mm: float,
    z0_mm: float,
    z1_mm: float,
    radius_mm: float,
) -> cq.Workplane:
    plate = (
        cq.Workplane("XY")
        .box(size_x_mm, size_y_mm, z1_mm - z0_mm, centered=(True, True, False))
        .translate((0.0, 0.0, z0_mm))
    )
    if radius_mm > 0.0:
        plate = plate.edges("|Z").fillet(radius_mm)
    return plate


def transformed_source_flange(params: Mapping[str, Any]) -> cq.Workplane:
    tool = params["tool_reference"]
    flange_thickness = float(tool["blue_flange_thickness_mm"])
    # The source STEP has its profile in XZ and thickness in +Y.  Rotate it to
    # a conventional XY profile and place its sensor-side face at Z=0.
    return (
        cq.importers.importStep(str(SOURCE_STEP))
        .rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90.0)
        .translate((0.0, 0.0, -flange_thickness))
    )


def build_clean_flange_replica(params: Mapping[str, Any]) -> cq.Workplane:
    """Rebuild the supplied flange's analytic profile for clean STL meshing.

    The source STEP is a valid solid, but its many split tangent faces create
    self-intersections in otherwise closed STL tessellations.  The source
    profile is exactly a R36 central disk, four 14 mm arms, four R7 lobes,
    eight R2 arm-root blends, and four 6.6 mm through-holes on PCD 88.  This
    reconstruction differs from the imported solid by less than 0.05 mm^3.
    """

    flange_thickness = float(params["tool_reference"]["blue_flange_thickness_mm"])
    mount = cylinder(72.0, -flange_thickness, 0.0)
    for x_mm, y_mm in ((0.0, 44.0), (44.0, 0.0), (0.0, -44.0), (-44.0, 0.0)):
        if abs(x_mm) < 1.0e-9:
            arm = (
                cq.Workplane("XY")
                .center(0.0, y_mm / 2.0)
                .rect(14.0, abs(y_mm))
                .extrude(flange_thickness)
                .translate((0.0, 0.0, -flange_thickness))
            )
        else:
            arm = (
                cq.Workplane("XY")
                .center(x_mm / 2.0, 0.0)
                .rect(abs(x_mm), 14.0)
                .extrude(flange_thickness)
                .translate((0.0, 0.0, -flange_thickness))
            )
        lobe = cylinder(14.0, -flange_thickness, 0.0, x_mm, y_mm)
        mount = mount.union(arm).union(lobe)

    root_edges = [
        edge
        for edge in mount.edges("|Z").vals()
        if 34.0 < math.hypot(edge.Center().x, edge.Center().y) < 38.0
    ]
    if len(root_edges) != 8:
        raise RuntimeError(f"expected 8 flange root edges, found {len(root_edges)}")
    mount = mount.newObject(root_edges).fillet(2.0)

    for x_mm, y_mm in ((44.0, 0.0), (-44.0, 0.0), (0.0, 44.0), (0.0, -44.0)):
        mount = mount.cut(cylinder(6.6, -flange_thickness - 0.2, 0.2, x_mm, y_mm))
    return mount


def resolve_peg_tip_geometry_mm(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    tip_profile_name: str,
) -> Dict[str, float]:
    """Resolve a source-exact or explicitly hardware-biased peg tip."""

    peg_params = params["peg"]
    profiles = peg_params["tip_geometry_profiles"]
    if tip_profile_name not in profiles:
        raise KeyError(f"unknown peg tip geometry profile: {tip_profile_name}")
    profile = profiles[tip_profile_name]
    bit_af = 1000.0 * float(sim_spec["key_af_m"])
    source_tip_face_af = bit_af * float(sim_spec["bit_tip_scale"])
    source_axial_length = 1000.0 * float(sim_spec["bit_tip_chamfer_m"])
    source_radial_setback = 0.5 * (bit_af - source_tip_face_af)
    if bool(profile["source_controlled"]):
        tip_face_af = source_tip_face_af
        axial_length = source_axial_length
        radial_setback = source_radial_setback
    else:
        radial_setback = float(
            profile["tip_chamfer_radial_setback_mm"][size_name]
        )
        axial_length = float(profile["tip_chamfer_axial_length_mm"][size_name])
        tip_face_af = bit_af - 2.0 * radial_setback
    if tip_face_af <= 0.0 or axial_length <= 0.0 or radial_setback <= 0.0:
        raise ValueError(f"{size_name}/{tip_profile_name}: invalid peg tip geometry")
    if not bool(profile["source_controlled"]) and not math.isclose(
        radial_setback,
        axial_length,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            f"{size_name}/{tip_profile_name}: hardware peg tip must retain a 45-degree chamfer"
        )
    return {
        "shaft_af_mm": bit_af,
        "tip_face_af_mm": tip_face_af,
        "tip_face_scale": tip_face_af / bit_af,
        "tip_chamfer_radial_setback_mm": radial_setback,
        "tip_chamfer_axial_length_mm": axial_length,
        "tip_chamfer_angle_from_axis_deg": math.degrees(
            math.atan2(radial_setback, axial_length)
        ),
        "source_tip_face_af_mm": source_tip_face_af,
        "source_tip_chamfer_axial_length_mm": source_axial_length,
        "source_tip_chamfer_radial_setback_mm": source_radial_setback,
    }


def build_integrated_flange_peg(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    tip_profile_name: str,
) -> cq.Workplane:
    """Build one solid flange-to-peg load path with a common Z=-42 mm tip.

    The sensor-side flange face is the intended print-bed datum.  After that
    face, every circular section becomes smaller with print height, so the
    failed supported shoulder of the retired cartridge design is eliminated.
    The lower 13 mm contact segment and the MuJoCo tip chamfer remain unchanged.
    """

    tool = params["tool_reference"]
    root = params["integrated_tool"]
    peg_params = params["peg"]
    peg_yaw_deg = float(peg_params["flange_relative_yaw_deg"])

    flange = build_clean_flange_replica(params)
    flange_thickness = float(tool["blue_flange_thickness_mm"])
    flange_tool_face_z = -flange_thickness
    tip_z = -float(tool["blue_sensor_plane_to_tip_mm"])

    first_taper_end_z = flange_tool_face_z - float(root["root_first_taper_length_mm"])
    boss_end_z = first_taper_end_z - float(root["root_boss_length_mm"])
    load_cone_end_z = boss_end_z - float(root["load_cone_length_mm"])
    collar_end_z = load_cone_end_z - float(root["lower_collar_length_mm"])

    root_flange_d = float(root["root_flange_diameter_mm"])
    boss_d = float(root["root_boss_diameter_mm"])
    lower_d = float(root["load_cone_lower_diameter_mm"])
    load_path = circular_frustum(
        boss_d,
        root_flange_d,
        first_taper_end_z,
        flange_tool_face_z + 0.03,
    )
    load_path = load_path.union(
        cylinder(boss_d, boss_end_z, first_taper_end_z + 0.03)
    )
    load_path = load_path.union(
        circular_frustum(
            lower_d,
            boss_d,
            load_cone_end_z,
            boss_end_z + 0.03,
        )
    )
    load_path = load_path.union(
        cylinder(lower_d, collar_end_z, load_cone_end_z + 0.03)
    )

    tip_geometry = resolve_peg_tip_geometry_mm(
        size_name,
        sim_spec,
        params,
        tip_profile_name,
    )
    bit_af = tip_geometry["shaft_af_mm"]
    tip_taper_h = tip_geometry["tip_chamfer_axial_length_mm"]
    tip_face_af = tip_geometry["tip_face_af_mm"]
    uniform_length = float(peg_params["uniform_contact_length_mm"])
    transition_length = float(peg_params["reinforcement_transition_length_mm"])
    upper_af = max(bit_af, float(peg_params["reinforcement_upper_af_mm"]))
    uniform_top_z = tip_z + uniform_length
    transition_top_z = uniform_top_z + transition_length
    if not math.isclose(transition_top_z, collar_end_z, abs_tol=1.0e-9):
        raise ValueError(
            f"{size_name}: integrated root ends at {collar_end_z:.3f} mm but "
            f"the AF14 reinforcement ends at {transition_top_z:.3f} mm"
        )
    required_collar_d = upper_af / math.cos(math.radians(30.0))
    if lower_d + 1.0e-9 < required_collar_d:
        raise ValueError(
            f"{size_name}: D{lower_d:.3f} collar cannot contain AF{upper_af:.3f}"
        )

    peg = hex_loft(
        tip_face_af,
        bit_af,
        tip_z,
        tip_z + tip_taper_h,
        peg_yaw_deg,
    )
    peg = peg.union(
        hex_prism(
            bit_af,
            tip_z + tip_taper_h - 0.03,
            uniform_top_z,
            peg_yaw_deg,
        )
    )
    peg = peg.union(
        hex_loft(
            bit_af,
            upper_af,
            uniform_top_z - 0.03,
            transition_top_z + 0.03,
            peg_yaw_deg,
        )
    )
    return flange.union(load_path).union(peg)


def build_modified_flange(params: Mapping[str, Any]) -> cq.Workplane:
    """Legacy v1.3 builder retained only for archived design reproduction."""
    tool = params["tool_reference"]
    receiver = params["retired_quick_change_v1_3"]["legacy_receiver_parameters"]
    flange_thickness = float(tool["blue_flange_thickness_mm"])
    # Use the analytically equivalent clean replica for printable outputs.  The
    # imported source remains the provenance/geometry check in the manifest.
    mount = build_clean_flange_replica(params)

    tool_face_z = -flange_thickness
    seat_z = tool_face_z - float(receiver["boss_length_below_blue_flange_mm"])
    boss_d = float(receiver["boss_outer_diameter_mm"])
    gusset_d = float(receiver["root_gusset_outer_diameter_mm"])
    gusset_h = float(receiver["root_gusset_height_mm"])

    boss = cylinder(boss_d, seat_z, tool_face_z - gusset_h + 0.15)
    gusset = circular_frustum(
        boss_d,
        gusset_d,
        tool_face_z - gusset_h,
        tool_face_z + 0.15,
    )
    modified = mount.union(boss).union(gusset)

    total_depth = float(receiver["female_hex_total_depth_mm"])
    entry_depth = float(receiver["female_hex_entry_depth_mm"])
    working_af = float(receiver["female_hex_working_af_mm"])
    entry_af = float(receiver["female_hex_entry_af_mm"])
    cavity_top_z = seat_z + total_depth
    entry_end_z = seat_z + entry_depth

    # A straight working pilot controls XY/yaw.  Only the first 2 mm is flared
    # for assembly; the hard shoulder at seat_z remains the axial datum.
    cavity = hex_loft(entry_af, working_af, seat_z - 0.2, entry_end_z)
    cavity = cavity.union(hex_prism(working_af, entry_end_z - 0.05, cavity_top_z))
    modified = modified.cut(cavity)

    # Tool-free positive retention: two horizontal channels admit a removable
    # U-clip around the cartridge's reduced neck.  The channel does not define
    # tool pose; the rigid shoulder and the short hex pilot remain the datum.
    lock = receiver["tool_free_retaining_clip"]
    groove_center_z = seat_z + float(lock["groove_center_above_seat_mm"])
    clip_t = float(lock["clip_thickness_mm"])
    channel_axial_clearance = float(lock["receiver_channel_axial_clearance_per_side_mm"])
    channel_z0 = groove_center_z - 0.5 * clip_t - channel_axial_clearance
    channel_z1 = groove_center_z + 0.5 * clip_t + channel_axial_clearance
    inner_gap = float(lock["clip_prong_inner_gap_mm"])
    prong_w = float(lock["clip_prong_width_mm"])
    channel_xy_clearance = float(lock["receiver_channel_xy_clearance_per_side_mm"])
    prong_center_y = 0.5 * (inner_gap + prong_w)
    clip_tip_x = float(lock["clip_tip_x_mm"])
    bridge_start_x = float(lock["clip_bridge_start_x_mm"])
    bridge_end_x = float(lock["clip_bridge_end_x_mm"])
    prong_channel_x0 = clip_tip_x - channel_xy_clearance
    prong_channel_x1 = bridge_end_x + channel_xy_clearance
    for sign in (-1.0, 1.0):
        channel = rectangular_prism(
            prong_channel_x1 - prong_channel_x0,
            prong_w + 2.0 * channel_xy_clearance,
            channel_z0,
            channel_z1,
            0.5 * (prong_channel_x0 + prong_channel_x1),
            sign * prong_center_y,
        )
        modified = modified.cut(channel)

    clip_outer_width = inner_gap + 2.0 * prong_w
    bridge_channel_x0 = bridge_start_x - channel_xy_clearance
    bridge_channel_x1 = bridge_end_x + channel_xy_clearance
    bridge_channel = rectangular_prism(
        bridge_channel_x1 - bridge_channel_x0,
        clip_outer_width + 2.0 * channel_xy_clearance,
        channel_z0,
        channel_z1,
        0.5 * (bridge_channel_x0 + bridge_channel_x1),
        0.0,
    )
    modified = modified.cut(bridge_channel)

    # Small pockets receive the clip's flexible detent bumps.  The long prongs
    # flex laterally during insertion and relax into these pockets at full seat.
    detent_x = float(lock["detent_x_mm"])
    detent_y = 0.5 * clip_outer_width
    detent_pocket_d = 2.0 * float(lock["detent_pocket_radius_mm"])
    for sign in (-1.0, 1.0):
        modified = modified.cut(
            cylinder(
                detent_pocket_d,
                channel_z0 - 0.05,
                channel_z1 + 0.05,
                detent_x,
                sign * detent_y,
            )
        )

    return modified


def build_cartridge(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
) -> cq.Workplane:
    tool = params["tool_reference"]
    receiver = params["retired_quick_change_v1_3"]["legacy_receiver_parameters"]
    peg_params = params["peg"]

    sensor_to_tip = float(tool["blue_sensor_plane_to_tip_mm"])
    flange_thickness = float(tool["blue_flange_thickness_mm"])
    seat_z = -flange_thickness - float(receiver["boss_length_below_blue_flange_mm"])
    tip_z = -sensor_to_tip

    shoulder_d = float(receiver["shoulder_diameter_mm"])
    shoulder_t = float(receiver["shoulder_thickness_mm"])
    shoulder_bottom_z = seat_z - shoulder_t
    shoulder = cylinder(shoulder_d, shoulder_bottom_z, seat_z)

    male_af = float(receiver["male_hex_working_af_mm"])
    lead_af = float(receiver["male_hex_lead_af_mm"])
    male_length = float(receiver["male_hex_length_mm"])
    lead_length = float(receiver["male_hex_lead_length_mm"])
    plug_top_z = seat_z + male_length
    lead_start_z = plug_top_z - lead_length
    lock = receiver["tool_free_retaining_clip"]
    groove_center_z = seat_z + float(lock["groove_center_above_seat_mm"])
    groove_h = float(lock["groove_axial_height_mm"])
    groove_bottom_z = groove_center_z - 0.5 * groove_h
    groove_top_z = groove_center_z + 0.5 * groove_h
    neck_d = float(lock["groove_neck_diameter_mm"])
    if groove_bottom_z <= seat_z or groove_top_z >= plug_top_z:
        raise ValueError("tool-free retaining groove does not fit inside the male pilot")

    plug = hex_prism(male_af, seat_z - 0.05, groove_bottom_z + 0.05)
    plug = plug.union(cylinder(neck_d, groove_bottom_z, groove_top_z))
    if groove_top_z < lead_start_z:
        plug = plug.union(hex_prism(male_af, groove_top_z - 0.05, lead_start_z + 0.05))
    plug = plug.union(hex_loft(male_af, lead_af, lead_start_z, plug_top_z))

    # Three short tapered ribs are intentionally local and sacrificial.  They
    # remove lateral play with a small printable contact patch while allowing
    # the hard shoulder to reach its stop.  They do not provide pull-out safety.
    ribs = receiver["local_interference_ribs"]
    rib_count = int(ribs["count"])
    rib_height = float(ribs["protrusion_mm"])
    rib_width = float(ribs["tangential_width_mm"])
    rib_length = float(ribs["axial_length_mm"])
    rib_ramp = float(ribs["end_ramp_mm"])
    rib_z0 = seat_z + 0.2
    rib_z1 = rib_z0 + rib_length
    if rib_z1 >= groove_bottom_z:
        raise ValueError("local interference ribs collide with the retaining groove")
    base_radius = 0.5 * male_af
    for index in range(rib_count):
        angle = index * 360.0 / rib_count
        rib = (
            cq.Workplane("XZ")
            .polyline(
                [
                    (base_radius, rib_z0),
                    (base_radius + rib_height, rib_z0 + rib_ramp),
                    (base_radius + rib_height, rib_z1 - rib_ramp),
                    (base_radius, rib_z1),
                ]
            )
            .close()
            .extrude(0.5 * rib_width, both=True)
            .rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle)
        )
        plug = plug.union(rib)

    bit_af = 1000.0 * float(sim_spec["key_af_m"])
    tip_taper_h = 1000.0 * float(sim_spec["bit_tip_chamfer_m"])
    tip_scale = float(sim_spec["bit_tip_scale"])
    uniform_length = float(peg_params["uniform_contact_length_mm"])
    transition_length = float(peg_params["reinforcement_transition_length_mm"])
    upper_af = max(bit_af, float(peg_params["reinforcement_upper_af_mm"]))
    collar_d = max(
        upper_af / math.cos(math.radians(30.0)),
        float(peg_params["root_collar_diameter_mm"]),
    )

    uniform_top_z = tip_z + uniform_length
    transition_top_z = uniform_top_z + transition_length
    collar_top_z = shoulder_bottom_z + 0.2
    if transition_top_z >= collar_top_z:
        raise ValueError(
            f"{size_name}: reinforcement stack collides with the shoulder; "
            "increase H or shorten the reinforcement"
        )

    peg = hex_loft(bit_af * tip_scale, bit_af, tip_z, tip_z + tip_taper_h)
    peg = peg.union(hex_prism(bit_af, tip_z + tip_taper_h - 0.03, uniform_top_z))
    peg = peg.union(hex_loft(bit_af, upper_af, uniform_top_z - 0.03, transition_top_z))
    peg = peg.union(
        circular_frustum(
            upper_af / math.cos(math.radians(30.0)),
            collar_d,
            transition_top_z - 0.03,
            min(transition_top_z + 2.0, collar_top_z),
        )
    )
    peg = peg.union(cylinder(collar_d, min(transition_top_z + 1.95, collar_top_z - 0.1), collar_top_z))

    return shoulder.union(plug).union(peg)


def build_tool_free_retaining_clip(params: Mapping[str, Any]) -> cq.Workplane:
    """Legacy v1.3 builder retained only for archived design reproduction."""
    receiver = params["retired_quick_change_v1_3"]["legacy_receiver_parameters"]
    lock = receiver["tool_free_retaining_clip"]
    thickness = float(lock["clip_thickness_mm"])
    inner_gap = float(lock["clip_prong_inner_gap_mm"])
    prong_w = float(lock["clip_prong_width_mm"])
    outer_width = inner_gap + 2.0 * prong_w
    tip_x = float(lock["clip_tip_x_mm"])
    bridge_start_x = float(lock["clip_bridge_start_x_mm"])
    bridge_end_x = float(lock["clip_bridge_end_x_mm"])
    handle_end_x = float(lock["clip_handle_end_x_mm"])
    handle_width = float(lock["clip_handle_width_mm"])

    clip = rectangular_prism(
        bridge_end_x - tip_x,
        outer_width,
        0.0,
        thickness,
        0.5 * (bridge_end_x + tip_x),
        0.0,
    )
    open_slot = rectangular_prism(
        bridge_start_x - tip_x + 0.4,
        inner_gap,
        -0.1,
        thickness + 0.1,
        0.5 * (bridge_start_x + tip_x - 0.4),
        0.0,
    )
    clip = clip.cut(open_slot)
    handle_x0 = bridge_end_x - 0.2
    clip = clip.union(
        rectangular_prism(
            handle_end_x - handle_x0,
            handle_width,
            0.0,
            thickness,
            0.5 * (handle_end_x + handle_x0),
            0.0,
        )
    )

    detent_x = float(lock["detent_x_mm"])
    detent_y = 0.5 * outer_width
    detent_d = 2.0 * float(lock["detent_bump_radius_mm"])
    for sign in (-1.0, 1.0):
        clip = clip.union(cylinder(detent_d, 0.0, thickness, detent_x, sign * detent_y))

    tether_x = handle_end_x - 0.35 * (handle_end_x - bridge_end_x)
    clip = clip.cut(
        cylinder(
            float(lock["clip_tether_hole_diameter_mm"]),
            -0.1,
            thickness + 0.1,
            tether_x,
            0.0,
        )
    )
    return clip


def retaining_clip_in_assembly(
    clip: cq.Workplane,
    params: Mapping[str, Any],
) -> cq.Workplane:
    tool = params["tool_reference"]
    receiver = params["retired_quick_change_v1_3"]["legacy_receiver_parameters"]
    lock = receiver["tool_free_retaining_clip"]
    seat_z = -float(tool["blue_flange_thickness_mm"]) - float(
        receiver["boss_length_below_blue_flange_mm"]
    )
    center_z = seat_z + float(lock["groove_center_above_seat_mm"])
    return clip.translate((0.0, 0.0, center_z - 0.5 * float(lock["clip_thickness_mm"])))


def resolve_socket_geometry_mm(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    socket_profile_name: str,
    printer_compensation_per_side_mm: float = 0.0,
) -> Dict[str, float]:
    """Resolve one explicit socket profile without hiding printer compensation.

    ``simulation_exact`` is checked against the MuJoCo source JSON.  Other
    profiles are deliberate hardware geometry variants.  Printer compensation
    translates both wall and top AF equally, so it does not alter the designed
    chamfer width or angle.
    """

    base = params["socket_base"]
    profiles = base["socket_geometry_profiles"]
    if socket_profile_name not in profiles:
        raise KeyError(f"unknown socket geometry profile: {socket_profile_name}")
    profile = profiles[socket_profile_name]

    source_wall_af = 1000.0 * float(sim_spec["derived"]["socket_wall_af_m"])
    source_top_af = 1000.0 * float(sim_spec["derived"]["socket_top_af_m"])
    source_width = 1000.0 * float(sim_spec["chamfer_width_m"])
    source_depth = 1000.0 * float(sim_spec["chamfer_depth_m"])
    if not math.isclose(
        source_top_af,
        source_wall_af + 2.0 * source_width,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{size_name}: MuJoCo top AF is inconsistent with chamfer width")

    wall_expansion = float(profile["wall_af_expansion_per_side_mm"])
    chamfer_width = float(profile["entry_chamfer_width_mm"][size_name])
    chamfer_depth = float(profile["entry_chamfer_depth_mm"][size_name])
    head_scale = float(profile["head_outer_diameter_scale"])
    support_neck_scale = float(profile["support_neck_diameter_scale"])
    if socket_profile_name == "simulation_exact":
        if not math.isclose(wall_expansion, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("simulation_exact cannot expand the MuJoCo wall AF")
        if not math.isclose(chamfer_width, source_width, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"{size_name}: simulation_exact chamfer width drifted from MuJoCo")
        if not math.isclose(chamfer_depth, source_depth, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"{size_name}: simulation_exact chamfer depth drifted from MuJoCo")
        if not math.isclose(head_scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("simulation_exact cannot scale the MuJoCo head diameter")
        if not math.isclose(support_neck_scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("simulation_exact cannot scale the support neck")
    if chamfer_width <= 0.0 or chamfer_depth <= 0.0:
        raise ValueError(f"{size_name}: socket chamfer dimensions must be positive")
    if not math.isclose(
        chamfer_width,
        chamfer_depth,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            f"{size_name}/{socket_profile_name}: the active socket profile must keep a 45-degree entry chamfer"
        )

    total_af_translation = 2.0 * (
        wall_expansion + float(printer_compensation_per_side_mm)
    )
    wall_af = source_wall_af + total_af_translation
    top_af = wall_af + 2.0 * chamfer_width
    source_head_diameter = 1000.0 * float(sim_spec["screw_head_diameter_m"])
    head_diameter = source_head_diameter * head_scale
    head_radius = 0.5 * head_diameter
    top_corner_radius = top_af / math.sqrt(3.0)
    if top_corner_radius >= head_radius - 0.5:
        raise ValueError(
            f"{size_name}: socket profile leaves less than 0.5 mm radial top rim"
        )

    return {
        "wall_af_mm": wall_af,
        "top_af_mm": top_af,
        "entry_chamfer_width_mm": chamfer_width,
        "entry_chamfer_depth_mm": chamfer_depth,
        "printer_compensation_per_side_mm": float(printer_compensation_per_side_mm),
        "source_wall_af_mm": source_wall_af,
        "source_top_af_mm": source_top_af,
        "source_head_outer_diameter_mm": source_head_diameter,
        "head_outer_diameter_scale": head_scale,
        "head_outer_diameter_mm": head_diameter,
        "support_neck_diameter_scale": support_neck_scale,
    }


def build_socket_head(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    socket_profile_name: str,
    printer_compensation_per_side_mm: float,
    include_exterior_ribs: bool = True,
) -> cq.Workplane:
    base = params["socket_base"]
    center_x, center_y = [float(value) for value in base["socket_centers_mm"][size_name]]
    top_z = float(base["socket_top_z_mm"])
    plate_top_z = float(base["plate_top_z_mm"])

    geometry = resolve_socket_geometry_mm(
        size_name,
        sim_spec,
        params,
        socket_profile_name,
        printer_compensation_per_side_mm,
    )
    head_d = geometry["head_outer_diameter_mm"]
    head_h = 1000.0 * float(sim_spec["screw_head_height_m"])
    head_bottom_z = top_z - head_h
    head = cylinder(head_d, head_bottom_z, top_z, center_x, center_y)

    neck_d = (
        float(base["support_neck_diameter_mm"][size_name])
        * geometry["support_neck_diameter_scale"]
    )
    if head_bottom_z > plate_top_z:
        head = head.union(cylinder(neck_d, plate_top_z - 0.1, head_bottom_z + 0.1, center_x, center_y))

    rib_params = base["exterior_vertical_ribs"]
    if include_exterior_ribs and bool(rib_params["enabled"]):
        guard_band = float(rib_params["smooth_guard_band_below_socket_top_mm"])
        bottom_margin = float(rib_params["bottom_margin_mm"])
        rib_z0 = head_bottom_z + bottom_margin
        rib_z1 = top_z - guard_band
        ramp = float(rib_params["end_ramp_mm"])
        if rib_z1 - rib_z0 <= 2.0 * ramp:
            raise ValueError(f"{size_name}: exterior rib span is too short for its ramps")
        rib_count = int(rib_params["count"][size_name])
        rib_root_width = float(rib_params["root_tangential_width_mm"])
        rib_tip_width = float(rib_params["tip_tangential_width_mm"])
        rib_height = float(rib_params["radial_height_mm"])
        head_radius = 0.5 * head_d
        root_overlap = float(rib_params["root_overlap_mm"])
        rib_root_radius = head_radius - root_overlap
        rib_outer_radius = head_radius + rib_height
        for index in range(rib_count):
            angle = index * 360.0 / rib_count
            axial_profile = (
                cq.Workplane("XZ")
                .polyline(
                    [
                        (rib_root_radius, rib_z0),
                        (rib_outer_radius, rib_z0 + ramp),
                        (rib_outer_radius, rib_z1 - ramp),
                        (rib_root_radius, rib_z1),
                    ]
                )
                .close()
                .extrude(0.5 * rib_root_width, both=True)
            )
            plan_profile = (
                cq.Workplane("XY")
                .polyline(
                    [
                        (rib_root_radius, -0.5 * rib_root_width),
                        (rib_outer_radius, -0.5 * rib_tip_width),
                        (rib_outer_radius, 0.5 * rib_tip_width),
                        (rib_root_radius, 0.5 * rib_root_width),
                    ]
                )
                .close()
                .extrude(rib_z1 - rib_z0)
                .translate((0.0, 0.0, rib_z0))
            )
            rib = (
                axial_profile.intersect(plan_profile)
                .rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle)
                .translate((center_x, center_y, 0.0))
            )
            head = head.union(rib)

    wall_af = geometry["wall_af_mm"]
    top_af = geometry["top_af_mm"]
    chamfer_depth = geometry["entry_chamfer_depth_mm"]
    socket_depth = 1000.0 * float(sim_spec["socket_depth_m"])

    cavity_bottom_z = top_z - socket_depth
    chamfer_bottom_z = top_z - chamfer_depth
    cavity = hex_prism(wall_af, cavity_bottom_z, chamfer_bottom_z + 0.03)
    # The target top-AF section must lie exactly on the physical socket mouth
    # at top_z.  Extending the taper itself to top_z + 0.15 would place the
    # target section above the part and silently shrink the real entry.  The
    # constant-AF prism is only a robust Boolean overcut above the top plane.
    cavity = cavity.union(hex_loft(wall_af, top_af, chamfer_bottom_z, top_z))
    cavity = cavity.union(hex_prism(top_af, top_z, top_z + 0.15))
    cavity = cavity.translate((center_x, center_y, 0.0))
    head = head.cut(cavity)

    return head


def build_socket_base(
    specs: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
    socket_profile_name: str = "simulation_exact",
    printer_compensation_per_side_mm: float = 0.0,
    include_exterior_ribs: bool = True,
) -> cq.Workplane:
    base = params["socket_base"]
    plate_top_z = float(base["plate_top_z_mm"])
    plate_bottom_z = plate_top_z - float(base["plate_thickness_mm"])
    fixture = rounded_plate(
        float(base["size_x_mm"]),
        float(base["size_y_mm"]),
        plate_bottom_z,
        plate_top_z,
        float(base["corner_radius_mm"]),
    )

    for size_name in SIZE_ORDER:
        fixture = fixture.union(
            build_socket_head(
                size_name,
                specs[size_name],
                params,
                socket_profile_name,
                printer_compensation_per_side_mm,
                include_exterior_ribs,
            )
        )

    # Cut the three center bores only after every head has been fused to the
    # common plate.  Cutting them inside each head before the union allowed the
    # plate to fill the lower portion, which looked like a pilot dot but was not
    # a true through-hole.
    if not bool(base["center_through_hole_final_body_cut"]):
        raise ValueError("the hardware fixture requires final-body center through-holes")
    center_hole_d = float(base["center_through_hole_diameter_mm"])
    for size_name in SIZE_ORDER:
        center_x, center_y = [float(value) for value in base["socket_centers_mm"][size_name]]
        socket_depth = 1000.0 * float(specs[size_name]["socket_depth_m"])
        cavity_bottom_z = float(base["socket_top_z_mm"]) - socket_depth
        fixture = fixture.cut(
            cylinder(
                center_hole_d,
                plate_bottom_z - 0.2,
                cavity_bottom_z + 0.2,
                center_x,
                center_y,
            )
        )

    mount_hole_d = float(base["mount_hole_diameter_mm"])
    for x, y in base["mount_hole_centers_mm"]:
        fixture = fixture.cut(cylinder(mount_hole_d, plate_bottom_z - 0.2, plate_top_z + 0.2, float(x), float(y)))

    datum_d = float(base["datum_round_hole_diameter_mm"])
    datum_x, datum_y = [float(value) for value in base["datum_round_hole_center_mm"]]
    fixture = fixture.cut(cylinder(datum_d, plate_bottom_z - 0.2, plate_top_z + 0.2, datum_x, datum_y))

    slot_width = float(base["datum_slot_width_mm"])
    slot_length = float(base["datum_slot_length_mm"])
    slot_x, slot_y = [float(value) for value in base["datum_slot_center_mm"]]
    slot = (
        cq.Workplane("XY")
        .center(slot_x, slot_y)
        .slot2D(slot_length, slot_width, angle=0.0)
        .extrude(float(base["plate_thickness_mm"]) + 0.4)
        .translate((0.0, 0.0, plate_bottom_z - 0.2))
    )
    fixture = fixture.cut(slot)

    # Large engraved identification survives FDM handling better than the tiny
    # raised v1.3 text observed in the first physical print.  It remains on the
    # recessed plate surface, outside the active contact geometry.
    markings = base["markings"]
    if markings["mode"] != "engraved":
        raise ValueError("active integrated tooling expects engraved fixture markings")
    marking_depth = float(markings["depth_mm"])
    marking_z = plate_top_z - marking_depth
    font_name = str(markings["font"])
    font_kind = str(markings["kind"])
    label_y = float(markings["size_label_y_mm"])
    for size_name in SIZE_ORDER:
        center_x, _ = [float(value) for value in base["socket_centers_mm"][size_name]]
        label = (
            cq.Workplane("XY", origin=(0.0, 0.0, marking_z))
            .center(center_x, label_y)
            .text(
                size_name.upper(),
                float(markings["size_label_font_mm"]),
                marking_depth + 0.05,
                combine=True,
                font=font_name,
                kind=font_kind,
                halign="center",
                valign="center",
            )
        )
        fixture = fixture.cut(label)

    arrow_y = float(markings["arrow_y_mm"])
    shaft_length = float(markings["arrow_shaft_length_mm"])
    shaft_width = float(markings["arrow_shaft_width_mm"])
    shaft_center_x = 5.0
    shaft_end_x = shaft_center_x + 0.5 * shaft_length
    head_length = float(markings["arrow_head_length_mm"])
    head_width = float(markings["arrow_head_width_mm"])
    arrow_shaft = (
        cq.Workplane("XY")
        .box(shaft_length, shaft_width, marking_depth + 0.05, centered=(True, True, False))
        .translate((shaft_center_x, arrow_y, marking_z))
    )
    arrow_head = (
        cq.Workplane("XY")
        .polyline(
            [
                (shaft_end_x - 0.5, arrow_y - 0.5 * head_width),
                (shaft_end_x + head_length, arrow_y),
                (shaft_end_x - 0.5, arrow_y + 0.5 * head_width),
            ]
        )
        .close()
        .extrude(marking_depth + 0.05)
        .translate((0.0, 0.0, marking_z))
    )
    axis_label = (
        cq.Workplane("XY", origin=(0.0, 0.0, marking_z))
        .center(43.0, arrow_y)
        .text(
            "+X",
            float(markings["axis_label_font_mm"]),
            marking_depth + 0.05,
            combine=True,
            font=font_name,
            kind=font_kind,
            halign="center",
            valign="center",
        )
    )
    return fixture.cut(arrow_shaft).cut(arrow_head).cut(axis_label)


def build_socket_coupon(
    size_name: str,
    sim_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    socket_profile_name: str,
) -> cq.Workplane:
    offsets = [float(value) for value in params["fit_coupons"]["socket_total_af_offsets_mm"]]
    pitch = 20.0
    plate_x = pitch * (len(offsets) - 1) + 24.0
    plate_y = 24.0
    plate_t = 4.0
    coupon = rounded_plate(plate_x, plate_y, 0.0, plate_t, 2.0)
    geometry = resolve_socket_geometry_mm(
        size_name,
        sim_spec,
        params,
        socket_profile_name,
        0.0,
    )
    target_af = geometry["wall_af_mm"]
    chamfer_width = geometry["entry_chamfer_width_mm"]
    chamfer_depth = geometry["entry_chamfer_depth_mm"]
    chamfer_bottom_z = plate_t - chamfer_depth
    start_x = -0.5 * pitch * (len(offsets) - 1)
    for index, offset in enumerate(offsets):
        wall_af = target_af + offset
        top_af = wall_af + 2.0 * chamfer_width
        hole = hex_prism(wall_af, -0.2, chamfer_bottom_z + 0.03)
        hole = hole.union(hex_loft(wall_af, top_af, chamfer_bottom_z, plate_t))
        hole = hole.union(hex_prism(top_af, plate_t, plate_t + 0.2))
        coupon = coupon.cut(hole.translate((start_x + index * pitch, 0.0, 0.0)))
    return coupon


def build_quick_change_coupon(params: Mapping[str, Any]) -> cq.Workplane:
    receiver = params["retired_quick_change_v1_3"]["legacy_receiver_parameters"]
    clearances = [
        float(value)
        for value in params["retired_quick_change_v1_3"][
            "legacy_quick_change_radial_clearances_mm"
        ]
    ]
    pitch = 27.0
    block_x = 24.0
    block_y = 23.0
    block_h = 8.0
    plate = rounded_plate(
        pitch * (len(clearances) - 1) + block_x,
        block_y,
        0.0,
        block_h,
        2.0,
    )
    working_af = float(receiver["male_hex_working_af_mm"])
    start_x = -0.5 * pitch * (len(clearances) - 1)
    for index, clearance in enumerate(clearances):
        female_af = working_af + 2.0 * clearance
        hole = hex_prism(female_af, -0.2, block_h + 0.2)
        plate = plate.cut(hole.translate((start_x + index * pitch, 0.0, 0.0)))
    return plate


def shape_record(name: str, shape: cq.Workplane) -> Dict[str, Any]:
    value = shape.val()
    bbox = value.BoundingBox()
    return {
        "name": name,
        "is_valid": bool(value.isValid()),
        "solid_count": len(value.Solids()),
        "volume_mm3": value.Volume(),
        "bounding_box_mm": {
            "x": [bbox.xmin, bbox.xmax],
            "y": [bbox.ymin, bbox.ymax],
            "z": [bbox.zmin, bbox.zmax],
        },
    }


def export_part(name: str, shape: cq.Workplane, params: Mapping[str, Any]) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_path = OUTPUT_DIR / f"{name}.step"
    stl_path = OUTPUT_DIR / f"{name}.stl"
    three_mf_path = OUTPUT_DIR / f"{name}.3mf"
    svg_path = OUTPUT_DIR / f"{name}.svg"
    cq.exporters.export(shape, str(step_path))
    cq.exporters.export(
        shape,
        str(stl_path),
        tolerance=float(params["export"]["stl_linear_tolerance_mm"]),
        angularTolerance=float(params["export"]["stl_angular_tolerance_rad"]),
    )
    cq.exporters.export(
        shape,
        str(three_mf_path),
        tolerance=float(params["export"]["stl_linear_tolerance_mm"]),
        angularTolerance=float(params["export"]["stl_angular_tolerance_rad"]),
    )
    try:
        cq.exporters.export(shape, str(svg_path))
    except Exception:
        svg_path = None

    reloaded = cq.importers.importStep(str(step_path))
    record = shape_record(name, shape)
    record["step_reimport_valid"] = bool(reloaded.val().isValid())
    record["step_reimport_volume_mm3"] = reloaded.val().Volume()
    record["files"] = {
        "step": {"path": step_path.name, "sha256": sha256(step_path)},
        "stl": {"path": stl_path.name, "sha256": sha256(stl_path)},
        "3mf": {"path": three_mf_path.name, "sha256": sha256(three_mf_path)},
    }
    if svg_path is not None and svg_path.exists():
        record["files"]["svg"] = {"path": svg_path.name, "sha256": sha256(svg_path)}
    return record


def export_assembly(
    name: str,
    receiver: cq.Workplane,
    cartridge: cq.Workplane,
    retaining_clip: cq.Workplane,
) -> Dict[str, Any]:
    path = OUTPUT_DIR / f"{name}.step"
    assembly = cq.Assembly(name=name)
    assembly.add(receiver, name="modified_blue_flange", color=cq.Color(0.15, 0.45, 0.95))
    assembly.add(cartridge, name="replaceable_cartridge", color=cq.Color(0.95, 0.65, 0.15))
    assembly.add(retaining_clip, name="tool_free_retaining_clip", color=cq.Color(0.20, 0.75, 0.40))
    assembly.save(str(path))
    return {"path": path.name, "sha256": sha256(path)}


def assert_geometry(
    tools: Mapping[str, cq.Workplane],
    bases: Mapping[str, cq.Workplane],
) -> None:
    all_shapes = {**tools, **bases}
    for name, shape in all_shapes.items():
        value = shape.val()
        if not value.isValid():
            raise RuntimeError(f"{name} is not a valid B-rep")
        if len(value.Solids()) != 1:
            raise RuntimeError(f"{name} must contain exactly one solid")

    for tool_name, tool in tools.items():
        bbox = tool.val().BoundingBox()
        if abs(bbox.zmax) > 1.0e-6 or abs(bbox.zmin + 42.0) > 0.05:
            raise RuntimeError(
                f"{tool_name} must span the common blue-plane/tip datum [0,-42] mm; "
                f"got [{bbox.zmax:.6f},{bbox.zmin:.6f}]"
            )


def measure_socket_cavity_af_mm(
    model: cq.Workplane,
    center_x_mm: float,
    center_y_mm: float,
    z_mm: float,
) -> float:
    """Measure the six-edge inner wire in a re-imported STEP section."""

    section = model.section(height=z_mm)
    candidates = []
    for wire in section.wires().vals():
        if len(wire.Edges()) != 6:
            continue
        box = wire.BoundingBox()
        center_x = 0.5 * (box.xmin + box.xmax)
        center_y = 0.5 * (box.ymin + box.ymax)
        distance_sq = (center_x - center_x_mm) ** 2 + (center_y - center_y_mm) ** 2
        candidates.append((distance_sq, box.xlen))
    if not candidates:
        raise RuntimeError(
            f"no six-edge socket wire found at ({center_x_mm}, {center_y_mm}, {z_mm})"
        )
    _, across_flats = min(candidates, key=lambda row: row[0])
    return float(across_flats)


def measure_center_hex_section_mm(
    model: cq.Workplane,
    z_mm: float,
    target_flat_normal_yaw_deg: float,
) -> Dict[str, float]:
    """Measure a centered hex without assuming its flats are axis-aligned."""

    candidates = []
    for wire in model.section(height=z_mm).wires().vals():
        if len(wire.Edges()) != 6:
            continue
        box = wire.BoundingBox()
        center_distance_sq = (0.5 * (box.xmin + box.xmax)) ** 2 + (
            0.5 * (box.ymin + box.ymax)
        ) ** 2
        candidates.append((center_distance_sq, wire))
    if not candidates:
        raise RuntimeError(f"no centered six-edge tool wire found at Z={z_mm:.6f} mm")
    _, wire = min(candidates, key=lambda row: row[0])
    box = wire.BoundingBox()
    center_x = 0.5 * (box.xmin + box.xmax)
    center_y = 0.5 * (box.ymin + box.ymax)

    flat_distances = []
    flat_normal_angles = []
    corner_radii = []
    for edge in wire.Edges():
        vertices = edge.Vertices()
        if len(vertices) != 2:
            raise RuntimeError(
                f"non-linear edge found in centered tool hex at Z={z_mm:.6f} mm"
            )
        x0, y0 = float(vertices[0].X), float(vertices[0].Y)
        x1, y1 = float(vertices[1].X), float(vertices[1].Y)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1.0e-12:
            raise RuntimeError(
                f"degenerate edge found in centered tool hex at Z={z_mm:.6f} mm"
            )
        flat_distances.append(
            abs(dx * (center_y - y0) - dy * (center_x - x0)) / length
        )
        midpoint_x = 0.5 * (x0 + x1) - center_x
        midpoint_y = 0.5 * (y0 + y1) - center_y
        flat_normal_angles.append(math.degrees(math.atan2(midpoint_y, midpoint_x)))
        corner_radii.extend(
            (
                math.hypot(x0 - center_x, y0 - center_y),
                math.hypot(x1 - center_x, y1 - center_y),
            )
        )

    period_deg = 60.0

    def periodic_delta_deg(value_deg: float, reference_deg: float) -> float:
        return (
            (value_deg - reference_deg + 0.5 * period_deg) % period_deg
            - 0.5 * period_deg
        )

    residuals = [
        periodic_delta_deg(angle, target_flat_normal_yaw_deg)
        for angle in flat_normal_angles
    ]
    measured_yaw = target_flat_normal_yaw_deg + sum(residuals) / len(residuals)
    spread = max(
        abs(periodic_delta_deg(angle, measured_yaw))
        for angle in flat_normal_angles
    )
    return {
        "center_x_mm": center_x,
        "center_y_mm": center_y,
        "across_flats_mm": 2.0 * sum(flat_distances) / len(flat_distances),
        "corner_span_mm": 2.0 * max(corner_radii),
        "selected_flat_normal_yaw_deg": measured_yaw,
        "flat_normal_yaw_spread_deg": spread,
    }


def validate_exported_integrated_tools(
    specs: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
    clean_flange: cq.Workplane,
    tip_profile_name: str,
) -> Dict[str, Any]:
    """Re-import each complete tool and verify datum, flange, and contact hex."""

    report: Dict[str, Any] = {}
    clean_flange_volume = clean_flange.val().Volume()
    tool_params = params["tool_reference"]
    peg_params = params["peg"]
    coordinate_system = params["coordinate_system"]
    tip_z = -float(tool_params["blue_sensor_plane_to_tip_mm"])
    uniform_top_z = tip_z + float(peg_params["uniform_contact_length_mm"])
    transition_length = float(peg_params["reinforcement_transition_length_mm"])
    target_flat_yaw_deg = float(
        coordinate_system["integrated_peg_selected_hex_flat_normal_deg"]
    )
    flange_reference_yaw_deg = float(
        coordinate_system["flange_reference_tangent_normal_deg"]
    )
    yaw_tolerance_deg = float(peg_params["cad_flat_normal_tolerance_deg"])
    tolerance_mm = 1.0e-3
    for size_name in SIZE_ORDER:
        step_path = OUTPUT_DIR / f"integrated_flange_peg_{size_name}.step"
        reloaded = cq.importers.importStep(str(step_path))
        solid = reloaded.val()
        if not solid.isValid() or len(solid.Solids()) != 1:
            raise RuntimeError(f"{size_name}: integrated tool STEP is not one valid solid")
        bbox = solid.BoundingBox()
        tip_geometry = resolve_peg_tip_geometry_mm(
            size_name,
            specs[size_name],
            params,
            tip_profile_name,
        )
        bit_af = tip_geometry["shaft_af_mm"]
        tip_face_af = tip_geometry["tip_face_af_mm"]
        tip_taper_h = tip_geometry["tip_chamfer_axial_length_mm"]
        tip_fraction = 0.25
        tip_sample_z = tip_z + tip_fraction * tip_taper_h
        tip_sample = measure_center_hex_section_mm(
            reloaded,
            tip_sample_z,
            target_flat_yaw_deg,
        )
        expected_tip_sample_af = tip_face_af + tip_fraction * (
            bit_af - tip_face_af
        )
        shaft_sample_z = min(tip_z + tip_taper_h + 0.2, uniform_top_z - 0.2)
        shaft_sample = measure_center_hex_section_mm(
            reloaded,
            shaft_sample_z,
            target_flat_yaw_deg,
        )
        transition_sample_z = uniform_top_z + 0.5 * transition_length
        transition_sample = measure_center_hex_section_mm(
            reloaded,
            transition_sample_z,
            target_flat_yaw_deg,
        )
        flange_overlap_volume = reloaded.intersect(clean_flange).val().Volume()
        checks = {
            "zmax_error_mm": abs(float(bbox.zmax)),
            "tip_z_error_mm": abs(float(bbox.zmin) - tip_z),
            "tip_sample_af_error_mm": abs(
                tip_sample["across_flats_mm"] - expected_tip_sample_af
            ),
            "uniform_shaft_af_error_mm": abs(
                shaft_sample["across_flats_mm"] - bit_af
            ),
            "tip_center_error_mm": math.hypot(
                tip_sample["center_x_mm"],
                tip_sample["center_y_mm"],
            ),
            "shaft_center_error_mm": math.hypot(
                shaft_sample["center_x_mm"],
                shaft_sample["center_y_mm"],
            ),
            "tip_flat_normal_yaw_error_deg": abs(
                tip_sample["selected_flat_normal_yaw_deg"]
                - target_flat_yaw_deg
            ),
            "shaft_flat_normal_yaw_error_deg": abs(
                shaft_sample["selected_flat_normal_yaw_deg"]
                - target_flat_yaw_deg
            ),
            "transition_flat_normal_yaw_error_deg": abs(
                transition_sample["selected_flat_normal_yaw_deg"]
                - target_flat_yaw_deg
            ),
            "selected_flat_to_flange_reference_error_deg": abs(
                shaft_sample["selected_flat_normal_yaw_deg"]
                - flange_reference_yaw_deg
            ),
            "clean_flange_overlap_volume_error_mm3": abs(
                flange_overlap_volume - clean_flange_volume
            ),
        }
        passed = (
            checks["zmax_error_mm"] <= tolerance_mm
            and checks["tip_z_error_mm"] <= tolerance_mm
            and checks["tip_sample_af_error_mm"] <= tolerance_mm
            and checks["uniform_shaft_af_error_mm"] <= tolerance_mm
            and checks["tip_center_error_mm"] <= tolerance_mm
            and checks["shaft_center_error_mm"] <= tolerance_mm
            and checks["tip_flat_normal_yaw_error_deg"] <= yaw_tolerance_deg
            and checks["shaft_flat_normal_yaw_error_deg"] <= yaw_tolerance_deg
            and checks["transition_flat_normal_yaw_error_deg"] <= yaw_tolerance_deg
            and checks["selected_flat_to_flange_reference_error_deg"]
            <= yaw_tolerance_deg
            and checks["clean_flange_overlap_volume_error_mm3"] <= 0.05
        )
        report[size_name] = {
            "step_file": step_path.name,
            "common_tip_z_mm": float(bbox.zmin),
            "sensor_mating_plane_z_mm": float(bbox.zmax),
            "uniform_contact_length_mm": float(peg_params["uniform_contact_length_mm"]),
            "expected_contact_af_mm": bit_af,
            "tip_profile_name": tip_profile_name,
            "target_tip_geometry_mm": tip_geometry,
            "tip_sample_z_mm": tip_sample_z,
            "expected_tip_sample_af_mm": expected_tip_sample_af,
            "measured_tip_section": tip_sample,
            "shaft_sample_z_mm": shaft_sample_z,
            "measured_uniform_shaft_section": shaft_sample,
            "transition_sample_z_mm": transition_sample_z,
            "measured_transition_section": transition_sample,
            "target_selected_flat_normal_yaw_deg": target_flat_yaw_deg,
            "target_flange_reference_tangent_normal_yaw_deg": (
                flange_reference_yaw_deg
            ),
            "flat_normal_yaw_tolerance_deg": yaw_tolerance_deg,
            "clean_flange_volume_mm3": clean_flange_volume,
            "clean_flange_overlap_volume_mm3": flange_overlap_volume,
            "errors": checks,
            "pass": passed,
        }
        if not passed:
            raise RuntimeError(f"{size_name}: exported integrated tool validation failed: {checks}")
    return report


def validate_exported_socket_profiles(
    specs: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
    base_profile_contracts: Mapping[str, Tuple[str, float]],
) -> Dict[str, Any]:
    """Re-import each STEP and verify wall, mid-chamfer, and mouth sections."""

    base = params["socket_base"]
    top_z = float(base["socket_top_z_mm"])
    report: Dict[str, Any] = {}
    tolerance_mm = 1.0e-3
    for base_name, (profile_name, printer_compensation) in base_profile_contracts.items():
        step_path = OUTPUT_DIR / f"{base_name}.step"
        reloaded = cq.importers.importStep(str(step_path))
        solid = reloaded.val()
        if not solid.isValid() or len(solid.Solids()) != 1:
            raise RuntimeError(f"{base_name}: exported STEP is not one valid solid")
        size_report: Dict[str, Any] = {}
        for size_name in SIZE_ORDER:
            sim_spec = specs[size_name]
            geometry = resolve_socket_geometry_mm(
                size_name,
                sim_spec,
                params,
                profile_name,
                printer_compensation,
            )
            center_x, center_y = [
                float(value) for value in base["socket_centers_mm"][size_name]
            ]
            wall_af = geometry["wall_af_mm"]
            top_af = geometry["top_af_mm"]
            depth = geometry["entry_chamfer_depth_mm"]
            epsilon = 1.0e-4
            samples = {
                "mouth_z0_minus": {
                    "z_mm": top_z - epsilon,
                    "expected_af_mm": top_af
                    - (top_af - wall_af) * epsilon / depth,
                },
                "chamfer_mid": {
                    "z_mm": top_z - 0.5 * depth,
                    "expected_af_mm": 0.5 * (wall_af + top_af),
                },
                "straight_wall": {
                    "z_mm": top_z - depth - 0.001,
                    "expected_af_mm": wall_af,
                },
            }
            max_error = 0.0
            for sample in samples.values():
                measured = measure_socket_cavity_af_mm(
                    reloaded,
                    center_x,
                    center_y,
                    float(sample["z_mm"]),
                )
                sample["measured_af_mm"] = measured
                sample["absolute_error_mm"] = abs(
                    measured - float(sample["expected_af_mm"])
                )
                max_error = max(max_error, float(sample["absolute_error_mm"]))

            peg_af = 1000.0 * float(sim_spec["key_af_m"])
            sim_top_af = 1000.0 * float(sim_spec["derived"]["socket_top_af_m"])
            sim_capture = 0.5 * (sim_top_af - peg_af)
            size_report[size_name] = {
                "target_geometry_mm": geometry,
                "wall_clearance_per_side_mm": 0.5 * (wall_af - peg_af),
                "mouth_capture_allowance_per_side_mm": 0.5 * (top_af - peg_af),
                "mouth_capture_allowance_vs_sim_ratio": (
                    0.5 * (top_af - peg_af) / sim_capture
                ),
                "chamfer_angle_from_axis_deg": math.degrees(
                    math.atan2(geometry["entry_chamfer_width_mm"], depth)
                ),
                "step_section_samples": samples,
                "maximum_section_error_mm": max_error,
                "pass": max_error <= tolerance_mm,
            }
            if max_error > tolerance_mm:
                raise RuntimeError(
                    f"{base_name}/{size_name} exported socket profile error "
                    f"{max_error:.6f} mm exceeds {tolerance_mm:.6f} mm"
                )
        report[base_name] = {
            "profile_name": profile_name,
            "printer_compensation_per_side_mm": printer_compensation,
            "section_tolerance_mm": tolerance_mm,
            "sizes": size_report,
            "pass": all(bool(row["pass"]) for row in size_report.values()),
        }
    return report


def evaluate_combined_hardware_funnel(
    specs: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
    socket_profile_name: str,
    tip_profile_name: str,
) -> Dict[str, Any]:
    """Quantify the easy socket + easy peg geometry without claiming capture."""

    rows: Dict[str, Any] = {}
    working_length = float(params["peg"]["uniform_contact_length_mm"])
    for size_name in SIZE_ORDER:
        sim_spec = specs[size_name]
        socket = resolve_socket_geometry_mm(
            size_name,
            sim_spec,
            params,
            socket_profile_name,
            0.0,
        )
        peg = resolve_peg_tip_geometry_mm(
            size_name,
            sim_spec,
            params,
            tip_profile_name,
        )
        shaft_af = peg["shaft_af_mm"]
        wall_clearance = 0.5 * (socket["wall_af_mm"] - shaft_af)
        geometric_guiding_allowance = (
            wall_clearance
            + socket["entry_chamfer_width_mm"]
            + peg["tip_chamfer_radial_setback_mm"]
        )
        source_wall_clearance = 0.5 * (
            socket["source_wall_af_mm"] - shaft_af
        )
        source_guiding_allowance = (
            source_wall_clearance
            + 1000.0 * float(sim_spec["chamfer_width_m"])
            + peg["source_tip_chamfer_radial_setback_mm"]
        )
        yaw_ratio = socket["wall_af_mm"] * math.sqrt(3.0) / (2.0 * shaft_af)
        centered_yaw_free_play = 30.0 if yaw_ratio >= 1.0 else 30.0 - math.degrees(
            math.acos(yaw_ratio)
        )
        socket_depth = 1000.0 * float(sim_spec["socket_depth_m"])
        full_af_straight_length = working_length - peg[
            "tip_chamfer_axial_length_mm"
        ]
        rows[size_name] = {
            "socket_profile_name": socket_profile_name,
            "peg_tip_profile_name": tip_profile_name,
            "shaft_af_mm": shaft_af,
            "tip_face_af_mm": peg["tip_face_af_mm"],
            "tip_face_area_ratio_vs_shaft": (
                peg["tip_face_af_mm"] / shaft_af
            )
            ** 2,
            "socket_wall_af_mm": socket["wall_af_mm"],
            "socket_mouth_af_mm": socket["top_af_mm"],
            "deep_wall_clearance_per_side_mm": wall_clearance,
            "socket_chamfer_width_and_depth_mm": socket[
                "entry_chamfer_width_mm"
            ],
            "peg_chamfer_width_and_depth_mm": peg[
                "tip_chamfer_radial_setback_mm"
            ],
            "aligned_flat_normal_geometric_guiding_allowance_mm": geometric_guiding_allowance,
            "source_control_guiding_allowance_mm": source_guiding_allowance,
            "guiding_allowance_vs_source_ratio": geometric_guiding_allowance
            / source_guiding_allowance,
            "centered_ideal_hex_yaw_free_play_each_side_deg": centered_yaw_free_play,
            "working_length_from_tip_to_reinforcement_mm": working_length,
            "full_af_straight_length_mm": full_af_straight_length,
            "socket_depth_mm": socket_depth,
            "full_af_length_beyond_socket_depth_mm": full_af_straight_length
            - socket_depth,
            "centered_straight_wall_tilt_clearance_approx_deg": math.degrees(
                math.atan2(wall_clearance, socket_depth)
            ),
            "pass": (
                math.isclose(
                    socket["entry_chamfer_width_mm"],
                    socket["entry_chamfer_depth_mm"],
                    abs_tol=1.0e-9,
                )
                and math.isclose(
                    peg["tip_chamfer_radial_setback_mm"],
                    peg["tip_chamfer_axial_length_mm"],
                    abs_tol=1.0e-9,
                )
                and full_af_straight_length > socket_depth
                and peg["tip_face_af_mm"] / shaft_af >= 0.8 - 1.0e-9
            ),
        }
        if not rows[size_name]["pass"]:
            raise RuntimeError(f"{size_name}: combined hardware funnel check failed")
    return {
        "sizes": rows,
        "interpretation": "The allowance is a flat-normal, yaw-aligned geometric estimate for first contact with the guiding region; it is not a guaranteed capture radius.",
        "unmodeled_failure_causes": [
            "friction and FDM layer ridges",
            "tip/socket yaw mismatch",
            "tool tilt and multi-point edge contact",
            "burrs, elephant foot, and as-printed dimensional error",
            "insufficient XY/tilt compliance or excessive axial force",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-manifest-only",
        action="store_true",
        help="Reserved for CI compatibility; full geometry is still regenerated.",
    )
    args = parser.parse_args()
    del args

    params = load_json(PARAMETERS_FILE)
    specs = {size: load_json(path) for size, path in SOURCE_CONFIGS.items()}
    remove_retired_active_outputs()

    source_flange = transformed_source_flange(params)
    clean_flange = build_clean_flange_replica(params)
    source_volume = source_flange.val().Volume()
    clean_volume = clean_flange.val().Volume()
    flange_volume_delta = clean_volume - source_volume
    if abs(flange_volume_delta) > 0.05:
        raise RuntimeError(
            "clean flange replica no longer matches the source STEP: "
            f"volume delta={flange_volume_delta:.6f} mm^3"
        )

    recommended_tip_profile = str(
        params["peg"]["recommended_hardware_tip_profile"]
    )
    integrated_tools = {
        f"integrated_flange_peg_{size}": build_integrated_flange_peg(
            size,
            specs[size],
            params,
            recommended_tip_profile,
        )
        for size in SIZE_ORDER
    }
    nominal_base = build_socket_base(specs, params, "simulation_exact", 0.0)
    recommended_profile = str(
        params["socket_base"]["recommended_hardware_socket_profile"]
    )
    hardware_easy_base = build_socket_base(specs, params, recommended_profile, 0.0)
    smooth_reference_bases = {
        "socket_base_nominal": build_socket_base(
            specs,
            params,
            "simulation_exact",
            0.0,
            include_exterior_ribs=False,
        ),
        "socket_base_hardware_easy_v2": build_socket_base(
            specs,
            params,
            recommended_profile,
            0.0,
            include_exterior_ribs=False,
        ),
    }
    fdm_comp = float(params["socket_base"]["fdm_trial_socket_compensation_per_side_mm"])
    fdm_trial_base = build_socket_base(specs, params, "simulation_exact", fdm_comp)
    bases = {
        "socket_base_nominal": nominal_base,
        "socket_base_hardware_easy_v2": hardware_easy_base,
        f"socket_base_fdm_trial_p{fdm_comp:.2f}mm_per_side".replace(".", "p"): fdm_trial_base,
    }
    base_profile_contracts = {
        "socket_base_nominal": ("simulation_exact", 0.0),
        "socket_base_hardware_easy_v2": (recommended_profile, 0.0),
        f"socket_base_fdm_trial_p{fdm_comp:.2f}mm_per_side".replace(".", "p"): (
            "simulation_exact",
            fdm_comp,
        ),
    }

    assert_geometry(integrated_tools, bases)

    records: List[Dict[str, Any]] = []
    for name, tool in integrated_tools.items():
        records.append(export_part(name, tool, params))
    integrated_tool_validation = validate_exported_integrated_tools(
        specs,
        params,
        clean_flange,
        recommended_tip_profile,
    )
    for name, base in bases.items():
        records.append(export_part(name, base, params))
    socket_profile_validation = validate_exported_socket_profiles(
        specs,
        params,
        base_profile_contracts,
    )
    combined_hardware_funnel_validation = evaluate_combined_hardware_funnel(
        specs,
        params,
        recommended_profile,
        recommended_tip_profile,
    )

    coupons: Dict[str, cq.Workplane] = {
        f"socket_fit_coupon_{size}": build_socket_coupon(
            size,
            specs[size],
            params,
            recommended_profile,
        )
        for size in SIZE_ORDER
    }
    for name, coupon in coupons.items():
        records.append(export_part(name, coupon, params))

    base_params = params["socket_base"]
    plate_bottom_z = float(base_params["plate_top_z_mm"]) - float(
        base_params["plate_thickness_mm"]
    )
    center_hole_d = float(base_params["center_through_hole_diameter_mm"])
    through_hole_validation = {}
    for base_name, base_shape in bases.items():
        size_validation = {}
        for size in SIZE_ORDER:
            center_x, center_y = [
                float(value) for value in base_params["socket_centers_mm"][size]
            ]
            cavity_bottom_z = float(base_params["socket_top_z_mm"]) - 1000.0 * float(
                specs[size]["socket_depth_m"]
            )
            axis_probe = cylinder(
                0.8 * center_hole_d,
                plate_bottom_z - 0.5,
                cavity_bottom_z + 0.5,
                center_x,
                center_y,
            )
            overlap = base_shape.intersect(axis_probe).val().Volume()
            size_validation[size] = {
                "nominal_diameter_mm": center_hole_d,
                "axis_probe_diameter_mm": 0.8 * center_hole_d,
                "axis_probe_solid_overlap_mm3": overlap,
                "pass": overlap < 1.0e-8,
            }
            if overlap >= 1.0e-8:
                raise RuntimeError(
                    f"{base_name}/{size} center bore is not open through the final fixture"
                )
        through_hole_validation[base_name] = size_validation

    rib_params = base_params["exterior_vertical_ribs"]
    guard_band = float(rib_params["smooth_guard_band_below_socket_top_mm"])
    protected_band = rectangular_prism(
        float(base_params["size_x_mm"]) + 2.0,
        float(base_params["size_y_mm"]) + 2.0,
        -guard_band,
        float(base_params["socket_top_z_mm"]) + 0.2,
    )
    active_contact_geometry_validation = {}
    for base_name in ("socket_base_nominal", "socket_base_hardware_easy_v2"):
        ribbed_base = bases[base_name]
        smooth_base = smooth_reference_bases[base_name]
        ribbed_guard_volume = ribbed_base.intersect(protected_band).val().Volume()
        smooth_guard_volume = smooth_base.intersect(protected_band).val().Volume()
        common_guard_volume = (
            ribbed_base.intersect(smooth_base).intersect(protected_band).val().Volume()
        )
        # Intersection-volume differences are more stable here than direct OCC
        # cuts between bodies with many coincident faces.
        added_in_guard = max(0.0, ribbed_guard_volume - common_guard_volume)
        removed_in_guard = max(0.0, smooth_guard_volume - common_guard_volume)
        passed = added_in_guard < 1.0e-6 and removed_in_guard < 1.0e-6
        active_contact_geometry_validation[base_name] = {
            "smooth_guard_band_z_mm": [
                -guard_band,
                float(base_params["socket_top_z_mm"]),
            ],
            "simulated_head_top_contact_thickness_mm": 0.08,
            "minimum_vertical_separation_from_simulated_top_contact_mm": guard_band
            - 0.08,
            "rib_added_volume_inside_guard_band_mm3": added_in_guard,
            "rib_removed_volume_inside_guard_band_mm3": removed_in_guard,
            "pass": passed,
            "claim_boundary": "Current MuJoCo screw_head_side_visual has collision disabled; this check preserves the physical top band but is not a future exterior-side collision validation.",
        }
        if not passed:
            raise RuntimeError(
                f"{base_name}: exterior ribs changed the protected socket-top guard band"
            )

    manifest = {
        "generator": str(Path(__file__).name),
        "cadquery_version": cq.__version__,
        "design_revision": params["design_revision"],
        "design_parameters_sha256": sha256(PARAMETERS_FILE),
        "source_step": {"path": str(SOURCE_STEP), "sha256": sha256(SOURCE_STEP)},
        "source_geometry_validation": {
            "source_flange_volume_mm3": source_volume,
            "clean_replica_volume_mm3": clean_volume,
            "clean_minus_source_volume_mm3": flange_volume_delta,
            "clean_replica_reason": "avoids self-intersecting STL triangles created by the source STEP's split tangent faces",
        },
        "source_configs": {
            size: {"path": str(path), "sha256": sha256(path)}
            for size, path in SOURCE_CONFIGS.items()
        },
        "coordinate_system": params["coordinate_system"],
        "peg_flange_yaw_contract": {
            "shape_rotation_about_positive_z_deg": params["peg"][
                "flange_relative_yaw_deg"
            ],
            "selected_peg_flat_normal_in_flange_deg": params[
                "coordinate_system"
            ]["integrated_peg_selected_hex_flat_normal_deg"],
            "flange_reference_tangent_normal_deg": params["coordinate_system"][
                "flange_reference_tangent_normal_deg"
            ],
            "socket_selected_flat_normal_deg": params["coordinate_system"][
                "socket_selected_hex_flat_normal_deg"
            ],
            "equivalent_hex_period_deg": params["coordinate_system"][
                "hex_equivalent_period_deg"
            ],
            "cad_tolerance_deg": params["peg"][
                "cad_flat_normal_tolerance_deg"
            ],
            "physical_acceptance_deg": params["peg"][
                "physical_flat_parallelism_acceptance_deg"
            ],
            "reference": params["peg"]["yaw_alignment_reference"],
            "scope": "The complete peg hex is clocked relative to the flange; socket geometry is intentionally unchanged.",
        },
        "tool_reference": params["tool_reference"],
        "tool_architecture": params["integrated_tool"],
        "retired_quick_change": params["retired_quick_change_v1_3"],
        "recommended_hardware_socket_profile": params["socket_base"][
            "recommended_hardware_socket_profile"
        ],
        "recommended_hardware_peg_tip_profile": recommended_tip_profile,
        "socket_fit_coupon_profile": recommended_profile,
        "socket_fit_coupon_total_wall_af_offsets_mm": params["fit_coupons"][
            "socket_total_af_offsets_mm"
        ],
        "parts": records,
        "integrated_tool_validation": integrated_tool_validation,
        "socket_profile_validation": socket_profile_validation,
        "combined_hardware_funnel_validation": combined_hardware_funnel_validation,
        "through_hole_validation": through_hole_validation,
        "active_contact_geometry_validation": active_contact_geometry_validation,
        "claim_boundary": [
            "Nominal CAD geometry only; no as-printed dimensional validation has been performed.",
            "The FDM trial compensation is a labeled trial, not a measured correction.",
            "socket_base_nominal is the strict MuJoCo geometry control; socket_base_hardware_easy_v2 deliberately enlarges the wall clearance, 45-degree entry funnel, and outer head and requires hardware-specific MuJoCo geometry for matched reruns.",
            "The one-piece flange tools eliminate the quick-change fit, cartridge shoulder, and retaining clip, but do not reduce the 42 mm blue-plane-to-tip moment arm.",
            "The hardware-easy-v2 peg tips use visible 45-degree chamfers while retaining 80 percent of each shaft AF at the tip face; this enlarges the guiding region but does not guarantee capture.",
            "The hardware-easy-v2 deep-wall clearance increases rotational free play before torque contact; terminal torque and dwell gates must be retuned on hardware.",
            "The integrated peg is clocked -15 degrees about the flange +Z axis so a selected flat normal is parallel to the +45 degree B/C tangent datum; this mechanical datum does not itself calibrate sensor, robot, or fixture yaw.",
            "The socket hexes remain at zero yaw. After printing, flange-datum-to-fixture teaching and measured peg/socket relative yaw are required before contact experiments.",
            "The full Bota measurement-origin-to-tip transform remains pending the white-flange stack measurement.",
            "The central through-holes are new hardware features and are not present in the current MuJoCo socket bottoms.",
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "parts": len(records), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
