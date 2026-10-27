"""Derive hardware-easy-v2 contact scenes without mutating frozen sources."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.allen_socket_model import ROOT, SPECS, build_xml, spec_to_config
from src.hardware_tooling_profile import DEFAULT_PROFILE, HardwareToolingProfile


PROFILE_NAME = "hardware_easy_v2"
DEFAULT_MANIFEST = (
    ROOT / "outputs" / "current" / "reports" / "hardware_easy_v2_scene_manifest.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive_spec(size: str, profile: HardwareToolingProfile):
    size = size.lower()
    source = SPECS[size]
    tool = profile.data["tools"][size]
    tip = tool["tip_geometry_profiles"][PROFILE_NAME]
    socket = profile.data["fixture"]["socket_geometry_profiles"][PROFILE_NAME][size]
    wall_af = float(socket["wall_af_m"])
    clearance = 0.5 * (wall_af - float(tool["peg_af_m"]))
    if clearance <= 0.0:
        raise ValueError(f"{size}: hardware socket must be larger than the peg")
    return replace(
        source,
        name=f"{size}_{PROFILE_NAME}",
        key_af_m=float(tool["peg_af_m"]),
        clearance_m=clearance,
        chamfer_width_m=float(socket["entry_chamfer_width_m"]),
        chamfer_depth_m=float(socket["entry_chamfer_depth_m"]),
        screw_head_diameter_m=float(socket["head_outer_diameter_m"]),
        bit_tip_chamfer_m=float(tip["tip_chamfer_axial_length_m"]),
        bit_tip_scale=float(tip["tip_face_scale"]),
        bounded_socket_shell=True,
    )


def build(size: str, profile: HardwareToolingProfile) -> dict:
    size = size.lower()
    frame = profile.data["frame_convention"]
    source_config = ROOT / "configs" / f"allen_{size}_source.json"
    source_model = ROOT / "models" / f"allen_{size}_source.xml"
    source_hashes_before = {
        "config_sha256": _sha256(source_config),
        "model_sha256": _sha256(source_model),
    }
    spec = derive_spec(size, profile)
    config_path = ROOT / "configs" / f"allen_{size}_{PROFILE_NAME}.json"
    model_path = ROOT / "models" / f"allen_{size}_{PROFILE_NAME}.xml"
    if config_path.name.endswith("_source.json") or model_path.name.endswith("_source.xml"):
        raise RuntimeError("derived hardware scenes may never target frozen source paths")

    config = spec_to_config(spec)
    peg_tip_setback = 0.5 * spec.key_af_m * (1.0 - spec.bit_tip_scale)
    config.update(
        {
            "size_name": size,
            "geometry_profile": PROFILE_NAME,
            "contact_geometry_source": str(profile.path),
            "contact_geometry_source_sha256": _sha256(profile.path),
            "frozen_source_config": str(source_config),
            "frozen_source_config_sha256": source_hashes_before["config_sha256"],
            "frozen_source_model": str(source_model),
            "frozen_source_model_sha256": source_hashes_before["model_sha256"],
            "paired_flat_normal_guidance_margin_m": (
                spec.clearance_m + spec.chamfer_width_m + peg_tip_setback
            ),
            "paired_guidance_margin_semantics": (
                "one-dimensional flat-normal geometric allowance only; not a guaranteed capture radius"
            ),
            "hardware_alignment_scope": (
                "top/socket contact-geometry-matched surrogate; FDM compliance, fixture sides, sensor origin, "
                "tool mass, COM, inertia, and calibrated wrench lever are not matched"
            ),
            "mechanical_clocking_contract": {
                "integrated_peg_shape_rotation_from_legacy_deg": frame[
                    "integrated_peg_shape_rotation_from_legacy_deg"
                ],
                "integrated_peg_selected_hex_flat_normal_deg": frame[
                    "integrated_peg_selected_hex_flat_normal_deg"
                ],
                "flange_reference_tangent_normal_deg": frame[
                    "flange_reference_tangent_normal_deg"
                ],
                "socket_selected_hex_flat_normal_deg": frame[
                    "socket_selected_hex_flat_normal_deg"
                ],
                "mujoco_relative_yaw_zero_semantics": (
                    "The MuJoCo bit and socket remain aligned at relative yaw zero. "
                    "Do not add the CAD -15 degree peg-to-flange clocking to the bit: "
                    "the physical flange datum is taught against fixture +X instead."
                ),
            },
            "hardware_aligned_profile_ready": False,
            "wrench_reference_to_tip_m": None,
            "wrench_reference_note": (
                "legacy simulation surrogate retained because the Bota measurement origin is not yet measured; "
                "the known 42 mm blue-plane datum is not substituted as a sensor lever arm"
            ),
            "collision_shell": {
                "mode": "72-sector circularly bounded ring shell",
                "outer_radius_m": spec.head_radius_m,
                "legacy_overlapping_wedges_used": False,
            },
        }
    )
    _atomic_json(config_path, config)
    model_path.write_text(build_xml(spec), encoding="utf-8")
    if source_hashes_before != {
        "config_sha256": _sha256(source_config),
        "model_sha256": _sha256(source_model),
    }:
        raise RuntimeError(f"{size}: frozen source artifact changed during derivation")
    return {
        "size": size,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "source": source_hashes_before,
        "wall_af_m": spec.socket_wall_af_m,
        "mouth_af_m": spec.socket_top_af_m,
        "head_outer_diameter_m": spec.screw_head_diameter_m,
        "tip_face_scale": spec.bit_tip_scale,
        "tip_chamfer_axial_m": spec.bit_tip_chamfer_m,
        "paired_flat_normal_guidance_margin_m": config[
            "paired_flat_normal_guidance_margin_m"
        ],
    }


def run(args: argparse.Namespace) -> int:
    profile = HardwareToolingProfile.load(args.profile)
    rows = [build(size, profile) for size in args.sizes]
    manifest = {
        "stage": "hardware_easy_v2_contact_scene_derivation",
        "profile": str(profile.path),
        "profile_sha256": _sha256(profile.path),
        "rows": rows,
        "source_files_mutated": False,
        "claim_boundary": (
            "Known CAD contact geometry is represented in simulation. This is not a calibrated real-wrench, "
            "FDM-compliance, fixture-side, or full sim-to-real model. The v1.5 peg-to-flange clocking is a "
            "physical mounting datum, not an extra simulated peg/socket relative yaw."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(Path(args.manifest).resolve(), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate hardware-easy-v2 MuJoCo contact scenes."
    )
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--sizes", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
