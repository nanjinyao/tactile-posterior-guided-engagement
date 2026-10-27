"""Fresh post-freeze strict/easy M8 validation tasks for adapter promotion."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scipy.stats import qmc

from src.sample_m8_conical_pilot import PilotTask
from src.sample_m8_hardware_easy_v2_direction import _geometry, _make_tasks
from src.sample_m8_independent_direction_audit import run as run_sampler


ROOT = Path(__file__).resolve().parents[1]
STRICT_SPEC = ROOT / "configs" / "allen_m8_source.json"
STRICT_MODEL = ROOT / "models" / "allen_m8_source.xml"
EASY_SPEC = ROOT / "configs" / "allen_m8_hardware_easy_v2.json"
EASY_MODEL = ROOT / "models" / "allen_m8_hardware_easy_v2.xml"
STRICT_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh_strict_untouched"
)
EASY_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh_easy_untouched"
)
STRICT_OUT_V2 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh2_strict_untouched"
)
EASY_OUT_V2 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh2_easy_untouched"
)
STRICT_OUT_V3 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh3_strict_untouched"
)
EASY_OUT_V3 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh3_easy_untouched"
)
STRICT_OUT_V4 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh4_strict_untouched"
)
EASY_OUT_V4 = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_fresh4_easy_untouched"
)


def build_fresh_easy_tasks(spec: dict) -> list[PilotTask]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.32 * guidance,
        0.68 * full_support,
        full_support - 0.00035,
        full_support + 0.00065,
        0.00492,
    )
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=(11.0, 41.0),
        repeats=3,
        nuisance_seed=20260724,
        prefix="easyfresh",
        noisy_repeats=True,
    )


def build_fresh2_easy_tasks(spec: dict) -> list[PilotTask]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.40 * guidance,
        0.75 * full_support,
        full_support - 0.00010,
        full_support + 0.00080,
        0.00499,
    )
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=(5.0, 35.0),
        repeats=3,
        nuisance_seed=20260725,
        prefix="easyfresh2",
        noisy_repeats=True,
    )


def build_fresh3_easy_tasks(spec: dict) -> list[PilotTask]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.50 * guidance,
        0.82 * full_support,
        full_support + 0.00005,
        full_support + 0.00069,
        0.00496,
    )
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=(21.0, 51.0),
        repeats=3,
        nuisance_seed=20260726,
        prefix="easyfresh3",
        noisy_repeats=True,
    )


def build_fresh4_easy_tasks(spec: dict) -> list[PilotTask]:
    """Final post-dense-adaptation audit; disjoint from all fitting decisions."""
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.58 * guidance,
        0.86 * full_support,
        full_support + 0.00012,
        full_support + 0.00058,
        0.00493,
    )
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=(14.0, 44.0),
        repeats=3,
        nuisance_seed=20260728,
        prefix="easyfresh4",
        noisy_repeats=True,
    )


def _build_strict_tasks(
    spec: dict,
    *,
    radii: tuple[float, ...],
    local_angles: tuple[float, ...],
    nuisance_seed: int,
    prefix: str,
) -> list[PilotTask]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    mouth_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    nuisance = qmc.Sobol(d=4, scramble=True, seed=nuisance_seed).random_base2(4)[:15]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, radius in enumerate(radii):
        outer_signed = head_radius - (radius + bit_radius)
        inner_signed = (radius - bit_radius) - mouth_apothem
        state = (
            "supported_search"
            if outer_signed > 0.0
            else "double_overhang"
            if inner_signed < 0.0
            else "outer_partial"
        )
        for repeat_index in range(3):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            for local_index, local_angle in enumerate(local_angles):
                orbit_id = (
                    f"{prefix}_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                )
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    angle_deg = local_angle + rotation_deg
                    angle_rad = math.radians(angle_deg)
                    base_pose_id = (
                        f"{state}_{prefix}_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=radius,
                            base_angle_deg=local_angle,
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=radius * math.cos(angle_rad),
                            y_m=radius * math.sin(angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=f"{prefix}_{radial_level:02d}_{repeat_index:02d}",
                            friction_slide=0.55 + 0.40 * float(sample[0]),
                            contact_timeconst_s=0.040 + 0.030 * float(sample[1]),
                            target_down_force_n=2.5 + float(sample[2]),
                            yaw_rad=math.radians(-1.5 + 3.0 * float(sample[3])),
                            nuisance_seed=nuisance_seed + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                        )
                    )
    return tasks


def build_fresh_strict_tasks(spec: dict) -> list[PilotTask]:
    return _build_strict_tasks(
        spec,
        radii=(0.00133, 0.00210, 0.00295, 0.00390, 0.00472),
        local_angles=(13.0, 53.0),
        nuisance_seed=20260724,
        prefix="strictfresh",
    )


def build_fresh2_strict_tasks(spec: dict) -> list[PilotTask]:
    return _build_strict_tasks(
        spec,
        radii=(0.00148, 0.00224, 0.00305, 0.00402, 0.00482),
        local_angles=(19.0, 49.0),
        nuisance_seed=20260725,
        prefix="strictfresh2",
    )


def build_fresh3_strict_tasks(spec: dict) -> list[PilotTask]:
    return _build_strict_tasks(
        spec,
        radii=(0.00162, 0.00236, 0.00320, 0.00412, 0.00486),
        local_angles=(9.0, 39.0),
        nuisance_seed=20260726,
        prefix="strictfresh3",
    )


def build_fresh4_strict_tasks(spec: dict) -> list[PilotTask]:
    return _build_strict_tasks(
        spec,
        radii=(0.00175, 0.00248, 0.00335, 0.00425, 0.00491),
        local_angles=(14.0, 44.0),
        nuisance_seed=20260728,
        prefix="strictfresh4",
    )


def run(args: argparse.Namespace) -> int:
    revision2 = args.revision == 2
    revision3 = args.revision == 3
    revision4 = args.revision == 4
    if args.domain == "strict":
        args.spec = str(STRICT_SPEC)
        args.model = str(STRICT_MODEL)
        args.task_builder = (
            build_fresh4_strict_tasks
            if revision4
            else build_fresh3_strict_tasks
            if revision3
            else build_fresh2_strict_tasks
            if revision2
            else build_fresh_strict_tasks
        )
        args.stage_name = (
            "M8_hardware_easy_v2_adapter_fresh4_strict_untouched"
            if revision4
            else "M8_hardware_easy_v2_adapter_fresh3_strict_untouched"
            if revision3
            else
            "M8_hardware_easy_v2_adapter_fresh2_strict_untouched"
            if revision2
            else "M8_hardware_easy_v2_adapter_fresh_strict_untouched"
        )
        args.output = args.output or str(
            STRICT_OUT_V4
            if revision4
            else STRICT_OUT_V3
            if revision3
            else STRICT_OUT_V2
            if revision2
            else STRICT_OUT
        )
        args.progress_description = (
            "M8 fresh4 strict"
            if revision4
            else "M8 fresh3 strict"
            if revision3
            else "M8 fresh2 strict"
            if revision2
            else "M8 fresh strict"
        )
    else:
        args.spec = str(EASY_SPEC)
        args.model = str(EASY_MODEL)
        args.task_builder = (
            build_fresh4_easy_tasks
            if revision4
            else build_fresh3_easy_tasks
            if revision3
            else build_fresh2_easy_tasks
            if revision2
            else build_fresh_easy_tasks
        )
        args.stage_name = (
            "M8_hardware_easy_v2_adapter_fresh4_easy_untouched"
            if revision4
            else "M8_hardware_easy_v2_adapter_fresh3_easy_untouched"
            if revision3
            else
            "M8_hardware_easy_v2_adapter_fresh2_easy_untouched"
            if revision2
            else "M8_hardware_easy_v2_adapter_fresh_easy_untouched"
        )
        args.output = args.output or str(
            EASY_OUT_V4
            if revision4
            else EASY_OUT_V3
            if revision3
            else EASY_OUT_V2
            if revision2
            else EASY_OUT
        )
        args.progress_description = (
            "M8 fresh4 easy"
            if revision4
            else "M8 fresh3 easy"
            if revision3
            else "M8 fresh2 easy"
            if revision2
            else "M8 fresh easy"
        )
    args.training_use = "forbidden"
    args.claim_boundary = (
        "Post-adaptation frozen simulation audit with disjoint radii, angles, nuisances, and task IDs. "
        "Never used for fitting, normalization, calibration, or gate selection."
    )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("strict", "easy"), required=True)
    parser.add_argument("--revision", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
