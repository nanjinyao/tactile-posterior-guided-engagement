"""Sample M8 direction traces in the derived hardware-easy-v2 contact scene."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.sample_m8_conical_pilot import PilotTask
from src.sample_m8_independent_direction_audit import run as run_sampler


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "allen_m8_hardware_easy_v2.json"
MODEL = ROOT / "models" / "allen_m8_hardware_easy_v2.xml"
TRAIN_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "datasets"
    / "m8_hardware_easy_v2_direction_training"
)
DENSE_TRAIN_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "datasets"
    / "m8_hardware_easy_v2_direction_dense_training"
)
AUDIT_OUT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_direction_untouched"
)


def _geometry(spec: dict) -> tuple[float, float, float, float]:
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius = 0.5 * float(spec["screw_head_diameter_m"])
    mouth_apothem = 0.5 * float(spec["derived"]["socket_top_af_m"])
    # Source geometries predate the explicit paired-chamfer field.  The generic
    # task builder only needs the first three values, so keep source-scene task
    # construction possible without inventing a guidance margin.
    guidance = float(spec.get("paired_flat_normal_guidance_margin_m", 0.0))
    return bit_radius, head_radius, mouth_apothem, guidance


def _state_and_distances(
    radius: float, bit_radius: float, head_radius: float, mouth_apothem: float
) -> tuple[str, float, float]:
    outer_signed = head_radius - (radius + bit_radius)
    inner_signed = (radius - bit_radius) - mouth_apothem
    state = (
        "supported_search"
        if outer_signed > 0.0
        else "double_overhang"
        if inner_signed < 0.0
        else "outer_partial"
    )
    return state, outer_signed, inner_signed


def _make_tasks(
    spec: dict,
    *,
    radii: tuple[float, ...],
    local_angles_deg: tuple[float, ...],
    repeats: int,
    nuisance_seed: int,
    prefix: str,
    noisy_repeats: bool,
) -> list[PilotTask]:
    bit_radius, head_radius, mouth_apothem, _ = _geometry(spec)
    count = len(radii) * repeats
    nuisance = qmc.Sobol(d=5, scramble=True, seed=nuisance_seed).random_base2(
        int(math.ceil(math.log2(count)))
    )[:count]
    tasks: list[PilotTask] = []
    nuisance_index = 0
    for radial_level, radius in enumerate(radii):
        state, outer_signed, inner_signed = _state_and_distances(
            float(radius), bit_radius, head_radius, mouth_apothem
        )
        for repeat_index in range(repeats):
            sample = nuisance[nuisance_index]
            nuisance_index += 1
            friction = 0.55 + 0.40 * float(sample[0])
            contact_timeconst = 0.040 + 0.030 * float(sample[1])
            down_force = 2.5 + float(sample[2])
            yaw_rad = math.radians(-2.0 + 4.0 * float(sample[3]))
            delay_s = 0.002 + 0.006 * float(sample[4])
            for local_index, local_angle_deg in enumerate(local_angles_deg):
                orbit_id = (
                    f"{prefix}_r{radial_level:02d}_l{local_index:02d}_n{repeat_index:02d}"
                )
                for sector_index in range(6):
                    rotation_deg = 60.0 * sector_index
                    physical_angle_deg = float(local_angle_deg) + rotation_deg
                    physical_angle_rad = math.radians(physical_angle_deg)
                    base_pose_id = (
                        f"{state}_{prefix}_r{radial_level:02d}_l{local_index:02d}_s{sector_index:02d}"
                    )
                    use_noise = noisy_repeats and (
                        repeats == 1 or repeat_index % 2 == 1
                    )
                    tasks.append(
                        PilotTask(
                            trial_id=f"{base_pose_id}_n{repeat_index:02d}",
                            base_pose_id=base_pose_id,
                            search_state=state,
                            radial_level=radial_level,
                            radius_m=float(radius),
                            base_angle_deg=float(local_angle_deg),
                            rotation_index=sector_index,
                            rotation_deg=rotation_deg,
                            x_m=float(radius) * math.cos(physical_angle_rad),
                            y_m=float(radius) * math.sin(physical_angle_rad),
                            nominal_tip_z_m=0.00010,
                            outer_signed_distance_m=outer_signed,
                            inner_signed_distance_m=inner_signed,
                            nuisance_id=(
                                f"{prefix}_{'sensor_prior' if use_noise else 'ideal'}_{repeat_index:02d}"
                            ),
                            friction_slide=friction,
                            contact_timeconst_s=contact_timeconst,
                            target_down_force_n=down_force,
                            yaw_rad=yaw_rad,
                            nuisance_seed=nuisance_seed + nuisance_index - 1,
                            symmetry_orbit_id=orbit_id,
                            sensor_noise_seed=(
                                nuisance_seed * 1000
                                + radial_level * 100
                                + repeat_index * 10
                                + sector_index
                                if use_noise
                                else None
                            ),
                            sensor_delay_s=delay_s if use_noise else 0.0,
                        )
                    )
    return tasks


def build_hardware_easy_v2_training_tasks(spec: dict) -> list[PilotTask]:
    """288 C6 scans outside the paired lead-in diagnostic margin."""
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    minimum = 1.15 * guidance
    radii = (
        minimum,
        0.80 * full_support,
        full_support + 0.00015,
        full_support + 0.00050,
        full_support + 0.00084,
        0.00500,
    )
    angle_samples = qmc.Sobol(d=1, scramble=True, seed=20260722).random_base2(2)[:, 0]
    angles = tuple(sorted(2.0 + 56.0 * float(value) for value in angle_samples))
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=angles,
        repeats=2,
        nuisance_seed=20260722,
        prefix="easytrain",
        noisy_repeats=True,
    )


def build_hardware_easy_v2_dense_training_tasks(spec: dict) -> list[PilotTask]:
    """768 C6 scans with dense angular and lead-in-boundary coverage.

    The first dataset established that the derived scene could be adapted at all,
    but its four local angles left 13-degree gaps in the fundamental 60-degree
    sector.  This second, independent training design covers every 7.5-degree
    direction bin and deliberately samples both sides of the stochastic capture
    transition.  Captured traces remain routing outcomes and are filtered before
    direction supervision.
    """
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.35 * guidance,
        1.50 * guidance,
        1.70 * guidance,
        0.78 * full_support,
        full_support - 0.00015,
        full_support + 0.00025,
        full_support + 0.00070,
        0.00500,
    )
    local_angles = tuple(3.75 + 7.5 * index for index in range(8))
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=local_angles,
        repeats=2,
        nuisance_seed=20260727,
        prefix="easydense",
        noisy_repeats=True,
    )


def build_hardware_easy_v2_untouched_tasks(spec: dict) -> list[PilotTask]:
    """180 disjoint easy-v2 scans, forbidden for fitting and gate selection."""
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    radii = (
        1.25 * guidance,
        0.72 * full_support,
        full_support - 0.00020,
        full_support + 0.00035,
        0.00478,
    )
    return _make_tasks(
        spec,
        radii=radii,
        local_angles_deg=(17.0, 47.0),
        repeats=3,
        nuisance_seed=20260723,
        prefix="easyaudit",
        noisy_repeats=True,
    )


def run(args: argparse.Namespace) -> int:
    args.spec = str(SPEC)
    args.model = str(MODEL)
    if args.role in ("train", "dense"):
        dense = args.role == "dense"
        args.task_builder = (
            build_hardware_easy_v2_dense_training_tasks
            if dense
            else build_hardware_easy_v2_training_tasks
        )
        args.stage_name = (
            "M8_hardware_easy_v2_direction_dense_training"
            if dense
            else "M8_hardware_easy_v2_direction_training"
        )
        args.training_use = "allowed_only_after_grouped_split_and_capture_filter"
        args.progress_description = "M8 easy-v2 dense train" if dense else "M8 easy-v2 train"
        args.claim_boundary = (
            "Hardware-easy-v2 contact-geometry-conditioned simulation training data. "
            "Captured/incomplete traces are routing outcomes and forbidden for direction supervision. "
            "Sensor noise is a pre-hardware engineering prior, not measured Bota behavior."
        )
    else:
        args.task_builder = build_hardware_easy_v2_untouched_tasks
        args.stage_name = "M8_hardware_easy_v2_direction_untouched"
        args.training_use = "forbidden"
        args.progress_description = "M8 easy-v2 untouched"
        args.claim_boundary = (
            "Frozen hardware-easy-v2 contact-geometry simulation audit. Never used for model fitting, "
            "normalization, calibration, gate selection, or active-sampling decisions."
        )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("train", "dense", "audit"), required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output is None:
        args.output = str(
            TRAIN_OUT
            if args.role == "train"
            else DENSE_TRAIN_OUT
            if args.role == "dense"
            else AUDIT_OUT
        )
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins != 360:
        parser.error("the direction interface requires 360 phase bins")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
