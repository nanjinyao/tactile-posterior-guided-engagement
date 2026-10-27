"""Sample dense train and untouched audit fingerprints for a shared M8/M10/M12 model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.sample_cross_size_s4_adaptation import deployment_geometry
from src.sample_m8_conical_pilot import PilotTask
from src.sample_m8_hardware_easy_v2_direction import _geometry, _make_tasks
from src.sample_m8_independent_direction_audit import run as run_sampler


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("m8", "m10", "m12")
TRAIN_SEEDS = {"m8": 20260732, "m10": 20260733, "m12": 20260734}
AUDIT_SEEDS = {"m8": 20260742, "m10": 20260743, "m12": 20260744}
STRICT_M8_SEED = 20260745
AUDIT2_SEEDS = {"m8": 20260752, "m10": 20260753, "m12": 20260754}
STRICT_M8_SEED2 = 20260755
AUDIT3_SEEDS = {"m8": 20260762, "m10": 20260763, "m12": 20260764}
STRICT_M8_SEED3 = 20260765
AUDIT4_SEEDS = {"m8": 20260772, "m10": 20260773, "m12": 20260774}
STRICT_M8_SEED4 = 20260775
AUDIT5_SEEDS = {"m8": 20260782, "m10": 20260783, "m12": 20260784}
STRICT_M8_SEED5 = 20260785
AUDIT6_SEEDS = {"m8": 20260792, "m10": 20260793, "m12": 20260794}
STRICT_M8_SEED6 = 20260795
AUDIT7_SEEDS = {"m8": 20260802, "m10": 20260803, "m12": 20260804}
STRICT_M8_SEED7 = 20260805
AUDIT8_SEEDS = {"m8": 20260812, "m10": 20260813, "m12": 20260814}
STRICT_M8_SEED8 = 20260815
AUDIT9_SEEDS = {"m8": 20260822, "m10": 20260823, "m12": 20260824}
STRICT_M8_SEED9 = 20260825
TRAIN_ROOT = ROOT / "outputs/current/datasets/hardware_easy_v2_multisize_dense"
AUDIT_ROOT = ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh"
AUDIT2_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh2"
)
AUDIT3_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh3"
)
AUDIT4_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh4"
)
AUDIT5_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh5"
)
AUDIT6_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh6"
)
AUDIT7_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh7"
)
AUDIT8_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh8"
)
AUDIT9_ROOT = (
    ROOT / "outputs/current/evaluations/hardware_easy_v2_multisize_fresh9"
)


def _dense_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    radii = (
        1.15 * guidance,
        1.35 * guidance,
        1.55 * guidance,
        0.85 * full_support,
        full_support - 0.00015,
        full_support + 0.30 * (visual_bound - full_support),
        full_support + 0.70 * (visual_bound - full_support),
        visual_bound,
    )
    if len(set(round(value, 12) for value in radii)) != 8:
        raise ValueError("multisize dense radii are not unique")
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize dense radii must be strictly increasing")
    return tuple(map(float, radii))


def build_dense_tasks(spec: dict, size: str) -> list[PilotTask]:
    if size not in SIZES:
        raise ValueError(size)
    angles = tuple(3.75 + 7.5 * index for index in range(8))
    return _make_tasks(
        spec,
        radii=_dense_radii(spec),
        local_angles_deg=angles,
        repeats=2,
        nuisance_seed=TRAIN_SEEDS[size],
        prefix=f"{size}easyjoint",
        noisy_repeats=True,
    )


def _audit_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    radii = (
        1.25 * guidance,
        0.82 * full_support,
        full_support + 0.02 * visual_bound,
        full_support + 0.11 * visual_bound,
        0.96 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize audit radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit2_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    radii = (
        1.18 * guidance,
        0.77 * full_support,
        full_support + 0.08 * (visual_bound - full_support),
        full_support + 0.48 * (visual_bound - full_support),
        0.92 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh2 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit3_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    radii = (
        1.30 * guidance,
        0.73 * full_support,
        full_support + 0.16 * (visual_bound - full_support),
        full_support + 0.60 * (visual_bound - full_support),
        0.95 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh3 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit4_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    radii = (
        1.22 * guidance,
        0.80 * full_support,
        full_support + 0.24 * (visual_bound - full_support),
        full_support + 0.78 * (visual_bound - full_support),
        0.98 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh4 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit5_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    residual = visual_bound - full_support
    radii = (
        1.28 * guidance,
        0.76 * full_support,
        full_support + 0.36 * residual,
        full_support + 0.72 * residual,
        0.985 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh5 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit6_radii(spec: dict) -> tuple[float, ...]:
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    residual = visual_bound - full_support
    radii = (
        1.32 * guidance,
        0.79 * full_support,
        full_support + 0.44 * residual,
        full_support + 0.66 * residual,
        0.975 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh6 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit7_radii(spec: dict) -> tuple[float, ...]:
    """Post-v5 final-audit rings, frozen before evaluating the clean model."""
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    residual = visual_bound - full_support
    radii = (
        1.36 * guidance,
        0.74 * full_support,
        full_support + 0.20 * residual,
        full_support + 0.62 * residual,
        0.965 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh7 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit8_radii(spec: dict) -> tuple[float, ...]:
    """Final post-selection rings; frozen before fresh8 is sampled."""
    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    residual = visual_bound - full_support
    radii = (
        1.38 * guidance,
        0.83 * full_support,
        full_support + 0.27 * residual,
        full_support + 0.56 * residual,
        0.945 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh8 radii must be strictly increasing")
    return tuple(map(float, radii))


def _audit9_radii(spec: dict) -> tuple[float, ...]:
    """Post-v8 final-audit rings, frozen before any fresh9 sampling."""

    _, head_radius, _, guidance = _geometry(spec)
    bit_radius = float(spec["key_af_m"]) / math.sqrt(3.0)
    full_support = head_radius - bit_radius
    visual_bound = float(max(spec["visual_offset_sweep_m"]))
    residual = visual_bound - full_support
    radii = (
        1.41 * guidance,
        0.805 * full_support,
        full_support + 0.33 * residual,
        full_support + 0.51 * residual,
        0.935 * visual_bound,
    )
    if tuple(sorted(radii)) != radii:
        raise ValueError("multisize fresh9 radii must be strictly increasing")
    return tuple(map(float, radii))


def build_easy_audit_tasks(
    spec: dict, size: str, generation: int = 1
) -> list[PilotTask]:
    if size not in SIZES:
        raise ValueError(size)
    if generation not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError(f"unsupported audit generation: {generation}")
    radii_by_generation = {
        1: _audit_radii(spec),
        2: _audit2_radii(spec),
        3: _audit3_radii(spec),
        4: _audit4_radii(spec),
        5: _audit5_radii(spec),
        6: _audit6_radii(spec),
        7: _audit7_radii(spec),
        8: _audit8_radii(spec),
        9: _audit9_radii(spec),
    }
    angles_by_generation = {
        1: (16.0, 46.0),
        2: (9.5, 39.5),
        3: (24.5, 54.5),
        4: (1.5, 31.5),
        5: (6.0, 36.0),
        6: (21.0, 51.0),
        7: (12.0, 42.0),
        8: (27.0, 57.0),
        9: (14.25, 44.25),
    }
    seeds_by_generation = {
        1: AUDIT_SEEDS,
        2: AUDIT2_SEEDS,
        3: AUDIT3_SEEDS,
        4: AUDIT4_SEEDS,
        5: AUDIT5_SEEDS,
        6: AUDIT6_SEEDS,
        7: AUDIT7_SEEDS,
        8: AUDIT8_SEEDS,
        9: AUDIT9_SEEDS,
    }
    return _make_tasks(
        spec,
        radii=radii_by_generation[generation],
        local_angles_deg=angles_by_generation[generation],
        repeats=3,
        nuisance_seed=seeds_by_generation[generation][size],
        prefix=f"{size}easyjointfresh{generation}",
        noisy_repeats=True,
    )


def build_m8_source_audit_tasks(
    spec: dict, generation: int = 1
) -> list[PilotTask]:
    if generation not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        raise ValueError(f"unsupported audit generation: {generation}")
    geometry = deployment_geometry(spec)
    full_support = float(geometry["full_support_radius_m"])
    visual_bound = float(geometry["visual_bound_m"])
    radius_min = float(geometry["radius_min_m"])
    residual = visual_bound - full_support
    radii_by_generation = {
        1: (
            max(1.10 * radius_min, 0.55 * full_support),
            0.82 * full_support,
            full_support + 0.10 * residual,
            full_support + 0.55 * residual,
            full_support + 0.90 * residual,
        ),
        2: (
            max(1.20 * radius_min, 0.62 * full_support),
            0.76 * full_support,
            full_support + 0.22 * residual,
            full_support + 0.68 * residual,
            full_support + 0.96 * residual,
        ),
        3: (
            max(1.30 * radius_min, 0.58 * full_support),
            0.88 * full_support,
            full_support + 0.34 * residual,
            full_support + 0.78 * residual,
            full_support + 0.99 * residual,
        ),
        4: (
            max(1.15 * radius_min, 0.67 * full_support),
            0.92 * full_support,
            full_support + 0.45 * residual,
            full_support + 0.86 * residual,
            full_support + 0.94 * residual,
        ),
        5: (
            max(1.25 * radius_min, 0.72 * full_support),
            0.84 * full_support,
            full_support + 0.58 * residual,
            full_support + 0.82 * residual,
            full_support + 0.97 * residual,
        ),
        6: (
            max(1.28 * radius_min, 0.64 * full_support),
            0.86 * full_support,
            full_support + 0.28 * residual,
            full_support + 0.74 * residual,
            full_support + 0.985 * residual,
        ),
        7: (
            max(1.34 * radius_min, 0.69 * full_support),
            0.89 * full_support,
            full_support + 0.40 * residual,
            full_support + 0.70 * residual,
            full_support + 0.955 * residual,
        ),
        8: (
            max(1.38 * radius_min, 0.66 * full_support),
            0.91 * full_support,
            full_support + 0.18 * residual,
            full_support + 0.64 * residual,
            full_support + 0.975 * residual,
        ),
        9: (
            max(1.42 * radius_min, 0.71 * full_support),
            0.87 * full_support,
            full_support + 0.31 * residual,
            full_support + 0.59 * residual,
            full_support + 0.935 * residual,
        ),
    }
    angles_by_generation = {
        1: (16.0, 46.0),
        2: (9.5, 39.5),
        3: (24.5, 54.5),
        4: (1.5, 31.5),
        5: (6.0, 36.0),
        6: (21.0, 51.0),
        7: (12.0, 42.0),
        8: (27.0, 57.0),
        9: (14.25, 44.25),
    }
    seeds_by_generation = {
        1: STRICT_M8_SEED,
        2: STRICT_M8_SEED2,
        3: STRICT_M8_SEED3,
        4: STRICT_M8_SEED4,
        5: STRICT_M8_SEED5,
        6: STRICT_M8_SEED6,
        7: STRICT_M8_SEED7,
        8: STRICT_M8_SEED8,
        9: STRICT_M8_SEED9,
    }
    return _make_tasks(
        spec,
        radii=tuple(map(float, radii_by_generation[generation])),
        local_angles_deg=angles_by_generation[generation],
        repeats=3,
        nuisance_seed=seeds_by_generation[generation],
        prefix=f"m8sourcejointfresh{generation}",
        noisy_repeats=True,
    )


def _validate_disjoint(spec: dict, size: str) -> None:
    train = build_dense_tasks(spec, size)
    audits = [
        build_easy_audit_tasks(spec, size, generation)
        for generation in (1, 2, 3, 4, 5, 6, 7, 8, 9)
    ]
    collections = [train, *audits]
    for left_index, left in enumerate(collections):
        for right in collections[left_index + 1 :]:
            if {task.radius_m for task in left} & {task.radius_m for task in right}:
                raise ValueError(f"{size}: train/audit generation radius leakage")
            if {task.base_angle_deg for task in left} & {
                task.base_angle_deg for task in right
            }:
                raise ValueError(f"{size}: train/audit generation angle leakage")
            if {task.trial_id for task in left} & {task.trial_id for task in right}:
                raise ValueError(f"{size}: train/audit generation task-id leakage")


def run(args: argparse.Namespace) -> int:
    if args.role == "m8-source-audit":
        spec_path = ROOT / "configs/allen_m8_source.json"
        model_path = ROOT / "models/allen_m8_source.xml"
        args.spec = str(spec_path)
        args.model = str(model_path)
        generation = args.audit_generation
        audit_root = (
            AUDIT9_ROOT
            if generation == 9
            else AUDIT8_ROOT
            if generation == 8
            else AUDIT7_ROOT
            if generation == 7
            else AUDIT6_ROOT
            if generation == 6
            else AUDIT5_ROOT
            if generation == 5
            else AUDIT4_ROOT
            if generation == 4
            else AUDIT3_ROOT
            if generation == 3
            else AUDIT2_ROOT
            if generation == 2
            else AUDIT_ROOT
        )
        args.task_builder = lambda selected: build_m8_source_audit_tasks(
            selected, generation
        )
        args.output = args.output or str(audit_root / "m8_source")
        args.stage_name = f"M8_source_multisize_joint_fresh{generation}_untouched"
        args.training_use = "forbidden"
        args.progress_description = "M8 source joint fresh"
        args.claim_boundary = (
            "Post-freeze M8 source-scene audit for the shared multisize model. "
            "Never used for fitting, normalization, calibration, or gate selection."
        )
        return run_sampler(args)

    size = args.size
    spec_path = ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json"
    model_path = ROOT / "models" / f"allen_{size}_hardware_easy_v2.xml"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _validate_disjoint(spec, size)
    args.spec = str(spec_path)
    args.model = str(model_path)
    if args.role == "train":
        args.task_builder = lambda selected: build_dense_tasks(selected, size)
        args.output = args.output or str(TRAIN_ROOT / size)
        args.stage_name = f"{size.upper()}_hardware_easy_v2_multisize_dense_training"
        args.training_use = "allowed_only_after_grouped_split_and_capture_filter"
        args.progress_description = f"{size.upper()} easy-v2 joint train"
        args.claim_boundary = (
            "Hardware-easy-v2 contact-geometry simulation data for shared multisize adaptation. "
            "Captured/incomplete traces are routing outcomes and forbidden for direction supervision. "
            "The sensor-to-tip lever, FDM compliance/friction and Bota noise are not hardware aligned."
        )
    else:
        generation = args.audit_generation
        audit_root = (
            AUDIT9_ROOT
            if generation == 9
            else AUDIT8_ROOT
            if generation == 8
            else AUDIT7_ROOT
            if generation == 7
            else AUDIT6_ROOT
            if generation == 6
            else AUDIT5_ROOT
            if generation == 5
            else AUDIT4_ROOT
            if generation == 4
            else AUDIT3_ROOT
            if generation == 3
            else AUDIT2_ROOT
            if generation == 2
            else AUDIT_ROOT
        )
        args.task_builder = lambda selected: build_easy_audit_tasks(
            selected, size, generation
        )
        args.output = args.output or str(audit_root / f"{size}_easy")
        args.stage_name = (
            f"{size.upper()}_hardware_easy_v2_multisize_fresh{generation}_untouched"
        )
        args.training_use = "forbidden"
        args.progress_description = f"{size.upper()} easy-v2 joint fresh"
        args.claim_boundary = (
            "Post-freeze hardware-easy-v2 simulation audit for the shared multisize model. "
            "Never used for fitting, normalization, calibration, or gate selection."
        )
    return run_sampler(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("train", "audit", "m8-source-audit"), required=True
    )
    parser.add_argument("--size", choices=SIZES)
    parser.add_argument(
        "--audit-generation",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        default=1,
    )
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--phase-bins", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.role != "m8-source-audit" and args.size is None:
        parser.error("--size is required for train/audit")
    if args.role == "train" and args.audit_generation != 1:
        parser.error("--audit-generation only applies to audit roles")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.phase_bins != 360:
        parser.error("the shared direction interface requires 360 phase bins")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
