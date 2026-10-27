from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.audit_m8_model_cross_size_zero_shot import _sample_size
from src.sample_m8_conical_pilot import PilotTask
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/current/datasets/cross_size_s4_adaptation"
SIZES = ("m10", "m12")
SEEDS = {"m10": 20261031, "m12": 20261032}

# A physical C6 orbit is never split across roles.  The local anchors are also
# disjoint, so validation/calibration measure interpolation between train
# anchors instead of replaying the same physical direction orbit.
ROLE_PHASES_DEG: dict[str, tuple[float, ...]] = {
    "train": (7.5, 37.5),
    "validation": (22.5,),
    "calibration": (52.5,),
}
ROLE_PREFIX = {"train": "s4tr", "validation": "s4va", "calibration": "s4ca"}
SECTORS = tuple(range(6))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deployment_geometry(spec: dict) -> dict[str, float]:
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    head_radius_m = 0.5 * float(spec["screw_head_diameter_m"])
    socket_top_apothem_m = 0.5 * float(spec["derived"]["socket_top_af_m"])
    capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    full_support_radius_m = head_radius_m - bit_radius_m
    inner_overlap_radius_m = bit_radius_m + socket_top_apothem_m
    visual_bound_m = float(max(spec["visual_offset_sweep_m"]))
    radius_min_m = max(
        float(spec["visual_offset_min_m"]), 4.0 * capture_radius_m
    )
    if not 0.0 < radius_min_m < full_support_radius_m < visual_bound_m:
        raise ValueError("invalid deployment support/visual radii")
    if visual_bound_m >= inner_overlap_radius_m:
        raise ValueError("S4 design assumes deployment remains inside inner overlap")
    return {
        "bit_radius_m": bit_radius_m,
        "head_radius_m": head_radius_m,
        "capture_radius_m": capture_radius_m,
        "radius_min_m": radius_min_m,
        "full_support_radius_m": full_support_radius_m,
        "inner_overlap_radius_m": inner_overlap_radius_m,
        "visual_bound_m": visual_bound_m,
    }


def deployment_rings(spec: dict) -> tuple[dict, ...]:
    geometry = deployment_geometry(spec)
    radius_min = geometry["radius_min_m"]
    full_support = geometry["full_support_radius_m"]
    visual_bound = geometry["visual_bound_m"]
    residual_span = visual_bound - full_support
    radii = (
        max(radius_min, 0.45 * full_support),
        0.80 * full_support,
        full_support + 0.15 * residual_span,
        full_support + 0.50 * residual_span,
        full_support + 0.85 * residual_span,
    )
    states = (
        "supported_search",
        "supported_search",
        "double_overhang",
        "double_overhang",
        "double_overhang",
    )
    rings = tuple(
        {
            "ring_index": index,
            "radius_m": float(radius),
            "normalized_visual_radius": float(radius / visual_bound),
            "search_state": state,
        }
        for index, (radius, state) in enumerate(zip(radii, states))
    )
    for ring in rings:
        radius = float(ring["radius_m"])
        if not radius_min <= radius <= visual_bound:
            raise ValueError("deployment ring is outside the frozen S4 annulus")
        expected = (
            "supported_search" if radius < full_support else "double_overhang"
        )
        if ring["search_state"] != expected:
            raise ValueError("deployment ring has an inconsistent CAD state")
    return rings


def build_role_tasks(spec: dict, size: str, seed: int) -> dict[str, list[PilotTask]]:
    if size not in SIZES:
        raise ValueError(size)
    geometry = deployment_geometry(spec)
    rings = deployment_rings(spec)
    orbit_specs = [
        (role, ring, phase_deg)
        for role, phases in ROLE_PHASES_DEG.items()
        for ring in rings
        for phase_deg in phases
    ]
    power = int(math.ceil(math.log2(len(orbit_specs))))
    nuisance = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(power)[
        : len(orbit_specs)
    ]
    role_tasks = {role: [] for role in ROLE_PHASES_DEG}
    for orbit_index, ((role, ring, phase_deg), sample) in enumerate(
        zip(orbit_specs, nuisance)
    ):
        radius_m = float(ring["radius_m"])
        state = str(ring["search_state"])
        ring_index = int(ring["ring_index"])
        friction = 0.55 + 0.40 * float(sample[0])
        contact_timeconst = 0.040 + 0.030 * float(sample[1])
        down_force = 2.5 + float(sample[2])
        # The current full-loop S3/S4 simulation starts at zero yaw.  Keep the
        # adaptation contact motion deployment-matched; hardware yaw nuisance
        # belongs in a separately frozen real-data contract.
        yaw_rad = 0.0
        sensor_delay_s = 0.002 + 0.006 * float(sample[3])
        nuisance_seed = seed * 100 + orbit_index
        sensor_seed = seed * 10000 + orbit_index
        phase_key = int(round(10.0 * phase_deg))
        orbit_id = f"{size}_{role}_{state}_r{ring_index}_p{phase_key:03d}"
        outer_signed = (
            geometry["head_radius_m"]
            - (radius_m + geometry["bit_radius_m"])
        )
        inner_signed = (
            radius_m
            - geometry["bit_radius_m"]
            - 0.5 * float(spec["derived"]["socket_top_af_m"])
        )
        for sector in SECTORS:
            angle_deg = phase_deg + 60.0 * sector
            angle_rad = math.radians(angle_deg)
            trial_id = (
                f"{ROLE_PREFIX[role]}_{size}_{state}_r{ring_index}_"
                f"p{phase_key:03d}_s{sector}"
            )
            role_tasks[role].append(
                PilotTask(
                    trial_id=trial_id,
                    base_pose_id=trial_id,
                    search_state=state,
                    radial_level=ring_index,
                    radius_m=radius_m,
                    base_angle_deg=float(phase_deg),
                    rotation_index=sector,
                    rotation_deg=60.0 * sector,
                    x_m=radius_m * math.cos(angle_rad),
                    y_m=radius_m * math.sin(angle_rad),
                    nominal_tip_z_m=0.00010,
                    outer_signed_distance_m=float(outer_signed),
                    inner_signed_distance_m=float(inner_signed),
                    nuisance_id=f"s4_{role}_{size}_sobol_{orbit_index:02d}",
                    friction_slide=float(friction),
                    contact_timeconst_s=float(contact_timeconst),
                    target_down_force_n=float(down_force),
                    yaw_rad=yaw_rad,
                    nuisance_seed=nuisance_seed,
                    symmetry_orbit_id=orbit_id,
                    sensor_noise_seed=sensor_seed,
                    sensor_delay_s=float(sensor_delay_s),
                )
            )
    validate_role_tasks(role_tasks, spec)
    return role_tasks


def validate_role_tasks(
    role_tasks: dict[str, list[PilotTask]], spec: dict
) -> None:
    expected_counts = {"train": 60, "validation": 30, "calibration": 30}
    geometry = deployment_geometry(spec)
    orbit_roles: dict[str, str] = {}
    for role, tasks in role_tasks.items():
        if role not in expected_counts or len(tasks) != expected_counts[role]:
            raise ValueError(f"{role}: unexpected S4 scan count")
        by_orbit: dict[str, list[PilotTask]] = {}
        for task in tasks:
            by_orbit.setdefault(task.symmetry_orbit_id, []).append(task)
            previous = orbit_roles.setdefault(task.symmetry_orbit_id, role)
            if previous != role:
                raise ValueError("a C6 orbit crossed S4 data roles")
            if not (
                geometry["radius_min_m"]
                <= task.radius_m
                <= geometry["visual_bound_m"]
            ):
                raise ValueError("S4 task is outside the deployment annulus")
            if task.sensor_noise_seed is None:
                raise ValueError("S4 adaptation requires the observation-chain prior")
            if not 0.002 <= task.sensor_delay_s <= 0.008:
                raise ValueError("S4 sensor delay is outside the frozen prior")
        for orbit_id, orbit in by_orbit.items():
            if len(orbit) != len(SECTORS):
                raise ValueError(f"{orbit_id}: incomplete C6 orbit")
            if {task.rotation_index for task in orbit} != set(SECTORS):
                raise ValueError(f"{orbit_id}: duplicate/missing C6 sector")
            invariant_fields = {
                (
                    task.radius_m,
                    task.base_angle_deg,
                    task.friction_slide,
                    task.contact_timeconst_s,
                    task.target_down_force_n,
                    task.sensor_noise_seed,
                    task.sensor_delay_s,
                )
                for task in orbit
            }
            if len(invariant_fields) != 1:
                raise ValueError(f"{orbit_id}: nuisance changed inside C6 orbit")
    intersections = [
        set(task.symmetry_orbit_id for task in role_tasks[left])
        & set(task.symmetry_orbit_id for task in role_tasks[right])
        for index, left in enumerate(ROLE_PHASES_DEG)
        for right in tuple(ROLE_PHASES_DEG)[index + 1 :]
    ]
    if any(intersections):
        raise ValueError("S4 C6 orbit leakage")


def build_dataset_contract(spec: dict, size: str, seed: int) -> dict:
    role_tasks = build_role_tasks(spec, size, seed)
    geometry = deployment_geometry(spec)
    rings = deployment_rings(spec)
    spec_path = ROOT / "configs" / f"allen_{size}_source.json"
    model_path = ROOT / "models" / f"allen_{size}_source.xml"
    role_orbits = {
        role: sorted({task.symmetry_orbit_id for task in tasks})
        for role, tasks in role_tasks.items()
    }
    observation_prior = WrenchObservationConfig().to_dict()
    observation_prior["seed"] = "per_orbit_frozen"
    observation_prior["delay_s"] = [0.002, 0.008]
    observation_prior.pop("delay_steps", None)
    return {
        "contract_version": "cross_size_s4_v1",
        "stage": "cross_size_S4_deployment_matched_state_direction_adaptation_data",
        "size": size,
        "generator_seed": seed,
        "phase_bins": 360,
        "data_roles_frozen_before_sampling": True,
        "split_unit": "complete physical C6 symmetry_orbit_id",
        "role_semantics": {
            "train": "state/direction head weight updates only",
            "validation": "early stopping and model selection only",
            "calibration": "temperature, APS and final descriptive metrics only",
            "future_full_loop_test": "forbidden until model/controller/gates are frozen",
        },
        "role_counts": {
            role: {
                "orbits": len(role_orbits[role]),
                "scans": len(role_tasks[role]),
                "state_scans": dict(
                    Counter(task.search_state for task in role_tasks[role])
                ),
            }
            for role in ROLE_PHASES_DEG
        },
        "role_phase_anchors_deg": {
            role: list(phases) for role, phases in ROLE_PHASES_DEG.items()
        },
        "role_orbit_ids": role_orbits,
        "pairwise_orbit_intersections": {
            f"{left}__{right}": sorted(set(role_orbits[left]) & set(role_orbits[right]))
            for index, left in enumerate(ROLE_PHASES_DEG)
            for right in tuple(ROLE_PHASES_DEG)[index + 1 :]
        },
        "deployment_geometry_mm": {
            key.removesuffix("_m") + "_mm": 1000.0 * value
            for key, value in geometry.items()
        },
        "deployment_rings": list(rings),
        "observation_prior": observation_prior,
        "nuisance_ranges": {
            "friction_slide": [0.55, 0.95],
            "contact_timeconst_s": [0.040, 0.070],
            "target_down_force_n": [2.5, 3.5],
            "yaw_rad": [0.0, 0.0],
            "sensor_delay_s": [0.002, 0.008],
        },
        "source_geometry": {
            "spec": str(spec_path.relative_to(ROOT)),
            "spec_sha256": _sha256(spec_path),
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": _sha256(model_path),
        },
        "source_provenance": {
            "contract_generator": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "dynamic_scan_sampler": {
                "path": "src/sample_m8_conical_pilot.py",
                "sha256": _sha256(ROOT / "src/sample_m8_conical_pilot.py"),
            },
            "observation_model": {
                "path": "src/wrench_observation.py",
                "sha256": _sha256(ROOT / "src/wrench_observation.py"),
            },
        },
        "excluded_artifacts": [
            "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3",
            "outputs/current/evaluations/m8_model_m10_m12_zero_shot",
            "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop",
        ],
        "claim_boundary": (
            "Simulation-only observed-wrench adaptation data inside the frozen visual "
            "deployment annulus; no full-loop test outcome or real F/T sample is used."
        ),
        "tasks": {
            role: [asdict(task) for task in role_tasks[role]]
            for role in ROLE_PHASES_DEG
        },
        "created_utc": _utc_now(),
    }


def _write_dataset_manifest(
    output: Path, size: str, contract: dict, *, sampled: bool
) -> dict:
    contract_path = output / size / "dataset_contract.json"
    if not sampled:
        manifest = {
            "status": "contract_only",
            "size": size,
            "dataset_contract_sha256": _sha256(contract_path),
            "role_counts": contract["role_counts"],
            "C6_orbit_leakage": False,
            "trial_files": [],
            "updated_utc": _utc_now(),
        }
        _atomic_json(output / size / "dataset_manifest.json", manifest)
        return manifest

    role_by_trial = {
        task["trial_id"]: role
        for role, tasks in contract["tasks"].items()
        for task in tasks
    }
    trial_paths = sorted((output / size / "trials").glob("*.npz"))
    if len(trial_paths) != len(role_by_trial):
        raise RuntimeError(
            f"{size}: expected {len(role_by_trial)} S4 scans, found {len(trial_paths)}"
        )
    files = []
    statuses: Counter[str] = Counter()
    for path in trial_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        trial_id = str(metadata["trial_id"])
        if trial_id not in role_by_trial:
            raise RuntimeError(f"{size}: uncontracted S4 trial {trial_id}")
        status = str(metadata.get("status", "unknown"))
        statuses[status] += 1
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "trial_id": trial_id,
                "symmetry_orbit_id": metadata["symmetry_orbit_id"],
                "role": role_by_trial[trial_id],
                "status": status,
            }
        )
    if statuses != {"ok": len(role_by_trial)}:
        raise RuntimeError(f"{size}: non-ok S4 scans {dict(statuses)}")
    manifest = {
        "status": "complete",
        "size": size,
        "dataset_contract_sha256": _sha256(contract_path),
        "role_counts": contract["role_counts"],
        "C6_orbit_leakage": False,
        "trial_files": files,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / size / "dataset_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    size_manifests = {}
    for size in args.sizes:
        spec = json.loads(
            (ROOT / "configs" / f"allen_{size}_source.json").read_text(
                encoding="utf-8"
            )
        )
        contract = build_dataset_contract(spec, size, SEEDS[size])
        contract_path = output / size / "dataset_contract.json"
        _atomic_json(contract_path, contract)
        if args.sample:
            tasks = [
                PilotTask(**task)
                for role in ROLE_PHASES_DEG
                for task in contract["tasks"][role]
            ]
            _sample_size(size, tasks, output / size, workers=args.workers)
        manifest = _write_dataset_manifest(
            output, size, contract, sampled=bool(args.sample)
        )
        size_manifests[size] = {
            "path": str((output / size / "dataset_manifest.json").relative_to(ROOT))
            if output.is_relative_to(ROOT)
            else str(output / size / "dataset_manifest.json"),
            "sha256": _sha256(output / size / "dataset_manifest.json"),
            "status": manifest["status"],
        }
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete" if args.sample else "contract_only",
            "stage": "cross_size_S4_deployment_matched_adaptation_data",
            "sizes": list(args.sizes),
            "sampling_explicitly_requested": bool(args.sample),
            "size_manifests": size_manifests,
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deployment-matched S4 C6 contracts; sampling is opt-in because "
            "the 240 dynamic scans are intentionally not launched by default."
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=min(24, max(1, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
