"""Build the leakage-safe S4b outer-boundary adaptation-data contract.

S4b is a *development successor* to S4.  The completed development-v2 audit
identified an aggregate observability gap near the full-loop visual boundary.
That finding is used only to choose a generic coverage axis: one additional
complete C6 ring at the geometry-declared visual bound.  This generator never
loads a development task pose, nuisance draw, rollout trace, or raw fingerprint.

The original ``cross_size_s4_adaptation`` tree is neither read nor written.
Dynamic sampling remains explicitly opt-in; the default invocation is contract
only.
"""

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
from src.sample_cross_size_s4_adaptation import (
    SECTORS,
    SIZES,
    deployment_geometry,
    deployment_rings as s4_deployment_rings,
)
from src.sample_m8_conical_pilot import PilotTask
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/current/datasets/cross_size_s4b_adaptation"
INTENDED_MODEL_OUTPUT = (
    ROOT / "outputs/current/models/cross_size_s4b_state_direction_heads"
)
S4_DATA_ROOT = ROOT / "outputs/current/datasets/cross_size_s4_adaptation"
DEVELOPMENT_V2_REPORT = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4_full_loop_development_v2/report.json"
)

# New seeds make every nuisance and observed-wrench seed distinct from S4.
SEEDS = {"m10": 20261131, "m12": 20261132}

# Complete C6 orbits, and not individual scans, are the split unit.  Phase
# anchors are disjoint across all three roles.
ROLE_PHASES_DEG: dict[str, tuple[float, ...]] = {
    "train": (7.5, 37.5),
    "validation": (22.5,),
    "calibration": (52.5,),
}
ROLE_PREFIX = {
    "train": "s4btr",
    "validation": "s4bva",
    "calibration": "s4bca",
}


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def deployment_rings(spec: dict) -> tuple[dict, ...]:
    """Return the five S4 rings plus a generic visual-boundary C6 ring."""

    geometry = deployment_geometry(spec)
    base_rings = tuple(dict(ring) for ring in s4_deployment_rings(spec))
    visual_bound = float(geometry["visual_bound_m"])
    outer_ring = {
        "ring_index": len(base_rings),
        "radius_m": visual_bound,
        "normalized_visual_radius": 1.0,
        "search_state": "double_overhang",
        "coverage_role": "generic_full_loop_visual_boundary",
        "radius_source": "max(source_geometry.visual_offset_sweep_m)",
    }
    rings = base_rings + (outer_ring,)
    if len(rings) != 6:
        raise ValueError("S4b requires exactly five inherited rings plus one boundary ring")
    if not math.isclose(
        float(rings[-1]["radius_m"]), visual_bound, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("S4b outer ring must equal the full-loop visual bound")
    if visual_bound >= float(geometry["inner_overlap_radius_m"]):
        raise ValueError("S4b visual-boundary ring left the double-overhang domain")
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
        sensor_delay_s = 0.002 + 0.006 * float(sample[3])
        nuisance_seed = seed * 100 + orbit_index
        sensor_seed = seed * 10000 + orbit_index
        phase_key = int(round(10.0 * phase_deg))
        orbit_id = f"{size}_{role}_{state}_s4b_r{ring_index}_p{phase_key:03d}"
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
                    nuisance_id=f"s4b_{role}_{size}_sobol_{orbit_index:02d}",
                    friction_slide=float(friction),
                    contact_timeconst_s=float(contact_timeconst),
                    target_down_force_n=float(down_force),
                    yaw_rad=0.0,
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
    rings = deployment_rings(spec)
    geometry = deployment_geometry(spec)
    expected_counts = {
        role: len(phases) * len(rings) * len(SECTORS)
        for role, phases in ROLE_PHASES_DEG.items()
    }
    orbit_roles: dict[str, str] = {}
    role_pose_sets: dict[str, set[tuple[float, float]]] = {}
    for role, tasks in role_tasks.items():
        if role not in expected_counts or len(tasks) != expected_counts[role]:
            raise ValueError(f"{role}: unexpected S4b scan count")
        by_orbit: dict[str, list[PilotTask]] = {}
        role_pose_sets[role] = set()
        for task in tasks:
            by_orbit.setdefault(task.symmetry_orbit_id, []).append(task)
            previous = orbit_roles.setdefault(task.symmetry_orbit_id, role)
            if previous != role:
                raise ValueError("a C6 orbit crossed S4b data roles")
            if not (
                geometry["radius_min_m"]
                <= task.radius_m
                <= geometry["visual_bound_m"]
            ):
                raise ValueError("S4b task is outside the deployment annulus")
            if task.sensor_noise_seed is None:
                raise ValueError("S4b adaptation requires the observation-chain prior")
            if not 0.002 <= task.sensor_delay_s <= 0.008:
                raise ValueError("S4b sensor delay is outside the frozen prior")
            role_pose_sets[role].add((round(task.x_m, 12), round(task.y_m, 12)))
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
    roles = tuple(ROLE_PHASES_DEG)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if {
                task.symmetry_orbit_id for task in role_tasks[left]
            } & {task.symmetry_orbit_id for task in role_tasks[right]}:
                raise ValueError("S4b C6 orbit leakage")
            if role_pose_sets[left] & role_pose_sets[right]:
                raise ValueError("S4b physical pose crossed data roles")


def build_dataset_contract(spec: dict, size: str, seed: int) -> dict:
    role_tasks = build_role_tasks(spec, size, seed)
    geometry = deployment_geometry(spec)
    rings = deployment_rings(spec)
    spec_path = ROOT / "configs" / f"allen_{size}_source.json"
    model_path = ROOT / "models" / f"allen_{size}_source.xml"
    generator_path = Path(__file__).resolve()
    s4_generator_path = ROOT / "src/sample_cross_size_s4_adaptation.py"
    role_orbits = {
        role: sorted({task.symmetry_orbit_id for task in tasks})
        for role, tasks in role_tasks.items()
    }
    observation_prior = WrenchObservationConfig().to_dict()
    observation_prior["seed"] = "new_per_orbit_frozen_S4b_seed"
    observation_prior["delay_s"] = [0.002, 0.008]
    observation_prior.pop("delay_steps", None)
    return {
        "contract_version": "cross_size_s4b_v1",
        "stage": "cross_size_S4b_generic_outer_boundary_state_direction_adaptation_data",
        "size": size,
        "generator_seed": seed,
        "phase_bins": 360,
        "data_roles_frozen_before_sampling": True,
        "split_unit": "complete physical C6 symmetry_orbit_id",
        "role_semantics": {
            "train": "state/direction head weight updates only",
            "validation": "early stopping and model selection only",
            "calibration": "temperature, APS and final descriptive metrics only",
            "future_full_loop_test": (
                "must be newly generated after S4b model/controller/gates are frozen; "
                "development-v2 tasks are permanently ineligible"
            ),
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
            f"{left}__{right}": sorted(
                set(role_orbits[left]) & set(role_orbits[right])
            )
            for index, left in enumerate(ROLE_PHASES_DEG)
            for right in tuple(ROLE_PHASES_DEG)[index + 1 :]
        },
        "pairwise_physical_pose_intersections": {
            f"{left}__{right}": []
            for index, left in enumerate(ROLE_PHASES_DEG)
            for right in tuple(ROLE_PHASES_DEG)[index + 1 :]
        },
        "deployment_geometry_mm": {
            key.removesuffix("_m") + "_mm": 1000.0 * value
            for key, value in geometry.items()
        },
        "deployment_rings": list(rings),
        "outer_boundary_addition": {
            "ring_index": 5,
            "radius_formula": "r_5 = max(source_geometry.visual_offset_sweep_m)",
            "normalized_visual_radius": 1.0,
            "state": "double_overhang",
            "complete_C6_orbits_per_role": {
                role: len(phases) for role, phases in ROLE_PHASES_DEG.items()
            },
            "construction": (
                "generic geometry-declared boundary; not copied from a failed "
                "full-loop task radius or direction"
            ),
        },
        "observation_prior": observation_prior,
        "nuisance_ranges": {
            "friction_slide": [0.55, 0.95],
            "contact_timeconst_s": [0.040, 0.070],
            "target_down_force_n": [2.5, 3.5],
            "yaw_rad": [0.0, 0.0],
            "sensor_delay_s": [0.002, 0.008],
        },
        "development_trigger": {
            "finding": "aggregate outer-boundary observability gap",
            "allowed_use": (
                "choose the generic visual-boundary coverage axis only; S4b remains "
                "development data and cannot support a confirmatory claim"
            ),
            "aggregate_report": {
                "path": _relative(DEVELOPMENT_V2_REPORT),
                "sha256": _sha256(DEVELOPMENT_V2_REPORT),
            },
            "old_task_pose_records_reused": 0,
            "old_nuisance_records_reused": 0,
            "old_raw_fingerprints_reused": 0,
            "development_trial_fields_loaded_by_generator": [],
        },
        "independence_contract": {
            "tasks_generated_ab_initio": True,
            "geometry_only_pose_design": True,
            "new_nuisance_and_sensor_seeds": True,
            "reads_existing_S4_dataset": False,
            "writes_existing_S4_dataset": False,
            "existing_S4_dataset_root": _relative(S4_DATA_ROOT),
            "intended_model_output": _relative(INTENDED_MODEL_OUTPUT),
        },
        "source_geometry": {
            "spec": _relative(spec_path),
            "spec_sha256": _sha256(spec_path),
            "model": _relative(model_path),
            "model_sha256": _sha256(model_path),
        },
        "source_provenance": {
            "contract_generator": {
                "path": _relative(generator_path),
                "sha256": _sha256(generator_path),
            },
            "five_ring_geometry_dependency": {
                "path": _relative(s4_generator_path),
                "sha256": _sha256(s4_generator_path),
                "role": "code/formula dependency only; no S4 dataset artifact is read",
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
        "prohibited_evaluation_artifacts": [
            "outputs/current/diagnostics/cross_size_s4_full_loop_development",
            "outputs/current/diagnostics/cross_size_s4_full_loop_development_v2",
            "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3",
        ],
        "claim_boundary": (
            "Simulation-only, development-informed but pose/nuisance/fingerprint-"
            "independent S4b adaptation contract. It adds generic complete-orbit "
            "coverage at the full-loop visual bound. No full-loop outcome, real F/T "
            "sample, or confirmatory evidence is created by this contract."
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
            "physical_pose_leakage": False,
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
            f"{size}: expected {len(role_by_trial)} S4b scans, found {len(trial_paths)}"
        )
    files = []
    statuses: Counter[str] = Counter()
    for path in trial_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        trial_id = str(metadata["trial_id"])
        if trial_id not in role_by_trial:
            raise RuntimeError(f"{size}: uncontracted S4b trial {trial_id}")
        status = str(metadata.get("status", "unknown"))
        statuses[status] += 1
        files.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "trial_id": trial_id,
                "symmetry_orbit_id": metadata["symmetry_orbit_id"],
                "role": role_by_trial[trial_id],
                "status": status,
            }
        )
    if statuses != {"ok": len(role_by_trial)}:
        raise RuntimeError(f"{size}: non-ok S4b scans {dict(statuses)}")
    manifest = {
        "status": "complete",
        "size": size,
        "dataset_contract_sha256": _sha256(contract_path),
        "role_counts": contract["role_counts"],
        "C6_orbit_leakage": False,
        "physical_pose_leakage": False,
        "trial_files": files,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / size / "dataset_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output == S4_DATA_ROOT.resolve():
        raise ValueError("S4b refuses to overwrite the existing S4 dataset root")
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
            "path": _relative(output / size / "dataset_manifest.json"),
            "sha256": _sha256(output / size / "dataset_manifest.json"),
            "status": manifest["status"],
        }
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete" if args.sample else "contract_only",
            "stage": "cross_size_S4b_generic_outer_boundary_adaptation_data",
            "sizes": list(args.sizes),
            "sampling_explicitly_requested": bool(args.sample),
            "existing_S4_dataset_written": False,
            "intended_model_output": _relative(INTENDED_MODEL_OUTPUT),
            "size_manifests": size_manifests,
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent S4b six-ring C6 contracts. Sampling is opt-in; "
            "the default invocation writes contracts only."
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
