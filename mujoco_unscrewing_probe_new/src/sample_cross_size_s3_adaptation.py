from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from src.audit_m8_model_cross_size_zero_shot import (
    _sample_size,
    build_cross_size_tasks,
)
from src.sample_m8_conical_pilot import PilotTask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/datasets/cross_size_s3_adaptation_pilot"
)
SIZES = ("m10", "m12")
SEEDS = {"m10": 20260831, "m12": 20260832}
EXPECTED_ORBITS_PER_ROLE = 6
EXPECTED_SCANS_PER_ORBIT = 6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_for_orbit(task: PilotTask) -> str:
    """Checkerboard split: two train and two calibration orbits per state."""

    local_index = int(str(task.symmetry_orbit_id).rsplit("_l", 1)[1])
    return "train" if (int(task.radial_level) + local_index) % 2 == 0 else "calibration"


def build_role_split(tasks: list[PilotTask]) -> dict[str, list[PilotTask]]:
    by_orbit: dict[str, list[PilotTask]] = defaultdict(list)
    for task in tasks:
        by_orbit[str(task.symmetry_orbit_id)].append(task)

    split: dict[str, list[PilotTask]] = {"train": [], "calibration": []}
    orbit_roles: dict[str, str] = {}
    for orbit_id, orbit in sorted(by_orbit.items()):
        if len(orbit) != EXPECTED_SCANS_PER_ORBIT:
            raise ValueError(f"{orbit_id}: expected a complete C6 orbit")
        sectors = {int(task.rotation_index) for task in orbit}
        if sectors != set(range(EXPECTED_SCANS_PER_ORBIT)):
            raise ValueError(f"{orbit_id}: incomplete or duplicate C6 sectors")
        roles = {_role_for_orbit(task) for task in orbit}
        if len(roles) != 1:
            raise ValueError(f"{orbit_id}: an orbit crossed data roles")
        role = roles.pop()
        orbit_roles[orbit_id] = role
        split[role].extend(orbit)

    train_orbits = {task.symmetry_orbit_id for task in split["train"]}
    calibration_orbits = {
        task.symmetry_orbit_id for task in split["calibration"]
    }
    if train_orbits & calibration_orbits:
        raise ValueError("C6 orbit leakage between train and calibration")
    if len(train_orbits) != EXPECTED_ORBITS_PER_ROLE:
        raise ValueError("unexpected number of training C6 orbits")
    if len(calibration_orbits) != EXPECTED_ORBITS_PER_ROLE:
        raise ValueError("unexpected number of calibration C6 orbits")

    for role, selected in split.items():
        state_orbits: dict[str, set[str]] = defaultdict(set)
        for task in selected:
            state_orbits[task.search_state].add(task.symmetry_orbit_id)
        if {state: len(orbits) for state, orbits in state_orbits.items()} != {
            "supported_search": 2,
            "double_overhang": 2,
            "outer_partial": 2,
        }:
            raise ValueError(f"{role}: state/orbit imbalance")
    return split


def _fresh_task(task: PilotTask, role: str) -> PilotTask:
    prefix = "s3tr" if role == "train" else "s3cal"
    trial_id = f"{prefix}_{task.trial_id}"
    return replace(
        task,
        trial_id=trial_id,
        base_pose_id=trial_id,
        nuisance_id=f"{prefix}_{task.nuisance_id}",
    )


def _write_contract(
    output: Path,
    size: str,
    seed: int,
    split: dict[str, list[PilotTask]],
) -> dict:
    role_tasks = {
        role: [_fresh_task(task, role) for task in tasks]
        for role, tasks in split.items()
    }
    train_orbits = sorted(
        {task.symmetry_orbit_id for task in role_tasks["train"]}
    )
    calibration_orbits = sorted(
        {task.symmetry_orbit_id for task in role_tasks["calibration"]}
    )
    spec_path = ROOT / "configs" / f"allen_{size}_source.json"
    model_path = ROOT / "models" / f"allen_{size}_source.xml"
    contract = {
        "stage": "cross_size_S3_fresh_direction_head_adaptation_data",
        "size": size,
        "generator_seed": seed,
        "phase_bins": 360,
        "data_roles_frozen_before_sampling": True,
        "split_unit": "complete physical C6 symmetry_orbit_id before sampling",
        "role_counts": {
            role: {
                "orbits": len({task.symmetry_orbit_id for task in tasks}),
                "scans": len(tasks),
                "state_orbits": dict(
                    Counter(
                        next(
                            task.search_state
                            for task in tasks
                            if task.symmetry_orbit_id == orbit
                        )
                        for orbit in {task.symmetry_orbit_id for task in tasks}
                    )
                ),
            }
            for role, tasks in role_tasks.items()
        },
        "train_orbit_ids": train_orbits,
        "calibration_orbit_ids": calibration_orbits,
        "orbit_intersection": sorted(set(train_orbits) & set(calibration_orbits)),
        "training_use": {
            "train": "allowed_only_for_direction_head_weight_updates",
            "calibration": "allowed_only_for_temperature_APS_and_reporting",
            "future_full_loop_test": "forbidden_until_all_models_and_gates_are_frozen",
        },
        "excluded_prior_artifacts": [
            "outputs/current/evaluations/m8_model_m10_m12_zero_shot",
            "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop",
        ],
        "source_geometry": {
            "spec": str(spec_path.relative_to(ROOT)),
            "spec_sha256": _sha256(spec_path),
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": _sha256(model_path),
        },
        "tasks": {
            role: [asdict(task) for task in tasks]
            for role, tasks in role_tasks.items()
        },
        "created_utc": _utc_now(),
    }
    _atomic_json(output / size / "dataset_contract.json", contract)
    return contract


def _write_completed_manifest(output: Path, size: str, contract: dict) -> None:
    trial_paths = sorted((output / size / "trials").glob("*.npz"))
    expected = sum(item["scans"] for item in contract["role_counts"].values())
    if len(trial_paths) != expected:
        raise RuntimeError(f"{size}: expected {expected} trials, found {len(trial_paths)}")
    role_by_trial = {
        task["trial_id"]: role
        for role, tasks in contract["tasks"].items()
        for task in tasks
    }
    files = []
    status_counts: Counter[str] = Counter()
    for path in trial_paths:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        trial_id = str(metadata["trial_id"])
        if trial_id not in role_by_trial:
            raise RuntimeError(f"{size}: uncontracted trial {trial_id}")
        status_counts[str(metadata.get("status", "unknown"))] += 1
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "trial_id": trial_id,
                "symmetry_orbit_id": metadata["symmetry_orbit_id"],
                "role": role_by_trial[trial_id],
                "status": metadata.get("status"),
            }
        )
    if status_counts != {"ok": expected}:
        raise RuntimeError(f"{size}: failed adaptation scans: {dict(status_counts)}")
    _atomic_json(
        output / size / "dataset_manifest.json",
        {
            "status": "complete",
            "size": size,
            "dataset_contract_sha256": _sha256(
                output / size / "dataset_contract.json"
            ),
            "role_counts": contract["role_counts"],
            "C6_orbit_leakage": False,
            "trial_files": files,
            "updated_utc": _utc_now(),
        },
    )


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        spec = json.loads(
            (ROOT / "configs" / f"allen_{size}_source.json").read_text(
                encoding="utf-8"
            )
        )
        seed = SEEDS[size]
        source_tasks = build_cross_size_tasks(spec, size, seed=seed)
        split = build_role_split(source_tasks)
        contract = _write_contract(output, size, seed, split)
        ordered_tasks = [
            PilotTask(**task)
            for role in ("train", "calibration")
            for task in contract["tasks"][role]
        ]
        if not args.skip_sampling:
            _sample_size(size, ordered_tasks, output / size, workers=args.workers)
        _write_completed_manifest(output, size, contract)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "stage": "cross_size_S3_fresh_adaptation_data",
            "sizes": list(args.sizes),
            "size_manifests": {
                size: {
                    "path": str(
                        (output / size / "dataset_manifest.json").relative_to(ROOT)
                    ),
                    "sha256": _sha256(output / size / "dataset_manifest.json"),
                }
                for size in args.sizes
            },
            "old_S1_S2_training_samples_used": 0,
            "updated_utc": _utc_now(),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample fresh, C6-isolated M10/M12 S3 adaptation data."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument(
        "--workers", type=int, default=min(24, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
