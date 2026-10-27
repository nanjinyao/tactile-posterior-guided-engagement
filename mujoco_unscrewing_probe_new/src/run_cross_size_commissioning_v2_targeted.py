from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_commissioning import commissioning_for_size
from src.funnel_guard import FunnelGuardConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
from src.run_cross_size_closed_loop_audit import (
    CrossSizeTask,
    _size_contract as base_size_contract,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4_full_loop_development_v2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/diagnostics/cross_size_commissioning_v2_targeted"
)
S4_MODEL_ROOT = ROOT / "outputs/current/models/cross_size_s4b_state_direction_heads"
ACTIVE_TIME_BUDGET_S = 130.0
METHODS = ("privileged_straight_to_center", "s4b_fine_tuned")
RUNTIME_METHOD = {
    "privileged_straight_to_center": "oracle",
    "s4b_fine_tuned": "hybrid",
}
BASE_R3_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)


_OUTPUT: Path | None = None
_METHODS: tuple[str, ...] = ()
_RUNTIME: M8DirectionEnsemble | None = None
_SIZE: str | None = None
_CONTRACT_SHA256: str | None = None


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
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _file_record(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        relative = resolved
    return {"path": str(relative), "sha256": _sha256(resolved)}


def _resolve_record_path(record: dict) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else ROOT / path


def _source_trial_paths(source: Path, size: str) -> list[Path]:
    return sorted((source / size / "trials").glob("*.json"))


def select_oracle_failure_pose_ids(source: Path, size: str) -> tuple[str, ...]:
    """Select only known development poses where the v1 oracle did not finish."""

    selected: list[str] = []
    for path in _source_trial_paths(source, size):
        record = json.loads(path.read_text(encoding="utf-8"))
        oracle = record["methods"]["privileged_straight_to_center"]
        if not _strict_bottom_success(oracle):
            selected.append(str(record["task"]["pose_id"]))
    if not selected:
        raise ValueError(f"{size}: source contains no failed oracle development pose")
    return tuple(selected)


def _load_selected_tasks(
    source: Path, size: str, pose_ids: tuple[str, ...]
) -> tuple[list[CrossSizeTask], dict[str, Path]]:
    requested = set(pose_ids)
    task_by_id: dict[str, CrossSizeTask] = {}
    path_by_id: dict[str, Path] = {}
    for path in _source_trial_paths(source, size):
        record = json.loads(path.read_text(encoding="utf-8"))
        task = CrossSizeTask(**record["task"])
        if task.pose_id in requested:
            task_by_id[task.pose_id] = task
            path_by_id[task.pose_id] = path
    missing = requested - set(task_by_id)
    if missing:
        raise ValueError(f"unknown development pose ids: {sorted(missing)}")
    ordered = [task_by_id[pose_id] for pose_id in pose_ids]
    return ordered, path_by_id


def _execution_files(
    source: Path, size: str, trial_paths: dict[str, Path]
) -> dict:
    model_root = S4_MODEL_ROOT / size / "fine_tuned"
    model_contract = json.loads(
        (model_root / "ensemble_interface_contract.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoints = []
    for value in model_contract["checkpoints"]:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        checkpoints.extend(
            [
                _file_record(path),
                _file_record(path.parent / "normalization.npz"),
            ]
        )
    size_contract = base_size_contract(size)
    return {
        "runner": _file_record(Path(__file__)),
        "commissioning": _file_record(ROOT / "src/cross_size_commissioning.py"),
        "rollout": _file_record(ROOT / "src/render_m8_search_comparison.py"),
        "source_report": _file_record(source / "report.json"),
        "source_trials": {
            pose_id: _file_record(path)
            for pose_id, path in sorted(trial_paths.items())
        },
        "cad_spec": _file_record(Path(size_contract["spec_path"])),
        "mujoco_model": _file_record(Path(size_contract["model_path"])),
        "s4_model_contract": _file_record(
            model_root / "ensemble_interface_contract.json"
        ),
        "s4_model_members": checkpoints,
    }


def _validate_execution_files(records: object) -> None:
    if isinstance(records, dict) and set(records) == {"path", "sha256"}:
        path = _resolve_record_path(records)
        if _sha256(path) != records["sha256"]:
            raise RuntimeError(f"targeted commissioning execution file drift: {path}")
        return
    if isinstance(records, dict):
        for value in records.values():
            _validate_execution_files(value)
        return
    if isinstance(records, list):
        for value in records:
            _validate_execution_files(value)


def build_contract(
    source: Path,
    size: str,
    pose_ids: tuple[str, ...],
    methods: tuple[str, ...],
) -> tuple[dict, list[CrossSizeTask]]:
    if size != "m10":
        raise ValueError("this targeted diagnosis is intentionally restricted to M10")
    if not pose_ids:
        raise ValueError("at least one known failed development pose is required")
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a subset of {METHODS}")
    tasks, trial_paths = _load_selected_tasks(source, size, pose_ids)
    commissioned = commissioning_for_size(
        size,
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    if tuple(commissioned.admittance.hard_wrench) != tuple(
        AdmittanceConfig().hard_wrench
    ):
        raise RuntimeError("commissioning v2 relaxed a raw hard-wrench threshold")
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if source_report.get("phase") != "development":
        raise ValueError("targeted commissioning may only reuse development tasks")
    contract = {
        "experiment": "cross_size_commissioning_v2_targeted_development_diagnostic",
        "phase": "development",
        "size": size,
        "pose_ids": list(pose_ids),
        "methods": list(methods),
        "tasks": [asdict(task) for task in tasks],
        "source": str(source.resolve()),
        "source_controller_label": "commissioning_v1_full_s_time_and_descent",
        "commissioning_v2": asdict(commissioned),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "fallback_budget_extension_s": 0.0,
        "raw_hard_wrench_thresholds": list(AdmittanceConfig().hard_wrench),
        "raw_hard_wrench_thresholds_relaxed": False,
        "selection_rule": (
            "known M10 development poses where the source privileged "
            "straight-to-center arm failed strict CAD-bottom completion"
        ),
        "untouched_tasks_or_seeds_read": False,
        "claim_boundary": (
            "Targeted reuse of known failed development poses. Results diagnose "
            "commissioning mechanics only and cannot support an untouched or "
            "confirmatory full-loop claim."
        ),
        "execution_files": _execution_files(source, size, trial_paths),
        "created_utc": _utc_now(),
    }
    return contract, tasks


def _worker_init(
    output: str,
    size: str,
    methods: tuple[str, ...],
    contract_sha256: str,
) -> None:
    global _OUTPUT, _METHODS, _RUNTIME, _SIZE, _CONTRACT_SHA256
    torch.set_num_threads(1)
    _OUTPUT = Path(output)
    _METHODS = methods
    _SIZE = size
    _CONTRACT_SHA256 = contract_sha256
    _RUNTIME = (
        M8DirectionEnsemble(S4_MODEL_ROOT / size / "fine_tuned")
        if "s4b_fine_tuned" in methods
        else None
    )


def _run_method(task: CrossSizeTask, method: str) -> dict:
    contract = base_size_contract(task.size)
    commissioned = commissioning_for_size(
        task.size,
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    return asdict(
        run_method(
            RUNTIME_METHOD[method],
            initial_xy_m=np.asarray([task.x_m, task.y_m], dtype=float),
            direction_runtime=_RUNTIME if method == "s4b_fine_tuned" else None,
            visual_sigma_m=float(contract["visual_sigma_m"]),
            visual_bound_m=float(contract["visual_bound_m"]),
            posterior_coverage=0.99,
            posterior_seed=task.posterior_seed,
            max_transport_m=float(contract["max_transport_m"]),
            baseline_phase_rad=task.baseline_phase_rad,
            baseline_max_radius_m=float(contract["baseline_max_radius_m"]),
            target_down_force_n=task.target_down_force_n,
            friction_slide=task.friction_slide,
            contact_timeconst_s=task.contact_timeconst_s,
            global_outward_fallback=True,
            active_time_budget_s=ACTIVE_TIME_BUDGET_S,
            continue_to_bottom=True,
            wrench_observation=WrenchObservationConfig(
                seed=task.sensor_noise_seed, delay_s=task.sensor_delay_s
            ),
            admittance_config=commissioned.admittance,
            observation_label=(
                f"targeted commissioning v2 {task.size} {method} development"
            ),
            spec_path=contract["spec_path"],
            model_path=contract["model_path"],
            spiral_pitch_m=float(contract["spiral_pitch_m"]),
            xy_speed_m_s=float(contract["xy_speed_m_s"]),
            cad_conditioned_insertion=True,
            conditioned_on_ambiguous_top_contact=True,
            funnel_guard=commissioned.funnel_guard,
            oracle_transport_soft_escape_recovery=(
                commissioned.oracle_transport_soft_escape_recovery
            ),
            oracle_center_tolerance_capture_fraction=(
                commissioned.oracle_center_tolerance_capture_fraction
            ),
        )
    )


def _source_snapshot(source: Path, task: CrossSizeTask, method: str) -> dict:
    record = json.loads(
        (source / task.size / "trials" / f"{task.pose_id}.json").read_text(
            encoding="utf-8"
        )
    )
    source_method = "s4_fine_tuned" if method == "s4b_fine_tuned" else method
    row = record["methods"][source_method]
    return {
        key: row.get(key)
        for key in (
            "result",
            "inner_wall_confirmed",
            "offline_capture_validated",
            "bottom_reached",
            "terminal_verification_mode",
            "total_active_time_s",
            "fallback_budget_extension_s",
            "hard_stop",
            "hard_stop_reason",
            "terminal_rejection_count",
            "funnel_recovery_count",
            "max_raw_physical_force_n",
            "max_raw_physical_torque_mnm",
            "stage_max_raw_physical_force_n",
            "stage_max_raw_physical_torque_mnm",
        )
    }


def _run_pose(task_dict: dict, source_value: str) -> dict:
    if (
        _OUTPUT is None
        or _SIZE is None
        or not _METHODS
        or _CONTRACT_SHA256 is None
    ):
        raise RuntimeError("targeted commissioning worker is not initialized")
    task = CrossSizeTask(**task_dict)
    if task.size != _SIZE:
        raise RuntimeError("targeted commissioning task size drift")
    source = Path(source_value)
    methods = {}
    for method in _METHODS:
        methods[method] = {
            "source_development_v2": _source_snapshot(source, task, method),
            "commissioning_v2": _run_method(task, method),
        }
    record = {
        "phase": "development",
        "task": asdict(task),
        "methods": methods,
        "task_contract_sha256": _CONTRACT_SHA256,
        "untouched_tasks_or_seeds_read": False,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _summary(records: list[dict], methods: tuple[str, ...]) -> dict:
    result: dict = {}
    for method in methods:
        old = [
            record["methods"][method]["source_development_v2"]
            for record in records
        ]
        new = [record["methods"][method]["commissioning_v2"] for record in records]
        result[method] = {
            "trials": len(new),
            "source_development_v2_strict_bottom_successes": sum(
                _strict_bottom_success(row) for row in old
            ),
            "commissioning_v2_strict_bottom_successes": sum(
                _strict_bottom_success(row) for row in new
            ),
            "source_development_v2_hard_stops": sum(
                bool(row["hard_stop"]) for row in old
            ),
            "commissioning_v2_hard_stops": sum(
                bool(row["hard_stop"]) for row in new
            ),
            "source_development_v2_terminal_rejections": sum(
                int(row.get("terminal_rejection_count") or 0) for row in old
            ),
            "commissioning_v2_terminal_rejections": sum(
                int(row.get("terminal_rejection_count") or 0) for row in new
            ),
            "source_development_v2_peak_raw_force_n": max(
                float(row["max_raw_physical_force_n"]) for row in old
            ),
            "commissioning_v2_peak_raw_force_n": max(
                float(row["max_raw_physical_force_n"]) for row in new
            ),
            "source_development_v2_peak_raw_torque_mnm": max(
                float(row["max_raw_physical_torque_mnm"]) for row in old
            ),
            "commissioning_v2_peak_raw_torque_mnm": max(
                float(row["max_raw_physical_torque_mnm"]) for row in new
            ),
        }
    return result


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    size = str(args.size).lower()
    methods = tuple(args.methods)
    pose_ids = (
        tuple(args.pose_ids)
        if args.pose_ids
        else select_oracle_failure_pose_ids(source, size)
    )
    contract, tasks = build_contract(source, size, pose_ids, methods)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_existing = json.loads(
            json.dumps(
                {k: v for k, v in existing.items() if k != "created_utc"},
                sort_keys=True,
            )
        )
        stable_candidate = json.loads(
            json.dumps(
                {k: v for k, v in contract.items() if k != "created_utc"},
                sort_keys=True,
            )
        )
        if stable_existing != stable_candidate:
            raise ValueError("existing targeted contract differs; use a new output")
        contract = existing
    else:
        _atomic_json(contract_path, contract)
    contract_sha256 = _sha256(contract_path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "frozen_contract_only",
                    "output": str(output),
                    "pose_ids": list(pose_ids),
                },
                indent=2,
            )
        )
        return 0

    _validate_execution_files(contract["execution_files"])
    records: list[dict] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(tasks)),
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(output), size, methods, contract_sha256),
    ) as pool:
        future_by_task = {
            pool.submit(_run_pose, asdict(task), str(source)): task for task in tasks
        }
        with tqdm(total=len(tasks), desc="M10 commissioning v2", unit="pose") as bar:
            for future in as_completed(future_by_task):
                task = future_by_task[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    _atomic_json(
                        output / "trials" / f"{task.pose_id}.error.json",
                        {"task": asdict(task), "error": repr(exc)},
                    )
                    raise
                finally:
                    bar.update(1)
    records.sort(key=lambda row: row["task"]["pose_id"])
    report = {
        "experiment": contract["experiment"],
        "phase": "development",
        "pose_ids": list(pose_ids),
        "methods": _summary(records, methods),
        "commissioning_v2": contract["commissioning_v2"],
        "raw_hard_wrench_thresholds_relaxed": False,
        "untouched_tasks_or_seeds_read": False,
        "claim_boundary": contract["claim_boundary"],
        "task_contract_sha256": contract_sha256,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "phase": "development",
            "poses": len(records),
            "methods": list(methods),
            "raw_hard_wrench_thresholds_relaxed": False,
            "untouched_tasks_or_seeds_read": False,
            "task_contract_sha256": contract_sha256,
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Target commissioning v2 only at known failed M10 development poses; "
            "never reads or creates untouched tasks."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", choices=("m10",), default="m10")
    parser.add_argument("--pose-ids", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
