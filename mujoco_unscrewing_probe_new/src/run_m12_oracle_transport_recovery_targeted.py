from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_commissioning import commissioning_for_size
from src.funnel_guard import FunnelGuardConfig
from src.render_m8_search_comparison import run_method
from src.run_cross_size_commissioning_v2_targeted import (
    _atomic_json,
    _file_record,
    _load_selected_tasks,
    _sha256,
    _utc_now,
    _validate_execution_files,
)
from src.run_cross_size_closed_loop_audit import CrossSizeTask
from src.run_cross_size_s4_full_loop_evaluation import _size_contract
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4b_full_loop_development_v3"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/diagnostics/"
    "cross_size_m12_oracle_transport_recovery_targeted"
)
POSE_IDS = (
    "m12_s3test_005",
    "m12_s3test_007",
    "m12_s3test_011",
)
ACTIVE_TIME_BUDGET_S = 130.0
BASE_R3_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)


_OUTPUT: Path | None = None
_CONTRACT_SHA256: str | None = None


def _source_trial(source: Path, pose_id: str) -> Path:
    return source / "m12" / "trials" / f"{pose_id}.json"


def _source_oracle_snapshot(source: Path, pose_id: str) -> dict:
    record = json.loads(
        _source_trial(source, pose_id).read_text(encoding="utf-8")
    )
    oracle = record["methods"]["privileged_straight_to_center"]
    return {
        key: oracle.get(key)
        for key in (
            "result",
            "offline_capture_validated",
            "bottom_reached",
            "terminal_verification_mode",
            "total_active_time_s",
            "hard_stop",
            "hard_stop_reason",
            "max_raw_physical_force_n",
            "max_raw_physical_torque_mnm",
            "stage_time_s",
            "stage_max_raw_physical_force_n",
            "stage_max_raw_physical_torque_mnm",
            "sensor_progress_candidate_count",
            "capture_handoff_used",
            "candidate_attempts",
        )
    }


def build_contract(source: Path) -> tuple[dict, list[CrossSizeTask]]:
    source = source.resolve()
    report_path = source / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("phase") != "development":
        raise ValueError("M12 targeted diagnosis may read development only")
    if "untouched" in source.name.lower():
        raise ValueError("untouched source is forbidden")

    tasks, paths = _load_selected_tasks(source, "m12", POSE_IDS)
    for task in tasks:
        trial = json.loads(paths[task.pose_id].read_text(encoding="utf-8"))
        if trial.get("phase") != "development":
            raise ValueError(f"{task.pose_id}: non-development trial is forbidden")
        oracle = trial["methods"]["privileged_straight_to_center"]
        if _strict_bottom_success(oracle):
            raise ValueError(f"{task.pose_id}: source oracle was not a failure")

    commissioned = commissioning_for_size(
        "m12",
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    if not commissioned.oracle_transport_soft_escape_recovery:
        raise RuntimeError("M12 commissioning did not enable oracle recovery")
    if commissioned.admittance.hard_wrench != AdmittanceConfig().hard_wrench:
        raise RuntimeError("raw hard-wrench limits changed")
    size_contract = _size_contract("m12")
    contract = {
        "experiment": "m12_oracle_transport_soft_escape_targeted_development",
        "phase": "development",
        "size": "m12",
        "pose_ids": list(POSE_IDS),
        "tasks": [asdict(task) for task in tasks],
        "source": str(source),
        "source_failure_classes": {
            "m12_s3test_005": "raw_Tx_hard_stop_during_off_center_insertion_handoff",
            "m12_s3test_007": "transport_soft_escape_then_search_radius_exhausted",
            "m12_s3test_011": "transport_soft_escape_then_search_radius_exhausted",
        },
        "intervention": (
            "oracle-only unchanged-soft-guard recovery: measured-tip freeze, "
            "low-force unload, admittance rebase, resume remaining true-center "
            "ray, then require measured center-tolerance dwell before the shared "
            "guarded descent and terminal verifier"
        ),
        "commissioning": asdict(commissioned),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "fallback_budget_extension_s": 0.0,
        "raw_hard_wrench_thresholds": list(AdmittanceConfig().hard_wrench),
        "raw_hard_wrench_thresholds_relaxed": False,
        "soft_wrench_thresholds_changed": False,
        "capture_thresholds_changed": False,
        "terminal_verifier_thresholds_changed": False,
        "non_oracle_behavior_changed": False,
        "untouched_tasks_or_seeds_read": False,
        "claim_boundary": (
            "Three preselected failed M12 development poses only. This is a "
            "mechanism regression, not untouched or confirmatory evidence."
        ),
        "execution_files": {
            "runner": _file_record(Path(__file__)),
            "commissioning": _file_record(
                ROOT / "src/cross_size_commissioning.py"
            ),
            "rollout": _file_record(
                ROOT / "src/render_m8_search_comparison.py"
            ),
            "admittance_rollout": _file_record(
                ROOT / "src/mujoco_admittance_rollout.py"
            ),
            "source_report": _file_record(report_path),
            "source_trials": {
                pose_id: _file_record(paths[pose_id]) for pose_id in POSE_IDS
            },
            "cad_spec": _file_record(Path(size_contract["spec_path"])),
            "mujoco_model": _file_record(Path(size_contract["model_path"])),
        },
        "created_utc": _utc_now(),
    }
    return contract, tasks


def _worker_init(output: str, contract_sha256: str) -> None:
    global _OUTPUT, _CONTRACT_SHA256
    torch.set_num_threads(1)
    _OUTPUT = Path(output)
    _CONTRACT_SHA256 = contract_sha256


def _run_pose(task_dict: dict, source_value: str) -> dict:
    if _OUTPUT is None or _CONTRACT_SHA256 is None:
        raise RuntimeError("M12 targeted worker is not initialized")
    task = CrossSizeTask(**task_dict)
    if task.size != "m12" or task.pose_id not in POSE_IDS:
        raise RuntimeError("M12 targeted task drift")
    contract = _size_contract("m12")
    commissioned = commissioning_for_size(
        "m12",
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    result = asdict(
        run_method(
            "oracle",
            initial_xy_m=np.asarray([task.x_m, task.y_m], dtype=float),
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
                seed=task.sensor_noise_seed,
                delay_s=task.sensor_delay_s,
            ),
            admittance_config=commissioned.admittance,
            observation_label=(
                f"M12 oracle soft-escape targeted development {task.pose_id}"
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
    source = Path(source_value)
    record = {
        "phase": "development",
        "task": asdict(task),
        "source_development_v3": _source_oracle_snapshot(
            source, task.pose_id
        ),
        "commissioning_oracle_recovery": result,
        "task_contract_sha256": _CONTRACT_SHA256,
        "untouched_tasks_or_seeds_read": False,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _summarize(records: list[dict]) -> dict:
    old = [row["source_development_v3"] for row in records]
    new = [row["commissioning_oracle_recovery"] for row in records]
    return {
        "trials": len(new),
        "source_strict_bottom_successes": sum(
            _strict_bottom_success(row) for row in old
        ),
        "recovery_strict_bottom_successes": sum(
            _strict_bottom_success(row) for row in new
        ),
        "source_hard_stops": sum(bool(row["hard_stop"]) for row in old),
        "recovery_hard_stops": sum(bool(row["hard_stop"]) for row in new),
        "recovery_counts": {
            row["task"]["pose_id"]: row[
                "commissioning_oracle_recovery"
            ]["oracle_transport_soft_escape_recovery_count"]
            for row in records
        },
        "final_center_residual_mm": {
            row["task"]["pose_id"]: row[
                "commissioning_oracle_recovery"
            ]["oracle_transport_final_center_residual_mm"]
            for row in records
        },
        "outcomes": {
            row["task"]["pose_id"]: {
                "result": row["commissioning_oracle_recovery"]["result"],
                "recovery_outcome": row[
                    "commissioning_oracle_recovery"
                ]["oracle_transport_recovery_outcome"],
                "bottom_reached": row[
                    "commissioning_oracle_recovery"
                ]["bottom_reached"],
                "hard_stop": row[
                    "commissioning_oracle_recovery"
                ]["hard_stop"],
                "hard_stop_reason": row[
                    "commissioning_oracle_recovery"
                ]["hard_stop_reason"],
            }
            for row in records
        },
    }


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    contract, tasks = build_contract(source)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        # Canonical JSON also normalizes tuple-valued dataclass fields to lists.
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
            raise ValueError("existing M12 targeted contract differs; use a new output")
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
                    "pose_ids": list(POSE_IDS),
                    "task_contract_sha256": contract_sha256,
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
        initargs=(str(output), contract_sha256),
    ) as pool:
        future_by_task = {
            pool.submit(_run_pose, asdict(task), str(source)): task
            for task in tasks
        }
        with tqdm(total=len(tasks), desc="M12 oracle recovery", unit="pose") as bar:
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
        "summary": _summarize(records),
        "raw_hard_wrench_thresholds_relaxed": False,
        "soft_wrench_thresholds_changed": False,
        "terminal_verifier_thresholds_changed": False,
        "non_oracle_behavior_changed": False,
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
            "raw_hard_wrench_thresholds_relaxed": False,
            "soft_wrench_thresholds_changed": False,
            "terminal_verifier_thresholds_changed": False,
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
            "Freeze and run only the three preselected failed M12 development "
            "oracle poses; untouched inputs are forbidden."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
