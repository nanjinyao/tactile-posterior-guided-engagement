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

import torch

from src.m8_direction_runtime import M8DirectionEnsemble
import src.run_cross_size_s4_full_loop_evaluation as s4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/diagnostics/"
    "cross_size_s4_residual_corridor_targeted_development"
)
DEFAULT_CASES = (
    "m10:006",
    "m10:010",
    "m12:003",
    "m12:004",
    "m12:011",
)


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


def _parse_case(value: str):
    try:
        size, index_text = value.lower().split(":", 1)
        index = int(index_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid case {value!r}; expected m10:006") from exc
    if size not in s4.SIZES or not 0 <= index < 12:
        raise ValueError(f"invalid development case {value!r}")
    return s4.build_phase_tasks("development", size, 12)[index]


def _run_task(task) -> dict:
    torch.set_num_threads(1)
    runtime = M8DirectionEnsemble(
        s4._model_root(task.size, "s4_fine_tuned")
    )
    s4._RUNTIMES = {"s4_fine_tuned": runtime}
    result = s4._run_one_method(
        task, "s4_fine_tuned", s4._size_contract(task.size)
    )
    payload = asdict(result)
    corridor_gates = [
        row
        for row in payload.get("hybrid_probe_gate_records") or []
        if row.get("stage") == "posterior_residual_corridor_coverage"
    ]
    return {
        "phase": "development",
        "confirmatory_use_forbidden": True,
        "task": asdict(task),
        "method": "s4_fine_tuned_plus_hybrid_residual_corridor",
        "result": payload,
        "residual_corridor_gates": corridor_gates,
        "created_utc": _utc_now(),
    }


def _contract(tasks: list) -> dict:
    source = ROOT / "src/render_m8_search_comparison.py"
    runner = ROOT / "src/run_cross_size_s4_full_loop_evaluation.py"
    commissioning = ROOT / "src/cross_size_commissioning.py"
    diagnostic = Path(__file__).resolve()
    sizes = sorted({task.size for task in tasks})
    return {
        "experiment": "cross_size_S4_residual_corridor_targeted_development",
        "phase": "development",
        "confirmatory_use_forbidden": True,
        "untouched_tasks_read_or_run": False,
        "task_source": str(s4.S3_DEVELOPMENT_SOURCE.relative_to(ROOT)),
        "tasks": [asdict(task) for task in tasks],
        "method": "S4_fine_tuned_plus_P_plus_hybrid_residual_corridor",
        "active_time_budget_s": s4.ACTIVE_TIME_BUDGET_S,
        "fallback_budget_extension_s": 0.0,
        "residual_coverage": asdict(s4.S4_HYBRID_RESIDUAL_COVERAGE),
        "size_contracts": {size: s4._size_contract(size) for size in sizes},
        "schedule_rule": (
            "shortcut+corridor+terminal and global-outward+terminal are "
            "alternative schedules, never cumulative prerequisites"
        ),
        "online_true_center_access": False,
        "raw_hard_wrench_thresholds_relaxed": False,
        "claim_boundary": (
            "conditional posterior-direction sector coverage; no guarantee of "
            "direction correctness, tracking, sensor capture, or terminal success"
        ),
        "source_sha256": {
            str(source.relative_to(ROOT)): _sha256(source),
            str(runner.relative_to(ROOT)): _sha256(runner),
            str(commissioning.relative_to(ROOT)): _sha256(commissioning),
            str(diagnostic.relative_to(ROOT)): _sha256(diagnostic),
        },
        "created_utc": _utc_now(),
    }


def _summary(records: list[dict]) -> dict:
    rows = []
    for record in records:
        task = record["task"]
        result = record["result"]
        gates = record["residual_corridor_gates"]
        rows.append(
            {
                "pose_id": task["pose_id"],
                "result": result["result"],
                "bottom_reached": result["bottom_reached"],
                "hard_stop": result["hard_stop"],
                "hard_stop_reason": result["hard_stop_reason"],
                "active_time_s": result["total_active_time_s"],
                "first_shortcut_planned_mm": result[
                    "first_shortcut_distance_mm"
                ],
                "first_shortcut_executed": result["first_shortcut_executed"],
                "control_direction_error_deg_offline": result[
                    "direction_error_deg"
                ],
                "corridor_executed": bool(gates and gates[-1]["executed"]),
                "corridor_outcome": (
                    gates[-1].get("outcome") if gates else "not_selected"
                ),
                "corridor_path_length_mm": (
                    gates[-1].get("corridor_path_length_mm") if gates else None
                ),
                "corridor_nominal_controller_restore": (
                    gates[-1].get("nominal_controller_restore")
                    if gates
                    else None
                ),
                "offline_min_nominal_center_distance_mm": (
                    gates[-1].get("offline_min_nominal_center_distance_mm")
                    if gates
                    else None
                ),
                "offline_min_measured_center_distance_mm": (
                    gates[-1].get("offline_min_measured_center_distance_mm")
                    if gates
                    else None
                ),
                "planar_tracking_error_p99_mm": (
                    gates[-1].get("planar_tracking_error_p99_mm")
                    if gates
                    else None
                ),
                "planar_tracking_error_max_mm": (
                    gates[-1].get("planar_tracking_error_max_mm")
                    if gates
                    else None
                ),
                "sensor_progress_candidates_during_corridor": (
                    gates[-1].get(
                        "sensor_progress_candidates_during_corridor"
                    )
                    if gates
                    else None
                ),
                "global_outward_fallback_used": result[
                    "global_outward_fallback_used"
                ],
                "fallback_skipped_reason": result["fallback_skipped_reason"],
                "budget_respected": result["total_active_time_s"]
                <= s4.ACTIVE_TIME_BUDGET_S + 1.0e-9,
            }
        )
    return {
        "phase": "development",
        "confirmatory_use_forbidden": True,
        "untouched_tasks_read_or_run": False,
        "trials": rows,
        "all_respect_130_s": all(row["budget_respected"] for row in rows),
        "hard_limits_relaxed": False,
        "updated_utc": _utc_now(),
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    tasks = [_parse_case(value) for value in args.cases]
    contract = _contract(tasks)
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_existing = {
            key: value for key, value in existing.items() if key != "created_utc"
        }
        stable_new = {
            key: value for key, value in contract.items() if key != "created_utc"
        }
        stable_existing = json.loads(
            json.dumps(stable_existing, sort_keys=True)
        )
        stable_new = json.loads(json.dumps(stable_new, sort_keys=True))
        if stable_existing != stable_new:
            raise ValueError("targeted development contract drifted; use a new output")
        contract = existing
    else:
        if args.execute:
            raise RuntimeError("freeze the development contract before --execute")
        _atomic_json(contract_path, contract)
    if not args.execute:
        print(json.dumps({"status": "frozen", "output": str(output)}, indent=2))
        return 0

    trial_dir = output / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    records = []
    pending = []
    for task in tasks:
        path = trial_dir / f"{task.pose_id}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)), mp_context=context
        ) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                record = future.result()
                _atomic_json(trial_dir / f"{task.pose_id}.json", record)
                records.append(record)
                print(
                    f"{task.pose_id}: {record['result']['result']} "
                    f"at {record['result']['total_active_time_s']:.3f} s",
                    flush=True,
                )
    by_id = {record["task"]["pose_id"]: record for record in records}
    ordered = [by_id[task.pose_id] for task in tasks]
    report = _summary(ordered)
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "phase": "development",
            "trial_count": len(ordered),
            "task_contract_sha256": _sha256(contract_path),
            "report_sha256": _sha256(output / "report.json"),
            "untouched_tasks_read_or_run": False,
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run named S4 development failures with residual coverage."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
