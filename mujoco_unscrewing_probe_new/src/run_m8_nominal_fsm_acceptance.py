from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

import src.run_m8_p2b_coverage_benchmark as p2b


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "current" / "evaluations" / "p2b_nominal_fsm_acceptance"
M8_HEAD_RADIUS_M = 0.0065
M8_BIT_RADIUS_M = 0.006 / np.sqrt(3.0)
NOMINAL_FULL_SUPPORT_RADIUS_M = M8_HEAD_RADIUS_M - M8_BIT_RADIUS_M
METHODS = ("learned", "baseline")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_tasks() -> list[p2b.CoverageTask]:
    tasks = [
        task
        for task in p2b.build_tasks()
        if task.method in METHODS
        and task.radius_m <= NOMINAL_FULL_SUPPORT_RADIUS_M + 1.0e-12
    ]
    poses = {task.pose_id for task in tasks}
    if len(tasks) != 36 or len(poses) != 18:
        raise ValueError(f"expected 18 nominal paired poses / 36 tasks, got {len(poses)} / {len(tasks)}")
    return tasks


def _summary(records: list[dict]) -> dict:
    by_method = {}
    for method in METHODS:
        subset = [r for r in records if r["task"]["method"] == method]
        successful = [r for r in subset if r["result"]["inner_wall_confirmed"]]
        by_method[method] = {
            "trials": len(subset),
            "successes": len(successful),
            "hard_stops": sum(r["result"]["hard_stop"] for r in subset),
            "termination_counts": p2b._termination_counts(subset),
            "success_time_median_s": (
                None
                if not successful
                else float(np.median([r["result"]["time_to_inner_wall_s"] for r in successful]))
            ),
            "contact_path_median_mm": float(
                np.median([r["result"]["contact_xy_path_mm"] for r in subset])
            ),
            "work_proxy_median_mj": float(
                np.median([r["result"]["wear_proxy_mj"] for r in subset])
            ),
            "max_force_p90_n": float(
                np.quantile([r["result"]["max_force_n"] for r in subset], 0.90)
            ),
            "max_torque_p90_mnm": float(
                np.quantile([r["result"]["max_torque_mnm"] for r in subset], 0.90)
            ),
        }

    by_pose = {}
    for record in records:
        by_pose.setdefault(record["task"]["pose_id"], {})[record["task"]["method"]] = record
    complete = [value for value in by_pose.values() if set(value) == set(METHODS)]
    both = [
        value
        for value in complete
        if value["learned"]["result"]["inner_wall_confirmed"]
        and value["baseline"]["result"]["inner_wall_confirmed"]
    ]

    def paired_values(key: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([value["learned"]["result"][key] for value in both], dtype=float),
            np.asarray([value["baseline"]["result"][key] for value in both], dtype=float),
        )

    paired = {
        "complete_pairs": len(complete),
        "both_success": len(both),
        "learned_only_success": sum(
            value["learned"]["result"]["inner_wall_confirmed"]
            and not value["baseline"]["result"]["inner_wall_confirmed"]
            for value in complete
        ),
        "baseline_only_success": sum(
            value["baseline"]["result"]["inner_wall_confirmed"]
            and not value["learned"]["result"]["inner_wall_confirmed"]
            for value in complete
        ),
    }
    for label, key in (
        ("time_s", "time_to_inner_wall_s"),
        ("path_mm", "contact_xy_path_mm"),
        ("work_mj", "wear_proxy_mj"),
        ("max_force_n", "max_force_n"),
        ("max_torque_mnm", "max_torque_mnm"),
    ):
        learned, baseline = paired_values(key)
        paired[f"learned_{label}_median"] = float(np.median(learned))
        paired[f"baseline_{label}_median"] = float(np.median(baseline))
        paired[f"learned_minus_baseline_{label}_median"] = float(
            np.median(learned - baseline)
        )
        paired[f"learned_lower_{label}_rate"] = float(np.mean(learned < baseline))

    return {
        "stage": "P2b_nominal_FSM_acceptance",
        "design": {
            "cad_full_support_radius_mm": 1000.0 * NOMINAL_FULL_SUPPORT_RADIUS_M,
            "radial_levels_mm": sorted(
                {round(1000.0 * float(r["task"]["radius_m"]), 6) for r in records}
            ),
            "physical_sectors_per_radius": 6,
            "initial_poses": len(complete),
            "methods": list(METHODS),
            "fsm_change": "sensor-depth commit latches below-rim progress and forbids planar-search resume",
        },
        "methods": by_method,
        "paired": paired,
        "claim_boundary": (
            "Development-set nominal M8 FSM regression after a trace-driven safety fix. "
            "It is not an independent test; capture still ends on privileged MuJoCo-tag gates."
        ),
        "updated_utc": _utc_now(),
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    (output / "trials").mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    records = []
    started = time.perf_counter()

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        rate = len(records) / elapsed
        _atomic_json(
            output / "progress.json",
            {
                "stage": "P2b_nominal_FSM_acceptance",
                "total": len(tasks),
                "completed": len(records),
                "successes": sum(r["result"]["inner_wall_confirmed"] for r in records),
                "hard_stops": sum(r["result"]["hard_stop"] for r in records),
                "rate_trials_s": rate,
                "eta_s": (len(tasks) - len(records)) / rate if rate > 0.0 else None,
                "updated_utc": _utc_now(),
            },
        )

    checkpoint()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        mp_context=context,
        initializer=p2b._worker_init,
        initargs=(str(output),),
    ) as pool:
        futures = {pool.submit(p2b._run_task, task): task for task in tasks}
        bar = tqdm(total=len(tasks), desc="nominal FSM acceptance", unit="trial", dynamic_ncols=True)
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "task": asdict(task),
                    "result": {"inner_wall_confirmed": False, "hard_stop": False, "result": "worker_exception"},
                    "error": repr(exc),
                    "created_utc": _utc_now(),
                }
                _atomic_json(output / "trials" / f"{task.task_id}.json", record)
            records.append(record)
            checkpoint()
            bar.update(1)
            bar.set_postfix(
                ok=sum(r["result"]["inner_wall_confirmed"] for r in records),
                hard=sum(r["result"]["hard_stop"] for r in records),
                refresh=False,
            )
        bar.close()

    errors = [record for record in records if "error" in record]
    report = _summary(records) if not errors else None
    if report is not None:
        _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "manifest.json",
        {
            "tasks": [asdict(task) for task in tasks],
            "completed": len(records),
            "worker_errors": len(errors),
            "claim_boundary": "Development-set nominal M8 FSM safety regression.",
            "updated_utc": _utc_now(),
        },
    )
    if errors:
        print(json.dumps({"worker_errors": errors}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=min(18, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
