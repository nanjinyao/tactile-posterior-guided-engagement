from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from src.funnel_guard import FunnelGuardConfig
from src.render_m8_search_comparison import ROOT, RunResult, run_method


OUT = ROOT / "outputs/current/evaluations/m8_funnel_guard_ablation"


@dataclass(frozen=True)
class FunnelTask:
    task_id: str
    initial_xy_m: tuple[float, float]
    insertion_bias_xy_m: tuple[float, float]
    friction_slide: float
    contact_timeconst_s: float
    target_down_force_n: float
    baseline_phase_rad: float


def build_tasks() -> list[FunnelTask]:
    tasks: list[FunnelTask] = []
    levels = (0.0, 0.5, 1.0)
    for index in range(12):
        angle = math.radians(60.0 * (index % 6))
        bias_angle = angle + math.radians(30.0 if index % 2 == 0 else -30.0)
        bias_radius = 0.00010 if index < 6 else 0.00025
        tasks.append(
            FunnelTask(
                task_id=f"funnel_{index:02d}",
                initial_xy_m=(0.0010 * math.cos(angle), 0.0010 * math.sin(angle)),
                insertion_bias_xy_m=(
                    bias_radius * math.cos(bias_angle),
                    bias_radius * math.sin(bias_angle),
                ),
                friction_slide=0.55 + 0.40 * levels[index % 3],
                contact_timeconst_s=0.040 + 0.030 * levels[(index // 2) % 3],
                target_down_force_n=2.5 + levels[(index // 3) % 3],
                baseline_phase_rad=math.radians(30.0 * index),
            )
        )
    return tasks


def _run(task: FunnelTask, enabled: bool) -> tuple[str, str, dict]:
    result = run_method(
        "baseline",
        initial_xy_m=np.asarray(task.initial_xy_m, dtype=float),
        baseline_max_radius_m=0.0020,
        baseline_phase_rad=task.baseline_phase_rad,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=30.0,
        continue_to_bottom=True,
        funnel_guard=FunnelGuardConfig(enabled=enabled),
        insertion_command_bias_xy_m=np.asarray(task.insertion_bias_xy_m, dtype=float),
    )
    return task.task_id, "funnel_guard" if enabled else "legacy_insertion", asdict(result)


def _summary(results: list[dict]) -> dict:
    bottom = [item for item in results if item["result"] == "cad_bottom_verified"]
    depth_overrun_mm = [max(0.0, -float(item["min_tip_z_mm"]) - 4.0) for item in results]
    return {
        "trials": len(results),
        "bottom_verified": len(bottom),
        "hard_stops": sum(bool(item["hard_stop"]) for item in results),
        "safe_aborts": sum(item["result"] == "funnel_safe_abort" for item in results),
        "bottom_not_verified": sum(item["result"] == "bottom_not_verified" for item in results),
        "recoveries": sum(int(item["funnel_recovery_count"]) for item in results),
        "cad_depth_safe_aborts": sum(
            item["funnel_abort_reason"] == "cad_depth_overshoot" for item in results
        ),
        "median_success_time_s": (
            None
            if not bottom
            else float(np.median([item["time_to_bottom_s"] for item in bottom]))
        ),
        "p90_max_force_n": float(np.quantile([item["max_force_n"] for item in results], 0.90)),
        "p90_max_torque_mnm": float(np.quantile([item["max_torque_mnm"] for item in results], 0.90)),
        "median_cad_depth_overrun_mm": float(np.median(depth_overrun_mm)),
        "max_cad_depth_overrun_mm": float(np.max(depth_overrun_mm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trials = OUT / "trials"
    trials.mkdir(exist_ok=True)
    tasks = build_tasks()
    futures = []
    records: dict[str, dict] = {
        task.task_id: {"task": asdict(task), "methods": {}} for task in tasks
    }
    total = 2 * len(tasks)
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for task in tasks:
            for enabled in (False, True):
                futures.append(executor.submit(_run, task, enabled))
        for future in as_completed(futures):
            task_id, method, result = future.result()
            records[task_id]["methods"][method] = result
            completed += 1
            width = 28
            filled = round(width * completed / total)
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r[{bar}] {completed:02d}/{total:02d}", end="", flush=True)
    print()

    for task_id, record in records.items():
        (trials / f"{task_id}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    legacy = [record["methods"]["legacy_insertion"] for record in records.values()]
    guarded = [record["methods"]["funnel_guard"] for record in records.values()]
    paired = {
        "both_bottom": sum(
            a["bottom_reached"] and b["bottom_reached"]
            for a, b in zip(legacy, guarded)
        ),
        "guard_only_bottom": sum(
            not a["bottom_reached"] and b["bottom_reached"]
            for a, b in zip(legacy, guarded)
        ),
        "legacy_only_bottom": sum(
            a["bottom_reached"] and not b["bottom_reached"]
            for a, b in zip(legacy, guarded)
        ),
        "neither_bottom": sum(
            not a["bottom_reached"] and not b["bottom_reached"]
            for a, b in zip(legacy, guarded)
        ),
    }
    report = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "paired_tasks": len(tasks),
            "initial_radius_mm": 1.0,
            "post_capture_command_bias_mm": [0.10, 0.25],
            "friction_range": [0.55, 0.95],
            "contact_timeconst_range_s": [0.040, 0.070],
            "down_force_range_n": [2.5, 3.5],
            "search_policy": "identical ordinary outward search; ablation starts only after capture",
        },
        "methods": {
            "legacy_insertion": _summary(legacy),
            "funnel_guard": _summary(guarded),
        },
        "paired": paired,
        "claim_boundary": (
            "Post-capture MuJoCo controller ablation with bounded synthetic TCP command bias and paired contact nuisance. "
            "It tests non-interference, safe recovery, and CAD-depth defense of an iiwa-style tip-frame impedance proxy. "
            "It is not hardware evidence, a search comparison, or a guarantee of M10/M12 parameter transfer."
        ),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
