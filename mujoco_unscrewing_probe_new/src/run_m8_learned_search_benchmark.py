from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import RunResult, run_method
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "current" / "evaluations" / "p2_m8_learned_search_benchmark"
METHODS = ("learned", "baseline", "oracle")
VISUAL_SIGMA_M = 0.0020
BASELINE_MAX_RADIUS_M = 0.0080
ACTIVE_TIME_BUDGET_S = 30.0


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    pose_id: str
    search_state: str
    method: str
    x_m: float
    y_m: float
    sector_index: int
    baseline_phase_rad: float
    posterior_seed: int


_RUNTIME: M8DirectionEnsemble | None = None
_OUTPUT: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_tasks() -> list[BenchmarkTask]:
    trials = training.load_trials(training.P0D)
    split = training.grouped_sector_split(trials, 20260710)
    test_groups = set(split["test"])
    selected = {}
    for trial in trials:
        pose_id = training._sector_pose_id(trial.metadata)
        sector = int(trial.metadata["rotation_index"])
        nuisance_id = str(trial.metadata["nuisance_id"])
        if pose_id not in test_groups or sector not in {0, 2, 4} or not nuisance_id.endswith("0"):
            continue
        selected[(pose_id, sector)] = trial.metadata
    if len(selected) != 18:
        raise ValueError(f"expected 18 held-out poses, got {len(selected)}")
    tasks = []
    for pose_index, ((pose_id, sector), metadata) in enumerate(sorted(selected.items())):
        for method in METHODS:
            tasks.append(
                BenchmarkTask(
                    task_id=f"{pose_id}_s{sector:02d}_{method}",
                    pose_id=f"{pose_id}_s{sector:02d}",
                    search_state=str(metadata["search_state"]),
                    method=method,
                    x_m=float(metadata["x_m"]),
                    y_m=float(metadata["y_m"]),
                    sector_index=sector,
                    baseline_phase_rad=(pose_index % 3) * 2.0 * np.pi / 3.0,
                    posterior_seed=20260710 + pose_index,
                )
            )
    return tasks


def _worker_init(output: str) -> None:
    global _RUNTIME, _OUTPUT
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble()
    _OUTPUT = Path(output)


def _run_task(task: BenchmarkTask) -> dict:
    assert _RUNTIME is not None and _OUTPUT is not None
    started = time.perf_counter()
    result: RunResult = run_method(
        task.method,
        initial_xy_m=np.array([task.x_m, task.y_m]),
        direction_runtime=_RUNTIME if task.method == "learned" else None,
        visual_sigma_m=VISUAL_SIGMA_M,
        posterior_seed=task.posterior_seed,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
    )
    record = {
        "task": asdict(task),
        "result": asdict(result),
        "worker_elapsed_s": time.perf_counter() - started,
        "claim_boundary": (
            "Held-out P0d-pose nominal-dynamics MuJoCo search rollout. Learned inference uses F/T and tip-z "
            "phase data; true center is used only for offline error and oracle. Progress gates still use "
            "privileged contact tags."
        ),
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.task_id}.json", record)
    return record


def _penalized_time(record: dict) -> float:
    value = record["result"]["time_to_inner_wall_s"]
    return ACTIVE_TIME_BUDGET_S if value is None else float(value)


def _summary(records: list[dict]) -> dict:
    method_summary = {}
    for method in METHODS:
        subset = [record for record in records if record["task"]["method"] == method]
        success = np.asarray(
            [record["result"]["inner_wall_confirmed"] for record in subset], dtype=bool
        )
        penalized = np.asarray([_penalized_time(record) for record in subset])
        work = np.asarray([record["result"]["wear_proxy_mj"] for record in subset])
        path = np.asarray([record["result"]["contact_xy_path_mm"] for record in subset])
        force = np.asarray([record["result"]["max_force_n"] for record in subset])
        torque = np.asarray([record["result"]["max_torque_mnm"] for record in subset])
        method_summary[method] = {
            "trials": len(subset),
            "successes": int(np.sum(success)),
            "success_rate": float(np.mean(success)),
            "penalized_time_median_s": float(np.median(penalized)),
            "penalized_time_p90_s": float(np.quantile(penalized, 0.90)),
            "contact_path_median_mm": float(np.median(path)),
            "wear_proxy_median_mj": float(np.median(work)),
            "max_force_p90_n": float(np.quantile(force, 0.90)),
            "max_torque_p90_mnm": float(np.quantile(torque, 0.90)),
        }
        if method == "learned":
            errors = np.asarray([record["result"]["direction_error_deg"] for record in subset])
            method_summary[method].update(
                {
                    "direction_within_7p5_rate": float(np.mean(errors <= 7.5)),
                    "direction_error_median_deg": float(np.median(errors)),
                    "direction_error_p90_deg": float(np.quantile(errors, 0.90)),
                    "transport_allowed_rate": float(
                        np.mean([record["result"]["transport_allowed"] for record in subset])
                    ),
                }
            )

    by_pose = {}
    for record in records:
        by_pose.setdefault(record["task"]["pose_id"], {})[
            record["task"]["method"]
        ] = record
    complete = {key: value for key, value in by_pose.items() if set(value) == set(METHODS)}
    learned_penalized = np.asarray([_penalized_time(value["learned"]) for value in complete.values()])
    baseline_penalized = np.asarray([_penalized_time(value["baseline"]) for value in complete.values()])
    oracle_penalized = np.asarray([_penalized_time(value["oracle"]) for value in complete.values()])
    paired = {
        "complete_pose_triplets": len(complete),
        "learned_minus_baseline_penalized_time_median_s": float(
            np.median(learned_penalized - baseline_penalized)
        ),
        "learned_faster_than_baseline_rate": float(
            np.mean(learned_penalized < baseline_penalized)
        ),
        "learned_minus_oracle_penalized_time_median_s": float(
            np.median(learned_penalized - oracle_penalized)
        ),
    }
    visual_95_radius_m = math.sqrt(5.991) * VISUAL_SIGMA_M
    strata = {}
    for stratum, predicate in (
        (
            "inside_synthetic_visual_95",
            lambda record: math.hypot(
                float(record["task"]["x_m"]), float(record["task"]["y_m"])
            )
            <= visual_95_radius_m,
        ),
        (
            "outside_synthetic_visual_95",
            lambda record: math.hypot(
                float(record["task"]["x_m"]), float(record["task"]["y_m"])
            )
            > visual_95_radius_m,
        ),
    ):
        strata[stratum] = {
            "visual_95_radius_mm": 1000.0 * visual_95_radius_m,
            "methods": {},
        }
        for method in METHODS:
            subset = [
                record
                for record in records
                if record["task"]["method"] == method and predicate(record)
            ]
            penalized = np.asarray([_penalized_time(record) for record in subset])
            strata[stratum]["methods"][method] = {
                "trials": len(subset),
                "successes": sum(
                    record["result"]["inner_wall_confirmed"] for record in subset
                ),
                "success_rate": float(
                    np.mean(
                        [record["result"]["inner_wall_confirmed"] for record in subset]
                    )
                ),
                "penalized_time_median_s": float(np.median(penalized)),
            }
    return {
        "stage": "P2_M8_learned_search_pilot",
        "design": {
            "held_out_spatial_groups": 6,
            "physical_sectors_per_group": 3,
            "initial_poses": 18,
            "methods": list(METHODS),
            "target_down_force_n": 3.0,
            "xy_speed_mm_s": 2.25,
            "spiral_pitch_mm_per_rev": 0.40,
            "common_active_budget_s": ACTIVE_TIME_BUDGET_S,
            "visual_prior_sigma_mm": 1000.0 * VISUAL_SIGMA_M,
            "baseline_max_radius_mm": 1000.0 * BASELINE_MAX_RADIUS_M,
        },
        "methods": method_summary,
        "paired": paired,
        "strata": strata,
        "claim_boundary": (
            "Small held-out-pose nominal-dynamics M8 simulation pilot. Learned probe cost is included; oracle "
            "uses true center only as a lower bound. The capture/progress verifier still uses privileged MuJoCo "
            "contact tags, and the visual covariance is synthetic rather than measured."
        ),
        "updated_utc": _utc_now(),
    }


def _write_report(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 P2 Learned Search Pilot",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "| method | success | penalized time median / p90 | contact path median | wear proxy median | force / torque p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        lines.append(
            f"| {method} | {item['successes']}/{item['trials']} ({item['success_rate']:.3f}) | "
            f"{item['penalized_time_median_s']:.2f} / {item['penalized_time_p90_s']:.2f} s | "
            f"{item['contact_path_median_mm']:.2f} mm | {item['wear_proxy_median_mj']:.3f} mJ | "
            f"{item['max_force_p90_n']:.2f} N / {item['max_torque_p90_mnm']:.2f} mNm |"
        )
    learned = report["methods"]["learned"]
    lines.extend(
        [
            "",
            f"Learned direction within 7.5 deg: {learned['direction_within_7p5_rate']:.3f}; "
            f"median / p90 error: {learned['direction_error_median_deg']:.2f} / "
            f"{learned['direction_error_p90_deg']:.2f} deg.",
            "",
            "## Paired Result",
            "",
            f"- learned - baseline penalized time median: {report['paired']['learned_minus_baseline_penalized_time_median_s']:+.2f} s",
            f"- learned faster than baseline: {report['paired']['learned_faster_than_baseline_rate']:.3f}",
            f"- learned - oracle penalized time median: {report['paired']['learned_minus_oracle_penalized_time_median_s']:+.2f} s",
            "",
            "## Visual-Prior Strata",
            "",
            f"Synthetic visual 95% radius: {report['strata']['inside_synthetic_visual_95']['visual_95_radius_mm']:.2f} mm.",
            "",
            "| stratum | learned | baseline | oracle |",
            "|---|---:|---:|---:|",
            *[
                "| "
                + stratum
                + " | "
                + " | ".join(
                    f"{values['methods'][method]['successes']}/{values['methods'][method]['trials']}"
                    for method in METHODS
                )
                + " |"
                for stratum, values in report["strata"].items()
            ],
            "",
            "## Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    (output / "trials").mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    existing, pending = [], []
    for task in tasks:
        path = output / "trials" / f"{task.task_id}.json"
        if args.resume and path.exists():
            existing.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)

    records = list(existing)
    progress_path = output / "progress.json"
    started = time.perf_counter()

    def checkpoint() -> None:
        completed_new = len(records) - len(existing)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        rate = completed_new / elapsed
        success = sum(record["result"]["inner_wall_confirmed"] for record in records)
        _atomic_json(
            progress_path,
            {
                "stage": "P2_M8_learned_search_pilot",
                "total": len(tasks),
                "completed": len(records),
                "successes": success,
                "failed_or_timeout": len(records) - success,
                "rate_trials_s": rate,
                "eta_s": (len(tasks) - len(records)) / rate if rate > 0.0 else None,
                "updated_utc": _utc_now(),
            },
        )

    checkpoint()
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_worker_init,
            initargs=(str(output),),
        ) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            bar = tqdm(
                total=len(tasks),
                initial=len(existing),
                desc="P2 learned search",
                unit="trial",
                dynamic_ncols=True,
                disable=args.no_progress,
            )
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "task": asdict(task),
                        "result": {
                            "inner_wall_confirmed": False,
                            "result": "worker_exception",
                        },
                        "error": repr(exc),
                        "created_utc": _utc_now(),
                    }
                    _atomic_json(output / "trials" / f"{task.task_id}.json", record)
                records.append(record)
                checkpoint()
                success = sum(item["result"]["inner_wall_confirmed"] for item in records)
                bar.update(1)
                bar.set_postfix(ok=success, miss=len(records) - success, refresh=False)
            bar.close()

    errors = [record for record in records if "error" in record]
    if not errors and len(records) == len(tasks):
        report = _summary(records)
        _write_report(output, report)
    checkpoint()
    _atomic_json(
        output / "manifest.json",
        {
            "tasks": [asdict(task) for task in tasks],
            "completed": len(records),
            "worker_errors": len(errors),
            "claim_boundary": (
                "Held-out P0d-pose nominal-dynamics simulation benchmark; not hardware or sensor-only capture evidence."
            ),
            "updated_utc": _utc_now(),
        },
    )
    if errors:
        print(json.dumps({"worker_errors": errors}, indent=2))
        return 1
    print(json.dumps(_summary(records), indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M8 learned-search pilot with live progress.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=min(18, max(1, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
