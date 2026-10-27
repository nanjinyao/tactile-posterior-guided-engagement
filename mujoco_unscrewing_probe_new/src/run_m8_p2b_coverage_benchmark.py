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
from src.render_m8_search_comparison import (
    COMMON_XY_SPEED_M_S,
    POST_PROBE_SETTLE_TIME_S,
    PROBE_RAMP_TIME_S,
    PROBE_RETURN_TIME_S,
    PROBE_SCAN_TIME_S,
    SPIRAL_PITCH_M,
    RunResult,
    archimedean_path,
    run_method,
)


ROOT = Path(__file__).resolve().parents[1]
P0D_MANIFEST = ROOT / "outputs" / "current" / "datasets" / "p0d_m8_sector_anchors" / "manifest.json"
DEFAULT_OUT = ROOT / "outputs" / "current" / "evaluations" / "p2b_m8_in_domain_coverage"
METHODS = ("learned", "baseline", "oracle")
VISUAL_SIGMA_M = 0.0020
VISUAL_BOUND_M = math.sqrt(5.991) * VISUAL_SIGMA_M
EFFECTIVE_CAPTURE_RADIUS_M = 0.00025
BASELINE_MAX_RADIUS_M = VISUAL_BOUND_M - EFFECTIVE_CAPTURE_RADIUS_M
POSTERIOR_COVERAGE = 0.99
GUARD_FACTOR = 2.25


def coverage_path_length_m() -> float:
    arc, _ = archimedean_path(
        np.zeros(2), SPIRAL_PITCH_M, BASELINE_MAX_RADIUS_M, 0.0
    )
    return float(arc[-1])


def derived_guard_budget_s() -> float:
    probe = (
        PROBE_RAMP_TIME_S
        + PROBE_SCAN_TIME_S
        + PROBE_RETURN_TIME_S
        + POST_PROBE_SETTLE_TIME_S
    )
    return probe + GUARD_FACTOR * coverage_path_length_m() / COMMON_XY_SPEED_M_S


ACTIVE_GUARD_BUDGET_S = derived_guard_budget_s()


@dataclass(frozen=True)
class CoverageTask:
    task_id: str
    pose_id: str
    search_state: str
    method: str
    radius_m: float
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


def build_tasks() -> list[CoverageTask]:
    payload = json.loads(P0D_MANIFEST.read_text(encoding="utf-8"))
    selected = {}
    for record in payload["records"]:
        radius = float(record["radius_m"])
        if radius > VISUAL_BOUND_M + 1.0e-12:
            continue
        if float(record["base_angle_deg"]) != 3.75:
            continue
        if str(record["nuisance_id"]) != "sector_sobol_0":
            continue
        key = (str(record["search_state"]), int(record["radial_level"]), int(record["rotation_index"]))
        selected[key] = record

    radii = sorted({round(float(record["radius_m"]), 12) for record in selected.values()})
    if len(radii) != 5 or len(selected) != 30:
        raise ValueError(f"expected five radii x six sectors, got {len(radii)} radii and {len(selected)} poses")

    tasks: list[CoverageTask] = []
    for pose_index, (key, record) in enumerate(sorted(selected.items())):
        state, radial_level, sector = key
        pose_id = f"{state}_r{radial_level:02d}_s{sector:02d}"
        for method in METHODS:
            tasks.append(
                CoverageTask(
                    task_id=f"{pose_id}_{method}",
                    pose_id=pose_id,
                    search_state=state,
                    method=method,
                    radius_m=float(record["radius_m"]),
                    x_m=float(record["x_m"]),
                    y_m=float(record["y_m"]),
                    sector_index=sector,
                    baseline_phase_rad=(pose_index % 6) * math.pi / 3.0,
                    posterior_seed=2026071000 + pose_index,
                )
            )
    return tasks


def _worker_init(output: str) -> None:
    global _RUNTIME, _OUTPUT
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble()
    _OUTPUT = Path(output)


def _run_task(task: CoverageTask) -> dict:
    assert _RUNTIME is not None and _OUTPUT is not None
    started = time.perf_counter()
    result: RunResult = run_method(
        task.method,
        initial_xy_m=np.array([task.x_m, task.y_m]),
        direction_runtime=_RUNTIME if task.method == "learned" else None,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        active_time_budget_s=ACTIVE_GUARD_BUDGET_S,
    )
    record = {
        "task": asdict(task),
        "result": asdict(result),
        "worker_elapsed_s": time.perf_counter() - started,
        "claim_boundary": (
            "M8 nominal-dynamics in-domain coverage rollout. The visual bound is a declared deployment "
            "assumption; learned inference uses F/T and tip-z, while capture still uses privileged MuJoCo tags."
        ),
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.task_id}.json", record)
    return record


def _termination_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        name = str(record["result"]["result"])
        if name == "common_time_budget_exhausted":
            name = "program_guard_exhausted"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _method_summary(records: list[dict], method: str) -> dict:
    subset = [record for record in records if record["task"]["method"] == method]
    successful = [record for record in subset if record["result"]["inner_wall_confirmed"]]
    all_time = np.asarray([record["result"]["total_active_time_s"] for record in subset])
    path = np.asarray([record["result"]["contact_xy_path_mm"] for record in subset])
    work = np.asarray([record["result"]["wear_proxy_mj"] for record in subset])
    force = np.asarray([record["result"]["max_force_n"] for record in subset])
    torque = np.asarray([record["result"]["max_torque_mnm"] for record in subset])
    success_time = np.asarray(
        [record["result"]["time_to_inner_wall_s"] for record in successful], dtype=float
    )
    result = {
        "trials": len(subset),
        "successes": len(successful),
        "success_rate": len(successful) / len(subset),
        "termination_counts": _termination_counts(subset),
        "success_time_median_s": None if len(success_time) == 0 else float(np.median(success_time)),
        "success_time_p90_s": None if len(success_time) == 0 else float(np.quantile(success_time, 0.90)),
        "all_trial_active_time_median_s": float(np.median(all_time)),
        "contact_path_median_mm": float(np.median(path)),
        "wear_proxy_median_mj": float(np.median(work)),
        "max_force_p90_n": float(np.quantile(force, 0.90)),
        "max_torque_p90_mnm": float(np.quantile(torque, 0.90)),
    }
    if method == "learned":
        errors = np.asarray([record["result"]["direction_error_deg"] for record in subset])
        result.update(
            {
                "direction_within_7p5_rate": float(np.mean(errors <= 7.5)),
                "direction_error_median_deg": float(np.median(errors)),
                "direction_error_p90_deg": float(np.quantile(errors, 0.90)),
                "transport_allowed_rate": float(
                    np.mean([record["result"]["transport_allowed"] for record in subset])
                ),
            }
        )
    return result


def _summary(records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    by_pose: dict[str, dict[str, dict]] = {}
    for record in records:
        by_pose.setdefault(record["task"]["pose_id"], {})[record["task"]["method"]] = record
    complete = {key: value for key, value in by_pose.items() if set(value) == set(METHODS)}
    both = [
        value
        for value in complete.values()
        if value["learned"]["result"]["inner_wall_confirmed"]
        and value["baseline"]["result"]["inner_wall_confirmed"]
    ]
    learned_times = np.asarray(
        [value["learned"]["result"]["time_to_inner_wall_s"] for value in both], dtype=float
    )
    baseline_times = np.asarray(
        [value["baseline"]["result"]["time_to_inner_wall_s"] for value in both], dtype=float
    )
    learned_paths = np.asarray(
        [value["learned"]["result"]["contact_xy_path_mm"] for value in both], dtype=float
    )
    baseline_paths = np.asarray(
        [value["baseline"]["result"]["contact_xy_path_mm"] for value in both], dtype=float
    )
    learned_work = np.asarray(
        [value["learned"]["result"]["wear_proxy_mj"] for value in both], dtype=float
    )
    baseline_work = np.asarray(
        [value["baseline"]["result"]["wear_proxy_mj"] for value in both], dtype=float
    )
    paired = {
        "complete_pose_triplets": len(complete),
        "both_learned_and_baseline_success": len(both),
        "learned_only_success": sum(
            value["learned"]["result"]["inner_wall_confirmed"]
            and not value["baseline"]["result"]["inner_wall_confirmed"]
            for value in complete.values()
        ),
        "baseline_only_success": sum(
            value["baseline"]["result"]["inner_wall_confirmed"]
            and not value["learned"]["result"]["inner_wall_confirmed"]
            for value in complete.values()
        ),
        "both_failed": sum(
            not value["baseline"]["result"]["inner_wall_confirmed"]
            and not value["learned"]["result"]["inner_wall_confirmed"]
            for value in complete.values()
        ),
        "learned_minus_baseline_time_median_s_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_times - baseline_times))
        ),
        "learned_time_median_s_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_times))
        ),
        "baseline_time_median_s_on_both_success": (
            None if len(both) == 0 else float(np.median(baseline_times))
        ),
        "learned_faster_rate_on_both_success": (
            None if len(both) == 0 else float(np.mean(learned_times < baseline_times))
        ),
        "learned_minus_baseline_path_median_mm_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_paths - baseline_paths))
        ),
        "learned_path_median_mm_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_paths))
        ),
        "baseline_path_median_mm_on_both_success": (
            None if len(both) == 0 else float(np.median(baseline_paths))
        ),
        "learned_shorter_path_rate_on_both_success": (
            None if len(both) == 0 else float(np.mean(learned_paths < baseline_paths))
        ),
        "learned_minus_baseline_work_median_mj_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_work - baseline_work))
        ),
        "learned_work_median_mj_on_both_success": (
            None if len(both) == 0 else float(np.median(learned_work))
        ),
        "baseline_work_median_mj_on_both_success": (
            None if len(both) == 0 else float(np.median(baseline_work))
        ),
        "learned_lower_work_rate_on_both_success": (
            None if len(both) == 0 else float(np.mean(learned_work < baseline_work))
        ),
    }
    by_radius = {}
    for radius in sorted({float(record["task"]["radius_m"]) for record in records}):
        key = f"{1000.0 * radius:.3f}_mm"
        by_radius[key] = {}
        for method in METHODS:
            subset = [
                record
                for record in records
                if record["task"]["method"] == method
                and math.isclose(float(record["task"]["radius_m"]), radius)
            ]
            by_radius[key][method] = {
                "trials": len(subset),
                "successes": sum(r["result"]["inner_wall_confirmed"] for r in subset),
                "termination_counts": _termination_counts(subset),
            }
    return {
        "stage": "P2b_M8_in_domain_coverage",
        "design": {
            "deployment_acceptance_radius_mm": 1000.0 * VISUAL_BOUND_M,
            "visual_prior_sigma_mm": 1000.0 * VISUAL_SIGMA_M,
            "radial_levels_mm": sorted(
                {round(1000.0 * float(record["task"]["radius_m"]), 6) for record in records}
            ),
            "physical_sectors_per_radius": 6,
            "initial_poses": len(complete),
            "methods": list(METHODS),
            "target_down_force_n": 3.0,
            "xy_speed_mm_s": 1000.0 * COMMON_XY_SPEED_M_S,
            "spiral_pitch_mm_per_rev": 1000.0 * SPIRAL_PITCH_M,
            "effective_capture_radius_mm": 1000.0 * EFFECTIVE_CAPTURE_RADIUS_M,
            "baseline_coverage_radius_mm": 1000.0 * BASELINE_MAX_RADIUS_M,
            "baseline_coverage_path_mm": 1000.0 * coverage_path_length_m(),
            "baseline_nominal_coverage_time_s": coverage_path_length_m() / COMMON_XY_SPEED_M_S,
            "program_guard_budget_s": ACTIVE_GUARD_BUDGET_S,
            "program_guard_factor": GUARD_FACTOR,
            "posterior_coverage": POSTERIOR_COVERAGE,
        },
        "methods": methods,
        "paired": paired,
        "by_radius": by_radius,
        "claim_boundary": (
            "Nominal-dynamics M8 simulation inside a declared 4.90 mm visual acceptance set. Search failure "
            "means coverage exhausted, a safety stop, or the non-performance program guard. "
            "Capture still uses privileged MuJoCo tags, and no nuisance or hardware result is included."
        ),
        "updated_utc": _utc_now(),
    }


def _write_report(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 P2b In-Domain Coverage Benchmark",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "| method | success | success time median / p90 | contact path median | work proxy median | force / torque p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        success_time = (
            "n/a"
            if item["success_time_median_s"] is None
            else f"{item['success_time_median_s']:.2f} / {item['success_time_p90_s']:.2f} s"
        )
        lines.append(
            f"| {method} | {item['successes']}/{item['trials']} ({item['success_rate']:.3f}) | "
            f"{success_time} | {item['contact_path_median_mm']:.2f} mm | "
            f"{item['wear_proxy_median_mj']:.2f} mJ | "
            f"{item['max_force_p90_n']:.2f} N / {item['max_torque_p90_mnm']:.2f} mNm |"
        )
    lines.extend(
        [
            "",
            "## Coverage Contract",
            "",
            f"- visual acceptance radius: {report['design']['deployment_acceptance_radius_mm']:.2f} mm",
            f"- baseline coverage radius/path/time: {report['design']['baseline_coverage_radius_mm']:.2f} mm / "
            f"{report['design']['baseline_coverage_path_mm']:.2f} mm / "
            f"{report['design']['baseline_nominal_coverage_time_s']:.2f} s",
            f"- derived program guard: {report['design']['program_guard_budget_s']:.2f} s; not a success threshold",
            "",
            "## Paired Result",
            "",
            *[f"- {key}: {value}" for key, value in report["paired"].items()],
            "",
            "## Radius Stratification",
            "",
            "| radius | learned | baseline | oracle |",
            "|---|---:|---:|---:|",
            *[
                "| "
                + radius
                + " | "
                + " | ".join(
                    f"{values[method]['successes']}/{values[method]['trials']}"
                    for method in METHODS
                )
                + " |"
                for radius, values in report["by_radius"].items()
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
    rerun_methods = set(args.rerun_method or [])
    rerun_results = set(args.rerun_result or [])
    for task in tasks:
        path = output / "trials" / f"{task.task_id}.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        force_rerun = task.method in rerun_methods or (
            saved is not None and saved["result"]["result"] in rerun_results
        )
        if args.resume and saved is not None and not force_rerun:
            existing.append(saved)
        else:
            pending.append(task)

    records = list(existing)
    progress_path = output / "progress.json"
    started = time.perf_counter()

    def checkpoint() -> None:
        completed_new = len(records) - len(existing)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        rate = completed_new / elapsed
        _atomic_json(
            progress_path,
            {
                "stage": "P2b_M8_in_domain_coverage",
                "total": len(tasks),
                "completed": len(records),
                "successes": sum(r["result"]["inner_wall_confirmed"] for r in records),
                "termination_counts": _termination_counts(records),
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
                desc="P2b in-domain coverage",
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
                        "result": {"inner_wall_confirmed": False, "result": "worker_exception"},
                        "error": repr(exc),
                        "created_utc": _utc_now(),
                    }
                    _atomic_json(output / "trials" / f"{task.task_id}.json", record)
                records.append(record)
                checkpoint()
                bar.update(1)
                bar.set_postfix(
                    ok=sum(r["result"]["inner_wall_confirmed"] for r in records),
                    miss=sum(not r["result"]["inner_wall_confirmed"] for r in records),
                    refresh=False,
                )
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
                "M8 in-domain nominal-dynamics coverage benchmark; not hardware or nuisance evidence."
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
    parser = argparse.ArgumentParser(
        description="Run the M8 P2b in-domain coverage benchmark with live progress."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=min(18, max(1, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-method", action="append", choices=METHODS)
    parser.add_argument("--rerun-result", action="append")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
