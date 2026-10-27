from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
    DIRECT_FIRST_FAILURE_POLICY_OVERLAY,
    RunResult,
    run_method,
)
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "outputs/current/evaluations/m8_fresh_24_pabc_direction_gate"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/evaluations/m8_strict_24_complete_loop"
)
ACTIVE_TIME_BUDGET_S = 130.0
VALID_TERMINAL_MODES = {
    "bottom_force_dwell",
    "cad_depth_low_wrench_dwell",
}

# Baseline B is represented by its first direct-ray attempt only. Its historical
# retry implementation starts a fresh simulator and excludes the physical reset,
# so it is not eligible for the strictly paired complete-loop table.
METHODS = (
    "proposed",
    "proposed_plus",
    "outward",
    "entropy_grid",
    "direct_first",
    "random",
    "oracle",
)
RUNTIME_METHOD = {
    "proposed": "learned",
    "proposed_plus": "hybrid",
    "outward": "baseline",
    "entropy_grid": "entropy_grid",
    "direct_first": "direct",
    "random": "random",
    "oracle": "oracle",
}
METHOD_ROLES = {
    "proposed": "paper-facing complete method",
    "proposed_plus": "adaptive research branch; no superiority claim",
    "outward": "geometry-only comparator",
    "entropy_grid": "strong transparent active-perception diagnostic",
    "direct_first": "mechanism diagnostic; no physical retry/reset claim",
    "random": "weak-search lower diagnostic",
    "oracle": "privileged upper-bound diagnostic",
}


_RUNTIME: M8DirectionEnsemble | None = None
_OUTPUT: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _worker_init(output: str) -> None:
    global _RUNTIME, _OUTPUT
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble(MODEL_ROOT)
    _OUTPUT = Path(output)


def _run_with_runtime(
    task: PoseTask,
    method: str,
    runtime: M8DirectionEnsemble,
    *,
    output_path: Path | None = None,
    trace_output_path: Path | None = None,
) -> RunResult:
    """Run one strict method with the frozen audit policy contract.

    ``output_path`` and ``trace_output_path`` only enable observation of the
    same rollout.  Keeping every policy/environment argument in this shared
    function prevents selected-case renderers from silently drifting away
    from the frozen strict audit.
    """

    runtime_method = RUNTIME_METHOD[method]
    requires_direction = runtime_method in {"learned", "hybrid", "direct"}
    return run_method(
        runtime_method,
        output_path=output_path,
        trace_output_path=trace_output_path,
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=runtime if requires_direction else None,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        max_transport_m=None,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=int(task.sensor_noise_seed), delay_s=task.sensor_delay_s
        ),
        observation_label="shared pre-hardware wrench engineering prior",
        random_search_seed=task.posterior_seed + 700000,
        cad_conditioned_insertion=True,
        conditioned_on_ambiguous_top_contact=True,
        direct_failure_policy_overlay=DIRECT_FIRST_FAILURE_POLICY_OVERLAY,
    )


def _run_one(task: PoseTask, method: str) -> RunResult:
    assert _RUNTIME is not None
    return _run_with_runtime(task, method, _RUNTIME)


def _run_pose(task: PoseTask) -> dict:
    assert _OUTPUT is not None
    started = time.perf_counter()
    methods = {method: asdict(_run_one(task, method)) for method in METHODS}
    fingerprint_hashes = {
        methods[method]["fingerprint_sha256"]
        for method in ("proposed", "proposed_plus", "direct_first")
    }
    record = {
        "task": asdict(task),
        "methods": methods,
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_seed": True,
            "same_strict_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "same_capture_handoff_and_terminal_verifier": True,
            "conditioned_on_ambiguous_top_contact": True,
            "proposed_proposed_plus_direct_first_fingerprint_match": (
                len(fingerprint_hashes) == 1
            ),
            "online_true_center_or_contact_tags": False,
            "oracle_uses_true_center_upper_bound_only": True,
            "direct_retry_excluded_because_physical_reset_is_not_simulated": True,
        },
        "created_utc": _utc_now(),
        "worker_elapsed_s": time.perf_counter() - started,
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _strict_bottom_success(row: dict) -> bool:
    return bool(
        row.get("inner_wall_confirmed")
        and row.get("offline_capture_validated")
        and row.get("bottom_reached")
        and row.get("result") == "cad_bottom_verified"
        and row.get("terminal_verification_mode") in VALID_TERMINAL_MODES
        and not row["hard_stop"]
        and float(row["total_active_time_s"]) <= ACTIVE_TIME_BUDGET_S + 1.0e-9
        and abs(float(row.get("fallback_budget_extension_s") or 0.0)) <= 1.0e-12
    )


def _capture_success(row: dict) -> bool:
    return bool(row.get("offline_capture_validated"))


def _failure_category(row: dict) -> str:
    if _strict_bottom_success(row):
        return "cad_bottom_verified"
    if float(row["total_active_time_s"]) > ACTIVE_TIME_BUDGET_S + 1.0e-9:
        return "budget_contract_violation"
    if abs(float(row.get("fallback_budget_extension_s") or 0.0)) > 1.0e-12:
        return "budget_extension_contract_violation"
    if row["hard_stop"]:
        return "hard_stop"
    if row["bottom_reached"]:
        return "strict_terminal_contract_violation"
    if row.get("funnel_abort_reason"):
        return f"funnel_safe_abort:{row['funnel_abort_reason']}"
    if row.get("inner_wall_confirmed") and not row.get("offline_capture_validated"):
        return "sensor_capture_false_positive"
    if row.get("offline_capture_validated"):
        return "capture_without_bottom"
    if row.get("inner_wall_confirmed"):
        return "sensor_handoff_without_offline_capture"
    if row.get("result") == "common_time_budget_exhausted":
        return "time_budget_exhausted"
    return str(row.get("result") or "no_capture")


def _median(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.median(values))


def _method_summary(records: list[dict], method: str) -> dict:
    rows = [record["methods"][method] for record in records]
    successful = [row for row in rows if _strict_bottom_success(row)]
    categories: dict[str, int] = {}
    for row in rows:
        category = _failure_category(row)
        categories[category] = categories.get(category, 0) + 1
    return {
        "role": METHOD_ROLES[method],
        "trials": len(rows),
        "strict_bottom_successes": len(successful),
        "strict_bottom_success_rate": len(successful) / max(1, len(rows)),
        "offline_capture_successes": sum(_capture_success(row) for row in rows),
        "sensor_handoff_count": sum(bool(row["inner_wall_confirmed"]) for row in rows),
        "sensor_handoff_false_positive_count": sum(
            bool(row["inner_wall_confirmed"])
            and not bool(row.get("offline_capture_validated"))
            for row in rows
        ),
        "hard_stops": sum(bool(row["hard_stop"]) for row in rows),
        "funnel_safe_aborts": sum(bool(row.get("funnel_abort_reason")) for row in rows),
        "budget_contract_violations": sum(
            float(row["total_active_time_s"]) > ACTIVE_TIME_BUDGET_S + 1.0e-9
            for row in rows
        ),
        "fallback_budget_extension_nonzero": sum(
            abs(float(row.get("fallback_budget_extension_s") or 0.0)) > 1.0e-12
            for row in rows
        ),
        "terminal_mode_counts": dict(
            sorted(Counter(str(row.get("terminal_verification_mode")) for row in rows).items())
        ),
        "preprobe_route_counts": dict(
            sorted(Counter(str(row.get("preprobe_route")) for row in rows).items())
        ),
        "fallback_skipped_reason_counts": dict(
            sorted(
                Counter(
                    str(row.get("fallback_skipped_reason")) for row in rows
                ).items()
            )
        ),
        "hard_stop_reason_counts": dict(
            sorted(
                Counter(
                    str(row.get("hard_stop_reason"))
                    for row in rows
                    if row.get("hard_stop")
                ).items()
            )
        ),
        "failure_categories": categories,
        "time_to_bottom_success_median_s": _median(successful, "time_to_bottom_s"),
        "total_active_time_all_median_s": _median(rows, "total_active_time_s"),
        "actual_path_success_median_mm": _median(successful, "actual_xy_path_mm"),
        "contact_path_success_median_mm": _median(successful, "contact_xy_path_mm"),
        "wear_success_median_mj": _median(successful, "wear_proxy_mj"),
        "max_force_p90_n": float(
            np.quantile([float(row["max_force_n"]) for row in rows], 0.90)
        ),
        "max_torque_p90_mnm": float(
            np.quantile([float(row["max_torque_mnm"]) for row in rows], 0.90)
        ),
        "max_raw_physical_force_all_steps_p90_n": float(
            np.quantile(
                [float(row.get("max_raw_physical_force_n") or 0.0) for row in rows],
                0.90,
            )
        ),
        "max_raw_physical_torque_all_steps_p90_mnm": float(
            np.quantile(
                [
                    float(row.get("max_raw_physical_torque_mnm") or 0.0)
                    for row in rows
                ],
                0.90,
            )
        ),
        "global_outward_fallback_count": sum(
            bool(row.get("global_outward_fallback_used")) for row in rows
        ),
        "hybrid_budget_reserve_switch_count": sum(
            bool(row.get("hybrid_budget_switch_reason")) for row in rows
        ),
        "hybrid_guarded_motion_failure_reason_counts": dict(
            sorted(
                Counter(
                    str(row.get("hybrid_guarded_motion_failure_reason"))
                    for row in rows
                    if row.get("hybrid_guarded_motion_failure_reason")
                ).items()
            )
        ),
        "post_terminal_event_display_new_hard_stop_count": sum(
            bool(row.get("post_terminal_event_display_new_hard_stop"))
            for row in rows
        ),
        "offline_capture_scoring_snapshot_counts": dict(
            sorted(
                Counter(
                    str(row.get("offline_capture_scoring_snapshot"))
                    for row in rows
                ).items()
            )
        ),
    }


def _paired(records: list[dict], left: str, right: str) -> dict:
    outcomes = [
        (
            _strict_bottom_success(record["methods"][left]),
            _strict_bottom_success(record["methods"][right]),
        )
        for record in records
    ]
    left_only = sum(pair == (True, False) for pair in outcomes)
    right_only = sum(pair == (False, True) for pair in outcomes)
    discordant = left_only + right_only
    exact_p = 1.0
    if discordant:
        tail = sum(
            math.comb(discordant, k)
            for k in range(min(left_only, right_only) + 1)
        )
        exact_p = min(1.0, 2.0 * tail / (2**discordant))
    common = [
        record
        for record, outcome in zip(records, outcomes)
        if outcome == (True, True)
    ]
    return {
        "both_success": sum(pair == (True, True) for pair in outcomes),
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_failed": sum(pair == (False, False) for pair in outcomes),
        "mcnemar_exact_two_sided_p": exact_p,
        "common_success_left_minus_right_time_median_s": (
            None
            if not common
            else float(
                np.median(
                    [
                        float(record["methods"][left]["time_to_bottom_s"])
                        - float(record["methods"][right]["time_to_bottom_s"])
                        for record in common
                    ]
                )
            )
        ),
        "common_success_left_minus_right_path_median_mm": (
            None
            if not common
            else float(
                np.median(
                    [
                        float(record["methods"][left]["actual_xy_path_mm"])
                        - float(record["methods"][right]["actual_xy_path_mm"])
                        for record in common
                    ]
                )
            )
        ),
    }


def _summary(records: list[dict], source: Path) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    return {
        "experiment": "M8_strict_24_pose_complete_closed_loop",
        "design": {
            "poses": len(records),
            "source_manifest": str((source / "manifest.json").relative_to(ROOT)),
            "source_seed": 20260723,
            "conditioned_on_ambiguous_top_contact": True,
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "budget_is_hard_accumulated_simulated_active_controller_time": True,
            "continue_to_cad_bottom": True,
            "online_true_center_or_contact_tags": False,
            "offline_capture_scoring_uses_mujoco_geometry": True,
            "direct_retry_excluded_from_strict_table": True,
            "raw_physical_peak_scope": "all physics steps including inactive settle/hold",
            "measured_peak_scope": "active-controller steps only",
            "strict_outcome_frozen_before_inactive_display_tail": True,
            "proposed_plus_schedule_contract": {
                "type": "conditional nominal reserve, not a success guarantee",
                "tracking_error_bound_mm": 0.25,
                "candidate_handling_contingency_s": 6.0,
                "continuous_contact_path_required": True,
                "local_spiral_fallback_enabled": False,
                "post_terminal_event_inactive_display_hold_charged_to_budget": False,
            },
            "direct_retry_reason": (
                "historical retry restarts the simulator and excludes reset motion; "
                "direct_first is retained only as a mechanism diagnostic"
            ),
        },
        "methods": methods,
        "paired": {
            f"proposed_vs_{right}": _paired(records, "proposed", right)
            for right in METHODS
            if right != "proposed"
        },
        "fingerprint_match_rate": float(
            np.mean(
                [
                    record["fairness"][
                        "proposed_proposed_plus_direct_first_fingerprint_match"
                    ]
                    for record in records
                ]
            )
        ),
        "claim_boundary": (
            "Frozen simulation-only M8 complete-loop audit conditioned on the "
            "declared ambiguous top-contact annulus. Strict success requires "
            "sensor-only handoff, offline-validated capture, CAD-bottom terminal "
            "verification, no hard stop, and completion within the shared 130 s "
            "active budget. It does not establish hardware transfer, passivity, "
            "motor coupling, threaded unscrewing, extraction, or universal "
            "superiority. Oracle is an upper-bound diagnostic; direct_first is "
            "not a physically reset retry baseline."
        ),
        "updated_utc": _utc_now(),
    }


def _write_summary(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 Strict 24-Pose Complete-Loop Audit",
        "",
        "Strict success = sensor handoff + offline capture validation + a declared "
        "CAD-bottom terminal dwell, no hard stop/extension, and active time <= 130 s.",
        "The task set is conditioned on ambiguous top contact; it is not the upstream visual/contact router.",
        "",
        "| method | role | strict CAD-bottom | offline capture | false handoff | hard stop | safe abort | budget violation | bottom time | path | wear |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        bottom_time = item["time_to_bottom_success_median_s"]
        path = item["actual_path_success_median_mm"]
        wear = item["wear_success_median_mj"]
        lines.append(
            f"| {method} | {item['role']} | {item['strict_bottom_successes']}/{item['trials']} | "
            f"{item['offline_capture_successes']}/{item['trials']} | "
            f"{item['sensor_handoff_false_positive_count']} | {item['hard_stops']} | "
            f"{item['funnel_safe_aborts']} | {item['budget_contract_violations']} | "
            f"{'n/a' if bottom_time is None else f'{bottom_time:.2f} s'} | "
            f"{'n/a' if path is None else f'{path:.2f} mm'} | "
            f"{'n/a' if wear is None else f'{wear:.2f} mJ'} |"
        )
    lines.extend(
        [
            "",
            "## Failure categories",
            "",
            *[
                f"- `{method}`: `{json.dumps(report['methods'][method]['failure_categories'], sort_keys=True)}`"
                for method in METHODS
            ],
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _load_tasks(source: Path, poses: int) -> list[PoseTask]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tasks = [PoseTask(**item) for item in manifest["tasks"]]
    if poses > len(tasks):
        raise ValueError(f"requested {poses} poses, source contains {len(tasks)}")
    selected = tasks[:poses]
    if any(task.sensor_noise_seed is None for task in selected):
        raise ValueError("strict audit requires a frozen sensor-noise seed per pose")
    return selected


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    (output / "trials").mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(source, args.poses)
    records: list[dict] = []
    pending: list[PoseTask] = []
    for task in tasks:
        path = output / "trials" / f"{task.pose_id}.json"
        if args.resume and path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)

    started = time.perf_counter()

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        completed_new = max(0, len(records) - (len(tasks) - len(pending)))
        rate = completed_new / elapsed
        _atomic_json(
            output / "progress.json",
            {
                "total_poses": len(tasks),
                "completed_poses": len(records),
                "rate_poses_s": rate,
                "eta_s": None if rate <= 0.0 else (len(tasks) - len(records)) / rate,
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
            futures = {pool.submit(_run_pose, task): task for task in pending}
            with tqdm(
                total=len(tasks),
                initial=len(records),
                desc="M8 strict complete loop",
                unit="pose",
                dynamic_ncols=True,
            ) as bar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "task": asdict(task),
                            "error": repr(exc),
                            "created_utc": _utc_now(),
                        }
                        _atomic_json(
                            output / "trials" / f"{task.pose_id}.json", record
                        )
                    records.append(record)
                    checkpoint()
                    bar.update(1)
                    valid = [item for item in records if "error" not in item]
                    bar.set_postfix(
                        P=sum(
                            _strict_bottom_success(item["methods"]["proposed"])
                            for item in valid
                        ),
                        Pp=sum(
                            _strict_bottom_success(item["methods"]["proposed_plus"])
                            for item in valid
                        ),
                        refresh=False,
                    )

    errors = [record for record in records if "error" in record]
    if errors:
        _atomic_json(output / "errors.json", {"errors": errors})
        print(json.dumps(errors, indent=2))
        return 1
    records.sort(key=lambda record: record["task"]["pose_id"])
    report = _summary(records, source)
    _write_summary(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "tasks": [asdict(task) for task in tasks],
            "completed": len(records),
            "methods": list(METHODS),
            "source": str(source.relative_to(ROOT)),
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict-budget M8 CAD-bottom complete-loop audit."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poses", type=int, default=24)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
