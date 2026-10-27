from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest
import torch
from tqdm import tqdm

from src.funnel_guard import FunnelGuardConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.m8_jin_style_direction import JinStyleDirectionRuntime
from src.render_m8_search_comparison import RunResult, run_method
from src.run_m8_fair_policy_benchmark import (
    BASELINE_MAX_RADIUS_M,
    MODEL_ROOT as IMPROVED_MODEL_ROOT,
    POSTERIOR_COVERAGE,
    VISUAL_BOUND_M,
    VISUAL_SIGMA_M,
    PoseTask,
)
from src.run_m8_strict_complete_loop_audit import (
    ACTIVE_TIME_BUDGET_S,
    DEFAULT_SOURCE,
    _failure_category,
    _strict_bottom_success,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
)
JIN_MODEL_ROOT = ROOT / "outputs/current/models/m8_jin_2021_style_direction_proxy"
METHODS = (
    "proposed",
    "proposed_plus",
    "outward",
    "improved_direct_retry3",
    "jin_direct_retry3",
)
METHOD_LABELS = {
    "proposed": "Proposed",
    "proposed_plus": "Proposed+",
    "outward": "Pure Outward",
    "improved_direct_retry3": "Improved fingerprint + direct x3",
    "jin_direct_retry3": "Jin-style 3-channel + direct x3",
}
DIRECT_FAILURE_POLICY = (
    "miss -> low-force unload/lift/perpendicular 0.25 mm shift/full re-probe; "
    "maximum 3 rays in the same rollout; no simulator reset"
)
SHARED_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)


_IMPROVED: M8DirectionEnsemble | None = None
_JIN: JinStyleDirectionRuntime | None = None
_OUTPUT: Path | None = None


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


def _load_tasks(source: Path, poses: int) -> list[PoseTask]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tasks = [PoseTask(**item) for item in manifest["tasks"]]
    if poses > len(tasks):
        raise ValueError(f"requested {poses} poses, source contains {len(tasks)}")
    selected = tasks[:poses]
    if any(task.sensor_noise_seed is None for task in selected):
        raise ValueError("five-method strict audit requires frozen sensor seeds")
    return selected


def _worker_init(output: str) -> None:
    global _IMPROVED, _JIN, _OUTPUT
    torch.set_num_threads(1)
    _IMPROVED = M8DirectionEnsemble(IMPROVED_MODEL_ROOT)
    _JIN = JinStyleDirectionRuntime(JIN_MODEL_ROOT)
    _OUTPUT = Path(output)


def _common_kwargs(task: PoseTask) -> dict:
    return {
        "initial_xy_m": np.asarray([task.x_m, task.y_m], dtype=float),
        "visual_sigma_m": VISUAL_SIGMA_M,
        "visual_bound_m": VISUAL_BOUND_M,
        "posterior_coverage": POSTERIOR_COVERAGE,
        "posterior_seed": task.posterior_seed,
        "max_transport_m": None,
        "baseline_phase_rad": task.baseline_phase_rad,
        "baseline_max_radius_m": BASELINE_MAX_RADIUS_M,
        "target_down_force_n": task.target_down_force_n,
        "friction_slide": task.friction_slide,
        "contact_timeconst_s": task.contact_timeconst_s,
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "continue_to_bottom": True,
        "wrench_observation": WrenchObservationConfig(
            seed=int(task.sensor_noise_seed),
            delay_s=task.sensor_delay_s,
        ),
        "observation_label": "shared strict measured-wrench contract",
        "cad_conditioned_insertion": True,
        "conditioned_on_ambiguous_top_contact": True,
        "funnel_guard": SHARED_FUNNEL,
    }


def _run_method(task: PoseTask, method: str) -> RunResult:
    assert _IMPROVED is not None and _JIN is not None
    kwargs = _common_kwargs(task)
    if method == "proposed":
        return run_method("learned", direction_runtime=_IMPROVED, **kwargs)
    if method == "proposed_plus":
        return run_method("hybrid", direction_runtime=_IMPROVED, **kwargs)
    if method == "outward":
        return run_method("baseline", **kwargs)
    if method == "improved_direct_retry3":
        return run_method(
            "direct",
            direction_runtime=_IMPROVED,
            direct_max_attempts=3,
            direct_adjustment_m=0.00025,
            direct_failure_policy_overlay=DIRECT_FAILURE_POLICY,
            fingerprint_display_mode="improved_four_channel",
            **kwargs,
        )
    if method == "jin_direct_retry3":
        return run_method(
            "direct",
            direction_runtime=_JIN,
            direct_max_attempts=3,
            direct_adjustment_m=0.00025,
            direct_failure_policy_overlay=DIRECT_FAILURE_POLICY,
            fingerprint_display_mode="jin_three_channel",
            **kwargs,
        )
    raise ValueError(method)


def _run_pose(task: PoseTask) -> dict:
    assert _OUTPUT is not None
    started = time.perf_counter()
    methods = {method: asdict(_run_method(task, method)) for method in METHODS}
    first_hashes = {
        methods[method]["fingerprint_sha256"]
        for method in (
            "proposed",
            "proposed_plus",
            "improved_direct_retry3",
            "jin_direct_retry3",
        )
    }
    record = {
        "task": asdict(task),
        "methods": methods,
        "fairness": {
            "same_initial_pose_contact_nuisance_and_sensor_seed": True,
            "same_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "zero_budget_extension": all(
                float(value.get("fallback_budget_extension_s", 0.0)) == 0.0
                for value in methods.values()
            ),
            "same_post_capture_funnel": asdict(SHARED_FUNNEL),
            "same_terminal_verifier": True,
            "first_probe_fingerprint_sha_match": len(first_hashes) == 1,
            "direct_attempts_include_physical_adjustment_time": True,
            "simulator_reset_count": sum(
                int(value.get("simulator_reset_count", 0))
                for value in methods.values()
            ),
            "online_true_center_or_contact_tags": False,
        },
        "worker_elapsed_s": time.perf_counter() - started,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _method_summary(records: list[dict], method: str) -> dict:
    rows = [record["methods"][method] for record in records]
    successes = [row for row in rows if _strict_bottom_success(row)]
    direct = method.endswith("retry3")

    def median(key: str, source: list[dict]) -> float | None:
        values = [float(row[key]) for row in source if row.get(key) is not None]
        return None if not values else float(np.median(values))

    direct_first_errors = []
    if direct:
        for row in rows:
            attempts = row.get("direct_attempt_records") or []
            if attempts and attempts[0].get("offline_control_error_deg") is not None:
                direct_first_errors.append(
                    float(attempts[0]["offline_control_error_deg"])
                )

    return {
        "label": METHOD_LABELS[method],
        "trials": len(rows),
        "strict_bottom_successes": len(successes),
        "strict_bottom_rate": len(successes) / max(1, len(rows)),
        "offline_capture_successes": sum(
            bool(row.get("offline_capture_validated")) for row in rows
        ),
        "hard_stops": sum(bool(row.get("hard_stop")) for row in rows),
        "funnel_safe_aborts": sum(bool(row.get("funnel_abort_reason")) for row in rows),
        "median_success_bottom_time_s": median("time_to_bottom_s", successes),
        "median_success_xy_path_mm": median("actual_xy_path_mm", successes),
        "median_success_wear_mj": median("wear_proxy_mj", successes),
        "median_all_max_force_n": median("max_force_n", rows),
        "median_all_max_torque_mnm": median("max_torque_mnm", rows),
        "post_capture_recoveries": sum(
            int(row.get("funnel_recovery_count", 0)) for row in rows
        ),
        "terminal_rejections": sum(
            int(row.get("terminal_rejection_count", 0)) for row in rows
        ),
        "direct_attempts_median": (
            median("direct_attempts_completed", rows) if direct else None
        ),
        "direct_successful_attempt_counts": (
            dict(
                Counter(
                    str(row.get("direct_successful_attempt") or "none")
                    for row in rows
                )
            )
            if direct
            else None
        ),
        "direct_first_ray_captures": (
            sum(int(row.get("direct_successful_attempt") or 0) == 1 for row in rows)
            if direct
            else None
        ),
        "direct_any_ray_captures": (
            sum(row.get("direct_successful_attempt") is not None for row in rows)
            if direct
            else None
        ),
        "first_probe_executed_direction_error_deg": (
            {
                "count": len(direct_first_errors),
                "median": float(np.median(direct_first_errors)),
                "p90": float(np.quantile(direct_first_errors, 0.90)),
                "within_15_deg": float(
                    np.mean(np.asarray(direct_first_errors) <= 15.0)
                ),
                "catastrophic_gt_90_deg": int(
                    np.sum(np.asarray(direct_first_errors) > 90.0)
                ),
                "source": (
                    "first direct_attempt_record.offline_control_error_deg; "
                    "the direction actually consumed by control"
                ),
            }
            if direct_first_errors
            else None
        ),
        "failure_categories": dict(
            Counter(_failure_category(row) for row in rows)
        ),
        "budget_violations": sum(
            float(row["total_active_time_s"]) > ACTIVE_TIME_BUDGET_S + 1.0e-9
            or float(row.get("fallback_budget_extension_s", 0.0)) != 0.0
            for row in rows
        ),
        "simulator_resets": sum(
            int(row.get("simulator_reset_count", 0)) for row in rows
        ),
    }


def _paired_test(records: list[dict], left: str, right: str) -> dict:
    pairs = [
        (
            _strict_bottom_success(record["methods"][left]),
            _strict_bottom_success(record["methods"][right]),
        )
        for record in records
    ]
    left_only = sum(pair == (True, False) for pair in pairs)
    right_only = sum(pair == (False, True) for pair in pairs)
    discordant = left_only + right_only
    return {
        "both_success": sum(pair == (True, True) for pair in pairs),
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_failed": sum(pair == (False, False) for pair in pairs),
        "exact_mcnemar_p": (
            1.0
            if discordant == 0
            else float(binomtest(left_only, discordant, 0.5).pvalue)
        ),
    }


def _paired_direct_metric(records: list[dict], metric: str) -> dict:
    def value(record: dict, method: str) -> bool:
        row = record["methods"][method]
        attempts = row.get("direct_attempt_records") or []
        if metric == "first_ray_capture":
            return int(row.get("direct_successful_attempt") or 0) == 1
        if metric == "any_ray_capture":
            return row.get("direct_successful_attempt") is not None
        if metric == "first_probe_within_15_deg":
            return bool(
                attempts
                and float(attempts[0]["offline_control_error_deg"]) <= 15.0
            )
        raise ValueError(metric)

    pairs = [
        (
            value(record, "improved_direct_retry3"),
            value(record, "jin_direct_retry3"),
        )
        for record in records
    ]
    improved_only = sum(pair == (True, False) for pair in pairs)
    jin_only = sum(pair == (False, True) for pair in pairs)
    discordant = improved_only + jin_only
    return {
        "metric": metric,
        "both_true": sum(pair == (True, True) for pair in pairs),
        "improved_only_true": improved_only,
        "jin_only_true": jin_only,
        "both_false": sum(pair == (False, False) for pair in pairs),
        "exact_mcnemar_p": (
            1.0
            if discordant == 0
            else float(binomtest(improved_only, discordant, 0.5).pvalue)
        ),
    }


def _write_summary(output: Path, report: dict) -> None:
    def metric(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# M8 Five-Method Closed-Loop Audit: Direct x3 + Shared R3 Funnel",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "Every method shares the same 130 s, zero-extension, sensor-only capture,",
        "tip-RCC, maximum-three post-capture recovery, and CAD-bottom verifier.",
        "The two direct methods additionally receive at most three same-rollout",
        "rays; every miss pays for unload/lift/shift/full re-probe.",
        "",
        "| method | CAD-bottom | capture | hard stop | median bottom time | median XY path | median wear | direct attempts | R3 recoveries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        lines.append(
            f"| {item['label']} | {item['strict_bottom_successes']}/{item['trials']} | "
            f"{item['offline_capture_successes']}/{item['trials']} | {item['hard_stops']} | "
            f"{metric(item['median_success_bottom_time_s'])} s | "
            f"{metric(item['median_success_xy_path_mm'])} mm | "
            f"{metric(item['median_success_wear_mj'])} mJ | "
            f"{metric(item['direct_attempts_median'], 1)} | "
            f"{item['post_capture_recoveries']} |"
        )
    lines.extend(
        [
            "",
            "## Direct x3 estimator/interface comparison",
            "",
            "| estimator | first-ray capture | any-ray capture | CAD-bottom | first-probe control error median / p90 | within 15 deg |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {report['methods'][method]['label']} | "
                    f"{report['methods'][method]['direct_first_ray_captures']}/{report['methods'][method]['trials']} | "
                    f"{report['methods'][method]['direct_any_ray_captures']}/{report['methods'][method]['trials']} | "
                    f"{report['methods'][method]['strict_bottom_successes']}/{report['methods'][method]['trials']} | "
                    f"{report['methods'][method]['first_probe_executed_direction_error_deg']['median']:.2f} / "
                    f"{report['methods'][method]['first_probe_executed_direction_error_deg']['p90']:.2f} deg | "
                    f"{report['methods'][method]['first_probe_executed_direction_error_deg']['within_15_deg']:.3f} |"
                )
                for method in ("improved_direct_retry3", "jin_direct_retry3")
            ],
            "",
            "## Paired improved-versus-Jin direct x3",
            "",
            f"- strict bottom: `{json.dumps(report['paired']['improved_vs_jin_direct_retry3'], sort_keys=True)}`",
            f"- first-ray capture: `{json.dumps(report['paired']['improved_vs_jin_first_ray_capture'], sort_keys=True)}`",
            f"- any-ray capture: `{json.dumps(report['paired']['improved_vs_jin_any_ray_capture'], sort_keys=True)}`",
            f"- first-probe within 15 deg: `{json.dumps(report['paired']['improved_vs_jin_first_probe_within_15_deg'], sort_keys=True)}`",
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


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    trials = output / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(source, args.poses)
    contract = {
        "experiment": "M8_five_method_retry3_shared_R3_closed_loop",
        "source_manifest": str((source / "manifest.json").relative_to(ROOT)),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "methods": list(METHODS),
        "tasks": [asdict(task) for task in tasks],
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "zero_extension": True,
        "direct_contract": {
            "maximum_rays": 3,
            "adjustment_m": 0.00025,
            "same_rollout": True,
            "simulator_reset": False,
            "failure_policy": DIRECT_FAILURE_POLICY,
        },
        "post_capture_contract": asdict(SHARED_FUNNEL),
        "model_contracts": {
            "improved": {
                "path": str(
                    (IMPROVED_MODEL_ROOT / "ensemble_interface_contract.json").relative_to(ROOT)
                ),
                "sha256": _sha256(
                    IMPROVED_MODEL_ROOT / "ensemble_interface_contract.json"
                ),
            },
            "jin_proxy": {
                "path": str((JIN_MODEL_ROOT / "runtime_contract.json").relative_to(ROOT)),
                "sha256": _sha256(JIN_MODEL_ROOT / "runtime_contract.json"),
            },
        },
        "created_utc": _utc_now(),
    }
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_existing = {k: v for k, v in existing.items() if k != "created_utc"}
        stable_new = {k: v for k, v in contract.items() if k != "created_utc"}
        if stable_existing != stable_new:
            raise ValueError("existing five-method task contract differs")
        contract = existing
    else:
        _atomic_json(contract_path, contract)

    records: list[dict] = []
    pending: list[PoseTask] = []
    for task in tasks:
        path = trials / f"{task.pose_id}.json"
        if args.resume and path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)
    started = time.perf_counter()
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
                desc="M8 five-method retry3/R3",
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
                        _atomic_json(trials / f"{task.pose_id}.json", record)
                    records.append(record)
                    _atomic_json(
                        output / "progress.json",
                        {
                            "completed": len(records),
                            "total": len(tasks),
                            "elapsed_s": time.perf_counter() - started,
                            "updated_utc": _utc_now(),
                        },
                    )
                    bar.update(1)

    errors = [record for record in records if "error" in record]
    if errors:
        _atomic_json(output / "errors.json", {"errors": errors})
        raise RuntimeError(f"{len(errors)} five-method rollout tasks failed")
    records.sort(key=lambda value: value["task"]["pose_id"])
    if not all(record["fairness"]["first_probe_fingerprint_sha_match"] for record in records):
        raise RuntimeError("first-probe fingerprints drifted across paired methods")
    if not all(record["fairness"]["simulator_reset_count"] == 0 for record in records):
        raise RuntimeError("a direct retry reported a simulator reset")
    report = {
        "experiment": contract["experiment"],
        "task_contract_sha256": _sha256(contract_path),
        "methods": {
            method: _method_summary(records, method) for method in METHODS
        },
        "paired": {
            "improved_vs_jin_direct_retry3": _paired_test(
                records,
                "improved_direct_retry3",
                "jin_direct_retry3",
            ),
            "proposed_vs_proposed_plus": _paired_test(
                records,
                "proposed",
                "proposed_plus",
            ),
            "improved_vs_jin_first_ray_capture": _paired_direct_metric(
                records, "first_ray_capture"
            ),
            "improved_vs_jin_any_ray_capture": _paired_direct_metric(
                records, "any_ray_capture"
            ),
            "improved_vs_jin_first_probe_within_15_deg": _paired_direct_metric(
                records, "first_probe_within_15_deg"
            ),
        },
        "all_first_probe_hashes_match": True,
        "all_budget_extensions_zero": all(
            record["fairness"]["zero_budget_extension"] for record in records
        ),
        "all_simulator_reset_counts_zero": True,
        "claim_boundary": (
            "Fresh paired M8 MuJoCo evidence under one 130 s complete-loop contract. "
            "The Jin arm is an M8 paper-style engineering proxy, not a faithful "
            "reproduction. Three rays are a bounded same-rollout recovery policy, "
            "not three independent trials. The shared R3 funnel can improve some "
            "recoverable contacts but does not guarantee CAD-bottom completion. "
            "No hardware, passivity, motor-coupling, or unscrewing claim follows."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output, report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "completed": len(records),
            "methods": list(METHODS),
            "task_contract_sha256": _sha256(contract_path),
            "report_sha256": _sha256(output / "report.json"),
            "summary_sha256": _sha256(output / "summary.md"),
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M8 five-method retry3/shared-R3 closed-loop audit."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poses", type=int, default=24)
    parser.add_argument(
        "--workers", type=int, default=min(24, max(1, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
