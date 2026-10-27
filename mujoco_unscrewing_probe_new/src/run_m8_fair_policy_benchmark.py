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
from scipy.stats import qmc
import torch
from tqdm import tqdm

from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import (
    COMMON_XY_SPEED_M_S,
    SPIRAL_PITCH_M,
    RunResult,
    run_method,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/current/evaluations/m8_four_method_entropy_audit"
MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
VISUAL_SIGMA_M = 0.0020
VISUAL_BOUND_M = math.sqrt(5.991) * VISUAL_SIGMA_M
EFFECTIVE_CAPTURE_RADIUS_M = 0.00025
BASELINE_MAX_RADIUS_M = VISUAL_BOUND_M - EFFECTIVE_CAPTURE_RADIUS_M
POSTERIOR_COVERAGE = 0.99
METHODS = ("proposed_posterior", "baseline_a", "baseline_b", "baseline_c_entropy")
RETRY_OFFSET_M = 0.00050
MAX_DIRECT_ATTEMPTS = 3


@dataclass(frozen=True)
class PoseTask:
    pose_id: str
    radius_m: float
    angle_rad: float
    x_m: float
    y_m: float
    friction_slide: float
    contact_timeconst_s: float
    target_down_force_n: float
    baseline_phase_rad: float
    posterior_seed: int
    sensor_noise_seed: int | None = None
    sensor_delay_s: float = 0.0


_RUNTIME: M8DirectionEnsemble | None = None
_OUTPUT: Path | None = None
_REUSE_PAB_FROM: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_tasks(count: int, seed: int, *, wrench_noise: bool = False) -> list[PoseTask]:
    power = int(math.ceil(math.log2(max(count, 1))))
    samples = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(power)[:count]
    # The learned classifier is used only for ambiguous top-contact states.
    # This radial interval matches the frozen training support that also lies
    # inside the declared visual 95% acceptance set.
    radius_min_m = 0.00150
    radius_max_m = VISUAL_BOUND_M
    tasks = []
    for index, sample in enumerate(samples):
        radius = math.sqrt(
            radius_min_m**2 + float(sample[0]) * (radius_max_m**2 - radius_min_m**2)
        )
        angle = 2.0 * math.pi * float(sample[1])
        tasks.append(
            PoseTask(
                pose_id=f"sobol_{index:03d}",
                radius_m=radius,
                angle_rad=angle,
                x_m=radius * math.cos(angle),
                y_m=radius * math.sin(angle),
                friction_slide=0.55 + 0.40 * float(sample[2]),
                contact_timeconst_s=0.040 + 0.030 * float(sample[3]),
                target_down_force_n=2.5 + float(sample[4]),
                baseline_phase_rad=2.0 * math.pi * float(sample[5]),
                posterior_seed=seed * 1000 + index,
                sensor_noise_seed=(seed * 10000 + index if wrench_noise else None),
                sensor_delay_s=(0.002 + 0.006 * ((index * 0.61803398875) % 1.0) if wrench_noise else 0.0),
            )
        )
    return tasks


def _worker_init(output: str, reuse_pab_from: str | None) -> None:
    global _RUNTIME, _OUTPUT, _REUSE_PAB_FROM
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble(MODEL_ROOT)
    _OUTPUT = Path(output)
    _REUSE_PAB_FROM = None if reuse_pab_from is None else Path(reuse_pab_from)


def _run_single(task: PoseTask, method: str, *, max_transport_m: float | None = 0.0015) -> RunResult:
    assert _RUNTIME is not None
    return run_method(
        method,
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=_RUNTIME if method in {"learned", "direct"} else None,
        visual_sigma_m=VISUAL_SIGMA_M,
        visual_bound_m=VISUAL_BOUND_M,
        posterior_coverage=POSTERIOR_COVERAGE,
        posterior_seed=task.posterior_seed,
        max_transport_m=max_transport_m,
        baseline_phase_rad=task.baseline_phase_rad,
        baseline_max_radius_m=BASELINE_MAX_RADIUS_M,
        target_down_force_n=task.target_down_force_n,
        friction_slide=task.friction_slide,
        contact_timeconst_s=task.contact_timeconst_s,
        active_time_budget_s=130.0,
        continue_to_bottom=False,
        wrench_observation=(
            None
            if task.sensor_noise_seed is None
            else WrenchObservationConfig(
                seed=task.sensor_noise_seed,
                delay_s=task.sensor_delay_s,
            )
        ),
        random_search_seed=task.posterior_seed + 700000,
    )


def _run_result_success(result: RunResult) -> bool:
    return bool(
        result.offline_capture_validated
        if result.offline_capture_validated is not None
        else result.inner_wall_confirmed
    )


def _aggregate_direct_attempts(attempts: list[RunResult]) -> dict:
    success_index = next(
        (index for index, item in enumerate(attempts) if _run_result_success(item)),
        None,
    )
    successful = None if success_index is None else attempts[success_index]
    first = attempts[0]
    return {
        "method": "baseline_b_shared_probe_direct_retry",
        "inner_wall_confirmed": successful is not None,
        "offline_capture_validated": successful is not None,
        "first_attempt_success": _run_result_success(first),
        "attempts_used": len(attempts),
        "result": successful.result if successful is not None else attempts[-1].result,
        "total_active_time_s": float(sum(item.total_active_time_s for item in attempts)),
        "time_to_inner_wall_s": (
            None
            if successful is None
            else float(
                sum(item.total_active_time_s for item in attempts[:success_index])
                + float(successful.time_to_inner_wall_s or 0.0)
            )
        ),
        "contact_xy_path_mm": float(sum(item.contact_xy_path_mm for item in attempts)),
        "wear_proxy_mj": float(sum(item.wear_proxy_mj for item in attempts)),
        "max_force_n": float(max(item.max_force_n for item in attempts)),
        "max_filtered_force_n": float(
            max(float(item.max_filtered_force_n or 0.0) for item in attempts)
        ),
        "max_torque_mnm": float(max(item.max_torque_mnm for item in attempts)),
        "hard_stop": bool(any(item.hard_stop for item in attempts)),
        "attempt_results": [asdict(item) for item in attempts],
        "first_fingerprint_sha256": first.fingerprint_sha256,
        "control_direction_deg": first.control_direction_deg,
        "top_direction_error_deg": first.direction_error_deg,
        "aps_set_size": first.aps_set_size,
        "circular_concentration": first.circular_concentration,
    }


def _run_pose(task: PoseTask) -> dict:
    assert _RUNTIME is not None and _OUTPUT is not None
    started = time.perf_counter()
    if _REUSE_PAB_FROM is not None:
        source = json.loads(
            (_REUSE_PAB_FROM / "trials" / f"{task.pose_id}.json").read_text(
                encoding="utf-8"
            )
        )
        source_task = source["task"]
        for key, value in asdict(task).items():
            if source_task.get(key) != value:
                raise ValueError(
                    f"frozen P/A/B task mismatch for {task.pose_id}: {key}"
                )
        baseline_c_entropy = _run_single(task, "entropy_grid")
        record = {
            "task": source_task,
            "methods": {
                "proposed_posterior": source["methods"]["proposed_posterior"],
                "baseline_a": source["methods"]["baseline_a"],
                "baseline_b": source["methods"]["baseline_b"],
                "baseline_c_entropy": asdict(baseline_c_entropy),
            },
            "fairness": {
                **source["fairness"],
                "pab_reused_byte_for_byte_from_frozen_audit": True,
                "baseline_c_same_initial_pose_nuisance_and_sensor_seed": True,
                "baseline_c_all_probe_motion_and_contact_cost_counted": True,
                "baseline_c_spiral_fallback": False,
            },
            "diagnostics": source["diagnostics"],
            "worker_elapsed_s": time.perf_counter() - started,
            "created_utc": _utc_now(),
        }
        _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
        return record

    initial_xy = np.array([task.x_m, task.y_m], dtype=float)
    proposed_posterior = _run_single(task, "learned", max_transport_m=None)
    baseline_a = _run_single(task, "baseline")
    baseline_c_entropy = _run_single(task, "entropy_grid")
    first_direct = _run_single(task, "direct")
    direct_attempts = [first_direct]
    if not _run_result_success(first_direct) and first_direct.control_direction_deg is not None:
        phi = math.radians(first_direct.control_direction_deg)
        normal = np.array([-math.sin(phi), math.cos(phi)], dtype=float)
        for sign in (1.0, -1.0):
            retry_start = initial_xy + sign * RETRY_OFFSET_M * normal
            retry_task = PoseTask(
                **{
                    **asdict(task),
                    "x_m": float(retry_start[0]),
                    "y_m": float(retry_start[1]),
                    "posterior_seed": task.posterior_seed + len(direct_attempts),
                    "sensor_noise_seed": (
                        None
                        if task.sensor_noise_seed is None
                        else task.sensor_noise_seed + len(direct_attempts)
                    ),
                }
            )
            direct_attempts.append(_run_single(retry_task, "direct"))
            if _run_result_success(direct_attempts[-1]) or len(direct_attempts) >= MAX_DIRECT_ATTEMPTS:
                break

    direct = _aggregate_direct_attempts(direct_attempts)
    proposed_dicts = {"proposed_posterior": asdict(proposed_posterior)}
    shared_hashes = {
        proposed_posterior.fingerprint_sha256,
        first_direct.fingerprint_sha256,
    }
    true_direction_deg = math.degrees(math.atan2(-task.y_m, -task.x_m)) % 360.0
    control_direction_deg = float(first_direct.control_direction_deg or 0.0)
    control_error_deg = abs((control_direction_deg - true_direction_deg + 180.0) % 360.0 - 180.0)
    direct_margin_mm = 1000.0 * (
        EFFECTIVE_CAPTURE_RADIUS_M
        - task.radius_m * abs(math.sin(math.radians(control_error_deg)))
    )
    record = {
        "task": asdict(task),
        "methods": {
            **proposed_dicts,
            "baseline_a": asdict(baseline_a),
            "baseline_b": direct,
            "baseline_c_entropy": asdict(baseline_c_entropy),
        },
        "fairness": {
            "first_probe_hash_match": len(shared_hashes) == 1,
            "first_probe_hash": proposed_posterior.fingerprint_sha256,
            "same_frozen_direction_model": True,
            "same_initial_pose_and_nuisance": True,
            "direct_retry_offsets_mm": [0.5, -0.5],
            "reset_motion_time_excluded": True,
        },
        "diagnostics": {
            "true_direction_deg": true_direction_deg,
            "control_direction_deg": control_direction_deg,
            "control_direction_error_deg": control_error_deg,
            "direct_ray_geometric_margin_mm": direct_margin_mm,
        },
        "worker_elapsed_s": time.perf_counter() - started,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{task.pose_id}.json", record)
    return record


def _success(value: dict) -> bool:
    return bool(value.get("offline_capture_validated", value["inner_wall_confirmed"]))


def _method_summary(records: list[dict], method: str) -> dict:
    values = [record["methods"][method] for record in records]
    successful = [value for value in values if _success(value)]
    def quantile(key: str, q: float, source: list[dict] = values) -> float | None:
        data = [float(value[key]) for value in source if value.get(key) is not None]
        return None if not data else float(np.quantile(data, q))
    result = {
        "trials": len(values),
        "successes": len(successful),
        "success_rate": len(successful) / len(values),
        "first_attempt_successes": sum(
            bool(
                value.get(
                    "first_attempt_success",
                    value.get("offline_capture_validated", value["inner_wall_confirmed"]),
                )
            )
            for value in values
        ),
        "time_success_median_s": quantile("time_to_inner_wall_s", 0.5, successful),
        "actual_path_median_mm": (
            None
            if not successful
            else float(
                np.median(
                    [
                        value.get(
                            "actual_xy_path_mm", value["contact_xy_path_mm"]
                        )
                        for value in successful
                    ]
                )
            )
        ),
        # Efficiency summaries are conditioned on a common valid endpoint.
        # Keep all-trial values separately so failed long searches are not
        # silently mixed into a quantity labelled as successful efficiency.
        "contact_path_median_mm": quantile(
            "contact_xy_path_mm", 0.5, successful
        ),
        "wear_proxy_median_mj": quantile("wear_proxy_mj", 0.5, successful),
        "contact_path_all_trials_median_mm": quantile(
            "contact_xy_path_mm", 0.5
        ),
        "actual_path_all_trials_median_mm": float(
            np.median(
                [
                    value.get("actual_xy_path_mm", value["contact_xy_path_mm"])
                    for value in values
                ]
            )
        ),
        "wear_proxy_all_trials_median_mj": quantile("wear_proxy_mj", 0.5),
        "max_force_p90_n": quantile("max_force_n", 0.9),
        "max_filtered_force_p90_n": quantile("max_filtered_force_n", 0.9),
        "max_torque_p90_mnm": quantile("max_torque_mnm", 0.9),
        "hard_stops": sum(bool(value["hard_stop"]) for value in values),
    }
    if method.startswith("proposed"):
        result["transport_median_mm"] = quantile("transport_distance_mm", 0.5)
        result["local_radius_median_mm"] = quantile("spiral_max_radius_mm", 0.5)
        result["global_fallback_count"] = sum(
            bool(value.get("global_outward_fallback_used")) for value in values
        )
    if method == "baseline_b":
        result["attempts_median"] = quantile("attempts_used", 0.5)
    if method == "baseline_c_entropy":
        result["probe_count_median"] = quantile("entropy_grid_probe_count", 0.5)
        result["final_entropy_median_nats"] = quantile(
            "entropy_grid_final_nats", 0.5
        )
        result["credible_radius_median_mm"] = quantile(
            "entropy_grid_credible_radius_mm", 0.5
        )
        result["spiral_fallback_count"] = sum(
            bool(value.get("entropy_grid_used_spiral")) for value in values
        )
    return result


def _paired_counts(records: list[dict], left: str, right: str) -> dict:
    left_only = right_only = both = neither = 0
    for record in records:
        a = _success(record["methods"][left])
        b = _success(record["methods"][right])
        if a and b:
            both += 1
        elif a:
            left_only += 1
        elif b:
            right_only += 1
        else:
            neither += 1
    discordant = left_only + right_only
    exact_p = 1.0
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        exact_p = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "both_success": both,
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_failed": neither,
        "mcnemar_exact_two_sided_p": exact_p,
    }


def _paired_efficiency(records: list[dict], left: str, right: str) -> dict:
    pairs = [
        (record["methods"][left], record["methods"][right])
        for record in records
        if _success(record["methods"][left]) and _success(record["methods"][right])
    ]
    result: dict[str, float | int | None] = {"both_success": len(pairs)}
    for key in (
        "time_to_inner_wall_s",
        "actual_xy_path_mm",
        "contact_xy_path_mm",
        "wear_proxy_mj",
        "max_force_n",
    ):
        if not pairs:
            result[f"left_minus_right_{key}_median"] = None
            result[f"left_lower_{key}_rate"] = None
            continue
        left_values = np.asarray(
            [
                float(
                    pair[0].get(key, pair[0].get("contact_xy_path_mm"))
                )
                for pair in pairs
            ]
        )
        right_values = np.asarray(
            [
                float(
                    pair[1].get(key, pair[1].get("contact_xy_path_mm"))
                )
                for pair in pairs
            ]
        )
        result[f"left_minus_right_{key}_median"] = float(
            np.median(left_values - right_values)
        )
        result[f"left_lower_{key}_rate"] = float(np.mean(left_values < right_values))
    return result


def _summary(records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    margins = np.asarray([record["diagnostics"]["direct_ray_geometric_margin_mm"] for record in records])
    direct_first = np.asarray(
        [record["methods"]["baseline_b"]["first_attempt_success"] for record in records], dtype=bool
    )
    return {
        "stage": "M8_fair_policy_benchmark",
        "design": {
            "poses": len(records),
            "visual_95_radius_mm": 1000.0 * VISUAL_BOUND_M,
            "random_position_design": "area-uniform Sobol in 1.50 mm to visual 95% radius",
            "friction_range": [0.55, 0.95],
            "contact_timeconst_range_s": [0.04, 0.07],
            "down_force_range_n": [2.5, 3.5],
            "same_probe_for_proposed_and_baseline_b_first_attempt": True,
            "baseline_b_max_attempts": MAX_DIRECT_ATTEMPTS,
            "baseline_b_retry_offset_mm": RETRY_OFFSET_M * 1000.0,
            "reset_motion_time_excluded_from_baseline_b": True,
            "online_capture_uses_privileged_mujoco_tags": False,
            "offline_success_scoring_uses_mujoco_geometry": True,
            "entropy_grid_baseline": (
                "Chen-inspired visual/CAD grid belief; each guarded low-force touchdown maximizes expected "
                "entropy reduction from a calibrated F/T edge-load cue; direct posterior-mean commit after "
                "the fixed probe budget; no conical learner and no spiral fallback"
            ),
            "posterior_residual_coverage": POSTERIOR_COVERAGE,
            "wrench_observation_model": (
                "none"
                if not records or records[0]["task"].get("sensor_noise_seed") is None
                else WrenchObservationConfig().to_dict()
            ),
            "sensor_delay_range_s": (
                None
                if not records or records[0]["task"].get("sensor_noise_seed") is None
                else [
                    min(float(item["task"]["sensor_delay_s"]) for item in records),
                    max(float(item["task"]["sensor_delay_s"]) for item in records),
                ]
            ),
            "proposed_global_outward_fallback": True,
            "baseline_c_spiral_fallback": False,
            "active_time_budget_s": 130.0,
        },
        "methods": methods,
        "paired_proposed_posterior_vs_baseline_b": _paired_counts(
            records, "proposed_posterior", "baseline_b"
        ),
        "paired_proposed_posterior_vs_baseline_a": _paired_counts(
            records, "proposed_posterior", "baseline_a"
        ),
        "paired_efficiency_proposed_vs_baseline_b": _paired_efficiency(
            records, "proposed_posterior", "baseline_b"
        ),
        "paired_efficiency_proposed_vs_baseline_a": _paired_efficiency(
            records, "proposed_posterior", "baseline_a"
        ),
        "paired_proposed_posterior_vs_baseline_c_entropy": _paired_counts(
            records, "proposed_posterior", "baseline_c_entropy"
        ),
        "paired_efficiency_proposed_vs_baseline_c_entropy": _paired_efficiency(
            records, "proposed_posterior", "baseline_c_entropy"
        ),
        "direct_margin_diagnostic": {
            "negative_margin_trials": int(np.sum(margins < 0.0)),
            "negative_margin_first_attempt_success_rate": (
                None if not np.any(margins < 0.0) else float(np.mean(direct_first[margins < 0.0]))
            ),
            "nonnegative_margin_first_attempt_success_rate": (
                None if not np.any(margins >= 0.0) else float(np.mean(direct_first[margins >= 0.0]))
            ),
        },
        "probe_hash_match_rate": float(
            np.mean([record["fairness"]["first_probe_hash_match"] for record in records])
        ),
        "claim_boundary": (
            "Frozen M8 simulation benchmark inside the declared visual prior. Position and bounded contact "
            "nuisances are unseen Sobol samples. When enabled, the F/T bias/noise/coupling/delay model is a "
            "pre-hardware engineering prior rather than a measured sensor distribution. Proposed and Baseline B "
            "share an identical first fingerprint "
            "and frozen learner. Baseline-B reset travel is not simulated or timed. Online handoff is sensor-only; "
            "MuJoCo geometry is used only for offline success scoring. This is not hardware evidence."
            " Baseline C uses a simulation-calibrated CAD/F-T likelihood that must be recalibrated on iiwa hardware."
        ),
        "updated_utc": _utc_now(),
    }


def _write_report(output: Path, report: dict) -> None:
    _atomic_json(output / "report.json", report)
    lines = [
        "# M8 Fair Policy Benchmark",
        "",
        "Efficiency medians are conditioned on successful offline-validated capture; force/torque p90 and hard stops use all trials.",
        "",
        "| method | eventual success | first attempt | capture time median | successful total/contact path | successful wear median | raw / filtered force p90 | torque p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        lines.append(
            f"| {method} | {item['successes']}/{item['trials']} | "
            f"{item['first_attempt_successes']}/{item['trials']} | "
            f"{item['time_success_median_s']} s | {item['actual_path_median_mm']:.2f} / "
            f"{item['contact_path_median_mm']:.2f} mm | "
            f"{item['wear_proxy_median_mj']:.2f} mJ | {item['max_force_p90_n']:.2f} / "
            f"{item['max_filtered_force_p90_n']:.2f} N | {item['max_torque_p90_mnm']:.2f} mNm |"
        )
    lines.extend(
        [
            "",
            "## Paired counts",
            "",
            *[
                f"- {key}: {value}"
                for key, value in report["paired_proposed_posterior_vs_baseline_b"].items()
            ],
            "",
            "### Proposed vs entropy-grid",
            *[
                f"- {key}: {value}"
                for key, value in report[
                    "paired_proposed_posterior_vs_baseline_c_entropy"
                ].items()
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
    tasks = build_tasks(args.poses, args.seed, wrench_noise=args.wrench_noise)
    existing: list[dict] = []
    pending: list[PoseTask] = []
    for task in tasks:
        path = output / "trials" / f"{task.pose_id}.json"
        if args.resume and path.exists():
            existing.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(task)
    records = list(existing)
    started = time.perf_counter()

    def checkpoint() -> None:
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        done_new = len(records) - len(existing)
        rate = done_new / elapsed
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
            initargs=(
                str(output),
                None
                if args.reuse_pab_from is None
                else str(Path(args.reuse_pab_from).resolve()),
            ),
        ) as pool:
            futures = {pool.submit(_run_pose, task): task for task in pending}
            with tqdm(
                total=len(tasks), initial=len(existing), desc="M8 fair policies", unit="pose", dynamic_ncols=True
            ) as bar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {"task": asdict(task), "error": repr(exc), "created_utc": _utc_now()}
                        _atomic_json(output / "trials" / f"{task.pose_id}.json", record)
                    records.append(record)
                    checkpoint()
                    bar.update(1)
                    valid = [item for item in records if "error" not in item]
                    bar.set_postfix(
                        prop=sum(_success(item["methods"]["proposed_posterior"]) for item in valid),
                        direct=sum(_success(item["methods"]["baseline_b"]) for item in valid),
                        entropy=sum(_success(item["methods"]["baseline_c_entropy"]) for item in valid),
                        refresh=False,
                    )
    errors = [record for record in records if "error" in record]
    if errors:
        print(json.dumps(errors, indent=2))
        return 1
    report = _summary(records)
    report["design"]["pab_reused_from_frozen_audit"] = (
        None
        if args.reuse_pab_from is None
        else str(Path(args.reuse_pab_from).resolve().relative_to(ROOT))
    )
    _write_report(output, report)
    _atomic_json(
        output / "manifest.json",
        {"tasks": [asdict(task) for task in tasks], "completed": len(records), "updated_utc": _utc_now()},
    )
    checkpoint()
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fair M8 Proposed/A/B/C paired benchmark.")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--poses", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--workers", type=int, default=min(18, max(1, os.cpu_count() or 1)))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-pab-from",
        default=None,
        help=(
            "reuse Proposed/A/B trial dictionaries from a frozen paired audit; "
            "only Baseline C is executed"
        ),
    )
    parser.add_argument(
        "--wrench-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the shared pre-hardware F/T observation prior",
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
