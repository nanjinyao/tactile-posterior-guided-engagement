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
from src.render_m8_search_comparison import RunResult, run_method
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop"
)
MODEL_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SIZES = ("m10", "m12")
METHODS = ("proposed", "outward", "oracle")
METHOD_TO_RUNTIME = {
    "proposed": "learned",
    "outward": "baseline",
    "oracle": "oracle",
}
M8_BIT_RADIUS_M = 0.006 / math.sqrt(3.0)
VISUAL_CHI95_RADIUS = math.sqrt(5.991)

# Frozen before the S2 rollouts. Oracle is an execution-feasibility gate;
# Proposed is the zero-shot task gate. Baseline performance does not decide
# whether the learned interface transfers.
MIN_ORACLE_BOTTOM_RATE = 0.90
MIN_PROPOSED_BOTTOM_RATE = 0.80
MAX_PROPOSED_HARD_STOPS = 0
MAX_PROPOSED_GT90_DIRECTION_ERRORS = 0


@dataclass(frozen=True)
class CrossSizeTask:
    size: str
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
    sensor_noise_seed: int
    sensor_delay_s: float


_RUNTIME: M8DirectionEnsemble | None = None
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


def _load_spec(size: str) -> dict:
    return json.loads(
        (ROOT / "configs" / f"allen_{size}_source.json").read_text(
            encoding="utf-8"
        )
    )


def _size_contract(size: str) -> dict:
    spec = _load_spec(size)
    bit_radius_m = float(spec["key_af_m"]) / math.sqrt(3.0)
    capture_radius_m = float(spec["derived"]["r_cap_top_m"])
    visual_bound_m = float(max(spec["visual_offset_sweep_m"]))
    visual_sigma_m = visual_bound_m / VISUAL_CHI95_RADIUS
    radius_min_m = max(
        float(spec["visual_offset_min_m"]),
        4.0 * capture_radius_m,
    )
    if radius_min_m >= visual_bound_m:
        raise ValueError(f"invalid visual deployment annulus for {size}")
    return {
        "size": size,
        "spec_path": str(ROOT / "configs" / f"allen_{size}_source.json"),
        "model_path": str(ROOT / "models" / f"allen_{size}_source.xml"),
        "bit_radius_m": bit_radius_m,
        "capture_radius_m": capture_radius_m,
        "visual_bound_m": visual_bound_m,
        "visual_sigma_m": visual_sigma_m,
        "radius_min_m": radius_min_m,
        "baseline_max_radius_m": visual_bound_m - capture_radius_m,
        "spiral_pitch_m": 2.0 * capture_radius_m * 0.80,
        "max_transport_m": 0.0015 * bit_radius_m / M8_BIT_RADIUS_M,
        "xy_speed_m_s": 0.00225,
        "active_time_budget_s": 180.0,
    }


def build_tasks(size: str, count: int, seed: int) -> list[CrossSizeTask]:
    contract = _size_contract(size)
    power = int(math.ceil(math.log2(max(1, count))))
    samples = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(power)[:count]
    r_min = float(contract["radius_min_m"])
    r_max = float(contract["visual_bound_m"])
    tasks = []
    for index, sample in enumerate(samples):
        radius = math.sqrt(r_min**2 + float(sample[0]) * (r_max**2 - r_min**2))
        angle = 2.0 * math.pi * float(sample[1])
        tasks.append(
            CrossSizeTask(
                size=size,
                pose_id=f"{size}_sobol_{index:03d}",
                radius_m=radius,
                angle_rad=angle,
                x_m=radius * math.cos(angle),
                y_m=radius * math.sin(angle),
                friction_slide=0.55 + 0.40 * float(sample[2]),
                contact_timeconst_s=0.040 + 0.030 * float(sample[3]),
                target_down_force_n=2.5 + float(sample[4]),
                baseline_phase_rad=2.0 * math.pi * float(sample[5]),
                posterior_seed=seed * 1000 + index,
                sensor_noise_seed=seed * 10000 + index,
                sensor_delay_s=0.002 + 0.006 * float(sample[6]),
            )
        )
    return tasks


def _worker_init(output: str) -> None:
    global _RUNTIME, _OUTPUT
    torch.set_num_threads(1)
    _RUNTIME = M8DirectionEnsemble(MODEL_ROOT)
    _OUTPUT = Path(output)


def _run_one_method(
    task: CrossSizeTask,
    method: str,
    contract: dict,
) -> RunResult:
    assert _RUNTIME is not None
    runtime_method = METHOD_TO_RUNTIME[method]
    return run_method(
        runtime_method,
        initial_xy_m=np.array([task.x_m, task.y_m], dtype=float),
        direction_runtime=_RUNTIME if runtime_method == "learned" else None,
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
        active_time_budget_s=float(contract["active_time_budget_s"]),
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=task.sensor_noise_seed,
            delay_s=task.sensor_delay_s,
        ),
        observation_label="shared pre-hardware wrench prior",
        spec_path=contract["spec_path"],
        model_path=contract["model_path"],
        spiral_pitch_m=float(contract["spiral_pitch_m"]),
        xy_speed_m_s=float(contract["xy_speed_m_s"]),
        cad_conditioned_insertion=True,
    )


def _run_pose(task: CrossSizeTask) -> dict:
    assert _OUTPUT is not None
    contract = _size_contract(task.size)
    methods = {
        method: asdict(_run_one_method(task, method, contract))
        for method in METHODS
    }
    record = {
        "task": asdict(task),
        "scale_contract": contract,
        "methods": methods,
        "created_utc": _utc_now(),
    }
    _atomic_json(
        _OUTPUT / task.size / "trials" / f"{task.pose_id}.json",
        record,
    )
    return record


def _bottom_success(result: dict) -> bool:
    return bool(result["bottom_reached"] and not result["hard_stop"])


def _failure_category(result: dict) -> str:
    if result["hard_stop"]:
        return "hard_stop"
    if result["bottom_reached"]:
        return "bottom_verified"
    if result.get("funnel_abort_reason"):
        return f"funnel_abort:{result['funnel_abort_reason']}"
    if result["inner_wall_confirmed"]:
        return "capture_without_bottom"
    return "no_sensor_capture"


def _method_summary(records: list[dict], method: str) -> dict:
    rows = [record["methods"][method] for record in records]
    successful = [row for row in rows if _bottom_success(row)]
    top1_direction_errors = [
        float(row["direction_error_deg"])
        for row in rows
        if row.get("direction_error_deg") is not None
    ]
    control_direction_errors = [
        abs(
            (
                float(row["control_direction_deg"])
                - float(row["true_direction_deg"])
                + 180.0
            )
            % 360.0
            - 180.0
        )
        for row in rows
        if row.get("control_direction_deg") is not None
        and row.get("true_direction_deg") is not None
    ]
    categories: dict[str, int] = {}
    for row in rows:
        category = _failure_category(row)
        categories[category] = categories.get(category, 0) + 1

    def successful_median(key: str) -> float | None:
        values = [float(row[key]) for row in successful if row.get(key) is not None]
        return None if not values else float(np.median(values))

    return {
        "trials": len(rows),
        "bottom_successes": len(successful),
        "bottom_success_rate": len(successful) / max(1, len(rows)),
        "capture_successes": sum(bool(row["inner_wall_confirmed"]) for row in rows),
        "hard_stops": sum(bool(row["hard_stop"]) for row in rows),
        "failure_categories": categories,
        "time_to_bottom_success_median_s": successful_median("time_to_bottom_s"),
        "total_path_success_median_mm": successful_median("actual_xy_path_mm"),
        "contact_path_success_median_mm": successful_median("contact_xy_path_mm"),
        "wear_success_median_mj": successful_median("wear_proxy_mj"),
        "max_force_p90_n": float(
            np.quantile([float(row["max_force_n"]) for row in rows], 0.90)
        ),
        "max_torque_p90_mnm": float(
            np.quantile([float(row["max_torque_mnm"]) for row in rows], 0.90)
        ),
        # Kept for compatibility: this is the top-1 direction-bin diagnostic.
        "direction_error_median_deg": (
            None
            if not top1_direction_errors
            else float(np.median(top1_direction_errors))
        ),
        "direction_error_p90_deg": (
            None
            if not top1_direction_errors
            else float(np.quantile(top1_direction_errors, 0.90))
        ),
        "direction_error_gt90_count": sum(
            error > 90.0 for error in top1_direction_errors
        ),
        # The planner executes the circular-mean direction, so this is the
        # load-bearing direction metric for the full-loop gate.
        "control_direction_error_median_deg": (
            None
            if not control_direction_errors
            else float(np.median(control_direction_errors))
        ),
        "control_direction_error_p90_deg": (
            None
            if not control_direction_errors
            else float(np.quantile(control_direction_errors, 0.90))
        ),
        "control_direction_error_gt90_count": sum(
            error > 90.0 for error in control_direction_errors
        ),
        "global_outward_fallback_count": sum(
            bool(row.get("global_outward_fallback_used")) for row in rows
        ),
    }


def _paired_summary(records: list[dict], left: str, right: str) -> dict:
    pairs = [
        (
            _bottom_success(record["methods"][left]),
            _bottom_success(record["methods"][right]),
        )
        for record in records
    ]
    both = [
        record
        for record, pair in zip(records, pairs)
        if pair == (True, True)
    ]
    return {
        "both_success": sum(pair == (True, True) for pair in pairs),
        "left_only_success": sum(pair == (True, False) for pair in pairs),
        "right_only_success": sum(pair == (False, True) for pair in pairs),
        "both_failed": sum(pair == (False, False) for pair in pairs),
        "common_success_left_minus_right_time_median_s": (
            None
            if not both
            else float(
                np.median(
                    [
                        float(record["methods"][left]["time_to_bottom_s"])
                        - float(record["methods"][right]["time_to_bottom_s"])
                        for record in both
                    ]
                )
            )
        ),
        "common_success_left_minus_right_path_median_mm": (
            None
            if not both
            else float(
                np.median(
                    [
                        float(record["methods"][left]["actual_xy_path_mm"])
                        - float(record["methods"][right]["actual_xy_path_mm"])
                        for record in both
                    ]
                )
            )
        ),
    }


def _summarize_size(size: str, records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    proposed = methods["proposed"]
    oracle = methods["oracle"]
    gate = bool(
        oracle["bottom_success_rate"] >= MIN_ORACLE_BOTTOM_RATE
        and proposed["bottom_success_rate"] >= MIN_PROPOSED_BOTTOM_RATE
        and proposed["hard_stops"] <= MAX_PROPOSED_HARD_STOPS
        and proposed["control_direction_error_gt90_count"]
        <= MAX_PROPOSED_GT90_DIRECTION_ERRORS
    )
    return {
        "size": size,
        "scale_contract": _size_contract(size),
        "methods": methods,
        "paired": {
            "proposed_vs_outward": _paired_summary(records, "proposed", "outward"),
            "proposed_vs_oracle": _paired_summary(records, "proposed", "oracle"),
        },
        "zero_shot_full_loop_gate_pass": gate,
        "gate_contract": {
            "oracle_bottom_rate_min": MIN_ORACLE_BOTTOM_RATE,
            "proposed_bottom_rate_min": MIN_PROPOSED_BOTTOM_RATE,
            "proposed_hard_stops_max": MAX_PROPOSED_HARD_STOPS,
            "proposed_gt90_direction_errors_max": MAX_PROPOSED_GT90_DIRECTION_ERRORS,
            "direction_metric": "executed circular-mean control direction",
        },
    }


def _write_summary(path: Path, report: dict) -> None:
    def metric(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Frozen-M8 M10/M12 Zero-Shot Full-Loop Audit",
        "",
        f"Updated: {report['updated_utc']}",
        "",
        "M10/M12 data are evaluation-only. The M8 P1v3 model, normalization, APS",
        "and learning thresholds remain frozen. CAD geometry, capture radius,",
        "coverage pitch, visual support and bounded transport cap are size-conditioned.",
        "",
        "| size | method | CAD-bottom | capture | hard stop | bottom time | path | force/torque p90 | control-dir med/p90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        size_report = report["sizes"][size]
        for method in METHODS:
            item = size_report["methods"][method]
            lines.append(
                f"| {size.upper()} | {method} | {item['bottom_successes']}/{item['trials']} | "
                f"{item['capture_successes']}/{item['trials']} | {item['hard_stops']} | "
                f"{metric(item['time_to_bottom_success_median_s'])} s | "
                f"{metric(item['total_path_success_median_mm'])} mm | "
                f"{item['max_force_p90_n']:.2f} N / {item['max_torque_p90_mnm']:.2f} mNm | "
                f"{metric(item['control_direction_error_median_deg'])}/{metric(item['control_direction_error_p90_deg'])} deg |"
            )
        lines.extend(
            [
                "",
                f"- {size.upper()} gate: **{'PASS' if size_report['zero_shot_full_loop_gate_pass'] else 'FAIL'}**",
                f"- {size.upper()} Proposed failure split: `{json.dumps(size_report['methods']['proposed']['failure_categories'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_sizes: dict[str, dict] = {}
    for size_index, size in enumerate(args.sizes):
        size_output = output / size
        (size_output / "trials").mkdir(parents=True, exist_ok=True)
        tasks = build_tasks(size, args.poses, args.seed + 1009 * size_index)
        records: list[dict] = []
        pending: list[CrossSizeTask] = []
        for task in tasks:
            trial_path = size_output / "trials" / f"{task.pose_id}.json"
            if args.resume and trial_path.exists():
                records.append(json.loads(trial_path.read_text(encoding="utf-8")))
            else:
                pending.append(task)

        started = time.perf_counter()

        def checkpoint() -> None:
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            new_done = max(0, len(records) - (len(tasks) - len(pending)))
            rate = new_done / elapsed
            _atomic_json(
                size_output / "progress.json",
                {
                    "size": size,
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
                    desc=f"{size.upper()} S2 full loop",
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
                                size_output / "trials" / f"{task.pose_id}.json",
                                record,
                            )
                        records.append(record)
                        checkpoint()
                        bar.update(1)
                        valid = [item for item in records if "error" not in item]
                        bar.set_postfix(
                            P=sum(
                                _bottom_success(item["methods"]["proposed"])
                                for item in valid
                            ),
                            A=sum(
                                _bottom_success(item["methods"]["outward"])
                                for item in valid
                            ),
                            O=sum(
                                _bottom_success(item["methods"]["oracle"])
                                for item in valid
                            ),
                            refresh=False,
                        )

        errors = [record for record in records if "error" in record]
        if errors:
            _atomic_json(size_output / "errors.json", {"errors": errors})
            raise RuntimeError(f"{size}: {len(errors)} rollout tasks failed")
        all_sizes[size] = _summarize_size(size, records)
        _atomic_json(
            size_output / "manifest.json",
            {
                "size": size,
                "tasks": [asdict(task) for task in tasks],
                "completed": len(records),
                "training_use": "forbidden",
                "updated_utc": _utc_now(),
            },
        )
        checkpoint()

    report = {
        "experiment": "frozen_M8_model_M10_M12_zero_shot_complete_closed_loop_S2",
        "sizes": all_sizes,
        "model_root": str(MODEL_ROOT.relative_to(ROOT)),
        "learning_contract": {
            "M8_model_frozen": True,
            "M10_M12_training_or_calibration_samples": 0,
            "M10_M12_evaluation_samples_reusable_for_training": False,
        },
        "all_size_gates_pass": all(
            item["zero_shot_full_loop_gate_pass"] for item in all_sizes.values()
        ),
        "claim_boundary": (
            "Simulation-only paired S2 audit inside size-specific synthetic visual 95% disks. "
            "It tests a frozen M8 direction learner with CAD-conditioned geometry and coverage, "
            "but retains simulation-initial control and terminal thresholds. A pass supports "
            "bounded zero-shot full-loop transfer in these MuJoCo fixtures; it does not establish "
            "probability calibration, real iiwa transfer, manufacturing tolerance, passivity, "
            "motor coupling, threaded unscrewing, or extraction. Oracle uses the true center only "
            "as an offline execution-feasibility upper bound."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output / "summary.md", report)
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-M8 zero-shot M10/M12 complete closed-loop S2 audit."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--poses", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(24, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
