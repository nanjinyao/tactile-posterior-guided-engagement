from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.funnel_guard import FunnelGuardConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import RunResult, run_method
from src.run_cross_size_closed_loop_audit import (
    CrossSizeTask,
    _size_contract as _legacy_size_contract,
    build_tasks as _build_tasks,
)
from src.run_m8_strict_complete_loop_audit import (
    _failure_category,
    _strict_bottom_success,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3"
)
DATA_ROOT = ROOT / "outputs/current/datasets/cross_size_s3_adaptation_pilot"
MODEL_ROOT = ROOT / "outputs/current/models/cross_size_s3_direction_head_pilot"
ZERO_SHOT_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"
SIZES = ("m10", "m12")
METHODS = ("zero_shot", "cal_only", "fine_tuned", "outward", "oracle")
METHOD_LABELS = {
    "zero_shot": "M8 zero-shot",
    "cal_only": "size calibration only",
    "fine_tuned": "direction-head fine-tuned",
    "outward": "pure outward",
    "oracle": "privileged straight-to-center",
}
ACTIVE_TIME_BUDGET_S = 130.0
EVALUATION_SEEDS = {"m10": 20260941, "m12": 20260942}
SHARED_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)
MIN_ORACLE_BOTTOM_RATE = 0.90
MIN_ADAPTED_BOTTOM_RATE = 0.80
MAX_ADAPTED_HARD_STOPS = 0
MAX_ADAPTED_GT90_CONTROL_ERRORS = 0


_RUNTIMES: dict[str, M8DirectionEnsemble] = {}
_OUTPUT: Path | None = None
_SIZE: str | None = None
_EXECUTION_BUNDLE_SHA256: str | None = None
_EXECUTION_SIZES: tuple[str, ...] = ()


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": _relative(resolved), "sha256": _sha256(resolved)}


def _ensemble_execution_files(model_root: Path) -> dict:
    contract_path = model_root / "ensemble_interface_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    members = {"contract": _file_record(contract_path), "seeds": []}
    for checkpoint_value in contract["checkpoints"]:
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        members["seeds"].append(
            {
                "checkpoint": _file_record(checkpoint),
                "normalization": _file_record(checkpoint.parent / "normalization.npz"),
                "metrics": _file_record(checkpoint.parent / "metrics.json"),
            }
        )
    return members


def _execution_bundle(sizes: tuple[str, ...] | list[str]) -> dict:
    # The rollout core has several local transitive dependencies.  Hashing the
    # complete source tree makes any mid-run local-code drift fail closed.
    source_paths = {
        str(path.relative_to(ROOT)): path
        for path in sorted((ROOT / "src").rglob("*.py"))
    }
    bundle = {
        "sources": {
            name: _file_record(path) for name, path in source_paths.items()
        },
        "sizes": {},
    }
    for size in sizes:
        contract = _size_contract(size)
        bundle["sizes"][size] = {
            "cad_spec": _file_record(Path(contract["spec_path"])),
            "mujoco_model": _file_record(Path(contract["model_path"])),
            "adaptation_data_contract": _file_record(
                DATA_ROOT / size / "dataset_contract.json"
            ),
            "adaptation_data_manifest": _file_record(
                DATA_ROOT / size / "dataset_manifest.json"
            ),
            "model_manifest": _file_record(MODEL_ROOT / size / "model_manifest.json"),
            "ensembles": {
                "zero_shot": _ensemble_execution_files(ZERO_SHOT_ROOT),
                "cal_only": _ensemble_execution_files(MODEL_ROOT / size / "cal_only"),
                "fine_tuned": _ensemble_execution_files(
                    MODEL_ROOT / size / "fine_tuned"
                ),
            },
        }
    return bundle


def _canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _size_contract(size: str) -> dict:
    contract = dict(_legacy_size_contract(size))
    contract["active_time_budget_s"] = ACTIVE_TIME_BUDGET_S
    contract["fallback_budget_extension_s"] = 0.0
    contract["post_capture_funnel"] = asdict(SHARED_FUNNEL)
    return contract


def build_untouched_tasks(size: str, count: int) -> list[CrossSizeTask]:
    tasks = _build_tasks(size, count, EVALUATION_SEEDS[size])
    return [
        replace(task, pose_id=f"{size}_s3test_{index:03d}")
        for index, task in enumerate(tasks)
    ]


def _worker_init(
    output: str,
    size: str,
    execution_sizes: tuple[str, ...],
    expected_execution_bundle_sha256: str,
) -> None:
    global _RUNTIMES, _OUTPUT, _SIZE, _EXECUTION_BUNDLE_SHA256, _EXECUTION_SIZES
    torch.set_num_threads(1)
    actual_bundle = _execution_bundle(execution_sizes)
    actual_sha256 = _canonical_json_sha256(actual_bundle)
    if actual_sha256 != expected_execution_bundle_sha256:
        raise RuntimeError("execution bundle drifted before worker initialization")
    _EXECUTION_BUNDLE_SHA256 = actual_sha256
    _EXECUTION_SIZES = tuple(execution_sizes)
    _SIZE = size
    _OUTPUT = Path(output)
    _RUNTIMES = {
        "zero_shot": M8DirectionEnsemble(ZERO_SHOT_ROOT),
        "cal_only": M8DirectionEnsemble(MODEL_ROOT / size / "cal_only"),
        "fine_tuned": M8DirectionEnsemble(MODEL_ROOT / size / "fine_tuned"),
    }


def _run_one_method(task: CrossSizeTask, method: str, contract: dict) -> RunResult:
    runtime_method = "learned" if method in _RUNTIMES else method
    if method == "outward":
        runtime_method = "baseline"
    return run_method(
        runtime_method,
        initial_xy_m=np.asarray([task.x_m, task.y_m], dtype=float),
        direction_runtime=_RUNTIMES.get(method),
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
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=task.sensor_noise_seed,
            delay_s=task.sensor_delay_s,
        ),
        observation_label="fresh paired cross-size S3 measured-wrench contract",
        spec_path=contract["spec_path"],
        model_path=contract["model_path"],
        spiral_pitch_m=float(contract["spiral_pitch_m"]),
        xy_speed_m_s=float(contract["xy_speed_m_s"]),
        cad_conditioned_insertion=True,
        conditioned_on_ambiguous_top_contact=True,
        funnel_guard=SHARED_FUNNEL,
    )


def _run_pose(task: CrossSizeTask) -> dict:
    assert (
        _OUTPUT is not None
        and _SIZE == task.size
        and _EXECUTION_BUNDLE_SHA256 is not None
        and _EXECUTION_SIZES
    )
    started = time.perf_counter()
    contract = _size_contract(task.size)
    methods = {
        method: asdict(_run_one_method(task, method, contract))
        for method in METHODS
    }
    probe_hashes = {
        methods[method].get("fingerprint_sha256")
        for method in ("zero_shot", "cal_only", "fine_tuned")
    }
    completion_bundle_sha256 = _canonical_json_sha256(
        _execution_bundle(_EXECUTION_SIZES)
    )
    if completion_bundle_sha256 != _EXECUTION_BUNDLE_SHA256:
        raise RuntimeError("execution bundle drifted during worker task")
    record = {
        "task": asdict(task),
        "scale_contract": contract,
        "methods": methods,
        "fairness": {
            "same_pose_contact_nuisance_and_sensor_seed": True,
            "same_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "zero_budget_extension": all(
                float(row.get("fallback_budget_extension_s", 0.0)) == 0.0
                for row in methods.values()
            ),
            "same_R3_post_capture_funnel": asdict(SHARED_FUNNEL),
            "same_terminal_verifier": True,
            "three_learned_first_probe_hashes_match": len(probe_hashes) == 1,
            "simulator_reset_count": sum(
                int(row.get("simulator_reset_count", 0))
                for row in methods.values()
            ),
            "non_oracle_online_true_center_or_contact_tags": False,
            "oracle_is_privileged_true_center_diagnostic": True,
        },
        "worker_elapsed_s": time.perf_counter() - started,
        "worker_init_execution_bundle_sha256": _EXECUTION_BUNDLE_SHA256,
        "worker_completion_execution_bundle_sha256": completion_bundle_sha256,
        "created_utc": _utc_now(),
    }
    _atomic_json(
        _OUTPUT / task.size / "trials" / f"{task.pose_id}.json",
        record,
    )
    return record


def _control_error(row: dict) -> float | None:
    if row.get("control_direction_deg") is None or row.get("true_direction_deg") is None:
        return None
    return abs(
        (
            float(row["control_direction_deg"])
            - float(row["true_direction_deg"])
            + 180.0
        )
        % 360.0
        - 180.0
    )


def _method_summary(records: list[dict], method: str) -> dict:
    rows = [record["methods"][method] for record in records]
    successful = [row for row in rows if _strict_bottom_success(row)]
    errors = [error for row in rows if (error := _control_error(row)) is not None]

    def median(key: str, source: list[dict]) -> float | None:
        values = [float(row[key]) for row in source if row.get(key) is not None]
        return None if not values else float(np.median(values))

    return {
        "label": METHOD_LABELS[method],
        "trials": len(rows),
        "strict_bottom_successes": len(successful),
        "strict_bottom_rate": len(successful) / max(1, len(rows)),
        "offline_capture_successes": sum(
            bool(row.get("offline_capture_validated")) for row in rows
        ),
        "hard_stops": sum(bool(row.get("hard_stop")) for row in rows),
        "failure_categories": dict(Counter(_failure_category(row) for row in rows)),
        "median_success_bottom_time_s": median("time_to_bottom_s", successful),
        "median_success_xy_path_mm": median("actual_xy_path_mm", successful),
        "median_success_wear_mj": median("wear_proxy_mj", successful),
        "control_direction_error_median_deg": (
            None if not errors else float(np.median(errors))
        ),
        "control_direction_error_p90_deg": (
            None if not errors else float(np.quantile(errors, 0.90))
        ),
        "control_direction_error_gt90_count": sum(error > 90.0 for error in errors),
        "post_capture_recoveries": sum(
            int(row.get("funnel_recovery_count", 0)) for row in rows
        ),
        "terminal_rejections": sum(
            int(row.get("terminal_rejection_count", 0)) for row in rows
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


def _paired_summary(records: list[dict], left: str, right: str) -> dict:
    pairs = [
        (
            _strict_bottom_success(record["methods"][left]),
            _strict_bottom_success(record["methods"][right]),
        )
        for record in records
    ]
    common = [
        record for record, pair in zip(records, pairs) if pair == (True, True)
    ]
    return {
        "both_success": sum(pair == (True, True) for pair in pairs),
        "left_only_success": sum(pair == (True, False) for pair in pairs),
        "right_only_success": sum(pair == (False, True) for pair in pairs),
        "both_failed": sum(pair == (False, False) for pair in pairs),
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
    }


def _summarize_size(size: str, records: list[dict]) -> dict:
    methods = {method: _method_summary(records, method) for method in METHODS}
    adapted = methods["fine_tuned"]
    oracle = methods["oracle"]
    oracle_gate = bool(oracle["strict_bottom_rate"] >= MIN_ORACLE_BOTTOM_RATE)
    adapted_task_gate = bool(
        adapted["strict_bottom_rate"] >= MIN_ADAPTED_BOTTOM_RATE
        and adapted["hard_stops"] <= MAX_ADAPTED_HARD_STOPS
        and adapted["control_direction_error_gt90_count"]
        <= MAX_ADAPTED_GT90_CONTROL_ERRORS
    )
    gate = oracle_gate and adapted_task_gate
    zero_task_gate = bool(
        methods["zero_shot"]["strict_bottom_rate"] >= MIN_ADAPTED_BOTTOM_RATE
        and methods["zero_shot"]["hard_stops"] == 0
        and methods["zero_shot"]["control_direction_error_gt90_count"] == 0
    )
    cal_task_gate = bool(
        methods["cal_only"]["strict_bottom_rate"] >= MIN_ADAPTED_BOTTOM_RATE
        and methods["cal_only"]["hard_stops"] == 0
        and methods["cal_only"]["control_direction_error_gt90_count"] == 0
    )
    zero_gate = oracle_gate and zero_task_gate
    cal_gate = oracle_gate and cal_task_gate
    if not oracle_gate:
        interpretation = "oracle_execution_feasibility_gate_failed"
    elif gate and not zero_gate and not cal_gate:
        interpretation = "direction_head_finetuning_required_to_pass_this_pilot_gate"
    elif gate:
        interpretation = "fine_tuning_passed_but_necessity_not_identified"
    else:
        interpretation = "direction_head_finetuning_not_sufficient_to_pass_this_pilot_gate"
    return {
        "size": size,
        "scale_contract": _size_contract(size),
        "methods": methods,
        "paired": {
            "fine_tuned_vs_zero_shot": _paired_summary(
                records, "fine_tuned", "zero_shot"
            ),
            "fine_tuned_vs_cal_only": _paired_summary(
                records, "fine_tuned", "cal_only"
            ),
            "fine_tuned_vs_outward": _paired_summary(
                records, "fine_tuned", "outward"
            ),
        },
        "fine_tuned_full_loop_gate_pass": gate,
        "oracle_execution_feasibility_gate_pass": oracle_gate,
        "fine_tuned_task_metrics_gate_pass": adapted_task_gate,
        "zero_shot_same_gate_pass": zero_gate,
        "cal_only_same_gate_pass": cal_gate,
        "zero_shot_task_metrics_gate_pass": zero_task_gate,
        "cal_only_task_metrics_gate_pass": cal_task_gate,
        "adaptation_interpretation": interpretation,
        "gate_contract": {
            "oracle_bottom_rate_min": MIN_ORACLE_BOTTOM_RATE,
            "adapted_bottom_rate_min": MIN_ADAPTED_BOTTOM_RATE,
            "adapted_hard_stops_max": MAX_ADAPTED_HARD_STOPS,
            "adapted_gt90_control_errors_max": MAX_ADAPTED_GT90_CONTROL_ERRORS,
            "frozen_before_rollout": True,
        },
    }


def _write_csv(output: Path, report: dict) -> None:
    with (output / "ppt_comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "size",
                "method",
                "offline_audited_strict_CAD_bottom_success",
                "trials",
                "success_rate",
                "median_success_time_s",
                "median_success_path_mm",
                "hard_stops",
                "control_error_median_deg",
                "control_error_p90_deg",
                "R3_recoveries",
            ]
        )
        for size in report["sizes"]:
            for method in METHODS:
                item = report["sizes"][size]["methods"][method]
                writer.writerow(
                    [
                        size.upper(),
                        METHOD_LABELS[method],
                        item["strict_bottom_successes"],
                        item["trials"],
                        f"{item['strict_bottom_rate']:.6f}",
                        item["median_success_bottom_time_s"],
                        item["median_success_xy_path_mm"],
                        item["hard_stops"],
                        item["control_direction_error_median_deg"],
                        item["control_direction_error_p90_deg"],
                        item["post_capture_recoveries"],
                    ]
                )


def _write_figure(output: Path, report: dict) -> None:
    colors = {
        "zero_shot": "#9ca3af",
        "cal_only": "#60a5fa",
        "fine_tuned": "#10b981",
        "outward": "#f59e0b",
        "oracle": "#8b5cf6",
    }
    sizes = list(report["sizes"])
    fig, axes_value = plt.subplots(
        1, len(sizes), figsize=(6.0 * len(sizes), 4.6), sharey=True
    )
    axes = np.atleast_1d(axes_value)
    for axis, size in zip(axes, sizes):
        values = [
            report["sizes"][size]["methods"][method]["strict_bottom_rate"]
            for method in METHODS
        ]
        x = np.arange(len(METHODS))
        bars = axis.bar(x, values, color=[colors[method] for method in METHODS])
        trials = report["sizes"][size]["methods"]["fine_tuned"]["trials"]
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.025,
                f"{int(round(value * trials))}/{trials}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.axhline(
            MIN_ADAPTED_BOTTOM_RATE,
            color="#dc2626",
            ls="--",
            lw=1.3,
            label="FT bottom sub-gate 0.80",
        )
        axis.axhline(
            MIN_ORACLE_BOTTOM_RATE,
            color="#7c3aed",
            ls=":",
            lw=1.3,
            label="oracle feasibility 0.90",
        )
        size_report = report["sizes"][size]
        axis.set_title(
            f"{size.upper()} · FT task "
            f"{'PASS' if size_report['fine_tuned_task_metrics_gate_pass'] else 'FAIL'}"
            " · oracle "
            f"{'PASS' if size_report['oracle_execution_feasibility_gate_pass'] else 'FAIL'}"
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["zero-shot", "cal-only", "head FT", "outward", "oracle"],
            rotation=18,
            ha="right",
        )
        axis.set_ylim(0.0, 1.12)
        axis.grid(axis="y", alpha=0.25)
        adapted = size_report["methods"]["fine_tuned"]
        axis.text(
            0.02,
            0.02,
            f"FT hard={adapted['hard_stops']}  >90deg={adapted['control_direction_error_gt90_count']}",
            transform=axis.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        axis.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0].set_ylabel("offline-audited strict CAD-bottom rate")
    fig.suptitle(
        "Fresh cross-size S3 pilot · 130 s · zero extension · shared R3 funnel",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output / "cross_size_s3_full_loop_pilot.png", dpi=200)
    plt.close(fig)


def _write_summary(output: Path, report: dict) -> None:
    def metric(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Fresh cross-size S3 untouched full-loop pilot",
        "",
        "All five methods share the same paired poses, 130 s zero-extension budget,",
        "tip-RCC/R3 post-capture funnel, and CAD-bottom terminal verifier. The four",
        "non-oracle controllers use sensor-only capture handoff; straight-to-center",
        "is a privileged execution diagnostic. Strict success is offline-audited",
        "with true-center/contact-tag geometry and is not an online terminal signal.",
        "The evaluation set was generated only after model and gate freeze.",
        f"Pre-registered gate: oracle >= {MIN_ORACLE_BOTTOM_RATE:.2f}; fine-tuned >= {MIN_ADAPTED_BOTTOM_RATE:.2f};",
        f"fine-tuned hard stops <= {MAX_ADAPTED_HARD_STOPS}; >90 deg control errors <= {MAX_ADAPTED_GT90_CONTROL_ERRORS}.",
        "",
        "| size | method | offline-audited strict CAD-bottom | capture | hard stop | bottom time | path | control error med/p90 | R3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in report["sizes"]:
        for method in METHODS:
            item = report["sizes"][size]["methods"][method]
            lines.append(
                f"| {size.upper()} | {item['label']} | "
                f"{item['strict_bottom_successes']}/{item['trials']} | "
                f"{item['offline_capture_successes']}/{item['trials']} | "
                f"{item['hard_stops']} | "
                f"{metric(item['median_success_bottom_time_s'])} s | "
                f"{metric(item['median_success_xy_path_mm'])} mm | "
                f"{metric(item['control_direction_error_median_deg'])}/"
                f"{metric(item['control_direction_error_p90_deg'])} deg | "
                f"{item['post_capture_recoveries']} |"
            )
        size_report = report["sizes"][size]
        lines.extend(
            [
                "",
                f"- {size.upper()} adapted gate: **{'PASS' if size_report['fine_tuned_full_loop_gate_pass'] else 'FAIL'}**",
                f"- Evidence interpretation: `{size_report['adaptation_interpretation']}`",
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
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_ppt_notes_cn(output: Path, report: dict) -> None:
    lines = [
        "# Cross-size S3 pilot：PPT 可用结论",
        "",
        f"预注册门槛：straight-to-center 执行门成功率不低于 {MIN_ORACLE_BOTTOM_RATE:.0%}；",
        f"direction-head fine-tuned 严格闭环成功率不低于 {MIN_ADAPTED_BOTTOM_RATE:.0%}；",
        f"fine-tuned hard stop 不超过 {MAX_ADAPTED_HARD_STOPS}，执行方向误差大于 90° 的样本不超过 {MAX_ADAPTED_GT90_CONTROL_ERRORS}。",
        "所有方法共享 130 s、零扩展、R3 物理漏斗和 CAD-bottom 终止验证。",
        "四个非 oracle 控制器使用 sensor-only capture；straight-to-center 是读取真中心的 privileged execution diagnostic。",
        "严格成功还接受离线 true-center/contact-tag capture 审计，不能冒充在线 terminal signal。",
        "",
    ]
    for size, size_report in report["sizes"].items():
        lines.extend([f"## {size.upper()}（n={size_report['methods']['fine_tuned']['trials']}）", ""])
        for method in METHODS:
            item = size_report["methods"][method]
            time_value = item["median_success_bottom_time_s"]
            time_text = "n/a" if time_value is None else f"{time_value:.2f} s"
            lines.append(
                f"- {item['label']}：{item['strict_bottom_successes']}/{item['trials']} "
                f"({item['strict_bottom_rate']:.1%})，hard stop={item['hard_stops']}，"
                f"成功样本中位时间={time_text}。"
            )
        lines.extend(
            [
                f"- fine-tuned task 指标门：{'通过' if size_report['fine_tuned_task_metrics_gate_pass'] else '未通过'}；"
                f"straight-to-center 执行门：{'通过' if size_report['oracle_execution_feasibility_gate_pass'] else '未通过'}；"
                f"综合门：{'通过' if size_report['fine_tuned_full_loop_gate_pass'] else '未通过'}。",
                f"- 证据解释：`{size_report['adaptation_interpretation']}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 必须保留的边界",
            "",
            "这是全新 MuJoCo 尺寸适配 pilot，不是真机数据实验。只有当 fine-tuned 通过同一预注册门、",
            "同时 zero-shot 与 cal-only 均未通过时，才能在该仿真分布内称方向头微调是必要的。",
            "即使仿真通过，也不能推出真机数据微调的必要性；后者仍需要独立真实 F/T train/cal/test。",
            "privileged straight-to-center 是直线路径执行诊断，不是保证更优的性能上界。",
            "",
        ]
    )
    (output / "ppt_notes_cn.md").write_text("\n".join(lines), encoding="utf-8")


def _load_model_contract_hashes(size: str) -> dict:
    paths = {
        "zero_shot": ZERO_SHOT_ROOT / "ensemble_interface_contract.json",
        "cal_only": MODEL_ROOT / size / "cal_only" / "ensemble_interface_contract.json",
        "fine_tuned": MODEL_ROOT / size / "fine_tuned" / "ensemble_interface_contract.json",
    }
    return {
        method: {"path": _relative(path), "sha256": _sha256(path)}
        for method, path in paths.items()
    }


def _write_provenance(output: Path, task_contract: dict) -> None:
    source_paths = {
        "S3_runner": Path(__file__).resolve(),
        "rollout_core": ROOT / "src/render_m8_search_comparison.py",
        "funnel_guard": ROOT / "src/funnel_guard.py",
        "direction_runtime": ROOT / "src/m8_direction_runtime.py",
        "direction_architecture": ROOT / "src/train_m8_direction_models.py",
    }
    _atomic_json(
        output / "provenance.json",
        {
            "model_and_data_contracts_were_frozen_before_rollout": True,
            "source_hash_recording_time": (
                "pre-rollout task contract, per-worker trial readback, and "
                "post-rollout parent verification"
            ),
            "execution_bundle_sha256": task_contract[
                "execution_bundle_sha256_frozen_before_rollout"
            ],
            "execution_bundle": task_contract[
                "execution_bundle_frozen_before_rollout"
            ],
            "sources": {
                name: {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "datasets": {
                size: {
                    "contract_sha256": _sha256(
                        DATA_ROOT / size / "dataset_contract.json"
                    ),
                    "manifest_sha256": _sha256(
                        DATA_ROOT / size / "dataset_manifest.json"
                    ),
                }
                for size in SIZES
            },
            "models": {
                size: {
                    "model_manifest_sha256": _sha256(
                        MODEL_ROOT / size / "model_manifest.json"
                    ),
                    "cal_only_contract_sha256": _sha256(
                        MODEL_ROOT
                        / size
                        / "cal_only/ensemble_interface_contract.json"
                    ),
                    "fine_tuned_contract_sha256": _sha256(
                        MODEL_ROOT
                        / size
                        / "fine_tuned/ensemble_interface_contract.json"
                    ),
                }
                for size in SIZES
            },
            "old_S1_S2_artifacts_loaded_for_weight_or_gate_selection": False,
            "updated_utc": _utc_now(),
        },
    )


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks_by_size = {
        size: build_untouched_tasks(size, args.poses) for size in args.sizes
    }
    execution_sizes = tuple(args.sizes)
    execution_bundle = _execution_bundle(execution_sizes)
    execution_bundle_sha256 = _canonical_json_sha256(execution_bundle)
    task_contract = {
        "experiment": "cross_size_S3_fresh_adaptation_untouched_full_loop_pilot",
        "methods": list(METHODS),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "zero_budget_extension": True,
        "post_capture_funnel": asdict(SHARED_FUNNEL),
        "terminal_verifier": "strict CAD-bottom with force/wrench dwell",
        "evaluation_set_role": "untouched_test_forbidden_for_training_calibration_or_gate_selection",
        "capture_and_scoring_contract": {
            "non_oracle_handoff": "sensor-only measured-wrench cue",
            "oracle_handoff": "privileged true-center execution diagnostic",
            "strict_success_scoring": (
                "offline true-center/contact-tag capture audit plus shared "
                "CAD-bottom terminal verifier"
            ),
            "offline_scoring_available_to_controller": False,
        },
        "evaluation_seeds": EVALUATION_SEEDS,
        "tasks": {
            size: [asdict(task) for task in tasks]
            for size, tasks in tasks_by_size.items()
        },
        "model_contracts": {
            size: _load_model_contract_hashes(size) for size in args.sizes
        },
        "adaptation_data_contracts": {
            size: {
                "path": _relative(DATA_ROOT / size / "dataset_contract.json"),
                "sha256": _sha256(DATA_ROOT / size / "dataset_contract.json"),
            }
            for size in args.sizes
        },
        "gates_frozen_before_rollout": {
            "minimum_oracle_bottom_rate": MIN_ORACLE_BOTTOM_RATE,
            "minimum_adapted_bottom_rate": MIN_ADAPTED_BOTTOM_RATE,
            "maximum_adapted_hard_stops": MAX_ADAPTED_HARD_STOPS,
            "maximum_adapted_gt90_control_errors": MAX_ADAPTED_GT90_CONTROL_ERRORS,
        },
        "old_S1_S2_rollouts_loaded": False,
        "execution_bundle_frozen_before_rollout": execution_bundle,
        "execution_bundle_sha256_frozen_before_rollout": execution_bundle_sha256,
        "created_utc": _utc_now(),
    }
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_existing = {key: value for key, value in existing.items() if key != "created_utc"}
        stable_new = {key: value for key, value in task_contract.items() if key != "created_utc"}
        if stable_existing != stable_new:
            raise ValueError("existing S3 full-loop task contract differs")
        task_contract = existing
    else:
        _atomic_json(contract_path, task_contract)

    size_reports = {}
    for size in args.sizes:
        size_output = output / size
        (size_output / "trials").mkdir(parents=True, exist_ok=True)
        tasks = tasks_by_size[size]
        records = []
        pending = []
        for task in tasks:
            path = size_output / "trials" / f"{task.pose_id}.json"
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
                initargs=(
                    str(output),
                    size,
                    execution_sizes,
                    execution_bundle_sha256,
                ),
            ) as pool:
                futures = {pool.submit(_run_pose, task): task for task in pending}
                with tqdm(
                    total=len(tasks),
                    initial=len(records),
                    desc=f"{size.upper()} S3 untouched x5",
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
                        _atomic_json(
                            size_output / "progress.json",
                            {
                                "size": size,
                                "completed": len(records),
                                "total": len(tasks),
                                "elapsed_s": time.perf_counter() - started,
                                "updated_utc": _utc_now(),
                            },
                        )
                        bar.update(1)
        errors = [record for record in records if "error" in record]
        if errors:
            _atomic_json(size_output / "errors.json", {"errors": errors})
            raise RuntimeError(f"{size}: {len(errors)} S3 rollout tasks failed")
        records.sort(key=lambda record: record["task"]["pose_id"])
        if not all(
            record.get("worker_init_execution_bundle_sha256")
            == execution_bundle_sha256
            and record.get("worker_completion_execution_bundle_sha256")
            == execution_bundle_sha256
            for record in records
        ):
            raise RuntimeError(f"{size}: per-worker execution bundle drifted")
        if _canonical_json_sha256(_execution_bundle(execution_sizes)) != execution_bundle_sha256:
            raise RuntimeError(f"{size}: execution bundle drifted during rollout")
        if not all(
            record["fairness"]["three_learned_first_probe_hashes_match"]
            for record in records
        ):
            raise RuntimeError(f"{size}: learned first-probe fingerprints drifted")
        if not all(record["fairness"]["zero_budget_extension"] for record in records):
            raise RuntimeError(f"{size}: nonzero budget extension")
        if not all(
            record["fairness"]["simulator_reset_count"] == 0
            for record in records
        ):
            raise RuntimeError(f"{size}: simulator reset reported")
        size_reports[size] = _summarize_size(size, records)
        _atomic_json(
            size_output / "manifest.json",
            {
                "status": "complete",
                "size": size,
                "completed": len(records),
                "evaluation_training_use": "forbidden",
                "all_first_probe_hashes_match": True,
                "all_budget_extensions_zero": True,
                "all_simulator_resets_zero": True,
                "all_worker_execution_bundles_match": True,
                "execution_bundle_sha256": execution_bundle_sha256,
                "task_contract_sha256": _sha256(contract_path),
                "updated_utc": _utc_now(),
            },
        )

    report = {
        "experiment": task_contract["experiment"],
        "sizes": size_reports,
        "all_adapted_size_gates_pass": all(
            report["fine_tuned_full_loop_gate_pass"]
            for report in size_reports.values()
        ),
        "adaptation_necessity_supported_for_all_sizes": all(
            report["adaptation_interpretation"]
            == "direction_head_finetuning_required_to_pass_this_pilot_gate"
            for report in size_reports.values()
        ),
        "task_contract_sha256": _sha256(contract_path),
        "execution_bundle_sha256": execution_bundle_sha256,
        "claim_boundary": (
            "Fresh paired MuJoCo S3 pilot only. A fine-tuned pass shows that a small "
            "size-specific direction-head update can recover this simulated full-loop "
            "gate under the stated contract. Fine-tuning is called necessary only when "
            "both zero-shot and calibration-only fail the same frozen gate. This cannot "
            "establish that real-robot data fine-tuning is necessary: no real F/T sample, "
            "manufacturing tolerance, passivity, motor coupling, threading, or extraction "
            "is represented. The privileged straight-to-center arm is an execution "
            "gate, not a guaranteed performance upper bound: its direct path can itself "
            "hard-stop. Strict success also includes offline true-center/contact-tag "
            "capture validation; that scorer is not available to non-oracle controllers "
            "and is not a deployable terminal signal. Hyperparameter/model selection was "
            "not independently preregistered, so a future positive necessity claim would "
            "require a frozen or nested model-selection protocol."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output, report)
    _write_ppt_notes_cn(output, report)
    _write_csv(output, report)
    _write_figure(output, report)
    if _canonical_json_sha256(_execution_bundle(execution_sizes)) != execution_bundle_sha256:
        raise RuntimeError("execution bundle drifted before final report commit")
    _write_provenance(output, task_contract)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "sizes": list(args.sizes),
            "poses_per_size": args.poses,
            "methods": list(METHODS),
            "report_sha256": _sha256(output / "report.json"),
            "summary_sha256": _sha256(output / "summary.md"),
            "ppt_notes_cn_sha256": _sha256(output / "ppt_notes_cn.md"),
            "csv_sha256": _sha256(output / "ppt_comparison_table.csv"),
            "figure_sha256": _sha256(output / "cross_size_s3_full_loop_pilot.png"),
            "provenance_sha256": _sha256(output / "provenance.json"),
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fresh, untouched M10/M12 S3 full-loop pilot."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--poses", type=int, default=12)
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
