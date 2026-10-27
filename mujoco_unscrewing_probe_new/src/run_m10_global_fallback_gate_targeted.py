from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_commissioning import commissioning_for_size
from src.funnel_guard import FunnelGuardConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
from src.run_cross_size_commissioning_v2_targeted import (
    _atomic_json,
    _file_record,
    _sha256,
    _utc_now,
    _validate_execution_files,
)
from src.run_cross_size_closed_loop_audit import CrossSizeTask
from src.run_cross_size_s4_full_loop_evaluation import (
    ACTIVE_TIME_BUDGET_S,
    S4_HYBRID_RESIDUAL_COVERAGE,
    _ensemble_execution_files,
    _size_contract,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4b_full_loop_development_v3"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/diagnostics/"
    "cross_size_m10_global_fallback_gate_targeted"
)
MODEL_ROOT = ROOT / "outputs/current/models/cross_size_s4b_state_direction_heads/m10"
POSE_ID = "m10_s3test_010"
METHODS = ("cal_only", "raw_s4b_fine_tuned")
MODEL_VARIANT = {
    "cal_only": "cal_only",
    "raw_s4b_fine_tuned": "fine_tuned",
}
SOURCE_METHOD = {
    "cal_only": "cal_only",
    "raw_s4b_fine_tuned": "s4_fine_tuned",
}
BASE_R3_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)


_OUTPUT: Path | None = None
_CONTRACT_SHA256: str | None = None


def _source_trial_path(source: Path) -> Path:
    return source / "m10" / "trials" / f"{POSE_ID}.json"


def _source_snapshot(source: Path, method: str) -> dict:
    trial = json.loads(_source_trial_path(source).read_text(encoding="utf-8"))
    row = trial["methods"][SOURCE_METHOD[method]]
    return {
        key: row.get(key)
        for key in (
            "result",
            "bottom_reached",
            "offline_capture_validated",
            "hard_stop",
            "hard_stop_reason",
            "total_active_time_s",
            "direction_error_deg",
            "global_outward_fallback_used",
            "capture_direction_consistency_gate_enabled",
            "candidate_attempts",
            "sensor_progress_candidate_count",
            "terminal_verification_mode",
        )
    }


def _execution_sources() -> dict:
    names = (
        "cartesian_admittance.py",
        "capture_yaw_assist.py",
        "cross_size_commissioning.py",
        "entropy_grid_baseline.py",
        "funnel_guard.py",
        "m8_direction_runtime.py",
        "mujoco_admittance_rollout.py",
        "posterior_search_plan.py",
        "render_m8_dynamic_videos.py",
        "render_m8_sampling_videos.py",
        "render_m8_search_comparison.py",
        "terminal_completion_verifier.py",
        "train_m8_direction_models.py",
        "wrench_observation.py",
    )
    return {
        name: _file_record(ROOT / "src" / name)
        for name in names
    }


def build_contract(source: Path) -> tuple[dict, CrossSizeTask]:
    source = source.resolve()
    if "untouched" in source.name.lower():
        raise ValueError("untouched source is forbidden")
    report_path = source / "report.json"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("phase") != "development":
        raise ValueError("global-fallback diagnostic may read development only")
    trial_path = _source_trial_path(source)
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if trial.get("phase") != "development":
        raise ValueError("targeted source trial is not development")
    task = CrossSizeTask(**trial["task"])
    if task.pose_id != POSE_ID or task.size != "m10":
        raise ValueError("targeted task identity drift")
    for method in METHODS:
        source_row = trial["methods"][SOURCE_METHOD[method]]
        if _strict_bottom_success(source_row):
            raise ValueError(f"{method}: source was not a failed full-loop case")
        if not source_row.get("global_outward_fallback_used", False):
            raise ValueError(f"{method}: source did not exercise global fallback")

    model_contracts = {}
    for method in METHODS:
        model_path = MODEL_ROOT / MODEL_VARIANT[method]
        interface = json.loads(
            (model_path / "ensemble_interface_contract.json").read_text(
                encoding="utf-8"
            )
        )
        adaptation = interface["adaptation_contract"]
        if adaptation.get("S3_full_loop_samples_seen") != 0:
            raise ValueError(f"{method}: S3 full-loop leakage")
        if adaptation.get("future_S4_full_loop_samples_seen") != 0:
            raise ValueError(f"{method}: future full-loop leakage")
        if adaptation.get("variant") != MODEL_VARIANT[method]:
            raise ValueError(f"{method}: model variant drift")
        model_contracts[method] = _ensemble_execution_files(model_path)

    commissioned = commissioning_for_size(
        "m10",
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    if commissioned.admittance.hard_wrench != AdmittanceConfig().hard_wrench:
        raise RuntimeError("raw hard-wrench thresholds changed")
    size_contract = _size_contract("m10")
    contract = {
        "experiment": "m10_global_fallback_direction_gate_retirement_targeted",
        "phase": "development",
        "size": "m10",
        "pose_id": POSE_ID,
        "methods": list(METHODS),
        "task": asdict(task),
        "source": str(source),
        "source_failure_mechanism": (
            "cal-only and raw S4b FT both used global outward fallback after a "
            "reverse learned posterior, but the initial learned-direction "
            "capture gate remained active and rejected opposite-side global capture"
        ),
        "intervention": (
            "permanently retire the initial learned-direction capture gate before "
            "global outward fallback; retain the existing persistent sensor depth/"
            "low-wrench capture gates and shared physical funnel"
        ),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "fallback_budget_extension_s": 0.0,
        "base_anchored_adapter_used": False,
        "raw_fine_tuned_runtime": (
            "M8DirectionEnsemble loaded directly from the S4b fine_tuned variant"
        ),
        "true_center_available_to_controller": False,
        "raw_hard_wrench_thresholds": list(AdmittanceConfig().hard_wrench),
        "raw_hard_wrench_thresholds_relaxed": False,
        "soft_wrench_thresholds_changed": False,
        "capture_thresholds_changed": False,
        "terminal_verifier_thresholds_changed": False,
        "untouched_tasks_or_seeds_read": False,
        "commissioning": asdict(commissioned),
        "execution_files": {
            "runner": _file_record(Path(__file__)),
            "sources": _execution_sources(),
            "source_report": _file_record(report_path),
            "source_trial": _file_record(trial_path),
            "cad_spec": _file_record(Path(size_contract["spec_path"])),
            "mujoco_model": _file_record(Path(size_contract["model_path"])),
            "models": model_contracts,
        },
        "claim_boundary": (
            "One preselected failed M10 development pose and two already-selected "
            "model variants. This is a controller-semantics regression only, not "
            "untouched or confirmatory evidence."
        ),
        "created_utc": _utc_now(),
    }
    return contract, task


def _worker_init(output: str, contract_sha256: str) -> None:
    global _OUTPUT, _CONTRACT_SHA256
    torch.set_num_threads(1)
    _OUTPUT = Path(output)
    _CONTRACT_SHA256 = contract_sha256


def _run_variant(method: str, task_dict: dict, source_value: str) -> dict:
    if _OUTPUT is None or _CONTRACT_SHA256 is None:
        raise RuntimeError("targeted global-fallback worker is not initialized")
    if method not in METHODS:
        raise ValueError(method)
    task = CrossSizeTask(**task_dict)
    if task.pose_id != POSE_ID or task.size != "m10":
        raise RuntimeError("targeted task drift")
    size_contract = _size_contract("m10")
    commissioned = commissioning_for_size(
        "m10",
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    runtime = M8DirectionEnsemble(MODEL_ROOT / MODEL_VARIANT[method])
    result = asdict(
        run_method(
            "hybrid",
            initial_xy_m=np.asarray([task.x_m, task.y_m], dtype=float),
            direction_runtime=runtime,
            visual_sigma_m=float(size_contract["visual_sigma_m"]),
            visual_bound_m=float(size_contract["visual_bound_m"]),
            posterior_coverage=0.99,
            posterior_seed=task.posterior_seed,
            max_transport_m=float(size_contract["max_transport_m"]),
            baseline_phase_rad=task.baseline_phase_rad,
            baseline_max_radius_m=float(size_contract["baseline_max_radius_m"]),
            target_down_force_n=task.target_down_force_n,
            friction_slide=task.friction_slide,
            contact_timeconst_s=task.contact_timeconst_s,
            global_outward_fallback=True,
            hybrid_residual_coverage_config=S4_HYBRID_RESIDUAL_COVERAGE,
            active_time_budget_s=ACTIVE_TIME_BUDGET_S,
            continue_to_bottom=True,
            wrench_observation=WrenchObservationConfig(
                seed=task.sensor_noise_seed,
                delay_s=task.sensor_delay_s,
            ),
            admittance_config=commissioned.admittance,
            observation_label=(
                f"M10 global-fallback gate retirement {method} development"
            ),
            spec_path=size_contract["spec_path"],
            model_path=size_contract["model_path"],
            spiral_pitch_m=float(size_contract["spiral_pitch_m"]),
            xy_speed_m_s=float(size_contract["xy_speed_m_s"]),
            cad_conditioned_insertion=True,
            conditioned_on_ambiguous_top_contact=True,
            funnel_guard=commissioned.funnel_guard,
            # Explicitly no base-anchored adapter and no true-center interface.
        )
    )
    source = Path(source_value)
    record = {
        "phase": "development",
        "method": method,
        "task": asdict(task),
        "source_development_v3": _source_snapshot(source, method),
        "gate_retirement_regression": result,
        "base_anchored_adapter_used": False,
        "true_center_available_to_controller": False,
        "task_contract_sha256": _CONTRACT_SHA256,
        "untouched_tasks_or_seeds_read": False,
        "created_utc": _utc_now(),
    }
    _atomic_json(_OUTPUT / "trials" / f"{method}.json", record)
    return record


def _summarize(records: list[dict]) -> dict:
    result = {}
    for record in records:
        method = record["method"]
        source = record["source_development_v3"]
        current = record["gate_retirement_regression"]
        retired_time = current["capture_direction_gate_retired_active_time_s"]
        fallback_time = current["global_outward_fallback_started_active_time_s"]
        result[method] = {
            "source_strict_bottom_success": _strict_bottom_success(source),
            "regression_strict_bottom_success": _strict_bottom_success(current),
            "source_result": source["result"],
            "regression_result": current["result"],
            "hard_stop": current["hard_stop"],
            "global_outward_fallback_used": current[
                "global_outward_fallback_used"
            ],
            "gate_initially_enabled": current[
                "capture_direction_consistency_gate_enabled"
            ],
            "gate_retired": current[
                "capture_direction_gate_retired_for_global_fallback"
            ],
            "gate_active_at_outcome": current[
                "capture_direction_gate_active_at_outcome"
            ],
            "gate_retired_active_time_s": retired_time,
            "gate_retirement_reason": current[
                "capture_direction_gate_retirement_reason"
            ],
            "fallback_started_active_time_s": fallback_time,
            "gate_retired_before_or_at_fallback_start": bool(
                retired_time is not None
                and fallback_time is not None
                and float(retired_time) <= float(fallback_time) + 1.0e-12
            ),
            "candidate_attempts": current["candidate_attempts"],
            "sensor_progress_candidate_count": current[
                "sensor_progress_candidate_count"
            ],
            "terminal_verification_mode": current[
                "terminal_verification_mode"
            ],
            "total_active_time_s": current["total_active_time_s"],
        }
    return result


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    contract, task = build_contract(source)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_existing = json.loads(
            json.dumps(
                {k: v for k, v in existing.items() if k != "created_utc"},
                sort_keys=True,
            )
        )
        stable_candidate = json.loads(
            json.dumps(
                {k: v for k, v in contract.items() if k != "created_utc"},
                sort_keys=True,
            )
        )
        if stable_existing != stable_candidate:
            raise ValueError("existing targeted contract differs; use a new output")
        contract = existing
    else:
        _atomic_json(contract_path, contract)
    contract_sha256 = _sha256(contract_path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "frozen_contract_only",
                    "output": str(output),
                    "pose_id": POSE_ID,
                    "methods": list(METHODS),
                    "task_contract_sha256": contract_sha256,
                },
                indent=2,
            )
        )
        return 0

    _validate_execution_files(contract["execution_files"])
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(METHODS)),
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(output), contract_sha256),
    ) as pool:
        future_by_method = {
            pool.submit(_run_variant, method, asdict(task), str(source)): method
            for method in METHODS
        }
        with tqdm(total=len(METHODS), desc="M10 fallback gate", unit="method") as bar:
            for future in as_completed(future_by_method):
                method = future_by_method[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    _atomic_json(
                        output / "trials" / f"{method}.error.json",
                        {"method": method, "task": asdict(task), "error": repr(exc)},
                    )
                    raise
                finally:
                    bar.update(1)
    records.sort(key=lambda row: METHODS.index(row["method"]))
    methods = _summarize(records)
    all_retired = all(
        row["gate_retired"] and row["gate_retired_before_or_at_fallback_start"]
        for row in methods.values()
    )
    report = {
        "experiment": contract["experiment"],
        "phase": "development",
        "pose_id": POSE_ID,
        "methods": methods,
        "all_exercised_global_fallback_gates_retired_before_motion": all_retired,
        "base_anchored_adapter_used": False,
        "true_center_available_to_controller": False,
        "raw_hard_wrench_thresholds_relaxed": False,
        "soft_wrench_thresholds_changed": False,
        "capture_thresholds_changed": False,
        "terminal_verifier_thresholds_changed": False,
        "untouched_tasks_or_seeds_read": False,
        "claim_boundary": contract["claim_boundary"],
        "task_contract_sha256": contract_sha256,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "phase": "development",
            "pose_id": POSE_ID,
            "methods": list(METHODS),
            "gate_retirement_telemetry": {
                method: {
                    "retired": row["gate_retired"],
                    "active_time_s": row["gate_retired_active_time_s"],
                    "reason": row["gate_retirement_reason"],
                    "fallback_started_active_time_s": row[
                        "fallback_started_active_time_s"
                    ],
                }
                for method, row in methods.items()
            },
            "all_exercised_global_fallback_gates_retired_before_motion": (
                all_retired
            ),
            "base_anchored_adapter_used": False,
            "raw_hard_wrench_thresholds_relaxed": False,
            "soft_wrench_thresholds_changed": False,
            "capture_thresholds_changed": False,
            "terminal_verifier_thresholds_changed": False,
            "untouched_tasks_or_seeds_read": False,
            "task_contract_sha256": contract_sha256,
            "report_sha256": _sha256(output / "report.json"),
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and regress M10 development pose010 global-fallback capture "
            "gate semantics for cal-only and raw S4b FT; untouched is forbidden."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
