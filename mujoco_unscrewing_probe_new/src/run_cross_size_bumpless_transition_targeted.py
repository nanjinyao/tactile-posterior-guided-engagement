from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_s4b_base_anchored_adapter import (
    BaseAnchoredS4bDirectionAdapter,
)
from src.cross_size_commissioning import commissioning_for_size
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import run_method
from src.run_cross_size_closed_loop_audit import CrossSizeTask
from src.run_cross_size_s4_full_loop_evaluation import (
    ACTIVE_TIME_BUDGET_S,
    BASE_R3_FUNNEL,
    RUNTIME_METHOD,
    S4_ADAPTER_ROOT,
    S4_HYBRID_RESIDUAL_COVERAGE,
    S4_MODEL_ROOT,
    ZERO_SHOT_ROOT,
    OutputExecutionLock,
    _execution_bundle,
    _method_availability_for_size,
    _size_contract,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/current/diagnostics/cross_size_s4b_full_loop_development_v4"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/current/diagnostics/"
    "cross_size_bumpless_transition_targeted_development"
)

# Frozen from the v4 failure-mechanism audit before this intervention.  These
# ten arms exercise the four transition classes while retaining paired controls.
TARGET_MATRIX: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "m10",
        "m10_s3test_011",
        ("s4_fine_tuned", "zero_shot", "cal_only"),
    ),
    (
        "m12",
        "m12_s3test_003",
        (
            "s4_fine_tuned",
            "pure_outward",
            "privileged_straight_to_center",
        ),
    ),
    (
        "m12",
        "m12_s3test_010",
        ("s4_fine_tuned", "cal_only", "privileged_straight_to_center"),
    ),
    (
        "m12",
        "m12_s3test_001",
        ("privileged_straight_to_center",),
    ),
)

LEARNED_METHODS = {"zero_shot", "cal_only", "s4_fine_tuned"}
SOURCE_FIELDS = (
    "result",
    "bottom_reached",
    "terminal_verification_mode",
    "hard_stop",
    "hard_stop_reason",
    "funnel_abort_reason",
    "funnel_recovery_count",
    "capture_mode",
    "total_active_time_s",
    "max_raw_physical_force_n",
    "max_raw_physical_torque_mnm",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _stable_contract(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key not in {"created_utc", "contract_sha256"}
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _trial_path(source: Path, size: str, pose_id: str) -> Path:
    return source / size / "trials" / f"{pose_id}.json"


def _load_matrix(source: Path) -> tuple[list[dict], dict[str, CrossSizeTask]]:
    source = source.resolve()
    if "untouched" in str(source).lower():
        raise ValueError("untouched source is forbidden")
    report_path = source / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("phase") != "development":
        raise ValueError("targeted transition diagnosis requires development source")

    rows: list[dict] = []
    tasks: dict[str, CrossSizeTask] = {}
    for size, pose_id, methods in TARGET_MATRIX:
        path = _trial_path(source, size, pose_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("phase") != "development":
            raise ValueError(f"{pose_id}: non-development source trial")
        task = CrossSizeTask(**payload["task"])
        if task.size != size or task.pose_id != pose_id:
            raise ValueError(f"{pose_id}: frozen task identity drift")
        tasks[pose_id] = task
        for method in methods:
            source_method = payload["methods"][method]
            rows.append(
                {
                    "size": size,
                    "pose_id": pose_id,
                    "method": method,
                    "source_v4": {
                        key: source_method.get(key) for key in SOURCE_FIELDS
                    },
                    "source_trial": _file_record(path),
                }
            )
    if len(rows) != 10:
        raise RuntimeError("targeted transition matrix must contain exactly 10 arms")
    return rows, tasks


def build_contract(source: Path = DEFAULT_SOURCE) -> tuple[dict, dict[str, CrossSizeTask]]:
    source = source.resolve()
    rows, tasks = _load_matrix(source)
    sizes = ("m10", "m12")
    availability = {
        size: _method_availability_for_size(size) for size in sizes
    }
    for row in rows:
        if not availability[row["size"]][row["method"]]["available"]:
            raise FileNotFoundError(
                f"{row['size']} {row['method']} unavailable: "
                f"{availability[row['size']][row['method']]['reason']}"
            )
    execution_bundle = _execution_bundle(
        "development",
        sizes,
        availability,
    )
    contract = {
        "experiment": "cross_size_bumpless_transition_targeted_development",
        "phase": "development",
        "source": str(source),
        "source_report": _file_record(source / "report.json"),
        "target_matrix": rows,
        "target_arm_count": len(rows),
        "pose_ids": [pose_id for _, pose_id, _ in TARGET_MATRIX],
        "methods_by_pose": {
            pose_id: list(methods)
            for _, pose_id, methods in TARGET_MATRIX
        },
        "mechanisms": {
            "terminal_CAD_set_entry": (
                "atomic admittance/held-command rebase at measured pose using "
                "the exact just-used RPY command; unchanged terminal verifier"
            ),
            "post_funnel_recovery": (
                "atomic measured-pose rebase, exit stale interpolation, restart "
                "descent from measured z with fraction zero"
            ),
            "oracle_endpoint_settle": (
                "oracle-only measured-pose rebase and smooth remaining center "
                "residual before unchanged center-tolerance dwell"
            ),
            "capture_release_to_insertion": (
                "measured-pose/held-command rebase, low-force release, sensor-only "
                "sustained requalification, smooth pose/orientation handoff; named "
                "safe abort when the cue is not sustained"
            ),
        },
        "fairness_and_safety": {
            "same_transition_and_physical_funnel_for_all_exercised_methods": True,
            "non_oracle_true_center_available": False,
            "non_oracle_offline_contact_tags_available": False,
            "oracle_true_center_use_limited_to_declared_destination": True,
            "raw_hard_wrench_thresholds": list(AdmittanceConfig().hard_wrench),
            "raw_hard_wrench_thresholds_relaxed": False,
            "soft_wrench_thresholds_changed": False,
            "capture_thresholds_changed": False,
            "terminal_verifier_thresholds_changed": False,
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        },
        "telemetry_contract": {
            "one_trace_csv_per_arm": True,
            "trace_contains_measured_six_axis_wrench": True,
            "trace_contains_raw_physical_six_axis_wrench": True,
            "trace_contains_filtered_six_axis_wrench": True,
            "transition_event_on_first_post_rebase_step": True,
            "structured_transition_telemetry_in_result": True,
            "post_release_requalification_telemetry_in_result": True,
            "oracle_endpoint_settle_telemetry_in_result": True,
            "exclusive_output_execution_lock_required": True,
            "run_id_recorded_per_arm_and_report": True,
        },
        "execution_bundle": execution_bundle,
        "execution_bundle_sha256": _canonical_sha256(execution_bundle),
        "execution_files": {
            "targeted_runner": _file_record(Path(__file__)),
            "policy_and_fsm": _file_record(
                ROOT / "src/render_m8_search_comparison.py"
            ),
            "admittance_rollout": _file_record(
                ROOT / "src/mujoco_admittance_rollout.py"
            ),
            "commissioning": _file_record(
                ROOT / "src/cross_size_commissioning.py"
            ),
            "full_loop_interface": _file_record(
                ROOT / "src/run_cross_size_s4_full_loop_evaluation.py"
            ),
        },
        "untouched_tasks_or_seeds_read": False,
        "full_development_campaign_launched": False,
        "claim_boundary": (
            "Ten preselected development arms from v4 only. This is a transition-"
            "mechanism regression and cannot be used as untouched or confirmatory "
            "evidence. A successful arm establishes neither universal closure nor "
            "real-robot validity."
        ),
        "created_utc": _utc_now(),
    }
    contract["contract_sha256"] = _canonical_sha256(_stable_contract(contract))
    return contract, tasks


def _runtime(size: str, method: str):
    if method == "s4_fine_tuned":
        return BaseAnchoredS4bDirectionAdapter(size, S4_ADAPTER_ROOT)
    if method == "zero_shot":
        return M8DirectionEnsemble(ZERO_SHOT_ROOT)
    if method == "cal_only":
        return M8DirectionEnsemble(S4_MODEL_ROOT / size / "cal_only")
    return None


def _run_arm(
    output: Path,
    contract_sha256: str,
    run_id: str,
    task: CrossSizeTask,
    method: str,
    source_snapshot: dict,
) -> dict:
    size_contract = _size_contract(task.size)
    commissioned = commissioning_for_size(
        task.size,
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
    runtime_method = RUNTIME_METHOD[method]
    trace_path = output / "traces" / task.size / f"{task.pose_id}__{method}.csv"
    runtime = _runtime(task.size, method)
    result = asdict(
        run_method(
            runtime_method,
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
            hybrid_residual_coverage_config=(
                S4_HYBRID_RESIDUAL_COVERAGE
                if runtime_method == "hybrid"
                else None
            ),
            active_time_budget_s=ACTIVE_TIME_BUDGET_S,
            trace_output_path=trace_path,
            continue_to_bottom=True,
            wrench_observation=WrenchObservationConfig(
                seed=task.sensor_noise_seed,
                delay_s=task.sensor_delay_s,
            ),
            admittance_config=commissioned.admittance,
            observation_label=(
                f"bumpless targeted development {task.pose_id} {method}"
            ),
            spec_path=size_contract["spec_path"],
            model_path=size_contract["model_path"],
            spiral_pitch_m=float(size_contract["spiral_pitch_m"]),
            xy_speed_m_s=float(size_contract["xy_speed_m_s"]),
            cad_conditioned_insertion=True,
            conditioned_on_ambiguous_top_contact=True,
            funnel_guard=commissioned.funnel_guard,
            oracle_transport_soft_escape_recovery=(
                commissioned.oracle_transport_soft_escape_recovery
            ),
            oracle_center_tolerance_capture_fraction=(
                commissioned.oracle_center_tolerance_capture_fraction
            ),
        )
    )
    record = {
        "phase": "development",
        "task": asdict(task),
        "method": method,
        "source_v4": source_snapshot,
        "bumpless_transition_result": result,
        "strict_bottom_success": _strict_bottom_success(result),
        "trace_csv": str(trace_path.relative_to(ROOT)),
        "task_contract_sha256": contract_sha256,
        "run_id": run_id,
        "untouched_tasks_or_seeds_read": False,
        "created_utc": _utc_now(),
    }
    _atomic_json(
        output / "trials" / task.size / task.pose_id / f"{method}.json",
        record,
    )
    return record


def _summarize(records: Iterable[dict]) -> dict:
    rows = list(records)
    return {
        "arms": len(rows),
        "strict_bottom_successes": sum(
            bool(row["strict_bottom_success"]) for row in rows
        ),
        "hard_stops": sum(
            bool(row["bumpless_transition_result"]["hard_stop"])
            for row in rows
        ),
        "named_safe_aborts": sum(
            row["bumpless_transition_result"].get("funnel_abort_reason")
            == "post_release_capture_requalification_failed"
            for row in rows
        ),
        "arms_with_transition_telemetry": sum(
            bool(
                row["bumpless_transition_result"].get(
                    "transition_telemetry_records"
                )
            )
            for row in rows
        ),
        "arms_with_post_release_requalification": sum(
            row["bumpless_transition_result"].get("post_release_requalified")
            is not None
            for row in rows
        ),
        "raw_hard_wrench_thresholds_relaxed": False,
        "soft_wrench_thresholds_changed": False,
        "terminal_verifier_thresholds_changed": False,
    }


def freeze_contract(output: Path, source: Path) -> dict:
    contract, _ = build_contract(source)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    _atomic_json(output / "task_contract.json", contract)
    return contract


def execute(
    output: Path,
    source: Path,
    *,
    recover_stale_lock: bool = False,
) -> dict:
    output = output.resolve()
    frozen_path = output / "task_contract.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("freeze task_contract.json before --execute")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    live, tasks = build_contract(source)
    if _stable_contract(frozen) != _stable_contract(live):
        raise RuntimeError("targeted contract or execution bundle drifted after freeze")
    if frozen.get("contract_sha256") != _canonical_sha256(
        _stable_contract(frozen)
    ):
        raise RuntimeError("frozen targeted contract hash mismatch")
    if (output / "report.json").exists() or (output / "trials").exists():
        raise FileExistsError("refusing to resume or overwrite targeted execution")

    execution_lock = OutputExecutionLock.acquire(
        output,
        phase="development",
        task_contract_sha256=frozen["contract_sha256"],
        execution_bundle_sha256=frozen["execution_bundle_sha256"],
        recover_stale=recover_stale_lock,
    )
    run_id = execution_lock.run_id
    try:
        torch.set_num_threads(1)
        source_by_arm = {
            (row["size"], row["pose_id"], row["method"]): row["source_v4"]
            for row in frozen["target_matrix"]
        }
        records: list[dict] = []
        for size, pose_id, methods in TARGET_MATRIX:
            task = tasks[pose_id]
            for method in methods:
                records.append(
                    _run_arm(
                        output,
                        frozen["contract_sha256"],
                        run_id,
                        task,
                        method,
                        source_by_arm[(size, pose_id, method)],
                    )
                )
        report = {
            "experiment": frozen["experiment"],
            "phase": "development",
            "run_id": run_id,
            "summary": _summarize(records),
            "records": records,
            "task_contract_sha256": frozen["contract_sha256"],
            "execution_bundle_sha256": frozen["execution_bundle_sha256"],
            "untouched_tasks_or_seeds_read": False,
            "claim_boundary": frozen["claim_boundary"],
            "created_utc": _utc_now(),
        }
        _atomic_json(output / "report.json", report)
        return report
    finally:
        execution_lock.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or execute the ten-arm development-only bumpless-transition "
            "mechanism regression. Default is read-only contract validation."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze-contract", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "replace a dead persistent execution lock only after explicit review"
        ),
    )
    args = parser.parse_args()
    if args.recover_stale_lock and not args.execute:
        parser.error("--recover-stale-lock requires --execute")
    return args


def main() -> int:
    args = parse_args()
    if args.freeze_contract:
        contract = freeze_contract(args.output, args.source)
        print(
            json.dumps(
                {
                    "contract": str(args.output / "task_contract.json"),
                    "contract_sha256": contract["contract_sha256"],
                    "target_arm_count": contract["target_arm_count"],
                },
                indent=2,
            )
        )
        return 0
    if args.execute:
        report = execute(
            args.output,
            args.source,
            recover_stale_lock=bool(args.recover_stale_lock),
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    contract, _ = build_contract(args.source)
    print(
        json.dumps(
            {
                "mode": "read_only_contract_validation",
                "contract_sha256": contract["contract_sha256"],
                "target_arm_count": contract["target_arm_count"],
                "pose_ids": contract["pose_ids"],
                "methods_by_pose": contract["methods_by_pose"],
                "untouched_tasks_or_seeds_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
