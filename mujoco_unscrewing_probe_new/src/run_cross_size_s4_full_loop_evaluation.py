from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_s4b_base_anchored_adapter import (
    BaseAnchoredS4bDirectionAdapter,
)
from src.cross_size_commissioning import commissioning_for_size
from src.funnel_guard import FunnelGuardConfig
from src.m8_direction_runtime import M8DirectionEnsemble
from src.render_m8_search_comparison import (
    HybridResidualCoverageConfig,
    RunResult,
    posterior_directional_corridor_path,
    run_method,
)
from src.run_cross_size_closed_loop_audit import (
    CrossSizeTask,
    _size_contract as _base_size_contract,
    build_tasks as _build_new_tasks,
)
from src.run_m8_strict_complete_loop_audit import (
    _failure_category,
    _strict_bottom_success,
)
from src.wrench_observation import WrenchObservationConfig


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("development", "untouched")
SIZES = ("m10", "m12")
DEFAULT_OUTPUTS = {
    "development": (
        ROOT
        / "outputs/current/diagnostics/cross_size_s4_full_loop_development"
    ),
    "untouched": (
        ROOT
        / "outputs/current/evaluations/cross_size_s4_full_loop_untouched"
    ),
}
S3_DEVELOPMENT_SOURCE = (
    ROOT
    / "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3"
    / "task_contract.json"
)
S4_DATA_ROOT = ROOT / "outputs/current/datasets/cross_size_s4b_adaptation"
S4_MODEL_ROOT = (
    ROOT / "outputs/current/models/cross_size_s4b_state_direction_heads"
)
S4_ADAPTER_ROOT = (
    ROOT / "outputs/current/models/cross_size_s4b_base_anchored_adapter"
)
ZERO_SHOT_ROOT = ROOT / "outputs/current/models/p1v3_m8_direction_models"

METHODS = (
    "zero_shot",
    "cal_only",
    "s4_fine_tuned",
    "pure_outward",
    "privileged_straight_to_center",
)
LEARNED_METHODS = ("zero_shot", "cal_only", "s4_fine_tuned")
REQUIRED_METHODS = (
    "zero_shot",
    "s4_fine_tuned",
    "pure_outward",
    "privileged_straight_to_center",
)
METHOD_LABELS = {
    "zero_shot": "M8 zero-shot + P+ hybrid",
    "cal_only": "S4b calibration-only + P+ hybrid",
    "s4_fine_tuned": "S4b base-anchored residual FT + P+ hybrid",
    "pure_outward": "pure outward",
    "privileged_straight_to_center": "privileged straight-to-center",
}
RUNTIME_METHOD = {
    "zero_shot": "hybrid",
    "cal_only": "hybrid",
    "s4_fine_tuned": "hybrid",
    # These two are deliberately not estimator variants.  Outward retains its
    # baseline identity and the privileged arm remains the execution gate.
    "pure_outward": "baseline",
    "privileged_straight_to_center": "oracle",
}

ACTIVE_TIME_BUDGET_S = 130.0
DEVELOPMENT_SEEDS = {"m10": 20260941, "m12": 20260942}
UNTOUCHED_SEEDS = {"m10": 20261141, "m12": 20261142}
BASE_R3_FUNNEL = FunnelGuardConfig(
    enabled=True,
    max_recoveries=3,
    retry_on_terminal_rejection=True,
)

# Frozen acceptance gates.  The oracle establishes execution feasibility;
# safety and direction gates are applied to the S4 fine-tuned arm.
MIN_ORACLE_BOTTOM_RATE = 0.90
MIN_FINE_TUNED_BOTTOM_RATE = 0.80
MAX_FINE_TUNED_HARD_STOPS = 0
# The safety gate applies only when a non-zero posterior-directed shortcut was
# actually dispatched.  Latent posterior errors remain reported separately.
MAX_FINE_TUNED_GT90_EXECUTED_DIRECTED_ERRORS = 0
# Backward-compatible symbol for downstream imports; it no longer denotes the
# latent error count over abstained posterior means.
MAX_FINE_TUNED_GT90_CONTROL_ERRORS = (
    MAX_FINE_TUNED_GT90_EXECUTED_DIRECTED_ERRORS
)

# Development-v2 exposed a scheduling deadlock on the outer M10/M12 annulus:
# a shortcut was rejected unless shortcut + *complete global spiral* + terminal
# fit, even when the complete global spiral itself could not fit 130 s.  This
# finite corridor is an explicitly conditional alternative.  It is frozen from
# measured posterior quantities only and retains the raw hard-wrench limits.
S4_HYBRID_RESIDUAL_COVERAGE = HybridResidualCoverageConfig(
    enabled=True,
    angular_half_width_deg=12.0,
    lane_spacing_capture_fraction=1.0,
    # Development engineering allowance, not an empirically validated bound.
    # It is frozen before the targeted rerun and must not be described as a
    # confidence interval on physical tracking error.
    tracking_error_allowance_m=0.00015,
    corridor_path_speed_m_s=0.0015,
    measured_tip_outer_servo_enabled=True,
    outer_servo_gain=1.5,
    outer_servo_max_correction_m=0.0010,
    outer_servo_max_correction_speed_m_s=0.0020,
    outer_servo_abort_tracking_error_m=0.0012,
    outer_servo_abort_dwell_s=0.15,
    minimum_direction_quality=0.10,
    maximum_reverse_probability=0.35,
    transition_overhead_s=1.0,
)


_RUNTIMES: dict[str, M8DirectionEnsemble] = {}
_ACTIVE_METHODS: tuple[str, ...] = ()
_OUTPUT: Path | None = None
_SIZE: str | None = None
_PHASE: str | None = None
_EXECUTION_SIZES: tuple[str, ...] = ()
_METHOD_AVAILABILITY: dict[str, dict[str, dict]] = {}
_EXECUTION_BUNDLE_SHA256: str | None = None
_TASK_CONTRACT_SHA256: str | None = None
_RUN_ID: str | None = None

EXECUTION_LOCK_FILENAME = ".s4_execution.lock.json"
EXECUTION_LOCK_GUARD_FILENAME = ".s4_execution.lock.guard"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _pid_is_alive(pid: int) -> bool:
    """Return whether a recorded local PID still denotes a live process."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_lock_pids(payload: dict) -> tuple[int, ...]:
    values = [payload.get("pid")]
    worker_pids = payload.get("worker_pids", [])
    if isinstance(worker_pids, list):
        values.extend(worker_pids)
    return tuple(
        sorted(
            {
                int(value)
                for value in values
                if isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            }
        )
    )


def _write_exclusive_json(path: Path, payload: dict) -> None:
    """Create a JSON file exactly once; never replace an existing lock."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    try:
        serialized = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(serialized):
            written = os.write(fd, serialized[offset:])
            if written <= 0:
                raise OSError("short write while creating S4 execution lock")
            offset += written
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


class OutputExecutionLock:
    """Exclusive, crash-auditable execution ownership for one output root.

    The persistent JSON lock is acquired with ``O_EXCL``.  A separate advisory
    guard serializes acquisition, stale recovery, ownership updates, and
    release.  The guard is automatically dropped by the kernel after a crash;
    the JSON lock intentionally remains so recovery must be explicit.
    """

    def __init__(self, path: Path, guard_path: Path, payload: dict) -> None:
        self.path = path
        self.guard_path = guard_path
        self.payload = payload
        self.run_id = str(payload["run_id"])
        self._released = False

    @staticmethod
    def _read_payload(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"unreadable S4 execution lock at {path}; "
                "explicit stale recovery is required"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"malformed S4 execution lock at {path}")
        return value

    @staticmethod
    def _acquire_guard(guard_path: Path) -> int:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(guard_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                f"another process is acquiring or updating the S4 execution "
                f"lock in {guard_path.parent}"
            ) from exc
        return fd

    @staticmethod
    def _release_guard(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @classmethod
    def acquire(
        cls,
        output: Path,
        *,
        phase: str,
        task_contract_sha256: str,
        execution_bundle_sha256: str,
        recover_stale: bool = False,
        run_id: str | None = None,
    ) -> "OutputExecutionLock":
        output.mkdir(parents=True, exist_ok=True)
        path = output / EXECUTION_LOCK_FILENAME
        guard_path = output / EXECUTION_LOCK_GUARD_FILENAME
        guard_fd = cls._acquire_guard(guard_path)
        try:
            payload = {
                "run_id": run_id or uuid.uuid4().hex,
                "pid": os.getpid(),
                "worker_pids": [],
                "phase": phase,
                "task_contract_sha256": task_contract_sha256,
                "execution_bundle_sha256": execution_bundle_sha256,
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
            }
            try:
                _write_exclusive_json(path, payload)
            except FileExistsError as exc:
                try:
                    existing = cls._read_payload(path)
                except RuntimeError:
                    if not recover_stale:
                        raise
                    # The guard proves no compliant writer is still filling the
                    # file.  An unreadable persistent lock can therefore be
                    # replaced only through the explicit recovery path.
                    path.unlink()
                    _write_exclusive_json(path, payload)
                    return cls(path, guard_path, payload)
                live_pids = tuple(
                    pid
                    for pid in _recorded_lock_pids(existing)
                    if _pid_is_alive(pid)
                )
                if live_pids:
                    raise RuntimeError(
                        f"live S4 execution lock at {path}; recorded live PIDs: "
                        f"{list(live_pids)}"
                    ) from exc
                if not recover_stale:
                    raise RuntimeError(
                        f"stale S4 execution lock at {path}; rerun with "
                        "--recover-stale-lock only after verifying the prior "
                        "execution is dead"
                    ) from exc
                # The advisory guard excludes every compliant acquirer while
                # the stale persistent record is removed and recreated.
                path.unlink()
                _write_exclusive_json(path, payload)
            return cls(path, guard_path, payload)
        finally:
            cls._release_guard(guard_fd)

    def update_worker_pids(self, worker_pids: tuple[int, ...]) -> None:
        if self._released:
            raise RuntimeError("cannot update a released S4 execution lock")
        guard_fd = self._acquire_guard(self.guard_path)
        try:
            existing = self._read_payload(self.path)
            if existing.get("run_id") != self.run_id:
                raise RuntimeError(
                    "S4 execution lock ownership changed before worker PID update"
                )
            updated = {
                **existing,
                "worker_pids": list(
                    sorted(
                        {
                            int(pid)
                            for pid in worker_pids
                            if isinstance(pid, int)
                            and not isinstance(pid, bool)
                            and pid > 0
                        }
                    )
                ),
                "updated_utc": _utc_now(),
            }
            _atomic_json(self.path, updated)
            self.payload = updated
        finally:
            self._release_guard(guard_fd)

    def release(self) -> None:
        if self._released:
            return
        guard_fd = self._acquire_guard(self.guard_path)
        try:
            if self.path.exists():
                existing = self._read_payload(self.path)
                if existing.get("run_id") != self.run_id:
                    raise RuntimeError(
                        "refusing to release an S4 execution lock owned by "
                        "another run"
                    )
                self.path.unlink()
            self._released = True
        finally:
            self._release_guard(guard_fd)


def _executor_worker_pids(pool: ProcessPoolExecutor) -> tuple[int, ...]:
    """Snapshot worker PIDs for crash-safe stale-lock validation."""

    processes = getattr(pool, "_processes", {})
    if not isinstance(processes, dict):
        return ()
    return tuple(
        sorted(
            {
                int(process.pid)
                for process in processes.values()
                if getattr(process, "pid", None) is not None
            }
        )
    )


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


def output_for_phase(phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(phase)
    return DEFAULT_OUTPUTS[phase]


def _residual_schedule_bound(size: str) -> dict:
    """Conservative L*=0 corridor schedule; no true task center is used."""

    contract = _size_contract(size)
    config = S4_HYBRID_RESIDUAL_COVERAGE
    capture_radius_m = float(contract["capture_radius_m"])
    robust_radius_m = capture_radius_m - config.tracking_error_allowance_m
    if robust_radius_m <= 0.0:
        raise ValueError("residual tracking allowance consumes capture radius")
    visual_bound_m = float(contract["visual_bound_m"])
    lane_spacing_m = (
        2.0 * robust_radius_m * config.lane_spacing_capture_fraction
    )
    arc, _, offsets = posterior_directional_corridor_path(
        np.zeros(2),
        np.array([1.0, 0.0]),
        axial_start_m=0.0,
        axial_end_m=visual_bound_m,
        half_width_m=visual_bound_m
        * np.sin(np.deg2rad(config.angular_half_width_deg)),
        lane_spacing_m=lane_spacing_m,
    )
    speed_m_s = (
        float(contract["xy_speed_m_s"])
        if config.corridor_path_speed_m_s <= 0.0
        else config.corridor_path_speed_m_s
    )
    spec = json.loads(Path(contract["spec_path"]).read_text(encoding="utf-8"))
    funnel = contract["commissioning"]["funnel_guard"]
    final_command_z_m = -float(spec["socket_depth_m"]) + 0.0010
    descent_s = max(
        6.0,
        abs(final_command_z_m - 0.00030)
        / float(funnel["descent_speed_m_s"]),
    )
    recovery_s = int(funnel["max_recoveries"]) * (
        float(funnel["recovery_unload_s"])
        + float(funnel["recovery_retract_s"])
        + float(funnel["recovery_retouch_s"])
    )
    terminal_s = max(
        12.0,
        float(funnel["capture_force_release_s"])
        + float(funnel["insertion_handoff_s"])
        + descent_s
        + recovery_s
        + 0.25,
    )
    fingerprint_s = 5.80
    zero_shortcut_s = 0.65
    corridor_s = float(arc[-1]) / speed_m_s + config.transition_overhead_s
    total_s = fingerprint_s + zero_shortcut_s + corridor_s + terminal_s
    return {
        "assumed_shortcut_distance_mm": 0.0,
        "lane_count": int(len(offsets)),
        "lane_spacing_mm": 1000.0 * lane_spacing_m,
        "corridor_path_mm": 1000.0 * float(arc[-1]),
        "corridor_speed_mm_s": 1000.0 * speed_m_s,
        "fingerprint_s": fingerprint_s,
        "zero_shortcut_s": zero_shortcut_s,
        "corridor_plus_transition_s": corridor_s,
        "terminal_reserve_s": terminal_s,
        "total_required_s": total_s,
        "fits_130_s": bool(total_s <= ACTIVE_TIME_BUDGET_S),
        "bound_uses_true_center": False,
    }


def phase_protocol(phase: str, sizes: tuple[str, ...], poses: int) -> dict:
    """Return the non-file protocol fields used by contract tests and runner."""

    if phase not in PHASES:
        raise ValueError(phase)
    if not sizes or any(size not in SIZES for size in sizes):
        raise ValueError("invalid S4 evaluation sizes")
    if poses <= 0:
        raise ValueError("poses must be positive")
    is_untouched = phase == "untouched"
    return {
        "phase": phase,
        "default_output": _relative(output_for_phase(phase)),
        "task_seed_source": (
            "fresh_S4_untouched_Sobol_seeds"
            if is_untouched
            else "reused_S3_v3_tasks_for_development_diagnostics"
        ),
        "evaluation_seeds": (
            UNTOUCHED_SEEDS if is_untouched else DEVELOPMENT_SEEDS
        ),
        "evaluation_use": (
            "forbidden_for_training_calibration_model_selection_controller_tuning_or_gate_selection"
            if is_untouched
            else "development_diagnostics_and_tuning_allowed; no confirmatory claim"
        ),
        "untouched_after_freeze": is_untouched,
        "poses_per_size": poses,
        "sizes": list(sizes),
        "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
        "fallback_budget_extension_s": 0.0,
        "hybrid_residual_coverage": asdict(S4_HYBRID_RESIDUAL_COVERAGE),
        "hybrid_residual_claim_boundary": (
            "conditional predicted-direction-sector coverage only; no true-center "
            "read, no hard-limit relaxation, and no guarantee of tracking, "
            "sensor capture handoff, or terminal completion"
        ),
        "hybrid_residual_worst_case_schedule_by_size": {
            size: _residual_schedule_bound(size) for size in sizes
        },
        "required_methods": list(REQUIRED_METHODS),
        "learned_controller_mapping": {
            method: RUNTIME_METHOD[method] for method in LEARNED_METHODS
        },
        "baseline_controller_mapping": {
            "pure_outward": RUNTIME_METHOD["pure_outward"],
            "privileged_straight_to_center": RUNTIME_METHOD[
                "privileged_straight_to_center"
            ],
        },
        "gates_frozen_before_rollout": {
            "minimum_oracle_bottom_rate": MIN_ORACLE_BOTTOM_RATE,
            "minimum_S4_fine_tuned_bottom_rate": MIN_FINE_TUNED_BOTTOM_RATE,
            "maximum_S4_fine_tuned_hard_stops": MAX_FINE_TUNED_HARD_STOPS,
            "maximum_S4_fine_tuned_gt90_executed_directed_errors": (
                MAX_FINE_TUNED_GT90_EXECUTED_DIRECTED_ERRORS
            ),
            "maximum_S4_fine_tuned_gt90_control_errors": (
                MAX_FINE_TUNED_GT90_CONTROL_ERRORS
            ),
            "gt90_gate_semantics": (
                "control-direction error is gated only when the first "
                "posterior-directed shortcut was dispatched with nonzero length; "
                "latent posterior errors for abstentions remain diagnostic"
            ),
        },
    }


def build_phase_tasks(phase: str, size: str, count: int) -> list[CrossSizeTask]:
    if phase not in PHASES or size not in SIZES or count <= 0:
        raise ValueError("invalid S4 task request")
    if phase == "development":
        source = json.loads(S3_DEVELOPMENT_SOURCE.read_text(encoding="utf-8"))
        source_tasks = source.get("tasks", {}).get(size, [])
        if source.get("evaluation_seeds", {}).get(size) != DEVELOPMENT_SEEDS[size]:
            raise ValueError("S3 v3 development seed contract drift")
        if count > len(source_tasks):
            raise ValueError(
                f"{size}: development phase has only {len(source_tasks)} frozen S3 tasks"
            )
        return [CrossSizeTask(**row) for row in source_tasks[:count]]

    tasks = _build_new_tasks(size, count, UNTOUCHED_SEEDS[size])
    return [
        replace(task, pose_id=f"{size}_s4_untouched_{index:03d}")
        for index, task in enumerate(tasks)
    ]


def _size_contract(size: str) -> dict:
    base = dict(_base_size_contract(size))
    commissioned = commissioning_for_size(
        size, base_funnel=BASE_R3_FUNNEL, base_admittance=AdmittanceConfig()
    )
    if tuple(commissioned.admittance.hard_wrench) != tuple(
        AdmittanceConfig().hard_wrench
    ):
        raise RuntimeError("commissioning relaxed the raw hard-wrench threshold")
    base.update(
        {
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "fallback_budget_extension_s": 0.0,
            "commissioning": asdict(commissioned),
            "raw_hard_wrench_thresholds_unchanged_from_M8": True,
        }
    )
    return base


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
                "normalization": _file_record(
                    checkpoint.parent / "normalization.npz"
                ),
                "metrics": _file_record(checkpoint.parent / "metrics.json"),
            }
        )
    return members


def _variant_legality(model_root: Path, size: str, variant: str) -> tuple[bool, str]:
    contract_path = model_root / "ensemble_interface_contract.json"
    if not contract_path.is_file():
        return False, "ensemble_interface_contract_missing"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        adaptation = contract["adaptation_contract"]
    except (KeyError, json.JSONDecodeError):
        return False, "malformed_S4_adaptation_contract"
    if adaptation.get("size") != size or adaptation.get("variant") != variant:
        return False, "size_or_variant_contract_mismatch"
    if int(adaptation.get("S3_full_loop_samples_seen", -1)) != 0:
        return False, "S3_full_loop_data_leakage"
    if int(adaptation.get("future_S4_full_loop_samples_seen", -1)) != 0:
        return False, "S4_full_loop_data_leakage"
    if variant == "cal_only":
        if (
            adaptation.get("weight_update_role") != "none"
            or adaptation.get("trainable_modules") not in ([], None)
            or not adaptation.get("all_checkpoint_weights_frozen", False)
            or adaptation.get("temperature_and_APS_role") != "calibration"
        ):
            return False, "cal_only_contract_allows_weight_updates"
        for checkpoint_value in contract.get("checkpoints", []):
            checkpoint = Path(checkpoint_value)
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            metrics_path = checkpoint.parent / "metrics.json"
            if not checkpoint.is_file() or not metrics_path.is_file():
                return False, "cal_only_seed_artifact_missing"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not metrics.get("checkpoint_byte_identical_to_base", False):
                return False, "cal_only_checkpoint_not_equal_to_base"
            base_checkpoint = Path(str(metrics.get("base_checkpoint", "")))
            if not base_checkpoint.is_absolute():
                base_checkpoint = ROOT / base_checkpoint
            if not base_checkpoint.is_file():
                return False, "cal_only_base_checkpoint_missing"
            checkpoint_sha256 = _sha256(checkpoint)
            base_sha256 = _sha256(base_checkpoint)
            if (
                checkpoint_sha256 != base_sha256
                or metrics.get("checkpoint_sha256") != checkpoint_sha256
                or metrics.get("base_checkpoint_sha256") != base_sha256
            ):
                return False, "cal_only_checkpoint_bytes_or_hashes_drifted"
    elif (
        adaptation.get("weight_update_role") != "train"
        or set(adaptation.get("trainable_modules", []))
        != {"state_head", "direction_head"}
        or adaptation.get("model_selection_role") != "validation"
        or adaptation.get("temperature_and_APS_role") != "calibration"
    ):
        return False, "fine_tuned_role_contract_mismatch"
    return True, "available"


def _adapter_legality(size: str) -> tuple[bool, str]:
    """Validate selection/evaluation separation for the S4b runtime adapter."""

    contract_path = S4_ADAPTER_ROOT / "adapter_contract.json"
    if not contract_path.is_file():
        return False, "base_anchored_adapter_contract_missing"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        access = contract["data_access_contract"]
        roots = contract["model_roots"]
        provenance = contract["provenance"]
    except (KeyError, json.JSONDecodeError):
        return False, "malformed_base_anchored_adapter_contract"
    if contract.get("status") != "train_validation_selected_frozen":
        return False, "base_anchored_adapter_not_frozen"
    if size not in contract.get("sizes", []):
        return False, "base_anchored_adapter_size_missing"
    if contract.get("selector_may_launch_full_loop") is not False:
        return False, "adapter_selector_evaluation_separation_missing"
    if contract.get("full_loop_execution_authorized") is not False:
        return False, "adapter_selector_improperly_authorizes_execution"
    if not contract.get(
        "eligible_for_separately_frozen_full_loop_evaluation", False
    ):
        return False, "adapter_not_eligible_for_separate_frozen_evaluation"
    forbidden_counts = (
        "calibration_examples_read",
        "calibration_labels_read",
        "full_loop_rollouts_read",
        "full_loop_pose_labels_read",
        "untouched_rollouts_read",
    )
    if any(int(access.get(key, -1)) != 0 for key in forbidden_counts):
        return False, "adapter_selection_data_leakage"
    if roots.get("base") != _relative(ZERO_SHOT_ROOT):
        return False, "adapter_base_parent_mismatch"
    expected_fine = _relative(S4_MODEL_ROOT / size / "fine_tuned")
    if roots.get("fine_tuned_by_size", {}).get(size) != expected_fine:
        return False, "adapter_fine_tuned_parent_mismatch"
    source_record = provenance.get("adapter_source", {})
    source_path = ROOT / str(source_record.get("path", ""))
    if (
        not source_path.is_file()
        or source_record.get("sha256") != _sha256(source_path)
    ):
        return False, "adapter_source_provenance_drift"
    for path_key, hash_key in (
        ("selection_report", "selection_report_sha256"),
        ("selection_readme", "selection_readme_sha256"),
    ):
        path = ROOT / str(contract.get(path_key, ""))
        if not path.is_file() or contract.get(hash_key) != _sha256(path):
            return False, f"adapter_{path_key}_provenance_drift"
    return True, "available"


def _method_availability_for_size(size: str) -> dict[str, dict]:
    fine_root = S4_MODEL_ROOT / size / "fine_tuned"
    cal_root = S4_MODEL_ROOT / size / "cal_only"
    fine_ok, fine_reason = _variant_legality(fine_root, size, "fine_tuned")
    adapter_ok, adapter_reason = _adapter_legality(size)
    cal_ok, cal_reason = _variant_legality(cal_root, size, "cal_only")
    zero_ok = (ZERO_SHOT_ROOT / "ensemble_interface_contract.json").is_file()
    return {
        "zero_shot": {
            "available": zero_ok,
            "reason": "available" if zero_ok else "base_ensemble_missing",
            "model_root": _relative(ZERO_SHOT_ROOT),
        },
        "cal_only": {
            "available": cal_ok,
            "reason": cal_reason,
            "model_root": _relative(cal_root),
        },
        "s4_fine_tuned": {
            "available": fine_ok and adapter_ok,
            "reason": fine_reason if not fine_ok else adapter_reason,
            "model_root": _relative(S4_ADAPTER_ROOT),
            "fine_tuned_parent_root": _relative(fine_root),
            "base_parent_root": _relative(ZERO_SHOT_ROOT),
        },
        "pure_outward": {
            "available": True,
            "reason": "nonlearned_baseline",
            "model_root": None,
        },
        "privileged_straight_to_center": {
            "available": True,
            "reason": "privileged_execution_diagnostic",
            "model_root": None,
        },
    }


def _model_root(size: str, method: str) -> Path:
    if method == "zero_shot":
        return ZERO_SHOT_ROOT
    if method == "cal_only":
        return S4_MODEL_ROOT / size / "cal_only"
    if method == "s4_fine_tuned":
        return S4_MODEL_ROOT / size / "fine_tuned"
    raise ValueError(method)


def _execution_bundle(
    phase: str,
    sizes: tuple[str, ...],
    availability: dict[str, dict[str, dict]],
) -> dict:
    source_paths = {
        str(path.relative_to(ROOT)): path
        for path in sorted((ROOT / "src").rglob("*.py"))
    }
    bundle: dict = {
        "phase": phase,
        "sources": {
            name: _file_record(path) for name, path in source_paths.items()
        },
        "sizes": {},
    }
    if phase == "development":
        bundle["development_task_source"] = _file_record(S3_DEVELOPMENT_SOURCE)
    for size in sizes:
        contract = _size_contract(size)
        size_bundle = {
            "cad_spec": _file_record(Path(contract["spec_path"])),
            "mujoco_model": _file_record(Path(contract["model_path"])),
            "adaptation_data_contract": _file_record(
                S4_DATA_ROOT / size / "dataset_contract.json"
            ),
            "adaptation_data_manifest": _file_record(
                S4_DATA_ROOT / size / "dataset_manifest.json"
            ),
            "model_manifest": _file_record(
                S4_MODEL_ROOT / size / "model_manifest.json"
            ),
            "ensembles": {},
        }
        for method in LEARNED_METHODS:
            if availability[size][method]["available"]:
                if method == "s4_fine_tuned":
                    adapter_contract = json.loads(
                        (S4_ADAPTER_ROOT / "adapter_contract.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    size_bundle["ensembles"][method] = {
                        "runtime": "base_anchored_residual_adapter",
                        "adapter_contract": _file_record(
                            S4_ADAPTER_ROOT / "adapter_contract.json"
                        ),
                        "adapter_manifest": _file_record(
                            S4_ADAPTER_ROOT / "manifest.json"
                        ),
                        "selection_report": _file_record(
                            ROOT / adapter_contract["selection_report"]
                        ),
                        "selection_readme": _file_record(
                            ROOT / adapter_contract["selection_readme"]
                        ),
                        "base_parent": _ensemble_execution_files(
                            ZERO_SHOT_ROOT
                        ),
                        "fine_tuned_parent": _ensemble_execution_files(
                            S4_MODEL_ROOT / size / "fine_tuned"
                        ),
                    }
                else:
                    size_bundle["ensembles"][method] = _ensemble_execution_files(
                        _model_root(size, method)
                    )
        bundle["sizes"][size] = size_bundle
    return bundle


def _build_task_contract(
    phase: str, sizes: tuple[str, ...], poses: int
) -> tuple[dict, dict[str, list[CrossSizeTask]]]:
    protocol = phase_protocol(phase, sizes, poses)
    tasks = {
        size: build_phase_tasks(phase, size, poses) for size in sizes
    }
    availability = {
        size: _method_availability_for_size(size) for size in sizes
    }
    for size in sizes:
        unavailable_required = [
            method
            for method in REQUIRED_METHODS
            if not availability[size][method]["available"]
        ]
        if unavailable_required:
            raise FileNotFoundError(
                f"{size}: required S4 methods unavailable: {unavailable_required}"
            )
    bundle = _execution_bundle(phase, sizes, availability)
    bundle_sha256 = _canonical_json_sha256(bundle)
    contract = {
        "experiment": f"cross_size_S4b_{phase}_full_loop_evaluation",
        "adaptation_cohort": (
            "S4b generic visual-boundary C6 with train-validation-selected "
            "base-anchored residual adapter"
        ),
        **protocol,
        "methods": list(METHODS),
        "method_availability": availability,
        "tasks": {
            size: [asdict(task) for task in size_tasks]
            for size, size_tasks in tasks.items()
        },
        "size_commissioning": {
            size: _size_contract(size)["commissioning"] for size in sizes
        },
        "shared_controller_contract": {
            "learned_estimator_variants_use_P_plus_hybrid": True,
            "S4_fine_tuned_runtime": (
                "same-fingerprint base-anchored residual posterior; "
                "p_A=p_B+alpha(p_F-p_B); conservative confidence/span; "
                "failed adapter gate abstains from directed shortcut"
            ),
            "pure_outward_is_the_declared_nonlearned_baseline_exception": True,
            "same_size_conditioned_admittance_and_R3_funnel_for_all_methods": True,
            "raw_hard_wrench_thresholds": list(
                AdmittanceConfig().hard_wrench
            ),
            "raw_hard_wrench_thresholds_relaxed": False,
            "active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "fallback_budget_extension_s": 0.0,
            "posterior_residual_coverage": asdict(
                S4_HYBRID_RESIDUAL_COVERAGE
            ),
            "posterior_residual_schedule_rule": (
                "compare shortcut+corridor+terminal against "
                "global-outward+terminal as alternative schedules; never require "
                "both to fit"
            ),
        },
        "capture_and_terminal_boundary": {
            "non_oracle_online_handoff": "sensor-only measured-wrench cue",
            "oracle_online_information": "privileged true-center execution diagnostic",
            "strict_success": (
                "offline true-center/contact-tag capture audit plus shared "
                "CAD-bottom terminal verifier"
            ),
            "offline_true_center_or_contact_tags_available_to_non_oracle": False,
            "offline_terminal_scorer_is_deployable_online_signal": False,
        },
        "necessity_rule_frozen_before_rollout": (
            "untouched only: oracle and S4b fine-tuned pass; zero-shot and legal "
            "S4 calibration-only both fail the same task gate"
        ),
        "cal_only_unavailable_policy": (
            "report unavailable and keep necessity unsupported; never substitute "
            "an S3 model or an uncalibrated alias"
        ),
        "probe_fairness_contract": {
            "same_pose_contact_sensor_nuisance": True,
            "same_conical_probe_controller_and_observation_chain": True,
            "learned_first_probe_fingerprint_hashes_must_match_when_recorded": True,
        },
        "direction_metric_contract": {
            "legacy_control_direction_error_fields": (
                "latent posterior circular-mean error over every available "
                "posterior, including abstentions"
            ),
            "latent_control_error_fields": (
                "explicit alias of the legacy all-posterior diagnostic"
            ),
            "directed_shortcut_accepted": (
                "first shortcut was dispatched and its frozen planned distance "
                "was greater than 1e-6 mm"
            ),
            "safety_gate_metric": "executed_directed_error_gt90_count",
            "adapter_last_telemetry": (
                "last base-anchored adapter inference within that method rollout; "
                "it is not asserted to be the first-posterior telemetry"
            ),
        },
        "execution_bundle_frozen_before_rollout": bundle,
        "execution_bundle_sha256_frozen_before_rollout": bundle_sha256,
        "contract_freeze_must_precede_execute": True,
        "created_utc": _utc_now(),
    }
    return contract, tasks


def _worker_init(
    output: str,
    size: str,
    phase: str,
    active_methods: tuple[str, ...],
    execution_sizes: tuple[str, ...],
    availability: dict[str, dict[str, dict]],
    expected_bundle_sha256: str,
    task_contract_sha256: str,
    run_id: str,
) -> None:
    global _RUNTIMES, _ACTIVE_METHODS, _OUTPUT, _SIZE, _PHASE
    global _EXECUTION_SIZES, _METHOD_AVAILABILITY
    global _EXECUTION_BUNDLE_SHA256, _TASK_CONTRACT_SHA256, _RUN_ID
    torch.set_num_threads(1)
    actual_bundle = _execution_bundle(phase, execution_sizes, availability)
    actual_sha256 = _canonical_json_sha256(actual_bundle)
    if actual_sha256 != expected_bundle_sha256:
        raise RuntimeError("S4 execution bundle drifted before worker initialization")
    _RUNTIMES = {}
    for method in active_methods:
        if method not in LEARNED_METHODS:
            continue
        if method == "s4_fine_tuned":
            _RUNTIMES[method] = BaseAnchoredS4bDirectionAdapter(
                size, S4_ADAPTER_ROOT
            )
        else:
            _RUNTIMES[method] = M8DirectionEnsemble(
                _model_root(size, method)
            )
    _ACTIVE_METHODS = active_methods
    _OUTPUT = Path(output)
    _SIZE = size
    _PHASE = phase
    _EXECUTION_SIZES = execution_sizes
    _METHOD_AVAILABILITY = availability
    _EXECUTION_BUNDLE_SHA256 = expected_bundle_sha256
    _TASK_CONTRACT_SHA256 = task_contract_sha256
    _RUN_ID = run_id


def _run_one_method(task: CrossSizeTask, method: str, contract: dict) -> RunResult:
    runtime_method = RUNTIME_METHOD[method]
    commissioned = commissioning_for_size(
        task.size,
        base_funnel=BASE_R3_FUNNEL,
        base_admittance=AdmittanceConfig(),
    )
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
        global_outward_fallback=True,
        hybrid_residual_coverage_config=(
            S4_HYBRID_RESIDUAL_COVERAGE
            if runtime_method == "hybrid"
            else None
        ),
        active_time_budget_s=ACTIVE_TIME_BUDGET_S,
        continue_to_bottom=True,
        wrench_observation=WrenchObservationConfig(
            seed=task.sensor_noise_seed, delay_s=task.sensor_delay_s
        ),
        admittance_config=commissioned.admittance,
        observation_label=(
            f"cross-size S4 {task.size} {method} shared observed-wrench contract"
        ),
        spec_path=contract["spec_path"],
        model_path=contract["model_path"],
        spiral_pitch_m=float(contract["spiral_pitch_m"]),
        xy_speed_m_s=float(contract["xy_speed_m_s"]),
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


def _directed_shortcut_accepted(row: dict) -> bool:
    """Whether the initial posterior direction caused nonzero physical motion.

    ``first_shortcut_executed`` records dispatch, including a legacy zero-length
    no-op.  The latter is an abstention, not an executed direction error.
    """

    explicit = row.get("directed_shortcut_accepted")
    if explicit is not None:
        return bool(explicit)
    distance_mm = row.get("first_shortcut_distance_mm")
    return bool(
        row.get("first_shortcut_executed")
        and distance_mm is not None
        and float(distance_mm) > 1.0e-6
    )


def _run_method_record(task: CrossSizeTask, method: str, contract: dict) -> dict:
    row = asdict(_run_one_method(task, method, contract))
    row["directed_shortcut_accepted"] = _directed_shortcut_accepted(row)
    if method == "s4_fine_tuned":
        runtime = _RUNTIMES.get(method)
        telemetry = (
            runtime.last_telemetry
            if isinstance(runtime, BaseAnchoredS4bDirectionAdapter)
            else None
        )
        row["adapter_last_telemetry"] = (
            None if telemetry is None else asdict(telemetry)
        )
    return row


def _run_pose(task: CrossSizeTask) -> dict:
    if not (
        _OUTPUT is not None
        and _SIZE == task.size
        and _PHASE in PHASES
        and _ACTIVE_METHODS
        and _EXECUTION_BUNDLE_SHA256
        and _TASK_CONTRACT_SHA256
        and _RUN_ID
    ):
        raise RuntimeError("S4 worker was not initialized")
    started = time.perf_counter()
    contract = _size_contract(task.size)
    methods = {
        method: _run_method_record(task, method, contract)
        for method in _ACTIVE_METHODS
    }
    learned_hashes = [
        methods[method].get("fingerprint_sha256")
        for method in _ACTIVE_METHODS
        if method in LEARNED_METHODS
    ]
    completion_bundle_sha256 = _canonical_json_sha256(
        _execution_bundle(_PHASE, _EXECUTION_SIZES, _METHOD_AVAILABILITY)
    )
    if completion_bundle_sha256 != _EXECUTION_BUNDLE_SHA256:
        raise RuntimeError("S4 execution bundle drifted during worker rollout")
    record = {
        "run_id": _RUN_ID,
        "phase": _PHASE,
        "task": asdict(task),
        "scale_contract": contract,
        "methods": methods,
        "fairness": {
            "run_id": _RUN_ID,
            "same_pose_contact_nuisance_and_sensor_seed": True,
            "same_active_time_budget_s": ACTIVE_TIME_BUDGET_S,
            "zero_budget_extension": all(
                abs(float(row.get("fallback_budget_extension_s") or 0.0))
                <= 1.0e-12
                for row in methods.values()
            ),
            "same_size_commissioned_admittance_and_R3_funnel": True,
            "raw_hard_wrench_thresholds_relaxed": False,
            "learned_first_probe_hashes": learned_hashes,
            "learned_first_probe_hashes_match": len(set(learned_hashes)) == 1,
            "learned_first_probe_hashes_all_present": all(
                value is not None for value in learned_hashes
            ),
            "same_offline_terminal_scorer": True,
            "offline_scorer_available_to_non_oracle": False,
            "simulator_reset_count": sum(
                int(row.get("simulator_reset_count", 0))
                for row in methods.values()
            ),
        },
        "task_contract_sha256": _TASK_CONTRACT_SHA256,
        "worker_init_execution_bundle_sha256": _EXECUTION_BUNDLE_SHA256,
        "worker_completion_execution_bundle_sha256": completion_bundle_sha256,
        "worker_elapsed_s": time.perf_counter() - started,
        "created_utc": _utc_now(),
    }
    _atomic_json(
        _OUTPUT / task.size / "trials" / f"{task.pose_id}.json", record
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
    latent_errors = [
        error for row in rows if (error := _control_error(row)) is not None
    ]
    executed_directed_errors = [
        error
        for row in rows
        if _directed_shortcut_accepted(row)
        and (error := _control_error(row)) is not None
    ]

    def median(key: str, source: list[dict]) -> float | None:
        values = [float(row[key]) for row in source if row.get(key) is not None]
        return None if not values else float(np.median(values))

    return {
        "available": True,
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
        # Legacy names remain latent-posterior diagnostics for compatibility.
        "control_direction_error_median_deg": (
            None if not latent_errors else float(np.median(latent_errors))
        ),
        "control_direction_error_p90_deg": (
            None
            if not latent_errors
            else float(np.quantile(latent_errors, 0.90))
        ),
        "control_direction_error_gt90_count": int(
            sum(error > 90.0 for error in latent_errors)
        ),
        "latent_control_error_median_deg": (
            None if not latent_errors else float(np.median(latent_errors))
        ),
        "latent_control_error_p90_deg": (
            None
            if not latent_errors
            else float(np.quantile(latent_errors, 0.90))
        ),
        "latent_control_error_gt90_count": int(
            sum(error > 90.0 for error in latent_errors)
        ),
        "directed_shortcut_accepted_count": sum(
            _directed_shortcut_accepted(row) for row in rows
        ),
        "executed_directed_error_median_deg": (
            None
            if not executed_directed_errors
            else float(np.median(executed_directed_errors))
        ),
        "executed_directed_error_p90_deg": (
            None
            if not executed_directed_errors
            else float(np.quantile(executed_directed_errors, 0.90))
        ),
        "executed_directed_error_gt90_count": int(
            sum(error > 90.0 for error in executed_directed_errors)
        ),
        "post_capture_recoveries": sum(
            int(row.get("funnel_recovery_count", 0)) for row in rows
        ),
        "terminal_rejections": sum(
            int(row.get("terminal_rejection_count", 0)) for row in rows
        ),
        "budget_violations": sum(
            float(row["total_active_time_s"]) > ACTIVE_TIME_BUDGET_S + 1.0e-9
            or abs(float(row.get("fallback_budget_extension_s") or 0.0))
            > 1.0e-12
            for row in rows
        ),
    }


def _task_metric_gate(method_summary: dict) -> bool:
    executed_gt90 = method_summary.get(
        "executed_directed_error_gt90_count",
        # Compatibility for callers holding a pre-schema summary.  New
        # summaries always provide the explicit executed-direction metric.
        method_summary.get("control_direction_error_gt90_count", 0),
    )
    return bool(
        method_summary.get("available")
        and method_summary["strict_bottom_rate"] >= MIN_FINE_TUNED_BOTTOM_RATE
        and method_summary["hard_stops"] <= MAX_FINE_TUNED_HARD_STOPS
        and executed_gt90 <= MAX_FINE_TUNED_GT90_EXECUTED_DIRECTED_ERRORS
    )


def _paired_summary(records: list[dict], left: str, right: str) -> dict:
    pairs = [
        (
            _strict_bottom_success(record["methods"][left]),
            _strict_bottom_success(record["methods"][right]),
        )
        for record in records
    ]
    return {
        "both_success": sum(pair == (True, True) for pair in pairs),
        "left_only_success": sum(pair == (True, False) for pair in pairs),
        "right_only_success": sum(pair == (False, True) for pair in pairs),
        "both_failed": sum(pair == (False, False) for pair in pairs),
    }


def _summarize_size(
    size: str,
    records: list[dict],
    availability: dict[str, dict],
    phase: str,
) -> dict:
    methods: dict[str, dict] = {}
    for method in METHODS:
        if availability[method]["available"]:
            methods[method] = _method_summary(records, method)
        else:
            methods[method] = {
                "available": False,
                "label": METHOD_LABELS[method],
                "reason": availability[method]["reason"],
            }
    fine_gate = _task_metric_gate(methods["s4_fine_tuned"])
    oracle_gate = bool(
        methods["privileged_straight_to_center"]["strict_bottom_rate"]
        >= MIN_ORACLE_BOTTOM_RATE
    )
    full_gate = oracle_gate and fine_gate
    zero_gate = _task_metric_gate(methods["zero_shot"])
    cal_available = bool(methods["cal_only"]["available"])
    cal_gate = _task_metric_gate(methods["cal_only"]) if cal_available else None
    necessity_numerical_condition = bool(
        full_gate and not zero_gate and cal_available and cal_gate is False
    )
    necessity_supported = bool(
        phase == "untouched" and necessity_numerical_condition
    )
    if phase == "development":
        interpretation = "development_diagnostic_only_no_confirmatory_claim"
    elif not oracle_gate:
        interpretation = "oracle_execution_feasibility_gate_failed"
    elif not fine_gate:
        interpretation = "S4_fine_tuning_not_sufficient_for_frozen_gate"
    elif not cal_available:
        interpretation = "fine_tuned_passed_but_necessity_untestable_cal_only_unavailable"
    elif zero_gate or cal_gate:
        interpretation = "fine_tuned_passed_but_necessity_not_identified"
    else:
        interpretation = "S4_fine_tuning_necessity_supported_in_this_simulation_contract"

    paired = {
        "fine_tuned_vs_zero_shot": _paired_summary(
            records, "s4_fine_tuned", "zero_shot"
        ),
        "fine_tuned_vs_pure_outward": _paired_summary(
            records, "s4_fine_tuned", "pure_outward"
        ),
    }
    if cal_available:
        paired["fine_tuned_vs_cal_only"] = _paired_summary(
            records, "s4_fine_tuned", "cal_only"
        )
    return {
        "size": size,
        "phase": phase,
        "scale_contract": _size_contract(size),
        "methods": methods,
        "paired": paired,
        "oracle_execution_feasibility_gate_pass": oracle_gate,
        "S4_fine_tuned_task_metrics_gate_pass": fine_gate,
        "S4_fine_tuned_full_loop_gate_pass": full_gate,
        "zero_shot_same_task_gate_pass": zero_gate,
        "cal_only_available": cal_available,
        "cal_only_same_task_gate_pass": cal_gate,
        "necessity_numerical_condition": necessity_numerical_condition,
        "adaptation_necessity_supported": necessity_supported,
        "interpretation": interpretation,
        "gate_contract": phase_protocol(phase, (size,), len(records))[
            "gates_frozen_before_rollout"
        ],
    }


def _write_csv(output: Path, report: dict) -> None:
    with (output / "ppt_comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phase",
                "size",
                "method",
                "availability",
                "strict_CAD_bottom_success",
                "trials",
                "success_rate",
                "median_success_time_s",
                "hard_stops",
                "control_error_median_deg",
                "control_error_p90_deg",
                "control_error_gt90_count",
                "latent_control_error_gt90_count",
                "directed_shortcut_accepted_count",
                "executed_directed_error_median_deg",
                "executed_directed_error_p90_deg",
                "executed_directed_error_gt90_count",
            ]
        )
        for size, size_report in report["sizes"].items():
            for method in METHODS:
                item = size_report["methods"][method]
                if not item["available"]:
                    writer.writerow(
                        [report["phase"], size.upper(), method, item["reason"]]
                    )
                    continue
                writer.writerow(
                    [
                        report["phase"],
                        size.upper(),
                        METHOD_LABELS[method],
                        "available",
                        item["strict_bottom_successes"],
                        item["trials"],
                        f"{item['strict_bottom_rate']:.6f}",
                        item["median_success_bottom_time_s"],
                        item["hard_stops"],
                        item["control_direction_error_median_deg"],
                        item["control_direction_error_p90_deg"],
                        item["control_direction_error_gt90_count"],
                        item["latent_control_error_gt90_count"],
                        item["directed_shortcut_accepted_count"],
                        item["executed_directed_error_median_deg"],
                        item["executed_directed_error_p90_deg"],
                        item["executed_directed_error_gt90_count"],
                    ]
                )


def _write_figure(output: Path, report: dict) -> Path:
    colors = {
        "zero_shot": "#9ca3af",
        "cal_only": "#60a5fa",
        "s4_fine_tuned": "#10b981",
        "pure_outward": "#f59e0b",
        "privileged_straight_to_center": "#8b5cf6",
    }
    sizes = list(report["sizes"])
    fig, axes_value = plt.subplots(
        1, len(sizes), figsize=(6.2 * len(sizes), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes_value)
    for axis, size in zip(axes, sizes):
        size_report = report["sizes"][size]
        values = [
            (
                size_report["methods"][method]["strict_bottom_rate"]
                if size_report["methods"][method]["available"]
                else 0.0
            )
            for method in METHODS
        ]
        x = np.arange(len(METHODS))
        bars = axis.bar(x, values, color=[colors[method] for method in METHODS])
        for method, bar, value in zip(METHODS, bars, values):
            item = size_report["methods"][method]
            if not item["available"]:
                bar.set_facecolor("white")
                bar.set_edgecolor("#64748b")
                bar.set_hatch("///")
                text = "N/A"
            else:
                text = f"{item['strict_bottom_successes']}/{item['trials']}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.025,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.axhline(
            MIN_FINE_TUNED_BOTTOM_RATE,
            color="#dc2626",
            ls="--",
            lw=1.2,
            label="FT gate 0.80",
        )
        axis.axhline(
            MIN_ORACLE_BOTTOM_RATE,
            color="#7c3aed",
            ls=":",
            lw=1.2,
            label="oracle gate 0.90",
        )
        axis.set_title(
            f"{size.upper()} · FT "
            f"{'PASS' if size_report['S4_fine_tuned_full_loop_gate_pass'] else 'FAIL'}"
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["zero", "cal", "S4 FT", "outward", "oracle"],
            rotation=18,
            ha="right",
        )
        axis.set_ylim(0.0, 1.12)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0].set_ylabel("offline-audited strict CAD-bottom rate")
    fig.suptitle(
        f"Cross-size S4b {report['phase']} · 130 s · zero extension · commissioned R3",
        fontsize=12,
    )
    fig.tight_layout()
    path = output / f"cross_size_s4_{report['phase']}_full_loop.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _write_summary(output: Path, report: dict) -> None:
    lines = [
        f"# Cross-size S4b {report['phase']} full-loop evaluation",
        "",
        "All estimator variants use the same Proposed+ hybrid controller. Pure",
        "outward remains the declared nonlearned baseline; straight-to-center is",
        "a privileged execution diagnostic. Every arm shares the size-conditioned",
        "commissioned admittance/R3 funnel, raw hard thresholds, 130 s budget, and",
        "zero extension. Strict success includes an offline scorer unavailable to",
        "the deployed non-oracle controller.",
        "",
        "| size | method | strict bottom | hard | executed >90 | latent >90 | median time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for size, size_report in report["sizes"].items():
        for method in METHODS:
            item = size_report["methods"][method]
            if not item["available"]:
                lines.append(
                    f"| {size.upper()} | {METHOD_LABELS[method]} | unavailable: "
                    f"{item['reason']} | — | — | — | — |"
                )
                continue
            time_value = item["median_success_bottom_time_s"]
            lines.append(
                f"| {size.upper()} | {METHOD_LABELS[method]} | "
                f"{item['strict_bottom_successes']}/{item['trials']} | "
                f"{item['hard_stops']} | "
                f"{item['executed_directed_error_gt90_count']} | "
                f"{item['latent_control_error_gt90_count']} | "
                f"{'n/a' if time_value is None else f'{time_value:.2f} s'} |"
            )
        lines.extend(
            [
                "",
                f"- Frozen FT full-loop gate: **{'PASS' if size_report['S4_fine_tuned_full_loop_gate_pass'] else 'FAIL'}**",
                f"- Necessity supported: **{'YES' if size_report['adaptation_necessity_supported'] else 'NO'}**",
                f"- Interpretation: `{size_report['interpretation']}`",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", report["claim_boundary"], ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_ppt_notes(output: Path, report: dict) -> None:
    cn = [
        f"# Cross-size S4b {report['phase']}：PPT 注释",
        "",
        "所有 estimator 变体统一使用 Proposed+ hybrid；纯 outward 是非学习基线，",
        "straight-to-center 是读取真中心的 privileged 执行诊断。所有方法共享尺寸",
        "commissioning 后的 admittance/R3 漏斗、未放宽的 raw hard threshold、",
        "130 s 与零预算扩展。严格成功中的离线 true-center/contact-tag 审计不可用于在线控制。",
        "",
    ]
    en = [
        f"# Cross-size S4b {report['phase']}: slide notes",
        "",
        "All estimator variants use Proposed+ hybrid. Pure outward is the nonlearned",
        "baseline and straight-to-center is a privileged execution diagnostic. Every",
        "method shares size-commissioned admittance/R3, unchanged raw hard thresholds,",
        "a 130 s budget, and zero extension. Offline true-center/contact-tag scoring is",
        "not available to the online controller.",
        "",
    ]
    for size, size_report in report["sizes"].items():
        cn.extend([f"## {size.upper()}", ""])
        en.extend([f"## {size.upper()}", ""])
        for method in METHODS:
            item = size_report["methods"][method]
            if not item["available"]:
                cn.append(f"- {METHOD_LABELS[method]}：不可用（{item['reason']}）。")
                en.append(f"- {METHOD_LABELS[method]}: unavailable ({item['reason']}).")
            else:
                value = (
                    f"{item['strict_bottom_successes']}/{item['trials']} "
                    f"({item['strict_bottom_rate']:.1%}), hard={item['hard_stops']}, "
                    f"executed>90deg={item['executed_directed_error_gt90_count']}, "
                    f"latent>90deg={item['latent_control_error_gt90_count']}"
                )
                cn.append(f"- {METHOD_LABELS[method]}：{value}。")
                en.append(f"- {METHOD_LABELS[method]}: {value}.")
        cn.extend(
            [
                f"- 结论：`{size_report['interpretation']}`。",
                "",
            ]
        )
        en.extend(
            [
                f"- Interpretation: `{size_report['interpretation']}`.",
                "",
            ]
        )
    if report["phase"] == "development":
        cn.append("该阶段允许调参，只能作为诊断，不得写成确认性结论。")
        en.append("This phase permits tuning and is diagnostic, never confirmatory.")
    else:
        cn.append("untouched 数据禁止用于任何调参；查看结果后该集合即永久退役为测试证据。")
        en.append(
            "The untouched set is forbidden for tuning and becomes permanently test-only after readout."
        )
    (output / "ppt_notes_cn.md").write_text("\n".join(cn) + "\n", encoding="utf-8")
    (output / "ppt_notes_en.md").write_text("\n".join(en) + "\n", encoding="utf-8")


def _write_provenance(
    output: Path, task_contract: dict, run_id: str
) -> None:
    _atomic_json(
        output / "provenance.json",
        {
            "run_id": run_id,
            "phase": task_contract["phase"],
            "contract_frozen_before_execute": True,
            "task_contract_sha256": _sha256(output / "task_contract.json"),
            "execution_bundle_sha256": task_contract[
                "execution_bundle_sha256_frozen_before_rollout"
            ],
            "execution_bundle": task_contract[
                "execution_bundle_frozen_before_rollout"
            ],
            "development_source": (
                _file_record(S3_DEVELOPMENT_SOURCE)
                if task_contract["phase"] == "development"
                else None
            ),
            "untouched_use_policy": task_contract["evaluation_use"],
            "offline_terminal_boundary": task_contract[
                "capture_and_terminal_boundary"
            ],
            "updated_utc": _utc_now(),
        },
    )


def _stable_contract(value: dict) -> dict:
    # The in-memory commissioning dataclasses retain tuple-valued vectors,
    # whereas their frozen JSON representation necessarily reads them back as
    # lists.  Compare the canonical JSON value rather than Python container
    # implementation details; otherwise a reviewed contract can never be
    # executed even when every physical and statistical field is unchanged.
    stable = {key: item for key, item in value.items() if key != "created_utc"}
    return json.loads(json.dumps(stable, sort_keys=True))


def _validate_resumed_record(
    record: dict,
    task: CrossSizeTask,
    phase: str,
    active_methods: tuple[str, ...],
    contract_sha256: str,
    bundle_sha256: str,
) -> None:
    if record.get("phase") != phase or record.get("task") != asdict(task):
        raise ValueError(f"{task.pose_id}: resumed S4 task/phase mismatch")
    if set(record.get("methods", {})) != set(active_methods):
        raise ValueError(f"{task.pose_id}: resumed S4 method set mismatch")
    if record.get("task_contract_sha256") != contract_sha256:
        raise ValueError(f"{task.pose_id}: resumed S4 task contract mismatch")
    if (
        record.get("worker_init_execution_bundle_sha256") != bundle_sha256
        or record.get("worker_completion_execution_bundle_sha256")
        != bundle_sha256
    ):
        raise ValueError(f"{task.pose_id}: resumed S4 execution bundle mismatch")


def _run_impl(args: argparse.Namespace) -> int:
    phase = str(args.phase)
    output = (
        output_for_phase(phase)
        if args.output is None
        else Path(args.output).resolve()
    )
    other_phase = "untouched" if phase == "development" else "development"
    if output.resolve() == output_for_phase(other_phase).resolve():
        raise ValueError("development and untouched outputs must remain distinct")
    output.mkdir(parents=True, exist_ok=True)
    sizes = tuple(args.sizes)
    candidate, tasks_by_size = _build_task_contract(phase, sizes, args.poses)
    contract_path = output / "task_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if _stable_contract(existing) != _stable_contract(candidate):
            raise ValueError(
                "existing S4 task contract differs; never mutate a frozen output"
            )
        task_contract = existing
    else:
        if args.execute:
            raise RuntimeError(
                "freeze and review the S4 task contract before --execute"
            )
        _atomic_json(contract_path, candidate)
        task_contract = candidate

    contract_sha256 = _sha256(contract_path)
    bundle_sha256 = task_contract[
        "execution_bundle_sha256_frozen_before_rollout"
    ]
    availability = task_contract["method_availability"]
    if not args.execute:
        _atomic_json(
            output / "manifest.json",
            {
                "status": "frozen_contract_only",
                "phase": phase,
                "rollout_started": False,
                "task_contract_sha256": contract_sha256,
                "execution_bundle_sha256": bundle_sha256,
                "evaluation_use": task_contract["evaluation_use"],
                "updated_utc": _utc_now(),
            },
        )
        print(json.dumps({"status": "frozen_contract_only", "output": str(output)}, indent=2))
        return 0

    execution_lock = OutputExecutionLock.acquire(
        output,
        phase=phase,
        task_contract_sha256=contract_sha256,
        execution_bundle_sha256=bundle_sha256,
        recover_stale=bool(getattr(args, "recover_stale_lock", False)),
    )
    setattr(args, "_execution_lock", execution_lock)
    run_id = execution_lock.run_id

    current_bundle = _execution_bundle(phase, sizes, availability)
    if _canonical_json_sha256(current_bundle) != bundle_sha256:
        raise RuntimeError("S4 execution bundle drifted after contract freeze")

    size_reports = {}
    for size in sizes:
        active_methods = tuple(
            method for method in METHODS if availability[size][method]["available"]
        )
        size_output = output / size
        trial_dir = size_output / "trials"
        trial_dir.mkdir(parents=True, exist_ok=True)
        tasks = tasks_by_size[size]
        records: list[dict] = []
        pending: list[CrossSizeTask] = []
        for task in tasks:
            path = trial_dir / f"{task.pose_id}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"{path}: use --resume or a new frozen output"
                    )
                record = json.loads(path.read_text(encoding="utf-8"))
                _validate_resumed_record(
                    record,
                    task,
                    phase,
                    active_methods,
                    contract_sha256,
                    bundle_sha256,
                )
                records.append(record)
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
                    phase,
                    active_methods,
                    sizes,
                    availability,
                    bundle_sha256,
                    contract_sha256,
                    run_id,
                ),
            ) as pool:
                futures = {pool.submit(_run_pose, task): task for task in pending}
                execution_lock.update_worker_pids(_executor_worker_pids(pool))
                with tqdm(
                    total=len(tasks),
                    initial=len(records),
                    desc=f"{size.upper()} S4 {phase}",
                    unit="pose",
                    dynamic_ncols=True,
                ) as bar:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            record = future.result()
                        except Exception as exc:
                            record = {
                                "run_id": run_id,
                                "phase": phase,
                                "task": asdict(task),
                                "error": repr(exc),
                                "task_contract_sha256": contract_sha256,
                                "created_utc": _utc_now(),
                            }
                            _atomic_json(
                                trial_dir / f"{task.pose_id}.json", record
                            )
                        records.append(record)
                        _atomic_json(
                            size_output / "progress.json",
                            {
                                "run_id": run_id,
                                "phase": phase,
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
            _atomic_json(
                size_output / "errors.json",
                {"run_id": run_id, "errors": errors},
            )
            raise RuntimeError(f"{size}: {len(errors)} S4 rollout tasks failed")
        records.sort(key=lambda record: record["task"]["pose_id"])
        for record, task in zip(records, sorted(tasks, key=lambda item: item.pose_id)):
            _validate_resumed_record(
                record,
                task,
                phase,
                active_methods,
                contract_sha256,
                bundle_sha256,
            )
        if not all(
            record["fairness"]["learned_first_probe_hashes_match"]
            for record in records
        ):
            raise RuntimeError(f"{size}: learned first-probe fingerprints drifted")
        if not all(record["fairness"]["zero_budget_extension"] for record in records):
            raise RuntimeError(f"{size}: nonzero fallback budget extension")
        if not all(
            record["fairness"]["simulator_reset_count"] == 0 for record in records
        ):
            raise RuntimeError(f"{size}: simulator reset violates S4 fairness")
        size_reports[size] = _summarize_size(
            size, records, availability[size], phase
        )
        _atomic_json(
            size_output / "manifest.json",
            {
                "status": "complete",
                "run_id": run_id,
                "phase": phase,
                "size": size,
                "completed": len(records),
                "active_methods": list(active_methods),
                "evaluation_use": task_contract["evaluation_use"],
                "all_learned_first_probe_hashes_match": True,
                "all_budget_extensions_zero": True,
                "all_simulator_resets_zero": True,
                "all_worker_execution_bundles_match": True,
                "execution_bundle_sha256": bundle_sha256,
                "task_contract_sha256": contract_sha256,
                "updated_utc": _utc_now(),
            },
        )

    report = {
        "run_id": run_id,
        "experiment": task_contract["experiment"],
        "phase": phase,
        "sizes": size_reports,
        "all_S4_fine_tuned_size_gates_pass": all(
            item["S4_fine_tuned_full_loop_gate_pass"]
            for item in size_reports.values()
        ),
        "adaptation_necessity_supported_for_all_sizes": all(
            item["adaptation_necessity_supported"]
            for item in size_reports.values()
        ),
        "task_contract_sha256": contract_sha256,
        "execution_bundle_sha256": bundle_sha256,
        "claim_boundary": (
            "MuJoCo-only cross-size S4 evaluation. Development results may guide "
            "tuning and can never be confirmatory. Untouched results are test-only "
            "and become permanently unavailable for later tuning after readout. A "
            "fine-tuned pass supports necessity only if both zero-shot and a legal "
            "weight-identical S4 calibration-only variant fail the same frozen gate. "
            "This does not establish a real-robot fine-tuning requirement. Strict "
            "success includes offline true-center/contact-tag scoring unavailable to "
            "the non-oracle controller; privileged straight-to-center is an execution "
            "diagnostic, not a deployable policy or guaranteed performance upper bound."
        ),
        "updated_utc": _utc_now(),
    }
    _atomic_json(output / "report.json", report)
    _write_summary(output, report)
    _write_csv(output, report)
    _write_ppt_notes(output, report)
    figure_path = _write_figure(output, report)
    if (
        _canonical_json_sha256(_execution_bundle(phase, sizes, availability))
        != bundle_sha256
    ):
        raise RuntimeError("S4 execution bundle drifted before report commit")
    _write_provenance(output, task_contract, run_id)
    _atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "run_id": run_id,
            "phase": phase,
            "sizes": list(sizes),
            "poses_per_size": args.poses,
            "methods": list(METHODS),
            "evaluation_use": task_contract["evaluation_use"],
            "report_sha256": _sha256(output / "report.json"),
            "summary_sha256": _sha256(output / "summary.md"),
            "csv_sha256": _sha256(output / "ppt_comparison_table.csv"),
            "ppt_notes_cn_sha256": _sha256(output / "ppt_notes_cn.md"),
            "ppt_notes_en_sha256": _sha256(output / "ppt_notes_en.md"),
            "figure_path": _relative(figure_path),
            "figure_sha256": _sha256(figure_path),
            "provenance_sha256": _sha256(output / "provenance.json"),
            "task_contract_sha256": contract_sha256,
            "execution_bundle_sha256": bundle_sha256,
            "updated_utc": _utc_now(),
        },
    )
    print(json.dumps(report, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    """Run with guaranteed release of any execution lock acquired by impl."""

    try:
        return _run_impl(args)
    finally:
        execution_lock = getattr(args, "_execution_lock", None)
        if execution_lock is not None:
            try:
                execution_lock.release()
            finally:
                delattr(args, "_execution_lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or execute the two-stage cross-size S4 full-loop protocol. "
            "The default freezes a contract only; --execute is explicit and requires "
            "that an identical contract already exists."
        )
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--poses", type=int, default=12)
    parser.add_argument(
        "--workers", type=int, default=min(24, max(1, os.cpu_count() or 1))
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "explicitly replace a stale output execution lock; still refuses "
            "recovery if its parent or any recorded worker PID is alive"
        ),
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.poses <= 0 or args.workers <= 0:
        parser.error("--poses and --workers must be positive")
    if args.recover_stale_lock and not args.execute:
        parser.error("--recover-stale-lock requires --execute")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
