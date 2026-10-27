from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cross_size_s4b_base_anchored_adapter import AdapterTelemetry
import src.run_cross_size_s4_full_loop_evaluation as s4


def _lock_payload(*, pid: int, worker_pids: list[int]) -> dict:
    return {
        "run_id": "old-run",
        "pid": pid,
        "worker_pids": worker_pids,
        "phase": "development",
        "task_contract_sha256": "contract-old",
        "execution_bundle_sha256": "bundle-old",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "updated_utc": "2026-01-01T00:00:00+00:00",
    }


def _acquire(root: Path, *, recover: bool = False, run_id: str = "new-run"):
    return s4.OutputExecutionLock.acquire(
        root,
        phase="development",
        task_contract_sha256="contract-new",
        execution_bundle_sha256="bundle-new",
        recover_stale=recover,
        run_id=run_id,
    )


def _method_row(
    *,
    control_direction_deg: float,
    shortcut_distance_mm: float,
    shortcut_executed: bool = True,
) -> dict:
    return {
        "inner_wall_confirmed": True,
        "offline_capture_validated": True,
        "bottom_reached": True,
        "result": "cad_bottom_verified",
        "terminal_verification_mode": "bottom_force_dwell",
        "hard_stop": False,
        "total_active_time_s": 20.0,
        "fallback_budget_extension_s": 0.0,
        "time_to_bottom_s": 20.0,
        "actual_xy_path_mm": 4.0,
        "wear_proxy_mj": 1.0,
        "control_direction_deg": control_direction_deg,
        "true_direction_deg": 0.0,
        "first_shortcut_executed": shortcut_executed,
        "first_shortcut_distance_mm": shortcut_distance_mm,
        "funnel_recovery_count": 0,
        "terminal_rejection_count": 0,
        "simulator_reset_count": 0,
    }


class OutputExecutionLockTests(unittest.TestCase):
    def test_o_excl_lock_records_required_provenance_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = _acquire(root)
            payload = json.loads(lock.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "new-run")
            self.assertEqual(payload["pid"], os.getpid())
            self.assertEqual(payload["phase"], "development")
            self.assertEqual(payload["task_contract_sha256"], "contract-new")
            self.assertEqual(payload["execution_bundle_sha256"], "bundle-new")
            self.assertEqual(payload["worker_pids"], [])
            lock.release()
            self.assertFalse(lock.path.exists())

    def test_live_parent_lock_is_refused_even_with_recovery_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = _acquire(root, run_id="owner")
            try:
                for recover in (False, True):
                    with self.assertRaisesRegex(RuntimeError, "live S4 execution"):
                        _acquire(root, recover=recover, run_id="contender")
            finally:
                owner.release()

    def test_stale_lock_requires_explicit_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / s4.EXECUTION_LOCK_FILENAME
            s4._write_exclusive_json(
                path, _lock_payload(pid=999_999_991, worker_pids=[])
            )
            with patch.object(s4, "_pid_is_alive", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "--recover-stale-lock"):
                    _acquire(root, recover=False)
                recovered = _acquire(root, recover=True)
            try:
                self.assertEqual(recovered.run_id, "new-run")
            finally:
                recovered.release()

    def test_recovery_refuses_a_live_recorded_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / s4.EXECUTION_LOCK_FILENAME
            s4._write_exclusive_json(
                path,
                _lock_payload(pid=999_999_991, worker_pids=[os.getpid()]),
            )
            with patch.object(
                s4, "_pid_is_alive", side_effect=lambda pid: pid == os.getpid()
            ):
                with self.assertRaisesRegex(RuntimeError, "recorded live PIDs"):
                    _acquire(root, recover=True)

    def test_worker_pid_update_is_owned_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = _acquire(Path(directory))
            try:
                lock.update_worker_pids((91, 42, 91))
                payload = json.loads(lock.path.read_text(encoding="utf-8"))
                self.assertEqual(payload["worker_pids"], [42, 91])
            finally:
                lock.release()

    def test_run_wrapper_releases_lock_on_clean_return_and_error(self) -> None:
        class FakeLock:
            def __init__(self) -> None:
                self.release_count = 0

            def release(self) -> None:
                self.release_count += 1

        clean_lock = FakeLock()
        clean_args = argparse.Namespace()

        def clean_impl(args: argparse.Namespace) -> int:
            args._execution_lock = clean_lock
            return 7

        with patch.object(s4, "_run_impl", side_effect=clean_impl):
            self.assertEqual(s4.run(clean_args), 7)
        self.assertEqual(clean_lock.release_count, 1)
        self.assertFalse(hasattr(clean_args, "_execution_lock"))

        error_lock = FakeLock()
        error_args = argparse.Namespace()

        def error_impl(args: argparse.Namespace) -> int:
            args._execution_lock = error_lock
            raise ValueError("synthetic failure")

        with patch.object(s4, "_run_impl", side_effect=error_impl):
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                s4.run(error_args)
        self.assertEqual(error_lock.release_count, 1)
        self.assertFalse(hasattr(error_args, "_execution_lock"))


class DirectionMetricSchemaTests(unittest.TestCase):
    def test_zero_length_abstention_is_latent_not_executed_error(self) -> None:
        row = _method_row(
            control_direction_deg=157.0,
            shortcut_distance_mm=0.0,
            shortcut_executed=True,
        )
        summary = s4._method_summary(
            [{"methods": {"s4_fine_tuned": row}}], "s4_fine_tuned"
        )
        self.assertEqual(summary["control_direction_error_gt90_count"], 1)
        self.assertEqual(summary["latent_control_error_gt90_count"], 1)
        self.assertEqual(summary["directed_shortcut_accepted_count"], 0)
        self.assertEqual(summary["executed_directed_error_gt90_count"], 0)
        self.assertTrue(s4._task_metric_gate(summary))

    def test_nonzero_executed_bad_direction_fails_safety_gate(self) -> None:
        row = _method_row(
            control_direction_deg=110.0,
            shortcut_distance_mm=1.0,
            shortcut_executed=True,
        )
        summary = s4._method_summary(
            [{"methods": {"s4_fine_tuned": row}}], "s4_fine_tuned"
        )
        self.assertEqual(summary["directed_shortcut_accepted_count"], 1)
        self.assertEqual(summary["executed_directed_error_gt90_count"], 1)
        self.assertFalse(s4._task_metric_gate(summary))

    def test_pre_run_id_resumed_record_remains_valid(self) -> None:
        task = s4.build_phase_tasks("development", "m10", 1)[0]
        record = {
            "phase": "development",
            "task": asdict(task),
            "methods": {"zero_shot": {}},
            "task_contract_sha256": "contract",
            "worker_init_execution_bundle_sha256": "bundle",
            "worker_completion_execution_bundle_sha256": "bundle",
        }
        # No run_id is intentional: old frozen trials remain resume-compatible.
        s4._validate_resumed_record(
            record,
            task,
            "development",
            ("zero_shot",),
            "contract",
            "bundle",
        )

    def test_method_record_serializes_adapter_telemetry_and_acceptance(self) -> None:
        @dataclass
        class FakeResult:
            first_shortcut_executed: bool = True
            first_shortcut_distance_mm: float = 0.0

        runtime = object.__new__(s4.BaseAnchoredS4bDirectionAdapter)
        runtime.last_telemetry = AdapterTelemetry(
            residual_alpha=0.92,
            minimum_fine_tuned_concentration=0.60,
            maximum_base_fine_tuned_disagreement_deg=90.0,
            base_direction_deg=10.0,
            fine_tuned_direction_deg=20.0,
            adapted_direction_deg=19.2,
            base_fine_tuned_disagreement_deg=10.0,
            base_concentration=0.7,
            fine_tuned_concentration=0.5,
            adapted_raw_concentration=0.6,
            conservative_concentration=0.0,
            conservative_span_deg=180.0,
            state_total_variation=0.1,
            aps_union_size=4,
            risk_gate_passed=False,
            risk_gate_reasons=("low_concentration",),
        )
        previous = s4._RUNTIMES
        s4._RUNTIMES = {"s4_fine_tuned": runtime}
        try:
            with patch.object(s4, "_run_one_method", return_value=FakeResult()):
                row = s4._run_method_record(
                    object(), "s4_fine_tuned", {}  # type: ignore[arg-type]
                )
        finally:
            s4._RUNTIMES = previous
        self.assertFalse(row["directed_shortcut_accepted"])
        self.assertFalse(row["adapter_last_telemetry"]["risk_gate_passed"])
        self.assertEqual(
            row["adapter_last_telemetry"]["risk_gate_reasons"],
            ("low_concentration",),
        )


if __name__ == "__main__":
    unittest.main()
