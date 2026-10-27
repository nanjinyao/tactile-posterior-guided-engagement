from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cartesian_admittance import AdmittanceConfig
from src.finetune_cross_size_s4_state_direction_heads import (
    _write_variant_checkpoint,
)
import src.run_cross_size_s4_full_loop_evaluation as s4


def _success_row(*, control_error_deg: float = 0.0) -> dict:
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
        "control_direction_deg": control_error_deg,
        "true_direction_deg": 0.0,
        "funnel_recovery_count": 0,
        "terminal_rejection_count": 0,
        "simulator_reset_count": 0,
    }


def _failure_row(*, hard_stop: bool = False) -> dict:
    return {
        "inner_wall_confirmed": False,
        "offline_capture_validated": False,
        "bottom_reached": False,
        "result": "common_time_budget_exhausted",
        "terminal_verification_mode": None,
        "hard_stop": hard_stop,
        "total_active_time_s": 130.0,
        "fallback_budget_extension_s": 0.0,
        "time_to_bottom_s": None,
        "actual_xy_path_mm": 10.0,
        "wear_proxy_mj": 2.0,
        "control_direction_deg": 0.0,
        "true_direction_deg": 0.0,
        "funnel_recovery_count": 0,
        "terminal_rejection_count": 0,
        "simulator_reset_count": 0,
    }


def _availability(*, cal_only: bool = True) -> dict[str, dict]:
    return {
        method: {
            "available": cal_only if method == "cal_only" else True,
            "reason": (
                "available"
                if method != "cal_only" or cal_only
                else "ensemble_interface_contract_missing"
            ),
        }
        for method in s4.METHODS
    }


def _records(
    *,
    oracle_successes: int = 9,
    fine_successes: int = 8,
    zero_successes: int = 7,
    cal_successes: int = 7,
) -> list[dict]:
    counts = {
        "privileged_straight_to_center": oracle_successes,
        "s4_fine_tuned": fine_successes,
        "zero_shot": zero_successes,
        "cal_only": cal_successes,
        "pure_outward": 5,
    }
    result = []
    for index in range(10):
        methods = {
            method: (
                _success_row() if index < successes else _failure_row()
            )
            for method, successes in counts.items()
        }
        result.append({"methods": methods})
    return result


class CrossSizeS4PhaseContractTests(unittest.TestCase):
    def test_frozen_json_contract_matches_tuple_valued_candidate(self) -> None:
        candidate = {
            "created_utc": "first",
            "commissioning": {"hard_wrench": (12.0, 12.0, 20.0)},
        }
        frozen = json.loads(json.dumps(candidate))
        frozen["created_utc"] = "second"
        self.assertEqual(
            s4._stable_contract(candidate),
            s4._stable_contract(frozen),
        )

    def test_development_and_untouched_are_physically_separate(self) -> None:
        development = s4.phase_protocol("development", ("m10", "m12"), 12)
        untouched = s4.phase_protocol("untouched", ("m10", "m12"), 12)
        self.assertNotEqual(development["default_output"], untouched["default_output"])
        self.assertNotEqual(
            development["evaluation_seeds"], untouched["evaluation_seeds"]
        )
        self.assertFalse(development["untouched_after_freeze"])
        self.assertTrue(untouched["untouched_after_freeze"])
        self.assertIn("tuning_allowed", development["evaluation_use"])
        self.assertIn("forbidden_for_training", untouched["evaluation_use"])

    def test_development_reuses_s3_v3_but_untouched_is_fresh(self) -> None:
        source = json.loads(s4.S3_DEVELOPMENT_SOURCE.read_text(encoding="utf-8"))
        development = s4.build_phase_tasks("development", "m10", 3)
        untouched = s4.build_phase_tasks("untouched", "m10", 3)
        self.assertEqual(
            [asdict(task) for task in development], source["tasks"]["m10"][:3]
        )
        self.assertTrue(
            all(task.pose_id.startswith("m10_s4_untouched_") for task in untouched)
        )
        self.assertEqual(
            {task.posterior_seed // 1000 for task in untouched},
            {s4.UNTOUCHED_SEEDS["m10"]},
        )
        self.assertFalse(
            {task.posterior_seed for task in development}
            & {task.posterior_seed for task in untouched}
        )

    def test_commissioning_is_shared_but_raw_hard_limits_are_not_relaxed(self) -> None:
        m10 = s4._size_contract("m10")
        m12 = s4._size_contract("m12")
        hard = AdmittanceConfig().hard_wrench
        self.assertEqual(m10["commissioning"]["admittance"]["hard_wrench"], hard)
        self.assertEqual(m12["commissioning"]["admittance"]["hard_wrench"], hard)
        self.assertTrue(m10["raw_hard_wrench_thresholds_unchanged_from_M8"])
        self.assertTrue(m12["raw_hard_wrench_thresholds_unchanged_from_M8"])
        self.assertNotEqual(
            m10["commissioning"]["bit_radius_scale"],
            m12["commissioning"]["bit_radius_scale"],
        )


class CrossSizeS4ControllerAndGateTests(unittest.TestCase):
    def test_s4_fine_tuned_availability_requires_frozen_adapter(self) -> None:
        for size in s4.SIZES:
            available = s4._method_availability_for_size(size)["s4_fine_tuned"]
            self.assertTrue(available["available"], available["reason"])
            self.assertEqual(
                available["model_root"], s4._relative(s4.S4_ADAPTER_ROOT)
            )
            adapter = s4.BaseAnchoredS4bDirectionAdapter(
                size, s4.S4_ADAPTER_ROOT
            )
            self.assertGreater(adapter.alpha, 0.0)
            self.assertLess(adapter.alpha, 1.0)

    def test_estimator_variants_use_hybrid_and_all_arms_share_commissioning(self) -> None:
        task = s4.build_phase_tasks("untouched", "m10", 1)[0]
        contract = s4._size_contract("m10")
        old_runtimes = s4._RUNTIMES
        s4._RUNTIMES = {
            "zero_shot": object(),
            "cal_only": object(),
            "s4_fine_tuned": object(),
        }
        try:
            calls = {}
            with patch.object(s4, "run_method", return_value=object()) as mocked:
                for method in s4.METHODS:
                    s4._run_one_method(task, method, contract)
                    calls[method] = mocked.call_args
        finally:
            s4._RUNTIMES = old_runtimes

        for method in s4.LEARNED_METHODS:
            self.assertEqual(calls[method].args[0], "hybrid")
            self.assertIsNotNone(calls[method].kwargs["direction_runtime"])
        self.assertEqual(calls["pure_outward"].args[0], "baseline")
        self.assertEqual(
            calls["privileged_straight_to_center"].args[0], "oracle"
        )
        for call in calls.values():
            self.assertEqual(call.kwargs["active_time_budget_s"], 130.0)
            self.assertEqual(
                call.kwargs["admittance_config"].hard_wrench,
                AdmittanceConfig().hard_wrench,
            )
            self.assertEqual(call.kwargs["funnel_guard"].max_recoveries, 3)
            self.assertTrue(call.kwargs["funnel_guard"].retry_on_terminal_rejection)

    def test_untouched_necessity_requires_zero_and_cal_fail_same_gate(self) -> None:
        report = s4._summarize_size(
            "m10", _records(), _availability(cal_only=True), "untouched"
        )
        self.assertTrue(report["oracle_execution_feasibility_gate_pass"])
        self.assertTrue(report["S4_fine_tuned_full_loop_gate_pass"])
        self.assertFalse(report["zero_shot_same_task_gate_pass"])
        self.assertFalse(report["cal_only_same_task_gate_pass"])
        self.assertTrue(report["adaptation_necessity_supported"])

        zero_passes = s4._summarize_size(
            "m10",
            _records(zero_successes=8),
            _availability(cal_only=True),
            "untouched",
        )
        self.assertFalse(zero_passes["adaptation_necessity_supported"])

    def test_development_never_supports_necessity_and_missing_cal_is_inconclusive(self) -> None:
        development = s4._summarize_size(
            "m12", _records(), _availability(cal_only=True), "development"
        )
        self.assertTrue(development["necessity_numerical_condition"])
        self.assertFalse(development["adaptation_necessity_supported"])
        self.assertIn("development_diagnostic_only", development["interpretation"])

        missing_cal_records = _records()
        for record in missing_cal_records:
            record["methods"].pop("cal_only")
        missing_cal = s4._summarize_size(
            "m12",
            missing_cal_records,
            _availability(cal_only=False),
            "untouched",
        )
        self.assertIsNone(missing_cal["cal_only_same_task_gate_pass"])
        self.assertFalse(missing_cal["adaptation_necessity_supported"])
        self.assertIn("cal_only_unavailable", missing_cal["interpretation"])

    def test_fine_tuned_hard_stop_or_gt90_error_fails_gate(self) -> None:
        passing = {
            "available": True,
            "strict_bottom_rate": 0.8,
            "hard_stops": 0,
            "control_direction_error_gt90_count": 0,
        }
        self.assertTrue(s4._task_metric_gate(passing))
        self.assertFalse(s4._task_metric_gate({**passing, "hard_stops": 1}))
        self.assertFalse(
            s4._task_metric_gate(
                {**passing, "control_direction_error_gt90_count": 1}
            )
        )


class CrossSizeS4CalOnlyContractTests(unittest.TestCase):
    def test_cal_only_export_is_byte_identical_by_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            target = root / "cal_only.pt"
            payload = b"opaque-torch-checkpoint-bytes"
            base.write_bytes(payload)
            _write_variant_checkpoint("cal_only", base, target, None)  # type: ignore[arg-type]
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(),
                hashlib.sha256(base.read_bytes()).hexdigest(),
            )

    def test_cal_only_requires_byte_identical_checkpoint_and_role_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seeds" / "seed_17"
            seed.mkdir(parents=True)
            base = root / "base.pt"
            checkpoint = seed / "circular_cnn_best.pt"
            payload = b"frozen-base-checkpoint"
            base.write_bytes(payload)
            checkpoint.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (seed / "metrics.json").write_text(
                json.dumps(
                    {
                        "base_checkpoint": str(base),
                        "base_checkpoint_sha256": digest,
                        "checkpoint_sha256": digest,
                        "checkpoint_byte_identical_to_base": True,
                    }
                ),
                encoding="utf-8",
            )
            contract = {
                "checkpoints": [str(checkpoint)],
                "adaptation_contract": {
                    "size": "m10",
                    "variant": "cal_only",
                    "trainable_modules": [],
                    "all_checkpoint_weights_frozen": True,
                    "weight_update_role": "none",
                    "model_selection_role": "none",
                    "temperature_and_APS_role": "calibration",
                    "S3_full_loop_samples_seen": 0,
                    "future_S4_full_loop_samples_seen": 0,
                },
            }
            (root / "ensemble_interface_contract.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            self.assertEqual(
                s4._variant_legality(root, "m10", "cal_only"),
                (True, "available"),
            )
            checkpoint.write_bytes(b"drift")
            legal, reason = s4._variant_legality(root, "m10", "cal_only")
            self.assertFalse(legal)
            self.assertIn("bytes_or_hashes_drifted", reason)


if __name__ == "__main__":
    unittest.main()
