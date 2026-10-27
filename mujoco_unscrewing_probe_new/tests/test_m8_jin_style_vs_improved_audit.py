from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.run_m8_fair_policy_benchmark import build_tasks
from src.run_m8_jin_style_vs_improved_audit import (
    AMENDMENT_FILENAME,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_DIRECTION_SOURCE,
    DEFAULT_IMPROVED_MODEL_ROOT,
    DEFAULT_JIN_MODEL_ROOT,
    DEFAULT_OUTPUT,
    EXPECTED_DIRECTION_TRIALS,
    TASK_COUNT,
    TASK_SEED,
    _atomic_json,
    _completed_manifest_is_valid,
    _continuous_direction_metrics,
    _exact_mcnemar,
    _experiment_contract,
    _load_verified_derived_rebuild_records,
    _method_task_summary,
    _paired_binary,
    _paired_base_pose_cluster_bootstrap,
    _paired_trial_record_set_sha256,
    _source_manifest,
    _validate_task_pair,
    _verify_contract_amendment,
    parse_args,
)


class TestM8JinStyleVsImprovedAudit(unittest.TestCase):
    def test_direction_metrics_include_continuous_and_common_sector_endpoints(self) -> None:
        rows = [
            {"test_error_deg": error, "test_sector_correct": correct}
            for error, correct in zip((0.0, 5.0, 10.0, 100.0), (True, True, True, False))
        ]
        metrics = _continuous_direction_metrics(rows, "test")
        self.assertEqual(metrics["samples"], 4)
        self.assertAlmostEqual(metrics["angle_error_median_deg"], 7.5)
        self.assertAlmostEqual(metrics["angle_error_p90_deg"], 73.0)
        self.assertAlmostEqual(metrics["angle_within_7p5_rate"], 0.5)
        self.assertAlmostEqual(metrics["angle_within_15_rate"], 0.75)
        self.assertEqual(metrics["catastrophic_gt90_count"], 1)
        self.assertAlmostEqual(metrics["common_12_sector_accuracy"], 0.75)

    def test_exact_mcnemar_and_paired_table_are_two_sided(self) -> None:
        self.assertAlmostEqual(_exact_mcnemar(1, 3), 0.625)
        self.assertEqual(_exact_mcnemar(0, 0), 1.0)
        events = ((True, True), (True, False), (False, True), (False, False))
        records = [
            {
                "methods": {
                    "improved_fingerprint": {"event": left},
                    "jin_2021_style_proxy": {"event": right},
                }
            }
            for left, right in events
        ]
        paired = _paired_binary(records, lambda row: row["event"])
        self.assertEqual(paired["both_event"], 1)
        self.assertEqual(paired["improved_only_event"], 1)
        self.assertEqual(paired["jin_proxy_only_event"], 1)
        self.assertEqual(paired["neither_event"], 1)
        self.assertEqual(paired["mcnemar_exact_two_sided_p"], 1.0)

    def test_paired_base_pose_bootstrap_is_clustered_and_deterministic(self) -> None:
        rows = [
            {
                "base_pose_id": cluster,
                "improved_error_deg": improved,
                "jin_error_deg": proxy,
            }
            for cluster, values in {
                "pose_a": ((1.0, 3.0), (2.0, 4.0), (3.0, 5.0)),
                "pose_b": ((4.0, 3.0), (5.0, 4.0), (6.0, 5.0)),
                "pose_c": ((2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
            }.items()
            for improved, proxy in values
        ]
        first = _paired_base_pose_cluster_bootstrap(rows, seed=17, resamples=500)
        second = _paired_base_pose_cluster_bootstrap(rows, seed=17, resamples=500)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 3)
        self.assertEqual(first["cluster_size_counts"], {"3": 3})
        self.assertEqual(first["samples"], 9)
        self.assertEqual(first["point_estimate_deg"], 0.0)
        self.assertEqual(first["seed"], 17)
        self.assertEqual(first["resamples"], 500)
        self.assertLessEqual(
            first["ci_percentile_low_deg"], first["point_estimate_deg"]
        )
        self.assertGreaterEqual(
            first["ci_percentile_high_deg"], first["point_estimate_deg"]
        )
        self.assertFalse(first["ci_excludes_zero"])
        self.assertIn("includes zero", first["interval_interpretation"])

    def test_pair_fairness_requires_matching_fingerprint_when_available(self) -> None:
        task = build_tasks(1, TASK_SEED, wrench_noise=True)[0]
        fairness = _validate_task_pair(
            task,
            {"fingerprint_sha256": "abc"},
            {"fingerprint_sha256": "abc"},
        )
        self.assertTrue(fairness["fingerprint_available_for_both"])
        self.assertTrue(fairness["fingerprint_sha256_match"])
        self.assertTrue(fairness["only_direction_estimator_changes"])
        with self.assertRaisesRegex(ValueError, "fingerprint SHA mismatch"):
            _validate_task_pair(
                task,
                {"fingerprint_sha256": "abc"},
                {"fingerprint_sha256": "def"},
            )
        with self.assertRaisesRegex(ValueError, "one-sided missing"):
            _validate_task_pair(
                task,
                {"fingerprint_sha256": None},
                {"fingerprint_sha256": "def"},
            )

    def test_task_summary_uses_executed_control_not_top_class_error(self) -> None:
        def method_row(control: float, true: float, top_class_error: float) -> dict:
            return {
                "control_direction_deg": control,
                "true_direction_deg": true,
                "direction_error_deg": top_class_error,
                "offline_capture_validated": False,
                "inner_wall_confirmed": False,
                "bottom_reached": False,
                "result": "search_radius_exhausted",
                "terminal_verification_mode": None,
                "hard_stop": False,
                "total_active_time_s": 1.0,
                "fallback_budget_extension_s": 0.0,
            }

        records = [
            {
                "methods": {
                    "improved_fingerprint": method_row(10.0, 10.0, 99.0),
                    "jin_2021_style_proxy": method_row(20.0, 10.0, 1.0),
                }
            }
        ]
        improved = _method_task_summary(records, "improved_fingerprint")
        proxy = _method_task_summary(records, "jin_2021_style_proxy")
        self.assertEqual(improved["control_direction_error_median_deg"], 0.0)
        self.assertEqual(proxy["control_direction_error_median_deg"], 10.0)
        self.assertEqual(
            improved["top_class_direction_diagnostic"]["error_median_deg"],
            99.0,
        )
        self.assertIn(
            "not the direct-first ray",
            improved["top_class_direction_diagnostic"]["role"],
        )
        self.assertNotIn("direction_error_median_deg", improved)

    def test_frozen_contract_records_fresh_48_pose_estimator_only_design(self) -> None:
        tasks = build_tasks(TASK_COUNT, TASK_SEED, wrench_noise=True)
        contract = _experiment_contract(
            direction_source=DEFAULT_DIRECTION_SOURCE,
            improved_model_root=Path("/tmp/improved"),
            jin_model_root=Path("/tmp/jin"),
            tasks=tasks,
            expected_direction_trials=EXPECTED_DIRECTION_TRIALS,
        )
        task_layer = contract["task_layer"]
        self.assertEqual(task_layer["seed"], TASK_SEED)
        self.assertEqual(task_layer["poses"], 48)
        self.assertTrue(task_layer["wrench_noise"])
        self.assertEqual(task_layer["active_time_budget_s"], 130.0)
        self.assertEqual(len(task_layer["tasks"]), 48)
        self.assertIn("Only the frozen direction estimator changes", task_layer["paired_execution"])
        self.assertIn("not a faithful paper reproduction", contract["claim_boundary"])
        self.assertIn("97.4%", contract["claim_boundary"])

    def test_completed_manifest_is_verified_and_not_reopened(self) -> None:
        tasks = build_tasks(1, TASK_SEED, wrench_noise=True)
        contract = _experiment_contract(
            direction_source=DEFAULT_DIRECTION_SOURCE,
            improved_model_root=Path("/tmp/improved"),
            jin_model_root=Path("/tmp/jin"),
            tasks=tasks,
            expected_direction_trials=EXPECTED_DIRECTION_TRIALS,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract_path = output / "task_contract.json"
            _atomic_json(contract_path, contract)
            digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            _atomic_json(
                output / "manifest.json",
                {
                    "status": "complete",
                    "task_contract_sha256": digest,
                    "artifacts": {},
                },
            )
            self.assertTrue(_completed_manifest_is_valid(output, contract))
            different = json.loads(json.dumps(contract))
            different["task_layer"]["poses"] = 2
            with self.assertRaisesRegex(ValueError, "contract does not match"):
                _completed_manifest_is_valid(output, different)

    def test_derived_rebuild_reads_hash_locked_trials_and_exact_control_errors(self) -> None:
        tasks = build_tasks(TASK_COUNT, TASK_SEED, wrench_noise=True)
        contract = _experiment_contract(
            direction_source=DEFAULT_DIRECTION_SOURCE,
            improved_model_root=DEFAULT_IMPROVED_MODEL_ROOT,
            jin_model_root=DEFAULT_JIN_MODEL_ROOT,
            tasks=tasks,
            expected_direction_trials=EXPECTED_DIRECTION_TRIALS,
        )
        records, manifest, previous_manifest_sha = (
            _load_verified_derived_rebuild_records(DEFAULT_OUTPUT, contract, tasks)
        )
        self.assertEqual(len(records), 48)
        self.assertEqual(manifest["completed_paired_task_poses"], 48)
        self.assertEqual(len(previous_manifest_sha), 64)
        self.assertEqual(
            manifest["paired_trial_record_set_sha256"],
            "4be92296c47ed0a4d7791fd852943c4106759536495f4fe949e90fea7d53077a",
        )
        improved = _method_task_summary(records, "improved_fingerprint")
        proxy = _method_task_summary(records, "jin_2021_style_proxy")
        self.assertEqual(improved["control_direction_error_samples"], 48)
        self.assertAlmostEqual(
            improved["control_direction_error_median_deg"],
            3.3408331774659956,
        )
        self.assertAlmostEqual(
            improved["control_direction_error_p90_deg"],
            20.553935618712543,
        )
        self.assertEqual(improved["control_direction_within_7p5_count"], 38)
        self.assertEqual(improved["control_direction_within_15_count"], 42)
        self.assertEqual(improved["control_direction_gt90_count"], 0)
        self.assertAlmostEqual(
            improved["top_class_direction_diagnostic"]["error_median_deg"],
            3.8706918619573116,
        )
        self.assertAlmostEqual(
            improved["top_class_direction_diagnostic"]["error_p90_deg"],
            12.161717534065247,
        )
        self.assertEqual(proxy["control_direction_error_samples"], 48)
        self.assertAlmostEqual(
            proxy["control_direction_error_median_deg"],
            8.175838384777308,
        )
        self.assertAlmostEqual(
            proxy["control_direction_error_p90_deg"],
            17.356161475181576,
        )
        self.assertEqual(proxy["control_direction_within_7p5_count"], 22)
        self.assertEqual(proxy["control_direction_within_15_count"], 40)
        self.assertEqual(proxy["control_direction_gt90_count"], 3)

    def test_post_completion_amendment_is_explicit_and_hash_bound(self) -> None:
        task_contract_sha = hashlib.sha256(
            (DEFAULT_OUTPUT / "task_contract.json").read_bytes()
        ).hexdigest()
        trial_set_sha = _paired_trial_record_set_sha256(DEFAULT_OUTPUT)
        payload, reference = _verify_contract_amendment(
            DEFAULT_OUTPUT,
            task_contract_sha256=task_contract_sha,
            trial_record_set_sha256=trial_set_sha,
        )
        self.assertEqual(reference["path"].split("/")[-1], AMENDMENT_FILENAME)
        self.assertEqual(len(reference["sha256"]), 64)
        self.assertTrue(reference["post_completion_protocol_correction"])
        self.assertFalse(reference["preregistered"])
        self.assertEqual(payload["original_contract"]["sha256"], task_contract_sha)
        self.assertEqual(
            payload["frozen_evidence"]["strict_paired_trial_record_set_sha256"],
            trial_set_sha,
        )
        self.assertIn("overbroad", payload["correction"])
        timing = payload["preregistration_timestamp_provenance"]
        self.assertEqual(timing["declared_created_utc"], "2026-07-15T00:00:00Z")
        self.assertIn(
            "placeholder", timing["declared_created_utc_interpretation"]
        )
        self.assertLess(
            timing["preregistration_file"]["observed_filesystem_mtime_ns"],
            timing["first_jin_proxy_checkpoint"]["observed_filesystem_mtime_ns"],
        )

    def test_committed_derived_artifacts_label_executed_control_exactly(self) -> None:
        report = json.loads((DEFAULT_OUTPUT / "report.json").read_text(encoding="utf-8"))
        task = report["task_layer"]
        self.assertEqual(task["direction_metric_schema_version"], 2)
        self.assertIn("control_direction_deg", task["operational_direction_scope"])
        self.assertIn(
            "non-operational diagnostic",
            task["top_class_direction_diagnostic_scope"],
        )
        correction = task["post_completion_derived_metric_correction"]
        self.assertEqual(
            correction["correct_operational_source"],
            "RunResult.control_direction_deg",
        )
        self.assertFalse(correction["simulator_rerun"])
        self.assertEqual(correction["frozen_trial_files_modified"], 0)
        improved = task["methods"]["improved_fingerprint"]
        proxy = task["methods"]["jin_2021_style_proxy"]
        self.assertAlmostEqual(
            improved["control_direction_error_median_deg"],
            3.3408331774659956,
        )
        self.assertAlmostEqual(
            improved["control_direction_error_p90_deg"],
            20.553935618712543,
        )
        self.assertAlmostEqual(
            proxy["control_direction_error_median_deg"],
            8.175838384777308,
        )
        self.assertAlmostEqual(
            proxy["control_direction_error_p90_deg"],
            17.356161475181576,
        )
        self.assertNotIn("direction_error_scope", task)
        with (DEFAULT_OUTPUT / "paired_tasks.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            columns = next(csv.reader(handle))
        self.assertIn("improved_control_direction_error_deg", columns)
        self.assertIn("improved_top_class_direction_error_deg", columns)
        self.assertNotIn("improved_direction_error_deg", columns)
        summary = (DEFAULT_OUTPUT / "summary.md").read_text(encoding="utf-8")
        self.assertIn("direction actually executed by the ray", summary)
        self.assertIn("Top-class diagnostic (not the executed ray)", summary)

        tasks = build_tasks(TASK_COUNT, TASK_SEED, wrench_noise=True)
        contract = _experiment_contract(
            direction_source=DEFAULT_DIRECTION_SOURCE,
            improved_model_root=DEFAULT_IMPROVED_MODEL_ROOT,
            jin_model_root=DEFAULT_JIN_MODEL_ROOT,
            tasks=tasks,
            expected_direction_trials=EXPECTED_DIRECTION_TRIALS,
        )
        self.assertTrue(_completed_manifest_is_valid(DEFAULT_OUTPUT, contract))
        manifest = json.loads(
            (DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        rebuild = manifest["derived_rebuild"]
        self.assertFalse(rebuild["simulation_rerun"])
        self.assertEqual(rebuild["trial_files_overwritten"], 0)
        self.assertEqual(
            rebuild["frozen_trial_record_set_sha256"],
            "4be92296c47ed0a4d7791fd852943c4106759536495f4fe949e90fea7d53077a",
        )

    def test_source_provenance_covers_models_checkpoints_and_audit_code(self) -> None:
        sources = _source_manifest(
            DEFAULT_OUTPUT,
            DEFAULT_DIRECTION_SOURCE,
            DEFAULT_IMPROVED_MODEL_ROOT,
            DEFAULT_JIN_MODEL_ROOT,
            [
                {
                    "trial_id": "hash_contract_smoke",
                    "shared_phase_sequence_sha256": "0" * 64,
                }
            ],
        )
        self.assertIn("preregistration", sources)
        self.assertIn("jin_proxy_training_report", sources)
        self.assertEqual(len(sources["jin_proxy_checkpoints"]), 3)
        self.assertEqual(len(sources["improved_ensemble_checkpoints"]), 3)
        self.assertEqual(len(sources["improved_ensemble_runtime_auxiliaries"]), 3)
        self.assertIn("comparison_audit_source", sources)
        self.assertIn("strict_complete_loop_audit_source", sources)
        self.assertIn("improved_model_definition_source", sources)
        self.assertIn("post_completion_contract_amendment", sources)
        self.assertEqual(
            sources["preregistration"]["declared_created_utc"],
            "2026-07-15T00:00:00Z",
        )
        self.assertTrue(
            sources["preregistration_timing_observation"][
                "created_utc_is_date_level_placeholder"
            ]
        )
        self.assertGreater(
            sources["preregistration_timing_observation"]["observed_gap_s"], 0.0
        )
        for family in ("jin_proxy_checkpoints", "improved_ensemble_checkpoints"):
            for item in sources[family]:
                self.assertEqual(len(item["sha256"]), 64)

    def test_cli_defaults_are_the_frozen_full_design(self) -> None:
        args = parse_args([])
        self.assertEqual(args.poses, TASK_COUNT)
        self.assertEqual(args.expected_direction_trials, EXPECTED_DIRECTION_TRIALS)
        self.assertTrue(args.resume)
        self.assertFalse(args.rebuild_derived)
        self.assertTrue(parse_args(["--rebuild-derived"]).rebuild_derived)
        self.assertEqual(BOOTSTRAP_SEED, 20260734)
        self.assertEqual(BOOTSTRAP_RESAMPLES, 20_000)
        with self.assertRaises(SystemExit):
            parse_args(["--workers", "0"])


if __name__ == "__main__":
    unittest.main()
