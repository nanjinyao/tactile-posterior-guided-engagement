from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/current/datasets/cross_size_s3_adaptation_pilot"
MODELS = ROOT / "outputs/current/models/cross_size_s3_direction_head_pilot"
EVAL = ROOT / "outputs/current/evaluations/cross_size_s3_full_loop_pilot_v3"
SIZES = ("m10", "m12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestCrossSizeS3AdaptationPilot(unittest.TestCase):
    def test_fresh_data_are_complete_C6_orbit_isolated_and_role_traced(self) -> None:
        top = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(top["status"], "complete")
        self.assertEqual(top["old_S1_S2_training_samples_used"], 0)
        for size in SIZES:
            contract = json.loads(
                (DATA / size / "dataset_contract.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (DATA / size / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            train = set(contract["train_orbit_ids"])
            calibration = set(contract["calibration_orbit_ids"])
            self.assertEqual(train & calibration, set())
            self.assertEqual(len(train), 6)
            self.assertEqual(len(calibration), 6)
            self.assertEqual(contract["role_counts"]["train"]["scans"], 36)
            self.assertEqual(contract["role_counts"]["calibration"]["scans"], 36)
            self.assertEqual(len(manifest["trial_files"]), 72)
            self.assertFalse(manifest["C6_orbit_leakage"])
            for role in ("train", "calibration"):
                counts = {}
                for item in manifest["trial_files"]:
                    if item["role"] == role:
                        counts[item["symmetry_orbit_id"]] = (
                            counts.get(item["symmetry_orbit_id"], 0) + 1
                        )
                self.assertEqual(set(counts.values()), {6})
            self.assertTrue(
                all(item["sha256"] == _sha256(ROOT / item["path"])
                    for item in manifest["trial_files"])
            )

    def test_three_seed_adaptation_changes_only_direction_head(self) -> None:
        report = json.loads((MODELS / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["frozen_modules"], ["encoder", "shared", "state_head"])
        self.assertEqual(report["trainable_module"], "direction_head")
        self.assertEqual(report["untouched_full_loop_samples_seen"], 0)
        for size in SIZES:
            model_manifest = json.loads(
                (MODELS / size / "model_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model_manifest["train_calibration_C6_intersection"], [])
            self.assertEqual(model_manifest["old_S1_S2_training_samples_used"], 0)
            for variant in ("cal_only", "fine_tuned"):
                variant_report = report["sizes"][size][variant]
                self.assertEqual(len(variant_report["seeds"]), 3)
                self.assertTrue(variant_report["all_frozen_components_exact"])
                runtime_contract = json.loads(
                    (
                        MODELS
                        / size
                        / variant
                        / "ensemble_interface_contract.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(len(runtime_contract["checkpoints"]), 3)
                self.assertEqual(runtime_contract["adaptation_contract"]["test_samples_seen"], 0)
                for seed in variant_report["seeds"]:
                    self.assertEqual(
                        seed["frozen_state_hash_before"],
                        seed["frozen_state_hash_after"],
                    )
                    checkpoint = ROOT / runtime_contract["checkpoints"][
                        variant_report["seeds"].index(seed)
                    ]
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertTrue(any(key.startswith("direction_head.") for key in state))
            self.assertTrue(
                report["sizes"][size]["cal_only"]["all_cal_only_checkpoints_equal_base"]
            )
            self.assertTrue(
                report["sizes"][size]["fine_tuned"]["all_direction_heads_changed"]
            )

    def test_untouched_full_loop_contract_and_artifacts(self) -> None:
        contract = json.loads((EVAL / "task_contract.json").read_text(encoding="utf-8"))
        report = json.loads((EVAL / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((EVAL / "manifest.json").read_text(encoding="utf-8"))
        correction = EVAL / "NARRATIVE_CORRECTION.md"
        self.assertEqual(
            contract["methods"],
            ["zero_shot", "cal_only", "fine_tuned", "outward", "oracle"],
        )
        self.assertEqual(contract["active_time_budget_s"], 130.0)
        self.assertTrue(contract["zero_budget_extension"])
        self.assertEqual(contract["post_capture_funnel"]["max_recoveries"], 3)
        self.assertTrue(
            contract["post_capture_funnel"]["retry_on_terminal_rejection"]
        )
        self.assertFalse(contract["old_S1_S2_rollouts_loaded"])
        capture = contract["capture_and_scoring_contract"]
        self.assertIn("sensor-only", capture["non_oracle_handoff"])
        self.assertIn("privileged true-center", capture["oracle_handoff"])
        self.assertFalse(capture["offline_scoring_available_to_controller"])
        bundle_sha256 = contract["execution_bundle_sha256_frozen_before_rollout"]
        self.assertEqual(len(bundle_sha256), 64)
        bundle = contract["execution_bundle_frozen_before_rollout"]
        self.assertIn("src/run_cross_size_s3_full_loop_pilot.py", bundle["sources"])
        self.assertIn("src/render_m8_search_comparison.py", bundle["sources"])
        self.assertGreaterEqual(len(bundle["sources"]), 80)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["report_sha256"], _sha256(EVAL / "report.json"))
        self.assertEqual(manifest["summary_sha256"], _sha256(EVAL / "summary.md"))
        self.assertTrue(correction.is_file())
        self.assertEqual(
            manifest["narrative_correction_sha256"], _sha256(correction)
        )
        self.assertIn("no fine-tuned full-loop gate pass", report["claim_boundary"])
        self.assertNotIn("A fine-tuned pass shows", report["claim_boundary"])
        self.assertTrue((EVAL / "ppt_comparison_table.csv").is_file())
        self.assertTrue((EVAL / "cross_size_s3_full_loop_pilot.png").is_file())
        for size in SIZES:
            size_manifest = json.loads(
                (EVAL / size / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(size_manifest["evaluation_training_use"], "forbidden")
            self.assertTrue(size_manifest["all_first_probe_hashes_match"])
            self.assertTrue(size_manifest["all_budget_extensions_zero"])
            self.assertTrue(size_manifest["all_simulator_resets_zero"])
            self.assertTrue(size_manifest["all_worker_execution_bundles_match"])
            self.assertEqual(size_manifest["execution_bundle_sha256"], bundle_sha256)
            trial_paths = sorted((EVAL / size / "trials").glob("*.json"))
            self.assertEqual(len(trial_paths), manifest["poses_per_size"])
            first_trial_time = min(
                datetime.fromisoformat(
                    json.loads(path.read_text(encoding="utf-8"))["created_utc"]
                )
                for path in trial_paths
            )
            self.assertLess(datetime.fromisoformat(contract["created_utc"]), first_trial_time)
            for path in trial_paths:
                trial = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    trial["worker_init_execution_bundle_sha256"], bundle_sha256
                )
                self.assertEqual(
                    trial["worker_completion_execution_bundle_sha256"],
                    bundle_sha256,
                )
                self.assertFalse(
                    trial["fairness"][
                        "non_oracle_online_true_center_or_contact_tags"
                    ]
                )
                self.assertTrue(
                    trial["fairness"][
                        "oracle_is_privileged_true_center_diagnostic"
                    ]
                )
            for method in contract["methods"]:
                item = report["sizes"][size]["methods"][method]
                self.assertEqual(item["trials"], manifest["poses_per_size"])
                self.assertEqual(item["budget_violations"], 0)
                self.assertEqual(item["simulator_resets"], 0)


if __name__ == "__main__":
    unittest.main()
