from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DENSE = (
    ROOT
    / "outputs"
    / "current"
    / "datasets"
    / "m8_hardware_easy_v2_direction_dense_training"
)
MODEL = (
    ROOT
    / "outputs"
    / "current"
    / "models"
    / "m8_hardware_easy_v2_robust_direction_v5"
)
AUDIT = (
    ROOT
    / "outputs"
    / "current"
    / "evaluations"
    / "m8_hardware_easy_v2_robustness_fresh4"
)


class TestHardwareEasyV2Artifacts(unittest.TestCase):
    def test_dense_dataset_is_complete_and_capture_filtered(self) -> None:
        progress = json.loads((DENSE / "progress.json").read_text(encoding="utf-8"))
        contract = json.loads(
            (DENSE / "sampling_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(progress["completed"], 768)
        self.assertEqual(progress["failed"], 0)
        self.assertEqual(contract["training_use"], "allowed_only_after_grouped_split_and_capture_filter")
        split = json.loads((MODEL / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(split["easy"]["all_ok_trials"], 768)
        self.assertEqual(split["easy"]["valid_full_fingerprints"], 675)
        self.assertEqual(split["easy"]["capture_or_incomplete_excluded"], 93)

    def test_promoted_model_contract_is_guarded_and_simulation_only(self) -> None:
        contract = json.loads(
            (MODEL / "ensemble_interface_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(contract["checkpoints"]), 3)
        self.assertLessEqual(contract["structured_transport_gate"]["max_aps_set_size"], 3)
        adaptation = contract["adaptation_contract"]
        self.assertEqual(
            adaptation["captured_or_incomplete_traces_used_for_direction_supervision"],
            0,
        )
        self.assertEqual(adaptation["final_promotion_audit_samples_seen_during_fitting"], 0)
        self.assertIn("Simulation-only", contract["claim_boundary"])

    def test_fresh4_is_untouched_and_promotion_gate_passes(self) -> None:
        for domain in ("strict", "easy"):
            source = (
                ROOT
                / "outputs"
                / "current"
                / "evaluations"
                / f"m8_hardware_easy_v2_fresh4_{domain}_untouched"
            )
            contract = json.loads(
                (source / "sampling_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contract["training_use"], "forbidden")
            self.assertEqual(contract["total"], 180)
        report = json.loads((AUDIT / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["promotion_ready_for_geometry_conditioned_simulation"])
        self.assertTrue(report["promotion_gates"]["strict_non_forgetting"]["passed"])
        self.assertTrue(report["promotion_gates"]["easy_v2_untouched"]["passed"])


if __name__ == "__main__":
    unittest.main()
