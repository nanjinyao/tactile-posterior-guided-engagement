from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from src.audit_clocking_clean_multisize_training import (
    DEFAULT_CLEAN_BASE,
    DEFAULT_DATA,
    DEFAULT_PROFILE,
    build_report,
)
from src.finetune_hardware_easy_v2_multisize import (
    _load_easy_dataset,
    _load_geometry_equivalence_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "profile": str(DEFAULT_PROFILE),
        "base_model": str(DEFAULT_CLEAN_BASE),
        "m8_easy_data": str(DEFAULT_DATA["m8"]),
        "m10_easy_data": str(DEFAULT_DATA["m10"]),
        "m12_easy_data": str(DEFAULT_DATA["m12"]),
        "output": "",
        "geometry_equivalence_audit": None,
        "m8_source_development_data": [],
        "easy_development_data": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestClockingCleanMultisizeTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report(_args())

    def test_clocking_is_construction_metadata_not_label_rotation(self) -> None:
        self.assertEqual(self.report["status"], "PASS")
        clocking = self.report["clocking_contract"]
        self.assertEqual(
            clocking["integrated_peg_shape_rotation_from_legacy_deg"], -15.0
        )
        self.assertEqual(
            clocking["canonical_mujoco_peg_socket_relative_yaw_deg"], 0.0
        )
        self.assertEqual(clocking["training_label_rotation_applied_deg"], 0.0)

    def test_all_canonical_contact_xmls_are_byte_identical(self) -> None:
        # Revalidated from immutable raw scans: continuous-contact force
        # unloading remains admissible, while privileged MuJoCo contact loss
        # and full-chamfer downward progress are quarantined.
        expected_valid = {"m8": 675, "m10": 611, "m12": 575}
        for size, item in self.report["datasets"].items():
            with self.subTest(size=size):
                self.assertTrue(item["spec_hash_changed_only_under_explicit_audit"])
                self.assertTrue(item["contact_xml_byte_identical"])
                self.assertEqual(
                    item["sampling_contract_model_sha256"],
                    item["current_model_sha256"],
                )
                self.assertEqual(
                    item["regenerated_model_sha256"],
                    item["current_model_sha256"],
                )
                self.assertEqual(
                    item["valid_complete_direction_fingerprints"],
                    expected_valid[size],
                )
                self.assertEqual(
                    item["trials_missing_explicit_direction_validity"], 0
                )

    def test_audited_base_and_policy_exclude_ambiguous_development_data(self) -> None:
        base = self.report["clean_base_model"]
        self.assertEqual(base["failed_audit_development_sets_consumed"], 0)
        self.assertEqual(base["untouched_promotion_samples_seen"], 0)
        self.assertEqual(base["captured_or_incomplete_direction_labels_used"], 0)
        policy = self.report["training_policy"]
        self.assertTrue(policy["canonical_datasets_only"])
        self.assertFalse(policy["failed_audit_development_datasets_allowed"])

    def test_trainer_requires_explicit_audit_for_spec_hash_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "spec hash drift"):
            _load_easy_dataset("m12", DEFAULT_DATA["m12"])
        contract, spec, trials = _load_easy_dataset(
            "m12",
            DEFAULT_DATA["m12"],
            geometry_equivalence_audit=self.report,
        )
        self.assertEqual(contract["total"], 768)
        self.assertEqual(spec["size_name"], "m12")
        self.assertEqual(len(trials), 768)

    def test_trainer_locks_audit_hash_base_and_no_development_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(
                json.dumps(self.report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = _args(geometry_equivalence_audit=str(path))
            report, record = _load_geometry_equivalence_audit(args)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(record["status"], "PASS")
            args.m8_source_development_data = ["forbidden_audit"]
            with self.assertRaisesRegex(ValueError, "forbidden"):
                _load_geometry_equivalence_audit(args)


if __name__ == "__main__":
    unittest.main()
