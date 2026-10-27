from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.audit_clocking_clean_multisize_model import (
    DEFAULT_GEOMETRY_AUDIT,
    DEFAULT_MODEL,
    build_report,
)


class TestClockingCleanMultisizeModelAudit(unittest.TestCase):
    def test_live_candidate_has_clean_lineage(self) -> None:
        report = build_report(DEFAULT_MODEL, DEFAULT_GEOMETRY_AUDIT)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["leakage_checks"],
            {
                "development_datasets_consumed": 0,
                "final_audit_samples_seen_during_fitting": 0,
                "captured_or_incomplete_direction_labels_used": 0,
                "group_intersections": 0,
            },
        )
        self.assertEqual(
            {
                size: item["valid_full_fingerprints"]
                for size, item in report["training_data"].items()
            },
            {"m8": 675, "m10": 624, "m12": 602},
        )
        self.assertEqual(len(report["checkpoints"]), 3)

    def test_tampered_development_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            for name in (
                "adaptation_report.json",
                "split_manifest.json",
                "ensemble_interface_contract.json",
            ):
                source = DEFAULT_MODEL / name
                target = copy / name
                target.write_bytes(source.read_bytes())
            adaptation_path = copy / "adaptation_report.json"
            adaptation = json.loads(adaptation_path.read_text())
            adaptation["m8_source_development_datasets_consumed"] = [
                "forbidden-failed-audit"
            ]
            adaptation_path.write_text(json.dumps(adaptation))
            with self.assertRaisesRegex(ValueError, "development"):
                build_report(copy, DEFAULT_GEOMETRY_AUDIT)


if __name__ == "__main__":
    unittest.main()
