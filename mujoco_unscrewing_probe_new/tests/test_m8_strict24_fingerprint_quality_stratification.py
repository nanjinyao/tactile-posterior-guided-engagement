from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.build_m8_strict24_fingerprint_quality_stratification import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    LOCKED_RULES,
    _in_clear_long,
    _in_fuzzy_short,
    build_report,
    write_artifact,
)
from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


EXPECTED_FUZZY = [
    "sobol_002",
    "sobol_008",
    "sobol_009",
    "sobol_015",
    "sobol_016",
    "sobol_023",
]
EXPECTED_CLEAR = [
    "sobol_000",
    "sobol_005",
    "sobol_007",
    "sobol_010",
    "sobol_013",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(
    (DEFAULT_SOURCE / "report.json").is_file(), "strict-24 source not generated"
)
class TestM8Strict24FingerprintQualityStratification(unittest.TestCase):
    def setUp(self) -> None:
        self.report = build_report(DEFAULT_SOURCE)

    def test_locked_rules_and_inclusive_boundaries(self) -> None:
        self.assertEqual(
            LOCKED_RULES["fuzzy_short"]["max_proposed_transport_mm"]["value"],
            1.2,
        )
        self.assertEqual(
            LOCKED_RULES["clear_long"]["circular_concentration"]["value"],
            0.995,
        )
        fuzzy_boundary = {
            "aps_set_size": 5,
            "circular_concentration": 0.2,
            "ensemble_top1_span_deg": 15.0,
            "proposed_transport_mm": 1.2,
            "proposed_plus_transport_mm": 1.2,
        }
        clear_boundary = {
            "aps_set_size": 3,
            "circular_concentration": 0.995,
            "ensemble_top1_span_deg": 7.5,
            "proposed_transport_mm": 2.7,
            "proposed_plus_transport_mm": 2.7,
        }
        self.assertTrue(_in_fuzzy_short(fuzzy_boundary))
        self.assertTrue(_in_clear_long(clear_boundary))

    def test_members_and_observed_strict_counts_are_frozen(self) -> None:
        fuzzy = self.report["strata"]["fuzzy_short"]
        clear = self.report["strata"]["clear_long"]
        self.assertEqual([row["pose_id"] for row in fuzzy["members"]], EXPECTED_FUZZY)
        self.assertEqual([row["pose_id"] for row in clear["members"]], EXPECTED_CLEAR)
        self.assertEqual(fuzzy["member_count"], 6)
        self.assertEqual(fuzzy["direct_first_strict_bottom_success_count"], 2)
        self.assertEqual(fuzzy["observed_count_ratio"], "2/6")
        self.assertEqual(clear["member_count"], 5)
        self.assertEqual(clear["direct_first_strict_bottom_success_count"], 5)
        self.assertEqual(clear["observed_count_ratio"], "5/5")

    def test_member_success_is_recomputed_with_full_strict_contract(self) -> None:
        member_rows = {
            row["pose_id"]: row
            for name in ("fuzzy_short", "clear_long")
            for row in self.report["strata"][name]["members"]
        }
        for pose_id, row in member_rows.items():
            trial = json.loads(
                (DEFAULT_SOURCE / "trials" / f"{pose_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            direct = trial["methods"]["direct_first"]
            self.assertEqual(
                row["direct_first_strict_bottom_success"],
                _strict_bottom_success(direct),
            )
        contract = self.report["analysis_design"]["strict_success_contract"]
        self.assertTrue(contract["capture_alone_is_not_success"])
        self.assertIn("bottom_reached", contract["requires"])

    def test_source_hashes_and_post_hoc_boundary_are_explicit(self) -> None:
        source = self.report["source"]
        self.assertEqual(
            source["manifest"]["sha256"], _sha256(DEFAULT_SOURCE / "manifest.json")
        )
        self.assertEqual(
            source["report"]["sha256"], _sha256(DEFAULT_SOURCE / "report.json")
        )
        design = self.report["analysis_design"]
        self.assertEqual(design["type"], "post_hoc_descriptive_stratification")
        self.assertFalse(design["preregistered"])
        self.assertTrue(design["no_probability_estimate"])
        self.assertTrue(design["no_significance_test"])
        self.assertIn("not probability estimates", self.report["claim_boundary"])

    def test_writer_is_deterministic_and_published_artifacts_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = write_artifact(DEFAULT_SOURCE, output)
            first_report_bytes = (output / "report.json").read_bytes()
            first_summary_bytes = (output / "summary.md").read_bytes()
            second = write_artifact(DEFAULT_SOURCE, output)
            self.assertEqual(first, second)
            self.assertEqual(first_report_bytes, (output / "report.json").read_bytes())
            self.assertEqual(first_summary_bytes, (output / "summary.md").read_bytes())

        self.assertTrue((DEFAULT_OUTPUT / "report.json").is_file())
        self.assertTrue((DEFAULT_OUTPUT / "summary.md").is_file())
        self.assertEqual(
            json.loads((DEFAULT_OUTPUT / "report.json").read_text(encoding="utf-8")),
            self.report,
        )


if __name__ == "__main__":
    unittest.main()
