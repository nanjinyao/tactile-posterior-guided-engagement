from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.audit_m8_finite_outward_boundary import (
    BOUNDARY_SAMPLE_COUNT,
    EFFECTIVE_CAPTURE_RADIUS_M,
    OUTPUT,
    PATH_SAMPLE_COUNT,
    SPIRAL_PITCH_M,
    build_report,
    render_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = OUTPUT / "report.json"
SUMMARY_PATH = OUTPUT / "summary.md"


class TestM8FiniteOutwardBoundaryAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not REPORT_PATH.is_file() or not SUMMARY_PATH.is_file():
            raise AssertionError(
                "run python3 -m src.audit_m8_finite_outward_boundary first"
            )
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_artifact_is_exactly_reproducible(self) -> None:
        rebuilt = build_report()
        self.assertEqual(rebuilt, self.report)
        self.assertEqual(render_summary(rebuilt), SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_shared_strict_geometry_and_pitch_bound_are_exact(self) -> None:
        geometry = self.report["geometry"]
        inputs = self.report["inputs"]
        self.assertEqual(inputs["path_sample_count"], PATH_SAMPLE_COUNT)
        self.assertEqual(inputs["boundary_sample_count"], BOUNDARY_SAMPLE_COUNT)
        self.assertEqual(
            inputs["path_generator"],
            "src.render_m8_search_comparison.archimedean_path",
        )
        self.assertEqual(
            geometry["strict_outward_pitch_m_per_rev"], SPIRAL_PITCH_M
        )
        self.assertEqual(
            geometry["effective_capture_radius_m"],
            EFFECTIVE_CAPTURE_RADIUS_M,
        )
        self.assertEqual(
            geometry["finite_path_max_radius_m"],
            geometry["visual_support_radius_m"]
            - geometry["effective_capture_radius_m"],
        )
        self.assertEqual(
            geometry["adjacent_turn_pitch_bound_m_per_rev"],
            2.0 * EFFECTIVE_CAPTURE_RADIUS_M,
        )
        self.assertEqual(
            geometry["pitch_to_adjacent_turn_bound_ratio"], 0.8
        )
        self.assertTrue(
            self.report["conclusions"][
                "adjacent_turn_pitch_bound_satisfied"
            ]
        )

    def test_high_resolution_boundary_gap_values_are_frozen(self) -> None:
        distance = self.report["boundary_nearest_distance"]
        self.assertEqual(distance["at_or_within_r_eff_count"], 1)
        self.assertEqual(distance["beyond_r_eff_count"], 65_535)
        self.assertEqual(
            distance["fraction_beyond_r_eff"], 65_535 / 65_536
        )
        self.assertAlmostEqual(
            distance["endpoint_aligned_sample_distance_m"],
            0.00025000000000000553,
            places=18,
        )
        self.assertAlmostEqual(
            distance["median_m"], 0.00044505936359602724, places=18
        )
        self.assertAlmostEqual(
            distance["worst_m"], 0.0006420530819360968, places=18
        )
        self.assertEqual(distance["worst_sample_index"], 1_295)
        self.assertEqual(distance["worst_nearest_segment_index"], 10_986)
        self.assertGreater(
            distance["worst_m"], EFFECTIVE_CAPTURE_RADIUS_M
        )
        self.assertGreater(
            self.report["finite_path"]["suffix_globality_margin_m"], 0.0
        )
        self.assertTrue(
            self.report["finite_path"][
                "suffix_is_proven_global_for_boundary_samples"
            ]
        )

    def test_no_complete_domain_certificate_is_claimed(self) -> None:
        conclusions = self.report["conclusions"]
        provenance = self.report["provenance_and_change_boundary"]
        self.assertFalse(conclusions["full_outer_boundary_within_r_eff"])
        self.assertFalse(conclusions["complete_visual_disk_coverage_certificate"])
        self.assertFalse(provenance["policy_changed"])
        self.assertFalse(provenance["trial_files_changed"])
        self.assertFalse(provenance["simulation_rerun"])
        self.assertIn(
            "empirical search comparator", self.report["claim_boundary"]
        )
        self.assertIn(
            "not a complete-domain coverage certificate",
            self.report["claim_boundary"],
        )

    def test_audit_has_no_scipy_dependency(self) -> None:
        source = (
            ROOT / "src" / "audit_m8_finite_outward_boundary.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("scipy", source.lower())


if __name__ == "__main__":
    unittest.main()
