from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "execute_m12_contact_engagement_policy.m"
ENTRY = ROOT / "run_m10_sampling_conical_fingerprints_26pt.m"
RESUME = ROOT / "run_m10_sampling_conical_fingerprints_26pt_resume.m"
DASHBOARD = ROOT / "m10_sampling_topview_dashboard.m"
SHARED_DASHBOARD = ROOT / "m12_sampling_topview_dashboard.m"
PLAN = ROOT / "configs" / "m10_printed_cad_conical_sampling_v2.json"
INSTALLATION = ROOT / "configs" / "m10_run_local_sampling_installation_v1.json"
GENERATOR = ROOT / "tools" / "generate_m10_printed_cad_sampling_plan.py"
CANONICAL_PNG = ROOT / "docs" / "m10_printed_cad_sampling_top_view.png"
OPERATOR_PNG = ROOT / "m10采样规划图.png"
PEG_STEP = (
    ROOT
    / "model"
    / "parametric_tooling"
    / "generated"
    / "integrated_flange_peg_m10.step"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM10SamplingCampaignStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = CORE.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.resume = RESUME.read_text(encoding="utf-8")
        cls.dashboard = DASHBOARD.read_text(encoding="utf-8")
        cls.shared_dashboard = SHARED_DASHBOARD.read_text(encoding="utf-8")
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.installation = json.loads(INSTALLATION.read_text(encoding="utf-8"))

    def test_generated_m10_plan_is_current_and_operator_image_is_m10(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("M10 printed-CAD sampling outputs are current", completed.stdout)
        self.assertEqual(OPERATOR_PNG.read_bytes(), CANONICAL_PNG.read_bytes())

    def test_m10_geometry_route_and_three_contact_states(self) -> None:
        self.assertEqual(self.plan["name"], "m10_printed_cad_conical_sampling_v2")
        expected = {
            "head_outer_diameter": 18.975,
            "socket_mouth_af": 10.54,
            "socket_deep_wall_af": 8.54,
            "socket_chamfer_width": 1.0,
            "socket_chamfer_depth": 1.0,
            "center_through_hole_diameter": 0.5,
            "peg_tip_face_af": 6.4,
            "peg_shaft_af": 8.0,
            "peg_chamfer_radial_setback": 0.8,
            "peg_chamfer_axial_length": 0.8,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(self.plan["geometry_mm"][field], value, places=9)
        self.assertEqual(
            self.plan["execution"]["attempt_indices"],
            [4, 14, 10, 5, 6, 25, 24, 12, 20, 16, 11, 22, 19,
             26, 23, 9, 18, 3, 15, 8, 21, 2, 17, 7, 13, 1],
        )
        self.assertLess(self.plan["execution"]["closed_route_distance_mm"], 98.01)
        self.assertEqual(
            self.plan["execution"]["sentinel_duplicate_execution_separations"],
            [20, 12],
        )
        self.assertEqual(
            self.plan["sampling_design"]["clean_contact_state_counts"],
            {
                "supported_search": 6,
                "double_overhang": 6,
                "outer_partial_support": 4,
            },
        )
        outer = [
            point
            for point in self.plan["points"]
            if point["partition"] == "clean_candidate"
            and point["contact_state"] == "outer_partial_support"
        ]
        self.assertEqual(len(outer), 4)
        self.assertTrue(
            all(
                point["nominal_touchdown_geometry"]["outer_overhang_fraction"]
                > 0.5
                and point["nominal_touchdown_geometry"]["mouth_overlap_fraction"]
                <= 1e-5
                and point["nominal_touchdown_geometry"]["head_top_support_fraction"]
                > 0.2
                for point in outer
            )
        )

    def test_plan_is_bound_to_the_printed_m10_step(self) -> None:
        sources = self.plan["source_files"]
        self.assertEqual(
            sources["printed_m10_integrated_peg_step"]["sha256"],
            sha256(PEG_STEP),
        )
        self.assertEqual(
            sha256(PEG_STEP),
            "17381968212daa2b7758dcf3795df32b3a4e687b42c482d3550111e03ae8daa2",
        )
        self.assertEqual(self.plan["fixture_centre_xy_mm"], [0.0, 0.0])

    def test_entry_and_resume_are_m10_only_facades(self) -> None:
        self.assertIn("DATASET_M10_CONICAL_FINGERPRINTS_26PT", self.entry)
        self.assertIn("DATASET_M10_CONICAL_FINGERPRINTS_26PT", self.resume)
        self.assertIn("M10_conical_fingerprints_26pt_*", self.resume)
        self.assertIn("m10_run_local_conical_fingerprints_26pt_v2", self.resume)
        self.assertIn("active_peg_step_sha256", self.resume)
        self.assertIn("if nargin == 0 || isempty(previousRunDir)", self.resume)
        for source in (self.entry, self.resume):
            for forbidden in ("movePTP", "sendEEf", "DirectServo", "KST("):
                self.assertNotIn(forbidden, source)

    def test_shared_core_applies_m10_before_loading_the_plan(self) -> None:
        setup = self.core[
            self.core.index("cfg = makeConfig") :
            self.core.index("runStamp = char(datetime")
        ]
        self.assertLess(
            setup.index("applySamplingSizeProfile"),
            setup.index("loadM12SamplingPlan"),
        )
        profile_start = self.core.index(
            "function cfg = applySamplingSizeProfile"
        )
        profile_end = self.core.index(
            "\nfunction plan = makePlan", profile_start
        )
        profile = self.core[profile_start:profile_end]
        for token in (
            "m10_printed_cad_conical_sampling_v2.json",
            "integrated_flange_peg_m10.step",
            "allen_m10_hardware_easy_v2.json",
            "cfg.conical.tip_face_af_mm = 6.4",
            "cfg.sampling.nominal_fixture_center_xy_mm = [0.0, 0.0]",
            "m10_sampling_topview_dashboard",
            "M10_SAMPLE",
        ):
            self.assertIn(token, profile)

    def test_m10_installation_and_gravity_lineage_are_explicit(self) -> None:
        self.assertEqual(self.installation["size"], "m10")
        self.assertEqual(
            self.installation["active_tcp"]["translation_mm"],
            [0.0, 0.0, 92.0],
        )
        self.assertFalse(
            self.installation["fixture_contract"]["as_built_metrology_claimed"]
        )
        self.assertIn("installed_tool_gravity_policy", self.core)
        self.assertIn("baseline_samples', 30", self.core)
        self.assertIn("raw Bota six-axis samples remain unchanged", self.core)

    def test_m10_topview_facade_pins_size_and_shared_geometry(self) -> None:
        self.assertIn("'M10'", self.dashboard)
        self.assertIn("m12_sampling_topview_dashboard", self.dashboard)
        for token in ("18.975", "10.54", "8.54", "6.4", "8.0"):
            self.assertIn(token, self.shared_dashboard)
        for token in ("+X_F", "+Y_F", "+Z_F", "measured endpoint", "command path"):
            self.assertIn(token, self.shared_dashboard)

    def test_directservo_loop_has_no_dashboard_or_extra_queries(self) -> None:
        cone = self.core[
            self.core.index("function [ctx, cone] = runConicalFingerprint") :
            self.core.index("function thresholds = conicalLowForceThresholds")
        ]
        for forbidden in (
            "callSamplingTopview",
            "m10_sampling_topview_dashboard",
            "m12_sampling_topview_dashboard",
            "drawnow",
            "timer(",
            "getEEFPos",
            "getJointsPos",
            "rossubscriber",
        ):
            self.assertNotIn(forbidden, cone)


if __name__ == "__main__":
    unittest.main()
