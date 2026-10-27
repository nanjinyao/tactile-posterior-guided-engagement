from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "execute_m12_contact_engagement_policy.m"
ENTRY = ROOT / "run_m8_sampling_conical_fingerprints_26pt.m"
RESUME = ROOT / "run_m8_sampling_conical_fingerprints_26pt_resume.m"
DASHBOARD = ROOT / "m8_sampling_topview_dashboard.m"
PLAN = ROOT / "configs" / "m8_printed_cad_conical_sampling_v2.json"
INSTALLATION = ROOT / "configs" / "m8_run_local_sampling_installation_v1.json"
GENERATOR = ROOT / "tools" / "generate_m8_printed_cad_sampling_plan.py"
CANONICAL_PNG = ROOT / "docs" / "m8_printed_cad_sampling_top_view.png"
SIZE_PNG = ROOT / "m8采样规划图.png"
OPERATOR_PNG = ROOT / "采样规划图.png"
PEG_STEP = (
    ROOT
    / "model"
    / "parametric_tooling"
    / "generated"
    / "integrated_flange_peg_m8.step"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM8SamplingCampaignStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = CORE.read_text(encoding="utf-8")
        cls.entry = ENTRY.read_text(encoding="utf-8")
        cls.resume = RESUME.read_text(encoding="utf-8")
        cls.dashboard = DASHBOARD.read_text(encoding="utf-8")
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.installation = json.loads(INSTALLATION.read_text(encoding="utf-8"))

    def test_generated_m8_plan_and_operator_images_are_current(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(GENERATOR), "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("M8 printed-CAD sampling outputs are current", completed.stdout)
        self.assertEqual(SIZE_PNG.read_bytes(), CANONICAL_PNG.read_bytes())
        self.assertEqual(OPERATOR_PNG.read_bytes(), CANONICAL_PNG.read_bytes())

    def test_m8_geometry_route_and_three_contact_states(self) -> None:
        self.assertEqual(self.plan["name"], "m8_printed_cad_conical_sampling_v2")
        expected = {
            "head_outer_diameter": 14.95,
            "socket_mouth_af": 8.1,
            "socket_deep_wall_af": 6.5,
            "socket_chamfer_width": 0.8,
            "socket_chamfer_depth": 0.8,
            "center_through_hole_diameter": 0.5,
            "peg_tip_face_af": 4.8,
            "peg_shaft_af": 6.0,
            "peg_chamfer_radial_setback": 0.6,
            "peg_chamfer_axial_length": 0.6,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(self.plan["geometry_mm"][field], value, places=9)
        self.assertEqual(
            self.plan["execution"]["attempt_indices"],
            [
                4, 14, 23, 9, 18, 3, 15, 8, 21, 2, 17, 7, 13,
                25, 6, 5, 10, 26, 19, 22, 11, 16, 20, 12, 24, 1,
            ],
        )
        self.assertLess(self.plan["execution"]["closed_route_distance_mm"], 75.83)
        self.assertEqual(
            self.plan["execution"]["sentinel_duplicate_execution_separations"],
            [12, 16],
        )
        self.assertEqual(
            self.plan["sampling_design"]["clean_contact_state_counts"],
            {
                "supported_search": 6,
                "double_overhang": 6,
                "outer_partial_support": 4,
            },
        )

    def test_plan_is_bound_to_the_printed_m8_step(self) -> None:
        sources = self.plan["source_files"]
        self.assertEqual(
            sources["printed_m8_integrated_peg_step"]["sha256"],
            sha256(PEG_STEP),
        )
        self.assertEqual(
            sha256(PEG_STEP),
            "22303873354eeb02d82c2492cb3a5d6b8ba73860b31a67804ca501347692f24b",
        )
        self.assertEqual(self.plan["fixture_centre_xy_mm"], [-45.0, 0.0])

    def test_entry_and_resume_are_m8_only_facades(self) -> None:
        self.assertIn("DATASET_M8_CONICAL_FINGERPRINTS_26PT", self.entry)
        self.assertIn("DATASET_M8_CONICAL_FINGERPRINTS_26PT", self.resume)
        self.assertIn("M8_conical_fingerprints_26pt_*", self.resume)
        self.assertIn("m8_run_local_conical_fingerprints_26pt_v2", self.resume)
        self.assertIn("active_peg_step_sha256", self.resume)
        self.assertIn("if nargin == 0 || isempty(previousRunDir)", self.resume)
        for source in (self.entry, self.resume):
            for forbidden in ("movePTP", "sendEEf", "DirectServo", "KST("):
                self.assertNotIn(forbidden, source)

    def test_shared_core_applies_m8_before_loading_the_plan(self) -> None:
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
            "m8_printed_cad_conical_sampling_v2.json",
            "integrated_flange_peg_m8.step",
            "allen_m8_hardware_easy_v2.json",
            "cfg.conical.tip_face_af_mm = 4.8",
            "cfg.sampling.nominal_fixture_center_xy_mm = [-45.0, 0.0]",
            "m8_sampling_topview_dashboard",
            "M8_SAMPLE",
        ):
            self.assertIn(token, profile)

    def test_m8_installation_and_gravity_lineage_are_explicit(self) -> None:
        self.assertEqual(self.installation["size"], "m8")
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

    def test_m8_dashboard_is_allowed_and_directservo_stays_ui_free(self) -> None:
        self.assertIn("'sampling_size_label', 'M8'", self.dashboard)
        self.assertIn("'m8_sampling_topview_dashboard'", self.core)
        cone = self.core[
            self.core.index("function [ctx, cone] = runConicalFingerprint") :
            self.core.index("function thresholds = conicalLowForceThresholds")
        ]
        for forbidden in (
            "callSamplingTopview",
            "m8_sampling_topview_dashboard",
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
