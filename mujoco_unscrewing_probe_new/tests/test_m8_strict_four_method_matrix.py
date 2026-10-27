from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from PIL import Image

from src.run_m8_strict_complete_loop_audit import _strict_bottom_success


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/active_comparison/strict_four_methods"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@unittest.skipUnless((OUT / "manifest.json").is_file(), "strict matrix not generated")
class TestM8StrictFourMethodMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (OUT / "manifest.json").read_text(encoding="utf-8")
        )
        source = ROOT / self.manifest["source"]["frozen_trial"]["path"]
        self.frozen_trial = json.loads(source.read_text(encoding="utf-8"))

    def test_selected_case_and_claim_boundary_are_explicit(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["pose_id"], "sobol_008")
        self.assertEqual(manifest["task"], self.frozen_trial["task"])
        self.assertEqual(
            manifest["method_order"],
            ["proposed", "proposed_plus", "outward", "entropy_grid"],
        )
        contract = manifest["contract"]
        self.assertTrue(contract["selected_mechanism_example_not_method_ranking"])
        self.assertTrue(contract["same_frozen_initial_pose_contact_nuisance_and_sensor_seed"])
        self.assertTrue(contract["same_capture_handoff_and_terminal_verifier"])
        self.assertEqual(contract["same_hard_accumulated_active_time_budget_s"], 130.0)
        self.assertFalse(contract["online_true_center_or_contact_tags"])
        self.assertIn("not a ranking", manifest["claim_boundary"])

    def test_each_rendered_result_matches_the_frozen_strict_trial(self) -> None:
        fields = (
            "result",
            "fingerprint_sha256",
            "time_to_inner_wall_s",
            "time_to_bottom_s",
            "total_active_time_s",
            "hard_stop",
            "hard_stop_reason",
            "terminal_verification_mode",
            "inner_wall_confirmed",
            "offline_capture_validated",
            "bottom_reached",
            "fallback_budget_extension_s",
        )
        for method_name in self.manifest["method_order"]:
            with self.subTest(method=method_name):
                rendered = self.manifest["methods"][method_name]
                frozen = self.frozen_trial["methods"][method_name]
                self.assertTrue(rendered["verification"]["all_fields_match"])
                self.assertTrue(
                    rendered["verification"]["strict_bottom_success"]["match"]
                )
                self.assertTrue(_strict_bottom_success(frozen))
                for field in fields:
                    actual = rendered["frozen_result"][field]
                    expected = frozen[field]
                    if isinstance(expected, float):
                        self.assertAlmostEqual(actual, expected, delta=1.0e-9)
                    else:
                        self.assertEqual(actual, expected)

    def test_composite_media_are_retained_and_components_are_removed(self) -> None:
        artifacts = self.manifest["artifacts"]
        for label in ("video", "poster", "keyframe"):
            with self.subTest(artifact=label):
                record = artifacts[label]
                path = ROOT / record["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(_sha256(path), record["sha256"])

        with Image.open(ROOT / artifacts["poster"]["path"]) as image:
            self.assertEqual(image.size, (1280, 800))
        with Image.open(ROOT / artifacts["keyframe"]["path"]) as image:
            self.assertEqual(image.size, (1280, 800))

        for method_name in ("proposed", "outward", "entropy_grid"):
            component = self.manifest["methods"][method_name]["component_video"]
            self.assertTrue(component["deleted_after_composition"])
            self.assertFalse(component["retained_after_composition"])
            path = ROOT / component["artifact_before_deletion"]["path"]
            self.assertFalse(path.exists())

        retention = json.loads(
            (ROOT / "outputs/current/VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        self.assertIn(artifacts["video"]["path"], retention["downgraded_videos"])
        self.assertNotIn(artifacts["video"]["path"], retention["active_videos"])


if __name__ == "__main__":
    unittest.main()
