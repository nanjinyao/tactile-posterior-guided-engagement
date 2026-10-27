from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MATRICES = (
    (
        "sobol_016",
        "comparison_strict_fingerprint_fuzzy_2x2",
        8,
        False,
        "fuzzy_short",
        "2/6",
    ),
    (
        "sobol_000",
        "comparison_strict_fingerprint_clear_2x2",
        3,
        True,
        "clear_long",
        "5/5",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestM8StrictFingerprintDirectMatrixArtifacts(unittest.TestCase):
    def test_both_frozen_cases_are_complete_and_renderer_invariant(self) -> None:
        for pose_id, stem, aps_size, direct_capture, stratum, ratio in MATRICES:
            with self.subTest(pose_id=pose_id):
                output = (
                    ROOT
                    / "outputs/current/active_comparison/strict_fingerprint_direct_matrices"
                    / pose_id
                )
                manifest = json.loads(
                    (output / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["pose_id"], pose_id)
                self.assertEqual(
                    manifest["method_order"],
                    ["proposed_plus", "proposed", "outward", "direct_first"],
                )
                contract = manifest["contract"]
                self.assertTrue(contract["same_frozen_initial_pose_contact_nuisance_and_sensor_seed"])
                self.assertTrue(contract["same_capture_handoff_and_terminal_verifier"])
                self.assertTrue(contract["direct_first_has_no_retry_or_simulator_reset"])
                self.assertTrue(contract["fingerprint_quality_label_is_post_hoc_descriptive"])
                self.assertTrue(contract["fingerprint_quality_label_is_not_probability_or_significance"])
                self.assertIn(
                    "executed control_direction_deg",
                    contract["summary_direction_metric"],
                )
                self.assertEqual(contract["same_hard_accumulated_active_time_budget_s"], 130.0)
                self.assertTrue(contract["all_substantive_runresult_fields_match_frozen_trial"])

                selection = manifest["source"][
                    "post_hoc_fingerprint_quality_selection"
                ]
                self.assertEqual(selection["pose_id"], pose_id)
                self.assertEqual(selection["stratum"], stratum)
                self.assertEqual(selection["direct_first_observed_count_ratio"], ratio)
                self.assertTrue(selection["post_hoc"])
                self.assertFalse(selection["preregistered"])
                self.assertTrue(selection["no_probability_estimate"])
                self.assertTrue(selection["no_significance_test"])
                selection_report = ROOT / selection["report"]["path"]
                self.assertTrue(selection_report.is_file())
                self.assertEqual(_sha256(selection_report), selection["report"]["sha256"])

                for label in ("video", "poster", "keyframe"):
                    artifact = manifest["artifacts"][label]
                    path = ROOT / artifact["path"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.stat().st_size, artifact["bytes"])
                    self.assertEqual(_sha256(path), artifact["sha256"])
                with Image.open(ROOT / manifest["artifacts"]["poster"]["path"]) as image:
                    self.assertEqual(image.size, (1280, 800))
                with Image.open(ROOT / manifest["artifacts"]["keyframe"]["path"]) as image:
                    self.assertEqual(image.size, (1280, 800))

                fingerprints = set()
                for method, record in manifest["methods"].items():
                    verification = record["verification"]
                    self.assertEqual(verification["substantive_mismatch_count"], 0)
                    self.assertTrue(
                        verification[
                            "all_fields_match_or_are_declared_renderer_outputs"
                        ]
                    )
                    component = record["component_video"]
                    self.assertTrue(component["deleted_after_composition"])
                    self.assertFalse(component["retained_after_composition"])
                    self.assertFalse(
                        (ROOT / component["artifact_before_deletion"]["path"]).exists()
                    )
                    result = record["frozen_result"]
                    if method != "outward":
                        fingerprints.add(result["fingerprint_sha256"])
                        self.assertEqual(result["aps_set_size"], aps_size)
                self.assertEqual(len(fingerprints), 1)
                direct = manifest["methods"]["direct_first"]["frozen_result"]
                self.assertEqual(direct["offline_capture_validated"], direct_capture)
                self.assertEqual(direct["bottom_reached"], direct_capture)
                self.assertEqual((output / f"{stem}.mp4"), ROOT / manifest["artifacts"]["video"]["path"])

    def test_retention_manifest_downgrades_both_one_ray_composites(self) -> None:
        retention = json.loads(
            (ROOT / "outputs/current/VIDEO_RETENTION.json").read_text(encoding="utf-8")
        )
        active = set(retention["active_videos"])
        downgraded = set(retention["downgraded_videos"])
        for pose_id, stem, _, _, _, _ in MATRICES:
            path = f"outputs/current/active_comparison/strict_fingerprint_direct_matrices/{pose_id}/{stem}.mp4"
            self.assertIn(path, downgraded)
            self.assertNotIn(path, active)


if __name__ == "__main__":
    unittest.main()
