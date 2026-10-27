from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "outputs/current/active_comparison/m8_retry3_selected_media"
AUDIT = ROOT / "outputs/current/evaluations/m8_five_method_retry3_r3_closed_loop"
CASES = (
    ("sobol_016", "comparison_fuzzy_retry3_2x2", "matrix", (1280, 800)),
    ("sobol_000", "comparison_clear_retry3_2x2", "matrix", (1280, 800)),
    ("sobol_005", "comparison_improved4_vs_jin3_retry3", "direct", (1280, 720)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestM8Retry3SelectedMedia(unittest.TestCase):
    def test_selected_media_are_frozen_rerenders(self) -> None:
        report_hash = _sha256(AUDIT / "report.json")
        for pose_id, stem, mode, resolution in CASES:
            with self.subTest(pose_id=pose_id, mode=mode):
                output = MEDIA / pose_id
                manifest = json.loads(
                    (output / f"{stem}_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["pose_id"], pose_id)
                self.assertEqual(manifest["mode"], mode)
                self.assertEqual(manifest["source_audit_report"]["sha256"], report_hash)
                contract = manifest["contract"]
                self.assertTrue(contract["same_frozen_task_nuisance_sensor_seed"])
                self.assertTrue(contract["same_first_raw_fingerprint_for_fingerprint_methods"])
                self.assertEqual(contract["direct_ray_attempts_max"], 3)
                self.assertTrue(contract["direct_retry_is_physical_same_rollout"])
                self.assertEqual(contract["shared_post_capture_recoveries_max"], 3)
                self.assertEqual(contract["simulator_resets"], 0)
                self.assertTrue(contract["selected_mechanism_not_distribution_ranking"])
                self.assertTrue(contract["online_true_center_or_contact_tags"] is False)
                for verification in manifest["verifications"].values():
                    self.assertEqual(verification["mismatch_count"], 0)
                    self.assertTrue(verification["render_fields_valid"])
                for label in ("video", "poster", "keyframe"):
                    artifact = manifest["artifacts"][label]
                    path = ROOT / artifact["path"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.stat().st_size, artifact["bytes"])
                    self.assertEqual(_sha256(path), artifact["sha256"])
                with Image.open(ROOT / manifest["artifacts"]["poster"]["path"]) as image:
                    self.assertEqual(image.size, resolution)
                with Image.open(ROOT / manifest["artifacts"]["keyframe"]["path"]) as image:
                    self.assertEqual(image.size, resolution)

    def test_direct_video_preserves_distinct_display_contracts(self) -> None:
        path = MEDIA / "sobol_005/comparison_improved4_vs_jin3_retry3_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["methods"],
            ["improved_direct_retry3", "jin_direct_retry3"],
        )
        improved = manifest["results"]["improved_direct_retry3"]
        jin = manifest["results"]["jin_direct_retry3"]
        self.assertEqual(improved["direct_successful_attempt"], 1)
        self.assertEqual(jin["direct_successful_attempt"], 3)
        self.assertTrue(improved["bottom_reached"])
        self.assertFalse(jin["bottom_reached"])
        self.assertTrue(jin["hard_stop"])


if __name__ == "__main__":
    unittest.main()
