from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import imageio.v2 as imageio
from PIL import Image

from src.render_m8_strict_entropy_topview_explainer import visible_text


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003"
)
OUT = (
    ROOT
    / "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003_en"
)
SOURCE_VIDEO = (
    "outputs/current/active_comparison/strict_entropy_grid_topview_sobol_003/"
    "m8_strict_entropy_grid_topview_sobol_003.mp4"
)


def _manifest() -> dict:
    return json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))


class TestM8StrictEntropyTopviewEnglish(unittest.TestCase):
    def test_english_visible_text_catalog_has_no_cjk(self) -> None:
        rendered_text = "\n".join(visible_text("en").values())
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", rendered_text))

    def test_manifest_proves_render_only_reuse_of_same_replay(self) -> None:
        manifest = _manifest()
        self.assertEqual(manifest["language"], "en")
        self.assertEqual(manifest["visible_text_language"], "en")
        replay = manifest["replay_contract"]
        self.assertFalse(replay["physics_rollout_rerun_for_this_render"])
        self.assertTrue(replay["render_only_reuse"])
        self.assertEqual(
            replay["reused_replay_from"],
            SOURCE.relative_to(ROOT).as_posix(),
        )
        self.assertTrue(replay["frozen_result_verification"]["exact_substantive_match"])
        self.assertEqual(
            replay["source_trace"]["sha256"],
            manifest["artifacts"]["trace"]["sha256"],
        )
        self.assertEqual(
            replay["source_result"]["sha256"],
            manifest["artifacts"]["result"]["sha256"],
        )
        self.assertEqual(replay["source_trace"]["path"], (SOURCE / "trace.csv").relative_to(ROOT).as_posix())
        self.assertEqual(replay["source_result"]["path"], (SOURCE / "result.json").relative_to(ROOT).as_posix())
        self.assertNotIn("video", replay["source_trace"])
        self.assertNotIn("video", replay["source_result"])

    def test_english_assets_are_self_contained_and_do_not_reference_chinese_video(self) -> None:
        manifest = _manifest()
        video = manifest["artifacts"]["video"]
        self.assertEqual(
            video["path"],
            (OUT / "m8_strict_entropy_grid_topview_sobol_003_en.mp4")
            .relative_to(ROOT)
            .as_posix(),
        )
        self.assertNotIn(SOURCE_VIDEO, json.dumps(manifest, sort_keys=True))
        for key in ("video", "poster", "planar_map", "mujoco_topview_base"):
            artifact = manifest["artifacts"][key]
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file(), key)
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertGreater(artifact["bytes"], 0)

        with Image.open(OUT / "poster.png") as poster:
            self.assertEqual(poster.size, (1600, 900))
        with Image.open(OUT / "planar_map.png") as planar:
            self.assertEqual(planar.size, (1600, 900))

        reader = imageio.get_reader(ROOT / video["path"])
        try:
            self.assertEqual(reader.count_frames(), manifest["video"]["frame_count"])
            last_frame = reader.get_data(manifest["video"]["frame_count"] - 1)
        finally:
            reader.close()
        self.assertEqual(last_frame.shape[:2], (900, 1600))


if __name__ == "__main__":
    unittest.main()
