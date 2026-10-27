import json
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "outputs" / "current"


class TestM8SamplingPositionMap(unittest.TestCase):
    def test_map_inventory_matches_current_manifests(self) -> None:
        inventory = json.loads(
            (CURRENT / "reports" / "m8_all_conical_sampling_positions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(inventory["total_dynamic_rollouts"], 1674)
        self.assertEqual(inventory["total_unique_xy_positions"], 651)
        self.assertEqual(len(inventory["datasets"]), 8)
        by_key = {dataset["key"]: dataset for dataset in inventory["datasets"]}
        self.assertEqual(by_key["p1v3_m8_targeted_direction_training"]["rollouts"], 288)
        self.assertEqual(by_key["p1v5_m8_final_untouched_direction_audit"]["unique_xy_positions"], 60)
        self.assertIn("no reuse", by_key["p1v5_m8_final_untouched_direction_audit"]["role"])

    def test_map_image_is_large_and_nonblank(self) -> None:
        path = CURRENT / "media" / "figures" / "m8_all_conical_sampling_positions_mujoco.png"
        with Image.open(path) as image:
            self.assertEqual(image.size, (1900, 1200))
            colors = image.resize((190, 120)).convert("RGB").getcolors(maxcolors=190 * 120)
        self.assertIsNotNone(colors)
        self.assertGreater(len(colors), 100)


if __name__ == "__main__":
    unittest.main()
