from __future__ import annotations

import json
from pathlib import Path
import unittest

from PIL import Image

from src.plot_m8_bilingual_core_paper_assets import (
    DEFAULT_DATASET_ROOT,
    TEXT,
    _contract_roles,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/current/datasets/cross_size_s4b_adaptation"
FIGURES = ROOT / "outputs/current/media/figures/paper"
MANIFEST = FIGURES / "m8_bilingual_core_paper_assets_manifest.json"


class BilingualCorePaperAssetTests(unittest.TestCase):
    def test_stable_information_boundary_labels(self) -> None:
        self.assertEqual(DEFAULT_DATASET_ROOT.resolve(), DATA.resolve())
        for language in ("cn", "en"):
            labels = TEXT[language]
            self.assertIn("state + direction", labels["hardware_lines"])
            self.assertIn("shared", labels["hardware_lines"])
            self.assertIn("camera", labels["hardware_lines"])
            self.assertIn("wrench + robot pose", labels["measurements"])
            self.assertIn(
                "contact" if language == "en" else "接触",
                labels["posterior"],
            )
            self.assertIn("posterior", labels["head_ft"])
            self.assertIn("99%", labels["residual_support"])

    def test_english_labels_contain_no_cjk(self) -> None:
        for key, label in TEXT["en"].items():
            with self.subTest(key=key):
                self.assertFalse(
                    any("\u3400" <= char <= "\u9fff" for char in label),
                    msg=f"English label {key!r} contains CJK text: {label!r}",
                )

    def test_s4_roles_are_drawn_separately(self) -> None:
        for size in ("m10", "m12"):
            contract = json.loads(
                (DATA / size / "dataset_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                _contract_roles(contract),
                ("train", "validation", "calibration"),
            )
            orbit_sets = {
                role: {
                    task["symmetry_orbit_id"]
                    for task in contract["tasks"][role]
                }
                for role in _contract_roles(contract)
            }
            self.assertFalse(orbit_sets["train"] & orbit_sets["validation"])
            self.assertFalse(orbit_sets["train"] & orbit_sets["calibration"])
            self.assertFalse(orbit_sets["validation"] & orbit_sets["calibration"])

    def test_bilingual_sampling_figures_are_current_and_nonempty(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        contract_paths = {
            Path(item["path"]).name
            for item in manifest["inputs"]["dataset_contracts"]
        }
        self.assertEqual(contract_paths, {"dataset_contract.json"})
        for language in ("cn", "en"):
            path = FIGURES / f"cross_size_finetune_c6_sampling_{language}.png"
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, 2000)
                self.assertGreaterEqual(image.height, 1000)
                self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
