from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT.parent / "group_meeting_trace"
ZH = TRACE / "unscrewing_contact_state_pipeline_slides_0709.md"
EN = TRACE / "unscrewing_contact_state_pipeline_slides_0709_en.md"

REFERENCE_PATTERN = re.compile(
    r'(?:src|href)="([^"]+)"|!\[[^\]]*\]\(([^)]+)\)'
)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

MEDIA_REPLACEMENTS = {
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_shortcut_posterior_geometry_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_shortcut_posterior_geometry_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "cross_size_finetune_c6_sampling_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "cross_size_finetune_c6_sampling_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_hardware_finetune_plan_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_hardware_finetune_plan_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_three_layer_control_flow_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_three_layer_control_flow_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_sobol016_proposed_plus_audit_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_sobol016_proposed_plus_audit_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_r3_paired_ablation_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_r3_paired_ablation_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_paired_shortcut_ablation_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_paired_shortcut_ablation_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "cross_size_transfer_result_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "cross_size_transfer_result_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
    "m8_sampling_positions_contract_cn.png":
        "../mujoco_unscrewing_probe_new/outputs/current/media/figures/paper/"
        "m8_sampling_positions_contract_en.png",
    "../mujoco_unscrewing_probe_new/outputs/current/active_comparison/"
    "strict_entropy_grid_topview_sobol_003/"
    "m8_strict_entropy_grid_topview_sobol_003.mp4":
        "../mujoco_unscrewing_probe_new/outputs/current/active_comparison/"
        "strict_entropy_grid_topview_sobol_003_en/"
        "m8_strict_entropy_grid_topview_sobol_003_en.mp4",
    "../mujoco_unscrewing_probe_new/outputs/current/active_comparison/"
    "strict_entropy_grid_topview_sobol_003/poster.png":
        "../mujoco_unscrewing_probe_new/outputs/current/active_comparison/"
        "strict_entropy_grid_topview_sobol_003_en/poster.png",
}


def _slide_count(text: str) -> int:
    return len(text.split("\n---\n")) - 1


def _references(text: str) -> list[str]:
    return [
        next(value for value in match.groups() if value is not None)
        for match in REFERENCE_PATTERN.finditer(text)
    ]


def _local_paths(markdown: Path, references: list[str]) -> list[Path]:
    return [
        (markdown.parent / value.split("#", 1)[0]).resolve()
        for value in references
        if value and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value)
    ]


class TestGroupMeetingBilingualMarkdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.zh = ZH.read_text(encoding="utf-8")
        cls.en = EN.read_text(encoding="utf-8")

    def test_english_and_chinese_decks_have_the_same_structure(self) -> None:
        self.assertEqual(_slide_count(self.en), _slide_count(self.zh))
        self.assertEqual(
            len(re.findall(r"(?m)^# ", self.en)),
            len(re.findall(r"(?m)^# ", self.zh)),
        )
        self.assertEqual(self.en.count("$$") // 2, self.zh.count("$$") // 2)

    def test_english_deck_contains_no_cjk(self) -> None:
        match = CJK_PATTERN.search(self.en)
        self.assertIsNone(
            match,
            f"CJK character remains at offset {match.start() if match else 'n/a'}",
        )

    def test_all_local_media_references_exist(self) -> None:
        references = _references(self.en)
        self.assertTrue(references)
        missing = [
            path for path in _local_paths(EN, references) if not path.is_file()
        ]
        self.assertEqual(missing, [])

    def test_media_order_matches_with_only_the_audited_english_substitutions(self) -> None:
        expected = [
            MEDIA_REPLACEMENTS.get(reference, reference)
            for reference in _references(self.zh)
        ]
        self.assertEqual(_references(self.en), expected)

    def test_english_deck_never_references_chinese_entropy_media(self) -> None:
        self.assertNotIn(
            "strict_entropy_grid_topview_sobol_003/"
            "m8_strict_entropy_grid_topview_sobol_003.mp4",
            self.en,
        )
        self.assertNotIn(
            "strict_entropy_grid_topview_sobol_003/poster.png",
            self.en,
        )
        self.assertNotIn("_cn.png", self.en)
        self.assertIn(
            "strict_entropy_grid_topview_sobol_003_en/"
            "m8_strict_entropy_grid_topview_sobol_003_en.mp4",
            self.en,
        )


if __name__ == "__main__":
    unittest.main()
