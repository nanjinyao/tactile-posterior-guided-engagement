from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT.parent / "group_meeting_trace"
PAPER = TRACE / "unscrewing_contact_funnels_ral_working_paper.md"
PAPER_FIGURES = ROOT / "outputs/current/media/figures/paper"
PAPER_FIGURE_MANIFEST = PAPER_FIGURES / "m8_bilingual_core_paper_assets_manifest.json"


def _paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


def _slide_blocks(text: str) -> list[str]:
    """Return content pages while excluding the YAML front matter."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("Marp front matter must start on line 1")
    try:
        front_matter_end = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration as error:
        raise AssertionError("Marp front matter is not closed") from error
    body = "\n".join(lines[front_matter_end + 1 :]).strip()
    return [block.strip() for block in re.split(r"(?m)^---$", body) if block.strip()]


class TestRalWorkingPaperMarkdown(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(PAPER.is_file())
        self.text = _paper_text()

    def test_is_exactly_eight_portrait_working_pages(self) -> None:
        self.assertEqual(len(_slide_blocks(self.text)), 8)
        self.assertRegex(self.text, r"(?m)^size:\s*17:22\s*$")
        self.assertIn("grid-template-columns: 1fr 1fr", self.text)
        self.assertIn(
            "NOT AN OFFICIAL IEEE RA-L SUBMISSION TEMPLATE",
            self.text,
        )

    def test_english_only_and_required_references(self) -> None:
        self.assertIsNone(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", self.text))
        for surname in ("Jin", "Díaz", "Chen", "Xue", "Liang"):
            with self.subTest(reference=surname):
                self.assertIn(surname, self.text)

    def test_shortcut_equations_match_the_frozen_definition(self) -> None:
        required = (
            r"p_-=\frac1N\sum_j\mathbf1[z_j\le0]",
            r"\kappa[\cos(\Delta/2)]_+(1-2p_-)",
            r"\alpha(q)=0.10+0.15q",
            r"\lambda(q)=0.35+0.30q",
            r"p_-\ge0.20",
            r"Q_{\alpha}(z)-0.25r_{cap}",
            r"\lambda Q_{0.99}(\|C-x_0\|)",
            r"x_s=x_0+L^*\hat d",
            r"R_{res}=\left[Q_{0.99}(\|C-x_s\|)-r_{cap}\right]_+",
            r"h=0.8(2r_{cap})",
        )
        for expression in required:
            with self.subTest(expression=expression):
                self.assertIn(expression, self.text)
        self.assertIn("Algorithm 1 — Posterior-risk-bounded shortcut", self.text)
        self.assertIn("Algorithm 2 — Three-role S4b joint-head adaptation", self.text)

    def test_s4_adaptation_contract_is_current_and_role_separated(self) -> None:
        required = (
            "Exact M10/M12 S4b dataset-contract points",
            "72 train, 36 validation, and 36 calibration scans",
            "12+6+6 pairwise-disjoint complete C6 orbits",
            "state and direction heads are jointly trainable: 6,579 parameters",
            "encoder and shared representation are bitwise frozen",
            "Train:</strong> update state and direction heads only",
            "Validation:</strong> select hyperparameters and epoch",
            "Calibration:</strong> on 36 disjoint scans",
            "fit state/direction temperatures and direction APS only",
            "2–8 ms delayed pre-hardware observation prior",
            "Historical S3 diagnosis — not S4b evidence",
            "one M10 control-error outlier above",
            "old development-v2 task is permanently ineligible",
        )
        for statement in required:
            with self.subTest(statement=statement):
                self.assertIn(statement, self.text)
        self.assertIn(
            r"\mathcal L=\mathcal L_s+\mathcal L_d",
            self.text,
        )
        self.assertIn(
            r"10^{-2}\!\sum_{u\in\{s,d\}}",
            self.text,
        )
        self.assertNotIn("Exact M10/M12 S3 dataset-contract points", self.text)
        self.assertNotIn("only the 6,192-parameter direction head", self.text)

    def test_figure_6_is_provenanced_to_the_s4_contracts(self) -> None:
        self.assertTrue(PAPER_FIGURE_MANIFEST.is_file())
        manifest = json.loads(PAPER_FIGURE_MANIFEST.read_text(encoding="utf-8"))
        output_paths = {record["path"] for record in manifest["outputs"]}
        self.assertIn(
            "outputs/current/media/figures/paper/"
            "cross_size_finetune_c6_sampling_en.png",
            output_paths,
        )
        contracts = manifest["inputs"]["dataset_contracts"]
        self.assertEqual(len(contracts), 2)
        self.assertTrue(
            all("cross_size_s4b_adaptation" in record["path"] for record in contracts)
        )
        for record in contracts:
            contract_path = ROOT / record["path"]
            self.assertTrue(contract_path.is_file())
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            counts = contract["role_counts"]
            self.assertEqual(
                {role: counts[role]["scans"] for role in counts},
                {"train": 72, "validation": 36, "calibration": 36},
            )
            self.assertEqual(
                {role: counts[role]["orbits"] for role in counts},
                {"train": 12, "validation": 6, "calibration": 6},
            )
            self.assertTrue(
                all(
                    not intersection
                    for intersection in contract["pairwise_orbit_intersections"].values()
                )
            )

    def test_cross_size_gate_is_an_unambiguous_conjunction(self) -> None:
        match = re.search(r"\$\$\s*G_s=(.*?)\s*\$\$", self.text, flags=re.DOTALL)
        self.assertIsNotNone(match)
        gate = match.group(1)
        self.assertEqual(gate.count(r"\land"), 3)
        self.assertNotIn(",", gate)
        for condition in (
            r"r_{bottom}\ge0.80",
            r"N_{hard}=0",
            r"N_{>90^\circ}=0",
            r"r_{oracle}\ge0.90",
        ):
            with self.subTest(condition=condition):
                self.assertIn(condition, gate)

    def test_s3_and_s4_claim_boundaries_are_explicit(self) -> None:
        self.assertEqual(self.text.count("<!-- S4_UNTOUCHED_RESULT_BLOCK -->"), 1)
        self.assertIn("S4b untouched result — evaluation in progress", self.text)
        self.assertIn("M10 head fine-tuned</td><td>10/12", self.text)
        self.assertIn("M12 head fine-tuned</td><td>9/12", self.text)
        self.assertIn("No M10 or M12 completion, safety, or timing value is reported", self.text)
        self.assertNotIn("S4b untouched result — PASS", self.text)

    def test_hardware_results_are_planned_and_all_result_cells_are_dashes(self) -> None:
        match = re.search(
            r"<!-- HARDWARE_RESULT_TABLE_BEGIN -->(.*?)"
            r"<!-- HARDWARE_RESULT_TABLE_END -->",
            self.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        block = match.group(1)
        self.assertIn("PLANNED, NOT EVALUATED", block)
        rows = re.findall(r"<tr>(.*?)</tr>", block, flags=re.DOTALL)
        result_rows = []
        for row in rows:
            cells = [
                re.sub(r"<[^>]+>", "", value).strip()
                for value in re.findall(r"<td>(.*?)</td>", row, flags=re.DOTALL)
            ]
            if cells:
                result_rows.append(cells)
        self.assertGreaterEqual(len(result_rows), 5)
        for cells in result_rows:
            with self.subTest(metric=cells[0]):
                self.assertEqual(len(cells), 4)
                self.assertEqual(cells[1:], ["—", "—", "—"])

    def test_every_local_figure_and_video_exists(self) -> None:
        references = re.findall(r'(?:src|href)="([^"]+)"', self.text)
        references.extend(re.findall(r"\]\(([^)]+)\)", self.text))
        local = {
            (PAPER.parent / value.split("#", 1)[0]).resolve()
            for value in references
            if value
            and not value.startswith("#")
            and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value)
        }
        self.assertGreaterEqual(len(local), 12)
        missing = sorted(str(path) for path in local if not path.is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
