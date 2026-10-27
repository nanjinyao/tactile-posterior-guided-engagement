from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

from src.run_m8_recoverable_funnel_r3_ablation import ARM_ORDER, build_tasks


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current/evaluations/m8_recoverable_funnel_r3_ablation"


class TestRecoverableFunnelR3Ablation(unittest.TestCase):
    def test_task_program_is_fixed_and_moderate(self) -> None:
        tasks = build_tasks()
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task.task_id for task in tasks}), 12)
        self.assertGreaterEqual(min(task.post_capture_tilt_deg for task in tasks), 1.0)
        self.assertLessEqual(max(task.post_capture_tilt_deg for task in tasks), 3.0)
        self.assertGreaterEqual(min(task.post_capture_lateral_m for task in tasks), 0.00015)
        self.assertLessEqual(max(task.post_capture_lateral_m for task in tasks), 0.00030)

    def test_contract_has_three_arms_and_no_privileged_rebase(self) -> None:
        contract = json.loads((OUT / "task_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["task_count"], 12)
        self.assertEqual(tuple(contract["arms"]), ARM_ORDER)
        self.assertFalse(contract["arms"]["no_funnel"]["enabled"])
        self.assertEqual(contract["arms"]["rcc_shield_r0"]["max_recoveries"], 0)
        self.assertEqual(contract["arms"]["rcc_shield_r3"]["max_recoveries"], 3)
        self.assertTrue(contract["arms"]["rcc_shield_r3"]["retry_on_terminal_rejection"])
        self.assertIn("measured snapshot.tip_position", contract["recovery_xy_rebase_source"])
        self.assertIn("no true socket center", contract["recovery_xy_rebase_source"])

    def test_all_trials_are_paired_and_have_zero_simulator_reset(self) -> None:
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report["methods"]), set(ARM_ORDER))
        self.assertIn("not a proof", report["claim_boundary"])
        for arm in ARM_ORDER:
            self.assertEqual(report["methods"][arm]["trials"], 12)
            self.assertEqual(report["methods"][arm]["simulator_resets"], 0)
        paired = list((OUT / "trials").glob("recoverable_[0-9][0-9].json"))
        self.assertEqual(len(paired), 12)
        for path in paired:
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(record["methods"]), set(ARM_ORDER))
            capture_times = []
            for arm, result in record["methods"].items():
                self.assertTrue(result["inner_wall_confirmed"], f"{path.name}/{arm}")
                capture_times.append(result["time_to_inner_wall_s"])
                self.assertEqual(result["simulator_reset_count"], 0)
                rebases = result.get("funnel_recovery_xy_rebases_mm") or []
                self.assertEqual(len(rebases), result["funnel_recovery_count"])
                if arm != "rcc_shield_r3":
                    self.assertEqual(rebases, [])
            self.assertAlmostEqual(max(capture_times), min(capture_times), places=12)

    def test_tables_and_poster_are_synchronized(self) -> None:
        with (OUT / "trials.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["arm"] for row in rows}, set(ARM_ORDER))
        self.assertTrue((OUT / "summary.csv").is_file())
        self.assertTrue((OUT / "paired_outcomes.csv").is_file())
        self.assertTrue((OUT / "report.md").is_file())
        self.assertTrue((OUT / "paired_recoverable_funnel_r3_poster.png").is_file())


if __name__ == "__main__":
    unittest.main()
