from __future__ import annotations

import json
from pathlib import Path
import unittest

import mujoco

from src.render_m8_dynamic_videos import build_cases, setup


ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "outputs/current/evaluations/m8_model_m10_m12_zero_shot_full_loop"


class TestCrossSizeFullLoopArtifacts(unittest.TestCase):
    def test_generic_dynamic_setup_loads_m10_and_m12(self) -> None:
        case = build_cases()[0]
        for size in ("m10", "m12"):
            with self.subTest(size=size):
                spec, model, _, rollout, renderer, _, _ = setup(
                    case,
                    render=False,
                    spec_path=ROOT / "configs" / f"allen_{size}_source.json",
                    model_path=ROOT / "models" / f"allen_{size}_source.xml",
                )
                self.assertIsNone(renderer)
                self.assertEqual(spec["name"], size)
                body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, rollout.body_id
                )
                self.assertEqual(body_name, f"allen_bit_{size}")

    def test_s2_is_frozen_evaluation_and_fails_both_scale_gates(self) -> None:
        report = json.loads((S2 / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(
            report["experiment"],
            "frozen_M8_model_M10_M12_zero_shot_complete_closed_loop_S2",
        )
        self.assertFalse(report["all_size_gates_pass"])
        self.assertTrue(report["learning_contract"]["M8_model_frozen"])
        self.assertEqual(
            report["learning_contract"]["M10_M12_training_or_calibration_samples"],
            0,
        )
        self.assertFalse(
            report["learning_contract"]["M10_M12_evaluation_samples_reusable_for_training"]
        )
        expected = {
            "m10": (23, 21, 27, 3),
            "m12": (21, 19, 30, 4),
        }
        for size, counts in expected.items():
            with self.subTest(size=size):
                item = report["sizes"][size]
                self.assertFalse(item["zero_shot_full_loop_gate_pass"])
                self.assertEqual(
                    item["gate_contract"]["direction_metric"],
                    "executed circular-mean control direction",
                )
                self.assertEqual(
                    (
                        item["methods"]["proposed"]["bottom_successes"],
                        item["methods"]["outward"]["bottom_successes"],
                        item["methods"]["oracle"]["bottom_successes"],
                        item["methods"]["proposed"]["hard_stops"],
                    ),
                    counts,
                )
                self.assertEqual(
                    item["methods"]["proposed"][
                        "control_direction_error_gt90_count"
                    ],
                    1,
                )
                manifest = json.loads(
                    (S2 / size / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["completed"], 30)
                self.assertEqual(manifest["training_use"], "forbidden")

    def test_paper_figure_and_machine_audit_expose_the_failed_gate(self) -> None:
        self.assertTrue(
            (
                ROOT
                / "outputs/current/media/figures/paper/cross_size_zero_shot_full_loop.png"
            ).is_file()
        )
        audit = json.loads(
            (
                ROOT / "outputs/current/reports/closed_loop_scientific_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(audit["cross_size_full_loop_S2"]["all_size_gates_pass"])
        self.assertEqual(
            audit["task"]["cross_size_full_loop_gate"],
            "failed for both M10 and M12",
        )


if __name__ == "__main__":
    unittest.main()
