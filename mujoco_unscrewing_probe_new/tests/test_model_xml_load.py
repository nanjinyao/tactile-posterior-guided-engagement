from __future__ import annotations

from pathlib import Path
import json
import unittest

import mujoco
import numpy as np

from src.mujoco_admittance_rollout import MuJoCoAdmittanceRollout


class TestGeneratedAllenModels(unittest.TestCase):
    def test_generated_models_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = [
            root / "models" / "allen_m8_source.xml",
            root / "models" / "allen_m10_source.xml",
            root / "models" / "allen_m12_source.xml",
        ]
        for path in expected:
            with self.subTest(path=path.name):
                model = mujoco.MjModel.from_xml_path(str(path))
                self.assertGreaterEqual(model.nbody, 3)
                self.assertGreaterEqual(model.ngeom, 80)
                self.assertGreaterEqual(model.nmesh, 70)
                self.assertEqual(model.nq, 7)

    def test_admittance_discovers_the_dynamic_bit_from_tip_site(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for size in ("m8", "m10", "m12"):
            with self.subTest(size=size):
                model = mujoco.MjModel.from_xml_path(
                    str(root / "models" / f"allen_{size}_source.xml")
                )
                data = mujoco.MjData(model)
                spec = json.loads(
                    (root / "configs" / f"allen_{size}_source.json").read_text(
                        encoding="utf-8"
                    )
                )
                rollout = MuJoCoAdmittanceRollout(
                    model,
                    data,
                    float(spec["bit_length_m"]),
                    float(spec["bit_mass_kg"]),
                )
                expected = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"allen_bit_{size}"
                )
                self.assertEqual(rollout.body_id, expected)

    def test_explicit_wrench_reference_is_independent_of_bit_length(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = mujoco.MjModel.from_xml_path(
            str(root / "models" / "allen_m8_source.xml")
        )
        data = mujoco.MjData(model)
        spec = json.loads(
            (root / "configs" / "allen_m8_source.json").read_text(encoding="utf-8")
        )
        explicit_distance = 0.042
        rollout = MuJoCoAdmittanceRollout(
            model,
            data,
            float(spec["bit_length_m"]),
            float(spec["bit_mass_kg"]),
            wrench_reference_to_tip_m=explicit_distance,
        )
        rollout.set_initial_tip_pose(np.array([0.001, -0.002, 0.006]))
        tip, _ = rollout.actual_tip_pose()
        sensor = rollout.wrench_reference_world()
        self.assertEqual(rollout.wrench_reference_mode, "explicit_calibrated_distance")
        self.assertTrue(np.allclose(sensor - tip, [0.0, 0.0, explicit_distance]))

    def test_invalid_explicit_wrench_reference_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = mujoco.MjModel.from_xml_path(
            str(root / "models" / "allen_m8_source.xml")
        )
        data = mujoco.MjData(model)
        with self.assertRaises(ValueError):
            MuJoCoAdmittanceRollout(model, data, 0.035, 0.5, wrench_reference_to_tip_m=0.0)


if __name__ == "__main__":
    unittest.main()
