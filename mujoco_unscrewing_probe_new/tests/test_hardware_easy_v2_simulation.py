from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.sample_m8_hardware_easy_v2_direction import (
    build_hardware_easy_v2_dense_training_tasks,
    build_hardware_easy_v2_training_tasks,
    build_hardware_easy_v2_untouched_tasks,
)
from src.sample_m8_hardware_easy_v2_fresh_validation import (
    build_fresh2_easy_tasks,
    build_fresh3_easy_tasks,
    build_fresh4_easy_tasks,
    build_fresh_easy_tasks,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestHardwareEasyV2Simulation(unittest.TestCase):
    def test_derived_contact_geometry_matches_hardware_contract(self) -> None:
        expected = {
            "m8": (0.00650, 0.00810, 0.014950, 0.80, 0.00060),
            "m10": (0.00854, 0.01054, 0.018975, 0.80, 0.00080),
            "m12": (0.01060, 0.01300, 0.021275, 0.80, 0.00100),
        }
        for size, values in expected.items():
            with self.subTest(size=size):
                path = ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                actual = (
                    data["derived"]["socket_wall_af_m"],
                    data["derived"]["socket_top_af_m"],
                    data["screw_head_diameter_m"],
                    data["bit_tip_scale"],
                    data["bit_tip_chamfer_m"],
                )
                np.testing.assert_allclose(actual, values, atol=1.0e-12, rtol=0.0)
                self.assertTrue(data["bounded_socket_shell"])
                self.assertFalse(data["hardware_aligned_profile_ready"])
                self.assertIsNone(data["wrench_reference_to_tip_m"])

    def test_derived_xmls_load_and_collision_shell_stays_inside_head(self) -> None:
        for size in ("m8", "m10", "m12"):
            with self.subTest(size=size):
                config = json.loads(
                    (ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json").read_text(
                        encoding="utf-8"
                    )
                )
                path = ROOT / "models" / f"allen_{size}_hardware_easy_v2.xml"
                model = mujoco.MjModel.from_xml_path(str(path))
                self.assertEqual(model.nq, 7)
                self.assertGreaterEqual(model.ngeom, 200)
                root = ET.parse(path).getroot()
                head_radius = 0.5 * float(config["screw_head_diameter_m"])
                checked = 0
                for mesh in root.findall("./asset/mesh"):
                    name = mesh.attrib["name"]
                    if "socket_chamfer_sector" not in name and "socket_wall_sector" not in name:
                        continue
                    vertices = np.asarray(
                        [float(value) for value in mesh.attrib["vertex"].split()], dtype=float
                    ).reshape(-1, 3)
                    self.assertLessEqual(
                        float(np.max(np.linalg.norm(vertices[:, :2], axis=1))),
                        head_radius + 1.0e-9,
                    )
                    checked += 1
                self.assertEqual(checked, 144)

    def test_frozen_source_hashes_are_recorded_and_unchanged(self) -> None:
        for size in ("m8", "m10", "m12"):
            data = json.loads(
                (ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                data["frozen_source_config_sha256"],
                _sha256(ROOT / "configs" / f"allen_{size}_source.json"),
            )
            self.assertEqual(
                data["frozen_source_model_sha256"],
                _sha256(ROOT / "models" / f"allen_{size}_source.xml"),
            )

    def test_easy_training_and_untouched_orbits_are_disjoint(self) -> None:
        spec = json.loads(
            (ROOT / "configs" / "allen_m8_hardware_easy_v2.json").read_text(
                encoding="utf-8"
            )
        )
        train = build_hardware_easy_v2_training_tasks(spec)
        dense = build_hardware_easy_v2_dense_training_tasks(spec)
        audit = build_hardware_easy_v2_untouched_tasks(spec)
        self.assertEqual(len(train), 288)
        self.assertEqual(len(dense), 768)
        self.assertEqual(len(audit), 180)
        self.assertFalse({task.radius_m for task in train} & {task.radius_m for task in audit})
        self.assertFalse(
            {task.base_angle_deg for task in train}
            & {task.base_angle_deg for task in audit}
        )
        guidance = float(spec["paired_flat_normal_guidance_margin_m"])
        self.assertGreaterEqual(min(task.radius_m for task in train), 1.15 * guidance)
        for tasks in (train, audit):
            for orbit_id in {task.symmetry_orbit_id for task in tasks}:
                orbit = [task for task in tasks if task.symmetry_orbit_id == orbit_id]
                self.assertEqual({task.rotation_index for task in orbit}, set(range(6)))
                self.assertEqual(len({task.radius_m for task in orbit}), 1)
                self.assertEqual(len({task.friction_slide for task in orbit}), 1)

    def test_all_easy_fresh_task_builders_return_complete_c6_audits(self) -> None:
        spec = json.loads(
            (ROOT / "configs" / "allen_m8_hardware_easy_v2.json").read_text(
                encoding="utf-8"
            )
        )
        builders = (
            build_fresh_easy_tasks,
            build_fresh2_easy_tasks,
            build_fresh3_easy_tasks,
            build_fresh4_easy_tasks,
        )
        task_sets = [builder(spec) for builder in builders]
        for tasks in task_sets:
            self.assertEqual(len(tasks), 180)
        for index, left in enumerate(task_sets):
            for right in task_sets[index + 1 :]:
                self.assertFalse(
                    {task.radius_m for task in left} & {task.radius_m for task in right}
                )


if __name__ == "__main__":
    unittest.main()
