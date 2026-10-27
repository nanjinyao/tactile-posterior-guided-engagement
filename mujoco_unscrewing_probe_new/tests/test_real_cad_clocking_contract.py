from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = SIM_ROOT.parent
REAL_ROOT = GROUP_ROOT / "real_device"
CAD_ROOT = REAL_ROOT / "model" / "parametric_tooling"

CANONICAL_RUNS = {
    "m8": "M8_conical_fingerprints_26pt_20260728_161249_458",
    "m10": "M10_conical_fingerprints_26pt_20260728_152808_877",
    "m12": "M12_conical_fingerprints_26pt_20260728_175502_318",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def periodic_error_deg(value: float, reference: float, period: float) -> float:
    return (value - reference + 0.5 * period) % period - 0.5 * period


def direction_deg(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0


def mesh_flat_normal_representative_deg(model_path: Path, suffix: str) -> float:
    root = ET.parse(model_path).getroot()
    mesh = next(
        item
        for item in root.findall("./asset/mesh")
        if item.attrib["name"].endswith(suffix)
    )
    values = np.fromstring(mesh.attrib["vertex"], sep=" ", dtype=float)
    first_ring = values.reshape(-1, 3)[:6, :2]
    midpoints = 0.5 * (first_ring + np.roll(first_ring, -1, axis=0))
    angles = np.mod(
        np.rad2deg(np.arctan2(midpoints[:, 1], midpoints[:, 0])), 360.0
    )
    return float(angles[np.argmin(np.abs(periodic_error_deg(angles, 0.0, 60.0)))])


class RealCadClockingContractTest(unittest.TestCase):
    def test_cad_step_sidecar_contract_is_self_consistent(self) -> None:
        design = load_json(CAD_ROOT / "design_parameters.json")
        manifest = load_json(CAD_ROOT / "generated" / "manifest.json")
        self.assertEqual(
            design["design_revision"],
            "v1.5-integrated-flange-tools-yaw-datum",
        )
        self.assertEqual(manifest["design_revision"], design["design_revision"])
        frame = design["coordinate_system"]
        yaw = manifest["peg_flange_yaw_contract"]
        self.assertEqual(frame, manifest["coordinate_system"])
        self.assertEqual(float(yaw["shape_rotation_about_positive_z_deg"]), -15.0)
        self.assertEqual(float(yaw["selected_peg_flat_normal_in_flange_deg"]), 45.0)
        self.assertEqual(float(yaw["flange_reference_tangent_normal_deg"]), 45.0)
        self.assertEqual(float(yaw["socket_selected_flat_normal_deg"]), 0.0)
        period = float(yaw["equivalent_hex_period_deg"])
        self.assertEqual(period, 60.0)
        self.assertAlmostEqual(
            periodic_error_deg(
                float(yaw["selected_peg_flat_normal_in_flange_deg"]),
                float(yaw["shape_rotation_about_positive_z_deg"]),
                period,
            ),
            0.0,
            places=12,
        )
        for size in CANONICAL_RUNS:
            validation = manifest["integrated_tool_validation"][size]
            self.assertTrue(validation["pass"])
            self.assertAlmostEqual(
                float(validation["target_selected_flat_normal_yaw_deg"]),
                45.0,
                places=9,
            )
            for section in (
                "measured_tip_section",
                "measured_uniform_shaft_section",
                "measured_transition_section",
            ):
                self.assertAlmostEqual(
                    periodic_error_deg(
                        float(validation[section]["selected_flat_normal_yaw_deg"]),
                        45.0,
                        period,
                    ),
                    0.0,
                    places=8,
                )

    def test_printed_plans_and_canonical_campaigns_preserve_frame_and_labels(
        self,
    ) -> None:
        for size, run_name in CANONICAL_RUNS.items():
            with self.subTest(size=size):
                plan_path = (
                    REAL_ROOT
                    / "configs"
                    / f"{size}_printed_cad_conical_sampling_v2.json"
                )
                run_root = REAL_ROOT / "data" / run_name
                plan = load_json(plan_path)
                campaign = load_json(run_root / "sampling_campaign.json")
                manifest = load_json(run_root / "manifest.json")
                summary = load_json(run_root / "summary.json")
                plan_hash = sha256(plan_path)
                self.assertEqual(campaign["sampling_plan_sha256"], plan_hash)
                self.assertEqual(manifest["sampling_plan_sha256"], plan_hash)
                self.assertEqual(summary["sampling_plan_sha256"], plan_hash)

                yaw = plan["coordinate_contract"]["yaw_contract"]
                self.assertEqual(
                    float(yaw["printed_peg_selected_flat_normal_in_flange_cad_deg"]),
                    45.0,
                )
                self.assertEqual(
                    float(yaw["commissioned_peg_flat_normal_in_fixture_deg"]),
                    0.0,
                )
                self.assertEqual(
                    float(yaw["socket_flat_normal_in_fixture_deg"]),
                    0.0,
                )
                self.assertIn("REQUIRES_RUN_LOCAL_CENTER_AND_YAW", plan["status"])

                for point in plan["points"]:
                    local = np.asarray(point["local_offset_fixture_mm"], dtype=float)
                    label = np.asarray(
                        point["true_direction_to_centre_fixture_unit"], dtype=float
                    )
                    self.assertAlmostEqual(
                        periodic_error_deg(
                            direction_deg(local),
                            float(point["angle_fixture_deg"]),
                            360.0,
                        ),
                        0.0,
                        places=7,
                    )
                    self.assertAlmostEqual(
                        periodic_error_deg(
                            direction_deg(label),
                            (float(point["angle_fixture_deg"]) + 180.0) % 360.0,
                            360.0,
                        ),
                        0.0,
                        places=7,
                    )

                rotation = np.asarray(
                    manifest["world_from_fixture_rotation_matrix"], dtype=float
                )
                self.assertTrue(np.allclose(rotation, np.eye(3), atol=1.0e-12))
                self.assertEqual(float(manifest["world_from_fixture_yaw_deg"]), 0.0)
                commissioning = manifest["tooling_commissioning_gate"]
                self.assertFalse(commissioning["geometry_commissioned"])
                self.assertEqual(commissioning["clocking_evidence_path"], "")
                self.assertEqual(commissioning["clocking_evidence_sha256"], "")

                self.assertEqual(len(campaign["attempts"]), 26)
                for attempt in campaign["attempts"]:
                    offset = np.asarray(
                        attempt["measured_contact_fixture_xy_mm"], dtype=float
                    )
                    expected = -offset / np.linalg.norm(offset)
                    fixture_direction = np.asarray(
                        attempt["true_direction_fixture_unit"], dtype=float
                    )
                    world_direction = np.asarray(
                        attempt["true_direction_world_xy"], dtype=float
                    )
                    self.assertTrue(
                        np.allclose(fixture_direction, expected, atol=1.0e-12)
                    )
                    self.assertTrue(
                        np.allclose(
                            world_direction,
                            rotation[:2, :2] @ fixture_direction,
                            atol=1.0e-12,
                        )
                    )
                    self.assertAlmostEqual(
                        periodic_error_deg(
                            direction_deg(fixture_direction),
                            float(attempt["true_direction_fixture_deg"]),
                            360.0,
                        ),
                        0.0,
                        places=9,
                    )

    def test_mujoco_hardware_easy_v2_is_zero_relative_yaw(self) -> None:
        for size in CANONICAL_RUNS:
            with self.subTest(size=size):
                config = load_json(
                    SIM_ROOT / "configs" / f"allen_{size}_hardware_easy_v2.json"
                )
                clocking = config["mechanical_clocking_contract"]
                self.assertEqual(
                    float(clocking["integrated_peg_shape_rotation_from_legacy_deg"]),
                    -15.0,
                )
                self.assertEqual(
                    float(clocking["integrated_peg_selected_hex_flat_normal_deg"]),
                    45.0,
                )
                self.assertEqual(
                    float(clocking["flange_reference_tangent_normal_deg"]),
                    45.0,
                )
                self.assertEqual(
                    float(clocking["socket_selected_hex_flat_normal_deg"]),
                    0.0,
                )
                self.assertIn(
                    "remain aligned at relative yaw zero",
                    clocking["mujoco_relative_yaw_zero_semantics"],
                )
                model_path = (
                    SIM_ROOT / "models" / f"allen_{size}_hardware_easy_v2.xml"
                )
                bit_yaw = mesh_flat_normal_representative_deg(
                    model_path, "_bit_mesh"
                )
                socket_yaw = mesh_flat_normal_representative_deg(
                    model_path, "_socket_bottom_mesh"
                )
                self.assertAlmostEqual(
                    periodic_error_deg(bit_yaw, socket_yaw, 60.0),
                    0.0,
                    places=9,
                )

    def test_exported_learning_artifacts_use_correct_c6_representation(self) -> None:
        transform_id = "REAL_FRAME_C6_V2_RPOS_PHASE_NEG_CAD_V1P5"
        dataset = load_json(
            REAL_ROOT
            / "offline_analysis"
            / "real_multisize_learning_v1"
            / "dataset"
            / "dataset_contract.json"
        )
        training = load_json(
            REAL_ROOT
            / "offline_analysis"
            / "real_multisize_learning_v1"
            / "training"
            / "training_report.json"
        )
        bundle_training = load_json(
            REAL_ROOT
            / "model_runtime"
            / "bundles"
            / "real_multisize_direction_v1"
            / "training_report.json"
        )
        self.assertEqual(dataset["c6_augmentation"]["transform_id"], transform_id)
        self.assertIn(
            "rotate Fx/Fy and Tx/Ty by +k*60 degrees",
            dataset["c6_augmentation"]["rotation_semantics"],
        )
        self.assertEqual(training["c6_transform_id"], transform_id)
        self.assertEqual(bundle_training["c6_transform_id"], transform_id)


if __name__ == "__main__":
    unittest.main()
