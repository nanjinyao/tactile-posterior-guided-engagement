from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from src.hardware_tooling_profile import (
    HardwareToolingProfile,
    IncompleteHardwareProfileError,
)


class TestHardwareToolingProfile(unittest.TestCase):
    def test_known_cad_geometry_loads_and_matches_common_datum(self) -> None:
        profile = HardwareToolingProfile.load()
        self.assertEqual(profile.data["schema_version"], 2)
        sensor = profile.data["installed_force_torque_sensor"]
        self.assertEqual(sensor["product_name"], "BFT-SENS-CAT2-B4")
        self.assertEqual(sensor["serial_number"], "SN001349")
        self.assertEqual(sensor["communication_interface_name"], "Bota_Binary")
        self.assertIn("/dev/serial/by-id/", sensor["host_device"])
        self.assertEqual(sensor["qualification"], "observation_logging_only")
        self.assertIsNone(sensor["measurement_range_variant"])
        self.assertIsNone(sensor["calibration_certificate"])
        origin = sensor["measurement_origin_frame_definition"]
        self.assertEqual(
            origin["status"], "official_mechanical_definition_confirmed"
        )
        self.assertEqual(
            origin["robot_side_to_measurement_origin_translation_m_in_bota_star"],
            [0.0, 0.0, 0.038],
        )
        self.assertAlmostEqual(origin["mounting_plane_spacing_m"], 0.038, places=12)
        self.assertNotIn("ati_metrology_required", profile.data)
        self.assertAlmostEqual(
            profile.pink_cad_reference_to_tip_m, 0.042, places=12
        )
        known = profile.data["known_rigid_geometry"]
        self.assertEqual(
            known["blue_reference_connector_observed_thickness_m_interval"],
            [0.010, 0.011],
        )
        self.assertNotIn("blue_sensor_plane_to_tip_translation_m", known)
        stack = profile.data["installed_stack_contract"]
        self.assertEqual(
            stack["ordered_rigid_chain"][1:6],
            [
                "white/silver adapter flange",
                "Bota SensONE Gen A robot-side mounting plane",
                "Bota SensONE Gen A tool-side mounting plane / wrench origin",
                "cement-gray adapter flange",
                "current pink tool CAD mating plane",
            ],
        )
        self.assertAlmostEqual(
            stack["known_axial_terms_m"][
                "bota_robot_side_to_tool_side_wrench_plane"
            ],
            0.038,
            places=12,
        )
        self.assertAlmostEqual(
            stack["provisional_whole_stack_observation"][
                "KUKA_media_flange_to_pink_CAD_mating_plane_m"
            ],
            0.050,
            places=12,
        )
        self.assertAlmostEqual(
            stack["pose_dependent_scene_observation"][
                "cement_gray_adapter_bottom_to_table_m"
            ],
            0.570,
            places=12,
        )
        self.assertAlmostEqual(
            stack["folded_start_scene_observation"][
                "cement_gray_adapter_bottom_to_white_table_vertical_distance_m"
            ],
            0.590,
            places=12,
        )
        self.assertEqual(
            stack["folded_start_scene_observation"]["pose_semantics"],
            "operator-defined default folded experiment start",
        )
        architecture = profile.data["tool_architecture"]
        self.assertEqual(
            architecture["revision"],
            "v1.5-integrated-flange-tools-yaw-datum",
        )
        self.assertTrue(architecture["one_piece_print"])
        self.assertFalse(architecture["separate_receiver"])
        self.assertFalse(architecture["separate_cartridge"])
        self.assertFalse(architecture["separate_retaining_clip"])
        self.assertEqual(
            architecture["swap_method"],
            "manually unbolt and replace the complete flange-tool",
        )
        self.assertTrue(
            profile.data["fixture"]["center_through_holes_open_through_final_body"]
        )
        fixture = profile.data["fixture"]
        frame = profile.data["frame_convention"]
        self.assertAlmostEqual(
            frame["integrated_peg_shape_rotation_from_legacy_deg"],
            -15.0,
            places=12,
        )
        self.assertAlmostEqual(
            frame["integrated_peg_selected_hex_flat_normal_deg"],
            45.0,
            places=12,
        )
        self.assertAlmostEqual(
            frame["flange_reference_tangent_normal_deg"],
            45.0,
            places=12,
        )
        self.assertAlmostEqual(
            frame["socket_selected_hex_flat_normal_deg"],
            0.0,
            places=12,
        )
        self.assertEqual(
            fixture["selected_hardware_socket_profile"], "hardware_easy_v2"
        )
        self.assertAlmostEqual(
            fixture["socket_geometry_profiles"]["simulation_exact"]["m8"]["mouth_af_m"],
            0.0065,
            places=12,
        )
        self.assertAlmostEqual(
            fixture["socket_geometry_profiles"]["hardware_easy_v2"]["m8"]["mouth_af_m"],
            0.0081,
            places=12,
        )
        self.assertAlmostEqual(
            fixture["socket_geometry_profiles"]["hardware_easy_v2"]["m12"][
                "head_outer_diameter_m"
            ],
            0.021275,
            places=12,
        )
        tools = profile.data["tools"]
        expected_easy_tips = {
            "m8": (0.0048, 0.0006),
            "m10": (0.0064, 0.0008),
            "m12": (0.008, 0.001),
        }
        for size, (tip_af, chamfer) in expected_easy_tips.items():
            self.assertEqual(tools[size]["selected_hardware_tip_profile"], "hardware_easy_v2")
            self.assertAlmostEqual(
                tools[size]["peg_shape_rotation_from_legacy_deg"],
                -15.0,
                places=12,
            )
            self.assertAlmostEqual(
                tools[size]["selected_hex_flat_normal_in_pink_cad_deg"],
                45.0,
                places=12,
            )
            easy = tools[size]["tip_geometry_profiles"]["hardware_easy_v2"]
            strict = tools[size]["tip_geometry_profiles"]["simulation_exact"]
            self.assertAlmostEqual(easy["tip_face_af_m"], tip_af, places=12)
            self.assertAlmostEqual(
                easy["tip_chamfer_radial_setback_m"], chamfer, places=12
            )
            self.assertAlmostEqual(easy["tip_chamfer_axial_length_m"], chamfer, places=12)
            self.assertAlmostEqual(easy["chamfer_angle_deg"], 45.0, places=12)
            self.assertAlmostEqual(strict["tip_face_scale"], 0.98, places=12)
        self.assertEqual(
            fixture["exterior_vertical_ribs"]["counts"],
            {"m8": 18, "m10": 24, "m12": 30},
        )
        self.assertFalse(profile.hardware_ready)
        clocking = profile.data["tool_clocking_commissioning"]
        self.assertEqual(
            clocking["status"],
            "new_v1_5_prints_and_per_size_low_load_requalification_pending",
        )
        self.assertEqual(
            clocking["existing_simulation_yaw_support_deg"],
            [-2.0, 2.0],
        )
        self.assertAlmostEqual(
            clocking["pink_cad_plane_to_tip_nominal_mm"], 42.0, places=12
        )
        self.assertAlmostEqual(
            clocking["pink_cad_plane_to_tip_acceptance_mm"], 0.25, places=12
        )
        self.assertAlmostEqual(
            clocking["tip_axis_runout_acceptance_mm"], 0.20, places=12
        )

    def test_generated_v1_5_tools_match_yaw_and_mesh_contracts(self) -> None:
        profile = HardwareToolingProfile.load()
        generated = (
            Path(__file__).resolve().parents[2]
            / "real_device"
            / "model"
            / "parametric_tooling"
            / "generated"
        )
        manifest = json.loads(
            (generated / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["design_revision"],
            "v1.5-integrated-flange-tools-yaw-datum",
        )
        yaw = manifest["peg_flange_yaw_contract"]
        self.assertAlmostEqual(
            yaw["shape_rotation_about_positive_z_deg"],
            -15.0,
            places=12,
        )
        self.assertAlmostEqual(
            yaw["selected_peg_flat_normal_in_flange_deg"],
            yaw["flange_reference_tangent_normal_deg"],
            places=12,
        )
        parts = {row["name"]: row for row in manifest["parts"]}
        for size in ("m8", "m10", "m12"):
            validation = manifest["integrated_tool_validation"][size]
            self.assertTrue(validation["pass"])
            self.assertAlmostEqual(
                validation["measured_uniform_shaft_section"][
                    "selected_flat_normal_yaw_deg"
                ],
                45.0,
                places=9,
            )
            self.assertEqual(
                parts[f"integrated_flange_peg_{size}"]["files"]["step"][
                    "sha256"
                ],
                profile.data["tools"][size][
                    "cad_integrated_flange_tool_step_sha256"
                ],
            )
        mesh = json.loads(
            (generated / "mesh_validation.json").read_text(encoding="utf-8")
        )
        self.assertTrue(mesh["all_pass"])
        self.assertEqual(mesh["failed"], [])
        self.assertEqual(mesh["validated_formats"], ["stl", "3mf"])
        self.assertEqual(len(mesh["meshes"]), 18)

    def test_incomplete_metrology_blocks_hardware_aligned_use(self) -> None:
        profile = HardwareToolingProfile.load()
        missing = profile.missing_metrology()
        self.assertIn(
            "sensor_metrology_required.sensor_measurement_origin_to_tip_translation_m",
            missing,
        )
        self.assertIn(
            "sensor_metrology_required."
            "T_sensor_measurement_origin_to_pink_tool_cad_mating_plane",
            missing,
        )
        self.assertIn(
            "installed_force_torque_sensor.calibration_certificate",
            missing,
        )
        self.assertIn("tools.m8.measured_complete_tool_mass_kg", missing)
        self.assertIn("fixture.base_to_robot_transform", missing)
        self.assertIn(
            "tool_clocking_commissioning."
            "flange_tangent_datum_to_fixture_x_yaw_rad",
            missing,
        )
        self.assertIn(
            "tool_clocking_commissioning.per_size.m12."
            "measured_peg_socket_relative_yaw_deg",
            missing,
        )
        self.assertIn(
            "tool_clocking_commissioning.per_size.m12."
            "measured_pink_cad_mating_plane_to_tip_mm",
            missing,
        )
        self.assertIn(
            "tool_clocking_commissioning.per_size.m12."
            "low_load_fingerprint_and_capture_requalified",
            missing,
        )
        with self.assertRaises(IncompleteHardwareProfileError):
            profile.require_hardware_ready()
        with self.assertRaises(IncompleteHardwareProfileError):
            profile.calibrated_wrench_reference_to_tip_m()

        bad_data = copy.deepcopy(profile.data)
        bad_clocking = bad_data["tool_clocking_commissioning"]
        bad_clocking["flange_tangent_datum_to_fixture_x_yaw_rad"] = 0.0
        bad_m12 = bad_clocking["per_size"]["m12"]
        bad_m12["as_built_selected_flat_to_flange_datum_error_deg"] = 0.0
        bad_m12["measured_peg_socket_relative_yaw_deg"] = 0.0
        bad_m12["measured_pink_cad_mating_plane_to_tip_mm"] = 42.26
        bad_m12["tip_axis_runout_mm"] = 0.201
        bad_m12["low_load_fingerprint_and_capture_requalified"] = True
        bad_profile = HardwareToolingProfile(profile.path, bad_data)
        bad_missing = bad_profile.missing_metrology()
        self.assertIn(
            "tool_clocking_commissioning.per_size.m12."
            "measured_pink_cad_mating_plane_to_tip_mm",
            bad_missing,
        )
        self.assertIn(
            "tool_clocking_commissioning.per_size.m12.tip_axis_runout_mm",
            bad_missing,
        )
        self.assertNotIn(
            "tool_clocking_commissioning.per_size.m12."
            "as_built_selected_flat_to_flange_datum_error_deg",
            bad_missing,
        )
        self.assertNotIn(
            "tool_clocking_commissioning.per_size.m12."
            "measured_peg_socket_relative_yaw_deg",
            bad_missing,
        )


if __name__ == "__main__":
    unittest.main()
