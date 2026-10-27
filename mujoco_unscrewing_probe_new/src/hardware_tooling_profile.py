"""Validated loader for the hardware tooling alignment contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs" / "real_tooling_v1.json"


class IncompleteHardwareProfileError(RuntimeError):
    """Raised when hardware-aligned simulation is requested before metrology."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HardwareToolingProfile:
    path: Path
    data: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE) -> "HardwareToolingProfile":
        resolved = Path(path).resolve()
        data = json.loads(resolved.read_text(encoding="utf-8"))
        profile = cls(resolved, data)
        profile.validate_known_geometry()
        return profile

    def validate_known_geometry(self) -> None:
        if int(self.data.get("schema_version", -1)) != 2:
            raise ValueError("unsupported hardware tooling schema")
        sensor = self.data["installed_force_torque_sensor"]
        expected_sensor_identity = {
            "manufacturer": "Bota Systems",
            "sensor_family": "SensONE Gen A (BFT-SENS-CAT2-B4)",
            "product_name": "BFT-SENS-CAT2-B4",
            "serial_number": "SN001349",
            "communication_interface_name": "Bota_Binary",
        }
        for key, expected in expected_sensor_identity.items():
            if sensor.get(key) != expected:
                raise ValueError(f"unexpected installed force/torque sensor {key}")
        if sensor.get("qualification") != "observation_logging_only":
            raise ValueError("installed Bota is currently qualified only for observation logging")
        origin = sensor.get("measurement_origin_frame_definition")
        if not isinstance(origin, Mapping):
            raise ValueError("official Bota measurement-origin definition is missing")
        if origin.get("status") != "official_mechanical_definition_confirmed":
            raise ValueError("Bota measurement-origin status is not confirmed")
        origin_translation = np.asarray(
            origin.get("robot_side_to_measurement_origin_translation_m_in_bota_star"),
            dtype=float,
        )
        if origin_translation.shape != (3,) or not np.allclose(
            origin_translation, [0.0, 0.0, 0.038], atol=1.0e-12
        ):
            raise ValueError("unexpected SensONE Gen A measurement-origin translation")
        if not np.isclose(
            float(origin.get("mounting_plane_spacing_m", float("nan"))),
            0.038,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected SensONE Gen A mounting-plane spacing")
        driver_config = Path(sensor["driver_config"])
        if driver_config.exists() and _sha256(driver_config) != sensor["driver_config_sha256"]:
            raise ValueError("Bota driver config changed without synchronizing real_tooling_v1.json")
        known = self.data["known_rigid_geometry"]
        pink_to_tip = np.asarray(
            known["pink_tool_cad_mating_plane_to_tip_translation_m"], dtype=float
        )
        if pink_to_tip.shape != (3,) or not np.all(np.isfinite(pink_to_tip)):
            raise ValueError(
                "pink_tool_cad_mating_plane_to_tip_translation_m must be a finite 3-vector"
            )
        if not np.allclose(pink_to_tip, [0.0, 0.0, -0.042], atol=1.0e-12):
            raise ValueError("delivered pink CAD expects a [0, 0, -42] mm tip datum")
        blue_thickness = np.asarray(
            known["blue_reference_connector_observed_thickness_m_interval"],
            dtype=float,
        )
        if blue_thickness.shape != (2,) or not np.allclose(
            blue_thickness, [0.010, 0.011], atol=1.0e-12
        ):
            raise ValueError("unexpected ruler observation for the distinct blue connector")
        if "blue_sensor_plane_to_tip_translation_m" in known:
            raise ValueError("the distinct blue connector must not alias the pink CAD datum")
        stack = self.data.get("installed_stack_contract")
        if not isinstance(stack, Mapping):
            raise ValueError("installed rigid-stack contract is missing")
        expected_chain = [
            "KUKA media flange",
            "white/silver adapter flange",
            "Bota SensONE Gen A robot-side mounting plane",
            "Bota SensONE Gen A tool-side mounting plane / wrench origin",
            "cement-gray adapter flange",
            "current pink tool CAD mating plane",
            "pink one-piece flange and peg tip",
        ]
        if stack.get("ordered_rigid_chain") != expected_chain:
            raise ValueError("unexpected installed rigid-stack ordering")
        axial = stack.get("known_axial_terms_m", {})
        if not np.isclose(
            float(axial.get("bota_robot_side_to_tool_side_wrench_plane", np.nan)),
            0.038,
            atol=1.0e-12,
        ):
            raise ValueError("installed stack lost the official 38 mm Bota term")
        if not np.isclose(
            float(axial.get("pink_cad_mating_plane_to_tip", np.nan)),
            0.042,
            atol=1.0e-12,
        ):
            raise ValueError("installed stack lost the pink CAD 42 mm term")
        whole = stack.get("provisional_whole_stack_observation", {})
        if not np.isclose(
            float(whole.get("KUKA_media_flange_to_pink_CAD_mating_plane_m", np.nan)),
            0.050,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected provisional whole-stack observation")
        scene = stack.get("pose_dependent_scene_observation", {})
        if not np.isclose(
            float(scene.get("cement_gray_adapter_bottom_to_table_m", np.nan)),
            0.570,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected cement-gray-to-table scene observation")
        if "not a rigid TCP term" not in str(scene.get("use", "")):
            raise ValueError("pose-dependent 570 mm scene height was misclassified")
        folded_scene = stack.get("folded_start_scene_observation", {})
        if folded_scene.get("pose_semantics") != (
            "operator-defined default folded experiment start"
        ):
            raise ValueError("folded-start scene observation lost its pose semantics")
        if not np.isclose(
            float(
                folded_scene.get(
                    "cement_gray_adapter_bottom_to_white_table_vertical_distance_m",
                    np.nan,
                )
            ),
            0.590,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected folded-start cement-gray-to-table observation")
        if "not a rigid TCP term" not in str(folded_scene.get("use", "")):
            raise ValueError("folded-start 590 mm scene height was misclassified")
        frame = self.data["frame_convention"]
        expected_angles = {
            "socket_selected_hex_flat_normal_deg": 0.0,
            "integrated_peg_shape_rotation_from_legacy_deg": -15.0,
            "integrated_peg_selected_hex_flat_normal_deg": 45.0,
            "flange_reference_tangent_normal_deg": 45.0,
            "hex_equivalent_period_deg": 60.0,
        }
        for key, expected in expected_angles.items():
            if not np.isclose(
                float(frame.get(key, np.nan)),
                expected,
                atol=1.0e-12,
            ):
                raise ValueError(f"unexpected tooling yaw contract field {key}")
        if (
            "not a sensor/robot/fixture yaw calibration"
            not in str(frame.get("runtime_frame_rule", ""))
        ):
            raise ValueError("mechanical clocking was misclassified as frame calibration")
        period_deg = float(frame["hex_equivalent_period_deg"])
        equivalent_residual_deg = (
            (
                float(frame["integrated_peg_selected_hex_flat_normal_deg"])
                - float(frame["integrated_peg_shape_rotation_from_legacy_deg"])
                + 0.5 * period_deg
            )
            % period_deg
            - 0.5 * period_deg
        )
        if abs(equivalent_residual_deg) > 1.0e-12:
            raise ValueError("selected peg flat is inconsistent with the shape rotation")
        if not np.isclose(
            float(frame["integrated_peg_selected_hex_flat_normal_deg"]),
            float(frame["flange_reference_tangent_normal_deg"]),
            atol=1.0e-12,
        ):
            raise ValueError("selected peg flat is not parallel to the flange datum")
        architecture = self.data["tool_architecture"]
        if (
            architecture.get("revision")
            != "v1.5-integrated-flange-tools-yaw-datum"
        ):
            raise ValueError(
                "tooling contract expects the v1.5 yaw-datum integrated tools"
            )
        for key in (
            "one_piece_print",
        ):
            if architecture.get(key) is not True:
                raise ValueError(f"integrated tooling expects {key}=true")
        for key in (
            "separate_receiver",
            "separate_cartridge",
            "separate_retaining_clip",
            "m4_hardware_required_for_tool_change",
        ):
            if architecture.get(key) is not False:
                raise ValueError(f"integrated tooling expects {key}=false")

        expected_tool_tips = {
            "m8": {
                "peg_af": 0.006,
                "simulation_exact": (0.98, 0.00588, 0.00006, 0.00035),
                "hardware_easy_v2": (0.8, 0.0048, 0.0006, 0.0006),
            },
            "m10": {
                "peg_af": 0.008,
                "simulation_exact": (0.98, 0.00784, 0.00008, 0.00045),
                "hardware_easy_v2": (0.8, 0.0064, 0.0008, 0.0008),
            },
            "m12": {
                "peg_af": 0.01,
                "simulation_exact": (0.98, 0.0098, 0.0001, 0.00055),
                "hardware_easy_v2": (0.8, 0.008, 0.001, 0.001),
            },
        }
        tools = self.data.get("tools", {})
        if set(tools) != set(expected_tool_tips):
            raise ValueError("unexpected integrated tool size set")
        for size, expected_tool in expected_tool_tips.items():
            tool = tools[size]
            if not np.isclose(float(tool["peg_af_m"]), expected_tool["peg_af"], atol=1.0e-12):
                raise ValueError(f"unexpected {size} peg AF")
            if tool.get("selected_hardware_tip_profile") != "hardware_easy_v2":
                raise ValueError(f"{size} must select the hardware_easy_v2 peg tip")
            if not np.isclose(
                float(tool.get("peg_shape_rotation_from_legacy_deg", np.nan)),
                -15.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"unexpected {size} peg-to-flange shape rotation")
            if not np.isclose(
                float(
                    tool.get(
                        "selected_hex_flat_normal_in_pink_cad_deg",
                        np.nan,
                    )
                ),
                45.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"unexpected {size} selected peg flat normal")
            step_hash = str(
                tool.get("cad_integrated_flange_tool_step_sha256", "")
            )
            if len(step_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in step_hash
            ):
                raise ValueError(f"invalid {size} integrated tool STEP hash")
            tip_profiles = tool.get("tip_geometry_profiles", {})
            if set(tip_profiles) != {"simulation_exact", "hardware_easy_v2"}:
                raise ValueError(f"unexpected {size} peg-tip profile set")
            for profile_name in ("simulation_exact", "hardware_easy_v2"):
                tip = tip_profiles[profile_name]
                actual = np.asarray(
                    [
                        tip["tip_face_scale"],
                        tip["tip_face_af_m"],
                        tip["tip_chamfer_radial_setback_m"],
                        tip["tip_chamfer_axial_length_m"],
                    ],
                    dtype=float,
                )
                if not np.allclose(
                    actual,
                    np.asarray(expected_tool[profile_name]),
                    atol=1.0e-12,
                ):
                    raise ValueError(f"unexpected {profile_name}/{size} peg-tip geometry")
                if not np.isclose(
                    actual[1],
                    float(tool["peg_af_m"]) - 2.0 * actual[2],
                    atol=1.0e-12,
                ):
                    raise ValueError(f"inconsistent {profile_name}/{size} tip face AF")
            if not np.isclose(
                float(tip_profiles["hardware_easy_v2"]["chamfer_angle_deg"]),
                45.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"hardware_easy_v2/{size} peg tip must be 45 degrees")
            volume = float(tool["cad_integrated_flange_tool_solid_volume_m3"])
            if not np.isfinite(volume) or volume <= 0.0:
                raise ValueError(f"invalid {size} integrated flange-tool CAD volume")
        fixture = self.data["fixture"]
        if fixture.get("center_through_holes_open_through_final_body") is not True:
            raise ValueError(
                "active integrated tooling expects three final-body center through-holes"
            )
        if not np.isclose(
            float(fixture["center_through_hole_diameter_m"]),
            0.0005,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected fixture center through-hole diameter")
        if fixture.get("selected_hardware_socket_profile") != "hardware_easy_v2":
            raise ValueError(
                "active integrated tooling expects the explicit hardware_easy_v2 socket profile"
            )
        expected_socket_profiles = {
            "simulation_exact": {
                "m8": (0.0062, 0.00015, 0.00015, 0.0065, 0.013),
                "m10": (0.00824, 0.00018, 0.00018, 0.0086, 0.0165),
                "m12": (0.0103, 0.00022, 0.00022, 0.01074, 0.0185),
            },
            "hardware_easy_v2": {
                "m8": (0.0065, 0.0008, 0.0008, 0.0081, 0.01495),
                "m10": (0.00854, 0.001, 0.001, 0.01054, 0.018975),
                "m12": (0.0106, 0.0012, 0.0012, 0.013, 0.021275),
            },
        }
        socket_profiles = fixture.get("socket_geometry_profiles", {})
        if set(socket_profiles) != set(expected_socket_profiles):
            raise ValueError("unexpected fixture socket geometry profile set")
        for profile_name, expected_sizes in expected_socket_profiles.items():
            profile = socket_profiles[profile_name]
            for size, expected in expected_sizes.items():
                row = profile[size]
                actual = np.asarray(
                    [
                        row["wall_af_m"],
                        row["entry_chamfer_width_m"],
                        row["entry_chamfer_depth_m"],
                        row["mouth_af_m"],
                        row["head_outer_diameter_m"],
                    ],
                    dtype=float,
                )
                if not np.allclose(actual, np.asarray(expected), atol=1.0e-12):
                    raise ValueError(f"unexpected {profile_name}/{size} socket geometry")
                if not np.isclose(
                    actual[3],
                    actual[0] + 2.0 * actual[1],
                    atol=1.0e-12,
                ):
                    raise ValueError(f"non-45-degree or inconsistent {profile_name}/{size} mouth")
                if not np.isclose(actual[1], actual[2], atol=1.0e-12):
                    raise ValueError(f"non-45-degree {profile_name}/{size} socket chamfer")
        exterior_ribs = fixture["exterior_vertical_ribs"]
        if not np.isclose(
            float(exterior_ribs["smooth_guard_band_below_top_m"]),
            0.002,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected fixture exterior-rib smooth guard band")
        if exterior_ribs["counts"] != {"m8": 18, "m10": 24, "m12": 30}:
            raise ValueError("unexpected fixture exterior-rib counts")
        centers = fixture["socket_poses"]
        expected_x = {"m8": -0.045, "m10": 0.0, "m12": 0.045}
        for size, x_expected in expected_x.items():
            translation = np.asarray(centers[size]["translation_m"], dtype=float)
            if translation.shape != (3,) or not np.allclose(
                translation, [x_expected, 0.0, 0.0], atol=1.0e-12
            ):
                raise ValueError(f"unexpected {size} socket pose")
            if not np.isclose(
                float(centers[size].get("yaw_rad", np.nan)),
                0.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"unexpected {size} socket yaw")

        clocking = self.data.get("tool_clocking_commissioning")
        if not isinstance(clocking, Mapping):
            raise ValueError("tool clocking commissioning contract is missing")
        if clocking.get("status") != (
            "new_v1_5_prints_and_per_size_low_load_requalification_pending"
        ):
            raise ValueError("unexpected tool clocking commissioning status")
        if not np.isclose(
            float(clocking.get("cad_parallelism_tolerance_deg", np.nan)),
            0.05,
            atol=1.0e-12,
        ) or not np.isclose(
            float(clocking.get("physical_parallelism_acceptance_deg", np.nan)),
            1.0,
            atol=1.0e-12,
        ):
            raise ValueError("unexpected CAD or physical clocking tolerance")
        if not np.allclose(
            np.asarray(clocking.get("existing_simulation_yaw_support_deg"), dtype=float),
            [-2.0, 2.0],
            atol=1.0e-12,
        ):
            raise ValueError("unexpected trained yaw-support declaration")
        dimensional_limits = {
            "pink_cad_plane_to_tip_nominal_mm": 42.0,
            "pink_cad_plane_to_tip_acceptance_mm": 0.25,
            "tip_axis_runout_acceptance_mm": 0.20,
        }
        for key, expected in dimensional_limits.items():
            if not np.isclose(
                float(clocking.get(key, np.nan)),
                expected,
                atol=1.0e-12,
            ):
                raise ValueError(f"unexpected tool commissioning limit {key}")
        clocking_sizes = clocking.get("per_size", {})
        if set(clocking_sizes) != set(expected_tool_tips):
            raise ValueError("unexpected clocking commissioning size set")
        for size, row in clocking_sizes.items():
            if row.get("expected_step_sha256") != tools[size].get(
                "cad_integrated_flange_tool_step_sha256"
            ):
                raise ValueError(f"{size} clocking STEP hash mismatch")

        source = self.data["cad_source"]
        configured_design_path = Path(source["design_parameters"])
        local_design_path = (
            ROOT.parent
            / "real_device"
            / "model"
            / "parametric_tooling"
            / "design_parameters.json"
        )
        design_path = next(
            (
                candidate
                for candidate in (configured_design_path, local_design_path)
                if candidate.exists()
            ),
            None,
        )
        if (
            design_path is not None
            and _sha256(design_path) != source["design_parameters_sha256"]
        ):
            raise ValueError("CAD design parameters changed without synchronizing real_tooling_v1.json")
        local_generated = (
            ROOT.parent / "real_device" / "model" / "parametric_tooling" / "generated"
        )
        if local_generated.exists():
            for size, tool in tools.items():
                step_path = local_generated / f"integrated_flange_peg_{size}.step"
                if step_path.exists() and _sha256(step_path) != tool[
                    "cad_integrated_flange_tool_step_sha256"
                ]:
                    raise ValueError(f"{size} integrated tool STEP changed")

    @property
    def pink_cad_reference_to_tip_m(self) -> float:
        vector = np.asarray(
            self.data["known_rigid_geometry"][
                "pink_tool_cad_mating_plane_to_tip_translation_m"
            ],
            dtype=float,
        )
        return float(np.linalg.norm(vector))

    def missing_metrology(self) -> list[str]:
        missing: list[str] = []
        sensor = self.data["installed_force_torque_sensor"]
        for key in (
            "measurement_range_variant",
            "calibration_certificate",
            "measurement_origin_frame_definition",
        ):
            if sensor.get(key) is None:
                missing.append(f"installed_force_torque_sensor.{key}")
        sensor_metrology = self.data["sensor_metrology_required"]
        for key in (
            "T_sensor_measurement_origin_to_pink_tool_cad_mating_plane",
            "sensor_measurement_origin_to_tip_translation_m",
        ):
            if sensor_metrology.get(key) is None:
                missing.append(f"sensor_metrology_required.{key}")
        for size, tool in self.data["tools"].items():
            for key in (
                "measured_complete_tool_mass_kg",
                "measured_com_in_sensor_measurement_frame_m",
                "measured_inertia_at_com_kg_m2",
            ):
                if tool.get(key) is None:
                    missing.append(f"tools.{size}.{key}")
        if self.data["fixture"].get("base_to_robot_transform") is None:
            missing.append("fixture.base_to_robot_transform")
        clocking = self.data["tool_clocking_commissioning"]
        parallelism_limit = float(clocking["physical_parallelism_acceptance_deg"])
        taught_yaw = clocking.get("flange_tangent_datum_to_fixture_x_yaw_rad")
        if (
            taught_yaw is None
            or not np.isfinite(float(taught_yaw))
            or abs(float(taught_yaw)) > np.deg2rad(parallelism_limit)
        ):
            missing.append(
                "tool_clocking_commissioning."
                "flange_tangent_datum_to_fixture_x_yaw_rad"
            )
        yaw_support = np.asarray(
            clocking["existing_simulation_yaw_support_deg"], dtype=float
        )
        tip_height_nominal = float(
            clocking["pink_cad_plane_to_tip_nominal_mm"]
        )
        tip_height_tolerance = float(
            clocking["pink_cad_plane_to_tip_acceptance_mm"]
        )
        runout_limit = float(clocking["tip_axis_runout_acceptance_mm"])
        for size, row in clocking["per_size"].items():
            prefix = f"tool_clocking_commissioning.per_size.{size}"
            parallelism = row.get(
                "as_built_selected_flat_to_flange_datum_error_deg"
            )
            if (
                parallelism is None
                or not np.isfinite(float(parallelism))
                or abs(float(parallelism)) > parallelism_limit
            ):
                missing.append(
                    f"{prefix}.as_built_selected_flat_to_flange_datum_error_deg"
                )
            relative_yaw = row.get("measured_peg_socket_relative_yaw_deg")
            if (
                relative_yaw is None
                or not np.isfinite(float(relative_yaw))
                or not yaw_support[0] <= float(relative_yaw) <= yaw_support[1]
            ):
                missing.append(f"{prefix}.measured_peg_socket_relative_yaw_deg")
            tip_height = row.get(
                "measured_pink_cad_mating_plane_to_tip_mm"
            )
            if (
                tip_height is None
                or not np.isfinite(float(tip_height))
                or abs(float(tip_height) - tip_height_nominal)
                > tip_height_tolerance
            ):
                missing.append(
                    f"{prefix}.measured_pink_cad_mating_plane_to_tip_mm"
                )
            runout = row.get("tip_axis_runout_mm")
            if (
                runout is None
                or not np.isfinite(float(runout))
                or not 0.0 <= float(runout) <= runout_limit
            ):
                missing.append(f"{prefix}.tip_axis_runout_mm")
            if row.get("low_load_fingerprint_and_capture_requalified") is not True:
                missing.append(
                    f"{prefix}.low_load_fingerprint_and_capture_requalified"
                )
        return missing

    @property
    def hardware_ready(self) -> bool:
        return not self.missing_metrology()

    def require_hardware_ready(self) -> None:
        missing = self.missing_metrology()
        if missing:
            raise IncompleteHardwareProfileError(
                "hardware-aligned simulation is blocked until metrology is filled: "
                + ", ".join(missing)
            )

    def calibrated_wrench_reference_to_tip_m(self) -> float:
        self.require_hardware_ready()
        vector = np.asarray(
            self.data["sensor_metrology_required"][
                "sensor_measurement_origin_to_tip_translation_m"
            ],
            dtype=float,
        )
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("sensor-to-tip translation must be a finite 3-vector")
        return float(np.linalg.norm(vector))
