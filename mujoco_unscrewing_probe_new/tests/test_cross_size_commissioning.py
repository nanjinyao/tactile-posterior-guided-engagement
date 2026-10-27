from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.cartesian_admittance import AdmittanceConfig
from src.cross_size_commissioning import (
    REFERENCE_BIT_RADIUS_M,
    commissioning_for_bit_radius,
    commissioning_for_size,
    funnel_guard_for_size,
)
from src.funnel_guard import FunnelGuardConfig
from src.render_m8_dynamic_videos import setup
from src.render_m8_search_comparison import comparison_case, run_method


ROOT = Path(__file__).resolve().parents[1]


class _SetupIntercepted(RuntimeError):
    pass


class TestCrossSizeCommissioning(unittest.TestCase):
    def test_v2_separates_geometry_and_fixed_mass_response_scaling(self) -> None:
        base_guard = FunnelGuardConfig()
        base_admittance = AdmittanceConfig()
        m8 = commissioning_for_size(
            "M8",
            base_funnel=base_guard,
            base_admittance=base_admittance,
        )
        m10 = commissioning_for_size(
            "m10",
            base_funnel=base_guard,
            base_admittance=base_admittance,
        )
        m12 = commissioning_for_size(
            "M12",
            base_funnel=base_guard,
            base_admittance=base_admittance,
        )

        self.assertAlmostEqual(m8.bit_radius_m, REFERENCE_BIT_RADIUS_M)
        self.assertAlmostEqual(m8.bit_radius_scale, 1.0)
        self.assertEqual(m8.funnel_guard, base_guard)
        self.assertEqual(m8.admittance, base_admittance)
        self.assertAlmostEqual(m10.bit_radius_scale, 4.0 / 3.0)
        self.assertAlmostEqual(m12.bit_radius_scale, 5.0 / 3.0)
        self.assertEqual(
            m10.commissioning_version,
            "v2_sqrt_time_separate_rotation_lever_bumpless_terminal_hold_"
            "oracle_soft_escape_resume",
        )

        for commissioned in (m10, m12):
            scale = commissioned.bit_radius_scale
            time_scale = math.sqrt(scale)
            guard = commissioned.funnel_guard
            outer = commissioned.admittance
            self.assertAlmostEqual(commissioned.response_time_scale, time_scale)
            self.assertAlmostEqual(
                guard.capture_force_release_s,
                base_guard.capture_force_release_s * time_scale,
            )
            self.assertAlmostEqual(
                guard.insertion_handoff_s,
                base_guard.insertion_handoff_s * time_scale,
            )
            self.assertAlmostEqual(
                guard.recovery_unload_s,
                base_guard.recovery_unload_s * time_scale,
            )
            self.assertAlmostEqual(
                guard.recovery_retract_s,
                base_guard.recovery_retract_s * time_scale,
            )
            self.assertAlmostEqual(
                guard.recovery_retouch_s,
                base_guard.recovery_retouch_s * time_scale,
            )
            self.assertTrue(guard.hold_terminal_set_on_retry)
            self.assertTrue(guard.bumpless_terminal_entry_rebase)
            self.assertTrue(guard.axial_position_hold_in_terminal_set)
            self.assertTrue(
                commissioned.oracle_transport_soft_escape_recovery
            )
            self.assertAlmostEqual(
                commissioned.oracle_center_tolerance_capture_fraction,
                0.25,
            )
            self.assertAlmostEqual(
                guard.descent_speed_m_s,
                base_guard.descent_speed_m_s / time_scale,
            )
            self.assertAlmostEqual(
                guard.planar_stiffness_n_m,
                base_guard.planar_stiffness_n_m / scale,
            )
            self.assertAlmostEqual(
                guard.planar_damping_n_s_m,
                base_guard.planar_damping_n_s_m / time_scale,
            )
            self.assertAlmostEqual(
                guard.deep_tilt_stiffness_nm_rad,
                base_guard.deep_tilt_stiffness_nm_rad / (scale * scale),
            )
            self.assertAlmostEqual(
                guard.deep_tilt_damping_nm_s_rad,
                base_guard.deep_tilt_damping_nm_s_rad / scale,
            )
            self.assertAlmostEqual(
                outer.stiffness[0], base_admittance.stiffness[0] / scale
            )
            self.assertAlmostEqual(
                outer.damping[0],
                base_admittance.damping[0] / time_scale,
            )
            self.assertAlmostEqual(
                outer.stiffness[3],
                base_admittance.stiffness[3] / (scale * scale),
            )
            self.assertAlmostEqual(
                outer.damping[3], base_admittance.damping[3] / scale,
            )
            self.assertAlmostEqual(
                outer.max_velocity[2],
                base_admittance.max_velocity[2] / time_scale,
            )
            self.assertAlmostEqual(
                outer.max_acceleration[2],
                base_admittance.max_acceleration[2] / scale,
            )

            # Commissioning changes compliance and nominal rate, never the raw
            # hard-stop vector or the funnel's independent safety thresholds.
            self.assertEqual(outer.hard_wrench, base_admittance.hard_wrench)
            self.assertEqual(
                guard.injected_power_limit_w, base_guard.injected_power_limit_w
            )
            self.assertEqual(guard.max_tilt_deg, base_guard.max_tilt_deg)
            self.assertEqual(guard.no_progress_energy_j, base_guard.no_progress_energy_j)

        self.assertFalse(m8.oracle_transport_soft_escape_recovery)

        self.assertLess(
            m12.funnel_guard.planar_stiffness_n_m,
            m10.funnel_guard.planar_stiffness_n_m,
        )
        self.assertLess(
            m12.funnel_guard.descent_speed_m_s,
            m10.funnel_guard.descent_speed_m_s,
        )
        self.assertEqual(funnel_guard_for_size("M12"), m12.funnel_guard)

    def test_invalid_or_smaller_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            commissioning_for_size("M14")
        with self.assertRaises(ValueError):
            commissioning_for_bit_radius(
                0.5 * REFERENCE_BIT_RADIUS_M,
                size="M6",
            )

    def test_setup_passes_custom_admittance_to_rollout(self) -> None:
        commissioned = commissioning_for_size("M10")
        spec, model, data, rollout, renderer, camera, scene_option = setup(
            comparison_case("baseline"),
            render=False,
            admittance_config=commissioned.admittance,
            spec_path=ROOT / "configs/allen_m10_source.json",
        )
        self.assertEqual(spec["name"], "m10")
        self.assertIsNone(renderer)
        self.assertEqual(
            rollout.controller.config.stiffness,
            commissioned.admittance.stiffness,
        )
        self.assertEqual(
            rollout.controller.config.hard_wrench,
            commissioned.admittance.hard_wrench,
        )
        # MuJoCoAdmittanceRollout owns the actual controller period and replaces
        # only dt; every commissioned physical parameter is preserved.
        self.assertAlmostEqual(
            rollout.controller.config.dt_s,
            rollout.controller_dt_s,
        )

    def test_run_method_forwards_admittance_to_setup(self) -> None:
        commissioned = commissioning_for_size("M12")
        with patch(
            "src.render_m8_search_comparison.setup",
            side_effect=_SetupIntercepted,
        ) as mocked_setup:
            with self.assertRaises(_SetupIntercepted):
                run_method(
                    "baseline",
                    admittance_config=commissioned.admittance,
                )
        self.assertIs(
            mocked_setup.call_args.kwargs["admittance_config"],
            commissioned.admittance,
        )


if __name__ == "__main__":
    unittest.main()
