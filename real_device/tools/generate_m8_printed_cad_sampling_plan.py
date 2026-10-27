#!/usr/bin/env python3
"""Generate the M8 26-point acquisition plan from the printed CAD."""

from __future__ import annotations

import generate_m12_printed_cad_sampling_plan as core


class M8ContactGeometry(core.ContactGeometry):
    """Keep the M8 outer-transition samples on both sides of the CAD boundary."""

    def radius_for_state(
        self, state: str, subtype: str, angle_deg: float
    ) -> tuple[float, str]:
        if (
            state == "outer_partial_support"
            and subtype == "landing_mouth_clear_cone_transition"
        ):
            landing_mouth_clear = self.solve_radius(
                angle_deg,
                "mouth_overlap_fraction",
                1e-6,
                decreasing=True,
            )
            return (
                landing_mouth_clear + 0.70,
                "M8 tip-face landing mouth-clear boundary plus 0.70 mm",
            )
        return super().radius_for_state(state, subtype, angle_deg)


def configure_m8_target() -> None:
    """Bind the proven printed-CAD planner to the independent M8 outputs."""

    core.TARGET_SIZE = "m8"
    core.TARGET_LABEL = "M8"
    core.PLAN_NAME = "m8_printed_cad_conical_sampling_v2"
    core.GENERATOR_NAME = "generate_m8_printed_cad_sampling_plan.py"
    core.PEG_STEP = core.GENERATED_ROOT / "integrated_flange_peg_m8.step"
    core.PLAN_PATH = (
        core.REAL_ROOT / "configs" / "m8_printed_cad_conical_sampling_v2.json"
    )
    core.CSV_PATH = (
        core.REAL_ROOT / "docs" / "m8_printed_cad_sampling_points.csv"
    )
    core.PNG_PATH = (
        core.REAL_ROOT / "docs" / "m8_printed_cad_sampling_top_view.png"
    )
    core.SVG_PATH = (
        core.REAL_ROOT / "docs" / "m8_printed_cad_sampling_top_view.svg"
    )
    core.OPERATOR_PNG_PATHS = (
        core.REAL_ROOT / "m8采样规划图.png",
        core.REAL_ROOT / "采样规划图.png",
    )
    core.EXPECTED_PEG_STEP_SHA256 = (
        "22303873354eeb02d82c2492cb3a5d6b8ba73860b31a67804ca501347692f24b"
    )
    core.EXPECTED_GEOMETRY = {
        "head_outer_diameter": 14.95,
        "socket_mouth_af": 8.1,
        "socket_deep_wall_af": 6.5,
        "socket_chamfer_width": 0.8,
        "socket_chamfer_depth": 0.8,
        "center_through_hole_diameter": 0.5,
        "peg_tip_face_af": 4.8,
        "peg_shaft_af": 6.0,
        "peg_chamfer_radial_setback": 0.6,
        "peg_chamfer_axial_length": 0.6,
    }
    core.NOMINAL_FIXTURE_CENTER_XY_MM = (-45.0, 0.0)
    core.ContactGeometry = M8ContactGeometry


if __name__ == "__main__":
    configure_m8_target()
    raise SystemExit(core.main())
