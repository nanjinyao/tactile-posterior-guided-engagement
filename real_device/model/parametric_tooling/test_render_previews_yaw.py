#!/usr/bin/env python3
"""Offline contract checks for the handed v1.5 yaw-datum preview."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from render_previews import cad_xy_to_tip_view, unit_xy


def wrap_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class TipSideYawPreviewTest(unittest.TestCase):
    def test_tip_side_projection_has_declared_handedness(self) -> None:
        projected = cad_xy_to_tip_view(np.array([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(projected[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(projected[1], [0.0, -1.0], atol=1e-12)

        positive_yaw_tip = cad_xy_to_tip_view(unit_xy(30.0))
        self.assertGreater(positive_yaw_tip[0], 0.0)
        self.assertLess(
            positive_yaw_tip[1],
            0.0,
            "positive +Z yaw must appear clockwise in the tip-side view",
        )

    def test_manifest_clocking_maps_once_to_fixture_x(self) -> None:
        manifest = json.loads(
            (HERE / "generated" / "manifest.json").read_text(encoding="utf-8")
        )
        contract = manifest["peg_flange_yaw_contract"]
        shape_rotation = float(contract["shape_rotation_about_positive_z_deg"])
        selected_flat = float(
            contract["selected_peg_flat_normal_in_flange_deg"]
        )
        flange_normal = float(contract["flange_reference_tangent_normal_deg"])
        socket_flat = float(contract["socket_selected_flat_normal_deg"])

        self.assertAlmostEqual(shape_rotation, -15.0)
        self.assertAlmostEqual(selected_flat, 45.0)
        self.assertAlmostEqual(flange_normal, 45.0)
        self.assertAlmostEqual(socket_flat, 0.0)
        self.assertAlmostEqual(wrap_degrees(selected_flat - flange_normal), 0.0)

        fixture_from_flange_yaw = -flange_normal
        installed_peg_flat = wrap_degrees(
            selected_flat + fixture_from_flange_yaw
        )
        self.assertAlmostEqual(installed_peg_flat, socket_flat)


if __name__ == "__main__":
    unittest.main()
