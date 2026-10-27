from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "m8_sampling_topview_dashboard.m"
SHARED = ROOT / "m12_sampling_topview_dashboard.m"


class TestM8SamplingDashboardStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facade = FACADE.read_text(encoding="utf-8")
        cls.shared = SHARED.read_text(encoding="utf-8")

    def test_m8_facade_pins_size_and_preserves_action_api(self) -> None:
        self.assertIn(
            "function dashboard = m8_sampling_topview_dashboard("
            "action, dashboard, varargin)",
            self.facade,
        )
        self.assertIn("'sampling_size_label', 'M8'", self.facade)
        self.assertRegex(
            self.facade,
            r"m12_sampling_topview_dashboard\(\s*\.\.\.\s*"
            r"action,\s*dashboard,\s*sizePayload,\s*varargin\{:\}\)",
        )

    def test_shared_renderer_validates_actual_printed_m8_geometry(self) -> None:
        geometry = re.search(
            r"if strcmp\(sizeLabel, 'M8'\)(.*?)"
            r"elseif strcmp\(sizeLabel, 'M10'\)",
            self.shared,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertIn(
            "expected = [14.95, 8.1, 6.5, 0.5, 4.8, 6.0]",
            geometry.group(1),
        )
        self.assertIn("strcmp(sizeLabel, 'M8')", self.shared)
        for token in (
            "head_outer_diameter",
            "socket_mouth_af",
            "socket_deep_wall_af",
            "center_through_hole_diameter",
            "peg_tip_face_af",
            "peg_shaft_af",
        ):
            self.assertIn(token, self.shared)

    def test_shared_renderer_keeps_lightweight_ui_contract(self) -> None:
        self.assertIn(
            "'Name', sprintf('%s 26-point sampling top view', sizeLabel)",
            self.shared,
        )
        self.assertIn(
            "%s sampling from +Z_F: +X_F left, +Y_F down, "
            "+Z_F toward viewer",
            self.shared,
        )
        self.assertIn("'maximum_refresh_hz', 2.0", self.shared)
        self.assertIn("'minimum_refresh_interval_s', 0.5", self.shared)
        self.assertIn(
            "~payloadLogical(payload, 'allow_drawnow', false)",
            self.shared,
        )
        self.assertIn("drawnow limitrate nocallbacks", self.shared)
        for forbidden in (
            "rossubscriber(",
            "getEEFPos(",
            "getJointsPos(",
            "getEEFForce(",
            "timer(",
        ):
            self.assertNotIn(forbidden, self.shared)


if __name__ == "__main__":
    unittest.main()
