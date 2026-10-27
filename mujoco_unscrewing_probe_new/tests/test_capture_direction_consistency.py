from __future__ import annotations

import unittest

import numpy as np

from src.render_m8_search_comparison import direction_consistent_capture


class TestCaptureDirectionConsistency(unittest.TestCase):
    def test_forward_and_small_local_candidates_are_allowed(self) -> None:
        initial = np.array([0.004, 0.0])
        self.assertTrue(
            direction_consistent_capture(
                initial,
                np.array([0.0001, 0.0001]),
                180.0,
                0.00025,
            )
        )
        self.assertTrue(
            direction_consistent_capture(
                initial,
                initial + np.array([0.0001, -0.0001]),
                180.0,
                0.00025,
            )
        )

    def test_outer_edge_drop_opposite_to_probe_direction_is_rejected(self) -> None:
        initial = np.array([-0.0006748962, 0.0047576079])
        candidate = np.array([0.0010161994, 0.0072570239])
        self.assertFalse(
            direction_consistent_capture(initial, candidate, 281.25, 0.00025)
        )

    def test_reverse_tolerance_is_cad_scaled(self) -> None:
        initial = np.zeros(2)
        self.assertTrue(
            direction_consistent_capture(
                initial,
                np.array([-0.0002, 0.0]),
                0.0,
                0.00025,
            )
        )
        self.assertFalse(
            direction_consistent_capture(
                initial,
                np.array([-0.0003, 0.0]),
                0.0,
                0.00025,
            )
        )


if __name__ == "__main__":
    unittest.main()
