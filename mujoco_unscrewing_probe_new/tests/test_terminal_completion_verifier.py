from __future__ import annotations

import unittest

import numpy as np

from src.terminal_completion_verifier import (
    TerminalCompletionVerifier,
    TerminalVerifierConfig,
)


class TestTerminalCompletionVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TerminalVerifierConfig(
            depth_tolerance_mm=0.15,
            bottom_force_n=0.70,
            low_lateral_force_n=0.25,
            low_torque_xy_mnm=1.5,
            stable_depth_span_mm=0.02,
            dwell_steps=4,
        )

    def test_bottom_force_dwell_remains_a_valid_terminal_cue(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for _ in range(4):
            mode = verifier.update(
                depth_error_mm=0.05,
                tip_z_mm=-4.05,
                measured_wrench=np.array([0.1, 0.0, 0.8, 0.0, 0.0, 0.0]),
            )
        self.assertEqual(mode, TerminalCompletionVerifier.FORCE_DWELL)

    def test_reverse_axial_force_cannot_pass_either_channel(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for _ in range(8):
            mode = verifier.update(
                depth_error_mm=0.01,
                tip_z_mm=-4.0,
                measured_wrench=np.array([0.0, 0.0, -0.8, 0.0, 0.0, 0.0]),
            )
        self.assertIsNone(mode)

    def test_lateral_jam_cannot_use_bottom_force_channel(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for _ in range(8):
            mode = verifier.update(
                depth_error_mm=0.01,
                tip_z_mm=-4.0,
                measured_wrench=np.array([0.4, 0.0, 0.8, 0.0, 0.0, 0.0]),
            )
        self.assertIsNone(mode)

    def test_stable_low_wrench_depth_dwell_accepts_free_fit_insertion(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for z in (-4.001, -4.004, -4.002, -4.003):
            mode = verifier.update(
                depth_error_mm=abs(z + 4.0),
                tip_z_mm=z,
                measured_wrench=np.array([0.02, -0.01, 0.05, 0.0002, 0.0001, 0.0]),
            )
        self.assertEqual(
            mode, TerminalCompletionVerifier.LOW_WRENCH_DEPTH_DWELL
        )

    def test_high_lateral_load_cannot_use_low_wrench_channel(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for _ in range(8):
            mode = verifier.update(
                depth_error_mm=0.01,
                tip_z_mm=-4.0,
                measured_wrench=np.array([0.4, 0.0, 0.1, 0.0, 0.0, 0.0]),
            )
        self.assertIsNone(mode)

    def test_unstable_depth_cannot_use_low_wrench_channel(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        mode = None
        for z in (-4.00, -4.04, -4.01, -4.05):
            mode = verifier.update(
                depth_error_mm=0.05,
                tip_z_mm=z,
                measured_wrench=np.zeros(6),
            )
        self.assertIsNone(mode)

    def test_diagnostics_preserve_depth_and_gate_reset_causes(self) -> None:
        verifier = TerminalCompletionVerifier(self.config)
        verifier.update(
            depth_error_mm=0.20,
            tip_z_mm=-3.80,
            measured_wrench=np.zeros(6),
        )
        for _ in range(3):
            verifier.update(
                depth_error_mm=0.10,
                tip_z_mm=-3.90,
                measured_wrench=np.array([0.40, 0.0, 0.80, 0.0, 0.0, 0.0]),
            )
        diagnostic = verifier.diagnostics()
        self.assertEqual(diagnostic["updates"], 4)
        self.assertEqual(diagnostic["depth_band_entries"], 1)
        self.assertEqual(diagnostic["depth_band_samples"], 3)
        self.assertEqual(diagnostic["max_consecutive_depth_band_samples"], 3)
        self.assertEqual(diagnostic["max_force_dwell_samples"], 0)
        self.assertEqual(diagnostic["gate_failure_counts"]["out_of_depth_band"], 1)
        self.assertEqual(diagnostic["gate_failure_counts"]["force_lateral"], 3)
        self.assertTrue(diagnostic["last"]["in_depth_band"])


if __name__ == "__main__":
    unittest.main()
