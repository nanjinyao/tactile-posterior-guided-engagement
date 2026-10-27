from __future__ import annotations

import unittest

import numpy as np

from src.render_m8_search_comparison import ConicalContactTransitionMonitor


def _wrench(
    *,
    fz_n: float,
    fx_n: float = 0.0,
    fy_n: float = 0.0,
    tx_nm: float = 0.0,
    ty_nm: float = 0.0,
) -> np.ndarray:
    return np.asarray([fx_n, fy_n, fz_n, tx_nm, ty_nm, 0.0], dtype=float)


class TestConicalContactTransitionMonitor(unittest.TestCase):
    def make_monitor(self) -> ConicalContactTransitionMonitor:
        return ConicalContactTransitionMonitor(
            start_time_s=0.0,
            start_tip_z_m=0.00030,
            target_down_force_n=3.0,
        )

    def test_brief_low_support_resumes_but_invalidates_fingerprint(self) -> None:
        monitor = self.make_monitor()
        decision = None
        for index in range(1, 5):
            decision = monitor.observe(
                time_s=0.002 * index,
                tip_z_m=0.00030,
                measured_wrench=_wrench(fz_n=0.5),
                evidence_relevant=True,
            )
            self.assertFalse(decision.stop_cone)
        decision = monitor.observe(
            time_s=0.010,
            tip_z_m=0.00030,
            measured_wrench=_wrench(fz_n=3.0),
            evidence_relevant=True,
        )
        self.assertFalse(decision.stop_cone)
        self.assertTrue(decision.fingerprint_incomplete)
        self.assertIsNone(decision.event)
        self.assertEqual(monitor.summary()["transient_low_support_count"], 1)

    def test_sustained_low_support_stops_remaining_cone(self) -> None:
        monitor = self.make_monitor()
        decision = None
        for index in range(1, 8):
            decision = monitor.observe(
                time_s=0.002 * index,
                tip_z_m=0.00030,
                measured_wrench=_wrench(fz_n=0.4),
                evidence_relevant=True,
            )
        assert decision is not None
        self.assertTrue(decision.stop_cone)
        self.assertEqual(
            decision.event,
            "sustained_low_support_during_conical_probe",
        )
        self.assertTrue(decision.fingerprint_incomplete)

    def test_safe_downward_progress_stops_even_without_force_loss(self) -> None:
        monitor = self.make_monitor()
        decision = None
        for index in range(1, 31):
            fraction = index / 30.0
            decision = monitor.observe(
                time_s=0.002 * index,
                tip_z_m=0.00030 - fraction * 0.00010,
                measured_wrench=_wrench(fz_n=3.0, fx_n=0.2),
                evidence_relevant=True,
            )
            if decision.stop_cone:
                break
        assert decision is not None
        self.assertTrue(decision.stop_cone)
        self.assertEqual(
            decision.event,
            "safe_downward_progress_during_conical_probe",
        )
        self.assertTrue(decision.safe_wrench)

    def test_lateral_load_blocks_safe_progress_classification(self) -> None:
        monitor = self.make_monitor()
        decision = None
        for index in range(1, 31):
            fraction = index / 30.0
            decision = monitor.observe(
                time_s=0.002 * index,
                tip_z_m=0.00030 - fraction * 0.00010,
                measured_wrench=_wrench(fz_n=3.0, fx_n=2.5),
                evidence_relevant=True,
            )
        assert decision is not None
        self.assertFalse(decision.stop_cone)
        self.assertFalse(decision.safe_wrench)
        self.assertIsNone(decision.event)

    def test_monitor_contract_declares_no_privileged_contact_inputs(self) -> None:
        summary = self.make_monitor().summary()
        self.assertFalse(summary["uses_contact_count"])
        self.assertFalse(summary["uses_contact_geometry_truth"])


if __name__ == "__main__":
    unittest.main()
