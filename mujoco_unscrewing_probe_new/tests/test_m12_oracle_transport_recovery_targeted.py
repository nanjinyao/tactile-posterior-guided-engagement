from __future__ import annotations

import json
import unittest

from src.run_m12_oracle_transport_recovery_targeted import (
    DEFAULT_SOURCE,
    POSE_IDS,
    build_contract,
)


class TestM12OracleTransportRecoveryTargeted(unittest.TestCase):
    def test_contract_is_frozen_to_three_development_failures(self) -> None:
        contract, tasks = build_contract(DEFAULT_SOURCE)
        self.assertEqual(contract["phase"], "development")
        self.assertEqual(contract["pose_ids"], list(POSE_IDS))
        self.assertEqual([task.pose_id for task in tasks], list(POSE_IDS))
        self.assertFalse(contract["untouched_tasks_or_seeds_read"])
        self.assertNotIn("untouched", json.dumps(contract["tasks"]).lower())

    def test_intervention_keeps_every_threshold_and_nonoracle_arm_fixed(self) -> None:
        contract, _ = build_contract(DEFAULT_SOURCE)
        self.assertTrue(
            contract["commissioning"][
                "oracle_transport_soft_escape_recovery"
            ]
        )
        self.assertAlmostEqual(
            contract["commissioning"][
                "oracle_center_tolerance_capture_fraction"
            ],
            0.25,
        )
        self.assertFalse(contract["raw_hard_wrench_thresholds_relaxed"])
        self.assertFalse(contract["soft_wrench_thresholds_changed"])
        self.assertFalse(contract["capture_thresholds_changed"])
        self.assertFalse(contract["terminal_verifier_thresholds_changed"])
        self.assertFalse(contract["non_oracle_behavior_changed"])


if __name__ == "__main__":
    unittest.main()
