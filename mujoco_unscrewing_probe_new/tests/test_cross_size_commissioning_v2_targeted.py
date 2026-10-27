from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.run_cross_size_commissioning_v2_targeted import (
    DEFAULT_SOURCE,
    METHODS,
    build_contract,
    select_oracle_failure_pose_ids,
)


class TestCrossSizeCommissioningV2Targeted(unittest.TestCase):
    def test_default_selection_is_exactly_the_known_failed_m10_oracle_set(self) -> None:
        self.assertEqual(
            select_oracle_failure_pose_ids(DEFAULT_SOURCE, "m10"),
            (
                "m10_s3test_002",
                "m10_s3test_004",
                "m10_s3test_006",
                "m10_s3test_007",
                "m10_s3test_010",
            ),
        )

    def test_contract_is_development_only_and_never_relaxes_raw_limits(self) -> None:
        pose_ids = select_oracle_failure_pose_ids(DEFAULT_SOURCE, "m10")
        contract, tasks = build_contract(DEFAULT_SOURCE, "m10", pose_ids, METHODS)
        self.assertEqual(contract["phase"], "development")
        self.assertFalse(contract["untouched_tasks_or_seeds_read"])
        self.assertFalse(contract["raw_hard_wrench_thresholds_relaxed"])
        self.assertEqual(contract["pose_ids"], list(pose_ids))
        self.assertEqual([task.pose_id for task in tasks], list(pose_ids))
        self.assertEqual(
            contract["commissioning_v2"]["commissioning_version"],
            "v2_sqrt_time_separate_rotation_lever_bumpless_terminal_hold_"
            "oracle_soft_escape_resume",
        )
        self.assertTrue(
            contract["commissioning_v2"][
                "oracle_transport_soft_escape_recovery"
            ]
        )
        self.assertEqual(
            list(contract["commissioning_v2"]["admittance"]["hard_wrench"]),
            contract["raw_hard_wrench_thresholds"],
        )
        self.assertNotIn("untouched", json.dumps(contract["tasks"]).lower())

    def test_non_m10_scope_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_contract(
                DEFAULT_SOURCE,
                "m12",
                ("m12_s3test_011",),
                METHODS,
            )


if __name__ == "__main__":
    unittest.main()
