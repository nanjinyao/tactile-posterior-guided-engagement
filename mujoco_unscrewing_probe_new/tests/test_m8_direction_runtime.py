import math
from pathlib import Path
import unittest

import numpy as np

from src.m8_direction_runtime import M8DirectionEnsemble
import src.train_m8_direction_models as training


ROOT = Path(__file__).resolve().parents[1]


class TestM8DirectionRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = M8DirectionEnsemble()
        cls.spec = training.SPEC

    def test_runtime_reproduces_grouped_p0d_direction_evidence(self) -> None:
        trials = training.load_trials(training.P0D)
        split = training.grouped_sector_split(trials, 20260710)
        selected = [
            trial
            for trial in trials
            if training._sector_pose_id(trial.metadata) in set(split["test"])
        ]
        bit_radius = float(self.spec["key_af_m"]) / math.sqrt(3.0)
        errors = []
        for trial in selected:
            indices = [
                trial.columns.index(name)
                for name in ("Fz_N", "Tx_Nm", "Ty_Nm", "tip_z_m")
            ]
            estimate = self.runtime.infer_phase_sequence(
                trial.sequence[indices],
                target_down_force_n=float(trial.metadata["target_down_force_n"]),
                bit_radius_m=bit_radius,
            )
            true_angle = training._direction_angle_deg(trial.metadata)
            errors.append(
                float(
                    training._circular_error_deg(
                        np.asarray([estimate.top_direction_deg]),
                        np.asarray([true_angle]),
                    )[0]
                )
            )
            self.assertAlmostEqual(float(np.sum(estimate.direction_probabilities)), 1.0, places=5)
            self.assertGreaterEqual(len(estimate.aps_direction_classes), 1)
            self.assertTrue(0.0 <= estimate.circular_mean_direction_deg < 360.0)
        self.assertEqual(len(errors), 72)
        self.assertGreaterEqual(float(np.mean(np.asarray(errors) <= 7.5)), 0.95)


if __name__ == "__main__":
    unittest.main()
