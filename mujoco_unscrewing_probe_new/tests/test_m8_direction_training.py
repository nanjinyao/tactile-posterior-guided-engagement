import unittest

import numpy as np

from src.train_m8_direction_models import (
    P0A,
    P0B,
    P0D,
    _augment_sequence,
    _direction_angle_deg,
    _direction_class,
    _circular_soft_targets,
    _select_confidence_gate,
    grouped_split,
    grouped_sector_split,
    grouped_targeted_split,
    load_trials,
)


class TestM8DirectionTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trials = load_trials(P0B)

    def test_grouped_split_is_disjoint_and_balanced(self) -> None:
        split = grouped_split(self.trials)
        self.assertEqual({key: len(value) for key, value in split.items()}, {"train": 72, "val": 24, "test": 24})
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["val"]) & set(split["test"]))

    def test_c6_augmentation_rotates_direction_class(self) -> None:
        trial = self.trials[0]
        base_class = _direction_class(_direction_angle_deg(trial.metadata, 0))
        rotated_class = _direction_class(_direction_angle_deg(trial.metadata, 1))
        self.assertEqual((rotated_class - base_class) % 48, 8)
        augmented = _augment_sequence(trial.sequence, trial.columns, 1)
        self.assertEqual(augmented.shape, trial.sequence.shape)
        self.assertTrue(np.all(np.isfinite(augmented)))

    def test_physical_rotation_is_included_in_direction_label(self) -> None:
        trial = next(item for item in load_trials(P0A) if item.metadata["rotation_index"] == 1)
        expected = (
            trial.metadata["base_angle_deg"] + trial.metadata["rotation_deg"] + 180.0
        ) % 360.0
        self.assertAlmostEqual(_direction_angle_deg(trial.metadata), expected)

    def test_p0d_split_is_spatially_disjoint_before_nuisance_expansion(self) -> None:
        split = grouped_sector_split(load_trials(P0D))
        self.assertEqual({key: len(value) for key, value in split.items()}, {"train": 15, "val": 6, "test": 6})
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["val"]) & set(split["test"]))

    def test_confidence_gate_is_selected_only_from_supplied_calibration_data(self) -> None:
        logits = np.full((30, 48), -4.0, dtype=float)
        angles = np.full(30, 3.75, dtype=float)
        logits[:, 0] = 4.0
        logits[0, 0] = -1.0
        logits[0, 24] = 1.0
        threshold = _select_confidence_gate(logits, angles, 1.0, min_accepted=24)
        self.assertGreater(threshold, 0.5)
        self.assertLessEqual(threshold, 1.0)

    def test_circular_soft_targets_are_normalized_and_wrap_at_zero(self) -> None:
        targets = _circular_soft_targets(np.asarray([0.0, 359.0]), 7.5)
        np.testing.assert_allclose(np.sum(targets, axis=1), 1.0, atol=1.0e-6)
        self.assertLess(np.argmax(targets[0]), 48)
        self.assertGreater(float(targets[0, 0] + targets[0, -1]), 0.3)

    def test_targeted_split_is_group_disjoint(self) -> None:
        targeted = load_trials(P0B.parent / "p1v3_m8_targeted_direction_training")
        if not targeted:
            self.skipTest("targeted dataset not sampled yet")
        split = grouped_targeted_split(targeted)
        self.assertFalse(set(split["train"]) & set(split["val"]))


if __name__ == "__main__":
    unittest.main()
