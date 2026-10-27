import unittest

import numpy as np

from src.render_m8_search_comparison import center_posterior_from_direction
from src.run_m8_learned_search_benchmark import METHODS, build_tasks
import src.train_m8_direction_models as training


class TestM8LearnedSearchBenchmark(unittest.TestCase):
    def test_benchmark_uses_held_out_pose_triplets(self) -> None:
        tasks = build_tasks()
        self.assertEqual(len(tasks), 54)
        pose_ids = {task.pose_id for task in tasks}
        self.assertEqual(len(pose_ids), 18)
        for pose_id in pose_ids:
            self.assertEqual(
                {task.method for task in tasks if task.pose_id == pose_id}, set(METHODS)
            )

    def test_direction_likelihood_shifts_visual_prior_without_true_center(self) -> None:
        probability = np.full(training.DIRECTION_CLASSES, 1.0e-4)
        zero_class = training._direction_class(3.75)
        probability[zero_class] = 1.0
        probability /= probability.sum()
        samples = center_posterior_from_direction(
            np.zeros(2), probability, 0.002, np.random.default_rng(4)
        )
        self.assertGreater(float(np.mean(samples[:, 0])), 0.001)
        self.assertLess(abs(float(np.mean(samples[:, 1]))), 0.0003)


if __name__ == "__main__":
    unittest.main()
