import unittest

from src.run_m8_nominal_fsm_acceptance import (
    METHODS,
    NOMINAL_FULL_SUPPORT_RADIUS_M,
    build_tasks,
)


class TestM8NominalFsmAcceptance(unittest.TestCase):
    def test_design_is_eighteen_nominal_paired_poses(self) -> None:
        tasks = build_tasks()
        self.assertEqual(len(tasks), 36)
        pose_ids = {task.pose_id for task in tasks}
        self.assertEqual(len(pose_ids), 18)
        self.assertEqual(len({round(task.radius_m, 12) for task in tasks}), 3)
        self.assertTrue(all(task.radius_m <= NOMINAL_FULL_SUPPORT_RADIUS_M for task in tasks))
        for pose_id in pose_ids:
            self.assertEqual(
                {task.method for task in tasks if task.pose_id == pose_id}, set(METHODS)
            )


if __name__ == "__main__":
    unittest.main()
