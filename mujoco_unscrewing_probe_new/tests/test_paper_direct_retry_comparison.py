from __future__ import annotations

import json
from pathlib import Path
import unittest


class TestPaperDirectRetryComparison(unittest.TestCase):
    def test_scripted_retry_stress_is_traceable_and_not_an_active_baseline(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                root
                / "outputs/current/diagnostics/paper_direct_scripted_stress/result.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(payload["paper_source"]["path"], "reference/锥形探索.pdf")
        self.assertIn("not an exact reproduction", payload["claim_boundary"])
        self.assertIn("not a random draw", payload["scripted_stress"]["selection_boundary"])
        self.assertIn("not an active baseline", payload["claim_boundary"].lower())
        self.assertEqual(
            payload["scripted_stress"]["direction_errors_deg"],
            [22.5, -22.5, 0.0],
        )
        self.assertTrue(payload["fairness"]["proposed_probe_cost_included"])
        self.assertTrue(payload["fairness"]["paper_retry_probe_cost_included"])

        paper = payload["paper_inspired"]
        self.assertTrue(paper["capture_confirmed"])
        self.assertEqual(paper["successful_attempt"], 3)
        self.assertFalse(paper["hard_stop"])
        self.assertEqual(
            [attempt["outcome"] for attempt in paper["attempts"]],
            ["failed_no_capture", "failed_no_capture", "capture_confirmed"],
        )

        proposed = payload["proposed"]
        self.assertLess(proposed["time_to_inner_wall_s"], paper["time_to_capture_s"])
        self.assertLess(proposed["actual_xy_path_mm"], paper["actual_xy_path_mm"])
        self.assertTrue(payload["comparison"]["proposed_faster"])
        self.assertTrue(payload["comparison"]["proposed_has_bounded_residual_coverage"])
        self.assertFalse(payload["comparison"]["paper_direct_has_residual_coverage"])

        for relative in payload["videos"].values():
            video = root / relative
            self.assertFalse(video.exists())
        for relative in payload["keyframes"].values():
            keyframe = root / relative
            self.assertTrue(keyframe.is_file())
            self.assertGreater(keyframe.stat().st_size, 50_000)


if __name__ == "__main__":
    unittest.main()
