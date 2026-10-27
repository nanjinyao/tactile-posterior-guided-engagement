"""updateDashboard must never call drawnow from the robot-command thread.

On this Ubuntu MATLAB Figures frontend, drawnow -- including drawnow limitrate
-- can wedge the figure system. The engine therefore accumulates points during
a run and leaves painting to the single post-cone phase-boundary drawnow and to
the return to the prompt.

On 2026-07-29 that rule was relaxed: the pure outward spiral has no cone, so it
has no phase-boundary drawnow either, and it showed an unpainted window for its
whole search, which an operator reasonably read as a hang. The "safe" repaint
added to fix it timed each drawnow against a budget and self-disabled on
overrun. That is not sufficient. The very first run that exercised the repaint
together with a cone hung: the timed call returned well inside its budget but
left the frontend wedged, the post-cone bring-to-front drawnow never returned,
and no window appeared at all.

A blank window is a display defect. That was a deadlock in the middle of a real
robot experiment, which is strictly worse. This test locks the revert.

If the blank-window problem is worth solving again, the fix must be validated on
a method that runs a cone (methods 1, 2 or 3), not only on the pure spiral.
"""

from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"


class TestDashboardNoCommandThreadDrawnow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = MAIN.read_text(encoding="utf-8", errors="replace")
        update = cls.main[cls.main.index("function dashboard = updateDashboard(") :]
        cls.update = update[: update.index("\nfunction ")]
        cls.update_code = "\n".join(
            line for line in cls.update.split("\n")
            if not line.strip().startswith("%")
        )

    def test_update_dashboard_never_paints(self):
        self.assertNotIn("drawnow", self.update_code)

    def test_the_reverted_repaint_machinery_is_gone(self):
        for token in (
            "search_paint_interval_s",
            "search_paint_budget_s",
            "painting_enabled",
            "last_paint_elapsed_s",
        ):
            self.assertNotIn(token, self.main)

    def test_the_reason_for_the_rule_is_recorded_where_it_is_enforced(self):
        # A bare "no drawnow here" invites the same well-meaning fix again.
        self.assertIn("can block indefinitely", self.update)
        self.assertIn("wedged", self.update)
        self.assertIn("validated on a method that runs a cone", self.update)


if __name__ == "__main__":
    unittest.main()
