"""Exhausting the sector-conditioned residual grid must widen, not give up.

The residual grid's admissible actions are conditioned to a sector around the
learned direction. That sector is only as good as the direction: when the
direction error exceeds the corridor half-width, the sector provably cannot
contain the hole, and every node in it can be swept without ever getting close.

Before 2026-07-30 that case broke out of the grid loop and handed control to the
blind phase-shifted coverage fallback, which re-walks ground the grid already
swept. The 39 deg run exhausted 11 s after the grid was bound, with essentially
no productive node, then spent 130 s and seven passes repeating itself -- the
operator watched what looked like a confused entropy search and was in fact
never the entropy search at all.

The fix releases the sector to the full residual disk once and continues with
the same information-driven selection. This test locks that behaviour, and in
particular that the release happens BEFORE the exhaustion break rather than
after it, which would make it dead code.
"""

from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "execute_m12_contact_engagement_policy.m"


class TestResidualSectorRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = MAIN.read_text(encoding="utf-8", errors="replace")

    def test_release_precedes_and_guards_the_exhaustion_break(self):
        release = self.main.index("RESIDUAL_SECTOR_RELEASED_TO_FULL_DISK")
        exhausted = self.main.index(
            "RESIDUAL_CAD_COVERAGE_GEOMETRICALLY_EXHAUSTED"
        )
        self.assertLess(
            release,
            exhausted,
            "the sector release must be reachable before the break",
        )
        self.assertIn(
            "if candidateIndex == 0 && directionQualified && ...\n"
            "                ~residualSectorReleased",
            self.main,
        )

    def test_release_widens_the_mask_and_continues(self):
        for token in (
            "residualSectorReleased = true;",
            "activeResidualActionMask = true(size(actionOffsets, 1), 1);",
            "cadCoverageMode = false;",
            "continue;",
        ):
            self.assertIn(token, self.main)
        # It must be a one-shot escalation: without the latch the loop could
        # release forever and never reach its own termination.
        self.assertIn("residualSectorReleased = false;", self.main)

    def test_release_also_reopens_the_posterior(self):
        # Widening the action mask alone is a no-op: selection is driven by the
        # posterior's 95% credible region, and the fallback radius is
        # min(q95 spread, ...).  Both stay collapsed unless the refuted
        # directional concentration is released from the belief too.  The
        # 02:18 run proved it -- released, then NO_UNTESTED_95PCT_CREDIBLE_NODE
        # with utility -Inf 12 ms later, then ten 0.75 mm fallback passes.
        start = self.main.index("residualSectorReleased = true;")
        block = self.main[start : start + 2500]
        self.assertIn("sector_release_uniform_weight", block)
        self.assertIn("posterior = posterior / sum(posterior);", block)
        self.assertIn("'sector_release_uniform_weight', 0.50", self.main)

    def test_release_is_recorded_in_the_result(self):
        for token in (
            "result.residual_sector_released = false;",
            "result.residual_sector_released = true;",
            "result.residual_sector_release_step = residualGridSteps;",
        ):
            self.assertIn(token, self.main)

    def test_release_does_not_touch_the_physical_guards(self):
        # Widening the admissible action set is a planning change only.  The
        # posterior, capture radius and reach must not be relaxed with it.
        start = self.main.index("RESIDUAL_SECTOR_RELEASED_TO_FULL_DISK")
        block = self.main[start - 2000 : start + 2000]
        for forbidden in (
            "bayes_capture_radius_mm =",
            "grid_radius_mm =",
            "action_reach_mm =",
            "maximum_grid_nodes =",
        ):
            self.assertNotIn(forbidden, block)


if __name__ == "__main__":
    unittest.main()
