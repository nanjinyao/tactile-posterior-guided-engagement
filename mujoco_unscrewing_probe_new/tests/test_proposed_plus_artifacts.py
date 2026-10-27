from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SIX = ROOT / "outputs/current/diagnostics/historical_comparisons/six_methods/manifest.json"
RACE = ROOT / "outputs/current/diagnostics/historical_comparisons/three_method_race/manifest.json"


class TestProposedPlusArtifacts(unittest.TestCase):
    def test_shortcut_precedes_task_grid_and_local_spiral_is_disabled(self) -> None:
        manifest = json.loads(SIX.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["proposed_plus_execution_order"],
            [
                "conical_fingerprint",
                "bounded_posterior_shortcut",
                "capture_probability_plus_information_grid_nodes",
                "direction_consistent_sensor_capture_or_full_visual_support_last_resort",
                "shared_post_capture_funnel",
            ],
        )
        result = manifest["results"]["hybrid"]
        gates = result["hybrid_probe_gate_records"]
        self.assertEqual(gates[0]["stage"], "posterior_shortcut_before_entropy_grid")
        probe_indices = [
            index for index, gate in enumerate(gates) if gate.get("probe") is not None
        ]
        self.assertTrue(probe_indices)
        self.assertGreater(min(probe_indices), 0)
        self.assertEqual(result["entropy_grid_probe_count"], len(probe_indices))
        self.assertFalse(result["entropy_grid_used_spiral"])
        self.assertFalse(result["hybrid_probe_config"]["local_spiral_fallback_enabled"])
        self.assertTrue(
            any(
                gate.get("action_mode")
                in {"capture_plus_information", "capture_weighted_coverage"}
                for gate in gates
            )
        )
        self.assertFalse(result["global_outward_fallback_used"])
        self.assertEqual(result["result"], "cad_bottom_verified")

    def test_race_reuses_the_same_frozen_hybrid_result(self) -> None:
        six = json.loads(SIX.read_text(encoding="utf-8"))
        race = json.loads(RACE.read_text(encoding="utf-8"))
        self.assertEqual(
            race["proposed_plus_execution_order"],
            six["proposed_plus_execution_order"],
        )
        self.assertEqual(race["results"]["hybrid"], six["results"]["hybrid"])
        self.assertFalse(race["display_fast_forward"]["changes_simulation_or_metrics"])


if __name__ == "__main__":
    unittest.main()
