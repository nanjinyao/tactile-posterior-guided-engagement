from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.render_m8_strict_fingerprint_direct_matrix import (
    DEFAULT_STRICT_MANIFEST,
    DISPLAY_NAMES,
    LEGACY_STRICT_FOUR_METHODS_OUTPUT,
    METHODS,
    _full_result_verification,
    _executed_control_direction_error_deg,
    _load_frozen_case,
    _resolve_output_and_stem,
    parse_args,
)
from src.render_m8_search_comparison import (
    DIRECT_FIRST_FAILURE_POLICY_OVERLAY,
    DIRECT_RETRY_FAILURE_POLICY_OVERLAY,
)
from src.run_m8_strict_complete_loop_audit import _run_with_runtime


ROOT = Path(__file__).resolve().parents[1]


class TestM8StrictFingerprintDirectMatrixRenderer(unittest.TestCase):
    def test_current_matrix_labels_do_not_claim_finite_spiral_coverage(self) -> None:
        self.assertIn("Archimedean search", DISPLAY_NAMES["outward"])
        self.assertIn("search reserve", DISPLAY_NAMES["proposed_plus"])
        self.assertNotIn("coverage", DISPLAY_NAMES["outward"].lower())
        self.assertNotIn("coverage", DISPLAY_NAMES["proposed_plus"].lower())

    def test_summary_error_uses_executed_control_direction_and_wraps(self) -> None:
        class Result:
            control_direction_deg = 358.0
            true_direction_deg = 2.0
            direction_error_deg = 37.5

        self.assertEqual(_executed_control_direction_error_deg(Result()), 4.0)

    def test_direct_first_overlay_cannot_advertise_the_historical_retry(self) -> None:
        self.assertIn("stop after 1 ray", DIRECT_FIRST_FAILURE_POLICY_OVERLAY)
        self.assertIn("no retry/reset", DIRECT_FIRST_FAILURE_POLICY_OVERLAY)
        self.assertNotIn("shifted reset", DIRECT_FIRST_FAILURE_POLICY_OVERLAY)
        self.assertIn("shifted reset", DIRECT_RETRY_FAILURE_POLICY_OVERLAY)
        self.assertNotEqual(
            DIRECT_FIRST_FAILURE_POLICY_OVERLAY,
            DIRECT_RETRY_FAILURE_POLICY_OVERLAY,
        )

    def test_strict_runner_explicitly_selects_the_no_retry_overlay(self) -> None:
        task, _, _ = _load_frozen_case(DEFAULT_STRICT_MANIFEST, "sobol_000")
        with patch(
            "src.run_m8_strict_complete_loop_audit.run_method"
        ) as runner:
            _run_with_runtime(task, "direct_first", object())
        self.assertEqual(
            runner.call_args.kwargs["direct_failure_policy_overlay"],
            DIRECT_FIRST_FAILURE_POLICY_OVERLAY,
        )

    def test_any_manifest_frozen_pose_can_be_loaded(self) -> None:
        for pose_id in ("sobol_000", "sobol_008", "sobol_023"):
            with self.subTest(pose_id=pose_id):
                task, trial_path, frozen = _load_frozen_case(
                    DEFAULT_STRICT_MANIFEST, pose_id
                )
                trial = json.loads(trial_path.read_text(encoding="utf-8"))
                self.assertEqual(task.pose_id, pose_id)
                self.assertEqual(asdict(task), trial["task"])
                self.assertEqual(tuple(frozen), METHODS)

    def test_only_observation_fields_may_change_during_render(self) -> None:
        _, _, frozen = _load_frozen_case(DEFAULT_STRICT_MANIFEST, "sobol_008")
        expected = frozen["proposed_plus"]
        actual = deepcopy(expected)
        component_path = (
            ROOT
            / "outputs/current/active_comparison/strict_fingerprint_direct_matrices"
            / "unit_test"
            / ".component.mp4"
        )
        actual["video_path"] = str(component_path.relative_to(ROOT))
        actual["frame_count"] = 123
        verification = _full_result_verification(
            "proposed_plus",
            actual,
            expected,
            component_path=component_path,
        )
        self.assertTrue(
            verification["all_fields_match_or_are_declared_renderer_outputs"]
        )
        self.assertEqual(verification["substantive_mismatch_count"], 0)
        self.assertGreater(verification["checked_substantive_leaf_fields"], 100)
        self.assertEqual(
            set(verification["excluded_from_value_equality"]),
            {"video_path", "frame_count"},
        )
        self.assertTrue(
            all(
                item["valid"]
                for item in verification["renderer_only_fields"].values()
            )
        )

    def test_policy_or_result_drift_is_rejected(self) -> None:
        _, _, frozen = _load_frozen_case(DEFAULT_STRICT_MANIFEST, "sobol_008")
        expected = frozen["direct_first"]
        actual = deepcopy(expected)
        component_path = (
            ROOT
            / "outputs/current/active_comparison/strict_fingerprint_direct_matrices"
            / "unit_test"
            / ".component.mp4"
        )
        actual["video_path"] = str(component_path.relative_to(ROOT))
        actual["frame_count"] = 123
        actual["total_active_time_s"] += 0.01
        verification = _full_result_verification(
            "direct_first", actual, expected, component_path=component_path
        )
        self.assertFalse(
            verification["all_fields_match_or_are_declared_renderer_outputs"]
        )
        self.assertEqual(verification["substantive_mismatch_count"], 1)
        self.assertEqual(
            verification["substantive_mismatches"][0]["path"],
            "RunResult.total_active_time_s",
        )

    def test_shared_fingerprint_contract_holds_in_frozen_trial(self) -> None:
        _, _, frozen = _load_frozen_case(DEFAULT_STRICT_MANIFEST, "sobol_008")
        hashes = {
            frozen[method]["fingerprint_sha256"]
            for method in ("proposed_plus", "proposed", "direct_first")
        }
        self.assertEqual(len(hashes), 1)
        self.assertNotIn(None, hashes)
        self.assertIsNone(frozen["outward"]["fingerprint_sha256"])

    def test_cli_requires_pose_and_protects_retained_legacy_output(self) -> None:
        args = parse_args(
            [
                "--pose-id",
                "sobol_008",
                "--output",
                str(LEGACY_STRICT_FOUR_METHODS_OUTPUT),
            ]
        )
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            _resolve_output_and_stem(args)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            args = parse_args(
                [
                    "--pose-id",
                    "sobol_008",
                    "--output",
                    directory,
                    "--artifact-stem",
                    "bad/stem",
                ]
            )
            with self.assertRaisesRegex(ValueError, "artifact stem"):
                _resolve_output_and_stem(args)


if __name__ == "__main__":
    unittest.main()
