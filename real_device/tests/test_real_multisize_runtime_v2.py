from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = REAL_ROOT / "model_runtime" / "infer_conical_direction.py"
BUNDLE_ROOT = (
    REAL_ROOT
    / "model_runtime"
    / "bundles"
    / "real_multisize_direction_v1"
)
MANIFEST_PATH = BUNDLE_ROOT / "deployment_manifest.json"
TRACE_PATH = (
    REAL_ROOT
    / "data"
    / "M12_conical_fingerprints_26pt_20260728_175502_318"
    / "sampling_attempts"
    / "01_M12-CLEAN_CANDIDATE-SUPPORTED-A003p75"
    / "conical_trace_raw.csv"
)
M10_TRACE_PATH = (
    REAL_ROOT
    / "data"
    / "M10_conical_fingerprints_26pt_20260728_152808_877"
    / "sampling_attempts"
    / "26_M10-CLEAN_CANDIDATE-SUPPORTED-A003p75"
    / "conical_trace_raw.csv"
)
M10_PILOT_PATH = (
    REAL_ROOT / "configs" / "m10_fixed_demo_pilot_authorization_v1.json"
)
PYTHON = REAL_ROOT / "model_runtime" / ".venv" / "bin" / "python"

SPEC = importlib.util.spec_from_file_location("real_runtime_v2", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class TestRealMultisizeRuntimeV2(unittest.TestCase):
    def test_manifest_binds_three_seeds_and_m12_only_control(self) -> None:
        manifest = RUNTIME.load_and_verify_manifest(
            MANIFEST_PATH, BUNDLE_ROOT
        )
        self.assertEqual(manifest["deployment_authority"], "CAPPED_SHORTCUT")
        self.assertEqual(manifest["_hardware_control_sizes"], ["m12"])
        self.assertEqual(
            manifest["_supported_sizes"], ["m8", "m10", "m12"]
        )
        self.assertEqual(len(manifest["seed_models"]), 3)
        self.assertTrue(manifest["_promotion_contract_verified"])

    def test_trace_adapter_is_schema2_without_cad_angle_patch(self) -> None:
        value = RUNTIME.trace_csv_to_input(TRACE_PATH)
        self.assertEqual(value["schema_version"], 2)
        self.assertIn("no additional CAD clocking compensation", value["frame_contract"])
        phase, columns = RUNTIME.validate_input(value)
        self.assertEqual(columns.shape, (len(phase), 7))
        self.assertTrue(np.allclose(value["world_from_task_xy"], np.eye(3)[:, :2]))

    def test_importable_three_seed_inference_is_finite_and_normalized(self) -> None:
        value = RUNTIME.trace_csv_to_input(TRACE_PATH)
        phase, columns = RUNTIME.validate_input(value)
        sequence = RUNTIME.phase_resample(phase, columns, bins=360)
        output = RUNTIME.run_ensemble(
            BUNDLE_ROOT,
            sequence,
            requested_size="m12",
            target_down_force_n=3.5,
        )
        self.assertEqual(len(output["seed_predictions"]), 3)
        self.assertAlmostEqual(sum(output["direction_probabilities"]), 1.0)
        self.assertAlmostEqual(sum(output["state_probabilities"]), 1.0)
        self.assertEqual(len(output["direction_probabilities"]), 48)
        self.assertEqual(len(output["state_probabilities"]), 3)
        self.assertTrue(np.isfinite(output["sample_quality"]))
        self.assertEqual(output["command_direction_source"], "TOP1_POSTERIOR_MODE")
        self.assertFalse(output["posterior_quality_is_validity_veto"])

    def test_cli_preflight_is_under_three_seconds_and_weak_complete_is_actionable(
        self,
    ) -> None:
        value = RUNTIME.trace_csv_to_input(TRACE_PATH)
        # Deliberately weak but complete/new/finite: quality may shrink the
        # route, but it must not revoke M12 high-level authority.
        for name in (
            "model_fx_n",
            "model_fy_n",
            "model_fz_n",
            "model_tx_nm",
            "model_ty_nm",
            "model_tz_nm",
        ):
            value[name] = (0.02 * np.asarray(value[name], dtype=float)).tolist()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "weak_complete.json"
            output_path = root / "output.json"
            input_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            started = time.monotonic()
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(RUNTIME_PATH),
                    "--input-json",
                    str(input_path),
                    "--output-json",
                    str(output_path),
                    "--model-root",
                    str(BUNDLE_ROOT),
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--size",
                    "m12",
                    "--target-down-force-n",
                    "3.5",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.monotonic() - started
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLess(elapsed, 3.0)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(output["hardware_control_allowed"])
            self.assertEqual(output["hardware_control_sizes"], ["m12"])
            self.assertFalse(output["posterior_quality_is_validity_veto"])
            self.assertTrue(output["weak_complete_fingerprint_remains_actionable"])
            self.assertTrue(np.all(np.isfinite(output["direction_probabilities"])))
            self.assertAlmostEqual(sum(output["direction_probabilities"]), 1.0)

    def test_nonfinite_and_incomplete_inputs_are_rejected(self) -> None:
        value = RUNTIME.trace_csv_to_input(TRACE_PATH)
        value["model_fx_n"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            RUNTIME.validate_input(value)
        value = RUNTIME.trace_csv_to_input(TRACE_PATH)
        value["phase_coverage_fraction"] = 0.90
        with self.assertRaisesRegex(ValueError, "below"):
            RUNTIME.validate_input(value)

    def test_m10_and_m8_are_inference_only_without_fixed_pilot(self) -> None:
        for size in ("m10", "m8"):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory) / f"{size}.json"
                completed = subprocess.run(
                    [
                        str(PYTHON),
                        str(RUNTIME_PATH),
                        "--trace-csv",
                        str(TRACE_PATH),
                        "--output-json",
                        str(output_path),
                        "--model-root",
                        str(BUNDLE_ROOT),
                        "--manifest",
                        str(MANIFEST_PATH),
                        "--size",
                        size,
                        "--target-down-force-n",
                        "3.5",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                output = json.loads(
                    output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(output["status"], "OK")
                self.assertEqual(output["hardware_control_sizes"], ["m12"])
                self.assertFalse(output["hardware_control_allowed"])
                self.assertEqual(
                    output["hardware_control_denial_reason"],
                    "requested_size_outside_M12_control_scope",
                )

    def test_exact_hash_pinned_m10_fixed_demo_pilot_is_narrowly_actionable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "m10_pilot.json"
            started = time.monotonic()
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(RUNTIME_PATH),
                    "--trace-csv",
                    str(M10_TRACE_PATH),
                    "--output-json",
                    str(output_path),
                    "--model-root",
                    str(BUNDLE_ROOT),
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--size",
                    "m10",
                    "--target-down-force-n",
                    "3.5",
                    "--pilot-authorization",
                    str(M10_PILOT_PATH),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.monotonic() - started
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLess(elapsed, 3.0)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(output["hardware_control_allowed"])
            self.assertEqual(output["hardware_control_sizes"], ["m12", "m10"])
            self.assertTrue(output["fixed_demo_pilot_authorization_verified"])
            self.assertEqual(
                output["fixed_demo_pilot_authorization_id"],
                RUNTIME.M10_FIXED_DEMO_PILOT_ID,
            )
            self.assertEqual(
                output["fixed_demo_pilot_authorization_sha256"],
                RUNTIME.M10_FIXED_DEMO_PILOT_SHA256,
            )
            self.assertIn(
                "not general M10 full-loop promotion",
                output["fixed_demo_pilot_scope"],
            )
            self.assertAlmostEqual(output["top_direction_deg"], 183.75)

    def test_copied_or_modified_m10_pilot_cannot_widen_runtime_scope(
        self,
    ) -> None:
        manifest = RUNTIME.load_and_verify_manifest(
            MANIFEST_PATH, BUNDLE_ROOT
        )
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "pilot.json"
            copied.write_bytes(M10_PILOT_PATH.read_bytes())
            with self.assertRaisesRegex(ValueError, "trust anchor changed"):
                RUNTIME.verify_fixed_demo_pilot_authorization(
                    copied, manifest, "m10"
                )


if __name__ == "__main__":
    unittest.main()
