from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "model_runtime" / "infer_conical_direction.py"
SPEC = importlib.util.spec_from_file_location("infer_conical_direction", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def _write_json(path: Path, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TestModelRuntimeAuthority(unittest.TestCase):
    def _manifest(self, model_id: str, authority: dict) -> dict:
        return {
            "schema_version": 1,
            "model_id": model_id,
            "artifacts": [],
            "hardware_authority": {
                "allowed_values": ["SHADOW_ONLY", "CAPPED_SHORTCUT"],
                **authority,
            },
        }

    def test_checked_in_shadow_manifest_is_immutable_and_retired(
        self,
    ) -> None:
        model_root = (
            ROOT.parent
            / "mujoco_unscrewing_probe_new"
            / "outputs"
            / "current"
            / "models"
            / "hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving"
        )
        manifest_path = ROOT / "model_runtime" / "deployment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["hardware_authority"]["default"], "SHADOW_ONLY"
        )
        self.assertEqual(
            manifest["hardware_authority"]["allowed_values"], ["SHADOW_ONLY"]
        )
        self.assertEqual(
            manifest["audit_decision"]["global_status"],
            "REJECTED_GLOBAL_AUDIT",
        )
        self.assertEqual(
            manifest["audit_decision"]["diagnostic_scope"],
            "M12_DEVELOPMENT_SHADOW_ONLY",
        )
        runtime_record = next(
            item
            for item in manifest["runtime_artifacts"]
            if item["path"] == "infer_conical_direction.py"
        )
        self.assertNotEqual(
            runtime_record["sha256"],
            hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest(),
            "The active v2 runtime must not be relabeled as the immutable "
            "historical shadow runtime.",
        )
        with self.assertRaisesRegex(
            ValueError, "frozen inference runtime artifact hash mismatch"
        ):
            RUNTIME.load_and_verify_manifest(manifest_path, model_root)

    def test_checked_in_venv_imports_the_hash_pinned_runtime_stack(
        self,
    ) -> None:
        venv_root = ROOT / "model_runtime" / ".venv"
        venv_python = venv_root / "bin" / "python"
        self.assertTrue(venv_python.is_file())
        completed = subprocess.run(
            [
                str(venv_python),
                "-c",
                (
                    "import json,sys,numpy,torch;"
                    "print(json.dumps({"
                    "'executable':sys.executable,"
                    "'prefix':sys.prefix,"
                    "'numpy':numpy.__version__,"
                    "'torch':torch.__version__"
                    "}))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        runtime = json.loads(completed.stdout)
        self.assertEqual(Path(runtime["executable"]), venv_python)
        self.assertEqual(Path(runtime["prefix"]), venv_root)
        self.assertEqual(runtime["numpy"], "1.24.4")
        self.assertEqual(runtime["torch"], "2.4.1+cpu")

        lock = (
            ROOT / "model_runtime" / "requirements-lock.txt"
        ).read_text(encoding="utf-8")
        self.assertIn(f"numpy=={runtime['numpy']}", lock)
        self.assertIn(f"torch=={runtime['torch']}", lock)

        main = (
            ROOT / "execute_m12_contact_engagement_policy.m"
        ).read_text(encoding="utf-8")
        learning = main[
            main.index("cfg.learning = struct(") :
            main.index("cfg.surface = struct(")
        ]
        self.assertIn(
            "'python_executable', fullfile(realDir, 'model_runtime', ...",
            learning,
        )
        self.assertIn("'.venv', 'bin', 'python')", learning)
        self.assertNotIn(
            "fullfile(filesep, 'usr', 'bin', 'python3')",
            learning,
        )

    def test_rejected_global_audit_cannot_expose_control_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            model_root.mkdir()
            manifest_path = root / "manifest.json"
            manifest = self._manifest(
                model_root.name,
                {
                    "default": "SHADOW_ONLY",
                    "promotion_contract_path": None,
                    "promotion_contract_sha256": None,
                },
            )
            manifest["audit_decision"] = {
                "global_status": "REJECTED_GLOBAL_AUDIT",
                "diagnostic_scope": "M12_DEVELOPMENT_SHADOW_ONLY",
            }
            manifest["hardware_authority"]["allowed_values"] = [
                "SHADOW_ONLY",
                "CAPPED_SHORTCUT",
            ]
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "must remain"):
                RUNTIME.load_and_verify_manifest(manifest_path, model_root)

    def test_immutable_m12_v7_shadow_replay_identity(self) -> None:
        replay_path = (
            ROOT
            / "model_runtime"
            / "replay_outputs"
            / "M12_event_trigger_insert_live_20260724_023025_306_v7_shadow.json"
        )
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "0f7a65b9fcf581e7e91aa4f8e0f3929af8b7fa425c3746adc9214321d2c548fa",
        )
        self.assertEqual(
            replay["model_id"],
            "hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving",
        )
        self.assertEqual(replay["size"], "m12")
        self.assertEqual(replay["deployment_authority"], "SHADOW_ONLY")
        self.assertFalse(replay["hardware_control_allowed"])
        self.assertEqual(
            replay["manifest_sha256"],
            hashlib.sha256(
                (ROOT / "model_runtime" / "deployment_manifest.json").read_bytes()
            ).hexdigest(),
        )
        input_path = Path(replay["input_path"])
        if not input_path.is_file():
            input_path = (
                ROOT
                / "data"
                / input_path.parent.name
                / input_path.name
            )
        self.assertTrue(input_path.is_file())
        self.assertEqual(
            replay["input_sha256"],
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
        )

    def test_real_config_retains_historical_shadow_evidence_only(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "unscrewing_real_v1.json").read_text(
                encoding="utf-8"
            )
        )
        learning = config["learning_hinge"]
        self.assertEqual(
            learning["model_id"],
            "hardware_easy_v2_multisize_clocking_clean_direction_v7_source_preserving",
        )
        self.assertEqual(
            learning["global_simulation_audit"], "REJECTED_GLOBAL_AUDIT"
        )
        self.assertEqual(
            learning["diagnostic_scope"], "M12_DEVELOPMENT_SHADOW_ONLY"
        )
        self.assertEqual(learning["allowed_authorities"], ["SHADOW_ONLY"])
        self.assertEqual(
            learning["deployment_manifest_sha256"],
            hashlib.sha256(
                (ROOT / "model_runtime" / "deployment_manifest.json").read_bytes()
            ).hexdigest(),
        )
        legacy_manifest = json.loads(
            (ROOT / "model_runtime" / "deployment_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        legacy_runtime = next(
            item
            for item in legacy_manifest["runtime_artifacts"]
            if item["path"] == "infer_conical_direction.py"
        )
        self.assertEqual(
            learning["runtime_sha256"], legacy_runtime["sha256"]
        )
        self.assertNotEqual(
            learning["runtime_sha256"],
            hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            learning["requirements_lock_sha256"],
            hashlib.sha256(
                (ROOT / "model_runtime" / "requirements-lock.txt").read_bytes()
            ).hexdigest(),
        )
        active = (
            ROOT / "execute_m12_contact_engagement_policy.m"
        ).read_text(encoding="utf-8")
        self.assertIn("'model_id', 'real_multisize_direction_v1'", active)
        self.assertIn(
            "'required_authority', 'CAPPED_SHORTCUT'", active
        )

    def test_every_remaining_full_loop_shadow_shortcut_receives_its_cone(
        self,
    ) -> None:
        main = (
            ROOT / "execute_m12_contact_engagement_policy.m"
        ).read_text(encoding="utf-8")
        calls = re.findall(
            r"(?m)^\s*\w+\s*=\s*planLearnedShortcut\(\s*\.\.\.\s*"
            r"(.*?)\);",
            main,
            flags=re.DOTALL,
        )
        # The commissioning sampler deliberately has no runtime model or
        # shadow shortcut.  At least one full-loop route still consumes the
        # cone associated with its frozen-model output.
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(
            main.count("planLearnedShortcut("),
            len(calls) + 1,
            "Every occurrence except the single function definition must "
            "be covered as a call site.",
        )
        for arguments in calls:
            compact = re.sub(r"\s+|\.\.\.", "", arguments)
            fields = compact.split(",")
            self.assertEqual(
                len(fields),
                5,
                f"Unexpected planLearnedShortcut signature: {compact}",
            )
            self.assertEqual(fields[2], "learned.output")
            self.assertIn(
                fields[3],
                {"cone", "conical"},
                "The fourth argument must be the fingerprint associated "
                "with the learned output.",
            )
            self.assertEqual(fields[4], "cfg")

    def test_conical_modes_preflight_three_seed_runtime_before_ros_and_kst(
        self,
    ) -> None:
        main = (
            ROOT / "execute_m12_contact_engagement_policy.m"
        ).read_text(encoding="utf-8")
        policy = main[
            main.index("function policy = executionPolicy") :
            main.index("function [result, manifest] = discloseExecutionPolicy")
        ]
        self.assertIn("'uses_conical_fingerprint', true", policy)
        pure_case = policy[
            policy.index("case 'FULL_LOOP_BASELINE_PURE_OUTWARD_SPIRAL'") :
            policy.index(
                "case 'FULL_LOOP_BASELINE_IMPROVED_DIRECT_X3_LEARNED'"
            )
        ]
        self.assertIn(
            "policy.uses_conical_fingerprint = false;", pure_case
        )
        self.assertEqual(
            policy.count("policy.uses_conical_fingerprint = false;"),
            1,
            "Pure Outward is the only mode allowed to skip inference preflight.",
        )

        entry = main[
            : main.index(
                "function [mode, resumeParentRunDir] = parseExecutionMode"
            )
        ]
        guarded_call = entry.index("if policy.uses_conical_fingerprint")
        preflight_call = entry.index(
            "validateLearnedRuntimePreflight(cfg, paths);", guarded_call
        )
        preflight_result = entry.index(
            "result.learned_runtime_preflight_passed = true;",
            preflight_call,
        )
        manifest_write = entry.index(
            "writeJson(paths.manifest, manifest);", preflight_result
        )
        kst_client = entry.index("kstClientDir = fullfile(", manifest_write)
        ros_start = entry.index("restartMatlabRosNode(", kst_client)
        first_subscriber = entry.index("rossubscriber(", ros_start)
        kst_constructor = entry.index("iiwa = KST(", first_subscriber)
        kst_connect = entry.index(
            "iiwa.net_establishConnection()", kst_constructor
        )
        self.assertLess(guarded_call, preflight_call)
        self.assertLess(preflight_call, preflight_result)
        self.assertLess(preflight_result, manifest_write)
        self.assertLess(manifest_write, kst_client)
        self.assertLess(kst_client, ros_start)
        self.assertLess(ros_start, first_subscriber)
        self.assertLess(first_subscriber, kst_constructor)
        self.assertLess(kst_constructor, kst_connect)

        learning = main[
            main.index("cfg.learning = struct(") :
            main.index("cfg.surface = struct(")
        ]
        for token in (
            "'model_id', 'real_multisize_direction_v1'",
            "'python_executable', fullfile(realDir, 'model_runtime', ...",
            "'.venv', 'bin', 'python')",
            "'inference_script', fullfile(realDir, 'model_runtime', ...",
            "'infer_conical_direction.py')",
            "'required_authority', 'CAPPED_SHORTCUT'",
            "'required_promotion_qualification', 'DEVELOPMENT_CV_CONTROL'",
            "'M12_event_trigger_insert_live_20260724_023025_306'",
            "'conical_directservo_trace.csv'",
            "'preflight_trace_sha256', ...",
            "'e2de7a701031175ffdd0a656f96f2489d28b06cb8f033487227cb5717d4b5aaa'",
        ):
            self.assertIn(token, learning)

        helper = main[
            main.index("function preflight = validateLearnedRuntimePreflight") :
            main.index("function validateSurfaceTransportContract")
        ]
        for token in (
            "validateModelDeploymentTrustAnchor(cfg);",
            "sha256File(cfg.learning.preflight_trace_csv)",
            "cfg.learning.preflight_trace_sha256",
            "'M12Demo:LearnedPreflightTraceTrustAnchor'",
            "/usr/bin/timeout %.0f %s %s --trace-csv %s ",
            "--output-json %s --model-root %s --manifest %s ",
            "--size %s --target-down-force-n %.9f %s 2>&1",
            "shellSafePath(cfg.learning.python_executable)",
            "shellSafePath(cfg.learning.inference_script)",
            "shellSafePath(cfg.learning.preflight_trace_csv)",
            "shellSafePath(paths.learnedPreflightOutput)",
            "shellSafePath(cfg.learning.model_root)",
            "shellSafePath(cfg.learning.deployment_manifest)",
            "learnedPilotAuthorizationCliArgument(cfg)",
            "[status, processOutput] = system(command);",
            "assert(status == 0",
            "jsondecode(fileread(paths.learnedPreflightOutput))",
            "assertActiveLearnedOutput(value, cfg, ...",
            "numel(directionProbability) == 48",
            "numel(stateProbability) == 3",
            "'PASS_BEFORE_KST_NO_MOTION'",
            "'LEARNED_RUNTIME_PREFLIGHT_PASS_NO_MOTION'",
            "venv/model/manifest/normalization inference complete",
        ):
            self.assertIn(token, helper)
        for motion_or_ros in (
            "rossubscriber",
            "rosinit",
            "net_establishConnection",
            "movePTP",
            "sendEEfPosition",
            "realTime_start",
        ):
            self.assertNotIn(motion_or_ros, helper)

        paths = main[
            main.index("function paths = initializeRunFiles") :
            main.index("function writePlan")
        ]
        self.assertIn(
            "'learnedPreflightOutput', fullfile(runDir, ...", paths
        )
        self.assertIn(
            "'learned_runtime_preflight_output.json')", paths
        )

        trace_path = (
            ROOT
            / "data"
            / "M12_event_trigger_insert_live_20260724_023025_306"
            / "conical_directservo_trace.csv"
        )
        model_root = (
            ROOT
            / "model_runtime"
            / "bundles"
            / "real_multisize_direction_v1"
        )
        manifest_path = model_root / "deployment_manifest.json"
        self.assertEqual(
            hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "e2de7a701031175ffdd0a656f96f2489d28b06cb8f033487227cb5717d4b5aaa",
        )

        manifest = RUNTIME.load_and_verify_manifest(
            manifest_path, model_root
        )
        self.assertEqual(manifest["_hardware_control_sizes"], ["m12"])
        contract = json.loads(
            (model_root / "ensemble_interface_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(contract["seeds"]), 3)
        self.assertEqual(
            [record["seed"] for record in contract["seeds"]],
            [20260728, 20260729, 20260730],
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "preflight.json"
            completed = subprocess.run(
                [
                    str(ROOT / "model_runtime" / ".venv" / "bin" / "python"),
                    str(ROOT / "model_runtime" / "infer_conical_direction.py"),
                    "--trace-csv",
                    str(trace_path),
                    "--output-json",
                    str(output_path),
                    "--model-root",
                    str(model_root),
                    "--manifest",
                    str(manifest_path),
                    "--size",
                    "m12",
                    "--target-down-force-n",
                    "3.5",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "OK")
            self.assertEqual(
                output["input_sha256"],
                hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                output["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(output["seed_predictions"]), 3)
            self.assertEqual(
                [record["seed"] for record in output["seed_predictions"]],
                [20260728, 20260729, 20260730],
            )
            self.assertEqual(
                output["deployment_authority"], "CAPPED_SHORTCUT"
            )
            self.assertEqual(output["hardware_control_sizes"], ["m12"])
            self.assertTrue(output["hardware_control_allowed"])
            self.assertEqual(
                output["command_direction_source"],
                "TOP1_POSTERIOR_MODE",
            )

    def test_capped_shortcut_rejects_missing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            model_root.mkdir()
            manifest_path = root / "manifest.json"
            _write_json(
                manifest_path,
                self._manifest(
                    model_root.name,
                    {
                        "default": "CAPPED_SHORTCUT",
                        "promotion_contract_path": "absent.json",
                        "promotion_contract_sha256": "0" * 64,
                    },
                ),
            )
            with self.assertRaises(FileNotFoundError):
                RUNTIME.load_and_verify_manifest(manifest_path, model_root)

    def test_capped_shortcut_requires_hash_bound_matching_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            model_root.mkdir()
            contract_path = root / "promotion.json"
            contract_hash = _write_json(
                contract_path,
                {
                    "schema_version": 1,
                    "status": "QUALIFIED",
                    "model_id": model_root.name,
                    "deployment_manifest_model_id": model_root.name,
                    "authorization": {
                        "capped_shortcut_hardware_control": True
                    },
                },
            )
            manifest_path = root / "manifest.json"
            _write_json(
                manifest_path,
                self._manifest(
                    model_root.name,
                    {
                        "default": "CAPPED_SHORTCUT",
                        "promotion_contract_path": contract_path.name,
                        "promotion_contract_sha256": contract_hash,
                    },
                ),
            )
            manifest = RUNTIME.load_and_verify_manifest(
                manifest_path, model_root
            )
            self.assertTrue(manifest["_promotion_contract_verified"])


if __name__ == "__main__":
    unittest.main()
