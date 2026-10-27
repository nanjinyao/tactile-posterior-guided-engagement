from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SIM_ROOT / "src" / "build_real_observation_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "build_real_observation_calibration", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


class TestRealObservationCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, cls.source_hashes = (
            CALIBRATION.load_allowlisted_campaigns()
        )
        cls.config = CALIBRATION.fit_calibration(
            cls.samples, cls.source_hashes
        )
        cls.arrays, cls.records = CALIBRATION.build_replay(
            cls.samples, cls.config
        )

    def test_fresh_run_allowlist_and_known_program_mismatch(self) -> None:
        expected = {
            "M12": "M12_conical_fingerprints_26pt_20260728_175502_318",
            "M10": "M10_conical_fingerprints_26pt_20260728_152808_877",
            "M8": "M8_conical_fingerprints_26pt_20260728_161249_458",
        }
        self.assertEqual(
            {
                size: run_dir.name
                for size, run_dir in CALIBRATION.RUN_ALLOWLIST.items()
            },
            expected,
        )
        self.assertEqual(
            CALIBRATION.KNOWN_REJECTED_RUN.name,
            "M8_conical_fingerprints_26pt_20260728_172108_490",
        )
        self.assertEqual(len(self.samples), 78)
        self.assertEqual(
            {
                partition: sum(
                    sample.partition == partition for sample in self.samples
                )
                for partition in CALIBRATION.PARTITION_COUNTS
            },
            {
                "clean_candidate": 48,
                "holdout": 12,
                "quarantine": 12,
                "sentinel": 6,
            },
        )
        self.assertTrue(
            all(
                details["attempt_hash_checks"] == 104
                for details in self.source_hashes.values()
            )
        )

    def test_stable_schema_and_fit_partition_boundary(self) -> None:
        required = {
            "frame_phase_transform",
            "channel_affine",
            "noise_cov",
            "drift",
            "timing",
            "force",
            "domain_randomization",
            "generation",
            "hashes",
        }
        self.assertTrue(required.issubset(self.config))
        policy = self.config["source_policy"]
        self.assertEqual(policy["fit_samples"], 48)
        self.assertEqual(policy["fit_partitions"], ["clean_candidate"])
        self.assertEqual(
            set(policy["excluded_from_fit"]),
            {"holdout", "quarantine", "sentinel"},
        )
        self.assertEqual(policy["sentinel_role"], "validation_only_never_fit")
        sentinel = self.config["validation"]["sentinel_repeatability"]
        self.assertEqual(sentinel["parameter_fit_effect"], "NONE")
        for size in CALIBRATION.SIZE_ORDER:
            drift = self.config["drift"]["by_size"][size]
            self.assertEqual(drift["fit_partition"], "clean_candidate")
            self.assertEqual(drift["clean_baseline_count"], 16)
            self.assertEqual(
                sentinel["by_size"][size]["pair_count"], 2
            )

    def test_frame_affine_noise_timing_and_force_are_finite(self) -> None:
        # allow_nan=False is also the serialization contract used by the builder.
        json.dumps(self.config, allow_nan=False)
        for size in CALIBRATION.SIZE_ORDER:
            frame = self.config["frame_phase_transform"]["by_size"][size]
            rotation = np.asarray(
                frame["fixture_to_world_xy_rotation"], dtype=float
            )
            np.testing.assert_allclose(
                rotation.T @ rotation, np.eye(2), atol=1.0e-9
            )
            self.assertGreaterEqual(
                frame["inward_response_peak_relative_phase_deg"], 0.0
            )
            self.assertLess(
                frame["inward_response_peak_relative_phase_deg"], 360.0
            )

            affine = self.config["channel_affine"]["by_size"][size]
            scale = np.asarray(affine["physical_robust_scale"], dtype=float)
            self.assertEqual(scale.shape, (7,))
            self.assertTrue(np.all(scale > 0.0))

            noise = self.config["noise_cov"]["by_size"][size]
            covariance = np.asarray(
                noise["white_difference_cov"], dtype=float
            )
            self.assertEqual(covariance.shape, (6, 6))
            np.testing.assert_allclose(
                covariance, covariance.T, atol=1.0e-12
            )
            self.assertGreater(
                self.config["timing"]["by_size"][size][
                    "effective_source_rate_hz_from_median_dt"
                ],
                0.0,
            )
            self.assertGreater(
                self.config["force"]["by_size"][size][
                    "contact_settle_normal_mean_n"
                ]["count"],
                0,
            )

    def test_replay_has_three_unique_seeds_per_all_geometry_tasks(self) -> None:
        self.assertEqual(len(self.records), 234)
        self.assertEqual(
            self.arrays["sequence_physical"].shape, (234, 7, 360)
        )
        self.assertEqual(
            self.arrays["sequence_normalized"].shape, (234, 7, 360)
        )
        self.assertEqual(len(np.unique(self.arrays["seed"])), 234)
        source_ids, counts = np.unique(
            self.arrays["sample_id"], return_counts=True
        )
        self.assertEqual(len(source_ids), 78)
        self.assertTrue(np.all(counts == 3))
        partition_counts = {
            partition: sum(
                record["partition"] == partition for record in self.records
            )
            for partition in CALIBRATION.PARTITION_COUNTS
        }
        self.assertEqual(
            partition_counts,
            {
                "clean_candidate": 144,
                "holdout": 36,
                "quarantine": 36,
                "sentinel": 18,
            },
        )
        self.assertTrue(
            all(
                (
                    record["training_use"]
                    == "eligible_real_calibrated_replay"
                )
                == (record["partition"] == "clean_candidate")
                for record in self.records
            )
        )

    def test_replay_is_deterministic_and_domain_randomization_is_unfitted(self) -> None:
        arrays, records = CALIBRATION.build_replay(
            self.samples, self.config
        )
        np.testing.assert_array_equal(
            arrays["sequence_physical"],
            self.arrays["sequence_physical"],
        )
        np.testing.assert_array_equal(
            arrays["sequence_normalized"],
            self.arrays["sequence_normalized"],
        )
        self.assertEqual(
            [record["seed"] for record in records],
            [record["seed"] for record in self.records],
        )
        domain = self.config["domain_randomization"]
        self.assertFalse(domain["identified_from_real_runs"])
        self.assertEqual(
            set(domain["variables"]),
            {
                "friction_slide",
                "contact_time_constant_s",
                "printed_polymer_stiffness_scale",
                "printed_polymer_damping_scale",
            },
        )

    def test_bundle_refuses_to_overwrite_before_loading_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "already_exists.json"
            config.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                CALIBRATION.build_bundle(config, root / "new_output")

            output = root / "already_exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                CALIBRATION.build_bundle(root / "new_config.json", output)


if __name__ == "__main__":
    unittest.main()
