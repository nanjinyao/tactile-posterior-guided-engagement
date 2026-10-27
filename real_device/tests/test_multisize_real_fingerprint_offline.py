from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REAL_ROOT / "tools" / "analyze_multisize_real_fingerprints.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_multisize_real_fingerprints", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class TestMultisizeRealFingerprintOffline(unittest.TestCase):
    def test_run_selection_is_an_explicit_fresh_allowlist(self) -> None:
        expected = {
            "M12": "M12_conical_fingerprints_26pt_20260728_175502_318",
            "M10": "M10_conical_fingerprints_26pt_20260728_152808_877",
            "M8": "M8_conical_fingerprints_26pt_20260728_161249_458",
        }
        self.assertEqual(
            {
                size: path.name
                for size, path in ANALYSIS.RUN_ALLOWLIST.items()
            },
            expected,
        )
        self.assertEqual(
            ANALYSIS.KNOWN_REJECTED_RUN.name,
            "M8_conical_fingerprints_26pt_20260728_172108_490",
        )
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('glob("M', source)
        self.assertIn('"DO_NOT_TRAIN.flag"', source)
        self.assertIn('"RUN_QUARANTINE.json"', source)

    def test_periodic_resample_preserves_phase_signal(self) -> None:
        phase = np.linspace(0.0, 2.0 * math.pi, 241)[1:]
        values = np.column_stack((np.sin(phase), np.cos(phase)))
        output = ANALYSIS.periodic_resample(phase, values, bins=360)
        grid = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        self.assertEqual(output.shape, (2, 360))
        self.assertLess(float(np.max(np.abs(output[0] - np.sin(grid)))), 0.001)
        self.assertLess(float(np.max(np.abs(output[1] - np.cos(grid)))), 0.001)

    def test_cad_ray_entry_is_finite_only_toward_hex(self) -> None:
        inward = ANALYSIS.ray_entry_distance_mm(
            np.asarray([10.0, 0.0]), np.asarray([-1.0, 0.0]), 5.0
        )
        outward = ANALYSIS.ray_entry_distance_mm(
            np.asarray([10.0, 0.0]), np.asarray([1.0, 0.0]), 5.0
        )
        inside = ANALYSIS.ray_entry_distance_mm(
            np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), 5.0
        )
        self.assertIsNotNone(inward)
        # regular_hexagon() has vertices at 30 + k*60 deg.  Therefore the
        # x-facing flats of an AF=5 hexagon lie at x=+/-2.5 mm; AF/sqrt(3)
        # is the circumradius, not the x intercept for this clocking.
        self.assertAlmostEqual(
            float(inward), 10.0 - 5.0 / 2.0, places=9
        )
        self.assertIsNone(outward)
        self.assertEqual(inside, 0.0)

    def test_peak_selection_preserves_separated_modes(self) -> None:
        score = np.zeros(360)
        score[10] = 1.0
        score[190] = 0.9
        score[18] = 0.8
        peaks = ANALYSIS.separated_peaks(
            score, count=2, separation_deg=20.0
        )
        self.assertEqual(peaks, [10, 190])

    def test_all_three_allowlisted_campaigns_pass_qc(self) -> None:
        for size, run_dir in ANALYSIS.RUN_ALLOWLIST.items():
            samples, quality = ANALYSIS.validate_campaign(size, run_dir)
            self.assertEqual(len(samples), 26)
            self.assertEqual(quality["hash_checks"], 104)
            self.assertEqual(quality["hash_mismatches"], 0)
            self.assertEqual(quality["partition_counts"], ANALYSIS.PARTITION_COUNTS)


if __name__ == "__main__":
    unittest.main()
