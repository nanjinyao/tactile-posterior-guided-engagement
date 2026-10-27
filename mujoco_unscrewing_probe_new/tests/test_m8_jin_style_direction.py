from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.m8_jin_style_direction import (
    DIRECTION_SECTORS,
    JinStyleDirectionRuntime,
    JinStylePolarCNN,
    direction_class,
    polar_binary_image,
)


class TestM8JinStyleDirection(unittest.TestCase):
    def test_direction_classes_wrap_at_sector_boundaries(self) -> None:
        self.assertEqual(direction_class(0.0), 0)
        self.assertEqual(direction_class(14.999), 0)
        self.assertEqual(direction_class(15.001), 1)
        self.assertEqual(direction_class(359.0), 0)
        self.assertEqual(direction_class(-30.0), DIRECTION_SECTORS - 1)

    def test_polar_raster_is_binary_deterministic_and_channel_preserving(self) -> None:
        phase = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        sequence = np.stack(
            (
                np.sin(phase),
                np.cos(2.0 * phase + 0.1),
                0.4 * np.sin(3.0 * phase - 0.3),
            )
        )
        first = polar_binary_image(sequence)
        second = polar_binary_image(sequence)
        self.assertEqual(first.shape, (3, 20, 20))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(set(np.unique(first)).issubset({0.0, 1.0}))
        self.assertTrue(np.all(first.sum(axis=(1, 2)) > 0))
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_network_has_paper_level_layer_contract(self) -> None:
        model = JinStylePolarCNN()
        convolution_count = sum(
            isinstance(module, torch.nn.Conv2d) for module in model.modules()
        )
        pooling_count = sum(
            isinstance(module, torch.nn.MaxPool2d) for module in model.modules()
        )
        linear_count = sum(
            isinstance(module, torch.nn.Linear) for module in model.modules()
        )
        self.assertEqual((convolution_count, pooling_count, linear_count), (2, 2, 1))
        output = model(torch.zeros(2, 3, 20, 20))
        self.assertEqual(tuple(output.shape), (2, DIRECTION_SECTORS))

    def test_runtime_uses_all_frozen_seed_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = []
            for index in range(3):
                path = root / f"seed_{index}.pt"
                torch.save(JinStylePolarCNN().state_dict(), path)
                checkpoints.append(path.name)
            (root / "runtime_contract.json").write_text(
                '{"checkpoints": ["seed_0.pt", "seed_1.pt", "seed_2.pt"]}\n',
                encoding="utf-8",
            )
            # Runtime-relative paths normally resolve from the project root.
            # Absolute paths keep this test independent of the workspace.
            (root / "runtime_contract.json").write_text(
                '{"checkpoints": ['
                + ", ".join(f'"{root / name}"' for name in checkpoints)
                + "]}\n",
                encoding="utf-8",
            )
            runtime = JinStyleDirectionRuntime(root)
            self.assertEqual(len(runtime.models), 3)


if __name__ == "__main__":
    unittest.main()
