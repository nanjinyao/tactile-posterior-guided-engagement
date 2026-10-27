from pathlib import Path
import tempfile
import unittest

from src.checkpoint_paths import resolve_checkpoint_path


class TestCheckpointRelocation(unittest.TestCase):
    def test_relocated_posix_and_windows_paths_keep_seed_and_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            checkpoint = root / "seeds" / "seed_123" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test fixture, not a model")
            for prefix in ("/old/workspace/", "C:\\old\\workspace\\"):
                value = prefix + "outputs/current/models/bundle/seeds/seed_123/model.pt"
                self.assertEqual(resolve_checkpoint_path(value, root), checkpoint)
            self.assertEqual(resolve_checkpoint_path(str(checkpoint), root), checkpoint)

    def test_missing_seed_or_different_bundle_is_not_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            for value in (
                "/old/outputs/current/models/bundle/seeds/missing/model.pt",
                "/old/outputs/current/models/different/seeds/seed_123/model.pt",
                "/old/outputs/current/models/bundle/seeds/../model.pt",
            ):
                with self.assertRaises(FileNotFoundError):
                    resolve_checkpoint_path(value, root)
