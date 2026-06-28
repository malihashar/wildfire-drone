import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision.paths import (
    DEFAULT_CONVLSTM_CHECKPOINT,
    resolve_convlstm_checkpoint,
)


class TestVisionPaths(unittest.TestCase):
    def test_default_convlstm_checkpoint_exists(self):
        self.assertTrue(DEFAULT_CONVLSTM_CHECKPOINT.exists())
        resolved = resolve_convlstm_checkpoint()
        self.assertEqual(resolved, DEFAULT_CONVLSTM_CHECKPOINT.resolve())

    def test_resolve_convlstm_checkpoint_from_directory(self):
        ckpt_dir = DEFAULT_CONVLSTM_CHECKPOINT.parent
        resolved = resolve_convlstm_checkpoint(ckpt_dir)
        self.assertEqual(resolved.name, "best_model.pt")


if __name__ == "__main__":
    unittest.main()
