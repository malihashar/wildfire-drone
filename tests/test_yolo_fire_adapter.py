import unittest
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision import (
    FireGridConfig,
    FireMaskDetection,
    build_convlstm_sequence,
    cell_to_image_rect,
    detections_to_fire_grid,
    overlay_grid_on_image,
)


class TestYoloFireAdapter(unittest.TestCase):
    def test_fire_mask_projects_to_binary_grid(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[50:100, 50:100] = 1.0
        det = FireMaskDetection(mask=mask, class_name="fire", confidence=0.9)

        result = detections_to_fire_grid(
            [det],
            image_shape=(200, 200),
            config=FireGridConfig(grid_shape=(100, 100), grid_threshold=0.2),
        )

        self.assertEqual(result.fire_state_grid.shape, (100, 100))
        self.assertEqual(result.evidence_grid.shape, (100, 100))
        self.assertEqual(result.fire_state_grid.dtype, np.uint8)
        self.assertGreater(result.fire_state_grid.sum(), 0)
        self.assertEqual(set(np.unique(result.fire_state_grid)).issubset({0, 1}), True)

    def test_low_confidence_fire_is_ignored(self):
        mask = np.ones((64, 64), dtype=np.float32)
        det = FireMaskDetection(mask=mask, class_name="fire", confidence=0.05)

        result = detections_to_fire_grid([det], image_shape=(64, 64))

        self.assertEqual(int(result.fire_state_grid.sum()), 0)
        self.assertEqual(len(result.used_detections), 0)

    def test_cell_to_image_rect_maps_grid_bounds(self):
        image_shape = (200, 400)

        self.assertEqual(cell_to_image_rect(0, 0, image_shape), (0, 0, 4, 2))
        self.assertEqual(cell_to_image_rect(99, 99, image_shape), (396, 198, 400, 200))

    def test_overlay_preserves_image_shape(self):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        grid = np.zeros((100, 100), dtype=np.float32)
        grid[25:75, 25:75] = 1.0

        overlay = overlay_grid_on_image(image, grid)

        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(overlay.sum(), 0)

    def test_build_convlstm_sequence_uses_existing_channel_contract(self):
        fire = np.zeros((100, 100), dtype=np.uint8)
        fire[40:50, 40:50] = 1
        terrain_weather = torch.zeros(9, 100, 100)

        seq = build_convlstm_sequence(fire, terrain_weather, timesteps=20)

        self.assertEqual(tuple(seq.shape), (20, 10, 100, 100))
        self.assertTrue(torch.equal(seq[:, 0], torch.as_tensor(fire, dtype=torch.float32).repeat(20, 1, 1)))
        self.assertTrue(torch.equal(seq[:, 1:], torch.zeros(20, 9, 100, 100)))


if __name__ == "__main__":
    unittest.main()
