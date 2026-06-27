import unittest
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision import (
    FireBoxDetection,
    FireGridConfig,
    FireMaskDetection,
    build_convlstm_sequence,
    cell_to_image_rect,
    compare_grid_distributions,
    compute_fire_grid_stats,
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

    def test_wildfire_box_projects_to_binary_grid(self):
        det = FireBoxDetection(
            box_xyxy=(50.0, 50.0, 100.0, 100.0),
            class_name="wildfire",
            confidence=0.9,
        )

        result = detections_to_fire_grid(
            [det],
            image_shape=(200, 200),
            config=FireGridConfig(grid_shape=(100, 100), grid_threshold=0.2),
        )

        self.assertGreater(result.fire_state_grid.sum(), 0)
        self.assertEqual(len(result.used_detections), 1)
        self.assertEqual(set(np.unique(result.fire_state_grid)).issubset({0, 1}), True)

    def test_multiple_fire_boxes_are_preserved_in_one_grid(self):
        detections = [
            FireBoxDetection(
                box_xyxy=(10.0, 10.0, 30.0, 30.0),
                class_name="wildfire",
                confidence=0.9,
            ),
            FireBoxDetection(
                box_xyxy=(140.0, 20.0, 170.0, 50.0),
                class_name="fire",
                confidence=0.8,
            ),
            FireBoxDetection(
                box_xyxy=(70.0, 140.0, 110.0, 180.0),
                class_name="flame",
                confidence=0.7,
            ),
        ]

        result = detections_to_fire_grid(
            detections,
            image_shape=(200, 200),
            config=FireGridConfig(grid_shape=(100, 100), grid_threshold=0.2),
        )

        self.assertEqual(len(result.used_detections), 3)
        self.assertGreater(result.fire_state_grid[5:20, 5:20].sum(), 0)
        self.assertGreater(result.fire_state_grid[10:30, 70:90].sum(), 0)
        self.assertGreater(result.fire_state_grid[70:95, 35:60].sum(), 0)

    def test_irregular_segmentation_mask_preserves_geometry_not_box(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[40:120, 40:70] = 1.0
        mask[90:120, 40:130] = 1.0
        det = FireMaskDetection(mask=mask, class_name="wildfire", confidence=0.9)

        result = detections_to_fire_grid(
            [det],
            image_shape=(200, 200),
            config=FireGridConfig(grid_shape=(100, 100), grid_threshold=0.2),
        )

        self.assertGreater(result.fire_state_grid[20:60, 20:35].sum(), 0)
        self.assertGreater(result.fire_state_grid[45:60, 20:65].sum(), 0)
        self.assertEqual(int(result.fire_state_grid[20:40, 45:65].sum()), 0)

    def test_tiny_isolated_mask_component_is_removed(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        mask[90:91, 90:91] = 1.0
        det = FireMaskDetection(mask=mask, class_name="fire", confidence=0.9)

        result = detections_to_fire_grid(
            [det],
            image_shape=(100, 100),
            config=FireGridConfig(
                grid_shape=(100, 100),
                grid_threshold=0.2,
                min_component_cells=4,
            ),
        )

        self.assertGreater(result.fire_state_grid[20:40, 20:40].sum(), 0)
        self.assertEqual(int(result.fire_state_grid[90, 90]), 0)

    def test_thin_fire_front_is_preserved(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 10:90] = 1.0
        det = FireMaskDetection(mask=mask, class_name="fire", confidence=0.9)

        result = detections_to_fire_grid(
            [det],
            image_shape=(100, 100),
            config=FireGridConfig(
                grid_shape=(100, 100),
                grid_threshold=0.2,
                min_component_cells=4,
            ),
        )

        self.assertGreater(result.fire_state_grid[50].sum(), 70)
        self.assertEqual(int(result.fire_state_grid[49].sum()), 0)
        self.assertEqual(int(result.fire_state_grid[51].sum()), 0)

    def test_smoke_box_uses_weaker_evidence(self):
        det = FireBoxDetection(
            box_xyxy=(0.0, 0.0, 100.0, 100.0),
            class_name="smoke",
            confidence=0.9,
        )

        result = detections_to_fire_grid(
            [det],
            image_shape=(100, 100),
            config=FireGridConfig(
                grid_shape=(100, 100),
                smoke_weight=0.35,
                grid_threshold=0.5,
            ),
        )

        self.assertEqual(int(result.fire_state_grid.sum()), 0)
        self.assertGreater(result.evidence_grid.sum(), 0)

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

    def test_fire_grid_stats_compare_vision_and_simulator_distributions(self):
        vision = np.zeros((100, 100), dtype=np.uint8)
        vision[20:30, 20:30] = 1
        vision[70:75, 70:75] = 1

        simulator = np.zeros((100, 100), dtype=np.uint8)
        simulator[40:70, 40:70] = 1

        stats = compute_fire_grid_stats(vision)
        comparison = compare_grid_distributions([vision], [simulator])

        self.assertEqual(stats.connected_component_count, 2)
        self.assertGreater(stats.burning_cell_percentage, 0.0)
        self.assertIn("vision", comparison)
        self.assertIn("simulator", comparison)
        self.assertIn("delta_vision_minus_simulator", comparison)

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
