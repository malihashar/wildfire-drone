import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.convlstm import WildfireConvLSTM
from src.vision.convlstm_bridge import (
    load_terrain_weather_from_simulation,
    predict_next_fire_from_grid,
)


class TestConvLSTMBridge(unittest.TestCase):
    def test_load_terrain_weather_from_simulation_normalizes_expected_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sim_path = tmp_path / "sim.pt"
            norm_path = tmp_path / "normalization.json"

            sim = torch.zeros(3, 10, 4, 4)
            sim[:, 1] = 12.0
            sim[:, 2] = 22.0
            sim[:, 3] = 32.0
            sim[:, 4] = 42.0
            sim[:, 7] = 72.0
            torch.save(sim, sim_path)

            norm_path.write_text(
                json.dumps(
                    {
                        "potential_ros": {"mean": 10.0, "std": 2.0},
                        "fireline_intensity": {"mean": 20.0, "std": 2.0},
                        "vegetation_density": {"mean": 30.0, "std": 2.0},
                        "slope": {"mean": 40.0, "std": 2.0},
                        "wind_speed": {"mean": 70.0, "std": 2.0},
                    }
                ),
                encoding="utf-8",
            )

            terrain_weather = load_terrain_weather_from_simulation(
                sim_path,
                start_t=0,
                timesteps=2,
                norm_json=norm_path,
            )

            self.assertEqual(tuple(terrain_weather.shape), (2, 9, 4, 4))
            self.assertTrue(torch.equal(terrain_weather[:, 0], torch.ones(2, 4, 4)))
            self.assertTrue(torch.equal(terrain_weather[:, 1], torch.ones(2, 4, 4)))
            self.assertTrue(torch.equal(terrain_weather[:, 2], torch.ones(2, 4, 4)))
            self.assertTrue(torch.equal(terrain_weather[:, 3], torch.ones(2, 4, 4)))
            self.assertTrue(torch.equal(terrain_weather[:, 6], torch.ones(2, 4, 4)))

    def test_predict_next_fire_from_grid_loads_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "best_model.pt"
            model = WildfireConvLSTM()
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            fire_grid = torch.zeros(8, 8)
            terrain_weather = torch.zeros(1, 9, 8, 8)

            pred = predict_next_fire_from_grid(
                fire_grid,
                terrain_weather,
                checkpoint_path=ckpt_path,
                timesteps=1,
                device="cpu",
            )

            self.assertEqual(tuple(pred.shape), (8, 8))
            self.assertTrue(torch.all(pred >= 0.0))
            self.assertTrue(torch.all(pred <= 1.0))


if __name__ == "__main__":
    unittest.main()
