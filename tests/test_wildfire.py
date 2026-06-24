import unittest
import numpy as np
import torch
import os
import shutil
from src.config import SimulationConfig, WindConfig, TerrainConfig
from src.physics import compute_slope_and_aspect, calculate_directional_ros, compute_fireline_intensity
from src.simulator import WildfireSimulator
from src.data_exporter import export_simulation_to_pytorch, export_simulation_to_numpy, WildfireDataset

class TestWildfireSimulator(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            rows=20,
            cols=20,
            dx=10.0,
            dy=10.0,
            wind=WindConfig(speed=10.0, direction=90.0), # Blowing East
            terrain=TerrainConfig(elevation_type="flat"),
            ignition_points=[(10, 10)]
        )
        self.test_dir = "test_outputs"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_slope_and_aspect_flat(self):
        elevation = np.zeros((10, 10))
        slope, aspect = compute_slope_and_aspect(elevation, 10.0, 10.0)
        self.assertTrue(np.allclose(slope, 0.0))
        self.assertTrue(np.allclose(aspect, 0.0))

    def test_slope_and_aspect_simple_gradient(self):
        # Elevation increases towards East (increasing columns)
        # Z = 0.1 * X
        y = np.linspace(0, 90, 10)
        x = np.linspace(0, 90, 10)
        X, Y = np.meshgrid(x, y)
        elevation = 0.1 * X
        
        slope, aspect = compute_slope_and_aspect(elevation, 10.0, 10.0)
        
        # dZ/dx = 0.1, dZ/dy = 0
        # tan(slope) = 0.1 => slope = arctan(0.1)
        expected_slope = np.arctan(0.1)
        self.assertTrue(np.allclose(slope, expected_slope, atol=1e-2))
        # aspect: uphill direction is East (+x axis) which is 90 degrees
        # check middle elements to avoid border effects
        self.assertTrue(np.allclose(aspect[2:-2, 2:-2], 90.0, atol=1e-1))

    def test_directional_ros_wind_effect(self):
        # East wind: blowing towards East (90 degrees)
        config = SimulationConfig(
            wind=WindConfig(speed=10.0, direction=90.0),
            base_ros_scale=1.0,
            wind_factor_coeff=0.5
        )
        veg = np.ones((5, 5))
        slope = np.zeros((5, 5))
        aspect = np.zeros((5, 5))
        
        # Case A: spreading towards East (from (2, 2) to (2, 3)), direction theta_d = 90
        ros_east = calculate_directional_ros((2, 2), (2, 3), config, veg, slope, aspect)
        
        # Case B: spreading towards West (from (2, 2) to (2, 1)), direction theta_d = 270
        ros_west = calculate_directional_ros((2, 2), (2, 1), config, veg, slope, aspect)
        
        # ROS East should be higher because of wind alignment
        self.assertGreater(ros_east, ros_west)

    def test_simulator_step(self):
        sim = WildfireSimulator(self.config)
        self.assertEqual(sim.state_grid[10, 10], 1)
        
        # Step once
        still_burning = sim.step()
        self.assertTrue(still_burning)
        # Check that state history contains recorded timesteps
        self.assertEqual(len(sim.history), 2) # step 0 and step 1

    def test_data_exporter(self):
        sim = WildfireSimulator(self.config)
        for _ in range(5):
            sim.step()
            
        pt_path = os.path.join(self.test_dir, "test_tensor.pt")
        npz_path = os.path.join(self.test_dir, "test_data.npz")
        
        tensor = export_simulation_to_pytorch(sim.history, pt_path)
        export_simulation_to_numpy(sim.history, npz_path)
        
        # Assertions
        self.assertTrue(os.path.exists(pt_path))
        self.assertTrue(os.path.exists(npz_path))
        
        self.assertEqual(tensor.shape[0], 6) # 0 to 5 timesteps
        self.assertEqual(tensor.shape[1], 10) # 10 channels
        self.assertEqual(tensor.shape[2], self.config.rows)
        self.assertEqual(tensor.shape[3], self.config.cols)

if __name__ == '__main__':
    unittest.main()
