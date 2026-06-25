import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class WildfireDataset(Dataset):
    """
    A custom PyTorch Dataset that loads saved simulation tensor files.
    Each item represents a single simulation sequence of shape:
    [Timesteps, Channels, Height, Width]
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.pt')
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns a tensor of shape [Timesteps, Channels, Height, Width]
        return torch.load(self.files[idx])

def export_simulation_to_numpy(history: List[Dict[str, Any]], filepath: str) -> None:
    """
    Export the raw simulation history to a compressed NumPy archive (.npz).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Pack each grid sequence
    state_grids = np.stack([step["state_grid"] for step in history])
    potential_ros_maps = np.stack([step["potential_ros_map"] for step in history])
    active_ros_maps = np.stack([step["active_ros_map"] for step in history])
    intensity_maps = np.stack([step["intensity_map"] for step in history])
    slope_maps = np.stack([step["slope"] for step in history])
    aspect_maps = np.stack([step["aspect"] for step in history])
    vegetation_density_maps = np.stack([step["vegetation_density"] for step in history])
    
    wind_speeds = np.array([step["wind_speed"] for step in history])
    wind_directions = np.array([step["wind_direction"] for step in history])
    timesteps = np.array([step["timestep"] for step in history])
    
    np.savez_compressed(
        filepath,
        timesteps=timesteps,
        state_grids=state_grids,
        potential_ros_maps=potential_ros_maps,
        active_ros_maps=active_ros_maps,
        intensity_maps=intensity_maps,
        slope_maps=slope_maps,
        aspect_maps=aspect_maps,
        vegetation_density_maps=vegetation_density_maps,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions
    )

def export_simulation_to_pytorch(history: List[Dict[str, Any]], filepath: str, compress: bool = False) -> torch.Tensor:
    """
    Export the simulation history as a PyTorch Tensor of shape [Timesteps, Channels, Height, Width].

    Parameters
    ----------
    history: List[Dict]
        Simulation history dictionaries.
    filepath: str
        Destination file path.
    compress: bool, optional
        If True, use PyTorch's newer zip‑serialization (compressed) format.
        If False, use the legacy uncompressed format.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    num_steps = len(history)
    rows, cols = history[0]["state_grid"].shape

    # 10 Channels total
    tensor_data = np.zeros((num_steps, 10, rows, cols), dtype=np.float32)

    for t, step in enumerate(history):
        tensor_data[t, 0] = step["state_grid"].astype(np.float32)
        tensor_data[t, 1] = step["potential_ros_map"]
        tensor_data[t, 2] = step["intensity_map"]
        tensor_data[t, 3] = step["vegetation_density"]
        tensor_data[t, 4] = step["slope"]
        aspect_rad = np.radians(step["aspect"])
        tensor_data[t, 5] = np.cos(aspect_rad)
        tensor_data[t, 6] = np.sin(aspect_rad)
        tensor_data[t, 7] = np.full((rows, cols), step["wind_speed"], dtype=np.float32)
        wind_rad = np.radians(step["wind_direction"])
        tensor_data[t, 8] = np.full((rows, cols), np.cos(wind_rad), dtype=np.float32)
        tensor_data[t, 9] = np.full((rows, cols), np.sin(wind_rad), dtype=np.float32)

    torch_tensor = torch.from_numpy(tensor_data)
    # torch.save supports a hidden arg _use_new_zipfile_serialization to toggle compression
    try:
        torch.save(torch_tensor, filepath, _use_new_zipfile_serialization=compress)
    except RuntimeError as e:
        # If compression fails (e.g., large tensors on Windows), fallback to uncompressed save
        print(f"[Warning] Compression failed for {filepath}: {e}. Saving without compression.")
        torch.save(torch_tensor, filepath, _use_new_zipfile_serialization=False)
    return torch_tensor
    """
    Export the simulation history as a PyTorch Tensor of shape [Timesteps, Channels, Height, Width].
    
    Channels:
      0: Fire State (0 = unburned, 1 = burning, 2 = burned)
      1: Potential ROS Map (m/s)
      2: Fireline Intensity Map (kW/m)
      3: Vegetation Density (0.0 to 1.0)
      4: Slope (angle in radians)
      5: Aspect Cosine (cos(aspect_angle)) - eliminates 0/360 boundary issues
      6: Aspect Sine (sin(aspect_angle))
      7: Wind Speed Map (uniform grid representing speed in m/s)
      8: Wind Direction Cosine (uniform grid representing cos(wind_angle))
      9: Wind Direction Sine (uniform grid representing sin(wind_angle))
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    num_steps = len(history)
    rows, cols = history[0]["state_grid"].shape
    
    # 10 Channels total
    tensor_data = np.zeros((num_steps, 10, rows, cols), dtype=np.float32)
    
    for t, step in enumerate(history):
        # Channel 0: Fire State
        tensor_data[t, 0] = step["state_grid"].astype(np.float32)
        
        # Channel 1: Potential ROS
        tensor_data[t, 1] = step["potential_ros_map"]
        
        # Channel 2: Fireline Intensity
        tensor_data[t, 2] = step["intensity_map"]
        
        # Channel 3: Vegetation Density
        tensor_data[t, 3] = step["vegetation_density"]
        
        # Channel 4: Slope
        tensor_data[t, 4] = step["slope"]
        
        # Aspect Cosine and Sine
        aspect_rad = np.radians(step["aspect"])
        tensor_data[t, 5] = np.cos(aspect_rad)
        tensor_data[t, 6] = np.sin(aspect_rad)
        
        # Wind features (expanded to full grid for spatial convolutional alignment)
        tensor_data[t, 7] = np.full((rows, cols), step["wind_speed"], dtype=np.float32)
        
        wind_rad = np.radians(step["wind_direction"])
        tensor_data[t, 8] = np.full((rows, cols), np.cos(wind_rad), dtype=np.float32)
        tensor_data[t, 9] = np.full((rows, cols), np.sin(wind_rad), dtype=np.float32)
        
    torch_tensor = torch.from_numpy(tensor_data)
    torch.save(torch_tensor, filepath)
    return torch_tensor
