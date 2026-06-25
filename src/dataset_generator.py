import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

# Ensure the project root is on the Python path when this file is executed directly
if __package__ is None:
    # script is run as a file, add the parent directory (project root) to sys.path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

from src.config import SimulationConfig, WindConfig, TerrainConfig, VegetationConfig
from src.simulator import WildfireSimulator
from src.data_exporter import export_simulation_to_pytorch

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def estimate_tensor_size(timesteps: int, rows: int = 100, cols: int = 100, channels: int = 10, dtype_size: int = 4) -> int:
    """Return an estimated size in bytes for a single simulation tensor.

    Args:
        timesteps: Number of time steps.
        rows, cols: Spatial dimensions (default 100×100).
        channels: Number of channels (default 10).
        dtype_size: Size of each element in bytes (float32 = 4).
    """
    return timesteps * channels * rows * cols * dtype_size

def format_bytes(num: int) -> str:
    """Human‑readable byte formatting (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"

# ---------------------------------------------------------------------------
# Cleanup short simulations
# ---------------------------------------------------------------------------

def cleanup_short_simulations(sim_dir: Path, metadata_dir: Path, min_timesteps: int = 50) -> None:
    """Delete simulations shorter than *min_timesteps* and clean metadata.

    The function scans ``sim_dir`` for ``*.pt`` files, loads each tensor, and
    removes those that do not meet the length requirement. Corresponding entries
    are removed from each split JSON (train/val/test)."""
    # Load all split metadata
    split_files = {name: metadata_dir / f"{name}.json" for name in ["train", "val", "test"]}
    split_meta: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in split_files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                split_meta[name] = json.load(f)
        else:
            split_meta[name] = []

    removed = 0
    for sim_file in sorted(sim_dir.glob("*.pt")):
        try:
            tensor = torch.load(sim_file)
        except Exception:
            continue
        if tensor.shape[0] < min_timesteps:
            # Delete file
            sim_file.unlink(missing_ok=True)
            # Remove from every split list
            rel_path = str(sim_file.relative_to(sim_file.parents[2]))  # relative to output root
            for meta in split_meta.values():
                meta[:] = [e for e in meta if e.get("path") != rel_path]
            removed += 1
    # Write back cleaned metadata
    for name, path in split_files.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_meta[name], f, indent=2)
    print(f"[Cleanup] Removed {removed} short simulations (<{min_timesteps} timesteps).")

# ---------------------------------------------------------------------------
# Main generation function with checkpointing and limits
# ---------------------------------------------------------------------------

def generate_dataset(
    num_simulations: int,
    output_root: str,
    max_steps: int = 500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    max_runtime_seconds: int = 7200,
    min_simulations: int = 1000,
    storage_limit_bytes: int = 20 * 1024 ** 3,  # 20 GB
) -> None:
    """Generate a wildfire dataset respecting runtime, storage, and checkpoint limits.

    The function creates ``output_root`` with the following layout:

    ``output_root/``
        ``simulations/`` – raw ``.pt`` tensors (compressed via torch's zip
        serialization).
        ``metadata/`` – ``train.json``, ``val.json``, ``test.json``.
        ``train/``, ``val/``, ``test/`` – *copies* of the simulation files for
        easy access.
    """
    start_time = time.time()
    output_path = Path(output_root)
    sim_dir = output_path / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load or create checkpoint
    # ---------------------------------------------------------------------
    checkpoint_path = output_path / "gen_checkpoint.json"
    if checkpoint_path.exists():
        # Use utf-8-sig to gracefully ignore a possible BOM that may have been added by Windows editors
        with open(checkpoint_path, "r", encoding="utf-8-sig") as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get("last_index", -1) + 1
        metadata = checkpoint.get("metadata", [])
        print(f"[Resume] Continuing from simulation {start_idx}.")
    else:
        start_idx = 0
        metadata = []
        checkpoint = {"last_index": -1, "metadata": [], "start_time": start_time}

    # ---------------------------------------------------------------------
    # Estimate storage requirement based on average length of existing data
    # (if any) and abort early if it would exceed the limit.
    # ---------------------------------------------------------------------
    existing_lengths = [entry["timesteps"] for entry in metadata]
    avg_len = sum(existing_lengths) / len(existing_lengths) if existing_lengths else 300
    est_size_per = estimate_tensor_size(avg_len)
    est_total = est_size_per * num_simulations
    print(f"[Estimate] Expected total size: {format_bytes(est_total)} (limit {format_bytes(storage_limit_bytes)})")
    compress = est_total > storage_limit_bytes
    if compress:
        print("[Info] Expected size exceeds limit – enabling torch compression (zip serialization).")

    # ---------------------------------------------------------------------
    # Prepare base configuration
    # ---------------------------------------------------------------------
    np.random.default_rng(random_seed)
    random.seed(random_seed)
    base_cfg = SimulationConfig()

    # ---------------------------------------------------------------------
    # Generation loop (with tqdm progress bar)
    # ---------------------------------------------------------------------
    pbar = tqdm(total=num_simulations, initial=start_idx, desc="Generating simulations")
    for i in range(start_idx, num_simulations):
        # Runtime guard
        if time.time() - start_time > max_runtime_seconds:
            print("[Timeout] Reached maximum runtime. Stopping generation.")
            break

        # Randomized config and reproducible seed per simulation
        sim_seed = random_seed + i
        cfg = random_simulation_config(base_cfg, seed=sim_seed)
        sim = WildfireSimulator(cfg)
        steps = 0
        while True:
            if not sim.step():
                break
            steps += 1
            if steps >= max_steps:
                print(f"[!] Max steps reached ({max_steps}) for simulation {i}, breaking.")
                break

        # Export tensor (compressed if needed)
        sim_path = sim_dir / f"sim_{i:05d}.pt"
        export_simulation_to_pytorch(sim.history, str(sim_path), compress=compress)

        # Build metadata entry with full parameter snapshot
        entry = {
            "id": f"sim_{i:05d}",
            "path": str(sim_path.relative_to(output_path)),
            "timesteps": len(sim.history),
            "seed": sim_seed,
            "params": {
                "wind": {"speed": cfg.wind.speed, "direction": cfg.wind.direction},
                "terrain": {
                    "type": cfg.terrain.elevation_type,
                    "slope_angle_deg": getattr(cfg.terrain, "slope_angle_deg", None),
                    "aspect_deg": getattr(cfg.terrain, "aspect_deg", None),
                },
                "vegetation": {"density": cfg.vegetation.base_density},
                "ignition": {"points": cfg.ignition_points},
            },
        }
        metadata.append(entry)

        # Periodic checkpoint (every 10 simulations)
        if (i + 1) % 10 == 0:
            checkpoint["last_index"] = i
            checkpoint["metadata"] = metadata
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)

        pbar.update(1)
    pbar.close()

    # ---------------------------------------------------------------------
    # Finalize splits (train/val/test) – copy files into split folders
    # ---------------------------------------------------------------------
    random.shuffle(metadata)
    n_total = len(metadata)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_meta = metadata[:n_train]
    val_meta = metadata[n_train:n_train + n_val]
    test_meta = metadata[n_train + n_val:]

    for name, data in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        # Write split metadata
        out_path = metadata_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Copy files to split folder
        split_dir = output_path / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for entry in data:
            src = output_path / entry["path"]
            dst = split_dir / src.name
            if not dst.exists():
                # Use shutil.copy2 to preserve metadata
                import shutil
                shutil.copy2(src, dst)

    # Cleanup checkpoint file – generation finished
    if checkpoint_path.exists():
        os.remove(checkpoint_path)

    print("[Done] Dataset generation complete.")
    print(f"    - {n_train} training simulations, {n_val} validation, {len(test_meta)} test.")

# ---------------------------------------------------------------------------
# Early entry point removed – will use proper guard at file end
# ---------------------------------------------------------------------------
# Entry point moved to end of file to ensure all functions are defined before execution.
# (You can run the script directly; the new entry point at the bottom will be used.)


import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

# Ensure the project root is on the Python path when this file is executed directly
if __package__ is None:
    # script is run as a file, add the parent directory (project root) to sys.path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))

from src.config import SimulationConfig, WindConfig, TerrainConfig, VegetationConfig
from src.simulator import WildfireSimulator
from src.data_exporter import export_simulation_to_pytorch


def random_simulation_config(base_config: SimulationConfig, seed: int | None = None) -> SimulationConfig:
    """Return a copy of *base_config* with randomized fields.

    Randomized parameters:
    - ignition location (random cell within the grid)
    - wind speed (0 – 20 m/s)
    - wind direction (0 – 360°)
    - vegetation base density (0.3 – 1.0)
    - terrain type ("flat", "slope", "sinusoidal")
    - for "slope" terrain also random slope_angle_deg (0 – 20°) and aspect_deg (0 – 360°)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Random ignition point inside the grid
    ir = random.randint(0, base_config.rows - 1)
    ic = random.randint(0, base_config.cols - 1)

    # Random wind
    wind_speed = random.uniform(0.0, 20.0)
    wind_dir = random.uniform(0.0, 360.0)
    wind_cfg = WindConfig(speed=wind_speed, direction=wind_dir)

    # Random terrain
    terrain_type = random.choice(["flat", "slope", "sinusoidal"])
    if terrain_type == "slope":
        slope_angle = random.uniform(0.0, 20.0)
        aspect = random.uniform(0.0, 360.0)
        terrain_cfg = TerrainConfig(
            elevation_type=terrain_type,
            max_elevation=base_config.terrain.max_elevation,
            slope_angle_deg=slope_angle,
            aspect_deg=aspect,
        )
    else:
        terrain_cfg = TerrainConfig(
            elevation_type=terrain_type,
            max_elevation=base_config.terrain.max_elevation,
            slope_angle_deg=base_config.terrain.slope_angle_deg,
            aspect_deg=base_config.terrain.aspect_deg,
        )

    # Random vegetation density
    veg_density = random.uniform(0.3, 1.0)
    veg_cfg = VegetationConfig(
        density_type=base_config.vegetation.density_type,
        base_density=veg_density,
        noise_scale=base_config.vegetation.noise_scale,
    )

    # Build new SimulationConfig – keep other numeric parameters unchanged
    cfg = SimulationConfig(
        rows=base_config.rows,
        cols=base_config.cols,
        dx=base_config.dx,
        dy=base_config.dy,
        dt=base_config.dt,
        base_ros_scale=base_config.base_ros_scale,
        wind_factor_coeff=base_config.wind_factor_coeff,
        slope_factor_coeff=base_config.slope_factor_coeff,
        heat_combustion=base_config.heat_combustion,
        max_fuel_load=base_config.max_fuel_load,
        wind=wind_cfg,
        terrain=terrain_cfg,
        vegetation=veg_cfg,
        ignition_points=[(ir, ic)],
    )
    return cfg



if __name__ == "__main__":
    generate_dataset(
        num_simulations=2000,
        output_root="dataset",
        max_steps=500,
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42,
        max_runtime_seconds=7200,
        min_simulations=1000,
        storage_limit_bytes=20 * 1024 ** 3,
    )
