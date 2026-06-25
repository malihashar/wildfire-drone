import os
import json
import random
from pathlib import Path
import torch
import numpy as np

# Configuration
SIM_DIR = Path('dataset/simulations')
METADATA_DIR = Path('dataset/metadata')
NORMALIZATION_PATH = Path('dataset/normalization.json')
REPORT_PATH = Path('artifacts/preprocess_report.md')
WINDOW_LEN = 20  # sequence length for ConvLSTM

# Ensure output directories exist
METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_tensor(pt_path):
    """Load a .pt tensor and return as a NumPy array (C, T, H, W)."""
    tensor = torch.load(pt_path)
    return tensor.numpy()

# Step 1: Scan simulations and collect statistics
all_files = sorted(SIM_DIR.glob('sim_*.pt'))
removed = []
kept = []
total_timesteps = 0
min_timesteps = None
max_timesteps = None

# Containers for per‑channel online statistics
channel_stats = {
    "ros": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
    "intensity": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
    "wind_speed": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
    "veg": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')},
    "slope": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": float('inf'), "max": float('-inf')}
}

for file in all_files:
    try:
        arr = load_tensor(file)  # shape: (T, C, H, W)
        timesteps = arr.shape[0]
    except Exception as e:
        # If loading fails, treat as corrupted and remove
        removed.append(file)
        continue
    if timesteps < 50:
        removed.append(file)
        continue
    # Keep this simulation
    kept.append(file)
    total_timesteps += timesteps
    min_timesteps = timesteps if min_timesteps is None else min(min_timesteps, timesteps)
    max_timesteps = timesteps if max_timesteps is None else max(max_timesteps, timesteps)
    # Channel indices based on dataset definition
    # 0: Fire State, 1: ROS, 2: Intensity, 3: Vegetation, 4: Slope, 5: Aspect Cos, 6: Aspect Sin, 7: Wind Speed, 8: Wind Dir Cos, 9: Wind Dir Sin
    # Update online statistics for each channel
    slice = arr[:, 1, :, :].ravel()
    cs = channel_stats["ros"]
    cs["sum"] += slice.sum()
    cs["sum_sq"] += np.square(slice, dtype=np.float64).sum()
    cs["count"] += slice.size
    cs["min"] = min(cs["min"], slice.min())
    cs["max"] = max(cs["max"], slice.max())

    slice = arr[:, 2, :, :].ravel()
    cs = channel_stats["intensity"]
    cs["sum"] += slice.sum()
    cs["sum_sq"] += np.square(slice, dtype=np.float64).sum()
    cs["count"] += slice.size
    cs["min"] = min(cs["min"], slice.min())
    cs["max"] = max(cs["max"], slice.max())

    slice = arr[:, 7, :, :].ravel()
    cs = channel_stats["wind_speed"]
    cs["sum"] += slice.sum()
    cs["sum_sq"] += np.square(slice, dtype=np.float64).sum()
    cs["count"] += slice.size
    cs["min"] = min(cs["min"], slice.min())
    cs["max"] = max(cs["max"], slice.max())

    slice = arr[:, 3, :, :].ravel()
    cs = channel_stats["veg"]
    cs["sum"] += slice.sum()
    cs["sum_sq"] += np.square(slice, dtype=np.float64).sum()
    cs["count"] += slice.size
    cs["min"] = min(cs["min"], slice.min())
    cs["max"] = max(cs["max"], slice.max())

    slice = arr[:, 4, :, :].ravel()
    cs = channel_stats["slope"]
    cs["sum"] += slice.sum()
    cs["sum_sq"] += np.square(slice, dtype=np.float64).sum()
    cs["count"] += slice.size
    cs["min"] = min(cs["min"], slice.min())
    cs["max"] = max(cs["max"], slice.max())

# Delete removed simulations
for file in removed:
    try:
        file.unlink()
    except Exception:
        pass

# Compute overall statistics
num_kept = len(kept)
avg_timesteps = total_timesteps / num_kept if num_kept else 0

# Helper function removed – using online aggregates

# Compute normalization statistics from online aggregates
def finalize_stats(cs):
    mean = cs["sum"] / cs["count"] if cs["count"] > 0 else 0.0
    variance = (cs["sum_sq"] / cs["count"] - mean ** 2) if cs["count"] > 0 else 0.0
    std = float(np.sqrt(max(variance, 0.0)))
    return {"mean": float(mean), "std": std, "min": float(cs["min"]), "max": float(cs["max"]) }
normalization = {"potential_ros": finalize_stats(channel_stats["ros"]), "fireline_intensity": finalize_stats(channel_stats["intensity"]), "wind_speed": finalize_stats(channel_stats["wind_speed"]), "vegetation_density": finalize_stats(channel_stats["veg"]), "slope": finalize_stats(channel_stats["slope"])}

# Write normalization JSON
with open(NORMALIZATION_PATH, 'w', encoding='utf-8') as f:
    json.dump(normalization, f, indent=2)

# Build minimal metadata entries for remaining simulations
metadata_entries = []
for file in kept:
    sim_id = file.stem  # e.g., "sim_00001"
    timesteps = torch.load(file).shape[0]
    entry = {
        "id": sim_id,
        "path": str(file.relative_to(Path('.'))),
        "timesteps": timesteps
    }
    metadata_entries.append(entry)

# Shuffle and split
random.shuffle(metadata_entries)
train_end = int(0.70 * num_kept)
val_end = train_end + int(0.15 * num_kept)
train_meta = metadata_entries[:train_end]
val_meta = metadata_entries[train_end:val_end]
test_meta = metadata_entries[val_end:]

# Write split JSON files
for name, data in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
    out_path = METADATA_DIR / f"{name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Compute sliding‑window sample count
window_samples = sum(max(0, entry["timesteps"] - WINDOW_LEN) for entry in metadata_entries)

# Compute storage usage for kept files (bytes)
storage_bytes = sum(file.stat().st_size for file in kept)

# Generate human‑readable report
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("# Dataset Pre‑processing Report\n\n")
    f.write(f"**Simulations removed**: {len(removed)}\n")
    f.write(f"**Simulations remaining**: {num_kept}\n")
    f.write(f"**Total timesteps (kept)**: {total_timesteps}\n")
    f.write(f"**Average timesteps per simulation**: {avg_timesteps:.2f}\n")
    f.write(f"**Min timesteps**: {min_timesteps}\n")
    f.write(f"**Max timesteps**: {max_timesteps}\n\n")
    f.write(f"**Storage usage (kept files)**: {storage_bytes / (1024**3):.2f} GB\n\n")
    f.write("## Normalization statistics\n\n")
    for channel, stats in normalization.items():
        f.write(f"- **{channel}**: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}\n")
    f.write("\n## Split counts\n\n")
    f.write(f"- Train: {len(train_meta)}\n")
    f.write(f"- Validation: {len(val_meta)}\n")
    f.write(f"- Test: {len(test_meta)}\n\n")
    f.write(f"**Sliding‑window (len={WINDOW_LEN}) training samples**: {window_samples}\n")

print("Pre-processing script completed successfully.")
