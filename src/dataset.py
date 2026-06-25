"""
WildfireDataset
---------------
Sliding-window dataset for ConvLSTM wildfire spread prediction.

Each sample is:
    input  : Tensor[WINDOW_LEN, C, H, W]  (normalised)
    target : Tensor[H, W]                  (binary fire mask at t+1)

Channel layout (10 total):
    0  Fire State          (0=unburned, 1=burning, 2=burned)
    1  Potential ROS       <- normalised
    2  Fireline Intensity  <- normalised
    3  Vegetation Density  <- normalised
    4  Slope               <- normalised
    5  Aspect Cosine
    6  Aspect Sine
    7  Wind Speed          <- normalised
    8  Wind Dir Cosine
    9  Wind Dir Sine
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

# ── channel indices ────────────────────────────────────────────────────────────
CHANNEL_FIRE      = 0
CHANNEL_ROS       = 1
CHANNEL_INTENSITY = 2
CHANNEL_VEG       = 3
CHANNEL_SLOPE     = 4
CHANNEL_WIND_SPD  = 7

# mapping: channel index → key in normalization.json
NORM_CHANNELS = {
    CHANNEL_ROS:       "potential_ros",
    CHANNEL_INTENSITY: "fireline_intensity",
    CHANNEL_VEG:       "vegetation_density",
    CHANNEL_SLOPE:     "slope",
    CHANNEL_WIND_SPD:  "wind_speed",
}


class WildfireDataset(Dataset):
    """
    Parameters
    ----------
    split_json : str | Path
        Path to train.json / val.json / test.json
    norm_json  : str | Path
        Path to dataset/normalization.json
    window_len : int
        Number of input frames (default 20)
    sim_dir    : str | Path
        Root directory containing simulation .pt files.
        Paths in the split JSON are relative to this directory's parent.
    """

    def __init__(
        self,
        split_json: str | Path,
        norm_json: str | Path,
        window_len: int = 20,
        sim_dir: str | Path = "dataset/simulations",
    ):
        super().__init__()
        self.window_len = window_len
        self.sim_dir    = Path(sim_dir)

        # ── load normalization statistics ──────────────────────────────────────
        with open(norm_json, "r", encoding="utf-8") as f:
            norm_data = json.load(f)

        # store as {channel_idx: (mean, std)}
        self.norm_stats: dict[int, tuple[float, float]] = {}
        for ch_idx, norm_key in NORM_CHANNELS.items():
            stats = norm_data[norm_key]
            mean  = float(stats["mean"])
            std   = float(stats["std"]) or 1.0  # guard divide-by-zero
            self.norm_stats[ch_idx] = (mean, std)

        # ── load split metadata ────────────────────────────────────────────────
        with open(split_json, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # ── build flat index of (sim_file, start_t) windows ───────────────────
        # Each entry: sim file has T timesteps -> T-window_len windows
        self.windows: list[tuple[Path, int]] = []  # (abs_path, start_t)

        for entry in entries:
            # Support both path formats in the JSON:
            #   "simulations\\sim_00001.pt"  (old, relative to dataset/)
            #   or absolute path
            raw_path = entry.get("path", "")
            # Normalise to just the filename part
            sim_name = Path(raw_path).name          # "sim_00001.pt"
            sim_path = self.sim_dir / sim_name

            T = int(entry["timesteps"])
            n_windows = T - window_len              # number of valid windows
            if n_windows > 0:
                for start_t in range(n_windows):
                    self.windows.append((sim_path, start_t))

    # ── cache: avoid reloading the same file for consecutive windows ──────────
    # LRU-style: keep last loaded tensor in memory between __getitem__ calls
    _cache_path: Path | None = None
    _cache_tensor: torch.Tensor | None = None

    def _load_sim(self, path: Path) -> torch.Tensor:
        if path == self._cache_path:
            return self._cache_tensor  # type: ignore[return-value]
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        WildfireDataset._cache_path   = path
        WildfireDataset._cache_tensor = tensor
        return tensor

    # ── dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sim_path, start_t = self.windows[idx]

        # shape: (T, C, H, W)
        tensor = self._load_sim(sim_path).float()

        # input sequence: [start_t, start_t + window_len)
        x = tensor[start_t : start_t + self.window_len].clone()  # (W, C, H, W)

        # target: fire state at next timestep (channel 0)
        target_frame = tensor[start_t + self.window_len, CHANNEL_FIRE]  # (H, W)

        # ── normalise selected channels in x ───────────────────────────────────
        for ch_idx, (mean, std) in self.norm_stats.items():
            x[:, ch_idx] = (x[:, ch_idx] - mean) / std

        # ── binarise target (burning=1, burning+burned=1, unburned=0) ──────────
        # Strategy: predict "on fire NOW or WILL burn" -> fire_state >= 1
        target_binary = (target_frame >= 1).float()  # (H, W)

        return x, target_binary


def build_loaders(
    dataset_root: str | Path = "dataset",
    window_len: int = 20,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple["DataLoader", "DataLoader", "DataLoader"]:  # type: ignore[name-defined]
    """
    Convenience function that returns (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    root     = Path(dataset_root)
    norm_json = root / "normalization.json"
    sim_dir   = root / "simulations"

    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = WildfireDataset(
            split_json=root / "metadata" / f"{split}.json",
            norm_json=norm_json,
            window_len=window_len,
            sim_dir=sim_dir,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    train_loader = _make("train", shuffle=True)
    val_loader   = _make("val",   shuffle=False)
    test_loader  = _make("test",  shuffle=False)
    return train_loader, val_loader, test_loader
