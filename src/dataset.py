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
from torch.utils.data import Dataset, DataLoader, Sampler

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


# ──────────────────────────────────────────────────────────────────────────────
# Sampler
# ──────────────────────────────────────────────────────────────────────────────

class SimulationGroupedSampler(Sampler):
    """
    Yields window indices grouped by simulation file.

    Simulations are shuffled each epoch; all windows within a simulation
    are emitted in sequence before the next simulation starts.

    This keeps the 1-file LRU cache in WildfireDataset permanently warm:
    each simulation file is loaded exactly once per epoch instead of once
    per sample, eliminating the disk thrashing caused by shuffle=True on
    the raw window index list.

    Parameters
    ----------
    sim_groups : list[list[int]]
        Outer list = one entry per simulation.
        Inner list = flat window indices belonging to that simulation.
    shuffle : bool
        If True, shuffles simulation order each iteration.
    """

    def __init__(self, sim_groups: list[list[int]], shuffle: bool = True) -> None:
        self.sim_groups = sim_groups
        self.shuffle    = shuffle

    def __iter__(self):
        order = list(range(len(self.sim_groups)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield from self.sim_groups[i]

    def __len__(self) -> int:
        return sum(len(g) for g in self.sim_groups)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

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
    stride     : int
        Step between consecutive window start frames (default 1).
        stride=10 reduces 130k windows to ~13k, cutting I/O and CPU
        time by ~10x with minimal loss of training diversity.
    """

    def __init__(
        self,
        split_json: str | Path,
        norm_json: str | Path,
        window_len: int = 20,
        sim_dir: str | Path = "dataset/simulations",
        stride: int = 1,
    ):
        super().__init__()
        self.window_len = window_len
        self.sim_dir    = Path(sim_dir)
        self.stride     = max(1, stride)

        # ── load normalization statistics ──────────────────────────────────────
        with open(norm_json, "r", encoding="utf-8") as f:
            norm_data = json.load(f)

        self.norm_stats: dict[int, tuple[float, float]] = {}
        for ch_idx, norm_key in NORM_CHANNELS.items():
            stats = norm_data[norm_key]
            mean  = float(stats["mean"])
            std   = float(stats["std"]) or 1.0
            self.norm_stats[ch_idx] = (mean, std)

        # ── load split metadata ────────────────────────────────────────────────
        with open(split_json, "r", encoding="utf-8") as f:
            entries = json.load(f)

        # ── build flat index of (sim_file, start_t) windows ───────────────────
        self.windows:    list[tuple[Path, int]] = []
        # sim_groups[i] = list of flat indices in self.windows for simulation i
        # Used by SimulationGroupedSampler to preserve cache locality.
        self.sim_groups: list[list[int]]        = []

        for entry in entries:
            raw_path = entry.get("path", "")
            sim_name = Path(raw_path).name
            sim_path = self.sim_dir / sim_name

            T = int(entry["timesteps"])
            n_windows = T - window_len
            if n_windows <= 0:
                continue

            group: list[int] = []
            for start_t in range(0, n_windows, self.stride):
                idx = len(self.windows)
                self.windows.append((sim_path, start_t))
                group.append(idx)

            if group:
                self.sim_groups.append(group)

    # ── cache: one file held in RAM; effective when indices are grouped by sim ─
    _cache_path:   Path | None         = None
    _cache_tensor: torch.Tensor | None = None

    def _load_sim(self, path: Path) -> torch.Tensor:
        if path == self._cache_path:
            return self._cache_tensor          # type: ignore[return-value]
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        WildfireDataset._cache_path   = path
        WildfireDataset._cache_tensor = tensor
        return tensor

    # ── dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sim_path, start_t = self.windows[idx]

        tensor = self._load_sim(sim_path).float()   # (T, C, H, W)

        x            = tensor[start_t : start_t + self.window_len].clone()
        target_frame = tensor[start_t + self.window_len, CHANNEL_FIRE]

        for ch_idx, (mean, std) in self.norm_stats.items():
            x[:, ch_idx] = (x[:, ch_idx] - mean) / std

        target_binary = (target_frame >= 1).float()
        return x, target_binary


# ──────────────────────────────────────────────────────────────────────────────
# Loader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(
    dataset_root: str | Path = "dataset",
    window_len:   int  = 20,
    batch_size:   int  = 8,
    num_workers:  int  = 0,
    pin_memory:   bool = False,
    stride:       int  = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Training loader uses SimulationGroupedSampler so each simulation
    file is loaded exactly once per epoch (cache-friendly).
    Val/test loaders use default sequential order.
    """
    root      = Path(dataset_root)
    norm_json = root / "normalization.json"
    sim_dir   = root / "simulations"

    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = WildfireDataset(
            split_json = root / "metadata" / f"{split}.json",
            norm_json  = norm_json,
            window_len = window_len,
            sim_dir    = sim_dir,
            stride     = stride,
        )
        if shuffle:
            sampler: Sampler | None = SimulationGroupedSampler(
                ds.sim_groups, shuffle=True
            )
            do_shuffle = False   # sampler owns shuffling; DataLoader must not
        else:
            sampler    = None
            do_shuffle = False

        return DataLoader(
            ds,
            batch_size         = batch_size,
            sampler            = sampler,
            shuffle            = do_shuffle,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = (num_workers > 0),
        )

    return _make("train", shuffle=True), _make("val", shuffle=False), _make("test", shuffle=False)
