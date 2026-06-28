"""
Bridge YOLO fire grids into the trained ConvLSTM predictor.

The YOLO adapter produces only channel 0: the observed fire-state grid. The
ConvLSTM still needs channels 1-9 from terrain/weather data in the same layout
used during simulator training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.convlstm import WildfireConvLSTM
from src.dataset import NORM_CHANNELS
from src.vision.paths import resolve_convlstm_checkpoint
from src.vision.yolo_fire_adapter import build_convlstm_sequence


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve `auto` to the best available torch device."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_yolo_device(device: str | torch.device = "auto") -> str:
    """Map a torch device choice to an Ultralytics YOLO device string."""
    torch_device = resolve_device(device)
    if torch_device.type == "cuda":
        return "0"
    if torch_device.type == "mps":
        return "mps"
    return "cpu"


def load_convlstm_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> WildfireConvLSTM:
    """Load a `best_model.pt`/`latest_model.pt` checkpoint for inference."""
    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    state_dict: dict[str, Any] = checkpoint.get("model_state_dict", checkpoint)

    # Be tolerant of older DataParallel checkpoints.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model = WildfireConvLSTM(
        in_channels=10,
        hidden_dims=[64, 64],
        kernel_size=3,
        proj_channels=32,
        dropout=0.1,
    )
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()
    return model


def load_terrain_weather_from_simulation(
    sim_path: str | Path,
    start_t: int = 0,
    timesteps: int = 20,
    norm_json: str | Path | None = None,
) -> torch.Tensor:
    """
    Load ConvLSTM channels 1-9 from a simulator tensor.

    Returns a tensor shaped `(T, 9, H, W)`. If `norm_json` is supplied, channels
    are normalized exactly like `WildfireDataset`.
    """
    tensor = torch.load(sim_path, map_location="cpu", weights_only=True).float()
    if tensor.ndim != 4 or tensor.shape[1] < 10:
        raise ValueError("simulation tensor must have shape (T, 10, H, W)")
    if start_t < 0:
        raise ValueError("start_t must be non-negative")
    if start_t + timesteps > tensor.shape[0]:
        raise ValueError(
            f"requested frames {start_t}:{start_t + timesteps}, "
            f"but simulation has {tensor.shape[0]} timesteps"
        )

    terrain_weather = tensor[start_t : start_t + timesteps, 1:10].clone()

    if norm_json is not None:
        with open(norm_json, "r", encoding="utf-8") as f:
            norm_data = json.load(f)
        for original_ch_idx, norm_key in NORM_CHANNELS.items():
            mean = float(norm_data[norm_key]["mean"])
            std = float(norm_data[norm_key]["std"]) or 1.0
            terrain_weather[:, original_ch_idx - 1] = (
                terrain_weather[:, original_ch_idx - 1] - mean
            ) / std

    return terrain_weather


def predict_next_fire_from_grid(
    fire_state_grid: torch.Tensor | Any,
    terrain_weather: torch.Tensor | Any,
    checkpoint_path: str | Path | None = None,
    timesteps: int = 20,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """
    Run ConvLSTM prediction from a YOLO fire grid and terrain/weather channels.

    When `checkpoint_path` is omitted, uses `models/convlstm/best_model.pt`.

    Returns a CPU tensor shaped `(H, W)` with probabilities in `[0, 1]`.
    """
    torch_device = resolve_device(device)
    ckpt_path = resolve_convlstm_checkpoint(checkpoint_path)
    model = load_convlstm_checkpoint(ckpt_path, torch_device)
    sequence = build_convlstm_sequence(
        fire_state_grid=fire_state_grid,
        terrain_weather=terrain_weather,
        timesteps=timesteps,
    )

    with torch.no_grad():
        pred = model(sequence.unsqueeze(0).to(torch_device))
    return pred.squeeze(0).detach().cpu()
