"""
End-to-end wildfire prediction: drone RGB image -> YOLO grid -> ConvLSTM forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.vision.convlstm_bridge import (
    load_terrain_weather_from_simulation,
    predict_next_fire_from_grid,
    resolve_yolo_device,
)
from src.vision.paths import DEFAULT_NORM_JSON, resolve_convlstm_checkpoint, resolve_path
from src.vision.paths import DEFAULT_YOLO_SEG_MODEL
from src.vision.yolo_fire_adapter import FireGridResult, YoloFireSegmenter


@dataclass
class WildfirePredictionResult:
    """Outputs from the YOLO + ConvLSTM product flow."""

    grid_result: FireGridResult
    next_fire_probability: torch.Tensor
    terrain_weather: torch.Tensor


def predict_wildfire_from_image(
    image: np.ndarray | str | Path,
    sim_path: str | Path,
    *,
    yolo_model_path: str | Path | None = None,
    convlstm_checkpoint: str | Path | None = None,
    norm_json: str | Path | None = None,
    start_t: int = 0,
    timesteps: int = 20,
    device: str | torch.device = "auto",
    yolo: YoloFireSegmenter | None = None,
) -> WildfirePredictionResult:
    """
    Run the full product flow on one image.

    1. YOLO segmentation produces the observed fire-state grid (channel 0).
    2. Terrain/weather channels 1-9 are loaded from a simulator `.pt` file.
    3. The trained ConvLSTM predicts next-fire probabilities on the 100x100 grid.
    """
    yolo_path = resolve_path(yolo_model_path, DEFAULT_YOLO_SEG_MODEL, "YOLO model")
    ckpt_path = resolve_convlstm_checkpoint(convlstm_checkpoint)

    segmenter = yolo or YoloFireSegmenter(
        model_path=yolo_path,
        device=resolve_yolo_device(device),
    )
    grid_result = segmenter.predict_grid(image)

    norm_path = None
    if norm_json is not None:
        norm_path = resolve_path(norm_json, DEFAULT_NORM_JSON, "normalization.json")
    elif DEFAULT_NORM_JSON.exists():
        norm_path = DEFAULT_NORM_JSON

    terrain_weather = load_terrain_weather_from_simulation(
        sim_path,
        start_t=start_t,
        timesteps=timesteps,
        norm_json=norm_path,
    )

    next_fire = predict_next_fire_from_grid(
        grid_result.fire_state_grid,
        terrain_weather,
        checkpoint_path=ckpt_path,
        timesteps=timesteps,
        device=device,
    )

    return WildfirePredictionResult(
        grid_result=grid_result,
        next_fire_probability=next_fire,
        terrain_weather=terrain_weather,
    )
