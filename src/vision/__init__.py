"""
Computer vision adapters for wildfire detection.

This package converts drone RGB imagery into the 100x100 fire-state grid used
by the existing ConvLSTM pipeline. It intentionally does not modify the
simulator, dataset tensor format, or ConvLSTM model.
"""

from src.vision.yolo_fire_adapter import (
    FireBoxDetection,
    FireGridConfig,
    FireMaskDetection,
    YoloFireSegmenter,
    build_convlstm_sequence,
    cell_to_image_rect,
    detections_to_fire_grid,
    overlay_grid_on_image,
    plot_fire_grid_diagnostics,
)
from src.vision.convlstm_bridge import (
    load_convlstm_checkpoint,
    load_terrain_weather_from_simulation,
    predict_next_fire_from_grid,
)
from src.vision.grid_analysis import (
    FireGridStats,
    compare_grid_distributions,
    compute_fire_grid_stats,
    load_simulator_fire_grids,
    summarize_fire_grid_stats,
)

__all__ = [
    "FireGridConfig",
    "FireBoxDetection",
    "FireMaskDetection",
    "YoloFireSegmenter",
    "build_convlstm_sequence",
    "cell_to_image_rect",
    "detections_to_fire_grid",
    "overlay_grid_on_image",
    "plot_fire_grid_diagnostics",
    "load_convlstm_checkpoint",
    "load_terrain_weather_from_simulation",
    "predict_next_fire_from_grid",
    "FireGridStats",
    "compute_fire_grid_stats",
    "summarize_fire_grid_stats",
    "compare_grid_distributions",
    "load_simulator_fire_grids",
]
