"""
Analysis helpers for comparing vision-generated fire grids to simulator grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class FireGridStats:
    """Shape statistics for a binary fire-state grid."""

    burning_cell_percentage: float
    connected_component_count: int
    average_connected_region_size: float
    boundary_complexity: float
    fire_coverage: float
    burning_cells: int
    total_cells: int


def compute_fire_grid_stats(grid: np.ndarray | torch.Tensor) -> FireGridStats:
    """
    Compute simulator-facing statistics for a binary fire-state grid.

    `boundary_complexity` is normalized by burning area so compact rectangles
    and irregular fire fronts can be compared across different fire sizes.
    """
    arr = torch.as_tensor(grid).detach().cpu().numpy() if isinstance(grid, torch.Tensor) else np.asarray(grid)
    binary = (arr >= 1).astype(np.uint8)
    total_cells = int(binary.size)
    burning_cells = int(binary.sum())
    fire_coverage = burning_cells / total_cells if total_cells else 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_sizes = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
    ]
    component_count = len(component_sizes)
    average_size = float(np.mean(component_sizes)) if component_sizes else 0.0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, closed=True) for contour in contours))
    boundary_complexity = perimeter / np.sqrt(max(1, burning_cells))

    return FireGridStats(
        burning_cell_percentage=fire_coverage * 100.0,
        connected_component_count=component_count,
        average_connected_region_size=average_size,
        boundary_complexity=boundary_complexity,
        fire_coverage=fire_coverage,
        burning_cells=burning_cells,
        total_cells=total_cells,
    )


def summarize_fire_grid_stats(grids: Iterable[np.ndarray | torch.Tensor]) -> dict[str, float]:
    """Average fire-grid statistics over a collection of grids."""
    stats = [compute_fire_grid_stats(grid) for grid in grids]
    if not stats:
        return {
            "burning_cell_percentage": 0.0,
            "connected_component_count": 0.0,
            "average_connected_region_size": 0.0,
            "boundary_complexity": 0.0,
            "fire_coverage": 0.0,
        }

    return {
        "burning_cell_percentage": float(np.mean([s.burning_cell_percentage for s in stats])),
        "connected_component_count": float(np.mean([s.connected_component_count for s in stats])),
        "average_connected_region_size": float(np.mean([s.average_connected_region_size for s in stats])),
        "boundary_complexity": float(np.mean([s.boundary_complexity for s in stats])),
        "fire_coverage": float(np.mean([s.fire_coverage for s in stats])),
    }


def compare_grid_distributions(
    vision_grids: Iterable[np.ndarray | torch.Tensor],
    simulator_grids: Iterable[np.ndarray | torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Compare average statistics for vision grids and simulator grids."""
    vision_summary = summarize_fire_grid_stats(vision_grids)
    simulator_summary = summarize_fire_grid_stats(simulator_grids)
    delta = {
        key: vision_summary[key] - simulator_summary[key]
        for key in vision_summary
    }
    return {
        "vision": vision_summary,
        "simulator": simulator_summary,
        "delta_vision_minus_simulator": delta,
    }


def load_simulator_fire_grids(
    sim_paths: Iterable[str | Path],
    max_frames_per_sim: int = 20,
) -> list[np.ndarray]:
    """Load channel-0 fire-state grids from simulator `.pt` tensors."""
    grids: list[np.ndarray] = []
    for sim_path in sim_paths:
        tensor = torch.load(sim_path, map_location="cpu", weights_only=True)
        if tensor.ndim != 4 or tensor.shape[1] < 1:
            raise ValueError(f"{sim_path} must contain a tensor shaped (T, C, H, W)")

        fire = tensor[:max_frames_per_sim, 0]
        for frame in fire:
            grids.append((frame >= 1).to(torch.uint8).numpy())
    return grids
