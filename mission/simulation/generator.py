"""
Synthetic suppression-target generator.

Phase 1: random targets on the grid, with two scenario dimensions beyond a
single uniform scatter: spatial clustering (multiple independent fire
fronts) and a wind-driven severity bias. Later: replace
``generate_synthetic_targets`` with a ConvLSTM → target adapter that maps
predicted fire intensity / spread cells into ``SuppressionTarget`` instances
with the same schema.
"""

from __future__ import annotations

import math

import numpy as np

from mission.config.settings import GridConfig, TargetGenerationConfig
from mission.simulation.targets import DroneState, SuppressionTarget
from mission.utils.geometry import euclidean_distance


def generate_drone_start(
    grid: GridConfig,
    rng: np.random.Generator,
) -> DroneState:
    """Sample a random continuous start pose inside the grid."""
    x = float(rng.uniform(0.0, grid.width - 1))
    y = float(rng.uniform(0.0, grid.height - 1))
    return DroneState(x=x, y=y)


def generate_synthetic_targets(
    grid: GridConfig,
    target_cfg: TargetGenerationConfig,
    drone: DroneState,
    rng: np.random.Generator,
) -> list[SuppressionTarget]:
    """
    Place suppression targets according to ``target_cfg.spatial_mode``.

    ``travel_cost`` is initialized to Euclidean distance from the drone start
    as a temporary stand-in until D* Lite provides path-aware costs.
    """
    n_targets = int(rng.integers(target_cfg.min_targets, target_cfg.max_targets + 1))
    dmg_lo, dmg_hi = target_cfg.damage_score_range
    pri_lo, pri_hi = target_cfg.priority_range

    if target_cfg.spatial_mode == "clustered":
        cells = _sample_clustered_cells(grid, n_targets, target_cfg, rng)
    else:
        cells = _sample_uniform_cells(grid, n_targets, rng)

    wind_rad = (
        math.radians(target_cfg.wind_direction_deg)
        if target_cfg.wind_direction_deg is not None
        else None
    )
    cx, cy = (grid.width - 1) / 2.0, (grid.height - 1) / 2.0

    targets: list[SuppressionTarget] = []
    for target_id, (cell_x, cell_y) in enumerate(cells):
        x, y = float(cell_x), float(cell_y)

        damage_score = float(rng.uniform(dmg_lo, dmg_hi))
        if wind_rad is not None and target_cfg.wind_bias_strength > 0.0:
            damage_score = _apply_wind_bias(
                damage_score, x, y, cx, cy, wind_rad, target_cfg.wind_bias_strength, dmg_hi
            )
        priority = float(rng.uniform(pri_lo, pri_hi))
        travel_cost = euclidean_distance(drone.x, drone.y, x, y)

        targets.append(
            SuppressionTarget(
                id=target_id,
                x=x,
                y=y,
                damage_score=damage_score,
                priority=priority,
                travel_cost=travel_cost,
            )
        )

    return targets


def _apply_wind_bias(
    damage_score: float,
    x: float,
    y: float,
    cx: float,
    cy: float,
    wind_rad: float,
    strength: float,
    dmg_hi: float,
) -> float:
    """
    Nudge ``damage_score`` upward for targets downwind of grid center.

    Documented proxy, not a physical spread model: projects the target's
    offset from center onto the wind direction unit vector; targets further
    downwind get a larger fraction of ``strength`` blended in toward
    ``dmg_hi``. Purely a scenario-generation knob for experiments.
    """
    wind_dx, wind_dy = math.cos(wind_rad), math.sin(wind_rad)
    max_extent = math.hypot(cx, cy) or 1.0
    downwind = ((x - cx) * wind_dx + (y - cy) * wind_dy) / max_extent
    downwind = max(0.0, min(1.0, downwind))  # only boost downwind side, clip to [0, 1]
    return float(damage_score * (1.0 - strength * downwind) + dmg_hi * strength * downwind)


def _sample_uniform_cells(
    grid: GridConfig,
    n_targets: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    for _ in range(n_targets):
        cell = _sample_unique_cell(grid, occupied, rng)
        occupied.add(cell)
        cells.append(cell)
    return cells


def _sample_clustered_cells(
    grid: GridConfig,
    n_targets: int,
    target_cfg: TargetGenerationConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Scatter targets around ``n_clusters`` random centers (multiple fire fronts)."""
    n_clusters = min(target_cfg.n_clusters, n_targets)
    centers = [
        (rng.uniform(0, grid.width - 1), rng.uniform(0, grid.height - 1))
        for _ in range(n_clusters)
    ]

    occupied: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    max_attempts = grid.width * grid.height
    for i in range(n_targets):
        center_x, center_y = centers[i % n_clusters]
        cell = None
        for _ in range(max_attempts):
            candidate_x = int(round(rng.normal(center_x, target_cfg.cluster_spread)))
            candidate_y = int(round(rng.normal(center_y, target_cfg.cluster_spread)))
            candidate_x = int(np.clip(candidate_x, 0, grid.width - 1))
            candidate_y = int(np.clip(candidate_y, 0, grid.height - 1))
            candidate = (candidate_x, candidate_y)
            if candidate not in occupied:
                cell = candidate
                break
        if cell is None:
            cell = _sample_unique_cell(grid, occupied, rng)
        occupied.add(cell)
        cells.append(cell)
    return cells


def _sample_unique_cell(
    grid: GridConfig,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample an integer grid cell not already occupied by another target."""
    max_attempts = grid.width * grid.height
    for _ in range(max_attempts):
        cell = (int(rng.integers(0, grid.width)), int(rng.integers(0, grid.height)))
        if cell not in occupied:
            return cell
    raise RuntimeError("Unable to place unique suppression targets on the grid.")
