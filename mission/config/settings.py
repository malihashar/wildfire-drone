"""
Mission-planner configuration.

Grid size matches the existing ConvLSTM / YOLO pipeline (100×100) so that
synthetic targets can later be replaced by predicted wildfire cells without
changing coordinate conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GridConfig:
    """Spatial domain for the wildfire mission environment."""

    width: int = 100
    height: int = 100

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive.")


@dataclass(frozen=True)
class TargetGenerationConfig:
    """
    Parameters for synthetic suppression-target generation.

    Later: ConvLSTM predicted-fire cells will feed the same target schema;
    these ranges only apply to the Phase-1 simulator.

    ``spatial_mode``:
      - ``"uniform"``  — targets scattered uniformly at random (original,
        default behaviour).
      - ``"clustered"`` — targets grouped around ``n_clusters`` random
        centers (Gaussian scatter, std ``cluster_spread``), representing
        multiple independent fire fronts rather than one diffuse scatter.

    ``wind_direction_deg`` / ``wind_bias_strength``: when set, targets
    downwind of ``wind_direction_deg`` (0 = +x axis, degrees CCW) get a
    higher ``damage_score`` — a simple, documented proxy for wind-driven
    fire intensity, not a physical spread model.
    """

    min_targets: int = 10
    max_targets: int = 20
    damage_score_range: tuple[float, float] = (0.1, 1.0)
    priority_range: tuple[float, float] = (0.1, 1.0)
    spatial_mode: str = "uniform"
    n_clusters: int = 3
    cluster_spread: float = 8.0
    wind_direction_deg: float | None = None
    wind_bias_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.min_targets < 1:
            raise ValueError("min_targets must be >= 1.")
        if self.max_targets < self.min_targets:
            raise ValueError("max_targets must be >= min_targets.")
        if self.spatial_mode not in {"uniform", "clustered"}:
            raise ValueError("spatial_mode must be 'uniform' or 'clustered'.")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if not 0.0 <= self.wind_bias_strength <= 1.0:
            raise ValueError("wind_bias_strength must be in [0, 1].")


@dataclass(frozen=True)
class VisualizationConfig:
    """Matplotlib scene rendering options."""

    figsize: tuple[float, float] = (10.0, 10.0)
    dpi: int = 120
    show_labels: bool = True
    save_dir: Path = field(default_factory=lambda: Path("outputs/mission"))


@dataclass(frozen=True)
class OptimizerConfig:
    """
    NSGA-II mission-optimization settings (pymoo).

    Inspired by the multi-objective firefighting task-planning framework in
    the 2025 multi-UAV paper, adapted here to a single UAV with permutation
    chromosomes over synthetic (later ConvLSTM) suppression targets.
    """

    population_size: int = 80
    n_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3
    # Soft/hard mission limits applied during chromosome decoding.
    max_mission_distance: float = 250.0
    max_mission_targets: int = 8
    # Objective 1 aggregation: "predicted_damage" or "priority".
    damage_metric: str = "predicted_damage"
    # Battery ≈ travel_distance * factor until a real battery model exists.
    battery_distance_factor: float = 1.0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4.")
        if self.n_generations < 1:
            raise ValueError("n_generations must be >= 1.")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1].")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1].")
        if self.max_mission_distance <= 0.0:
            raise ValueError("max_mission_distance must be positive.")
        if self.max_mission_targets < 1:
            raise ValueError("max_mission_targets must be >= 1.")
        if self.damage_metric not in {"predicted_damage", "priority"}:
            raise ValueError(
                "damage_metric must be 'predicted_damage' or 'priority'."
            )
        if self.battery_distance_factor < 0.0:
            raise ValueError("battery_distance_factor must be >= 0.")


@dataclass(frozen=True)
class MissionConfig:
    """Top-level configuration for the mission-planning research prototype."""

    grid: GridConfig = field(default_factory=GridConfig)
    targets: TargetGenerationConfig = field(default_factory=TargetGenerationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int | None = 42


DEFAULT_CONFIG = MissionConfig()
