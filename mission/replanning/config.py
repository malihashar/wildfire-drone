"""
Online replanning configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mission.fitness.mission_selection import MissionSelectionConfig
from mission.simulation.dynamics import SyntheticDynamicsConfig


@dataclass(frozen=True)
class OnlineReplanConfig:
    """Parameters for the online replanning architecture demo."""

    seed: int = 42
    grid_size: int = 100
    n_targets_initial: int = 15

    # Optimizer budget per replan (kept modest for interactive demos).
    population_size: int = 50
    n_generations: int = 50
    max_mission_distance: float = 250.0
    max_mission_targets: int = 8
    damage_metric: str = "predicted_damage"

    # Simulation / prediction stream.
    n_replan_events: int = 6
    dynamics: SyntheticDynamicsConfig = field(
        default_factory=lambda: SyntheticDynamicsConfig(max_ticks=6)
    )
    selection: MissionSelectionConfig = field(default_factory=MissionSelectionConfig)

    # Fly to and suppress (remove) the next live target of the current
    # mission each tick before the prediction update / replan. Disabling
    # this keeps the UAV parked and targets un-suppressed (useful for
    # isolating pure prediction-driven replanning from mission progress).
    advance_drone: bool = True

    # Animation
    animation_fps: float = 1.25
    hold_frames_per_event: int = 10
    figsize: tuple[float, float] = (11.0, 9.0)
    dpi: int = 120

    output_dir: Path = field(default_factory=lambda: Path("results/online_replan"))

    def __post_init__(self) -> None:
        # Keep dynamics tick budget aligned with requested replan events.
        object.__setattr__(
            self,
            "dynamics",
            SyntheticDynamicsConfig(
                min_patches_per_tick=self.dynamics.min_patches_per_tick,
                max_patches_per_tick=self.dynamics.max_patches_per_tick,
                priority_delta_range=self.dynamics.priority_delta_range,
                move_delta_range=self.dynamics.move_delta_range,
                damage_delta_range=self.dynamics.damage_delta_range,
                p_priority_up=self.dynamics.p_priority_up,
                p_priority_down=self.dynamics.p_priority_down,
                p_add=self.dynamics.p_add,
                p_remove=self.dynamics.p_remove,
                p_move=self.dynamics.p_move,
                p_damage=self.dynamics.p_damage,
                min_targets=self.dynamics.min_targets,
                max_targets=self.dynamics.max_targets,
                max_ticks=self.n_replan_events,
            ),
        )
