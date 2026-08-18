"""Constraint helpers for decoded UAV missions."""

from __future__ import annotations

from mission.config.settings import OptimizerConfig


def violates_max_targets(n_targets: int, config: OptimizerConfig) -> bool:
    """True if the mission visits more targets than allowed."""
    return n_targets > config.max_mission_targets


def violates_max_distance(travel_distance: float, config: OptimizerConfig) -> bool:
    """True if the mission tour exceeds the configured distance budget."""
    return travel_distance > config.max_mission_distance


def constraint_max_targets(n_targets: int, config: OptimizerConfig) -> float:
    """
    Inequality constraint g <= 0 for pymoo.

    g = n_targets - max_mission_targets
    """
    return float(n_targets - config.max_mission_targets)


def constraint_max_distance(travel_distance: float, config: OptimizerConfig) -> float:
    """
    Inequality constraint g <= 0 for pymoo.

    g = travel_distance - max_mission_distance
    """
    return float(travel_distance - config.max_mission_distance)
