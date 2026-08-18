"""Mission fitness / scoring."""

from __future__ import annotations

__all__ = [
    "FitnessVector",
    "MissionObjectives",
    "MissionPlan",
    "MissionScorer",
    "objective_battery_usage",
    "objective_damage_prevented",
    "objective_travel_distance",
]


def __getattr__(name: str):
    if name in {
        "FitnessVector",
        "MissionObjectives",
        "MissionPlan",
        "MissionScorer",
    }:
        from mission.fitness import scoring

        return getattr(scoring, name)
    if name in {
        "objective_battery_usage",
        "objective_damage_prevented",
        "objective_travel_distance",
    }:
        from mission.fitness import objectives

        return getattr(objectives, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
