"""
Isolated multi-objective fitness functions for UAV suppression missions.

pymoo minimizes all objectives; maximization of damage prevented is therefore
returned as a negated value by the problem wrapper, not here.
"""

from __future__ import annotations

from mission.simulation.targets import DroneState, SuppressionTarget
from mission.utils.geometry import euclidean_distance


def objective_damage_prevented(
    mission_targets: list[SuppressionTarget],
    damage_metric: str,
) -> float:
    """
    Objective 1 (to maximize): predicted damage prevented by the mission.

    ``damage_metric`` selects the aggregation attribute:
      - ``"predicted_damage"`` → sum of ``target.predicted_damage``
      - ``"priority"`` → sum of ``target.priority``
    """
    if damage_metric == "priority":
        return float(sum(t.priority for t in mission_targets))
    if damage_metric == "predicted_damage":
        return float(sum(t.predicted_damage for t in mission_targets))
    raise ValueError(f"Unknown damage_metric: {damage_metric!r}")


def objective_travel_distance(
    drone: DroneState,
    mission_targets: list[SuppressionTarget],
) -> float:
    """
    Objective 2 (to minimize): SURROGATE travel-cost estimate for NSGA-II's
    fitness evaluation — total Euclidean tour length, NOT the actual
    obstacle-aware route.

    This is a deliberate approximation, not an oversight: evaluating real
    D* Lite routing for every chromosome in every generation would be far
    too expensive (population_size x n_generations D* Lite searches per
    optimization run) and isn't required for NSGA-II to usefully rank
    candidate missions relative to each other. After a mission is selected,
    ``mission.replanning.executor.DStarLiteMissionExecutor`` computes the
    real obstacle-aware route and records both costs on
    ``MissionExecutionResult`` (``path_length`` = actual,
    ``straight_line_length`` = this surrogate, ``deviation_ratio`` = their
    ratio) so the gap between estimate and actual can be measured, not
    silently assumed to be zero.

    Path: drone start → target_1 → … → target_k  (no return-to-depot).
    """
    if not mission_targets:
        return 0.0

    total = 0.0
    prev_x, prev_y = drone.x, drone.y
    for target in mission_targets:
        total += euclidean_distance(prev_x, prev_y, target.x, target.y)
        prev_x, prev_y = target.x, target.y
    return float(total)


def objective_battery_usage(
    travel_distance: float,
    battery_distance_factor: float,
) -> float:
    """
    Objective 3 (to minimize): approximate battery consumption.

    Phase-2 proxy: ``battery = travel_distance * battery_distance_factor``.
    Replace this function body later with a real energy model.
    """
    return float(travel_distance * battery_distance_factor)
