"""
Scalar mission scoring for selecting one plan from a Pareto set.

Multi-objective NSGA-II returns a front; online replanning needs a single
mission to execute (and later hand to D* Lite). Scores are normalized within
the current Pareto set so weights remain interpretable across ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mission.fitness.scoring import MissionPlan
from mission.optimizer.nsga2 import OptimizationResult


@dataclass(frozen=True)
class MissionSelectionConfig:
    """Weights for the scalarized mission score (higher is better)."""

    w_damage: float = 1.0
    w_travel: float = 1.0
    w_battery: float = 0.5


@dataclass(frozen=True)
class ScoredMission:
    """A Pareto plan paired with its scalar selection score."""

    plan: MissionPlan
    score: float

    @property
    def target_ids(self) -> tuple[int, ...]:
        return self.plan.target_ids

    @property
    def mission_order(self) -> list[int]:
        return self.plan.mission_order


def score_mission(
    plan: MissionPlan,
    result: OptimizationResult,
    config: MissionSelectionConfig | None = None,
) -> float:
    """
    Scalar mission score (higher is better).

    ``score = w_d * dmg_n - w_t * travel_n - w_b * battery_n``
    with each term min-max normalized over the current Pareto set.
    """
    cfg = config or MissionSelectionConfig()
    if result.n_solutions == 0:
        return float("-inf")

    damages = np.array([p.objectives.damage_prevented for p in result.plans], dtype=float)
    travels = np.array([p.objectives.travel_distance for p in result.plans], dtype=float)
    batteries = np.array([p.objectives.battery_usage for p in result.plans], dtype=float)

    dmg_n = _norm_max_better(plan.objectives.damage_prevented, damages)
    travel_n = _norm_min_better(plan.objectives.travel_distance, travels)
    battery_n = _norm_min_better(plan.objectives.battery_usage, batteries)

    return float(
        cfg.w_damage * dmg_n
        - cfg.w_travel * travel_n
        - cfg.w_battery * battery_n
    )


def select_highest_scoring_mission(
    result: OptimizationResult,
    config: MissionSelectionConfig | None = None,
) -> ScoredMission:
    """Select the Pareto plan with the maximum scalar score."""
    if result.n_solutions == 0:
        raise ValueError("Cannot select a mission from an empty Pareto set.")

    cfg = config or MissionSelectionConfig()
    scored = [
        ScoredMission(plan=plan, score=score_mission(plan, result, cfg))
        for plan in result.plans
    ]
    return max(scored, key=lambda s: s.score)


def _norm_max_better(value: float, values: np.ndarray) -> float:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return 1.0
    return float((value - lo) / (hi - lo))


def _norm_min_better(value: float, values: np.ndarray) -> float:
    """Normalize so smaller raw values map toward 0 cost (0 = best)."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return 0.0
    return float((value - lo) / (hi - lo))
