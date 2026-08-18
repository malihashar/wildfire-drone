"""
Mission plan data model and fitness evaluation façade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mission.config.settings import OptimizerConfig
from mission.fitness.constraints import constraint_max_distance, constraint_max_targets
from mission.fitness.objectives import (
    objective_battery_usage,
    objective_damage_prevented,
    objective_travel_distance,
)
from mission.optimizer.chromosome import DecodedMission, decode_permutation
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.targets import SuppressionTarget


@dataclass(frozen=True)
class MissionObjectives:
    """Human-readable (maximization-friendly) objective values."""

    damage_prevented: float
    travel_distance: float
    battery_usage: float


@dataclass(frozen=True)
class MissionPlan:
    """One Pareto-optimal (or candidate) suppression mission."""

    index: int
    target_ids: tuple[int, ...]
    objectives: MissionObjectives
    # Raw pymoo minimization vector: [-damage, travel, battery]
    F: tuple[float, float, float]

    @property
    def mission_order(self) -> list[int]:
        """Ordered list of suppression target IDs to visit."""
        return list(self.target_ids)

    def summary(self) -> str:
        order = " → ".join(f"T{i}" for i in self.target_ids) or "(empty)"
        return (
            f"Mission {self.index}: {order}\n"
            f"  damage_prevented = {self.objectives.damage_prevented:.4f}\n"
            f"  travel_distance  = {self.objectives.travel_distance:.4f}\n"
            f"  battery_usage    = {self.objectives.battery_usage:.4f}"
        )


@dataclass(frozen=True)
class FitnessVector:
    """Evaluated objectives and inequality constraints for one chromosome."""

    damage_prevented: float
    travel_distance: float
    battery_usage: float
    # pymoo minimization objectives
    F: tuple[float, float, float]
    # pymoo inequality constraints (g <= 0)
    G: tuple[float, float]
    decoded: DecodedMission


class MissionScorer:
    """
    Evaluate permutation chromosomes against the three research objectives.

    Objectives are isolated in ``mission.fitness.objectives``; this class
    only orchestrates decode → score → constraint packaging for pymoo.
    """

    def __init__(
        self,
        env: WildfireEnvironment,
        config: OptimizerConfig,
    ) -> None:
        self.env = env
        self.config = config
        self._id_to_target: dict[int, SuppressionTarget] = {
            t.id: t for t in env.targets
        }

    def evaluate_permutation(self, permutation: np.ndarray | list[int]) -> FitnessVector:
        """Decode a chromosome and compute objectives + constraints."""
        decoded = decode_permutation(
            permutation,
            self.env.targets,
            self.env.drone,
            self.config,
        )
        return self.evaluate_decoded(decoded)

    def evaluate_decoded(self, decoded: DecodedMission) -> FitnessVector:
        """Score an already-decoded feasible (or near-feasible) mission."""
        mission_targets = list(decoded.targets)

        damage = objective_damage_prevented(
            mission_targets,
            damage_metric=self.config.damage_metric,
        )
        # Recompute travel via the objective function for a single source of truth.
        travel = objective_travel_distance(self.env.drone, mission_targets)
        battery = objective_battery_usage(
            travel,
            battery_distance_factor=self.config.battery_distance_factor,
        )

        # pymoo minimizes: negate damage so larger prevented-damage is better.
        F = (-damage, travel, battery)
        G = (
            constraint_max_targets(decoded.n_targets, self.config),
            constraint_max_distance(travel, self.config),
        )
        return FitnessVector(
            damage_prevented=damage,
            travel_distance=travel,
            battery_usage=battery,
            F=F,
            G=G,
            decoded=decoded,
        )

    def score(self, mission: Any) -> FitnessVector:
        """
        Evaluate a mission chromosome or ID sequence.

        Accepts a permutation ``np.ndarray`` / ``list[int]`` of indices, or a
        ``DecodedMission``.
        """
        if isinstance(mission, DecodedMission):
            return self.evaluate_decoded(mission)
        return self.evaluate_permutation(mission)

    def plan_from_fitness(self, index: int, fitness: FitnessVector) -> MissionPlan:
        """Package a fitness evaluation into a ``MissionPlan``."""
        return MissionPlan(
            index=index,
            target_ids=fitness.decoded.target_ids,
            objectives=MissionObjectives(
                damage_prevented=fitness.damage_prevented,
                travel_distance=fitness.travel_distance,
                battery_usage=fitness.battery_usage,
            ),
            F=fitness.F,
        )
