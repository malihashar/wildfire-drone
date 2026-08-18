"""
pymoo problem definition for single-UAV multi-target suppression planning.

Decision variables are a permutation of candidate target indices.
Objectives (all minimized by pymoo):

  f1 = -damage_prevented   (maximize damage prevented)
  f2 =  travel_distance
  f3 =  battery_usage

Inequality constraints (g <= 0), enforced primarily by chromosome decoding
and reported explicitly for transparency:

  g1 = n_targets - max_mission_targets
  g2 = travel_distance - max_mission_distance
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from mission.config.settings import OptimizerConfig
from mission.fitness.scoring import MissionScorer
from mission.simulation.environment import WildfireEnvironment


class SuppressionMissionProblem(ElementwiseProblem):
    """
    Multi-objective permutation problem for wildfire suppression missions.

    Parameters
    ----------
    env:
        Synthetic (or later ConvLSTM-derived) wildfire environment.
    config:
        Optimizer / constraint / objective configuration.
    """

    def __init__(
        self,
        env: WildfireEnvironment,
        config: OptimizerConfig,
    ) -> None:
        if env.n_targets < 1:
            raise ValueError("Environment must contain at least one target.")

        self.env = env
        self.config = config
        self.scorer = MissionScorer(env, config)

        super().__init__(
            n_var=env.n_targets,
            n_obj=3,
            n_ieq_constr=2,
            xl=0,
            xu=env.n_targets - 1,
            vtype=int,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args: object, **kwargs: object) -> None:
        fitness = self.scorer.evaluate_permutation(x)
        out["F"] = list(fitness.F)
        out["G"] = list(fitness.G)
