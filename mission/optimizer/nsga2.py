"""
Single-UAV NSGA-II mission optimizer using official pymoo operators.

Adapts the multi-objective firefighting task-planning philosophy from
"Multi-Target Firefighting Task Planning Strategy for Multiple UAVs Under
Dynamic Forest Fire Environment" (2025) to:

  - one UAV (not a fleet)
  - synthetic suppression targets (ConvLSTM-ready schema)
  - permutation chromosomes (visit order + feasible subset via decoding)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mission.config.settings import MissionConfig, OptimizerConfig
from mission.fitness.scoring import MissionPlan, MissionScorer
from mission.optimizer.chromosome import decode_permutation
from mission.optimizer.problem import SuppressionMissionProblem
from mission.simulation.environment import WildfireEnvironment


@dataclass(frozen=True)
class OptimizationResult:
    """Pareto set returned by ``NSGA2MissionOptimizer.optimize``."""

    plans: tuple[MissionPlan, ...]
    # Minimization objective matrix (n_solutions × 3)
    F: np.ndarray
    # Permutation chromosomes (n_solutions × n_targets)
    X: np.ndarray

    @property
    def n_solutions(self) -> int:
        return len(self.plans)

    def best_damage_plan(self) -> MissionPlan:
        """Plan with the highest damage prevented (among Pareto set)."""
        return max(self.plans, key=lambda p: p.objectives.damage_prevented)

    def best_distance_plan(self) -> MissionPlan:
        """Plan with the lowest travel distance (among Pareto set)."""
        return min(self.plans, key=lambda p: p.objectives.travel_distance)

    def best_battery_plan(self) -> MissionPlan:
        """Plan with the lowest battery usage (among Pareto set)."""
        return min(self.plans, key=lambda p: p.objectives.battery_usage)

    def summary(self) -> str:
        lines = [f"Pareto set: {self.n_solutions} mission plan(s)"]
        for plan in self.plans:
            lines.append(plan.summary())
            lines.append("")
        return "\n".join(lines).rstrip()


class NSGA2MissionOptimizer:
    """
    Run pymoo NSGA-II over suppression-target permutations.

    Operators (official pymoo — not reinvented):
      - sampling:  ``PermutationRandomSampling``
      - crossover: ``OrderCrossover`` (OX)
      - mutation:  ``InversionMutation``
      - selection / survival: NSGA-II defaults (tournament + crowding)
    """

    def __init__(
        self,
        env: WildfireEnvironment,
        config: OptimizerConfig | MissionConfig | None = None,
    ) -> None:
        if isinstance(config, MissionConfig):
            opt_cfg = config.optimizer
            self.seed = config.seed
        elif isinstance(config, OptimizerConfig):
            opt_cfg = config
            self.seed = None
        else:
            opt_cfg = OptimizerConfig()
            self.seed = None

        self.env = env
        self.config = opt_cfg
        self.problem = SuppressionMissionProblem(env, opt_cfg)
        self.scorer = MissionScorer(env, opt_cfg)

    def optimize(
        self,
        seed: int | None = None,
        callback: object | None = None,
    ) -> OptimizationResult:
        """
        Execute NSGA-II and return the non-dominated mission plans.

        Parameters
        ----------
        seed:
            Optional RNG seed override. Falls back to the value stored when
            the optimizer was constructed from a ``MissionConfig``.
        callback:
            Optional pymoo ``Callback`` (e.g. convergence logging).
        """
        rng_seed = seed if seed is not None else self.seed

        algorithm = NSGA2(
            pop_size=self.config.population_size,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(prob=self.config.crossover_prob),
            mutation=InversionMutation(prob=self.config.mutation_prob),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", self.config.n_generations)

        minimize_kwargs: dict = {
            "seed": rng_seed,
            "verbose": self.config.verbose,
        }
        if callback is not None:
            minimize_kwargs["callback"] = callback

        result = minimize(
            self.problem,
            algorithm,
            termination,
            **minimize_kwargs,
        )

        return self._build_result(result.X, result.F)

    def _build_result(
        self,
        X: np.ndarray | None,
        F: np.ndarray | None,
    ) -> OptimizationResult:
        if X is None or F is None or len(np.atleast_2d(X)) == 0:
            return OptimizationResult(plans=(), F=np.zeros((0, 3)), X=np.zeros((0, 0)))

        X_arr = np.atleast_2d(np.asarray(X, dtype=int))
        F_arr = np.atleast_2d(np.asarray(F, dtype=float))

        # Deduplicate identical mission ID sequences while keeping Pareto F.
        plans: list[MissionPlan] = []
        seen: set[tuple[int, ...]] = set()
        kept_X: list[np.ndarray] = []
        kept_F: list[np.ndarray] = []

        for row_x, row_f in zip(X_arr, F_arr, strict=True):
            decoded = decode_permutation(
                row_x,
                self.env.targets,
                self.env.drone,
                self.config,
            )
            key = decoded.target_ids
            if key in seen:
                continue
            seen.add(key)

            fitness = self.scorer.evaluate_decoded(decoded)
            plan = self.scorer.plan_from_fitness(len(plans), fitness)
            plans.append(plan)
            kept_X.append(row_x)
            kept_F.append(np.asarray(fitness.F, dtype=float))

        if not plans:
            return OptimizationResult(plans=(), F=np.zeros((0, 3)), X=np.zeros((0, 0)))

        # Re-filter after decode/dedup: chromosome→mission mapping can leave
        # weakly dominated plans when battery ≈ travel distance.
        nd_mask = _nondominated_mask(np.vstack(kept_F))
        plans_nd = []
        X_nd: list[np.ndarray] = []
        F_nd: list[np.ndarray] = []
        for keep, plan, row_x, row_f in zip(nd_mask, plans, kept_X, kept_F, strict=True):
            if not keep:
                continue
            plans_nd.append(
                MissionPlan(
                    index=len(plans_nd),
                    target_ids=plan.target_ids,
                    objectives=plan.objectives,
                    F=plan.F,
                )
            )
            X_nd.append(row_x)
            F_nd.append(row_f)

        return OptimizationResult(
            plans=tuple(plans_nd),
            F=np.vstack(F_nd) if F_nd else np.zeros((0, 3)),
            X=np.vstack(X_nd) if X_nd else np.zeros((0, self.env.n_targets)),
        )


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows for a minimization matrix ``F``."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i if j is <= all objs and < at least one.
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    return keep
