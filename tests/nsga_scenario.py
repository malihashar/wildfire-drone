"""
Reproducible random wildfire-suppression scenario generator, shared by
``tests/nsga_benchmark.py`` and ``tests/nsga_mavsdk_sitl.py``.

Reuses the project's actual optimizer end-to-end -- nothing here reimplements
NSGA-II:
  - ``mission.optimizer.nsga2.NSGA2MissionOptimizer`` (pymoo NSGA-II)
  - ``mission.simulation.targets.SuppressionTarget`` / ``DroneState``
  - ``mission.simulation.environment.WildfireEnvironment``
  - ``mission.config.settings.OptimizerConfig`` / ``TargetGenerationConfig``
    (severity ranges only -- the *sampling scheme* mirrors
    ``test_five_points_random.py``: ``r = R * sqrt(U)``, uniform angle,
    minimum spacing, all within ``RADIUS_M`` of home)

Coordinates are LOCAL NORTH/EAST METRE OFFSETS from home ``(0, 0)`` -- the
same convention ``test_five_points_random.py`` and
``mission.flight.mavsdk_controller`` use for GPS conversion. This is NOT the
abstract 100x100 mission grid used elsewhere in the pipeline (D* Lite /
ConvLSTM); NSGA-II's objectives (``mission.fitness.objectives``) only ever
consume ``DroneState``/``SuppressionTarget`` ``.x``/``.y`` and don't care
what units they're in, so this is a drop-in scenario, not a fork of the
optimizer.

``run_nsga2_with_deadline`` drives the SAME pymoo ``NSGA2`` algorithm and
operators ``NSGA2MissionOptimizer.optimize`` uses (order crossover, inversion
mutation, permutation sampling) -- it exists only because ``optimize()``
hardcodes a fixed-generation ``get_termination("n_gen", ...)`` and has no
wall-clock option. Swapping in pymoo's own ``TimeBasedTermination`` is a
termination-criterion change, not a reimplementation of NSGA-II.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination

from mission.config.settings import OptimizerConfig, TargetGenerationConfig
from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.targets import DroneState, SuppressionTarget

RADIUS_M = 10.0
MIN_SPACING_M = 5.0
MIN_TARGETS = 7
MAX_TARGETS = 10
MAX_SAMPLE_ATTEMPTS = 10_000

# Severity (damage_score/priority) ranges: reused verbatim from the project's
# own synthetic-target generator config, not invented for this script.
_TARGET_CFG = TargetGenerationConfig()


@dataclass(frozen=True)
class Scenario:
    seed: int
    env: WildfireEnvironment
    optimizer_config: OptimizerConfig

    @property
    def n_targets(self) -> int:
        return self.env.n_targets


def _sample_offsets(rng: np.random.Generator, n: int) -> list[tuple[float, float]]:
    """Uniform-in-disc sampling (r = R*sqrt(U)) with minimum spacing, capped at RADIUS_M."""
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n:
        attempts += 1
        if attempts > MAX_SAMPLE_ATTEMPTS:
            raise RuntimeError(
                f"Could not sample {n} points >= {MIN_SPACING_M}m apart within "
                f"a {RADIUS_M}m radius after {MAX_SAMPLE_ATTEMPTS} attempts."
            )
        r = RADIUS_M * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2 * math.pi)
        north_m = r * math.cos(theta)
        east_m = r * math.sin(theta)
        if any(math.hypot(north_m - n_, east_m - e_) < MIN_SPACING_M for n_, e_ in points):
            continue
        points.append((north_m, east_m))
    return points


def generate_scenario(seed: int, optimizer_config: OptimizerConfig | None = None) -> Scenario:
    """
    Build one reproducible scenario: 7-10 targets, each within RADIUS_M of
    home, at least MIN_SPACING_M apart, with severity drawn from the
    project's existing ``TargetGenerationConfig`` ranges.
    """
    rng = np.random.default_rng(seed)
    n_targets = int(rng.integers(MIN_TARGETS, MAX_TARGETS + 1))
    offsets = _sample_offsets(rng, n_targets)

    dmg_lo, dmg_hi = _TARGET_CFG.damage_score_range
    pri_lo, pri_hi = _TARGET_CFG.priority_range

    drone = DroneState(x=0.0, y=0.0)  # home, local north/east metres
    targets = [
        SuppressionTarget(
            id=i,
            x=north_m,
            y=east_m,
            damage_score=float(rng.uniform(dmg_lo, dmg_hi)),
            priority=float(rng.uniform(pri_lo, pri_hi)),
            travel_cost=math.hypot(north_m, east_m),
        )
        for i, (north_m, east_m) in enumerate(offsets)
    ]

    cfg = optimizer_config or OptimizerConfig()
    span = int(math.ceil(2 * RADIUS_M)) + 1
    env = WildfireEnvironment(width=span, height=span, drone=drone, targets=targets, seed=seed)
    return Scenario(seed=seed, env=env, optimizer_config=cfg)


def scenario_with_generations(scenario: Scenario, n_generations: int) -> Scenario:
    """Same targets/severity/config, only ``n_generations`` changed (fair-comparison benchmark)."""
    return replace(scenario, optimizer_config=replace(scenario.optimizer_config, n_generations=n_generations))


def run_nsga2_with_deadline(
    optimizer: NSGA2MissionOptimizer,
    seed: int,
    deadline_s: float,
) -> tuple[OptimizationResult, float, int]:
    """
    Run the project's exact NSGA-II operators (order crossover, inversion
    mutation, permutation sampling -- copied from
    ``NSGA2MissionOptimizer.optimize`` verbatim) against a wall-clock
    deadline instead of a fixed generation count, then decode/select the
    final Pareto set via the SAME ``_build_result`` the optimizer normally
    uses. Returns ``(result, elapsed_s, generations_completed)``.
    """
    config = optimizer.config
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=config.crossover_prob),
        mutation=InversionMutation(prob=config.mutation_prob),
        eliminate_duplicates=True,
    )
    termination = TimeBasedTermination(deadline_s)

    start = time.perf_counter()
    result = minimize(optimizer.problem, algorithm, termination, seed=seed, verbose=False)
    elapsed = time.perf_counter() - start

    opt_result = optimizer._build_result(result.X, result.F)  # noqa: SLF001 - reuse, not reimplement
    return opt_result, elapsed, int(result.algorithm.n_gen)
