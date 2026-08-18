"""
Online NSGA-II replanning loop for dynamic wildfire predictions.

Inspired by CREDS-style online replanning architecture: when the predicted
environment changes, re-optimize, score the Pareto set, and select a mission
for (future) D* Lite execution.

Each tick, the UAV first suppresses (and removes) the next target of its
current mission, then a new prediction update is applied and NSGA-II
re-optimizes over the *remaining* live targets only. Already-suppressed
targets can never be reselected — only the unfinished part of the mission is
subject to replanning, matching the "complete A, then replan B → D → E, not
blindly continue B → C → D" requirement.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from mission.config.settings import (
    GridConfig,
    MissionConfig,
    OptimizerConfig,
    TargetGenerationConfig,
)
from mission.fitness.mission_selection import (
    ScoredMission,
    select_highest_scoring_mission,
)
from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult
from mission.replanning.config import OnlineReplanConfig
from mission.replanning.executor import (
    DStarLiteMissionExecutor,
    MissionExecutionRequest,
    MissionExecutionResult,
    MissionExecutor,
    build_execution_request,
)
from mission.simulation.dynamics import (
    SyntheticPredictionSource,
    apply_prediction_update,
)
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.prediction_source import (
    EnvironmentDiff,
    PredictionSource,
    PredictionUpdate,
)
from mission.simulation.targets import DroneState


@dataclass(frozen=True)
class ReplanEvent:
    """One online replanning cycle after a prediction update."""

    tick: int
    update: PredictionUpdate
    diff: EnvironmentDiff
    previous_mission: ScoredMission
    new_mission: ScoredMission
    previous_score: float
    new_score: float
    optimization_runtime_s: float
    n_targets: int
    n_pareto: int
    execution_request: MissionExecutionRequest
    execution_result: MissionExecutionResult | None
    env_before: WildfireEnvironment
    env_after: WildfireEnvironment
    why: str
    # Target suppressed (flown to + removed) immediately before this tick's
    # prediction update, or None if the current mission was already empty.
    suppressed_target_id: int | None
    # Full Pareto result for this tick's NSGA-II run (best-in-front stats,
    # CSV export, and future plotting all read from here).
    opt_result: OptimizationResult


@dataclass
class OnlineReplanResult:
    """Full online-replanning demo transcript."""

    initial_env: WildfireEnvironment
    initial_mission: ScoredMission
    initial_runtime_s: float
    initial_n_pareto: int
    initial_result: OptimizationResult
    initial_execution_result: MissionExecutionResult | None
    events: list[ReplanEvent]
    config: OnlineReplanConfig

    @property
    def n_replan_events(self) -> int:
        return len(self.events)


class OnlineReplanner:
    """
    Orchestrates prediction updates → NSGA-II → mission scoring → hand-off.

    Parameters
    ----------
    prediction_source:
        Synthetic by default; pass ``ConvLSTMPredictionSource`` for real
        model-driven replanning.
    executor:
        Receives ``MissionExecutionRequest`` after each selection. Defaults
        to ``DStarLiteMissionExecutor`` (real obstacle-aware local planning).
    callback_factory:
        Optional zero-arg factory returning a fresh pymoo ``Callback`` for
        each NSGA-II run (initial + every replan event). Used by
        ``mission.experiments.online_convergence`` to record per-generation
        history without changing this class's default behaviour (``None``
        keeps every existing call site unaffected). Each callback's
        ``.history`` (or the callback itself, if it has no ``.history``) is
        appended to ``self.tick_histories`` in run order.
    """

    def __init__(
        self,
        config: OnlineReplanConfig | None = None,
        prediction_source: PredictionSource | None = None,
        executor: MissionExecutor | None = None,
        callback_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or OnlineReplanConfig()
        self.prediction_source = prediction_source or SyntheticPredictionSource(
            config=self.config.dynamics,
            seed=self.config.seed + 7,
        )
        self.executor = executor or DStarLiteMissionExecutor()
        self.callback_factory = callback_factory
        self.tick_histories: list[Any] = []

    def run(self) -> OnlineReplanResult:
        """Execute the initial plan plus ``n_replan_events`` online revisions."""
        cfg = self.config
        env = self._create_initial_environment()
        initial_env_snapshot = _snapshot_env(env)

        initial_result, initial_runtime = self._optimize(env, seed=cfg.seed)
        initial_mission = select_highest_scoring_mission(initial_result, cfg.selection)
        initial_execution_result = self.executor.execute(
            build_execution_request(env, initial_mission, tick=0)
        )

        events: list[ReplanEvent] = []
        current_mission = initial_mission

        for tick in range(1, cfg.n_replan_events + 1):
            env_before = _snapshot_env(env)
            prev_mission = current_mission
            prev_score = current_mission.score

            suppressed_id = None
            if cfg.advance_drone:
                suppressed_id = _suppress_next_target(env, current_mission)

            update = self.prediction_source.next_update(env, tick=tick)
            if update is None:
                break

            diff = apply_prediction_update(env, update)
            opt_result, opt_runtime = self._optimize(env, seed=cfg.seed + tick)
            new_mission = select_highest_scoring_mission(opt_result, cfg.selection)
            request = build_execution_request(env, new_mission, tick=tick, risk_map=update.risk_map)
            execution_result = self.executor.execute(request)

            events.append(
                ReplanEvent(
                    tick=tick,
                    update=update,
                    diff=diff,
                    previous_mission=prev_mission,
                    new_mission=new_mission,
                    previous_score=prev_score,
                    new_score=new_mission.score,
                    optimization_runtime_s=opt_runtime,
                    n_targets=env.n_targets,
                    n_pareto=opt_result.n_solutions,
                    execution_request=request,
                    execution_result=execution_result,
                    env_before=env_before,
                    env_after=_snapshot_env(env),
                    why=_explain_replan(update, diff, prev_mission, new_mission),
                    suppressed_target_id=suppressed_id,
                    opt_result=opt_result,
                )
            )
            current_mission = new_mission

        return OnlineReplanResult(
            initial_env=initial_env_snapshot,
            initial_mission=initial_mission,
            initial_runtime_s=initial_runtime,
            initial_execution_result=initial_execution_result,
            initial_n_pareto=initial_result.n_solutions,
            initial_result=initial_result,
            events=events,
            config=cfg,
        )

    def _create_initial_environment(self) -> WildfireEnvironment:
        cfg = self.config
        mission_cfg = MissionConfig(
            grid=GridConfig(width=cfg.grid_size, height=cfg.grid_size),
            targets=TargetGenerationConfig(
                min_targets=cfg.n_targets_initial,
                max_targets=cfg.n_targets_initial,
            ),
            optimizer=self._optimizer_config(),
            seed=cfg.seed,
        )
        return WildfireEnvironment.create_synthetic(mission_cfg)

    def _optimizer_config(self) -> OptimizerConfig:
        cfg = self.config
        return OptimizerConfig(
            population_size=cfg.population_size,
            n_generations=cfg.n_generations,
            max_mission_distance=cfg.max_mission_distance,
            max_mission_targets=cfg.max_mission_targets,
            damage_metric=cfg.damage_metric,
            verbose=False,
        )

    def _optimize(
        self,
        env: WildfireEnvironment,
        seed: int,
    ) -> tuple[OptimizationResult, float]:
        mission_cfg = MissionConfig(optimizer=self._optimizer_config(), seed=seed)
        optimizer = NSGA2MissionOptimizer(env, mission_cfg)
        callback = self.callback_factory() if self.callback_factory is not None else None
        t0 = time.perf_counter()
        result = optimizer.optimize(seed=seed, callback=callback)
        runtime = time.perf_counter() - t0
        if callback is not None:
            self.tick_histories.append(getattr(callback, "history", callback))
        return result, runtime


def _snapshot_env(env: WildfireEnvironment) -> WildfireEnvironment:
    return deepcopy(env)


def _suppress_next_target(env: WildfireEnvironment, mission: ScoredMission) -> int | None:
    """
    Fly to and suppress the first still-live target of ``mission``.

    Suppression removes the target from the environment's live target set so
    it can never be reselected by a later NSGA-II replan — only the
    unfinished remainder of the mission is subject to re-optimization.
    Returns the suppressed target's id, or None if the mission had no
    remaining live targets to suppress.
    """
    if env.n_targets <= 1:
        # Never suppress the last remaining target here; let the next
        # prediction update / replan decide the environment's fate.
        id_map = {t.id: t for t in env.targets}
        for tid in mission.target_ids:
            target = id_map.get(tid)
            if target is not None:
                env.drone = DroneState(x=target.x, y=target.y)
                return None
        return None

    id_map = {t.id: t for t in env.targets}
    for tid in mission.target_ids:
        target = id_map.get(tid)
        if target is not None:
            env.drone = DroneState(x=target.x, y=target.y)
            env.targets = [t for t in env.targets if t.id != tid]
            return tid
    return None


def _explain_replan(
    update: PredictionUpdate,
    diff: EnvironmentDiff,
    previous: ScoredMission,
    new: ScoredMission,
) -> str:
    prev_order = " → ".join(f"T{i}" for i in previous.target_ids) or "(empty)"
    new_order = " → ".join(f"T{i}" for i in new.target_ids) or "(empty)"
    return (
        f"{update.note}. Diff: {diff.describe()}. "
        f"Mission {prev_order} (score={previous.score:.3f}) → "
        f"{new_order} (score={new.score:.3f})."
    )
