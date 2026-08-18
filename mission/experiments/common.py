"""
Shared helpers for optimization-performance experiments.

Keeps environment construction, metric extraction, CSV I/O, and plotting
style consistent across independent experiment modules.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mission.config.settings import (
    GridConfig,
    MissionConfig,
    OptimizerConfig,
    TargetGenerationConfig,
    VisualizationConfig,
)
from mission.experiments.experiment_config import ExperimentConfig, ExperimentPaths
from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult
from mission.simulation.environment import WildfireEnvironment


OBJECTIVE_NAMES = ("damage_prevented", "travel_distance", "battery_usage")


@dataclass(frozen=True)
class TimedOptimization:
    """Optimization result plus wall-clock runtime in seconds."""

    result: OptimizationResult
    runtime_s: float


def build_mission_config(
    exp: ExperimentConfig,
    *,
    n_targets: int | None = None,
    population_size: int | None = None,
    n_generations: int | None = None,
    seed: int | None = None,
) -> MissionConfig:
    """Build a ``MissionConfig`` for a specific experimental condition."""
    n = n_targets if n_targets is not None else exp.n_targets
    return MissionConfig(
        grid=GridConfig(width=exp.grid_size, height=exp.grid_size),
        targets=TargetGenerationConfig(min_targets=n, max_targets=n),
        visualization=VisualizationConfig(save_dir=exp.paths.plots),
        optimizer=OptimizerConfig(
            population_size=population_size or exp.convergence_population,
            n_generations=n_generations or max(exp.convergence_generations),
            crossover_prob=exp.crossover_prob,
            mutation_prob=exp.mutation_prob,
            max_mission_distance=exp.max_mission_distance,
            max_mission_targets=exp.max_mission_targets,
            damage_metric=exp.damage_metric,
            verbose=False,
        ),
        seed=exp.seed if seed is None else seed,
    )


def make_environment(
    exp: ExperimentConfig,
    *,
    n_targets: int | None = None,
    seed: int | None = None,
) -> tuple[WildfireEnvironment, MissionConfig]:
    """Create a reproducible synthetic environment for an experiment."""
    cfg = build_mission_config(exp, n_targets=n_targets, seed=seed)
    return WildfireEnvironment.create_synthetic(cfg), cfg


def run_timed_optimization(
    env: WildfireEnvironment,
    config: MissionConfig | OptimizerConfig,
    *,
    seed: int | None = None,
    callback: object | None = None,
) -> TimedOptimization:
    """Run NSGA-II and measure wall-clock runtime."""
    optimizer = NSGA2MissionOptimizer(env, config)
    t0 = time.perf_counter()
    result = optimizer.optimize(seed=seed, callback=callback)
    runtime_s = time.perf_counter() - t0
    return TimedOptimization(result=result, runtime_s=runtime_s)


def hypervolume_reference_point(config: OptimizerConfig) -> np.ndarray:
    """
    Fixed reference point for fair HV comparison across generations.

    Objectives are minimized as ``F = [-damage, travel, battery]``.
    The reference must be strictly dominated by feasible points (component-wise
    larger). Damage is at most ``max_mission_targets`` (scores in [0, 1]), so
    ``-damage`` is at least ``-max_mission_targets``; the anti-optimal damage
    pole used here is ``0.1`` (worse than any non-empty mission's ``-damage``).
    """
    travel_cap = config.max_mission_distance * 1.1
    battery_cap = travel_cap * max(config.battery_distance_factor, 1e-9)
    return np.array([0.1, travel_cap, battery_cap], dtype=float)


def compute_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume of the non-dominated set in ``F`` (minimization)."""
    if F.size == 0:
        return 0.0
    F = np.atleast_2d(np.asarray(F, dtype=float))
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    front = F[nds]
    # Discard points that do not dominate the reference (invalid for HV).
    valid = np.all(front < ref_point, axis=1)
    front = front[valid]
    if len(front) == 0:
        return 0.0
    return float(HV(ref_point=ref_point)(front))


def population_objective_stats(F: np.ndarray) -> dict[str, float]:
    """
    Best / average objective stats in human-readable (maximize-damage) form.

    ``F`` is the pymoo minimization matrix ``[-damage, travel, battery]``.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.size == 0:
        return {
            "best_damage": 0.0,
            "avg_damage": 0.0,
            "best_travel": 0.0,
            "avg_travel": 0.0,
            "best_battery": 0.0,
            "avg_battery": 0.0,
            "n_pareto": 0,
        }

    damage = -F[:, 0]
    travel = F[:, 1]
    battery = F[:, 2]
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return {
        "best_damage": float(np.max(damage)),
        "avg_damage": float(np.mean(damage)),
        "best_travel": float(np.min(travel)),
        "avg_travel": float(np.mean(travel)),
        "best_battery": float(np.min(battery)),
        "avg_battery": float(np.mean(battery)),
        "n_pareto": int(len(nds)),
    }


def result_objective_stats(result: OptimizationResult) -> dict[str, float]:
    """Summarize a final Pareto set in human-readable objective units."""
    if result.n_solutions == 0:
        return population_objective_stats(np.zeros((0, 3)))
    damages = [p.objectives.damage_prevented for p in result.plans]
    travels = [p.objectives.travel_distance for p in result.plans]
    batteries = [p.objectives.battery_usage for p in result.plans]
    return {
        "best_damage": float(max(damages)),
        "avg_damage": float(np.mean(damages)),
        "best_travel": float(min(travels)),
        "avg_travel": float(np.mean(travels)),
        "best_battery": float(min(batteries)),
        "avg_battery": float(np.mean(batteries)),
        "n_pareto": float(result.n_solutions),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """Write a list of dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        names = list(fieldnames or [])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=names)
            writer.writeheader()
        return path

    names = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})
    return path


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a matplotlib figure as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a consistent research-plot style."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.45)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)


def ensure_paths(paths: ExperimentPaths) -> ExperimentPaths:
    paths.ensure()
    return paths


def write_config_snapshot(exp: ExperimentConfig) -> Path:
    """Dump the exact ``ExperimentConfig`` used for a run, for reproducibility."""
    import dataclasses
    import json

    paths = ensure_paths(exp.paths)
    payload = dataclasses.asdict(exp)
    payload.pop("paths", None)
    payload["results_root"] = str(exp.paths.root)
    path = paths.config_snapshot
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
