"""
Experiment 1 — NSGA-II convergence analysis.

Runs one optimization to the maximum generation budget while a pymoo callback
records best/average objectives, hypervolume, and Pareto-front size each
generation. Checkpoint generations (25, 50, 100, 200, 500) are extracted from
the same history for tabular reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.callback import Callback

from mission.experiments.common import (
    compute_hypervolume,
    ensure_paths,
    hypervolume_reference_point,
    make_environment,
    population_objective_stats,
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig


@dataclass
class ConvergenceHistory:
    """Per-generation optimization metrics."""

    generations: list[int] = field(default_factory=list)
    best_damage: list[float] = field(default_factory=list)
    avg_damage: list[float] = field(default_factory=list)
    best_travel: list[float] = field(default_factory=list)
    avg_travel: list[float] = field(default_factory=list)
    best_battery: list[float] = field(default_factory=list)
    avg_battery: list[float] = field(default_factory=list)
    hypervolume: list[float] = field(default_factory=list)
    n_pareto: list[int] = field(default_factory=list)

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, gen in enumerate(self.generations):
            rows.append(
                {
                    "generation": gen,
                    "best_damage": self.best_damage[i],
                    "avg_damage": self.avg_damage[i],
                    "best_travel": self.best_travel[i],
                    "avg_travel": self.avg_travel[i],
                    "best_battery": self.best_battery[i],
                    "avg_battery": self.avg_battery[i],
                    "hypervolume": self.hypervolume[i],
                    "n_pareto": self.n_pareto[i],
                }
            )
        return rows

    def checkpoint_rows(self, checkpoints: tuple[int, ...]) -> list[dict[str, Any]]:
        by_gen = {row["generation"]: row for row in self.as_rows()}
        out: list[dict[str, Any]] = []
        for g in checkpoints:
            if g in by_gen:
                out.append(by_gen[g])
            elif self.generations:
                # Nearest available generation if callback indexing differs.
                nearest = min(self.generations, key=lambda x: abs(x - g))
                row = dict(by_gen[nearest])
                row["requested_generation"] = g
                out.append(row)
        return out


class ConvergenceCallback(Callback):
    """Record population quality metrics after every generation."""

    def __init__(self, ref_point: np.ndarray) -> None:
        super().__init__()
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.history = ConvergenceHistory()

    def notify(self, algorithm: object) -> None:
        pop = algorithm.pop  # type: ignore[attr-defined]
        F = np.atleast_2d(np.asarray(pop.get("F"), dtype=float))
        stats = population_objective_stats(F)
        hv = compute_hypervolume(F, self.ref_point)
        gen = int(algorithm.n_gen)  # type: ignore[attr-defined]

        self.history.generations.append(gen)
        self.history.best_damage.append(stats["best_damage"])
        self.history.avg_damage.append(stats["avg_damage"])
        self.history.best_travel.append(stats["best_travel"])
        self.history.avg_travel.append(stats["avg_travel"])
        self.history.best_battery.append(stats["best_battery"])
        self.history.avg_battery.append(stats["avg_battery"])
        self.history.hypervolume.append(hv)
        self.history.n_pareto.append(int(stats["n_pareto"]))


@dataclass(frozen=True)
class ConvergenceExperimentResult:
    history: ConvergenceHistory
    runtime_s: float
    n_targets: int
    population_size: int
    max_generations: int
    csv_history: Path
    csv_checkpoints: Path
    plot_best: Path
    plot_avg: Path
    plot_hv: Path
    plot_pareto_size: Path


def run_convergence_experiment(
    exp: ExperimentConfig | None = None,
) -> ConvergenceExperimentResult:
    """
    Evaluate NSGA-II convergence on a fixed synthetic environment.

    A single run proceeds to ``max(convergence_generations)``; metrics at
    25 / 50 / 100 / 200 / 500 generations are taken from the recorded history.
    """
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    max_gen = max(exp.convergence_generations)
    env, cfg = make_environment(exp, n_targets=exp.n_targets, seed=exp.seed)
    # Override optimizer knobs for this experiment.
    from dataclasses import replace

    opt = replace(
        cfg.optimizer,
        population_size=exp.convergence_population,
        n_generations=max_gen,
    )
    cfg = replace(cfg, optimizer=opt)

    ref = hypervolume_reference_point(cfg.optimizer)
    callback = ConvergenceCallback(ref)
    timed = run_timed_optimization(env, cfg, seed=exp.seed, callback=callback)
    history = callback.history

    csv_history = write_csv(paths.csv / "convergence_history.csv", history.as_rows())
    csv_checkpoints = write_csv(
        paths.csv / "convergence_checkpoints.csv",
        history.checkpoint_rows(exp.convergence_generations),
    )

    plot_best = _plot_best_fitness(history, paths.plots / "convergence_best_fitness.png")
    plot_avg = _plot_avg_fitness(history, paths.plots / "convergence_avg_fitness.png")
    plot_hv = _plot_hypervolume(history, paths.plots / "convergence_hypervolume.png")
    plot_pareto = _plot_pareto_size(
        history, paths.plots / "convergence_pareto_size.png"
    )

    return ConvergenceExperimentResult(
        history=history,
        runtime_s=timed.runtime_s,
        n_targets=env.n_targets,
        population_size=exp.convergence_population,
        max_generations=max_gen,
        csv_history=csv_history,
        csv_checkpoints=csv_checkpoints,
        plot_best=plot_best,
        plot_avg=plot_avg,
        plot_hv=plot_hv,
        plot_pareto_size=plot_pareto,
    )


def _plot_best_fitness(history: ConvergenceHistory, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].plot(history.generations, history.best_damage, color="#b85c38", lw=2)
    style_axes(axes[0], "Best Damage Prevented vs Generation", "Generation", "Best Damage")

    axes[1].plot(history.generations, history.best_travel, color="#1f4e79", lw=2, label="Travel")
    axes[1].plot(
        history.generations,
        history.best_battery,
        color="#5a7d4e",
        lw=2,
        linestyle="--",
        label="Battery",
    )
    style_axes(axes[1], "Best Cost Objectives vs Generation", "Generation", "Best Value")
    axes[1].legend(framealpha=0.9)

    fig.suptitle("NSGA-II Convergence — Best Fitness", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, path)


def _plot_avg_fitness(history: ConvergenceHistory, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].plot(history.generations, history.avg_damage, color="#b85c38", lw=2)
    style_axes(axes[0], "Average Damage Prevented vs Generation", "Generation", "Avg Damage")

    axes[1].plot(history.generations, history.avg_travel, color="#1f4e79", lw=2, label="Travel")
    axes[1].plot(
        history.generations,
        history.avg_battery,
        color="#5a7d4e",
        lw=2,
        linestyle="--",
        label="Battery",
    )
    style_axes(axes[1], "Average Cost Objectives vs Generation", "Generation", "Avg Value")
    axes[1].legend(framealpha=0.9)

    fig.suptitle("NSGA-II Convergence — Average Fitness", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, path)


def _plot_hypervolume(history: ConvergenceHistory, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(history.generations, history.hypervolume, color="#6b3fa0", lw=2)
    style_axes(ax, "Hypervolume vs Generation", "Generation", "Hypervolume")
    fig.tight_layout()
    return save_figure(fig, path)


def _plot_pareto_size(history: ConvergenceHistory, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(history.generations, history.n_pareto, color="#c45c26", lw=2)
    style_axes(ax, "Pareto Front Size vs Generation", "Generation", "# Non-dominated Solutions")
    fig.tight_layout()
    return save_figure(fig, path)
