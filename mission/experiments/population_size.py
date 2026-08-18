"""
Experiment 3 — Effect of NSGA-II population size.

Sweeps population size while holding the environment and generation budget
fixed (multiple trials per setting).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mission.experiments.common import (
    ensure_paths,
    make_environment,
    mean_std,
    result_objective_stats,
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig


@dataclass(frozen=True)
class PopulationSizeResult:
    trial_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    csv_trials: Path
    csv_summary: Path
    plot_runtime: Path
    plot_objectives: Path
    plot_pareto_size: Path


def run_population_size_experiment(
    exp: ExperimentConfig | None = None,
) -> PopulationSizeResult:
    """Evaluate how population size affects runtime and solution quality."""
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    # Fixed environment across all population sizes / trials for fair comparison
    # of search effort; trial seeds only vary the optimizer RNG.
    base_env, base_cfg = make_environment(exp, n_targets=exp.n_targets, seed=exp.seed)

    trial_rows: list[dict[str, Any]] = []
    for pop_size in exp.population_sizes:
        for trial in range(exp.population_trials):
            seed = exp.seed + 17 * pop_size + trial
            opt = replace(
                base_cfg.optimizer,
                population_size=pop_size,
                n_generations=exp.population_generations,
            )
            cfg = replace(base_cfg, optimizer=opt, seed=seed)
            timed = run_timed_optimization(base_env, cfg, seed=seed)
            stats = result_objective_stats(timed.result)

            trial_rows.append(
                {
                    "population_size": pop_size,
                    "trial": trial,
                    "seed": seed,
                    "n_targets": base_env.n_targets,
                    "n_generations": exp.population_generations,
                    "runtime_s": timed.runtime_s,
                    "n_pareto": timed.result.n_solutions,
                    "best_damage": stats["best_damage"],
                    "best_travel": stats["best_travel"],
                    "best_battery": stats["best_battery"],
                    "avg_damage": stats["avg_damage"],
                    "avg_travel": stats["avg_travel"],
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for pop_size in exp.population_sizes:
        subset = [r for r in trial_rows if r["population_size"] == pop_size]
        rt_mean, rt_std = mean_std(r["runtime_s"] for r in subset)
        dmg_mean, dmg_std = mean_std(r["best_damage"] for r in subset)
        travel_mean, travel_std = mean_std(r["best_travel"] for r in subset)
        pareto_mean, pareto_std = mean_std(r["n_pareto"] for r in subset)
        summary_rows.append(
            {
                "population_size": pop_size,
                "n_trials": len(subset),
                "mean_runtime_s": rt_mean,
                "std_runtime_s": rt_std,
                "mean_best_damage": dmg_mean,
                "std_best_damage": dmg_std,
                "mean_best_travel": travel_mean,
                "std_best_travel": travel_std,
                "mean_n_pareto": pareto_mean,
                "std_n_pareto": pareto_std,
            }
        )

    csv_trials = write_csv(paths.csv / "population_size_trials.csv", trial_rows)
    csv_summary = write_csv(paths.csv / "population_size_summary.csv", summary_rows)

    plot_runtime = _plot_pop_runtime(
        summary_rows, paths.plots / "population_runtime.png"
    )
    plot_objectives = _plot_pop_objectives(
        summary_rows, paths.plots / "population_best_objectives.png"
    )
    plot_pareto = _plot_pop_pareto(
        summary_rows, paths.plots / "population_pareto_size.png"
    )

    return PopulationSizeResult(
        trial_rows=trial_rows,
        summary_rows=summary_rows,
        csv_trials=csv_trials,
        csv_summary=csv_summary,
        plot_runtime=plot_runtime,
        plot_objectives=plot_objectives,
        plot_pareto_size=plot_pareto,
    )


def _plot_pop_runtime(summary_rows: list[dict[str, Any]], path: Path) -> Path:
    xs = [r["population_size"] for r in summary_rows]
    ys = [r["mean_runtime_s"] for r in summary_rows]
    yerr = [r["std_runtime_s"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color="#1f4e79", ecolor="#7a9bb8", capsize=4, lw=2)
    style_axes(ax, "Runtime vs Population Size", "Population Size", "Runtime (s)")
    fig.tight_layout()
    return save_figure(fig, path)


def _plot_pop_objectives(summary_rows: list[dict[str, Any]], path: Path) -> Path:
    xs = [r["population_size"] for r in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].errorbar(
        xs,
        [r["mean_best_damage"] for r in summary_rows],
        yerr=[r["std_best_damage"] for r in summary_rows],
        fmt="-o",
        color="#b85c38",
        ecolor="#d4a090",
        capsize=4,
        lw=2,
    )
    style_axes(axes[0], "Best Damage vs Population Size", "Population Size", "Best Damage")

    axes[1].errorbar(
        xs,
        [r["mean_best_travel"] for r in summary_rows],
        yerr=[r["std_best_travel"] for r in summary_rows],
        fmt="-o",
        color="#1f4e79",
        ecolor="#7a9bb8",
        capsize=4,
        lw=2,
    )
    style_axes(axes[1], "Best Travel vs Population Size", "Population Size", "Best Travel Distance")

    fig.suptitle("Effect of Population Size on Best Objectives", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, path)


def _plot_pop_pareto(summary_rows: list[dict[str, Any]], path: Path) -> Path:
    xs = [r["population_size"] for r in summary_rows]
    ys = [r["mean_n_pareto"] for r in summary_rows]
    yerr = [r["std_n_pareto"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color="#c45c26", ecolor="#e0a888", capsize=4, lw=2)
    style_axes(ax, "Pareto Set Size vs Population Size", "Population Size", "Mean # Pareto Solutions")
    # Keep y-axis non-negative for counts.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=max(ymax, 1.0))
    fig.tight_layout()
    return save_figure(fig, path)
