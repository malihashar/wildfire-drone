"""
Experiment 5 — Generations-to-convergence-threshold vs number of targets.

For each problem size, runs NSGA-II while recording best damage-prevented
fitness per generation, then finds the earliest generation at which fitness
reaches 95%, 98%, and 99% of that run's final best value (reusing the
threshold logic from ``fitness_runtime``). Answers: does the optimizer need
more generations to converge as the problem grows?
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
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig
from mission.experiments.fitness_runtime import RuntimeFitnessCallback, find_threshold_hits


@dataclass(frozen=True)
class ConvergenceScalingResult:
    trial_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    csv_trials: Path
    csv_summary: Path
    plot_generations: Path


def run_convergence_scaling_experiment(
    exp: ExperimentConfig | None = None,
) -> ConvergenceScalingResult:
    """Measure generations-to-threshold as a function of problem size."""
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    trial_rows: list[dict[str, Any]] = []

    for n_targets in exp.threshold_target_counts:
        for trial in range(exp.threshold_trials):
            seed = exp.seed + 2000 * n_targets + trial
            env, cfg = make_environment(exp, n_targets=n_targets, seed=seed)
            opt = replace(
                cfg.optimizer,
                population_size=exp.threshold_population,
                n_generations=exp.threshold_generations,
            )
            cfg = replace(cfg, optimizer=opt)

            callback = RuntimeFitnessCallback()
            timed = run_timed_optimization(env, cfg, seed=seed, callback=callback)
            hits = find_threshold_hits(callback.history)
            by_frac = {h.fraction: h for h in hits}

            trial_rows.append(
                {
                    "n_targets": n_targets,
                    "trial": trial,
                    "seed": seed,
                    "population_size": exp.threshold_population,
                    "max_generations": exp.threshold_generations,
                    "final_best_fitness": callback.history.final_best,
                    "runtime_s": timed.runtime_s,
                    "gen_95": by_frac[0.95].generation if by_frac[0.95].reached else "",
                    "gen_98": by_frac[0.98].generation if by_frac[0.98].reached else "",
                    "gen_99": by_frac[0.99].generation if by_frac[0.99].reached else "",
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for n_targets in exp.threshold_target_counts:
        rows = [r for r in trial_rows if r["n_targets"] == n_targets]
        row: dict[str, Any] = {"n_targets": n_targets, "n_trials": len(rows)}
        for key in ("gen_95", "gen_98", "gen_99"):
            vals = [r[key] for r in rows if r[key] != ""]
            mean_g, std_g = mean_std(vals)
            row[f"mean_{key}"] = mean_g
            row[f"std_{key}"] = std_g
            row[f"n_reached_{key}"] = len(vals)
        summary_rows.append(row)

    csv_trials = write_csv(paths.csv / "convergence_scaling_trials.csv", trial_rows)
    csv_summary = write_csv(paths.csv / "convergence_scaling_summary.csv", summary_rows)
    plot_generations = _plot_generations(
        summary_rows, paths.plots / "generations_to_threshold_vs_targets.png"
    )

    return ConvergenceScalingResult(
        trial_rows=trial_rows,
        summary_rows=summary_rows,
        csv_trials=csv_trials,
        csv_summary=csv_summary,
        plot_generations=plot_generations,
    )


def _plot_generations(summary_rows: list[dict[str, Any]], path: Path) -> Path:
    xs = [r["n_targets"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    colors = {"gen_95": "#2e7d32", "gen_98": "#c45c26", "gen_99": "#6b3fa0"}
    labels = {"gen_95": "95% of final", "gen_98": "98% of final", "gen_99": "99% of final"}
    for key in ("gen_95", "gen_98", "gen_99"):
        ys = [r[f"mean_{key}"] for r in summary_rows]
        yerr = [r[f"std_{key}"] for r in summary_rows]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="-o",
            color=colors[key],
            ecolor=colors[key],
            alpha=0.9,
            capsize=4,
            lw=2,
            markersize=6,
            label=labels[key],
        )
    style_axes(
        ax,
        "Generations to Reach Convergence Threshold vs Number of Targets",
        "Number of Targets",
        "Generation (mean ± std)",
    )
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return save_figure(fig, path)
