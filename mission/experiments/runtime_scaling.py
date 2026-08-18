"""
Experiment 2 — Runtime scaling with problem size.

Generates environments with increasing numbers of suppression targets and
measures NSGA-II wall-clock runtime over multiple independent trials.
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


@dataclass(frozen=True)
class RuntimeScalingResult:
    trial_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    csv_trials: Path
    csv_summary: Path
    plot_runtime: Path


def run_runtime_scaling_experiment(
    exp: ExperimentConfig | None = None,
) -> RuntimeScalingResult:
    """Benchmark optimization runtime versus number of targets."""
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    trial_rows: list[dict[str, Any]] = []

    for n_targets in exp.runtime_target_counts:
        for trial in range(exp.runtime_trials):
            seed = exp.seed + 1000 * n_targets + trial
            env, cfg = make_environment(exp, n_targets=n_targets, seed=seed)
            opt = replace(
                cfg.optimizer,
                population_size=exp.runtime_population,
                n_generations=exp.runtime_generations,
            )
            cfg = replace(cfg, optimizer=opt)
            timed = run_timed_optimization(env, cfg, seed=seed)

            trial_rows.append(
                {
                    "n_targets": n_targets,
                    "trial": trial,
                    "seed": seed,
                    "population_size": exp.runtime_population,
                    "n_generations": exp.runtime_generations,
                    "runtime_s": timed.runtime_s,
                    "n_pareto": timed.result.n_solutions,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for n_targets in exp.runtime_target_counts:
        runtimes = [r["runtime_s"] for r in trial_rows if r["n_targets"] == n_targets]
        mean_rt, std_rt = mean_std(runtimes)
        summary_rows.append(
            {
                "n_targets": n_targets,
                "n_trials": len(runtimes),
                "mean_runtime_s": mean_rt,
                "std_runtime_s": std_rt,
                "min_runtime_s": float(min(runtimes)) if runtimes else 0.0,
                "max_runtime_s": float(max(runtimes)) if runtimes else 0.0,
            }
        )

    csv_trials = write_csv(paths.csv / "runtime_scaling_trials.csv", trial_rows)
    csv_summary = write_csv(paths.csv / "runtime_scaling_summary.csv", summary_rows)
    plot_runtime = _plot_runtime(summary_rows, paths.plots / "runtime_vs_targets.png")

    return RuntimeScalingResult(
        trial_rows=trial_rows,
        summary_rows=summary_rows,
        csv_trials=csv_trials,
        csv_summary=csv_summary,
        plot_runtime=plot_runtime,
    )


def _plot_runtime(summary_rows: list[dict[str, Any]], path: Path) -> Path:
    xs = [r["n_targets"] for r in summary_rows]
    ys = [r["mean_runtime_s"] for r in summary_rows]
    yerr = [r["std_runtime_s"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="-o",
        color="#1f4e79",
        ecolor="#7a9bb8",
        capsize=4,
        lw=2,
        markersize=7,
        label="Mean ± std",
    )
    style_axes(ax, "NSGA-II Runtime vs Number of Targets", "Number of Targets", "Runtime (s)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return save_figure(fig, path)
