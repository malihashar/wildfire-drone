"""
Validation Experiment 1 — NSGA-II scaling across problem size.

For each target count, run NSGA-II across multiple random seeds and record
runtime, generation budget used, Pareto-front size, best damage-prevention
proxy, best travel distance, and hypervolume. This validates how the
optimizer behaves as problem size grows, independent of any single lucky
(or unlucky) seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mission.experiments.common import (
    compute_hypervolume,
    ensure_paths,
    hypervolume_reference_point,
    make_environment,
    result_objective_stats,
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig


@dataclass(frozen=True)
class NSGA2ScalingResult:
    trial_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    csv_trials: Path
    csv_summary: Path
    plot_path: Path


def run_nsga2_scaling_experiment(
    exp: ExperimentConfig | None = None,
    target_counts: tuple[int, ...] = (10, 25, 50, 100, 200),
    n_seeds: int = 20,
) -> NSGA2ScalingResult:
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    trial_rows: list[dict[str, Any]] = []
    for n_targets in target_counts:
        for seed_offset in range(n_seeds):
            seed = exp.seed + seed_offset
            env, cfg = make_environment(exp, n_targets=n_targets, seed=seed)
            timed = run_timed_optimization(env, cfg, seed=seed)
            stats = result_objective_stats(timed.result)
            ref = hypervolume_reference_point(cfg.optimizer)
            hv = compute_hypervolume(timed.result.F, ref)

            trial_rows.append(
                {
                    "n_targets": n_targets,
                    "seed": seed,
                    "runtime_s": timed.runtime_s,
                    "generations": cfg.optimizer.n_generations,
                    "n_pareto": timed.result.n_solutions,
                    "best_damage": stats["best_damage"],
                    "best_travel": stats["best_travel"],
                    "hypervolume": hv,
                }
            )

    summary_rows = _summarize(trial_rows, target_counts)

    csv_trials = write_csv(paths.csv / "nsga2_scaling_trials.csv", trial_rows)
    csv_summary = write_csv(paths.csv / "nsga2_scaling_summary.csv", summary_rows)
    plot_path = _plot(summary_rows, paths.plots / "nsga2_scaling.png")

    return NSGA2ScalingResult(
        trial_rows=trial_rows,
        summary_rows=summary_rows,
        csv_trials=csv_trials,
        csv_summary=csv_summary,
        plot_path=plot_path,
    )


def _summarize(trial_rows: list[dict[str, Any]], target_counts: tuple[int, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for n_targets in target_counts:
        rows = [r for r in trial_rows if r["n_targets"] == n_targets]
        if not rows:
            continue
        summary.append(
            {
                "n_targets": n_targets,
                "n_trials": len(rows),
                "mean_runtime_s": float(np.mean([r["runtime_s"] for r in rows])),
                "std_runtime_s": float(np.std([r["runtime_s"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "mean_n_pareto": float(np.mean([r["n_pareto"] for r in rows])),
                "mean_best_damage": float(np.mean([r["best_damage"] for r in rows])),
                "mean_best_travel": float(np.mean([r["best_travel"] for r in rows])),
                "mean_hypervolume": float(np.mean([r["hypervolume"] for r in rows])),
                "std_hypervolume": float(np.std([r["hypervolume"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
        )
    return summary


def _plot(summary_rows: list[dict[str, Any]], path: Path):
    n_targets = [r["n_targets"] for r in summary_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)

    ax = axes[0][0]
    ax.errorbar(n_targets, [r["mean_runtime_s"] for r in summary_rows],
                yerr=[r["std_runtime_s"] for r in summary_rows], marker="o", color="#4a7c59", capsize=3)
    style_axes(ax, "Runtime vs Target Count", "Targets", "Runtime (s)")

    ax = axes[0][1]
    ax.plot(n_targets, [r["mean_n_pareto"] for r in summary_rows], marker="o", color="#8a4baf")
    style_axes(ax, "Pareto-Front Size vs Target Count", "Targets", "Mean # Pareto Solutions")

    ax = axes[1][0]
    ax.plot(n_targets, [r["mean_best_damage"] for r in summary_rows], marker="o", color="#c45c26", label="Best damage")
    ax2 = ax.twinx()
    ax2.plot(n_targets, [r["mean_best_travel"] for r in summary_rows], marker="^", color="#1f4e79", label="Best travel")
    style_axes(ax, "Best Objectives vs Target Count", "Targets", "Best Damage (proxy)")
    ax2.set_ylabel("Best Travel Distance")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax = axes[1][1]
    ax.errorbar(n_targets, [r["mean_hypervolume"] for r in summary_rows],
                yerr=[r["std_hypervolume"] for r in summary_rows], marker="o", color="#6b3fa0", capsize=3)
    style_axes(ax, "Hypervolume vs Target Count", "Targets", "Mean Hypervolume")

    fig.suptitle(f"NSGA-II Scaling ({summary_rows[0]['n_trials']} seeds per target count)")
    fig.tight_layout()
    return save_figure(fig, path)
