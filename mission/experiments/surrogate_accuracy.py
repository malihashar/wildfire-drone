"""
Validation Experiment 2 — surrogate (NSGA-II Euclidean) vs. D* Lite actual
route accuracy.

For many independently generated missions, select the best-scoring mission,
execute it via D* Lite, and record both the surrogate distance NSGA-II used
internally and D* Lite's real obstacle-aware distance. Validates the
decision (documented in mission.fitness.objectives) to NOT run D* Lite
inside every NSGA-II evaluation: if the surrogate tracks the actual route
cost closely, that decision is justified; if not, it needs revisiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mission.experiments.common import ensure_paths, make_environment, write_csv, save_figure, style_axes
from mission.experiments.experiment_config import ExperimentConfig
from mission.fitness.mission_selection import select_highest_scoring_mission
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from mission.replanning.executor import DStarLiteMissionExecutor, build_execution_request


@dataclass(frozen=True)
class SurrogateAccuracyResult:
    trial_rows: list[dict[str, Any]]
    stats: dict[str, float]
    csv_path: Path
    plot_path: Path


def run_surrogate_accuracy_experiment(
    exp: ExperimentConfig | None = None,
    n_trials: int = 60,
    target_counts: tuple[int, ...] = (10, 20, 30, 50),
) -> SurrogateAccuracyResult:
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    rows: list[dict[str, Any]] = []
    trial = 0
    for n_targets in target_counts:
        per_count = max(1, n_trials // len(target_counts))
        for i in range(per_count):
            seed = exp.seed + trial
            trial += 1
            env, cfg = make_environment(exp, n_targets=n_targets, seed=seed)
            optimizer = NSGA2MissionOptimizer(env, cfg)
            result = optimizer.optimize(seed=seed)
            if result.n_solutions == 0:
                continue
            scored = select_highest_scoring_mission(result)
            request = build_execution_request(env, scored, tick=0)
            executor = DStarLiteMissionExecutor()
            exec_result = executor.execute(request)
            if exec_result is None or not exec_result.feasible:
                continue

            rows.append(
                {
                    "trial": trial,
                    "n_targets": n_targets,
                    "seed": seed,
                    "surrogate_distance": exec_result.straight_line_length,
                    "actual_distance": exec_result.path_length,
                    "deviation_ratio": exec_result.deviation_ratio,
                }
            )

    ratios = np.array([r["deviation_ratio"] for r in rows], dtype=float)
    stats = {
        "n_trials": len(rows),
        "mean_deviation_ratio": float(np.mean(ratios)),
        "median_deviation_ratio": float(np.median(ratios)),
        "p95_deviation_ratio": float(np.percentile(ratios, 95)),
        "min_deviation_ratio": float(np.min(ratios)),
        "max_deviation_ratio": float(np.max(ratios)),
    }

    csv_path = write_csv(paths.csv / "surrogate_accuracy_trials.csv", rows)
    plot_path = _plot(rows, stats, paths.plots / "surrogate_accuracy.png")

    return SurrogateAccuracyResult(trial_rows=rows, stats=stats, csv_path=csv_path, plot_path=plot_path)


def _plot(rows: list[dict[str, Any]], stats: dict[str, float], path: Path):
    ratios = [r["deviation_ratio"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].hist(ratios, bins=20, color="#4a7c59", edgecolor="black", alpha=0.85)
    axes[0].axvline(stats["mean_deviation_ratio"], color="#c45c26", linestyle="--", label=f"mean={stats['mean_deviation_ratio']:.3f}")
    axes[0].axvline(stats["p95_deviation_ratio"], color="#8a4baf", linestyle=":", label=f"p95={stats['p95_deviation_ratio']:.3f}")
    style_axes(axes[0], "Deviation Ratio Distribution\n(D* Lite actual / NSGA-II surrogate)", "Deviation ratio", "Count")
    axes[0].legend(fontsize=8)

    surrogate = [r["surrogate_distance"] for r in rows]
    actual = [r["actual_distance"] for r in rows]
    axes[1].scatter(surrogate, actual, alpha=0.6, color="#1f4e79", edgecolors="black", linewidths=0.4)
    lims = [0, max(max(surrogate), max(actual)) * 1.05]
    axes[1].plot(lims, lims, color="#c62828", linestyle="--", label="y = x (perfect surrogate)")
    style_axes(axes[1], "Surrogate vs Actual Distance", "NSGA-II surrogate (Euclidean)", "D* Lite actual")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Surrogate Accuracy — n={stats['n_trials']} missions across varied problem sizes")
    fig.tight_layout()
    return save_figure(fig, path)
