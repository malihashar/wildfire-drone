"""
Experiment 5 — Scenario diversity.

Runs NSGA-II across spatially and severity-wise distinct scenarios (not
just the fixed uniform-random scatter every other experiment uses) to check
whether the optimizer's behavior generalizes: a single uniformly-scattered
scenario is not evidence the optimizer works on realistic multi-front,
wind-driven fires.

Scenarios:
  - uniform          — original synthetic scatter (baseline)
  - clustered         — targets grouped into multiple independent fire fronts
  - clustered + wind  — clustered, plus a wind-driven severity bias
    (downwind targets get higher damage_score)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mission.config.settings import (
    GridConfig,
    MissionConfig,
    OptimizerConfig,
    TargetGenerationConfig,
    VisualizationConfig,
)
from mission.experiments.common import (
    ensure_paths,
    result_objective_stats,
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig
from mission.simulation.environment import WildfireEnvironment
from mission.visualization.plot_scene import plot_mission_scene

SCENARIOS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("uniform", {"spatial_mode": "uniform"}),
    ("clustered", {"spatial_mode": "clustered", "n_clusters": 3, "cluster_spread": 6.0}),
    (
        "clustered_wind",
        {
            "spatial_mode": "clustered",
            "n_clusters": 3,
            "cluster_spread": 6.0,
            "wind_direction_deg": 45.0,
            "wind_bias_strength": 0.6,
        },
    ),
)


@dataclass(frozen=True)
class ScenarioDiversityResult:
    rows: list[dict[str, Any]]
    csv_path: Path
    comparison_plot: Path
    scene_plots: dict[str, Path]


def run_scenario_diversity_experiment(
    exp: ExperimentConfig | None = None,
) -> ScenarioDiversityResult:
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    rows: list[dict[str, Any]] = []
    scene_plots: dict[str, Path] = {}

    for name, target_kwargs in SCENARIOS:
        mission_cfg = MissionConfig(
            grid=GridConfig(width=exp.grid_size, height=exp.grid_size),
            targets=TargetGenerationConfig(
                min_targets=exp.n_targets, max_targets=exp.n_targets, **target_kwargs
            ),
            visualization=VisualizationConfig(save_dir=paths.plots),
            optimizer=OptimizerConfig(
                population_size=exp.convergence_population,
                n_generations=max(exp.convergence_generations),
                crossover_prob=exp.crossover_prob,
                mutation_prob=exp.mutation_prob,
                max_mission_distance=exp.max_mission_distance,
                max_mission_targets=exp.max_mission_targets,
                damage_metric=exp.damage_metric,
                verbose=False,
            ),
            seed=exp.seed,
        )
        env = WildfireEnvironment.create_synthetic(mission_cfg)

        timed = run_timed_optimization(env, mission_cfg, seed=exp.seed)
        stats = result_objective_stats(timed.result)

        rows.append(
            {
                "scenario": name,
                "n_targets": env.n_targets,
                "runtime_s": timed.runtime_s,
                "n_pareto": timed.result.n_solutions,
                **stats,
            }
        )

        fig, _ = plot_mission_scene(
            env, viz_cfg=mission_cfg.visualization, title=f"Scenario: {name}"
        )
        scene_path = paths.plots / f"scenario_{name}.png"
        scene_plots[name] = save_figure(fig, scene_path)

    csv_path = write_csv(paths.csv / "scenario_diversity.csv", rows)
    comparison_plot = _plot_comparison(rows, paths.plots / "scenario_diversity_comparison.png")

    return ScenarioDiversityResult(
        rows=rows, csv_path=csv_path, comparison_plot=comparison_plot, scene_plots=scene_plots
    )


def _plot_comparison(rows: list[dict[str, Any]], path: Path):
    names = [r["scenario"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    axes[0].bar(names, [r["best_damage"] for r in rows], color="#c45c26")
    style_axes(axes[0], "Best Damage Prevented", "Scenario", "Damage")

    axes[1].bar(names, [r["best_travel"] for r in rows], color="#1f4e79")
    style_axes(axes[1], "Best Travel Distance", "Scenario", "Distance")

    axes[2].bar(names, [r["runtime_s"] * 1000 for r in rows], color="#4a7c59")
    style_axes(axes[2], "Optimization Runtime", "Scenario", "Runtime (ms)")

    fig.suptitle("NSGA-II Across Scenario Types (uniform vs. multi-front vs. wind-driven)")
    fig.tight_layout()
    return save_figure(fig, path)
