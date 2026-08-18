"""
NSGA-II convergence, collected from the *online* in-mission replanning loop.

``mission.experiments.convergence`` runs one static NSGA-II optimization on a
fixed initial scene. This module instead drives
``mission.replanning.online_replanner.OnlineReplanner``: NSGA-II is re-run
from scratch for the initial mission and again after every suppression +
synthetic wildfire-prediction-update tick, exactly as the online replanning
demo does. A ``ConvergenceCallback`` (reused from ``mission.experiments.
convergence``) is attached to each of those NSGA-II runs, and the resulting
per-generation histories are averaged pointwise across all runs (initial +
every replan event) to produce a single "Average Damage / Cost vs
Generation" curve in the same visual style as the static convergence plot —
but built entirely from the new online-replanning code path, not from the
old experiment's data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mission.config.settings import OptimizerConfig
from mission.experiments.common import hypervolume_reference_point, save_figure, style_axes
from mission.experiments.convergence import ConvergenceCallback, ConvergenceHistory
from mission.experiments.experiment_config import ExperimentPaths
from mission.replanning.config import OnlineReplanConfig
from mission.replanning.online_replanner import OnlineReplanner, OnlineReplanResult

RAW_CSV_FIELDS = [
    "tick",
    "generation",
    "best_damage",
    "avg_damage",
    "best_travel",
    "avg_travel",
    "best_battery",
    "avg_battery",
    "hypervolume",
    "n_pareto",
]
AGGREGATED_CSV_FIELDS = [
    "generation",
    "best_damage",
    "avg_damage",
    "best_travel",
    "avg_travel",
    "best_battery",
    "avg_battery",
    "hypervolume",
    "n_pareto",
]


@dataclass(frozen=True)
class OnlineConvergenceResult:
    """Per-generation histories from every online-replanning NSGA-II run."""

    online_result: OnlineReplanResult
    tick_histories: list[ConvergenceHistory]
    aggregated: ConvergenceHistory
    csv_raw: Path
    csv_aggregated: Path
    plot_path: Path

    @property
    def n_runs(self) -> int:
        return len(self.tick_histories)


def run_online_convergence_experiment(
    config: OnlineReplanConfig | None = None,
    results_paths: ExperimentPaths | None = None,
) -> OnlineConvergenceResult:
    """Run the online replanning loop with a convergence callback on every tick."""
    cfg = config or OnlineReplanConfig()
    paths = results_paths or ExperimentPaths()
    paths.ensure()

    opt_cfg = OptimizerConfig(
        population_size=cfg.population_size,
        n_generations=cfg.n_generations,
        max_mission_distance=cfg.max_mission_distance,
        max_mission_targets=cfg.max_mission_targets,
        damage_metric=cfg.damage_metric,
        verbose=False,
    )
    ref = hypervolume_reference_point(opt_cfg)

    replanner = OnlineReplanner(config=cfg, callback_factory=lambda: ConvergenceCallback(ref))
    online_result = replanner.run()
    histories: list[ConvergenceHistory] = replanner.tick_histories

    aggregated = _aggregate(histories)

    csv_raw = _write_raw_csv(histories, paths.csv / "online_convergence_raw.csv")
    csv_aggregated = _write_aggregated_csv(
        aggregated, paths.csv / "online_convergence_aggregated.csv"
    )
    plot_path = _plot_avg_fitness_online(
        aggregated,
        n_runs=len(histories),
        n_events=online_result.n_replan_events,
        path=paths.plots / "nsga_convergence_online_replanning.png",
    )

    return OnlineConvergenceResult(
        online_result=online_result,
        tick_histories=histories,
        aggregated=aggregated,
        csv_raw=csv_raw,
        csv_aggregated=csv_aggregated,
        plot_path=plot_path,
    )


def _aggregate(histories: list[ConvergenceHistory]) -> ConvergenceHistory:
    """Pointwise mean across every online NSGA-II run's per-generation history."""
    if not histories:
        return ConvergenceHistory()

    n_gen = min(len(h.generations) for h in histories)
    agg = ConvergenceHistory()
    agg.generations = list(histories[0].generations[:n_gen])
    for i in range(n_gen):
        agg.best_damage.append(float(np.mean([h.best_damage[i] for h in histories])))
        agg.avg_damage.append(float(np.mean([h.avg_damage[i] for h in histories])))
        agg.best_travel.append(float(np.mean([h.best_travel[i] for h in histories])))
        agg.avg_travel.append(float(np.mean([h.avg_travel[i] for h in histories])))
        agg.best_battery.append(float(np.mean([h.best_battery[i] for h in histories])))
        agg.avg_battery.append(float(np.mean([h.avg_battery[i] for h in histories])))
        agg.hypervolume.append(float(np.mean([h.hypervolume[i] for h in histories])))
        agg.n_pareto.append(int(round(float(np.mean([h.n_pareto[i] for h in histories])))))
    return agg


def _write_raw_csv(histories: list[ConvergenceHistory], path: Path) -> Path:
    """One row per (tick, generation): tick 0 = initial mission, tick i = replan event i."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for tick, history in enumerate(histories):
            for i, gen in enumerate(history.generations):
                writer.writerow(
                    {
                        "tick": tick,
                        "generation": gen,
                        "best_damage": history.best_damage[i],
                        "avg_damage": history.avg_damage[i],
                        "best_travel": history.best_travel[i],
                        "avg_travel": history.avg_travel[i],
                        "best_battery": history.best_battery[i],
                        "avg_battery": history.avg_battery[i],
                        "hypervolume": history.hypervolume[i],
                        "n_pareto": history.n_pareto[i],
                    }
                )
    return path


def _write_aggregated_csv(history: ConvergenceHistory, path: Path) -> Path:
    """The exact per-generation series used to draw the summary plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGGREGATED_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history.as_rows())
    return path


def _plot_avg_fitness_online(
    history: ConvergenceHistory,
    n_runs: int,
    n_events: int,
    path: Path,
) -> Path:
    """Same 2-panel layout/styling as the static convergence plot's avg-fitness figure."""
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

    fig.suptitle(
        f"NSGA-II Convergence — Online Replanning\n"
        f"(mean over {n_runs} in-mission NSGA-II runs: 1 initial + {n_events} replan events; "
        f"each run optimizes only the remaining, unsuppressed targets)",
        fontsize=10.5,
    )
    fig.tight_layout()
    return save_figure(fig, path)
