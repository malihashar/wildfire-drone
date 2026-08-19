"""
Validation Experiment 4 — online replanning policy comparison.

Compares three replanning policies, each run as its own live loop from the
same initial scenario (same seed), so their environment trajectories can
legitimately diverge based on what each policy actually decides to do —
that divergence is the point, not a bug:

  - "none"      — optimize once at tick 0, never re-run NSGA-II again.
  - "periodic"  — re-run NSGA-II every ``periodic_interval`` ticks,
    regardless of how much the prediction changed.
  - "triggered" — re-run NSGA-II only when the prediction update's diff
    exceeds ``trigger_threshold`` changed cells; otherwise keep flying the
    current mission.

Measures cumulative realized damage prevented (sum of suppressed targets'
damage_score at the moment each was suppressed), total D* Lite travel
distance, a battery proxy, replan count, and total optimization runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mission.config.settings import GridConfig, MissionConfig, OptimizerConfig, TargetGenerationConfig
from mission.experiments.common import ensure_paths, save_figure, style_axes, write_csv
from mission.experiments.experiment_config import ExperimentConfig
from mission.fitness.mission_selection import select_highest_scoring_mission
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from mission.replanning.executor import DStarLiteMissionExecutor, build_execution_request
from mission.replanning.online_replanner import _suppress_next_target
from mission.simulation.dynamics import SyntheticPredictionSource, apply_prediction_update
from mission.simulation.environment import WildfireEnvironment

POLICIES = ("none", "periodic", "triggered")


@dataclass(frozen=True)
class OnlineReplanningComparisonResult:
    trial_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    csv_trials: Path
    csv_summary: Path
    plot_path: Path


def run_online_replanning_comparison(
    exp: ExperimentConfig | None = None,
    n_trials: int = 12,
    n_targets: int = 15,
    grid_size: int = 40,
    n_ticks: int = 8,
    population_size: int = 40,
    n_generations: int = 60,
    periodic_interval: int = 3,
    trigger_threshold: int = 2,
) -> OnlineReplanningComparisonResult:
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    rows: list[dict[str, Any]] = []
    for trial in range(n_trials):
        seed = exp.seed + trial
        for policy in POLICIES:
            metrics = _run_policy(
                policy, seed, n_targets, grid_size, n_ticks,
                population_size, n_generations, periodic_interval, trigger_threshold,
            )
            metrics["trial"] = trial
            metrics["seed"] = seed
            rows.append(metrics)

    summary_rows = _summarize(rows)
    csv_trials = write_csv(paths.csv / "online_replanning_comparison_trials.csv", rows)
    csv_summary = write_csv(paths.csv / "online_replanning_comparison_summary.csv", summary_rows)
    plot_path = _plot(summary_rows, paths.plots / "online_replanning_comparison.png")

    return OnlineReplanningComparisonResult(
        trial_rows=rows, summary_rows=summary_rows,
        csv_trials=csv_trials, csv_summary=csv_summary, plot_path=plot_path,
    )


def _run_policy(
    policy: str,
    seed: int,
    n_targets: int,
    grid_size: int,
    n_ticks: int,
    population_size: int,
    n_generations: int,
    periodic_interval: int,
    trigger_threshold: int,
) -> dict[str, Any]:
    opt_cfg = OptimizerConfig(
        population_size=population_size, n_generations=n_generations,
        max_mission_targets=6, verbose=False,
    )
    mission_cfg = MissionConfig(
        grid=GridConfig(width=grid_size, height=grid_size),
        targets=TargetGenerationConfig(min_targets=n_targets, max_targets=n_targets),
        optimizer=opt_cfg,
        seed=seed,
    )
    env = WildfireEnvironment.create_synthetic(mission_cfg)
    prediction_source = SyntheticPredictionSource(seed=seed + 7)
    executor = DStarLiteMissionExecutor()

    def optimize(call_seed: int):
        m_cfg = MissionConfig(optimizer=opt_cfg, seed=call_seed)
        optimizer = NSGA2MissionOptimizer(env, m_cfg)
        t0 = time.perf_counter()
        result = optimizer.optimize(seed=call_seed)
        return select_highest_scoring_mission(result), time.perf_counter() - t0

    current_mission, runtime0 = optimize(seed)
    total_runtime = runtime0
    n_replans = 1
    total_damage = 0.0
    total_travel = 0.0

    request0 = build_execution_request(env, current_mission, tick=0)
    result0 = executor.execute(request0)
    if result0 is not None and result0.feasible:
        total_travel += result0.path_length

    for tick in range(1, n_ticks + 1):
        id_map = {t.id: t for t in env.targets}
        next_id = next((tid for tid in current_mission.target_ids if tid in id_map), None)
        if next_id is not None:
            total_damage += id_map[next_id].damage_score
        _suppress_next_target(env, current_mission)

        update = prediction_source.next_update(env, tick=tick)
        if update is None:
            break
        diff = apply_prediction_update(env, update)

        n_changed = (
            len(diff.added_ids) + len(diff.removed_ids)
            + len(diff.priority_changed_ids) + len(diff.damage_changed_ids)
        )
        should_replan = {
            "none": False,
            "periodic": tick % periodic_interval == 0,
            "triggered": n_changed >= trigger_threshold,
        }[policy]

        if should_replan:
            current_mission, rt = optimize(seed + tick)
            total_runtime += rt
            n_replans += 1

        request = build_execution_request(env, current_mission, tick=tick)
        result = executor.execute(request)
        if result is not None and result.feasible:
            total_travel += result.path_length

    return {
        "policy": policy,
        "total_damage_prevented": total_damage,
        "total_travel_distance": total_travel,
        "total_battery_proxy": total_travel * opt_cfg.battery_distance_factor,
        "n_replans": n_replans,
        "total_runtime_s": total_runtime,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        summary.append(
            {
                "policy": policy,
                "n_trials": len(subset),
                "mean_damage_prevented": float(np.mean([r["total_damage_prevented"] for r in subset])),
                "mean_travel_distance": float(np.mean([r["total_travel_distance"] for r in subset])),
                "mean_battery_proxy": float(np.mean([r["total_battery_proxy"] for r in subset])),
                "mean_n_replans": float(np.mean([r["n_replans"] for r in subset])),
                "mean_runtime_s": float(np.mean([r["total_runtime_s"] for r in subset])),
            }
        )
    return summary


def _plot(summary_rows: list[dict[str, Any]], path: Path):
    policies = [r["policy"] for r in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    axes[0].bar(policies, [r["mean_damage_prevented"] for r in summary_rows], color="#c45c26")
    style_axes(axes[0], "Cumulative Damage Prevented", "Policy", "Damage")

    axes[1].bar(policies, [r["mean_travel_distance"] for r in summary_rows], color="#1f4e79")
    style_axes(axes[1], "Total Travel Distance (D* Lite actual)", "Policy", "Distance")

    ax = axes[2]
    ax.bar(policies, [r["mean_runtime_s"] for r in summary_rows], color="#4a7c59", alpha=0.8, label="Runtime (s)")
    ax2 = ax.twinx()
    ax2.plot(policies, [r["mean_n_replans"] for r in summary_rows], marker="o", color="#8a4baf", label="# Replans")
    style_axes(ax, "Optimization Cost", "Policy", "Total Runtime (s)")
    ax2.set_ylabel("Mean # Replans")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle(f"Online Replanning Policy Comparison ({summary_rows[0]['n_trials']} trials/policy)")
    fig.tight_layout()
    return save_figure(fig, path)
