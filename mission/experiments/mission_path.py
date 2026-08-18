"""
Experiment 4 — Visualize one Pareto-optimal mission path.

Selects a representative plan (knee / balanced damage–distance trade-off) and
renders the ordered UAV tour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from mission.experiments.common import (
    ensure_paths,
    make_environment,
    result_objective_stats,
    run_timed_optimization,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig
from mission.fitness.scoring import MissionPlan
from mission.optimizer.nsga2 import OptimizationResult
from mission.simulation.environment import WildfireEnvironment
from mission.visualization.plot_mission_path import save_mission_path
from mission.visualization.plot_pareto import save_pareto_front


@dataclass(frozen=True)
class MissionPathExperimentResult:
    env: WildfireEnvironment
    result: OptimizationResult
    selected_plan: MissionPlan
    runtime_s: float
    csv_plans: Path
    plot_path: Path
    plot_pareto: Path


def select_representative_plan(result: OptimizationResult) -> MissionPlan:
    """
    Pick a balanced Pareto mission near the damage–distance knee.

    Uses normalized distance to the utopia point
    (max damage, min travel) within the Pareto set.
    """
    if result.n_solutions == 0:
        raise ValueError("Cannot select a mission from an empty Pareto set.")
    if result.n_solutions == 1:
        return result.plans[0]

    damages = np.array([p.objectives.damage_prevented for p in result.plans], dtype=float)
    travels = np.array([p.objectives.travel_distance for p in result.plans], dtype=float)

    dmg_span = max(float(damages.max() - damages.min()), 1e-12)
    tr_span = max(float(travels.max() - travels.min()), 1e-12)
    # Utopia in normalized [0,1] maximize-damage / minimize-travel space.
    dmg_n = (damages - damages.min()) / dmg_span
    tr_n = (travels - travels.min()) / tr_span
    # Distance to utopia (1, 0).
    dist = np.sqrt((1.0 - dmg_n) ** 2 + (tr_n - 0.0) ** 2)
    return result.plans[int(np.argmin(dist))]


def run_mission_path_experiment(
    exp: ExperimentConfig | None = None,
) -> MissionPathExperimentResult:
    """Optimize once and export a mission-path figure for a knee solution."""
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    env, cfg = make_environment(exp, n_targets=exp.n_targets, seed=exp.seed)
    opt = replace(
        cfg.optimizer,
        population_size=exp.path_population,
        n_generations=exp.path_generations,
    )
    cfg = replace(cfg, optimizer=opt)

    timed = run_timed_optimization(env, cfg, seed=exp.seed)
    if timed.result.n_solutions == 0:
        raise RuntimeError("NSGA-II returned an empty Pareto set for path visualization.")

    selected = select_representative_plan(timed.result)
    stats = result_objective_stats(timed.result)

    plan_rows: list[dict[str, Any]] = []
    for plan in timed.result.plans:
        plan_rows.append(
            {
                "mission_index": plan.index,
                "selected": int(plan.index == selected.index),
                "target_order": "->".join(str(i) for i in plan.target_ids),
                "n_targets_visited": len(plan.target_ids),
                "damage_prevented": plan.objectives.damage_prevented,
                "travel_distance": plan.objectives.travel_distance,
                "battery_usage": plan.objectives.battery_usage,
            }
        )

    csv_plans = write_csv(paths.csv / "mission_path_plans.csv", plan_rows)
    plot_path = save_mission_path(
        env,
        selected,
        paths.plots / "mission_path.png",
        viz_cfg=cfg.visualization,
    )
    plot_pareto = save_pareto_front(
        timed.result,
        path=paths.plots / "mission_path_pareto.png",
        viz_cfg=cfg.visualization,
    )

    # Attach summary stats for the report via row metadata file.
    write_csv(
        paths.csv / "mission_path_summary.csv",
        [
            {
                "n_targets": env.n_targets,
                "population_size": exp.path_population,
                "n_generations": exp.path_generations,
                "runtime_s": timed.runtime_s,
                "n_pareto": timed.result.n_solutions,
                "selected_mission_index": selected.index,
                "selected_damage": selected.objectives.damage_prevented,
                "selected_travel": selected.objectives.travel_distance,
                "selected_battery": selected.objectives.battery_usage,
                "best_damage": stats["best_damage"],
                "best_travel": stats["best_travel"],
            }
        ],
    )

    return MissionPathExperimentResult(
        env=env,
        result=timed.result,
        selected_plan=selected,
        runtime_s=timed.runtime_s,
        csv_plans=csv_plans,
        plot_path=plot_path,
        plot_pareto=plot_pareto,
    )
