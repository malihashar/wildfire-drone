"""
Markdown report generator for optimization-performance experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

from mission.experiments.convergence import ConvergenceExperimentResult
from mission.experiments.convergence_scaling import ConvergenceScalingResult
from mission.experiments.experiment_config import ExperimentConfig
from mission.experiments.mission_path import MissionPathExperimentResult
from mission.experiments.population_size import PopulationSizeResult
from mission.experiments.runtime_scaling import RuntimeScalingResult
from mission.experiments.scenario_diversity import ScenarioDiversityResult


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _img(path: Path, report_dir: Path, caption: str) -> str:
    rel = os.path.relpath(path, start=report_dir)
    return f"![{caption}]({Path(rel).as_posix()})\n\n*{caption}*\n"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def generate_report(
    *,
    exp: ExperimentConfig,
    convergence: ConvergenceExperimentResult | None = None,
    runtime: RuntimeScalingResult | None = None,
    population: PopulationSizeResult | None = None,
    mission_path: MissionPathExperimentResult | None = None,
    threshold: ConvergenceScalingResult | None = None,
    scenario_diversity: ScenarioDiversityResult | None = None,
) -> Path:
    """
    Write ``results/experiment_report.md`` embedding all available figures.
    """
    root = exp.paths.root
    root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Wildfire UAV Mission Optimization — Experiment Report",
        "",
        "Research evaluation of **pymoo NSGA-II** for single-UAV suppression",
        "mission planning (synthetic targets; ConvLSTM integration pending).",
        "",
        "## Shared Experimental Settings",
        "",
        f"- Grid: `{exp.grid_size} x {exp.grid_size}`",
        f"- Default target count (Exps 1/3/4): **{exp.n_targets}**",
        f"- Max mission targets: **{exp.max_mission_targets}**",
        f"- Max mission distance: **{exp.max_mission_distance}**",
        f"- Damage metric: `{exp.damage_metric}`",
        f"- Seed: `{exp.seed}`",
        "",
    ]

    if convergence is not None:
        lines.extend(_section_convergence(convergence, root, exp))
    if runtime is not None:
        lines.extend(_section_runtime(runtime, root, exp))
    if population is not None:
        lines.extend(_section_population(population, root, exp))
    if mission_path is not None:
        lines.extend(_section_mission_path(mission_path, root, exp))
    if threshold is not None:
        lines.extend(_section_threshold(threshold, root, exp))

    if scenario_diversity is not None:
        lines.extend(_section_scenario_diversity(scenario_diversity, root, exp))

    lines.extend(
        [
            "## Data Artifacts",
            "",
            f"- CSV directory: `{_rel(exp.paths.csv, root)}`",
            f"- Plot directory: `{_rel(exp.paths.plots, root)}`",
            f"- Config snapshot: `{_rel(exp.paths.config_snapshot, root)}`",
            "",
        ]
    )

    report_path = exp.paths.report
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _section_convergence(
    result: ConvergenceExperimentResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    checkpoints = result.history.checkpoint_rows(exp.convergence_generations)
    lines = [
        "## Experiment 1 — Convergence",
        "",
        f"- Number of targets: **{result.n_targets}**",
        f"- Population size: **{result.population_size}**",
        f"- Generations (max): **{result.max_generations}**",
        f"- Runtime: **{_fmt(result.runtime_s, 3)} s**",
        "",
        "### Checkpoint Summary",
        "",
        "| Generation | Best Damage | Avg Damage | Best Travel | Hypervolume | # Pareto |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoints:
        gen = row.get("requested_generation", row["generation"])
        lines.append(
            f"| {gen} | {_fmt(row['best_damage'])} | {_fmt(row['avg_damage'])} | "
            f"{_fmt(row['best_travel'])} | {_fmt(row['hypervolume'])} | {int(row['n_pareto'])} |"
        )
    lines.extend(
        [
            "",
            "### Figures",
            "",
            _img(result.plot_best, exp.paths.reports, "Best fitness vs generation"),
            _img(result.plot_avg, exp.paths.reports, "Average fitness vs generation"),
            _img(result.plot_hv, exp.paths.reports, "Hypervolume vs generation"),
            _img(result.plot_pareto_size, exp.paths.reports, "Pareto front size vs generation"),
            f"CSV: `{_rel(result.csv_history, root)}`, `{_rel(result.csv_checkpoints, root)}`",
            "",
        ]
    )
    return lines


def _section_runtime(
    result: RuntimeScalingResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    lines = [
        "## Experiment 2 — Runtime Scaling",
        "",
        f"- Population size: **{exp.runtime_population}**",
        f"- Generations per run: **{exp.runtime_generations}**",
        f"- Trials per target count: **{exp.runtime_trials}**",
        "",
        "| Targets | Mean Runtime (s) | Std (s) | Min | Max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result.summary_rows:
        lines.append(
            f"| {row['n_targets']} | {_fmt(row['mean_runtime_s'], 3)} | "
            f"{_fmt(row['std_runtime_s'], 3)} | {_fmt(row['min_runtime_s'], 3)} | "
            f"{_fmt(row['max_runtime_s'], 3)} |"
        )
    lines.extend(
        [
            "",
            "### Figures",
            "",
            _img(result.plot_runtime, exp.paths.reports, "Runtime vs number of targets"),
            f"CSV: `{_rel(result.csv_trials, root)}`, `{_rel(result.csv_summary, root)}`",
            "",
        ]
    )
    return lines


def _section_population(
    result: PopulationSizeResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    lines = [
        "## Experiment 3 — Population Size",
        "",
        f"- Number of targets: **{exp.n_targets}**",
        f"- Generations: **{exp.population_generations}**",
        f"- Trials per population size: **{exp.population_trials}**",
        "",
        "| Pop. Size | Mean Runtime (s) | Best Damage (mean±std) | Best Travel (mean±std) | Pareto Size (mean±std) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result.summary_rows:
        lines.append(
            f"| {row['population_size']} | {_fmt(row['mean_runtime_s'], 3)} | "
            f"{_fmt(row['mean_best_damage'])} ± {_fmt(row['std_best_damage'])} | "
            f"{_fmt(row['mean_best_travel'])} ± {_fmt(row['std_best_travel'])} | "
            f"{_fmt(row['mean_n_pareto'], 2)} ± {_fmt(row['std_n_pareto'], 2)} |"
        )
    lines.extend(
        [
            "",
            "### Figures",
            "",
            _img(result.plot_runtime, exp.paths.reports, "Runtime vs population size"),
            _img(result.plot_objectives, exp.paths.reports, "Best objectives vs population size"),
            _img(result.plot_pareto_size, exp.paths.reports, "Pareto set size vs population size"),
            f"CSV: `{_rel(result.csv_trials, root)}`, `{_rel(result.csv_summary, root)}`",
            "",
        ]
    )
    return lines


def _section_mission_path(
    result: MissionPathExperimentResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    plan = result.selected_plan
    order = " → ".join(f"T{i}" for i in plan.target_ids)
    lines = [
        "## Experiment 4 — Mission Path Visualization",
        "",
        f"- Number of targets: **{result.env.n_targets}**",
        f"- Population size: **{exp.path_population}**",
        f"- Generations: **{exp.path_generations}**",
        f"- Runtime: **{_fmt(result.runtime_s, 3)} s**",
        f"- Number of Pareto solutions: **{result.result.n_solutions}**",
        f"- Selected mission index: **{plan.index}** (knee / utopia-nearest)",
        f"- Mission order: `{order}`",
        f"- Best objectives (selected): damage={_fmt(plan.objectives.damage_prevented)}, "
        f"travel={_fmt(plan.objectives.travel_distance)}, "
        f"battery={_fmt(plan.objectives.battery_usage)}",
        "",
        "### Figures",
        "",
        _img(result.plot_path, exp.paths.reports, "Selected Pareto mission path"),
        _img(result.plot_pareto, exp.paths.reports, "Pareto front for path experiment"),
        f"CSV: `{_rel(result.csv_plans, root)}`",
        "",
    ]
    return lines


def _section_threshold(
    result: ConvergenceScalingResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    lines = [
        "## Experiment 5 — Generations to Convergence Threshold vs Problem Size",
        "",
        f"- Population size: **{exp.threshold_population}**",
        f"- Max generations per run: **{exp.threshold_generations}**",
        f"- Trials per target count: **{exp.threshold_trials}**",
        "",
        "| Targets | Gen@95% | Gen@98% | Gen@99% |",
        "|---:|---:|---:|---:|",
    ]
    for row in result.summary_rows:
        lines.append(
            f"| {row['n_targets']} | {_fmt(row['mean_gen_95'], 1)} ± {_fmt(row['std_gen_95'], 1)} | "
            f"{_fmt(row['mean_gen_98'], 1)} ± {_fmt(row['std_gen_98'], 1)} | "
            f"{_fmt(row['mean_gen_99'], 1)} ± {_fmt(row['std_gen_99'], 1)} |"
        )
    lines.extend(
        [
            "",
            "### Figures",
            "",
            _img(
                result.plot_generations,
                exp.paths.reports,
                "Generations to reach 95/98/99% of final best fitness vs number of targets",
            ),
            f"CSV: `{_rel(result.csv_trials, root)}`, `{_rel(result.csv_summary, root)}`",
            "",
        ]
    )
    return lines


def _section_scenario_diversity(
    result: ScenarioDiversityResult,
    root: Path,
    exp: ExperimentConfig,
) -> list[str]:
    lines = [
        "## Experiment 6 — Scenario Diversity",
        "",
        "Same target count, three spatial/severity distributions: a single",
        "uniform scatter is not evidence NSGA-II generalizes to realistic",
        "multi-front, wind-driven fire scenarios.",
        "",
        "| Scenario | Best Damage | Best Travel | Runtime (s) | Pareto Size |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.rows:
        lines.append(
            f"| {row['scenario']} | {_fmt(row['best_damage'])} | {_fmt(row['best_travel'])} | "
            f"{_fmt(row['runtime_s'], 3)} | {row['n_pareto']} |"
        )
    lines.extend(
        [
            "",
            "### Figures",
            "",
            _img(result.comparison_plot, exp.paths.reports, "Objectives/runtime across scenario types"),
        ]
    )
    for name, path in result.scene_plots.items():
        lines.append(_img(path, exp.paths.reports, f"Scenario scene: {name}"))
    lines.extend(
        [
            "",
            f"CSV: `{_rel(result.csv_path, root)}`",
            "",
        ]
    )
    return lines
