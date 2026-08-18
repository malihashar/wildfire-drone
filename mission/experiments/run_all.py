#!/usr/bin/env python3
"""
CLI for research optimization-performance experiments.

Examples
--------
python -m mission.experiments.run_all
python -m mission.experiments.run_all --only convergence
python -m mission.experiments.run_all --only runtime,population,path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mission.experiments.common import write_config_snapshot  # noqa: E402
from mission.experiments.convergence import run_convergence_experiment  # noqa: E402
from mission.experiments.convergence_scaling import (  # noqa: E402
    run_convergence_scaling_experiment,
)
from mission.experiments.experiment_config import ExperimentConfig, ExperimentPaths  # noqa: E402
from mission.experiments.mission_path import run_mission_path_experiment  # noqa: E402
from mission.experiments.population_size import run_population_size_experiment  # noqa: E402
from mission.experiments.report import generate_report  # noqa: E402
from mission.experiments.runtime_scaling import run_runtime_scaling_experiment  # noqa: E402
from mission.experiments.scenario_diversity import (  # noqa: E402
    run_scenario_diversity_experiment,
)


EXPERIMENT_NAMES = (
    "convergence", "runtime", "population", "path", "threshold", "scenario", "all",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NSGA-II optimization-performance experiments."
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated: convergence,runtime,population,path,threshold,scenario,all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-targets", type=int, default=15)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory for csv/, plots/, and the Markdown report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter budgets for a fast smoke run (not for paper figures).",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    paths = ExperimentPaths(root=args.results_dir)
    if not args.quick:
        return ExperimentConfig(paths=paths, seed=args.seed, n_targets=args.n_targets)

    # Fast smoke configuration.
    return ExperimentConfig(
        paths=paths,
        seed=args.seed,
        n_targets=min(args.n_targets, 12),
        convergence_generations=(10, 25, 50),
        convergence_population=30,
        runtime_target_counts=(10, 20, 30),
        runtime_trials=2,
        runtime_generations=15,
        runtime_population=25,
        population_sizes=(25, 50, 100),
        population_generations=25,
        population_trials=2,
        path_population=40,
        path_generations=40,
        threshold_target_counts=(10, 20, 30),
        threshold_trials=1,
        threshold_population=30,
        threshold_generations=60,
    )


def selected_experiments(spec: str) -> set[str]:
    parts = {p.strip().lower() for p in spec.split(",") if p.strip()}
    all_names = {"convergence", "runtime", "population", "path", "threshold", "scenario"}
    if not parts or "all" in parts:
        return all_names
    unknown = parts - all_names - {"all"}
    if unknown:
        raise ValueError(f"Unknown experiment name(s): {sorted(unknown)}")
    return parts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")

    exp = build_config(args)
    chosen = selected_experiments(args.only)
    snapshot_path = write_config_snapshot(exp)
    print(f"Config snapshot → {snapshot_path}")

    convergence = runtime = population = mission_path = threshold = scenario = None

    if "convergence" in chosen:
        print("=== Experiment 1: Convergence ===")
        convergence = run_convergence_experiment(exp)
        print(f"  runtime={convergence.runtime_s:.2f}s  plots → {exp.paths.plots}")

    if "runtime" in chosen:
        print("=== Experiment 2: Runtime Scaling ===")
        runtime = run_runtime_scaling_experiment(exp)
        print(f"  trials={len(runtime.trial_rows)}  plot → {runtime.plot_runtime}")

    if "population" in chosen:
        print("=== Experiment 3: Population Size ===")
        population = run_population_size_experiment(exp)
        print(f"  trials={len(population.trial_rows)}  plots → {exp.paths.plots}")

    if "path" in chosen:
        print("=== Experiment 4: Mission Path ===")
        mission_path = run_mission_path_experiment(exp)
        print(
            f"  selected mission {mission_path.selected_plan.index}  "
            f"path → {mission_path.plot_path}"
        )

    if "threshold" in chosen:
        print("=== Experiment 5: Generations-to-Threshold Scaling ===")
        threshold = run_convergence_scaling_experiment(exp)
        print(f"  trials={len(threshold.trial_rows)}  plot → {threshold.plot_generations}")

    if "scenario" in chosen:
        print("=== Experiment 6: Scenario Diversity ===")
        scenario = run_scenario_diversity_experiment(exp)
        print(f"  scenarios={len(scenario.rows)}  plot → {scenario.comparison_plot}")

    report_path = generate_report(
        exp=exp,
        convergence=convergence,
        runtime=runtime,
        population=population,
        mission_path=mission_path,
        threshold=threshold,
        scenario_diversity=scenario,
    )
    print(f"\nReport written → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
