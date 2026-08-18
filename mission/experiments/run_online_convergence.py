#!/usr/bin/env python3
"""
CLI for the online-replanning NSGA-II convergence experiment.

Example
-------
python -m mission.experiments.run_online_convergence
python -m mission.experiments.run_online_convergence --events 8 --generations 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mission.experiments.experiment_config import ExperimentPaths  # noqa: E402
from mission.experiments.online_convergence import (  # noqa: E402
    run_online_convergence_experiment,
)
from mission.replanning.config import OnlineReplanConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online (in-mission) NSGA-II convergence experiment."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--targets", type=int, default=15, help="Initial suppression targets.")
    p.add_argument("--events", type=int, default=6, help="Number of replan events.")
    p.add_argument("--pop-size", type=int, default=60)
    p.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Generation budget per NSGA-II run (initial + each replan event).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root for the shared csv/ and plots/ output.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Faster smoke settings (not for supervisor demos).",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> OnlineReplanConfig:
    if args.quick:
        return OnlineReplanConfig(
            seed=args.seed,
            n_targets_initial=min(args.targets, 12),
            n_replan_events=min(args.events, 3),
            population_size=30,
            n_generations=30,
        )
    return OnlineReplanConfig(
        seed=args.seed,
        n_targets_initial=args.targets,
        n_replan_events=args.events,
        population_size=args.pop_size,
        n_generations=args.generations,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")

    cfg = build_config(args)
    result = run_online_convergence_experiment(
        cfg, results_paths=ExperimentPaths(root=args.results_dir)
    )
    print(f"NSGA-II runs (initial + replan events): {result.n_runs}")
    print(f"Raw per-generation CSV        → {result.csv_raw}")
    print(f"Aggregated per-generation CSV → {result.csv_aggregated}")
    print(f"Plot                          → {result.plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
