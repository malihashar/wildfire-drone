#!/usr/bin/env python3
"""
Mission-planning research prototype entry point.

Pipeline status
---------------
Environmental Data          … external / future
ConvLSTM Prediction         … existing ``src/`` (not wired yet)
Predicted Spread Map        … future adapter
Candidate Targets           … synthetic generator (Phase 1)
NSGA-II Mission Optimizer   … pymoo NSGA-II (this phase)
Mission Scoring             … objective evaluation (this phase)
Selected Mission            … choose from Pareto set (manual / later)
D* Lite Local Planner       … later (stub only)
Drone Execution             … later
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python mission/main.py`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mission.config.settings import (  # noqa: E402
    GridConfig,
    MissionConfig,
    OptimizerConfig,
    TargetGenerationConfig,
    VisualizationConfig,
)
from mission.experiments.optimize_demo import run_optimization_demo  # noqa: E402
from mission.experiments.phase1_demo import run_phase1_demo  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wildfire UAV mission planning research prototype."
    )
    parser.add_argument(
        "--mode",
        choices=("scene", "optimize"),
        default="optimize",
        help="scene = Phase-1 visualization only; optimize = NSGA-II (default).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument("--min-targets", type=int, default=10)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=100, help="Square grid side length.")
    parser.add_argument("--pop-size", type=int, default=80, help="NSGA-II population size.")
    parser.add_argument("--generations", type=int, default=100, help="NSGA-II generations.")
    parser.add_argument(
        "--max-mission-distance",
        type=float,
        default=250.0,
        help="Maximum allowed tour length.",
    )
    parser.add_argument(
        "--max-mission-targets",
        type=int,
        default=8,
        help="Maximum targets visited per mission.",
    )
    parser.add_argument(
        "--damage-metric",
        choices=("predicted_damage", "priority"),
        default="predicted_damage",
        help="Objective-1 aggregation attribute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scene PNG path (default under outputs/mission/).",
    )
    parser.add_argument(
        "--pareto-output",
        type=Path,
        default=None,
        help="Pareto PNG path (default under outputs/mission/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print pymoo generation logs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib windows after saving.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> MissionConfig:
    return MissionConfig(
        grid=GridConfig(width=args.grid_size, height=args.grid_size),
        targets=TargetGenerationConfig(
            min_targets=args.min_targets,
            max_targets=args.max_targets,
        ),
        visualization=VisualizationConfig(),
        optimizer=OptimizerConfig(
            population_size=args.pop_size,
            n_generations=args.generations,
            max_mission_distance=args.max_mission_distance,
            max_mission_targets=args.max_mission_targets,
            damage_metric=args.damage_metric,
            verbose=args.verbose,
        ),
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    config = build_config(args)

    if args.mode == "scene":
        run_phase1_demo(config=config, show=args.show, output_path=args.output)
    else:
        run_optimization_demo(
            config=config,
            show=args.show,
            scene_path=args.output,
            pareto_path=args.pareto_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
