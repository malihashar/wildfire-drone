#!/usr/bin/env python3
"""
CLI for the online replanning architecture demo.

Example
-------
python -m mission.experiments.run_online_replan
python -m mission.experiments.run_online_replan --events 4 --fps 2 --hold 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mission.experiments.experiment_config import ExperimentPaths  # noqa: E402
from mission.experiments.online_replan_demo import run_online_replan_demo  # noqa: E402
from mission.replanning.config import OnlineReplanConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online replanning architecture demo.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--targets", type=int, default=15, help="Initial suppression targets.")
    p.add_argument("--events", type=int, default=6, help="Number of replan events.")
    p.add_argument("--pop-size", type=int, default=50)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--fps", type=float, default=1.25, help="Animation frames per second.")
    p.add_argument(
        "--hold",
        type=int,
        default=10,
        help="How many times each event frame is repeated (slower = larger).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/online_replan"),
        help="Animation frames/GIF/MP4/report directory.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root for the shared csv/ and plots/ output (per-event CSV + summary plot).",
    )
    p.add_argument(
        "--no-advance-drone",
        action="store_true",
        help="Keep UAV at the initial start pose between replans.",
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
            n_generations=25,
            animation_fps=args.fps,
            hold_frames_per_event=max(2, args.hold // 2),
            advance_drone=not args.no_advance_drone,
            output_dir=args.output_dir,
        )
    return OnlineReplanConfig(
        seed=args.seed,
        n_targets_initial=args.targets,
        n_replan_events=args.events,
        population_size=args.pop_size,
        n_generations=args.generations,
        animation_fps=args.fps,
        hold_frames_per_event=args.hold,
        advance_drone=not args.no_advance_drone,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")

    cfg = build_config(args)
    run_online_replan_demo(cfg, results_paths=ExperimentPaths(root=args.results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
