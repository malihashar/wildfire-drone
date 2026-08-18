"""Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization."""

from __future__ import annotations

from pathlib import Path

from mission.config.settings import MissionConfig
from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult
from mission.simulation.environment import WildfireEnvironment
from mission.visualization.plot_pareto import save_pareto_front
from mission.visualization.plot_scene import save_mission_scene


def run_optimization_demo(
    config: MissionConfig | None = None,
    show: bool = False,
    scene_path: Path | str | None = None,
    pareto_path: Path | str | None = None,
) -> tuple[WildfireEnvironment, OptimizationResult, Path, Path]:
    """
    Generate a synthetic environment, run NSGA-II, and save visualizations.

    Returns
    -------
    env, result, scene_figure_path, pareto_figure_path
    """
    cfg = config or MissionConfig()
    env = WildfireEnvironment.create_synthetic(cfg)
    print(env.summary())
    print()

    scene_fig = save_mission_scene(env, path=scene_path, viz_cfg=cfg.visualization)
    print(f"Saved mission scene → {scene_fig}")

    optimizer = NSGA2MissionOptimizer(env, cfg)
    result = optimizer.optimize()
    print()
    print(result.summary())

    pareto_fig = save_pareto_front(result, path=pareto_path, viz_cfg=cfg.visualization)
    print(f"\nSaved Pareto front → {pareto_fig}")

    if show:
        import matplotlib.pyplot as plt

        from mission.visualization.plot_pareto import plot_pareto_front
        from mission.visualization.plot_scene import plot_mission_scene

        plot_mission_scene(env, viz_cfg=cfg.visualization)
        plot_pareto_front(result, viz_cfg=cfg.visualization)
        plt.show()

    return env, result, scene_fig, pareto_fig
