"""Phase-1 experiment: generate a synthetic scene and save a visualization."""

from __future__ import annotations

from pathlib import Path

from mission.config.settings import MissionConfig
from mission.simulation.environment import WildfireEnvironment
from mission.visualization.plot_scene import save_mission_scene


def run_phase1_demo(
    config: MissionConfig | None = None,
    show: bool = False,
    output_path: Path | str | None = None,
) -> tuple[WildfireEnvironment, Path]:
    """
    Create a synthetic wildfire mission scene and export a matplotlib figure.

    Returns the environment and the path to the saved figure.
    """
    cfg = config or MissionConfig()
    env = WildfireEnvironment.create_synthetic(cfg)
    print(env.summary())

    figure_path = save_mission_scene(env, path=output_path, viz_cfg=cfg.visualization)
    print(f"Saved mission scene → {figure_path}")

    if show:
        from mission.visualization.plot_scene import plot_mission_scene
        import matplotlib.pyplot as plt

        plot_mission_scene(env, viz_cfg=cfg.visualization)
        plt.show()

    return env, figure_path
