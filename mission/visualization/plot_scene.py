"""
Mission scene visualization.

Shows the UAV start pose and labeled suppression targets on the grid.
Later phases will overlay NSGA-II mission sequences and D* Lite paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mission.config.settings import VisualizationConfig
from mission.simulation.environment import WildfireEnvironment


def plot_mission_scene(
    env: WildfireEnvironment,
    viz_cfg: VisualizationConfig | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Render drone location and suppression targets.

    Marker size scales with ``priority``; color intensity scales with
    ``damage_score`` to preview the multi-objective signals NSGA-II will use.
    """
    cfg = viz_cfg or VisualizationConfig()
    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    else:
        fig = ax.figure

    _draw_grid_frame(ax, env.width, env.height)
    _draw_targets(ax, env, show_labels=cfg.show_labels)
    _draw_drone(ax, env)

    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    ax.set_title(title or f"Synthetic Wildfire Mission Scene ({env.n_targets} targets)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def save_mission_scene(
    env: WildfireEnvironment,
    path: Path | str | None = None,
    viz_cfg: VisualizationConfig | None = None,
) -> Path:
    """Plot and save the mission scene; return the output path."""
    cfg = viz_cfg or VisualizationConfig()
    out_dir = cfg.save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(path) if path is not None else out_dir / "phase1_mission_scene.png"

    fig, _ = plot_mission_scene(env, viz_cfg=cfg)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_grid_frame(ax: Axes, width: int, height: int) -> None:
    ax.set_facecolor("#f7f4ef")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.axhline(height - 1, color="#888888", linewidth=0.8)
    ax.axvline(width - 1, color="#888888", linewidth=0.8)


def _draw_drone(ax: Axes, env: WildfireEnvironment) -> None:
    ax.scatter(
        [env.drone.x],
        [env.drone.y],
        s=220,
        c="#1f4e79",
        marker="^",
        edgecolors="white",
        linewidths=1.2,
        zorder=5,
        label="UAV start",
    )
    ax.annotate(
        "UAV",
        (env.drone.x, env.drone.y),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        fontweight="bold",
        color="#1f4e79",
    )


def _draw_targets(ax: Axes, env: WildfireEnvironment, show_labels: bool) -> None:
    if not env.targets:
        return

    xs = np.array([t.x for t in env.targets])
    ys = np.array([t.y for t in env.targets])
    damage = np.array([t.damage_score for t in env.targets])
    priority = np.array([t.priority for t in env.targets])

    sizes = 60 + 180 * priority
    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=damage,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#4a1c00",
        linewidths=0.8,
        zorder=4,
        label="Suppression targets",
    )
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("damage_score")

    if show_labels:
        for t in env.targets:
            ax.annotate(
                f"T{t.id}",
                (t.x, t.y),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
                color="#333333",
            )
