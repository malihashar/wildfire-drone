"""Pareto-front visualization for NSGA-II mission plans."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mission.config.settings import VisualizationConfig
from mission.optimizer.nsga2 import OptimizationResult


def plot_pareto_front(
    result: OptimizationResult,
    viz_cfg: VisualizationConfig | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Scatter plot: travel distance (x) vs damage prevented (y).

    Each Pareto solution is labeled with its mission index.
    """
    cfg = viz_cfg or VisualizationConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    else:
        fig = ax.figure

    if result.n_solutions == 0:
        ax.set_title(title or "Pareto Front (empty)")
        ax.set_xlabel("Travel Distance")
        ax.set_ylabel("Damage Prevented")
        fig.tight_layout()
        return fig, ax

    xs = [p.objectives.travel_distance for p in result.plans]
    ys = [p.objectives.damage_prevented for p in result.plans]

    ax.scatter(
        xs,
        ys,
        s=70,
        c="#c45c26",
        edgecolors="#4a1c00",
        linewidths=0.8,
        zorder=3,
        label="Pareto missions",
    )

    # Connect points sorted by travel distance to suggest the front shape.
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ax.plot(
        [xs[i] for i in order],
        [ys[i] for i in order],
        color="#c45c26",
        alpha=0.35,
        linewidth=1.2,
        zorder=2,
    )

    for plan, x, y in zip(result.plans, xs, ys, strict=True):
        ax.annotate(
            str(plan.index),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Travel Distance")
    ax.set_ylabel("Damage Prevented")
    ax.set_title(title or f"Pareto Front ({result.n_solutions} missions)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def save_pareto_front(
    result: OptimizationResult,
    path: Path | str | None = None,
    viz_cfg: VisualizationConfig | None = None,
) -> Path:
    """Plot and save the Pareto front; return the output path."""
    cfg = viz_cfg or VisualizationConfig()
    out_dir = cfg.save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(path) if path is not None else out_dir / "pareto_front.png"

    fig, _ = plot_pareto_front(result, viz_cfg=cfg)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
