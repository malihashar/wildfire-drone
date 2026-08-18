"""
Ordered mission-path visualization for a selected Pareto plan.

Draws the UAV start, numbered visit order, polyline path, and direction arrows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mission.config.settings import VisualizationConfig
from mission.fitness.scoring import MissionPlan
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.targets import SuppressionTarget


def plot_mission_path(
    env: WildfireEnvironment,
    plan: MissionPlan,
    viz_cfg: VisualizationConfig | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Render one mission as an ordered path through suppression targets.

    Visit order is annotated as 1..k on the selected targets. Non-visited
    targets are shown faded in the background.
    """
    cfg = viz_cfg or VisualizationConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    else:
        fig = ax.figure

    id_to_target = {t.id: t for t in env.targets}
    ordered = [_require_target(id_to_target, tid) for tid in plan.target_ids]

    ax.set_facecolor("#f7f4ef")
    _draw_background_targets(ax, env.targets, visited_ids=set(plan.target_ids))

    # Path coordinates: drone → targets in order.
    xs = [env.drone.x] + [t.x for t in ordered]
    ys = [env.drone.y] + [t.y for t in ordered]

    ax.plot(xs, ys, color="#1f4e79", linewidth=2.0, alpha=0.85, zorder=3, label="Mission path")
    _draw_direction_arrows(ax, xs, ys)

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

    if ordered:
        ox = [t.x for t in ordered]
        oy = [t.y for t in ordered]
        ax.scatter(
            ox,
            oy,
            s=120,
            c="#c45c26",
            edgecolors="#4a1c00",
            linewidths=0.9,
            zorder=4,
            label="Visited targets",
        )
        for step, target in enumerate(ordered, start=1):
            ax.annotate(
                f"{step}\n(T{target.id})",
                (target.x, target.y),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color="#333333",
                fontweight="bold",
            )

    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    ax.grid(True, linestyle=":", alpha=0.4)

    obj = plan.objectives
    default_title = (
        f"Mission {plan.index} Path  |  "
        f"damage={obj.damage_prevented:.3f}, "
        f"travel={obj.travel_distance:.1f}, "
        f"battery={obj.battery_usage:.1f}"
    )
    ax.set_title(title or default_title)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def save_mission_path(
    env: WildfireEnvironment,
    plan: MissionPlan,
    path: Path | str,
    viz_cfg: VisualizationConfig | None = None,
) -> Path:
    """Plot and save a mission path figure."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_mission_path(env, plan, viz_cfg=viz_cfg)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def _require_target(
    id_to_target: dict[int, SuppressionTarget],
    target_id: int,
) -> SuppressionTarget:
    if target_id not in id_to_target:
        raise KeyError(f"Mission references unknown target id {target_id}.")
    return id_to_target[target_id]


def _draw_background_targets(
    ax: Axes,
    targets: list[SuppressionTarget],
    visited_ids: set[int],
) -> None:
    unused = [t for t in targets if t.id not in visited_ids]
    if not unused:
        return
    ax.scatter(
        [t.x for t in unused],
        [t.y for t in unused],
        s=45,
        c="#bbbbbb",
        edgecolors="#888888",
        linewidths=0.5,
        alpha=0.55,
        zorder=2,
        label="Unused targets",
    )
    for t in unused:
        ax.annotate(
            f"T{t.id}",
            (t.x, t.y),
            textcoords="offset points",
            xytext=(4, -9),
            fontsize=7,
            color="#777777",
        )


def _draw_direction_arrows(ax: Axes, xs: list[float], ys: list[float]) -> None:
    """Place mid-segment arrows indicating mission direction."""
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) + abs(dy) < 1e-9:
            continue
        # Arrow centered on the segment.
        ax.annotate(
            "",
            xy=(x0 + 0.62 * dx, y0 + 0.62 * dy),
            xytext=(x0 + 0.38 * dx, y0 + 0.38 * dy),
            arrowprops=dict(
                arrowstyle="->",
                color="#1f4e79",
                lw=1.6,
                mutation_scale=14,
            ),
            zorder=3,
        )
