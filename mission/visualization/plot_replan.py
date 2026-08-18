"""
Frame rendering for online replanning visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mission.fitness.mission_selection import ScoredMission
from mission.replanning.executor import MissionExecutionResult
from mission.replanning.online_replanner import OnlineReplanResult
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.prediction_source import EnvironmentDiff


@dataclass(frozen=True)
class FrameStats:
    """HUD statistics drawn on each animation frame."""

    tick: int
    n_replan_events: int
    optimization_runtime_s: float
    n_targets: int
    n_pareto: int
    best_mission_score: float
    previous_score: float | None
    status: str


def render_replan_frame(
    env: WildfireEnvironment,
    previous_mission: ScoredMission | None,
    new_mission: ScoredMission,
    diff: EnvironmentDiff | None,
    stats: FrameStats,
    *,
    env_before: WildfireEnvironment | None = None,
    execution_result: MissionExecutionResult | None = None,
    figsize: tuple[float, float] = (11.0, 9.0),
    dpi: int = 120,
    title: str | None = None,
) -> Figure:
    """
    Draw one research-quality replanning frame.

    Colors (matches wildfire-GIS / drone-GCS convention: fire = red/orange
    threat, UAV/route = blue-family response, kept visually distinct)
    ------
    Old mission order: muted red dashed (thin reference)
    NSGA-II target order (new): thin blue dotted reference line
    D* Lite route (actual, obstacle-aware): solid teal with direction arrows
    Added targets: green
    Removed targets: red X at last-known positions
    Priority-changed: gold rings
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.15], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    ax.set_facecolor("#f7f4ef")
    _draw_targets(ax, env, diff)
    if previous_mission is not None:
        _draw_path(
            ax,
            env_before or env,
            previous_mission,
            color="#b85c38",
            linestyle="--",
            linewidth=1.6,
            alpha=0.6,
            label="Previous mission",
            arrows=False,
            zorder=3,
        )
    _draw_path(
        ax,
        env,
        new_mission,
        color="#1f4e79",
        linestyle=":",
        linewidth=1.6,
        alpha=0.75,
        label="NSGA-II target order",
        arrows=False,
        zorder=4,
    )
    if execution_result is not None and execution_result.feasible and execution_result.cell_path:
        _draw_dstar_path(ax, execution_result, zorder=5)
    _draw_drone(ax, env)

    ax.set_xlim(-2, env.width + 1)
    ax.set_ylim(-2, env.height + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(title or f"Online Replanning — tick {stats.tick}")

    legend_elements = [
        Line2D([0], [0], color="#00897b", lw=2.4, label="D* Lite route (actual)"),
        Line2D([0], [0], color="#1f4e79", lw=1.6, linestyle=":", label="NSGA-II target order"),
        Line2D([0], [0], color="#b85c38", lw=1.6, linestyle="--", label="Previous mission"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32", markersize=9, label="Added target"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c45c26",
            markeredgecolor="#d4a017",
            markeredgewidth=2,
            markersize=9,
            label="Priority changed",
        ),
        Line2D([0], [0], marker="x", color="#c62828", markersize=9, label="Removed target"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f4e79", markersize=10, label="UAV"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.92, fontsize=8)

    _draw_stats_panel(ax_info, stats, previous_mission, new_mission, diff, execution_result)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06, wspace=0.25)
    return fig


def build_animation_frames(result: OnlineReplanResult) -> list[Figure]:
    """Create one figure per replan event (plus an initial frame)."""
    frames: list[Figure] = []
    cfg = result.config

    frames.append(
        render_replan_frame(
            result.initial_env,
            previous_mission=None,
            new_mission=result.initial_mission,
            diff=EnvironmentDiff(),
            stats=FrameStats(
                tick=0,
                n_replan_events=0,
                optimization_runtime_s=result.initial_runtime_s,
                n_targets=result.initial_env.n_targets,
                n_pareto=result.initial_n_pareto,
                best_mission_score=result.initial_mission.score,
                previous_score=None,
                status="Initial mission (pre-replan)",
            ),
            figsize=cfg.figsize,
            dpi=cfg.dpi,
            title="Initial NSGA-II Mission",
            execution_result=result.initial_execution_result,
        )
    )

    for event in result.events:
        frames.append(
            render_replan_frame(
                event.env_after,
                previous_mission=event.previous_mission,
                new_mission=event.new_mission,
                diff=event.diff,
                stats=FrameStats(
                    tick=event.tick,
                    n_replan_events=event.tick,
                    optimization_runtime_s=event.optimization_runtime_s,
                    n_targets=event.n_targets,
                    n_pareto=event.n_pareto,
                    best_mission_score=event.new_score,
                    previous_score=event.previous_score,
                    status=event.why,
                ),
                env_before=event.env_before,
                execution_result=event.execution_result,
                figsize=cfg.figsize,
                dpi=cfg.dpi,
                title=f"Replan Event {event.tick}",
            )
        )
    return frames


def _draw_drone(ax: Axes, env: WildfireEnvironment) -> None:
    ax.scatter(
        [env.drone.x],
        [env.drone.y],
        s=220,
        c="#1f4e79",
        marker="^",
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
    )


def _draw_targets(
    ax: Axes,
    env: WildfireEnvironment,
    diff: EnvironmentDiff | None,
) -> None:
    diff = diff or EnvironmentDiff()
    added = set(diff.added_ids)
    priority_changed = set(diff.priority_changed_ids)
    moved = set(diff.moved_ids)

    for target in env.targets:
        color = "#c45c26"
        edge = "#4a1c00"
        ew = 0.8
        if target.id in added:
            color = "#2e7d32"
            edge = "#1b5e20"
        elif target.id in priority_changed:
            edge = "#d4a017"
            ew = 2.2
        elif target.id in moved:
            edge = "#6a1b9a"
            ew = 1.6

        ax.scatter(
            [target.x],
            [target.y],
            s=70 + 120 * target.priority,
            c=color,
            edgecolors=edge,
            linewidths=ew,
            zorder=5,
        )
        ax.annotate(
            f"T{target.id}",
            (target.x, target.y),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7,
            color="#333333",
        )

    # Annotate removed targets at last-known coordinates.
    for rid, (rx, ry) in (diff.removed_positions or {}).items():
        ax.scatter(
            [rx],
            [ry],
            s=90,
            marker="x",
            linewidths=2.0,
            c="#c62828",
            zorder=5,
        )
        ax.annotate(
            f"T{rid}∅",
            (rx, ry),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7,
            color="#c62828",
        )


def _draw_path(
    ax: Axes,
    env: WildfireEnvironment,
    mission: ScoredMission,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    label: str,
    arrows: bool,
    zorder: int,
) -> None:
    id_map = {t.id: t for t in env.targets}
    xs = [env.drone.x]
    ys = [env.drone.y]
    for tid in mission.target_ids:
        t = id_map.get(tid)
        if t is None:
            continue
        xs.append(t.x)
        ys.append(t.y)

    if len(xs) < 2:
        return

    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    if arrows:
        for i in range(len(xs) - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            if abs(dx) + abs(dy) < 1e-9:
                continue
            ax.annotate(
                "",
                xy=(xs[i] + 0.62 * dx, ys[i] + 0.62 * dy),
                xytext=(xs[i] + 0.38 * dx, ys[i] + 0.38 * dy),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    mutation_scale=12,
                ),
                zorder=zorder + 1,
            )


def _draw_dstar_path(
    ax: Axes,
    execution_result: MissionExecutionResult,
    *,
    zorder: int,
) -> None:
    """Draw the actual obstacle-aware D* Lite route (solid teal, with arrows)."""
    xs = [c[0] for c in execution_result.cell_path]
    ys = [c[1] for c in execution_result.cell_path]
    if len(xs) < 2:
        return

    ax.plot(
        xs,
        ys,
        color="#00897b",
        linestyle="-",
        linewidth=2.2,
        alpha=0.95,
        label="D* Lite route (actual)",
        zorder=zorder,
    )
    step = max(1, len(xs) // 12)  # sparse arrows so a long route isn't cluttered
    for i in range(0, len(xs) - 1, step):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        if abs(dx) + abs(dy) < 1e-9:
            continue
        ax.annotate(
            "",
            xy=(xs[i] + 0.7 * dx, ys[i] + 0.7 * dy),
            xytext=(xs[i] + 0.3 * dx, ys[i] + 0.3 * dy),
            arrowprops=dict(arrowstyle="->", color="#00897b", lw=1.3, mutation_scale=10),
            zorder=zorder + 1,
        )


def _draw_stats_panel(
    ax: Axes,
    stats: FrameStats,
    previous: ScoredMission | None,
    new: ScoredMission,
    diff: EnvironmentDiff | None,
    execution_result: MissionExecutionResult | None = None,
) -> None:
    prev_score = "—" if stats.previous_score is None else f"{stats.previous_score:.3f}"
    prev_order = (
        "—"
        if previous is None
        else (" → ".join(f"T{i}" for i in previous.target_ids) or "(empty)")
    )
    new_order = " → ".join(f"T{i}" for i in new.target_ids) or "(empty)"
    diff = diff or EnvironmentDiff()

    dstar_lines: list[str] = []
    if execution_result is not None:
        if execution_result.feasible:
            extra = execution_result.path_length - execution_result.straight_line_length
            dstar_lines = [
                "",
                "D* LITE ROUTE",
                f"Actual:      {execution_result.path_length:.1f} cells",
                f"Straight-line: {execution_result.straight_line_length:.1f} cells",
                f"Detour cost: +{max(extra, 0.0):.1f} cells",
            ]
        else:
            dstar_lines = ["", "D* LITE ROUTE", "INFEASIBLE (blocked by obstacles)"]

    lines = [
        "LIVE STATISTICS",
        "",
        f"Tick: {stats.tick}",
        f"Replan events: {stats.n_replan_events}",
        f"Opt. runtime: {stats.optimization_runtime_s:.3f} s",
        f"# Targets: {stats.n_targets}",
        f"# Pareto sols: {stats.n_pareto}",
        f"Best score: {stats.best_mission_score:.3f}",
        f"Prev score: {prev_score}",
        "",
        "PREVIOUS MISSION",
        prev_order,
        "",
        "NEW MISSION",
        new_order,
        "",
        "CHANGES",
        f"Added: {diff.added_ids or '—'}",
        f"Removed: {diff.removed_ids or '—'}",
        f"Priority Δ: {diff.priority_changed_ids or '—'}",
        f"Moved: {diff.moved_ids or '—'}",
        *dstar_lines,
        "",
        "STATUS",
        _wrap(stats.status, width=34),
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
        linespacing=1.35,
    )


def _wrap(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if len(trial) <= width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)
