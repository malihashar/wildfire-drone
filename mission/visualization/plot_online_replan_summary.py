"""
Research-style summary plot for the online (in-mission) NSGA-II replanning
transcript.

Designed to make one thing visually obvious: the executed suppression
mission changes across replan events because the (synthetic) wildfire
prediction changes between suppression steps — not because of a single
static initial plan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mission.replanning.online_replanner import OnlineReplanResult


def plot_online_replan_summary(result: OnlineReplanResult, path: Path) -> Path:
    """Render a 2x2 research figure and save it to ``path``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ticks = [0] + [e.tick for e in result.events]
    missions = [result.initial_mission.target_ids] + [
        e.new_mission.target_ids for e in result.events
    ]
    suppressed = [None] + [e.suppressed_target_id for e in result.events]
    scores = [result.initial_mission.score] + [e.new_score for e in result.events]
    runtimes = [result.initial_runtime_s] + [e.optimization_runtime_s for e in result.events]
    n_pareto = [result.initial_n_pareto] + [e.n_pareto for e in result.events]
    damage = [result.initial_mission.plan.objectives.damage_prevented] + [
        e.new_mission.plan.objectives.damage_prevented for e in result.events
    ]
    travel = [result.initial_mission.plan.objectives.travel_distance] + [
        e.new_mission.plan.objectives.travel_distance for e in result.events
    ]
    ev_ticks = [e.tick for e in result.events]
    ev_prev_scores = [e.previous_score for e in result.events]

    fig, ((ax_comp, ax_score), (ax_runtime, ax_effect)) = plt.subplots(
        2, 2, figsize=(13, 9), dpi=150
    )

    _plot_composition(ax_comp, ticks, missions, suppressed)
    _plot_score(ax_score, ticks, scores, ev_ticks, ev_prev_scores)
    _plot_runtime(ax_runtime, ticks, runtimes, n_pareto)
    _plot_effectiveness(ax_effect, ticks, damage, travel)

    fig.suptitle(
        "Online NSGA-II Replanning During an Active Suppression Mission\n"
        "Synthetic wildfire-prediction updates trigger re-optimization of the "
        "remaining (unsuppressed) mission — the initial plan is not followed blindly.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_composition(ax, ticks, missions, suppressed) -> None:
    cmap = plt.get_cmap("tab20")
    suppressed_label_used = False
    for tick, mission in zip(ticks, missions):
        for order_pos, tid in enumerate(mission):
            ax.scatter(
                tick,
                tid,
                s=max(25, 90 - 6 * order_pos),
                color=cmap(order_pos % 20),
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
    for tick, sid in zip(ticks, suppressed):
        if sid is None:
            continue
        ax.scatter(
            tick,
            sid,
            marker="*",
            s=260,
            color="#c62828",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
            label="Suppressed target" if not suppressed_label_used else None,
        )
        suppressed_label_used = True

    all_ids = sorted({tid for m in missions for tid in m})
    for tid in all_ids:
        xs = [t for t, m in zip(ticks, missions) if tid in m]
        if len(xs) > 1:
            ax.plot(xs, [tid] * len(xs), color="#999999", linewidth=0.6, alpha=0.5, zorder=1)

    ax.set_title("Selected Mission Composition per Replan Event")
    ax.set_xlabel("Replan event (tick)")
    ax.set_ylabel("Target ID (marker size = visit order, larger = earlier)")
    ax.set_xticks(ticks)
    ax.grid(True, linestyle=":", alpha=0.4)
    if suppressed_label_used:
        ax.legend(loc="upper right", fontsize=8)


def _plot_score(ax, ticks, scores, ev_ticks, ev_prev_scores) -> None:
    ax.plot(
        ticks, scores, marker="o", color="#1f4e79", linewidth=2, label="Selected mission score"
    )
    if ev_ticks:
        ax.scatter(
            ev_ticks,
            ev_prev_scores,
            marker="x",
            s=60,
            color="#b85c38",
            label="Score before this replan",
            zorder=3,
        )
    ax.set_title("Mission Score vs Replan Event")
    ax.set_xlabel("Replan event (tick)")
    ax.set_ylabel("Scalarized mission score")
    ax.set_xticks(ticks)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)


def _plot_runtime(ax, ticks, runtimes, n_pareto) -> None:
    ax.bar(ticks, runtimes, color="#4a7c59", width=0.5, label="NSGA-II runtime (s)")
    ax.set_title("NSGA-II Optimization Runtime per Replan Event")
    ax.set_xlabel("Replan event (tick)")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks(ticks)
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")

    ax2 = ax.twinx()
    ax2.plot(ticks, n_pareto, marker="s", color="#8a4baf", label="Pareto-front size")
    ax2.set_ylabel("Pareto-front size")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")


def _plot_effectiveness(ax, ticks, damage, travel) -> None:
    ax.plot(ticks, damage, marker="o", color="#c45c26", label="Damage prevented (selected)")
    ax.set_xlabel("Replan event (tick)")
    ax.set_ylabel("Damage prevented", color="#c45c26")
    ax.tick_params(axis="y", labelcolor="#c45c26")
    ax.set_xticks(ticks)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("Suppression Effectiveness vs Travel Cost")

    ax2 = ax.twinx()
    ax2.plot(ticks, travel, marker="^", color="#1f4e79", label="Travel distance (selected)")
    ax2.set_ylabel("Travel distance", color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
