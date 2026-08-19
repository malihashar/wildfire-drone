"""
Validation Experiment 3 — risk-weighted D* Lite routing ablation.

Fixed start/goal on a grid with an elevated-risk band directly on the
shortest path. Sweep ``risk_weight`` and measure the tradeoff between route
length and risk exposure -- validates that risk-weighting actually trades
distance for safety in a controllable, monotonic way rather than doing
nothing or something erratic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mission.experiments.common import ensure_paths, save_figure, style_axes, write_csv
from mission.experiments.experiment_config import ExperimentConfig
from mission.optimizer.dstar_lite import DStarLite, path_length

GRID = 40
START = (2, 2)
GOAL = (37, 37)
_BLOB_CENTER = (20, 20)  # sits squarely on the direct diagonal start->goal line
_BLOB_RADIUS = 9.0


def _build_risk_field() -> dict[tuple[int, int], float]:
    """
    A localized high-risk blob directly on the shortest path, small enough
    that the grid has room to route around it on either side -- unlike a
    grid-spanning band, which would make crossing topologically mandatory
    regardless of weight and give risk_weight nothing to trade against.
    """
    risk: dict[tuple[int, int], float] = {}
    cx, cy = _BLOB_CENTER
    for x in range(0, GRID):
        for y in range(0, GRID):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= _BLOB_RADIUS:
                risk[(x, y)] = max(0.0, 1.0 - dist / _BLOB_RADIUS)
    return risk


@dataclass(frozen=True)
class RiskRoutingAblationResult:
    trial_rows: list[dict[str, Any]]
    csv_path: Path
    plot_path: Path


def run_risk_routing_ablation(
    exp: ExperimentConfig | None = None,
    # Finer-grained than the originally-requested (0,1,10,50,100) -- that
    # set only samples two plateaus of what turns out to be a 3-step
    # staircase (straight-through -> partial detour -> full avoidance);
    # these extra points actually resolve the transition zone.
    risk_weights: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 10.0, 50.0, 100.0),
) -> RiskRoutingAblationResult:
    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)
    risk_field = _build_risk_field()

    rows: list[dict[str, Any]] = []
    for weight in risk_weights:
        planner = DStarLite(GRID, GRID, blocked=set(), risk=risk_field, risk_weight=weight)
        path = planner.plan(START, GOAL)
        if path is None:
            continue
        distance = path_length(path)
        risk_exposure = sum(risk_field.get(c, 0.0) for c in path)
        combined_cost = distance + weight * risk_exposure
        rows.append(
            {
                "risk_weight": weight,
                "actual_distance": distance,
                "risk_exposure": risk_exposure,
                "combined_cost": combined_cost,
                "path_len_cells": len(path),
            }
        )

    csv_path = write_csv(paths.csv / "risk_routing_ablation.csv", rows)
    plot_path = _plot(rows, paths.plots / "risk_routing_ablation.png")

    return RiskRoutingAblationResult(trial_rows=rows, csv_path=csv_path, plot_path=plot_path)


def _plot(rows: list[dict[str, Any]], path: Path):
    weights = [r["risk_weight"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].plot(weights, [r["actual_distance"] for r in rows], marker="o", color="#1f4e79", label="Distance")
    ax2 = axes[0].twinx()
    ax2.plot(weights, [r["risk_exposure"] for r in rows], marker="s", color="#c45c26", label="Risk exposure")
    axes[0].set_xscale("symlog", linthresh=0.5)  # weights span 0..100; symlog resolves the 0-4 transition zone
    style_axes(axes[0], "Distance vs Risk Exposure Tradeoff", "risk_weight (symlog)", "Route distance (cells)")
    ax2.set_ylabel("Risk exposure (sum along path)")
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    # Group weights that land on the same (risk, distance) point (the
    # staircase plateaus) so labels don't overlap illegibly.
    grouped: dict[tuple[float, float], list[float]] = {}
    for r in rows:
        key = (round(r["risk_exposure"], 3), round(r["actual_distance"], 3))
        grouped.setdefault(key, []).append(r["risk_weight"])

    axes[1].scatter([k[0] for k in grouped], [k[1] for k in grouped],
                     s=80, color="#8a4baf", edgecolors="black", zorder=3)
    for (risk, dist), ws in grouped.items():
        label = "w=" + ",".join(f"{w:g}" for w in sorted(ws))
        axes[1].annotate(label, (risk, dist), textcoords="offset points", xytext=(8, 4), fontsize=7.5)
    style_axes(axes[1], "Risk-Distance Pareto Curve", "Risk exposure", "Route distance (cells)")

    fig.suptitle("Risk-Weighted D* Lite Routing Ablation")
    fig.tight_layout()
    return save_figure(fig, path)
