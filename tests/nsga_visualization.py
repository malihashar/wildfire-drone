"""
One-shot 2D mission-plan visualization for the NSGA-II scenario/mission
pipeline. Called exactly ONCE, after NSGA-II has already finished and a
final target ordering has been selected -- never from inside the fitness
function, the optimization loop, or per-generation callbacks.

This module does not import anything from ``mission.optimizer`` /
``mission.fitness`` and does not touch NSGA-II's population, chromosomes,
operators, or objectives. It only draws the (x, y) target coordinates and
the already-decided visit order that the caller hands it -- the exact same
``target_ids``/coordinates that go on to become GPS waypoints for the
Pixhawk mission. Nothing here feeds back into the optimizer or the flight
scripts; it is a pure side-effect (a saved PNG file).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window, safe for scripted/SITL/CI runs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DEFAULT_OUTPUT_DIR = Path("nsga_run_images")


def save_mission_visualization(
    seed: int,
    all_targets: list,  # list[mission.simulation.targets.SuppressionTarget] -- typed loosely to avoid importing optimizer internals here
    target_ids: list[int],
    damage_prevented: float,
    travel_distance: float,
    radius_m: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Draw and save a single 2D map of the planned mission:
      - home at (0, 0)
      - every generated target, labeled by ID
      - the NSGA-II-selected targets highlighted and connected in visit order
      - the scenario's radius circle
      - North/East axes in metres, with a legend

    ``all_targets`` / ``target_ids`` must be the SAME objects/IDs the caller
    is about to convert into GPS waypoints -- this function does not
    regenerate, reorder, or randomize anything itself.

    Returns the saved file path. Does not print (the caller prints the
    "Saved mission visualization: ..." message so callers can control
    exactly when/whether that happens).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_target = {t.id: t for t in all_targets}
    selected = set(target_ids)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scenario radius (10m circle centered on home).
    circle = Circle((0, 0), radius_m, fill=False, linestyle="--", color="gray", linewidth=1.5, label=f"{radius_m:.0f}m scenario radius")
    ax.add_patch(circle)

    # Home at (0, 0). Targets store (x=north_m, y=east_m); plotted as (East, North)
    # so the figure reads like a standard top-down map (North = up).
    ax.scatter([0], [0], marker="*", s=350, color="black", zorder=5, label="Home (0, 0)")
    ax.annotate("Home", (0, 0), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")

    # Every generated target, labeled by ID. Non-selected targets (dropped by
    # NSGA-II's chromosome-decode feasibility cutoff, if any) are shown dimmed.
    for t in all_targets:
        east_m, north_m = t.y, t.x
        if t.id in selected:
            continue  # drawn below, on top of the path, in a distinct style
        ax.scatter([east_m], [north_m], marker="o", s=90, color="lightgray", edgecolor="dimgray", zorder=3)
        ax.annotate(f"T{t.id}", (east_m, north_m), textcoords="offset points", xytext=(6, 6), fontsize=8, color="dimgray")

    # Selected mission targets, connected in the exact NSGA-II visit order.
    path_east = [0.0] + [id_to_target[tid].y for tid in target_ids]
    path_north = [0.0] + [id_to_target[tid].x for tid in target_ids]
    ax.plot(path_east, path_north, "-", color="tab:blue", linewidth=2, zorder=4, label="Planned path (visit order)")
    for i in range(len(path_east) - 1):
        ax.annotate(
            "",
            xy=(path_east[i + 1], path_north[i + 1]),
            xytext=(path_east[i], path_north[i]),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5, alpha=0.7),
            zorder=4,
        )

    for order, tid in enumerate(target_ids, start=1):
        t = id_to_target[tid]
        east_m, north_m = t.y, t.x
        ax.scatter([east_m], [north_m], marker="o", s=160, color="tab:orange", edgecolor="black", zorder=6)
        ax.annotate(
            f"#{order} T{tid}",
            (east_m, north_m),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    # Return-to-launch leg (dashed, visually distinct from the outbound mission path).
    if target_ids:
        last = id_to_target[target_ids[-1]]
        ax.plot([last.y, 0.0], [last.x, 0.0], "--", color="tab:red", linewidth=1.5, zorder=4, label="Return to launch")

    ax.scatter([], [], marker="o", s=90, color="lightgray", edgecolor="dimgray", label="Generated target (not selected)")
    ax.scatter([], [], marker="o", s=160, color="tab:orange", edgecolor="black", label="Selected mission target")

    margin = radius_m * 0.3
    ax.set_xlim(-radius_m - margin, radius_m + margin)
    ax.set_ylim(-radius_m - margin, radius_m + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(
        f"NSGA-II Mission Plan -- seed={seed}\n"
        f"{len(target_ids)}/{len(all_targets)} targets selected  "
        f"damage_prevented={damage_prevented:.3f}  travel_distance={travel_distance:.2f}m"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    output_path = output_dir / f"nsga2_mission_seed_{seed}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
