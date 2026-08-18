#!/usr/bin/env python3
"""
Render docs/architecture_pipeline.drawio as a PNG.

No draw.io desktop app / CLI is available in this environment, so this
redraws the same block diagram (same boxes, order, and REPLAN loop)
directly with matplotlib rather than exporting the .drawio file.
If draw.io desktop is later installed, prefer:
    drawio --export --format png --output docs/architecture_pipeline.png \
        docs/architecture_pipeline.drawio

Style: plain black-on-white boxes. Solid border = existing / this task.
Dashed border = not yet implemented (future work). The only accent color
is a plain blue fill on the NSGA-II box (this task's contribution).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/architecture_pipeline.png")

WHITE = "#ffffff"
BLUE = "#dae8fc"
BLACK = "#000000"

BOX_W = 4.2
CX = 5.0

# key, title, body, fill, dashed, bold
BOXES = [
    ("static", "STATIC DATA", "Terrain / elevation\nVegetation / fuel\nSlope", WHITE, False, False),
    ("dynamic", "DYNAMIC DATA", "Current fire state, wind,\ntemperature, humidity", WHITE, False, False),
    ("convlstm", "ConvLSTM MODEL", "Future wildfire prediction\n(fire spread / severity)\n— not implemented in this task —", WHITE, True, False),
    ("targets", "SUPPRESSION TARGETS", "Location, predicted severity,\npredicted damage / risk\n(currently: documented synthetic placeholder)", WHITE, False, False),
    ("nsga2", "NSGA-II — decides WHAT to do", "Multi-objective mission optimization\nmaximize suppression effectiveness\nminimize travel cost\n(single-UAV, this project)", BLUE, False, True),
    ("pareto", "PARETO FRONT", "Multiple trade-off\nmission candidates", WHITE, False, False),
    ("scoring", "FINAL MISSION SCORING", "Select one mission according to\nproject priorities (knee / utopia-nearest)", WHITE, False, False),
    ("dstar", "D* LITE — decides HOW to travel", "Detailed local path, obstacle avoidance,\ndynamic replanning\n— not implemented in this task (stub) —", WHITE, True, True),
    ("uav", "UAV EXECUTION", "", WHITE, False, False),
    ("newpred", "NEW FIRE PREDICTION", "", WHITE, False, False),
]

HEIGHTS = {"static": 1.3, "dynamic": 1.3, "convlstm": 1.6, "targets": 1.6,
           "nsga2": 1.9, "pareto": 1.3, "scoring": 1.4, "dstar": 1.9,
           "uav": 0.9, "newpred": 0.9}

GAP = 0.55


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 16), dpi=200)

    tops = []
    cursor = 0.0
    for key, *_ in BOXES:
        h = HEIGHTS[key]
        tops.append((key, cursor, h))
        cursor += h + GAP
    total_h = cursor - GAP

    centers: dict[str, tuple[float, float, float]] = {}

    for key, title, body, fill, dashed, bold in BOXES:
        _, top, h = next(t for t in tops if t[0] == key)
        y0 = total_h - top - h
        box = FancyBboxPatch(
            (CX - BOX_W / 2, y0), BOX_W, h,
            boxstyle="square,pad=0.02",
            linewidth=1.6, edgecolor=BLACK, facecolor=fill,
            linestyle="dashed" if dashed else "solid",
            zorder=2,
        )
        ax.add_patch(box)
        cy = y0 + h / 2
        centers[key] = (CX, y0 + h, y0)
        title_fs = 12 if bold else 11
        weight = "bold" if bold else "normal"
        if body:
            ax.text(CX, y0 + h - 0.28, title, ha="center", va="top",
                     fontsize=title_fs, fontweight=weight, color=BLACK, zorder=3)
            ax.text(CX, y0 + h * 0.42, body, ha="center", va="center",
                     fontsize=8.6, color=BLACK, zorder=3, linespacing=1.5)
        else:
            ax.text(CX, cy, title, ha="center", va="center",
                     fontsize=title_fs, fontweight=weight, color=BLACK, zorder=3)

    order = [b[0] for b in BOXES]
    for a, b in zip(order[:-1], order[1:]):
        _, top_a, bottom_a = centers[a]
        _, top_b, bottom_b = centers[b]
        arrow = FancyArrowPatch(
            (CX, bottom_a), (CX, top_b),
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.4, color=BLACK, zorder=1,
        )
        ax.add_patch(arrow)

    # REPLAN loop: bottom (newpred) back up to DYNAMIC DATA, on the left side.
    _, top_new, bottom_new = centers["newpred"]
    _, top_dyn, bottom_dyn = centers["dynamic"]
    loop_x = CX - BOX_W / 2 - 1.1
    ax.plot(
        [CX - BOX_W / 2, loop_x, loop_x, CX - BOX_W / 2],
        [bottom_new + (top_new - bottom_new) / 2, bottom_new + (top_new - bottom_new) / 2,
         top_dyn - (top_dyn - bottom_dyn) / 2, top_dyn - (top_dyn - bottom_dyn) / 2],
        color=BLACK, linewidth=1.4, zorder=1,
    )
    ax.annotate(
        "", xy=(CX - BOX_W / 2, top_dyn - (top_dyn - bottom_dyn) / 2),
        xytext=(loop_x, top_dyn - (top_dyn - bottom_dyn) / 2),
        arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4),
        zorder=1,
    )
    ax.text(loop_x - 0.15, (bottom_new + top_dyn) / 2, "REPLAN",
            rotation=90, ha="center", va="center", fontsize=10,
            fontweight="bold", fontstyle="italic", color=BLACK)

    # Note box explaining NSGA-II vs D* Lite roles.
    note_x0, note_y0, note_w, note_h = 7.4, total_h - 12.6, 2.9, 2.6
    note = FancyBboxPatch(
        (note_x0, note_y0), note_w, note_h,
        boxstyle="square,pad=0.02",
        linewidth=1.4, edgecolor=BLACK, facecolor=WHITE, zorder=2,
    )
    ax.add_patch(note)
    ax.text(
        note_x0 + note_w / 2, note_y0 + note_h - 0.25,
        "NSGA-II and D* Lite are\nsequential stages, not\ncompeting optimizers",
        ha="center", va="top", fontsize=8.3, fontweight="bold", style="italic",
        color=BLACK,
    )
    ax.text(
        note_x0 + note_w / 2, note_y0 + note_h * 0.35,
        "NSGA-II selects WHICH\nsuppression targets and in\nwhat order (the mission).\n\n"
        "D* Lite plans the detailed\nflight path BETWEEN those\nalready-chosen targets.",
        ha="center", va="center", fontsize=7.8, color=BLACK, linespacing=1.6,
    )

    # Legend.
    leg_x0, leg_y0, leg_w, leg_h = 7.4, total_h - 2.3, 2.9, 2.6
    legend = FancyBboxPatch(
        (leg_x0, leg_y0), leg_w, leg_h,
        boxstyle="square,pad=0.02",
        linewidth=1.4, edgecolor=BLACK, facecolor=WHITE, zorder=2,
    )
    ax.add_patch(legend)
    ax.text(leg_x0 + leg_w / 2, leg_y0 + leg_h - 0.25, "Legend",
            ha="center", va="top", fontsize=9.5, fontweight="bold", color=BLACK)
    ax.text(
        leg_x0 + leg_w / 2, leg_y0 + leg_h * 0.38,
        "Solid border = existing /\nthis task\n\n"
        "Dashed border = not yet\nimplemented (future work)\n\n"
        "Blue fill = NSGA-II\n(this task's contribution)",
        ha="center", va="center", fontsize=7.8, color=BLACK, linespacing=1.6,
    )

    ax.set_xlim(-0.3, 10.4)
    ax.set_ylim(-0.3, total_h + 0.3)
    ax.axis("off")
    ax.set_title(
        "Path-Planning Architecture — Wildfire Suppression UAV\n"
        "(Single-UAV NSGA-II mission selection + D* Lite local planning)",
        fontsize=13, fontweight="bold", pad=14, color=BLACK,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
