"""
Experiment runner for the online replanning architecture demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mission.experiments.experiment_config import ExperimentPaths
from mission.replanning.config import OnlineReplanConfig
from mission.replanning.csv_export import write_replan_csv
from mission.replanning.online_replanner import OnlineReplanner, OnlineReplanResult
from mission.replanning.report import write_replan_report
from mission.visualization.animate_replan import save_replan_outputs
from mission.visualization.plot_online_replan_summary import plot_online_replan_summary


@dataclass(frozen=True)
class OnlineReplanDemoResult:
    result: OnlineReplanResult
    artifacts: dict[str, Path]
    report_path: Path


def run_online_replan_demo(
    config: OnlineReplanConfig | None = None,
    results_paths: ExperimentPaths | None = None,
) -> OnlineReplanDemoResult:
    """
    Run synthetic online replanning, export animations, CSV, plot, and a report.

    ``results_paths`` controls where the CSV and summary plot land (defaults
    to the shared ``results/csv`` and ``results/plots`` directories used by
    the other optimization-performance experiments). Animation frames/GIF/MP4
    and the Markdown report still go to ``config.output_dir``
    (``results/online_replan`` by default).
    """
    cfg = config or OnlineReplanConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = results_paths or ExperimentPaths()
    paths.ensure()

    print("Running online replanner…")
    replanner = OnlineReplanner(config=cfg)
    result = replanner.run()
    print(
        f"  initial score={result.initial_mission.score:.3f}  "
        f"events={result.n_replan_events}"
    )

    print("Rendering animation artifacts…")
    artifacts = save_replan_outputs(result)

    csv_path = write_replan_csv(result, paths.csv / "online_replan_events.csv")
    plot_path = plot_online_replan_summary(result, paths.plots / "online_replan_summary.png")
    artifacts = {**artifacts, "csv": csv_path, "summary_plot": plot_path}

    report_path = write_replan_report(result, artifacts)
    print(f"  gif → {artifacts['gif']}")
    print(f"  mp4 → {artifacts['mp4']}")
    print(f"  csv → {csv_path}")
    print(f"  plot → {plot_path}")
    print(f"  report → {report_path}")

    return OnlineReplanDemoResult(
        result=result,
        artifacts=artifacts,
        report_path=report_path,
    )
