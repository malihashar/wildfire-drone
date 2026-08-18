"""
Markdown report for the online replanning architecture demo.
"""

from __future__ import annotations

from pathlib import Path

from mission.replanning.online_replanner import OnlineReplanResult


def write_replan_report(
    result: OnlineReplanResult,
    artifacts: dict[str, Path],
    report_path: Path | None = None,
) -> Path:
    """Generate a Markdown report describing each replan event."""
    out_dir = result.config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(report_path) if report_path is not None else out_dir / "online_replan_report.md"

    def rel(p: Path) -> str:
        try:
            return p.relative_to(out_dir).as_posix()
        except ValueError:
            return p.as_posix()

    lines: list[str] = [
        "# Online Replanning Architecture Demo",
        "",
        "Synthetic wildfire-prediction updates trigger NSGA-II replanning.",
        "This demonstrates the CREDS-inspired online loop **before** ConvLSTM",
        "and D* Lite are integrated.",
        "",
        "## Configuration",
        "",
        f"- Initial targets: **{result.config.n_targets_initial}**",
        f"- Replan events: **{result.n_replan_events}**",
        f"- Population size: **{result.config.population_size}**",
        f"- Generations / replan: **{result.config.n_generations}**",
        f"- Animation FPS: **{result.config.animation_fps}**",
        f"- Hold frames / event: **{result.config.hold_frames_per_event}**",
        f"- Seed: **{result.config.seed}**",
        "",
        "## Initial Mission",
        "",
        f"- Optimization runtime: **{result.initial_runtime_s:.3f} s**",
        f"- Pareto solutions: **{result.initial_n_pareto}**",
        f"- Mission score: **{result.initial_mission.score:.4f}**",
        f"- Order: `{' → '.join(f'T{i}' for i in result.initial_mission.target_ids)}`",
        "",
        "## Artifacts",
        "",
        f"- GIF: `{rel(artifacts['gif'])}`",
        f"- MP4: `{rel(artifacts['mp4'])}`",
        f"- Frames: `{rel(artifacts['frames_dir'])}/`",
    ]
    if "csv" in artifacts:
        lines.append(f"- Per-event CSV: `{artifacts['csv'].as_posix()}`")
    if "summary_plot" in artifacts:
        lines.append(f"- Summary plot: `{artifacts['summary_plot'].as_posix()}`")
    lines.extend(
        [
            "",
            f"![Animation GIF]({rel(artifacts['gif'])})",
            "",
        ]
    )

    for event in result.events:
        prev = " → ".join(f"T{i}" for i in event.previous_mission.target_ids)
        new = " → ".join(f"T{i}" for i in event.new_mission.target_ids)
        delta = event.new_score - event.previous_score
        suppressed = (
            f"T{event.suppressed_target_id}"
            if event.suppressed_target_id is not None
            else "none (previous mission had no live targets left)"
        )
        lines.extend(
            [
                f"## Replan Event {event.tick}",
                "",
                "### Suppression completed this tick",
                "",
                f"- Suppressed (flown to + removed, permanently excluded from future missions): **{suppressed}**",
                "",
                "### Why replanning occurred",
                "",
                f"- Prediction source: `{event.update.source_name}`",
                f"- Update note: {event.update.note}",
                f"- Environment diff: {event.diff.describe()}",
                "",
                "### Mission change",
                "",
                f"- Previous: `{prev}`",
                f"- New: `{new}`",
                f"- Score before: **{event.previous_score:.4f}**",
                f"- Score after: **{event.new_score:.4f}** (Δ = {delta:+.4f})",
                f"- Optimization runtime: **{event.optimization_runtime_s:.3f} s**",
                f"- Targets after update: **{event.n_targets}**",
                f"- Pareto solutions: **{event.n_pareto}**",
                "",
                f"Summary: {event.why}",
                "",
            ]
        )

    lines.extend(
        [
            "## Future Integration Notes",
            "",
            "1. Replace `SyntheticPredictionSource` with `ConvLSTMPredictionSource`",
            "   that emits the same `PredictionUpdate` schema from predicted maps.",
            "2. Pass each `MissionExecutionRequest` into `DStarLiteMissionExecutor`",
            "   to locally refine waypoint-to-waypoint paths.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
