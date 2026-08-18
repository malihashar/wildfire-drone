"""
Shared configuration for research evaluation experiments.

Defaults are chosen for publishable curves while remaining runnable on a laptop.
Override via CLI flags in ``run_all``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Output layout for CSVs, plots, and the Markdown report."""

    root: Path = field(default_factory=lambda: Path("results"))

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def csv(self) -> Path:
        return self.root / "csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def report(self) -> Path:
        return self.reports / "experiment_report.md"

    @property
    def config_snapshot(self) -> Path:
        return self.experiments / "config_used.json"

    def ensure(self) -> None:
        self.plots.mkdir(parents=True, exist_ok=True)
        self.csv.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.experiments.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared across independent optimization experiments."""

    paths: ExperimentPaths = field(default_factory=ExperimentPaths)
    seed: int = 42
    grid_size: int = 100
    # Fixed scene size used by convergence / population / path experiments.
    n_targets: int = 15
    max_mission_distance: float = 250.0
    max_mission_targets: int = 8
    damage_metric: str = "predicted_damage"
    crossover_prob: float = 0.9
    mutation_prob: float = 0.3

    # Experiment 1 — Convergence
    convergence_generations: tuple[int, ...] = (25, 50, 100, 200, 500)
    convergence_population: int = 60

    # Experiment 2 — Runtime scaling
    runtime_target_counts: tuple[int, ...] = (10, 20, 30, 40, 50)
    runtime_trials: int = 5
    runtime_generations: int = 120
    runtime_population: int = 100

    # Experiment 3 — Population size
    population_sizes: tuple[int, ...] = (25, 50, 100, 200)
    population_generations: int = 80
    population_trials: int = 3

    # Experiment 4 — Mission path (uses same fixed scene as convergence)
    path_population: int = 80
    path_generations: int = 100

    # Experiment 5 — Generations-to-threshold vs problem size
    threshold_target_counts: tuple[int, ...] = (10, 15, 20, 30, 40, 50)
    threshold_trials: int = 3
    threshold_population: int = 60
    threshold_generations: int = 200
