"""
Best-fitness vs wall-clock runtime analysis.

Records per-generation wall-clock time and best damage prevented (Objective 1),
then finds the earliest generation where best fitness reaches 95%, 98%, and 99%
of the final best value.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.callback import Callback

from mission.experiments.common import (
    ensure_paths,
    make_environment,
    population_objective_stats,
    run_timed_optimization,
    save_figure,
    style_axes,
    write_csv,
)
from mission.experiments.experiment_config import ExperimentConfig, ExperimentPaths


THRESHOLDS = (0.95, 0.98, 0.99)


@dataclass
class RuntimeFitnessHistory:
    """Per-generation wall-clock runtime and best fitness."""

    generations: list[int] = field(default_factory=list)
    runtime_ms: list[float] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "generation": g,
                "runtime_ms": rt,
                "best_fitness": bf,
                "fraction_of_final": (bf / self.final_best) if self.final_best > 0 else 0.0,
            }
            for g, rt, bf in zip(
                self.generations, self.runtime_ms, self.best_fitness, strict=True
            )
        ]

    @property
    def final_best(self) -> float:
        return float(self.best_fitness[-1]) if self.best_fitness else 0.0


@dataclass(frozen=True)
class ThresholdHit:
    """Earliest generation where best fitness reaches a fraction of the final best."""

    fraction: float
    target_fitness: float
    generation: int | None
    runtime_ms: float | None
    best_fitness: float | None
    reached: bool


class RuntimeFitnessCallback(Callback):
    """Log elapsed wall-clock time and best damage after every generation."""

    def __init__(self) -> None:
        super().__init__()
        self.t0 = time.perf_counter()
        self.history = RuntimeFitnessHistory()

    def notify(self, algorithm: object) -> None:
        elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
        pop = algorithm.pop  # type: ignore[attr-defined]
        F = np.atleast_2d(np.asarray(pop.get("F"), dtype=float))
        stats = population_objective_stats(F)
        gen = int(algorithm.n_gen)  # type: ignore[attr-defined]

        self.history.generations.append(gen)
        self.history.runtime_ms.append(float(elapsed_ms))
        # Objective 1 (maximize): predicted damage prevented.
        self.history.best_fitness.append(float(stats["best_damage"]))


def find_threshold_hits(
    history: RuntimeFitnessHistory,
    fractions: tuple[float, ...] = THRESHOLDS,
) -> list[ThresholdHit]:
    """Earliest generation at which best fitness >= fraction * final_best."""
    final_best = history.final_best
    hits: list[ThresholdHit] = []
    for frac in fractions:
        target = frac * final_best
        hit_gen = hit_rt = hit_bf = None
        reached = False
        for g, rt, bf in zip(
            history.generations, history.runtime_ms, history.best_fitness, strict=True
        ):
            if bf >= target - 1e-12:
                hit_gen, hit_rt, hit_bf = g, rt, bf
                reached = True
                break
        hits.append(
            ThresholdHit(
                fraction=frac,
                target_fitness=target,
                generation=hit_gen,
                runtime_ms=hit_rt,
                best_fitness=hit_bf,
                reached=reached,
            )
        )
    return hits


@dataclass(frozen=True)
class FitnessRuntimeResult:
    history: RuntimeFitnessHistory
    thresholds: tuple[ThresholdHit, ...]
    total_runtime_ms: float
    n_targets: int
    population_size: int
    n_generations: int
    csv_history: Path
    csv_thresholds: Path
    plot_path: Path


def run_fitness_runtime_experiment(
    exp: ExperimentConfig | None = None,
    *,
    n_generations: int | None = None,
    population_size: int | None = None,
) -> FitnessRuntimeResult:
    """
    Optimize while recording best fitness and wall-clock time each generation.
    """
    exp = exp or ExperimentConfig(
        paths=ExperimentPaths(root=Path("results")),
    )
    paths = ensure_paths(exp.paths)
    n_gen = n_generations if n_generations is not None else max(exp.convergence_generations)
    pop = population_size if population_size is not None else exp.convergence_population

    env, cfg = make_environment(exp, n_targets=exp.n_targets, seed=exp.seed)
    cfg = replace(
        cfg,
        optimizer=replace(cfg.optimizer, population_size=pop, n_generations=n_gen),
    )

    callback = RuntimeFitnessCallback()
    timed = run_timed_optimization(env, cfg, seed=exp.seed, callback=callback)
    history = callback.history
    hits = find_threshold_hits(history)

    csv_history = write_csv(
        paths.csv / "fitness_vs_runtime_history.csv",
        history.as_rows(),
    )
    csv_thresholds = write_csv(
        paths.csv / "fitness_vs_runtime_thresholds.csv",
        [
            {
                "threshold_pct": int(round(h.fraction * 100)),
                "fraction": h.fraction,
                "target_fitness": h.target_fitness,
                "generation": h.generation if h.reached else "",
                "runtime_ms": h.runtime_ms if h.reached else "",
                "best_fitness_at_hit": h.best_fitness if h.reached else "",
                "reached": int(h.reached),
                "final_best_fitness": history.final_best,
            }
            for h in hits
        ],
    )
    plot_path = _plot_fitness_vs_runtime(
        history,
        hits,
        paths.plots / "best_fitness_vs_runtime.png",
    )

    return FitnessRuntimeResult(
        history=history,
        thresholds=tuple(hits),
        total_runtime_ms=timed.runtime_s * 1000.0,
        n_targets=env.n_targets,
        population_size=pop,
        n_generations=n_gen,
        csv_history=csv_history,
        csv_thresholds=csv_thresholds,
        plot_path=plot_path,
    )


def _plot_fitness_vs_runtime(
    history: RuntimeFitnessHistory,
    hits: list[ThresholdHit],
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    ax.plot(
        history.runtime_ms,
        history.best_fitness,
        color="#1f4e79",
        lw=2.2,
        label="Best fitness (damage prevented)",
    )

    colors = {0.95: "#2e7d32", 0.98: "#c45c26", 0.99: "#6b3fa0"}
    final = history.final_best
    for hit in hits:
        y = hit.target_fitness
        color = colors.get(hit.fraction, "#333333")
        pct = int(round(hit.fraction * 100))
        ax.axhline(
            y,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"{pct}% of final ({y:.4f})",
        )
        if hit.reached and hit.runtime_ms is not None and hit.best_fitness is not None:
            ax.scatter(
                [hit.runtime_ms],
                [hit.best_fitness],
                s=70,
                c=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"{pct}% @ gen {hit.generation}\n{hit.runtime_ms:.1f} ms",
                (hit.runtime_ms, hit.best_fitness),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color=color,
            )

    ax.axhline(
        final,
        color="#888888",
        linestyle=":",
        linewidth=1.2,
        label=f"Final best ({final:.4f})",
    )
    style_axes(ax, "Best Fitness vs Runtime", "Wall-clock Runtime (ms)", "Best Fitness")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, path)


def print_threshold_report(result: FitnessRuntimeResult) -> None:
    """Print a concise research-style summary to stdout."""
    h = result.history
    print("=== Best Fitness vs Runtime ===")
    print(f"targets={result.n_targets}  pop={result.population_size}  gens={result.n_generations}")
    print(f"final best fitness (damage prevented) = {h.final_best:.6f}")
    print(f"total wall-clock runtime = {result.total_runtime_ms:.2f} ms")
    print()
    print(f"{'Threshold':>10}  {'Target':>10}  {'Gen':>6}  {'Runtime (ms)':>14}  {'Fitness':>10}")
    print("-" * 60)
    for hit in result.thresholds:
        pct = f"{int(round(hit.fraction * 100))}%"
        if not hit.reached:
            print(f"{pct:>10}  {hit.target_fitness:10.4f}  {'—':>6}  {'not reached':>14}  {'—':>10}")
            continue
        print(
            f"{pct:>10}  {hit.target_fitness:10.4f}  {hit.generation:6d}  "
            f"{hit.runtime_ms:14.2f}  {hit.best_fitness:10.4f}"
        )
    print()
    print(f"CSV history   → {result.csv_history}")
    print(f"CSV thresholds→ {result.csv_thresholds}")
    print(f"Plot          → {result.plot_path}")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    result = run_fitness_runtime_experiment(
        ExperimentConfig(paths=ExperimentPaths(root=Path("results"))),
        n_generations=500,
        population_size=60,
    )
    print_threshold_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
