"""
Fair NSGA-II generation-count benchmark for the wildfire-suppression optimizer.

Uses the project's REAL ``NSGA2MissionOptimizer`` (mission/optimizer/nsga2.py)
unmodified -- no reimplementation of NSGA-II. For each random scenario (see
``tests/nsga_scenario.py``), the EXACT SAME targets/severity/config are run
at every generation count in ``GENERATION_COUNTS`` so the effect of extra
computation can be compared fairly. Only ``NSGA2MissionOptimizer.optimize()``
is timed with ``time.perf_counter()`` -- scenario generation, printing, and
result packaging are excluded.

Run standalone, no MAVSDK / PX4 SITL involved:

    python tests/nsga_benchmark.py
    python tests/nsga_benchmark.py --seeds 42 43 7 --output-dir results/csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pymoo.core.callback import Callback

from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from nsga_scenario import generate_scenario, scenario_with_generations

GENERATION_COUNTS = (50, 70, 90, 100, 200, 300, 400, 500)
DEFAULT_SEEDS = (42, 43)
CONVERGENCE_CHECKPOINTS = (10, 20, 30, 50, 70, 90, 100, 200, 500)
TWO_SECOND_BUDGET_S = 2.0


class ConvergenceRecorder(Callback):
    """
    Records the Pareto front's best per-objective values at each requested
    generation checkpoint. Multi-objective -- there is no single scalar
    "best fitness"; ``best_damage_prevented`` / ``min_travel_distance`` /
    ``min_battery_usage`` are each the best value ACROSS the current
    non-dominated set (``algorithm.opt``), not a fabricated composite score.
    """

    def __init__(self, checkpoints: tuple[int, ...]) -> None:
        super().__init__()
        self.checkpoints = set(checkpoints)
        self.history: list[dict] = []

    def notify(self, algorithm) -> None:
        gen = int(algorithm.n_gen)
        if gen not in self.checkpoints and gen != algorithm.termination.n_max_gen:
            return
        F = algorithm.opt.get("F")
        if F is None or len(F) == 0:
            return
        self.history.append(
            {
                "generation": gen,
                # F columns: [-damage_prevented, travel_distance, battery_usage]
                "best_damage_prevented": float(-F[:, 0].min()),
                "min_travel_distance": float(F[:, 1].min()),
                "min_battery_usage": float(F[:, 2].min()),
                "pareto_front_size": int(len(F)),
            }
        )


def run_benchmark(seeds: tuple[int, ...], generation_counts: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []

    for seed in seeds:
        base_scenario = generate_scenario(seed)
        print(f"\n=== Scenario seed={seed}  targets={base_scenario.n_targets} ===")

        for n_gen in generation_counts:
            scenario = scenario_with_generations(base_scenario, n_gen)
            optimizer = NSGA2MissionOptimizer(scenario.env, scenario.optimizer_config)
            recorder = ConvergenceRecorder(CONVERGENCE_CHECKPOINTS + (n_gen,))

            start = time.perf_counter()  # NSGA-II optimization ONLY
            result = optimizer.optimize(seed=seed, callback=recorder)
            elapsed_s = time.perf_counter() - start

            if result.n_solutions == 0:
                print(f"  gen={n_gen:>4}: no feasible solutions")
                continue

            best = result.best_damage_plan()
            gen_per_s = n_gen / elapsed_s if elapsed_s > 0 else float("inf")

            print(
                f"  gen={n_gen:>4}  time={elapsed_s:7.4f}s  gen/s={gen_per_s:8.1f}  "
                f"pareto={result.n_solutions:>3}  "
                f"damage={best.objectives.damage_prevented:6.3f}  "
                f"travel={best.objectives.travel_distance:6.2f}  "
                f"battery={best.objectives.battery_usage:6.2f}  "
                f"path={best.mission_order}"
            )
            print("    convergence (generation -> best_damage_prevented / min_travel_distance):")
            for point in recorder.history:
                print(
                    f"      {point['generation']:>4}  "
                    f"{point['best_damage_prevented']:6.3f} / {point['min_travel_distance']:6.2f}  "
                    f"(pareto size {point['pareto_front_size']})"
                )

            rows.append(
                {
                    "seed": seed,
                    "n_targets": scenario.n_targets,
                    "population_size": scenario.optimizer_config.population_size,
                    "n_generations": n_gen,
                    "time_s": elapsed_s,
                    "generations_per_s": gen_per_s,
                    "pareto_front_size": result.n_solutions,
                    "best_damage_prevented": best.objectives.damage_prevented,
                    "best_travel_distance": best.objectives.travel_distance,
                    "best_battery_usage": best.objectives.battery_usage,
                    "final_path_target_ids": list(best.mission_order),
                    "convergence_history": recorder.history,
                }
            )

    return rows


def summarize_generation_budget(rows: list[dict], budget_s: float = TWO_SECOND_BUDGET_S) -> str:
    """
    Determine, FROM THE MEASURED DATA, roughly how many generations fit in
    ``budget_s`` seconds. Reports a linear-fit R^2 per seed so a nonlinear
    relationship is surfaced rather than silently extrapolated.
    """
    by_seed: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append((r["n_generations"], r["time_s"]))

    lines = [f"\n=== Practical generation limit within {budget_s:.1f}s (measured, not assumed) ==="]
    for seed, pairs in sorted(by_seed.items()):
        pairs.sort()
        gens = np.array([p[0] for p in pairs], dtype=float)
        times = np.array([p[1] for p in pairs], dtype=float)

        slope, intercept = np.polyfit(gens, times, 1)
        pred = slope * gens + intercept
        ss_res = float(np.sum((times - pred) ** 2))
        ss_tot = float(np.sum((times - times.mean()) ** 2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
        linearity = "roughly linear" if r2 > 0.9 else f"NONLINEAR (r^2={r2:.3f}) -- do not extrapolate linearly"

        below = [(g, t) for g, t in pairs if t <= budget_s]
        above = [(g, t) for g, t in pairs if t > budget_s]
        if below and above:
            g_lo, t_lo = below[-1]
            g_hi, t_hi = above[0]
            interp = g_lo + (budget_s - t_lo) * (g_hi - g_lo) / (t_hi - t_lo)
            estimate = (
                f"~{interp:.0f} generations "
                f"(interpolated between {g_lo} gens @ {t_lo:.3f}s and {g_hi} gens @ {t_hi:.3f}s)"
            )
        elif below:
            estimate = f">= {below[-1][0]} generations (every measured count finished under budget)"
        else:
            estimate = f"< {pairs[0][0]} generations (even the smallest tested count exceeded budget)"

        lines.append(f"  seed {seed}: {estimate}; runtime-vs-generations fit r^2={r2:.3f} ({linearity})")
    return "\n".join(lines)


def print_final_table(rows: list[dict]) -> None:
    print("\n=== Final benchmark table ===")
    header = f"{'Seed':>5} {'Targets':>7} {'Pop':>5} {'Gens':>5} {'Time(s)':>9} {'Gen/s':>9} {'Damage':>8} {'Travel':>8} {'Battery':>8}  Path"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['seed']:>5} {r['n_targets']:>7} {r['population_size']:>5} {r['n_generations']:>5} "
            f"{r['time_s']:>9.4f} {r['generations_per_s']:>9.1f} "
            f"{r['best_damage_prevented']:>8.3f} {r['best_travel_distance']:>8.2f} {r['best_battery_usage']:>8.2f}  "
            f"{r['final_path_target_ids']}"
        )


def save_results(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "nsga_benchmark_results.json"
    csv_path = output_dir / "nsga_benchmark_results.csv"

    json_path.write_text(json.dumps(rows, indent=2))

    csv_rows = [{k: v for k, v in r.items() if k != "convergence_history"} for r in rows]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--generations", type=int, nargs="+", default=list(GENERATION_COUNTS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/csv"))
    args = parser.parse_args()

    rows = run_benchmark(tuple(args.seeds), tuple(args.generations))
    print_final_table(rows)
    print(summarize_generation_budget(rows))

    json_path, csv_path = save_results(rows, args.output_dir)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
