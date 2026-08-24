"""
TEST 8 -- per-stage performance timing for the NSGA-II + MAVSDK software
pipeline. Measures each stage SEPARATELY over many repetitions and reports
mean/median/p95/min/max. Pure software except for stage 5, which times only
local ``mavsdk.System()`` construction -- no network connection is opened,
so this script never touches hardware.

Usage:
    python tests/nsga_pipeline_timing.py
    python tests/nsga_pipeline_timing.py --reps 50 --seed 42 --generations 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mavsdk import System
from mavsdk.mission import MissionItem

import nsga_scenario as ns
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from nsga_pixhawk_mission import ACCEPTANCE_RADIUS_M, CRUISE_ALTITUDE_M, _offset_latlon

HOME_LAT, HOME_LON = 37.7749, -122.4194


@dataclass
class StageTimings:
    name: str
    samples_s: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples_s)

    @property
    def median(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def p95(self) -> float:
        sorted_samples = sorted(self.samples_s)
        idx = min(len(sorted_samples) - 1, int(round(0.95 * (len(sorted_samples) - 1))))
        return sorted_samples[idx]

    @property
    def minimum(self) -> float:
        return min(self.samples_s)

    @property
    def maximum(self) -> float:
        return max(self.samples_s)


def time_scenario_generation(reps: int, seed_base: int) -> StageTimings:
    samples = []
    for i in range(reps):
        start = time.perf_counter()
        ns.generate_scenario(seed_base + i)
        samples.append(time.perf_counter() - start)
    return StageTimings("1. scenario generation", samples)


def time_nsga2_optimization(reps: int, seed_base: int, n_generations: int) -> StageTimings:
    samples = []
    for i in range(reps):
        scenario = ns.generate_scenario(seed_base + i)
        scenario = ns.scenario_with_generations(scenario, n_generations)
        optimizer = NSGA2MissionOptimizer(scenario.env, scenario.optimizer_config)
        start = time.perf_counter()
        optimizer.optimize(seed=scenario.seed)
        samples.append(time.perf_counter() - start)
    return StageTimings(f"2. NSGA-II optimization ({n_generations} gens)", samples)


def time_coordinate_conversion(reps: int, seed_base: int) -> StageTimings:
    samples = []
    for i in range(reps):
        scenario = ns.generate_scenario(seed_base + i)
        offsets = [(t.x, t.y) for t in scenario.env.targets]
        start = time.perf_counter()
        for north_m, east_m in offsets:
            _offset_latlon(HOME_LAT, HOME_LON, north_m, east_m)
        samples.append(time.perf_counter() - start)
    return StageTimings("3. coordinate conversion", samples)


def time_mission_item_construction(reps: int, seed_base: int) -> StageTimings:
    samples = []
    for i in range(reps):
        scenario = ns.generate_scenario(seed_base + i)
        waypoints = [_offset_latlon(HOME_LAT, HOME_LON, t.x, t.y) for t in scenario.env.targets]
        start = time.perf_counter()
        items = [
            MissionItem(
                lat, lon, CRUISE_ALTITUDE_M, 5.0, True,
                float("nan"), float("nan"), MissionItem.CameraAction.NONE,
                float("nan"), float("nan"), ACCEPTANCE_RADIUS_M,
                float("nan"), float("nan"), MissionItem.VehicleAction.NONE,
            )
            for lat, lon in waypoints
        ]
        samples.append(time.perf_counter() - start)
        assert len(items) == len(waypoints)
    return StageTimings("4. mission-item construction", samples)


def time_telemetry_object_init(reps: int) -> StageTimings:
    """
    Local ``mavsdk.System()`` construction ONLY -- no ``connect()`` is called,
    so this measures object/client init overhead, not real connection
    handshake latency. Actual connection time depends on the radio/SITL and
    must be measured separately with hardware attached (see
    tests/telemetry_diagnostic.py's printed timing, or add --time-connect
    there if you want it instrumented).
    """
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        System()
        samples.append(time.perf_counter() - start)
    return StageTimings("5. telemetry client init (System(), no connection)", samples)


def print_table(stages: list[StageTimings]) -> None:
    header = f"{'Stage':<58} {'Mean(ms)':>10} {'Median(ms)':>11} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}  N"
    print(header)
    print("-" * len(header))
    for s in stages:
        print(
            f"{s.name:<58} {s.mean * 1000:>10.3f} {s.median * 1000:>11.3f} "
            f"{s.p95 * 1000:>10.3f} {s.minimum * 1000:>10.3f} {s.maximum * 1000:>10.3f}  {len(s.samples_s)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=30, help="repetitions per stage")
    parser.add_argument("--seed", type=int, default=1000, help="base seed; each rep uses seed+i")
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()

    print(f"Running {args.reps} repetitions per stage (base seed={args.seed})...\n")

    stages = [
        time_scenario_generation(args.reps, args.seed),
        time_nsga2_optimization(args.reps, args.seed, args.generations),
        time_coordinate_conversion(args.reps, args.seed),
        time_mission_item_construction(args.reps, args.seed),
        time_telemetry_object_init(args.reps),
    ]
    print_table(stages)
    print(
        "\nNote: stage 5 times local System() construction only -- it does NOT open a "
        "network/serial connection, so it never touches hardware. Real connection "
        "latency depends on your radio/SITL and is reported live (not averaged) by "
        "tests/telemetry_diagnostic.py."
    )


if __name__ == "__main__":
    main()
