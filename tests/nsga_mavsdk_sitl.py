"""
PX4 SITL integration test: random scenario -> NSGA-II (2s wall-clock budget)
-> ordered path -> GPS waypoints -> MAVSDK mission -> PX4 SITL execution.

    Random scenario (tests/nsga_scenario.py)
          |
    NSGA-II (mission.optimizer.nsga2.NSGA2MissionOptimizer, REAL, unmodified)
          |
    2-second wall-clock optimization budget
          |
    Final selected solution (best_damage_plan() of the Pareto set)
          |
    Ordered path (MissionPlan.mission_order -> target_ids)
          |
    GPS waypoints (local north/east metres -> lat/lon, same flat-earth
                   conversion as test_five_points_random.py)
          |
    MAVSDK MissionItem / MissionPlan upload
          |
    PX4 SITL (UDP only -- see SYSTEM_ADDRESS)
          |
    Mission execution + progress/position monitoring
          |
    Return to launch + land

This script is intentionally separate from the D* Lite / grid-cell flight
layer in mission/flight/mavsdk_controller.py: NSGA-II's output here is a
flat ordered list of GPS suppression targets (no obstacle-aware routing is
in scope for this test), so it uploads a MAVSDK ``MissionPlan`` directly,
mirroring test_five_points_random.py's proven upload/monitor/RTL pattern
instead of D* Lite's grid-cell executor. ``arm_drone`` is reused from
mission.flight.mavsdk_controller since arming/health-check logic is
identical regardless of what produced the waypoints.

PX4 SITL ONLY -- never point SYSTEM_ADDRESS at a serial device or a real
Pixhawk. The physical flight-control layer (mission/flight/mavsdk_controller
.mission(), used with real D* Lite routes) is untouched by this script.

Usage:
    python tests/nsga_mavsdk_sitl.py                  # seed 42
    python tests/nsga_mavsdk_sitl.py --seed 7
    python tests/nsga_mavsdk_sitl.py --deadline 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.mission import MissionItem, MissionPlan

from mission.flight.mavsdk_controller import arm_drone
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from nsga_scenario import RADIUS_M, generate_scenario, run_nsga2_with_deadline
from nsga_visualization import save_mission_visualization

# PX4 SITL UDP only. Do NOT point this at /dev/cu.*, /dev/tty.*, a USB
# telemetry radio, or a physical Pixhawk.
SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"

CRUISE_ALTITUDE_M = 25.0
ACCEPTANCE_RADIUS_M = 3.0
OPTIMIZATION_DEADLINE_S = 2.0

CONNECT_TIMEOUT_S = 30.0
HEALTH_TIMEOUT_S = 60.0
HOME_TIMEOUT_S = 30.0
TAKEOFF_TIMEOUT_S = 30.0
MISSION_TIMEOUT_S = 300.0
LAND_TIMEOUT_S = 60.0

_EARTH_RADIUS_M = 6_371_000.0


def _offset_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Flat-earth local NE offset -> GPS. Identical formula to test_five_points_random.py."""
    d_lat = (north_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
    d_lon = (east_m / (_EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))) * (180.0 / math.pi)
    return lat_deg + d_lat, lon_deg + d_lon


async def _connect_loop(drone: System) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            return


async def _health_loop(drone: System) -> None:
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("GPS and home position OK.")
            return


async def _home_loop(drone: System) -> tuple[float, float]:
    async for home in drone.telemetry.home():
        print(f"Home: lat={home.latitude_deg:.7f}, lon={home.longitude_deg:.7f}")
        return home.latitude_deg, home.longitude_deg
    raise RuntimeError("Home position stream ended without a value.")


async def _connect_and_wait(drone: System) -> None:
    print(f"Connecting to PX4 SITL on {SYSTEM_ADDRESS} ...")
    await drone.connect(system_address=SYSTEM_ADDRESS)
    print("Waiting for connection...")
    await _connect_loop(drone)


async def _wait_connected(drone: System) -> None:
    # ``drone.connect()`` itself (not just the connection_state() poll) can
    # block indefinitely if the mavsdk_server subprocess never comes up --
    # both must be inside the same timeout, not just the poll loop after it.
    try:
        await asyncio.wait_for(_connect_and_wait(drone), timeout=CONNECT_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"No PX4 SITL connection on {SYSTEM_ADDRESS} within {CONNECT_TIMEOUT_S:.0f}s. "
            "Is `make px4_sitl` running?"
        ) from exc


async def _wait_global_position_ready(drone: System) -> None:
    print("Waiting for global position and home position lock...")
    try:
        await asyncio.wait_for(_health_loop(drone), timeout=HEALTH_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Global/home position not OK within {HEALTH_TIMEOUT_S:.0f}s."
        ) from exc


async def _fetch_home(drone: System) -> tuple[float, float]:
    print("Fetching home position...")
    try:
        return await asyncio.wait_for(_home_loop(drone), timeout=HOME_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"No home position within {HOME_TIMEOUT_S:.0f}s.") from exc


def _plan_mission(seed: int, deadline_s: float) -> tuple[list[int], list[tuple[float, float]], float, int]:
    """
    Random scenario -> NSGA-II under a wall-clock deadline -> ordered
    (north_m, east_m) waypoints. Returns (target_ids, ne_offsets, elapsed_s,
    generations_completed). No MAVSDK/SITL code runs in here.
    """
    scenario = generate_scenario(seed)
    print(f"Scenario: seed={seed}  targets={scenario.n_targets}")

    optimizer_seed = seed
    optimizer = NSGA2MissionOptimizer(scenario.env, scenario.optimizer_config)

    print(f"Running NSGA-II with a {deadline_s:.1f}s wall-clock deadline...")
    result, elapsed_s, n_gen = run_nsga2_with_deadline(optimizer, optimizer_seed, deadline_s)
    gen_per_s = n_gen / elapsed_s if elapsed_s > 0 else float("inf")
    print(
        f"NSGA-II stopped after {elapsed_s:.3f}s  generations={n_gen}  "
        f"gen/s={gen_per_s:.1f}  pareto_size={result.n_solutions}"
    )

    if result.n_solutions == 0:
        raise RuntimeError("NSGA-II produced no feasible mission within the deadline.")

    # Existing repo selection method: highest damage-prevented plan in the
    # Pareto set (see OptimizationResult.best_damage_plan in mission/optimizer/nsga2.py).
    best = result.best_damage_plan()
    print(f"Selected path: {best.mission_order}")
    print(best.summary())

    id_to_target = {t.id: t for t in scenario.env.targets}
    target_ids = list(best.mission_order)
    ne_offsets = [(id_to_target[tid].x, id_to_target[tid].y) for tid in target_ids]

    # Visualization only: runs exactly once, after NSGA-II is fully done and
    # the final order is already decided -- optimizer runtime above (elapsed_s)
    # was captured before this line, so plotting time is never counted in it.
    # Draws the SAME scenario.env.targets / target_ids about to become GPS
    # waypoints; nothing here feeds back into the optimizer or SITL execution.
    viz_path = save_mission_visualization(
        seed=seed,
        all_targets=scenario.env.targets,
        target_ids=target_ids,
        damage_prevented=best.objectives.damage_prevented,
        travel_distance=best.objectives.travel_distance,
        radius_m=RADIUS_M,
    )
    print(f"Saved mission visualization: {viz_path}")

    return target_ids, ne_offsets, elapsed_s, n_gen


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline", type=float, default=OPTIMIZATION_DEADLINE_S)
    args = parser.parse_args()

    # --- NSGA-II planning (no MAVSDK involved) ---
    target_ids, ne_offsets, elapsed_s, n_gen = _plan_mission(args.seed, args.deadline)

    # --- MAVSDK / PX4 SITL execution ---
    drone = System()
    try:
        await _wait_connected(drone)
        await _wait_global_position_ready(drone)
        home_lat, home_lon = await _fetch_home(drone)

        print(f"Converting {len(ne_offsets)} NSGA-II waypoints to GPS:")
        waypoints: list[tuple[float, float]] = []
        for tid, (north_m, east_m) in zip(target_ids, ne_offsets):
            lat, lon = _offset_latlon(home_lat, home_lon, north_m, east_m)
            waypoints.append((lat, lon))
            print(f"  T{tid}: north={north_m:+.2f}m east={east_m:+.2f}m -> lat={lat:.7f}, lon={lon:.7f}")

        mission_items = [
            MissionItem(
                lat, lon,
                CRUISE_ALTITUDE_M,
                5.0,
                True,
                float("nan"),
                float("nan"),
                MissionItem.CameraAction.NONE,
                float("nan"),
                float("nan"),
                ACCEPTANCE_RADIUS_M,
                float("nan"),
                float("nan"),
                MissionItem.VehicleAction.NONE,
            )
            for lat, lon in waypoints
        ]

        print(f"Uploading mission plan ({len(mission_items)} waypoints)...")
        await drone.mission.upload_mission(MissionPlan(mission_items))

        arm_result = await arm_drone(drone)
        if not arm_result.success:
            raise RuntimeError(f"Arm failed: {arm_result.message}")
        print("Armed.")

        await _takeoff(drone)

        print("Starting mission...")
        await drone.mission.start_mission()

        total_waypoints = len(mission_items)
        await asyncio.wait_for(_run_mission(drone, total_waypoints), timeout=MISSION_TIMEOUT_S)

        print(f"All {total_waypoints} waypoints complete.")
        print("Returning to launch and landing...")
        await _return_home_and_land(drone)
        print("Landed.")

    except Exception as exc:  # noqa: BLE001 - top-level mission safety net, must never leak the drone armed
        print(f"\nMission aborted due to error: {exc}")
        print("Attempting safe recovery (return-to-launch + land)...")
        try:
            await _return_home_and_land(drone)
            print("Recovered: returned home and landed.")
        except Exception as recovery_exc:  # noqa: BLE001 - best-effort safety net
            print(f"Recovery ALSO failed: {recovery_exc}. Manual intervention required in SITL.")
        raise

    print(
        f"\nDone. Plan: seed={args.seed} targets_visited={target_ids} "
        f"nsga2_time={elapsed_s:.3f}s generations={n_gen}"
    )


async def _run_mission(drone: System, total_waypoints: int) -> None:
    last_reported = -1

    async def report_progress() -> None:
        nonlocal last_reported
        async for progress in drone.mission.mission_progress():
            if progress.current != last_reported:
                last_reported = progress.current
                print(f"Waypoint {progress.current}/{total_waypoints}")
            if progress.current >= total_waypoints:
                return

    async def report_position() -> None:
        async for position in drone.telemetry.position():
            print(
                f"  pos: lat={position.latitude_deg:.7f}, "
                f"lon={position.longitude_deg:.7f}, "
                f"alt={position.relative_altitude_m:.1f}m"
            )

    position_task = asyncio.ensure_future(report_position())
    try:
        await report_progress()
    finally:
        position_task.cancel()


async def _return_home_and_land(drone: System) -> None:
    await drone.action.return_to_launch()
    try:
        await asyncio.wait_for(_wait_landed(drone), timeout=LAND_TIMEOUT_S)
    except asyncio.TimeoutError:
        print(f"Return-to-launch did not land within {LAND_TIMEOUT_S:.0f}s; forcing land().")
        await drone.action.land()
        await asyncio.wait_for(_wait_landed(drone), timeout=LAND_TIMEOUT_S)


async def _wait_landed(drone: System) -> None:
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            return


async def _wait_in_air(drone: System, *, timeout_s: float) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async for in_air in drone.telemetry.in_air():
        if in_air:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False


async def _takeoff(drone: System) -> None:
    try:
        await drone.action.set_takeoff_altitude(CRUISE_ALTITUDE_M)
    except ActionError:
        pass  # optional; fall back to the vehicle's configured takeoff altitude

    print("Taking off...")
    try:
        await drone.action.takeoff()
    except ActionError as exc:
        raise RuntimeError(f"Takeoff rejected: {exc}") from exc

    if not await _wait_in_air(drone, timeout_s=TAKEOFF_TIMEOUT_S):
        raise RuntimeError(f"Vehicle did not report in-air state within {TAKEOFF_TIMEOUT_S:.0f}s of takeoff.")
    print("Airborne.")


if __name__ == "__main__":
    asyncio.run(main())
