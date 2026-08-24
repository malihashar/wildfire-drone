

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

from mission.flight.mavsdk_controller import (
    HealthWatch,
    arm_drone,
    format_gps,
    health_checks,
    status_log,
    watch_gps_info,
    watch_status_text,
)
from mission.flight.preflight_confirm import (
    print_mission_plan,
    print_qgc_clear_reminder,
    verify_disarmed_and_on_ground,
    verify_mission_uploaded,
)
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from nsga_scenario import RADIUS_M, generate_scenario, run_nsga2_with_deadline
from nsga_visualization import save_mission_visualization

# Real Pixhawk over serial telemetry -- see test_pixhawk.py for provenance.
# NEVER point this at udpin://... (that's SITL) unless you deliberately want
# a simulated vehicle; never point tests/nsga_mavsdk_sitl.py at this address.
DEFAULT_SERIAL_PORT = "/dev/cu.usbserial-DU0D65S7"
DEFAULT_BAUD = 57600

CRUISE_ALTITUDE_M = 25.0
ACCEPTANCE_RADIUS_M = 3.0
WAYPOINT_HOLD_TIME_S = 5.0
OPTIMIZATION_DEADLINE_S = 2.0

CONNECT_TIMEOUT_S = 30.0
HEALTH_TIMEOUT_S = 60.0
HOME_TIMEOUT_S = 30.0
MISSION_TIMEOUT_S = 300.0
LAND_TIMEOUT_S = 60.0

# Health flags (from mission.flight.mavsdk_controller.health_checks) that must
# all read OK before it's safe to proceed towards arming. This mirrors exactly
# what arm_drone()'s own pre-arm wait requires, so a bad flag (e.g. an
# uncalibrated magnetometer) is caught here instead of only surfacing later as
# an arm rejection after the mission has already been uploaded.
REQUIRED_HEALTH_CHECKS = (
    "global position",
    "home position",
    "local position",
    "gyrometer calibration",
    "accelerometer calibration",
    "magnetometer calibration",
)

_EARTH_RADIUS_M = 6_371_000.0


def _build_serial_address(port: str, baud: int) -> str:
    """
    Normalizes ``port`` into a MAVSDK serial system address.

    Accepts either a bare device path (``/dev/cu.usbserial-XXXX``) or an
    already-prefixed and/or baud-suffixed address (e.g.
    ``serial:///dev/cu.usbserial-XXXX:57600``) and always returns exactly one
    ``serial://`` prefix and one baud suffix, so a caller passing a
    full address for ``--port`` can't double up the prefix or the baud.
    """
    device = port[len("serial://"):] if port.startswith("serial://") else port
    head, sep, tail = device.rpartition(":")
    if sep and tail.isdigit():
        device = head
    return f"serial://{device}:{baud}"


def _offset_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Flat-earth local NE offset -> GPS. Identical formula to test_five_points_random.py."""
    d_lat = (north_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
    d_lon = (east_m / (_EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))) * (180.0 / math.pi)
    return lat_deg + d_lat, lon_deg + d_lon


def _require_site_check(args: argparse.Namespace) -> None:
    if not args.i_have_verified_the_site:
        raise SystemExit(
            "\nRefusing to run: pass --i-have-verified-the-site once you have physically\n"
            "confirmed props/area/RC-failsafe/regulations (see the safety block at the\n"
            "top of this file). This is not a formality -- read it first.\n"
        )


def _require_typed_confirmation() -> None:
    print("\n" + "=" * 70)
    print("THIS WILL ARM A REAL DRONE AND MAKE IT TAKE OFF.")
    print("Props on? Area clear? RC transmitter in hand with failsafe ready?")
    print("=" * 70)
    answer = input('Type "FLY" (all caps) to proceed, anything else aborts: ').strip()
    if answer != "FLY":
        raise SystemExit("Aborted: confirmation not given.")


async def _connect_loop(drone: System) -> None:
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            return


def _required_check_status(health) -> dict[str, bool | None]:
    """
    Maps every ``REQUIRED_HEALTH_CHECKS`` name to its reported OK/FAIL state.

    Looks each name up explicitly rather than filtering ``health_checks()``'s
    output -- if a required name is ever missing from what ``health_checks()``
    reports (e.g. a future rename in mavsdk_controller.py), that entry maps to
    ``None`` here and is treated as failing everywhere below, never as
    "not applicable" or silently dropped.
    """
    available = dict(health_checks(health))
    return {name: available.get(name) for name in REQUIRED_HEALTH_CHECKS}


def _format_preflight_snapshot(health) -> str:
    lines = []
    for name, ok in _required_check_status(health).items():
        state = "OK" if ok is True else ("MISSING" if ok is None else "FAIL")
        lines.append(f"  {name}: {state}")
    return "Preflight health:\n" + "\n".join(lines)


async def _preflight_health_loop(drone: System, watch: HealthWatch) -> None:
    last_printed: str | None = None
    async for health in drone.telemetry.health():
        watch.health = health
        snapshot = _format_preflight_snapshot(health)
        if snapshot != last_printed:
            print(snapshot)
            last_printed = snapshot
        if all(ok is True for ok in _required_check_status(health).values()):
            print("Preflight health checks passed.")
            return


async def _home_loop(drone: System) -> tuple[float, float]:
    async for home in drone.telemetry.home():
        print(f"Home: lat={home.latitude_deg:.7f}, lon={home.longitude_deg:.7f}")
        return home.latitude_deg, home.longitude_deg
    raise RuntimeError("Home position stream ended without a value.")


async def _connect_and_wait(drone: System, system_address: str) -> None:
    print(f"Connecting to Pixhawk on {system_address} ...")
    await drone.connect(system_address=system_address)
    print("Waiting for connection...")
    await _connect_loop(drone)


async def _wait_connected(drone: System, system_address: str) -> None:
    try:
        await asyncio.wait_for(_connect_and_wait(drone, system_address), timeout=CONNECT_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"No connection to {system_address} within {CONNECT_TIMEOUT_S:.0f}s. "
            "Check the radio is plugged in and the port is correct (ls /dev/cu.*)."
        ) from exc


async def _wait_preflight_health_ready(drone: System) -> None:
    """
    Waits for every flag in ``REQUIRED_HEALTH_CHECKS`` to read OK, printing a
    live health snapshot and any vehicle STATUSTEXT messages while it waits.

    This is the same set of flags arm_drone() itself requires, checked here
    -- before mission upload -- so a real failure (e.g. magnetometer
    calibration) aborts safely up front instead of only being discovered
    after arming is attempted.
    """
    print("Waiting for preflight health checks (position, home, local position, gyro/accel/mag calibration)...")
    watch = HealthWatch()
    watchers = [
        asyncio.create_task(watch_gps_info(drone, watch)),
        asyncio.create_task(watch_status_text(drone, watch)),
    ]
    try:
        try:
            await asyncio.wait_for(_preflight_health_loop(drone, watch), timeout=HEALTH_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            if watch.health is not None:
                failing = [name for name, ok in _required_check_status(watch.health).items() if ok is not True]
                reason = ", ".join(failing) if failing else "unspecified (flags flipped between samples)"
                snapshot = _format_preflight_snapshot(watch.health)
            else:
                reason = ", ".join(REQUIRED_HEALTH_CHECKS) + " (no health telemetry received from the vehicle)"
                snapshot = "Preflight health:\n" + "\n".join(f"  {name}: MISSING" for name in REQUIRED_HEALTH_CHECKS)
            gps_line = format_gps(watch.gps_info) if watch.gps_info is not None else "  [GPS] no gps_info telemetry received"
            raise RuntimeError(
                f"Preflight health checks did not pass within {HEALTH_TIMEOUT_S:.0f}s. "
                f"Failing checks: {reason}.\n"
                f"{snapshot}\n{gps_line}\n"
                f"Recent vehicle status messages:\n{status_log(watch)}"
            ) from exc
    finally:
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)


async def _fetch_home(drone: System) -> tuple[float, float]:
    print("Fetching home position...")
    try:
        return await asyncio.wait_for(_home_loop(drone), timeout=HOME_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"No home position within {HOME_TIMEOUT_S:.0f}s.") from exc


def _plan_mission(seed: int, deadline_s: float) -> tuple[list[int], list[tuple[float, float]], float, int]:
    """
    Random scenario -> NSGA-II under a wall-clock deadline -> ordered
    (north_m, east_m) waypoints. No MAVSDK/hardware code runs in here.
    """
    scenario = generate_scenario(seed)
    print(f"Scenario: seed={seed}  targets={scenario.n_targets}")

    optimizer = NSGA2MissionOptimizer(scenario.env, scenario.optimizer_config)

    print(f"Running NSGA-II with a {deadline_s:.1f}s wall-clock deadline...")
    result, elapsed_s, n_gen = run_nsga2_with_deadline(optimizer, seed, deadline_s)
    gen_per_s = n_gen / elapsed_s if elapsed_s > 0 else float("inf")
    print(
        f"NSGA-II stopped after {elapsed_s:.3f}s  generations={n_gen}  "
        f"gen/s={gen_per_s:.1f}  pareto_size={result.n_solutions}"
    )

    if result.n_solutions == 0:
        raise RuntimeError("NSGA-II produced no feasible mission within the deadline.")

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
    # waypoints; nothing here feeds back into the optimizer or the flight plan.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline", type=float, default=OPTIMIZATION_DEADLINE_S)
    parser.add_argument("--port", type=str, default=DEFAULT_SERIAL_PORT, help="Serial device, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--altitude", type=float, default=CRUISE_ALTITUDE_M, help="Cruise altitude, metres AGL")
    parser.add_argument(
        "--i-have-verified-the-site",
        action="store_true",
        help="Required. Confirms you've checked props/area/RC-failsafe/regulations.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive typed 'FLY' confirmation (for scripted/CI use only -- NOT recommended for a real flight).",
    )
    args = parser.parse_args()

    _require_site_check(args)
    system_address = _build_serial_address(args.port, args.baud)

    # --- NSGA-II planning (no hardware involved) ---
    target_ids, ne_offsets, elapsed_s, n_gen = _plan_mission(args.seed, args.deadline)

    # --- MAVSDK / real Pixhawk execution ---
    drone = System()
    try:
        await _wait_connected(drone, system_address)
        await _wait_preflight_health_ready(drone)
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
                args.altitude,
                5.0,
                False,
                float("nan"),
                float("nan"),
                MissionItem.CameraAction.NONE,
                WAYPOINT_HOLD_TIME_S,
                float("nan"),
                ACCEPTANCE_RADIUS_M,
                float("nan"),
                float("nan"),
                MissionItem.VehicleAction.NONE,
            )
            for lat, lon in waypoints
        ]

        print_mission_plan(mission_items, description="NSGA-II mission plan about to be flown")

        print(f"Uploading mission plan ({len(mission_items)} waypoints)...")
        await drone.mission.upload_mission(MissionPlan(mission_items))
        await verify_mission_uploaded(drone, mission_items)

        await verify_disarmed_and_on_ground(drone)
        print_qgc_clear_reminder()

        if not args.yes:
            _require_typed_confirmation()
        else:
            print("--yes passed: skipping interactive confirmation (make sure you meant to).")

        arm_result = await arm_drone(drone)
        if not arm_result.success:
            raise RuntimeError(f"Arm failed: {arm_result.message}")
        print("Armed.")

        print("Starting mission...")
        await drone.mission.start_mission()

        total_waypoints = len(mission_items)
        await asyncio.wait_for(_run_mission(drone, total_waypoints), timeout=MISSION_TIMEOUT_S)

        print(f"All {total_waypoints} waypoints complete.")
        print("Mission finished. Auto-landing disabled -- take manual control to land.")

    except Exception as exc:  # noqa: BLE001 - top-level mission safety net, must never leak the drone armed
        print(f"\nMission aborted due to error: {exc}")
        airborne = await _confirm_airborne(drone)
        if airborne is True:
            print("Vehicle is confirmed airborne -- attempting safe recovery (return-to-launch + land)...")
            try:
                await _return_home_and_land(drone)
                print("Recovered: returned home and landed.")
            except Exception as recovery_exc:  # noqa: BLE001 - best-effort safety net
                print(
                    f"Recovery ALSO failed: {recovery_exc}. TAKE MANUAL RC CONTROL NOW."
                )
        elif airborne is False:
            print("Vehicle is not airborne -- no airborne recovery necessary (return-to-launch not issued).")
        else:
            print(
                "Could not confirm airborne state (telemetry unavailable) -- not issuing "
                "return-to-launch. If the vehicle is armed or flying, take manual RC control now."
            )
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


async def _confirm_airborne(drone: System, *, timeout_s: float = 5.0) -> bool | None:
    """
    Best-effort, timeout-bounded read of the vehicle's in-air state.

    Returns ``True``/``False`` only when telemetry actually reports it;
    returns ``None`` (never guesses) if it can't be read in time, e.g.
    because the failure that triggered recovery was the connection itself.
    Callers must treat ``None`` the same as "not airborne" for the purposes
    of deciding whether to issue return-to-launch -- RTL is only safe to
    attempt once the vehicle's in-air state is positively confirmed.
    """
    async def _read_once() -> bool:
        async for in_air in drone.telemetry.in_air():
            return bool(in_air)
        raise RuntimeError("in_air() stream ended without a value.")

    try:
        return await asyncio.wait_for(_read_once(), timeout=timeout_s)
    except Exception:  # noqa: BLE001 - telemetry may be unavailable this deep into a failure; treat as unknown
        return None


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


if __name__ == "__main__":
    asyncio.run(main())
