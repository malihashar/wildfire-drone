

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Deque, Sequence

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.core import ConnectionState

from mission.optimizer.dstar_lite import Cell
from mission.replanning.executor import MissionExecutionResult

_EARTH_RADIUS_M = 6_371_000.0


# ─────────────────────────── geo conversion ────────────────────────────


@dataclass(frozen=True)
class GeoOrigin:
    """
    Local-tangent-plane mapping from mission-grid cells to GPS.

    The mission grid (see ``mission.config.settings.GridConfig``) has no
    inherent georeference -- ``meters_per_cell`` and the home lat/lon are
    supplied by the deployment site, not derived from anything upstream.
    """

    home_latitude_deg: float
    home_longitude_deg: float
    home_absolute_altitude_m: float
    meters_per_cell: float = 1.0

    def cell_to_latlon(self, cell: Cell) -> tuple[float, float]:
        """Flat-earth (equirectangular) approximation -- adequate at mission-grid scale."""
        dx_m = cell[0] * self.meters_per_cell
        dy_m = cell[1] * self.meters_per_cell
        d_lat = (dy_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
        d_lon = (
            (dx_m / (_EARTH_RADIUS_M * math.cos(math.radians(self.home_latitude_deg))))
            * (180.0 / math.pi)
        )
        return self.home_latitude_deg + d_lat, self.home_longitude_deg + d_lon


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


# ────────────────────────────── results ─────────────────────────────────


class MissionFailureReason(Enum):
    NONE = "none"
    CONNECTION_LOST = "connection_lost"
    HEALTH_CHECK_FAILED = "health_check_failed"
    ARM_REJECTED = "arm_rejected"
    TAKEOFF_FAILED = "takeoff_failed"
    UNEXPECTED_FLIGHT_MODE = "unexpected_flight_mode"
    WAYPOINT_TIMEOUT = "waypoint_timeout"
    WAYPOINT_COMMAND_REJECTED = "waypoint_command_rejected"
    LANDING_FAILED = "landing_failed"
    ABORTED = "aborted"
    NO_ROUTES = "no_routes"


@dataclass(frozen=True)
class ArmResult:
    success: bool
    message: str


@dataclass(frozen=True)
class WaypointOutcome:
    leg_tick: int
    cell: Cell
    reached: bool
    distance_m: float
    elapsed_s: float


@dataclass
class MissionResult:
    success: bool
    failure_reason: MissionFailureReason
    message: str
    legs_completed: int
    legs_total: int
    waypoints: list[WaypointOutcome] = field(default_factory=list)
    returned_home: bool = False
    landed: bool = False


# ─────────────────────── arm_drone diagnostics ──────────────────────────


@dataclass
class HealthWatch:
    """Latest values from background telemetry subscriptions, for diagnostics only."""

    health: object | None = None
    gps_info: object | None = None
    status_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=20))


def health_checks(health) -> list[tuple[str, bool]]:
    return [
        ("global position", health.is_global_position_ok),
        ("home position", health.is_home_position_ok),
        ("local position", health.is_local_position_ok),
        ("gyrometer calibration", health.is_gyrometer_calibration_ok),
        ("accelerometer calibration", health.is_accelerometer_calibration_ok),
        ("magnetometer calibration", health.is_magnetometer_calibration_ok),
        ("armable", health.is_armable),
    ]


def format_health(health) -> str:
    return "\n".join(
        f"  [{'OK' if ok else 'FAIL'}] {name}" for name, ok in health_checks(health)
    )


def format_gps(gps_info) -> str:
    return f"  [GPS] fix={gps_info.fix_type.name} satellites={gps_info.num_satellites}"


async def watch_health(drone: System, watch: HealthWatch) -> None:
    async for health in drone.telemetry.health():
        watch.health = health


async def watch_gps_info(drone: System, watch: HealthWatch) -> None:
    async for gps_info in drone.telemetry.gps_info():
        watch.gps_info = gps_info


async def watch_status_text(drone: System, watch: HealthWatch) -> None:
    async for status in drone.telemetry.status_text():
        line = f"[STATUSTEXT/{status.type.name}] {status.text}"
        print(line)
        watch.status_lines.append(line)


def status_log(watch: HealthWatch) -> str:
    if not watch.status_lines:
        return "  (no STATUSTEXT messages received)"
    return "\n".join(watch.status_lines)


def health_failure_report(watch: HealthWatch) -> str:
    """Builds the specific underlying reason for a pre-arm health check timeout."""
    if watch.health is not None:
        failing = [name for name, ok in health_checks(watch.health) if not ok]
        reason = (
            ", ".join(failing)
            if failing
            else "unspecified (all known flags read OK, but health_all_ok() never reported ready)"
        )
        snapshot = format_health(watch.health)
    else:
        reason = "no health telemetry received from the vehicle"
        snapshot = "  (no health telemetry received)"

    gps_line = format_gps(watch.gps_info) if watch.gps_info is not None else "  [GPS] no gps_info telemetry received"

    return (
        "Pre-arm health check did not pass within timeout. "
        f"Failing checks: {reason}.\n"
        f"Health snapshot:\n{snapshot}\n{gps_line}\n"
        f"Recent vehicle status messages:\n{status_log(watch)}"
    )


# ────────────────────────────── arm_drone ───────────────────────────────


async def arm_drone(
    drone: System,
    *,
    health_timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
) -> ArmResult:
    """
    Verify the vehicle is healthy enough to arm, then arm it.

    Waits for ``telemetry.health_all_ok()`` to report ready (global position,
    home position, calibration, etc.) up to ``health_timeout_s`` before
    attempting ``action.arm()``. Does not connect and does not take off --
    connection is the caller's responsibility (see ``mission`` for the
    full connect -> verify -> arm -> fly sequence).

    While waiting, background subscriptions track the individual health
    flags, GPS fix, and STATUSTEXT messages so that a timeout or a rejected
    arm can be reported with the specific underlying reason rather than a
    generic message.
    """
    watch = HealthWatch()
    watchers = [
        asyncio.create_task(watch_health(drone, watch)),
        asyncio.create_task(watch_gps_info(drone, watch)),
        asyncio.create_task(watch_status_text(drone, watch)),
    ]

    try:
        deadline = asyncio.get_event_loop().time() + health_timeout_s
        healthy = False
        async for ok in drone.telemetry.health_all_ok():
            if ok:
                healthy = True
                break
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(poll_interval_s)

        if not healthy:
            return ArmResult(False, health_failure_report(watch))

        print("Pre-arm telemetry snapshot (health check passed):")
        if watch.health is not None:
            print(format_health(watch.health))
        if watch.gps_info is not None:
            print(format_gps(watch.gps_info))

        try:
            await drone.action.arm()
        except ActionError as exc:
            return ArmResult(
                False,
                f"Arm rejected: {exc}\nRecent vehicle status messages:\n{status_log(watch)}",
            )

        return ArmResult(True, "Armed.")
    finally:
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)


# ──────────────────────────── connection ────────────────────────────────


async def _connect_and_verify(drone: System, *, connect_timeout_s: float) -> bool:
    deadline = asyncio.get_event_loop().time() + connect_timeout_s
    async for state in drone.core.connection_state():
        state = state  # type: ConnectionState
        if state.is_connected:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
    return False


# ────────────────────────────── mission ─────────────────────────────────


ReplanProvider = Callable[[int], Awaitable["MissionExecutionResult | None"]]
"""Polled between waypoints with the index of the leg about to fly next.
Return a replacement ``MissionExecutionResult`` for that leg, or ``None`` to
keep flying the originally supplied route. No-op unless the caller wires one
up to the online D* Lite replanner (``mission.replanning.online_replanner``)."""


async def mission(
    drone: System,
    routes: Sequence[MissionExecutionResult],
    origin: GeoOrigin,
    *,
    system_address: str | None = None,
    connect_timeout_s: float = 30.0,
    cruise_altitude_m: float = 30.0,
    waypoint_tolerance_m: float = 3.0,
    waypoint_timeout_s: float = 60.0,
    position_poll_interval_s: float = 0.5,
    yaw_deg: float = 0.0,
    replan_provider: ReplanProvider | None = None,
) -> MissionResult:
    """
    Fly the ordered D* Lite routes for the NSGA-II-selected mission.

    ``routes`` is the exact, already-planned output of
    ``DStarLiteMissionExecutor.execute`` (or equivalent), one entry per
    mission leg, in execution order. This function performs no target
    selection and no path planning -- it only flies ``route.cell_path`` for
    each ``route`` in order, converting grid cells to GPS via ``origin``,
    then returns to ``origin`` (home) and lands.

    Every waypoint is confirmed via live telemetry (position + in-air state)
    against ``waypoint_tolerance_m``, not assumed reached after the command
    is sent. Loss of connection, an infeasible/empty route, an unexpected
    flight mode, or a waypoint that isn't reached within
    ``waypoint_timeout_s`` all abort the mission safely (attempt
    return-to-launch, else land in place) and are reported in the returned
    ``MissionResult`` rather than raised or silently swallowed.
    """
    legs_total = len(routes)
    outcomes: list[WaypointOutcome] = []

    if system_address is not None:
        await drone.connect(system_address=system_address)

    if not await _connect_and_verify(drone, connect_timeout_s=connect_timeout_s):
        return MissionResult(
            False, MissionFailureReason.CONNECTION_LOST,
            "Could not establish a connection to the vehicle.",
            0, legs_total, outcomes,
        )

    if legs_total == 0 or any(len(r.cell_path) < 1 for r in routes) or not any(
        len(r.cell_path) > 0 for r in routes
    ):
        return MissionResult(
            False, MissionFailureReason.NO_ROUTES,
            "No D* Lite routes supplied to execute.",
            0, legs_total, outcomes,
        )

    arm_result = await arm_drone(drone)
    if not arm_result.success:
        return MissionResult(
            False, MissionFailureReason.ARM_REJECTED, arm_result.message,
            0, legs_total, outcomes,
        )

    try:
        await drone.action.set_takeoff_altitude(cruise_altitude_m)
    except ActionError:
        pass  # optional; fall back to the vehicle's configured takeoff altitude

    try:
        await drone.action.takeoff()
    except ActionError as exc:
        return await _abort(
            drone, MissionFailureReason.TAKEOFF_FAILED, f"Takeoff rejected: {exc}",
            0, legs_total, outcomes,
        )

    if not await _wait_in_air(drone, timeout_s=waypoint_timeout_s):
        return await _abort(
            drone, MissionFailureReason.TAKEOFF_FAILED,
            "Vehicle did not report in-air state after takeoff.",
            0, legs_total, outcomes,
        )

    queue: Deque[MissionExecutionResult] = deque(routes)
    legs_completed = 0
    leg_index = 0

    while queue:
        if replan_provider is not None:
            try:
                replacement = await replan_provider(leg_index)
            except Exception:  # noqa: BLE001 - a misbehaving provider must not down the mission
                replacement = None
            if replacement is not None:
                queue[0] = replacement

        leg = queue.popleft()
        cells_to_fly = leg.cell_path[1:] if len(leg.cell_path) > 1 else leg.cell_path

        if not leg.feasible or not cells_to_fly:
            return await _abort(
                drone, MissionFailureReason.WAYPOINT_COMMAND_REJECTED,
                f"Leg {leg_index} (targets {leg.target_ids}) has no feasible route.",
                legs_completed, legs_total, outcomes,
            )

        for cell in cells_to_fly:
            outcome = await _fly_to_cell(
                drone, cell, origin,
                cruise_altitude_m=cruise_altitude_m,
                yaw_deg=yaw_deg,
                tolerance_m=waypoint_tolerance_m,
                timeout_s=waypoint_timeout_s,
                poll_interval_s=position_poll_interval_s,
                leg_tick=leg.tick,
            )
            outcomes.append(outcome)
            if not outcome.reached:
                return await _abort(
                    drone, MissionFailureReason.WAYPOINT_TIMEOUT,
                    f"Waypoint {cell} on leg {leg_index} (targets {leg.target_ids}) "
                    f"not reached within {waypoint_timeout_s:.0f}s "
                    f"(last distance {outcome.distance_m:.1f}m).",
                    legs_completed, legs_total, outcomes,
                )

            if not await _connected(drone):
                return await _abort(
                    drone, MissionFailureReason.CONNECTION_LOST,
                    "Lost connection to the vehicle mid-waypoint.",
                    legs_completed, legs_total, outcomes,
                )

        legs_completed += 1
        leg_index += 1

    returned_home, landed, land_msg = await _return_home_and_land(drone)
    if not landed:
        return MissionResult(
            False, MissionFailureReason.LANDING_FAILED, land_msg,
            legs_completed, legs_total, outcomes,
            returned_home=returned_home, landed=False,
        )

    return MissionResult(
        True, MissionFailureReason.NONE,
        f"Mission complete: {legs_completed}/{legs_total} legs flown, returned home and landed.",
        legs_completed, legs_total, outcomes,
        returned_home=returned_home, landed=True,
    )


# ───────────────────────────── helpers ──────────────────────────────────


async def _connected(drone: System) -> bool:
    async for state in drone.core.connection_state():
        return bool(state.is_connected)
    return False


async def _wait_in_air(drone: System, *, timeout_s: float, poll_interval_s: float = 0.5) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async for in_air in drone.telemetry.in_air():
        if in_air:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(poll_interval_s)
    return False


async def _fly_to_cell(
    drone: System,
    cell: Cell,
    origin: GeoOrigin,
    *,
    cruise_altitude_m: float,
    yaw_deg: float,
    tolerance_m: float,
    timeout_s: float,
    poll_interval_s: float,
    leg_tick: int,
) -> WaypointOutcome:
    lat, lon = origin.cell_to_latlon(cell)
    target_alt = origin.home_absolute_altitude_m + cruise_altitude_m

    start = asyncio.get_event_loop().time()
    try:
        await drone.action.goto_location(lat, lon, target_alt, yaw_deg)
    except ActionError:
        return WaypointOutcome(leg_tick, cell, False, math.inf, 0.0)

    deadline = start + timeout_s
    last_distance = math.inf
    while True:
        async for pos in drone.telemetry.position():
            last_distance = _haversine_m(pos.latitude_deg, pos.longitude_deg, lat, lon)
            alt_error = abs(pos.absolute_altitude_m - target_alt)
            if last_distance <= tolerance_m and alt_error <= max(tolerance_m, 2.0):
                elapsed = asyncio.get_event_loop().time() - start
                return WaypointOutcome(leg_tick, cell, True, last_distance, elapsed)
            break  # only take the latest reading per poll tick
        if asyncio.get_event_loop().time() >= deadline:
            elapsed = asyncio.get_event_loop().time() - start
            return WaypointOutcome(leg_tick, cell, False, last_distance, elapsed)
        await asyncio.sleep(poll_interval_s)


async def _return_home_and_land(drone: System) -> tuple[bool, bool, str]:
    try:
        await drone.action.return_to_launch()
    except ActionError as exc:
        try:
            await drone.action.land()
        except ActionError as land_exc:
            return False, False, f"Return-to-launch failed ({exc}); land also failed: {land_exc}"
        landed = await _wait_landed(drone, timeout_s=60.0)
        return False, landed, "Return-to-launch failed; landed in place." if landed else "Landing did not complete."

    landed = await _wait_landed(drone, timeout_s=90.0)
    if not landed:
        return True, False, "Return-to-launch did not complete landing within timeout."
    return True, True, "Returned home and landed."


async def _wait_landed(drone: System, *, timeout_s: float, poll_interval_s: float = 1.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(poll_interval_s)
    return False


async def _abort(
    drone: System,
    reason: MissionFailureReason,
    message: str,
    legs_completed: int,
    legs_total: int,
    outcomes: list[WaypointOutcome],
) -> MissionResult:
    """Best-effort safe recovery (return-to-launch, else land) after a mid-mission failure."""
    returned_home, landed = False, False
    try:
        returned_home, landed, _ = await _return_home_and_land(drone)
    except Exception:  # noqa: BLE001 - the vehicle may already be unreachable; report the original failure regardless
        pass

    return MissionResult(
        False, reason, message, legs_completed, legs_total, outcomes,
        returned_home=returned_home, landed=landed,
    )
