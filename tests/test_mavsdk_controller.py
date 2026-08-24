"""
Tests for the MAVSDK flight-control layer (mission.flight.mavsdk_controller).

Uses lightweight fakes for mavsdk.System's core/telemetry/action namespaces
instead of a real vehicle/SITL connection, so these run in CI without
hardware. Fakes reproduce just the async-generator / coroutine shapes the
controller actually calls (verified against the installed mavsdk package).
"""

from __future__ import annotations

import asyncio

import pytest
from mavsdk.action import ActionError, ActionResult
from mavsdk.core import ConnectionState
from mavsdk.telemetry import FixType, GpsInfo, Health, Position, StatusText, StatusTextType

from mission.flight.mavsdk_controller import (
    GeoOrigin,
    MissionFailureReason,
    arm_drone,
    mission,
)
from mission.replanning.executor import MissionExecutionResult


def _make_leg(tick: int, target_ids: tuple[int, ...], cell_path: tuple[tuple[int, int], ...]) -> MissionExecutionResult:
    return MissionExecutionResult(
        tick=tick,
        target_ids=target_ids,
        cell_path=cell_path,
        path_length=float(len(cell_path)),
        straight_line_length=float(len(cell_path)),
        feasible=True,
    )


class _FakeCore:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected

    async def connection_state(self):
        yield ConnectionState(is_connected=self.connected)


class _FakeTelemetry:
    def __init__(
        self,
        origin: GeoOrigin,
        *,
        converge: bool = True,
        healthy: bool = True,
        health: Health | None = None,
        gps_info: GpsInfo | None = None,
        status_texts: tuple[StatusText, ...] = (),
    ) -> None:
        self._origin = origin
        self._converge = converge
        self.lat = origin.home_latitude_deg
        self.lon = origin.home_longitude_deg
        self.alt = origin.home_absolute_altitude_m
        self._in_air = False
        self._healthy = healthy
        self._health = health or Health(True, True, True, True, True, True, True)
        self._gps_info = gps_info or GpsInfo(12, FixType.FIX_3D)
        self._status_texts = status_texts

    async def health_all_ok(self):
        yield self._healthy

    async def health(self):
        yield self._health

    async def gps_info(self):
        yield self._gps_info

    async def status_text(self):
        for status in self._status_texts:
            yield status

    async def in_air(self):
        yield self._in_air

    async def position(self):
        yield Position(self.lat, self.lon, self.alt, self.alt - self._origin.home_absolute_altitude_m)


class _FakeAction:
    def __init__(self, telemetry: _FakeTelemetry, *, converge: bool = True) -> None:
        self._telemetry = telemetry
        self._converge = converge
        self.armed = False
        self.rtl_called = False
        self.land_called = False

    async def arm(self):
        self.armed = True

    async def set_takeoff_altitude(self, alt):
        pass

    async def takeoff(self):
        self._telemetry._in_air = True

    async def goto_location(self, lat, lon, alt, yaw):
        if self._converge:
            self._telemetry.lat = lat
            self._telemetry.lon = lon
            self._telemetry.alt = alt

    async def return_to_launch(self):
        self.rtl_called = True
        self._telemetry._in_air = False

    async def land(self):
        self.land_called = True
        self._telemetry._in_air = False


class _FakeSystem:
    def __init__(
        self,
        origin: GeoOrigin,
        *,
        connected: bool = True,
        converge: bool = True,
        healthy: bool = True,
        health: Health | None = None,
        gps_info: GpsInfo | None = None,
        status_texts: tuple[StatusText, ...] = (),
    ) -> None:
        self.core = _FakeCore(connected=connected)
        self.telemetry = _FakeTelemetry(
            origin,
            converge=converge,
            healthy=healthy,
            health=health,
            gps_info=gps_info,
            status_texts=status_texts,
        )
        self.action = _FakeAction(self.telemetry, converge=converge)

    async def connect(self, system_address=None):
        pass


ORIGIN = GeoOrigin(
    home_latitude_deg=37.0,
    home_longitude_deg=-122.0,
    home_absolute_altitude_m=100.0,
    meters_per_cell=2.0,
)


@pytest.mark.asyncio
async def test_arm_drone_succeeds_when_healthy():
    drone = _FakeSystem(ORIGIN)
    result = await arm_drone(drone)
    assert result.success
    assert drone.action.armed


@pytest.mark.asyncio
async def test_mission_flies_routes_returns_home_and_lands():
    drone = _FakeSystem(ORIGIN)
    routes = [
        _make_leg(0, (1,), ((0, 0), (1, 0), (2, 0))),
        _make_leg(1, (2,), ((2, 0), (2, 1))),
    ]
    result = await mission(
        drone, routes, ORIGIN,
        waypoint_timeout_s=2.0, position_poll_interval_s=0.01,
    )
    assert result.success, result.message
    assert result.legs_completed == 2
    assert result.returned_home
    assert result.landed
    assert drone.action.rtl_called
    assert len(result.waypoints) == 3  # 2 + 1 non-start cells across both legs
    assert all(w.reached for w in result.waypoints)


@pytest.mark.asyncio
async def test_mission_reports_waypoint_timeout_and_attempts_recovery():
    drone = _FakeSystem(ORIGIN, converge=False)
    routes = [_make_leg(0, (1,), ((0, 0), (5, 5)))]
    result = await mission(
        drone, routes, ORIGIN,
        waypoint_timeout_s=0.05, position_poll_interval_s=0.01,
    )
    assert not result.success
    assert result.failure_reason == MissionFailureReason.WAYPOINT_TIMEOUT
    assert result.legs_completed == 0
    assert drone.action.rtl_called  # abort path attempted safe recovery


@pytest.mark.asyncio
async def test_mission_fails_fast_on_no_connection():
    drone = _FakeSystem(ORIGIN, connected=False)
    routes = [_make_leg(0, (1,), ((0, 0), (1, 0)))]
    result = await mission(drone, routes, ORIGIN, connect_timeout_s=0.05)
    assert not result.success
    assert result.failure_reason == MissionFailureReason.CONNECTION_LOST
    assert not drone.action.armed


@pytest.mark.asyncio
async def test_mission_fails_on_infeasible_route():
    drone = _FakeSystem(ORIGIN)
    infeasible = MissionExecutionResult(
        tick=0, target_ids=(1,), cell_path=(), path_length=0.0,
        straight_line_length=0.0, feasible=False,
    )
    result = await mission(drone, [infeasible], ORIGIN)
    assert not result.success
    assert result.failure_reason == MissionFailureReason.NO_ROUTES


@pytest.mark.asyncio
async def test_arm_drone_reports_rejected_arm():
    drone = _FakeSystem(ORIGIN)

    async def _reject():
        raise ActionError(ActionResult(ActionResult.Result.COMMAND_DENIED, "denied"), "arm")

    drone.action.arm = _reject
    result = await arm_drone(drone)
    assert not result.success
    assert "rejected" in result.message.lower() or "denied" in result.message.lower()


@pytest.mark.asyncio
async def test_arm_drone_reports_specific_reason_on_health_timeout():
    bad_health = Health(
        is_gyrometer_calibration_ok=True,
        is_accelerometer_calibration_ok=True,
        is_magnetometer_calibration_ok=True,
        is_local_position_ok=True,
        is_global_position_ok=False,
        is_home_position_ok=False,
        is_armable=False,
    )
    drone = _FakeSystem(
        ORIGIN,
        healthy=False,
        health=bad_health,
        gps_info=GpsInfo(3, FixType.NO_FIX),
        status_texts=(StatusText(StatusTextType.WARNING, "Preflight Fail: no global position"),),
    )
    result = await arm_drone(drone, health_timeout_s=0.05, poll_interval_s=0.01)

    assert not result.success
    assert not drone.action.armed
    # the specific failing checks must be named in the "Failing checks:" summary line,
    # not just the generic timeout sentence
    failing_line = next(line for line in result.message.splitlines() if "Failing checks:" in line)
    assert "global position" in failing_line
    assert "home position" in failing_line
    assert "armable" in failing_line
    assert "gyrometer" not in failing_line
    # GPS fix and STATUSTEXT context must also be surfaced
    assert "NO_FIX" in result.message
    assert "Preflight Fail: no global position" in result.message
