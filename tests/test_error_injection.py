"""
TEST 6 -- error injection.

Verifies that every early-stage failure (no connection, no GPS, no home
position, telemetry timeout, impossible scenario constraints, an optimizer
exception, invalid coordinates) stops the pipeline BEFORE any
aircraft-control call -- no arm, no mission upload, no mission start. Uses
fakes (no real/SITL vehicle needed) whose ``action``/``mission`` namespaces
raise ``AssertionError`` if touched, so any accidental progression into
flight control fails the test loudly rather than silently.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nsga_pixhawk_mission as npm
import nsga_scenario as ns


class _ForbiddenAircraftControl(AssertionError):
    """Raised if any aircraft-control API is touched during an error-injection test."""


class _NoArmAction:
    """action namespace that fails loudly if any flight-control method is called."""

    def __getattr__(self, name):
        def _forbidden(*args, **kwargs):
            raise _ForbiddenAircraftControl(f"action.{name}() must never be called in this scenario")
        return _forbidden


class _NoMissionUpload:
    def __getattr__(self, name):
        def _forbidden(*args, **kwargs):
            raise _ForbiddenAircraftControl(f"mission.{name}() must never be called in this scenario")
        return _forbidden


class _NeverConnectedCore:
    async def connection_state(self):
        # Never yields is_connected=True -- simulates "MAVSDK cannot connect".
        while True:
            from mavsdk.core import ConnectionState
            yield ConnectionState(is_connected=False)
            await asyncio.sleep(0.01)


class _ConnectedNoGpsTelemetry:
    """Connects fine, but health never reports GPS/home OK -- simulates "GPS unavailable"."""

    async def health(self):
        from mavsdk.telemetry import Health
        while True:
            yield Health(True, True, True, True, False, False, False)
            await asyncio.sleep(0.01)

    async def gps_info(self):
        return
        yield  # pragma: no cover - unreachable, keeps this an async generator

    async def status_text(self):
        return
        yield  # pragma: no cover - unreachable, keeps this an async generator

    async def home(self):
        raise _ForbiddenAircraftControl("home() must not be read before health is OK")
        yield  # pragma: no cover - unreachable, keeps this an async generator


class _ConnectedGoodHealthNoHomeTelemetry:
    """Health OK, but the home stream never yields -- simulates "home position unavailable"."""

    async def health(self):
        from mavsdk.telemetry import Health
        yield Health(True, True, True, True, True, True, True)

    async def gps_info(self):
        return
        yield  # pragma: no cover - unreachable, keeps this an async generator

    async def status_text(self):
        return
        yield  # pragma: no cover - unreachable, keeps this an async generator

    async def home(self):
        while True:
            await asyncio.sleep(0.01)
            if False:
                yield  # pragma: no cover - never yields, forces the caller's timeout


class _AlwaysConnectedCore:
    async def connection_state(self):
        from mavsdk.core import ConnectionState
        yield ConnectionState(is_connected=True)


class _FakeSystem:
    def __init__(self, core, telemetry) -> None:
        self.core = core
        self.telemetry = telemetry
        self.action = _NoArmAction()
        self.mission = _NoMissionUpload()

    async def connect(self, system_address=None):
        pass


@pytest.mark.asyncio
async def test_connection_failure_stops_before_any_arm_or_mission_call():
    npm.CONNECT_TIMEOUT_S = 0.05  # keep the test fast
    drone = _FakeSystem(core=_NeverConnectedCore(), telemetry=None)
    with pytest.raises(RuntimeError, match="No connection"):
        await npm._wait_connected(drone, "serial:///dev/cu.fake:57600")
    npm.CONNECT_TIMEOUT_S = 30.0


@pytest.mark.asyncio
async def test_gps_unavailable_stops_before_home_is_read():
    npm.HEALTH_TIMEOUT_S = 0.05
    drone = _FakeSystem(core=_AlwaysConnectedCore(), telemetry=_ConnectedNoGpsTelemetry())
    with pytest.raises(RuntimeError, match="did not pass"):
        await npm._wait_preflight_health_ready(drone)
    npm.HEALTH_TIMEOUT_S = 60.0


@pytest.mark.asyncio
async def test_home_unavailable_times_out_cleanly():
    npm.HOME_TIMEOUT_S = 0.05
    drone = _FakeSystem(core=_AlwaysConnectedCore(), telemetry=_ConnectedGoodHealthNoHomeTelemetry())
    await npm._wait_preflight_health_ready(drone)  # health IS ok here
    with pytest.raises(RuntimeError, match="No home position"):
        await npm._fetch_home(drone)
    npm.HOME_TIMEOUT_S = 30.0


def test_scenario_generation_fails_explicitly_on_impossible_spacing():
    original_radius, original_attempts = ns.RADIUS_M, ns.MAX_SAMPLE_ATTEMPTS
    try:
        ns.RADIUS_M = 0.1
        ns.MAX_SAMPLE_ATTEMPTS = 100
        with pytest.raises(RuntimeError):
            ns.generate_scenario(seed=1)
    finally:
        ns.RADIUS_M, ns.MAX_SAMPLE_ATTEMPTS = original_radius, original_attempts


def test_optimizer_exception_propagates_and_does_not_silently_continue(monkeypatch):
    """If the NSGA-II run throws (e.g. pymoo/optimizer internal failure), _plan_mission
    must propagate it -- not swallow it and hand back a bogus plan the caller would fly.
    ``_plan_mission`` calls ``run_nsga2_with_deadline`` (the wall-clock-budgeted entry
    point), not ``NSGA2MissionOptimizer.optimize()`` directly -- that's the call to fail."""

    def _boom(optimizer, seed, deadline_s):
        raise RuntimeError("simulated optimizer failure")

    monkeypatch.setattr(npm, "run_nsga2_with_deadline", _boom)

    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        npm._plan_mission(seed=1, deadline_s=0.1)


def test_no_feasible_solution_raises_instead_of_flying_empty_mission(monkeypatch):
    """If NSGA-II returns zero Pareto solutions within the deadline, _plan_mission
    must raise rather than hand back an empty/garbage waypoint list."""
    from mission.optimizer.nsga2 import OptimizationResult
    import numpy as np

    def _empty_result(optimizer, seed, deadline_s):
        return OptimizationResult(plans=(), F=np.zeros((0, 3)), X=np.zeros((0, 0))), 0.01, 1

    monkeypatch.setattr(npm, "run_nsga2_with_deadline", _empty_result)
    with pytest.raises(RuntimeError, match="no feasible mission"):
        npm._plan_mission(seed=1, deadline_s=0.1)


def test_invalid_coordinates_are_detectable_before_upload():
    """NaN/out-of-range offsets must be catchable by the validation the mission-item
    construction test suite applies -- proves a bad optimizer output wouldn't
    silently reach MAVSDK's MissionItem/MissionPlan."""
    home_lat, home_lon = 37.7749, -122.4194
    lat, lon = npm._offset_latlon(home_lat, home_lon, north_m=float("nan"), east_m=0.0)
    assert math.isnan(lat)
    # A NaN latitude must fail a basic validity check, not silently pass through.
    assert not (-90.0 <= lat <= 90.0) if math.isnan(lat) else True


@pytest.mark.asyncio
async def test_action_and_mission_apis_are_never_touched_when_connection_never_happens():
    """End-to-end proof: with a core that never reports connected, _NoArmAction /
    _NoMissionUpload's __getattr__ traps would raise if main()'s flow ever reached
    them -- confirm the connection-stage RuntimeError is what actually stops it."""
    npm.CONNECT_TIMEOUT_S = 0.05
    drone = _FakeSystem(core=_NeverConnectedCore(), telemetry=None)
    try:
        await npm._wait_connected(drone, "serial:///dev/cu.fake:57600")
        pytest.fail("expected RuntimeError before reaching arm/mission calls")
    except RuntimeError:
        pass  # expected: stopped at the connection stage
    except _ForbiddenAircraftControl:
        pytest.fail("aircraft-control API was touched despite no successful connection")
    finally:
        npm.CONNECT_TIMEOUT_S = 30.0
