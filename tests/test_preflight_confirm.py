"""
Tests for mission.flight.preflight_confirm -- the shared pre-arm safety
checks used by every real-Pixhawk script (tests/nsga_pixhawk_mission.py,
test_five_points_random.py, test_five_points_square.py).

Uses lightweight fakes, no real/SITL vehicle needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mavsdk.mission import MissionItem, MissionPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission.flight.preflight_confirm import (
    require_fly_confirmation,
    verify_disarmed_and_on_ground,
    verify_mission_uploaded,
)


def _mi(lat: float, lon: float, alt: float = 25.0, accept_radius: float = 3.0) -> MissionItem:
    return MissionItem(
        lat, lon, alt, 5.0, False,
        float("nan"), float("nan"),
        MissionItem.CameraAction.NONE,
        float("nan"), float("nan"),
        accept_radius,
        float("nan"), float("nan"),
        MissionItem.VehicleAction.NONE,
    )


class _FakeTelemetry:
    def __init__(self, *, armed: bool | None, in_air: bool | None) -> None:
        self._armed = armed
        self._in_air = in_air

    async def armed(self):
        if self._armed is not None:
            yield self._armed

    async def in_air(self):
        if self._in_air is not None:
            yield self._in_air


class _FakeMission:
    def __init__(self, mission_items: list[MissionItem]) -> None:
        self._plan = MissionPlan(mission_items)

    async def download_mission(self):
        return self._plan


class _FakeSystem:
    def __init__(self, telemetry: _FakeTelemetry, mission: _FakeMission | None = None) -> None:
        self.telemetry = telemetry
        self.mission = mission


@pytest.mark.asyncio
async def test_verify_disarmed_and_on_ground_passes_when_disarmed_and_grounded():
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=False))
    await verify_disarmed_and_on_ground(drone)  # must not raise


@pytest.mark.asyncio
async def test_verify_disarmed_and_on_ground_raises_when_armed():
    drone = _FakeSystem(_FakeTelemetry(armed=True, in_air=False))
    with pytest.raises(RuntimeError, match="ARMED"):
        await verify_disarmed_and_on_ground(drone)


@pytest.mark.asyncio
async def test_verify_disarmed_and_on_ground_raises_when_in_air():
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=True))
    with pytest.raises(RuntimeError, match="IN AIR"):
        await verify_disarmed_and_on_ground(drone)


@pytest.mark.asyncio
async def test_verify_disarmed_and_on_ground_raises_when_telemetry_unavailable():
    """No armed()/in_air() sample at all (e.g. a dropped stream) must refuse
    to proceed rather than assume a safe state."""
    drone = _FakeSystem(_FakeTelemetry(armed=None, in_air=None))
    with pytest.raises(RuntimeError, match="Could not read"):
        await verify_disarmed_and_on_ground(drone)


@pytest.mark.asyncio
async def test_verify_mission_uploaded_passes_on_exact_match():
    items = [_mi(37.0001, -122.0001), _mi(37.0002, -122.0002)]
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=False), _FakeMission(items))
    await verify_mission_uploaded(drone, items)  # must not raise


@pytest.mark.asyncio
async def test_verify_mission_uploaded_tolerates_small_float_noise():
    """The vehicle rarely echoes floats back bit-for-bit -- small noise within
    tolerance must still be accepted as a match."""
    uploaded = [_mi(37.0001000, -122.0001000, alt=25.0)]
    echoed_back = [_mi(37.0001000004, -122.0000999997, alt=25.05)]
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=False), _FakeMission(echoed_back))
    await verify_mission_uploaded(drone, uploaded)  # must not raise


@pytest.mark.asyncio
async def test_verify_mission_uploaded_raises_on_waypoint_count_mismatch():
    """The classic symptom of a stale QGroundControl mission still on the vehicle."""
    uploaded = [_mi(37.0001, -122.0001)]
    stale_on_vehicle = [_mi(37.0001, -122.0001), _mi(37.5, -122.5)]
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=False), _FakeMission(stale_on_vehicle))
    with pytest.raises(RuntimeError, match="QGroundControl"):
        await verify_mission_uploaded(drone, uploaded)


@pytest.mark.asyncio
async def test_verify_mission_uploaded_raises_on_coordinate_mismatch():
    uploaded = [_mi(37.0001, -122.0001)]
    wrong_on_vehicle = [_mi(38.0, -120.0)]
    drone = _FakeSystem(_FakeTelemetry(armed=False, in_air=False), _FakeMission(wrong_on_vehicle))
    with pytest.raises(RuntimeError, match="waypoint 1"):
        await verify_mission_uploaded(drone, uploaded)


def test_require_fly_confirmation_accepts_exact_fly(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "FLY")
    require_fly_confirmation()  # must not raise


def test_require_fly_confirmation_rejects_anything_else(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "fly")
    with pytest.raises(SystemExit):
        require_fly_confirmation()


def test_require_fly_confirmation_rejects_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    with pytest.raises(SystemExit):
        require_fly_confirmation()
