"""
TEST 5 -- mission-item construction WITHOUT execution.

Runs the exact same NSGA-II -> GPS-waypoint code path
``tests/nsga_pixhawk_mission.py``/``tests/nsga_mavsdk_sitl.py`` use
(``_plan_mission`` + ``_offset_latlon`` + the ``MissionItem`` construction
loop), but stops before any ``drone.connect()`` call -- no MAVSDK System is
ever created here. Prints a full validation table and asserts the
properties a bad waypoint list could violate.

Run as a script to print the table for manual inspection:
    python tests/test_mission_item_construction.py --seed 42

Run as a test suite:
    pytest tests/test_mission_item_construction.py -v
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mavsdk.mission import MissionItem
from nsga_pixhawk_mission import ACCEPTANCE_RADIUS_M, CRUISE_ALTITUDE_M, _offset_latlon, _plan_mission
from nsga_scenario import RADIUS_M


@dataclass(frozen=True)
class WaypointRow:
    index: int
    target_id: int
    north_m: float
    east_m: float
    lat: float
    lon: float
    altitude_m: float
    distance_from_home_m: float
    distance_from_prev_m: float


def build_mission_table(
    seed: int,
    home_lat: float = 37.7749,
    home_lon: float = -122.4194,
    altitude_m: float = CRUISE_ALTITUDE_M,
    deadline_s: float = 2.0,
) -> tuple[list[WaypointRow], list[MissionItem]]:
    """
    NSGA-II plan -> GPS waypoints -> MissionItems, stopping short of any
    MAVSDK System / connection. Returns (validation rows, MissionItem list).
    """
    target_ids, ne_offsets, _, _ = _plan_mission(seed, deadline_s)

    rows: list[WaypointRow] = []
    mission_items: list[MissionItem] = []
    prev_lat, prev_lon = home_lat, home_lon

    for i, (tid, (north_m, east_m)) in enumerate(zip(target_ids, ne_offsets)):
        lat, lon = _offset_latlon(home_lat, home_lon, north_m, east_m)
        dist_home = math.hypot(north_m, east_m)
        dist_prev = _haversine_m(prev_lat, prev_lon, lat, lon)
        rows.append(
            WaypointRow(
                index=i,
                target_id=tid,
                north_m=north_m,
                east_m=east_m,
                lat=lat,
                lon=lon,
                altitude_m=altitude_m,
                distance_from_home_m=dist_home,
                distance_from_prev_m=dist_prev,
            )
        )
        mission_items.append(
            MissionItem(
                lat, lon, altitude_m, 5.0, True,
                float("nan"), float("nan"), MissionItem.CameraAction.NONE,
                float("nan"), float("nan"), ACCEPTANCE_RADIUS_M,
                float("nan"), float("nan"), MissionItem.VehicleAction.NONE,
            )
        )
        prev_lat, prev_lon = lat, lon

    return rows, mission_items


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    _EARTH_RADIUS_M = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def print_validation_table(rows: list[WaypointRow]) -> None:
    header = (
        f"{'idx':>3} {'target_id':>9} {'north_m':>9} {'east_m':>9} "
        f"{'lat':>12} {'lon':>13} {'alt_m':>7} {'dist_home_m':>12} {'dist_prev_m':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.index:>3} {r.target_id:>9} {r.north_m:>9.3f} {r.east_m:>9.3f} "
            f"{r.lat:>12.7f} {r.lon:>13.7f} {r.altitude_m:>7.1f} "
            f"{r.distance_from_home_m:>12.3f} {r.distance_from_prev_m:>12.3f}"
        )


@pytest.fixture(scope="module")
def table():
    rows, items = build_mission_table(seed=42)
    return rows, items


def test_waypoint_order_matches_nsga2_selected_path(table):
    rows, _ = table
    target_ids, _, _, _ = _plan_mission(42, 2.0)
    assert [r.target_id for r in rows] == target_ids
    assert [r.index for r in rows] == list(range(len(rows)))


def test_altitude_is_correct_on_every_item(table):
    rows, items = table
    for r, item in zip(rows, items):
        assert r.altitude_m == pytest.approx(CRUISE_ALTITUDE_M)
        assert item.relative_altitude_m == pytest.approx(CRUISE_ALTITUDE_M)


def test_coordinates_are_valid(table):
    rows, items = table
    for r, item in zip(rows, items):
        assert -90.0 <= r.lat <= 90.0
        assert -180.0 <= r.lon <= 180.0
        assert item.latitude_deg == pytest.approx(r.lat)
        assert item.longitude_deg == pytest.approx(r.lon)


def test_radius_constraint_maintained(table):
    rows, _ = table
    for r in rows:
        assert r.distance_from_home_m <= RADIUS_M + 1e-6, (
            f"waypoint {r.index} (target {r.target_id}) is {r.distance_from_home_m:.3f}m "
            f"from home, exceeds RADIUS_M={RADIUS_M}"
        )


def test_no_waypoint_is_accidentally_duplicated(table):
    rows, _ = table
    seen_targets = [r.target_id for r in rows]
    assert len(seen_targets) == len(set(seen_targets)), "duplicate target_id in mission order"

    seen_coords = [(round(r.lat, 9), round(r.lon, 9)) for r in rows]
    assert len(seen_coords) == len(set(seen_coords)), "two waypoints resolved to the identical GPS coordinate"


def test_no_lat_lon_inversion(table):
    """A swapped (lon, lat) pair for a Bay-Area home would put latitude
    outside [-90, 90] or flip the sign convention -- catch it explicitly."""
    rows, _ = table
    home_lat, home_lon = 37.7749, -122.4194
    for r in rows:
        assert abs(r.lat - home_lat) < 1.0  # target is within 10m; 1 degree is a generous ceiling
        assert abs(r.lon - home_lon) < 1.0
        assert r.lat > 0  # San Francisco is in the northern hemisphere
        assert r.lon < 0  # and the western hemisphere


def test_mission_item_count_matches_target_count(table):
    rows, items = table
    assert len(rows) == len(items)
    target_ids, ne_offsets, _, _ = _plan_mission(42, 2.0)
    assert len(items) == len(target_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--home-lat", type=float, default=37.7749)
    parser.add_argument("--home-lon", type=float, default=-122.4194)
    args = parser.parse_args()

    rows, items = build_mission_table(args.seed, args.home_lat, args.home_lon)
    print(f"\nMission-item validation table (seed={args.seed}, no MAVSDK connection made):\n")
    print_validation_table(rows)
