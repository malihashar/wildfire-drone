"""
TEST 3 -- north/east metre-offset -> latitude/longitude conversion.

Tests the flat-earth ``_offset_latlon`` formula shared verbatim across
tests/nsga_pixhawk_mission.py, tests/nsga_mavsdk_sitl.py, and
test_five_points_random.py. Pure math -- no MAVSDK, no hardware, no network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nsga_pixhawk_mission import _EARTH_RADIUS_M, _offset_latlon

# A spread of real-world home coordinates: equator, mid-latitude (both
# hemispheres), high latitude, and the antimeridian-adjacent case.
HOME_COORDS = [
    (0.0, 0.0),
    (37.7749, -122.4194),  # San Francisco
    (-33.8688, 151.2093),  # Sydney (southern hemisphere)
    (64.1466, -21.9426),  # Reykjavik (high latitude, cos(lat) small-ish)
    (1.3521, 103.8198),  # Singapore (near equator)
]


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
def test_zero_offset_returns_exact_home(home_lat, home_lon):
    lat, lon = _offset_latlon(home_lat, home_lon, 0.0, 0.0)
    assert lat == pytest.approx(home_lat, abs=1e-12)
    assert lon == pytest.approx(home_lon, abs=1e-12)


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
@pytest.mark.parametrize(
    "north_m,east_m",
    [
        (10.0, 0.0),
        (-10.0, 0.0),
        (0.0, 10.0),
        (0.0, -10.0),
        (7.0, 7.0),
        (-7.0, 7.0),
        (7.0, -7.0),
        (-7.0, -7.0),
        (9.99, 0.0),
    ],
)
def test_round_trip_distance_matches_within_tolerance(home_lat, home_lon, north_m, east_m):
    lat, lon = _offset_latlon(home_lat, home_lon, north_m, east_m)
    expected_dist = math.hypot(north_m, east_m)
    actual_dist = _haversine_m(home_lat, home_lon, lat, lon)
    # Flat-earth approximation error is negligible at <=10m scale (<<1cm).
    assert actual_dist == pytest.approx(expected_dist, abs=0.01)


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
def test_no_target_exceeds_radius_after_conversion(home_lat, home_lon):
    radius_m = 10.0
    rng_points = [
        (radius_m * math.cos(theta), radius_m * math.sin(theta))
        for theta in (i * math.pi / 6 for i in range(12))
    ]
    for north_m, east_m in rng_points:
        lat, lon = _offset_latlon(home_lat, home_lon, north_m, east_m)
        dist = _haversine_m(home_lat, home_lon, lat, lon)
        assert dist <= radius_m + 0.01


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
def test_north_only_offset_changes_latitude_not_longitude(home_lat, home_lon):
    lat, lon = _offset_latlon(home_lat, home_lon, north_m=8.0, east_m=0.0)
    assert lat != pytest.approx(home_lat, abs=1e-12)
    assert lon == pytest.approx(home_lon, abs=1e-9)


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
def test_east_only_offset_changes_longitude_not_latitude(home_lat, home_lon):
    lat, lon = _offset_latlon(home_lat, home_lon, north_m=0.0, east_m=8.0)
    assert lon != pytest.approx(home_lon, abs=1e-12)
    assert lat == pytest.approx(home_lat, abs=1e-9)


def test_positive_north_increases_latitude():
    lat_pos, _ = _offset_latlon(0.0, 0.0, north_m=5.0, east_m=0.0)
    lat_neg, _ = _offset_latlon(0.0, 0.0, north_m=-5.0, east_m=0.0)
    assert lat_pos > 0.0 > lat_neg


def test_positive_east_increases_longitude_at_equator():
    _, lon_pos = _offset_latlon(0.0, 0.0, north_m=0.0, east_m=5.0)
    _, lon_neg = _offset_latlon(0.0, 0.0, north_m=0.0, east_m=-5.0)
    assert lon_pos > 0.0 > lon_neg


@pytest.mark.parametrize("home_lat,home_lon", HOME_COORDS)
def test_lat_lon_ordering_is_never_swapped(home_lat, home_lon):
    """A large north-only offset must stay within a plausible latitude delta and
    leave longitude untouched -- catches an accidental (lat, lon) <-> (lon, lat) swap."""
    lat, lon = _offset_latlon(home_lat, home_lon, north_m=10.0, east_m=0.0)
    # 10m of latitude is ~9e-5 degrees; a swap would instead show up as a
    # longitude-scale delta (which, away from the poles, is a different
    # magnitude) or would leave latitude completely unchanged.
    assert abs(lat - home_lat) < 1e-3
    assert abs(lat - home_lat) > 1e-7
    assert lon == pytest.approx(home_lon, abs=1e-9)
