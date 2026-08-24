"""
Shared pre-arm safety checks for scripts that fly a real Pixhawk.

Every real-hardware script (tests/nsga_pixhawk_mission.py, and the root-level
test_five_points_random.py / test_five_points_square.py) uses the exact same
sequence right before arming: print the waypoints that are about to be flown,
confirm the vehicle is disarmed and on the ground, confirm the mission the
vehicle actually has stored matches what was just uploaded (so a leftover
QGroundControl mission can't be flown by accident), remind the operator to
clear/replace any old QGroundControl mission if needed, and require a typed
"FLY" confirmation immediately before the arm call. Centralizing it here
means the three scripts can't drift out of sync on what "safe to arm" means.

Nothing here bypasses, weakens, or auto-clears anything on the vehicle -- a
verification failure always raises/aborts rather than proceeding.
"""

from __future__ import annotations

from mavsdk import System
from mavsdk.mission import MissionItem


def print_mission_plan(mission_items: list[MissionItem], *, description: str = "Mission plan about to be flown") -> None:
    """Prints the exact waypoints about to be flown, in flight order."""
    print(f"\n{description} ({len(mission_items)} waypoints):")
    for i, item in enumerate(mission_items, start=1):
        print(
            f"  {i}: lat={item.latitude_deg:.7f}, lon={item.longitude_deg:.7f}, "
            f"alt={item.relative_altitude_m:.1f}m, "
            f"acceptance_radius={item.acceptance_radius_m:.1f}m"
        )


async def verify_disarmed_and_on_ground(drone: System) -> None:
    """
    Confirms the vehicle is connected, disarmed, and on the ground.

    Raises RuntimeError (never proceeds) if either check fails -- arming must
    only ever be attempted from a known, freshly-confirmed disarmed/on-ground
    state, never assumed.
    """
    armed: bool | None = None
    async for is_armed in drone.telemetry.armed():
        armed = bool(is_armed)
        break

    in_air: bool | None = None
    async for airborne in drone.telemetry.in_air():
        in_air = bool(airborne)
        break

    if armed is None or in_air is None:
        raise RuntimeError(
            "Could not read armed/in_air telemetry from the vehicle -- refusing to arm "
            "without confirming its state first."
        )
    if armed:
        raise RuntimeError(
            "Vehicle already reports ARMED -- refusing to proceed. Disarm it (RC/GCS) and retry."
        )
    if in_air:
        raise RuntimeError(
            "Vehicle reports IN AIR -- refusing to proceed. Land and confirm it is on the ground first."
        )
    print(f"Vehicle state verified: armed={armed}, in_air={in_air} (disarmed, on the ground).")


async def verify_mission_uploaded(
    drone: System,
    expected_mission_items: list[MissionItem],
    *,
    lat_lon_tolerance_deg: float = 1e-6,
    alt_tolerance_m: float = 0.5,
) -> None:
    """
    Downloads the mission currently stored on the vehicle and confirms it
    matches ``expected_mission_items`` waypoint-for-waypoint (within a small
    tolerance -- the vehicle rarely echoes floats back bit-for-bit).

    Raises RuntimeError (never proceeds) on any mismatch, including a wrong
    waypoint count -- the most common symptom of a stale QGroundControl
    mission still being on the vehicle instead of the one just uploaded.
    """
    downloaded = await drone.mission.download_mission()
    got_items = downloaded.mission_items

    if len(got_items) != len(expected_mission_items):
        raise RuntimeError(
            f"Uploaded mission verification FAILED: vehicle reports {len(got_items)} waypoints, "
            f"expected {len(expected_mission_items)}. A stale QGroundControl (or other GCS) mission "
            "may still be on the vehicle -- clear/replace it in QGroundControl and re-upload before arming."
        )

    for i, (got, want) in enumerate(zip(got_items, expected_mission_items), start=1):
        lat_ok = abs(got.latitude_deg - want.latitude_deg) <= lat_lon_tolerance_deg
        lon_ok = abs(got.longitude_deg - want.longitude_deg) <= lat_lon_tolerance_deg
        alt_ok = abs(got.relative_altitude_m - want.relative_altitude_m) <= alt_tolerance_m
        if not (lat_ok and lon_ok and alt_ok):
            raise RuntimeError(
                f"Uploaded mission verification FAILED at waypoint {i}: vehicle reports "
                f"lat={got.latitude_deg:.7f}, lon={got.longitude_deg:.7f}, alt={got.relative_altitude_m:.1f}m; "
                f"expected lat={want.latitude_deg:.7f}, lon={want.longitude_deg:.7f}, alt={want.relative_altitude_m:.1f}m. "
                "A stale QGroundControl (or other GCS) mission may still be on the vehicle -- clear/replace it "
                "in QGroundControl and re-upload before arming."
            )

    print(f"Verified: {len(got_items)} uploaded waypoints on the vehicle match the intended mission.")


def print_qgc_clear_reminder() -> None:
    """Reminds the operator to clear/replace any old QGroundControl mission -- never done automatically."""
    print(
        "\nIMPORTANT: if QGroundControl (or any other ground control station) has a previously "
        "stored mission on this vehicle, clear or replace it now in QGroundControl before "
        "continuing. This script uploads its own mission and verifies it above, but it does NOT "
        "automatically clear anything on the vehicle -- that is a deliberate, manual step.\n"
    )


def require_fly_confirmation() -> None:
    """
    The typed "FLY" gate, required immediately before arming.

    Must be called right before the arm attempt (after the waypoint print,
    the disarmed/on-ground check, and the uploaded-mission verification),
    not earlier in the script, so it reflects the actual state the operator
    is about to arm into.
    """
    print("\n" + "=" * 70)
    print("THIS WILL ARM A REAL DRONE AND MAKE IT TAKE OFF.")
    print("Props on? Area clear? RC transmitter in hand with failsafe ready?")
    print("=" * 70)
    answer = input('Type "FLY" (all caps) to arm now, anything else aborts: ').strip()
    if answer != "FLY":
        raise SystemExit("Aborted: confirmation not given.")
