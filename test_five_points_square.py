import asyncio
import math

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

from mission.flight.preflight_confirm import (
    print_mission_plan,
    print_qgc_clear_reminder,
    require_fly_confirmation,
    verify_disarmed_and_on_ground,
    verify_mission_uploaded,
)

SYSTEM_ADDRESS = "serial:///dev/cu.usbserial-DU0D65S7:57600"

CRUISE_ALTITUDE_M = 25.0
SQUARE_SIDE_M = 5.0
ACCEPTANCE_RADIUS_M = 3.0

_EARTH_RADIUS_M = 6_371_000.0


def _offset_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    d_lat = (north_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
    d_lon = (east_m / (_EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))) * (180.0 / math.pi)
    return lat_deg + d_lat, lon_deg + d_lon


async def main() -> None:
    drone = System()

    print(f"Connecting to Pixhawk on {SYSTEM_ADDRESS} ...")
    await drone.connect(system_address=SYSTEM_ADDRESS)

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone.")
            break

    print("Waiting for global position and home position lock...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("GPS and home position OK.")
            break

    print("Fetching home position...")
    async for home in drone.telemetry.home():
        home_lat, home_lon = home.latitude_deg, home.longitude_deg
        break
    print(f"Home: lat={home_lat:.7f}, lon={home_lon:.7f}")

    offsets = [
        (SQUARE_SIDE_M, 0.0),
        (SQUARE_SIDE_M, SQUARE_SIDE_M),
        (0.0, SQUARE_SIDE_M),
        (0.0, 0.0),
    ]

    waypoints = [(_offset_latlon(home_lat, home_lon, n, e), n, e) for n, e in offsets]

    print("Waypoints (5 m square):")
    for i, ((lat, lon), n, e) in enumerate(waypoints, start=1):
        print(f"  {i}: north={n:+.1f}m east={e:+.1f}m -> lat={lat:.7f}, lon={lon:.7f}")

    mission_items = []
    for (lat, lon), _n, _e in waypoints:
        mission_items.append(
            MissionItem(
                lat,
                lon,
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
        )

    print_mission_plan(mission_items)

    print("Uploading mission plan (4 waypoints)...")
    await drone.mission.upload_mission(MissionPlan(mission_items))
    await verify_mission_uploaded(drone, mission_items)

    await verify_disarmed_and_on_ground(drone)
    print_qgc_clear_reminder()
    require_fly_confirmation()

    print("Arming...")
    await drone.action.arm()

    print("Starting mission...")
    await drone.mission.start_mission()

    total_waypoints = len(mission_items)
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

    print("All 4 waypoints complete.")
    print("Mission finished. Auto-landing disabled -- take manual control to land.")


if __name__ == "__main__":
    asyncio.run(main())
