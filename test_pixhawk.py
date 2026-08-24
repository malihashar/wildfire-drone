import asyncio
from mavsdk import System


async def main():
    drone = System()

    print("Connecting to Pixhawk...")

    await drone.connect(
        system_address="serial:///dev/cu.usbserial-DU0D65S7:57600"
    )

    print("Waiting for connection...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("CONNECTED TO PIXHAWK!")
            break

    async for position in drone.telemetry.position():
        print(
            f"Position: "
            f"lat={position.latitude_deg}, "
            f"lon={position.longitude_deg}, "
            f"alt={position.absolute_altitude_m}"
        )


if __name__ == "__main__":
    asyncio.run(main())
