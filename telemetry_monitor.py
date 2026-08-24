import asyncio
from mavsdk import System

SERIAL = "serial:///dev/ttyUSB0:57600"



async def main():
    drone = System()

    print("Connecting...")
    await drone.connect(system_address=SERIAL)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("CONNECTED")
            break

    async def position():
        async for p in drone.telemetry.position():
            print(
                f"GPS: {p.latitude_deg:.7f}, "
                f"{p.longitude_deg:.7f}, "
                f"alt={p.absolute_altitude_m:.1f}m"
            )

    async def flight_mode():
        async for mode in drone.telemetry.flight_mode():
            print(f"Flight mode: {mode}")

    await asyncio.gather(position(), flight_mode())


if __name__ == "__main__":
    asyncio.run(main())
