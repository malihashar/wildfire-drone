"""
TEST 4 -- telemetry-only diagnostic. Connects to a vehicle (real Pixhawk over
serial, or PX4 SITL over UDP) and READS telemetry only.

This script NEVER arms, NEVER takes off, NEVER starts motors, and NEVER
uploads or starts a mission. It only calls read-only MAVSDK telemetry/core
APIs (``core.connection_state``, ``telemetry.health``, ``telemetry.position``,
``telemetry.home``, ``telemetry.armed``, ``telemetry.in_air``,
``telemetry.gps_info``, ``telemetry.battery``) and prints a diagnostic
report. Safe to run against a real telemetry radio at any time.

Usage:
    python tests/telemetry_diagnostic.py                                  # real Pixhawk, default port
    python tests/telemetry_diagnostic.py --address serial:///dev/cu.usbserial-XXXX:57600
    python tests/telemetry_diagnostic.py --address udpin://0.0.0.0:14540  # PX4 SITL
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mavsdk import System

DEFAULT_SERIAL_PORT = "serial:///dev/cu.usbserial-DU0D65S7:57600"
DEFAULT_BAUD = 57600

CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 15.0


@dataclass
class DiagnosticReport:
    address: str
    connected: bool = False
    connect_error: str | None = None

    gps_received: bool = False
    gps_fix_type: str | None = None
    gps_num_satellites: int | None = None

    home_received: bool = False
    home_lat: float | None = None
    home_lon: float | None = None

    health_received: bool = False
    health_fields: dict = field(default_factory=dict)

    position_received: bool = False
    position_lat: float | None = None
    position_lon: float | None = None
    position_rel_alt_m: float | None = None

    armed_state_received: bool = False
    armed: bool | None = None

    battery_received: bool = False
    battery_percent: float | None = None

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("TELEMETRY DIAGNOSTIC REPORT (read-only -- nothing was armed)")
        print("=" * 60)
        print(f"Address:            {self.address}")
        print(f"Connected:          {self.connected}" + (f"  (error: {self.connect_error})" if self.connect_error else ""))
        if not self.connected:
            print("=" * 60)
            return
        print(f"GPS received:       {self.gps_received}  fix_type={self.gps_fix_type}  satellites={self.gps_num_satellites}")
        print(f"Home received:      {self.home_received}  lat={self.home_lat}  lon={self.home_lon}")
        print(f"Health received:    {self.health_received}  {self.health_fields}")
        print(
            f"Position received:  {self.position_received}  "
            f"lat={self.position_lat}  lon={self.position_lon}  rel_alt_m={self.position_rel_alt_m}"
        )
        print(f"Armed state:        {self.armed_state_received}  armed={self.armed}")
        print(f"Battery:            {self.battery_received}  percent={self.battery_percent}")
        print("=" * 60)


async def _read_one(agen, timeout_s: float):
    """Read exactly one item from an async generator, with a timeout. None on timeout/failure."""
    try:
        return await asyncio.wait_for(agen.__anext__(), timeout=timeout_s)
    except (asyncio.TimeoutError, StopAsyncIteration, Exception):
        return None


async def run_diagnostic(address: str) -> DiagnosticReport:
    report = DiagnosticReport(address=address)
    drone = System()

    print(f"Connecting (read-only) to {address} ...")
    try:
        await asyncio.wait_for(drone.connect(system_address=address), timeout=CONNECT_TIMEOUT_S)
    except asyncio.TimeoutError:
        report.connect_error = f"connect() did not return within {CONNECT_TIMEOUT_S:.0f}s"
        return report
    except Exception as exc:  # noqa: BLE001 - reported in the diagnostic, not raised
        report.connect_error = str(exc) or f"{type(exc).__name__} (no message)"
        return report

    try:
        state = await asyncio.wait_for(_wait_for_connected(drone), timeout=CONNECT_TIMEOUT_S)
    except asyncio.TimeoutError:
        report.connect_error = f"No connection within {CONNECT_TIMEOUT_S:.0f}s"
        return report

    if state is None:
        report.connect_error = "connection_state stream ended without ever reporting is_connected"
        return report
    report.connected = True

    gps = await _read_one(drone.telemetry.gps_info(), READ_TIMEOUT_S)
    if gps is not None:
        report.gps_received = True
        report.gps_fix_type = str(gps.fix_type)
        report.gps_num_satellites = gps.num_satellites

    health = await _read_one(drone.telemetry.health(), READ_TIMEOUT_S)
    if health is not None:
        report.health_received = True
        report.health_fields = {
            "gyrometer_calibration_ok": health.is_gyrometer_calibration_ok,
            "accelerometer_calibration_ok": health.is_accelerometer_calibration_ok,
            "magnetometer_calibration_ok": health.is_magnetometer_calibration_ok,
            "local_position_ok": health.is_local_position_ok,
            "global_position_ok": health.is_global_position_ok,
            "home_position_ok": health.is_home_position_ok,
            "armable": health.is_armable,
        }

    home = await _read_one(drone.telemetry.home(), READ_TIMEOUT_S)
    if home is not None:
        report.home_received = True
        report.home_lat = home.latitude_deg
        report.home_lon = home.longitude_deg

    position = await _read_one(drone.telemetry.position(), READ_TIMEOUT_S)
    if position is not None:
        report.position_received = True
        report.position_lat = position.latitude_deg
        report.position_lon = position.longitude_deg
        report.position_rel_alt_m = position.relative_altitude_m

    armed = await _read_one(drone.telemetry.armed(), READ_TIMEOUT_S)
    if armed is not None:
        report.armed_state_received = True
        report.armed = bool(armed)

    battery = await _read_one(drone.telemetry.battery(), READ_TIMEOUT_S)
    if battery is not None:
        report.battery_received = True
        report.battery_percent = battery.remaining_percent

    return report


async def _wait_for_connected(drone: System):
    """Return the first ConnectionState with is_connected=True. The caller wraps this
    whole coroutine in asyncio.wait_for -- if connection_state() never yields at all
    (e.g. a dead channel), wait_for's own timeout still fires because it cancels this
    coroutine outright, not because this function polls a deadline internally."""
    async for state in drone.core.connection_state():
        if state.is_connected:
            return state


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--address",
        type=str,
        default=f"serial://{DEFAULT_SERIAL_PORT}:{DEFAULT_BAUD}",
        help="MAVSDK system_address, e.g. serial:///dev/cu.usbserial-XXXX:57600 or udpin://0.0.0.0:14540",
    )
    args = parser.parse_args()

    report = await run_diagnostic(args.address)
    report.print_report()


if __name__ == "__main__":
    asyncio.run(main())
