"""
TEST 4 (audit) -- static + behavioral proof that tests/telemetry_diagnostic.py
is read-only: it must never reference ``.action.`` or ``.mission.`` (the only
MAVSDK namespaces that can arm/fly/upload a mission), and must fail cleanly
(not hang) when no vehicle is reachable.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import telemetry_diagnostic as td


def test_source_never_references_action_or_mission_namespaces():
    source = inspect.getsource(td)
    assert ".action." not in source, "telemetry_diagnostic.py must never touch drone.action.*"
    assert ".mission." not in source, "telemetry_diagnostic.py must never touch drone.mission.*"


def test_diagnostic_report_has_no_arm_or_takeoff_fields():
    """The report dataclass itself must not even have a place to record an arm/takeoff
    attempt -- reinforces read-only-by-construction, not just by convention."""
    fields = {f for f in vars(td.DiagnosticReport) if not f.startswith("_")}
    for forbidden in ("armed_by_us", "took_off", "flew", "mission_started"):
        assert forbidden not in fields


@pytest.mark.asyncio
async def test_unreachable_address_fails_cleanly_within_timeout():
    td.CONNECT_TIMEOUT_S = 1.0
    report = await asyncio.wait_for(
        td.run_diagnostic("serial:///dev/cu.this-does-not-exist:57600"), timeout=5.0
    )
    assert report.connected is False
    assert report.connect_error  # non-empty, human-readable
    td.CONNECT_TIMEOUT_S = 30.0
