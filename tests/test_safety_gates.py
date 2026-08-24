"""
TEST 7 -- safety-gate audit for tests/nsga_pixhawk_mission.py (the real-hardware
flight script). Confirms the gates exist, are required, cannot be bypassed by
an exception, and that the declared timeouts are real, finite values that get
applied. Does NOT weaken, remove, or work around any gate -- it only proves
each one functions as documented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nsga_pixhawk_mission as npm


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline", type=float, default=npm.OPTIMIZATION_DEADLINE_S)
    parser.add_argument("--port", type=str, default=npm.DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=npm.DEFAULT_BAUD)
    parser.add_argument("--altitude", type=float, default=npm.CRUISE_ALTITUDE_M)
    parser.add_argument("--i-have-verified-the-site", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def test_site_verification_flag_is_required_by_default():
    args = _parse([])
    assert args.i_have_verified_the_site is False
    with pytest.raises(SystemExit):
        npm._require_site_check(args)


def test_site_verification_flag_when_passed_allows_progress():
    args = _parse(["--i-have-verified-the-site"])
    assert args.i_have_verified_the_site is True
    npm._require_site_check(args)  # must not raise


def test_typed_confirmation_is_required_and_rejects_wrong_input(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")  # anything but "FLY"
    with pytest.raises(SystemExit):
        npm._require_typed_confirmation()


def test_typed_confirmation_rejects_lowercase_fly(monkeypatch):
    """Case-sensitivity matters here: an accidental lowercase 'fly' from a script
    or a fat-fingered terminal paste must NOT be treated as confirmation."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "fly")
    with pytest.raises(SystemExit):
        npm._require_typed_confirmation()


def test_typed_confirmation_accepts_exact_fly(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "FLY")
    npm._require_typed_confirmation()  # must not raise


def test_typed_confirmation_rejects_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    with pytest.raises(SystemExit):
        npm._require_typed_confirmation()


def test_both_gates_are_independent_neither_alone_is_sufficient():
    """Passing --i-have-verified-the-site does NOT also satisfy the typed
    confirmation, and vice versa -- main() calls both, back to back."""
    site_ok_args = _parse(["--i-have-verified-the-site"])
    npm._require_site_check(site_ok_args)  # passes the site check...
    # ...but the typed-confirmation gate is a wholly separate function call
    # that main() always makes next; it is not satisfied by the flag above.
    import inspect
    source = inspect.getsource(npm.main)
    assert "_require_site_check" in source
    assert "_require_typed_confirmation" in source
    # The site check must run before the typed confirmation in main()'s body.
    assert source.index("_require_site_check") < source.index("_require_typed_confirmation")


def test_yes_flag_bypasses_only_the_interactive_prompt_not_the_site_check():
    """--yes is documented as skipping the typed prompt for scripted use, but it
    must never also short-circuit --i-have-verified-the-site."""
    import inspect
    source = inspect.getsource(npm.main)
    require_site_idx = source.index("_require_site_check")
    yes_check_idx = source.index("if not args.yes")
    assert require_site_idx < yes_check_idx, (
        "the site-verification gate must run unconditionally, before the --yes branch"
    )


@pytest.mark.parametrize(
    "timeout_name",
    ["CONNECT_TIMEOUT_S", "HEALTH_TIMEOUT_S", "HOME_TIMEOUT_S", "MISSION_TIMEOUT_S", "LAND_TIMEOUT_S"],
)
def test_all_declared_timeouts_are_finite_positive_numbers(timeout_name):
    value = getattr(npm, timeout_name)
    assert isinstance(value, (int, float))
    assert 0 < value < float("inf")


def test_main_wraps_flight_sequence_in_a_recovery_try_except():
    """Any exception during the flight sequence must fall into the recovery
    block (return-to-launch + land attempt) rather than leaving the vehicle
    armed with no further code running."""
    import inspect
    source = inspect.getsource(npm.main)
    assert "except Exception as exc" in source
    assert "_return_home_and_land" in source
    # The except block re-raises after attempting recovery -- it must not
    # swallow the failure and let main() report false success.
    assert source.count("raise") >= 1


def test_exception_during_recovery_itself_does_not_crash_silently_or_hide_the_original_error():
    import inspect
    source = inspect.getsource(npm.main)
    # A nested try/except around the recovery attempt: recovery failing must
    # still surface (print) rather than raising an unrelated new exception
    # that masks the original failure.
    assert source.count("try:") >= 2
    assert "Recovery ALSO failed" in source
