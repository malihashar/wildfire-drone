"""
TEST 1 -- scenario generation properties (tests/nsga_scenario.py).

Verifies the random-scenario generator used by the NSGA-II benchmark and the
Pixhawk/SITL mission scripts: radius/spacing/count constraints, explicit
failure when constraints are impossible, randomization of every parameter
the real objectives consume, and seed reproducibility. Pure software -- no
MAVSDK, no hardware, no network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nsga_scenario as ns
from mission.config.settings import TargetGenerationConfig


def test_radius_m_is_a_configurable_module_constant():
    """RADIUS_M is a plain module-level float the operator can edit, not hardcoded per-call."""
    assert isinstance(ns.RADIUS_M, float)
    original = ns.RADIUS_M
    try:
        ns.RADIUS_M = 15.0
        scenario = ns.generate_scenario(seed=999)
        for t in scenario.env.targets:
            assert math.hypot(t.x, t.y) <= ns.RADIUS_M + 1e-6
    finally:
        ns.RADIUS_M = original


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 43, 1000])
def test_every_target_within_radius(seed):
    scenario = ns.generate_scenario(seed)
    for t in scenario.env.targets:
        dist = math.hypot(t.x, t.y)
        assert dist <= ns.RADIUS_M + 1e-6, f"target {t.id} is {dist:.3f}m from home (> {ns.RADIUS_M}m)"


@pytest.mark.parametrize("seed", [1, 2, 3, 42, 43, 1000])
def test_minimum_spacing_enforced(seed):
    scenario = ns.generate_scenario(seed)
    targets = scenario.env.targets
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            dist = math.hypot(targets[i].x - targets[j].x, targets[i].y - targets[j].y)
            assert dist >= ns.MIN_SPACING_M - 1e-6, (
                f"targets {targets[i].id} and {targets[j].id} are only {dist:.3f}m apart"
            )


@pytest.mark.parametrize("seed", range(20))
def test_target_count_within_bounds(seed):
    scenario = ns.generate_scenario(seed)
    assert ns.MIN_TARGETS <= scenario.n_targets <= ns.MAX_TARGETS


def test_generation_fails_explicitly_when_constraints_are_impossible():
    """A radius too small to fit MIN_TARGETS points MIN_SPACING_M apart must raise, not silently violate spacing."""
    original_radius, original_attempts = ns.RADIUS_M, ns.MAX_SAMPLE_ATTEMPTS
    try:
        ns.RADIUS_M = 0.5  # far too small for 7 points >= 5m apart
        ns.MAX_SAMPLE_ATTEMPTS = 200  # keep the test fast
        with pytest.raises(RuntimeError):
            ns.generate_scenario(seed=1)
    finally:
        ns.RADIUS_M, ns.MAX_SAMPLE_ATTEMPTS = original_radius, original_attempts


def test_target_locations_are_randomized_across_seeds():
    a = ns.generate_scenario(1)
    b = ns.generate_scenario(2)
    coords_a = {(round(t.x, 6), round(t.y, 6)) for t in a.env.targets}
    coords_b = {(round(t.x, 6), round(t.y, 6)) for t in b.env.targets}
    assert coords_a != coords_b


def test_severity_and_priority_are_randomized_within_configured_ranges():
    """damage_score/priority (the objective functions' inputs) come from the project's own
    TargetGenerationConfig ranges, not fixed constants, and vary target-to-target."""
    cfg = TargetGenerationConfig()
    dmg_lo, dmg_hi = cfg.damage_score_range
    pri_lo, pri_hi = cfg.priority_range

    scenario = ns.generate_scenario(seed=7)
    damages = [t.damage_score for t in scenario.env.targets]
    priorities = [t.priority for t in scenario.env.targets]

    for d in damages:
        assert dmg_lo <= d <= dmg_hi
    for p in priorities:
        assert pri_lo <= p <= pri_hi
    # Randomized, not all identical (would indicate a constant-fill bug).
    assert len(set(round(d, 6) for d in damages)) > 1
    assert len(set(round(p, 6) for p in priorities)) > 1


def test_travel_cost_is_derived_not_random():
    """SuppressionTarget.travel_cost is the distance-from-home the objectives can consume;
    it must be *computed*, not independently randomized (that would desync it from x/y)."""
    scenario = ns.generate_scenario(seed=11)
    for t in scenario.env.targets:
        assert t.travel_cost == pytest.approx(math.hypot(t.x, t.y))


@pytest.mark.parametrize("seed", [42, 43, 100])
def test_seed_reproducibility(seed):
    a = ns.generate_scenario(seed)
    b = ns.generate_scenario(seed)
    assert a.n_targets == b.n_targets
    for ta, tb in zip(a.env.targets, b.env.targets):
        assert ta.id == tb.id
        assert ta.x == tb.x and ta.y == tb.y
        assert ta.damage_score == tb.damage_score
        assert ta.priority == tb.priority


def test_scenario_with_generations_preserves_targets_and_severity():
    """The fair-comparison helper must change ONLY n_generations, not the scenario itself."""
    base = ns.generate_scenario(seed=42)
    varied = ns.scenario_with_generations(base, 300)
    assert varied.optimizer_config.n_generations == 300
    assert varied.env.targets == base.env.targets
    assert varied.env.drone == base.env.drone
