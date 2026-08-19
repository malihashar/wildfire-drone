"""
Tests for the architecture-correctness fixes:

  1. P(fire) vs. expected-damage-proxy separation
  2. PredictionRiskMap preserved through the pipeline
  3. D* Lite executor genuine incremental reuse (not fresh-per-call)
  4. Risk-weighted D* Lite traversal cost
  5. NSGA-II surrogate vs. D* Lite actual travel cost recorded separately
  6. Permutation decoding represents a genuine feasible subset, not just order
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission.config.settings import OptimizerConfig
from mission.optimizer.chromosome import decode_permutation
from mission.optimizer.dstar_lite import DStarLite, DStarLiteSession
from mission.replanning.executor import (
    DStarLiteMissionExecutor,
    MissionExecutionRequest,
)
from mission.simulation.prediction_source import (
    ConvLSTMSourceConfig,
    ExpectedDamageConfig,
    PredictionRiskMap,
    _compute_risk_grid,
)
from mission.simulation.targets import DroneState, SuppressionTarget


# ── Issue 1: P(fire) vs expected-damage-proxy ──────────────────────────────


class TestExpectedDamageProxy(unittest.TestCase):
    def test_probability_grid_stays_a_probability(self):
        """The raw ConvLSTM output must never be mutated by the proxy transform."""
        pred = np.array([[0.9, 0.1], [0.5, 0.0]], dtype=np.float32)
        pred_copy = pred.copy()
        severity = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        fuel = np.array([[0.8, 0.8], [0.8, 0.8]], dtype=np.float32)
        norm = {
            "potential_ros": {"min": 0.0, "max": 1.0},
            "vegetation_density": {"min": 0.0, "max": 1.0},
        }
        _compute_risk_grid(pred, severity, fuel, norm, ExpectedDamageConfig())
        np.testing.assert_array_equal(pred, pred_copy)
        self.assertTrue(bool(np.all((pred >= 0) & (pred <= 1))))

    def test_damage_score_is_not_probability_when_factors_differ_from_one(self):
        pred = np.array([[0.9]], dtype=np.float32)
        severity = np.array([[0.5]], dtype=np.float32)  # normalizes to 0.5 given [0,1] range
        fuel = np.array([[0.4]], dtype=np.float32)
        norm = {
            "potential_ros": {"min": 0.0, "max": 1.0},
            "vegetation_density": {"min": 0.0, "max": 1.0},
        }
        risk = _compute_risk_grid(pred, severity, fuel, norm, ExpectedDamageConfig())
        # 0.9 * 0.5 * 0.4 = 0.18, clearly not equal to the raw probability 0.9.
        self.assertAlmostEqual(float(risk[0, 0]), 0.18, places=6)
        self.assertNotAlmostEqual(float(risk[0, 0]), float(pred[0, 0]), places=2)

    def test_transformation_is_deterministic(self):
        pred = np.random.default_rng(3).uniform(0, 1, size=(5, 5)).astype(np.float32)
        severity = np.random.default_rng(4).uniform(0, 1, size=(5, 5)).astype(np.float32)
        fuel = np.random.default_rng(5).uniform(0, 1, size=(5, 5)).astype(np.float32)
        norm = {
            "potential_ros": {"min": 0.0, "max": 1.0},
            "vegetation_density": {"min": 0.0, "max": 1.0},
        }
        cfg = ExpectedDamageConfig()
        risk_a = _compute_risk_grid(pred, severity, fuel, norm, cfg)
        risk_b = _compute_risk_grid(pred, severity, fuel, norm, cfg)
        np.testing.assert_array_equal(risk_a, risk_b)

    def test_higher_severity_or_fuel_increases_expected_damage(self):
        pred = np.array([[0.8]], dtype=np.float32)
        norm = {
            "potential_ros": {"min": 0.0, "max": 1.0},
            "vegetation_density": {"min": 0.0, "max": 1.0},
        }
        cfg = ExpectedDamageConfig()
        low = _compute_risk_grid(
            pred, np.array([[0.2]], dtype=np.float32), np.array([[0.2]], dtype=np.float32), norm, cfg
        )
        high = _compute_risk_grid(
            pred, np.array([[0.9]], dtype=np.float32), np.array([[0.9]], dtype=np.float32), norm, cfg
        )
        self.assertGreater(float(high[0, 0]), float(low[0, 0]))

    def test_disabling_factors_falls_back_toward_raw_probability(self):
        pred = np.array([[0.7]], dtype=np.float32)
        severity = np.array([[0.1]], dtype=np.float32)
        fuel = np.array([[0.1]], dtype=np.float32)
        norm = {
            "potential_ros": {"min": 0.0, "max": 1.0},
            "vegetation_density": {"min": 0.0, "max": 1.0},
        }
        cfg_off = ExpectedDamageConfig(use_severity_factor=False, use_fuel_factor=False)
        risk = _compute_risk_grid(pred, severity, fuel, norm, cfg_off)
        self.assertAlmostEqual(float(risk[0, 0]), 0.7, places=6)


# ── Issue 2: PredictionRiskMap ──────────────────────────────────────────────


class TestPredictionRiskMap(unittest.TestCase):
    def test_risk_at_samples_correct_cell(self):
        fire_prob = np.zeros((4, 4), dtype=np.float32)
        risk = np.zeros((4, 4), dtype=np.float32)
        risk[2, 3] = 0.77  # row=2 (y), col=3 (x)
        rmap = PredictionRiskMap(fire_probability=fire_prob, risk=risk, width=4, height=4, tick=1)
        self.assertAlmostEqual(rmap.risk_at(x=3, y=2), 0.77, places=6)

    def test_as_cell_risk_dict_matches_dstar_cell_convention(self):
        risk = np.zeros((3, 2), dtype=np.float32)  # height=3, width=2
        risk[1, 0] = 0.5  # row=1, col=0 -> Cell (x=0, y=1)
        rmap = PredictionRiskMap(
            fire_probability=np.zeros((3, 2)), risk=risk, width=2, height=3, tick=0
        )
        d = rmap.as_cell_risk_dict()
        self.assertAlmostEqual(d[(0, 1)], 0.5, places=6)
        self.assertEqual(len(d), 6)


# ── Issue 3: genuine incremental D* Lite reuse ──────────────────────────────


class TestDStarLiteSessionIncremental(unittest.TestCase):
    def test_same_planner_state_survives_move_and_obstacle_change(self):
        """
        plan() -> UAV moves -> obstacle changes -> update_start()/
        update_obstacles() -- using the SAME DStarLite instance's g/rhs/U,
        not a freshly reconstructed planner.
        """
        session = DStarLiteSession(10, 10, blocked=set())
        dstar_instance = session._dstar  # same object identity checked below
        first = session.start_leg((0, 0), (9, 9))
        self.assertIsNotNone(first)
        self.assertIs(session._dstar, dstar_instance)  # never replaced

        moved = session.move_to((1, 1))
        self.assertIsNotNone(moved)
        self.assertIs(session._dstar, dstar_instance)
        self.assertGreater(dstar_instance._km, 0.0)  # km bookkeeping actually happened

        g_before = dict(dstar_instance._g)
        repaired = session.add_obstacles({(5, 5), (5, 6), (5, 4)})
        self.assertIsNotNone(repaired)
        self.assertIs(session._dstar, dstar_instance)
        # Incremental repair changed only some g-values, not a full reset
        # (a fresh plan() call clears _g entirely via _initialize).
        self.assertTrue(len(dstar_instance._g) >= len(g_before) - 5)


class TestExecutorGenuineIncrementalReuse(unittest.TestCase):
    def _request(self, tick, obstacles=()):
        return MissionExecutionRequest(
            target_ids=(1,),
            waypoints=((0.0, 0.0), (9.0, 9.0)),
            start=DroneState(x=0.0, y=0.0),
            grid_width=15,
            grid_height=15,
            mission_score=0.5,
            tick=tick,
            obstacle_cells=obstacles,
        )

    def test_second_call_with_same_goal_reuses_session(self):
        executor = DStarLiteMissionExecutor()
        r1 = executor.execute(self._request(tick=0))
        self.assertIsNotNone(r1)
        self.assertFalse(r1.used_incremental_replan)  # first call: fresh, correctly
        session_after_first = executor._session

        r2 = executor.execute(self._request(tick=1, obstacles=((5, 5),)))
        self.assertIsNotNone(r2)
        self.assertTrue(r2.used_incremental_replan)  # SAME goal -> real reuse
        self.assertIs(executor._session, session_after_first)  # not rebuilt

    def test_different_goal_forces_a_new_session(self):
        executor = DStarLiteMissionExecutor()
        executor.execute(self._request(tick=0))
        session_after_first = executor._session

        different_goal_request = MissionExecutionRequest(
            target_ids=(2,),  # different next target id
            waypoints=((0.0, 0.0), (2.0, 2.0)),
            start=DroneState(x=0.0, y=0.0),
            grid_width=15,
            grid_height=15,
            mission_score=0.5,
            tick=1,
        )
        r2 = executor.execute(different_goal_request)
        self.assertFalse(r2.used_incremental_replan)  # genuinely new problem
        self.assertIsNot(executor._session, session_after_first)

    def test_does_not_require_reconstructing_the_executor_itself(self):
        executor = DStarLiteMissionExecutor()
        for tick in range(5):
            result = executor.execute(self._request(tick=tick, obstacles=((3, 3),) if tick > 2 else ()))
            self.assertIsNotNone(result)
        # Same executor instance handled all 5 ticks; no external rebuild needed.


# ── Issue 4: risk-weighted traversal cost ───────────────────────────────────


class TestRiskWeightedRouting(unittest.TestCase):
    def test_higher_risk_region_causes_a_longer_lower_risk_detour(self):
        width = height = 12
        risky_band = {(x, 5) for x in range(2, 10)} | {(x, 6) for x in range(2, 10)}
        risk = {c: 1.0 for c in risky_band}

        no_risk = DStarLite(width, height, blocked=set(), risk=risk, risk_weight=0.0)
        path_ignoring_risk = no_risk.plan((5, 0), (5, 11))

        weighted = DStarLite(width, height, blocked=set(), risk=risk, risk_weight=50.0)
        path_avoiding_risk = weighted.plan((5, 0), (5, 11))

        self.assertIsNotNone(path_ignoring_risk)
        self.assertIsNotNone(path_avoiding_risk)

        cells_in_band_ignoring = sum(1 for c in path_ignoring_risk if c in risky_band)
        cells_in_band_avoiding = sum(1 for c in path_avoiding_risk if c in risky_band)
        self.assertLess(cells_in_band_avoiding, cells_in_band_ignoring)

    def test_zero_risk_weight_reproduces_pure_distance_routing(self):
        width = height = 10
        risk = {(5, 5): 100.0}
        planner = DStarLite(width, height, blocked=set(), risk=risk, risk_weight=0.0)
        path = planner.plan((0, 0), (9, 9))
        self.assertIsNotNone(path)
        # Should still be willing to pass near/through (5,5) since weight is 0.
        from mission.optimizer.dstar_lite import path_length
        self.assertAlmostEqual(path_length(path), 9 * (2 ** 0.5), places=6)


# ── Issue 5: surrogate vs actual cost separation ────────────────────────────


class TestSurrogateVsActualCost(unittest.TestCase):
    def test_deviation_ratio_computed_from_both_recorded_costs(self):
        executor = DStarLiteMissionExecutor()
        request = MissionExecutionRequest(
            target_ids=(1,),
            waypoints=((0.0, 0.0), (9.0, 0.0)),
            start=DroneState(x=0.0, y=0.0),
            grid_width=15,
            grid_height=15,
            mission_score=0.5,
            tick=0,
            obstacle_cells=((4, 0), (5, 0), (6, 0)),  # forces a detour
        )
        result = executor.execute(request)
        self.assertIsNotNone(result)
        self.assertTrue(result.feasible)
        self.assertGreater(result.path_length, result.straight_line_length)
        self.assertGreater(result.deviation_ratio, 1.0)

    def test_reported_surrogate_is_bit_identical_to_nsga2s_own_objective(self):
        """
        Regression test for a real bug: the executor used to RE-DERIVE the
        surrogate distance from rounded-to-integer-cell waypoints (differing
        from NSGA-II's actual continuous-coordinate objective by ~0.2% on
        average), while claiming to report "what NSGA-II used". It now
        receives and reports the literal value instead.
        """
        from mission.config.settings import (
            GridConfig,
            MissionConfig,
            TargetGenerationConfig,
        )
        from mission.fitness.mission_selection import select_highest_scoring_mission
        from mission.optimizer.nsga2 import NSGA2MissionOptimizer
        from mission.simulation.environment import WildfireEnvironment

        mission_cfg = MissionConfig(
            grid=GridConfig(width=30, height=30),
            targets=TargetGenerationConfig(min_targets=12, max_targets=12),
            optimizer=OptimizerConfig(population_size=20, n_generations=20, max_mission_targets=5),
            seed=7,
        )
        env = WildfireEnvironment.create_synthetic(mission_cfg)
        opt_result = NSGA2MissionOptimizer(env, mission_cfg).optimize(seed=7)
        self.assertGreater(opt_result.n_solutions, 0)
        scored = select_highest_scoring_mission(opt_result)
        true_nsga2_value = scored.plan.objectives.travel_distance

        from mission.replanning.executor import build_execution_request

        request = build_execution_request(env, scored, tick=0)
        self.assertEqual(request.nsga2_travel_distance, true_nsga2_value)

        result = DStarLiteMissionExecutor().execute(request)
        self.assertIsNotNone(result)
        self.assertEqual(result.straight_line_length, true_nsga2_value)  # bit-identical, not just close


# ── Issue 6: permutation decoding is a genuine feasible subset ─────────────


class TestPermutationDecodingIsASubset(unittest.TestCase):
    def _targets(self, n, spacing=100.0):
        return [
            SuppressionTarget(id=i, x=float(i) * spacing, y=0.0, damage_score=0.5, priority=0.5)
            for i in range(n)
        ]

    def test_full_permutation_visits_every_target_when_unconstrained(self):
        targets = self._targets(5, spacing=1.0)
        drone = DroneState(x=0.0, y=0.0)
        cfg = OptimizerConfig(max_mission_targets=10, max_mission_distance=1e6)
        decoded = decode_permutation([4, 3, 2, 1, 0], targets, drone, cfg)
        self.assertEqual(decoded.n_targets, 5)
        self.assertEqual(decoded.target_ids, (4, 3, 2, 1, 0))  # order preserved

    def test_max_mission_targets_forces_a_real_subset(self):
        targets = self._targets(10, spacing=1.0)
        drone = DroneState(x=0.0, y=0.0)
        cfg = OptimizerConfig(max_mission_targets=3, max_mission_distance=1e6)
        decoded = decode_permutation(list(range(10)), targets, drone, cfg)
        self.assertEqual(decoded.n_targets, 3)
        self.assertLess(decoded.n_targets, len(targets))  # genuinely a subset

    def test_max_mission_distance_forces_a_real_subset(self):
        targets = self._targets(10, spacing=100.0)
        drone = DroneState(x=0.0, y=0.0)
        cfg = OptimizerConfig(max_mission_targets=100, max_mission_distance=250.0)
        decoded = decode_permutation(list(range(10)), targets, drone, cfg)
        self.assertLess(decoded.n_targets, len(targets))
        self.assertLessEqual(decoded.travel_distance, 250.0)

    def test_decoded_ids_are_unique(self):
        targets = self._targets(8, spacing=50.0)
        drone = DroneState(x=0.0, y=0.0)
        cfg = OptimizerConfig(max_mission_targets=8, max_mission_distance=1e6)
        decoded = decode_permutation([3, 1, 4, 0, 6, 2, 5, 7], targets, drone, cfg)
        self.assertEqual(len(decoded.target_ids), len(set(decoded.target_ids)))

    def test_invalid_permutation_raises(self):
        targets = self._targets(4)
        drone = DroneState(x=0.0, y=0.0)
        cfg = OptimizerConfig()
        with self.assertRaises(ValueError):
            decode_permutation([0, 1, 1, 3], targets, drone, cfg)  # not a permutation


if __name__ == "__main__":
    unittest.main()
