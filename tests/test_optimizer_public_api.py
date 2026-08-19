"""
Tests for the packaged public API: ``from mission.optimizer import ...``.

Confirms the optimizer is usable exactly as documented, without reaching
into submodules, and that going through the public surface produces
identical behavior to importing from mission.optimizer.nsga2 directly
(the package re-export is a thin alias, not a different code path).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestOptimizerPublicAPI(unittest.TestCase):
    def test_documented_import_shape_works(self):
        from mission.optimizer import NSGA2MissionOptimizer, OptimizationResult, OptimizerConfig

        self.assertTrue(callable(NSGA2MissionOptimizer))
        self.assertTrue(callable(OptimizationResult))
        self.assertTrue(callable(OptimizerConfig))

    def test_public_alias_is_the_same_class_as_the_submodule(self):
        from mission.optimizer import NSGA2MissionOptimizer as PublicOptimizer
        from mission.optimizer.nsga2 import NSGA2MissionOptimizer as DirectOptimizer

        self.assertIs(PublicOptimizer, DirectOptimizer)

    def test_end_to_end_optimize_via_public_import_only(self):
        from mission.config.settings import GridConfig, MissionConfig, TargetGenerationConfig
        from mission.optimizer import NSGA2MissionOptimizer, OptimizerConfig
        from mission.simulation.environment import WildfireEnvironment

        mission_cfg = MissionConfig(
            grid=GridConfig(width=20, height=20),
            targets=TargetGenerationConfig(min_targets=8, max_targets=8),
            optimizer=OptimizerConfig(population_size=16, n_generations=15, max_mission_targets=4),
            seed=11,
        )
        env = WildfireEnvironment.create_synthetic(mission_cfg)

        optimizer = NSGA2MissionOptimizer(env, mission_cfg)
        result = optimizer.optimize(seed=11)

        self.assertGreater(result.n_solutions, 0)
        best = result.best_damage_plan()
        self.assertGreater(len(best.target_ids), 0)
        self.assertEqual(len(best.target_ids), len(set(best.target_ids)))  # unique

    def test_dstar_lite_planner_also_reachable_from_the_package(self):
        from mission.optimizer import DStarLitePlanner

        planner = DStarLitePlanner(10, 10, blocked=set())
        path, cost = planner.plan_mission([(0, 0), (5, 5)])
        self.assertGreater(len(path), 0)
        self.assertGreater(cost, 0.0)

    def test_unknown_attribute_raises_cleanly(self):
        import mission.optimizer as optimizer_pkg

        with self.assertRaises(AttributeError):
            _ = optimizer_pkg.NotARealSymbol


if __name__ == "__main__":
    unittest.main()
