import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission.optimizer.dstar_lite import DStarLite, path_length
from mission.optimizer.planner_stub import DStarLitePlanner


class TestDStarLite(unittest.TestCase):
    def test_open_grid_finds_straight_diagonal_path(self):
        planner = DStarLite(10, 10, blocked=set())
        path = planner.plan((0, 0), (5, 5))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 5))
        # Open grid: optimal path is 5 diagonal steps.
        self.assertAlmostEqual(path_length(path), 5 * (2 ** 0.5), places=6)

    def test_routes_around_a_wall(self):
        blocked = {(3, y) for y in range(0, 8)}  # vertical wall, gap at y=8,9
        planner = DStarLite(10, 10, blocked=blocked)
        path = planner.plan((0, 5), (6, 5))
        self.assertIsNotNone(path)
        for cell in path:
            self.assertNotIn(cell, blocked)

    def test_unreachable_goal_returns_none(self):
        # Fully enclose the goal.
        blocked = {(4, y) for y in range(0, 10)} | {(6, y) for y in range(0, 10)} | \
                  {(x, 4) for x in range(4, 7)} | {(x, 6) for x in range(4, 7)}
        planner = DStarLite(10, 10, blocked=blocked)
        path = planner.plan((0, 0), (5, 5))
        self.assertIsNone(path)

    def test_update_obstacles_forces_reroute(self):
        planner = DStarLite(10, 10, blocked=set())
        first = planner.plan((0, 0), (9, 0))
        self.assertIsNotNone(first)
        self.assertTrue(all(c[1] == 0 for c in first))  # straight along y=0

        # Block the straight route; incremental update should find a detour.
        blocked_row = {(x, 0) for x in range(2, 8)}
        second = planner.update_obstacles(added=blocked_row)
        self.assertIsNotNone(second)
        for cell in second:
            self.assertNotIn(cell, blocked_row)

    def test_multi_leg_mission_plan(self):
        planner = DStarLitePlanner(20, 20, blocked=set())
        path, cost = planner.plan_mission([(0, 0), (5, 0), (5, 5)])
        self.assertGreater(len(path), 0)
        self.assertAlmostEqual(cost, 10.0, places=6)

    def test_mission_plan_raises_when_leg_infeasible(self):
        blocked = {(4, y) for y in range(0, 10)} | {(6, y) for y in range(0, 10)} | \
                  {(x, 4) for x in range(4, 7)} | {(x, 6) for x in range(4, 7)}
        planner = DStarLitePlanner(10, 10, blocked=blocked)
        with self.assertRaises(RuntimeError):
            planner.plan_mission([(0, 0), (5, 5)])


if __name__ == "__main__":
    unittest.main()
