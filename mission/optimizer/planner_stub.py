"""
Local path planner: multi-leg D* Lite mission execution.

NSGA-II decides WHAT suppression targets to visit and in what order.
``DStarLitePlanner`` decides HOW to actually fly between them — planning
each consecutive leg on an obstacle-aware grid (``mission.optimizer.
dstar_lite.DStarLite``) rather than assuming a straight line.
"""

from __future__ import annotations

from mission.optimizer.dstar_lite import Cell, DStarLite, path_length


class DStarLitePlanner:
    """
    Plans a full multi-leg mission route using D* Lite per leg.

    Each leg gets a fresh ``DStarLite`` instance (batch planning of an
    entire multi-target route up front, e.g. for legs the UAV isn't
    actively flying yet). For a single in-progress leg that needs true
    incremental repair as the UAV moves and the environment changes, use
    ``mission.optimizer.dstar_lite.DStarLiteSession`` instead, which keeps
    one persistent planner instance alive across updates.
    """

    def __init__(
        self,
        width: int,
        height: int,
        blocked: set[Cell] | None = None,
        risk: dict[Cell, float] | None = None,
        risk_weight: float = 0.0,
    ) -> None:
        self.width = width
        self.height = height
        self.blocked: set[Cell] = set(blocked or ())
        self.risk: dict[Cell, float] | None = risk
        self.risk_weight = risk_weight

    def plan_leg(self, start: Cell, goal: Cell) -> list[Cell] | None:
        """Plan a single obstacle-aware leg from ``start`` to ``goal``."""
        planner = DStarLite(
            self.width, self.height, self.blocked, risk=self.risk, risk_weight=self.risk_weight
        )
        return planner.plan(start, goal)

    def plan_mission(self, waypoints: list[Cell]) -> tuple[list[Cell], float]:
        """
        Plan the full route through ``waypoints`` (``[start, t1, ..., tk]``).

        Returns the concatenated cell path and its total D* Lite length.
        Raises ``RuntimeError`` if any leg is infeasible given the current
        obstacles.
        """
        if len(waypoints) < 2:
            return list(waypoints), 0.0

        full_path: list[Cell] = [waypoints[0]]
        total_cost = 0.0
        for a, b in zip(waypoints, waypoints[1:]):
            leg = self.plan_leg(a, b)
            if leg is None:
                raise RuntimeError(
                    f"No feasible D* Lite path between {a} and {b} given current obstacles."
                )
            full_path.extend(leg[1:])
            total_cost += path_length(leg)
        return full_path, total_cost
