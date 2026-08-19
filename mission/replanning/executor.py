"""
Execution hand-off interface for the selected mission.

After online replanning selects a mission (NSGA-II — decides WHAT to visit),
``DStarLiteMissionExecutor`` consumes a ``MissionExecutionRequest`` and plans
the actual obstacle-aware grid route (D* Lite — decides HOW to get there).

The executor maintains a persistent ``DStarLiteSession`` for the leg the UAV
is currently flying (its head waypoint), reusing that session's D* Lite
state across ticks via ``move_to``/``add_obstacles``/``remove_obstacles``
whenever the immediate next target hasn't changed — genuine incremental
repair, not a fresh planner constructed every call. A new session starts
only when the active leg's goal actually changes (a different next target),
which is a genuinely new planning problem D* Lite's incrementality doesn't
apply to. Legs beyond the immediate one are planned fresh each call (the
UAV isn't flying them yet, so there is no prior state to reuse for them).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from mission.fitness.mission_selection import ScoredMission
from mission.optimizer.dstar_lite import Cell, DStarLiteSession, path_length
from mission.optimizer.planner_stub import DStarLitePlanner
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.prediction_source import PredictionRiskMap
from mission.simulation.targets import DroneState


@dataclass(frozen=True)
class MissionExecutionRequest:
    """
    Payload passed from the mission selector to a local planner / executor.

    ``waypoints`` is ordered as ``[drone_start, target_1, ..., target_k]`` in
    grid coordinates. D* Lite plans consecutive segments between them.
    ``obstacle_cells`` are other currently-live (not-yet-suppressed) fire
    targets outside this mission — cells the route should avoid flying
    through. ``risk_map``, if the prediction source provided one, carries
    the full spatial expected-damage-proxy grid for optional risk-weighted
    routing (see ``DStarLiteMissionExecutor(risk_weight=...)``).
    """

    target_ids: tuple[int, ...]
    waypoints: tuple[tuple[float, float], ...]
    start: DroneState
    grid_width: int
    grid_height: int
    mission_score: float
    tick: int
    obstacle_cells: tuple[tuple[int, int], ...] = ()
    risk_map: PredictionRiskMap | None = None
    # The ACTUAL value mission.fitness.objectives.objective_travel_distance
    # computed for this mission during NSGA-II's fitness evaluation (continuous
    # float coordinates, no rounding) -- not a re-derivation. None only for
    # synthetic/test requests with no real NSGA-II mission behind them, in
    # which case the executor falls back to its own rounded-cell estimate.
    nsga2_travel_distance: float | None = None


def build_execution_request(
    env: WildfireEnvironment,
    scored: ScoredMission,
    tick: int,
    risk_map: PredictionRiskMap | None = None,
) -> MissionExecutionRequest:
    """Build a D*-Lite-ready request from the environment and selected mission."""
    id_map = {t.id: t for t in env.targets}
    waypoints: list[tuple[float, float]] = [(env.drone.x, env.drone.y)]
    valid_ids: list[int] = []
    for tid in scored.target_ids:
        target = id_map.get(tid)
        if target is None:
            continue
        valid_ids.append(tid)
        waypoints.append((target.x, target.y))

    mission_ids = set(valid_ids)
    obstacle_cells = tuple(
        (
            int(round(t.x)),
            int(round(t.y)),
        )
        for t in env.targets
        if t.id not in mission_ids
    )

    return MissionExecutionRequest(
        target_ids=tuple(valid_ids),
        waypoints=tuple(waypoints),
        start=env.drone,
        grid_width=env.width,
        grid_height=env.height,
        mission_score=scored.score,
        tick=tick,
        obstacle_cells=obstacle_cells,
        risk_map=risk_map,
        nsga2_travel_distance=scored.plan.objectives.travel_distance,
    )


@dataclass(frozen=True)
class MissionExecutionResult:
    """
    Outcome of executing/locally-planning a ``MissionExecutionRequest``.

    ``path_length`` is D* Lite's actual obstacle-aware route cost;
    ``straight_line_length`` is the SAME value ``mission.fitness.objectives.
    objective_travel_distance`` computed during NSGA-II's fitness evaluation
    for this exact mission (passed through via ``MissionExecutionRequest.
    nsga2_travel_distance``, not re-derived on rounded grid cells — an
    earlier version recomputed it from integer-rounded waypoints, which was
    off by ~0.2% on average from the true NSGA-II value; fixed to use the
    literal figure). The two are recorded separately so the deviation
    between the surrogate and the real route can be measured
    (``path_length / straight_line_length``), not conflated.
    ``used_incremental_replan`` is True only when the active leg's D* Lite
    session state (g/rhs/queue) was genuinely reused this tick rather than
    a fresh planner being constructed.
    """

    tick: int
    target_ids: tuple[int, ...]
    cell_path: tuple[Cell, ...]
    path_length: float
    straight_line_length: float
    feasible: bool
    used_incremental_replan: bool = False

    @property
    def deviation_ratio(self) -> float:
        """``path_length / straight_line_length``; 1.0 = surrogate matched actual exactly."""
        if not self.feasible or self.straight_line_length <= 0:
            return float("nan")
        return self.path_length / self.straight_line_length


class MissionExecutor(ABC):
    """Interface for executing / locally refining a selected mission."""

    @abstractmethod
    def execute(self, request: MissionExecutionRequest) -> MissionExecutionResult | None:
        """Execute or refine ``request`` (e.g. via D* Lite)."""


class DStarLiteMissionExecutor(MissionExecutor):
    """
    Plans the real obstacle-aware route for a selected mission using D* Lite.

    Treats other currently-live suppression targets (fire not part of this
    leg) as obstacles the UAV must route around, rather than a straight line
    through them. The leg toward the immediate next target is planned via a
    persistent ``DStarLiteSession`` reused across ticks (true incremental
    repair) whenever that target hasn't changed; later legs are planned
    fresh each call. Keeps the most recent result for inspection/plotting.

    Parameters
    ----------
    risk_weight:
        If > 0 and a request carries a ``risk_map``, blends per-cell
        expected-damage-proxy risk into D* Lite's edge cost (see
        ``mission.optimizer.dstar_lite.DStarLite``). Default 0 reproduces
        pure-distance routing exactly (no behaviour change unless opted in).
    """

    def __init__(self, risk_weight: float = 0.0) -> None:
        self.last_result: MissionExecutionResult | None = None
        self.risk_weight = risk_weight
        self._session: DStarLiteSession | None = None
        self._active_goal_id: int | None = None
        self._active_dims: tuple[int, int] | None = None
        self._active_obstacles: set[Cell] = set()

    def execute(self, request: MissionExecutionRequest) -> MissionExecutionResult | None:
        if len(request.waypoints) < 2:
            self.last_result = None
            return None

        cells = [self._to_cell(x, y, request) for x, y in request.waypoints]
        obstacles = set(request.obstacle_cells)
        dims = (request.grid_width, request.grid_height)
        risk = (
            request.risk_map.as_cell_risk_dict()
            if (request.risk_map is not None and self.risk_weight > 0.0)
            else None
        )

        first_goal_id = request.target_ids[0] if request.target_ids else None
        start_cell, first_goal_cell = cells[0], cells[1]

        can_reuse = (
            self._session is not None
            and self._active_goal_id == first_goal_id
            and self._active_dims == dims
        )
        if can_reuse:
            assert self._session is not None
            self._session.move_to(start_cell)
            added = obstacles - self._active_obstacles
            removed = self._active_obstacles - obstacles
            if added:
                self._session.add_obstacles(added)
            if removed:
                self._session.remove_obstacles(removed)
            first_leg_path = self._session.current_path()
            used_incremental = True
        else:
            self._session = DStarLiteSession(
                request.grid_width,
                request.grid_height,
                blocked=obstacles,
                risk=risk,
                risk_weight=self.risk_weight,
            )
            first_leg_path = self._session.start_leg(start_cell, first_goal_cell)
            used_incremental = False

        self._active_goal_id = first_goal_id
        self._active_dims = dims
        self._active_obstacles = obstacles

        rest_planner = DStarLitePlanner(
            request.grid_width, request.grid_height, blocked=obstacles, risk=risk, risk_weight=self.risk_weight
        )
        try:
            rest_path, rest_cost = rest_planner.plan_mission(cells[1:])
            feasible_rest = True
        except RuntimeError:
            rest_path, rest_cost, feasible_rest = [], math.inf, False

        if first_leg_path is None:
            feasible = False
            full_path: list[Cell] = []
            total_cost = math.inf
        else:
            feasible = feasible_rest
            full_path = list(first_leg_path) + rest_path[1:]
            total_cost = path_length(first_leg_path) + rest_cost

        straight = (
            request.nsga2_travel_distance
            if request.nsga2_travel_distance is not None
            else sum(_euclid(a, b) for a, b in zip(cells, cells[1:]))
        )

        result = MissionExecutionResult(
            tick=request.tick,
            target_ids=request.target_ids,
            cell_path=tuple(full_path),
            path_length=total_cost,
            straight_line_length=straight,
            feasible=feasible,
            used_incremental_replan=used_incremental,
        )
        self.last_result = result
        return result

    @staticmethod
    def _to_cell(x: float, y: float, request: MissionExecutionRequest) -> Cell:
        return (
            int(round(min(max(x, 0), request.grid_width - 1))),
            int(round(min(max(y, 0), request.grid_height - 1))),
        )


def _euclid(a: Cell, b: Cell) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class NullMissionExecutor(MissionExecutor):
    """No-op executor used by the architecture demo."""

    def execute(self, request: MissionExecutionRequest) -> None:
        return None
