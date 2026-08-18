"""
D* Lite grid path planner (Koenig & Likhachev, 2002).

Decides HOW the UAV travels between two points NSGA-II has already chosen
(WHAT to visit) — an 8-connected grid search that treats currently-active
suppression targets (fire cells) as obstacles, so the planned route routes
around live fire instead of flying a straight line through it.

Supports the algorithm's actual incremental-replanning behaviour, not a
"rebuild and re-search from scratch" substitute:
  - ``update_start``    — the UAV moves; only ``km`` bookkeeping changes.
  - ``update_obstacles`` — cells become blocked/unblocked; only affected
    vertices are re-evaluated.
  - ``update_risk``     — spatially-varying traversal risk changes; same
    incremental vertex-update mechanism.
All three preserve ``g``/``rhs``/the priority queue between calls — no
reinitialization. ``DStarLiteSession`` wraps one persistent instance so a
caller can plan once and then repeatedly move/repair without ever
reconstructing the planner.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

Cell = tuple[int, int]

_DIAGONAL_COST = math.sqrt(2.0)
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _DIAGONAL_COST), (1, -1, _DIAGONAL_COST),
    (-1, 1, _DIAGONAL_COST), (-1, -1, _DIAGONAL_COST),
)


@dataclass(order=True)
class _Key:
    k1: float
    k2: float


class DStarLite:
    """
    Incremental grid path planner.

    Parameters
    ----------
    width, height:
        Grid dimensions in cells.
    blocked:
        Set of ``(col, row)`` cells that are impassable (e.g. active fire).
    risk:
        Optional ``{(col, row): risk}`` traversal-risk values in a
        configurable, application-defined unit (this project uses the
        ConvLSTM expected-damage proxy — see
        ``mission.simulation.prediction_source.ExpectedDamageConfig``).
        NOT a physically calibrated hazard cost. Blended into edge cost via
        ``risk_weight``; ``risk_weight=0`` (default) reproduces the original
        pure-distance behaviour exactly.
    risk_weight:
        Multiplier on the destination cell's risk value, added to the base
        movement cost: ``cost = base_move_cost + risk_weight * risk[b]``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        blocked: set[Cell] | None = None,
        risk: dict[Cell, float] | None = None,
        risk_weight: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        self.width = width
        self.height = height
        self.blocked: set[Cell] = set(blocked or ())
        self.risk: dict[Cell, float] = dict(risk or {})
        self.risk_weight = float(risk_weight)

        self._g: dict[Cell, float] = {}
        self._rhs: dict[Cell, float] = {}
        self._U: list[tuple[_Key, Cell]] = []
        self._U_members: set[Cell] = set()
        self._km = 0.0
        self._start: Cell | None = None
        self._goal: Cell | None = None
        self._last_start: Cell | None = None

    # ── public API ──────────────────────────────────────────────────────

    def plan(self, start: Cell, goal: Cell) -> list[Cell] | None:
        """Compute a full path from ``start`` to ``goal``. Returns None if unreachable."""
        self._initialize(start, goal)
        self._compute_shortest_path()
        return self._extract_path()

    def update_start(self, new_start: Cell) -> list[Cell] | None:
        """
        Move the UAV's current position without resetting planner state.

        Per D* Lite, moving the start increases ``km`` by
        ``h(last_start, new_start)`` so previously-computed keys stay
        consistent; ``g``/``rhs``/the queue are otherwise untouched — no
        reinitialization, no fresh search from scratch.
        """
        if self._start is None or self._goal is None:
            raise RuntimeError("Call plan() once before update_start().")
        if new_start != self._start:
            self._km += self._heuristic(self._last_start, new_start)
            self._last_start = new_start
            self._start = new_start
        self._compute_shortest_path()
        return self._extract_path()

    def update_obstacles(
        self,
        added: set[Cell] | None = None,
        removed: set[Cell] | None = None,
    ) -> list[Cell] | None:
        """
        Incrementally repair the path after obstacle cells change.

        Only vertices adjacent to a changed cell are re-evaluated, matching
        D* Lite's actual incremental-replanning behaviour rather than a full
        from-scratch search. Does not touch ``km``/start bookkeeping — that
        is ``update_start``'s responsibility.
        """
        if self._start is None or self._goal is None:
            raise RuntimeError("Call plan() once before update_obstacles().")

        added = added or set()
        removed = removed or set()
        changed = (added | removed) & self._in_bounds_cells(added | removed)
        if not changed:
            return self._extract_path()

        self.blocked |= added
        self.blocked -= removed

        self._propagate_vertex_updates(changed)
        self._compute_shortest_path()
        return self._extract_path()

    def update_risk(self, risk_updates: dict[Cell, float]) -> list[Cell] | None:
        """
        Incrementally repair the path after spatially-varying risk changes.

        Same incremental mechanism as ``update_obstacles`` — only cells
        adjacent to a changed risk value are re-evaluated.
        """
        if self._start is None or self._goal is None:
            raise RuntimeError("Call plan() once before update_risk().")
        if not risk_updates:
            return self._extract_path()

        changed: set[Cell] = set()
        for cell, value in risk_updates.items():
            if self._in_bounds(cell):
                self.risk[cell] = value
                changed.add(cell)
        if not changed:
            return self._extract_path()

        self._propagate_vertex_updates(changed)
        self._compute_shortest_path()
        return self._extract_path()

    def current_path(self) -> list[Cell] | None:
        """Return the currently-computed path without triggering a new search."""
        return self._extract_path()

    # ── D* Lite core ────────────────────────────────────────────────────

    def _propagate_vertex_updates(self, changed: set[Cell]) -> None:
        affected: set[Cell] = set()
        for cell in changed:
            affected.add(cell)
            affected.update(self._neighbors(cell))
        for u in affected:
            self._update_vertex(u)

    def _initialize(self, start: Cell, goal: Cell) -> None:
        self._start = start
        self._last_start = start
        self._goal = goal
        self._km = 0.0
        self._g.clear()
        self._rhs.clear()
        self._U.clear()
        self._U_members.clear()
        self._rhs[goal] = 0.0
        heapq.heappush(self._U, (self._calculate_key(goal), goal))
        self._U_members.add(goal)

    def _calculate_key(self, s: Cell) -> _Key:
        g_rhs = min(self._g.get(s, math.inf), self._rhs.get(s, math.inf))
        return _Key(g_rhs + self._heuristic(self._start, s) + self._km, g_rhs)

    def _heuristic(self, a: Cell | None, b: Cell) -> float:
        if a is None:
            return 0.0
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def _in_bounds_cells(self, cells: set[Cell]) -> set[Cell]:
        return {c for c in cells if self._in_bounds(c)}

    def _neighbors(self, s: Cell) -> list[Cell]:
        out = []
        for dx, dy, _ in _NEIGHBOR_OFFSETS:
            n = (s[0] + dx, s[1] + dy)
            if self._in_bounds(n):
                out.append(n)
        return out

    def _cost(self, a: Cell, b: Cell) -> float:
        if a in self.blocked or b in self.blocked:
            return math.inf
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        base = _DIAGONAL_COST if dx and dy else 1.0
        if self.risk_weight > 0.0 and self.risk:
            base += self.risk_weight * self.risk.get(b, 0.0)
        return base

    def _update_vertex(self, u: Cell) -> None:
        if u != self._goal:
            self._rhs[u] = min(
                (self._cost(u, s) + self._g.get(s, math.inf) for s in self._neighbors(u)),
                default=math.inf,
            )
        if u in self._U_members:
            self._U = [(k, c) for k, c in self._U if c != u]
            heapq.heapify(self._U)
            self._U_members.discard(u)
        if self._g.get(u, math.inf) != self._rhs.get(u, math.inf):
            heapq.heappush(self._U, (self._calculate_key(u), u))
            self._U_members.add(u)

    def _compute_shortest_path(self) -> None:
        assert self._start is not None
        while self._U and (
            self._U[0][0] < self._calculate_key(self._start)
            or self._rhs.get(self._start, math.inf) != self._g.get(self._start, math.inf)
        ):
            k_old, u = heapq.heappop(self._U)
            self._U_members.discard(u)
            k_new = self._calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self._U, (k_new, u))
                self._U_members.add(u)
                continue
            if self._g.get(u, math.inf) > self._rhs.get(u, math.inf):
                self._g[u] = self._rhs[u]
                for s in self._neighbors(u):
                    self._update_vertex(s)
            else:
                self._g[u] = math.inf
                for s in self._neighbors(u) + [u]:
                    self._update_vertex(s)

    def _extract_path(self) -> list[Cell] | None:
        assert self._start is not None and self._goal is not None
        if self._g.get(self._start, math.inf) == math.inf:
            return None
        path = [self._start]
        current = self._start
        guard = 0
        max_steps = self.width * self.height + 4
        while current != self._goal:
            guard += 1
            if guard > max_steps:
                return None
            neighbors = self._neighbors(current)
            current = min(
                neighbors,
                key=lambda s: self._cost(current, s) + self._g.get(s, math.inf),
            )
            if self._cost(path[-1], current) == math.inf or self._g.get(current, math.inf) == math.inf:
                return None
            path.append(current)
        return path


class DStarLiteSession:
    """
    Persistent D* Lite planning session for one active leg (fixed goal).

    Wraps a single ``DStarLite`` instance and exposes the incremental API
    so the SAME ``g``/``rhs``/priority-queue state survives across calls,
    rather than constructing a fresh planner (and re-running a full search)
    every time something changes. Use ``start_leg`` once, then repeatedly
    ``move_to`` / ``add_obstacles`` / ``remove_obstacles`` / ``update_risk``
    as the UAV flies and the environment changes; start a NEW session only
    when the goal itself changes (a genuinely different planning problem).
    """

    def __init__(
        self,
        width: int,
        height: int,
        blocked: set[Cell] | None = None,
        risk: dict[Cell, float] | None = None,
        risk_weight: float = 0.0,
    ) -> None:
        self._dstar = DStarLite(width, height, blocked=blocked, risk=risk, risk_weight=risk_weight)
        self.goal: Cell | None = None

    def start_leg(self, start: Cell, goal: Cell) -> list[Cell] | None:
        """Begin planning toward ``goal``. Resets internal state (new problem)."""
        self.goal = goal
        return self._dstar.plan(start, goal)

    def move_to(self, new_start: Cell) -> list[Cell] | None:
        """Report the UAV's new position; incrementally repairs, does not reset."""
        return self._dstar.update_start(new_start)

    def add_obstacles(self, cells: set[Cell]) -> list[Cell] | None:
        """Mark cells as newly blocked; incrementally repairs, does not reset."""
        return self._dstar.update_obstacles(added=set(cells))

    def remove_obstacles(self, cells: set[Cell]) -> list[Cell] | None:
        """Mark cells as newly passable; incrementally repairs, does not reset."""
        return self._dstar.update_obstacles(removed=set(cells))

    def update_risk(self, risk_updates: dict[Cell, float]) -> list[Cell] | None:
        """Update spatially-varying traversal risk; incrementally repairs."""
        return self._dstar.update_risk(risk_updates)

    def current_path(self) -> list[Cell] | None:
        """The most recently computed path, without triggering a new search."""
        return self._dstar.current_path()

    @property
    def blocked(self) -> set[Cell]:
        return self._dstar.blocked


def path_length(path: list[Cell]) -> float:
    """Total Euclidean length of a cell path (8-connected step costs)."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        total += _DIAGONAL_COST if dx and dy else 1.0
    return total
