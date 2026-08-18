"""
Mission optimizer package.

Uses official pymoo NSGA-II with permutation operators, plus a real D* Lite
grid planner (``dstar_lite.py`` / ``planner_stub.py``) for local execution.

Import concrete symbols from submodules, e.g.::

    from mission.optimizer.nsga2 import NSGA2MissionOptimizer
"""

from __future__ import annotations

__all__ = [
    "NSGA2MissionOptimizer",
    "OptimizationResult",
    "SuppressionMissionProblem",
    "DStarLitePlanner",
]


def __getattr__(name: str):
    if name in {"NSGA2MissionOptimizer", "OptimizationResult"}:
        from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult

        return {
            "NSGA2MissionOptimizer": NSGA2MissionOptimizer,
            "OptimizationResult": OptimizationResult,
        }[name]
    if name == "SuppressionMissionProblem":
        from mission.optimizer.problem import SuppressionMissionProblem

        return SuppressionMissionProblem
    if name == "DStarLitePlanner":
        from mission.optimizer.planner_stub import DStarLitePlanner

        return DStarLitePlanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
