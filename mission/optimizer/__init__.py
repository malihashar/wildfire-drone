"""
Mission optimizer package — public entry point for suppression-mission
optimization.

Uses official pymoo NSGA-II with permutation operators, plus a real D* Lite
grid planner (``dstar_lite.py`` / ``planner_stub.py``) for local execution.
NSGA-II decides WHAT suppression targets to visit; D* Lite decides HOW to
physically route between them — two separate concerns, both importable from
here.

Typical usage::

    from mission.optimizer import NSGA2MissionOptimizer, OptimizerConfig

    optimizer = NSGA2MissionOptimizer(environment, OptimizerConfig(...))
    result = optimizer.optimize(seed=42)
    best = result.best_damage_plan()

Everything below is lazily imported on first access (``__getattr__``), so
``import mission.optimizer`` stays cheap even though it re-exports symbols
from pymoo-backed submodules.
"""

from __future__ import annotations

__all__ = [
    "NSGA2MissionOptimizer",
    "OptimizationResult",
    "OptimizerConfig",
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
    if name == "OptimizerConfig":
        # Lives in mission.config.settings (shared by other mission
        # subsystems too), re-exported here for ergonomic one-line use
        # alongside NSGA2MissionOptimizer.
        from mission.config.settings import OptimizerConfig

        return OptimizerConfig
    if name == "SuppressionMissionProblem":
        from mission.optimizer.problem import SuppressionMissionProblem

        return SuppressionMissionProblem
    if name == "DStarLitePlanner":
        from mission.optimizer.planner_stub import DStarLitePlanner

        return DStarLitePlanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
