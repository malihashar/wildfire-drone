"""
Autonomous wildfire-suppression UAV — mission planning research package.

Phase 1: synthetic environment + visualization.
Phase 2: pymoo NSGA-II Pareto mission optimization.
Later: ConvLSTM target adapter, D* Lite local planning.
"""

from __future__ import annotations

from mission.config.settings import DEFAULT_CONFIG, MissionConfig, OptimizerConfig
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.targets import DroneState, SuppressionTarget

__all__ = [
    "DEFAULT_CONFIG",
    "MissionConfig",
    "OptimizerConfig",
    "WildfireEnvironment",
    "DroneState",
    "SuppressionTarget",
    "NSGA2MissionOptimizer",
    "OptimizationResult",
    "MissionPlan",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    if name in {"NSGA2MissionOptimizer", "OptimizationResult"}:
        from mission.optimizer.nsga2 import NSGA2MissionOptimizer, OptimizationResult

        return {
            "NSGA2MissionOptimizer": NSGA2MissionOptimizer,
            "OptimizationResult": OptimizationResult,
        }[name]
    if name == "MissionPlan":
        from mission.fitness.scoring import MissionPlan

        return MissionPlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
