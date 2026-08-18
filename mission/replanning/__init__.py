"""
Online replanning package (CREDS-inspired architecture demo).

Synthetic prediction updates → NSGA-II replan → mission scoring →
``MissionExecutionRequest`` hand-off for future D* Lite.
"""

from mission.replanning.config import OnlineReplanConfig
from mission.replanning.executor import (
    DStarLiteMissionExecutor,
    MissionExecutionRequest,
    NullMissionExecutor,
    build_execution_request,
)
from mission.replanning.online_replanner import OnlineReplanner, OnlineReplanResult, ReplanEvent

__all__ = [
    "OnlineReplanConfig",
    "OnlineReplanner",
    "OnlineReplanResult",
    "ReplanEvent",
    "MissionExecutionRequest",
    "build_execution_request",
    "NullMissionExecutor",
    "DStarLiteMissionExecutor",
]
