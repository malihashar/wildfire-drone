"""Lightweight geometry helpers shared across simulation and (later) planning."""

from __future__ import annotations

import math


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points in the mission grid."""
    return math.hypot(x2 - x1, y2 - y1)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))
