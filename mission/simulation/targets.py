"""
Suppression target data model.

Aligned with the multi-target firefighting planning philosophy in
"Multi-Target Firefighting Task Planning Strategy for Multiple UAVs Under
Dynamic Forest Fire Environment" (2025), adapted here for a single UAV.

``damage_score`` and ``priority`` are intended as objectives / weights for
future NSGA-II mission optimization (GARST-inspired prioritization).
``travel_cost`` is a placeholder that D* Lite (or a simpler distance metric)
will populate during local planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SuppressionTarget:
    """A candidate wildfire cell the UAV may suppress."""

    id: int
    x: float
    y: float
    damage_score: float
    priority: float
    travel_cost: float = field(default=0.0)

    @property
    def predicted_damage(self) -> float:
        """
        Alias for ``damage_score``.

        Named to match the research problem statement and future ConvLSTM
        predicted-damage fields without breaking the existing schema.
        """
        return self.damage_score

    def as_dict(self) -> dict[str, float | int]:
        """Serialize for logging / experiment dumps."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "damage_score": self.damage_score,
            "predicted_damage": self.predicted_damage,
            "priority": self.priority,
            "travel_cost": self.travel_cost,
        }


@dataclass(frozen=True)
class DroneState:
    """UAV pose in the mission grid (Phase 1: 2-D position only)."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
