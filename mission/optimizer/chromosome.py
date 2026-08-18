"""
Permutation chromosome encoding / decoding for single-UAV missions.

Chromosome
----------
A permutation of all candidate target indices, e.g. ``[4, 7, 2, 9, 1, ...]``.
Index ``i`` in the array refers to ``targets[i]`` (which has ``targets[i].id``).

Decoding
--------
The ordered list implies visit priority. The feasible mission is the longest
prefix of that order that respects ``max_mission_targets`` and
``max_mission_distance``. Targets beyond the feasible prefix are skipped.

This yields both:
  1. which targets to visit (the feasible prefix / subset)
  2. the order to visit them
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mission.config.settings import OptimizerConfig
from mission.simulation.targets import DroneState, SuppressionTarget
from mission.utils.geometry import euclidean_distance


@dataclass(frozen=True)
class DecodedMission:
    """Feasible mission extracted from a permutation chromosome."""

    target_ids: tuple[int, ...]
    targets: tuple[SuppressionTarget, ...]
    travel_distance: float

    @property
    def n_targets(self) -> int:
        return len(self.target_ids)


def decode_permutation(
    permutation: np.ndarray | list[int],
    targets: list[SuppressionTarget],
    drone: DroneState,
    config: OptimizerConfig,
) -> DecodedMission:
    """
    Convert a full permutation into a constraint-feasible ordered mission.

    Walks the permutation in order, appending targets until either the
    target-count or distance budget would be exceeded.
    """
    if len(targets) == 0:
        return DecodedMission(target_ids=(), targets=(), travel_distance=0.0)

    order = [int(i) for i in np.asarray(permutation, dtype=int).tolist()]
    n = len(targets)
    if sorted(order) != list(range(n)):
        raise ValueError(
            f"Chromosome must be a permutation of 0..{n - 1}, got {order!r}."
        )

    selected: list[SuppressionTarget] = []
    total_distance = 0.0
    prev_x, prev_y = drone.x, drone.y

    for idx in order:
        if len(selected) >= config.max_mission_targets:
            break

        candidate = targets[idx]
        step = euclidean_distance(prev_x, prev_y, candidate.x, candidate.y)
        if total_distance + step > config.max_mission_distance:
            break

        selected.append(candidate)
        total_distance += step
        prev_x, prev_y = candidate.x, candidate.y

    return DecodedMission(
        target_ids=tuple(t.id for t in selected),
        targets=tuple(selected),
        travel_distance=float(total_distance),
    )


def target_id_sequence(decoded: DecodedMission) -> list[int]:
    """Return the mission as an ordered list of target IDs."""
    return list(decoded.target_ids)
