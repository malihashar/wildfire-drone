"""
Synthetic wildfire mission environment.

Holds the 100×100 spatial domain, UAV start state, and candidate suppression
targets. Designed so the target list can later be produced from ConvLSTM
predicted spread maps instead of the synthetic generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mission.config.settings import MissionConfig
from mission.simulation.generator import generate_drone_start, generate_synthetic_targets
from mission.simulation.targets import DroneState, SuppressionTarget
from mission.utils.seed import set_seed


@dataclass
class WildfireEnvironment:
    """
    Phase-1 research environment for mission planning.

    Attributes
    ----------
    width, height:
        Grid dimensions (match ConvLSTM output resolution).
    drone:
        UAV starting location.
    targets:
        Candidate suppression points (synthetic now; ConvLSTM-derived later).
    seed:
        RNG seed used to build this instance (for reproducibility).
    """

    width: int
    height: int
    drone: DroneState
    targets: list[SuppressionTarget] = field(default_factory=list)
    seed: int | None = None
    # Monotonic counter for fresh target ids; never recomputed from the
    # current target list, so an id is never reused after its target is
    # suppressed/removed (which would otherwise make a brand-new target look
    # like a previously-suppressed one reappearing).
    _next_id_counter: int = field(default=-1, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._next_id_counter < 0:
            self._next_id_counter = max((t.id for t in self.targets), default=-1) + 1

    def allocate_target_id(self) -> int:
        """Return a fresh target id that has never been used before."""
        tid = self._next_id_counter
        self._next_id_counter += 1
        return tid

    @classmethod
    def create_synthetic(cls, config: MissionConfig | None = None) -> WildfireEnvironment:
        """Build a fully synthetic scene from ``MissionConfig``."""
        cfg = config or MissionConfig()
        rng = set_seed(cfg.seed)

        drone = generate_drone_start(cfg.grid, rng)
        targets = generate_synthetic_targets(cfg.grid, cfg.targets, drone, rng)

        return cls(
            width=cfg.grid.width,
            height=cfg.grid.height,
            drone=drone,
            targets=targets,
            seed=cfg.seed,
        )

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    def target_xy(self) -> np.ndarray:
        """Return an ``(N, 2)`` array of target coordinates."""
        if not self.targets:
            return np.zeros((0, 2), dtype=float)
        return np.array([[t.x, t.y] for t in self.targets], dtype=float)

    def summary(self) -> str:
        """Human-readable scene summary for logs / demos."""
        lines = [
            f"WildfireEnvironment {self.width}x{self.height} "
            f"(seed={self.seed}, targets={self.n_targets})",
            f"  drone start: ({self.drone.x:.2f}, {self.drone.y:.2f})",
        ]
        for t in self.targets:
            lines.append(
                f"  T{t.id:02d} @ ({t.x:5.1f}, {t.y:5.1f})  "
                f"damage={t.damage_score:.3f}  priority={t.priority:.3f}  "
                f"travel_cost={t.travel_cost:.2f}"
            )
        return "\n".join(lines)
