"""
Synthetic wildfire-prediction dynamics.

Randomly edits priorities, positions, and membership of suppression targets to
stand in for evolving ConvLSTM forecasts during online replanning demos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mission.simulation.environment import WildfireEnvironment
from mission.simulation.prediction_source import (
    EnvironmentDiff,
    PredictionSource,
    PredictionUpdate,
    TargetPatch,
    UpdateKind,
)
from mission.simulation.targets import SuppressionTarget
from mission.utils.geometry import clamp, euclidean_distance


@dataclass(frozen=True)
class SyntheticDynamicsConfig:
    """Controls how aggressively the synthetic predictor mutates the scene."""

    min_patches_per_tick: int = 1
    max_patches_per_tick: int = 3
    priority_delta_range: tuple[float, float] = (0.05, 0.25)
    move_delta_range: tuple[float, float] = (1.0, 5.0)
    damage_delta_range: tuple[float, float] = (0.05, 0.20)
    # Probabilities are renormalized over currently feasible actions.
    p_priority_up: float = 0.25
    p_priority_down: float = 0.20
    p_add: float = 0.15
    p_remove: float = 0.15
    p_move: float = 0.15
    p_damage: float = 0.10
    min_targets: int = 8
    max_targets: int = 25
    max_ticks: int = 6


class SyntheticPredictionSource(PredictionSource):
    """
    Emits random but reproducible prediction updates for architecture demos.

    Swap this class for ``ConvLSTMPredictionSource`` once real forecasts exist.
    """

    def __init__(
        self,
        config: SyntheticDynamicsConfig | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = 42,
    ) -> None:
        self.config = config or SyntheticDynamicsConfig()
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._ticks_emitted = 0

    @property
    def source_name(self) -> str:
        return "synthetic"

    def next_update(self, env: WildfireEnvironment, tick: int) -> PredictionUpdate | None:
        if self._ticks_emitted >= self.config.max_ticks:
            return None

        n_patches = int(
            self.rng.integers(
                self.config.min_patches_per_tick,
                self.config.max_patches_per_tick + 1,
            )
        )
        patches: list[TargetPatch] = []
        for _ in range(n_patches):
            patch = self._sample_patch(env, existing=patches)
            if patch is not None:
                patches.append(patch)

        if not patches:
            # Force at least a mild priority tweak when possible.
            patch = self._force_priority_patch(env)
            if patch is not None:
                patches.append(patch)

        self._ticks_emitted += 1
        note = f"synthetic prediction tick {tick} ({len(patches)} patch(es))"
        return PredictionUpdate(
            tick=tick,
            patches=tuple(patches),
            source_name=self.source_name,
            note=note,
        )

    def _sample_patch(
        self,
        env: WildfireEnvironment,
        existing: list[TargetPatch],
    ) -> TargetPatch | None:
        actions = self._feasible_actions(env, existing)
        if not actions:
            return None
        kinds = [a[0] for a in actions]
        weights = np.asarray([a[1] for a in actions], dtype=float)
        weights = weights / weights.sum()
        idx = int(self.rng.choice(len(kinds), p=weights))
        return self._build_patch(env, kinds[idx], existing)

    def _feasible_actions(
        self,
        env: WildfireEnvironment,
        existing: list[TargetPatch],
    ) -> list[tuple[UpdateKind, float]]:
        cfg = self.config
        n = env.n_targets
        pending_removes = {p.target_id for p in existing if p.kind == UpdateKind.REMOVE_TARGET}
        pending_adds = sum(1 for p in existing if p.kind == UpdateKind.ADD_TARGET)
        effective_n = n - len(pending_removes) + pending_adds

        actions: list[tuple[UpdateKind, float]] = []
        if n > 0:
            actions.append((UpdateKind.PRIORITY_INCREASE, cfg.p_priority_up))
            actions.append((UpdateKind.PRIORITY_DECREASE, cfg.p_priority_down))
            actions.append((UpdateKind.MOVE_TARGET, cfg.p_move))
            actions.append((UpdateKind.DAMAGE_CHANGE, cfg.p_damage))
        if effective_n < cfg.max_targets:
            actions.append((UpdateKind.ADD_TARGET, cfg.p_add))
        if effective_n > cfg.min_targets and n - len(pending_removes) > 0:
            actions.append((UpdateKind.REMOVE_TARGET, cfg.p_remove))
        return actions

    def _build_patch(
        self,
        env: WildfireEnvironment,
        kind: UpdateKind,
        existing: list[TargetPatch],
    ) -> TargetPatch | None:
        cfg = self.config
        if kind == UpdateKind.ADD_TARGET:
            x = float(self.rng.uniform(0, env.width - 1))
            y = float(self.rng.uniform(0, env.height - 1))
            return TargetPatch(
                kind=kind,
                new_x=x,
                new_y=y,
                new_priority=float(self.rng.uniform(0.2, 1.0)),
                new_damage=float(self.rng.uniform(0.2, 1.0)),
                reason="new predicted ignition / spread cell",
            )

        target = self._random_existing_target(env, existing, exclude_removed=True)
        if target is None:
            return None

        if kind == UpdateKind.REMOVE_TARGET:
            return TargetPatch(
                kind=kind,
                target_id=target.id,
                reason="target predicted extinguished / no longer critical",
            )

        if kind == UpdateKind.PRIORITY_INCREASE:
            delta = float(self.rng.uniform(*cfg.priority_delta_range))
            return TargetPatch(
                kind=kind,
                target_id=target.id,
                delta_priority=delta,
                reason="priority increased by updated forecast",
            )

        if kind == UpdateKind.PRIORITY_DECREASE:
            delta = -float(self.rng.uniform(*cfg.priority_delta_range))
            return TargetPatch(
                kind=kind,
                target_id=target.id,
                delta_priority=delta,
                reason="priority decreased by updated forecast",
            )

        if kind == UpdateKind.MOVE_TARGET:
            mag = float(self.rng.uniform(*cfg.move_delta_range))
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            return TargetPatch(
                kind=kind,
                target_id=target.id,
                delta_x=mag * float(np.cos(angle)),
                delta_y=mag * float(np.sin(angle)),
                reason="predicted fire centroid drifted",
            )

        if kind == UpdateKind.DAMAGE_CHANGE:
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            delta = sign * float(self.rng.uniform(*cfg.damage_delta_range))
            return TargetPatch(
                kind=kind,
                target_id=target.id,
                delta_damage=delta,
                reason="predicted damage intensity revised",
            )
        return None

    def _force_priority_patch(self, env: WildfireEnvironment) -> TargetPatch | None:
        if env.n_targets == 0:
            return None
        target = env.targets[int(self.rng.integers(0, env.n_targets))]
        delta = float(self.rng.uniform(*self.config.priority_delta_range))
        if self.rng.random() < 0.5:
            delta = -delta
        kind = (
            UpdateKind.PRIORITY_INCREASE if delta > 0 else UpdateKind.PRIORITY_DECREASE
        )
        return TargetPatch(
            kind=kind,
            target_id=target.id,
            delta_priority=delta,
            reason="forced synthetic priority revision",
        )

    def _random_existing_target(
        self,
        env: WildfireEnvironment,
        existing: list[TargetPatch],
        exclude_removed: bool,
    ) -> SuppressionTarget | None:
        removed = {p.target_id for p in existing if p.kind == UpdateKind.REMOVE_TARGET}
        candidates = [
            t
            for t in env.targets
            if not (exclude_removed and t.id in removed)
        ]
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]


def apply_prediction_update(
    env: WildfireEnvironment,
    update: PredictionUpdate,
) -> EnvironmentDiff:
    """
    Apply a prediction update in-place and return the resulting diff.

    This is the single mutation gateway used by both synthetic and (future)
    ConvLSTM sources.
    """
    diff = EnvironmentDiff(patches=list(update.patches))
    id_to_idx = {t.id: i for i, t in enumerate(env.targets)}

    for patch in update.patches:
        if patch.kind == UpdateKind.ADD_TARGET:
            target = SuppressionTarget(
                id=env.allocate_target_id(),
                x=float(patch.new_x if patch.new_x is not None else 0.0),
                y=float(patch.new_y if patch.new_y is not None else 0.0),
                damage_score=float(patch.new_damage if patch.new_damage is not None else 0.5),
                priority=float(patch.new_priority if patch.new_priority is not None else 0.5),
                travel_cost=euclidean_distance(
                    env.drone.x,
                    env.drone.y,
                    float(patch.new_x or 0.0),
                    float(patch.new_y or 0.0),
                ),
            )
            env.targets.append(target)
            diff.added_ids.append(target.id)
            id_to_idx[target.id] = len(env.targets) - 1
            continue

        if patch.target_id is None or patch.target_id not in id_to_idx:
            continue
        idx = id_to_idx[patch.target_id]
        target = env.targets[idx]

        if patch.kind == UpdateKind.REMOVE_TARGET:
            diff.removed_ids.append(target.id)
            diff.removed_positions[target.id] = (target.x, target.y)
            env.targets.pop(idx)
            id_to_idx = {t.id: i for i, t in enumerate(env.targets)}
            continue

        if patch.kind in {UpdateKind.PRIORITY_INCREASE, UpdateKind.PRIORITY_DECREASE}:
            delta = float(patch.delta_priority or 0.0)
            target.priority = clamp(target.priority + delta, 0.05, 1.0)
            diff.priority_changed_ids.append(target.id)
            continue

        if patch.kind == UpdateKind.MOVE_TARGET:
            target.x = clamp(target.x + float(patch.delta_x or 0.0), 0.0, env.width - 1)
            target.y = clamp(target.y + float(patch.delta_y or 0.0), 0.0, env.height - 1)
            target.travel_cost = euclidean_distance(
                env.drone.x, env.drone.y, target.x, target.y
            )
            diff.moved_ids.append(target.id)
            continue

        if patch.kind == UpdateKind.DAMAGE_CHANGE:
            delta = float(patch.delta_damage or 0.0)
            target.damage_score = clamp(target.damage_score + delta, 0.05, 1.0)
            diff.damage_changed_ids.append(target.id)
            continue

    # De-duplicate highlight lists while preserving order.
    diff.added_ids = _unique(diff.added_ids)
    diff.removed_ids = _unique(diff.removed_ids)
    diff.priority_changed_ids = _unique(diff.priority_changed_ids)
    diff.moved_ids = _unique(diff.moved_ids)
    diff.damage_changed_ids = _unique(diff.damage_changed_ids)
    return diff


def _unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
