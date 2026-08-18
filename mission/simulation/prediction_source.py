"""
Wildfire prediction update contracts.

Synthetic updates stand in for future ConvLSTM predicted-spread revisions.
Any ``PredictionSource`` implementation must emit ``PredictionUpdate`` objects
that mutate a ``WildfireEnvironment`` through the same apply path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mission.simulation.environment import WildfireEnvironment


class UpdateKind(str, Enum):
    """Kinds of prediction-driven environment edits."""

    PRIORITY_INCREASE = "priority_increase"
    PRIORITY_DECREASE = "priority_decrease"
    ADD_TARGET = "add_target"
    REMOVE_TARGET = "remove_target"
    MOVE_TARGET = "move_target"
    DAMAGE_CHANGE = "damage_change"


@dataclass(frozen=True)
class TargetPatch:
    """One atomic change applied to the live target set."""

    kind: UpdateKind
    target_id: int | None = None
    # Used when adding a brand-new target.
    new_x: float | None = None
    new_y: float | None = None
    new_priority: float | None = None
    new_damage: float | None = None
    # Used for priority / damage deltas and moves.
    delta_priority: float | None = None
    delta_damage: float | None = None
    delta_x: float | None = None
    delta_y: float | None = None
    reason: str = ""


@dataclass
class EnvironmentDiff:
    """Summarizes how the target set changed between two prediction ticks."""

    added_ids: list[int] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)
    removed_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    priority_changed_ids: list[int] = field(default_factory=list)
    moved_ids: list[int] = field(default_factory=list)
    damage_changed_ids: list[int] = field(default_factory=list)
    patches: list[TargetPatch] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.patches

    def describe(self) -> str:
        parts: list[str] = []
        if self.added_ids:
            parts.append(f"added {self.added_ids}")
        if self.removed_ids:
            parts.append(f"removed {self.removed_ids}")
        if self.priority_changed_ids:
            parts.append(f"priorityΔ {self.priority_changed_ids}")
        if self.moved_ids:
            parts.append(f"moved {self.moved_ids}")
        if self.damage_changed_ids:
            parts.append(f"damageΔ {self.damage_changed_ids}")
        return "; ".join(parts) if parts else "no change"


@dataclass(frozen=True)
class PredictionRiskMap:
    """
    Full spatial prediction, preserved rather than discarded after target
    extraction.

    ``fire_probability`` is the ConvLSTM's raw output — a genuine
    probability, unmodified. ``risk`` is the same *expected-damage-proxy*
    transform used for ``SuppressionTarget.damage_score`` (see
    ``ConvLSTMPredictionSource`` docstring), but evaluated at every grid
    cell instead of only at target locations — a documented proxy, not a
    validated physical hazard field. Consumers (e.g. D* Lite's optional
    risk-weighted routing) read ``risk``; nothing downstream needs the raw
    10-channel ConvLSTM tensor.
    """

    fire_probability: np.ndarray
    risk: np.ndarray
    width: int
    height: int
    tick: int

    def risk_at(self, x: float, y: float) -> float:
        """Sample ``risk`` at a (possibly fractional) mission-grid coordinate."""
        row = int(np.clip(round(y), 0, self.height - 1))
        col = int(np.clip(round(x), 0, self.width - 1))
        return float(self.risk[row, col])

    def as_cell_risk_dict(self) -> dict[tuple[int, int], float]:
        """
        Convert to a ``{(x, y): risk}`` dict keyed the same way D* Lite's
        ``Cell`` type is (col, row) = (x, y) — for risk-weighted routing.
        """
        out: dict[tuple[int, int], float] = {}
        for row in range(self.height):
            for col in range(self.width):
                out[(col, row)] = float(self.risk[row, col])
        return out


@dataclass(frozen=True)
class PredictionUpdate:
    """
    One online prediction revision.

    ``tick`` is a discrete simulation time index (maps to “every few seconds”
    in the animation). ``source_name`` records whether the update came from the
    synthetic simulator or (later) ConvLSTM. ``risk_map`` is optional — only
    sources that produce a full spatial prediction (e.g. ConvLSTM) populate
    it; ``SyntheticPredictionSource`` leaves it ``None``.
    """

    tick: int
    patches: tuple[TargetPatch, ...]
    source_name: str
    note: str = ""
    risk_map: PredictionRiskMap | None = None


class PredictionSource(ABC):
    """
    Interface for wildfire-prediction providers.

    Replace ``SyntheticPredictionSource`` with a ConvLSTM-backed source that
    converts predicted fire maps into ``PredictionUpdate`` / ``TargetPatch``
    objects — the replanner does not need to change.
    """

    @abstractmethod
    def next_update(self, env: WildfireEnvironment, tick: int) -> PredictionUpdate | None:
        """
        Return the next prediction revision, or ``None`` if the stream ended.
        """


@dataclass(frozen=True)
class ExpectedDamageConfig:
    """
    Configuration for the fire-probability -> expected-damage-proxy transform.

    ``SuppressionTarget.damage_score`` is NOT set equal to the ConvLSTM's
    predicted fire probability. P(fire) is a probability; "expected damage"
    is a different quantity (how bad it would be *if* the fire reaches that
    cell), and conflating the two silently was a modeling error. Instead:

        expected_damage_proxy = fire_probability
                                 * normalized_severity_factor   (optional)
                                 * normalized_fuel_factor        (optional)

    This is a documented PROXY, not a validated physical wildfire-damage
    model — do not present it as one.

    ``severity_factor`` is read from the driving simulator's
    ``potential_ros_map`` (potential rate of spread) at that cell, min-max
    normalized using the real training-time stats in
    ``dataset/normalization.json``. Potential ROS was chosen over
    ``intensity_map`` deliberately: intensity is only nonzero where fire is
    *currently* burning, which is nearly always false at a newly-predicted
    hotspot — using it would make the proxy collapse to zero at exactly the
    cells this is meant to evaluate. Potential ROS is defined everywhere
    (a terrain/fuel/wind-derived "how fast/intense would it get here"
    quantity) regardless of current burn state.

    ``fuel_factor`` is the simulator's ``vegetation_density`` at that cell,
    likewise min-max normalized against its real observed training range.

    Both factors are genuine values already flowing through this pipeline's
    driving ``WildfireSimulator`` — not invented numbers.
    """

    use_severity_factor: bool = True
    use_fuel_factor: bool = True


@dataclass(frozen=True)
class ConvLSTMSourceConfig:
    """Configuration for :class:`ConvLSTMPredictionSource`."""

    # Trained checkpoint / normalization stats. Defaults to
    # models/convlstm/best_model.pt and dataset/normalization.json.
    checkpoint_path: str | Path | None = None
    normalization_json: str | Path | None = None
    device: str = "auto"
    # ConvLSTM input sequence length. Must have this many frames of history
    # before a real prediction can be made (see "warm-up" note below).
    history_len: int = 20
    # Cellular-automata steps the driving WildfireSimulator advances per
    # online-replanning tick, to build fresh history for the next inference.
    steps_per_tick: int = 3
    # Predicted P(fire) above which a brand-new suppression target is proposed.
    hotspot_threshold: float = 0.55
    # Predicted P(fire) below which an existing target is dropped (extinguished).
    remove_threshold: float = 0.08
    max_new_targets_per_tick: int = 2
    seed: int | None = 42
    expected_damage: ExpectedDamageConfig = field(default_factory=ExpectedDamageConfig)


class ConvLSTMPredictionSource(PredictionSource):
    """
    ConvLSTM-backed prediction source.

    Each tick, this class:

      1. Advances a real ``src.simulator.WildfireSimulator`` (the exact
         cellular-automata physics used to generate the ConvLSTM's training
         data) a few steps, building genuine, spatially-correlated
         terrain/weather/fire history rather than synthetic random edits.
      2. Packs the rolling history into the ``(T, 10, H, W)`` channel layout
         documented in ``src.dataset`` (fire state, potential ROS, fireline
         intensity, vegetation density, slope, aspect cos/sin, wind speed,
         wind direction cos/sin) and applies the same normalization stats
         used at training time.
      3. Runs the trained ``WildfireConvLSTM`` checkpoint to obtain a real
         predicted fire-probability map for the grid.
      4. Translates that map into ``TargetPatch`` edits on the mission's
         live ``SuppressionTarget`` list: predicted probability at each
         existing target's cell nudges its ``damage_score``/``priority``;
         strong new hotspots become ``ADD_TARGET`` patches; targets whose
         predicted probability has collapsed are ``REMOVE_TARGET``'d.

    The resulting ``PredictionUpdate`` is applied through the exact same
    ``apply_prediction_update`` path as ``SyntheticPredictionSource``, so
    ``OnlineReplanner`` / NSGA-II require no changes to consume it: NSGA-II's
    damage-prevention objective already reads ``target.damage_score`` /
    ``target.priority``, which this class now derives from a real ConvLSTM
    forward pass instead of synthetic random deltas.

    This intentionally reuses the checkpoint trained by ``src/train.py``
    (the 10-channel schema ``src/dataset.py`` documents) rather than
    introducing a second, incompatible channel schema — retraining against a
    new schema is out of scope for this integration and would break the
    existing checkpoint, the YOLO bridge, and existing tests.

    Representation gap (documented, not hidden): ``WildfireEnvironment`` is a
    point/target-based mission representation (drone + suppression targets),
    while ``WildfireSimulator``/ConvLSTM operate on dense terrain/weather/fire
    rasters. This class bridges the gap by seeding the driving simulator's
    ignition points from the mission environment's current targets (weighted
    toward higher-damage targets) so the simulated fire starts in a
    spatially-consistent place, then feeds the simulator's own subsequent
    physics — not the mission targets — into ConvLSTM. The simulator's
    terrain/weather (elevation, vegetation, wind) are therefore synthetic,
    matching how the checkpoint was trained; there is currently no real
    GIS/weather feed wired into this class (see project docs for the real
    datasets that would replace this).
    """

    def __init__(self, config: ConvLSTMSourceConfig | None = None) -> None:
        self.config = config or ConvLSTMSourceConfig()
        self._model: Any = None
        self._device: Any = None
        self._simulator: Any = None
        self._norm: dict[str, dict[str, float]] | None = None

    def next_update(self, env: WildfireEnvironment, tick: int) -> PredictionUpdate | None:
        import torch

        self._ensure_ready(env)

        for _ in range(self.config.steps_per_tick):
            still_burning = self._simulator.step()
            if not still_burning:
                break

        history_len = self.config.history_len
        history_slice = self._simulator.history[-history_len:]
        if len(history_slice) < history_len:
            return PredictionUpdate(
                tick=tick,
                patches=(),
                source_name="convlstm",
                note=(
                    f"ConvLSTM warm-up: {len(history_slice)}/{history_len} "
                    "simulator frames collected, no prediction yet"
                ),
            )

        frames = _pack_frames(history_slice)
        frames = _normalize_frames(frames, self._norm)

        x = torch.from_numpy(frames).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred = self._model(x).squeeze(0).detach().cpu().numpy()

        latest = history_slice[-1]
        risk_grid = _compute_risk_grid(
            pred,
            severity_grid=latest["potential_ros_map"],
            fuel_grid=latest["vegetation_density"],
            norm=self._norm,
            damage_cfg=self.config.expected_damage,
        )
        risk_map = PredictionRiskMap(
            fire_probability=pred,
            risk=risk_grid,
            width=env.width,
            height=env.height,
            tick=tick,
        )

        patches = _predicted_map_to_patches(env, pred, risk_grid, self.config)
        note = (
            f"ConvLSTM predicted fire map (max P={float(pred.max()):.3f}, "
            f"mean P={float(pred.mean()):.3f}); {len(patches)} patch(es)"
        )
        return PredictionUpdate(
            tick=tick,
            patches=tuple(patches),
            source_name="convlstm",
            note=note,
            risk_map=risk_map,
        )

    def _ensure_ready(self, env: WildfireEnvironment) -> None:
        if self._simulator is not None:
            return

        import json

        from src.config import SimulationConfig
        from src.simulator import WildfireSimulator
        from src.vision.convlstm_bridge import load_convlstm_checkpoint, resolve_device
        from src.vision.paths import DEFAULT_NORM_JSON, resolve_convlstm_checkpoint

        self._device = resolve_device(self.config.device)
        ckpt_path = resolve_convlstm_checkpoint(self.config.checkpoint_path)
        self._model = load_convlstm_checkpoint(ckpt_path, self._device)

        norm_path = Path(self.config.normalization_json or DEFAULT_NORM_JSON)
        with norm_path.open("r", encoding="utf-8") as fh:
            self._norm = json.load(fh)

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        ignition_points = _ignition_points_from_targets(env)
        sim_cfg = SimulationConfig(
            rows=env.height,
            cols=env.width,
            ignition_points=ignition_points,
        )
        self._simulator = WildfireSimulator(sim_cfg)


def _ignition_points_from_targets(env: WildfireEnvironment) -> list[tuple[int, int]]:
    """Seed the driving simulator's ignition cells from the mission's current targets."""
    if not env.targets:
        mid = (env.height // 2, env.width // 2)
        return [mid]
    ranked = sorted(env.targets, key=lambda t: t.damage_score, reverse=True)
    points: list[tuple[int, int]] = []
    for target in ranked[:3]:
        row = int(np.clip(round(target.y), 0, env.height - 1))
        col = int(np.clip(round(target.x), 0, env.width - 1))
        points.append((row, col))
    return points


def _pack_frames(history_slice: list[dict[str, Any]]) -> np.ndarray:
    """
    Pack a slice of ``WildfireSimulator.history`` into ``(T, 10, H, W)``.

    Channel layout mirrors ``src.data_exporter.export_simulation_to_pytorch``
    exactly (fire state, potential ROS, fireline intensity, vegetation
    density, slope, aspect cos/sin, wind speed, wind direction cos/sin) —
    duplicated here (rather than imported) to keep this prediction source
    free of a hard dependency on the file-writing exporter.
    """
    rows, cols = history_slice[0]["state_grid"].shape
    frames = np.zeros((len(history_slice), 10, rows, cols), dtype=np.float32)
    for t, step in enumerate(history_slice):
        frames[t, 0] = step["state_grid"].astype(np.float32)
        frames[t, 1] = step["potential_ros_map"]
        frames[t, 2] = step["intensity_map"]
        frames[t, 3] = step["vegetation_density"]
        frames[t, 4] = step["slope"]
        aspect_rad = np.radians(step["aspect"])
        frames[t, 5] = np.cos(aspect_rad)
        frames[t, 6] = np.sin(aspect_rad)
        frames[t, 7] = np.full((rows, cols), step["wind_speed"], dtype=np.float32)
        wind_rad = np.radians(step["wind_direction"])
        frames[t, 8] = np.full((rows, cols), np.cos(wind_rad), dtype=np.float32)
        frames[t, 9] = np.full((rows, cols), np.sin(wind_rad), dtype=np.float32)
    return frames


# channel index -> normalization.json key, mirrors src.dataset.NORM_CHANNELS.
_NORM_CHANNELS = {
    1: "potential_ros",
    2: "fireline_intensity",
    3: "vegetation_density",
    4: "slope",
    7: "wind_speed",
}


def _normalize_frames(frames: np.ndarray, norm: dict[str, dict[str, float]] | None) -> np.ndarray:
    """Apply the training-time (mean, std) normalization to channels 1,2,3,4,7."""
    if norm is None:
        return frames
    out = frames.copy()
    for channel_idx, key in _NORM_CHANNELS.items():
        mean = float(norm[key]["mean"])
        std = float(norm[key]["std"]) or 1.0
        out[:, channel_idx] = (out[:, channel_idx] - mean) / std
    return out


def _normalize_grid(grid: np.ndarray, key: str, norm: dict[str, dict[str, float]] | None) -> np.ndarray:
    """Min-max normalize ``grid`` using real training-time stats for ``key``."""
    if norm is None or key not in norm:
        return np.ones_like(grid, dtype=np.float32)
    lo, hi = float(norm[key]["min"]), float(norm[key]["max"])
    if hi <= lo:
        return np.zeros_like(grid, dtype=np.float32)
    return np.clip((grid.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _compute_risk_grid(
    pred: np.ndarray,
    severity_grid: np.ndarray,
    fuel_grid: np.ndarray,
    norm: dict[str, dict[str, float]] | None,
    damage_cfg: ExpectedDamageConfig,
) -> np.ndarray:
    """
    Expected-damage-proxy grid: ``fire_probability * severity_norm * fuel_norm``.

    Documented proxy (see ``ExpectedDamageConfig``), not a physical model.
    """
    risk = pred.astype(np.float32).copy()
    if damage_cfg.use_severity_factor:
        risk = risk * _normalize_grid(severity_grid, "potential_ros", norm)
    if damage_cfg.use_fuel_factor:
        risk = risk * _normalize_grid(fuel_grid, "vegetation_density", norm)
    return np.clip(risk, 0.0, 1.0)


def _predicted_map_to_patches(
    env: WildfireEnvironment,
    pred: np.ndarray,
    risk_grid: np.ndarray,
    config: ConvLSTMSourceConfig,
) -> list[TargetPatch]:
    """
    Convert a predicted (H, W) fire-probability map into target patches.

    ``pred`` (raw P(fire)) drives urgency/``priority`` and the
    add/remove-target thresholds (whether fire is present at all).
    ``risk_grid`` (the expected-damage proxy, see ``ExpectedDamageConfig``)
    drives ``damage_score`` — a deliberately different quantity, not P(fire)
    relabeled.
    """
    patches: list[TargetPatch] = []
    covered_cells: set[tuple[int, int]] = set()

    for target in env.targets:
        row = int(np.clip(round(target.y), 0, env.height - 1))
        col = int(np.clip(round(target.x), 0, env.width - 1))
        covered_cells.add((row, col))
        p = float(pred[row, col])
        expected_damage = float(risk_grid[row, col])

        if p < config.remove_threshold:
            patches.append(
                TargetPatch(
                    kind=UpdateKind.REMOVE_TARGET,
                    target_id=target.id,
                    reason=f"ConvLSTM predicts P(fire)={p:.3f} (extinguished)",
                )
            )
            continue

        delta_damage = expected_damage - target.damage_score
        if abs(delta_damage) > 1e-3:
            patches.append(
                TargetPatch(
                    kind=UpdateKind.DAMAGE_CHANGE,
                    target_id=target.id,
                    delta_damage=float(np.clip(delta_damage, -1.0, 1.0)),
                    reason=f"expected-damage-proxy {expected_damage:.3f} (P(fire)={p:.3f})",
                )
            )

        delta_priority = p - target.priority
        if abs(delta_priority) > 1e-3:
            kind = UpdateKind.PRIORITY_INCREASE if delta_priority > 0 else UpdateKind.PRIORITY_DECREASE
            patches.append(
                TargetPatch(
                    kind=kind,
                    target_id=target.id,
                    delta_priority=float(np.clip(delta_priority, -1.0, 1.0)),
                    reason=f"ConvLSTM predicted priority (P(fire)={p:.3f})",
                )
            )

    for row, col, p in _find_hotspots(pred, config, covered_cells):
        expected_damage = float(risk_grid[row, col])
        patches.append(
            TargetPatch(
                kind=UpdateKind.ADD_TARGET,
                new_x=float(col),
                new_y=float(row),
                new_priority=float(p),
                new_damage=expected_damage,
                reason=(
                    f"ConvLSTM predicts new hotspot P(fire)={p:.3f}, "
                    f"expected-damage-proxy={expected_damage:.3f}"
                ),
            )
        )

    return patches


def _find_hotspots(
    pred: np.ndarray,
    config: ConvLSTMSourceConfig,
    covered_cells: set[tuple[int, int]],
    min_separation: int = 5,
) -> list[tuple[int, int, float]]:
    """Greedy non-max-suppression peak search for new, uncovered hotspots."""
    candidates = np.argwhere(pred >= config.hotspot_threshold)
    if candidates.size == 0:
        return []

    scored = sorted(
        ((int(r), int(c), float(pred[r, c])) for r, c in candidates),
        key=lambda item: item[2],
        reverse=True,
    )

    chosen: list[tuple[int, int, float]] = []
    for row, col, p in scored:
        if len(chosen) >= config.max_new_targets_per_tick:
            break
        if (row, col) in covered_cells:
            continue
        too_close = any(
            abs(row - cr) < min_separation and abs(col - cc) < min_separation
            for cr, cc, _ in chosen
        )
        if too_close:
            continue
        chosen.append((row, col, p))
    return chosen
