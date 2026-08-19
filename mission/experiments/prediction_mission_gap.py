"""
Validation Experiment 5 — prediction quality -> mission quality gap.

Not "is ConvLSTM accurate", but "does a better prediction actually produce a
better suppression mission". For the same target layout, builds two severity
assignments:

  - "ideal"     — from the REAL future simulator state (one real step beyond
    the history window the ConvLSTM saw) -- perfect-knowledge upper bound.
  - "predicted" — from the ConvLSTM's actual prediction at that same point.

Runs NSGA-II on both (same seed), then measures the actual end-to-end gap:
if you flew the mission NSGA-II chose from the (possibly wrong) prediction,
how much REAL damage (scored under the ideal/true severities) would you
actually have prevented, versus the best possible mission chosen with
perfect foresight? This is a single end-to-end metric rather than judging
ConvLSTM and NSGA-II in isolation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mission.config.settings import GridConfig, MissionConfig, OptimizerConfig, TargetGenerationConfig
from mission.experiments.common import ensure_paths, save_figure, style_axes, write_csv
from mission.experiments.experiment_config import ExperimentConfig
from mission.fitness.mission_selection import select_highest_scoring_mission
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from mission.simulation.environment import WildfireEnvironment


@dataclass(frozen=True)
class PredictionMissionGapResult:
    trial_rows: list[dict[str, Any]]
    stats: dict[str, float]
    csv_path: Path
    plot_path: Path


def run_prediction_mission_gap_experiment(
    exp: ExperimentConfig | None = None,
    n_trials: int = 8,
    n_targets: int = 12,
    grid_size: int = 30,
    history_len: int = 20,
    steps_per_tick: int = 2,
) -> PredictionMissionGapResult:
    import torch

    from src.config import SimulationConfig
    from src.simulator import WildfireSimulator
    from src.vision.convlstm_bridge import load_convlstm_checkpoint, resolve_device
    from src.vision.paths import DEFAULT_NORM_JSON, resolve_convlstm_checkpoint
    import json as _json

    from mission.simulation.prediction_source import (
        ExpectedDamageConfig,
        _compute_risk_grid,
        _normalize_frames,
        _pack_frames,
    )

    exp = exp or ExperimentConfig()
    paths = ensure_paths(exp.paths)

    device = resolve_device("auto")
    model = load_convlstm_checkpoint(resolve_convlstm_checkpoint(None), device)
    with open(DEFAULT_NORM_JSON) as f:
        norm = _json.load(f)
    damage_cfg = ExpectedDamageConfig()

    rows: list[dict[str, Any]] = []
    for trial in range(n_trials):
        seed = exp.seed + trial

        sim_cfg = SimulationConfig(rows=grid_size, cols=grid_size, ignition_points=[(grid_size // 2, grid_size // 2)])
        simulator = WildfireSimulator(sim_cfg)
        # Warm-up burn before recording history: right after ignition the fire
        # covers only a handful of cells, so randomly-scattered targets almost
        # never overlap it, making the "ideal" damage degenerate to ~0 for
        # most trials. Let it spread first so target overlap is meaningful.
        warmup_steps = 30
        for _ in range(warmup_steps + history_len + steps_per_tick):
            if not simulator.step():
                break
        if len(simulator.history) < history_len + 1:
            continue

        history_slice = simulator.history[-(history_len + 1):-1]
        future_frame = simulator.history[-1]

        frames = _pack_frames(history_slice)
        frames = _normalize_frames(frames, norm)
        x = torch.from_numpy(frames).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).detach().cpu().numpy()

        truth = (future_frame["state_grid"] >= 1).astype(np.float32)

        pred_risk = _compute_risk_grid(
            pred, history_slice[-1]["potential_ros_map"], history_slice[-1]["vegetation_density"], norm, damage_cfg
        )
        truth_risk = _compute_risk_grid(
            truth, future_frame["potential_ros_map"], future_frame["vegetation_density"], norm, damage_cfg
        )

        opt_cfg = OptimizerConfig(population_size=40, n_generations=60, max_mission_targets=5, verbose=False)
        mission_cfg = MissionConfig(
            grid=GridConfig(width=grid_size, height=grid_size),
            targets=TargetGenerationConfig(min_targets=n_targets, max_targets=n_targets),
            optimizer=opt_cfg,
            seed=seed,
        )
        env_predicted = WildfireEnvironment.create_synthetic(mission_cfg)
        env_ideal = copy.deepcopy(env_predicted)

        for t in env_predicted.targets:
            r, c = _cell(t, grid_size)
            t.priority = float(pred[r, c])
            t.damage_score = float(pred_risk[r, c])
        for t in env_ideal.targets:
            r, c = _cell(t, grid_size)
            t.priority = float(truth[r, c])
            t.damage_score = float(truth_risk[r, c])

        result_pred = NSGA2MissionOptimizer(env_predicted, mission_cfg).optimize(seed=seed)
        result_ideal = NSGA2MissionOptimizer(env_ideal, mission_cfg).optimize(seed=seed)
        if result_pred.n_solutions == 0 or result_ideal.n_solutions == 0:
            continue

        predicted_mission = select_highest_scoring_mission(result_pred)
        # Reference for the "best achievable" bound: the highest-damage
        # Pareto point under TRUE future severities, not the balanced
        # (knee/utopia) selection -- the latter trades off against travel
        # distance too, so it isn't guaranteed to have the max damage and
        # would make gap_ratio exceed 1 in a way that isn't a real error.
        ideal_mission = result_ideal.best_damage_plan()

        ideal_damage_map = {t.id: t.damage_score for t in env_ideal.targets}
        realized_damage = sum(ideal_damage_map.get(tid, 0.0) for tid in predicted_mission.target_ids)
        ideal_damage = ideal_mission.objectives.damage_prevented

        overlap = _jaccard(set(predicted_mission.target_ids), set(ideal_mission.target_ids))

        rows.append(
            {
                "trial": trial,
                "seed": seed,
                "predicted_mission_damage_under_own_prediction": predicted_mission.plan.objectives.damage_prevented,
                "predicted_mission_realized_damage_under_truth": realized_damage,
                "ideal_mission_damage": ideal_damage,
                "gap": max(0.0, ideal_damage - realized_damage),
                "gap_ratio": (realized_damage / ideal_damage) if ideal_damage > 0 else float("nan"),
                "target_set_jaccard_overlap": overlap,
            }
        )

    gap_ratios = [r["gap_ratio"] for r in rows if not np.isnan(r["gap_ratio"])]
    stats = {
        "n_trials": len(rows),
        "mean_gap": float(np.mean([r["gap"] for r in rows])) if rows else float("nan"),
        "mean_gap_ratio": float(np.mean(gap_ratios)) if gap_ratios else float("nan"),
        "mean_target_overlap": float(np.mean([r["target_set_jaccard_overlap"] for r in rows])) if rows else float("nan"),
    }

    csv_path = write_csv(paths.csv / "prediction_mission_gap.csv", rows)
    plot_path = _plot(rows, stats, paths.plots / "prediction_mission_gap.png")

    return PredictionMissionGapResult(trial_rows=rows, stats=stats, csv_path=csv_path, plot_path=plot_path)


def _cell(target, grid_size: int) -> tuple[int, int]:
    r = int(np.clip(round(target.y), 0, grid_size - 1))
    c = int(np.clip(round(target.x), 0, grid_size - 1))
    return r, c


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _plot(rows: list[dict[str, Any]], stats: dict[str, float], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    trials = [r["trial"] for r in rows]
    axes[0].bar([t - 0.2 for t in trials], [r["ideal_mission_damage"] for r in rows], width=0.4, color="#4a7c59", label="Ideal mission (true future)")
    axes[0].bar([t + 0.2 for t in trials], [r["predicted_mission_realized_damage_under_truth"] for r in rows], width=0.4, color="#c45c26", label="Predicted mission (realized under truth)")
    style_axes(axes[0], "Ideal vs Predicted Mission — Realized Damage", "Trial", "Damage (scored under true future)")
    axes[0].legend(fontsize=8)

    axes[1].scatter([r["gap_ratio"] for r in rows], [r["target_set_jaccard_overlap"] for r in rows],
                     s=80, color="#8a4baf", edgecolors="black")
    style_axes(axes[1], "Mission Quality Gap vs Target-Set Overlap", "gap_ratio (realized/ideal damage)", "Jaccard overlap")

    fig.suptitle(
        f"Prediction -> Mission Quality Gap (n={stats['n_trials']}, "
        f"mean gap ratio={stats['mean_gap_ratio']:.3f}, mean overlap={stats['mean_target_overlap']:.3f})"
    )
    fig.tight_layout()
    return save_figure(fig, path)
