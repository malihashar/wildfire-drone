"""
Lightweight integration tests: ConvLSTM prediction -> NSGA-II mission planning.

Deliberately small (tiny grid, few targets, small NSGA-II budget, short
ConvLSTM history) so this runs in seconds using the real trained checkpoint
under models/convlstm/ -- no training and no large dataset generation here.
"""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mission.config.settings import (
    GridConfig,
    MissionConfig,
    OptimizerConfig,
    TargetGenerationConfig,
)
from mission.optimizer.nsga2 import NSGA2MissionOptimizer
from mission.replanning.config import OnlineReplanConfig
from mission.replanning.online_replanner import OnlineReplanner
from mission.simulation.dynamics import apply_prediction_update
from mission.simulation.environment import WildfireEnvironment
from mission.simulation.prediction_source import (
    ConvLSTMPredictionSource,
    ConvLSTMSourceConfig,
    _normalize_frames,
    _pack_frames,
)

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "models" / "convlstm" / "best_model.pt"


def _small_env(seed: int = 7, n_targets: int = 5, grid: int = 20) -> WildfireEnvironment:
    cfg = MissionConfig(
        grid=GridConfig(width=grid, height=grid),
        targets=TargetGenerationConfig(min_targets=n_targets, max_targets=n_targets),
        optimizer=OptimizerConfig(population_size=8, n_generations=4, max_mission_targets=3),
        seed=seed,
    )
    return WildfireEnvironment.create_synthetic(cfg)


def _small_source_config(grid: int = 20) -> ConvLSTMSourceConfig:
    return ConvLSTMSourceConfig(
        checkpoint_path=CHECKPOINT_PATH,
        history_len=4,
        steps_per_tick=2,
        hotspot_threshold=0.5,
        remove_threshold=0.05,
        max_new_targets_per_tick=1,
        seed=7,
    )


@unittest.skipUnless(CHECKPOINT_PATH.exists(), "trained ConvLSTM checkpoint not present")
class TestConvLSTMToNSGA2Integration(unittest.TestCase):
    def test_dataset_to_model_tensor_shapes(self):
        """Simulator history packs into the (T, 10, H, W) shape ConvLSTM expects."""
        from src.config import SimulationConfig
        from src.simulator import WildfireSimulator

        sim = WildfireSimulator(SimulationConfig(rows=12, cols=12, ignition_points=[(6, 6)]))
        for _ in range(5):
            sim.step()

        frames = _pack_frames(sim.history[-4:])
        self.assertEqual(frames.shape, (4, 10, 12, 12))

        normalized = _normalize_frames(frames, {
            "potential_ros": {"mean": 0.0, "std": 1.0},
            "fireline_intensity": {"mean": 0.0, "std": 1.0},
            "vegetation_density": {"mean": 0.0, "std": 1.0},
            "slope": {"mean": 0.0, "std": 1.0},
            "wind_speed": {"mean": 0.0, "std": 1.0},
        })
        self.assertEqual(normalized.shape, frames.shape)

    def test_model_to_prediction_shape(self):
        """WildfireConvLSTM produces one probability value per grid cell."""
        import torch

        from src.convlstm import WildfireConvLSTM
        from src.vision.convlstm_bridge import load_convlstm_checkpoint, resolve_device

        device = resolve_device("cpu")
        model = load_convlstm_checkpoint(CHECKPOINT_PATH, device)
        x = torch.zeros(1, 4, 10, 12, 12)
        with torch.no_grad():
            pred = model(x)
        self.assertEqual(tuple(pred.shape), (1, 12, 12))
        self.assertTrue(bool(((pred >= 0) & (pred <= 1)).all()))

    def test_prediction_updates_target_severity_nsga2_compatible(self):
        """A ConvLSTM update mutates SuppressionTarget damage/priority in place."""
        env = _small_env()
        before = {t.id: (t.damage_score, t.priority) for t in env.targets}

        source = ConvLSTMPredictionSource(_small_source_config())
        update = None
        for tick in range(1, 6):
            update = source.next_update(env, tick=tick)
            if update is not None and update.patches:
                apply_prediction_update(env, update)
                break

        self.assertIsNotNone(update, "prediction source returned no update at all")
        self.assertTrue(update.patches, "ConvLSTM never produced a non-empty update within 5 ticks")
        self.assertEqual(update.source_name, "convlstm")

        after = {t.id: (t.damage_score, t.priority) for t in env.targets}
        changed = any(
            tid not in after or after.get(tid) != vals for tid, vals in before.items()
        ) or (set(after) != set(before))
        self.assertTrue(changed, "ConvLSTM update did not change any target's severity")

        # NSGA-II must be able to consume the mutated environment unmodified.
        optimizer = NSGA2MissionOptimizer(env, env_config_or_default(env))
        result = optimizer.optimize(seed=1)
        self.assertGreaterEqual(result.n_solutions, 1)

    def test_nsga2_optimization_changes_with_predicted_fire_map(self):
        """Same seed, same NSGA-II config: prediction changes -> mission changes."""
        env_before = _small_env()
        opt_cfg = OptimizerConfig(population_size=8, n_generations=4, max_mission_targets=3)

        result_before = NSGA2MissionOptimizer(env_before, opt_cfg).optimize(seed=1)
        damage_before = result_before.best_damage_plan().objectives.damage_prevented

        env_after = deepcopy(env_before)
        source = ConvLSTMPredictionSource(_small_source_config())
        applied = False
        for tick in range(1, 6):
            update = source.next_update(env_after, tick=tick)
            if update is not None and update.patches:
                apply_prediction_update(env_after, update)
                applied = True
                break
        self.assertTrue(applied, "ConvLSTM never produced a usable update within 5 ticks")

        result_after = NSGA2MissionOptimizer(env_after, opt_cfg).optimize(seed=1)
        damage_after = result_after.best_damage_plan().objectives.damage_prevented

        self.assertNotAlmostEqual(
            damage_before,
            damage_after,
            places=6,
            msg="NSGA-II's best objective did not change after the ConvLSTM prediction update",
        )

    def test_existing_synthetic_replanner_still_works_independently(self):
        """Regression guard: the pre-existing synthetic online-replanning path is untouched."""
        cfg = OnlineReplanConfig(
            n_targets_initial=6,
            n_replan_events=2,
            population_size=8,
            n_generations=4,
            max_mission_targets=3,
        )
        result = OnlineReplanner(config=cfg).run()
        self.assertGreaterEqual(result.initial_n_pareto, 1)
        self.assertLessEqual(len(result.events), cfg.n_replan_events)


def env_config_or_default(env: WildfireEnvironment) -> OptimizerConfig:
    return OptimizerConfig(population_size=8, n_generations=4, max_mission_targets=3)


if __name__ == "__main__":
    unittest.main()
