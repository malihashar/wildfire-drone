# Graph Report - wildfire-drone  (2026-08-16)

## Corpus Check
- 90 files · ~106,905 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 801 nodes · 2018 edges · 45 communities (34 shown, 11 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 169 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b9cb61dd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- common.py
- VisualizationConfig
- convergence.py
- MissionConfig
- fitness_runtime.py
- WildfireEnvironment
- convlstm_bridge.py
- OnlineReplanResult
- detections_to_fire_grid
- train.py
- ExperimentConfig
- online_replanner.py
- yolo_fire_adapter.py
- population_size.py
- vision/__init__.py
- YOLO Fire Detection Integration
- SyntheticPredictionSource
- ConvLSTMCell
- .mission_order
- .predicted_damage
- fitness/__init__.py
- SuppressionTarget
- score_mission
- OptimizationResult
- Wildfire UAV Mission Optimization — Experiment Report
- DecodedMission
- dataset.py
- .as_dict
- WildfireConvLSTM
- ndarray
- YoloFireSegmenter
- plot_mission_path.py
- population_objective_stats
- WildfireDataset
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLitePlanner
- preprocess_dataset.py
- .target_xy
- ._evaluate
- setup_mac.sh
- upload_helper.py
- upload_to_kaggle.sh

## God Nodes (most connected - your core abstractions)
1. `WildfireEnvironment` - 81 edges
2. `ExperimentConfig` - 40 edges
3. `OptimizerConfig` - 38 edges
4. `OptimizationResult` - 37 edges
5. `OnlineReplanResult` - 31 edges
6. `SuppressionTarget` - 29 edges
7. `OnlineReplanner` - 28 edges
8. `MissionConfig` - 28 edges
9. `MissionPlan` - 26 edges
10. `DroneState` - 25 edges

## Surprising Connections (you probably didn't know these)
- `TestConvLSTMBridge` --uses--> `WildfireConvLSTM`  [INFERRED]
  tests/test_convlstm_bridge.py → src/convlstm.py
- `TestWildfireSimulator` --uses--> `WildfireDataset`  [INFERRED]
  tests/test_wildfire.py → src/data_exporter.py
- `TestWildfireSimulator` --uses--> `WildfireSimulator`  [INFERRED]
  tests/test_wildfire.py → src/simulator.py
- `main()` --calls--> `predict_wildfire_from_image()`  [EXTRACTED]
  scripts/predict_wildfire.py → src/vision/pipeline.py
- `main()` --calls--> `SimulationConfig`  [EXTRACTED]
  run_simulation.py → src/config.py

## Import Cycles
- None detected.

## Communities (45 total, 11 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (54): FuncAnimation, ndarray, main(), SimulationConfig, Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction. (+46 more)

### Community 1 - "common.py"
Cohesion: 0.11
Nodes (35): ensure_paths(), make_environment(), mean_std(), Any, Path, Shared helpers for optimization-performance experiments. Keeps environment…, Write a list of dict rows to CSV., Dump the exact ``ExperimentConfig`` used for a run, for reproducibility. (+27 more)

### Community 2 - "VisualizationConfig"
Cohesion: 0.13
Nodes (26): Matplotlib scene rendering options., VisualizationConfig, Path, Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), Visualization utilities for mission scenes, Pareto fronts, and paths., plot_pareto_front() (+18 more)

### Community 3 - "convergence.py"
Cohesion: 0.13
Nodes (24): Axes, Figure, Save a matplotlib figure as PNG., Apply a consistent research-plot style., save_figure(), style_axes(), ConvergenceCallback, ConvergenceHistory (+16 more)

### Community 4 - "MissionConfig"
Cohesion: 0.10
Nodes (30): Configuration for the mission-planning prototype., GridConfig, MissionConfig, Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, Top-level configuration for the mission-planning research prototype., Spatial domain for the wildfire mission environment., Parameters for synthetic suppression-target generation. Later: ConvLSTM…, TargetGenerationConfig (+22 more)

### Community 5 - "fitness_runtime.py"
Cohesion: 0.11
Nodes (20): ExperimentPaths, Path, Output layout for CSVs, plots, and the Markdown report., FitnessRuntimeResult, main(), _plot_fitness_vs_runtime(), print_threshold_report(), Any (+12 more)

### Community 6 - "WildfireEnvironment"
Cohesion: 0.09
Nodes (30): ElementwiseProblem, OptimizerConfig, NSGA-II mission-optimization settings (pymoo). Inspired by the multi-objective…, Optimization result plus wall-clock runtime in seconds., TimedOptimization, constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions. (+22 more)

### Community 7 - "convlstm_bridge.py"
Cohesion: 0.09
Nodes (31): load_convlstm_checkpoint(), load_terrain_weather_from_simulation(), predict_next_fire_from_grid(), Any, device, Path, Tensor, Bridge YOLO fire grids into the trained ConvLSTM predictor. The YOLO adapter… (+23 more)

### Community 8 - "OnlineReplanResult"
Cohesion: 0.07
Nodes (47): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, and write a report., run_online_replan_demo(), build_config(), main(), parse_args(), Namespace (+39 more)

### Community 9 - "detections_to_fire_grid"
Cohesion: 0.18
Nodes (15): cell_to_image_rect(), _class_weight(), detections_to_fire_grid(), _filter_fire_grid(), FireBoxDetection, FireGridConfig, FireMaskDetection, _passes_threshold() (+7 more)

### Community 10 - "train.py"
Cohesion: 0.14
Nodes (18): Module, Optimizer, aggregate_metrics(), bce_loss(), compute_metrics(), FocalLoss, load_model_state(), main() (+10 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.36
Nodes (19): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+11 more)

### Community 12 - "online_replanner.py"
Cohesion: 0.11
Nodes (28): Scalar mission scoring for selecting one plan from a Pareto set. Multi-…, Select the Pareto plan with the maximum scalar score., select_highest_scoring_mission(), build_execution_request(), DStarLiteMissionExecutor, MissionExecutionRequest, MissionExecutor, NullMissionExecutor (+20 more)

### Community 13 - "yolo_fire_adapter.py"
Cohesion: 0.13
Nodes (18): main(), _area_pool_to_grid(), _box_to_mask(), build_convlstm_sequence(), FireGridResult, _max_pool_to_grid(), overlay_grid_on_image(), plot_fire_grid_diagnostics() (+10 more)

### Community 14 - "population_size.py"
Cohesion: 0.27
Nodes (11): Summarize a final Pareto set in human-readable objective units., result_objective_stats(), Experiment runners for the mission-planning research prototype., _plot_pop_objectives(), _plot_pop_pareto(), _plot_pop_runtime(), Any, Path (+3 more)

### Community 15 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "SyntheticPredictionSource"
Cohesion: 0.12
Nodes (19): Enum, Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, ConvLSTMPredictionSource, PredictionSource, PredictionUpdate (+11 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.15
Nodes (9): ConvLSTM, ConvLSTMCell, Tensor, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no… (+1 more)

### Community 22 - "SuppressionTarget"
Cohesion: 0.11
Nodes (23): objective_damage_prevented(), objective_travel_distance(), Isolated multi-objective fitness functions for UAV suppression missions. pymoo…, Objective 1 (to maximize): predicted damage prevented by the mission.…, Objective 2 (to minimize): total Euclidean tour length. Path: drone start →…, decode_permutation(), ndarray, Permutation chromosome encoding / decoding for single-UAV missions. Chromosome… (+15 more)

### Community 23 - "score_mission"
Cohesion: 0.40
Nodes (6): _norm_max_better(), _norm_min_better(), ndarray, Scalar mission score (higher is better). ``score = w_d * dmg_n - w_t * travel_n…, Normalize so smaller raw values map toward 0 cost (0 = best)., score_mission()

### Community 24 - "OptimizationResult"
Cohesion: 0.15
Nodes (10): MissionPlan, One Pareto-optimal (or candidate) suppression mission., _nondominated_mask(), OptimizationResult, ndarray, Boolean mask of non-dominated rows for a minimization matrix ``F``., Pareto set returned by ``NSGA2MissionOptimizer.optimize``., Plan with the highest damage prevented (among Pareto set). (+2 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - "DecodedMission"
Cohesion: 0.13
Nodes (14): objective_battery_usage(), Objective 3 (to minimize): approximate battery consumption. Phase-2 proxy:…, FitnessVector, Any, ndarray, Score an already-decoded feasible (or near-feasible) mission., Evaluate a mission chromosome or ID sequence. Accepts a permutation…, Package a fitness evaluation into a ``MissionPlan``. (+6 more)

### Community 27 - "dataset.py"
Cohesion: 0.18
Nodes (7): DataLoader, Sampler, build_loaders(), WildfireDataset --------------- Sliding-window dataset for ConvLSTM wildfire…, Returns (train_loader, val_loader, test_loader). Training loader uses…, Yields window indices grouped by simulation file. Simulations are shuffled each…, SimulationGroupedSampler

### Community 29 - "WildfireConvLSTM"
Cohesion: 0.36
Nodes (6): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, WildfireConvLSTM, check(), main(), Smoke-test for WildfireDataset + WildfireConvLSTM. Run from project root:…

### Community 31 - "YoloFireSegmenter"
Cohesion: 0.24
Nodes (7): FireDetection, Any, Path, Thin wrapper around Ultralytics YOLO11 for fire-grid inference. Ultralytics is…, Run YOLO inference and return the 100x100 fire grid result., _resize_mask(), YoloFireSegmenter

### Community 34 - "plot_mission_path.py"
Cohesion: 0.23
Nodes (12): _draw_background_targets(), _draw_direction_arrows(), plot_mission_path(), Axes, Figure, Path, Ordered mission-path visualization for a selected Pareto plan. Draws the UAV…, Plot and save a mission path figure. (+4 more)

### Community 35 - "population_objective_stats"
Cohesion: 0.25
Nodes (7): compute_hypervolume(), hypervolume_reference_point(), population_objective_stats(), ndarray, Hypervolume of the non-dominated set in ``F`` (minimization)., Best / average objective stats in human-readable (maximize-damage) form. ``F``…, Fixed reference point for fair HV comparison across generations. Objectives are…

### Community 36 - "WildfireDataset"
Cohesion: 0.31
Nodes (5): Dataset, Path, Tensor, Parameters ---------- split_json : str | Path Path to train.json / val.json /…, WildfireDataset

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.25
Nodes (7): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Research Comparison — Zhu et al. (2025) vs This Project, What has been demonstrated so far, What was deliberately changed, What was taken from the paper

### Community 38 - "DStarLitePlanner"
Cohesion: 0.40
Nodes (3): DStarLitePlanner, Local path planner interface (future — not implemented yet). D* Lite will…, Placeholder for future D* Lite local planning.

## Knowledge Gaps
- **27 isolated node(s):** `ConvLSTM Integration`, `Failure Cases`, `Image-To-Grid Flow`, `Model Choice`, `Output Contract` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `common.py`, `VisualizationConfig`, `plot_mission_path.py`, `MissionConfig`, `OnlineReplanResult`, `.target_xy`, `ExperimentConfig`, `online_replanner.py`, `SyntheticPredictionSource`, `SuppressionTarget`, `OptimizationResult`, `DecodedMission`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `OptimizationResult` connect `OptimizationResult` to `common.py`, `VisualizationConfig`, `MissionConfig`, `WildfireEnvironment`, `OnlineReplanResult`, `ExperimentConfig`, `online_replanner.py`, `population_size.py`, `score_mission`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `ExperimentConfig` connect `ExperimentConfig` to `common.py`, `convergence.py`, `MissionConfig`, `fitness_runtime.py`, `WildfireEnvironment`, `population_size.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `FitnessVector`) actually correct?**
  _`OptimizerConfig` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `OptimizationResult` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`OptimizationResult` has 13 INFERRED edges - model-reasoned connections that need verification._