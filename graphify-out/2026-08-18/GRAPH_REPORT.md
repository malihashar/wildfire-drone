# Graph Report - wildfire-drone  (2026-08-16)

## Corpus Check
- 95 files · ~196,748 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 807 nodes · 2034 edges · 43 communities (37 shown, 6 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 169 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `19bc85ff`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- common.py
- VisualizationConfig
- convergence.py
- MissionConfig
- fitness_runtime.py
- OptimizerConfig
- WildfireConvLSTM
- OnlineReplanResult
- yolo_fire_adapter.py
- train.py
- ExperimentConfig
- ScoredMission
- EnvironmentDiff
- population_size.py
- vision/__init__.py
- YOLO Fire Detection Integration
- WildfireEnvironment
- ConvLSTMCell
- online_replanner.py
- DecodedMission
- dynamics.py
- SuppressionTarget
- OnlineReplanConfig
- OptimizationResult
- Wildfire UAV Mission Optimization — Experiment Report
- .evaluate_decoded
- resolve_convlstm_checkpoint
- constraints.py
- OnlineReplanner
- ndarray
- predict_wildfire_from_image
- eval_convlstm_held_out.py
- population_objective_stats
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLitePlanner
- preprocess_dataset.py
- .target_xy
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
- `main()` --calls--> `WildfireConvLSTM`  [EXTRACTED]
  scripts/eval_convlstm_held_out.py → src/convlstm.py
- `main()` --calls--> `aggregate_metrics()`  [EXTRACTED]
  scripts/eval_convlstm_held_out.py → src/train.py
- `main()` --calls--> `compute_metrics()`  [EXTRACTED]
  scripts/eval_convlstm_held_out.py → src/train.py
- `main()` --calls--> `resolve_device()`  [EXTRACTED]
  scripts/eval_convlstm_held_out.py → src/vision/convlstm_bridge.py
- `TestWildfireSimulator` --uses--> `WildfireDataset`  [INFERRED]
  tests/test_wildfire.py → src/data_exporter.py

## Import Cycles
- None detected.

## Communities (43 total, 6 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (54): FuncAnimation, ndarray, main(), SimulationConfig, Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction. (+46 more)

### Community 1 - "common.py"
Cohesion: 0.11
Nodes (35): ensure_paths(), make_environment(), mean_std(), Any, Path, Shared helpers for optimization-performance experiments. Keeps environment…, Write a list of dict rows to CSV., Dump the exact ``ExperimentConfig`` used for a run, for reproducibility. (+27 more)

### Community 2 - "VisualizationConfig"
Cohesion: 0.06
Nodes (50): Matplotlib scene rendering options., VisualizationConfig, Path, Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), Path, Phase-1 experiment: generate a synthetic scene and save a visualization. (+42 more)

### Community 3 - "convergence.py"
Cohesion: 0.13
Nodes (24): Axes, Figure, Save a matplotlib figure as PNG., Apply a consistent research-plot style., save_figure(), style_axes(), ConvergenceCallback, ConvergenceHistory (+16 more)

### Community 4 - "MissionConfig"
Cohesion: 0.13
Nodes (19): Configuration for the mission-planning prototype., GridConfig, MissionConfig, Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, Top-level configuration for the mission-planning research prototype., Spatial domain for the wildfire mission environment., Parameters for synthetic suppression-target generation. Later: ConvLSTM…, TargetGenerationConfig (+11 more)

### Community 5 - "fitness_runtime.py"
Cohesion: 0.11
Nodes (20): ExperimentPaths, Path, Output layout for CSVs, plots, and the Markdown report., FitnessRuntimeResult, main(), _plot_fitness_vs_runtime(), print_threshold_report(), Any (+12 more)

### Community 6 - "OptimizerConfig"
Cohesion: 0.17
Nodes (13): ElementwiseProblem, OptimizerConfig, NSGA-II mission-optimization settings (pymoo). Inspired by the multi-objective…, MissionScorer, Evaluate permutation chromosomes against the three research objectives.…, Mission optimizer package. Uses official pymoo NSGA-II with permutation…, NSGA2MissionOptimizer, Single-UAV NSGA-II mission optimizer using official pymoo operators. Adapts the… (+5 more)

### Community 7 - "WildfireConvLSTM"
Cohesion: 0.13
Nodes (23): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, WildfireConvLSTM, check(), main(), Smoke-test for WildfireDataset + WildfireConvLSTM. Run from project root:…, load_convlstm_checkpoint(), load_terrain_weather_from_simulation() (+15 more)

### Community 8 - "OnlineReplanResult"
Cohesion: 0.15
Nodes (19): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, and write a report., run_online_replan_demo(), OnlineReplanResult, Full online-replanning demo transcript., Path, Markdown report for the online replanning architecture demo. (+11 more)

### Community 9 - "yolo_fire_adapter.py"
Cohesion: 0.10
Nodes (29): _area_pool_to_grid(), _box_to_mask(), build_convlstm_sequence(), cell_to_image_rect(), _class_weight(), detections_to_fire_grid(), _filter_fire_grid(), FireBoxDetection (+21 more)

### Community 10 - "train.py"
Cohesion: 0.07
Nodes (30): DataLoader, Module, Optimizer, Sampler, build_loaders(), Dataset, Path, Tensor (+22 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.36
Nodes (19): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+11 more)

### Community 12 - "ScoredMission"
Cohesion: 0.15
Nodes (15): A Pareto plan paired with its scalar selection score., ScoredMission, build_execution_request(), DStarLiteMissionExecutor, MissionExecutionRequest, MissionExecutor, NullMissionExecutor, ABC (+7 more)

### Community 13 - "EnvironmentDiff"
Cohesion: 0.21
Nodes (16): EnvironmentDiff, Summarizes how the target set changed between two prediction ticks., build_animation_frames(), _draw_drone(), _draw_path(), _draw_stats_panel(), _draw_targets(), FrameStats (+8 more)

### Community 14 - "population_size.py"
Cohesion: 0.27
Nodes (11): Summarize a final Pareto set in human-readable objective units., result_objective_stats(), Experiment runners for the mission-planning research prototype., _plot_pop_objectives(), _plot_pop_pareto(), _plot_pop_runtime(), Any, Path (+3 more)

### Community 15 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "WildfireEnvironment"
Cohesion: 0.17
Nodes (13): Enum, Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, Phase-1 research environment for mission planning. Attributes ---------- width,…, Human-readable scene summary for logs / demos., WildfireEnvironment (+5 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.15
Nodes (9): ConvLSTM, ConvLSTMCell, Tensor, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no… (+1 more)

### Community 19 - "online_replanner.py"
Cohesion: 0.18
Nodes (13): _explain_replan(), Online NSGA-II replanning loop for dynamic wildfire predictions. Inspired by…, One online replanning cycle after a prediction update., ReplanEvent, ConvLSTMPredictionSource, PredictionSource, PredictionUpdate, ABC (+5 more)

### Community 20 - "DecodedMission"
Cohesion: 0.21
Nodes (10): decode_permutation(), DecodedMission, ndarray, Permutation chromosome encoding / decoding for single-UAV missions. Chromosome…, Feasible mission extracted from a permutation chromosome., Convert a full permutation into a constraint-feasible ordered mission. Walks…, Return the mission as an ordered list of target IDs., target_id_sequence() (+2 more)

### Community 21 - "dynamics.py"
Cohesion: 0.25
Nodes (8): apply_prediction_update(), Synthetic wildfire-prediction dynamics. Randomly edits priorities, positions,…, Apply a prediction update in-place and return the resulting diff. This is the…, _unique(), clamp(), Lightweight geometry helpers shared across simulation and (later) planning., Clamp ``value`` into ``[low, high]``., Shared utilities for the mission package.

### Community 22 - "SuppressionTarget"
Cohesion: 0.11
Nodes (19): Mission fitness / scoring., objective_battery_usage(), objective_damage_prevented(), objective_travel_distance(), Isolated multi-objective fitness functions for UAV suppression missions. pymoo…, Objective 1 (to maximize): predicted damage prevented by the mission.…, Objective 2 (to minimize): total Euclidean tour length. Path: drone start →…, Objective 3 (to minimize): approximate battery consumption. Phase-2 proxy:… (+11 more)

### Community 23 - "OnlineReplanConfig"
Cohesion: 0.14
Nodes (19): build_config(), main(), parse_args(), Namespace, MissionSelectionConfig, _norm_max_better(), _norm_min_better(), ndarray (+11 more)

### Community 24 - "OptimizationResult"
Cohesion: 0.13
Nodes (11): MissionPlan, One Pareto-optimal (or candidate) suppression mission., Ordered list of suppression target IDs to visit., _nondominated_mask(), OptimizationResult, ndarray, Boolean mask of non-dominated rows for a minimization matrix ``F``., Pareto set returned by ``NSGA2MissionOptimizer.optimize``. (+3 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - ".evaluate_decoded"
Cohesion: 0.18
Nodes (10): FitnessVector, MissionObjectives, Any, ndarray, Score an already-decoded feasible (or near-feasible) mission., Evaluate a mission chromosome or ID sequence. Accepts a permutation…, Package a fitness evaluation into a ``MissionPlan``., Human-readable (maximization-friendly) objective values. (+2 more)

### Community 27 - "resolve_convlstm_checkpoint"
Cohesion: 0.27
Nodes (7): Path, Default model and data paths for the vision + ConvLSTM product flow., Resolve an optional user path, falling back to a project default., Resolve a ConvLSTM checkpoint file or checkpoint directory., resolve_convlstm_checkpoint(), resolve_path(), TestVisionPaths

### Community 28 - "constraints.py"
Cohesion: 0.20
Nodes (9): constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions., True if the mission tour exceeds the configured distance budget., Inequality constraint g <= 0 for pymoo. g = n_targets - max_mission_targets, Inequality constraint g <= 0 for pymoo. g = travel_distance -…, True if the mission visits more targets than allowed., violates_max_distance() (+1 more)

### Community 29 - "OnlineReplanner"
Cohesion: 0.29
Nodes (6): _advance_drone_one_hop(), OnlineReplanner, Execute the initial plan plus ``n_replan_events`` online revisions., Move the UAV to the first still-existing waypoint of the current mission., Orchestrates prediction updates → NSGA-II → mission scoring → hand-off.…, _snapshot_env()

### Community 31 - "predict_wildfire_from_image"
Cohesion: 0.13
Nodes (18): FireDetection, main(), predict_wildfire_from_image(), device, ndarray, Path, Outputs from the YOLO + ConvLSTM product flow., Run the full product flow on one image. 1. YOLO segmentation produces the… (+10 more)

### Community 32 - "eval_convlstm_held_out.py"
Cohesion: 0.53
Nodes (5): local_overlap(), main(), normalize(), save_comparison_plot(), Tensor

### Community 35 - "population_objective_stats"
Cohesion: 0.25
Nodes (7): compute_hypervolume(), hypervolume_reference_point(), population_objective_stats(), ndarray, Hypervolume of the non-dominated set in ``F`` (minimization)., Best / average objective stats in human-readable (maximize-damage) form. ``F``…, Fixed reference point for fair HV comparison across generations. Objectives are…

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.25
Nodes (7): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Research Comparison — Zhu et al. (2025) vs This Project, What has been demonstrated so far, What was deliberately changed, What was taken from the paper

### Community 38 - "DStarLitePlanner"
Cohesion: 0.40
Nodes (3): DStarLitePlanner, Local path planner interface (future — not implemented yet). D* Lite will…, Placeholder for future D* Lite local planning.

## Knowledge Gaps
- **27 isolated node(s):** `ConvLSTM Integration`, `Failure Cases`, `Image-To-Grid Flow`, `Model Choice`, `Output Contract` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `common.py`, `VisualizationConfig`, `MissionConfig`, `OptimizerConfig`, `OnlineReplanResult`, `.target_xy`, `ExperimentConfig`, `ScoredMission`, `EnvironmentDiff`, `online_replanner.py`, `dynamics.py`, `SuppressionTarget`, `OptimizationResult`, `.evaluate_decoded`, `OnlineReplanner`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `OptimizationResult` connect `OptimizationResult` to `common.py`, `VisualizationConfig`, `MissionConfig`, `OptimizerConfig`, `OnlineReplanResult`, `ExperimentConfig`, `ScoredMission`, `population_size.py`, `WildfireEnvironment`, `online_replanner.py`, `SuppressionTarget`, `OnlineReplanConfig`, `OnlineReplanner`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Why does `ExperimentConfig` connect `ExperimentConfig` to `common.py`, `convergence.py`, `MissionConfig`, `fitness_runtime.py`, `population_size.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `FitnessVector`) actually correct?**
  _`OptimizerConfig` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `OptimizationResult` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`OptimizationResult` has 13 INFERRED edges - model-reasoned connections that need verification._