# Graph Report - wildfire-drone  (2026-08-19)

## Corpus Check
- 113 files · ~249,707 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1198 nodes · 3309 edges · 65 communities (59 shown, 6 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 384 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `77c2e856`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- MissionExecutionResult
- optimize_demo.py
- style_axes
- MissionConfig
- ExperimentPaths
- NSGA2MissionOptimizer
- WildfireConvLSTM
- OnlineReplanConfig
- detections_to_fire_grid
- train.py
- ExperimentConfig
- prediction_mission_gap.py
- ScoredMission
- save_figure
- vision/__init__.py
- YOLO Fire Detection Integration
- prediction_source.py
- ConvLSTMCell
- DStarLiteMissionExecutor
- mission_selection.py
- scoring.py
- WildfireDataset
- online_replanner.py
- OptimizationResult
- Wildfire UAV Mission Optimization — Experiment Report
- DStarLitePlanner
- resolve_convlstm_checkpoint
- .target_xy
- ConvLSTM Wildfire Data Architecture — Critical Review
- dynamics.py
- VisualizationConfig
- eval_convlstm_held_out.py
- SimulationGroupedSampler
- Online Replanning Architecture Demo
- common.py
- yolo_fire_adapter.py
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLite
- Wildfire UAV Mission Optimization — Experiment Report
- preprocess_dataset.py
- DecodedMission
- WildfireEnvironment
- SuppressionTarget
- setup_mac.sh
- upload_helper.py
- upload_to_kaggle.sh
- eval_convlstm_real_firms.py
- executor.py
- pipeline.py
- DStarLiteSession
- GridConfig
- OptimizerConfig
- risk_routing_ablation.py
- online_replanning_comparison.py
- plot_mission_path.py
- ExpectedDamageConfig
- csv_export.py
- animate_replan.py
- plot_online_replan_summary
- OnlineReplanResult
- population_objective_stats
- cell_to_image_rect

## God Nodes (most connected - your core abstractions)
1. `WildfireEnvironment` - 110 edges
2. `OptimizerConfig` - 72 edges
3. `ExperimentConfig` - 59 edges
4. `MissionConfig` - 52 edges
5. `NSGA2MissionOptimizer` - 48 edges
6. `GridConfig` - 44 edges
7. `DStarLite` - 42 edges
8. `DroneState` - 42 edges
9. `TargetGenerationConfig` - 41 edges
10. `OptimizationResult` - 39 edges

## Surprising Connections (you probably didn't know these)
- `TestDStarLiteSessionIncremental` --uses--> `GridConfig`  [INFERRED]
  tests/test_architecture_fixes.py → mission/config/settings.py
- `TestExecutorGenuineIncrementalReuse` --uses--> `GridConfig`  [INFERRED]
  tests/test_architecture_fixes.py → mission/config/settings.py
- `TestExpectedDamageProxy` --uses--> `GridConfig`  [INFERRED]
  tests/test_architecture_fixes.py → mission/config/settings.py
- `TestPermutationDecodingIsASubset` --uses--> `GridConfig`  [INFERRED]
  tests/test_architecture_fixes.py → mission/config/settings.py
- `TestPredictionRiskMap` --uses--> `GridConfig`  [INFERRED]
  tests/test_architecture_fixes.py → mission/config/settings.py

## Import Cycles
- None detected.

## Communities (65 total, 6 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (53): FuncAnimation, main(), Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction., SimulationConfig, TerrainConfig (+45 more)

### Community 1 - "MissionExecutionResult"
Cohesion: 0.08
Nodes (41): asyncio, _abort(), arm_drone(), ArmResult, _connect_and_verify(), _connected(), _fly_to_cell(), GeoOrigin (+33 more)

### Community 2 - "optimize_demo.py"
Cohesion: 0.17
Nodes (17): Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Path, Phase-1 experiment: generate a synthetic scene and save a visualization., Create a synthetic wildfire mission scene and export a matplotlib figure.…, run_phase1_demo(), Visualization utilities for mission scenes, Pareto fronts, and paths., _draw_drone(), _draw_grid_frame() (+9 more)

### Community 3 - "style_axes"
Cohesion: 0.10
Nodes (34): hypervolume_reference_point(), Axes, Apply a consistent research-plot style., Fixed reference point for fair HV comparison across generations. Objectives are…, style_axes(), ConvergenceCallback, ConvergenceHistory, _plot_avg_fitness() (+26 more)

### Community 4 - "MissionConfig"
Cohesion: 0.13
Nodes (14): Configuration for the mission-planning prototype., MissionConfig, Top-level configuration for the mission-planning research prototype., Parameters for synthetic suppression-target generation. Later: ConvLSTM…, TargetGenerationConfig, build_mission_config(), Optimization result plus wall-clock runtime in seconds., Build a ``MissionConfig`` for a specific experimental condition. (+6 more)

### Community 5 - "ExperimentPaths"
Cohesion: 0.11
Nodes (22): ExperimentPaths, Path, Output layout for CSVs, plots, and the Markdown report., find_threshold_hits(), FitnessRuntimeResult, main(), _plot_fitness_vs_runtime(), print_threshold_report() (+14 more)

### Community 6 - "NSGA2MissionOptimizer"
Cohesion: 0.16
Nodes (12): ElementwiseProblem, Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, MissionScorer, Evaluate permutation chromosomes against the three research objectives.…, Mission optimizer package — public entry point for suppression-mission…, NSGA2MissionOptimizer, Single-UAV NSGA-II mission optimizer using official pymoo operators. Adapts the…, Run pymoo NSGA-II over suppression-target permutations. Operators (official… (+4 more)

### Community 7 - "WildfireConvLSTM"
Cohesion: 0.20
Nodes (12): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, WildfireConvLSTM, load_terrain_weather_from_simulation(), predict_next_fire_from_grid(), Any, Path, Tensor (+4 more)

### Community 8 - "OnlineReplanConfig"
Cohesion: 0.19
Nodes (15): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, CSV, plot, and a report.…, run_online_replan_demo(), build_config(), main(), parse_args(), Namespace (+7 more)

### Community 9 - "detections_to_fire_grid"
Cohesion: 0.19
Nodes (14): FireDetection, _class_weight(), detections_to_fire_grid(), FireBoxDetection, FireGridConfig, FireMaskDetection, _passes_threshold(), Any (+6 more)

### Community 10 - "train.py"
Cohesion: 0.15
Nodes (16): Module, Optimizer, bce_loss(), FocalLoss, load_model_state(), main(), parse_args(), device (+8 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.34
Nodes (21): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+13 more)

### Community 12 - "prediction_mission_gap.py"
Cohesion: 0.12
Nodes (21): _cell(), _jaccard(), _plot(), PredictionMissionGapResult, Any, Path, Validation Experiment 5 — prediction quality -> mission quality gap. Not "is…, run_prediction_mission_gap_experiment() (+13 more)

### Community 13 - "ScoredMission"
Cohesion: 0.15
Nodes (21): A Pareto plan paired with its scalar selection score., ScoredMission, _explain_replan(), EnvironmentDiff, Summarizes how the target set changed between two prediction ticks., build_animation_frames(), _draw_drone(), _draw_dstar_path() (+13 more)

### Community 14 - "save_figure"
Cohesion: 0.10
Nodes (26): Figure, Save a matplotlib figure as PNG., save_figure(), _plot_generations(), Any, Path, _plot(), Any (+18 more)

### Community 15 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "prediction_source.py"
Cohesion: 0.12
Nodes (23): ConvLSTMPredictionSource, ConvLSTMSourceConfig, _ignition_points_from_targets(), PredictionSource, ABC, Wildfire prediction update contracts. Synthetic updates stand in for future…, Interface for wildfire-prediction providers. Replace…, Configuration for :class:`ConvLSTMPredictionSource`. (+15 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.15
Nodes (9): ConvLSTM, ConvLSTMCell, Tensor, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no… (+1 more)

### Community 19 - "DStarLiteMissionExecutor"
Cohesion: 0.23
Nodes (8): DStarLiteMissionExecutor, _euclid(), MissionExecutionRequest, Cell, Plans the real obstacle-aware route for a selected mission using D* Lite.…, Payload passed from the mission selector to a local planner / executor.…, TestExecutorGenuineIncrementalReuse, TestSurrogateVsActualCost

### Community 20 - "mission_selection.py"
Cohesion: 0.26
Nodes (11): MissionSelectionConfig, _norm_max_better(), _norm_min_better(), ndarray, Scalar mission scoring for selecting one plan from a Pareto set. Multi-…, Weights for the scalarized mission score (higher is better)., Scalar mission score (higher is better). ``score = w_d * dmg_n - w_t * travel_n…, Select the Pareto plan with the maximum scalar score. (+3 more)

### Community 21 - "scoring.py"
Cohesion: 0.11
Nodes (18): constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions., True if the mission tour exceeds the configured distance budget., Inequality constraint g <= 0 for pymoo. g = n_targets - max_mission_targets, Inequality constraint g <= 0 for pymoo. g = travel_distance -…, True if the mission visits more targets than allowed., violates_max_distance() (+10 more)

### Community 22 - "WildfireDataset"
Cohesion: 0.16
Nodes (12): DataLoader, build_loaders(), Dataset, Path, Tensor, WildfireDataset --------------- Sliding-window dataset for ConvLSTM wildfire…, Returns (train_loader, val_loader, test_loader). Training loader uses…, Parameters ---------- split_json : str | Path Path to train.json / val.json /… (+4 more)

### Community 23 - "online_replanner.py"
Cohesion: 0.19
Nodes (9): Online replanning package (CREDS-inspired architecture demo). Synthetic…, OnlineReplanner, Online NSGA-II replanning loop for dynamic wildfire predictions. Inspired by…, Orchestrates prediction updates → NSGA-II → mission scoring → hand-off.…, Execute the initial plan plus ``n_replan_events`` online revisions., One online replanning cycle after a prediction update., ReplanEvent, _snapshot_env() (+1 more)

### Community 24 - "OptimizationResult"
Cohesion: 0.11
Nodes (13): MissionPlan, One Pareto-optimal (or candidate) suppression mission., Ordered list of suppression target IDs to visit., Autonomous wildfire-suppression UAV — mission planning research package. Phase…, _nondominated_mask(), OptimizationResult, ndarray, Execute NSGA-II and return the non-dominated mission plans. Parameters… (+5 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - "DStarLitePlanner"
Cohesion: 0.13
Nodes (11): _Key, path_length(), D* Lite grid path planner (Koenig & Likhachev, 2002). Decides HOW the UAV…, Total Euclidean length of a cell path (8-connected step costs)., DStarLitePlanner, Cell, Local path planner: multi-leg D* Lite mission execution. NSGA-II decides WHAT…, Plans a full multi-leg mission route using D* Lite per leg. Each leg gets a… (+3 more)

### Community 27 - "resolve_convlstm_checkpoint"
Cohesion: 0.27
Nodes (7): Path, Default model and data paths for the vision + ConvLSTM product flow., Resolve an optional user path, falling back to a project default., Resolve a ConvLSTM checkpoint file or checkpoint directory., resolve_convlstm_checkpoint(), resolve_path(), TestVisionPaths

### Community 29 - "ConvLSTM Wildfire Data Architecture — Critical Review"
Cohesion: 0.06
Nodes (30): 10. Weather representation, 11. Fuel / vegetation representation, 12. Final tensor / channel specification, 13. Data preprocessing pipeline, 14. Training/validation split strategy, 15. Real vs. simulation strategy, 16. ConvLSTM vs. alternatives, 17. Physics-informed feature integration (+22 more)

### Community 30 - "dynamics.py"
Cohesion: 0.14
Nodes (16): apply_prediction_update(), Synthetic wildfire-prediction dynamics. Randomly edits priorities, positions,…, Apply a prediction update in-place and return the resulting diff. This is the…, _unique(), PredictionUpdate, One online prediction revision. ``tick`` is a discrete simulation time index…, Return the next prediction revision, or ``None`` if the stream ended., clamp() (+8 more)

### Community 31 - "VisualizationConfig"
Cohesion: 0.16
Nodes (17): Matplotlib scene rendering options., VisualizationConfig, Path, Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), build_config(), main(), parse_args() (+9 more)

### Community 32 - "eval_convlstm_held_out.py"
Cohesion: 0.48
Nodes (6): local_overlap(), main(), normalize(), Tensor, save_comparison_plot(), aggregate_metrics()

### Community 33 - "SimulationGroupedSampler"
Cohesion: 0.33
Nodes (3): Sampler, Yields window indices grouped by simulation file. Simulations are shuffled each…, SimulationGroupedSampler

### Community 34 - "Online Replanning Architecture Demo"
Cohesion: 0.11
Nodes (17): Artifacts, Configuration, Future Integration Notes, Initial Mission, Mission change, Mission change, Mission change, Online Replanning Architecture Demo (+9 more)

### Community 35 - "common.py"
Cohesion: 0.10
Nodes (42): ensure_paths(), make_environment(), mean_std(), Any, Path, Shared helpers for optimization-performance experiments. Keeps environment…, Summarize a final Pareto set in human-readable objective units., Write a list of dict rows to CSV. (+34 more)

### Community 36 - "yolo_fire_adapter.py"
Cohesion: 0.14
Nodes (15): _area_pool_to_grid(), _box_to_mask(), build_convlstm_sequence(), _filter_fire_grid(), _max_pool_to_grid(), overlay_grid_on_image(), ndarray, Tensor (+7 more)

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.15
Nodes (12): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Precise implementation status, Research Comparison — Zhu et al. (2025) vs This Project, Update: architecture-correctness pass (P(fire) vs. damage, risk map,, Update: ConvLSTM is now connected (was "not yet implemented" as of the, Update: D* Lite is now implemented (was "not yet implemented" as of the (+4 more)

### Community 38 - "DStarLite"
Cohesion: 0.20
Nodes (8): DStarLite, Cell, Move the UAV's current position without resetting planner state. Per D* Lite,…, Incrementally repair the path after obstacle cells change. Only vertices…, Incrementally repair the path after spatially-varying risk changes. Same…, Return the currently-computed path without triggering a new search., Incremental grid path planner. Parameters ---------- width, height: Grid…, Compute a full path from ``start`` to ``goal``. Returns None if unreachable.

### Community 39 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.12
Nodes (16): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Experiment 5 — Generations to Convergence Threshold vs Problem Size, Experiment 6 — Scenario Diversity (+8 more)

### Community 41 - "DecodedMission"
Cohesion: 0.13
Nodes (14): FitnessVector, MissionObjectives, Any, ndarray, Score an already-decoded feasible (or near-feasible) mission., Evaluate a mission chromosome or ID sequence. Accepts a permutation…, Package a fitness evaluation into a ``MissionPlan``., Human-readable (maximization-friendly) objective values. (+6 more)

### Community 42 - "WildfireEnvironment"
Cohesion: 0.13
Nodes (15): Any, Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, Phase-1 research environment for mission planning. Attributes ---------- width,…, Return a fresh target id that has never been used before., Human-readable scene summary for logs / demos. (+7 more)

### Community 43 - "SuppressionTarget"
Cohesion: 0.16
Nodes (11): Permutation chromosome encoding / decoding for single-UAV missions. Chromosome…, Synthetic wildfire mission environment. Holds the 100×100 spatial domain, UAV…, Synthetic wildfire environment and suppression-target generation., Suppression target data model. Aligned with the multi-target firefighting…, A candidate wildfire cell the UAV may suppress., Alias for ``damage_score``. Named to match the research problem statement and…, Serialize for logging / experiment dumps., SuppressionTarget (+3 more)

### Community 49 - "eval_convlstm_real_firms.py"
Cohesion: 0.36
Nodes (9): build_input_sequence(), load_real_fire_pair(), main(), ndarray, Tensor, rasterize(), (WINDOW, 10, GRID, GRID): real fire history (replicated, see limitation above)…, save_comparison() (+1 more)

### Community 50 - "executor.py"
Cohesion: 0.15
Nodes (12): MissionExecutor, NullMissionExecutor, ABC, Execution hand-off interface for the selected mission. After online replanning…, Interface for executing / locally refining a selected mission., Execute or refine ``request`` (e.g. via D* Lite)., No-op executor used by the architecture demo., PredictionRiskMap (+4 more)

### Community 51 - "pipeline.py"
Cohesion: 0.13
Nodes (19): main(), Map a torch device choice to an Ultralytics YOLO device string., resolve_yolo_device(), predict_wildfire_from_image(), device, ndarray, Path, End-to-end wildfire prediction: drone RGB image -> YOLO grid -> ConvLSTM… (+11 more)

### Community 52 - "DStarLiteSession"
Cohesion: 0.12
Nodes (9): DStarLiteSession, Persistent D* Lite planning session for one active leg (fixed goal). Wraps a…, Begin planning toward ``goal``. Resets internal state (new problem)., Report the UAV's new position; incrementally repairs, does not reset., Mark cells as newly blocked; incrementally repairs, does not reset., Mark cells as newly passable; incrementally repairs, does not reset., Update spatially-varying traversal risk; incrementally repairs., The most recently computed path, without triggering a new search. (+1 more)

### Community 53 - "GridConfig"
Cohesion: 0.21
Nodes (14): GridConfig, Spatial domain for the wildfire mission environment., _apply_wind_bias(), generate_drone_start(), generate_synthetic_targets(), Synthetic suppression-target generator. Phase 1: random targets on the grid,…, Scatter targets around ``n_clusters`` random centers (multiple fire fronts)., Sample an integer grid cell not already occupied by another target. (+6 more)

### Community 54 - "OptimizerConfig"
Cohesion: 0.30
Nodes (8): OptimizerConfig, NSGA-II mission-optimization settings (pymoo). Inspired by the multi-objective…, decode_permutation(), ndarray, Convert a full permutation into a constraint-feasible ordered mission. Walks…, DroneState, UAV pose in the mission grid (Phase 1: 2-D position only)., TestPermutationDecodingIsASubset

### Community 55 - "risk_routing_ablation.py"
Cohesion: 0.31
Nodes (8): _build_risk_field(), _plot(), Any, Path, Validation Experiment 3 — risk-weighted D* Lite routing ablation. Fixed…, A localized high-risk blob directly on the shortest path, small enough that the…, RiskRoutingAblationResult, run_risk_routing_ablation()

### Community 56 - "online_replanning_comparison.py"
Cohesion: 0.26
Nodes (12): OnlineReplanningComparisonResult, _plot(), Any, Path, Validation Experiment 4 — online replanning policy comparison. Compares three…, run_online_replanning_comparison(), _run_policy(), _summarize() (+4 more)

### Community 57 - "plot_mission_path.py"
Cohesion: 0.23
Nodes (12): _draw_background_targets(), _draw_direction_arrows(), plot_mission_path(), Axes, Figure, Path, Ordered mission-path visualization for a selected Pareto plan. Draws the UAV…, Plot and save a mission path figure. (+4 more)

### Community 58 - "ExpectedDamageConfig"
Cohesion: 0.36
Nodes (6): _compute_risk_grid(), ExpectedDamageConfig, Configuration for the fire-probability -> expected-damage-proxy transform.…, Expected-damage-proxy grid: ``fire_probability * severity_norm * fuel_norm``.…, The raw ConvLSTM output must never be mutated by the proxy transform., TestExpectedDamageProxy

### Community 59 - "csv_export.py"
Cohesion: 0.38
Nodes (9): _best(), _dstar_fields(), _event_row(), _initial_row(), _order_str(), Path, CSV export of the online (in-mission) NSGA-II replanning transcript. One row…, Write one CSV row per NSGA-II run in ``result`` (initial + replans). (+1 more)

### Community 60 - "animate_replan.py"
Cohesion: 0.33
Nodes (9): _crop_to_common(), Path, Save online-replanning animations as PNG frames, GIF, and MP4., Convert RGBA/RGB frames to RGB ndarray., Render and export PNG frames + GIF + MP4 for an online-replan run.…, Encode MP4 using the imageio-ffmpeg binary (no system ffmpeg required)., save_replan_outputs(), _write_gif() (+1 more)

### Community 61 - "plot_online_replan_summary"
Cohesion: 0.33
Nodes (8): _plot_composition(), _plot_effectiveness(), plot_online_replan_summary(), _plot_runtime(), _plot_score(), Path, Research-style summary plot for the online (in-mission) NSGA-II replanning…, Render a 2x2 research figure and save it to ``path``.

### Community 62 - "OnlineReplanResult"
Cohesion: 0.29
Nodes (6): OnlineReplanResult, Full online-replanning demo transcript., Path, Markdown report for the online replanning architecture demo., Generate a Markdown report describing each replan event., write_replan_report()

### Community 63 - "population_objective_stats"
Cohesion: 0.33
Nodes (5): compute_hypervolume(), population_objective_stats(), ndarray, Hypervolume of the non-dominated set in ``F`` (minimization)., Best / average objective stats in human-readable (maximize-damage) form. ``F``…

## Knowledge Gaps
- **81 isolated node(s):** `setup_mac.sh script`, `upload_to_kaggle.sh script`, `Copy assets from Windows`, `Kaggle API (Mac)`, `Run the pipeline` (+76 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `MissionExecutionResult`, `optimize_demo.py`, `MissionConfig`, `NSGA2MissionOptimizer`, `ExperimentConfig`, `prediction_mission_gap.py`, `ScoredMission`, `prediction_source.py`, `DStarLiteMissionExecutor`, `scoring.py`, `online_replanner.py`, `OptimizationResult`, `.target_xy`, `dynamics.py`, `VisualizationConfig`, `common.py`, `DecodedMission`, `SuppressionTarget`, `executor.py`, `OptimizerConfig`, `online_replanning_comparison.py`, `plot_mission_path.py`, `ExpectedDamageConfig`, `OnlineReplanResult`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `WildfireConvLSTM` connect `WildfireConvLSTM` to `eval_convlstm_held_out.py`, `train.py`, `eval_convlstm_real_firms.py`, `ConvLSTMCell`, `prediction_source.py`, `WildfireDataset`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `MissionExecutionResult` connect `MissionExecutionResult` to `WildfireEnvironment`, `ScoredMission`, `executor.py`, `DStarLiteMissionExecutor`, `DStarLiteSession`, `OptimizerConfig`, `online_replanner.py`, `DStarLitePlanner`, `OnlineReplanResult`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `OnlineConvergenceResult`) actually correct?**
  _`OptimizerConfig` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `MissionConfig` (e.g. with `TimedOptimization` and `OnlineReplanningComparisonResult`) actually correct?**
  _`MissionConfig` has 19 INFERRED edges - model-reasoned connections that need verification._