# Graph Report - wildfire-drone  (2026-08-20)

## Corpus Check
- 123 files · ~256,248 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1276 nodes · 3503 edges · 59 communities (54 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 397 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `48fd3825`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- MissionExecutionResult
- OptimizationResult
- online_convergence.py
- online_replanner.py
- fitness_runtime.py
- nsga2.py
- convlstm_bridge.py
- OnlineReplanConfig
- yolo_fire_adapter.py
- train.py
- ExperimentConfig
- WildfireConvLSTM
- ScoredMission
- environment.py
- SuppressionTarget
- YOLO Fire Detection Integration
- NSGA2MissionOptimizer
- ConvLSTMCell
- OptimizerConfig
- predict_wildfire_from_image
- write_csv
- WildfireDataset
- scoring.py
- MissionPlan
- Wildfire UAV Mission Optimization — Experiment Report
- nsga_mavsdk_real.py
- vision/__init__.py
- chromosome.py
- ConvLSTM Wildfire Data Architecture — Critical Review
- nsga_benchmark.py
- save_figure
- eval_convlstm_held_out.py
- dataset.py
- Online Replanning Architecture Demo
- common.py
- build_convlstm_sequence
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLite
- Wildfire UAV Mission Optimization — Experiment Report
- preprocess_dataset.py
- resolve_convlstm_checkpoint
- WildfireEnvironment
- ExperimentPaths
- setup_mac.sh
- upload_helper.py
- upload_to_kaggle.sh
- eval_convlstm_real_firms.py
- DStarLiteMissionExecutor
- mission_path.py
- plot_online_replan_summary
- OnlineConvergenceResult
- run_epoch
- .target_xy
- csv_export.py
- animate_replan.py

## God Nodes (most connected - your core abstractions)
1. `WildfireEnvironment` - 113 edges
2. `OptimizerConfig` - 75 edges
3. `ExperimentConfig` - 59 edges
4. `NSGA2MissionOptimizer` - 58 edges
5. `MissionConfig` - 52 edges
6. `DroneState` - 45 edges
7. `GridConfig` - 44 edges
8. `TargetGenerationConfig` - 43 edges
9. `DStarLite` - 42 edges
10. `OptimizationResult` - 42 edges

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

## Communities (59 total, 5 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (54): FuncAnimation, main(), Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction., SimulationConfig, TerrainConfig (+46 more)

### Community 1 - "MissionExecutionResult"
Cohesion: 0.06
Nodes (47): asyncio, _abort(), arm_drone(), ArmResult, _connect_and_verify(), _connected(), _fly_to_cell(), GeoOrigin (+39 more)

### Community 2 - "OptimizationResult"
Cohesion: 0.05
Nodes (57): Matplotlib scene rendering options., VisualizationConfig, Path, Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), Path, Phase-1 experiment: generate a synthetic scene and save a visualization. (+49 more)

### Community 3 - "online_convergence.py"
Cohesion: 0.15
Nodes (19): ConvergenceHistory, Any, Per-generation optimization metrics., _aggregate(), _plot_avg_fitness_online(), Path, NSGA-II convergence, collected from the *online* in-mission replanning loop.…, Pointwise mean across every online NSGA-II run's per-generation history. (+11 more)

### Community 4 - "online_replanner.py"
Cohesion: 0.09
Nodes (33): _plot(), Any, Path, Validation Experiment 4 — online replanning policy comparison. Compares three…, run_online_replanning_comparison(), _run_policy(), _summarize(), Scalar mission scoring for selecting one plan from a Pareto set. Multi-… (+25 more)

### Community 5 - "fitness_runtime.py"
Cohesion: 0.14
Nodes (19): find_threshold_hits(), FitnessRuntimeResult, main(), _plot_fitness_vs_runtime(), print_threshold_report(), Any, Callback, Path (+11 more)

### Community 6 - "nsga2.py"
Cohesion: 0.14
Nodes (11): ElementwiseProblem, Configuration for the mission-planning prototype., Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, MissionScorer, Evaluate permutation chromosomes against the three research objectives.…, Mission optimizer package — public entry point for suppression-mission…, Single-UAV NSGA-II mission optimizer using official pymoo operators. Adapts the…, ndarray (+3 more)

### Community 7 - "convlstm_bridge.py"
Cohesion: 0.15
Nodes (17): load_convlstm_checkpoint(), load_terrain_weather_from_simulation(), predict_next_fire_from_grid(), Any, device, Path, Tensor, Bridge YOLO fire grids into the trained ConvLSTM predictor. The YOLO adapter… (+9 more)

### Community 8 - "OnlineReplanConfig"
Cohesion: 0.15
Nodes (17): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, CSV, plot, and a report.…, run_online_replan_demo(), build_config(), main(), parse_args(), Namespace (+9 more)

### Community 9 - "yolo_fire_adapter.py"
Cohesion: 0.18
Nodes (16): cell_to_image_rect(), _class_weight(), detections_to_fire_grid(), _filter_fire_grid(), FireBoxDetection, FireGridConfig, FireMaskDetection, _passes_threshold() (+8 more)

### Community 10 - "train.py"
Cohesion: 0.19
Nodes (12): Module, bce_loss(), FocalLoss, load_model_state(), main(), parse_args(), Namespace, Tensor (+4 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.34
Nodes (21): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+13 more)

### Community 12 - "WildfireConvLSTM"
Cohesion: 0.27
Nodes (7): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, WildfireConvLSTM, check(), main(), Smoke-test for WildfireDataset + WildfireConvLSTM. Run from project root:…

### Community 13 - "ScoredMission"
Cohesion: 0.16
Nodes (20): A Pareto plan paired with its scalar selection score., ScoredMission, EnvironmentDiff, Summarizes how the target set changed between two prediction ticks., build_animation_frames(), _draw_drone(), _draw_dstar_path(), _draw_path() (+12 more)

### Community 14 - "environment.py"
Cohesion: 0.12
Nodes (20): Synthetic wildfire mission environment. Holds the 100×100 spatial domain, UAV…, _apply_wind_bias(), generate_drone_start(), generate_synthetic_targets(), Synthetic suppression-target generator. Phase 1: random targets on the grid,…, Scatter targets around ``n_clusters`` random centers (multiple fire fronts)., Sample an integer grid cell not already occupied by another target., Sample a random continuous start pose inside the grid. (+12 more)

### Community 15 - "SuppressionTarget"
Cohesion: 0.13
Nodes (13): Autonomous wildfire-suppression UAV — mission planning research package. Phase…, Synthetic wildfire environment and suppression-target generation., Suppression target data model. Aligned with the multi-target firefighting…, A candidate wildfire cell the UAV may suppress., Alias for ``damage_score``. Named to match the research problem statement and…, Serialize for logging / experiment dumps., SuppressionTarget, generate_scenario() (+5 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "NSGA2MissionOptimizer"
Cohesion: 0.10
Nodes (33): GridConfig, MissionConfig, Top-level configuration for the mission-planning research prototype., Spatial domain for the wildfire mission environment., Parameters for synthetic suppression-target generation. Later: ConvLSTM…, TargetGenerationConfig, Optimization result plus wall-clock runtime in seconds., TimedOptimization (+25 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.17
Nodes (8): ConvLSTM, ConvLSTMCell, Tensor, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no…, Time a single ConvLSTMCell for T steps.

### Community 19 - "OptimizerConfig"
Cohesion: 0.30
Nodes (8): OptimizerConfig, NSGA-II mission-optimization settings (pymoo). Inspired by the multi-objective…, decode_permutation(), ndarray, Convert a full permutation into a constraint-feasible ordered mission. Walks…, DroneState, UAV pose in the mission grid (Phase 1: 2-D position only)., TestPermutationDecodingIsASubset

### Community 20 - "predict_wildfire_from_image"
Cohesion: 0.12
Nodes (19): FireDetection, main(), predict_wildfire_from_image(), device, ndarray, Path, Outputs from the YOLO + ConvLSTM product flow., Run the full product flow on one image. 1. YOLO segmentation produces the… (+11 more)

### Community 21 - "write_csv"
Cohesion: 0.14
Nodes (19): ensure_paths(), Any, Path, Write a list of dict rows to CSV., write_csv(), _build_risk_field(), _plot(), Any (+11 more)

### Community 22 - "WildfireDataset"
Cohesion: 0.31
Nodes (5): Dataset, Path, Tensor, Parameters ---------- split_json : str | Path Path to train.json / val.json /…, WildfireDataset

### Community 23 - "scoring.py"
Cohesion: 0.09
Nodes (25): constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions., True if the mission tour exceeds the configured distance budget., Inequality constraint g <= 0 for pymoo. g = n_targets - max_mission_targets, Inequality constraint g <= 0 for pymoo. g = travel_distance -…, True if the mission visits more targets than allowed., violates_max_distance() (+17 more)

### Community 24 - "MissionPlan"
Cohesion: 0.17
Nodes (9): MissionObjectives, MissionPlan, Package a fitness evaluation into a ``MissionPlan``., Human-readable (maximization-friendly) objective values., One Pareto-optimal (or candidate) suppression mission., Ordered list of suppression target IDs to visit., _nondominated_mask(), ndarray (+1 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - "nsga_mavsdk_real.py"
Cohesion: 0.11
Nodes (39): _connect_and_wait(), _connect_loop(), _fetch_home(), _health_loop(), _home_loop(), main(), _offset_latlon(), _plan_mission() (+31 more)

### Community 27 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 28 - "chromosome.py"
Cohesion: 0.33
Nodes (5): DecodedMission, Permutation chromosome encoding / decoding for single-UAV missions. Chromosome…, Feasible mission extracted from a permutation chromosome., Return the mission as an ordered list of target IDs., target_id_sequence()

### Community 29 - "ConvLSTM Wildfire Data Architecture — Critical Review"
Cohesion: 0.06
Nodes (30): 10. Weather representation, 11. Fuel / vegetation representation, 12. Final tensor / channel specification, 13. Data preprocessing pipeline, 14. Training/validation split strategy, 15. Real vs. simulation strategy, 16. ConvLSTM vs. alternatives, 17. Physics-informed feature integration (+22 more)

### Community 30 - "nsga_benchmark.py"
Cohesion: 0.18
Nodes (13): ConvergenceRecorder, main(), print_final_table(), Callback, Path, Fair NSGA-II generation-count benchmark for the wildfire-suppression optimizer.…, Determine, FROM THE MEASURED DATA, roughly how many generations fit in…, Records the Pareto front's best per-objective values at each requested… (+5 more)

### Community 31 - "save_figure"
Cohesion: 0.15
Nodes (26): Axes, Figure, Save a matplotlib figure as PNG., Apply a consistent research-plot style., save_figure(), style_axes(), _plot_avg_fitness(), _plot_best_fitness() (+18 more)

### Community 32 - "eval_convlstm_held_out.py"
Cohesion: 0.48
Nodes (6): local_overlap(), main(), normalize(), Tensor, save_comparison_plot(), aggregate_metrics()

### Community 33 - "dataset.py"
Cohesion: 0.18
Nodes (7): DataLoader, Sampler, build_loaders(), WildfireDataset --------------- Sliding-window dataset for ConvLSTM wildfire…, Returns (train_loader, val_loader, test_loader). Training loader uses…, Yields window indices grouped by simulation file. Simulations are shuffled each…, SimulationGroupedSampler

### Community 34 - "Online Replanning Architecture Demo"
Cohesion: 0.11
Nodes (17): Artifacts, Configuration, Future Integration Notes, Initial Mission, Mission change, Mission change, Mission change, Online Replanning Architecture Demo (+9 more)

### Community 35 - "common.py"
Cohesion: 0.10
Nodes (28): build_mission_config(), compute_hypervolume(), hypervolume_reference_point(), population_objective_stats(), ndarray, Shared helpers for optimization-performance experiments. Keeps environment…, Hypervolume of the non-dominated set in ``F`` (minimization)., Best / average objective stats in human-readable (maximize-damage) form. ``F``… (+20 more)

### Community 36 - "build_convlstm_sequence"
Cohesion: 0.14
Nodes (12): _area_pool_to_grid(), _box_to_mask(), build_convlstm_sequence(), _max_pool_to_grid(), overlay_grid_on_image(), ndarray, Tensor, Rasterize an `xyxy` detection box into a binary image-space mask. (+4 more)

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.15
Nodes (12): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Precise implementation status, Research Comparison — Zhu et al. (2025) vs This Project, Update: architecture-correctness pass (P(fire) vs. damage, risk map,, Update: ConvLSTM is now connected (was "not yet implemented" as of the, Update: D* Lite is now implemented (was "not yet implemented" as of the (+4 more)

### Community 38 - "DStarLite"
Cohesion: 0.07
Nodes (25): DStarLite, _Key, path_length(), Cell, D* Lite grid path planner (Koenig & Likhachev, 2002). Decides HOW the UAV…, Move the UAV's current position without resetting planner state. Per D* Lite,…, Incrementally repair the path after obstacle cells change. Only vertices…, Incrementally repair the path after spatially-varying risk changes. Same… (+17 more)

### Community 39 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.12
Nodes (16): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Experiment 5 — Generations to Convergence Threshold vs Problem Size, Experiment 6 — Scenario Diversity (+8 more)

### Community 41 - "resolve_convlstm_checkpoint"
Cohesion: 0.24
Nodes (8): Path, Default model and data paths for the vision + ConvLSTM product flow., Resolve an optional user path, falling back to a project default., Resolve a ConvLSTM checkpoint file or checkpoint directory., resolve_convlstm_checkpoint(), resolve_path(), End-to-end wildfire prediction: drone RGB image -> YOLO grid -> ConvLSTM…, TestVisionPaths

### Community 42 - "WildfireEnvironment"
Cohesion: 0.08
Nodes (33): Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, Phase-1 research environment for mission planning. Attributes ---------- width,…, Return a fresh target id that has never been used before., Human-readable scene summary for logs / demos., WildfireEnvironment (+25 more)

### Community 43 - "ExperimentPaths"
Cohesion: 0.31
Nodes (3): ExperimentPaths, Path, Output layout for CSVs, plots, and the Markdown report.

### Community 49 - "eval_convlstm_real_firms.py"
Cohesion: 0.36
Nodes (9): build_input_sequence(), load_real_fire_pair(), main(), ndarray, Tensor, rasterize(), (WINDOW, 10, GRID, GRID): real fire history (replicated, see limitation above)…, save_comparison() (+1 more)

### Community 50 - "DStarLiteMissionExecutor"
Cohesion: 0.08
Nodes (34): DStarLiteSession, Persistent D* Lite planning session for one active leg (fixed goal). Wraps a…, DStarLiteMissionExecutor, _euclid(), MissionExecutionRequest, MissionExecutor, NullMissionExecutor, ABC (+26 more)

### Community 51 - "mission_path.py"
Cohesion: 0.12
Nodes (26): make_environment(), mean_std(), Create a reproducible synthetic environment for an experiment., _plot_generations(), Any, Path, Experiment 5 — Generations-to-convergence-threshold vs number of targets. For…, Measure generations-to-threshold as a function of problem size. (+18 more)

### Community 53 - "plot_online_replan_summary"
Cohesion: 0.33
Nodes (8): _plot_composition(), _plot_effectiveness(), plot_online_replan_summary(), _plot_runtime(), _plot_score(), Path, Research-style summary plot for the online (in-mission) NSGA-II replanning…, Render a 2x2 research figure and save it to ``path``.

### Community 54 - "OnlineConvergenceResult"
Cohesion: 0.25
Nodes (6): ConvergenceCallback, Callback, ndarray, Record population quality metrics after every generation., OnlineConvergenceResult, Per-generation histories from every online-replanning NSGA-II run.

### Community 55 - "run_epoch"
Cohesion: 0.50
Nodes (4): Optimizer, device, Run one epoch. Returns (avg_loss, avg_metrics)., run_epoch()

### Community 58 - "csv_export.py"
Cohesion: 0.38
Nodes (9): _best(), _dstar_fields(), _event_row(), _initial_row(), _order_str(), Path, CSV export of the online (in-mission) NSGA-II replanning transcript. One row…, Write one CSV row per NSGA-II run in ``result`` (initial + replans). (+1 more)

### Community 59 - "animate_replan.py"
Cohesion: 0.33
Nodes (9): _crop_to_common(), Path, Save online-replanning animations as PNG frames, GIF, and MP4., Convert RGBA/RGB frames to RGB ndarray., Render and export PNG frames + GIF + MP4 for an online-replan run.…, Encode MP4 using the imageio-ffmpeg binary (no system ffmpeg required)., save_replan_outputs(), _write_gif() (+1 more)

## Knowledge Gaps
- **81 isolated node(s):** `setup_mac.sh script`, `upload_to_kaggle.sh script`, `Copy assets from Windows`, `Kaggle API (Mac)`, `Run the pipeline` (+76 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `MissionExecutionResult`, `OptimizationResult`, `common.py`, `online_replanner.py`, `nsga2.py`, `convlstm_bridge.py`, `ExperimentConfig`, `ScoredMission`, `environment.py`, `SuppressionTarget`, `NSGA2MissionOptimizer`, `DStarLiteMissionExecutor`, `mission_path.py`, `OptimizerConfig`, `scoring.py`, `MissionPlan`, `.target_xy`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._
- **Why does `WildfireConvLSTM` connect `WildfireConvLSTM` to `eval_convlstm_held_out.py`, `convlstm_bridge.py`, `train.py`, `eval_convlstm_real_firms.py`, `ConvLSTMCell`, `NSGA2MissionOptimizer`, `run_epoch`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `load_convlstm_checkpoint()` connect `convlstm_bridge.py` to `NSGA2MissionOptimizer`, `WildfireEnvironment`, `vision/__init__.py`, `WildfireConvLSTM`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `OnlineConvergenceResult`) actually correct?**
  _`OptimizerConfig` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `NSGA2MissionOptimizer` (e.g. with `TimedOptimization` and `OnlineReplanningComparisonResult`) actually correct?**
  _`NSGA2MissionOptimizer` has 24 INFERRED edges - model-reasoned connections that need verification._