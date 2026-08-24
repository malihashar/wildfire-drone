# Graph Report - wildfire-drone  (2026-08-21)

## Corpus Check
- 135 files · ~272,868 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1506 nodes · 3970 edges · 66 communities (61 shown, 5 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 410 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `48fd3825`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- GeoOrigin
- VisualizationConfig
- style_axes
- run_online_replanning_comparison
- ExperimentPaths
- WildfireEnvironment
- resolve_device
- OnlineReplanResult
- detections_to_fire_grid
- train.py
- ExperimentConfig
- test_error_injection.py
- MissionExecutionResult
- OptimizerConfig
- test_architecture_fixes.py
- YOLO Fire Detection Integration
- test_convlstm_nsga2_integration.py
- ConvLSTMCell
- save_figure
- resolve_convlstm_checkpoint
- online_replanner.py
- test_safety_gates.py
- SuppressionTarget
- prediction_mission_gap.py
- Wildfire UAV Mission Optimization — Experiment Report
- nsga_mavsdk_sitl.py
- vision/__init__.py
- .next_update
- ConvLSTM Wildfire Data Architecture — Critical Review
- nsga_benchmark.py
- common.py
- telemetry_diagnostic.py
- SimulationGroupedSampler
- Online Replanning Architecture Demo
- yolo_fire_adapter.py
- test_scenario_generation.py
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLite
- Wildfire UAV Mission Optimization — Experiment Report
- preprocess_dataset.py
- .target_xy
- prediction_source.py
- surrogate_accuracy.py
- setup_mac.sh
- upload_helper.py
- upload_to_kaggle.sh
- _offset_latlon
- DStarLiteMissionExecutor
- OptimizationResult
- NSGA2MissionOptimizer
- arm_drone
- risk_routing_ablation.py
- nsga_pixhawk_mission.py
- WildfireConvLSTM
- test_preflight_confirm.py
- mavsdk_controller.py
- test_mavsdk_controller.py
- DroneState
- build_convlstm_sequence
- population_objective_stats
- target_count_scenarios

## God Nodes (most connected - your core abstractions)
1. `WildfireEnvironment` - 113 edges
2. `OptimizerConfig` - 75 edges
3. `NSGA2MissionOptimizer` - 65 edges
4. `ExperimentConfig` - 59 edges
5. `MissionConfig` - 52 edges
6. `OptimizationResult` - 52 edges
7. `TargetGenerationConfig` - 45 edges
8. `DroneState` - 45 edges
9. `GridConfig` - 44 edges
10. `DStarLite` - 42 edges

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

## Communities (66 total, 5 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (54): FuncAnimation, main(), Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction., SimulationConfig, TerrainConfig (+46 more)

### Community 1 - "GeoOrigin"
Cohesion: 0.08
Nodes (11): GpsInfo, Health, GeoOrigin, Cell, Local-tangent-plane mapping from mission-grid cells to GPS. The mission grid…, Flat-earth (equirectangular) approximation -- adequate at mission-grid scale., StatusText, main() (+3 more)

### Community 2 - "VisualizationConfig"
Cohesion: 0.11
Nodes (30): Matplotlib scene rendering options., VisualizationConfig, Path, Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), Path, Phase-1 experiment: generate a synthetic scene and save a visualization. (+22 more)

### Community 3 - "style_axes"
Cohesion: 0.10
Nodes (34): hypervolume_reference_point(), Axes, Apply a consistent research-plot style., Fixed reference point for fair HV comparison across generations. Objectives are…, style_axes(), ConvergenceCallback, ConvergenceHistory, _plot_avg_fitness() (+26 more)

### Community 4 - "run_online_replanning_comparison"
Cohesion: 0.50
Nodes (5): _plot(), Any, Path, run_online_replanning_comparison(), _summarize()

### Community 5 - "ExperimentPaths"
Cohesion: 0.11
Nodes (22): ExperimentPaths, Path, Output layout for CSVs, plots, and the Markdown report., find_threshold_hits(), FitnessRuntimeResult, main(), _plot_fitness_vs_runtime(), print_threshold_report() (+14 more)

### Community 6 - "WildfireEnvironment"
Cohesion: 0.12
Nodes (13): ElementwiseProblem, MissionScorer, Evaluate permutation chromosomes against the three research objectives.…, Mission optimizer package — public entry point for suppression-mission…, Single-UAV NSGA-II mission optimizer using official pymoo operators. Adapts the…, ndarray, pymoo problem definition for single-UAV multi-target suppression planning.…, Multi-objective permutation problem for wildfire suppression missions.… (+5 more)

### Community 7 - "resolve_device"
Cohesion: 0.18
Nodes (14): _ignition_points_from_targets(), Seed the driving simulator's ignition cells from the mission's current targets., load_convlstm_checkpoint(), predict_next_fire_from_grid(), Any, device, Path, Run ConvLSTM prediction from a YOLO fire grid and terrain/weather channels.… (+6 more)

### Community 8 - "OnlineReplanResult"
Cohesion: 0.07
Nodes (48): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, CSV, plot, and a report.…, run_online_replan_demo(), build_config(), main(), parse_args(), Namespace (+40 more)

### Community 9 - "detections_to_fire_grid"
Cohesion: 0.17
Nodes (16): FireDetection, cell_to_image_rect(), _class_weight(), detections_to_fire_grid(), _filter_fire_grid(), FireBoxDetection, FireGridConfig, FireMaskDetection (+8 more)

### Community 10 - "train.py"
Cohesion: 0.09
Nodes (34): DataLoader, Module, Optimizer, local_overlap(), main(), normalize(), Tensor, save_comparison_plot() (+26 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.34
Nodes (21): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+13 more)

### Community 12 - "test_error_injection.py"
Cohesion: 0.07
Nodes (26): AssertionError, _AlwaysConnectedCore, _ConnectedGoodHealthNoHomeTelemetry, _ConnectedNoGpsTelemetry, _FakeSystem, _ForbiddenAircraftControl, _NeverConnectedCore, _NoArmAction (+18 more)

### Community 13 - "MissionExecutionResult"
Cohesion: 0.16
Nodes (21): MissionExecutionResult, Outcome of executing/locally-planning a ``MissionExecutionRequest``.…, ``path_length / straight_line_length``; 1.0 = surrogate matched actual exactly., EnvironmentDiff, Summarizes how the target set changed between two prediction ticks., build_animation_frames(), _draw_drone(), _draw_dstar_path() (+13 more)

### Community 14 - "OptimizerConfig"
Cohesion: 0.08
Nodes (38): Configuration for the mission-planning prototype., GridConfig, MissionConfig, OptimizerConfig, Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, Top-level configuration for the mission-planning research prototype., Spatial domain for the wildfire mission environment., Parameters for synthetic suppression-target generation. Later: ConvLSTM… (+30 more)

### Community 15 - "test_architecture_fixes.py"
Cohesion: 0.13
Nodes (20): DStarLiteSession, Persistent D* Lite planning session for one active leg (fixed goal). Wraps a…, _compute_risk_grid(), ConvLSTMSourceConfig, ExpectedDamageConfig, PredictionRiskMap, Sample ``risk`` at a (possibly fractional) mission-grid coordinate., Convert to a ``{(x, y): risk}`` dict keyed the same way D* Lite's ``Cell`` type… (+12 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "test_convlstm_nsga2_integration.py"
Cohesion: 0.11
Nodes (23): apply_prediction_update(), Synthetic wildfire-prediction dynamics. Randomly edits priorities, positions,…, Apply a prediction update in-place and return the resulting diff. This is the…, _unique(), ConvLSTMPredictionSource, ConvLSTM-backed prediction source. Each tick, this class: 1. Advances a real…, clamp(), euclidean_distance() (+15 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.15
Nodes (9): ConvLSTM, ConvLSTMCell, Tensor, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no… (+1 more)

### Community 19 - "save_figure"
Cohesion: 0.14
Nodes (20): Figure, Save a matplotlib figure as PNG., save_figure(), _plot_generations(), Any, Path, _plot(), Any (+12 more)

### Community 20 - "resolve_convlstm_checkpoint"
Cohesion: 0.27
Nodes (7): Path, Default model and data paths for the vision + ConvLSTM product flow., Resolve an optional user path, falling back to a project default., Resolve a ConvLSTM checkpoint file or checkpoint directory., resolve_convlstm_checkpoint(), resolve_path(), TestVisionPaths

### Community 21 - "online_replanner.py"
Cohesion: 0.12
Nodes (26): A Pareto plan paired with its scalar selection score., ScoredMission, build_execution_request(), MissionExecutionRequest, MissionExecutor, NullMissionExecutor, ABC, Execution hand-off interface for the selected mission. After online replanning… (+18 more)

### Community 22 - "test_safety_gates.py"
Cohesion: 0.12
Nodes (15): _parse(), Namespace, parametrize, TEST 7 -- safety-gate audit for tests/nsga_pixhawk_mission.py (the real-…, Any exception during the flight sequence must fall into the recovery block…, Case-sensitivity matters here: an accidental lowercase 'fly' from a script or a…, Passing --i-have-verified-the-site does NOT also satisfy the typed…, --yes is documented as skipping the typed prompt for scripted use, but it must… (+7 more)

### Community 23 - "SuppressionTarget"
Cohesion: 0.05
Nodes (45): constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions., True if the mission tour exceeds the configured distance budget., Inequality constraint g <= 0 for pymoo. g = n_targets - max_mission_targets, Inequality constraint g <= 0 for pymoo. g = travel_distance -…, True if the mission visits more targets than allowed., violates_max_distance() (+37 more)

### Community 24 - "prediction_mission_gap.py"
Cohesion: 0.14
Nodes (20): _cell(), _jaccard(), _plot(), Any, Path, Validation Experiment 5 — prediction quality -> mission quality gap. Not "is…, run_prediction_mission_gap_experiment(), MissionSelectionConfig (+12 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - "nsga_mavsdk_sitl.py"
Cohesion: 0.08
Nodes (39): _connect_and_wait(), _connect_loop(), _fetch_home(), _health_loop(), _home_loop(), main(), _offset_latlon(), _plan_mission() (+31 more)

### Community 27 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 28 - ".next_update"
Cohesion: 0.22
Nodes (10): _find_hotspots(), _normalize_grid(), _pack_frames(), _predicted_map_to_patches(), Any, ndarray, Pack a slice of ``WildfireSimulator.history`` into ``(T, 10, H, W)``. Channel…, Min-max normalize ``grid`` using real training-time stats for ``key``. (+2 more)

### Community 29 - "ConvLSTM Wildfire Data Architecture — Critical Review"
Cohesion: 0.06
Nodes (30): 10. Weather representation, 11. Fuel / vegetation representation, 12. Final tensor / channel specification, 13. Data preprocessing pipeline, 14. Training/validation split strategy, 15. Real vs. simulation strategy, 16. ConvLSTM vs. alternatives, 17. Physics-informed feature integration (+22 more)

### Community 30 - "nsga_benchmark.py"
Cohesion: 0.18
Nodes (13): ConvergenceRecorder, main(), print_final_table(), Callback, Path, Fair NSGA-II generation-count benchmark for the wildfire-suppression optimizer.…, Determine, FROM THE MEASURED DATA, roughly how many generations fit in…, Records the Pareto front's best per-objective values at each requested… (+5 more)

### Community 31 - "common.py"
Cohesion: 0.11
Nodes (40): ensure_paths(), make_environment(), mean_std(), Any, Path, Shared helpers for optimization-performance experiments. Keeps environment…, Summarize a final Pareto set in human-readable objective units., Write a list of dict rows to CSV. (+32 more)

### Community 32 - "telemetry_diagnostic.py"
Cohesion: 0.15
Nodes (14): DiagnosticReport, main(), System, TEST 4 -- telemetry-only diagnostic. Connects to a vehicle (real Pixhawk over…, Return the first ConnectionState with is_connected=True. The caller wraps this…, Read exactly one item from an async generator, with a timeout. None on…, _read_one(), run_diagnostic() (+6 more)

### Community 33 - "SimulationGroupedSampler"
Cohesion: 0.33
Nodes (3): Sampler, Yields window indices grouped by simulation file. Simulations are shuffled each…, SimulationGroupedSampler

### Community 34 - "Online Replanning Architecture Demo"
Cohesion: 0.11
Nodes (17): Artifacts, Configuration, Future Integration Notes, Initial Mission, Mission change, Mission change, Mission change, Online Replanning Architecture Demo (+9 more)

### Community 35 - "yolo_fire_adapter.py"
Cohesion: 0.09
Nodes (29): main(), predict_wildfire_from_image(), device, ndarray, Path, End-to-end wildfire prediction: drone RGB image -> YOLO grid -> ConvLSTM…, Outputs from the YOLO + ConvLSTM product flow., Run the full product flow on one image. 1. YOLO segmentation produces the… (+21 more)

### Community 36 - "test_scenario_generation.py"
Cohesion: 0.13
Nodes (16): parametrize, TEST 1 -- scenario generation properties (tests/nsga_scenario.py). Verifies the…, SuppressionTarget.travel_cost is the distance-from-home the objectives can…, The fair-comparison helper must change ONLY n_generations, not the scenario…, RADIUS_M is a plain module-level float the operator can edit, not hardcoded…, A radius too small to fit MIN_TARGETS points MIN_SPACING_M apart must raise,…, damage_score/priority (the objective functions' inputs) come from the project's…, test_every_target_within_radius() (+8 more)

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.15
Nodes (12): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Precise implementation status, Research Comparison — Zhu et al. (2025) vs This Project, Update: architecture-correctness pass (P(fire) vs. damage, risk map,, Update: ConvLSTM is now connected (was "not yet implemented" as of the, Update: D* Lite is now implemented (was "not yet implemented" as of the (+4 more)

### Community 38 - "DStarLite"
Cohesion: 0.06
Nodes (25): DStarLite, _Key, path_length(), Cell, D* Lite grid path planner (Koenig & Likhachev, 2002). Decides HOW the UAV…, Move the UAV's current position without resetting planner state. Per D* Lite,…, Incrementally repair the path after obstacle cells change. Only vertices…, Incrementally repair the path after spatially-varying risk changes. Same… (+17 more)

### Community 39 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.12
Nodes (16): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Experiment 5 — Generations to Convergence Threshold vs Problem Size, Experiment 6 — Scenario Diversity (+8 more)

### Community 42 - "prediction_source.py"
Cohesion: 0.12
Nodes (16): Any, Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, PredictionSource, ABC, Enum (+8 more)

### Community 43 - "surrogate_accuracy.py"
Cohesion: 0.38
Nodes (6): _plot(), Any, Path, Validation Experiment 2 — surrogate (NSGA-II Euclidean) vs. D* Lite actual…, run_surrogate_accuracy_experiment(), SurrogateAccuracyResult

### Community 49 - "_offset_latlon"
Cohesion: 0.13
Nodes (24): main(), print_table(), TEST 8 -- per-stage performance timing for the NSGA-II + MAVSDK software…, Local ``mavsdk.System()`` construction ONLY -- no ``connect()`` is called, so…, StageTimings, time_coordinate_conversion(), time_mission_item_construction(), time_nsga2_optimization() (+16 more)

### Community 50 - "DStarLiteMissionExecutor"
Cohesion: 0.26
Nodes (5): DStarLiteMissionExecutor, _euclid(), Cell, Plans the real obstacle-aware route for a selected mission using D* Lite.…, TestExecutorGenuineIncrementalReuse

### Community 51 - "OptimizationResult"
Cohesion: 0.09
Nodes (22): Pick a balanced Pareto mission near the damage–distance knee. Uses normalized…, select_representative_plan(), MissionPlan, One Pareto-optimal (or candidate) suppression mission., Ordered list of suppression target IDs to visit., OptimizationResult, Pareto set returned by ``NSGA2MissionOptimizer.optimize``., Plan with the highest damage prevented (among Pareto set). (+14 more)

### Community 53 - "NSGA2MissionOptimizer"
Cohesion: 0.17
Nodes (14): _nondominated_mask(), NSGA2MissionOptimizer, ndarray, Execute NSGA-II and return the non-dominated mission plans. Parameters…, Boolean mask of non-dominated rows for a minimization matrix ``F``., Run pymoo NSGA-II over suppression-target permutations. Operators (official…, parametrize, TEST 2 -- NSGA-II optimization correctness and convergence… (+6 more)

### Community 55 - "arm_drone"
Cohesion: 0.20
Nodes (19): arm_drone(), ArmResult, format_gps(), format_health(), health_checks(), health_failure_report(), HealthWatch, Latest values from background telemetry subscriptions, for diagnostics only. (+11 more)

### Community 59 - "risk_routing_ablation.py"
Cohesion: 0.31
Nodes (8): _build_risk_field(), _plot(), Any, Path, Validation Experiment 3 — risk-weighted D* Lite routing ablation. Fixed…, A localized high-risk blob directly on the shortest path, small enough that the…, RiskRoutingAblationResult, run_risk_routing_ablation()

### Community 60 - "nsga_pixhawk_mission.py"
Cohesion: 0.25
Nodes (17): _build_serial_address(), _confirm_airborne(), _connect_and_wait(), _connect_loop(), _fetch_home(), _home_loop(), main(), Namespace (+9 more)

### Community 61 - "WildfireConvLSTM"
Cohesion: 0.11
Nodes (17): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, WildfireConvLSTM, Dataset, Path, Tensor, WildfireDataset --------------- Sliding-window dataset for ConvLSTM wildfire…, Parameters ---------- split_json : str | Path Path to train.json / val.json /… (+9 more)

### Community 63 - "test_preflight_confirm.py"
Cohesion: 0.10
Nodes (40): print_mission_plan(), print_qgc_clear_reminder(), MissionItem, System, Shared pre-arm safety checks for scripts that fly a real Pixhawk. Every real-…, Reminds the operator to clear/replace any old QGroundControl mission -- never…, The typed "FLY" gate, required immediately before arming. Must be called right…, Prints the exact waypoints about to be flown, in flight order. (+32 more)

### Community 66 - "mavsdk_controller.py"
Cohesion: 0.27
Nodes (16): _abort(), _connect_and_verify(), _connected(), _fly_to_cell(), _haversine_m(), mission(), MissionResult, System (+8 more)

### Community 69 - "test_mavsdk_controller.py"
Cohesion: 0.30
Nodes (13): MissionFailureReason, Enum, _FakeSystem, _make_leg(), asyncio, Tests for the MAVSDK flight-control layer (mission.flight.mavsdk_controller).…, test_arm_drone_reports_rejected_arm(), test_arm_drone_reports_specific_reason_on_health_timeout() (+5 more)

### Community 71 - "DroneState"
Cohesion: 0.35
Nodes (6): decode_permutation(), ndarray, Convert a full permutation into a constraint-feasible ordered mission. Walks…, DroneState, UAV pose in the mission grid (Phase 1: 2-D position only)., TestPermutationDecodingIsASubset

### Community 72 - "build_convlstm_sequence"
Cohesion: 0.50
Nodes (3): build_convlstm_sequence(), Tensor, Build a `(T, 10, 100, 100)` tensor for the existing ConvLSTM. `terrain_weather`…

### Community 76 - "population_objective_stats"
Cohesion: 0.33
Nodes (5): compute_hypervolume(), population_objective_stats(), ndarray, Hypervolume of the non-dominated set in ``F`` (minimization)., Best / average objective stats in human-readable (maximize-damage) form. ``F``…

### Community 77 - "target_count_scenarios"
Cohesion: 0.40
Nodes (5): range, fixture, At least one real generated scenario for each of 7, 8, 9, 10 targets., _scenarios_covering_target_counts(), target_count_scenarios()

## Knowledge Gaps
- **81 isolated node(s):** `setup_mac.sh script`, `upload_to_kaggle.sh script`, `Copy assets from Windows`, `Kaggle API (Mac)`, `Run the pipeline` (+76 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `VisualizationConfig`, `resolve_device`, `OnlineReplanResult`, `ExperimentConfig`, `MissionExecutionResult`, `OptimizerConfig`, `test_architecture_fixes.py`, `test_convlstm_nsga2_integration.py`, `online_replanner.py`, `SuppressionTarget`, `prediction_mission_gap.py`, `.next_update`, `common.py`, `.target_xy`, `prediction_source.py`, `DStarLiteMissionExecutor`, `OptimizationResult`, `NSGA2MissionOptimizer`, `DroneState`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `NSGA2MissionOptimizer` connect `NSGA2MissionOptimizer` to `VisualizationConfig`, `WildfireEnvironment`, `DroneState`, `OnlineReplanResult`, `surrogate_accuracy.py`, `OptimizerConfig`, `test_architecture_fixes.py`, `_offset_latlon`, `DStarLiteMissionExecutor`, `OptimizationResult`, `test_convlstm_nsga2_integration.py`, `online_replanner.py`, `SuppressionTarget`, `prediction_mission_gap.py`, `nsga_mavsdk_sitl.py`, `nsga_pixhawk_mission.py`, `nsga_benchmark.py`, `common.py`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `WildfireConvLSTM` connect `WildfireConvLSTM` to `test_convlstm_nsga2_integration.py`, `train.py`, `ConvLSTMCell`, `resolve_device`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `OnlineConvergenceResult`) actually correct?**
  _`OptimizerConfig` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `NSGA2MissionOptimizer` (e.g. with `TimedOptimization` and `OnlineReplanningComparisonResult`) actually correct?**
  _`NSGA2MissionOptimizer` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 18 INFERRED edges - model-reasoned connections that need verification._