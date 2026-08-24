# Graph Report - wildfire-drone  (2026-08-21)

## Corpus Check
- 133 files · ~271,680 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1464 nodes · 3852 edges · 79 communities (74 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 414 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `48fd3825`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- WildfireSimulator
- GeoOrigin
- WildfireEnvironment
- online_convergence.py
- online_replanning_comparison.py
- ExperimentPaths
- OptimizerConfig
- convlstm_bridge.py
- OnlineReplanResult
- detections_to_fire_grid
- train.py
- ExperimentConfig
- test_error_injection.py
- MissionExecutionResult
- MissionConfig
- test_architecture_fixes.py
- YOLO Fire Detection Integration
- test_convlstm_nsga2_integration.py
- ConvLSTMCell
- style_axes
- resolve_convlstm_checkpoint
- online_replanner.py
- test_safety_gates.py
- SuppressionTarget
- prediction_mission_gap.py
- Wildfire UAV Mission Optimization — Experiment Report
- nsga_mavsdk_sitl.py
- vision/__init__.py
- test_mission_item_construction.py
- ConvLSTM Wildfire Data Architecture — Critical Review
- generate_scenario
- common.py
- telemetry_diagnostic.py
- dataset.py
- Online Replanning Architecture Demo
- yolo_fire_adapter.py
- test_scenario_generation.py
- Research Comparison — Zhu et al. (2025) vs This Project
- DStarLite
- Wildfire UAV Mission Optimization — Experiment Report
- preprocess_dataset.py
- run_all.py
- SyntheticPredictionSource
- save_figure
- setup_mac.sh
- upload_helper.py
- upload_to_kaggle.sh
- _offset_latlon
- DStarLiteMissionExecutor
- OptimizationResult
- NSGA2MissionOptimizer
- executor.py
- arm_drone
- DStarLiteSession
- .evaluate_decoded
- eval_convlstm_real_firms.py
- risk_routing_ablation.py
- nsga_pixhawk_mission.py
- WildfireConvLSTM
- WildfireDataset
- MissionItem
- _plan_mission
- PredictionRiskMap
- mavsdk_controller.py
- eval_convlstm_held_out.py
- test_mavsdk_controller.py
- set_seed
- DroneState
- build_convlstm_sequence
- csv_export.py
- animate_replan.py
- .predict_grid
- population_objective_stats
- target_count_scenarios
- cell_to_image_rect

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

## Communities (79 total, 5 thin omitted)

### Community 0 - "WildfireSimulator"
Cohesion: 0.06
Nodes (54): FuncAnimation, main(), Configuration for terrain slope and elevation grid generation., Configuration for fuel/vegetation density grid generation., Overall wildfire simulator configurations., Configuration for wind speed and direction., SimulationConfig, TerrainConfig (+46 more)

### Community 1 - "GeoOrigin"
Cohesion: 0.08
Nodes (11): GpsInfo, Health, GeoOrigin, Cell, Local-tangent-plane mapping from mission-grid cells to GPS. The mission grid…, Flat-earth (equirectangular) approximation -- adequate at mission-grid scale., StatusText, main() (+3 more)

### Community 2 - "WildfireEnvironment"
Cohesion: 0.09
Nodes (30): Path, Phase-2 experiment: synthetic scene + pymoo NSGA-II Pareto optimization., Generate a synthetic environment, run NSGA-II, and save visualizations. Returns…, run_optimization_demo(), Path, Phase-1 experiment: generate a synthetic scene and save a visualization., Create a synthetic wildfire mission scene and export a matplotlib figure.…, run_phase1_demo() (+22 more)

### Community 3 - "online_convergence.py"
Cohesion: 0.10
Nodes (28): ConvergenceCallback, Callback, ndarray, Record population quality metrics after every generation., _aggregate(), OnlineConvergenceResult, Path, NSGA-II convergence, collected from the *online* in-mission replanning loop.… (+20 more)

### Community 4 - "online_replanning_comparison.py"
Cohesion: 0.27
Nodes (11): _plot(), Any, Path, Validation Experiment 4 — online replanning policy comparison. Compares three…, run_online_replanning_comparison(), _run_policy(), _summarize(), build_execution_request() (+3 more)

### Community 5 - "ExperimentPaths"
Cohesion: 0.09
Nodes (28): _plot_generations(), Any, Path, Experiment 5 — Generations-to-convergence-threshold vs number of targets. For…, Measure generations-to-threshold as a function of problem size., run_convergence_scaling_experiment(), ExperimentPaths, Path (+20 more)

### Community 6 - "OptimizerConfig"
Cohesion: 0.09
Nodes (23): ElementwiseProblem, Configuration for the mission-planning prototype., OptimizerConfig, Mission-planner configuration. Grid size matches the existing ConvLSTM / YOLO…, NSGA-II mission-optimization settings (pymoo). Inspired by the multi-objective…, constraint_max_distance(), constraint_max_targets(), Constraint helpers for decoded UAV missions. (+15 more)

### Community 7 - "convlstm_bridge.py"
Cohesion: 0.14
Nodes (18): _ignition_points_from_targets(), Seed the driving simulator's ignition cells from the mission's current targets., load_convlstm_checkpoint(), load_terrain_weather_from_simulation(), predict_next_fire_from_grid(), Any, device, Path (+10 more)

### Community 8 - "OnlineReplanResult"
Cohesion: 0.13
Nodes (21): OnlineReplanDemoResult, Experiment runner for the online replanning architecture demo., Run synthetic online replanning, export animations, CSV, plot, and a report.…, run_online_replan_demo(), OnlineReplanner, OnlineReplanResult, Orchestrates prediction updates → NSGA-II → mission scoring → hand-off.…, Full online-replanning demo transcript. (+13 more)

### Community 9 - "detections_to_fire_grid"
Cohesion: 0.21
Nodes (13): _class_weight(), detections_to_fire_grid(), _filter_fire_grid(), FireBoxDetection, FireGridConfig, FireMaskDetection, _passes_threshold(), Remove tiny isolated detections while preserving thin connected fronts. We… (+5 more)

### Community 10 - "train.py"
Cohesion: 0.13
Nodes (19): DataLoader, Module, Optimizer, build_loaders(), Returns (train_loader, val_loader, test_loader). Training loader uses…, bce_loss(), FocalLoss, load_model_state() (+11 more)

### Community 11 - "ExperimentConfig"
Cohesion: 0.34
Nodes (21): ConvergenceExperimentResult, ConvergenceScalingResult, ExperimentConfig, Parameters shared across independent optimization experiments., MissionPathExperimentResult, PopulationSizeResult, _fmt(), generate_report() (+13 more)

### Community 12 - "test_error_injection.py"
Cohesion: 0.07
Nodes (26): AssertionError, _AlwaysConnectedCore, _ConnectedGoodHealthNoHomeTelemetry, _ConnectedNoGpsTelemetry, _FakeSystem, _ForbiddenAircraftControl, _NeverConnectedCore, _NoArmAction (+18 more)

### Community 13 - "MissionExecutionResult"
Cohesion: 0.14
Nodes (23): A Pareto plan paired with its scalar selection score., ScoredMission, MissionExecutionResult, Outcome of executing/locally-planning a ``MissionExecutionRequest``.…, ``path_length / straight_line_length``; 1.0 = surrogate matched actual exactly., EnvironmentDiff, Summarizes how the target set changed between two prediction ticks., build_animation_frames() (+15 more)

### Community 14 - "MissionConfig"
Cohesion: 0.09
Nodes (28): GridConfig, MissionConfig, Top-level configuration for the mission-planning research prototype., Spatial domain for the wildfire mission environment., Parameters for synthetic suppression-target generation. Later: ConvLSTM…, TargetGenerationConfig, build_mission_config(), Optimization result plus wall-clock runtime in seconds. (+20 more)

### Community 15 - "test_architecture_fixes.py"
Cohesion: 0.19
Nodes (10): _compute_risk_grid(), ConvLSTMSourceConfig, ExpectedDamageConfig, Configuration for the fire-probability -> expected-damage-proxy transform.…, Configuration for :class:`ConvLSTMPredictionSource`., Expected-damage-proxy grid: ``fire_probability * severity_norm * fuel_norm``.…, Tests for the architecture-correctness fixes: 1. P(fire) vs. expected-damage-…, The raw ConvLSTM output must never be mutated by the proxy transform. (+2 more)

### Community 16 - "YOLO Fire Detection Integration"
Cohesion: 0.12
Nodes (15): ConvLSTM Integration, Failure Cases, Image-To-Grid Flow, Model Choice, Output Contract, Product Flow (Default Model Paths), Public Fine-Tuning Datasets, Quality Checks (+7 more)

### Community 17 - "test_convlstm_nsga2_integration.py"
Cohesion: 0.26
Nodes (10): ConvLSTMPredictionSource, ConvLSTM-backed prediction source. Each tick, this class: 1. Advances a real…, skipUnless, env_config_or_default(), Lightweight integration tests: ConvLSTM prediction -> NSGA-II mission planning.…, Same seed, same NSGA-II config: prediction changes -> mission changes., A ConvLSTM update mutates SuppressionTarget damage/priority in place., _small_env() (+2 more)

### Community 18 - "ConvLSTMCell"
Cohesion: 0.15
Nodes (9): ConvLSTM, ConvLSTMCell, Tensor, Parameters ---------- x : Tensor shape (B, T, C, H, W) Returns ------- Tensor…, A single ConvLSTM cell. Parameters ---------- in_channels : int — number of…, Stacked ConvLSTM encoder. Processes a sequence (T, C, H, W) through num_layers…, profile_cell(), Single-batch CPU profiler. Uses a synthetic in-memory tensor — no disk, no… (+1 more)

### Community 19 - "style_axes"
Cohesion: 0.17
Nodes (19): mean_std(), Axes, Apply a consistent research-plot style., style_axes(), Experiment runners for the mission-planning research prototype., _plot_pop_objectives(), _plot_pop_pareto(), _plot_pop_runtime() (+11 more)

### Community 20 - "resolve_convlstm_checkpoint"
Cohesion: 0.27
Nodes (7): Path, Default model and data paths for the vision + ConvLSTM product flow., Resolve an optional user path, falling back to a project default., Resolve a ConvLSTM checkpoint file or checkpoint directory., resolve_convlstm_checkpoint(), resolve_path(), TestVisionPaths

### Community 21 - "online_replanner.py"
Cohesion: 0.14
Nodes (19): _explain_replan(), Online NSGA-II replanning loop for dynamic wildfire predictions. Inspired by…, Execute the initial plan plus ``n_replan_events`` online revisions., One online replanning cycle after a prediction update., ReplanEvent, _snapshot_env(), apply_prediction_update(), Synthetic wildfire-prediction dynamics. Randomly edits priorities, positions,… (+11 more)

### Community 22 - "test_safety_gates.py"
Cohesion: 0.12
Nodes (15): _parse(), Namespace, parametrize, TEST 7 -- safety-gate audit for tests/nsga_pixhawk_mission.py (the real-…, Any exception during the flight sequence must fall into the recovery block…, Case-sensitivity matters here: an accidental lowercase 'fly' from a script or a…, Passing --i-have-verified-the-site does NOT also satisfy the typed…, --yes is documented as skipping the typed prompt for scripted use, but it must… (+7 more)

### Community 23 - "SuppressionTarget"
Cohesion: 0.05
Nodes (45): Matplotlib scene rendering options., VisualizationConfig, Mission fitness / scoring., objective_battery_usage(), objective_damage_prevented(), objective_travel_distance(), Isolated multi-objective fitness functions for UAV suppression missions. pymoo…, Objective 1 (to maximize): predicted damage prevented by the mission.… (+37 more)

### Community 24 - "prediction_mission_gap.py"
Cohesion: 0.09
Nodes (30): _cell(), _jaccard(), _plot(), Any, Path, Validation Experiment 5 — prediction quality -> mission quality gap. Not "is…, run_prediction_mission_gap_experiment(), MissionSelectionConfig (+22 more)

### Community 25 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.15
Nodes (12): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Figures, Figures (+4 more)

### Community 26 - "nsga_mavsdk_sitl.py"
Cohesion: 0.27
Nodes (17): _connect_and_wait(), _connect_loop(), _fetch_home(), _health_loop(), _home_loop(), main(), _offset_latlon(), System (+9 more)

### Community 27 - "vision/__init__.py"
Cohesion: 0.20
Nodes (15): compare_grid_distributions(), compute_fire_grid_stats(), FireGridStats, load_simulator_fire_grids(), ndarray, Path, Tensor, Analysis helpers for comparing vision-generated fire grids to simulator grids. (+7 more)

### Community 28 - "test_mission_item_construction.py"
Cohesion: 0.15
Nodes (14): _plan_mission(), Random scenario -> NSGA-II under a wall-clock deadline -> ordered (north_m,…, build_mission_table(), _haversine_m(), print_validation_table(), fixture, TEST 5 -- mission-item construction WITHOUT execution. Runs the exact same…, A swapped (lon, lat) pair for a Bay-Area home would put latitude outside [-90,… (+6 more)

### Community 29 - "ConvLSTM Wildfire Data Architecture — Critical Review"
Cohesion: 0.06
Nodes (30): 10. Weather representation, 11. Fuel / vegetation representation, 12. Final tensor / channel specification, 13. Data preprocessing pipeline, 14. Training/validation split strategy, 15. Real vs. simulation strategy, 16. ConvLSTM vs. alternatives, 17. Physics-informed feature integration (+22 more)

### Community 30 - "generate_scenario"
Cohesion: 0.15
Nodes (16): ConvergenceRecorder, main(), print_final_table(), Callback, Path, Fair NSGA-II generation-count benchmark for the wildfire-suppression optimizer.…, Determine, FROM THE MEASURED DATA, roughly how many generations fit in…, Records the Pareto front's best per-objective values at each requested… (+8 more)

### Community 31 - "common.py"
Cohesion: 0.13
Nodes (29): ensure_paths(), hypervolume_reference_point(), make_environment(), Any, Shared helpers for optimization-performance experiments. Keeps environment…, Summarize a final Pareto set in human-readable objective units., Write a list of dict rows to CSV., Create a reproducible synthetic environment for an experiment. (+21 more)

### Community 32 - "telemetry_diagnostic.py"
Cohesion: 0.15
Nodes (14): DiagnosticReport, main(), System, TEST 4 -- telemetry-only diagnostic. Connects to a vehicle (real Pixhawk over…, Return the first ConnectionState with is_connected=True. The caller wraps this…, Read exactly one item from an async generator, with a timeout. None on…, _read_one(), run_diagnostic() (+6 more)

### Community 33 - "dataset.py"
Cohesion: 0.25
Nodes (4): Sampler, WildfireDataset --------------- Sliding-window dataset for ConvLSTM wildfire…, Yields window indices grouped by simulation file. Simulations are shuffled each…, SimulationGroupedSampler

### Community 34 - "Online Replanning Architecture Demo"
Cohesion: 0.11
Nodes (17): Artifacts, Configuration, Future Integration Notes, Initial Mission, Mission change, Mission change, Mission change, Online Replanning Architecture Demo (+9 more)

### Community 35 - "yolo_fire_adapter.py"
Cohesion: 0.18
Nodes (16): main(), predict_wildfire_from_image(), device, ndarray, Path, End-to-end wildfire prediction: drone RGB image -> YOLO grid -> ConvLSTM…, Outputs from the YOLO + ConvLSTM product flow., Run the full product flow on one image. 1. YOLO segmentation produces the… (+8 more)

### Community 36 - "test_scenario_generation.py"
Cohesion: 0.13
Nodes (16): parametrize, TEST 1 -- scenario generation properties (tests/nsga_scenario.py). Verifies the…, SuppressionTarget.travel_cost is the distance-from-home the objectives can…, The fair-comparison helper must change ONLY n_generations, not the scenario…, RADIUS_M is a plain module-level float the operator can edit, not hardcoded…, A radius too small to fit MIN_TARGETS points MIN_SPACING_M apart must raise,…, damage_score/priority (the objective functions' inputs) come from the project's…, test_every_target_within_radius() (+8 more)

### Community 37 - "Research Comparison — Zhu et al. (2025) vs This Project"
Cohesion: 0.15
Nodes (12): Limitations, Next steps, Not yet implemented (future work, not claimed as done), Precise implementation status, Research Comparison — Zhu et al. (2025) vs This Project, Update: architecture-correctness pass (P(fire) vs. damage, risk map,, Update: ConvLSTM is now connected (was "not yet implemented" as of the, Update: D* Lite is now implemented (was "not yet implemented" as of the (+4 more)

### Community 38 - "DStarLite"
Cohesion: 0.22
Nodes (8): DStarLite, Cell, Move the UAV's current position without resetting planner state. Per D* Lite,…, Incrementally repair the path after obstacle cells change. Only vertices…, Incrementally repair the path after spatially-varying risk changes. Same…, Return the currently-computed path without triggering a new search., Incremental grid path planner. Parameters ---------- width, height: Grid…, Compute a full path from ``start`` to ``goal``. Returns None if unreachable.

### Community 39 - "Wildfire UAV Mission Optimization — Experiment Report"
Cohesion: 0.12
Nodes (16): Checkpoint Summary, Data Artifacts, Experiment 1 — Convergence, Experiment 2 — Runtime Scaling, Experiment 3 — Population Size, Experiment 4 — Mission Path Visualization, Experiment 5 — Generations to Convergence Threshold vs Problem Size, Experiment 6 — Scenario Diversity (+8 more)

### Community 41 - "run_all.py"
Cohesion: 0.36
Nodes (8): Path, Dump the exact ``ExperimentConfig`` used for a run, for reproducibility., write_config_snapshot(), build_config(), main(), parse_args(), Namespace, selected_experiments()

### Community 42 - "SyntheticPredictionSource"
Cohesion: 0.16
Nodes (11): Any, Controls how aggressively the synthetic predictor mutates the scene., Emits random but reproducible prediction updates for architecture demos. Swap…, SyntheticDynamicsConfig, SyntheticPredictionSource, Enum, Kinds of prediction-driven environment edits., One atomic change applied to the live target set. (+3 more)

### Community 43 - "save_figure"
Cohesion: 0.14
Nodes (22): Figure, Save a matplotlib figure as PNG., save_figure(), ConvergenceHistory, _plot_avg_fitness(), _plot_best_fitness(), _plot_hypervolume(), _plot_pareto_size() (+14 more)

### Community 49 - "_offset_latlon"
Cohesion: 0.13
Nodes (24): main(), print_table(), TEST 8 -- per-stage performance timing for the NSGA-II + MAVSDK software…, Local ``mavsdk.System()`` construction ONLY -- no ``connect()`` is called, so…, StageTimings, time_coordinate_conversion(), time_mission_item_construction(), time_nsga2_optimization() (+16 more)

### Community 50 - "DStarLiteMissionExecutor"
Cohesion: 0.21
Nodes (9): DStarLiteMissionExecutor, _euclid(), MissionExecutionRequest, Cell, Execute or refine ``request`` (e.g. via D* Lite)., Plans the real obstacle-aware route for a selected mission using D* Lite.…, Payload passed from the mission selector to a local planner / executor.…, TestExecutorGenuineIncrementalReuse (+1 more)

### Community 51 - "OptimizationResult"
Cohesion: 0.11
Nodes (14): Pick a balanced Pareto mission near the damage–distance knee. Uses normalized…, select_representative_plan(), OptimizationResult, Pareto set returned by ``NSGA2MissionOptimizer.optimize``., Plan with the highest damage prevented (among Pareto set)., Plan with the lowest travel distance (among Pareto set)., Plan with the lowest battery usage (among Pareto set)., plot_pareto_front() (+6 more)

### Community 53 - "NSGA2MissionOptimizer"
Cohesion: 0.15
Nodes (15): _nondominated_mask(), NSGA2MissionOptimizer, ndarray, Execute NSGA-II and return the non-dominated mission plans. Parameters…, Boolean mask of non-dominated rows for a minimization matrix ``F``., Run pymoo NSGA-II over suppression-target permutations. Operators (official…, Scenario, parametrize (+7 more)

### Community 54 - "executor.py"
Cohesion: 0.10
Nodes (17): _Key, path_length(), D* Lite grid path planner (Koenig & Likhachev, 2002). Decides HOW the UAV…, Total Euclidean length of a cell path (8-connected step costs)., DStarLitePlanner, Cell, Local path planner: multi-leg D* Lite mission execution. NSGA-II decides WHAT…, Plans a full multi-leg mission route using D* Lite per leg. Each leg gets a… (+9 more)

### Community 55 - "arm_drone"
Cohesion: 0.20
Nodes (19): arm_drone(), ArmResult, format_gps(), format_health(), health_checks(), health_failure_report(), HealthWatch, Latest values from background telemetry subscriptions, for diagnostics only. (+11 more)

### Community 56 - "DStarLiteSession"
Cohesion: 0.12
Nodes (10): DStarLiteSession, Persistent D* Lite planning session for one active leg (fixed goal). Wraps a…, Begin planning toward ``goal``. Resets internal state (new problem)., Report the UAV's new position; incrementally repairs, does not reset., Mark cells as newly blocked; incrementally repairs, does not reset., Mark cells as newly passable; incrementally repairs, does not reset., Update spatially-varying traversal risk; incrementally repairs., The most recently computed path, without triggering a new search. (+2 more)

### Community 57 - ".evaluate_decoded"
Cohesion: 0.21
Nodes (8): FitnessVector, Any, ndarray, Score an already-decoded feasible (or near-feasible) mission., Evaluate a mission chromosome or ID sequence. Accepts a permutation…, Package a fitness evaluation into a ``MissionPlan``., Evaluated objectives and inequality constraints for one chromosome., Decode a chromosome and compute objectives + constraints.

### Community 58 - "eval_convlstm_real_firms.py"
Cohesion: 0.36
Nodes (9): build_input_sequence(), load_real_fire_pair(), main(), ndarray, Tensor, rasterize(), (WINDOW, 10, GRID, GRID): real fire history (replicated, see limitation above)…, save_comparison() (+1 more)

### Community 59 - "risk_routing_ablation.py"
Cohesion: 0.31
Nodes (8): _build_risk_field(), _plot(), Any, Path, Validation Experiment 3 — risk-weighted D* Lite routing ablation. Fixed…, A localized high-risk blob directly on the shortest path, small enough that the…, RiskRoutingAblationResult, run_risk_routing_ablation()

### Community 60 - "nsga_pixhawk_mission.py"
Cohesion: 0.25
Nodes (17): _build_serial_address(), _confirm_airborne(), _connect_and_wait(), _connect_loop(), _fetch_home(), _home_loop(), main(), Namespace (+9 more)

### Community 61 - "WildfireConvLSTM"
Cohesion: 0.29
Nodes (7): ConvLSTM -------- Convolutional LSTM implementation for spatiotemporal wildfire…, End-to-end wildfire spread prediction model. Architecture: - Optional 2D input…, WildfireConvLSTM, check(), main(), Smoke-test for WildfireDataset + WildfireConvLSTM. Run from project root:…, TestConvLSTMBridge

### Community 62 - "WildfireDataset"
Cohesion: 0.31
Nodes (5): Dataset, Path, Tensor, Parameters ---------- split_json : str | Path Path to train.json / val.json /…, WildfireDataset

### Community 63 - "MissionItem"
Cohesion: 0.33
Nodes (7): MissionItem, _generate_offsets(), main(), _offset_latlon(), _validate_offsets(), main(), _offset_latlon()

### Community 64 - "_plan_mission"
Cohesion: 0.25
Nodes (7): _plan_mission(), Random scenario -> NSGA-II under a wall-clock deadline -> ordered (north_m,…, run_nsga2_with_deadline(), Path, One-shot 2D mission-plan visualization for the NSGA-II scenario/mission…, Draw and save a single 2D map of the planned mission: - home at (0, 0) - every…, save_mission_visualization()

### Community 65 - "PredictionRiskMap"
Cohesion: 0.28
Nodes (5): PredictionRiskMap, Sample ``risk`` at a (possibly fractional) mission-grid coordinate., Convert to a ``{(x, y): risk}`` dict keyed the same way D* Lite's ``Cell`` type…, Full spatial prediction, preserved rather than discarded after target…, TestPredictionRiskMap

### Community 66 - "mavsdk_controller.py"
Cohesion: 0.27
Nodes (16): _abort(), _connect_and_verify(), _connected(), _fly_to_cell(), _haversine_m(), mission(), MissionResult, System (+8 more)

### Community 68 - "eval_convlstm_held_out.py"
Cohesion: 0.48
Nodes (6): local_overlap(), main(), normalize(), Tensor, save_comparison_plot(), aggregate_metrics()

### Community 69 - "test_mavsdk_controller.py"
Cohesion: 0.30
Nodes (13): MissionFailureReason, Enum, _FakeSystem, _make_leg(), asyncio, Tests for the MAVSDK flight-control layer (mission.flight.mavsdk_controller).…, test_arm_drone_reports_rejected_arm(), test_arm_drone_reports_specific_reason_on_health_timeout() (+5 more)

### Community 70 - "set_seed"
Cohesion: 0.50
Nodes (3): Reproducibility helpers for research experiments., Seed Python and NumPy RNGs. Returns a NumPy Generator for use in simulation…, set_seed()

### Community 71 - "DroneState"
Cohesion: 0.35
Nodes (6): decode_permutation(), ndarray, Convert a full permutation into a constraint-feasible ordered mission. Walks…, DroneState, UAV pose in the mission grid (Phase 1: 2-D position only)., TestPermutationDecodingIsASubset

### Community 72 - "build_convlstm_sequence"
Cohesion: 0.14
Nodes (12): _area_pool_to_grid(), _box_to_mask(), build_convlstm_sequence(), _max_pool_to_grid(), overlay_grid_on_image(), ndarray, Tensor, Rasterize an `xyxy` detection box into a binary image-space mask. (+4 more)

### Community 73 - "csv_export.py"
Cohesion: 0.38
Nodes (9): _best(), _dstar_fields(), _event_row(), _initial_row(), _order_str(), Path, CSV export of the online (in-mission) NSGA-II replanning transcript. One row…, Write one CSV row per NSGA-II run in ``result`` (initial + replans). (+1 more)

### Community 74 - "animate_replan.py"
Cohesion: 0.33
Nodes (9): _crop_to_common(), Path, Save online-replanning animations as PNG frames, GIF, and MP4., Convert RGBA/RGB frames to RGB ndarray., Render and export PNG frames + GIF + MP4 for an online-replan run.…, Encode MP4 using the imageio-ffmpeg binary (no system ffmpeg required)., save_replan_outputs(), _write_gif() (+1 more)

### Community 75 - ".predict_grid"
Cohesion: 0.25
Nodes (5): FireDetection, Any, Path, Run YOLO inference and return the 100x100 fire grid result., _resize_mask()

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

- **Why does `WildfireEnvironment` connect `WildfireEnvironment` to `online_replanning_comparison.py`, `OptimizerConfig`, `convlstm_bridge.py`, `OnlineReplanResult`, `ExperimentConfig`, `MissionExecutionResult`, `MissionConfig`, `test_architecture_fixes.py`, `test_convlstm_nsga2_integration.py`, `online_replanner.py`, `SuppressionTarget`, `prediction_mission_gap.py`, `generate_scenario`, `common.py`, `SyntheticPredictionSource`, `DStarLiteMissionExecutor`, `OptimizationResult`, `NSGA2MissionOptimizer`, `executor.py`, `DStarLiteSession`, `.evaluate_decoded`, `PredictionRiskMap`, `DroneState`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `NSGA2MissionOptimizer` connect `NSGA2MissionOptimizer` to `WildfireEnvironment`, `online_replanning_comparison.py`, `OptimizerConfig`, `OnlineReplanResult`, `MissionConfig`, `test_architecture_fixes.py`, `test_convlstm_nsga2_integration.py`, `online_replanner.py`, `SuppressionTarget`, `prediction_mission_gap.py`, `nsga_mavsdk_sitl.py`, `test_mission_item_construction.py`, `generate_scenario`, `common.py`, `_offset_latlon`, `DStarLiteMissionExecutor`, `DStarLiteSession`, `nsga_pixhawk_mission.py`, `_plan_mission`, `PredictionRiskMap`, `DroneState`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `MissionExecutionResult` connect `MissionExecutionResult` to `GeoOrigin`, `mavsdk_controller.py`, `WildfireEnvironment`, `PredictionRiskMap`, `test_mavsdk_controller.py`, `DroneState`, `OnlineReplanResult`, `DStarLiteMissionExecutor`, `online_replanner.py`, `executor.py`, `arm_drone`, `DStarLiteSession`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `WildfireEnvironment` (e.g. with `TimedOptimization` and `MissionPathExperimentResult`) actually correct?**
  _`WildfireEnvironment` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `OptimizerConfig` (e.g. with `TimedOptimization` and `OnlineConvergenceResult`) actually correct?**
  _`OptimizerConfig` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `NSGA2MissionOptimizer` (e.g. with `TimedOptimization` and `OnlineReplanningComparisonResult`) actually correct?**
  _`NSGA2MissionOptimizer` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `ExperimentConfig` (e.g. with `TimedOptimization` and `ConvergenceCallback`) actually correct?**
  _`ExperimentConfig` has 18 INFERRED edges - model-reasoned connections that need verification._