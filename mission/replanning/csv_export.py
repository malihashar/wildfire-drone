"""
CSV export of the online (in-mission) NSGA-II replanning transcript.

One row per NSGA-II run: the initial pre-mission optimization (tick 0) plus
every subsequent online replan event triggered by a synthetic wildfire
prediction update.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mission.replanning.online_replanner import OnlineReplanResult

FIELDNAMES = [
    "tick",
    "scenario_id",
    "prediction_source",
    "update_note",
    "n_targets_remaining",
    "suppressed_target_id",
    "selected_target_order",
    "n_pareto",
    "optimization_runtime_s",
    "selected_damage_prevented",
    "selected_travel_distance",
    "selected_battery_usage",
    "pareto_best_damage_prevented",
    "pareto_best_travel_distance",
    "pareto_best_battery_usage",
    "mission_score",
    "score_delta",
    # D* Lite actual route vs. NSGA-II's surrogate (selected_travel_distance
    # above) -- kept separate on purpose, see objectives.objective_travel_distance.
    "dstar_actual_distance",
    "dstar_feasible",
    "dstar_deviation_ratio",
    "dstar_used_incremental_replan",
]


def write_replan_csv(result: OnlineReplanResult, path: Path) -> Path:
    """Write one CSV row per NSGA-II run in ``result`` (initial + replans)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_id = f"seed{result.config.seed}"
    rows: list[dict[str, object]] = [_initial_row(result, scenario_id)]
    rows.extend(_event_row(event, scenario_id) for event in result.events)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _initial_row(result: OnlineReplanResult, scenario_id: str) -> dict[str, object]:
    plan = result.initial_mission.plan
    opt = result.initial_result
    return {
        "tick": 0,
        "scenario_id": scenario_id,
        "prediction_source": "initial",
        "update_note": "initial NSGA-II mission (pre-replan)",
        "n_targets_remaining": result.initial_env.n_targets,
        "suppressed_target_id": "",
        "selected_target_order": _order_str(result.initial_mission.target_ids),
        "n_pareto": result.initial_n_pareto,
        "optimization_runtime_s": result.initial_runtime_s,
        "selected_damage_prevented": plan.objectives.damage_prevented,
        "selected_travel_distance": plan.objectives.travel_distance,
        "selected_battery_usage": plan.objectives.battery_usage,
        "pareto_best_damage_prevented": _best(opt, "damage"),
        "pareto_best_travel_distance": _best(opt, "travel"),
        "pareto_best_battery_usage": _best(opt, "battery"),
        "mission_score": result.initial_mission.score,
        "score_delta": "",
        **_dstar_fields(result.initial_execution_result),
    }


def _event_row(event, scenario_id: str) -> dict[str, object]:
    plan = event.new_mission.plan
    opt = event.opt_result
    return {
        "tick": event.tick,
        "scenario_id": scenario_id,
        "prediction_source": event.update.source_name,
        "update_note": event.update.note,
        "n_targets_remaining": event.n_targets,
        "suppressed_target_id": (
            event.suppressed_target_id if event.suppressed_target_id is not None else ""
        ),
        "selected_target_order": _order_str(event.new_mission.target_ids),
        "n_pareto": event.n_pareto,
        "optimization_runtime_s": event.optimization_runtime_s,
        "selected_damage_prevented": plan.objectives.damage_prevented,
        "selected_travel_distance": plan.objectives.travel_distance,
        "selected_battery_usage": plan.objectives.battery_usage,
        "pareto_best_damage_prevented": _best(opt, "damage"),
        "pareto_best_travel_distance": _best(opt, "travel"),
        "pareto_best_battery_usage": _best(opt, "battery"),
        "mission_score": event.new_score,
        "score_delta": event.new_score - event.previous_score,
        **_dstar_fields(event.execution_result),
    }


def _dstar_fields(execution_result) -> dict[str, object]:
    if execution_result is None:
        return {
            "dstar_actual_distance": "",
            "dstar_feasible": "",
            "dstar_deviation_ratio": "",
            "dstar_used_incremental_replan": "",
        }
    return {
        "dstar_actual_distance": execution_result.path_length,
        "dstar_feasible": execution_result.feasible,
        "dstar_deviation_ratio": execution_result.deviation_ratio,
        "dstar_used_incremental_replan": execution_result.used_incremental_replan,
    }


def _order_str(target_ids: tuple[int, ...]) -> str:
    return " ".join(str(i) for i in target_ids)


def _best(opt, which: str) -> float | str:
    if opt.n_solutions == 0:
        return ""
    if which == "damage":
        return opt.best_damage_plan().objectives.damage_prevented
    if which == "travel":
        return opt.best_distance_plan().objectives.travel_distance
    return opt.best_battery_plan().objectives.battery_usage
