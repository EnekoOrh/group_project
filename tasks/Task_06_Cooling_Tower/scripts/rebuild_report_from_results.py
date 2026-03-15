import csv
import json
import os
import shutil
import sys
from typing import Iterable, List

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.benchmarks.cooling_tower import evaluate_decision, get_scenario, scenario_to_dict
from tasks.Task_06_Cooling_Tower.run_experiments import (
    ALGORITHMS,
    ALGO_SEED_OFFSETS,
    _generate_report,
    _pick_best_run,
)


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cast_row(row: dict, float_fields: Iterable[str], int_fields: Iterable[str]) -> dict:
    out = dict(row)
    for key in float_fields:
        if key in out and out[key] not in ("", None):
            out[key] = float(out[key])
    for key in int_fields:
        if key in out and out[key] not in ("", None):
            out[key] = int(float(out[key]))
    return out


def _infer_runs(raw_rows: List[dict]) -> int:
    if not raw_rows:
        return 0
    return max(int(row["run_index"]) for row in raw_rows) + 1


def _infer_seed_offset(raw_rows: List[dict]) -> int:
    offsets = []
    for row in raw_rows:
        algo = str(row["algorithm"])
        if algo not in ALGO_SEED_OFFSETS:
            continue
        offsets.append(int(row["seed"]) - ALGO_SEED_OFFSETS[algo] - int(row["run_index"]))
    return min(offsets) if offsets else 0


def main() -> None:
    task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(task_dir, "results")
    data_dir = os.path.join(results_dir, "data")
    figures_dir = os.path.join(results_dir, "figures")
    report_outdir = os.path.join(results_dir, "reports")
    report_path = os.path.join(report_outdir, "Report.md")
    root_report_path = os.path.join(task_dir, "Report.md")

    raw_rows = _read_csv(os.path.join(data_dir, "raw_runs.csv"))
    scenario_summary_rows = [
        _cast_row(
            row,
            float_fields=[
                "mean_area",
                "median_area",
                "area_iqr_q1",
                "area_iqr_q3",
                "area_ci95",
                "std_area",
                "mean_rel_volume_error",
                "mean_rel_height_error",
                "mean_monotonic_violation",
                "mean_evals",
                "mean_time_s",
                "feasibility_rate",
                "engineering_feasibility_rate",
                "feasible_area_mean",
                "feasible_area_median",
                "engineering_feasible_area_mean",
                "engineering_feasible_area_median",
                "best_penalized_objective",
            ],
            int_fields=["runs", "feasible_runs", "engineering_feasible_runs"],
        )
        for row in _read_csv(os.path.join(data_dir, "scenario_summary.csv"))
    ]
    algorithm_summary_rows = [
        _cast_row(
            row,
            float_fields=[
                "mean_area",
                "median_area",
                "area_iqr_q1",
                "area_iqr_q3",
                "area_ci95",
                "mean_rel_volume_error",
                "mean_rel_height_error",
                "mean_evals",
                "mean_time_s",
                "overall_feasibility_rate",
                "overall_engineering_feasibility_rate",
                "feasible_area_mean",
                "feasible_area_median",
                "engineering_feasible_area_mean",
                "engineering_feasible_area_median",
            ],
            int_fields=["total_runs", "feasible_runs", "engineering_feasible_runs"],
        )
        for row in _read_csv(os.path.join(data_dir, "algorithm_summary.csv"))
    ]

    scenario_ids = sorted({str(row["scenario_id"]) for row in scenario_summary_rows})
    scenario_definitions = [scenario_to_dict(get_scenario(sid)) for sid in scenario_ids]
    visual_selection_by_scenario = {}

    for sid in scenario_ids:
        scenario = get_scenario(sid)
        visual_selection_by_scenario[sid] = {}
        for algo_name in ALGORITHMS:
            rows = [row for row in raw_rows if row["scenario_id"] == sid and row["algorithm"] == algo_name]
            if not rows:
                continue

            best_row = _pick_best_run(rows)
            best_x = np.asarray(json.loads(best_row["decision_vector"]), dtype=float)
            best_metrics = evaluate_decision(scenario, best_x)
            selection_basis = (
                "best_feasible_area" if str(best_row["feasible"]).lower() == "true" else "best_penalized_objective"
            )
            visual_selection_by_scenario[sid][algo_name] = {
                "run_index": int(best_row["run_index"]),
                "seed": int(best_row["seed"]),
                "selection_basis": selection_basis,
                "metrics": {
                    "feasible": bool(best_metrics["feasible"]),
                    "area": float(best_metrics["area"]),
                    "rel_volume_error": float(best_metrics["rel_volume_error"]),
                    "rel_height_error": float(best_metrics["rel_height_error"]),
                    "monotonic_violation": float(best_metrics["monotonic_violation"]),
                    "radii": np.asarray(best_metrics["radii"], dtype=float).tolist(),
                    "heights": np.asarray(best_metrics["heights"], dtype=float).tolist(),
                },
            }

    _generate_report(
        report_path=report_path,
        scenario_ids=scenario_ids,
        scenario_definitions=scenario_definitions,
        scenario_summary_rows=scenario_summary_rows,
        algorithm_summary_rows=algorithm_summary_rows,
        visual_selection_by_scenario=visual_selection_by_scenario,
        include_3d=True,
        runs=_infer_runs(raw_rows),
        seed_offset=_infer_seed_offset(raw_rows),
        figures_dir=figures_dir,
    )

    shutil.copyfile(report_path, root_report_path)

    print(f"Regenerated markdown report: {report_path}")
    print(f"Synced root report: {root_report_path}")


if __name__ == "__main__":
    main()
