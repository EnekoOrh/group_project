import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from task7_common import (
    STATUS_ENGINEERING,
    STATUS_MATH_FALLBACK,
    STATUS_WARNING,
    build_case_payload,
    load_csv,
    load_json,
    select_task7_case,
)


STATUS_STYLE = {
    STATUS_ENGINEERING: {"linestyle": "-", "alpha": 1.0, "suffix": "engineering-feasible"},
    STATUS_MATH_FALLBACK: {"linestyle": "--", "alpha": 0.95, "suffix": "math fallback"},
    STATUS_WARNING: {"linestyle": ":", "alpha": 0.90, "suffix": "warning"},
}
ALGORITHM_COLORS = {"SA": "#C0392B", "PSO": "#1F618D", "BFGS": "#117A65"}


def _write_profile_csv(path: Path, z_values, radii):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_m", "radius_m"])
        for z_val, radius in zip(z_values, radii):
            writer.writerow([f"{z_val:.6f}", f"{radius:.6f}"])


def _plot_profiles(cases, output_path: Path):
    grouped = {}
    for case in cases:
        grouped.setdefault(case["scenario_id"], []).append(case)

    scenario_ids = sorted(grouped.keys(), key=lambda value: int(value[1:]))
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=False, sharey=False)
    axes = axes.flatten()

    for axis, scenario_id in zip(axes, scenario_ids):
        scenario_cases = sorted(grouped[scenario_id], key=lambda row: row["algorithm"])
        for case in scenario_cases:
            style = STATUS_STYLE[case["selection_status"]]
            label = f"{case['algorithm']} ({style['suffix']})"
            color = ALGORITHM_COLORS[case["algorithm"]]
            axis.plot(case["radii_m"], case["z_m"], label=label, linewidth=2.0, color=color, linestyle=style["linestyle"], alpha=style["alpha"])
            axis.plot([-value for value in case["radii_m"]], case["z_m"], linewidth=2.0, color=color, linestyle=style["linestyle"], alpha=style["alpha"])
        axis.set_title(f"{scenario_id}: {scenario_cases[0]['scenario_description']}", fontsize=10)
        axis.set_xlabel("Radius (m)")
        axis.set_ylabel("Height z (m)")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=8, loc="best")

    fig.suptitle("Task 7 candidate cooling-tower profiles selected from Task 6", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export Task 6 cooling-tower candidates for Task 7 Abaqus.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--raw-runs", default=str(repo_root / "tasks" / "Task_06_Cooling_Tower" / "results" / "data" / "raw_runs.csv"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus"))
    args = parser.parse_args()

    config = load_json(Path(args.config))
    raw_runs_csv = Path(args.raw_runs)
    output_dir = Path(args.output_dir)
    candidates_dir = output_dir / "candidates"
    profiles_dir = candidates_dir / "profiles"
    figures_dir = output_dir / "results" / "figures"
    data_dir = output_dir / "results" / "data"

    profiles_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(raw_runs_csv)
    exported_cases = []
    warning_scenarios = set(config.get("selection", {}).get("warning_scenarios", []))
    scenario_ids = config.get("selection", {}).get("scenario_ids", [f"S{index}" for index in range(1, 9)])
    algorithms = config.get("selection", {}).get("algorithms", ["SA", "PSO", "BFGS"])

    for scenario_id in scenario_ids:
        for algorithm in algorithms:
            selected_row, selection_status, selection_basis, selection_note = select_task7_case(
                rows, scenario_id, algorithm, warning_scenarios=warning_scenarios
            )
            case_payload = build_case_payload(
                selected_row,
                scenario_id=scenario_id,
                algorithm=algorithm,
                selection_status=selection_status,
                selection_basis=selection_basis,
                selection_note=selection_note,
            )
            exported_cases.append(case_payload)
            _write_profile_csv(profiles_dir / f"{case_payload['case_id']}.csv", case_payload["z_m"], case_payload["radii_m"])

    exported_cases.sort(key=lambda case: (int(case["scenario_id"][1:]), case["algorithm"]))

    with (candidates_dir / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(exported_cases, handle, indent=2)

    with (candidates_dir / "candidate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "model_name",
                "label",
                "scenario_id",
                "algorithm",
                "selection_status",
                "selection_basis",
                "source_run_index",
                "decision_mode",
                "task6_area_m2",
                "task6_volume_m3",
                "task6_total_height_m",
                "task6_penalized_objective",
            ]
        )
        for case in exported_cases:
            writer.writerow(
                [
                    case["case_id"],
                    case["model_name"],
                    case["label"],
                    case["scenario_id"],
                    case["algorithm"],
                    case["selection_status"],
                    case["selection_basis"],
                    case["source_run_index"],
                    case["decision_mode"],
                    f"{case['task6_area_m2']:.6f}",
                    f"{case['task6_volume_m3']:.6f}",
                    f"{case['task6_total_height_m']:.6f}",
                    f"{case['task6_penalized_objective']:.6f}",
                ]
            )

    _plot_profiles(exported_cases, figures_dir / "candidate_profiles.png")

    with (data_dir / "selected_case_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"cases": exported_cases}, handle, indent=2)

    status_counts = {}
    for case in exported_cases:
        status_counts[case["selection_status"]] = status_counts.get(case["selection_status"], 0) + 1

    print(f"Exported {len(exported_cases)} Task 7 candidate cases from Task 6.")
    for status, count in sorted(status_counts.items()):
        print(f"  - {status}: {count}")


if __name__ == "__main__":
    main()
