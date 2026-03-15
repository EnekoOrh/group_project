import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.cooling_tower import decode_decision_vector, get_scenario, tower_area_volume  # noqa: E402


SELECTIONS = [
    {
        "case_id": "radius_baseline",
        "label": "S2 / BFGS radius baseline",
        "scenario_id": "S2",
        "algorithm": "BFGS",
        "description": "Radius-only optimized tower used as the clean Task 6 baseline candidate.",
    },
    {
        "case_id": "height_variant",
        "label": "S6 / BFGS height variant",
        "scenario_id": "S6",
        "algorithm": "BFGS",
        "description": "Height-only optimized tower used to test structural sensitivity to vertical segmentation.",
    },
    {
        "case_id": "joint_reference",
        "label": "S7 / BFGS joint reference",
        "scenario_id": "S7",
        "algorithm": "BFGS",
        "description": "Joint radii-height optimized tower used as the main realistic reference for Task 7.",
    },
]


def _load_rows(raw_runs_csv: Path):
    with raw_runs_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_best_engineering_feasible(rows, scenario_id, algorithm):
    subset = [
        row
        for row in rows
        if row["scenario_id"] == scenario_id
        and row["algorithm"] == algorithm
        and row["engineering_feasible"].strip().lower() == "true"
    ]
    if not subset:
        raise ValueError(f"No engineering-feasible runs found for {scenario_id} / {algorithm}")
    return min(subset, key=lambda row: float(row["area"]))


def _write_profile_csv(path: Path, z_values, radii):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_m", "radius_m"])
        for z_val, radius in zip(z_values, radii):
            writer.writerow([f"{z_val:.6f}", f"{radius:.6f}"])


def _plot_profiles(cases, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "radius_baseline": "#C0392B",
        "height_variant": "#1F618D",
        "joint_reference": "#117A65",
    }
    for case in cases:
        color = colors.get(case["case_id"], "#444444")
        ax.plot(case["radii_m"], case["z_m"], label=case["label"], linewidth=2.2, color=color)
        ax.plot([-value for value in case["radii_m"]], case["z_m"], linewidth=2.2, color=color)

    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Height z (m)")
    ax.set_title("Task 7 Candidate Cooling-Tower Profiles from Task 6")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export Task 6 cooling-tower candidates for Task 7 Abaqus.")
    parser.add_argument(
        "--raw-runs",
        default=str(REPO_ROOT / "tasks" / "Task_06_Cooling_Tower" / "results" / "data" / "raw_runs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tasks" / "Task_07_Abaqus"),
    )
    args = parser.parse_args()

    raw_runs_csv = Path(args.raw_runs)
    output_dir = Path(args.output_dir)
    candidates_dir = output_dir / "candidates"
    profiles_dir = candidates_dir / "profiles"
    figures_dir = output_dir / "results" / "figures"
    data_dir = output_dir / "results" / "data"

    profiles_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(raw_runs_csv)
    exported_cases = []

    for selection in SELECTIONS:
        row = _select_best_engineering_feasible(rows, selection["scenario_id"], selection["algorithm"])
        scenario = get_scenario(selection["scenario_id"])
        decision_vector = np.array(json.loads(row["decision_vector"]), dtype=float)
        radii, heights, z_values = decode_decision_vector(scenario, decision_vector)
        area, volume = tower_area_volume(radii, heights)

        case_payload = {
            "case_id": selection["case_id"],
            "label": selection["label"],
            "description": selection["description"],
            "scenario_id": selection["scenario_id"],
            "algorithm": selection["algorithm"],
            "source_run_index": int(row["run_index"]),
            "decision_mode": scenario.decision_mode,
            "task6_area_m2": float(area),
            "task6_volume_m3": float(volume),
            "task6_total_height_m": float(z_values[-1]),
            "task6_rel_volume_error": float(row["rel_volume_error"]),
            "task6_rel_height_error": float(row["rel_height_error"]),
            "task6_engineering_failed_checks": row["engineering_failed_checks"],
            "radii_m": [float(value) for value in radii.tolist()],
            "heights_m": [float(value) for value in heights.tolist()],
            "z_m": [float(value) for value in z_values.tolist()],
        }
        exported_cases.append(case_payload)
        _write_profile_csv(profiles_dir / f"{selection['case_id']}.csv", z_values, radii)

    with (candidates_dir / "selected_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(exported_cases, handle, indent=2)

    with (candidates_dir / "candidate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "label",
                "scenario_id",
                "algorithm",
                "source_run_index",
                "decision_mode",
                "task6_area_m2",
                "task6_volume_m3",
                "task6_total_height_m",
            ]
        )
        for case in exported_cases:
            writer.writerow(
                [
                    case["case_id"],
                    case["label"],
                    case["scenario_id"],
                    case["algorithm"],
                    case["source_run_index"],
                    case["decision_mode"],
                    f"{case['task6_area_m2']:.6f}",
                    f"{case['task6_volume_m3']:.6f}",
                    f"{case['task6_total_height_m']:.6f}",
                ]
            )

    _plot_profiles(exported_cases, figures_dir / "candidate_profiles.png")

    with (data_dir / "selected_case_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"cases": exported_cases}, handle, indent=2)

    print(f"Exported {len(exported_cases)} Task 7 candidate cases from Task 6.")


if __name__ == "__main__":
    main()
