import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.cooling_tower import decode_decision_vector, get_scenario, tower_area_volume  # noqa: E402


STATUS_ENGINEERING = "engineering_feasible"
STATUS_MATH_FALLBACK = "mathematical_fallback"
STATUS_WARNING = "warning_noncompliant"


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scenario_algorithm_case_id(scenario_id: str, algorithm: str) -> str:
    return f"{scenario_id.lower()}_{algorithm.lower()}"


def scenario_algorithm_model_name(scenario_id: str, algorithm: str) -> str:
    return f"{scenario_id}_{algorithm}"


def scenario_algorithm_label(scenario_id: str, algorithm: str) -> str:
    return f"{scenario_id} / {algorithm}"


def refined_job_name(case_payload: dict) -> str:
    return f"task7_{case_payload['case_id']}_refined"


def coarse_job_name(case_payload: dict) -> str:
    return f"task7_{case_payload['case_id']}_coarse"


def status_rank(case_payload: dict) -> int:
    status = case_payload.get("selection_status", "")
    if status == STATUS_ENGINEERING:
        return 0
    if status == STATUS_MATH_FALLBACK:
        return 1
    return 2


def select_task7_case(rows, scenario_id: str, algorithm: str, warning_scenarios=None):
    warning_scenarios = set(warning_scenarios or [])
    subset = [row for row in rows if row["scenario_id"] == scenario_id and row["algorithm"] == algorithm]
    if not subset:
        raise ValueError(f"No Task 6 rows found for {scenario_id} / {algorithm}")

    def by_area(row):
        return float(row["area"])

    def by_penalized(row):
        return float(row["penalized_objective"])

    engineering = [row for row in subset if row["engineering_feasible"].strip().lower() == "true"]
    mathematical = [row for row in subset if row["feasible"].strip().lower() == "true"]

    if scenario_id in warning_scenarios:
        selected = min(subset, key=by_penalized)
        return (
            selected,
            STATUS_WARNING,
            "best_penalized_warning",
            f"{scenario_id} has no compliant Task 6 target by design in this workflow, so this case is kept as a warning-only structural comparison.",
        )

    if engineering:
        selected = min(engineering, key=by_area)
        return (
            selected,
            STATUS_ENGINEERING,
            "lowest_area_engineering_feasible",
            "Lowest-area engineering-feasible Task 6 tower selected for this scenario/optimizer pair.",
        )

    if mathematical:
        selected = min(mathematical, key=by_area)
        return (
            selected,
            STATUS_MATH_FALLBACK,
            "lowest_area_mathematical_fallback",
            "No engineering-feasible Task 6 tower was available for this scenario/optimizer pair, so the lowest-area mathematically compliant fallback was selected.",
        )

    selected = min(subset, key=by_penalized)
    return (
        selected,
        STATUS_WARNING,
        "best_penalized_warning",
        "No engineering-feasible or mathematically compliant Task 6 tower was available for this scenario/optimizer pair, so the best penalized result is kept as a warning-only structural comparison.",
    )


def build_case_payload(selected_row: dict, scenario_id: str, algorithm: str, selection_status: str, selection_basis: str, selection_note: str) -> dict:
    scenario = get_scenario(scenario_id)
    decision_vector = np.array(json.loads(selected_row["decision_vector"]), dtype=float)
    radii, heights, z_values = decode_decision_vector(scenario, decision_vector)
    area, volume = tower_area_volume(radii, heights)

    return {
        "case_id": scenario_algorithm_case_id(scenario_id, algorithm),
        "model_name": scenario_algorithm_model_name(scenario_id, algorithm),
        "label": scenario_algorithm_label(scenario_id, algorithm),
        "scenario_id": scenario_id,
        "scenario_description": scenario.description,
        "algorithm": algorithm,
        "selection_status": selection_status,
        "selection_basis": selection_basis,
        "selection_note": selection_note,
        "is_warning_case": selection_status == STATUS_WARNING,
        "source_run_index": int(selected_row["run_index"]),
        "decision_mode": scenario.decision_mode,
        "task6_penalized_objective": float(selected_row["penalized_objective"]),
        "task6_area_m2": float(area),
        "task6_volume_m3": float(volume),
        "task6_total_height_m": float(z_values[-1]),
        "task6_rel_volume_error": float(selected_row["rel_volume_error"]),
        "task6_rel_height_error": float(selected_row["rel_height_error"]),
        "task6_math_feasible": selected_row["feasible"].strip().lower() == "true",
        "task6_engineering_feasible": selected_row["engineering_feasible"].strip().lower() == "true",
        "task6_engineering_failed_checks": selected_row["engineering_failed_checks"],
        "radii_m": [float(value) for value in radii.tolist()],
        "heights_m": [float(value) for value in heights.tolist()],
        "z_m": [float(value) for value in z_values.tolist()],
    }


def interpolate_profile(z_values, radii, subdivisions_per_segment):
    ring_z = [float(z_values[0])]
    ring_r = [float(radii[0])]
    for index in range(len(z_values) - 1):
        z0 = float(z_values[index])
        z1 = float(z_values[index + 1])
        r0 = float(radii[index])
        r1 = float(radii[index + 1])
        for sub_index in range(1, subdivisions_per_segment + 1):
            t = sub_index / float(subdivisions_per_segment)
            ring_z.append((1.0 - t) * z0 + t * z1)
            ring_r.append((1.0 - t) * r0 + t * r1)
    return ring_z, ring_r


def build_mesh(case_data, n_theta, axial_subdivisions):
    ring_z, ring_r = interpolate_profile(case_data["z_m"], case_data["radii_m"], axial_subdivisions)
    nodes = []
    ring_node_ids = []
    node_id = 1

    for radius, height_y in zip(ring_r, ring_z):
        current_ring = []
        for theta_index in range(n_theta):
            theta = 2.0 * math.pi * theta_index / n_theta
            x_coord = radius * math.cos(theta)
            z_coord = radius * math.sin(theta)
            nodes.append((node_id, x_coord, height_y, z_coord))
            current_ring.append(node_id)
            node_id += 1
        ring_node_ids.append(current_ring)

    elements = []
    sector_sets = {index: [] for index in range(n_theta)}
    element_id = 1
    for ring_index in range(len(ring_node_ids) - 1):
        lower_ring = ring_node_ids[ring_index]
        upper_ring = ring_node_ids[ring_index + 1]
        for theta_index in range(n_theta):
            next_theta = (theta_index + 1) % n_theta
            n1 = lower_ring[theta_index]
            n2 = lower_ring[next_theta]
            n3 = upper_ring[next_theta]
            n4 = upper_ring[theta_index]
            elements.append((element_id, n1, n2, n3, n4))
            sector_sets[theta_index].append(element_id)
            element_id += 1

    return {
        "nodes": nodes,
        "elements": elements,
        "bottom_nodes": ring_node_ids[0],
        "top_nodes": ring_node_ids[-1],
        "ring_count": len(ring_node_ids),
        "ring_z": ring_z,
        "ring_r": ring_r,
        "sector_sets": sector_sets,
    }


def pressure_coefficient(theta_rad):
    raw_value = 0.8 * math.cos(theta_rad)
    return max(-0.5, min(0.8, raw_value))


def geometry_verification(case_data, mesh_data, axial_subdivisions):
    stride = int(axial_subdivisions)
    source_z = [float(value) for value in case_data["z_m"]]
    source_r = [float(value) for value in case_data["radii_m"]]
    sampled_z = [float(mesh_data["ring_z"][index]) for index in range(0, len(mesh_data["ring_z"]), stride)]
    sampled_r = [float(mesh_data["ring_r"][index]) for index in range(0, len(mesh_data["ring_r"]), stride)]

    if len(sampled_z) != len(source_z):
        raise ValueError(f"Geometry verification sampling mismatch for {case_data['case_id']}")

    max_radius_abs_diff = max(abs(a - b) for a, b in zip(sampled_r, source_r)) if source_r else 0.0
    max_z_abs_diff = max(abs(a - b) for a, b in zip(sampled_z, source_z)) if source_z else 0.0
    mesh_total_height = mesh_data["ring_z"][-1] - mesh_data["ring_z"][0]
    source_total_height = source_z[-1] - source_z[0]

    return {
        "source_total_height_m": source_total_height,
        "mesh_total_height_m": mesh_total_height,
        "abs_height_diff_m": abs(mesh_total_height - source_total_height),
        "max_source_ring_radius_abs_diff_m": max_radius_abs_diff,
        "max_source_ring_z_abs_diff_m": max_z_abs_diff,
    }
