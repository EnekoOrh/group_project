import argparse
import csv
import json
import math
from pathlib import Path

from task7_common import build_mesh, geometry_verification, load_json, pressure_coefficient


def _write_nset(handle, set_name, node_ids):
    handle.write(f"*NSET, NSET={set_name}\n")
    for start in range(0, len(node_ids), 16):
        chunk = node_ids[start : start + 16]
        handle.write(", ".join(str(node_id) for node_id in chunk) + "\n")


def _write_elset(handle, set_name, element_ids):
    handle.write(f"*ELSET, ELSET={set_name}\n")
    for start in range(0, len(element_ids), 16):
        chunk = element_ids[start : start + 16]
        handle.write(", ".join(str(element_id) for element_id in chunk) + "\n")


def _write_input_file(path: Path, mesh_data, config, case_data):
    material = config["material"]
    shell = config["shell"]
    wind = config["wind"]
    gravity = config["gravity"]
    buckling = config["buckling"]
    n_theta = len(mesh_data["sector_sets"])
    q_ref = 0.5 * float(wind["air_density_kg_m3"]) * float(wind["reference_speed_m_s"]) ** 2

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("*HEADING\n")
        handle.write(f"** Task 7 generated input for {case_data['label']} ({case_data['selection_status']})\n")
        handle.write("*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n")
        handle.write("*NODE\n")
        for node_id, x_coord, y_coord, z_coord in mesh_data["nodes"]:
            handle.write(f"{node_id}, {x_coord:.6f}, {y_coord:.6f}, {z_coord:.6f}\n")

        handle.write("*ELEMENT, TYPE=S4R, ELSET=EALL\n")
        for element_id, n1, n2, n3, n4 in mesh_data["elements"]:
            handle.write(f"{element_id}, {n1}, {n2}, {n3}, {n4}\n")

        _write_nset(handle, "NALL", [node[0] for node in mesh_data["nodes"]])
        _write_nset(handle, "NBASE", mesh_data["bottom_nodes"])
        _write_nset(handle, "NTOP", mesh_data["top_nodes"])

        for sector_index, element_ids in mesh_data["sector_sets"].items():
            _write_elset(handle, f"WIND_SEC_{sector_index + 1:02d}", element_ids)

        handle.write(f"*MATERIAL, NAME={material['name']}\n")
        handle.write("*ELASTIC\n")
        handle.write(f"{material['youngs_modulus_pa']:.6f}, {material['poisson_ratio']:.6f}\n")
        handle.write("*DENSITY\n")
        handle.write(f"{material['density_kg_m3']:.6f}\n")
        handle.write(f"*SHELL SECTION, ELSET=EALL, MATERIAL={material['name']}\n")
        handle.write(f"{shell['thickness_m']:.6f}\n")

        handle.write("*BOUNDARY\n")
        handle.write("NBASE, 1, 6\n")

        direction = gravity["direction"]

        def write_load_pattern():
            handle.write("*DLOAD\n")
            handle.write(
                f"EALL, GRAV, {gravity['acceleration_m_s2']:.6f}, {direction[0]:.6f}, {direction[1]:.6f}, {direction[2]:.6f}\n"
            )
            for sector_index in range(n_theta):
                theta_center = 2.0 * math.pi * (sector_index + 0.5) / n_theta
                cp = pressure_coefficient(theta_center)
                pressure_value = -q_ref * cp
                if abs(pressure_value) < 1e-9:
                    continue
                handle.write(f"WIND_SEC_{sector_index + 1:02d}, P, {pressure_value:.6f}\n")

        handle.write("*STEP, NAME=STATIC_WIND, NLGEOM=NO\n")
        handle.write("*STATIC\n")
        handle.write("0.1, 1.0, 1e-05, 1.0\n")
        write_load_pattern()
        handle.write("*OUTPUT, FIELD, FREQUENCY=1\n")
        handle.write("*NODE OUTPUT, NSET=NALL\n")
        handle.write("U\n")
        handle.write("*NODE OUTPUT, NSET=NBASE\n")
        handle.write("RF\n")
        handle.write("*ELEMENT OUTPUT, ELSET=EALL\n")
        handle.write("S, E\n")
        handle.write("*END STEP\n")

        handle.write("*STEP, NAME=BUCKLING, PERTURBATION\n")
        handle.write("*BUCKLE, EIGENSOLVER=LANCZOS\n")
        handle.write(f"{buckling['num_eigenvalues']},\n")
        write_load_pattern()
        handle.write("*OUTPUT, FIELD, FREQUENCY=1\n")
        handle.write("*NODE OUTPUT, NSET=NALL\n")
        handle.write("U\n")
        handle.write("*END STEP\n")

    return q_ref


def main():
    parser = argparse.ArgumentParser(description="Build Abaqus input decks for Task 7 cooling-tower cases.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--cases", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results"))
    args = parser.parse_args()

    config = load_json(Path(args.config))
    cases = load_json(Path(args.cases))
    results_dir = Path(args.output_dir)
    inputs_dir = results_dir / "inputs"
    data_dir = results_dir / "data"
    jobs_dir = results_dir / "jobs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    mesh_config = config["mesh"]
    variants = [
        {"mesh_level": "coarse", "n_theta": int(mesh_config["circumferential_divisions"]), "axial_subdivisions": int(mesh_config["axial_subdivisions_per_segment"]), "job_suffix": "_coarse"},
        {"mesh_level": "refined", "n_theta": int(mesh_config["refined_circumferential_divisions"]), "axial_subdivisions": int(mesh_config["refined_axial_subdivisions_per_segment"]), "job_suffix": "_refined"},
    ]

    manifest_cases = []
    mesh_rows = []
    verification_rows = []
    q_ref = None

    for case in cases:
        for variant in variants:
            mesh_data = build_mesh(case, variant["n_theta"], variant["axial_subdivisions"])
            job_name = f"task7_{case['case_id']}{variant['job_suffix']}"
            input_path = inputs_dir / f"{job_name}.inp"
            q_ref = _write_input_file(input_path, mesh_data, config, case)
            geom_check = geometry_verification(case, mesh_data, variant["axial_subdivisions"])

            manifest_cases.append(
                {
                    "job_name": job_name,
                    "model_name": case["model_name"],
                    "case_id": case["case_id"],
                    "case_label": case["label"],
                    "scenario_id": case["scenario_id"],
                    "algorithm": case["algorithm"],
                    "selection_status": case["selection_status"],
                    "selection_basis": case["selection_basis"],
                    "selection_note": case["selection_note"],
                    "is_warning_case": case["is_warning_case"],
                    "case_variant": "comparison",
                    "mesh_level": variant["mesh_level"],
                    "input_path": str(input_path),
                    "odb_path": str(jobs_dir / f"{job_name}.odb"),
                    "circumferential_divisions": variant["n_theta"],
                    "axial_subdivisions_per_segment": variant["axial_subdivisions"],
                    "task6_area_m2": case["task6_area_m2"],
                    "task6_penalized_objective": case["task6_penalized_objective"],
                }
            )
            mesh_rows.append(
                {
                    "job_name": job_name,
                    "case_id": case["case_id"],
                    "case_label": case["label"],
                    "scenario_id": case["scenario_id"],
                    "algorithm": case["algorithm"],
                    "selection_status": case["selection_status"],
                    "mesh_level": variant["mesh_level"],
                    "nodes": len(mesh_data["nodes"]),
                    "elements": len(mesh_data["elements"]),
                    "circumferential_divisions": variant["n_theta"],
                    "axial_subdivisions_per_segment": variant["axial_subdivisions"],
                }
            )
            verification_rows.append(
                {
                    "job_name": job_name,
                    "case_id": case["case_id"],
                    "case_label": case["label"],
                    "scenario_id": case["scenario_id"],
                    "algorithm": case["algorithm"],
                    "selection_status": case["selection_status"],
                    "mesh_level": variant["mesh_level"],
                    **geom_check,
                }
            )

    manifest = {
        "study_name": config["study_name"],
        "reference_dynamic_pressure_pa": q_ref,
        "wind_model": config["wind"],
        "material_model": config["material"],
        "shell": config["shell"],
        "job_defaults": config.get("jobs", {}),
        "convergence_criteria_percent": config.get("convergence_criteria_percent", {}),
        "presentation_mesh_level": config.get("presentation", {}).get("mesh_level", "refined"),
        "cases": manifest_cases,
    }

    with (data_dir / "job_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with (data_dir / "mesh_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mesh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mesh_rows)

    with (data_dir / "geometry_verification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verification_rows[0].keys()))
        writer.writeheader()
        writer.writerows(verification_rows)

    print(f"Built {len(manifest_cases)} Abaqus input deck(s).")


if __name__ == "__main__":
    main()
