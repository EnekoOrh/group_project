import argparse
import csv
import json
import math
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _interpolate_profile(z_values, radii, subdivisions_per_segment):
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


def _build_mesh(case_data, n_theta, axial_subdivisions):
    ring_z, ring_r = _interpolate_profile(case_data["z_m"], case_data["radii_m"], axial_subdivisions)
    nodes = []
    ring_node_ids = []
    node_id = 1

    for radius, z_val in zip(ring_r, ring_z):
        current_ring = []
        for theta_index in range(n_theta):
            theta = 2.0 * math.pi * theta_index / n_theta
            x_coord = radius * math.cos(theta)
            y_coord = radius * math.sin(theta)
            nodes.append((node_id, x_coord, y_coord, z_val))
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
        "sector_sets": sector_sets,
    }


def _pressure_coefficient(theta_rad):
    raw_value = 0.8 * math.cos(theta_rad)
    return max(-0.5, min(0.8, raw_value))


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
        handle.write(f"** Task 7 generated input for {case_data['case_id']}\n")
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
                cp = _pressure_coefficient(theta_center)
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

    config = _load_json(Path(args.config))
    cases = _load_json(Path(args.cases))
    results_dir = Path(args.output_dir)
    inputs_dir = results_dir / "inputs"
    data_dir = results_dir / "data"
    jobs_dir = results_dir / "jobs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    mesh_rows = []
    q_ref = None

    for case in cases:
        is_reference = case["case_id"] == config["mesh"]["reference_case_for_sensitivity"]
        case_variants = [
            {
                "suffix": "",
                "label": "comparison",
                "n_theta": int(config["mesh"]["circumferential_divisions"]),
                "axial_subdivisions": int(config["mesh"]["axial_subdivisions_per_segment"]),
            }
        ]
        if is_reference:
            case_variants.append(
                {
                    "suffix": "_refined",
                    "label": "mesh_sensitivity",
                    "n_theta": int(config["mesh"]["refined_circumferential_divisions"]),
                    "axial_subdivisions": int(config["mesh"]["refined_axial_subdivisions_per_segment"]),
                }
            )

        for variant in case_variants:
            mesh_data = _build_mesh(case, variant["n_theta"], variant["axial_subdivisions"])
            job_name = f"task7_{case['case_id']}{variant['suffix']}"
            input_path = inputs_dir / f"{job_name}.inp"
            q_ref = _write_input_file(input_path, mesh_data, config, case)
            manifest_cases.append(
                {
                    "job_name": job_name,
                    "case_id": case["case_id"],
                    "case_label": case["label"],
                    "case_variant": variant["label"],
                    "input_path": str(input_path),
                    "odb_path": str(jobs_dir / f"{job_name}.odb"),
                    "circumferential_divisions": variant["n_theta"],
                    "axial_subdivisions_per_segment": variant["axial_subdivisions"],
                    "task6_area_m2": case["task6_area_m2"],
                }
            )
            mesh_rows.append(
                {
                    "job_name": job_name,
                    "case_id": case["case_id"],
                    "case_variant": variant["label"],
                    "nodes": len(mesh_data["nodes"]),
                    "elements": len(mesh_data["elements"]),
                    "circumferential_divisions": variant["n_theta"],
                    "axial_subdivisions_per_segment": variant["axial_subdivisions"],
                }
            )

    manifest = {
        "study_name": config["study_name"],
        "reference_dynamic_pressure_pa": q_ref,
        "wind_model": config["wind"],
        "material_model": config["material"],
        "shell": config["shell"],
        "cases": manifest_cases,
    }

    with (data_dir / "job_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with (data_dir / "mesh_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mesh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mesh_rows)

    print(f"Built {len(manifest_cases)} Abaqus input deck(s).")


if __name__ == "__main__":
    main()
