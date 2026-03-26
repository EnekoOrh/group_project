import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from task7_common import load_json, pressure_coefficient


MATERIAL_RE = re.compile(r"^\*MATERIAL\s*,\s*NAME\s*=\s*([^,]+)\s*$", re.IGNORECASE)
SHELL_SECTION_RE = re.compile(r"^\*SHELL SECTION\b", re.IGNORECASE)
ELSET_WIND_RE = re.compile(r"^\*ELSET\s*,\s*ELSET\s*=\s*WIND_SEC_(\d+)\s*$", re.IGNORECASE)
GRAV_RE = re.compile(
    r"^EALL\s*,\s*GRAV\s*,\s*([+\-0-9.eE]+)\s*,\s*([+\-0-9.eE]+)\s*,\s*([+\-0-9.eE]+)\s*,\s*([+\-0-9.eE]+)\s*$",
    re.IGNORECASE,
)
WIND_PRESSURE_RE = re.compile(r"^WIND_SEC_(\d+)\s*,\s*P\s*,\s*([+\-0-9.eE]+)\s*$", re.IGNORECASE)


def _resolve(path_text, repo_root):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _next_data_line(lines, start_index):
    for idx in range(start_index, len(lines)):
        value = lines[idx].strip()
        if not value or value.startswith("**"):
            continue
        if value.startswith("*"):
            continue
        return value
    return None


def _parse_pair(line):
    if line is None:
        return None
    parts = [part.strip() for part in line.split(",") if part.strip()]
    if len(parts) < 2:
        return None
    return float(parts[0]), float(parts[1])


def _parse_single(line):
    if line is None:
        return None
    parts = [part.strip() for part in line.split(",") if part.strip()]
    if not parts:
        return None
    return float(parts[0])


def _close(value, expected, tol=1.0e-6):
    return abs(float(value) - float(expected)) <= tol


def _parse_input_contract(input_path):
    lines = input_path.read_text(encoding="utf-8").splitlines()
    material_name = None
    elastic_pair = None
    density_value = None
    shell_thickness_value = None
    sector_ids = set()
    grav_entries = []
    wind_entries = []
    node_coords = []
    in_node_block = False

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue

        if line.startswith("*"):
            in_node_block = line.upper().startswith("*NODE")

        if in_node_block and not line.startswith("*"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                node_coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue

        material_match = MATERIAL_RE.match(line)
        if material_match:
            material_name = material_match.group(1).strip()
            continue

        if line.upper() == "*ELASTIC":
            elastic_pair = _parse_pair(_next_data_line(lines, idx + 1))
            continue

        if line.upper() == "*DENSITY":
            density_value = _parse_single(_next_data_line(lines, idx + 1))
            continue

        if SHELL_SECTION_RE.match(line):
            shell_thickness_value = _parse_single(_next_data_line(lines, idx + 1))
            continue

        wind_elset_match = ELSET_WIND_RE.match(line)
        if wind_elset_match:
            sector_ids.add(int(wind_elset_match.group(1)))
            continue

        grav_match = GRAV_RE.match(line)
        if grav_match:
            grav_entries.append(
                (
                    float(grav_match.group(1)),
                    float(grav_match.group(2)),
                    float(grav_match.group(3)),
                    float(grav_match.group(4)),
                )
            )
            continue

        wind_match = WIND_PRESSURE_RE.match(line)
        if wind_match:
            wind_entries.append((int(wind_match.group(1)), float(wind_match.group(2))))

    bounds = None
    if node_coords:
        xs = [value[0] for value in node_coords]
        ys = [value[1] for value in node_coords]
        zs = [value[2] for value in node_coords]
        bounds = {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "z_min": min(zs),
            "z_max": max(zs),
            "max_abs_coord": max(abs(value) for triplet in node_coords for value in triplet),
        }

    return {
        "material_name": material_name,
        "elastic_pair": elastic_pair,
        "density_value": density_value,
        "shell_thickness_value": shell_thickness_value,
        "sector_ids": sector_ids,
        "grav_entries": grav_entries,
        "wind_entries": wind_entries,
        "node_bounds": bounds,
    }


def _expected_pressure_map(n_theta, q_ref):
    expected = {}
    for sector_index in range(n_theta):
        theta_center = 2.0 * math.pi * (sector_index + 0.5) / n_theta
        cp = pressure_coefficient(theta_center)
        pressure_value = -q_ref * cp
        if abs(pressure_value) >= 1.0e-9:
            expected[sector_index + 1] = pressure_value
    return expected


def _validate_case(case, config, repo_root):
    input_path = _resolve(case["input_path"], repo_root)
    parsed = _parse_input_contract(input_path)
    material = config["material"]
    shell = config["shell"]
    gravity = config["gravity"]
    wind = config["wind"]
    q_ref = 0.5 * float(wind["air_density_kg_m3"]) * float(wind["reference_speed_m_s"]) ** 2

    n_theta = int(case["circumferential_divisions"])
    expected_sectors = set(range(1, n_theta + 1))
    expected_pressures = _expected_pressure_map(n_theta, q_ref)
    step_count = 2 if case.get("has_buckling_step", True) else 1
    load_case = str(case.get("load_case", "combined"))
    include_gravity = load_case in ("combined", "gravity_only")
    include_wind = load_case in ("combined", "wind_only")
    expected_gravity_entry_count = step_count if include_gravity else 0
    expected_wind_entry_count = step_count if include_wind else 0
    expected_wind_sector_ids = set(expected_pressures.keys()) if include_wind else set()

    errors = []

    if parsed["material_name"] != material["name"]:
        errors.append(f"material_name={parsed['material_name']} expected={material['name']}")

    elastic = parsed["elastic_pair"]
    if not elastic:
        errors.append("missing_elastic_block")
    else:
        if not _close(elastic[0], float(material["youngs_modulus_pa"])):
            errors.append(f"youngs_modulus={elastic[0]} expected={material['youngs_modulus_pa']}")
        if not _close(elastic[1], float(material["poisson_ratio"])):
            errors.append(f"poisson_ratio={elastic[1]} expected={material['poisson_ratio']}")

    density = parsed["density_value"]
    if density is None:
        errors.append("missing_density_block")
    elif not _close(density, float(material["density_kg_m3"])):
        errors.append(f"density={density} expected={material['density_kg_m3']}")

    shell_thickness = parsed["shell_thickness_value"]
    if shell_thickness is None:
        errors.append("missing_shell_section")
    elif not _close(shell_thickness, float(shell["thickness_m"])):
        errors.append(f"shell_thickness={shell_thickness} expected={shell['thickness_m']}")

    if parsed["sector_ids"] != expected_sectors:
        errors.append(f"wind_sector_ids={sorted(parsed['sector_ids'])} expected={list(range(1, n_theta + 1))}")

    grav_entries = parsed["grav_entries"]
    if len(grav_entries) != expected_gravity_entry_count:
        errors.append(f"gravity_entry_count={len(grav_entries)} expected={expected_gravity_entry_count}")
    expected_grav = (
        float(gravity["acceleration_m_s2"]),
        float(gravity["direction"][0]),
        float(gravity["direction"][1]),
        float(gravity["direction"][2]),
    )
    for idx, entry in enumerate(grav_entries):
        if not all(_close(entry[i], expected_grav[i]) for i in range(4)):
            errors.append(f"gravity_entry_{idx}={entry} expected={expected_grav}")

    wind_entries = parsed["wind_entries"]
    by_sector = defaultdict(list)
    for sector_id, pressure_value in wind_entries:
        by_sector[int(sector_id)].append(float(pressure_value))

    if set(by_sector.keys()) != expected_wind_sector_ids:
        errors.append(
            "wind_dload_sectors={0} expected_nonzero={1}".format(
                sorted(by_sector.keys()),
                sorted(expected_wind_sector_ids),
            )
        )

    for sector_id, expected_pressure in expected_pressures.items():
        observed = by_sector.get(sector_id, [])
        if len(observed) != expected_wind_entry_count:
            errors.append(f"wind_sector_{sector_id:02d}_count={len(observed)} expected={expected_wind_entry_count}")
            continue
        if not include_wind:
            continue
        for observed_value in observed:
            if not _close(observed_value, expected_pressure, tol=1.0e-5):
                errors.append(f"wind_sector_{sector_id:02d}_pressure={observed_value} expected={expected_pressure}")

    bounds = parsed["node_bounds"] or {}
    max_abs_coord = float(bounds.get("max_abs_coord", float("nan")))
    geometry_scale_m_ok = math.isfinite(max_abs_coord) and 1.0 <= max_abs_coord <= 500.0

    unit_checks = {
        "unit_declared_si": str(config.get("units", "")).upper() == "SI",
        "geometry_scale_m_ok": geometry_scale_m_ok,
        "shell_thickness_m_ok": shell_thickness is not None and 0.05 <= float(shell_thickness) <= 2.0,
        "elastic_pa_ok": elastic is not None and 1.0e9 <= float(elastic[0]) <= 1.0e12,
        "poisson_ratio_ok": elastic is not None and 0.0 < float(elastic[1]) < 0.5,
        "density_kg_m3_ok": density is not None and 500.0 <= float(density) <= 6000.0,
        "gravity_m_s2_ok": _close(float(gravity["acceleration_m_s2"]), 9.81, tol=1.0e-3),
        "gravity_direction_ok": all(
            _close(float(value), expected, tol=1.0e-6)
            for value, expected in zip(gravity["direction"], [0.0, -1.0, 0.0])
        ),
        "wind_speed_m_s_ok": 1.0 <= float(wind["reference_speed_m_s"]) <= 100.0,
        "wind_pressure_basis_ok": q_ref > 0.0 and bool(expected_pressures),
    }
    unit_status = "pass" if all(unit_checks.values()) else "fail"

    contract_row = {
        "job_name": case["job_name"],
        "case_id": case["case_id"],
        "mesh_level": case["mesh_level"],
        "case_variant": case.get("case_variant", "comparison"),
        "load_case": case.get("load_case", "combined"),
        "input_path": str(input_path),
        "material_ok": "yes" if not any("material" in err or "youngs_modulus" in err or "poisson_ratio" in err or "density" in err for err in errors) else "no",
        "gravity_ok": "yes" if not any(err.startswith("gravity") for err in errors) else "no",
        "wind_ok": "yes" if not any(err.startswith("wind_") for err in errors) else "no",
        "status": "pass" if not errors else "fail",
        "details": "; ".join(errors),
    }

    unit_row = {
        "job_name": case["job_name"],
        "case_id": case["case_id"],
        "mesh_level": case["mesh_level"],
        "case_variant": case.get("case_variant", "comparison"),
        "load_case": case.get("load_case", "combined"),
        "unit_system": config.get("units", ""),
        "q_ref_pa": f"{q_ref:.6f}",
        "max_abs_coord_m": f"{max_abs_coord:.6f}" if math.isfinite(max_abs_coord) else "nan",
        "shell_thickness_m": f"{shell_thickness:.6f}" if shell_thickness is not None else "",
        "youngs_modulus_pa": f"{elastic[0]:.6f}" if elastic else "",
        "poisson_ratio": f"{elastic[1]:.6f}" if elastic else "",
        "density_kg_m3": f"{density:.6f}" if density is not None else "",
        "gravity_acceleration_m_s2": f"{gravity['acceleration_m_s2']:.6f}",
        "gravity_direction": ",".join(f"{float(value):.6f}" for value in gravity["direction"]),
        "wind_speed_m_s": f"{float(wind['reference_speed_m_s']):.6f}",
        "wind_direction_axis": str(wind.get("wind_direction_axis", "")),
        "unit_declared_si_ok": "yes" if unit_checks["unit_declared_si"] else "no",
        "geometry_scale_m_ok": "yes" if unit_checks["geometry_scale_m_ok"] else "no",
        "shell_thickness_m_ok": "yes" if unit_checks["shell_thickness_m_ok"] else "no",
        "elastic_pa_ok": "yes" if unit_checks["elastic_pa_ok"] else "no",
        "poisson_ratio_ok": "yes" if unit_checks["poisson_ratio_ok"] else "no",
        "density_kg_m3_ok": "yes" if unit_checks["density_kg_m3_ok"] else "no",
        "gravity_m_s2_ok": "yes" if unit_checks["gravity_m_s2_ok"] else "no",
        "gravity_direction_ok": "yes" if unit_checks["gravity_direction_ok"] else "no",
        "wind_speed_m_s_ok": "yes" if unit_checks["wind_speed_m_s_ok"] else "no",
        "wind_pressure_basis_ok": "yes" if unit_checks["wind_pressure_basis_ok"] else "no",
        "status": unit_status,
    }

    return contract_row, unit_row, errors


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate Task 7 Abaqus input deck material/load contracts.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--manifest", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "job_manifest.json"))
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--output-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "input_contract_audit.csv"))
    parser.add_argument(
        "--unit-output-csv",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "unit_load_contract_audit.csv"),
    )
    args = parser.parse_args()

    config = load_json(Path(args.config))
    manifest = load_json(Path(args.manifest))

    contract_rows = []
    unit_rows = []
    failures = []
    unit_failures = []
    for case in manifest["cases"]:
        contract_row, unit_row, errors = _validate_case(case, config, repo_root)
        contract_rows.append(contract_row)
        unit_rows.append(unit_row)
        if errors:
            failures.append((case["job_name"], errors))
        if unit_row["status"] != "pass":
            unit_failures.append(case["job_name"])

    _write_csv(
        Path(args.output_csv),
        [
            "job_name",
            "case_id",
            "mesh_level",
            "case_variant",
            "load_case",
            "input_path",
            "material_ok",
            "gravity_ok",
            "wind_ok",
            "status",
            "details",
        ],
        contract_rows,
    )

    _write_csv(
        Path(args.unit_output_csv),
        [
            "job_name",
            "case_id",
            "mesh_level",
            "case_variant",
            "load_case",
            "unit_system",
            "q_ref_pa",
            "max_abs_coord_m",
            "shell_thickness_m",
            "youngs_modulus_pa",
            "poisson_ratio",
            "density_kg_m3",
            "gravity_acceleration_m_s2",
            "gravity_direction",
            "wind_speed_m_s",
            "wind_direction_axis",
            "unit_declared_si_ok",
            "geometry_scale_m_ok",
            "shell_thickness_m_ok",
            "elastic_pa_ok",
            "poisson_ratio_ok",
            "density_kg_m3_ok",
            "gravity_m_s2_ok",
            "gravity_direction_ok",
            "wind_speed_m_s_ok",
            "wind_pressure_basis_ok",
            "status",
        ],
        unit_rows,
    )

    if failures or unit_failures:
        print("Task 7 input contract validation FAILED:")
        for job_name, errors in failures:
            print(f"  - {job_name}:")
            for error in errors:
                print(f"      {error}")
        if unit_failures:
            print("Unit/load chain checks failed for: " + ", ".join(unit_failures[:10]))
        raise SystemExit(1)

    print(f"Validated {len(contract_rows)} Task 7 input deck(s) successfully.")


if __name__ == "__main__":
    main()
