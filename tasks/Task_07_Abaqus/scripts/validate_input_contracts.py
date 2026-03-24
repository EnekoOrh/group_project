import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from task7_common import load_json, pressure_coefficient


MATERIAL_RE = re.compile(r"^\*MATERIAL\s*,\s*NAME\s*=\s*([^,]+)\s*$", re.IGNORECASE)
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
    sector_ids = set()
    grav_entries = []
    wind_entries = []

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("**"):
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

    return {
        "material_name": material_name,
        "elastic_pair": elastic_pair,
        "density_value": density_value,
        "sector_ids": sector_ids,
        "grav_entries": grav_entries,
        "wind_entries": wind_entries,
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
    gravity = config["gravity"]
    wind = config["wind"]
    q_ref = 0.5 * float(wind["air_density_kg_m3"]) * float(wind["reference_speed_m_s"]) ** 2

    n_theta = int(case["circumferential_divisions"])
    expected_sectors = set(range(1, n_theta + 1))
    expected_pressures = _expected_pressure_map(n_theta, q_ref)

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

    if parsed["sector_ids"] != expected_sectors:
        errors.append(f"wind_sector_ids={sorted(parsed['sector_ids'])} expected={list(range(1, n_theta + 1))}")

    grav_entries = parsed["grav_entries"]
    if len(grav_entries) != 2:
        errors.append(f"gravity_entry_count={len(grav_entries)} expected=2")
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

    expected_sector_ids = set(expected_pressures.keys())
    if set(by_sector.keys()) != expected_sector_ids:
        errors.append(
            "wind_dload_sectors={0} expected_nonzero={1}".format(
                sorted(by_sector.keys()),
                sorted(expected_sector_ids),
            )
        )

    for sector_id, expected_pressure in expected_pressures.items():
        observed = by_sector.get(sector_id, [])
        if len(observed) != 2:
            errors.append(f"wind_sector_{sector_id:02d}_count={len(observed)} expected=2")
            continue
        for observed_value in observed:
            if not _close(observed_value, expected_pressure, tol=1.0e-5):
                errors.append(
                    f"wind_sector_{sector_id:02d}_pressure={observed_value} expected={expected_pressure}"
                )

    return {
        "job_name": case["job_name"],
        "case_id": case["case_id"],
        "mesh_level": case["mesh_level"],
        "input_path": str(input_path),
        "material_ok": "yes" if not any("material" in err or "youngs_modulus" in err or "poisson_ratio" in err or "density" in err for err in errors) else "no",
        "gravity_ok": "yes" if not any(err.startswith("gravity") for err in errors) else "no",
        "wind_ok": "yes" if not any(err.startswith("wind_") for err in errors) else "no",
        "status": "pass" if not errors else "fail",
        "details": "; ".join(errors),
    }, errors


def main():
    parser = argparse.ArgumentParser(description="Validate Task 7 Abaqus input deck material/load contracts.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--manifest", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "job_manifest.json"))
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--output-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "input_contract_audit.csv"))
    args = parser.parse_args()

    config = load_json(Path(args.config))
    manifest = load_json(Path(args.manifest))

    rows = []
    failures = []
    for case in manifest["cases"]:
        row, errors = _validate_case(case, config, repo_root)
        rows.append(row)
        if errors:
            failures.append((case["job_name"], errors))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_name", "case_id", "mesh_level", "input_path", "material_ok", "gravity_ok", "wind_ok", "status", "details"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        print("Task 7 input contract validation FAILED:")
        for job_name, errors in failures:
            print(f"  - {job_name}:")
            for error in errors:
                print(f"      {error}")
        raise SystemExit(1)

    print(f"Validated {len(rows)} Task 7 input deck(s) successfully.")


if __name__ == "__main__":
    main()
