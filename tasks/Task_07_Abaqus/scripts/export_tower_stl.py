import argparse
import csv
import json
import math
import struct
from pathlib import Path


def load_cases(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of case payloads in {path}")
    return payload


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def normalize(vector):
    length = math.sqrt(dot(vector, vector))
    if length <= 1.0e-15:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def orient_side_triangle_outward(v1, v2, v3):
    n_raw = cross(sub(v2, v1), sub(v3, v1))
    centroid = (
        (v1[0] + v2[0] + v3[0]) / 3.0,
        (v1[1] + v2[1] + v3[1]) / 3.0,
        (v1[2] + v2[2] + v3[2]) / 3.0,
    )
    radial = (centroid[0], 0.0, centroid[2])
    if dot(n_raw, radial) < 0.0:
        v2, v3 = v3, v2
        n_raw = cross(sub(v2, v1), sub(v3, v1))
    return v1, v2, v3, normalize(n_raw)


def orient_cap_triangle(v1, v2, v3, up_sign):
    n_raw = cross(sub(v2, v1), sub(v3, v1))
    target = (0.0, float(up_sign), 0.0)
    if dot(n_raw, target) < 0.0:
        v2, v3 = v3, v2
        n_raw = cross(sub(v2, v1), sub(v3, v1))
    return v1, v2, v3, normalize(n_raw)


def rings_from_profile(z_values, radii, theta_divisions):
    rings = []
    for height_y, radius in zip(z_values, radii):
        ring = []
        for k in range(theta_divisions):
            theta = 2.0 * math.pi * k / float(theta_divisions)
            ring.append((radius * math.cos(theta), height_y, radius * math.sin(theta)))
        rings.append(ring)
    return rings


def triangles_from_case(case_payload, theta_divisions, include_caps):
    z_values = [float(v) for v in case_payload["z_m"]]
    radii = [float(v) for v in case_payload["radii_m"]]
    if len(z_values) != len(radii):
        raise ValueError(
            f"Case {case_payload.get('case_id', '?')} has mismatched z/r lengths: {len(z_values)} != {len(radii)}"
        )
    if len(z_values) < 2:
        raise ValueError(f"Case {case_payload.get('case_id', '?')} has fewer than 2 rings")

    rings = rings_from_profile(z_values, radii, theta_divisions)
    triangles = []

    for ring_index in range(len(rings) - 1):
        lower = rings[ring_index]
        upper = rings[ring_index + 1]
        for k in range(theta_divisions):
            k_next = (k + 1) % theta_divisions

            v00 = lower[k]
            v01 = lower[k_next]
            v10 = upper[k]
            v11 = upper[k_next]

            triangles.append(orient_side_triangle_outward(v00, v10, v11))
            triangles.append(orient_side_triangle_outward(v00, v11, v01))

    if include_caps:
        # Bottom cap (normal toward -Y)
        bottom_ring = rings[0]
        bottom_center = (0.0, z_values[0], 0.0)
        for k in range(theta_divisions):
            k_next = (k + 1) % theta_divisions
            triangles.append(
                orient_cap_triangle(bottom_center, bottom_ring[k_next], bottom_ring[k], up_sign=-1.0)
            )

        # Top cap (normal toward +Y)
        top_ring = rings[-1]
        top_center = (0.0, z_values[-1], 0.0)
        for k in range(theta_divisions):
            k_next = (k + 1) % theta_divisions
            triangles.append(orient_cap_triangle(top_center, top_ring[k], top_ring[k_next], up_sign=1.0))

    return triangles


def write_binary_stl(path: Path, solid_name: str, oriented_triangles):
    header = f"Task7 {solid_name}".encode("ascii", errors="ignore")[:80]
    header = header + b" " * (80 - len(header))

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(oriented_triangles)))
        for v1, v2, v3, normal in oriented_triangles:
            handle.write(
                struct.pack(
                    "<12fH",
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    float(v1[0]),
                    float(v1[1]),
                    float(v1[2]),
                    float(v2[0]),
                    float(v2[1]),
                    float(v2[2]),
                    float(v3[0]),
                    float(v3[1]),
                    float(v3[2]),
                    0,
                )
            )


def write_manifest(path: Path, rows):
    fieldnames = [
        "case_id",
        "model_name",
        "scenario_id",
        "algorithm",
        "selection_status",
        "theta_divisions",
        "rings",
        "triangles",
        "min_radius_m",
        "max_radius_m",
        "height_m",
        "caps_included",
        "stl_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export Task 7 cooling-tower cases to STL files.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--cases",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "candidates" / "selected_cases.json"),
        help="Path to selected_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "models" / "stl"),
        help="Output directory for STL files",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "models" / "stl_manifest.csv"),
        help="CSV manifest path",
    )
    parser.add_argument(
        "--theta-divisions",
        type=int,
        default=180,
        help="Circumferential divisions used for STL tessellation",
    )
    parser.add_argument(
        "--include-caps",
        action="store_true",
        help="Include top and bottom caps to produce closed STL solids",
    )
    args = parser.parse_args()

    if args.theta_divisions < 12:
        raise ValueError("--theta-divisions must be >= 12")

    cases_path = Path(args.cases)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    rows = []

    for case in sorted(cases, key=lambda c: (c["scenario_id"], c["algorithm"])):
        case_id = case["case_id"]
        model_name = case["model_name"]
        stl_name = f"{case_id}.stl"
        stl_path = output_dir / stl_name

        triangles = triangles_from_case(case, args.theta_divisions, args.include_caps)
        write_binary_stl(stl_path, model_name, triangles)

        radii = [float(v) for v in case["radii_m"]]
        z_values = [float(v) for v in case["z_m"]]
        rows.append(
            {
                "case_id": case_id,
                "model_name": model_name,
                "scenario_id": case["scenario_id"],
                "algorithm": case["algorithm"],
                "selection_status": case["selection_status"],
                "theta_divisions": args.theta_divisions,
                "rings": len(z_values),
                "triangles": len(triangles),
                "min_radius_m": min(radii),
                "max_radius_m": max(radii),
                "height_m": z_values[-1] - z_values[0],
                "caps_included": bool(args.include_caps),
                "stl_path": str(stl_path),
            }
        )

    write_manifest(manifest_path, rows)
    print(f"Exported {len(rows)} STL files to {output_dir}")
    print(f"Wrote STL manifest to {manifest_path}")


if __name__ == "__main__":
    main()
