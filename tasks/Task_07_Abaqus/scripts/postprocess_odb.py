import argparse
import csv
import json
import math
import re

from odbAccess import openOdb


EIGENVALUE_RE = re.compile(r"EigenValue\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_eigenvalue(frame):
    match = EIGENVALUE_RE.search(frame.description)
    if match:
        return float(match.group(1))
    return float("nan")


def _sum_field_vectors(field_values):
    total = [0.0, 0.0, 0.0]
    for value in field_values:
        data = value.data
        for index in range(min(3, len(data))):
            total[index] += float(data[index])
    return total


def _process_case(case):
    odb = openOdb(path=case["odb_path"], readOnly=True)
    try:
        static_step = odb.steps["STATIC_WIND"]
        static_frame = static_step.frames[-1]
        u_field = static_frame.fieldOutputs["U"]
        s_field = static_frame.fieldOutputs["S"]

        max_disp = max(float(value.magnitude) for value in u_field.values)
        max_mises = max(float(value.mises) for value in s_field.values)

        if "RF" in static_frame.fieldOutputs.keys():
            rf_field = static_frame.fieldOutputs["RF"]
        else:
            rf_field = None
        if rf_field is None:
            reaction = [float("nan"), float("nan"), float("nan")]
        else:
            reaction = _sum_field_vectors(rf_field.values)

        buckle_step = odb.steps["BUCKLING"]
        first_mode = buckle_step.frames[1] if len(buckle_step.frames) > 1 else buckle_step.frames[0]
        buckling_factor = _extract_eigenvalue(first_mode)

        return {
            "job_name": case["job_name"],
            "case_id": case["case_id"],
            "case_label": case["case_label"],
            "case_variant": case["case_variant"],
            "task6_area_m2": float(case["task6_area_m2"]),
            "max_displacement_m": max_disp,
            "max_mises_pa": max_mises,
            "base_reaction_fx_n": reaction[0],
            "base_reaction_fy_n": reaction[1],
            "base_reaction_fz_n": reaction[2],
            "base_reaction_resultant_n": math.sqrt(reaction[0] ** 2 + reaction[1] ** 2 + reaction[2] ** 2),
            "buckling_factor_1": buckling_factor,
        }
    finally:
        odb.close()


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_mesh_sensitivity(path, rows):
    baseline = None
    refined = None
    for row in rows:
        if row["case_id"] != "joint_reference":
            continue
        if row["case_variant"] == "comparison":
            baseline = row
        elif row["case_variant"] == "mesh_sensitivity":
            refined = row

    if baseline is None or refined is None:
        return

    def rel_change(refined_value, base_value):
        if abs(base_value) < 1e-12:
            return float("nan")
        return 100.0 * (refined_value - base_value) / base_value

    output = [
        {
            "metric": "max_displacement_m",
            "baseline": baseline["max_displacement_m"],
            "refined": refined["max_displacement_m"],
            "relative_change_percent": rel_change(refined["max_displacement_m"], baseline["max_displacement_m"]),
        },
        {
            "metric": "max_mises_pa",
            "baseline": baseline["max_mises_pa"],
            "refined": refined["max_mises_pa"],
            "relative_change_percent": rel_change(refined["max_mises_pa"], baseline["max_mises_pa"]),
        },
        {
            "metric": "buckling_factor_1",
            "baseline": baseline["buckling_factor_1"],
            "refined": refined["buckling_factor_1"],
            "relative_change_percent": rel_change(refined["buckling_factor_1"], baseline["buckling_factor_1"]),
        },
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)


def main():
    parser = argparse.ArgumentParser(description="Extract Task 7 Abaqus results from ODB files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mesh-csv", required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    rows = []
    for case in manifest["cases"]:
        rows.append(_process_case(case))

    _write_csv(args.output_csv, rows)
    _write_mesh_sensitivity(args.mesh_csv, rows)
    print(f"Postprocessed {len(rows)} ODB file(s).")


if __name__ == "__main__":
    main()
