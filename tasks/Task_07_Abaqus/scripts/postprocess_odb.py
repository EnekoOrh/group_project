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
        reaction = _sum_field_vectors(static_frame.fieldOutputs["RF"].values) if "RF" in static_frame.fieldOutputs.keys() else [float("nan")] * 3

        buckle_step = odb.steps["BUCKLING"]
        first_mode = buckle_step.frames[1] if len(buckle_step.frames) > 1 else buckle_step.frames[0]
        buckling_factor = _extract_eigenvalue(first_mode)

        return {
            "job_name": case["job_name"],
            "model_name": case["model_name"],
            "case_id": case["case_id"],
            "case_label": case["case_label"],
            "scenario_id": case["scenario_id"],
            "algorithm": case["algorithm"],
            "selection_status": case["selection_status"],
            "selection_basis": case["selection_basis"],
            "selection_note": case["selection_note"],
            "is_warning_case": case["is_warning_case"],
            "case_variant": case["case_variant"],
            "mesh_level": case.get("mesh_level", "coarse"),
            "task6_area_m2": float(case["task6_area_m2"]),
            "task6_penalized_objective": float(case["task6_penalized_objective"]),
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


def _rel_change(refined_value, coarse_value):
    if abs(coarse_value) < 1e-12:
        return float("nan")
    return 100.0 * (refined_value - coarse_value) / coarse_value


def _comparison_pairs(rows):
    case_map = {}
    for row in rows:
        if row["case_variant"] != "comparison":
            continue
        if row["case_id"] not in case_map:
            case_map[row["case_id"]] = {"case_label": row["case_label"]}
        case_map[row["case_id"]][row["mesh_level"]] = row
    return case_map


def _bool_str(value):
    return "yes" if value else "no"


def _write_mesh_sensitivity(path, rows, thresholds):
    pairs = _comparison_pairs(rows)
    output = []
    ordered_case_ids = sorted(pairs.keys(), key=lambda case_id: (int(case_id.split("_")[0][1:]), case_id.split("_")[1]))

    for case_id in ordered_case_ids:
        case_rows = pairs[case_id]
        coarse = case_rows.get("coarse")
        refined = case_rows.get("refined")
        if coarse is None or refined is None:
            continue

        disp_delta = _rel_change(refined["max_displacement_m"], coarse["max_displacement_m"])
        stress_delta = _rel_change(refined["max_mises_pa"], coarse["max_mises_pa"])
        buckle_delta = _rel_change(refined["buckling_factor_1"], coarse["buckling_factor_1"])

        pass_disp = abs(disp_delta) <= float(thresholds.get("max_displacement_m", 5.0))
        pass_stress = abs(stress_delta) <= float(thresholds.get("max_mises_pa", 5.0))
        pass_buckle = abs(buckle_delta) <= float(thresholds.get("buckling_factor_1", 10.0))

        output.append(
            {
                "case_id": case_id,
                "case_label": coarse["case_label"],
                "scenario_id": coarse["scenario_id"],
                "algorithm": coarse["algorithm"],
                "selection_status": coarse["selection_status"],
                "coarse_job_name": coarse["job_name"],
                "refined_job_name": refined["job_name"],
                "coarse_max_displacement_m": coarse["max_displacement_m"],
                "refined_max_displacement_m": refined["max_displacement_m"],
                "delta_max_displacement_percent": disp_delta,
                "pass_max_displacement": _bool_str(pass_disp),
                "coarse_max_mises_pa": coarse["max_mises_pa"],
                "refined_max_mises_pa": refined["max_mises_pa"],
                "delta_max_mises_percent": stress_delta,
                "pass_max_mises": _bool_str(pass_stress),
                "coarse_buckling_factor_1": coarse["buckling_factor_1"],
                "refined_buckling_factor_1": refined["buckling_factor_1"],
                "delta_buckling_factor_1_percent": buckle_delta,
                "pass_buckling_factor_1": _bool_str(pass_buckle),
                "all_criteria_pass": _bool_str(pass_disp and pass_stress and pass_buckle),
            }
        )

    if output:
        _write_csv(path, output)


def main():
    parser = argparse.ArgumentParser(description="Extract Task 7 Abaqus results from ODB files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mesh-csv", required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    rows = [_process_case(case) for case in manifest["cases"]]
    mesh_order = {"coarse": 0, "refined": 1, "confirmation": 2}
    rows.sort(key=lambda row: (int(row["scenario_id"][1:]), row["algorithm"], mesh_order.get(row["mesh_level"], 99)))
    _write_csv(args.output_csv, rows)
    _write_mesh_sensitivity(args.mesh_csv, rows, manifest.get("convergence_criteria_percent", {}))
    print(f"Postprocessed {len(rows)} ODB file(s).")


if __name__ == "__main__":
    main()
