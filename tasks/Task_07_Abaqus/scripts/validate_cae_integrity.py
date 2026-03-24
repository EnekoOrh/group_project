import argparse
import csv
import json
import os
import traceback

from abaqus import openMdb, mdb


EXPECTED_STEP_NAMES = ["Initial", "STATIC_WIND", "BUCKLING"]
EXPECTED_LOAD_NAMES = ["SelfWeight", "Wind"]
EXPECTED_BC_NAMES = ["FixBase"]


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _node_band_check(node_min, node_max):
    return node_min >= 840 and node_max <= 1100 and node_max >= 980


def main():
    parser = argparse.ArgumentParser(description="Validate Task 7 CAE integrity and model contracts.")
    parser.add_argument("--cae-path", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cases = _load_json(args.cases)
    expected_models = sorted(case["model_name"] for case in cases)
    rows = []
    failures = []

    try:
        openMdb(pathName=args.cae_path)
    except Exception as exc:
        with open(args.output_json, "w") as handle:
            json.dump(
                {
                    "status": "fail",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "cae_path": args.cae_path,
                },
                handle,
                indent=2,
            )
        raise SystemExit(1)

    model_names = sorted(mdb.models.keys())
    if model_names != expected_models:
        failures.append("model_name_set mismatch expected={0} actual={1}".format(expected_models, model_names))

    node_counts = []
    for model_name in model_names:
        model = mdb.models[model_name]
        steps = sorted(model.steps.keys())
        loads = sorted(model.loads.keys())
        bcs = sorted(model.boundaryConditions.keys())

        if "TowerShell" not in model.parts:
            failures.append("{0}: missing part TowerShell".format(model_name))
            continue

        part = model.parts["TowerShell"]
        node_count = len(part.nodes)
        element_count = len(part.elements)
        node_counts.append(node_count)

        steps_ok = steps == sorted(EXPECTED_STEP_NAMES)
        loads_ok = loads == sorted(EXPECTED_LOAD_NAMES)
        bcs_ok = bcs == sorted(EXPECTED_BC_NAMES)
        row_status = "pass" if (steps_ok and loads_ok and bcs_ok) else "fail"

        rows.append(
            {
                "model_name": model_name,
                "nodes": node_count,
                "elements": element_count,
                "steps": ";".join(steps),
                "loads": ";".join(loads),
                "bcs": ";".join(bcs),
                "status": row_status,
            }
        )

        if not steps_ok:
            failures.append("{0}: steps mismatch expected={1} actual={2}".format(model_name, EXPECTED_STEP_NAMES, steps))
        if not loads_ok:
            failures.append("{0}: loads mismatch expected={1} actual={2}".format(model_name, EXPECTED_LOAD_NAMES, loads))
        if not bcs_ok:
            failures.append("{0}: bcs mismatch expected={1} actual={2}".format(model_name, EXPECTED_BC_NAMES, bcs))

    node_min = min(node_counts) if node_counts else 0
    node_max = max(node_counts) if node_counts else 0
    if not _node_band_check(node_min, node_max):
        failures.append("node_count_band mismatch expected_min>=840 max<=1100 and max>=980 actual_min={0} actual_max={1}".format(node_min, node_max))

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "nodes", "elements", "steps", "loads", "bcs", "status"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "pass" if not failures else "fail",
        "cae_path": args.cae_path,
        "expected_model_count": len(expected_models),
        "actual_model_count": len(model_names),
        "node_count_min": node_min,
        "node_count_max": node_max,
        "failures": failures,
    }
    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    if failures:
        print("Task 7 CAE integrity validation FAILED")
        for failure in failures:
            print(" - {0}".format(failure))
        raise SystemExit(1)

    print("Validated Task 7 CAE integrity successfully.")


if __name__ == "__main__":
    main()
