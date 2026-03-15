import argparse
import csv
import json
from collections import defaultdict

from odbAccess import openOdb


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _node_coordinates(instance):
    return {int(node.label): tuple(float(value) for value in node.coordinates) for node in instance.nodes}


def _element_connectivity(instance):
    return {int(element.label): [int(node_id) for node_id in element.connectivity] for element in instance.elements}


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_vector(data):
    return [float(data[i]) if i < len(data) else 0.0 for i in range(3)]


def _export_case(case, output_dir):
    odb = openOdb(path=case["odb_path"], readOnly=True)
    try:
        instance_name = list(odb.rootAssembly.instances.keys())[0]
        instance = odb.rootAssembly.instances[instance_name]
        node_xyz = _node_coordinates(instance)
        element_conn = _element_connectivity(instance)

        mesh_payload = {
            "job_name": case["job_name"],
            "case_id": case["case_id"],
            "case_label": case["case_label"],
            "case_variant": case["case_variant"],
            "instance_name": instance_name,
            "nodes": [{"label": label, "x": coords[0], "y": coords[1], "z": coords[2]} for label, coords in sorted(node_xyz.items())],
            "elements": [
                {"label": label, "connectivity": connectivity}
                for label, connectivity in sorted(element_conn.items())
            ],
        }
        with open(output_dir + "/" + case["job_name"] + "_mesh.json", "w", encoding="utf-8") as handle:
            json.dump(mesh_payload, handle, indent=2)

        static_frame = odb.steps["STATIC_WIND"].frames[-1]
        u_field = static_frame.fieldOutputs["U"]
        nodal_rows = []
        nodal_umag = {}
        for value in u_field.values:
            label = int(value.nodeLabel)
            x0, y0, z0 = node_xyz[label]
            ux, uy, uz = _safe_vector(value.data)
            umag = float(value.magnitude)
            nodal_umag[label] = umag
            nodal_rows.append(
                {
                    "node_label": label,
                    "x": x0,
                    "y": y0,
                    "z": z0,
                    "ux": ux,
                    "uy": uy,
                    "uz": uz,
                    "umag": umag,
                    "deformed_x": x0 + ux,
                    "deformed_y": y0 + uy,
                    "deformed_z": z0 + uz,
                }
            )
        _write_csv(
            output_dir + "/" + case["job_name"] + "_static_nodal.csv",
            ["node_label", "x", "y", "z", "ux", "uy", "uz", "umag", "deformed_x", "deformed_y", "deformed_z"],
            nodal_rows,
        )

        s_field = static_frame.fieldOutputs["S"]
        stress_accumulator = defaultdict(list)
        for value in s_field.values:
            stress_accumulator[int(value.elementLabel)].append(float(value.mises))
        stress_rows = []
        for element_label, connectivity in sorted(element_conn.items()):
            coords = [node_xyz[node_id] for node_id in connectivity]
            cx = sum(point[0] for point in coords) / float(len(coords))
            cy = sum(point[1] for point in coords) / float(len(coords))
            cz = sum(point[2] for point in coords) / float(len(coords))
            mises_values = stress_accumulator[element_label]
            stress_rows.append(
                {
                    "element_label": element_label,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "centroid_z": cz,
                    "mises_pa_avg": sum(mises_values) / float(len(mises_values)) if mises_values else 0.0,
                    "mises_pa_max": max(mises_values) if mises_values else 0.0,
                }
            )
        _write_csv(
            output_dir + "/" + case["job_name"] + "_static_element.csv",
            ["element_label", "centroid_x", "centroid_y", "centroid_z", "mises_pa_avg", "mises_pa_max"],
            stress_rows,
        )

        buckle_step = odb.steps["BUCKLING"]
        buckle_frame = buckle_step.frames[1] if len(buckle_step.frames) > 1 else buckle_step.frames[0]
        buckle_u = buckle_frame.fieldOutputs["U"]
        mode_vectors = {}
        mode_magnitudes = []
        for value in buckle_u.values:
            label = int(value.nodeLabel)
            ux, uy, uz = _safe_vector(value.data)
            umag = float(value.magnitude)
            mode_vectors[label] = (ux, uy, uz, umag)
            mode_magnitudes.append(umag)

        z_values = [coords[2] for coords in node_xyz.values()]
        tower_height = max(z_values) - min(z_values)
        max_mode = max(mode_magnitudes) if mode_magnitudes else 1.0
        display_scale = 0.10 * tower_height / max(max_mode, 1e-12)

        buckle_rows = []
        for label, (x0, y0, z0) in sorted(node_xyz.items()):
            ux, uy, uz, umag = mode_vectors[label]
            buckle_rows.append(
                {
                    "node_label": label,
                    "x": x0,
                    "y": y0,
                    "z": z0,
                    "ux": ux,
                    "uy": uy,
                    "uz": uz,
                    "umag": umag,
                    "scale_factor": display_scale,
                    "scaled_x": x0 + display_scale * ux,
                    "scaled_y": y0 + display_scale * uy,
                    "scaled_z": z0 + display_scale * uz,
                }
            )
        _write_csv(
            output_dir + "/" + case["job_name"] + "_buckling_mode1_nodal.csv",
            ["node_label", "x", "y", "z", "ux", "uy", "uz", "umag", "scale_factor", "scaled_x", "scaled_y", "scaled_z"],
            buckle_rows,
        )

        return {
            "job_name": case["job_name"],
            "case_id": case["case_id"],
            "case_variant": case["case_variant"],
            "node_count": len(node_xyz),
            "element_count": len(element_conn),
            "buckling_scale_factor": display_scale,
        }
    finally:
        odb.close()


def main():
    parser = argparse.ArgumentParser(description="Export structured Task 7 field data from Abaqus ODB files.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir
    export_summary = []

    for case in manifest["cases"]:
        export_summary.append(_export_case(case, output_dir))

    with open(output_dir + "/field_export_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"cases": export_summary}, handle, indent=2)

    print("Exported Task 7 ODB field data.")


if __name__ == "__main__":
    main()
