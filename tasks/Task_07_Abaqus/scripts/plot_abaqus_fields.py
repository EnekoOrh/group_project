import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _readable_case_name(case_id):
    mapping = {
        "radius_baseline": "Radius baseline (S2 / BFGS)",
        "height_variant": "Height variant (S6 / BFGS)",
        "joint_reference": "Joint reference (S7 / BFGS)",
    }
    return mapping.get(case_id, case_id)


def _build_face_polygons(mesh, node_map, coord_keys):
    polygons = []
    element_labels = []
    for element in mesh["elements"]:
        coords = []
        for node_id in element["connectivity"]:
            node = node_map[node_id]
            coords.append([node[key] for key in coord_keys])
        polygons.append(coords)
        element_labels.append(element["label"])
    return polygons, element_labels


def _set_equal_3d(ax, points):
    arr = np.asarray(points, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _base_axes_style(ax):
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _plot_surface(polygons, scalar_values, cmap_name, title, subtitle, output_path: Path):
    scalar_array = np.asarray(scalar_values, dtype=float)
    norm = colors.Normalize(vmin=float(np.min(scalar_array)), vmax=float(np.max(scalar_array)))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    facecolors = cmap(norm(scalar_array))

    fig = plt.figure(figsize=(8.2, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(polygons, facecolors=facecolors, linewidths=0.15, edgecolors=(0.1, 0.1, 0.1, 0.18))
    ax.add_collection3d(poly)
    flat_points = [vertex for polygon in polygons for vertex in polygon]
    _set_equal_3d(ax, flat_points)
    _base_axes_style(ax)
    fig.suptitle(title, fontsize=15, y=0.96)
    fig.text(0.5, 0.03, subtitle, ha="center", va="center", fontsize=10)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.66, pad=0.03)
    fig.tight_layout(rect=[0, 0.05, 0.93, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_case(case, summary_row, mesh_summary_map, fields_dir: Path, output_dir: Path):
    job_name = case["job_name"]
    mesh = _load_json(fields_dir / f"{job_name}_mesh.json")
    static_nodal = _load_csv(fields_dir / f"{job_name}_static_nodal.csv")
    static_element = _load_csv(fields_dir / f"{job_name}_static_element.csv")
    buckle_nodal = _load_csv(fields_dir / f"{job_name}_buckling_mode1_nodal.csv")

    expected = mesh_summary_map[job_name]
    if len(mesh["nodes"]) != int(expected["nodes"]) or len(mesh["elements"]) != int(expected["elements"]):
        raise ValueError(f"Field export count mismatch for {job_name}")

    original_node_map = {
        int(row["node_label"]): {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
        }
        for row in static_nodal
    }
    deformed_node_map = {
        int(row["node_label"]): {
            "x": float(row["deformed_x"]),
            "y": float(row["deformed_y"]),
            "z": float(row["deformed_z"]),
            "umag": float(row["umag"]),
        }
        for row in static_nodal
    }
    buckle_node_map = {
        int(row["node_label"]): {
            "x": float(row["scaled_x"]),
            "y": float(row["scaled_y"]),
            "z": float(row["scaled_z"]),
            "umag": float(row["umag"]),
            "scale_factor": float(row["scale_factor"]),
        }
        for row in buckle_nodal
    }
    element_stress = {int(row["element_label"]): float(row["mises_pa_avg"]) / 1e6 for row in static_element}

    original_polygons, element_labels = _build_face_polygons(mesh, original_node_map, ["x", "y", "z"])
    deformed_polygons, _ = _build_face_polygons(mesh, deformed_node_map, ["x", "y", "z"])
    buckle_polygons, _ = _build_face_polygons(mesh, buckle_node_map, ["x", "y", "z"])

    stress_values = [element_stress[label] for label in element_labels]
    disp_values = [
        float(np.mean([deformed_node_map[node_id]["umag"] for node_id in element["connectivity"]])) * 1000.0
        for element in mesh["elements"]
    ]
    buckle_values = [
        float(np.mean([buckle_node_map[node_id]["umag"] for node_id in element["connectivity"]]))
        for element in mesh["elements"]
    ]

    case_name = _readable_case_name(case["case_id"])
    _plot_surface(
        original_polygons,
        stress_values,
        "inferno",
        f"{case_name} - Static stress field",
        "Actual Abaqus element Mises stress on undeformed shell (MPa). Max Mises = {0:.3f} MPa".format(
            float(summary_row["max_mises_pa"]) / 1e6
        ),
        output_dir / f"{job_name}_stress.png",
    )
    _plot_surface(
        deformed_polygons,
        disp_values,
        "viridis",
        f"{case_name} - Static displacement field",
        "Deformed shell from actual Abaqus nodal displacement magnitude (mm). Max displacement = {0:.3f} mm".format(
            1000.0 * float(summary_row["max_displacement_m"])
        ),
        output_dir / f"{job_name}_displacement.png",
    )
    _plot_surface(
        buckle_polygons,
        buckle_values,
        "cividis",
        f"{case_name} - Buckling mode 1",
        "Mode shape from actual Abaqus eigenvector field with explicit visualization scaling. Buckling factor = {0:.3f}".format(
            float(summary_row["buckling_factor_1"])
        ),
        output_dir / f"{job_name}_buckling_mode1.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Render Task 7 custom plots from exported Abaqus field data.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--manifest", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "job_manifest.json"))
    parser.add_argument("--mesh-summary", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_summary.csv"))
    parser.add_argument("--summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv"))
    parser.add_argument("--fields-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "fields"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "figures"))
    args = parser.parse_args()

    manifest = _load_json(Path(args.manifest))
    mesh_summary = _load_csv(Path(args.mesh_summary))
    summary_rows = _load_csv(Path(args.summary_csv))
    fields_dir = Path(args.fields_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_summary_map = {row["job_name"]: row for row in mesh_summary}
    summary_map = {row["job_name"]: row for row in summary_rows}

    for case in manifest["cases"]:
        if case["case_variant"] != "comparison":
            continue
        _plot_case(case, summary_map[case["job_name"]], mesh_summary_map, fields_dir, output_dir)

    print("Rendered Task 7 custom plots from exported field data.")


if __name__ == "__main__":
    main()
