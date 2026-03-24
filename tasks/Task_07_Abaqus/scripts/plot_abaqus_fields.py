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

from task7_common import STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING, status_rank


STATUS_COLORS = {
    STATUS_ENGINEERING: "#117A65",
    STATUS_MATH_FALLBACK: "#B9770E",
    STATUS_WARNING: "#922B21",
}
STATUS_LABELS = {
    STATUS_ENGINEERING: "engineering-feasible",
    STATUS_MATH_FALLBACK: "math fallback",
    STATUS_WARNING: "warning",
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _case_rows(summary_rows):
    grouped = {}
    for row in summary_rows:
        if row["case_variant"] != "comparison":
            continue
        grouped.setdefault(row["case_id"], {})[row["mesh_level"]] = row
    return grouped


def _preferred_row(case_id, row_map):
    case_map = row_map[case_id]
    return case_map.get("refined", case_map.get("coarse"))


def _rank_rows(rows):
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            status_rank(row),
            -float(row["buckling_factor_1"]),
            float(row["max_displacement_m"]),
            float(row["max_mises_pa"]),
            float(row["task6_area_m2"]),
        )
    )
    return ranked


def _display_label(row):
    return f"{row['scenario_id']}/{row['algorithm']}"


def _comparison_plot(refined_rows, output_path: Path):
    x = np.arange(len(refined_rows))
    labels = [_display_label(row) for row in refined_rows]
    metrics = [
        ("max_displacement_m", "Max displacement", 1000.0, "mm"),
        ("max_mises_pa", "Max von Mises stress", 1.0e-6, "MPa"),
        ("buckling_factor_1", "First buckling factor", 1.0, "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for axis, (key, title, scale, unit) in zip(axes, metrics):
        values = [scale * float(row[key]) for row in refined_rows]
        colors_list = [STATUS_COLORS[row["selection_status"]] for row in refined_rows]
        bars = axis.bar(x, values, color=colors_list)
        for bar, row in zip(bars, refined_rows):
            if row["selection_status"] != STATUS_ENGINEERING:
                bar.set_hatch("//")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=75, ha="right")
        axis.grid(axis="y", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[key], label=STATUS_LABELS[key]) for key in [STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING]]
    axes[0].legend(handles=handles, loc="upper left")
    fig.suptitle("Task 7 refined structural comparison across all 24 scenario/optimizer cases")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _warning_plot(refined_rows, output_path: Path):
    warning_rows = [row for row in refined_rows if row["scenario_id"] == "S8"]
    if not warning_rows:
        return

    labels = [_display_label(row) for row in warning_rows]
    x = np.arange(len(warning_rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    disp_vals = [1000.0 * float(row["max_displacement_m"]) for row in warning_rows]
    stress_vals = [float(row["max_mises_pa"]) / 1.0e6 for row in warning_rows]
    buckling_vals = [float(row["buckling_factor_1"]) for row in warning_rows]
    ax.bar(x - width, disp_vals, width=width, label="Displacement (mm)", color="#5DADE2")
    ax.bar(x, stress_vals, width=width, label="Stress (MPa)", color="#F5B041")
    ax.bar(x + width, buckling_vals, width=width, label="Buckling factor", color="#CD6155")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Scenario S8 warning-only structural comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _winner_mesh_comparison(row_map, winner_row, output_path: Path):
    case_id = winner_row["case_id"]
    case_rows = row_map[case_id]
    coarse = case_rows.get("coarse")
    refined = case_rows.get("refined")
    if coarse is None or refined is None:
        return

    metrics = [
        ("max_displacement_m", "Max displacement", 1000.0, "mm"),
        ("max_mises_pa", "Max von Mises stress", 1.0e-6, "MPa"),
        ("buckling_factor_1", "First buckling factor", 1.0, "-"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    for axis, (key, title, scale, unit) in zip(axes, metrics):
        values = [scale * float(coarse[key]), scale * float(refined[key])]
        axis.bar(["Coarse", "Refined"], values, color=["#6C7A89", "#117A65"])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle(f"{winner_row['case_label']} - coarse vs refined mesh comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_case(selected_row, mesh_summary_map, fields_dir: Path, output_dir: Path):
    job_name = selected_row["job_name"]
    mesh = _load_json(fields_dir / f"{job_name}_mesh.json")
    static_nodal = _load_csv(fields_dir / f"{job_name}_static_nodal.csv")
    static_element = _load_csv(fields_dir / f"{job_name}_static_element.csv")
    buckle_nodal = _load_csv(fields_dir / f"{job_name}_buckling_mode1_nodal.csv")

    expected = mesh_summary_map[job_name]
    if len(mesh["nodes"]) != int(expected["nodes"]) or len(mesh["elements"]) != int(expected["elements"]):
        raise ValueError(f"Field export count mismatch for {job_name}")

    original_node_map = {int(row["node_label"]): {"x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"])} for row in static_nodal}
    deformed_node_map = {
        int(row["node_label"]): {"x": float(row["deformed_x"]), "y": float(row["deformed_y"]), "z": float(row["deformed_z"]), "umag": float(row["umag"])}
        for row in static_nodal
    }
    buckle_node_map = {
        int(row["node_label"]): {"x": float(row["scaled_x"]), "y": float(row["scaled_y"]), "z": float(row["scaled_z"]), "umag": float(row["umag"])}
        for row in buckle_nodal
    }
    element_stress = {int(row["element_label"]): float(row["mises_pa_avg"]) / 1e6 for row in static_element}

    original_polygons, element_labels = _build_face_polygons(mesh, original_node_map, ["x", "y", "z"])
    deformed_polygons, _ = _build_face_polygons(mesh, deformed_node_map, ["x", "y", "z"])
    buckle_polygons, _ = _build_face_polygons(mesh, buckle_node_map, ["x", "y", "z"])

    stress_values = [element_stress[label] for label in element_labels]
    disp_values = [float(np.mean([deformed_node_map[node_id]["umag"] for node_id in element["connectivity"]])) * 1000.0 for element in mesh["elements"]]
    buckle_values = [float(np.mean([buckle_node_map[node_id]["umag"] for node_id in element["connectivity"]])) for element in mesh["elements"]]

    case_name = selected_row["case_label"]
    mesh_label = selected_row["mesh_level"]
    _plot_surface(
        original_polygons,
        stress_values,
        "inferno",
        f"{case_name} - Static stress field",
        f"{mesh_label.capitalize()} mesh, true-scale equal-axis view. Abaqus element Mises stress on undeformed shell (MPa). Max Mises = {float(selected_row['max_mises_pa']) / 1e6:.3f} MPa",
        output_dir / f"task7_{selected_row['case_id']}_stress.png",
    )
    _plot_surface(
        deformed_polygons,
        disp_values,
        "viridis",
        f"{case_name} - Static displacement field",
        f"{mesh_label.capitalize()} mesh, true-scale equal-axis view. Deformed shell from Abaqus nodal displacement magnitude (mm). Max displacement = {1000.0 * float(selected_row['max_displacement_m']):.3f} mm",
        output_dir / f"task7_{selected_row['case_id']}_displacement.png",
    )
    _plot_surface(
        buckle_polygons,
        buckle_values,
        "cividis",
        f"{case_name} - Buckling mode 1",
        f"{mesh_label.capitalize()} mesh, true-scale equal-axis view. Mode shape from Abaqus eigenvector field with explicit visualization scaling. Buckling factor = {float(selected_row['buckling_factor_1']):.3f}",
        output_dir / f"task7_{selected_row['case_id']}_buckling_mode1.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Render Task 7 custom plots from exported Abaqus field data.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--mesh-summary", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_summary.csv"))
    parser.add_argument("--summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv"))
    parser.add_argument("--fields-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "fields"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "figures"))
    args = parser.parse_args()

    mesh_summary = _load_csv(Path(args.mesh_summary))
    summary_rows = _load_csv(Path(args.summary_csv))
    fields_dir = Path(args.fields_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_summary_map = {row["job_name"]: row for row in mesh_summary}
    row_map = _case_rows(summary_rows)
    refined_rows = [_preferred_row(case_id, row_map) for case_id in sorted(row_map.keys(), key=lambda value: (int(value.split('_')[0][1:]), value.split('_')[1]))]
    ranked_rows = _rank_rows(refined_rows)
    winner_row = ranked_rows[0]

    _comparison_plot(refined_rows, output_dir / "comparison_metrics.png")
    _warning_plot(refined_rows, output_dir / "s8_warning_metrics.png")
    _winner_mesh_comparison(row_map, winner_row, output_dir / f"task7_{winner_row['case_id']}_mesh_comparison.png")

    for selected_row in refined_rows:
        _plot_case(selected_row, mesh_summary_map, fields_dir, output_dir)

    print("Rendered Task 7 custom plots from exported field data.")


if __name__ == "__main__":
    main()
